"""
Stage 7, step 6 — prove the sliced `lm_head` computes the same loss as `labels=`.

`step_loss` no longer asks the LLM for its own loss. It takes hidden states from `llm.model`
and runs `lm_head` on the ~250 positions that actually carry a label, instead of on all ~3,400.
The motive is memory: Qwen3's vocabulary is 151,936 tokens, 4.75× Vicuna's, and the discarded
logits are what pushes a 29.6 GB model past a 40 GB card.

The motive is memory, but the risk is arithmetic, and it is silent in the usual way. An
off-by-one in the shift, a mean taken over the wrong denominator, or padding leaking into the
selection all produce a plausible loss that falls during training. Nothing raises. So the claim
"mathematically identical" is checked here against the `labels=` path itself rather than argued.

Three checks, on a 4-layer random Qwen3 on CPU in float32 — the weights are irrelevant, only
the bookkeeping is under test:

  1. equivalence, no prefix — isolates the slicing from the injection;
  2. equivalence through `PrefixCache`, the exact call `step_loss` makes;
  3. padding neutrality — a right-padded batch of two unequal sequences must give the
     token-weighted mean of the same two computed singly. This is the Stage-4 check, re-run
     because the selection now indexes the label tensor directly, where a padded row would
     be a wrong answer rather than an error.

Usage:
    cd src/adapter
    python verify_sliced_loss.py     # ~10 s on a laptop CPU
"""

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from train import PrefixCache, sliced_loss

TOL = 1e-4
PREFIX_LEN, PROMPT_LEN = 8, 24        # prompt tokens are -100, like a real example


def make_example(config, length, prompt_len=PROMPT_LEN):
    """-> (input_ids, labels) for one sequence: a masked prompt then supervised target tokens."""
    input_ids = torch.randint(0, config.vocab_size, (1, length))
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    return input_ids, labels


def reference_loss(llm, input_ids, labels, mask, cache=None, cache_position=None):
    """The path being replaced: full logits, transformers' own ForCausalLMLoss."""
    with torch.no_grad():
        return llm(input_ids=input_ids, attention_mask=mask, past_key_values=cache,
                   cache_position=cache_position, use_cache=False, labels=labels).loss


def our_loss(llm, input_ids, labels, mask, cache=None, cache_position=None):
    """The path in step_loss: hidden states, then lm_head on the supervised rows only."""
    with torch.no_grad():
        hidden = llm.model(input_ids=input_ids, attention_mask=mask, past_key_values=cache,
                           cache_position=cache_position, use_cache=False).last_hidden_state
        return sliced_loss(llm, hidden, labels)


def main():
    torch.manual_seed(0)

    # float32, not bf16: the tolerance is meant to catch a wrong denominator or a shifted
    # window, and bf16 rounding would swamp both. sliding_window pinned off to match Qwen3-14B.
    config = Qwen3Config(vocab_size=256, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=16, sliding_window=None, use_sliding_window=False)
    llm = Qwen3ForCausalLM(config).eval()

    # --- 1. the slicing alone -------------------------------------------------
    input_ids, labels = make_example(config, 64)
    mask = torch.ones_like(input_ids)
    want = reference_loss(llm, input_ids, labels, mask)
    got = our_loss(llm, input_ids, labels, mask)
    error = abs(want.item() - got.item())
    assert error < TOL, f"sliced loss disagrees with the labels= path by {error:.2e}"

    supervised = int((labels[:, 1:] != -100).sum())
    print(f"no prefix:   {want.item():.6f} vs {got.item():.6f}  (error {error:.1e}), "
          f"head ran on {supervised}/{input_ids.size(1)} positions")

    # --- 2. the same, through PrefixCache — the call step_loss actually makes --
    prefix = torch.randn(1, config.num_hidden_layers, 2, config.num_key_value_heads,
                         PREFIX_LEN, config.head_dim)
    length = input_ids.size(1)
    prefixed_mask = torch.ones(1, PREFIX_LEN + length, dtype=torch.long)
    position = torch.arange(PREFIX_LEN, PREFIX_LEN + length)

    want = reference_loss(llm, input_ids, labels, prefixed_mask,
                          PrefixCache(prefix), position)
    got = our_loss(llm, input_ids, labels, prefixed_mask, PrefixCache(prefix), position)
    error = abs(want.item() - got.item())
    assert error < TOL, f"with a prefix, sliced loss disagrees by {error:.2e}"
    print(f"with prefix: {want.item():.6f} vs {got.item():.6f}  (error {error:.1e})")

    # --- 3. padding neutrality ------------------------------------------------
    # Right-padded, exactly as collate() builds a batch: pad_token_id in input_ids, -100 in
    # labels, zeros in the mask over the padding.
    singles = [make_example(config, n) for n in (40, 72)]
    width = max(ids.size(1) for ids, _ in singles)

    batch_ids = torch.zeros(len(singles), width, dtype=torch.long)
    batch_labels = torch.full((len(singles), width), -100, dtype=torch.long)
    batch_mask = torch.zeros(len(singles), width, dtype=torch.long)
    for i, (ids, lab) in enumerate(singles):
        batch_ids[i, :ids.size(1)] = ids
        batch_labels[i, :lab.size(1)] = lab
        batch_mask[i, :ids.size(1)] = 1

    prefix_mask = torch.ones(len(singles), PREFIX_LEN, dtype=torch.long)
    batch_prefix = prefix.expand(len(singles), -1, -1, -1, -1, -1)
    batched = our_loss(llm, batch_ids, batch_labels,
                       torch.cat([prefix_mask, batch_mask], dim=1),
                       PrefixCache(batch_prefix),
                       torch.arange(PREFIX_LEN, PREFIX_LEN + width))

    loss_sum = token_sum = 0.0
    for ids, lab in singles:
        n = ids.size(1)
        single = our_loss(llm, ids, lab, torch.ones(1, PREFIX_LEN + n, dtype=torch.long),
                          PrefixCache(prefix), torch.arange(PREFIX_LEN, PREFIX_LEN + n))
        tokens = int((lab[:, 1:] != -100).sum())
        loss_sum += single.item() * tokens
        token_sum += tokens
    weighted = loss_sum / token_sum

    error = abs(batched.item() - weighted)
    assert error < TOL, (
        f"padding is not neutral: batched {batched.item():.6f} vs token-weighted singles "
        f"{weighted:.6f}, differing by {error:.2e}")
    print(f"padding:     {batched.item():.6f} batched vs {weighted:.6f} token-weighted  "
          f"(error {error:.1e}, {width - singles[0][0].size(1)} tokens of padding)")

    print("\nthe sliced head computes the loss the labels= path computed")


if __name__ == "__main__":
    main()
