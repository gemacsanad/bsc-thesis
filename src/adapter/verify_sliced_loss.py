"""Check that the sliced `lm_head` computes the same loss as `labels=`."""

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

    # float32, so bf16 rounding does not hide a wrong denominator or a shifted window.
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
