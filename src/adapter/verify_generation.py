"""
Stage 5, step 1 — prove the prefix survives `generate()` before spending GPU hours.

The failure this guards against is silent: a `cache_position` off by P produces fluent,
plausible abstracts, not an error. A generation run that got it wrong would be
indistinguishable from one that got it right, after the fact.

Method, same as the Stage 4 injection check: compare **two runs of the same model**, so the
weights are irrelevant and a 4-layer random Llama on CPU exercises the same `forward`,
`_update_causal_mask`, RoPE and `DynamicCache.update` as Vicuna-7B.

The ground truth is text. Split a token sequence into `prefix ++ body`, then compare

  A. plain greedy generation over the whole `prefix ++ body`
  B. greedy generation over `body` alone, with the *real* key/values of `prefix` injected

A zero or random prefix would prove nothing — a broken mask still yields plausible tokens. The
injected cache has to be one whose correct effect is already known, and the real KV of a real
token span is exactly that: B must reproduce A token for token.

The negative control matters as much: the same comparison **without** `patch_for_prefix`
must fail. If it passed, the patch would be untested and the whole check vacuous.

Usage:
    python adapter/evaluation/verify_generation.py     # ~10 s on a laptop CPU
"""

import torch
import transformers
from transformers import LlamaConfig, LlamaForCausalLM

from prefix_generation import generate_with_prefix, patch_for_prefix

# The Komondor container pins this; the `prepare_inputs_for_generation` internals the patch
# delegates to are version-specific, so a silent upgrade must not go unnoticed.
assert transformers.__version__ == "4.39.3", f"expected 4.39.3, got {transformers.__version__}"

PREFIX_LEN, BODY_LEN, NEW_TOKENS = 10, 16, 20


def real_prefix(llm, prefix_ids):
    """Run `prefix_ids` through the model and reshape its cache into the adapter's layout.

    -> (1, layers, 2, heads, tokens, head_dim), the shape `PrefixAdapter.forward` returns.
    """
    with torch.no_grad():
        cache = llm(prefix_ids, use_cache=True).past_key_values
    return torch.stack([torch.stack([k, v], dim=1) for k, v in cache], dim=1)


def main():
    torch.manual_seed(0)

    config = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4)
    llm = LlamaForCausalLM(config).eval()

    tokens = torch.randint(0, config.vocab_size, (1, PREFIX_LEN + BODY_LEN))
    prefix_ids, body_ids = tokens[:, :PREFIX_LEN], tokens[:, PREFIX_LEN:]
    prefix = real_prefix(llm, prefix_ids)
    assert prefix.shape == (1, config.num_hidden_layers, 2, config.num_attention_heads,
                            PREFIX_LEN, config.hidden_size // config.num_attention_heads)

    # A — the reference: no cache tricks, the prefix is just more tokens
    with torch.no_grad():
        reference = llm.generate(tokens, max_new_tokens=NEW_TOKENS, do_sample=False,
                                 pad_token_id=0)[0, tokens.size(1):]

    # B — the prefix as a cache, unpatched. Expected to disagree; if it does not, the rest of
    # this file proves nothing.
    unpatched = generate_with_prefix(llm, body_ids, prefix, max_new_tokens=NEW_TOKENS,
                                     do_sample=False, pad_token_id=0)
    assert not torch.equal(reference, unpatched), \
        "unpatched generation already matches — the negative control is broken, so a passing " \
        "patched run would say nothing about the patch"
    disagree = int((reference != unpatched).sum())
    print(f"unpatched: {disagree}/{NEW_TOKENS} tokens differ from the reference (as expected)")

    # B' — the same, patched. This must match token for token.
    patch_for_prefix(llm, PREFIX_LEN)
    patched = generate_with_prefix(llm, body_ids, prefix, max_new_tokens=NEW_TOKENS,
                                   do_sample=False, pad_token_id=0)
    assert torch.equal(reference, patched), (
        f"patched generation still differs at "
        f"{int((reference != patched).sum())}/{NEW_TOKENS} tokens\n"
        f"  reference {reference.tolist()}\n  patched   {patched.tolist()}"
    )
    print(f"patched:   {NEW_TOKENS}/{NEW_TOKENS} tokens match the reference exactly")

    # Logits, not just argmax: matching tokens can hide a small corruption that sampling or a
    # longer run would surface.
    with torch.no_grad():
        want = llm(tokens).logits[0, PREFIX_LEN:]
        mask = torch.ones(1, PREFIX_LEN + BODY_LEN, dtype=torch.long)
        from prefix_generation import prefix_to_cache
        got = llm(body_ids, past_key_values=prefix_to_cache(prefix), attention_mask=mask,
                  cache_position=torch.arange(PREFIX_LEN, PREFIX_LEN + BODY_LEN),
                  use_cache=False).logits[0]
    error = (want - got).abs().max().item()
    assert error < 1e-4, f"prefilled forward pass disagrees by {error:.2e}"
    print(f"forward pass: max logit error {error:.1e} against the plain run")

    print("\nprefix injection survives generate()")


if __name__ == "__main__":
    main()
