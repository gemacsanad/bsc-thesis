"""
Stage 5, step 1 — prove the prefix survives `generate()` before spending GPU hours.

The failure this guards against is silent: a wrong `cache_position` produces fluent, plausible
abstracts, not an error. A generation run that got it wrong would be indistinguishable from one
that got it right, after the fact.

Method, same as the Stage 4 injection check: compare **two runs of the same model**, so the
weights are irrelevant and a 4-layer random Qwen3 on CPU exercises the same `forward`,
`_update_causal_mask`, RoPE and `DynamicCache.update` as Qwen3-14B.

The ground truth is text. Split a token sequence into `prefix ++ body`, then compare

  A. plain greedy generation over the whole `prefix ++ body`
  B. greedy generation over `body` alone, with the *real* key/values of `prefix` injected

A zero or random prefix would prove nothing — a broken mask still yields plausible tokens. The
injected cache has to be one whose correct effect is already known, and the real KV of a real
token span is exactly that: B must reproduce A token for token.

The negative control matters as much: the same comparison **without** `patch_for_prefix` must
fail. If it passed, the patch would be untested and the whole check vacuous.

Grouped-query attention
-----------------------
The test config is deliberately `num_attention_heads=4, num_key_value_heads=2`. Every earlier
version of this file used `kv == q`, so the cache layout could have been wrong in exactly the
dimension Qwen3 introduces and this script would still have passed. Qwen3-14B serves 40 query
heads from 8 KV heads, and the prefix is sized by the **KV** count.

Usage:
    cd src/adapter
    python verify_generation.py     # ~10 s on a laptop CPU
"""

import torch
import transformers
from transformers import Qwen3Config, Qwen3ForCausalLM

from prefix_generation import generate_with_prefix, patch_for_prefix

# The Komondor container pins this. The internals the patch overrides are version-specific —
# 4.39.3 and 4.51.3 get `cache_position` wrong in *different* ways (see prefix_generation.py) —
# so a silent upgrade must not go unnoticed.
assert transformers.__version__ == "4.51.3", f"expected 4.51.3, got {transformers.__version__}"

PREFIX_LEN, BODY_LEN, NEW_TOKENS = 10, 16, 20


def real_prefix(llm, prefix_ids):
    """Run `prefix_ids` through the model and reshape its cache into the adapter's layout.

    -> (1, layers, 2, kv_heads, tokens, head_dim), the shape `PrefixAdapter.forward` returns.
    """
    with torch.no_grad():
        cache = llm(prefix_ids, use_cache=True).past_key_values
    return torch.stack([torch.stack([k, v], dim=1) for k, v in cache], dim=1)


def main():
    torch.manual_seed(0)

    # sliding_window is pinned off: Qwen3Config defaults it to 4096, while Qwen3-14B ships
    # null. A window would let prefix positions fall out of attention as generation advances.
    config = Qwen3Config(vocab_size=256, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=16, sliding_window=None, use_sliding_window=False)
    llm = Qwen3ForCausalLM(config).eval()

    tokens = torch.randint(0, config.vocab_size, (1, PREFIX_LEN + BODY_LEN))
    prefix_ids, body_ids = tokens[:, :PREFIX_LEN], tokens[:, PREFIX_LEN:]
    prefix = real_prefix(llm, prefix_ids)
    assert prefix.shape == (1, config.num_hidden_layers, 2, config.num_key_value_heads,
                            PREFIX_LEN, config.head_dim), \
        f"cache layout is {tuple(prefix.shape)}, not the adapter's"
    print(f"prefix layout {tuple(prefix.shape)} — "
          f"{config.num_attention_heads // config.num_key_value_heads}:1 GQA exercised")

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
        "patched run would say nothing about the patch. If transformers has fixed this, " \
        "delete patch_for_prefix rather than weakening this assertion."
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

    # The patch is installed for a fixed P. A prefix of another width is a silent
    # config/checkpoint mismatch, so it must raise rather than generate.
    try:
        generate_with_prefix(llm, body_ids, prefix[:, :, :, :, :PREFIX_LEN - 2],
                             max_new_tokens=1, do_sample=False, pad_token_id=0)
        raise SystemExit("a prefix of the wrong width was accepted — the guard is dead")
    except AssertionError:
        print("width guard: a prefix that disagrees with NUM_PREFIX_TOKENS raises")

    # Logits, not just argmax: matching tokens can hide a small corruption that sampling or a
    # longer run would surface. This is the train.py path — explicit cache_position, no
    # generate() bookkeeping involved — so it is checked separately from the two runs above.
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

    # Everything above exercises `DynamicCache`. Training uses `PrefixCache`, a different class
    # with a different contract, so it needs its own check — otherwise "prefix injection works"
    # would only ever have been shown for the generation path.
    from train import PrefixCache
    cache = PrefixCache(prefix)
    assert cache.get_seq_length() == PREFIX_LEN
    assert cache.get_max_cache_shape() is None, "an unbounded cache must report no maximum"
    with torch.no_grad():
        train_path = llm(body_ids, past_key_values=cache, attention_mask=mask,
                         cache_position=torch.arange(PREFIX_LEN, PREFIX_LEN + BODY_LEN),
                         use_cache=False).logits[0]
    error = (want - train_path).abs().max().item()
    assert error < 1e-4, f"PrefixCache disagrees with the plain run by {error:.2e}"

    # The defining property: it stores nothing. Gradient checkpointing runs each layer twice,
    # and a cache that appended would have grown by BODY_LEN on the second pass — silently
    # changing the loss rather than raising.
    with torch.no_grad():
        again = llm(body_ids, past_key_values=cache, attention_mask=mask,
                    cache_position=torch.arange(PREFIX_LEN, PREFIX_LEN + BODY_LEN),
                    use_cache=False).logits[0]
    assert torch.equal(train_path, again), "PrefixCache is not read-only — it grew on reuse"
    assert cache.get_seq_length() == PREFIX_LEN, "PrefixCache stored something"
    print(f"PrefixCache:  max logit error {error:.1e}, read-only across repeated passes")

    print("\nprefix injection survives generate()")


if __name__ == "__main__":
    main()
