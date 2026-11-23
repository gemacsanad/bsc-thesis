"""Check that the prefix reaches the model unchanged during `generate()`."""

import torch
import transformers
from transformers import Qwen3Config, Qwen3ForCausalLM

from prefix_generation import generate_with_prefix, patch_for_prefix

assert transformers.__version__ == "4.51.3", f"expected 4.51.3, got {transformers.__version__}"

PREFIX_LEN, BODY_LEN, NEW_TOKENS = 10, 16, 20


def real_prefix(llm, prefix_ids):
    """Run `prefix_ids` through the model and reshape its cache into the adapter's layout."""
    with torch.no_grad():
        cache = llm(prefix_ids, use_cache=True).past_key_values
    return torch.stack([torch.stack([k, v], dim=1) for k, v in cache], dim=1)


def main():
    torch.manual_seed(0)

    # sliding_window off, as in Qwen3-14B's config (Qwen3Config defaults it to 4096).
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

    try:
        generate_with_prefix(llm, body_ids, prefix[:, :, :, :, :PREFIX_LEN - 2],
                             max_new_tokens=1, do_sample=False, pad_token_id=0)
        raise SystemExit("a prefix of the wrong width was accepted — the guard is dead")
    except AssertionError:
        print("width guard: a prefix that disagrees with NUM_PREFIX_TOKENS raises")

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
