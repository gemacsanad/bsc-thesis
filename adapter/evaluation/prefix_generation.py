"""
Stage 5 — injecting a trained prefix into `generate()`.

Training built the KV cache by hand and called the model directly (`train.py`). Generation
cannot: `generate()` owns the cache and the position bookkeeping, and in transformers 4.39.3
it gets both wrong for a pre-filled cache — **silently**, producing fluent text rather than
an exception.

Two things have to be repaired.

`cache_position`
----------------
Every search method in `generation/utils.py` opens with
`model_kwargs["cache_position"] = torch.arange(cur_len)`, where `cur_len` is the length of
`input_ids` alone — the P prefix positions already sitting in the cache are not counted. That
value reaches `LlamaForCausalLM.prepare_inputs_for_generation`, which passes it straight
through (`cache_position = cache_position[-input_length:]`), and `cache_position` is what
sizes the causal mask. Every row shifts by P, so each token attends mostly to prefix positions
instead of to the tokens before it. Passing `cache_position=` to `generate()` does not help —
it is overwritten before the loop starts.

`prepare_inputs_for_generation` is the last thing to touch `cache_position` before the forward
pass, so the correction goes there. It is `+P` on **every** step, not only the first:
`_update_model_kwargs_for_generation` increments the dict `generate` holds, which never sees
what this override returns, so the value arriving here is always short by exactly P.

Correcting it on entry rather than on exit also fixes `past_length`, which the stock method
derives from `cache_position[0]` and uses to decide how much of `input_ids` is unprocessed.

The cache object
----------------
`PrefixCache` from `train.py` cannot be reused. It is deliberately read-only — it stores
nothing, so that gradient checkpointing can recompute a layer without the cache growing.
Incremental decoding needs the opposite: a cache that keeps every key it is handed. So the
prefix goes into a real `DynamicCache`, which then grows P -> P+L -> P+L+1 ... on its own.

Positions
---------
The stock `position_ids` path (cumsum over the P+L mask) yields `P .. P+L-1` for the body,
which is exactly the convention `train.py` trains under. Nothing to correct there — but the
two must not be changed independently, or the prefix is read at an offset it never saw.
"""

import torch
from transformers.cache_utils import DynamicCache


def prefix_to_cache(prefix):
    """(batch, layers, 2, heads, tokens, head_dim) -> a `DynamicCache` holding only the prefix.

    Built through `.update()` rather than by assigning `key_cache`, so `_seen_tokens` is set
    the way `generate()` expects to find it.
    """
    cache = DynamicCache()
    for layer in range(prefix.size(1)):
        cache.update(prefix[:, layer, 0], prefix[:, layer, 1], layer)
    return cache


def patch_for_prefix(llm, num_prefix_tokens):
    """Shift `cache_position` by `num_prefix_tokens` on every generation step.

    Call once, before any `generate()` with a prefix. The catch-all must be named `**kwargs`:
    `_validate_model_kwargs` inspects this signature by name and only forwards forward()'s
    arguments when it finds `kwargs` or `model_kwargs` there.
    """
    original = llm.prepare_inputs_for_generation

    def prepare(input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None,
                cache_position=None, **kwargs):
        if cache_position is not None:
            cache_position = cache_position + num_prefix_tokens
        return original(input_ids, past_key_values=past_key_values,
                        attention_mask=attention_mask, inputs_embeds=inputs_embeds,
                        cache_position=cache_position, **kwargs)

    llm.prepare_inputs_for_generation = prepare


@torch.no_grad()
def generate_with_prefix(llm, input_ids, prefix, **generate_kwargs):
    """Greedy/sampled continuation of `input_ids` with `prefix` prepended as a KV cache.

    `input_ids` is (1, L) — batch size 1, so no padding is involved and the mask is all ones.
    Returns only the newly generated token ids.
    """
    length = input_ids.size(1)
    mask = torch.ones(1, prefix.size(4) + length, dtype=torch.long, device=input_ids.device)

    out = llm.generate(
        input_ids=input_ids,
        attention_mask=mask,
        past_key_values=prefix_to_cache(prefix),
        **generate_kwargs,
    )
    return out[0, length:]
