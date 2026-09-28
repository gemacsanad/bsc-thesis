"""
Stage 5 — injecting a trained prefix into `generate()`.

Training built the KV cache by hand and called the model directly (`train.py`). Generation
cannot: `generate()` owns the cache and the position bookkeeping, and it gets them wrong for a
pre-filled cache — **silently**, producing fluent text rather than an exception.

The bug is version-specific, and it changed between transformers 4.39.3 and 4.51.3. This file
documents the 4.51.3 one. The 4.39.3 analysis is kept below because the difference is the whole
reason this had to be re-derived rather than ported.

`cache_position` in 4.51.3
--------------------------
`_get_initial_cache_position` (generation/utils.py) builds the position vector like this:

    cache_position = arange(len(input_ids))         # [0 .. L-1]
    past_length    = past_key_values.get_seq_length()
    cache_position = cache_position[past_length:]   # [P .. L-1]   ← the mistake, for us

It assumes the cache holds a **prefix of `input_ids`** — the ordinary prefill-then-decode case,
where the first `past_length` tokens have already been processed and only the remainder is new.
Our cache holds no `input_ids` at all: it holds P graph positions, and every one of the L body
tokens is unprocessed.

The damage lands one step later, in `_cache_dependant_input_preparation`. With a `cache_position`
of length L-P against `input_ids` of length L, it takes the default branch,
`input_ids = input_ids[:, cache_position]`, and **the first P body tokens are dropped**. The
model then generates a fluent continuation of a truncated prompt.

The fix is to stop the slice and offset instead — every body token is new, they simply start at
position P:

    cache_position = arange(L) + P

That also repairs the input slicing for free. With `cache_position[-1] == P+L-1 >= L`,
`_cache_dependant_input_preparation` takes its Exception 3 branch,
`input_ids[:, -len(cache_position):]`, which is all L tokens. On every later step
`_update_model_kwargs_for_generation` carries the last position forward and adds one, so the
decode steps stay correct without further help.

How 4.39.3 differed, and why porting would have been wrong
----------------------------------------------------------
There, `cache_position` was `arange(cur_len)` with no slice at all, so the P prefix positions
were simply uncounted: every row of the causal mask shifted by P, and the correction was to add
P inside `prepare_inputs_for_generation` on every step. Applied to 4.51.3 that would be wrong in
both halves — the offset is already partly accounted for, and the truncation it cannot see would
remain. Hence the override goes on `_get_initial_cache_position`, which is the function that
actually makes the decision, and `prepare_inputs_for_generation` is left alone.

Note that the **forward pass was never broken** in 4.51.3. Calling the model directly with a
pre-filled cache agrees with the un-prefixed run to ~3e-7 max logit error whether
`cache_position` is passed explicitly or omitted, which is why `train.py` needs no patch. Only
`generate()`'s bookkeeping is at fault.

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


def patch_for_prefix(llm, num_prefix_tokens, max_extra=0):
    """Make `generate()` treat all of `input_ids` as new, starting after the cache.

    Call once, before any `generate()` with a prefix. It replaces `_get_initial_cache_position`
    outright rather than correcting its result, because the value that function returns has
    already lost the information needed to reconstruct the right one — its length is L-P, and
    which P entries were dropped is not recoverable from it.

    The patched model is **only** valid for prefixed generation: the assertion below fires on a
    plain `generate()`, where the cache is empty. That is deliberate. `generate.py` applies the
    patch only when `--mode` is not `baseline`, and each mode is its own process, so the control
    arm always runs an unpatched model.

    The assertion also catches the prefix width drifting from `NUM_PREFIX_TOKENS` — which is a
    live risk, since that constant moved from 10 to 32 for Qwen3's grouped-query attention.

    `max_extra` is the stage-9 neighbour-token allowance. Those tokens are prefilled into the
    cache before `generate()` is called (see `prefill_neighbours`), so the cache is P+k long
    rather than P, and k varies per node with its out-degree. The check therefore becomes a
    range rather than an equality — but a *bounded* one, so a cache of the wrong length still
    fails loudly instead of decoding at a silent offset.
    """
    def get_initial_cache_position(input_ids, model_kwargs):
        cache = model_kwargs.get("past_key_values")
        past = cache.get_seq_length() if cache is not None else 0
        assert num_prefix_tokens <= past <= num_prefix_tokens + max_extra, (
            f"cache holds {past} positions but the patch was installed for "
            f"{num_prefix_tokens}..{num_prefix_tokens + max_extra}; the prefix width and "
            f"NUM_PREFIX_TOKENS (or the neighbour-token count) disagree")
        model_kwargs["cache_position"] = (
            torch.arange(input_ids.shape[1], device=input_ids.device) + past)
        return model_kwargs

    llm._get_initial_cache_position = get_initial_cache_position


@torch.no_grad()
def prefill_neighbours(llm, cache, neighbour_tokens):
    """Run the neighbour tokens through the model and append their real KV to `cache`.

    Training feeds `[neighbour tokens ++ prompt]` to the model in ONE pass as `inputs_embeds`
    (train.step_loss). Generation cannot do the same, because `generate()` in transformers
    4.51.3 only forwards `inputs_embeds` when `cache_position[0] == 0` — and with a prefix in
    the cache it never is, so the embeddings would be silently dropped for an empty
    `input_ids`. Prefilling them here instead keeps `generate()` on the `input_ids` path it
    already works on, and leaves `patch_for_prefix` as the only override needed.

    Splitting the pass at this boundary is exact, not an approximation: attention is causal,
    so a neighbour token's keys and values depend only on the KV prefix and the neighbour
    tokens at or before it — never on the prompt that follows. The two-pass cache is therefore
    the one-pass cache. verify_neighbour_tokens.py holds that claim numerically.

    -> the number of positions appended.
    """
    if neighbour_tokens is None or neighbour_tokens.size(1) == 0:
        return 0

    past, k = cache.get_seq_length(), neighbour_tokens.size(1)
    llm.model(
        inputs_embeds=neighbour_tokens,
        past_key_values=cache,
        attention_mask=torch.ones(1, past + k, dtype=torch.long,
                                  device=neighbour_tokens.device),
        cache_position=torch.arange(past, past + k, device=neighbour_tokens.device),
        use_cache=True,
    )
    return k


@torch.no_grad()
def generate_with_prefix(llm, input_ids, prefix, neighbour_tokens=None, **generate_kwargs):
    """Greedy/sampled continuation of `input_ids` behind the graph's two channels.

    `prefix` is the pooled KV prefix; `neighbour_tokens` is (1, k, hidden) of input embeddings,
    or None. The sequence the model sees is the one train.step_loss documents:

        positions 0 .. P-1        the KV prefix
        positions P .. P+k-1      the neighbour tokens
        positions P+k ..          `input_ids`

    `input_ids` is (1, L) — batch size 1, so no padding is involved and the mask is all ones.
    Returns only the newly generated token ids.
    """
    length = input_ids.size(1)
    cache = prefix_to_cache(prefix)
    k = prefill_neighbours(llm, cache, neighbour_tokens)
    mask = torch.ones(1, prefix.size(4) + k + length, dtype=torch.long,
                      device=input_ids.device)

    out = llm.generate(
        input_ids=input_ids,
        attention_mask=mask,
        past_key_values=cache,
        **generate_kwargs,
    )
    return out[0, length:]
