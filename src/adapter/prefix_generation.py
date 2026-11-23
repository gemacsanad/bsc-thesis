"""Generation with a trained prefix written into the key-value cache of the LLM."""

import torch
from transformers.cache_utils import DynamicCache


def prefix_to_cache(prefix):
    """(batch, layers, 2, heads, tokens, head_dim) -> a `DynamicCache` holding only the prefix."""
    cache = DynamicCache()
    for layer in range(prefix.size(1)):
        cache.update(prefix[:, layer, 0], prefix[:, layer, 1], layer)
    return cache


def patch_for_prefix(llm, num_prefix_tokens, max_extra=0):
    """Make `generate()` treat all of `input_ids` as new, starting after the cache."""
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
    """Run the neighbour tokens through the model and append their real KV to `cache`."""
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
    """Greedy/sampled continuation of `input_ids` behind the graph's two channels."""
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
