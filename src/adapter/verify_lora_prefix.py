"""
Stage 4b, step 1 — prove LoRA does not disturb the prefix path, before spending GPU hours.

Same method as `verify_generation.py`: a 4-layer random Llama on CPU exercises the same
`forward`, `_update_causal_mask`, RoPE and cache code as Vicuna-7B, so the weights are
irrelevant and the whole check costs seconds.

Three failures are possible here, and all three are silent — each one yields fluent text
rather than an exception:

  1. LoRA is injected but never trains, because the parameters are frozen or the gradient
     does not survive gradient checkpointing. The run would look normal and produce the
     no-LoRA result.
  2. The prefix stops reaching the model once the module tree is edited, so the graph is
     quietly disconnected and only LoRA is learning.
  3. Generation does not reproduce training — a dtype, a dropout left live, or a state dict
     that loaded only part of the adapter.

Usage:
    python adapter/verify_lora_prefix.py     # ~15 s on a laptop CPU
"""

import copy

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from config import NUM_PREFIX_TOKENS
from lora import add_lora, load_lora_state, lora_state_dict
from prefix_generation import prefix_to_cache
from train import PrefixCache

BODY_LEN = 16


def tiny_llama(seed=0):
    torch.manual_seed(seed)
    config = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4)
    return LlamaForCausalLM(config).eval()


def open_the_adapter(llm):
    """peft zero-initialises `lora_B`, so a freshly injected adapter is exactly the identity.

    Every check below would pass vacuously against it. Filling B with noise is what makes
    "LoRA changed the output" and "generation reproduces training" mean anything.
    """
    with torch.no_grad():
        for name, param in llm.named_parameters():
            if "lora_B" in name:
                param.normal_(std=0.02)


def forward_with_prefix(llm, input_ids, prefix, labels=None):
    """The training call, verbatim from `train.step_loss` — PrefixCache, explicit
    cache_position, mask covering the prefix, use_cache off."""
    p, (b, length) = prefix.size(4), input_ids.shape
    mask = torch.ones(b, p + length, dtype=torch.long)
    return llm(input_ids=input_ids,
               past_key_values=PrefixCache(prefix),
               attention_mask=mask,
               cache_position=torch.arange(p, p + length),
               use_cache=False,
               labels=labels)


def main():
    p = NUM_PREFIX_TOKENS
    base = tiny_llama()
    layers = base.config.num_hidden_layers
    heads = base.config.num_attention_heads
    head_dim = base.config.hidden_size // heads

    body = torch.randint(0, base.config.vocab_size, (1, BODY_LEN))
    prefix = torch.randn(1, layers, 2, heads, p, head_dim) * 0.02

    # --- 1. LoRA is live --------------------------------------------------
    with torch.no_grad():
        before = forward_with_prefix(base, body, prefix).logits

    llm = copy.deepcopy(base)
    lora_params = add_lora(llm)
    open_the_adapter(llm)
    llm.eval()

    n_lora = sum(x.numel() for x in lora_params)
    assert lora_params, "no LoRA parameters were injected"
    assert all(x.dtype == torch.float32 for x in lora_params), "LoRA parameters are not fp32"
    assert not any(x.requires_grad for name, x in llm.named_parameters() if "lora_" not in name), \
        "a base weight is still trainable — the LLM is no longer frozen outside the adapter"

    with torch.no_grad():
        after = forward_with_prefix(llm, body, prefix).logits
    moved = (before - after).abs().max().item()
    assert moved > 1e-3, (
        f"LoRA changed the logits by only {moved:.2e} — the adapter is not on the forward "
        f"path, so everything below would pass vacuously")
    print(f"1. injected {n_lora:,} LoRA parameters in {len(lora_params) // 2} modules; "
          f"logits move by {moved:.3f}")

    # --- 2. gradients reach both channels, through checkpointing ----------
    # `use_reentrant=False` because no tensor input to layer 0 requires a gradient: the two
    # trainable channels enter at the cache and inside the layers, never at the embedding.
    llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    llm.train()

    trained_prefix = prefix.clone().requires_grad_(True)       # stands in for PrefixAdapter
    labels = body.clone()
    forward_with_prefix(llm, body, trained_prefix, labels=labels).loss.backward()

    assert trained_prefix.grad is not None and trained_prefix.grad.abs().sum() > 0, \
        "no gradient reached the prefix — the graph channel is disconnected under LoRA"
    dead = [name for name, param in llm.named_parameters()
            if "lora_" in name and (param.grad is None or param.grad.abs().sum() == 0)]
    assert not dead, f"{len(dead)} LoRA tensors got no gradient, e.g. {dead[:3]}"
    print(f"2. gradients reach the prefix and all {len(lora_params)} LoRA tensors "
          f"through non-reentrant checkpointing")

    # --- 3. a reloaded adapter reproduces this model exactly --------------
    llm.eval()
    state = lora_state_dict(llm)

    restored = copy.deepcopy(base)
    add_lora(restored)
    load_lora_state(restored, state)
    restored.eval()

    with torch.no_grad():
        want = forward_with_prefix(llm, body, prefix).logits
        got = forward_with_prefix(restored, body, prefix).logits
    assert torch.equal(want, got), (
        f"a reloaded checkpoint disagrees by "
        f"{(want - got).abs().max().item():.2e} — generation would not reproduce training")
    print(f"3. {len(state)} saved tensors reload bit-identically")

    # The generation path builds the same prefix as a DynamicCache instead of a PrefixCache.
    # `verify_generation.py` proves that equivalence without LoRA; this re-proves it with the
    # module tree edited, which is the thing that could have broken it.
    with torch.no_grad():
        mask = torch.ones(1, p + BODY_LEN, dtype=torch.long)
        decoded = restored(body, past_key_values=prefix_to_cache(prefix), attention_mask=mask,
                           cache_position=torch.arange(p, p + BODY_LEN),
                           use_cache=False).logits
    error = (want - decoded).abs().max().item()
    assert error < 1e-4, f"the generation cache path disagrees with training by {error:.2e}"
    print(f"4. generation cache path matches training, max logit error {error:.1e}")

    print("\nLoRA trains alongside the prefix and survives the round trip to generation")


if __name__ == "__main__":
    main()
