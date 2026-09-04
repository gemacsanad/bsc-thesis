"""
Stage 4b, step 1 — prove LoRA does not disturb the prefix path, before spending GPU hours.

Same method as `verify_generation.py`: a 4-layer random Qwen3 on CPU exercises the same
`forward`, `_update_causal_mask`, RoPE and cache code as Qwen3-14B, so the weights are
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

Grouped-query attention
-----------------------
`num_attention_heads=4, num_key_value_heads=2`, for the same reason `verify_generation.py`
gives: every earlier version of this file used `kv == q`, so the cache layout could have been
wrong in exactly the dimension Qwen3 introduces and this script would still have passed.

It matters twice over here. `LORA_TARGETS` is `q_proj` and `v_proj`, and under GQA those two
projections **no longer have the same output width** — `q_proj` serves 4 query heads, `v_proj`
2 KV heads. With `kv == q` a LoRA bug that assumed one shape for both would have been
invisible. Qwen3-14B is 40 query heads over 8 KV heads.

Usage:
    python adapter/verify_lora_prefix.py     # ~15 s on a laptop CPU
"""

import copy

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from config import NUM_PREFIX_TOKENS
from lora import add_lora, load_lora_state, lora_state_dict
from prefix_generation import prefix_to_cache
from train import PrefixCache, sliced_loss

BODY_LEN = 16
PROMPT_LEN = 4          # labels are -100 here, so the sliced head is actually exercised


def tiny_qwen(seed=0):
    torch.manual_seed(seed)
    # sliding_window pinned off: Qwen3Config defaults it to 4096, while Qwen3-14B ships null.
    config = Qwen3Config(vocab_size=256, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=16, sliding_window=None, use_sliding_window=False)
    return Qwen3ForCausalLM(config).eval()


def open_the_adapter(llm):
    """peft zero-initialises `lora_B`, so a freshly injected adapter is exactly the identity.

    Every check below would pass vacuously against it. Filling B with noise is what makes
    "LoRA changed the output" and "generation reproduces training" mean anything.
    """
    with torch.no_grad():
        for name, param in llm.named_parameters():
            if "lora_B" in name:
                param.normal_(std=0.02)


def forward_with_prefix(llm, input_ids, prefix):
    """-> logits. The injection as train.py performs it — PrefixCache, explicit
    cache_position, mask covering the prefix, use_cache off.

    Full logits, unlike `train.step_loss`: the checks below compare two models position by
    position, and that is what the sliced head throws away.
    """
    p, (b, length) = prefix.size(4), input_ids.shape
    mask = torch.ones(b, p + length, dtype=torch.long)
    return llm(input_ids=input_ids,
               past_key_values=PrefixCache(prefix),
               attention_mask=mask,
               cache_position=torch.arange(p, p + length),
               use_cache=False).logits


def loss_with_prefix(llm, input_ids, prefix, labels):
    """The training call, verbatim from `train.step_loss` since Stage 7 step 6.

    Hidden states from `llm.model`, then `lm_head` on the supervised rows only. The head is
    bypassed on the way in, so this is also what proves the gradient still reaches LoRA when
    the loss is computed that way — `peft` injects into `llm.model`, which is exactly the
    submodule now being called directly.
    """
    p, (b, length) = prefix.size(4), input_ids.shape
    hidden = llm.model(input_ids=input_ids,
                       past_key_values=PrefixCache(prefix),
                       attention_mask=torch.ones(b, p + length, dtype=torch.long),
                       cache_position=torch.arange(p, p + length),
                       use_cache=False).last_hidden_state
    return sliced_loss(llm, hidden, labels)


def main():
    p = NUM_PREFIX_TOKENS
    base = tiny_qwen()
    layers = base.config.num_hidden_layers
    kv_heads = base.config.num_key_value_heads
    head_dim = base.config.head_dim

    body = torch.randint(0, base.config.vocab_size, (1, BODY_LEN))
    prefix = torch.randn(1, layers, 2, kv_heads, p, head_dim) * 0.02
    print(f"prefix layout {tuple(prefix.shape)} — "
          f"{base.config.num_attention_heads // kv_heads}:1 GQA exercised")

    # --- 1. LoRA is live --------------------------------------------------
    with torch.no_grad():
        before = forward_with_prefix(base, body, prefix)

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
        after = forward_with_prefix(llm, body, prefix)
    moved = (before - after).abs().max().item()
    assert moved > 1e-3, (
        f"LoRA changed the logits by only {moved:.2e} — the adapter is not on the forward "
        f"path, so everything below would pass vacuously")

    # Under GQA the two targets have different output widths, and a checkpoint that silently
    # adapted only the square one would still load. Name them rather than count them.
    widths = {name.split(".")[-4]: param.shape[0]
              for name, param in llm.named_parameters() if "lora_B" in name}
    assert len(set(widths.values())) > 1, \
        "both LoRA targets have the same output width — GQA is not being exercised"
    print(f"1. injected {n_lora:,} LoRA parameters in {len(lora_params) // 2} modules; "
          f"logits move by {moved:.3f}; lora_B widths {widths}")

    # --- 2. gradients reach both channels, through checkpointing ----------
    # `use_reentrant=False` because no tensor input to layer 0 requires a gradient: the two
    # trainable channels enter at the cache and inside the layers, never at the embedding.
    llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    llm.train()

    trained_prefix = prefix.clone().requires_grad_(True)       # stands in for PrefixAdapter
    labels = body.clone()
    labels[:, :PROMPT_LEN] = -100                              # as collate() masks a prompt
    loss_with_prefix(llm, body, trained_prefix, labels).backward()

    assert trained_prefix.grad is not None and trained_prefix.grad.abs().sum() > 0, \
        "no gradient reached the prefix — the graph channel is disconnected under LoRA"
    dead = [name for name, param in llm.named_parameters()
            if "lora_" in name and (param.grad is None or param.grad.abs().sum() == 0)]
    assert not dead, f"{len(dead)} LoRA tensors got no gradient, e.g. {dead[:3]}"
    print(f"2. gradients reach the prefix and all {len(lora_params)} LoRA tensors "
          f"through non-reentrant checkpointing and the sliced head")

    # --- 3. a reloaded adapter reproduces this model exactly --------------
    llm.eval()
    state = lora_state_dict(llm)

    restored = copy.deepcopy(base)
    add_lora(restored)
    load_lora_state(restored, state)
    restored.eval()

    with torch.no_grad():
        want = forward_with_prefix(llm, body, prefix)
        got = forward_with_prefix(restored, body, prefix)
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
