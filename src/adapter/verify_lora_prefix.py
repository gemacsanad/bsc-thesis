"""Check that LoRA does not disturb the prefix path."""

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
    """peft zero-initialises `lora_B`, so a freshly injected adapter is exactly the identity."""
    with torch.no_grad():
        for name, param in llm.named_parameters():
            if "lora_B" in name:
                param.normal_(std=0.02)


def forward_with_prefix(llm, input_ids, prefix):
    """-> logits. The injection as train.py performs it — PrefixCache, explicit cache_position, mask
    covering the prefix, use_cache off.
    """
    p, (b, length) = prefix.size(4), input_ids.shape
    mask = torch.ones(b, p + length, dtype=torch.long)
    return llm(input_ids=input_ids,
               past_key_values=PrefixCache(prefix),
               attention_mask=mask,
               cache_position=torch.arange(p, p + length),
               use_cache=False).logits


def loss_with_prefix(llm, input_ids, prefix, labels):
    """The training call, as in `train.step_loss`."""
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

    widths = {name.split(".")[-4]: param.shape[0]
              for name, param in llm.named_parameters() if "lora_B" in name}
    assert len(set(widths.values())) > 1, \
        "both LoRA targets have the same output width — GQA is not being exercised"
    print(f"1. injected {n_lora:,} LoRA parameters in {len(lora_params) // 2} modules; "
          f"logits move by {moved:.3f}; lora_B widths {widths}")

    # --- 2. gradients reach both channels, through checkpointing ----------
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
