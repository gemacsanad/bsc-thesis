"""
Stage 4b — LoRA adapters on the otherwise frozen LLM.

The KV prefix enters the LLM through attention, as extra keys and values. It supplies those
directly, so `k_proj` and `v_proj` never touch it and `q_proj` alone decides how much of it
the model can read. That was the whole target list until stage 9, and the reasoning behind
stopping there still stands: wider targets (`o_proj`, the MLP) are more capacity for the LLM
to learn the task on its own — which is the same capacity that lets it ignore the graph.
Step 5 measured that happening. Given a prefix carrying little, r=16 on q/v spent itself
learning a template: 453 distinct abstracts out of 945, and judge accuracy *below* the frozen
run it was meant to improve on.

`k_proj` joins the list in stage 9, and only because the neighbour tokens are a different
kind of object. They are ordinary input embeddings, so unlike the prefix their keys are
*computed* by `k_proj` at every layer. With it frozen the model has no way to make the new
tokens attendable, whatever else it learns — so this is the projection the new channel
depends on, not general capacity. See config.LORA_TARGETS.

`inject_adapter_in_model`, not `get_peft_model`
-----------------------------------------------
Injection edits the module tree in place and leaves the object a `LlamaForCausalLM`. A
`PeftModel` wrapper would not: its `generate()` reassigns `prepare_inputs_for_generation`,
which is the exact attribute `prefix_generation.patch_for_prefix` monkeypatches, and its
`forward` would have to relay `past_key_values`, `cache_position` and `use_cache` through
`**kwargs` for `PrefixCache` to keep working. Injection keeps every one of those paths
untouched, and needs no merge step at generation time whose numerics could drift from
training's.

Everything here is deliberately explicit rather than delegated to peft's defaults: which
parameters are trainable, what dtype they are in, and which tensors go into the checkpoint.
"""

from peft import LoraConfig, inject_adapter_in_model

from config import LORA_ALPHA, LORA_DROPOUT, LORA_R, LORA_TARGETS


def lora_config():
    """The one place the adapter's shape is defined — train.py and generate.py must agree."""
    return LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                      target_modules=list(LORA_TARGETS), bias="none", task_type="CAUSAL_LM")


def add_lora(llm):
    """Inject LoRA into `llm` in place. -> the list of new trainable parameters.

    The parameters are moved to fp32 in training *and* generation. AdamW on bf16 parameters
    loses updates below bf16's precision, and peft casts activations to the adapter's dtype
    on the way in — so if the two halves disagreed, generation would not be running what
    training produced.
    """
    inject_adapter_in_model(lora_config(), llm)

    params = []
    for name, param in llm.named_parameters():
        param.requires_grad_("lora_" in name)      # set here, not left to peft's default
        if param.requires_grad:
            param.data = param.data.float()
            params.append(param)
    return params


def lora_state_dict(llm):
    """Just the adapter tensors — the base weights are frozen and already on disk."""
    return {key: value for key, value in llm.state_dict().items() if "lora_" in key}


def load_lora_state(llm, state):
    """Restore `state` into an already-injected `llm`.

    The key sets are compared before loading. `load_state_dict(strict=False)` reports every
    base weight as missing here, so its own checks say nothing useful; a rank or target-module
    mismatch between the checkpoint and config.py has to be caught explicitly or it would load
    a partial adapter and generate fluent text from it.
    """
    have = {key for key in llm.state_dict() if "lora_" in key}
    assert have == set(state), (
        f"the checkpoint's LoRA tensors do not match the injected adapter — "
        f"{len(set(state) - have)} unexpected, {len(have - set(state))} missing; "
        f"config.py's LORA_R or LORA_TARGETS has changed since this checkpoint was trained")
    llm.load_state_dict(state, strict=False)
