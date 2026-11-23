"""LoRA adapters on the attention projections of the otherwise frozen LLM."""

from peft import LoraConfig, inject_adapter_in_model

from config import LORA_ALPHA, LORA_DROPOUT, LORA_R, LORA_TARGETS


def lora_config(targets=None):
    """The one place the adapter's shape is defined — train.py and generate.py must agree."""
    return LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                      target_modules=list(targets or LORA_TARGETS), bias="none",
                      task_type="CAUSAL_LM")


def targets_of(state):
    """-> the target modules a saved LoRA state was trained on, read off its keys."""
    return sorted({part for key in state for part in key.split(".") if part.endswith("_proj")})


def add_lora(llm, targets=None):
    """Inject LoRA into `llm` in place. -> the list of new trainable parameters."""
    inject_adapter_in_model(lora_config(targets), llm)

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
    """Restore `state` into an already-injected `llm`."""
    have = {key for key in llm.state_dict() if "lora_" in key}
    assert have == set(state), (
        f"the checkpoint's LoRA tensors do not match the injected adapter — "
        f"{len(set(state) - have)} unexpected, {len(have - set(state))} missing; "
        f"config.py's LORA_R or LORA_TARGETS has changed since this checkpoint was trained")
    llm.load_state_dict(state, strict=False)
