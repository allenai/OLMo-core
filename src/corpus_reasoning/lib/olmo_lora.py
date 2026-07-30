"""LoRA for the olmo-core Qwen3-dense path.

olmo-core has no native LoRA/PEFT support, so we add it on top of its existing
primitives. The approach is **in-place augmentation**: for each target
``nn.Linear`` we register ``lora_A``/``lora_B`` parameters *on the existing
module* and patch its ``forward`` to add ``scaling * (x A^T) B^T``. Crucially the
base weight keeps its original dotted name (e.g. ``blocks.0.attention.w_q.weight``)
so the converted-base distcp checkpoint loads into it unchanged — the LoRA params
are simply *extra* params that the base checkpoint doesn't contain.

Lifecycle (see scripts/train/olmo_train.py):
  1. build model on meta -> ``inject_lora`` -> ``freeze_base``  (BEFORE the train
     module / optimizer build, so FSDP shards the LoRA params and the optimizer
     captures only them).
  2. train module build runs ``to_empty`` + ``reset_parameters``; ``nn.Linear``'s
     ``reset_parameters`` only touches ``weight``/``bias``, so the LoRA params are
     left uninitialized here.
  3. the base distcp loads with ``strict=False`` (the LoRA keys are absent and get
     pruned from the to-load dict).
  4. a ``pre_train`` callback calls ``init_lora_params`` (Kaiming ``A``, zeros
     ``B``) — the authoritative init, run after the base load, before step 0.

Export (single-process, plain tensors — see scripts/train/export_olmo_to_hf.py):
  - ``merge_lora`` folds the adapter into the base weight for a normal HF export.
  - ``lora_peft_state_dict`` / ``write_peft_adapter`` emit a standalone PEFT
    adapter (causal-LM key layout, for ``evaluate.py --lora-path``).

The attribute names are olmo-core's: attention ``w_q/w_k/w_v/w_out``, SwiGLU
feed-forward ``w1`` (gate) / ``w3`` (up) / ``w2`` (down). ``lm_head.w_out`` is an
nn.Linear too but is excluded because it is not under ``blocks.``.
"""

from __future__ import annotations

import json
import math
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

# olmo-core linear attribute -> HF projection name (used for the PEFT-adapter export).
ATTN_MAP = {"w_q": "q_proj", "w_k": "k_proj", "w_v": "v_proj", "w_out": "o_proj"}
FF_MAP = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}

# blocks.<idx>.(attention|feed_forward).<wname>
_TARGET_RE = re.compile(r"^blocks\.(\d+)\.(attention|feed_forward)\.(\w+)$")


def _target_kind(name: str, target: str):
    """Return ('attention'|'feed_forward', wname) if `name` is a LoRA target, else None."""
    m = _TARGET_RE.match(name)
    if m is None:
        return None
    kind, wname = m.group(2), m.group(3)
    if kind == "attention" and wname in ATTN_MAP:
        return kind, wname
    if kind == "feed_forward" and wname in FF_MAP and target == "all_linear":
        return kind, wname
    return None


def _local(t: torch.Tensor) -> torch.Tensor:
    """The local shard of a (possibly DTensor) param; identity for plain tensors."""
    return t.to_local() if hasattr(t, "to_local") else t


# --------------------------------------------------------------------------- #
# Injection / freezing / init
# --------------------------------------------------------------------------- #
def _augment_linear(module: nn.Linear, *, r: int, alpha: float, dropout: float) -> None:
    out_f, in_f = module.out_features, module.in_features
    w = module.weight  # matches dtype/device, including meta at build time
    a = nn.Parameter(torch.empty((r, in_f), dtype=w.dtype, device=w.device))
    b = nn.Parameter(torch.empty((out_f, r), dtype=w.dtype, device=w.device))
    module.register_parameter("lora_A", a)
    module.register_parameter("lora_B", b)
    module.lora_r = r
    module.lora_alpha = alpha
    module.lora_scaling = alpha / r
    module.lora_dropout_p = dropout

    orig_forward = module.forward  # bound nn.Linear.forward

    def lora_forward(x, _orig=orig_forward, _m=module):
        out = _orig(x)
        h = x
        if _m.lora_dropout_p > 0.0 and _m.training:
            h = F.dropout(h, p=_m.lora_dropout_p)
        # (x A^T) B^T -- low-rank delta; A:[r,in], B:[out,r]
        delta = F.linear(F.linear(h, _m.lora_A), _m.lora_B)
        return out + _m.lora_scaling * delta

    module.forward = lora_forward  # instance attr shadows the class method


def inject_lora(model: nn.Module, *, r: int, alpha: float,
                dropout: float = 0.0, target: str = "all_linear") -> list[str]:
    """Augment target nn.Linear modules in place. Returns the list of patched names.

    `target`: "all_linear" (attn q/k/v/o + MLP gate/up/down) or "attn_only".
    Call BEFORE the train-module build so the optimizer/FSDP see the LoRA params.
    """
    if target not in ("all_linear", "attn_only"):
        raise ValueError(f"unknown lora target {target!r}")
    patched: list[str] = []
    for name, module in model.named_modules():
        if _target_kind(name, target) is None:
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(f"LoRA target {name!r} is not nn.Linear: {type(module).__name__}")
        _augment_linear(module, r=r, alpha=alpha, dropout=dropout)
        patched.append(name)
    if not patched:
        raise RuntimeError(
            f"inject_lora matched no modules (target={target!r}); "
            "model structure may differ from the expected olmo-core Qwen3 layout.")
    return patched


def freeze_base(model: nn.Module) -> tuple[int, int]:
    """requires_grad=True only for lora_*; everything else frozen. Returns (trainable, frozen)."""
    n_train = n_frozen = 0
    for name, p in model.named_parameters():
        is_lora = name.endswith(".lora_A") or name.endswith(".lora_B")
        p.requires_grad = is_lora
        if is_lora:
            n_train += p.numel()
        else:
            n_frozen += p.numel()
    return n_train, n_frozen


def _kaiming_uniform_(param: torch.Tensor) -> None:
    """nn.Linear-style kaiming_uniform_(a=sqrt(5)), computed from the GLOBAL fan_in so it
    is correct whether `param` is a plain tensor or a sharded DTensor."""
    fan_in = param.shape[1]
    gain = math.sqrt(2.0 / (1.0 + 5.0))  # a=sqrt(5) -> gain = sqrt(2/(1+a^2))
    bound = math.sqrt(3.0) * gain / math.sqrt(fan_in)
    with torch.no_grad():
        _local(param).uniform_(-bound, bound)


def init_lora_params(model: nn.Module) -> int:
    """Kaiming `A`, zeros `B` (adapter starts as identity). Returns #modules initialized."""
    n = 0
    for module in model.modules():
        if not hasattr(module, "lora_A"):
            continue
        _kaiming_uniform_(module.lora_A)
        with torch.no_grad():
            _local(module.lora_B).zero_()
        n += 1
    return n


# --------------------------------------------------------------------------- #
# Export-side: merge + PEFT adapter (single process, plain tensors)
# --------------------------------------------------------------------------- #
def merge_lora(model: nn.Module) -> int:
    """In-place fold each adapter into its base weight, drop lora params, restore forward.

    Operates on full tensors (use `.full_tensor()` to gather DTensors first if sharded);
    intended for the single-process exporter where params are plain. Returns #merged.
    """
    n = 0
    for module in model.modules():
        if not hasattr(module, "lora_A"):
            continue
        a = module.lora_A.full_tensor() if hasattr(module.lora_A, "full_tensor") else module.lora_A
        b = module.lora_B.full_tensor() if hasattr(module.lora_B, "full_tensor") else module.lora_B
        with torch.no_grad():
            delta = (module.lora_scaling * (b @ a)).to(module.weight.dtype)
            module.weight.add_(delta)
        del module._parameters["lora_A"]
        del module._parameters["lora_B"]
        module.__dict__.pop("forward", None)  # restore class nn.Linear.forward
        n += 1
    return n


def lora_peft_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Map olmo-core lora params -> PEFT causal-LM adapter keys.

    e.g. blocks.3.attention.w_q.{lora_A,lora_B} ->
         base_model.model.model.layers.3.self_attn.q_proj.{lora_A,lora_B}.weight
    (the layout AutoModelForCausalLM + PEFT expects, matching evaluate.py --lora-path).
    """
    sd: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A"):
            continue
        m = _TARGET_RE.match(name)
        layer, kind, wname = m.group(1), m.group(2), m.group(3)
        proj = ATTN_MAP[wname] if kind == "attention" else FF_MAP[wname]
        sub = "self_attn" if kind == "attention" else "mlp"
        prefix = f"base_model.model.model.layers.{layer}.{sub}.{proj}"
        a = module.lora_A.full_tensor() if hasattr(module.lora_A, "full_tensor") else module.lora_A
        b = module.lora_B.full_tensor() if hasattr(module.lora_B, "full_tensor") else module.lora_B
        sd[f"{prefix}.lora_A.weight"] = a.detach().to(torch.float32).cpu().contiguous()
        sd[f"{prefix}.lora_B.weight"] = b.detach().to(torch.float32).cpu().contiguous()
    return sd


def write_peft_adapter(out_dir: str, state_dict: dict[str, torch.Tensor], *,
                       base_model: str, r: int, alpha: float, dropout: float,
                       target: str) -> None:
    """Write a PEFT-loadable adapter (adapter_model.safetensors + adapter_config.json)."""
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    save_file(state_dict, os.path.join(out_dir, "adapter_model.safetensors"))
    targets = sorted(set(ATTN_MAP.values()) |
                     (set(FF_MAP.values()) if target == "all_linear" else set()))
    cfg = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_model,
        "r": r,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "target_modules": targets,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "modules_to_save": None,
        "use_rslora": False,
        "use_dora": False,
        "rank_pattern": {},
        "alpha_pattern": {},
    }
    with open(os.path.join(out_dir, "adapter_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


# --------------------------------------------------------------------------- #
# Sidecar (training writes it; exporter reads it to rebuild a matching skeleton)
# --------------------------------------------------------------------------- #
def write_lora_sidecar(save_folder: str, *, base_model: str, r: int, alpha: float,
                       dropout: float, target: str) -> str:
    os.makedirs(save_folder, exist_ok=True)
    path = os.path.join(save_folder, "lora_config.json")
    with open(path, "w") as f:
        json.dump({"enabled": True, "base_model": base_model, "r": r,
                   "alpha": alpha, "dropout": dropout, "target": target}, f, indent=2)
    return path


def read_lora_sidecar(save_folder: str):
    """Return the lora sidecar dict, or None if this is a full-FT checkpoint."""
    path = os.path.join(save_folder, "lora_config.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)
