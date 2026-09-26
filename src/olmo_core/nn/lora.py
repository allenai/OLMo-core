"""
LoRA (low-rank adaptation) for OLMo-core models.

Adds a trainable ``B @ A`` update alongside a frozen ``nn.Linear``, so a finetune touches
a few percent of the parameters instead of all of them. The motivating use here is cheap
Molmo2 stage-2 dataset/mixture ablations: the base weights stop needing optimizer state,
weight-gradient GEMMs, and gradient reduce-scatter, while the forward pass is unchanged.

Two design choices are load-bearing and should not be "simplified" away:

**``LoRALinear`` subclasses ``nn.Linear`` rather than wrapping it.** A wrapper would
rename every adapted parameter (``...w_q.weight`` -> ``...w_q.base.weight``), which breaks
checkpoint loading and violates the model/optimizer key-space invariant documented on
:meth:`~olmo_core.nn.vision.MultimodalLM.legacy_vision_key_mapping`. Subclassing leaves
every pre-existing parameter name untouched and adds exactly two new names per adapted
layer, ``lora_A`` and ``lora_B``.

**``lora_B`` is zero-initialised**, so a freshly adapted model is *bitwise* identical to
the base model on the forward pass. That makes "did the checkpoint load correctly?"
testable: step-0 loss must match a full-finetune run exactly.

Typical use, from a train module::

    freeze_params(model, ["lm.*"])          # or the train module's own freeze loop
    names = apply_lora(model, LoRAConfig(rank=64, alpha=128.0))

and, offline, before evaluating the resulting checkpoint::

    merge_lora_(model)                      # W += (alpha/rank) * B @ A, adapters removed
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from ..exceptions import OLMoConfigurationError

__all__ = [
    "LLM_LORA_TARGET_MODULES",
    "LoRAConfig",
    "LoRALinear",
    "apply_lora",
    "lora_param_names",
    "merge_lora_",
]

log = logging.getLogger(__name__)


LLM_LORA_TARGET_MODULES: List[str] = [
    "lm.blocks.*.attention.w_q",
    "lm.blocks.*.attention.w_k",
    "lm.blocks.*.attention.w_v",
    "lm.blocks.*.attention.w_out",
    "lm.blocks.*.feed_forward.w1",
    "lm.blocks.*.feed_forward.w2",
    "lm.blocks.*.feed_forward.w3",
]
"""Attention + MLP projections of the language model. Deliberately excludes embeddings,
the LM head (tied to the embeddings on Molmo2-4B), and every norm."""


class LoRALinear(nn.Linear):
    """
    An :class:`nn.Linear` with an additive low-rank update.

    ``forward(x) = F.linear(x, weight, bias) + (dropout(x) @ A.T) @ B.T * (alpha / rank)``

    ``weight`` and ``bias`` keep their names and are expected to be frozen; only
    ``lora_A`` and ``lora_B`` require grad.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        if rank <= 0:
            raise OLMoConfigurationError(f"LoRA rank must be positive, got {rank}")
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank, device=device, dtype=dtype))

    def reset_lora_parameters(self, generator: Optional[torch.Generator] = None) -> None:
        """Standard LoRA init: ``A`` Kaiming-uniform, ``B`` zeros (so the delta starts at 0)."""
        # `nn.init.kaiming_uniform_` does not accept a generator, so do the bound by hand:
        # gain for a=sqrt(5) leaves bound = sqrt(3 / fan_in), matching nn.Linear's default.
        bound = math.sqrt(3.0 / self.lora_A.shape[1])
        with torch.no_grad():
            self.lora_A.uniform_(-bound, bound, generator=generator)
            self.lora_B.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = F.linear(x, self.weight, self.bias)
        lora_x = self.lora_dropout(x)
        return out + F.linear(F.linear(lora_x, self.lora_A), self.lora_B) * self.scaling

    def merged_weight(self) -> torch.Tensor:
        """``weight + scaling * B @ A``, in ``weight``'s dtype."""
        delta = (self.lora_B @ self.lora_A) * self.scaling
        return self.weight + delta.to(self.weight.dtype)

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, rank={self.rank}, alpha={self.alpha}"


@dataclass
class LoRAConfig(Config):
    """Configuration for :func:`apply_lora`."""

    rank: int = 64
    alpha: float = 128.0
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: list(LLM_LORA_TARGET_MODULES))
    """fnmatch globs over *module* fully-qualified names (not parameter names)."""
    init_seed: int = 7919
    """Seeds ``lora_A`` identically on every rank, so no collective is needed to agree."""

    def __post_init__(self):
        if self.rank <= 0:
            raise OLMoConfigurationError(f"LoRA rank must be positive, got {self.rank}")
        if not self.target_modules:
            raise OLMoConfigurationError("LoRA 'target_modules' must not be empty")


def _matches(name: str, patterns: List[str]) -> bool:
    return any(fnmatch(name, pat) for pat in patterns)


def apply_lora(model: nn.Module, config: LoRAConfig) -> List[str]:
    """
    Replace every ``nn.Linear`` whose FQN matches ``config.target_modules`` with a
    :class:`LoRALinear`, freeze its ``weight``/``bias``, and initialise the adapters.

    The original ``weight``/``bias`` :class:`~torch.nn.Parameter` objects are *reused*, not
    copied — the model may be several GB and, in the Molmo2 flow, was just materialised by
    ``to_empty()``. Call this *after* the model is on its real device and *before*
    activation checkpointing, ``torch.compile``, and FSDP wrapping.

    :returns: The new parameter names (``*.lora_A`` / ``*.lora_B``), sorted.

    :raises OLMoConfigurationError: If any pattern matches no ``nn.Linear``. A silent
        no-match would mean training zero parameters, which is not a state worth warning
        about and continuing from.
    """
    targets: List[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            if _matches(name, config.target_modules):
                raise OLMoConfigurationError(
                    f"'{name}' is already a LoRALinear; apply_lora was called twice"
                )
            continue
        if isinstance(module, nn.Linear) and _matches(name, config.target_modules):
            targets.append((name, module))

    matched_patterns = {
        pat for pat in config.target_modules for name, _ in targets if fnmatch(name, pat)
    }
    unmatched = [pat for pat in config.target_modules if pat not in matched_patterns]
    if unmatched:
        raise OLMoConfigurationError(
            f"LoRA target_modules patterns matched no nn.Linear: {unmatched}"
        )

    generator = torch.Generator(device="cpu").manual_seed(config.init_seed)
    new_param_names: List[str] = []
    for name, linear in targets:
        adapted = LoRALinear(
            linear.in_features,
            linear.out_features,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
            bias=linear.bias is not None,
            device="meta",
            dtype=linear.weight.dtype,
        )
        # Reuse the base tensors rather than allocating a second copy of the model.
        adapted.weight = linear.weight
        adapted.bias = linear.bias
        adapted.weight.requires_grad_(False)
        if adapted.bias is not None:
            adapted.bias.requires_grad_(False)
        # `lora_A`/`lora_B` are still on meta; materialise them where the base weight lives.
        device = linear.weight.device
        adapted.lora_A = nn.Parameter(
            torch.empty(config.rank, linear.in_features, device=device, dtype=linear.weight.dtype)
        )
        adapted.lora_B = nn.Parameter(
            torch.empty(linear.out_features, config.rank, device=device, dtype=linear.weight.dtype)
        )
        adapted.reset_lora_parameters(generator=None if device.type == "meta" else generator)
        _set_submodule(model, name, adapted)
        new_param_names.extend([f"{name}.lora_A", f"{name}.lora_B"])

    n_lora = sum(p.numel() for n, p in model.named_parameters() if n in set(new_param_names))
    log.info(
        "Applied LoRA (rank=%d, alpha=%.1f) to %d linear layers: %s trainable adapter params",
        config.rank,
        config.alpha,
        len(targets),
        f"{n_lora:,d}",
    )
    return sorted(new_param_names)


def merge_lora_(model: nn.Module) -> List[str]:
    """
    Fold every :class:`LoRALinear` back into a plain :class:`nn.Linear`, in place.

    After this the model's parameter names and shapes are identical to an unadapted model,
    so a merged checkpoint is interchangeable with a full-finetune one — which is what
    keeps the downstream eval path unchanged.

    :returns: The FQNs that were merged, sorted.
    """
    merged: List[str] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, LoRALinear):
            continue
        with torch.no_grad():
            weight = module.merged_weight()
        plain = nn.Linear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device="meta",
            dtype=module.weight.dtype,
        )
        plain.weight = nn.Parameter(weight, requires_grad=module.weight.requires_grad)
        plain.bias = module.bias
        _set_submodule(model, name, plain)
        merged.append(name)
    return sorted(merged)


def lora_param_names(model: nn.Module) -> List[str]:
    """Every ``*.lora_A`` / ``*.lora_B`` parameter name currently in ``model``."""
    return sorted(
        name
        for name, _ in model.named_parameters()
        if name.endswith((".lora_A", ".lora_B"))
    )


def _set_submodule(root: nn.Module, name: str, new_module: nn.Module) -> None:
    parent_name, _, attr = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, attr, new_module)
