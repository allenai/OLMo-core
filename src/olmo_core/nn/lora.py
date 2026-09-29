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
    "apply_trainable_token_rows",
    "fold_token_rows",
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
            if generator is None:
                self.lora_A.uniform_(-bound, bound)
            else:
                # Draw on the generator's own device and copy in. `Tensor.uniform_` requires
                # the generator to match the tensor's device ("Expected a 'cuda' device type
                # for generator but found 'cpu'"), and the generator is deliberately a CPU
                # one: seeding it by rank-independent value is what makes every rank produce
                # identical adapters without a collective. Drawing on CPU and copying keeps
                # that property on any device, and is free at these sizes.
                values = torch.empty(
                    self.lora_A.shape, device=generator.device, dtype=torch.float32
                ).uniform_(-bound, bound, generator=generator)
                self.lora_A.copy_(values.to(self.lora_A.dtype))
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
    trainable_token_ids: Optional[List[int]] = None
    """Vocabulary rows that get a trainable per-row delta on the (tied) embedding table.

    Motivation, measured on Molmo2-4B: the rows for ``<think>`` / ``</think>`` /
    ``<tool_call>`` / ``</tool_call>`` are near-initialisation vectors (norm 0.37 vs 1.09
    for ordinary tokens) that are almost collinear with each other (cos 0.92-0.99), and
    ``freeze_params=["lm.*"]`` keeps them frozen under LoRA. The head therefore cannot
    raise ``<think>`` without raising its siblings, and P(<think> | first token) plateaued
    at ~0.07 in every scratchpad arm regardless of dose or loss weight. The delta reaches
    both the input lookup and the tied LM head; it is folded into the table by
    :func:`fold_token_rows` / ``merge_lora_checkpoint.py``.
    """

    def __post_init__(self):
        if self.rank <= 0:
            raise OLMoConfigurationError(f"LoRA rank must be positive, got {self.rank}")
        if not self.target_modules:
            raise OLMoConfigurationError("LoRA 'target_modules' must not be empty")
        if self.trainable_token_ids is not None:
            if not self.trainable_token_ids:
                raise OLMoConfigurationError(
                    "LoRA 'trainable_token_ids' must not be empty when set"
                )
            if len(set(self.trainable_token_ids)) != len(self.trainable_token_ids):
                raise OLMoConfigurationError("LoRA 'trainable_token_ids' must be unique")


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
        if device.type == "meta":
            # Nothing to initialise on meta; the real values are written once the module is
            # materialised. `apply_lora` is normally called after `to_empty()`, so this only
            # covers config-inspection paths that never train.
            adapted.reset_lora_parameters(generator=None)
        else:
            adapted.reset_lora_parameters(generator=generator)
        _set_submodule(model, name, adapted)
        new_param_names.extend([f"{name}.lora_A", f"{name}.lora_B"])

    if config.trainable_token_ids:
        new_param_names.extend(apply_trainable_token_rows(model, config.trainable_token_ids))

    n_lora = sum(p.numel() for n, p in model.named_parameters() if n in set(new_param_names))
    log.info(
        "Applied LoRA (rank=%d, alpha=%.1f) to %d linear layers: %s trainable adapter params",
        config.rank,
        config.alpha,
        len(targets),
        f"{n_lora:,d}",
    )
    return sorted(new_param_names)


TOKEN_DELTA_NAME = "token_delta"
TOKEN_DELTA_IDS_NAME = "token_delta_ids"


def _find_tied_embedding(model: nn.Module):
    """Return ``(fqn, embedding_module, head_linear_or_None)`` for the LM's word embedding.

    Works on a bare ``Transformer`` or anything wrapping one under ``.lm``; the head is
    returned only when its weight *is* the embedding weight (tied), because an untied head
    keeps its own rows and would be adjusted separately.
    """
    for name, module in model.named_modules():
        emb = getattr(module, "embeddings", None)
        head = getattr(module, "lm_head", None)
        if emb is None or head is None or not hasattr(emb, "weight"):
            continue
        w_out = getattr(head, "w_out", None)
        tied = w_out is not None and getattr(w_out, "weight", None) is emb.weight
        emb_fqn = f"{name}.embeddings" if name else "embeddings"
        return emb_fqn, emb, (w_out if tied else None)
    raise OLMoConfigurationError("No module with both 'embeddings' and 'lm_head' found")


def apply_trainable_token_rows(model: nn.Module, token_ids: List[int]) -> List[str]:
    """
    Give the vocabulary rows ``token_ids`` a trainable additive delta while the embedding
    table itself stays frozen.

    The delta is registered on the embedding module as ``token_delta`` (zero-initialised,
    ``(len(token_ids), d_model)``) with a persistent ``token_delta_ids`` buffer, and reaches:

    * the **input lookup** -- ``embedding.forward`` is wrapped to add ``delta[row]`` wherever
      ``input_ids`` hits one of the rows;
    * the **tied LM head** -- a forward hook on ``lm_head.w_out`` adds ``h @ delta.T`` into
      the corresponding logit columns.

    Zero init keeps the model bit-identical to the unadapted one at step 0. Call after
    ``freeze_params`` has been applied (as the train module does for LoRA), so the new
    parameter is left trainable.

    :returns: The FQNs of the new parameter and of its ``token_delta_ids`` buffer. Both are
        absent from a base checkpoint, and the train module's non-strict-load guard uses
        this list as the set of keys allowed to be missing; the buffer's value is fixed at
        construction, so a missing checkpoint entry costs nothing.
    """
    emb_fqn, emb, w_out = _find_tied_embedding(model)
    if hasattr(emb, TOKEN_DELTA_NAME):
        raise OLMoConfigurationError("trainable token rows were already applied")
    weight = emb.weight
    ids = torch.tensor(sorted(set(int(i) for i in token_ids)), dtype=torch.long)
    if int(ids.max()) >= weight.shape[0]:
        raise OLMoConfigurationError(
            f"trainable_token_ids up to {int(ids.max())} exceed the base vocab {weight.shape[0]}"
        )
    device = weight.device
    delta = nn.Parameter(
        torch.zeros(len(ids), weight.shape[1], device=device, dtype=weight.dtype)
        if device.type != "meta"
        else torch.empty(len(ids), weight.shape[1], device=device, dtype=weight.dtype)
    )
    emb.register_parameter(TOKEN_DELTA_NAME, delta)
    emb.register_buffer(TOKEN_DELTA_IDS_NAME, ids.to(device), persistent=True)

    orig_forward = emb.forward

    def forward_with_rows(input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        out = orig_forward(input)
        d = getattr(emb, TOKEN_DELTA_NAME)
        rows = getattr(emb, TOKEN_DELTA_IDS_NAME)
        # positions whose id is one of the trainable rows -> add that row's delta
        match = input.unsqueeze(-1) == rows  # (..., n_rows)
        if match.any():
            out = out + (match.to(d.dtype) @ d.to(out.dtype)).to(out.dtype)
        return out

    emb.forward = forward_with_rows  # type: ignore[method-assign]

    if w_out is not None:

        def head_hook(module, inputs, output):
            d = getattr(emb, TOKEN_DELTA_NAME)
            rows = getattr(emb, TOKEN_DELTA_IDS_NAME)
            h = inputs[0]
            extra = (h.to(d.dtype) @ d.T).to(output.dtype)  # (..., n_rows)
            return output.index_add(-1, rows, extra)

        w_out.register_forward_hook(head_hook)
    log.info(
        "Trainable token rows on %s: %d rows (%s), tied head %s",
        emb_fqn,
        len(ids),
        ids.tolist(),
        "adjusted" if w_out is not None else "not tied -- head rows unchanged",
    )
    return [f"{emb_fqn}.{TOKEN_DELTA_NAME}", f"{emb_fqn}.{TOKEN_DELTA_IDS_NAME}"]


def fold_token_rows(tensors: dict, *, weight_key: str) -> int:
    """
    Fold a saved ``token_delta`` into the embedding table inside a plain state dict
    (``{name: tensor}``), removing the delta and its ids. Used by
    ``merge_lora_checkpoint.py``; returns the number of rows folded (0 if none present).
    """
    prefix = weight_key[: -len(".weight")]
    dkey, ikey = f"{prefix}.{TOKEN_DELTA_NAME}", f"{prefix}.{TOKEN_DELTA_IDS_NAME}"
    if dkey not in tensors:
        return 0
    if ikey not in tensors:
        raise RuntimeError(f"{dkey} present without {ikey}; cannot tell which rows it applies to")
    delta = tensors.pop(dkey)
    ids = tensors.pop(ikey).long()
    weight = tensors[weight_key]
    w = weight.float()
    w[ids] = w[ids] + delta.float()
    tensors[weight_key] = w.to(weight.dtype)
    return int(len(ids))


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
        name for name, _ in model.named_parameters() if name.endswith((".lora_A", ".lora_B"))
    )


def _set_submodule(root: nn.Module, name: str, new_module: nn.Module) -> None:
    parent_name, _, attr = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, attr, new_module)
