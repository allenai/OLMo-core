"""
Monitor attention gate scores and log per-layer, per-head stats to W&B.
"""

from __future__ import annotations

import dataclasses
import functools as ft
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from olmo_core.nn.attention import Attention, GateGranularity

from ..common import ReduceType
from ..train_module import TransformerTrainModule
from .callback import Callback

log = logging.getLogger(__name__)

_BLOCK_IDX_RE = re.compile(r"(?:^|\.)blocks\.(\d+)\.attention(?:\.|$)")


@dataclass
class HeadGateStats:
    """Additive per-head gate statistics (sums/counts over gate score elements)."""

    sum: float
    count: int
    count_lt_1e2: int
    count_lt_1e3: int

    @property
    def mean(self) -> float:
        return self.sum / self.count if self.count else 0.0

    @property
    def frac_lt_1e2(self) -> float:
        return self.count_lt_1e2 / self.count if self.count else 0.0

    @property
    def frac_lt_1e3(self) -> float:
        return self.count_lt_1e3 / self.count if self.count else 0.0

    def update(self, other: "HeadGateStats") -> None:
        self.sum += other.sum
        self.count += other.count
        self.count_lt_1e2 += other.count_lt_1e2
        self.count_lt_1e3 += other.count_lt_1e3


def per_head_gate_stats(
    gate_logits: torch.Tensor,
    *,
    n_heads: int,
    head_dim: int,
    granularity: GateGranularity,
) -> List[HeadGateStats]:
    """
    Compute per-head additive gate statistics from raw ``w_g`` logits.

    :param gate_logits: Pre-sigmoid gate projection output. Shape ``(..., H)`` for headwise
        or ``(..., H * D)`` for elementwise.
    :param n_heads: Number of attention heads.
    :param head_dim: Head dimension (used for elementwise reshape).
    :param granularity: Gate granularity.

    :returns: One :class:`HeadGateStats` per head, in head order. Fractions derived from
        these stats are in ``[0, 1]``.
    """
    gates = torch.sigmoid(gate_logits.detach().float())
    if granularity == GateGranularity.headwise:
        if gates.shape[-1] != n_heads:
            raise ValueError(
                f"headwise gate last dim {gates.shape[-1]} does not match n_heads={n_heads}"
            )
        # (H, N) with N = product of leading dims
        per_head = gates.reshape(-1, n_heads).transpose(0, 1)
    elif granularity == GateGranularity.elementwise:
        if gates.shape[-1] != n_heads * head_dim:
            raise ValueError(
                f"elementwise gate last dim {gates.shape[-1]} does not match "
                f"n_heads*head_dim={n_heads * head_dim}"
            )
        # (..., H, D) -> (H, N)
        per_head = gates.view(*gates.shape[:-1], n_heads, head_dim).reshape(
            -1, n_heads, head_dim
        )
        per_head = per_head.permute(1, 0, 2).reshape(n_heads, -1)
    else:
        raise ValueError(f"unsupported gate granularity: {granularity}")

    count = per_head.shape[-1]
    sums = per_head.sum(dim=-1)
    counts_lt_1e2 = (per_head < 1e-2).sum(dim=-1)
    counts_lt_1e3 = (per_head < 1e-3).sum(dim=-1)

    return [
        HeadGateStats(
            sum=float(sums[h]),
            count=count,
            count_lt_1e2=int(counts_lt_1e2[h]),
            count_lt_1e3=int(counts_lt_1e3[h]),
        )
        for h in range(n_heads)
    ]


def _parse_block_idx(module_name: str) -> Optional[int]:
    match = _BLOCK_IDX_RE.search(module_name)
    if match is None:
        return None
    return int(match.group(1))


@dataclass
class GatedAttnMonitorCallback(Callback):
    """
    Log per-layer, per-head attention gate statistics under the ``gated_attn/`` namespace.

    For each gated :class:`~olmo_core.nn.attention.Attention` layer ``i``, records layer-wide:

    - ``gated_attn/layer-{i}/1e-2`` — fraction of gate scores ``< 1e-2``
    - ``gated_attn/layer-{i}/1e-3`` — fraction of gate scores ``< 1e-3``
    - ``gated_attn/layer-{i}/mean`` — mean gate score

    and the same three metrics per head ``j`` under ``gated_attn/layer-{i}/head-{j}/...``.

    ``i`` is the transformer block index. Supports both headwise and elementwise gates.
    """

    enabled: bool = True
    """Master switch."""

    interval: int = 10
    """How often (in steps) to measure and log statistics."""

    _handles: Optional[list] = dataclasses.field(default=None, repr=False)
    _layer_meta: Optional[Dict[int, Tuple[int, int, GateGranularity]]] = dataclasses.field(
        default=None, repr=False
    )
    _accum: Optional[Dict[int, List[HeadGateStats]]] = dataclasses.field(default=None, repr=False)
    _dry_run_complete: bool = dataclasses.field(default=False, repr=False)

    def post_attach(self):
        if not self.enabled:
            return
        if not isinstance(self.trainer.train_module, TransformerTrainModule):
            raise ValueError(f"{type(self).__name__} only works with the TransformerTrainModule.")
        if self.interval <= 0:
            raise ValueError(f"interval must be positive, got {self.interval}")

    def pre_train(self):
        if not self.enabled:
            return

        assert isinstance(self.trainer.train_module, TransformerTrainModule)
        self._reset_hooks()

        layer_meta: Dict[int, Tuple[int, int, GateGranularity]] = {}
        handles: List[torch.utils.hooks.RemovableHandle] = []

        for name, module in self.trainer.train_module.model.named_modules():
            if not isinstance(module, Attention) or module.w_g is None:
                continue
            if module.gate is None:
                continue

            block_idx = _parse_block_idx(name)
            if block_idx is None:
                log.warning(
                    "Skipping gated Attention at '%s': could not parse block index from name", name
                )
                continue

            if block_idx in layer_meta:
                log.warning(
                    "Duplicate gated Attention for block %d (module '%s'); keeping first",
                    block_idx,
                    name,
                )
                continue

            layer_meta[block_idx] = (module.n_heads, module.head_dim, module.gate.granularity)
            handles.append(
                module.w_g.register_forward_hook(ft.partial(self._gate_hook, block_idx=block_idx))
            )
            log.info(
                "Monitoring attention gates at block %d (%s, n_heads=%d)",
                block_idx,
                module.gate.granularity,
                module.n_heads,
            )

        self._handles = handles
        self._layer_meta = layer_meta
        self._accum = None

        if not layer_meta:
            log.warning(
                "%s enabled but no gated Attention modules found; disabling",
                type(self).__name__,
            )
            self.enabled = False
            self._reset_hooks()

    def pre_step(self, batch: Dict[str, Any]):
        del batch
        if not self.enabled:
            return
        self._dry_run_complete = True
        if self.step % self.interval == 0:
            self._reset_accum()

    def pre_optim_step(self):
        if not self.enabled or not self._dry_run_complete:
            return
        if self.step % self.interval != 0:
            return
        self._flush_metrics()

    def close(self):
        self._reset_hooks()
        self._accum = None
        self._dry_run_complete = False

    @torch._dynamo.disable()
    @torch.no_grad()
    def _gate_hook(
        self,
        module: nn.Module,
        args: tuple,
        output: torch.Tensor,
        *,
        block_idx: int,
    ):
        del module, args
        if not self.enabled or not self._dry_run_complete:
            return
        if self.step % self.interval != 0:
            return
        if self._layer_meta is None or self._accum is None:
            return
        if block_idx not in self._layer_meta:
            return

        n_heads, head_dim, granularity = self._layer_meta[block_idx]
        stats = per_head_gate_stats(
            output, n_heads=n_heads, head_dim=head_dim, granularity=granularity
        )

        head_accums = self._accum[block_idx]
        for h, head_stats in enumerate(stats):
            head_accums[h].update(head_stats)

    def _flush_metrics(self):
        if self._accum is None or self._layer_meta is None:
            return

        for block_idx, head_accums in self._accum.items():
            layer_acc = HeadGateStats(sum=0.0, count=0, count_lt_1e2=0, count_lt_1e3=0)
            for h, acc in enumerate(head_accums):
                if acc.count == 0:
                    continue
                layer_acc.update(acc)
                prefix = f"gated_attn/layer-{block_idx}/head-{h}"
                self.trainer.record_metric(
                    f"{prefix}/1e-2",
                    acc.frac_lt_1e2,
                    reduce_type=ReduceType.mean,
                )
                self.trainer.record_metric(
                    f"{prefix}/1e-3",
                    acc.frac_lt_1e3,
                    reduce_type=ReduceType.mean,
                )
                self.trainer.record_metric(
                    f"{prefix}/mean",
                    acc.mean,
                    reduce_type=ReduceType.mean,
                )

            if layer_acc.count == 0:
                continue
            layer_prefix = f"gated_attn/layer-{block_idx}"
            self.trainer.record_metric(
                f"{layer_prefix}/1e-2",
                layer_acc.frac_lt_1e2,
                reduce_type=ReduceType.mean,
            )
            self.trainer.record_metric(
                f"{layer_prefix}/1e-3",
                layer_acc.frac_lt_1e3,
                reduce_type=ReduceType.mean,
            )
            self.trainer.record_metric(
                f"{layer_prefix}/mean",
                layer_acc.mean,
                reduce_type=ReduceType.mean,
            )

        self._accum = None

    def _reset_accum(self):
        if self._layer_meta is None:
            self._accum = None
            return
        self._accum = {
            block_idx: [
                HeadGateStats(sum=0.0, count=0, count_lt_1e2=0, count_lt_1e3=0)
                for _ in range(n_heads)
            ]
            for block_idx, (n_heads, _, _) in self._layer_meta.items()
        }

    def _reset_hooks(self):
        if self._handles is not None:
            for h in self._handles:
                h.remove()
            self._handles = None
        self._layer_meta = None
