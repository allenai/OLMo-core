"""
Learned per-token block skipping (mixture-of-depths style) -- the third routing dimension next to
the nested-width FFN router (:mod:`olmo_core.nn.nested_ffn_moe`) and the KV-cache router
(:mod:`olmo_core.nn.attention.kv_route`).

Per routed transformer block, a ``Linear(d_model, 1)`` router reads the block input and decides
for every token whether it **runs** the block or **skips** it. A skipped token's hidden state
passes the residual stream unchanged, and it is neither a query nor a key in that block's
attention (nothing later attends to it there), so the whole block -- Q/K/V/O projections, FFN and
attention scores -- is saved for that token. This is what reaches the ~50% of per-token FLOPs the
other two routers cannot touch (projections, and on Qwen3.5 the recurrent layers).

Same recipe as the other routers so the three compose under one budget:

- hard decision in the forward, straight-through gradient through the coefficient
  ``1 + p_sel - p_sel.detach()`` on the block's residual update (value 1, gradient 1);
- keep-all init (router bias +10) so an enabled-but-untrained model reproduces its base;
- two-sided budget on the mean keep probability over tokens and routed blocks, annealed target.

Implementation note: skipping is realised as a **mask**, not a gather -- the block runs on every
token and the skipped tokens' outputs are discarded (``where(keep, block(h), h)``), while the
attention layer receives ``block_keep`` and excludes skipped tokens as keys (through the KV-route
masked path, which also drops them from the KV cache at prefill). The maths is exactly
mixture-of-depths; the wall-clock saving is not realised in this version (a compacted variant that
gathers kept tokens, re-derives ``cu_doc_lens``/``position_ids`` and scatters back is the
follow-up). FLOPs are priced analytically by the trainer's meter.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

__all__ = [
    "BlockSkipHolder",
    "BlockSkipRouter",
    "install_block_skip",
    "block_skip_forward",
    "reset_block_skip_extras",
    "KEEP_INIT_BIAS",
]

KEEP_INIT_BIAS = 10.0


class BlockSkipHolder:
    """Per-forward accumulators, schedules and the decision cache shared by all routed blocks."""

    def __init__(
        self,
        *,
        target: float = 0.5,
        budget_weight: float = 1.0,
        two_sided: bool = True,
        target_anneal_calls: int = 0,
        seed: int = 0,
        start_layer: int = 0,
        n_layers: int = 0,
    ):
        self.target = float(target)
        self.budget_weight = float(budget_weight)
        self.two_sided = bool(two_sided)
        self.target_anneal_calls = int(target_anneal_calls)
        self.seed = int(seed)
        self.start_layer = int(start_layer)
        self.n_layers = int(n_layers)
        self.enabled = True
        self.calls = 0
        self.collect_loss = True
        self.routed_layers: List[int] = []
        self._choice_cache: Dict[tuple, torch.Tensor] = {}
        self._exp_keep: List[torch.Tensor] = []
        self._exp_layers: List[int] = []
        self._hard_kept: Dict[int, float] = {}
        self._n_tokens: Dict[int, int] = {}
        self._depth_count: Optional[torch.Tensor] = None
        self.last_per_layer_keep: Dict[int, float] = {}
        self.last_depth_hist: List[float] = []
        self.cum_kept = 0.0
        self.cum_tokens = 0

    def _reset(self) -> None:
        self._exp_keep, self._exp_layers, self._hard_kept, self._n_tokens = [], [], {}, {}
        self._depth_count = None

    def begin_forward(self, *, collect_loss: bool = True) -> None:
        if self._n_tokens:
            self.last_per_layer_keep = {
                li: self._hard_kept[li] / max(1, self._n_tokens[li]) for li in self._n_tokens
            }
            if self._depth_count is not None:
                n_r = max(1, len(self.routed_layers))
                hist = torch.bincount(self._depth_count, minlength=n_r + 1).float()
                self.last_depth_hist = (hist / hist.sum().clamp(min=1)).tolist()
        self._reset()
        self._choice_cache = {}
        self.collect_loss = bool(collect_loss)
        self.calls += 1

    def set_calls(self, calls: int) -> None:
        self.calls = int(calls)

    def current_target(self) -> float:
        if self.target_anneal_calls <= 0:
            return self.target
        return 1.0 + (self.target - 1.0) * min(1.0, self.calls / self.target_anneal_calls)

    def accumulate(self, *, exp_keep: torch.Tensor, keep: torch.Tensor, layer_idx: int) -> None:
        if self.collect_loss:
            self._exp_keep.append(exp_keep)
            self._exp_layers.append(layer_idx)
        kept = float(keep.sum().item())
        n = int(keep.numel())
        self._hard_kept[layer_idx] = kept
        self._n_tokens[layer_idx] = n
        self.cum_kept += kept
        self.cum_tokens += n
        flat = keep.reshape(-1).to(torch.int64)
        self._depth_count = (
            flat
            if self._depth_count is None or self._depth_count.numel() != n
            else self._depth_count + flat
        )

    def regularization_loss(self) -> Optional[torch.Tensor]:
        if not self._exp_keep or self.budget_weight <= 0:
            return None
        gap = torch.stack(self._exp_keep).mean() - self.current_target()
        dev = gap.abs() if self.two_sided else torch.clamp(gap, min=0.0)
        return self.budget_weight * dev

    def mean_keep(self, *, last_forward: bool = True) -> float:
        d = (
            self.last_per_layer_keep
            if last_forward
            else {li: self._hard_kept[li] / max(1, self._n_tokens[li]) for li in self._n_tokens}
        )
        return sum(d.values()) / len(d) if d else 1.0

    def per_layer_keep(self, *, last_forward: bool = True) -> Dict[int, float]:
        if last_forward:
            return dict(self.last_per_layer_keep)
        return {li: self._hard_kept[li] / max(1, self._n_tokens[li]) for li in self._n_tokens}

    def cumulative_metrics(self) -> Dict[str, float]:
        return {
            "block_skip/cum_keep": self.cum_kept / max(1, self.cum_tokens),
            "block_skip/cum_tokens": float(self.cum_tokens),
        }


class BlockSkipRouter(nn.Module):
    """Per-token run/skip logit, initialised to "run everything"."""

    def __init__(
        self,
        d_model: int,
        *,
        init_bias: float = KEEP_INIT_BIAS,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.w = nn.Linear(d_model, 1, bias=True, device=device, dtype=dtype)
        self.init_bias = float(init_bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.w.weight)
        nn.init.constant_(self.w.bias, self.init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w(x).squeeze(-1).float()


def block_skip_forward(block: nn.Module, h: torch.Tensor, kwargs: Dict[str, Any]) -> torch.Tensor:
    """
    Run ``block`` on ``h`` with per-token skipping.

    :param block: A transformer block carrying ``_bskip`` (holder + layer index) and
        ``_bskip_router``.
    :param h: ``(B, T, d_model)`` block input.
    :param kwargs: The block kwargs; ``block_keep`` is added for the attention layer.
    """
    cfg = block._bskip  # type: ignore[attr-defined]
    holder: BlockSkipHolder = cfg["holder"]
    layer_idx: int = cfg["layer_idx"]
    if not holder.enabled:
        return block(h, **kwargs)
    B, T, _ = h.shape
    # Decode step (single new token with a live cache): generated tokens always run every block --
    # a per-row skip would need per-row cache lengths, and the cost is negligible.
    kvm = getattr(getattr(block, "attention", None), "kv_cache_manager", None)
    if kvm is not None and T == 1:
        return block(h, **kwargs)

    logits = block._bskip_router(h)  # type: ignore[attr-defined]  (B, T)
    p = torch.sigmoid(logits)
    key = (layer_idx, B, T)
    cache = holder._choice_cache
    if block.training and key in cache:
        keep = cache[key]
    else:
        keep = logits.detach() > 0
        if block.training:
            cache[key] = keep
    holder.accumulate(exp_keep=p.mean(), keep=keep, layer_idx=layer_idx)

    if "block_keep" in kwargs and kwargs["block_keep"] is not None:
        keep_attn = kwargs["block_keep"] & keep
    else:
        keep_attn = keep
    out = block(h, **{**kwargs, "block_keep": keep_attn})
    # Straight-through on the residual UPDATE of kept tokens (value: out; grad to p_keep through
    # <dL/dy, out - h>); skipped tokens pass h through untouched.
    p_sel = torch.where(keep, p, 1.0 - p)
    coef = (1.0 + p_sel - p_sel.detach()).to(h.dtype)[:, :, None]
    mixed = h + coef * (out - h)
    return torch.where(keep[:, :, None], mixed, h)


def install_block_skip(
    blocks: nn.ModuleDict, holder: BlockSkipHolder, *, start_layer: int = 0
) -> List[int]:
    """Attach a router to every block at or after ``start_layer``; returns the routed indices."""
    routed: List[int] = []
    for key, block in blocks.items():
        li = int(key)
        if li < start_layer:
            continue
        attn = getattr(block, "attention", None)
        ref = getattr(attn, "w_q", None)
        if ref is None:  # recurrent / other mixers: skipping needs the attention-side key mask
            continue
        block._bskip_router = BlockSkipRouter(  # type: ignore[attr-defined]
            ref.in_features, device=ref.weight.device, dtype=ref.weight.dtype
        )
        block._bskip = {"holder": holder, "layer_idx": li}  # type: ignore[attr-defined]
        routed.append(li)
    holder.routed_layers = routed
    return routed


def reset_block_skip_extras(block: nn.Module) -> None:
    """Re-run the deterministic router init (after a strict=False base load)."""
    block._bskip_router.reset_parameters()  # type: ignore[attr-defined]


def enable_from_config_block(model: Any, block: Dict[str, Any]) -> None:
    """Enable block skipping on a built model as the trainer's ``config.json`` ``block_skip`` block says."""
    model.enable_block_skip(start_layer=int(block.get("start_layer", 0)))
    print(
        f"[block-skip] routing enabled from config.json: routed blocks {model._block_skip['routed']}",
        flush=True,
    )
