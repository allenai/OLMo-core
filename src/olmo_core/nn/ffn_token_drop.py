"""
Random per-token FFN dropping ("drop-CPT"): train a dense model to tolerate the null FFN rung.

The learned FFN router (:mod:`olmo_core.nn.nested_ffn_moe`) was never compute-optimal against dense
SFT. Its losses looked like an *adaptation* problem: all-layer routing collapsed on 30-150-step SFT
runs and recovered only with a long horizon. This module moves that adaptation into continued
pretraining: during training, every token independently skips the FFN of a block (identity
residual, exactly the AdaMoE null expert) with a random probability. There is **no router and no
new parameter** -- the checkpoint stays a plain dense checkpoint that any loader accepts, and the
routed SFT stage (``--variant ffnmoe``) then starts from a model already robust to null FFNs.

Drop schedule (all draws are training-only; eval/inference always runs the full FFN):

- each (forward, batch row) draws a rate ``r ~ U[0, max_rate]``, shared by every layer, so the
  model sees the whole range of budgets rather than one fixed rate;
- each (forward, batch row, layer) is, with probability ``layer_prob``, dropped **whole** for that
  row -- trained routers mostly prune whole layers, so pure per-token noise alone would train a
  different robustness than the router later uses;
- otherwise each token of that row skips the layer's FFN with probability ``r``;
- layers below ``start_layer`` are never dropped (every trained router kept layer 0 dense).

No ``1/(1-p)`` rescaling: a routed null token later gets exactly zero FFN output, and training must
match that.

Kept tokens are gathered and the FFN runs on the compacted set, so the drop is a real FLOP saving.
Draws come from a generator seeded by ``(seed, forward call, rank, layer)`` -- activation
checkpointing re-runs the block in backward and must reproduce the same token split, and
``calls`` advances only in :meth:`FFNTokenDropHolder.begin_forward`, which recompute does not
re-enter (the same contract as the nested-FFN exploration draw).

Enabled via :meth:`~olmo_core.nn.transformer.model.Transformer.enable_ffn_token_drop`.
"""

import logging
from types import MethodType
from typing import List, Optional

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

__all__ = ["FFNTokenDropHolder", "install_ffn_token_drop"]


class FFNTokenDropHolder:
    """
    Shared state of the drop schedule: rates, seed, and the forward-call counter.

    :param max_rate: Upper end of the per-row drop rate ``r ~ U[0, max_rate]``.
    :param layer_prob: Probability that a (row, layer) pair drops the layer's FFN for every token.
    :param seed: Base seed for the draws.
    :param rank: Data-parallel rank, mixed into the seed so ranks draw different patterns.
    """

    def __init__(
        self,
        *,
        max_rate: float,
        layer_prob: float = 0.0,
        seed: int = 0,
        rank: int = 0,
    ):
        if not 0.0 <= max_rate <= 1.0:
            raise ValueError(f"max_rate must be in [0, 1], got {max_rate}")
        if not 0.0 <= layer_prob <= 1.0:
            raise ValueError(f"layer_prob must be in [0, 1], got {layer_prob}")
        self.max_rate = float(max_rate)
        self.layer_prob = float(layer_prob)
        self.seed = int(seed)
        self.rank = int(rank)
        self.calls = 0
        self.enabled = True
        self._row_rates: Optional[torch.Tensor] = None
        self._dropped = 0.0
        self._total = 0.0

    def begin_forward(self, training: bool) -> None:
        """Advance the call counter (training forwards only) and clear the cached row rates."""
        if training:
            self.calls += 1
        self._row_rates = None

    def generator(self, device: torch.device, layer: int) -> torch.Generator:
        """A generator seeded by ``(seed, calls, rank, layer)``; ``layer=-1`` for the row rates."""
        gen = torch.Generator(device=device)
        s = ((self.seed * 1_000_003 + self.calls) * 1_000_003 + self.rank) * 1_000_003 + layer + 1
        gen.manual_seed(s % (2**63 - 1))
        return gen

    def row_rates(self, n_rows: int, device: torch.device) -> torch.Tensor:
        """Per-row drop rates for this forward, shared by every layer."""
        if self._row_rates is None or self._row_rates.shape[0] != n_rows:
            gen = self.generator(device, -1)
            self._row_rates = (
                torch.rand(n_rows, device=device, generator=gen) * self.max_rate
            )
        return self._row_rates

    def record(self, dropped: float, total: float) -> None:
        self._dropped += dropped
        self._total += total

    def pop_metrics(self) -> dict:
        """Realized drop fraction over the tokens seen since the last call, then reset."""
        out = {"ffn_drop/frac": self._dropped / self._total} if self._total else {}
        self._dropped = self._total = 0.0
        return out


def _drop_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    holder: FFNTokenDropHolder = self._fdrop_holder  # type: ignore[attr-defined]
    orig = self._fdrop_orig_forward  # type: ignore[attr-defined]
    if not (self.training and holder.enabled) or (holder.max_rate <= 0 and holder.layer_prob <= 0):
        return orig(x)

    layer: int = self._fdrop_layer_idx  # type: ignore[attr-defined]
    rows = x.reshape(-1, x.shape[-2], x.shape[-1]) if x.dim() >= 2 else x.reshape(1, 1, -1)
    n_rows, seq = rows.shape[0], rows.shape[1]
    gen = holder.generator(x.device, layer)
    rates = holder.row_rates(n_rows, x.device)
    whole = torch.rand(n_rows, device=x.device, generator=gen) < holder.layer_prob
    rates = torch.where(whole, torch.ones_like(rates), rates)
    keep = torch.rand(n_rows, seq, device=x.device, generator=gen) >= rates[:, None]

    flat = rows.reshape(n_rows * seq, rows.shape[-1])
    keep_idx = keep.reshape(-1).nonzero().squeeze(1)
    n_keep = int(keep_idx.numel())
    holder.record(float(flat.shape[0] - n_keep), float(flat.shape[0]))
    out = torch.zeros_like(flat)
    if n_keep == 0:
        # Keep every FFN parameter in the graph with a zero gradient: a rank whose whole layer was
        # dropped must still produce grads for FSDP's reduce-scatter.
        return (out + 0.0 * orig(flat[:1]).sum()).reshape(x.shape)
    out = out.index_copy(0, keep_idx, orig(flat.index_select(0, keep_idx)).to(out.dtype))
    return out.reshape(x.shape)


def install_ffn_token_drop(
    blocks: nn.ModuleDict, holder: FFNTokenDropHolder, *, start_layer: int = 1
) -> List[str]:
    """
    Shadow ``feed_forward.forward`` of every block at or after ``start_layer`` with the random-drop
    version. Adds no parameters and no state-dict keys.

    :param blocks: The model's block dict (keys are layer indices as strings).
    :param holder: Shared drop state.
    :param start_layer: First layer that may drop.

    :returns: The block keys patched.
    """
    patched = []
    for key, block in blocks.items():
        if int(key) < start_layer:
            continue
        ff = getattr(block, "feed_forward", None)
        if ff is None or hasattr(ff, "_fdrop_orig_forward"):
            continue
        if hasattr(ff, "_nffn_orig_forward"):
            raise ValueError(f"block {key}: FFN token drop cannot be combined with nested-FFN routing")
        ff._fdrop_holder = holder
        ff._fdrop_layer_idx = int(key)
        ff._fdrop_orig_forward = ff.forward
        ff.forward = MethodType(_drop_forward, ff)
        patched.append(key)
    return patched
