"""
Deliver router budget gradients THROUGH the block output instead of as a separate loss term.

Every router (nested-FFN, KV-route, block-skip) computes a differentiable expectation inside its
block -- inside the block's activation-checkpoint region. If those scalars are summed into the
loss as a separate term, autograd reaches them first in backward (they are direct children of the
loss), which forces the recompute of EVERY checkpointed block at the start of backward; all 36
blocks' recomputed intermediates then stay alive together. On Qwen3-4B at 65k that was 48 GB of
RMSNorm intermediates (memory attribution 2026-09-05: routed peak 68.6 GB vs dense 26.5 GB) and
the OOM of every three-router / all-layer attempt.

:func:`attach_budget_grads` is an identity on the block output whose backward hands each budget
scalar a fixed gradient coefficient. The budget's gradient therefore enters the router graph at
the moment the block's own backward runs, so each block is recomputed once, in order. The
coefficient is the budget's derivative w.r.t. that scalar, linearised with the PREVIOUS forward's
values (``d|cost - target|/d e_i = lambda * sign(cost - target) * (dcost/de_i)``): a one-step lag
on a slowly annealed target. The loss value itself no longer contains the budget term; the
holders still report it for logging.
"""

from __future__ import annotations

from typing import Sequence

import torch

__all__ = ["attach_budget_grads"]


class _AttachGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, out: torch.Tensor, coefs: Sequence[float], *terms: torch.Tensor):  # type: ignore[override]
        ctx.coefs = tuple(float(c) for c in coefs)
        ctx.n = len(terms)
        return out.view_as(out)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        grads = tuple(
            torch.full((), c, dtype=torch.float32, device=grad_out.device) for c in ctx.coefs
        )
        return (grad_out, None) + grads


def attach_budget_grads(
    out: torch.Tensor, terms: Sequence[torch.Tensor], coefs: Sequence[float]
) -> torch.Tensor:
    """
    Return ``out`` unchanged, arranging for ``terms[i]`` to receive gradient ``coefs[i]`` when
    ``out``'s gradient is computed.

    :param out: The block output (any tensor on the main backward path).
    :param terms: Scalar tensors (router expectations) computed inside the block.
    :param coefs: The gradient each term should receive (its budget derivative).
    """
    live = [(t, c) for t, c in zip(terms, coefs) if t is not None and t.requires_grad and c != 0.0]
    if not live or not out.requires_grad:
        return out
    return _AttachGrad.apply(out, [c for _, c in live], *[t for t, _ in live])
