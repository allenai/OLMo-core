"""
Memory-neutral router evaluation on a block INPUT, outside the block's checkpoint region.

Why outside: a router evaluated inside a checkpointed block (reading the block's attention-norm
output and handing a scalar to the budget holder) kept every block's norm intermediates alive
under FSDP2 + full activation checkpointing (KV router: 61 GB peak vs 27 GB with the router graph
removed; FFN router, which stays inside, does not leak -- memory attribution 2026-09-05). Routers
that run *before* the block, on the block input ``h``, keep the region free of router graph.

Why a custom Function: the naive ``linear(rms_norm(h))`` saves an fp32 copy of ``h`` (the norm)
and a bf16 normalised copy (the linear) per layer -- 1 GB per layer at 65k, which is the retention
we are avoiding. :class:`RouterOnInput` saves only ``h`` (already kept as the checkpoint region's
input, so no extra memory) plus a per-token scale, recomputing the normalised input in backward.
The RMS scale is treated as a constant in the backward (its dependence on ``h`` is dropped), which
is immaterial for a 1-logit router trained from scratch.
"""

from __future__ import annotations

from typing import Optional

import torch

__all__ = ["router_logits"]


class RouterOnInput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor], eps: float):  # type: ignore[override]
        with torch.no_grad():
            scale = torch.rsqrt(h.float().pow(2).mean(-1, keepdim=True) + eps).to(h.dtype)
            x = h * scale
            logits = torch.nn.functional.linear(x, w.to(h.dtype), None if b is None else b.to(h.dtype))
        ctx.save_for_backward(h, scale, w)
        ctx.has_bias = b is not None
        return logits.float()

    @staticmethod
    def backward(ctx, g: torch.Tensor):  # type: ignore[override]
        h, scale, w = ctx.saved_tensors
        g = g.to(h.dtype)
        x = h * scale
        gw = g.reshape(-1, g.shape[-1]).t().mm(x.reshape(-1, x.shape[-1])).to(w.dtype)
        gb = g.reshape(-1, g.shape[-1]).sum(0).to(w.dtype) if ctx.has_bias else None
        gx = g @ w.to(h.dtype)
        gh = gx * scale
        return gh, gw, gb, None


def router_logits(h: torch.Tensor, router: torch.nn.Module, eps: float = 1e-6) -> torch.Tensor:
    """
    ``router.w`` applied to the RMS-normalised block input ``h``, saving nothing but ``h`` and a
    per-token scale. Returns fp32 logits of shape ``h.shape[:-1] + (out_features,)``.
    """
    lin = router.w
    return RouterOnInput.apply(h, lin.weight, lin.bias, eps)
