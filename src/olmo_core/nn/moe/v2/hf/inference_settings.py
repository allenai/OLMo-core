"""Explicit inference precision diagnostics using existing PyTorch/FLA operations only."""

import torch
from torch.nn import functional as F

_INSTALLED = None


def install(*, linear: str = "native", sdpa: str = "native", recurrent: bool = False) -> dict:
    """Promote accumulation without changing stored parameters or returned dtypes."""
    if linear not in ("native", "float32", "float64") or sdpa not in ("native", "float64"):
        raise ValueError("Unsupported diagnostic precision")
    global _INSTALLED
    settings = dict(linear=linear, sdpa=sdpa, recurrent=recurrent, reduced_bf16_reduction=False)
    if _INSTALLED is not None:
        if settings != _INSTALLED:
            raise RuntimeError("Cannot change inference precision after installing it")
        return settings
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if linear != "native":
        original_linear = F.linear
        dtype = getattr(torch, linear)

        def precise_linear(x, weight, bias=None):
            return original_linear(
                x.to(dtype), weight.to(dtype), None if bias is None else bias.to(dtype)
            ).to(x.dtype)

        F.linear = precise_linear
    if sdpa != "native":
        original_sdpa = F.scaled_dot_product_attention

        def precise_sdpa(q, k, v, attn_mask=None, *args, **kwargs):
            if attn_mask is not None and attn_mask.is_floating_point():
                attn_mask = attn_mask.double()
            return original_sdpa(q.double(), k.double(), v.double(), attn_mask, *args, **kwargs).to(
                q.dtype
            )

        F.scaled_dot_product_attention = precise_sdpa
    if recurrent:
        import fla.ops.kda as kda_ops
        from fla.modules.l2norm import l2norm_fwd

        def recurrent_prefill(q, k, v, **kwargs):
            if kwargs.pop("use_qk_l2norm_in_kernel", False):
                q, _ = l2norm_fwd(q)
                k, _ = l2norm_fwd(k)
            return kda_ops.fused_recurrent_kda(q=q, k=k, v=v, **kwargs)

        kda_ops.chunk_kda = recurrent_prefill
    _INSTALLED = settings
    return settings
