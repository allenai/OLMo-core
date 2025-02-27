from typing import Optional

import torch
import torch.distributed as dist

try:
    from flash_attn import flash_attn_func as _flash_attn_func
    from flash_attn import flash_attn_qkvpacked_func as _flash_attn_qkvpacked_func
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func
    from flash_attn import (
        flash_attn_varlen_qkvpacked_func as _flash_attn_varlen_qkvpacked_func,
    )
except ImportError:
    _flash_attn_func = None
    _flash_attn_varlen_func = None
    _flash_attn_qkvpacked_func = None
    _flash_attn_varlen_qkvpacked_func = None

try:
    from ring_flash_attn import (
        zigzag_ring_flash_attn_func as _zigzag_ring_flash_attn_func,
    )
    from ring_flash_attn import (
        zigzag_ring_flash_attn_qkvpacked_func as _zigzag_ring_flash_attn_qkvpacked_func,
    )
    from ring_flash_attn import (
        zigzag_ring_flash_attn_varlen_func as _zigzag_ring_flash_attn_varlen_func,
    )
    from ring_flash_attn import (
        zigzag_ring_flash_attn_varlen_qkvpacked_func as _zigzag_ring_flash_attn_varlen_qkvpacked_func,
    )
except ImportError:
    _zigzag_ring_flash_attn_func = None
    _zigzag_ring_flash_attn_varlen_func = None
    _zigzag_ring_flash_attn_qkvpacked_func = None
    _zigzag_ring_flash_attn_varlen_qkvpacked_func = None


def flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    if _flash_attn_func is None:
        raise RuntimeError("flash-attn is required!")
    return _flash_attn_func(
        q, k, v, dropout_p=dropout_p, causal=causal, softmax_scale=softmax_scale
    )


def flash_attn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    *,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    if _flash_attn_varlen_func is None:
        raise RuntimeError("flash-attn is required!")
    return _flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
    )


def flash_attn_qkvpacked(
    qkv: torch.Tensor,
    *,
    dropout_p: float = 0.0,
    causal: bool = False,
) -> torch.Tensor:
    if _flash_attn_qkvpacked_func is None:
        raise RuntimeError("flash-attn is required!")
    return _flash_attn_qkvpacked_func(qkv, dropout_p=dropout_p, causal=causal)


def flash_attn_varlen_qkvpacked(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    *,
    dropout_p: float = 0.0,
    causal: bool = False,
) -> torch.Tensor:
    if _flash_attn_varlen_qkvpacked_func is None:
        raise RuntimeError("flash-attn is required!")
    return _flash_attn_varlen_qkvpacked_func(
        qkv,
        cu_seqlens,
        max_seqlen,
        dropout_p=dropout_p,
        causal=causal,
    )


def zigzag_ring_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    if _zigzag_ring_flash_attn_func is None:
        raise RuntimeError("flash-attn and ring-flash-attn are required!")
    out = _zigzag_ring_flash_attn_func(
        q,
        k,
        v,
        dropout_p=dropout_p,
        causal=causal,
        softmax_scale=softmax_scale,
        group=group,
    )
    return out  # type: ignore


def zigzag_ring_flash_attn_qkvpacked(
    qkv: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    dropout_p: float = 0.0,
    causal: bool = False,
) -> torch.Tensor:
    if _zigzag_ring_flash_attn_qkvpacked_func is None:
        raise RuntimeError("flash-attn and ring-flash-attn are required!")
    out = _zigzag_ring_flash_attn_qkvpacked_func(
        qkv,
        dropout_p=dropout_p,
        causal=causal,
        group=group,
    )
    return out  # type: ignore


@torch._dynamo.disable()
def zigzag_ring_flash_attn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    *,
    group: dist.ProcessGroup,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    if _zigzag_ring_flash_attn_varlen_func is None:
        raise RuntimeError("flash-attn and ring-flash-attn are required!")
    out = _zigzag_ring_flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens,
        max_seqlen,
        dropout_p=dropout_p,
        causal=causal,
        softmax_scale=softmax_scale,
        group=group,
    )
    return out  # type: ignore


@torch._dynamo.disable()
def zigzag_ring_flash_attn_varlen_qkvpacked(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    *,
    group: dist.ProcessGroup,
    dropout_p: float = 0.0,
    causal: bool = False,
) -> torch.Tensor:
    if _zigzag_ring_flash_attn_varlen_qkvpacked_func is None:
        raise RuntimeError("flash-attn and ring-flash-attn are required!")
    out = _zigzag_ring_flash_attn_varlen_qkvpacked_func(
        qkv,
        cu_seqlens,
        max_seqlen,
        dropout_p=dropout_p,
        causal=causal,
        group=group,
    )
    return out  # type: ignore
