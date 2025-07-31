import logging
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from .ring import RingAttentionLoadBalancerType

try:
    import flash_attn  # type: ignore
except ImportError:
    flash_attn = None

try:
    import ring_flash_attn  # type: ignore
except ImportError:
    ring_flash_attn = None

log = logging.getLogger(__name__)


def _flatten_batch_dim(x: torch.Tensor) -> torch.Tensor:
    B, T, *other = x.shape
    return x.view(B * T, *other)


def dispatch_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    if flash_attn is None:
        raise RuntimeError("flash-attn is required!")

    if cu_seqlens is not None:
        if cu_seqlens_q is None:
            cu_seqlens_q = cu_seqlens
        if cu_seqlens_k is None:
            cu_seqlens_k = cu_seqlens
    if max_seqlen is not None:
        if max_seqlen_q is None:
            max_seqlen_q = max_seqlen
        if max_seqlen_k is None:
            max_seqlen_k = max_seqlen

    varlen = all(x is not None for x in (cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k))

    if varlen:
        return flash_attn.flash_attn_varlen_func(
            _flatten_batch_dim(q),
            _flatten_batch_dim(k),
            _flatten_batch_dim(v),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
        )
    else:
        return flash_attn.flash_attn_func(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
        )


def dispatch_flash_attn_qkvpacked(
    qkv: torch.Tensor,
    *,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int],
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    if flash_attn is None:
        raise RuntimeError("flash-attn is required!")

    if cu_seqlens is not None and max_seqlen is not None:
        return flash_attn.flash_attn_varlen_qkvpacked_func(
            _flatten_batch_dim(qkv),
            cu_seqlens,
            max_seqlen,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
        )
    else:
        return flash_attn.flash_attn_qkvpacked_func(
            qkv,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
        )


@torch._dynamo.disable()
def dispatch_flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    rotary_interleaved: bool = True,
) -> torch.Tensor:
    if flash_attn is None:
        raise RuntimeError("flash-attn is required!")
    if k is not None and q.dtype != k.dtype:
        log.warning(f"q.dtype ({q.dtype}) != k.dtype ({k.dtype})")
        k = k.to(q.dtype)
    if q.dtype != k_cache.dtype:
        log.warning(f"q.dtype ({q.dtype}) != k_cache.dtype ({k_cache.dtype})")

    return flash_attn.flash_attn_with_kvcache(
        q,
        k_cache,  # updated in-place if k/v are provided
        v_cache,  # updated in-place if k/v are provided
        k=k,
        v=v,
        cache_seqlens=cache_seqlens,
        cache_leftpad=cache_leftpad,
        block_table=block_table,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        rotary_interleaved=rotary_interleaved,
    )


@torch._dynamo.disable()
def dispatch_ring_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    strategy: "RingAttentionLoadBalancerType",
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    heads_k_stride: Optional[int] = None,
    local_k_slice: Optional[slice] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    if ring_flash_attn is None:
        raise RuntimeError("flash-attn and ring-flash-attn are required!")

    if strategy == RingAttentionLoadBalancerType.zig_zag:
        if any(x is not None for x in (cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)):
            raise RuntimeError(
                f"{strategy} load balancing strategy requires unified QK doc lengths"
            )

        if local_k_slice is not None:
            raise RuntimeError(f"'local_k_slice' is invalid for {strategy} load balancing strategy")

        if cu_seqlens is not None and max_seqlen is not None:
            out = ring_flash_attn.zigzag_ring_flash_attn_varlen_func(
                _flatten_batch_dim(q),
                _flatten_batch_dim(k),
                _flatten_batch_dim(v),
                cu_seqlens,
                max_seqlen,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                group=group,
                window_size=window_size,
            )
        else:
            out = ring_flash_attn.zigzag_ring_flash_attn_func(
                q,
                k,
                v,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                group=group,
                window_size=window_size,
            )
    elif strategy == RingAttentionLoadBalancerType.llama3:
        if any(x is not None for x in (cu_seqlens, max_seqlen)):
            raise RuntimeError(
                f"{strategy} load balancing strategy requires seperate QK doc lengths"
            )

        if (
            cu_seqlens_q is None
            or cu_seqlens_k is None
            or max_seqlen_q is None
            or max_seqlen_k is None
            or heads_k_stride is None
            or local_k_slice is None
        ):
            raise RuntimeError(
                f"{strategy} load balancing strategy is only implemented for 'varlen' variant.\n"
                "The following arguments are required: 'cu_seqlens_(q|k)', 'max_seqlen_(q|k)', "
                "'heads_k_stride', and 'local_k_slice'."
            )

        out = ring_flash_attn.llama3_flash_attn_varlen_func(
            _flatten_batch_dim(q),
            _flatten_batch_dim(k),
            _flatten_batch_dim(v),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            heads_k_stride,
            local_k_slice,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            group=group,
            window_size=window_size,
        )
    else:
        raise NotImplementedError(strategy)

    return out  # type: ignore


@torch._dynamo.disable()
def dispatch_ring_flash_attn_qkvpacked(
    qkv: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    strategy: RingAttentionLoadBalancerType,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int],
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    if ring_flash_attn is None:
        raise RuntimeError("flash-attn and ring-flash-attn are required!")

    if strategy == RingAttentionLoadBalancerType.zig_zag:
        if cu_seqlens is not None and max_seqlen is not None:
            out = ring_flash_attn.zigzag_ring_flash_attn_varlen_qkvpacked_func(
                _flatten_batch_dim(qkv),
                cu_seqlens,
                max_seqlen,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                group=group,
                window_size=window_size,
            )
        else:
            out = ring_flash_attn.zigzag_ring_flash_attn_qkvpacked_func(
                qkv,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                group=group,
                window_size=window_size,
            )
    else:
        raise NotImplementedError(strategy)

    return out  # type: ignore
