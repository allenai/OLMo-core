from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast

import torch
import torch.distributed as dist

from olmo_core._nvtx import nvtx
from olmo_core.distributed.utils import get_rank
from olmo_core.kernels.mxfp8_utils import quantize_rows_to_mxfp8

from ...moe.utils import wait_stream_no_compile
from .ep_config import ExpertParallelPath
from .ep_no_sync_buffers import compute_ep_no_sync_rank_capacity
from .ep_no_sync_common import sync_tail_drop_allowed_splits_single_a2a
from .ep_no_sync_rowwise_helpers import (
    accumulate_ep_no_sync_rowwise_metrics,
    build_rowwise_route_maps,
    should_accumulate_ep_no_sync_rowwise_metrics,
)
from .fp8 import shared_experts_forward_rowwise_fp8
from .routed_experts import (
    ExpertActivation,
    requires_host_side_split_sizes,
    use_torch_grouped_mm,
)

if TYPE_CHECKING:
    from olmo_core.nn.ddp.block import OLMoDDPTransformerBlock

try:
    _torch_compile_disable = torch.compiler.disable
except AttributeError:

    def _torch_compile_disable(fn):
        return fn


@dataclass
class _DeepEpV2Runtime:
    # The DeepEP package is an untyped optional dependency (lazily imported), so its module and
    # Buffer handle are typed as Any.
    deep_ep: Any
    buffer: Any
    num_max_tokens_per_rank: int
    hidden: int
    num_topk: int
    num_experts: int
    num_local_experts: int
    expert_alignment: int
    num_sms: int
    num_qps: int
    async_with_compute_stream: bool
    use_fp8_dispatch: bool


@dataclass(frozen=True)
class _DeepEpV2RuntimeKey:
    ep_pg_id: int
    hidden: int
    num_topk: int
    num_experts: int
    num_local_experts: int
    expert_alignment: int
    path: Optional[str]
    num_sms: int
    num_qps: int
    num_allocated_qps: int
    async_mode: bool
    prefer_overlap_with_compute: bool
    allow_hybrid_mode: bool
    allow_multiple_reduction: bool
    use_fp8_dispatch: bool


@lru_cache(maxsize=None)
def _import_deepep_cached(deepep_path: Optional[str]) -> object:
    resolved_path = deepep_path or os.getenv("OLMO_DEEPEP_PATH", "/workspace/DeepEP")
    if resolved_path:
        resolved_path = os.path.abspath(resolved_path)
        if os.path.isdir(resolved_path) and resolved_path not in sys.path:
            sys.path.insert(0, resolved_path)
    try:
        import deep_ep  # type: ignore[import-not-found]
    except (ImportError, OSError) as e:
        # Only treat an actually-missing package (ImportError) or a shared-library load failure
        # (OSError, e.g. a missing .so) as "DeepEP unavailable". Any other exception raised during
        # module init is a real error and should surface rather than be reported as absence.
        raise RuntimeError(
            "Failed to import DeepEP for EP path='deepep_v2'. "
            "Build/install DeepEP first, set OLMO_DEEPEP_PATH, or set "
            "ep.deepep.path. "
            f"Original error: {type(e).__name__}: {e}"
        ) from e
    return deep_ep


@_torch_compile_disable
def _import_deepep(deepep_path: Optional[str]) -> object:
    return _import_deepep_cached(deepep_path)


def is_deepep_available(deepep_path: Optional[str] = None) -> bool:
    try:
        _import_deepep_cached(deepep_path)
    except RuntimeError:
        return False
    return True


def _deep_ep_wait(event: Any, *, async_with_compute_stream: bool) -> None:
    if async_with_compute_stream:
        event.current_stream_wait()


def _deepep_v2_uses_fp8(block: OLMoDDPTransformerBlock) -> bool:
    cfg = getattr(block, "rowwise_fp8", None)
    return cfg is not None and cfg.enabled


def _pack_deepep_mxfp8_scales(scales: torch.Tensor) -> torch.Tensor:
    """Pack four UE8M0 scales into each DeepEP ``sf_pack_t`` entry."""
    if scales.dtype != torch.float8_e8m0fnu:
        raise RuntimeError(
            "DeepEP MXFP8 dispatch expects float8_e8m0fnu scales, " f"got {scales.dtype}"
        )
    if scales.ndim != 2 or scales.shape[1] % 4 != 0:
        raise RuntimeError(
            "DeepEP MXFP8 dispatch requires a rank-2 scale tensor with four-scale "
            f"packing, got shape={tuple(scales.shape)}"
        )
    return scales.contiguous().view(torch.int32)


def _unpack_deepep_mxfp8_scales(packed_scales: torch.Tensor) -> torch.Tensor:
    """Expose opaque DeepEP ``sf_pack_t`` payloads as OLMo UE8M0 scales."""
    if packed_scales.dtype != torch.int32 or packed_scales.ndim != 2:
        raise RuntimeError(
            "DeepEP MXFP8 received scales must be a rank-2 int32 tensor, "
            f"got dtype={packed_scales.dtype} shape={tuple(packed_scales.shape)}"
        )
    return packed_scales.contiguous().view(torch.float8_e8m0fnu)


def _logical_rank2_tensor(
    shape: tuple[int, int],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    # The scaled grouped-MM reads q/scales in forward. This storage-free BF16
    # view supplies the logical shape and the leaf whose gradient DeepEP sends
    # back to the source ranks.
    return torch.empty((), dtype=dtype, device=device).expand(shape)


def _expanded_expert_counts(handle: Any, expert_alignment: int) -> torch.Tensor:
    psum = handle.psum_num_recv_tokens_per_expert
    if psum.ndim != 1:
        raise RuntimeError(
            "DeepEP handle.psum_num_recv_tokens_per_expert must be 1D "
            f"(got shape={tuple(psum.shape)})"
        )
    starts = torch.empty_like(psum)
    starts.fill_(0)
    if psum.numel() > 1:
        previous = psum[:-1]
        if expert_alignment == 1:
            starts[1:] = previous
        else:
            starts[1:] = ((previous + expert_alignment - 1) // expert_alignment) * expert_alignment
    return (psum - starts).to(dtype=torch.int32)


@_torch_compile_disable
def _expanded_weight_grad_to_topk_grad(
    *,
    block: OLMoDDPTransformerBlock,
    runtime: _DeepEpV2Runtime,
    handle: Any,
    expanded_weight_grad: torch.Tensor,
    local_num_tokens: int,
    topk_weights_dtype: torch.dtype,
) -> torch.Tensor:
    metadata = handle.recv_src_metadata
    if metadata.ndim != 2 or metadata.shape[1] < 2 + runtime.num_topk:
        raise RuntimeError(
            "DeepEP expanded recv metadata must be [num_recv_tokens, 2 + top_k], "
            f"got shape={tuple(metadata.shape)} for top_k={runtime.num_topk}"
        )
    if block.ep_pg is None:
        raise RuntimeError("deepep_v2 top-k weight backward requires an EP process group")

    grad_flat = expanded_weight_grad.reshape(-1).to(dtype=torch.float32)
    grad_by_src = torch.zeros(
        (block.ep_world_size, runtime.num_max_tokens_per_rank, runtime.num_topk),
        device=grad_flat.device,
        dtype=torch.float32,
    )
    src_global = metadata[:, 0].to(dtype=torch.long)
    metadata_row = torch.arange(metadata.shape[0], device=metadata.device, dtype=torch.long)
    actual_recv_tokens = handle.psum_num_recv_tokens_per_scaleup_rank[-1].to(dtype=torch.long)
    src_rank = torch.div(
        src_global,
        runtime.num_max_tokens_per_rank,
        rounding_mode="floor",
    )
    src_token = src_global - src_rank * runtime.num_max_tokens_per_rank
    expanded_slots_by_lane = metadata[:, 2 : 2 + runtime.num_topk].to(dtype=torch.long)
    flat_grad_by_src = grad_by_src.reshape(-1)
    grad_flat_numel = grad_flat.numel()
    if grad_flat_numel > 0:
        safe_src_rank = src_rank.clamp(0, block.ep_world_size - 1)
        safe_src_token = src_token.clamp(0, runtime.num_max_tokens_per_rank - 1)
        base_flat_dst = (
            safe_src_rank * runtime.num_max_tokens_per_rank + safe_src_token
        ) * runtime.num_topk
        lane_ids = torch.arange(runtime.num_topk, device=metadata.device, dtype=torch.long)
        flat_dst = base_flat_dst.unsqueeze(1) + lane_ids.unsqueeze(0)
        valid_rows = (
            (metadata_row < actual_recv_tokens)
            & (src_rank >= 0)
            & (src_rank < block.ep_world_size)
            & (src_token >= 0)
            & (src_token < runtime.num_max_tokens_per_rank)
        )
        valid_slots = (expanded_slots_by_lane >= 0) & (expanded_slots_by_lane < grad_flat_numel)
        valid = valid_rows.unsqueeze(1) & valid_slots
        safe_slots = expanded_slots_by_lane.clamp(0, grad_flat_numel - 1)
        slot_grad = grad_flat.index_select(0, safe_slots.reshape(-1))
        slot_grad = slot_grad * valid.reshape(-1).to(dtype=slot_grad.dtype)
        flat_grad_by_src.scatter_add_(0, flat_dst.reshape(-1), slot_grad)

    # Every receiving rank owns a different subset of expanded rows. Sum those
    # per-source contributions so the original token owner can return a normal
    # [local_tokens, top_k] gradient to its router.
    dist.all_reduce(grad_by_src, group=block.ep_pg)
    local_ep_rank = get_rank(block.ep_pg)
    grad_topk = grad_by_src[local_ep_rank, :local_num_tokens, :]
    if grad_topk.dtype != topk_weights_dtype:
        grad_topk = grad_topk.to(dtype=topk_weights_dtype)
    return grad_topk


def _validate_deepep_v2_hidden_size(hidden: int) -> None:
    if hidden % 256 != 0:
        raise RuntimeError(
            "deepep_v2 BF16 combine requires routed hidden size divisible by 256 " f"(got {hidden})"
        )


def _validate_deepep_v2_block(
    block: OLMoDDPTransformerBlock,
    x: torch.Tensor,
) -> None:
    if block.ep.path != ExpertParallelPath.deepep_v2:
        raise RuntimeError(
            "combined_forward_ep_deepep_v2 requires " f"path={ExpertParallelPath.deepep_v2!r}"
        )
    if not x.is_cuda:
        raise RuntimeError("deepep_v2 EP requires CUDA input")
    if x.dtype != torch.bfloat16:
        raise RuntimeError(f"deepep_v2 EP currently supports bf16 only, got {x.dtype}")
    _validate_deepep_v2_hidden_size(x.shape[-1])
    if block.ep_pg is None:
        raise RuntimeError("deepep_v2 EP requires block.ep_pg to be initialized")
    if block.routed_experts is None or block.routed_experts_router is None:
        raise RuntimeError("deepep_v2 EP requires routed experts and a routed router")
    if block.num_local_routed_experts is None:
        raise RuntimeError("deepep_v2 EP requires local routed expert count")
    deepep_cfg = block.ep.deepep
    if deepep_cfg.expert_alignment != 1:
        raise RuntimeError(
            "deepep_v2 model path currently requires deepep.expert_alignment=1. "
            "The expanded dispatch layout may contain aligned padding, while "
            "RoutedExperts.forward consumes packed per-expert rows."
        )
    if block.routed_experts.b_down is not None:
        raise RuntimeError(
            "deepep_v2 EP with deepep.weighting='swiglu' requires bias-free "
            "routed expert down projections."
        )
    if _deepep_v2_uses_fp8(block):
        assert block.rowwise_fp8 is not None
        routed_fp8 = block.routed_experts.rowwise_fp8
        if routed_fp8 is None or not routed_fp8.enabled:
            raise RuntimeError(
                "deepep_v2 FP8 requires rowwise FP8 on both the transformer "
                "block and its routed experts"
            )
        if block.routed_experts.b_up_gate is not None:
            raise RuntimeError("deepep_v2 FP8 routed experts do not support expert biases")
        if block.rowwise_fp8.block_size != 32:
            raise RuntimeError(
                "deepep_v2 FP8 dispatch requires MXFP8 block_size=32, "
                f"got {block.rowwise_fp8.block_size}"
            )
        if (
            block.shared_experts is not None
            and block.shared_experts.activation != ExpertActivation.swiglu
        ):
            raise RuntimeError("deepep_v2 FP8 shared experts currently require swiglu")
    if not use_torch_grouped_mm():
        raise RuntimeError("deepep_v2 EP requires torch.grouped_mm support")
    if requires_host_side_split_sizes():
        raise RuntimeError("deepep_v2 EP does not support host-side split sizes")


@_torch_compile_disable
def _global_num_max_tokens_per_rank(
    block: OLMoDDPTransformerBlock,
    requested_tokens: int,
    device: torch.device,
) -> int:
    # Construct the scalar on-device. torch.tensor([value], device="cuda")
    # stages through pageable CPU memory and synchronizes the CUDA stream.
    requested_tensor = torch.empty((1,), device=device, dtype=torch.long)
    requested_tensor.fill_(requested_tokens)
    dist.all_reduce(requested_tensor, op=dist.ReduceOp.MAX, group=block.ep_pg)
    return int(requested_tensor.item())


def _requested_num_max_tokens_per_rank(
    local_tokens: int,
) -> int:
    if local_tokens <= 0:
        raise ValueError(f"DeepEP requires a positive source-token capacity, got {local_tokens}")
    # DeepEP uses this value as a source-token bound and as the stride in
    # `src_token_global_idx = src_rank * num_max_tokens_per_rank + src_token`.
    # It is not the destination expanded-route capacity; top-k and
    # capacity_factor are already accounted for by `rank_capacity` below.
    return int(local_tokens)


def _warm_deepep_v2_process_group(
    block: OLMoDDPTransformerBlock,
    device: torch.device,
) -> None:
    # DeepEP reuses PyTorch's NCCL communicator by default. For a freshly
    # created ProcessGroupNCCL, backend._comm_ptr() is null until that exact
    # group has launched its first CUDA collective. Passing the null pointer to
    # DeepEP's ncclTeamWorld() segfaults during ElasticBuffer size calculation.
    # Initialize it entirely on-device; no host readback is needed.
    if not int(os.getenv("EP_REUSE_NCCL_COMM", "1")):
        return
    ready = torch.empty((1,), device=device, dtype=torch.int32)
    ready.zero_()
    dist.all_reduce(ready, group=block.ep_pg)
    # ElasticBuffer queries NCCL's device-team metadata immediately after this
    # helper. Finish the one-time communicator initialization before exposing
    # its raw pointer to DeepEP. ElasticBuffer initialization synchronizes
    # anyway; this never runs in forward or recompute.
    torch.cuda.synchronize(device)


def _runtime_key(block: OLMoDDPTransformerBlock, hidden: int, top_k: int) -> _DeepEpV2RuntimeKey:
    assert block.ep_pg is not None
    assert block.routed_experts_router is not None
    assert block.num_local_routed_experts is not None
    deepep_cfg = block.ep.deepep
    return _DeepEpV2RuntimeKey(
        ep_pg_id=id(block.ep_pg),
        hidden=hidden,
        num_topk=top_k,
        num_experts=block.routed_experts_router.num_experts,
        num_local_experts=block.num_local_routed_experts,
        expert_alignment=deepep_cfg.expert_alignment,
        path=deepep_cfg.path,
        num_sms=deepep_cfg.num_sms,
        num_qps=deepep_cfg.num_qps,
        num_allocated_qps=deepep_cfg.num_allocated_qps,
        async_mode=deepep_cfg.async_mode,
        prefer_overlap_with_compute=deepep_cfg.prefer_overlap_with_compute,
        allow_hybrid_mode=deepep_cfg.allow_hybrid_mode,
        allow_multiple_reduction=deepep_cfg.allow_multiple_reduction,
        use_fp8_dispatch=_deepep_v2_uses_fp8(block),
    )


@_torch_compile_disable
def _get_deepep_v2_runtime(
    block: OLMoDDPTransformerBlock,
    *,
    local_tokens: int,
    hidden: int,
    top_k: int,
    device: torch.device,
    static_num_max_tokens_per_rank: Optional[int] = None,
) -> _DeepEpV2Runtime:
    assert block.ep_pg is not None
    assert block.routed_experts_router is not None
    assert block.num_local_routed_experts is not None

    requested_tokens = _requested_num_max_tokens_per_rank(local_tokens)
    runtime_cache = getattr(block, "_deepep_v2_runtime_cache", None)
    if runtime_cache is None:
        runtime_cache = {}
        block._deepep_v2_runtime_cache = runtime_cache
    key = _runtime_key(block, hidden, top_k)
    runtime = runtime_cache.get(key)
    if runtime is not None:
        # The model shares this cache across matching DeepEP blocks. Capacity
        # was fixed when the runtime was created, so the normal fixed-shape hot
        # path only needs this host-side local bound check. Do
        # not repeat the CUDA scalar copy, NCCL MAX, and .item() here: those
        # operations introduced two stream synchronizations per DeepEP call.
        if requested_tokens <= runtime.num_max_tokens_per_rank:
            block._deepep_v2_runtime = runtime
            return runtime
        raise RuntimeError(
            "deepep_v2 EP saw more tokens than its shared ElasticBuffer "
            f"capacity: local_requested={requested_tokens}, "
            f"capacity={runtime.num_max_tokens_per_rank}. Prewarm deepep_v2 "
            "with the largest configured local microbatch token count."
        )

    if key.use_fp8_dispatch:
        assert block.rowwise_fp8 is not None
        # Runtime capability discovery can query the CUDA device. Keep it in
        # this one-time, compile-disabled construction path rather than the
        # per-block forward validation.
        block.rowwise_fp8.assert_runtime_supported()

    if static_num_max_tokens_per_rank is not None:
        # Normal training prewarms from the configured rank microbatch size,
        # which is identical across the EP group. This avoids capacity MAX
        # negotiation and host readback even on the first model forward; the
        # startup-only collective below merely makes the NCCL communicator
        # safe for DeepEP to reuse.
        num_max_tokens_per_rank = int(static_num_max_tokens_per_rank)
        if num_max_tokens_per_rank < requested_tokens:
            raise ValueError(
                "Static DeepEP source-token capacity is smaller than the "
                f"requested shape: capacity={num_max_tokens_per_rank}, "
                f"requested={requested_tokens}"
            )
        _warm_deepep_v2_process_group(block, device)
    else:
        # Fallback for direct callers that bypass model prewarm. ElasticBuffer
        # construction requires a host capacity identical on every EP rank, so
        # this cold path must negotiate and read back the maximum once.
        num_max_tokens_per_rank = _global_num_max_tokens_per_rank(
            block,
            requested_tokens,
            device,
        )

    deepep_cfg = block.ep.deepep
    deep_ep = _import_deepep(deepep_cfg.path)
    num_allocated_qps = max(deepep_cfg.num_allocated_qps, deepep_cfg.num_qps)
    buffer = deep_ep.ElasticBuffer(
        block.ep_pg,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden,
        num_topk=top_k,
        use_fp8_dispatch=key.use_fp8_dispatch,
        deterministic=False,
        allow_hybrid_mode=deepep_cfg.allow_hybrid_mode,
        allow_multiple_reduction=deepep_cfg.allow_multiple_reduction,
        prefer_overlap_with_compute=deepep_cfg.prefer_overlap_with_compute,
        num_allocated_qps=num_allocated_qps,
        explicitly_destroy=False,
    )
    for required_method in ("dispatch_expanded_into", "dispatch_cached_expanded_into"):
        if not hasattr(buffer, required_method):
            raise RuntimeError(
                "deepep_v2 model path requires the modified DeepEP working copy "
                f"with ElasticBuffer.{required_method}. Use ep.deepep.path or "
                "OLMO_DEEPEP_PATH to point at /workspace/DeepEP."
            )
    num_sms = (
        int(deepep_cfg.num_sms)
        if deepep_cfg.num_sms != 0
        else int(buffer.get_theoretical_num_sms(block.routed_experts_router.num_experts, top_k))
    )
    num_qps = (
        int(deepep_cfg.num_qps)
        if deepep_cfg.num_qps != 0
        else int(buffer.get_theoretical_num_qps(num_sms))
    )
    runtime = _DeepEpV2Runtime(
        deep_ep=deep_ep,
        buffer=buffer,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden,
        num_topk=top_k,
        num_experts=block.routed_experts_router.num_experts,
        num_local_experts=block.num_local_routed_experts,
        expert_alignment=deepep_cfg.expert_alignment,
        num_sms=num_sms,
        num_qps=num_qps,
        async_with_compute_stream=deepep_cfg.async_mode,
        use_fp8_dispatch=key.use_fp8_dispatch,
    )
    runtime_cache[key] = runtime
    block._deepep_v2_runtime = runtime
    return runtime


def _routed_experts_need_grad(routed_experts: Optional[Any]) -> bool:
    """Whether the routed experts have trainable state (plain params or optimizer-owned fp8 stores)."""
    if routed_experts is None:
        return False
    if any(p.requires_grad for p in routed_experts.parameters()):
        return True
    named_stores = getattr(routed_experts, "named_fp8_weight_stores", None)
    if named_stores is not None:
        return any(getattr(store, "optimizer_enabled", False) for _, store in named_stores())
    return False


def prewarm_deepep_v2_runtime(
    block: OLMoDDPTransformerBlock,
    *,
    max_local_tokens: int,
    hidden: int,
    top_k: int,
    device: torch.device,
) -> _DeepEpV2Runtime:
    """Create a DeepEP runtime from a configured, globally identical token bound."""
    return _get_deepep_v2_runtime(
        block,
        local_tokens=max_local_tokens,
        hidden=hidden,
        top_k=top_k,
        device=device,
        static_num_max_tokens_per_rank=max_local_tokens,
    )


class _DeepEpV2Autograd(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        source_input: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        block: OLMoDDPTransformerBlock,
        runtime: _DeepEpV2Runtime,
        rank_capacity: int,
        grad_anchor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del grad_anchor  # only present to force this Function into the autograd graph
        recv_topk_weights_out = torch.empty(
            (int(rank_capacity),),
            device=topk_weights.device,
            dtype=topk_weights.dtype,
        )
        if runtime.use_fp8_dispatch:
            assert block.rowwise_fp8 is not None
            source_q, source_scales = quantize_rows_to_mxfp8(
                source_input,
                block_size=int(block.rowwise_fp8.block_size),
                scale_mode=block.rowwise_fp8.scale_mode.value,
            )
            source_packed_scales = _pack_deepep_mxfp8_scales(source_scales)
            recv_q_out = torch.empty(
                (int(rank_capacity), runtime.hidden),
                device=source_input.device,
                dtype=source_q.dtype,
            )
            recv_packed_scales_out = torch.empty(
                (int(rank_capacity), source_packed_scales.shape[1]),
                device=source_input.device,
                dtype=source_packed_scales.dtype,
            )
            (
                recv_payload,
                _recv_topk_idx,
                expanded_topk_weights,
                handle,
                event,
            ) = runtime.buffer.dispatch_expanded_into(
                (source_q, source_packed_scales),
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                recv_x_out=recv_q_out,
                recv_sf_out=recv_packed_scales_out,
                recv_topk_weights_out=recv_topk_weights_out,
                num_experts=runtime.num_experts,
                num_max_tokens_per_rank=runtime.num_max_tokens_per_rank,
                expert_alignment=runtime.expert_alignment,
                num_sms=runtime.num_sms,
                num_qps=runtime.num_qps,
                async_with_compute_stream=runtime.async_with_compute_stream,
                do_cpu_sync=False,
            )
        else:
            recv_x_out = torch.empty(
                (int(rank_capacity), runtime.hidden),
                device=source_input.device,
                dtype=source_input.dtype,
            )
            (
                recv_payload,
                _recv_topk_idx,
                expanded_topk_weights,
                handle,
                event,
            ) = runtime.buffer.dispatch_expanded_into(
                source_input,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                recv_x_out=recv_x_out,
                recv_topk_weights_out=recv_topk_weights_out,
                num_experts=runtime.num_experts,
                num_max_tokens_per_rank=runtime.num_max_tokens_per_rank,
                expert_alignment=runtime.expert_alignment,
                num_sms=runtime.num_sms,
                num_qps=runtime.num_qps,
                async_with_compute_stream=runtime.async_with_compute_stream,
                do_cpu_sync=False,
            )
        _deep_ep_wait(event, async_with_compute_stream=runtime.async_with_compute_stream)
        if expanded_topk_weights is None:
            raise RuntimeError("deepep_v2 expanded dispatch did not return top-k weights")

        batch_size_per_expert = _expanded_expert_counts(handle, runtime.expert_alignment)
        need_grad_topk_weights = ctx.needs_input_grad[2]
        expanded_weights = expanded_topk_weights.reshape(-1, 1)
        if need_grad_topk_weights:
            expanded_weights = expanded_weights.detach().requires_grad_(True)
        assert block.routed_experts is not None
        if runtime.use_fp8_dispatch:
            if not isinstance(recv_payload, tuple) or len(recv_payload) != 2:
                raise RuntimeError("DeepEP FP8 expanded dispatch did not return (q, scales)")
            recv_q, recv_packed_scales = recv_payload
            recv_scales = _unpack_deepep_mxfp8_scales(recv_packed_scales)
            recv_x_for_experts = (
                _logical_rank2_tensor(
                    (int(rank_capacity), runtime.hidden),
                    dtype=source_input.dtype,
                    device=source_input.device,
                )
                .detach()
                .requires_grad_(True)
            )
            with torch.enable_grad():
                unweighted_expert_out = block.routed_experts(
                    recv_x_for_experts,
                    batch_size_per_expert,
                    use_rowwise_fp8=True,
                    rowwise_fp8_input_q=recv_q,
                    rowwise_fp8_input_scales=recv_scales,
                )
                # DeepEP combine is BF16-only. Weighting the bias-free down
                # projection output is mathematically equivalent to weighting
                # its SwiGLU input, while keeping the received payload MXFP8.
                expert_graph_out = unweighted_expert_out * expanded_weights.to(
                    dtype=unweighted_expert_out.dtype
                )
        else:
            if isinstance(recv_payload, tuple):
                raise RuntimeError(
                    "DeepEP BF16 expanded dispatch unexpectedly returned FP8 payload"
                )
            recv_x_for_experts = recv_payload.detach().requires_grad_(True)
            with torch.enable_grad():
                expert_graph_out = block.routed_experts(
                    recv_x_for_experts,
                    batch_size_per_expert,
                    row_weights=expanded_weights,
                )

        combined_x, _combined_topk_weights, event = runtime.buffer.combine(
            expert_graph_out,
            handle=handle,
            num_sms=runtime.num_sms,
            num_qps=runtime.num_qps,
            async_with_compute_stream=runtime.async_with_compute_stream,
        )
        _deep_ep_wait(event, async_with_compute_stream=runtime.async_with_compute_stream)

        ctx.block = block
        ctx.runtime = runtime
        ctx.handle = handle
        ctx.local_num_tokens = int(topk_weights.shape[0])
        ctx.topk_weights_dtype = topk_weights.dtype
        ctx.need_grad_topk_weights = need_grad_topk_weights
        ctx.save_for_backward(recv_x_for_experts, expert_graph_out, expanded_weights)
        return combined_x

    @staticmethod
    def backward(ctx, grad_combined_x: torch.Tensor):  # type: ignore[override]
        runtime: _DeepEpV2Runtime = ctx.runtime
        block: OLMoDDPTransformerBlock = ctx.block
        handle = ctx.handle
        recv_x, expert_graph_out, expanded_weights = ctx.saved_tensors

        grad_weighted_expert_out = torch.empty_like(recv_x)
        (
            _grad_weighted_expert_out,
            _grad_topk_idx,
            _grad_topk_weights,
            _handle,
            event,
        ) = runtime.buffer.dispatch_cached_expanded_into(
            grad_combined_x.contiguous(),
            handle=handle,
            recv_x_out=grad_weighted_expert_out,
            num_sms=runtime.num_sms,
            num_qps=runtime.num_qps,
            async_with_compute_stream=runtime.async_with_compute_stream,
        )
        _deep_ep_wait(event, async_with_compute_stream=runtime.async_with_compute_stream)

        torch.autograd.backward(expert_graph_out, grad_weighted_expert_out)
        if recv_x.grad is None:
            raise RuntimeError("deepep_v2 expert backward did not produce grad for recv_x")
        grad_topk_weights = None
        if ctx.need_grad_topk_weights:
            if expanded_weights.grad is None:
                raise RuntimeError(
                    "deepep_v2 expert backward did not produce grad for expanded top-k weights"
                )
            grad_topk_weights = _expanded_weight_grad_to_topk_grad(
                block=block,
                runtime=runtime,
                handle=handle,
                expanded_weight_grad=expanded_weights.grad,
                local_num_tokens=ctx.local_num_tokens,
                topk_weights_dtype=ctx.topk_weights_dtype,
            )

        combined_grad_x, _combined_grad_topk_weights, event = runtime.buffer.combine(
            recv_x.grad,
            handle=handle,
            num_sms=runtime.num_sms,
            num_qps=runtime.num_qps,
            async_with_compute_stream=runtime.async_with_compute_stream,
        )
        _deep_ep_wait(event, async_with_compute_stream=runtime.async_with_compute_stream)

        # Trailing None is for the (non-differentiable) grad_anchor input.
        return combined_grad_x, None, grad_topk_weights, None, None, None, None


def combined_forward_ep_deepep_v2(
    block: OLMoDDPTransformerBlock,
    x: torch.Tensor,
    *,
    accumulate_routed_aux_loss_metrics: Optional[bool] = None,
    accumulate_router_aux_loss_metrics: Optional[bool] = None,
    loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
    **kwargs,
) -> torch.Tensor:
    """Forward with DeepEP V2 expanded dispatch/combine.

    This is intentionally a narrow first model-path integration of the
    standalone benchmark backend. It skips OLMo symmetric-memory systems and
    uses DeepEP's own ElasticBuffer on the EP process group.
    """
    self = block
    assert self.routed_experts is not None
    assert self.routed_experts_router is not None

    block_inp = x
    del x

    segment_ids = kwargs.pop("segment_ids", None)
    attn_res_out = self._checkpointed_res_norm_attn(block_inp, **kwargs)

    kwargs.pop("max_doc_len", None)
    kwargs.pop("cu_doc_lens", None)
    moe_inp = self._prepare_moe_input(attn_res_out)
    routed_moe_inp = self._prepare_routed_moe_input(moe_inp)
    _validate_deepep_v2_block(self, routed_moe_inp)

    (
        local_x_global_routed_expert_weights,
        local_x_global_routed_expert_indices,
        local_batch_size_per_global_routed_expert,
        routed_expert_router_aux_loss_info,
    ) = self.routed_experts_router(
        moe_inp,
        False,
        loss_div_factor=loss_div_factor,
        segment_ids=segment_ids,
    )

    wait_stream_no_compile(
        this_stream=self.get_dense_stream(),
        other_stream=torch.cuda.current_stream(),
    )

    with torch.cuda.stream(self.get_dense_stream()):
        if self.shared_experts_router:
            (
                local_x_global_shared_expert_weights,
                _,
                _,
                _,
            ) = self.shared_experts_router(
                moe_inp,
                True,
                loss_div_factor=loss_div_factor,
            )
        else:
            local_x_global_shared_expert_weights = None

    routed_in_shape = routed_moe_inp.size()

    mixed_shared_out = None
    if self.shared_experts is not None:
        wait_stream_no_compile(
            this_stream=self.get_dense_stream(),
            other_stream=torch.cuda.current_stream(),
        )
        with torch.cuda.stream(self.get_dense_stream()):
            if _deepep_v2_uses_fp8(self):
                assert self.rowwise_fp8 is not None
                shared_out = shared_experts_forward_rowwise_fp8(
                    self,
                    moe_inp,
                    use_fast_accum=self.rowwise_fp8.use_fast_accum,
                )
            else:
                shared_out = self.shared_experts(moe_inp)
            mixed_shared_out = self._mix_shared_out(
                shared_out,
                local_x_global_shared_expert_weights,
                attn_res_out.shape,
            )

    routed_moe_inp = routed_moe_inp.view(-1, routed_in_shape[-1])
    top_k = self.routed_experts_router.top_k
    routing_map = local_x_global_routed_expert_indices.view(-1, top_k)
    route_weights = local_x_global_routed_expert_weights.view(-1, top_k)
    num_out_tokens = routing_map.numel()
    with torch.no_grad():
        rank_capacity = compute_ep_no_sync_rank_capacity(self, num_out_tokens)
        (
            allowed_splits,
            recv_splits_by_src_local,
            _drop_token_cnt,
            keep_from_src_dest_local,
        ) = cast(
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            sync_tail_drop_allowed_splits_single_a2a(
                self,
                local_batch_size_per_global_routed_expert.to(dtype=torch.long),
                rank_capacity=rank_capacity,
                return_keep_matrix=True,
            ),
        )
        _dst_ranks, dst_rows = build_rowwise_route_maps(
            self,
            routing_map=routing_map,
            allowed_splits=allowed_splits,
            keep_from_src_dest_local=keep_from_src_dest_local,
        )
        route_keep_mask = dst_rows >= 0
        if should_accumulate_ep_no_sync_rowwise_metrics(accumulate_routed_aux_loss_metrics):
            accumulate_ep_no_sync_rowwise_metrics(
                self,
                drop_token_cnt=_drop_token_cnt,
                num_out_tokens=num_out_tokens,
                recv_splits_by_src_local=recv_splits_by_src_local,
                rank_capacity=rank_capacity,
            )

    runtime = _get_deepep_v2_runtime(
        self,
        local_tokens=routed_moe_inp.shape[0],
        hidden=routed_moe_inp.shape[-1],
        top_k=top_k,
        device=routed_moe_inp.device,
    )
    # Reuse rowwise's deterministic tail-drop policy, then present dropped
    # routes to DeepEP as invalid top-k slots. DeepEP ignores negative expert
    # ids during dispatch counting/copy and combine reduction.
    topk_idx = (
        torch.where(
            route_keep_mask,
            routing_map,
            routing_map.new_full(routing_map.shape, -1),
        )
        .to(
            dtype=runtime.deep_ep.topk_idx_t,
        )
        .contiguous()
    )
    topk_weights = (
        torch.where(
            route_keep_mask,
            route_weights,
            torch.zeros_like(route_weights),
        )
        .to(
            dtype=torch.float32,
        )
        .contiguous()
    )

    # The routed-expert weights are reached only through ``self`` inside the Function's inner
    # (enable_grad) graph, not as tensor inputs. When neither tensor input requires grad -- e.g.
    # frozen lower layers and a frozen router with trainable experts -- the Function would get no
    # grad_fn, its backward would never run, and expert weight grads would be silently dropped. Add
    # a differentiable anchor in that case to keep the Function in the graph.
    grad_anchor: Optional[torch.Tensor] = None
    if (
        torch.is_grad_enabled()
        and not (routed_moe_inp.requires_grad or topk_weights.requires_grad)
        and _routed_experts_need_grad(self.routed_experts)
    ):
        grad_anchor = torch.zeros(
            (),
            device=routed_moe_inp.device,
            dtype=routed_moe_inp.dtype,
            requires_grad=True,
        )

    with nvtx.annotate("deepep_v2/routed", color="green"):
        routed_out = _DeepEpV2Autograd.apply(
            routed_moe_inp,
            topk_idx,
            topk_weights,
            self,
            runtime,
            rank_capacity,
            grad_anchor,
        )

    x_moe = routed_out.view(routed_in_shape)
    x_moe = self._restore_routed_moe_output(x_moe)
    wait_stream_no_compile(torch.cuda.current_stream(), self.get_dense_stream())

    mlp_out = self._merge_routed_and_shared(x_moe, mixed_shared_out)
    final_out = self._res_norm_mlp(attn_res_out, mlp_out)
    return self._attach_routed_aux_loss(
        final_out,
        routed_expert_router_aux_loss_info,
        accumulate_metrics=accumulate_router_aux_loss_metrics,
    )
