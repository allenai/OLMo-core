from __future__ import annotations

import importlib
import os
from typing import Optional
import nvtx

import torch

from .mxfp8_utils import quantize_rows_to_mxfp8, reduce_gathered_rows_from_mxfp8

_EXTENSION_MODULE_NAME = "olmo_core.kernels._symm_mem_vdev2d_ext_gpu"
_CUDA_EXTENSION = None
_CUDA_EXTENSION_ERROR: Optional[Exception] = None


def _load_cuda_extension():
    global _CUDA_EXTENSION
    global _CUDA_EXTENSION_ERROR
    if _CUDA_EXTENSION is not None:
        return _CUDA_EXTENSION

    auto_build = os.getenv("OLMO_SYMM_VDEV2D_AUTO_BUILD", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    build_backend = os.getenv("OLMO_SYMM_VDEV2D_BUILD_BACKEND", "cmake")
    if auto_build:
        try:
            from .build_symm_mem_vdev2d_ext import build_extension

            build_extension(
                inplace=True,
                verbose=False,
                force=False,
                backend=build_backend,
            )
        except Exception as e:
            _CUDA_EXTENSION_ERROR = e
            raise RuntimeError(f"Failed to auto-build CUDA symm_mem_vdev2d extension: {e}") from e

    try:
        _CUDA_EXTENSION = importlib.import_module(_EXTENSION_MODULE_NAME)
    except Exception as e:
        _CUDA_EXTENSION_ERROR = e
        raise RuntimeError(
            "GPU-side symm_mem_vdev2d extension is unavailable. Build it first with:\n"
            "  python -m olmo_core.kernels.build_symm_mem_vdev2d_ext --inplace --backend cmake\n"
            "Or set OLMO_SYMM_VDEV2D_AUTO_BUILD=1 to build automatically at import time."
        ) from e

    return _CUDA_EXTENSION


@torch.compiler.disable
def nvshmem_world_barrier() -> None:
    """Enqueue an NVSHMEM_TEAM_WORLD barrier on the current CUDA stream."""
    ext = _load_cuda_extension()
    ext.olmo_symm_world_barrier()


@torch.compiler.disable
def rowwise_signal_peers_on_stream(
    signals: torch.Tensor,
    signal_row: int,
    generation: int,
    group_name: str,
    *,
    quiet_before_signal: bool = True,
) -> None:
    """Signal every peer in ``group_name`` for one row of a symmetric signal tensor."""
    ext = _load_cuda_extension()
    ext.rowwise_signal_peers_on_stream(
        signals,
        int(signal_row),
        int(generation),
        group_name,
        bool(quiet_before_signal),
    )


@torch.compiler.disable
def rowwise_wait_signal_peers_on_stream(
    signals: torch.Tensor,
    signal_row: int,
    generation: int,
    group_name: str,
) -> None:
    """Wait on the current CUDA stream until every peer has signaled one row."""
    ext = _load_cuda_extension()
    ext.rowwise_wait_signal_peers_on_stream(
        signals,
        int(signal_row),
        int(generation),
        group_name,
    )


@torch.compiler.disable
def olmo_symm_peer_base_ptrs(tensor: torch.Tensor, group_name: str) -> torch.Tensor:
    """Return device int64 peer-visible base pointers for an OLMo symmetric tensor."""
    ext = _load_cuda_extension()
    return ext.olmo_symm_peer_base_ptrs(tensor, group_name)


@torch.compiler.disable
def all_to_all_vdev_2d_nblocks(
    input: torch.Tensor,
    out: torch.Tensor,
    in_splits: torch.Tensor,
    out_splits_offsets: torch.Tensor,
    group_name: str,
    *,
    major_align: int = 1,
    nblocks: int = 0,
) -> None:
    ext = _load_cuda_extension()
    ext.all_to_all_vdev_2d_nblocks(
        input,
        out,
        in_splits,
        out_splits_offsets,
        group_name,
        major_align,
        nblocks,
    )


@torch.compiler.disable
def all_to_all_vdev_2d_offset_nblocks(
    input: torch.Tensor,
    out: torch.Tensor,
    in_splits_offsets: torch.Tensor,
    out_splits_offsets: torch.Tensor,
    group_name: str,
    *,
    nblocks: int = 0,
) -> None:
    ext = _load_cuda_extension()
    ext.all_to_all_vdev_2d_offset_nblocks(
        input,
        out,
        in_splits_offsets,
        out_splits_offsets,
        group_name,
        nblocks,
    )


@torch.compiler.disable
def rowwise_dispatch_put(
    input: torch.Tensor,
    out: torch.Tensor,
    dst_ranks: torch.Tensor,
    dst_rows: torch.Tensor,
    group_name: str,
    *,
    probs: Optional[torch.Tensor] = None,
    nblocks: int = 0,
    pre_barrier: bool = False,
    post_barrier: bool = True,
) -> None:
    ext = _load_cuda_extension()
    ext.rowwise_dispatch_put(
        input,
        out,
        dst_ranks,
        dst_rows,
        probs,
        group_name,
        nblocks,
        pre_barrier,
        post_barrier,
    )


@torch.compiler.disable
def rowwise_build_compact_route_records(
    dst_ranks: torch.Tensor,
    dst_rows: torch.Tensor,
    route_experts: torch.Tensor,
    *,
    num_local_experts: int,
    num_waves: int,
    nblocks: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if dst_ranks.ndim != 2 or dst_rows.ndim != 2 or route_experts.ndim != 2:
        raise ValueError("dst_ranks, dst_rows, and route_experts must be rank-2 [N, K]")
    if tuple(dst_ranks.shape) != tuple(dst_rows.shape) or tuple(dst_ranks.shape) != tuple(route_experts.shape):
        raise ValueError("dst_ranks, dst_rows, and route_experts must have identical shapes")
    ext = _load_cuda_extension()
    num_routes = dst_ranks.numel()
    route_records = torch.empty((num_routes, 4), device=dst_ranks.device, dtype=torch.long)
    wave_counts = torch.empty((int(num_waves),), device=dst_ranks.device, dtype=torch.long)
    wave_fill_counts = torch.empty_like(wave_counts)
    wave_offsets = torch.empty((int(num_waves) + 1,), device=dst_ranks.device, dtype=torch.long)
    ext.rowwise_build_compact_route_records(
        dst_ranks,
        dst_rows,
        route_experts,
        route_records,
        wave_counts,
        wave_fill_counts,
        wave_offsets,
        int(num_local_experts),
        int(num_waves),
        int(nblocks),
    )
    return route_records, wave_offsets


@torch.compiler.disable
def rowwise_dispatch_put_compact(
    input: torch.Tensor,
    out: torch.Tensor,
    route_records: torch.Tensor,
    wave_offsets: torch.Tensor,
    wave_idx: int,
    group_name: str,
    *,
    nblocks: int = 0,
    pre_barrier: bool = False,
    post_barrier: bool = True,
) -> None:
    ext = _load_cuda_extension()
    ext.rowwise_dispatch_put_compact(
        input,
        out,
        route_records,
        wave_offsets,
        int(wave_idx),
        group_name,
        nblocks,
        pre_barrier,
        post_barrier,
    )


@torch.compiler.disable
def rowwise_inverse_route_meta_put_compact(
    inverse_route_meta: torch.Tensor,
    route_records: torch.Tensor,
    wave_offsets: torch.Tensor,
    *,
    src_rank: int,
    group_name: str,
    nblocks: int = 0,
    pre_barrier: bool = False,
    post_barrier: bool = True,
) -> None:
    ext = _load_cuda_extension()
    ext.rowwise_inverse_route_meta_put_compact(
        inverse_route_meta,
        route_records,
        wave_offsets,
        int(src_rank),
        group_name,
        nblocks,
        pre_barrier,
        post_barrier,
    )


@torch.compiler.disable
def rowwise_dispatch_put_scaled(
    input_hp: torch.Tensor,
    out_q: torch.Tensor,
    out_scales: torch.Tensor,
    dst_ranks: torch.Tensor,
    dst_rows: torch.Tensor,
    group_name: str,
    *,
    block_size: int = 32,
    nblocks: int = 0,
    pre_barrier: bool = False,
    post_barrier: bool = True,
    zero_unwritten: bool = False,
) -> None:
    # The rowwise dispatch kernel only writes rows referenced by valid route
    # maps. Keep an opt-in safety init for diagnostics or callers that knowingly
    # include padded rows in downstream math.
    if zero_unwritten or os.getenv("OLMO_ROWWISE_FP8_DISPATCH_INIT_OUT", "0") == "1":
        out_q.zero_()
        out_scales.fill_(1.0)

    qdata, scales = quantize_rows_to_mxfp8(input_hp, block_size=block_size)
    rowwise_dispatch_put(
        qdata,
        out_q,
        dst_ranks,
        dst_rows,
        group_name,
        nblocks=nblocks,
        pre_barrier=pre_barrier,
        post_barrier=False,
    )
    rowwise_dispatch_put(
        scales,
        out_scales,
        dst_ranks,
        dst_rows,
        group_name,
        nblocks=nblocks,
        pre_barrier=False,
        post_barrier=post_barrier,
    )


@torch.compiler.disable
def rowwise_combine_get(
    expert_out: torch.Tensor,
    out: torch.Tensor,
    src_ranks: torch.Tensor,
    src_rows: torch.Tensor,
    group_name: str,
    *,
    probs: Optional[torch.Tensor] = None,
    nblocks: int = 0,
    gathered_out: Optional[torch.Tensor] = None,
    pre_barrier: bool = True,
    post_barrier: bool = False,
) -> None:
    ext = _load_cuda_extension()
    ext.rowwise_combine_get(
        expert_out,
        out,
        src_ranks,
        src_rows,
        probs,
        group_name,
        nblocks,
        gathered_out,
        pre_barrier,
        post_barrier,
    )


@torch.compiler.disable
def rowwise_combine_put(
    expert_out: torch.Tensor,
    gathered_out: torch.Tensor,
    inverse_route_meta: torch.Tensor,
    row_start: torch.Tensor,
    num_rows: torch.Tensor,
    group_name: str,
    *,
    nblocks: int = 0,
    pre_barrier: bool = False,
    post_barrier: bool = True,
) -> None:
    ext = _load_cuda_extension()
    if gathered_out.ndim == 3:
        gathered_flat = gathered_out.view(gathered_out.shape[0] * gathered_out.shape[1], gathered_out.shape[2])
    else:
        gathered_flat = gathered_out
    ext.rowwise_combine_put(
        expert_out,
        gathered_flat,
        inverse_route_meta,
        row_start,
        num_rows,
        group_name,
        nblocks,
        pre_barrier,
        post_barrier,
    )


@torch.compiler.disable
def rowwise_reduce_gathered_routes(
    gathered: torch.Tensor,
    probs: torch.Tensor,
    out: torch.Tensor,
    route_ranks: Optional[torch.Tensor] = None,
) -> None:
    ext = _load_cuda_extension()
    ext.rowwise_reduce_gathered_routes(gathered, probs, out, route_ranks)

@nvtx.annotate("rowwise_combine_get_scaled")
def rowwise_combine_get_scaled(
    expert_out_q: torch.Tensor,
    expert_out_scales: torch.Tensor,
    out: torch.Tensor,
    src_ranks: torch.Tensor,
    src_rows: torch.Tensor,
    group_name: str,
    *,
    probs: Optional[torch.Tensor] = None,
    block_size: int = 32,
    nblocks: int = 0,
    gathered_out: Optional[torch.Tensor] = None,
    gathered_q_out: Optional[torch.Tensor] = None,
    gathered_scales_out: Optional[torch.Tensor] = None,
    pre_barrier: bool = True,
    post_barrier: bool = False,
) -> None:
    if src_ranks.ndim != 2 or src_rows.ndim != 2:
        raise ValueError(
            f"src_ranks/src_rows must be [N,K], got {tuple(src_ranks.shape)} and {tuple(src_rows.shape)}"
        )
    n, k = src_ranks.shape
    if src_rows.shape != src_ranks.shape:
        raise ValueError("src_ranks/src_rows shape mismatch")

    valid_rows = (src_rows >= 0) & (src_rows < expert_out_q.shape[0])
    valid = (src_ranks >= 0) & valid_rows
    safe_ranks = torch.where(valid, src_ranks, torch.full_like(src_ranks, -1))
    safe_rows = torch.where(valid, src_rows, torch.full_like(src_rows, -1))

    flat_ranks = safe_ranks.reshape(-1, 1).contiguous()
    flat_rows = safe_rows.reshape(-1, 1).contiguous()

    expected_q_shape = (n, k, expert_out_q.shape[1])
    if gathered_q_out is not None:
        if tuple(gathered_q_out.shape) != expected_q_shape:
            raise ValueError(
                f"gathered_q_out shape mismatch: expected {expected_q_shape}, got {tuple(gathered_q_out.shape)}"
            )
        if gathered_q_out.dtype != expert_out_q.dtype:
            raise ValueError(
                f"gathered_q_out dtype mismatch: expected {expert_out_q.dtype}, got {gathered_q_out.dtype}"
            )
        if gathered_q_out.device != out.device:
            raise ValueError(
                f"gathered_q_out device mismatch: expected {out.device}, got {gathered_q_out.device}"
            )
        if not gathered_q_out.is_contiguous():
            raise ValueError("gathered_q_out must be contiguous")
        gathered_q_3d = gathered_q_out
        gathered_q = gathered_q_3d.view(n * k, -1)
    else:
        gathered_q = torch.empty(
            (n * k, expert_out_q.shape[1]),
            device=out.device,
            dtype=expert_out_q.dtype,
        )
        gathered_q_3d = gathered_q.view(n, k, -1)

    rowwise_gather_get(
        expert_out_q,
        gathered_q,
        flat_ranks,
        flat_rows,
        group_name,
        nblocks=nblocks,
        pre_barrier=pre_barrier,
        post_barrier=False,
    )

    expected_scales_shape = (n, k, expert_out_scales.shape[1])
    if gathered_scales_out is not None:
        if tuple(gathered_scales_out.shape) != expected_scales_shape:
            raise ValueError(
                "gathered_scales_out shape mismatch: "
                f"expected {expected_scales_shape}, got {tuple(gathered_scales_out.shape)}"
            )
        if gathered_scales_out.dtype != expert_out_scales.dtype:
            raise ValueError(
                "gathered_scales_out dtype mismatch: "
                f"expected {expert_out_scales.dtype}, got {gathered_scales_out.dtype}"
            )
        if gathered_scales_out.device != out.device:
            raise ValueError(
                "gathered_scales_out device mismatch: "
                f"expected {out.device}, got {gathered_scales_out.device}"
            )
        if not gathered_scales_out.is_contiguous():
            raise ValueError("gathered_scales_out must be contiguous")
        gathered_scales_3d = gathered_scales_out
        gathered_scales = gathered_scales_3d.view(n * k, -1)
    else:
        gathered_scales = torch.empty(
            (n * k, expert_out_scales.shape[1]),
            device=out.device,
            dtype=expert_out_scales.dtype,
        )
        gathered_scales_3d = gathered_scales.view(n, k, -1)

    rowwise_gather_get(
        expert_out_scales,
        gathered_scales,
        flat_ranks,
        flat_rows,
        group_name,
        nblocks=nblocks,
        pre_barrier=False,
        post_barrier=post_barrier,
    )

    reduce_gathered_rows_from_mxfp8(
        gathered_q_3d,
        gathered_scales_3d,
        out,
        probs=probs,
        valid_mask=valid,
        block_size=block_size,
        gathered_out=gathered_out,
    )


@torch.compiler.disable
def rowwise_combine_get_fused(
    expert_out: torch.Tensor,
    out: torch.Tensor,
    src_ranks: torch.Tensor,
    src_rows: torch.Tensor,
    group_name: str,
    *,
    probs: Optional[torch.Tensor] = None,
    nblocks: int = 0,
    gathered_out: Optional[torch.Tensor] = None,
    pre_barrier: bool = True,
    post_barrier: bool = False,
) -> None:
    ext = _load_cuda_extension()
    ext.rowwise_combine_get_fused(
        expert_out,
        out,
        src_ranks,
        src_rows,
        probs,
        group_name,
        nblocks,
        gathered_out,
        pre_barrier,
        post_barrier,
    )


@torch.compiler.disable
def rowwise_gather_get(
    expert_out: torch.Tensor,
    out: torch.Tensor,
    src_ranks: torch.Tensor,
    src_rows: torch.Tensor,
    group_name: str,
    *,
    nblocks: int = 0,
    pre_barrier: bool = True,
    post_barrier: bool = False,
) -> None:
    ext = _load_cuda_extension()
    ext.rowwise_gather_get(
        expert_out,
        out,
        src_ranks,
        src_rows,
        group_name,
        nblocks,
        pre_barrier,
        post_barrier,
    )


# Compatibility re-exports. The wave/MegaMoE implementation is built and loaded
# by olmo_core.kernels.wave_mega_ep; keep these names here for existing callers.
from .wave_mega_ep import (  # noqa: E402,F401
    rowwise_bf16_mega_moe_combine_debug,
    rowwise_bf16_mega_moe_forward_config,
    rowwise_bf16_mega_moe_forward_debug,
    rowwise_bf16_mega_moe_forward_persistent,
    rowwise_bf16_mega_moe_forward_plan_debug,
    rowwise_bf16_mega_moe_grouped_gemm_metadata_debug,
    rowwise_bf16_mega_moe_grouped_gemm_tile_debug,
    rowwise_bf16_mega_moe_local_full_forward_megakernel_debug,
    rowwise_bf16_mega_moe_local_persistent_forward_debug,
    rowwise_bf16_mega_moe_local_umma_compute,
    rowwise_bf16_mega_moe_local_umma_compute_debug,
    rowwise_bf16_mega_moe_peer_route_metadata_debug,
    rowwise_bf16_mega_moe_peer_window_combine_debug,
    rowwise_bf16_mega_moe_peer_window_dispatch_debug,
    rowwise_bf16_mega_moe_route_counts_debug,
    rowwise_bf16_mega_moe_route_pack_debug,
    rowwise_bf16_mega_moe_route_pack_inputs_debug,
    rowwise_bf16_mega_moe_sm100_tma_load_contract_debug,
    rowwise_bf16_mega_moe_sm100_tma_umma_contract_debug,
    rowwise_bf16_mega_moe_sm100_tma_umma_tile_contract_debug,
    rowwise_bf16_mega_moe_sm100_tma_umma_tile_forward_b_mn_debug,
    rowwise_bf16_mega_moe_sm100_tma_umma_tile_forward_debug,
    rowwise_bf16_mega_moe_standard_ep_dispatch_metadata_debug,
    rowwise_bf16_mega_moe_standard_ep_dispatch_metadata_peer_map_debug,
    rowwise_bf16_mega_moe_standard_ep_dispatch_pack_inputs_debug,
    rowwise_bf16_mega_moe_standard_ep_forward_from_dispatch_debug,
    rowwise_bf16_mega_moe_standard_ep_forward_from_dispatch_umma_debug,
    rowwise_bf16_mega_moe_standard_ep_forward_persistent,
    rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace,
    rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace_collective_world,
    rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace_collective_world_umma,
    rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace_peer_group,
    rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace_peer_group_umma,
    rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace_umma,
    rowwise_bf16_mega_moe_standard_ep_full_forward_megakernel,
    rowwise_bf16_mega_moe_standard_ep_full_forward_megakernel_debug,
    rowwise_bf16_mega_moe_standard_ep_workspace_config,
    rowwise_bf16_mega_moe_standard_scheduler_debug,
    rowwise_bf16_mega_moe_w1_wmma_debug,
)
