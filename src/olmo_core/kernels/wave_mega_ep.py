from __future__ import annotations

import importlib
import os
from typing import Optional

import torch

_EXTENSION_MODULE_NAME = "olmo_core.kernels._wave_mega_ep_ext_gpu"
_CUDA_EXTENSION = None
_CUDA_EXTENSION_ERROR: Optional[Exception] = None

__all__ = [
    "rowwise_bf16_mega_moe_forward_config",
    "rowwise_bf16_mega_moe_sm100_tma_umma_contract_debug",
    "rowwise_bf16_mega_moe_sm100_tma_load_contract_debug",
    "rowwise_bf16_mega_moe_sm100_tma_umma_tile_contract_debug",
    "rowwise_bf16_mega_moe_sm100_tma_umma_tile_forward_debug",
    "rowwise_bf16_mega_moe_sm100_tma_umma_tile_forward_b_mn_debug",
    "rowwise_bf16_mega_moe_forward_plan_debug",
    "rowwise_bf16_mega_moe_route_counts_debug",
    "rowwise_bf16_mega_moe_route_pack_debug",
    "rowwise_bf16_mega_moe_route_pack_inputs_debug",
    "rowwise_bf16_mega_moe_peer_route_metadata_debug",
    "rowwise_bf16_mega_moe_peer_window_dispatch_debug",
    "rowwise_bf16_mega_moe_peer_window_combine_debug",
    "rowwise_bf16_mega_moe_grouped_gemm_metadata_debug",
    "rowwise_bf16_mega_moe_grouped_gemm_tile_debug",
    "rowwise_bf16_mega_moe_combine_debug",
    "rowwise_bf16_mega_moe_w1_wmma_debug",
    "rowwise_bf16_mega_moe_forward_debug",
    "rowwise_bf16_mega_moe_local_persistent_forward_debug",
    "rowwise_bf16_mega_moe_local_full_forward_megakernel_debug",
    "rowwise_bf16_mega_moe_standard_scheduler_debug",
    "rowwise_bf16_mega_moe_standard_ep_dispatch_metadata_debug",
    "rowwise_bf16_mega_moe_standard_ep_dispatch_metadata_peer_map_debug",
    "rowwise_bf16_mega_moe_standard_ep_dispatch_pack_inputs_debug",
    "rowwise_bf16_mega_moe_standard_ep_forward_from_dispatch_debug",
    "rowwise_bf16_mega_moe_standard_ep_forward_from_dispatch_umma_debug",
    "rowwise_bf16_mega_moe_local_umma_compute_debug",
    "rowwise_bf16_mega_moe_local_umma_compute",
    "rowwise_bf16_mega_moe_standard_ep_full_forward_megakernel_debug",
    "rowwise_bf16_mega_moe_standard_ep_full_forward_megakernel",
    "rowwise_bf16_mega_moe_standard_ep_workspace_config",
    "rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace",
    "rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace_peer_group",
    "rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace_umma",
    "rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace_peer_group_umma",
    "rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace_collective_world_umma",
    "rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace_collective_world",
    "rowwise_bf16_mega_moe_standard_ep_forward_persistent",
    "rowwise_bf16_mega_moe_forward_persistent",
]


def _load_cuda_extension():
    global _CUDA_EXTENSION
    global _CUDA_EXTENSION_ERROR
    if _CUDA_EXTENSION is not None:
        return _CUDA_EXTENSION

    auto_build = os.getenv("OLMO_WAVE_MEGA_EP_AUTO_BUILD", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if auto_build:
        try:
            from .build_wave_mega_ep_ext import build_extension

            build_extension(
                inplace=True,
                verbose=os.getenv("OLMO_WAVE_MEGA_EP_BUILD_VERBOSE", "").strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                },
                force=os.getenv("OLMO_WAVE_MEGA_EP_BUILD_FORCE", "").strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                },
            )
        except Exception as e:
            _CUDA_EXTENSION_ERROR = e
            raise RuntimeError(f"Failed to auto-build CUDA wave_mega_ep extension: {e}") from e

    try:
        _CUDA_EXTENSION = importlib.import_module(_EXTENSION_MODULE_NAME)
    except Exception as e:
        _CUDA_EXTENSION_ERROR = e
        raise RuntimeError(
            "GPU-side wave_mega_ep extension is unavailable. Build it first with:\n"
            "  python -m olmo_core.kernels.build_wave_mega_ep_ext --inplace\n"
            "Or set OLMO_WAVE_MEGA_EP_AUTO_BUILD=1 to build automatically at import time."
        ) from e

    return _CUDA_EXTENSION


@torch.compiler.disable
def rowwise_bf16_mega_moe_forward_config(
    *,
    num_rows: int,
    top_k: int,
    hidden: int,
    intermediate: int,
    num_local_experts: int,
    num_sms: Optional[int] = None,
) -> dict[str, int]:
    if num_sms is None:
        if not torch.cuda.is_available():
            raise RuntimeError("num_sms is required when CUDA is unavailable")
        num_sms = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
    ext = _load_cuda_extension()
    values = ext.rowwise_bf16_mega_moe_forward_config(
        int(num_rows),
        int(top_k),
        int(hidden),
        int(intermediate),
        int(num_local_experts),
        int(num_sms),
    )
    keys = (
        "block_m",
        "block_n",
        "block_k",
        "store_block_m",
        "num_experts_per_wave",
        "num_expert_waves",
        "num_dispatch_threads",
        "num_non_epilogue_threads",
        "num_epilogue_threads",
        "num_bytes_per_pull",
        "cluster_size",
        "load_block_m",
        "load_block_n",
        "num_max_pool_tokens",
        "num_max_pool_blocks",
        "workspace_bytes",
        "runtime_buffer_bytes",
        "total_symmetric_bytes",
        "num_stages",
        "smem_size",
        "grid_sms",
        "num_ranks",
        "num_experts_per_rank",
        "f1_dispatch_sms",
        "f1_finalize_sms",
        "f1_gemm_sms",
        "f1_expected_tokens_per_expert",
        "f1_gemm_m_tiles_per_expert",
        "f1_gemm_n_tiles",
        "f1_dispatch_route_tasks",
        "f1_finalize_expert_tasks",
        "f1_gemm_tasks",
        "f1_total_tasks",
        "f2_combine_sms",
        "f2_reduce_sms",
        "f2_gemm_sms",
        "f2_expected_tokens_per_expert",
        "f2_gemm_m_tiles_per_expert",
        "f2_gemm_n_tiles",
        "f2_combine_scatter_tasks",
        "f2_combine_reduce_tasks",
        "f2_gemm_tasks",
        "f2_total_tasks",
    )
    return {key: int(value) for key, value in zip(keys, values)}


@torch.compiler.disable
def rowwise_bf16_mega_moe_sm100_tma_umma_contract_debug() -> torch.Tensor:
    ext = _load_cuda_extension()
    return ext.rowwise_bf16_mega_moe_sm100_tma_umma_contract_debug()


@torch.compiler.disable
def rowwise_bf16_mega_moe_sm100_tma_load_contract_debug(
    source: torch.Tensor,
) -> torch.Tensor:
    ext = _load_cuda_extension()
    return ext.rowwise_bf16_mega_moe_sm100_tma_load_contract_debug(source)


@torch.compiler.disable
def rowwise_bf16_mega_moe_sm100_tma_umma_tile_contract_debug(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    ext = _load_cuda_extension()
    return ext.rowwise_bf16_mega_moe_sm100_tma_umma_tile_contract_debug(a, b)


@torch.compiler.disable
def rowwise_bf16_mega_moe_sm100_tma_umma_tile_forward_debug(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    ext = _load_cuda_extension()
    return ext.rowwise_bf16_mega_moe_sm100_tma_umma_tile_forward_debug(a, b)


@torch.compiler.disable
def rowwise_bf16_mega_moe_sm100_tma_umma_tile_forward_b_mn_debug(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    ext = _load_cuda_extension()
    return ext.rowwise_bf16_mega_moe_sm100_tma_umma_tile_forward_b_mn_debug(a, b)


@torch.compiler.disable
def rowwise_bf16_mega_moe_forward_plan_debug(
    *,
    num_rows: int,
    top_k: int,
    hidden: int,
    intermediate: int,
    num_local_experts: int,
    num_sms: Optional[int] = None,
) -> torch.Tensor:
    if num_sms is None:
        if not torch.cuda.is_available():
            raise RuntimeError("num_sms is required when CUDA is unavailable")
        num_sms = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
    ext = _load_cuda_extension()
    return ext.rowwise_bf16_mega_moe_forward_plan_debug(
        int(num_rows),
        int(top_k),
        int(hidden),
        int(intermediate),
        int(num_local_experts),
        int(num_sms),
    )


@torch.compiler.disable
def rowwise_bf16_mega_moe_route_counts_debug(
    route_expert_indices: torch.Tensor,
    *,
    num_local_experts: int,
) -> torch.Tensor:
    ext = _load_cuda_extension()
    return ext.rowwise_bf16_mega_moe_route_counts_debug(
        route_expert_indices,
        int(num_local_experts),
    )


@torch.compiler.disable
def rowwise_bf16_mega_moe_route_pack_debug(
    route_expert_indices: torch.Tensor,
    *,
    num_local_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ext = _load_cuda_extension()
    expert_offsets, packed_token_topk_indices = ext.rowwise_bf16_mega_moe_route_pack_debug(
        route_expert_indices,
        int(num_local_experts),
    )
    return expert_offsets, packed_token_topk_indices


@torch.compiler.disable
def rowwise_bf16_mega_moe_route_pack_inputs_debug(
    source_input: torch.Tensor,
    route_expert_indices: torch.Tensor,
    probs: torch.Tensor,
    *,
    num_local_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ext = _load_cuda_extension()
    expert_offsets, packed_token_topk_indices, packed_input, packed_probs = (
        ext.rowwise_bf16_mega_moe_route_pack_inputs_debug(
            source_input,
            route_expert_indices,
            probs,
            int(num_local_experts),
        )
    )
    return expert_offsets, packed_token_topk_indices, packed_input, packed_probs


@torch.compiler.disable
def rowwise_bf16_mega_moe_peer_route_metadata_debug(
    dst_ranks: torch.Tensor,
    dst_rows: torch.Tensor,
    probs: torch.Tensor,
    *,
    ep_world_size: int,
    rank_capacity: int,
    static_route_budget: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ext = _load_cuda_extension()
    route_records_i32, route_record_probs, routes_per_rank, rank_offsets, overflow_by_rank = (
        ext.rowwise_bf16_mega_moe_peer_route_metadata_debug(
            dst_ranks,
            dst_rows,
            probs,
            int(ep_world_size),
            int(rank_capacity),
            int(static_route_budget),
        )
    )
    return route_records_i32, route_record_probs, routes_per_rank, rank_offsets, overflow_by_rank


@torch.compiler.disable
def rowwise_bf16_mega_moe_peer_window_dispatch_debug(
    source_input: torch.Tensor,
    dst_ranks: torch.Tensor,
    dst_rows: torch.Tensor,
    *,
    ep_world_size: int,
    rank_capacity: int,
) -> torch.Tensor:
    ext = _load_cuda_extension()
    return ext.rowwise_bf16_mega_moe_peer_window_dispatch_debug(
        source_input,
        dst_ranks,
        dst_rows,
        int(ep_world_size),
        int(rank_capacity),
    )


@torch.compiler.disable
def rowwise_bf16_mega_moe_peer_window_combine_debug(
    peer_payload: torch.Tensor,
    src_ranks: torch.Tensor,
    src_rows: torch.Tensor,
    probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    ext = _load_cuda_extension()
    gathered_out, out = ext.rowwise_bf16_mega_moe_peer_window_combine_debug(
        peer_payload,
        src_ranks,
        src_rows,
        probs,
    )
    return gathered_out, out


@torch.compiler.disable
def rowwise_bf16_mega_moe_grouped_gemm_metadata_debug(
    route_expert_indices: torch.Tensor,
    *,
    num_local_experts: int,
    block_m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ext = _load_cuda_extension()
    expert_counts, token_offsets, tile_counts, tile_offsets, num_total_m_tiles = (
        ext.rowwise_bf16_mega_moe_grouped_gemm_metadata_debug(
            route_expert_indices,
            int(num_local_experts),
            int(block_m),
        )
    )
    return expert_counts, token_offsets, tile_counts, tile_offsets, num_total_m_tiles


@torch.compiler.disable
def rowwise_bf16_mega_moe_grouped_gemm_tile_debug(
    route_expert_indices: torch.Tensor,
    *,
    num_local_experts: int,
    block_m: int,
    n_tiles: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ext = _load_cuda_extension()
    expert_counts, token_offsets, tile_counts, tile_offsets, num_total_m_tiles, debug = (
        ext.rowwise_bf16_mega_moe_grouped_gemm_tile_debug(
            route_expert_indices,
            int(num_local_experts),
            int(block_m),
            int(n_tiles),
        )
    )
    return expert_counts, token_offsets, tile_counts, tile_offsets, num_total_m_tiles, debug


@torch.compiler.disable
def rowwise_bf16_mega_moe_combine_debug(
    packed_expert_out: torch.Tensor,
    packed_token_topk_indices: torch.Tensor,
    probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    ext = _load_cuda_extension()
    gathered_out, out = ext.rowwise_bf16_mega_moe_combine_debug(
        packed_expert_out,
        packed_token_topk_indices,
        probs,
    )
    return gathered_out, out


@torch.compiler.disable
def rowwise_bf16_mega_moe_w1_wmma_debug(
    source_input: torch.Tensor,
    route_expert_indices: torch.Tensor,
    up_gate_weight: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    ext = _load_cuda_extension()
    return tuple(
        ext.rowwise_bf16_mega_moe_w1_wmma_debug(
            source_input,
            route_expert_indices,
            up_gate_weight,
        )
    )


@torch.compiler.disable
def rowwise_bf16_mega_moe_forward_debug(
    source_input: torch.Tensor,
    route_expert_indices: torch.Tensor,
    probs: torch.Tensor,
    up_gate_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    ext = _load_cuda_extension()
    return tuple(
        ext.rowwise_bf16_mega_moe_forward_debug(
            source_input,
            route_expert_indices,
            probs,
            up_gate_weight,
            down_weight,
        )
    )


@torch.compiler.disable
def rowwise_bf16_mega_moe_local_persistent_forward_debug(
    source_input: torch.Tensor,
    route_expert_indices: torch.Tensor,
    probs: torch.Tensor,
    up_gate_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    ext = _load_cuda_extension()
    return tuple(
        ext.rowwise_bf16_mega_moe_local_persistent_forward_debug(
            source_input,
            route_expert_indices,
            probs,
            up_gate_weight,
            down_weight,
        )
    )


@torch.compiler.disable
def rowwise_bf16_mega_moe_local_full_forward_megakernel_debug(
    source_input: torch.Tensor,
    route_expert_indices: torch.Tensor,
    probs: torch.Tensor,
    up_gate_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    ext = _load_cuda_extension()
    return tuple(
        ext.rowwise_bf16_mega_moe_local_full_forward_megakernel_debug(
            source_input,
            route_expert_indices,
            probs,
            up_gate_weight,
            down_weight,
        )
    )


@torch.compiler.disable
def rowwise_bf16_mega_moe_standard_scheduler_debug(
    expert_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    ext = _load_cuda_extension()
    workspace, debug = ext.rowwise_bf16_mega_moe_standard_scheduler_debug(expert_counts)
    return workspace, debug


@torch.compiler.disable
def rowwise_bf16_mega_moe_standard_ep_dispatch_metadata_debug(
    route_expert_indices: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    ext = _load_cuda_extension()
    return tuple(
        ext.rowwise_bf16_mega_moe_standard_ep_dispatch_metadata_debug(
            route_expert_indices,
        )
    )


@torch.compiler.disable
def rowwise_bf16_mega_moe_standard_ep_dispatch_metadata_peer_map_debug(
    route_expert_indices: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    ext = _load_cuda_extension()
    return tuple(
        ext.rowwise_bf16_mega_moe_standard_ep_dispatch_metadata_peer_map_debug(
            route_expert_indices,
        )
    )


@torch.compiler.disable
def rowwise_bf16_mega_moe_standard_ep_dispatch_pack_inputs_debug(
    source_input: torch.Tensor,
    route_expert_indices: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    ext = _load_cuda_extension()
    return tuple(
        ext.rowwise_bf16_mega_moe_standard_ep_dispatch_pack_inputs_debug(
            source_input,
            route_expert_indices,
        )
    )


@torch.compiler.disable
def rowwise_bf16_mega_moe_standard_ep_forward_from_dispatch_debug(
    source_input: torch.Tensor,
    route_expert_indices: torch.Tensor,
    probs: torch.Tensor,
    up_gate_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    ext = _load_cuda_extension()
    return tuple(
        ext.rowwise_bf16_mega_moe_standard_ep_forward_from_dispatch_debug(
            source_input,
            route_expert_indices,
            probs,
            up_gate_weight,
            down_weight,
        )
    )


@torch.compiler.disable
def rowwise_bf16_mega_moe_standard_ep_forward_from_dispatch_umma_debug(
    source_input: torch.Tensor,
    route_expert_indices: torch.Tensor,
    probs: torch.Tensor,
    up_gate_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    ext = _load_cuda_extension()
    return tuple(
        ext.rowwise_bf16_mega_moe_standard_ep_forward_from_dispatch_umma_debug(
            source_input,
            route_expert_indices,
            probs,
            up_gate_weight,
            down_weight,
        )
    )


@torch.compiler.disable
def rowwise_bf16_mega_moe_local_umma_compute_debug(
    packed_input: torch.Tensor,
    expert_counts: torch.Tensor,
    up_gate_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    ext = _load_cuda_extension()
    return tuple(
        ext.rowwise_bf16_mega_moe_local_umma_compute_debug(
            packed_input,
            expert_counts,
            up_gate_weight,
            down_weight,
        )
    )


@torch.compiler.disable
def rowwise_bf16_mega_moe_local_umma_compute(
    packed_input: torch.Tensor,
    expert_counts: torch.Tensor,
    up_gate_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    ext = _load_cuda_extension()
    return ext.rowwise_bf16_mega_moe_local_umma_compute(
        packed_input,
        expert_counts,
        up_gate_weight,
        down_weight,
    )


@torch.compiler.disable
def rowwise_bf16_mega_moe_standard_ep_full_forward_megakernel_debug(
    source_input: torch.Tensor,
    route_expert_indices: torch.Tensor,
    probs: torch.Tensor,
    up_gate_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    ext = _load_cuda_extension()
    return tuple(
        ext.rowwise_bf16_mega_moe_standard_ep_full_forward_megakernel_debug(
            source_input,
            route_expert_indices,
            probs,
            up_gate_weight,
            down_weight,
        )
    )


@torch.compiler.disable
def rowwise_bf16_mega_moe_standard_ep_full_forward_megakernel(
    source_input: torch.Tensor,
    route_expert_indices: torch.Tensor,
    probs: torch.Tensor,
    up_gate_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    ext = _load_cuda_extension()
    gathered_out, out = ext.rowwise_bf16_mega_moe_standard_ep_full_forward_megakernel(
        source_input,
        route_expert_indices,
        probs,
        up_gate_weight,
        down_weight,
    )
    return gathered_out, out


@torch.compiler.disable
def rowwise_bf16_mega_moe_standard_ep_workspace_config(
    num_tokens: int,
    hidden: int,
    intermediate: int,
) -> dict[str, int]:
    ext = _load_cuda_extension()
    values = list(
        ext.rowwise_bf16_mega_moe_standard_ep_workspace_config(
            int(num_tokens),
            int(hidden),
            int(intermediate),
        )
    )
    keys = (
        "workspace_bytes",
        "workspace_stride_bytes",
        "num_route_slots",
        "local_packed_capacity",
        "num_ranks",
        "num_total_experts",
        "num_local_experts",
        "top_k",
        "barrier_state_len",
        "packed_values",
        "h_values",
    )
    if len(values) != len(keys):
        raise RuntimeError(
            "rowwise_bf16_mega_moe_standard_ep_workspace_config returned "
            f"{len(values)} values, expected {len(keys)}"
        )
    return {key: int(value) for key, value in zip(keys, values)}


@torch.compiler.disable
def rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace(
    source_input: torch.Tensor,
    gathered_out: torch.Tensor,
    out: torch.Tensor,
    route_expert_indices: torch.Tensor,
    probs: torch.Tensor,
    up_gate_weight: torch.Tensor,
    down_weight: torch.Tensor,
    workspace: torch.Tensor,
    rank_workspace_bases: torch.Tensor,
    global_counts: torch.Tensor,
    global_offsets: torch.Tensor,
    expert_cursors: torch.Tensor,
    packed_route: torch.Tensor,
    route_to_slot: torch.Tensor,
    packed_input: torch.Tensor,
    h: torch.Tensor,
    packed_expert_out: torch.Tensor,
    barrier_state: torch.Tensor,
    caller_rank_idx: int = 0,
    use_peer_workspace_bases: bool = False,
    enable_cross_rank_barriers: bool = False,
    rank_local_expert_owner: bool = False,
) -> None:
    if enable_cross_rank_barriers:
        raise RuntimeError(
            "standard EP cross-rank barriers require a collective-launch wrapper"
        )
    ext = _load_cuda_extension()
    ext.rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace(
        source_input,
        gathered_out,
        out,
        route_expert_indices,
        probs,
        up_gate_weight,
        down_weight,
        workspace,
        rank_workspace_bases,
        global_counts,
        global_offsets,
        expert_cursors,
        packed_route,
        route_to_slot,
        packed_input,
        h,
        packed_expert_out,
        barrier_state,
        int(caller_rank_idx),
        bool(use_peer_workspace_bases),
        False,
        bool(rank_local_expert_owner),
        False,
    )


@torch.compiler.disable
def rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace_peer_group(
    source_input: torch.Tensor,
    gathered_out: torch.Tensor,
    out: torch.Tensor,
    route_expert_indices: torch.Tensor,
    probs: torch.Tensor,
    up_gate_weight: torch.Tensor,
    down_weight: torch.Tensor,
    workspace: torch.Tensor,
    rank_workspace_bases: torch.Tensor,
    global_counts: torch.Tensor,
    global_offsets: torch.Tensor,
    expert_cursors: torch.Tensor,
    packed_route: torch.Tensor,
    route_to_slot: torch.Tensor,
    packed_input: torch.Tensor,
    h: torch.Tensor,
    packed_expert_out: torch.Tensor,
    barrier_state: torch.Tensor,
    caller_rank_idx: int,
) -> None:
    """Launch standard EP rank-local forward over direct peer workspaces.

    All ranks in the EP group must launch this on the same CUDA stream order.
    Cross-rank phase ordering is handled inside the kernel through OLMo-owned
    symmetric workspace barriers; no NVSHMEM world collective launch is used.
    """

    ext = _load_cuda_extension()
    ext.rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace(
        source_input,
        gathered_out,
        out,
        route_expert_indices,
        probs,
        up_gate_weight,
        down_weight,
        workspace,
        rank_workspace_bases,
        global_counts,
        global_offsets,
        expert_cursors,
        packed_route,
        route_to_slot,
        packed_input,
        h,
        packed_expert_out,
        barrier_state,
        int(caller_rank_idx),
        True,
        True,
        True,
        False,
    )


@torch.compiler.disable
def rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace_umma(
    source_input: torch.Tensor,
    gathered_out: torch.Tensor,
    out: torch.Tensor,
    route_expert_indices: torch.Tensor,
    probs: torch.Tensor,
    up_gate_weight: torch.Tensor,
    down_weight: torch.Tensor,
    workspace: torch.Tensor,
    rank_workspace_bases: torch.Tensor,
    global_counts: torch.Tensor,
    global_offsets: torch.Tensor,
    expert_cursors: torch.Tensor,
    packed_route: torch.Tensor,
    route_to_slot: torch.Tensor,
    packed_input: torch.Tensor,
    h: torch.Tensor,
    packed_expert_out: torch.Tensor,
    barrier_state: torch.Tensor,
    w1_up: torch.Tensor,
    w1_gate: torch.Tensor,
    caller_rank_idx: int = 0,
    use_peer_workspace_bases: bool = False,
    enable_cross_rank_barriers: bool = False,
    rank_local_expert_owner: bool = False,
    use_nvshmem_world_collective: bool = False,
) -> None:
    ext = _load_cuda_extension()
    ext.rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace(
        source_input,
        gathered_out,
        out,
        route_expert_indices,
        probs,
        up_gate_weight,
        down_weight,
        workspace,
        rank_workspace_bases,
        global_counts,
        global_offsets,
        expert_cursors,
        packed_route,
        route_to_slot,
        packed_input,
        h,
        packed_expert_out,
        barrier_state,
        int(caller_rank_idx),
        bool(use_peer_workspace_bases),
        bool(enable_cross_rank_barriers),
        bool(rank_local_expert_owner),
        bool(use_nvshmem_world_collective),
        w1_up,
        w1_gate,
        True,
    )


@torch.compiler.disable
def rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace_peer_group_umma(
    source_input: torch.Tensor,
    gathered_out: torch.Tensor,
    out: torch.Tensor,
    route_expert_indices: torch.Tensor,
    probs: torch.Tensor,
    up_gate_weight: torch.Tensor,
    down_weight: torch.Tensor,
    workspace: torch.Tensor,
    rank_workspace_bases: torch.Tensor,
    global_counts: torch.Tensor,
    global_offsets: torch.Tensor,
    expert_cursors: torch.Tensor,
    packed_route: torch.Tensor,
    route_to_slot: torch.Tensor,
    packed_input: torch.Tensor,
    h: torch.Tensor,
    packed_expert_out: torch.Tensor,
    barrier_state: torch.Tensor,
    w1_up: torch.Tensor,
    w1_gate: torch.Tensor,
    caller_rank_idx: int,
) -> None:
    rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace_umma(
        source_input,
        gathered_out,
        out,
        route_expert_indices,
        probs,
        up_gate_weight,
        down_weight,
        workspace,
        rank_workspace_bases,
        global_counts,
        global_offsets,
        expert_cursors,
        packed_route,
        route_to_slot,
        packed_input,
        h,
        packed_expert_out,
        barrier_state,
        w1_up,
        w1_gate,
        caller_rank_idx=caller_rank_idx,
        use_peer_workspace_bases=True,
        enable_cross_rank_barriers=True,
        rank_local_expert_owner=True,
        use_nvshmem_world_collective=False,
    )


@torch.compiler.disable
def rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace_collective_world_umma(
    source_input: torch.Tensor,
    gathered_out: torch.Tensor,
    out: torch.Tensor,
    route_expert_indices: torch.Tensor,
    probs: torch.Tensor,
    up_gate_weight: torch.Tensor,
    down_weight: torch.Tensor,
    workspace: torch.Tensor,
    rank_workspace_bases: torch.Tensor,
    global_counts: torch.Tensor,
    global_offsets: torch.Tensor,
    expert_cursors: torch.Tensor,
    packed_route: torch.Tensor,
    route_to_slot: torch.Tensor,
    packed_input: torch.Tensor,
    h: torch.Tensor,
    packed_expert_out: torch.Tensor,
    barrier_state: torch.Tensor,
    w1_up: torch.Tensor,
    w1_gate: torch.Tensor,
    caller_rank_idx: int,
) -> None:
    rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace_umma(
        source_input,
        gathered_out,
        out,
        route_expert_indices,
        probs,
        up_gate_weight,
        down_weight,
        workspace,
        rank_workspace_bases,
        global_counts,
        global_offsets,
        expert_cursors,
        packed_route,
        route_to_slot,
        packed_input,
        h,
        packed_expert_out,
        barrier_state,
        w1_up,
        w1_gate,
        caller_rank_idx=caller_rank_idx,
        use_peer_workspace_bases=True,
        enable_cross_rank_barriers=True,
        rank_local_expert_owner=True,
        use_nvshmem_world_collective=True,
    )


@torch.compiler.disable
def rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace_collective_world(
    source_input: torch.Tensor,
    gathered_out: torch.Tensor,
    out: torch.Tensor,
    route_expert_indices: torch.Tensor,
    probs: torch.Tensor,
    up_gate_weight: torch.Tensor,
    down_weight: torch.Tensor,
    workspace: torch.Tensor,
    rank_workspace_bases: torch.Tensor,
    global_counts: torch.Tensor,
    global_offsets: torch.Tensor,
    expert_cursors: torch.Tensor,
    packed_route: torch.Tensor,
    route_to_slot: torch.Tensor,
    packed_input: torch.Tensor,
    h: torch.Tensor,
    packed_expert_out: torch.Tensor,
    barrier_state: torch.Tensor,
    caller_rank_idx: int,
) -> None:
    """Launch standard EP rank-local forward through NVSHMEM_TEAM_WORLD.

    This is intentionally narrow: EP must be the NVSHMEM bootstrap world and
    have size 4. Subgroup EP launch needs a separate team/cache path.
    """

    ext = _load_cuda_extension()
    ext.rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace(
        source_input,
        gathered_out,
        out,
        route_expert_indices,
        probs,
        up_gate_weight,
        down_weight,
        workspace,
        rank_workspace_bases,
        global_counts,
        global_offsets,
        expert_cursors,
        packed_route,
        route_to_slot,
        packed_input,
        h,
        packed_expert_out,
        barrier_state,
        int(caller_rank_idx),
        True,
        True,
        True,
        True,
    )


@torch.compiler.disable
def rowwise_bf16_mega_moe_standard_ep_forward_persistent(
    source_input: torch.Tensor,
    gathered_out: torch.Tensor,
    out: torch.Tensor,
    route_expert_indices: torch.Tensor,
    probs: torch.Tensor,
    up_gate_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> None:
    ext = _load_cuda_extension()
    ext.rowwise_bf16_mega_moe_standard_ep_forward_persistent(
        source_input,
        gathered_out,
        out,
        route_expert_indices,
        probs,
        up_gate_weight,
        down_weight,
    )


@torch.compiler.disable
def rowwise_bf16_mega_moe_forward_persistent(
    source_input: torch.Tensor,
    gathered_out: torch.Tensor,
    out: torch.Tensor,
    route_dst_ranks: torch.Tensor,
    route_dst_rows: torch.Tensor,
    route_expert_indices: torch.Tensor,
    probs: torch.Tensor,
    up_gate_weight: torch.Tensor,
    down_weight: torch.Tensor,
    expert_offsets: torch.Tensor,
    group_name: str,
    *,
    route_done_counts: Optional[torch.Tensor] = None,
    symm_probs: Optional[torch.Tensor] = None,
    pre_barrier: bool = True,
    post_barrier: bool = False,
) -> None:
    ext = _load_cuda_extension()
    ext.rowwise_bf16_mega_moe_forward_persistent(
        source_input,
        gathered_out,
        out,
        route_dst_ranks,
        route_dst_rows,
        route_expert_indices,
        probs,
        up_gate_weight,
        down_weight,
        expert_offsets,
        group_name,
        route_done_counts,
        symm_probs,
        pre_barrier,
        post_barrier,
    )
