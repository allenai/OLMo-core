from __future__ import annotations

import importlib
import os
from typing import Optional

import torch
import torch.distributed as dist

_EXTENSION_MODULE_NAME = "olmo_core.kernels._tma_ibgda_ep_ext_gpu"
_CUDA_EXTENSION = None
_CUDA_EXTENSION_ERROR: Optional[Exception] = None
_SYMM_INIT_KEY: tuple[str, int] | None = None


class TmaIbgdaKernelUnavailable(RuntimeError):
    """Raised when the OLMo-owned TMA/IBGDA CUDA extension is unavailable."""


def _load_cuda_extension():
    global _CUDA_EXTENSION
    global _CUDA_EXTENSION_ERROR
    if _CUDA_EXTENSION is not None:
        return _CUDA_EXTENSION
    auto_build = os.getenv("OLMO_TMA_IBGDA_EP_AUTO_BUILD", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if auto_build:
        try:
            from .build_tma_ibgda_ep_ext import build_extension

            build_extension(
                inplace=True,
                verbose=False,
                force=False,
            )
        except Exception as e:
            _CUDA_EXTENSION_ERROR = e
            raise TmaIbgdaKernelUnavailable(
                f"Failed to auto-build OLMo TMA/IBGDA EP CUDA extension: {e}"
            ) from e

    try:
        _CUDA_EXTENSION = importlib.import_module(_EXTENSION_MODULE_NAME)
    except Exception as e:
        _CUDA_EXTENSION_ERROR = e
        raise TmaIbgdaKernelUnavailable(
            "OLMo TMA/IBGDA EP CUDA extension is unavailable. This backend is "
            "scaffolded but not built yet. Build it with:\n"
            "  python -m olmo_core.kernels.build_tma_ibgda_ep_ext --inplace\n"
            "Or set OLMO_TMA_IBGDA_EP_AUTO_BUILD=1 to build on import."
        ) from e
    return _CUDA_EXTENSION


def is_available() -> bool:
    try:
        _load_cuda_extension()
    except TmaIbgdaKernelUnavailable:
        return False
    return True


@torch.compiler.disable
def extension_contract() -> dict[str, object]:
    ext = _load_cuda_extension()
    return dict(ext.extension_contract())


@torch.compiler.disable
def plan_peer_window_layout(
    *,
    num_routes: int,
    ep_world_size: int,
    rank_capacity: int,
    hidden_size: int,
    dtype_bytes: int,
) -> dict[str, int]:
    ext = _load_cuda_extension()
    return dict(
        ext.plan_peer_window_layout(
            int(num_routes),
            int(ep_world_size),
            int(rank_capacity),
            int(hidden_size),
            int(dtype_bytes),
        )
    )


@torch.compiler.disable
def enable_peer_access_for_all_visible_devices() -> None:
    ext = _load_cuda_extension()
    ext.enable_peer_access_for_all_visible_devices()


@torch.compiler.disable
def init_symmetric_memory(group: dist.ProcessGroup, device: torch.device) -> None:
    global _SYMM_INIT_KEY
    if not dist.is_available() or not dist.is_initialized():
        raise TmaIbgdaKernelUnavailable(
            "TMA/IBGDA symmetric memory requires torch.distributed to be initialized"
        )
    world_size = dist.get_world_size(group=group)
    group_name = getattr(group, "group_name", None) or f"group:{id(group)}"
    init_key = (str(group_name), world_size)
    if _SYMM_INIT_KEY == init_key:
        return
    if _SYMM_INIT_KEY is not None and _SYMM_INIT_KEY != init_key:
        raise TmaIbgdaKernelUnavailable(
            "TMA/IBGDA symmetric memory is already initialized for "
            f"{_SYMM_INIT_KEY}, cannot reinitialize for {init_key}"
        )

    ext = _load_cuda_extension()
    uid = list(ext.get_unique_id())
    gathered: list[list[int] | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, uid, group=group)
    if any(item is None for item in gathered):
        raise TmaIbgdaKernelUnavailable("failed to gather TMA/IBGDA NVSHMEM unique IDs")
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise TmaIbgdaKernelUnavailable(f"TMA/IBGDA symmetric memory requires CUDA, got {resolved}")
    device_idx = resolved.index if resolved.index is not None else torch.cuda.current_device()
    ext.init(
        [list(item) for item in gathered if item is not None],
        dist.get_rank(group=group),
        world_size,
        device_idx,
    )
    _SYMM_INIT_KEY = init_key


@torch.compiler.disable
def empty_symmetric(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    init_symmetric_memory(group, device)
    ext = _load_cuda_extension()
    return ext.empty(tuple(int(dim) for dim in shape), dtype, device)


@torch.compiler.disable
def barrier_all_on_stream(device: torch.device) -> None:
    ext = _load_cuda_extension()
    ext.barrier_all_on_stream(torch.device(device))


@torch.compiler.disable
def signal_all_and_wait(
    signals: torch.Tensor,
    *,
    generation: int,
    world_size: int,
) -> None:
    ext = _load_cuda_extension()
    ext.signal_all_and_wait(signals, int(generation), int(world_size))


@torch.compiler.disable
def preprocess_routes(
    dst_ranks: torch.Tensor,
    dst_rows: torch.Tensor,
    *,
    ep_world_size: int,
    rank_capacity: int,
    static_route_budget: Optional[int] = None,
    probs: Optional[torch.Tensor] = None,
    nblocks: int = 0,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    ext = _load_cuda_extension()
    return ext.preprocess_routes(
        dst_ranks,
        dst_rows,
        int(ep_world_size),
        int(rank_capacity),
        -1 if static_route_budget is None else int(static_route_budget),
        int(nblocks),
        probs,
    )


@torch.compiler.disable
def preprocess_routes_into(
    dst_ranks: torch.Tensor,
    dst_rows: torch.Tensor,
    *,
    route_records: torch.Tensor,
    routes_per_rank: torch.Tensor,
    rank_offsets: torch.Tensor,
    overflow_by_rank: torch.Tensor,
    route_ordinals: torch.Tensor,
    errors: torch.Tensor,
    ep_world_size: int,
    rank_capacity: int,
    static_route_budget: Optional[int] = None,
    probs: Optional[torch.Tensor] = None,
    nblocks: int = 0,
) -> None:
    ext = _load_cuda_extension()
    ext.preprocess_routes_into(
        dst_ranks,
        dst_rows,
        route_records,
        routes_per_rank,
        rank_offsets,
        overflow_by_rank,
        route_ordinals,
        errors,
        int(ep_world_size),
        int(rank_capacity),
        -1 if static_route_budget is None else int(static_route_budget),
        int(nblocks),
        probs,
    )


@torch.compiler.disable
def route_records_with_probs(
    route_records: torch.Tensor,
    probs: torch.Tensor,
) -> torch.Tensor:
    ext = _load_cuda_extension()
    return ext.route_records_with_probs(route_records, probs)


@torch.compiler.disable
def dispatch_bf16_peer(
    input: torch.Tensor,
    out: torch.Tensor,
    dst_ranks: torch.Tensor,
    dst_rows: torch.Tensor,
    peer_out_ptrs: torch.Tensor,
    probs: Optional[torch.Tensor] = None,
    *,
    nblocks: int = 0,
) -> None:
    ext = _load_cuda_extension()
    ext.dispatch_bf16_peer(input, out, dst_ranks, dst_rows, peer_out_ptrs, probs, nblocks)


@torch.compiler.disable
def combine_bf16_peer(
    expert_out: torch.Tensor,
    out: torch.Tensor,
    src_ranks: torch.Tensor,
    src_rows: torch.Tensor,
    peer_expert_out_ptrs: torch.Tensor,
    probs: Optional[torch.Tensor] = None,
) -> None:
    ext = _load_cuda_extension()
    ext.combine_bf16_peer(expert_out, out, src_ranks, src_rows, peer_expert_out_ptrs, probs)


@torch.compiler.disable
def route_dot_bf16_peer(
    expert_out: torch.Tensor,
    grad_out: torch.Tensor,
    src_ranks: torch.Tensor,
    src_rows: torch.Tensor,
    peer_expert_out_ptrs: torch.Tensor,
    out: torch.Tensor,
) -> None:
    ext = _load_cuda_extension()
    ext.route_dot_bf16_peer(
        expert_out,
        grad_out,
        src_ranks,
        src_rows,
        peer_expert_out_ptrs,
        out,
    )


@torch.compiler.disable
def dispatch_bf16_ibgda(
    input: torch.Tensor,
    out: torch.Tensor,
    dst_ranks: torch.Tensor,
    dst_rows: torch.Tensor,
    probs: Optional[torch.Tensor] = None,
    *,
    nblocks: int = 0,
) -> None:
    ext = _load_cuda_extension()
    ext.dispatch_bf16_ibgda(input, out, dst_ranks, dst_rows, probs, nblocks)


@torch.compiler.disable
def dispatch_bf16_ibgda_records(
    input: torch.Tensor,
    out: torch.Tensor,
    route_records: torch.Tensor,
    *,
    nblocks: int = 0,
) -> None:
    ext = _load_cuda_extension()
    ext.dispatch_bf16_ibgda_records(input, out, route_records, nblocks)


@torch.compiler.disable
def dispatch_bf16_ibgda_records_tma(
    input: torch.Tensor,
    out: torch.Tensor,
    route_records: torch.Tensor,
    *,
    nblocks: int = 0,
) -> None:
    ext = _load_cuda_extension()
    ext.dispatch_bf16_ibgda_records_tma(input, out, route_records, nblocks)


@torch.compiler.disable
def combine_bf16_ibgda(
    expert_out: torch.Tensor,
    out: torch.Tensor,
    src_ranks: torch.Tensor,
    src_rows: torch.Tensor,
    probs: Optional[torch.Tensor] = None,
    *,
    nblocks: int = 0,
) -> None:
    ext = _load_cuda_extension()
    ext.combine_bf16_ibgda(expert_out, out, src_ranks, src_rows, probs, nblocks)


@torch.compiler.disable
def combine_bf16_ibgda_records(
    expert_out: torch.Tensor,
    out: torch.Tensor,
    route_records: torch.Tensor,
    *,
    top_k: int,
    nblocks: int = 0,
) -> None:
    ext = _load_cuda_extension()
    ext.combine_bf16_ibgda_records(expert_out, out, route_records, int(top_k), nblocks)


@torch.compiler.disable
def route_dot_bf16_ibgda(
    expert_out: torch.Tensor,
    grad_out: torch.Tensor,
    src_ranks: torch.Tensor,
    src_rows: torch.Tensor,
    out: torch.Tensor,
) -> None:
    ext = _load_cuda_extension()
    ext.route_dot_bf16_ibgda(expert_out, grad_out, src_ranks, src_rows, out)


@torch.compiler.disable
def route_dot_bf16_ibgda_records(
    expert_out: torch.Tensor,
    grad_out: torch.Tensor,
    route_records: torch.Tensor,
    *,
    top_k: int,
    out: torch.Tensor,
) -> None:
    ext = _load_cuda_extension()
    ext.route_dot_bf16_ibgda_records(expert_out, grad_out, route_records, int(top_k), out)
