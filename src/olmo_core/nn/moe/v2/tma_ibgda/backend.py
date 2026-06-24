from __future__ import annotations

from dataclasses import dataclass
import os
import socket
import threading
from typing import Any, Optional

import torch
import torch.distributed as dist

from .metadata import TmaIbgdaRouteMetadata, build_tma_ibgda_route_metadata
from .workspace import (
    TMA_IBGDA_COMPLETION_BYTES,
    TMA_IBGDA_DOORBELL_BYTES,
    TMA_IBGDA_ROUTE_RECORD_BYTES,
    TMA_IBGDA_WORKSPACE_ALIGNMENT,
    TmaIbgdaPeerWindowPlan,
    plan_tma_ibgda_peer_windows,
)


class TmaIbgdaBackendUnavailable(RuntimeError):
    """Raised when the OLMo-owned TMA/IBGDA backend cannot run in this process."""


@dataclass(frozen=True)
class TmaIbgdaBackendConfig:
    """Host-side launch/config knobs for the rowwise TMA/IBGDA backend."""

    num_sms_dispatch: Optional[int] = None
    num_sms_combine: Optional[int] = None
    num_sms_preprocess: Optional[int] = None
    static_route_budget: Optional[int] = None
    require_registered_memory: bool = False
    require_intra_node: bool = True
    synchronize_after_dispatch: bool = True
    synchronize_before_combine: bool = False
    use_signal_barrier: bool = True
    use_stream_ordered_barrier: bool = False
    reuse_symmetric_buffers: bool = True
    return_symmetric_dispatch_buffer: bool = False
    use_gpu_route_preprocess: bool = True
    validate_gpu_route_preprocess: bool = True
    use_tma_load_dispatch: bool = True
    write_expert_out_to_symmetric: bool = False


@dataclass(frozen=True)
class TmaIbgdaRoutePreprocess:
    """Device metadata produced by the TMA/IBGDA route-preprocess kernel."""

    route_records: torch.Tensor
    routes_per_rank: torch.Tensor
    rank_offsets: torch.Tensor
    overflow_by_rank: torch.Tensor
    route_ordinals: torch.Tensor
    errors: torch.Tensor


@dataclass(frozen=True)
class TmaIbgdaDispatchHandle:
    """Metadata handle produced by dispatch and consumed by combine/backward."""

    metadata: TmaIbgdaRouteMetadata
    group_name: str
    config: TmaIbgdaBackendConfig
    process_group: Any
    peer_out_ptrs: torch.Tensor
    preprocess: Optional[TmaIbgdaRoutePreprocess] = None
    peer_window_plan: Optional[TmaIbgdaPeerWindowPlan] = None
    dst_ranks_version: Optional[int] = None
    dst_rows_version: Optional[int] = None


@dataclass(frozen=True)
class TmaIbgdaPeerWindowViews:
    """Typed tensor views carved out of one symmetric peer-window allocation."""

    window: torch.Tensor
    route_records: torch.Tensor
    routes_per_rank: torch.Tensor
    rank_offsets: torch.Tensor
    overflow_by_rank: torch.Tensor
    payload: torch.Tensor
    send_doorbells: torch.Tensor
    recv_completions: torch.Tensor


def is_tma_ibgda_backend_available() -> bool:
    """Return whether compiled TMA/IBGDA kernels are available."""

    try:
        from olmo_core.kernels import tma_ibgda_ep
    except Exception:
        return False
    try:
        if not tma_ibgda_ep.is_available():
            return False
        _check_kernel_contract(tma_ibgda_ep)
    except Exception:
        return False
    return True


def _check_bf16_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.dtype != torch.bfloat16:
        raise RuntimeError(f"{name} must be torch.bfloat16 for the TMA/IBGDA BF16 backend")
    if tensor.ndim != 2:
        raise RuntimeError(f"{name} must be rank-2 [rows, hidden], got {tuple(tensor.shape)}")
    if not tensor.is_contiguous():
        raise RuntimeError(f"{name} must be contiguous")


def _check_probs(probs: Optional[torch.Tensor], route_shape: torch.Size) -> None:
    if probs is None:
        return
    if tuple(probs.shape) != tuple(route_shape):
        raise RuntimeError(
            f"probs shape mismatch: got {tuple(probs.shape)}, expected {tuple(route_shape)}"
        )
    if probs.dtype != torch.float32:
        raise RuntimeError("probs must be torch.float32 for the TMA/IBGDA BF16 backend")
    if not probs.is_contiguous():
        raise RuntimeError("probs must be contiguous")


_INTRA_NODE_GROUPS_CHECKED: set[tuple[int, int]] = set()
_MIN_MULTIRANK_IBGDA_BLOCKS = 32
_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}


def _debug_tma_ibgda_enabled(name: str) -> bool:
    return os.environ.get(name, "").lower() in _TRUE_ENV_VALUES


def _debug_tma_ibgda(message: str) -> None:
    if not _debug_tma_ibgda_enabled("OLMO_TMA_IBGDA_DEBUG"):
        return
    try:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    except Exception:
        rank = "?"
    print(f"[tma_ibgda][rank={rank}] {message}", flush=True)


def _debug_tma_ibgda_sync(message: str, device: torch.device) -> None:
    if not _debug_tma_ibgda_enabled("OLMO_TMA_IBGDA_DEBUG_SYNC"):
        return
    _debug_tma_ibgda(f"{message}: cuda sync begin")
    torch.cuda.synchronize(device)
    _debug_tma_ibgda(f"{message}: cuda sync end")


def _tensor_version(tensor: torch.Tensor) -> Optional[int]:
    return getattr(tensor, "_version", None)


class _SymmetricScratchCache:
    """Small process-local cache for NVSHMEM symmetric transport tensors."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buffers: dict[tuple[str, int, torch.dtype, int, str, int], torch.Tensor] = {}

    @staticmethod
    def _device_index(device: torch.device) -> int:
        if device.index is not None:
            return int(device.index)
        return int(torch.cuda.current_device())

    @staticmethod
    def _group_key(process_group: Any) -> str:
        group_name = getattr(process_group, "group_name", None)
        if group_name is not None:
            return str(group_name)
        return f"group:{id(process_group)}"

    def get(
        self,
        *,
        kind: str,
        shape: tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
        process_group: Any,
        world_size: int,
    ) -> torch.Tensor:
        rows, hidden = (int(shape[0]), int(shape[1]))
        key = (
            self._group_key(process_group),
            self._device_index(device),
            dtype,
            hidden,
            kind,
            int(world_size),
        )
        with self._lock:
            tensor = self._buffers.get(key)
            if tensor is None or tensor.shape[0] < rows:
                from olmo_core.kernels import tma_ibgda_ep

                tensor = tma_ibgda_ep.empty_symmetric(
                    (rows, hidden),
                    dtype=dtype,
                    device=device,
                    group=process_group,
                )
                self._buffers[key] = tensor
            return tensor[:rows, :hidden]


_SYMMETRIC_SCRATCH_CACHE = _SymmetricScratchCache()


def _check_peer_window_storage(
    window: torch.Tensor,
    plan: TmaIbgdaPeerWindowPlan,
) -> None:
    if window.dtype != torch.uint8:
        raise RuntimeError(f"TMA/IBGDA peer window must be torch.uint8, got {window.dtype}")
    if not window.is_contiguous():
        raise RuntimeError("TMA/IBGDA peer window must be contiguous")
    if window.numel() < plan.rank_stride_bytes:
        raise RuntimeError(
            "TMA/IBGDA peer window is too small for rank stride: "
            f"got {window.numel()} bytes, need {plan.rank_stride_bytes}"
        )


def _peer_window_section_view(
    window: torch.Tensor,
    *,
    offset: int,
    num_bytes: int,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    name: str,
) -> torch.Tensor:
    dtype_bytes = torch.empty((), dtype=dtype).element_size()
    if offset % dtype_bytes != 0:
        raise RuntimeError(f"TMA/IBGDA peer-window {name} offset is not dtype-aligned")
    if num_bytes % dtype_bytes != 0:
        raise RuntimeError(f"TMA/IBGDA peer-window {name} size is not dtype-aligned")
    expected_numel = 1
    for dim in shape:
        expected_numel *= int(dim)
    if expected_numel * dtype_bytes != num_bytes:
        raise RuntimeError(
            f"TMA/IBGDA peer-window {name} size mismatch: "
            f"{num_bytes} bytes for shape {shape} and dtype {dtype}"
        )
    return window.narrow(0, int(offset), int(num_bytes)).view(dtype).view(shape)


def _payload_view_from_peer_window(
    window: torch.Tensor,
    plan: TmaIbgdaPeerWindowPlan,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    _check_peer_window_storage(window, plan)
    dtype_bytes = torch.empty((), dtype=dtype).element_size()
    expected_bytes = plan.rank_capacity * plan.hidden_size * dtype_bytes
    if plan.payload_window_bytes_per_rank != expected_bytes:
        raise RuntimeError(
            "TMA/IBGDA peer-window payload size mismatch: "
            f"plan has {plan.payload_window_bytes_per_rank}, expected {expected_bytes}"
        )
    return _peer_window_section_view(
        window,
        offset=plan.payload_window_offset,
        num_bytes=plan.payload_window_bytes_per_rank,
        dtype=dtype,
        shape=(plan.rank_capacity, plan.hidden_size),
        name="payload",
    )


def _peer_window_views_from_window(
    window: torch.Tensor,
    plan: TmaIbgdaPeerWindowPlan,
) -> TmaIbgdaPeerWindowViews:
    _check_peer_window_storage(window, plan)
    if plan.route_records_bytes % TMA_IBGDA_ROUTE_RECORD_BYTES != 0:
        raise RuntimeError("TMA/IBGDA peer-window route-record bytes are misaligned")
    num_routes = plan.route_records_bytes // TMA_IBGDA_ROUTE_RECORD_BYTES
    route_record_words = TMA_IBGDA_ROUTE_RECORD_BYTES // torch.empty(
        (),
        dtype=torch.int32,
    ).element_size()
    payload = _payload_view_from_peer_window(window, plan, dtype=plan.dtype)
    return TmaIbgdaPeerWindowViews(
        window=window,
        route_records=_peer_window_section_view(
            window,
            offset=plan.route_records_offset,
            num_bytes=plan.route_records_bytes,
            dtype=torch.int32,
            shape=(num_routes, route_record_words),
            name="route_records",
        ),
        routes_per_rank=_peer_window_section_view(
            window,
            offset=plan.routes_per_rank_offset,
            num_bytes=plan.routes_per_rank_bytes,
            dtype=torch.long,
            shape=(plan.ep_world_size,),
            name="routes_per_rank",
        ),
        rank_offsets=_peer_window_section_view(
            window,
            offset=plan.rank_offsets_offset,
            num_bytes=plan.rank_offsets_bytes,
            dtype=torch.long,
            shape=(plan.ep_world_size + 1,),
            name="rank_offsets",
        ),
        overflow_by_rank=_peer_window_section_view(
            window,
            offset=plan.overflow_by_rank_offset,
            num_bytes=plan.overflow_by_rank_bytes,
            dtype=torch.bool,
            shape=(plan.ep_world_size,),
            name="overflow_by_rank",
        ),
        payload=payload,
        send_doorbells=_peer_window_section_view(
            window,
            offset=plan.send_doorbells_offset,
            num_bytes=plan.send_doorbells_bytes,
            dtype=torch.long,
            shape=(plan.ep_world_size,),
            name="send_doorbells",
        ),
        recv_completions=_peer_window_section_view(
            window,
            offset=plan.recv_completions_offset,
            num_bytes=plan.recv_completions_bytes,
            dtype=torch.long,
            shape=(plan.ep_world_size,),
            name="recv_completions",
        ),
    )


class _PeerWindowScratchCache:
    """Process-local cache for symmetric peer windows backing payload views."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._windows: dict[
            tuple[str, int, torch.dtype, str, int, int, int, int, int, int],
            torch.Tensor,
        ] = {}

    @staticmethod
    def _device_index(device: torch.device) -> int:
        if device.index is not None:
            return int(device.index)
        return int(torch.cuda.current_device())

    @staticmethod
    def _group_key(process_group: Any) -> str:
        group_name = getattr(process_group, "group_name", None)
        if group_name is not None:
            return str(group_name)
        return f"group:{id(process_group)}"

    def get_views(
        self,
        *,
        plan: TmaIbgdaPeerWindowPlan,
        device: torch.device,
        process_group: Any,
        world_size: int,
        kind: str,
    ) -> TmaIbgdaPeerWindowViews:
        key = (
            self._group_key(process_group),
            self._device_index(device),
            plan.dtype,
            kind,
            int(world_size),
            int(plan.rank_stride_bytes),
            int(plan.payload_window_offset),
            int(plan.payload_window_bytes_per_rank),
            int(plan.rank_capacity),
            int(plan.hidden_size),
        )
        with self._lock:
            window = self._windows.get(key)
            if window is None:
                from olmo_core.kernels import tma_ibgda_ep

                window = tma_ibgda_ep.empty_symmetric(
                    (int(plan.rank_stride_bytes),),
                    dtype=torch.uint8,
                    device=device,
                    group=process_group,
                )
                self._windows[key] = window
            return _peer_window_views_from_window(window, plan)

    def get_payload(
        self,
        *,
        plan: TmaIbgdaPeerWindowPlan,
        dtype: torch.dtype,
        device: torch.device,
        process_group: Any,
        world_size: int,
        kind: str,
    ) -> torch.Tensor:
        if plan.dtype != dtype:
            raise RuntimeError(
                f"TMA/IBGDA peer-window dtype mismatch: plan has {plan.dtype}, got {dtype}"
            )
        return self.get_views(
            plan=plan,
            device=device,
            process_group=process_group,
            world_size=world_size,
            kind=kind,
        ).payload


_PEER_WINDOW_SCRATCH_CACHE = _PeerWindowScratchCache()


class _SignalScratchCache:
    """Process-local cache for reusable NVSHMEM readiness signal slots."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._signals: dict[tuple[str, int, str, int], torch.Tensor] = {}
        self._generations: dict[tuple[str, int, str, int], int] = {}

    @staticmethod
    def _device_index(device: torch.device) -> int:
        if device.index is not None:
            return int(device.index)
        return int(torch.cuda.current_device())

    @staticmethod
    def _group_key(process_group: Any) -> str:
        group_name = getattr(process_group, "group_name", None)
        if group_name is not None:
            return str(group_name)
        return f"group:{id(process_group)}"

    def next(
        self,
        *,
        kind: str,
        device: torch.device,
        process_group: Any,
        world_size: int,
    ) -> tuple[torch.Tensor, int]:
        key = (
            self._group_key(process_group),
            self._device_index(device),
            kind,
            int(world_size),
        )
        with self._lock:
            signal = self._signals.get(key)
            if signal is None:
                from olmo_core.kernels import tma_ibgda_ep

                signal = tma_ibgda_ep.empty_symmetric(
                    (int(world_size), 1),
                    dtype=torch.long,
                    device=device,
                    group=process_group,
                ).view(-1)
                signal.zero_()
                self._signals[key] = signal
                self._generations[key] = 0
            generation = self._generations.get(key, 0) + 1
            self._generations[key] = generation
            return signal, generation


_SIGNAL_SCRATCH_CACHE = _SignalScratchCache()


def _check_route_tensors(dst_ranks: torch.Tensor, dst_rows: torch.Tensor) -> None:
    if dst_ranks.device != dst_rows.device:
        raise RuntimeError("route maps must be on the same device")
    if dst_ranks.device.type != "cuda":
        raise RuntimeError("route maps must be CUDA tensors")
    if dst_ranks.dtype != torch.long or dst_rows.dtype != torch.long:
        raise RuntimeError("route maps must be torch.long")
    if dst_ranks.ndim != 2 or dst_rows.ndim != 2:
        raise RuntimeError(
            f"route maps must be rank-2 [rows, top_k], got {tuple(dst_ranks.shape)} and {tuple(dst_rows.shape)}"
        )
    if tuple(dst_ranks.shape) != tuple(dst_rows.shape):
        raise RuntimeError("route map shape mismatch")
    if not dst_ranks.is_contiguous() or not dst_rows.is_contiguous():
        raise RuntimeError("route maps must be contiguous")


def _require_kernel_module():
    try:
        from olmo_core.kernels import tma_ibgda_ep
    except Exception as e:
        raise TmaIbgdaBackendUnavailable(
            "OLMo TMA/IBGDA EP kernels could not be imported. Build the standalone "
            "_tma_ibgda_ep_ext_gpu target with:\n"
            "  python -m olmo_core.kernels.build_tma_ibgda_ep_ext --inplace"
        ) from e
    if not tma_ibgda_ep.is_available():
        raise TmaIbgdaBackendUnavailable(
            "OLMo TMA/IBGDA EP kernels are unavailable. Build the standalone "
            "_tma_ibgda_ep_ext_gpu target with:\n"
            "  python -m olmo_core.kernels.build_tma_ibgda_ep_ext --inplace"
        )
    _check_kernel_contract(tma_ibgda_ep)
    return tma_ibgda_ep


def _check_kernel_contract(tma_ibgda_ep: Any) -> None:
    if not hasattr(tma_ibgda_ep, "extension_contract"):
        raise TmaIbgdaBackendUnavailable(
            "OLMo TMA/IBGDA EP kernels do not expose extension_contract. "
            "Rebuild the standalone _tma_ibgda_ep_ext_gpu target."
        )
    try:
        contract = dict(tma_ibgda_ep.extension_contract())
    except Exception as e:
        raise TmaIbgdaBackendUnavailable(
            "OLMo TMA/IBGDA EP kernels failed the extension_contract probe. "
            "Rebuild the standalone _tma_ibgda_ep_ext_gpu target."
        ) from e

    expected = {
        "extension_module": "_tma_ibgda_ep_ext_gpu",
        "route_record_bytes": TMA_IBGDA_ROUTE_RECORD_BYTES,
        "route_record_words": TMA_IBGDA_ROUTE_RECORD_BYTES // 4,
        "route_flag_valid": 1,
        "workspace_alignment": TMA_IBGDA_WORKSPACE_ALIGNMENT,
        "doorbell_bytes": TMA_IBGDA_DOORBELL_BYTES,
        "completion_bytes": TMA_IBGDA_COMPLETION_BYTES,
        "bf16_only": True,
        "has_gpu_route_preprocess": True,
        "has_ibgda_dispatch": True,
        "has_tma_load_dispatch": True,
        "has_ibgda_combine": True,
        "has_route_dot_backward": True,
        "has_peer_window_layout_planner": True,
    }
    size_mismatches = []
    for key in ("peer_window_layout_bytes", "kernel_launch_config_bytes"):
        if not isinstance(contract.get(key), int) or contract[key] <= 0:
            size_mismatches.append(
                f"{key}: expected positive int, got {contract.get(key)!r}"
            )
    mismatches = [
        f"{key}: expected {expected_value!r}, got {contract.get(key)!r}"
        for key, expected_value in expected.items()
        if contract.get(key) != expected_value
    ] + size_mismatches
    if mismatches:
        raise TmaIbgdaBackendUnavailable(
            "OLMo TMA/IBGDA EP extension contract mismatch. "
            "Rebuild the standalone _tma_ibgda_ep_ext_gpu target. "
            + "; ".join(mismatches)
        )


def _resolve_world(
    *,
    ep_world_size: int,
    process_group: Any,
) -> tuple[int, int]:
    if ep_world_size <= 0:
        raise RuntimeError(f"ep_world_size must be > 0, got {ep_world_size}")
    if ep_world_size == 1 and (not dist.is_available() or not dist.is_initialized()):
        return 0, 1
    if not dist.is_available() or not dist.is_initialized():
        raise TmaIbgdaBackendUnavailable(
            "TMA/IBGDA peer-visible transport requires torch.distributed to be initialized "
            "when ep_world_size > 1"
        )
    world_size = dist.get_world_size(group=process_group)
    if world_size != ep_world_size:
        raise RuntimeError(
            f"ep_world_size mismatch for TMA/IBGDA process group: got {ep_world_size}, group has {world_size}"
        )
    return dist.get_rank(group=process_group), world_size


def _barrier(process_group: Any, device: torch.device) -> None:
    if not dist.is_available() or not dist.is_initialized():
        return
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    try:
        dist.barrier(group=process_group, device_ids=[device_index])
    except TypeError:
        dist.barrier(group=process_group)


def _transport_barrier(
    process_group: Any,
    device: torch.device,
    *,
    world_size: int,
    config: TmaIbgdaBackendConfig,
    signal_kind: str,
) -> None:
    if world_size <= 1:
        return
    if config.use_signal_barrier:
        tma_ibgda_ep = _require_kernel_module()
        if not hasattr(tma_ibgda_ep, "signal_all_and_wait"):
            raise TmaIbgdaBackendUnavailable(
                "OLMo TMA/IBGDA EP kernels do not expose signal_all_and_wait. "
                "Rebuild the standalone _tma_ibgda_ep_ext_gpu target."
            )
        signals, generation = _SIGNAL_SCRATCH_CACHE.next(
            kind=signal_kind,
            device=device,
            process_group=process_group,
            world_size=world_size,
        )
        tma_ibgda_ep.signal_all_and_wait(
            signals,
            generation=generation,
            world_size=world_size,
        )
        return
    if config.use_stream_ordered_barrier:
        tma_ibgda_ep = _require_kernel_module()
        if not hasattr(tma_ibgda_ep, "barrier_all_on_stream"):
            raise TmaIbgdaBackendUnavailable(
                "OLMo TMA/IBGDA EP kernels do not expose barrier_all_on_stream. "
                "Rebuild the standalone _tma_ibgda_ep_ext_gpu target."
            )
        tma_ibgda_ep.barrier_all_on_stream(device)
        return
    torch.cuda.synchronize(device)
    _barrier(process_group, device)


def _all_gather_cuda_i64(value: int, world_size: int, device: torch.device, process_group: Any) -> torch.Tensor:
    local = torch.tensor([value], dtype=torch.long, device=device)
    if world_size == 1:
        return local
    gathered = torch.empty((world_size,), dtype=torch.long, device=device)
    if hasattr(dist, "all_gather_into_tensor"):
        dist.all_gather_into_tensor(gathered, local, group=process_group)
    else:
        pieces = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_gather(pieces, local, group=process_group)
        gathered.copy_(torch.cat(pieces, dim=0))
    return gathered


def _assert_intra_node_peer_visible(
    *,
    world_size: int,
    process_group: Any,
    device: torch.device,
    config: TmaIbgdaBackendConfig,
) -> None:
    if world_size <= 1:
        return
    if not config.require_intra_node:
        return
    group_id = id(process_group) if process_group is not None else 0
    cache_key = (group_id, world_size)
    if cache_key in _INTRA_NODE_GROUPS_CHECKED:
        return

    if torch.cuda.device_count() < world_size:
        raise TmaIbgdaBackendUnavailable(
            "TMA/IBGDA peer-visible transport requires every process to see the peer GPUs. "
            f"CUDA exposes {torch.cuda.device_count()} visible device(s), but the EP group has {world_size} rank(s). "
            "Launch with node-local peer GPUs visible to each process."
        )

    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    local_info = (socket.gethostname(), int(device_index))
    gathered: list[tuple[str, int] | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_info, group=process_group)

    hostnames = {info[0] for info in gathered if info is not None}
    if len(hostnames) != 1:
        raise TmaIbgdaBackendUnavailable(
            "TMA/IBGDA peer-visible transport is intra-node only; EP group spans hosts "
            f"{sorted(hostnames)}"
        )

    peer_devices = [info[1] for info in gathered if info is not None]
    for peer_device in peer_devices:
        if peer_device == device_index:
            continue
        if not torch.cuda.can_device_access_peer(device_index, peer_device):
            raise TmaIbgdaBackendUnavailable(
                f"CUDA device {device_index} cannot peer-access visible device {peer_device}; "
                "TMA/IBGDA peer-visible transport cannot run on this placement"
            )
    _INTRA_NODE_GROUPS_CHECKED.add(cache_key)


def _prepare_direct_peer_transport(
    *,
    ep_world_size: int,
    process_group: Any,
    device: torch.device,
    config: TmaIbgdaBackendConfig,
):
    rank, world_size = _resolve_world(
        ep_world_size=ep_world_size,
        process_group=process_group,
    )
    del rank
    _assert_intra_node_peer_visible(
        world_size=world_size,
        process_group=process_group,
        device=device,
        config=config,
    )
    tma_ibgda_ep = _require_kernel_module()
    return tma_ibgda_ep, world_size


def _resolve_ibgda_nblocks(
    requested: Optional[int],
    *,
    world_size: int,
    operation: str,
) -> int:
    if world_size <= 1:
        return int(requested or 0)
    if requested is None or requested == 0:
        return _MIN_MULTIRANK_IBGDA_BLOCKS
    if requested < _MIN_MULTIRANK_IBGDA_BLOCKS:
        raise TmaIbgdaBackendUnavailable(
            f"TMA/IBGDA {operation} needs at least {_MIN_MULTIRANK_IBGDA_BLOCKS} "
            f"blocks in the current multi-rank implementation; got {requested}. "
            "Lower-SM launches are intentionally disabled until the transport "
            "kernel has a true low-SM progress design."
        )
    return int(requested)


def _empty_transport_buffer(
    shape: tuple[int, int],
    *,
    dtype: torch.dtype,
    device: torch.device,
    process_group: Any,
    world_size: int,
    kind: str,
    config: TmaIbgdaBackendConfig,
    peer_window_plan: Optional[TmaIbgdaPeerWindowPlan] = None,
) -> torch.Tensor:
    if world_size <= 1:
        return torch.empty(shape, device=device, dtype=dtype)
    if process_group is None:
        raise TmaIbgdaBackendUnavailable(
            "multi-rank TMA/IBGDA buffers require an explicit process group"
        )
    if config.reuse_symmetric_buffers:
        if peer_window_plan is not None:
            if tuple(shape) != (peer_window_plan.rank_capacity, peer_window_plan.hidden_size):
                raise RuntimeError(
                    "TMA/IBGDA peer-window payload shape mismatch: "
                    f"shape={tuple(shape)}, plan payload="
                    f"{(peer_window_plan.rank_capacity, peer_window_plan.hidden_size)}"
                )
            if peer_window_plan.dtype != dtype:
                raise RuntimeError(
                    "TMA/IBGDA peer-window payload dtype mismatch: "
                    f"shape dtype={dtype}, plan dtype={peer_window_plan.dtype}"
                )
            return _PEER_WINDOW_SCRATCH_CACHE.get_payload(
                plan=peer_window_plan,
                dtype=dtype,
                device=device,
                process_group=process_group,
                world_size=world_size,
                kind=kind,
            )
        return _SYMMETRIC_SCRATCH_CACHE.get(
            kind=kind,
            shape=shape,
            dtype=dtype,
            device=device,
            process_group=process_group,
            world_size=world_size,
        )
    from olmo_core.kernels import tma_ibgda_ep

    return tma_ibgda_ep.empty_symmetric(
        shape,
        dtype=dtype,
        device=device,
        group=process_group,
    )


def _make_peer_window_plan_for_routes(
    dst_ranks: torch.Tensor,
    *,
    ep_world_size: int,
    rank_capacity: int,
    hidden_size: int,
    dtype: torch.dtype,
) -> TmaIbgdaPeerWindowPlan:
    num_tokens, top_k = dst_ranks.shape
    metadata = TmaIbgdaRouteMetadata(
        dst_ranks=dst_ranks,
        dst_rows=dst_ranks,
        valid_mask=torch.empty_like(dst_ranks, dtype=torch.bool),
        source_rows=torch.empty_like(dst_ranks),
        topk_slots=torch.empty_like(dst_ranks),
        routes_per_rank=torch.empty(
            (int(ep_world_size),),
            dtype=torch.long,
            device=dst_ranks.device,
        ),
        rank_offsets=torch.empty(
            (int(ep_world_size) + 1,),
            dtype=torch.long,
            device=dst_ranks.device,
        ),
        route_ordinals=torch.empty_like(dst_ranks),
        overflow_by_rank=torch.empty(
            (int(ep_world_size),),
            dtype=torch.bool,
            device=dst_ranks.device,
        ),
        num_tokens=int(num_tokens),
        top_k=int(top_k),
        ep_world_size=int(ep_world_size),
        rank_capacity=int(rank_capacity),
        static_route_budget=None,
    )
    return plan_tma_ibgda_peer_windows(
        metadata,
        hidden_size=hidden_size,
        dtype=dtype,
    )


def _empty_peer_window_views(
    plan: TmaIbgdaPeerWindowPlan,
    *,
    device: torch.device,
    process_group: Any,
    world_size: int,
    kind: str,
) -> TmaIbgdaPeerWindowViews:
    if world_size <= 1:
        window = torch.empty(
            (int(plan.rank_stride_bytes),),
            dtype=torch.uint8,
            device=device,
        )
        return _peer_window_views_from_window(window, plan)
    if process_group is None:
        raise TmaIbgdaBackendUnavailable(
            "multi-rank TMA/IBGDA peer windows require an explicit process group"
        )
    return _PEER_WINDOW_SCRATCH_CACHE.get_views(
        plan=plan,
        device=device,
        process_group=process_group,
        world_size=world_size,
        kind=kind,
    )


def tma_ibgda_empty_symmetric_expert_out(
    shape: tuple[int, int],
    *,
    dtype: torch.dtype,
    device: torch.device,
    ep_world_size: int,
    process_group: Any,
    config: Optional[TmaIbgdaBackendConfig] = None,
) -> torch.Tensor:
    """Allocate the expert-output tensor in the peer-visible transport window."""

    cfg = config or TmaIbgdaBackendConfig()
    _tma_ibgda_ep, world_size = _prepare_direct_peer_transport(
        ep_world_size=ep_world_size,
        process_group=process_group,
        device=device,
        config=cfg,
    )
    return _empty_transport_buffer(
        shape,
        dtype=dtype,
        device=device,
        process_group=process_group,
        world_size=world_size,
        kind="expert_out_direct",
        config=cfg,
    )


def _preprocess_routes_on_gpu(
    dst_ranks: torch.Tensor,
    dst_rows: torch.Tensor,
    *,
    probs: Optional[torch.Tensor],
    ep_world_size: int,
    rank_capacity: Optional[int],
    config: TmaIbgdaBackendConfig,
    peer_window_views: Optional[TmaIbgdaPeerWindowViews] = None,
) -> Optional[TmaIbgdaRoutePreprocess]:
    if not config.use_gpu_route_preprocess or rank_capacity is None:
        _debug_tma_ibgda(
            "route preprocess skipped "
            f"use_gpu={config.use_gpu_route_preprocess} rank_capacity={rank_capacity}"
        )
        return None
    tma_ibgda_ep = _require_kernel_module()
    if peer_window_views is None:
        _debug_tma_ibgda(
            "route preprocess allocate launch "
            f"tokens={dst_ranks.shape[0]} top_k={dst_ranks.shape[1]} "
            f"ep={ep_world_size} capacity={int(rank_capacity)} "
            f"static_budget={config.static_route_budget}"
        )
        (
            route_records,
            routes_per_rank,
            rank_offsets,
            overflow_by_rank,
            route_ordinals,
            errors,
        ) = tma_ibgda_ep.preprocess_routes(
            dst_ranks,
            dst_rows,
            ep_world_size=ep_world_size,
            rank_capacity=int(rank_capacity),
            static_route_budget=config.static_route_budget,
            probs=probs,
            nblocks=config.num_sms_preprocess or 0,
        )
        _debug_tma_ibgda_sync("route preprocess allocate", dst_ranks.device)
    else:
        _debug_tma_ibgda(
            "route preprocess into peer-window launch "
            f"tokens={dst_ranks.shape[0]} top_k={dst_ranks.shape[1]} "
            f"ep={ep_world_size} capacity={int(rank_capacity)} "
            f"static_budget={config.static_route_budget}"
        )
        route_ordinals = torch.empty_like(dst_ranks)
        errors = torch.empty((3,), dtype=torch.int32, device=dst_ranks.device)
        tma_ibgda_ep.preprocess_routes_into(
            dst_ranks,
            dst_rows,
            route_records=peer_window_views.route_records,
            routes_per_rank=peer_window_views.routes_per_rank,
            rank_offsets=peer_window_views.rank_offsets,
            overflow_by_rank=peer_window_views.overflow_by_rank,
            route_ordinals=route_ordinals,
            errors=errors,
            ep_world_size=ep_world_size,
            rank_capacity=int(rank_capacity),
            static_route_budget=config.static_route_budget,
            probs=probs,
            nblocks=config.num_sms_preprocess or 0,
        )
        _debug_tma_ibgda_sync("route preprocess into peer-window", dst_ranks.device)
        route_records = peer_window_views.route_records
        routes_per_rank = peer_window_views.routes_per_rank
        rank_offsets = peer_window_views.rank_offsets
        overflow_by_rank = peer_window_views.overflow_by_rank
    preprocess = TmaIbgdaRoutePreprocess(
        route_records=route_records,
        routes_per_rank=routes_per_rank,
        rank_offsets=rank_offsets,
        overflow_by_rank=overflow_by_rank,
        route_ordinals=route_ordinals,
        errors=errors,
    )
    if config.validate_gpu_route_preprocess:
        _debug_tma_ibgda("route preprocess validation begin")
        _check_gpu_preprocess_errors(preprocess)
        _debug_tma_ibgda("route preprocess validation end")
    _debug_tma_ibgda("route preprocess done")
    return preprocess


def _clone_route_preprocess(
    preprocess: Optional[TmaIbgdaRoutePreprocess],
) -> Optional[TmaIbgdaRoutePreprocess]:
    if preprocess is None:
        return None
    return TmaIbgdaRoutePreprocess(
        route_records=preprocess.route_records.clone(memory_format=torch.contiguous_format),
        routes_per_rank=preprocess.routes_per_rank.clone(memory_format=torch.contiguous_format),
        rank_offsets=preprocess.rank_offsets.clone(memory_format=torch.contiguous_format),
        overflow_by_rank=preprocess.overflow_by_rank.clone(memory_format=torch.contiguous_format),
        route_ordinals=preprocess.route_ordinals.clone(memory_format=torch.contiguous_format),
        errors=preprocess.errors.clone(memory_format=torch.contiguous_format),
    )


def _check_gpu_preprocess_errors(preprocess: TmaIbgdaRoutePreprocess) -> None:
    errors = preprocess.errors.detach().cpu().tolist()
    if len(errors) != 3:
        raise RuntimeError(
            f"TMA/IBGDA route preprocess returned {len(errors)} error flag(s), expected 3"
        )
    messages = []
    if errors[0]:
        messages.append("dropped routes must have both rank and row negative")
    if errors[1]:
        messages.append("dst_ranks contains a valid route outside ep_world_size")
    if errors[2]:
        messages.append("dst_rows contains a valid route outside rank_capacity")
    if messages:
        raise RuntimeError("TMA/IBGDA route preprocess failed: " + "; ".join(messages))


def _preprocess_with_route_probs(
    preprocess: TmaIbgdaRoutePreprocess,
    probs: torch.Tensor,
) -> TmaIbgdaRoutePreprocess:
    """Reuse route structure from dispatch while patching combine probabilities on GPU."""

    _check_probs(probs, preprocess.route_ordinals.shape)
    tma_ibgda_ep = _require_kernel_module()
    route_records = tma_ibgda_ep.route_records_with_probs(
        preprocess.route_records,
        probs,
    )
    return TmaIbgdaRoutePreprocess(
        route_records=route_records,
        routes_per_rank=preprocess.routes_per_rank,
        rank_offsets=preprocess.rank_offsets,
        overflow_by_rank=preprocess.overflow_by_rank,
        route_ordinals=preprocess.route_ordinals,
        errors=preprocess.errors,
    )


def _check_dispatch_handle_matches_combine(
    handle: TmaIbgdaDispatchHandle,
    *,
    group_name: str,
    src_ranks: torch.Tensor,
    src_rows: torch.Tensor,
    ep_world_size: int,
    rank_capacity: Optional[int],
    process_group: Any,
    config: Optional[TmaIbgdaBackendConfig],
    payload_hidden_size: Optional[int] = None,
    payload_dtype: Optional[torch.dtype] = None,
) -> None:
    metadata = handle.metadata
    if handle.group_name != group_name:
        raise RuntimeError(
            "TMA/IBGDA combine handle group mismatch: "
            f"handle has {handle.group_name!r}, combine got {group_name!r}"
        )
    if metadata.ep_world_size != int(ep_world_size):
        raise RuntimeError(
            "TMA/IBGDA combine handle ep_world_size mismatch: "
            f"handle has {metadata.ep_world_size}, combine got {ep_world_size}"
        )
    if rank_capacity is not None and metadata.rank_capacity != int(rank_capacity):
        raise RuntimeError(
            "TMA/IBGDA combine handle rank_capacity mismatch: "
            f"handle has {metadata.rank_capacity}, combine got {rank_capacity}"
        )
    if process_group is not None and process_group is not handle.process_group:
        raise RuntimeError("TMA/IBGDA combine handle process_group mismatch")
    if config is not None and config != handle.config:
        raise RuntimeError("TMA/IBGDA combine handle config mismatch")
    if handle.peer_window_plan is not None:
        plan = handle.peer_window_plan
        if plan.ep_world_size != metadata.ep_world_size:
            raise RuntimeError("TMA/IBGDA combine handle peer-window ep_world_size mismatch")
        if plan.rank_capacity != metadata.rank_capacity:
            raise RuntimeError("TMA/IBGDA combine handle peer-window rank_capacity mismatch")
        if payload_hidden_size is not None and plan.hidden_size != int(payload_hidden_size):
            raise RuntimeError(
                "TMA/IBGDA combine handle peer-window hidden_size mismatch: "
                f"handle has {plan.hidden_size}, combine got {payload_hidden_size}"
            )
        if payload_dtype is not None and plan.dtype != payload_dtype:
            raise RuntimeError(
                "TMA/IBGDA combine handle peer-window dtype mismatch: "
                f"handle has {plan.dtype}, combine got {payload_dtype}"
            )

    if tuple(metadata.dst_ranks.shape) != tuple(src_ranks.shape) or tuple(
        metadata.dst_rows.shape
    ) != tuple(src_rows.shape):
        raise RuntimeError(
            "TMA/IBGDA combine handle route shape mismatch: "
            f"handle has {tuple(metadata.dst_ranks.shape)}/"
            f"{tuple(metadata.dst_rows.shape)}, combine got "
            f"{tuple(src_ranks.shape)}/{tuple(src_rows.shape)}"
        )
    same_ranks = (
        metadata.dst_ranks.data_ptr() == src_ranks.data_ptr()
        and metadata.dst_ranks.stride() == src_ranks.stride()
        and metadata.dst_ranks.device == src_ranks.device
        and metadata.dst_ranks.dtype == src_ranks.dtype
    )
    same_rows = (
        metadata.dst_rows.data_ptr() == src_rows.data_ptr()
        and metadata.dst_rows.stride() == src_rows.stride()
        and metadata.dst_rows.device == src_rows.device
        and metadata.dst_rows.dtype == src_rows.dtype
    )
    if not same_ranks or not same_rows:
        raise RuntimeError(
            "TMA/IBGDA combine handle route tensor mismatch. Reusing packed "
            "route records requires the same route-map tensors returned by dispatch."
        )
    current_ranks_version = _tensor_version(src_ranks)
    current_rows_version = _tensor_version(src_rows)
    if (
        handle.dst_ranks_version is not None
        and current_ranks_version is not None
        and handle.dst_ranks_version != current_ranks_version
    ):
        raise RuntimeError(
            "TMA/IBGDA combine handle route tensor version mismatch for ranks. "
            "Route maps were modified after dispatch."
        )
    if (
        handle.dst_rows_version is not None
        and current_rows_version is not None
        and handle.dst_rows_version != current_rows_version
    ):
        raise RuntimeError(
            "TMA/IBGDA combine handle route tensor version mismatch for rows. "
            "Route maps were modified after dispatch."
        )


def _metadata_from_preprocess(
    dst_ranks: torch.Tensor,
    dst_rows: torch.Tensor,
    *,
    ep_world_size: int,
    rank_capacity: int,
    static_route_budget: Optional[int],
    preprocess: TmaIbgdaRoutePreprocess,
) -> TmaIbgdaRouteMetadata:
    num_tokens, top_k = dst_ranks.shape
    records = preprocess.route_records
    valid_mask = (records[:, 5] != 0).view(num_tokens, top_k)
    source_rows = records[:, 0].to(dtype=torch.long).view(num_tokens, top_k)
    topk_slots = records[:, 1].to(dtype=torch.long).view(num_tokens, top_k)
    return TmaIbgdaRouteMetadata(
        dst_ranks=dst_ranks,
        dst_rows=dst_rows,
        valid_mask=valid_mask,
        source_rows=source_rows,
        topk_slots=topk_slots,
        routes_per_rank=preprocess.routes_per_rank,
        rank_offsets=preprocess.rank_offsets,
        route_ordinals=preprocess.route_ordinals,
        overflow_by_rank=preprocess.overflow_by_rank,
        num_tokens=num_tokens,
        top_k=top_k,
        ep_world_size=ep_world_size,
        rank_capacity=rank_capacity,
        static_route_budget=static_route_budget,
    )


def _dispatch_bf16_peer_raw(
    input: torch.Tensor,
    dst_ranks: torch.Tensor,
    dst_rows: torch.Tensor,
    *,
    probs: Optional[torch.Tensor],
    ep_world_size: int,
    rank_capacity: Optional[int],
    process_group: Any,
    config: TmaIbgdaBackendConfig,
    preprocess: Optional[TmaIbgdaRoutePreprocess] = None,
) -> tuple[
    torch.Tensor,
    TmaIbgdaRouteMetadata,
    torch.Tensor,
    Optional[TmaIbgdaRoutePreprocess],
]:
    _check_bf16_tensor("input", input)
    _check_route_tensors(dst_ranks, dst_rows)
    _check_probs(probs, dst_ranks.shape)
    _debug_tma_ibgda(
        "dispatch raw begin "
        f"input={tuple(input.shape)} routes={tuple(dst_ranks.shape)} "
        f"ep={ep_world_size} rank_capacity={rank_capacity}"
    )
    tma_ibgda_ep, world_size = _prepare_direct_peer_transport(
        ep_world_size=ep_world_size,
        process_group=process_group,
        device=input.device,
        config=config,
    )
    _debug_tma_ibgda(f"dispatch transport ready world_size={world_size}")
    peer_window_plan: Optional[TmaIbgdaPeerWindowPlan] = None
    peer_window_views: Optional[TmaIbgdaPeerWindowViews] = None
    if preprocess is None:
        if (
            rank_capacity is not None
            and config.use_gpu_route_preprocess
            and config.reuse_symmetric_buffers
        ):
            peer_window_plan = _make_peer_window_plan_for_routes(
                dst_ranks,
                ep_world_size=ep_world_size,
                rank_capacity=int(rank_capacity),
                hidden_size=input.shape[1],
                dtype=input.dtype,
            )
            peer_window_views = _empty_peer_window_views(
                peer_window_plan,
                device=input.device,
                process_group=process_group,
                world_size=world_size,
                kind="dispatch",
            )
        preprocess = _preprocess_routes_on_gpu(
            dst_ranks,
            dst_rows,
            probs=probs,
            ep_world_size=ep_world_size,
            rank_capacity=rank_capacity,
            config=config,
            peer_window_views=peer_window_views,
        )
    if preprocess is not None and rank_capacity is not None:
        metadata = _metadata_from_preprocess(
            dst_ranks,
            dst_rows,
            ep_world_size=ep_world_size,
            rank_capacity=int(rank_capacity),
            static_route_budget=config.static_route_budget,
            preprocess=preprocess,
        )
    else:
        metadata = build_tma_ibgda_route_metadata(
            dst_ranks,
            dst_rows,
            ep_world_size=ep_world_size,
            rank_capacity=rank_capacity,
            static_route_budget=config.static_route_budget,
        )
    _debug_tma_ibgda(
        "dispatch metadata ready "
        f"tokens={metadata.num_tokens} top_k={metadata.top_k} "
        f"rank_capacity={metadata.rank_capacity} "
        f"static_route_budget={metadata.static_route_budget}"
    )
    if peer_window_plan is None:
        peer_window_plan = plan_tma_ibgda_peer_windows(
            metadata,
            hidden_size=input.shape[1],
            dtype=input.dtype,
        )
    if peer_window_views is not None:
        dispatch_out = peer_window_views.payload
    else:
        dispatch_out = _empty_transport_buffer(
            (metadata.rank_capacity, input.shape[1]),
            device=input.device,
            dtype=input.dtype,
            process_group=process_group,
            world_size=world_size,
            kind="dispatch",
            config=config,
            peer_window_plan=peer_window_plan,
        )
    if world_size > 1:
        dispatch_nblocks = _resolve_ibgda_nblocks(
            config.num_sms_dispatch,
            world_size=world_size,
            operation="dispatch",
        )
        if preprocess is not None:
            if config.use_tma_load_dispatch:
                _debug_tma_ibgda(
                    "dispatch records_tma launch "
                    f"nblocks={dispatch_nblocks} out={tuple(dispatch_out.shape)}"
                )
                tma_ibgda_ep.dispatch_bf16_ibgda_records_tma(
                    input,
                    dispatch_out,
                    preprocess.route_records,
                    nblocks=dispatch_nblocks,
                )
                _debug_tma_ibgda_sync("dispatch records_tma", input.device)
            else:
                _debug_tma_ibgda(
                    "dispatch records launch "
                    f"nblocks={dispatch_nblocks} out={tuple(dispatch_out.shape)}"
                )
                tma_ibgda_ep.dispatch_bf16_ibgda_records(
                    input,
                    dispatch_out,
                    preprocess.route_records,
                    nblocks=dispatch_nblocks,
                )
                _debug_tma_ibgda_sync("dispatch records", input.device)
        else:
            _debug_tma_ibgda(
                "dispatch route-map launch "
                f"nblocks={dispatch_nblocks} out={tuple(dispatch_out.shape)}"
            )
            tma_ibgda_ep.dispatch_bf16_ibgda(
                input,
                dispatch_out,
                dst_ranks,
                dst_rows,
                probs,
                nblocks=dispatch_nblocks,
            )
            _debug_tma_ibgda_sync("dispatch route-map", input.device)
        peer_out_ptrs = torch.empty((0,), dtype=torch.long, device=input.device)
    else:
        peer_out_ptrs = _all_gather_cuda_i64(
            dispatch_out.data_ptr(),
            world_size,
            input.device,
            process_group,
        )
        _debug_tma_ibgda(f"dispatch peer launch out={tuple(dispatch_out.shape)}")
        tma_ibgda_ep.dispatch_bf16_peer(
            input,
            dispatch_out,
            dst_ranks,
            dst_rows,
            peer_out_ptrs,
            probs,
            nblocks=config.num_sms_dispatch or 0,
        )
        _debug_tma_ibgda_sync("dispatch peer", input.device)
    if config.synchronize_after_dispatch:
        _debug_tma_ibgda("dispatch transport barrier begin")
        _transport_barrier(
            process_group,
            input.device,
            world_size=world_size,
            config=config,
            signal_kind="dispatch_out",
        )
        _debug_tma_ibgda("dispatch transport barrier end")
    elif world_size > 1 and config.reuse_symmetric_buffers and not config.return_symmetric_dispatch_buffer:
        raise RuntimeError(
            "TMA/IBGDA dispatch must synchronize before cloning from the symmetric "
            "transport window. Set synchronize_after_dispatch=True or "
            "return_symmetric_dispatch_buffer=True."
        )
    if world_size > 1 and config.reuse_symmetric_buffers and not config.return_symmetric_dispatch_buffer:
        _debug_tma_ibgda("dispatch clone from symmetric begin")
        dispatch_out = dispatch_out.clone(memory_format=torch.contiguous_format)
        _debug_tma_ibgda_sync("dispatch clone from symmetric", input.device)
        _debug_tma_ibgda("dispatch clone from symmetric end")
    _debug_tma_ibgda(f"dispatch raw end out={tuple(dispatch_out.shape)}")
    return dispatch_out, metadata, peer_out_ptrs, preprocess


def _combine_bf16_peer_raw(
    expert_out: torch.Tensor,
    src_ranks: torch.Tensor,
    src_rows: torch.Tensor,
    *,
    probs: Optional[torch.Tensor],
    ep_world_size: int,
    rank_capacity: Optional[int],
    num_tokens: Optional[int],
    process_group: Any,
    config: TmaIbgdaBackendConfig,
    preprocess: Optional[TmaIbgdaRoutePreprocess] = None,
    expert_out_is_symmetric: bool = False,
) -> torch.Tensor:
    _check_bf16_tensor("expert_out", expert_out)
    _check_route_tensors(src_ranks, src_rows)
    _check_probs(probs, src_ranks.shape)
    _debug_tma_ibgda(
        "combine raw begin "
        f"expert_out={tuple(expert_out.shape)} routes={tuple(src_ranks.shape)} "
        f"ep={ep_world_size} rank_capacity={rank_capacity} "
        f"expert_out_is_symmetric={expert_out_is_symmetric}"
    )
    if preprocess is not None and rank_capacity is not None:
        metadata = _metadata_from_preprocess(
            src_ranks,
            src_rows,
            ep_world_size=ep_world_size,
            rank_capacity=int(rank_capacity),
            static_route_budget=config.static_route_budget,
            preprocess=preprocess,
        )
    else:
        metadata = build_tma_ibgda_route_metadata(
            src_ranks,
            src_rows,
            ep_world_size=ep_world_size,
            rank_capacity=rank_capacity,
            static_route_budget=config.static_route_budget,
        )
    if num_tokens is not None and metadata.num_tokens != int(num_tokens):
        raise RuntimeError(
            f"num_tokens mismatch for TMA/IBGDA combine: got {num_tokens}, metadata has {metadata.num_tokens}"
        )
    _debug_tma_ibgda(
        "combine metadata ready "
        f"tokens={metadata.num_tokens} top_k={metadata.top_k} "
        f"rank_capacity={metadata.rank_capacity} "
        f"static_route_budget={metadata.static_route_budget}"
    )
    tma_ibgda_ep, world_size = _prepare_direct_peer_transport(
        ep_world_size=ep_world_size,
        process_group=process_group,
        device=expert_out.device,
        config=config,
    )
    _debug_tma_ibgda(f"combine transport ready world_size={world_size}")
    if config.synchronize_before_combine:
        _debug_tma_ibgda("combine pre-barrier begin")
        _transport_barrier(
            process_group,
            expert_out.device,
            world_size=world_size,
            config=config,
            signal_kind="pre_combine",
        )
        _debug_tma_ibgda("combine pre-barrier end")

    out = torch.empty(
        (metadata.num_tokens, expert_out.shape[1]),
        device=expert_out.device,
        dtype=expert_out.dtype,
    )
    if world_size > 1:
        combine_nblocks = _resolve_ibgda_nblocks(
            config.num_sms_combine,
            world_size=world_size,
            operation="combine",
        )
        if preprocess is None:
            preprocess = _preprocess_routes_on_gpu(
                src_ranks,
                src_rows,
                probs=probs,
                ep_world_size=ep_world_size,
                rank_capacity=metadata.rank_capacity,
                config=config,
            )
        if expert_out_is_symmetric:
            expert_transport = expert_out
        else:
            expert_transport = _empty_transport_buffer(
                (expert_out.shape[0], expert_out.shape[1]),
                device=expert_out.device,
                dtype=expert_out.dtype,
                process_group=process_group,
                world_size=world_size,
                kind="expert_out",
                config=config,
                peer_window_plan=plan_tma_ibgda_peer_windows(
                    metadata,
                    hidden_size=expert_out.shape[1],
                    dtype=expert_out.dtype,
                ),
            )
            _debug_tma_ibgda("combine expert_out copy begin")
            expert_transport.copy_(expert_out)
            _debug_tma_ibgda_sync("combine expert_out copy", expert_out.device)
            _debug_tma_ibgda("combine expert_out copy end")
        _debug_tma_ibgda("combine expert_out barrier begin")
        _transport_barrier(
            process_group,
            expert_out.device,
            world_size=world_size,
            config=config,
            signal_kind="combine_expert_out",
        )
        _debug_tma_ibgda("combine expert_out barrier end")
        if preprocess is not None:
            _debug_tma_ibgda(
                "combine records launch "
                f"nblocks={combine_nblocks} out={tuple(out.shape)}"
            )
            tma_ibgda_ep.combine_bf16_ibgda_records(
                expert_transport,
                out,
                preprocess.route_records,
                top_k=metadata.top_k,
                nblocks=combine_nblocks,
            )
            _debug_tma_ibgda_sync("combine records", expert_out.device)
        else:
            _debug_tma_ibgda(
                "combine route-map launch "
                f"nblocks={combine_nblocks} out={tuple(out.shape)}"
            )
            tma_ibgda_ep.combine_bf16_ibgda(
                expert_transport,
                out,
                src_ranks,
                src_rows,
                probs,
                nblocks=combine_nblocks,
            )
            _debug_tma_ibgda_sync("combine route-map", expert_out.device)
    else:
        peer_expert_out_ptrs = _all_gather_cuda_i64(
            expert_out.data_ptr(),
            world_size,
            expert_out.device,
            process_group,
        )
        _debug_tma_ibgda(f"combine peer launch out={tuple(out.shape)}")
        tma_ibgda_ep.combine_bf16_peer(
            expert_out,
            out,
            src_ranks,
            src_rows,
            peer_expert_out_ptrs,
            probs,
        )
        _debug_tma_ibgda_sync("combine peer", expert_out.device)
    _debug_tma_ibgda(f"combine raw end out={tuple(out.shape)}")
    return out


def _route_dot_bf16_peer_raw(
    expert_out: torch.Tensor,
    grad_out: torch.Tensor,
    src_ranks: torch.Tensor,
    src_rows: torch.Tensor,
    *,
    ep_world_size: int,
    process_group: Any,
    config: TmaIbgdaBackendConfig,
    preprocess: Optional[TmaIbgdaRoutePreprocess] = None,
    expert_out_is_symmetric: bool = False,
) -> torch.Tensor:
    _check_bf16_tensor("expert_out", expert_out)
    _check_bf16_tensor("grad_out", grad_out)
    _check_route_tensors(src_ranks, src_rows)
    if src_ranks.shape[0] != grad_out.shape[0]:
        raise RuntimeError("route maps first dim must match grad_out rows")
    _debug_tma_ibgda(
        "route-dot raw begin "
        f"expert_out={tuple(expert_out.shape)} grad_out={tuple(grad_out.shape)} "
        f"routes={tuple(src_ranks.shape)} ep={ep_world_size} "
        f"expert_out_is_symmetric={expert_out_is_symmetric}"
    )
    rank_capacity = int(expert_out.shape[0])
    if preprocess is not None:
        metadata = _metadata_from_preprocess(
            src_ranks,
            src_rows,
            ep_world_size=ep_world_size,
            rank_capacity=rank_capacity,
            static_route_budget=config.static_route_budget,
            preprocess=preprocess,
        )
    else:
        metadata = build_tma_ibgda_route_metadata(
            src_ranks,
            src_rows,
            ep_world_size=ep_world_size,
            rank_capacity=rank_capacity,
            static_route_budget=config.static_route_budget,
        )
    _debug_tma_ibgda(
        "route-dot metadata ready "
        f"tokens={metadata.num_tokens} top_k={metadata.top_k} "
        f"rank_capacity={metadata.rank_capacity} "
        f"static_route_budget={metadata.static_route_budget}"
    )
    tma_ibgda_ep, world_size = _prepare_direct_peer_transport(
        ep_world_size=ep_world_size,
        process_group=process_group,
        device=expert_out.device,
        config=config,
    )
    _debug_tma_ibgda(f"route-dot transport ready world_size={world_size}")
    if config.synchronize_before_combine:
        _debug_tma_ibgda("route-dot pre-barrier begin")
        _transport_barrier(
            process_group,
            expert_out.device,
            world_size=world_size,
            config=config,
            signal_kind="pre_route_dot",
        )
        _debug_tma_ibgda("route-dot pre-barrier end")
    out = torch.empty_like(src_ranks, dtype=torch.float32)
    if world_size > 1:
        if expert_out_is_symmetric:
            expert_transport = expert_out
        else:
            expert_transport = _empty_transport_buffer(
                (expert_out.shape[0], expert_out.shape[1]),
                device=expert_out.device,
                dtype=expert_out.dtype,
                process_group=process_group,
                world_size=world_size,
                kind="expert_out",
                config=config,
                peer_window_plan=plan_tma_ibgda_peer_windows(
                    metadata,
                    hidden_size=expert_out.shape[1],
                    dtype=expert_out.dtype,
                ),
            )
            _debug_tma_ibgda("route-dot expert_out copy begin")
            expert_transport.copy_(expert_out)
            _debug_tma_ibgda_sync("route-dot expert_out copy", expert_out.device)
            _debug_tma_ibgda("route-dot expert_out copy end")
        _debug_tma_ibgda("route-dot expert_out barrier begin")
        _transport_barrier(
            process_group,
            expert_out.device,
            world_size=world_size,
            config=config,
            signal_kind="route_dot_expert_out",
        )
        _debug_tma_ibgda("route-dot expert_out barrier end")
        if preprocess is not None:
            _debug_tma_ibgda(f"route-dot records launch out={tuple(out.shape)}")
            tma_ibgda_ep.route_dot_bf16_ibgda_records(
                expert_transport,
                grad_out,
                preprocess.route_records,
                top_k=src_ranks.shape[1],
                out=out,
            )
            _debug_tma_ibgda_sync("route-dot records", expert_out.device)
        else:
            _debug_tma_ibgda(f"route-dot route-map launch out={tuple(out.shape)}")
            tma_ibgda_ep.route_dot_bf16_ibgda(
                expert_transport,
                grad_out,
                src_ranks,
                src_rows,
                out,
            )
            _debug_tma_ibgda_sync("route-dot route-map", expert_out.device)
    else:
        peer_expert_out_ptrs = _all_gather_cuda_i64(
            expert_out.data_ptr(),
            world_size,
            expert_out.device,
            process_group,
        )
        _debug_tma_ibgda(f"route-dot peer launch out={tuple(out.shape)}")
        tma_ibgda_ep.route_dot_bf16_peer(
            expert_out,
            grad_out,
            src_ranks,
            src_rows,
            peer_expert_out_ptrs,
            out,
        )
        _debug_tma_ibgda_sync("route-dot peer", expert_out.device)
    _debug_tma_ibgda(f"route-dot raw end out={tuple(out.shape)}")
    return out


class _TmaIbgdaDispatchAutograd(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        input: torch.Tensor,
        dst_ranks: torch.Tensor,
        dst_rows: torch.Tensor,
        probs: Optional[torch.Tensor],
        ep_world_size: int,
        rank_capacity: Optional[int],
        process_group: Any,
        config: TmaIbgdaBackendConfig,
        preprocess: Optional[TmaIbgdaRoutePreprocess],
        capture: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        dispatch_out, metadata, _peer_out_ptrs, _preprocess = _dispatch_bf16_peer_raw(
            input,
            dst_ranks,
            dst_rows,
            probs=probs,
            ep_world_size=int(ep_world_size),
            rank_capacity=rank_capacity,
            process_group=process_group,
            config=config,
            preprocess=preprocess,
        )
        saved_preprocess = _clone_route_preprocess(_preprocess)
        if capture is not None:
            capture["metadata"] = metadata
            capture["peer_out_ptrs"] = _peer_out_ptrs
            capture["preprocess"] = saved_preprocess
        tensors: list[torch.Tensor] = [dst_ranks, dst_rows]
        if probs is not None:
            tensors.extend([probs, input])
        ctx.save_for_backward(*tensors)
        ctx.has_probs = probs is not None
        ctx.ep_world_size = int(ep_world_size)
        ctx.rank_capacity = metadata.rank_capacity
        ctx.process_group = process_group
        ctx.config = config
        ctx.preprocess = saved_preprocess
        return dispatch_out

    @staticmethod
    def backward(ctx, grad_dispatch_out: torch.Tensor):  # type: ignore[override]
        saved = ctx.saved_tensors
        dst_ranks = saved[0]
        dst_rows = saved[1]
        probs = saved[2] if ctx.has_probs else None
        input = saved[3] if ctx.has_probs else None
        grad_dispatch_out = grad_dispatch_out.contiguous()
        grad_input = _combine_bf16_peer_raw(
            grad_dispatch_out,
            dst_ranks,
            dst_rows,
            probs=probs,
            ep_world_size=ctx.ep_world_size,
            rank_capacity=ctx.rank_capacity,
            num_tokens=dst_ranks.shape[0],
            process_group=ctx.process_group,
            config=ctx.config,
            preprocess=ctx.preprocess,
        )
        grad_probs = None
        if ctx.has_probs and input is not None:
            grad_probs = _route_dot_bf16_peer_raw(
                grad_dispatch_out,
                input,
                dst_ranks,
                dst_rows,
                ep_world_size=ctx.ep_world_size,
                process_group=ctx.process_group,
                config=ctx.config,
                preprocess=ctx.preprocess,
            )
        return grad_input, None, None, grad_probs, None, None, None, None, None, None


class _TmaIbgdaCombineAutograd(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        expert_out: torch.Tensor,
        src_ranks: torch.Tensor,
        src_rows: torch.Tensor,
        probs: Optional[torch.Tensor],
        ep_world_size: int,
        rank_capacity: Optional[int],
        process_group: Any,
        config: TmaIbgdaBackendConfig,
        preprocess: Optional[TmaIbgdaRoutePreprocess],
        expert_out_is_symmetric: bool,
    ) -> torch.Tensor:
        out = _combine_bf16_peer_raw(
            expert_out,
            src_ranks,
            src_rows,
            probs=probs,
            ep_world_size=int(ep_world_size),
            rank_capacity=rank_capacity,
            num_tokens=src_ranks.shape[0],
            process_group=process_group,
            config=config,
            preprocess=preprocess,
            expert_out_is_symmetric=bool(expert_out_is_symmetric),
        )
        tensors: list[torch.Tensor] = [src_ranks, src_rows, expert_out]
        if probs is not None:
            tensors.append(probs)
        ctx.save_for_backward(*tensors)
        ctx.has_probs = probs is not None
        ctx.ep_world_size = int(ep_world_size)
        ctx.rank_capacity = expert_out.shape[0]
        ctx.process_group = process_group
        ctx.config = config
        ctx.preprocess = preprocess
        ctx.expert_out_is_symmetric = bool(expert_out_is_symmetric)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        saved = ctx.saved_tensors
        src_ranks = saved[0]
        src_rows = saved[1]
        expert_out = saved[2]
        probs = saved[3] if ctx.has_probs else None
        grad_out = grad_out.contiguous()
        grad_expert_out, _metadata, _peer_ptrs, _preprocess = _dispatch_bf16_peer_raw(
            grad_out,
            src_ranks,
            src_rows,
            probs=probs,
            ep_world_size=ctx.ep_world_size,
            rank_capacity=ctx.rank_capacity,
            process_group=ctx.process_group,
            config=ctx.config,
            preprocess=ctx.preprocess,
        )
        grad_probs = None
        if ctx.has_probs:
            grad_probs = _route_dot_bf16_peer_raw(
                expert_out,
                grad_out,
                src_ranks,
                src_rows,
                ep_world_size=ctx.ep_world_size,
                process_group=ctx.process_group,
                config=ctx.config,
                preprocess=ctx.preprocess,
                expert_out_is_symmetric=ctx.expert_out_is_symmetric,
            )
        return grad_expert_out, None, None, grad_probs, None, None, None, None, None, None


def tma_ibgda_rowwise_dispatch_bf16(
    input: torch.Tensor,
    dst_ranks: torch.Tensor,
    dst_rows: torch.Tensor,
    group_name: str,
    *,
    ep_world_size: int,
    probs: Optional[torch.Tensor] = None,
    rank_capacity: Optional[int] = None,
    process_group: Any = None,
    config: Optional[TmaIbgdaBackendConfig] = None,
) -> tuple[torch.Tensor, TmaIbgdaDispatchHandle]:
    """BF16 rowwise dispatch through intra-node peer-visible output buffers."""

    cfg = config or TmaIbgdaBackendConfig()
    _check_bf16_tensor("input", input)
    _check_route_tensors(dst_ranks, dst_rows)
    _check_probs(probs, dst_ranks.shape)
    _debug_tma_ibgda(
        "rowwise dispatch begin "
        f"group={group_name} input={tuple(input.shape)} ep={ep_world_size}"
    )
    capture: dict[str, Any] = {}
    dispatch_out = _TmaIbgdaDispatchAutograd.apply(
        input,
        dst_ranks,
        dst_rows,
        probs,
        ep_world_size,
        rank_capacity,
        process_group,
        cfg,
        None,
        capture,
    )
    metadata = capture["metadata"]
    preprocess = capture.get("preprocess")
    peer_out_ptrs = capture.get(
        "peer_out_ptrs",
        torch.empty((0,), dtype=torch.long, device=input.device),
    )

    handle = TmaIbgdaDispatchHandle(
        metadata=metadata,
        group_name=group_name,
        config=cfg,
        process_group=process_group,
        peer_out_ptrs=peer_out_ptrs,
        preprocess=preprocess,
        peer_window_plan=plan_tma_ibgda_peer_windows(
            metadata,
            hidden_size=input.shape[1],
            dtype=input.dtype,
        ),
        dst_ranks_version=_tensor_version(dst_ranks),
        dst_rows_version=_tensor_version(dst_rows),
    )
    _debug_tma_ibgda(f"rowwise dispatch end out={tuple(dispatch_out.shape)}")
    return dispatch_out, handle


def tma_ibgda_rowwise_combine_bf16(
    expert_out: torch.Tensor,
    src_ranks: torch.Tensor,
    src_rows: torch.Tensor,
    group_name: str,
    *,
    ep_world_size: int,
    probs: Optional[torch.Tensor],
    handle: Optional[TmaIbgdaDispatchHandle] = None,
    rank_capacity: Optional[int] = None,
    process_group: Any = None,
    config: Optional[TmaIbgdaBackendConfig] = None,
    expert_out_is_symmetric: bool = False,
) -> torch.Tensor:
    """BF16 rowwise combine through intra-node peer-visible expert-output buffers."""

    if handle is not None:
        _check_dispatch_handle_matches_combine(
            handle,
            group_name=group_name,
            src_ranks=src_ranks,
            src_rows=src_rows,
            ep_world_size=ep_world_size,
            rank_capacity=rank_capacity,
            process_group=process_group,
            config=config,
            payload_hidden_size=expert_out.shape[1],
            payload_dtype=expert_out.dtype,
        )
    cfg = config or (handle.config if handle is not None else TmaIbgdaBackendConfig())
    pg = process_group if process_group is not None else (handle.process_group if handle is not None else None)
    resolved_rank_capacity = (
        rank_capacity
        if rank_capacity is not None
        else (handle.metadata.rank_capacity if handle is not None else None)
    )
    preprocess = None
    if handle is not None and handle.preprocess is not None:
        if probs is None:
            preprocess = handle.preprocess
        else:
            preprocess = _preprocess_with_route_probs(handle.preprocess, probs)
    else:
        preprocess = _preprocess_routes_on_gpu(
            src_ranks,
            src_rows,
            probs=probs,
            ep_world_size=ep_world_size,
            rank_capacity=resolved_rank_capacity,
            config=cfg,
        )
    _debug_tma_ibgda(
        "rowwise combine begin "
        f"group={group_name} expert_out={tuple(expert_out.shape)} ep={ep_world_size} "
        f"rank_capacity={resolved_rank_capacity}"
    )
    out = _TmaIbgdaCombineAutograd.apply(
        expert_out,
        src_ranks,
        src_rows,
        probs,
        ep_world_size,
        resolved_rank_capacity,
        pg,
        cfg,
        preprocess,
        bool(expert_out_is_symmetric),
    )
    _debug_tma_ibgda(f"rowwise combine end out={tuple(out.shape)}")
    return out
