from __future__ import annotations

from dataclasses import dataclass
import socket
import threading
from typing import Any, Optional

import torch
import torch.distributed as dist

from .metadata import TmaIbgdaRouteMetadata, build_tma_ibgda_route_metadata


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


def is_tma_ibgda_backend_available() -> bool:
    """Return whether compiled TMA/IBGDA kernels are available."""

    try:
        from olmo_core.kernels import tma_ibgda_ep
    except Exception:
        return False
    return tma_ibgda_ep.is_available()


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
    return tma_ibgda_ep


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
) -> torch.Tensor:
    if world_size <= 1:
        return torch.empty(shape, device=device, dtype=dtype)
    if process_group is None:
        raise TmaIbgdaBackendUnavailable(
            "multi-rank TMA/IBGDA buffers require an explicit process group"
        )
    if config.reuse_symmetric_buffers:
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
) -> Optional[TmaIbgdaRoutePreprocess]:
    if not config.use_gpu_route_preprocess or rank_capacity is None:
        return None
    tma_ibgda_ep = _require_kernel_module()
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
    preprocess = TmaIbgdaRoutePreprocess(
        route_records=route_records,
        routes_per_rank=routes_per_rank,
        rank_offsets=rank_offsets,
        overflow_by_rank=overflow_by_rank,
        route_ordinals=route_ordinals,
        errors=errors,
    )
    if config.validate_gpu_route_preprocess:
        _check_gpu_preprocess_errors(preprocess)
    return preprocess


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
    if preprocess is None:
        preprocess = _preprocess_routes_on_gpu(
            dst_ranks,
            dst_rows,
            probs=probs,
            ep_world_size=ep_world_size,
            rank_capacity=rank_capacity,
            config=config,
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
    tma_ibgda_ep, world_size = _prepare_direct_peer_transport(
        ep_world_size=ep_world_size,
        process_group=process_group,
        device=input.device,
        config=config,
    )
    dispatch_out = _empty_transport_buffer(
        (metadata.rank_capacity, input.shape[1]),
        device=input.device,
        dtype=input.dtype,
        process_group=process_group,
        world_size=world_size,
        kind="dispatch",
        config=config,
    )
    if world_size > 1:
        dispatch_nblocks = _resolve_ibgda_nblocks(
            config.num_sms_dispatch,
            world_size=world_size,
            operation="dispatch",
        )
        if preprocess is not None:
            if config.use_tma_load_dispatch:
                tma_ibgda_ep.dispatch_bf16_ibgda_records_tma(
                    input,
                    dispatch_out,
                    preprocess.route_records,
                    nblocks=dispatch_nblocks,
                )
            else:
                tma_ibgda_ep.dispatch_bf16_ibgda_records(
                    input,
                    dispatch_out,
                    preprocess.route_records,
                    nblocks=dispatch_nblocks,
                )
        else:
            tma_ibgda_ep.dispatch_bf16_ibgda(
                input,
                dispatch_out,
                dst_ranks,
                dst_rows,
                probs,
                nblocks=dispatch_nblocks,
            )
        peer_out_ptrs = torch.empty((0,), dtype=torch.long, device=input.device)
    else:
        peer_out_ptrs = _all_gather_cuda_i64(
            dispatch_out.data_ptr(),
            world_size,
            input.device,
            process_group,
        )
        tma_ibgda_ep.dispatch_bf16_peer(
            input,
            dispatch_out,
            dst_ranks,
            dst_rows,
            peer_out_ptrs,
            probs,
            nblocks=config.num_sms_dispatch or 0,
        )
    if config.synchronize_after_dispatch:
        _transport_barrier(
            process_group,
            input.device,
            world_size=world_size,
            config=config,
            signal_kind="dispatch_out",
        )
    elif world_size > 1 and config.reuse_symmetric_buffers and not config.return_symmetric_dispatch_buffer:
        raise RuntimeError(
            "TMA/IBGDA dispatch must synchronize before cloning from the symmetric "
            "transport window. Set synchronize_after_dispatch=True or "
            "return_symmetric_dispatch_buffer=True."
        )
    if world_size > 1 and config.reuse_symmetric_buffers and not config.return_symmetric_dispatch_buffer:
        dispatch_out = dispatch_out.clone(memory_format=torch.contiguous_format)
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
    tma_ibgda_ep, world_size = _prepare_direct_peer_transport(
        ep_world_size=ep_world_size,
        process_group=process_group,
        device=expert_out.device,
        config=config,
    )
    if config.synchronize_before_combine:
        _transport_barrier(
            process_group,
            expert_out.device,
            world_size=world_size,
            config=config,
            signal_kind="pre_combine",
        )

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
            )
            expert_transport.copy_(expert_out)
        _transport_barrier(
            process_group,
            expert_out.device,
            world_size=world_size,
            config=config,
            signal_kind="combine_expert_out",
        )
        if preprocess is not None:
            tma_ibgda_ep.combine_bf16_ibgda_records(
                expert_transport,
                out,
                preprocess.route_records,
                top_k=metadata.top_k,
                nblocks=combine_nblocks,
            )
        else:
            tma_ibgda_ep.combine_bf16_ibgda(
                expert_transport,
                out,
                src_ranks,
                src_rows,
                probs,
                nblocks=combine_nblocks,
            )
    else:
        peer_expert_out_ptrs = _all_gather_cuda_i64(
            expert_out.data_ptr(),
            world_size,
            expert_out.device,
            process_group,
        )
        tma_ibgda_ep.combine_bf16_peer(
            expert_out,
            out,
            src_ranks,
            src_rows,
            peer_expert_out_ptrs,
            probs,
        )
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
    tma_ibgda_ep, world_size = _prepare_direct_peer_transport(
        ep_world_size=ep_world_size,
        process_group=process_group,
        device=expert_out.device,
        config=config,
    )
    if config.synchronize_before_combine:
        _transport_barrier(
            process_group,
            expert_out.device,
            world_size=world_size,
            config=config,
            signal_kind="pre_route_dot",
        )
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
            )
            expert_transport.copy_(expert_out)
        _transport_barrier(
            process_group,
            expert_out.device,
            world_size=world_size,
            config=config,
            signal_kind="route_dot_expert_out",
        )
        if preprocess is not None:
            tma_ibgda_ep.route_dot_bf16_ibgda_records(
                expert_transport,
                grad_out,
                preprocess.route_records,
                top_k=src_ranks.shape[1],
                out=out,
            )
        else:
            tma_ibgda_ep.route_dot_bf16_ibgda(
                expert_transport,
                grad_out,
                src_ranks,
                src_rows,
                out,
            )
    else:
        peer_expert_out_ptrs = _all_gather_cuda_i64(
            expert_out.data_ptr(),
            world_size,
            expert_out.device,
            process_group,
        )
        tma_ibgda_ep.route_dot_bf16_peer(
            expert_out,
            grad_out,
            src_ranks,
            src_rows,
            peer_expert_out_ptrs,
            out,
        )
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
        tensors: list[torch.Tensor] = [dst_ranks, dst_rows]
        if probs is not None:
            tensors.extend([probs, input])
        ctx.save_for_backward(*tensors)
        ctx.has_probs = probs is not None
        ctx.ep_world_size = int(ep_world_size)
        ctx.rank_capacity = metadata.rank_capacity
        ctx.process_group = process_group
        ctx.config = config
        ctx.preprocess = _preprocess
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
        return grad_input, None, None, grad_probs, None, None, None, None, None


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
                expert_out_is_symmetric=False,
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
    preprocess = _preprocess_routes_on_gpu(
        dst_ranks,
        dst_rows,
        probs=probs,
        ep_world_size=ep_world_size,
        rank_capacity=rank_capacity,
        config=cfg,
    )
    if preprocess is not None and rank_capacity is not None:
        metadata = _metadata_from_preprocess(
            dst_ranks,
            dst_rows,
            ep_world_size=ep_world_size,
            rank_capacity=int(rank_capacity),
            static_route_budget=cfg.static_route_budget,
            preprocess=preprocess,
        )
    else:
        metadata = build_tma_ibgda_route_metadata(
            dst_ranks,
            dst_rows,
            ep_world_size=ep_world_size,
            rank_capacity=rank_capacity,
            static_route_budget=cfg.static_route_budget,
        )
    dispatch_out = _TmaIbgdaDispatchAutograd.apply(
        input,
        dst_ranks,
        dst_rows,
        probs,
        ep_world_size,
        metadata.rank_capacity,
        process_group,
        cfg,
        preprocess,
    )

    return dispatch_out, TmaIbgdaDispatchHandle(
        metadata=metadata,
        group_name=group_name,
        config=cfg,
        process_group=process_group,
        peer_out_ptrs=torch.empty((0,), dtype=torch.long, device=input.device),
        preprocess=preprocess,
    )


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
    del group_name
    return _TmaIbgdaCombineAutograd.apply(
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
