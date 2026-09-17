"""Manage NVSHMEM initialization, group registration and symmetric tensor allocation."""

from __future__ import annotations

import os
from typing import Iterable

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

from olmo_core.distributed.utils import get_node_hostname

from .symm_mem_vdev2d import _load_cuda_extension

_BOOTSTRAP_GLOBAL_RANKS: tuple[int, ...] | None = None
_REGISTERED_GROUPS: set[str] = set()


def is_enabled() -> bool:
    """Whether to use the OLMo-owned symmetric-memory runtime."""
    raw = os.getenv("OLMO_USE_OWN_SYMM_MEM", "1")
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _group_global_ranks(group: dist.ProcessGroup) -> tuple[int, ...]:
    ranks = c10d._world.pg_group_ranks[group]
    return tuple(global_rank for global_rank, _ in sorted(ranks.items(), key=lambda item: item[1]))


def _ensure_cuda_device(device: torch.device | str | int | None) -> torch.device:
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("OLMo symmetric memory requires CUDA")
        return torch.device("cuda", torch.cuda.current_device())
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise RuntimeError(f"OLMo symmetric memory requires a CUDA device, got {resolved}")
    if resolved.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return resolved


def _all_gather_unique_ids(group: dist.ProcessGroup) -> list[list[int]]:
    ext = _load_cuda_extension()
    uid = list(ext.olmo_symm_get_unique_id())
    gathered: list[list[int] | None] = [None for _ in range(dist.get_world_size(group))]
    dist.all_gather_object(gathered, uid, group=group)
    if any(item is None for item in gathered):
        raise RuntimeError("Failed to gather NVSHMEM unique IDs for OLMo symmetric memory")
    return [list(item) for item in gathered if item is not None]


def _validate_local_only_group(group: dist.ProcessGroup, device: torch.device) -> None:
    host, active_uuid, error = "", "", ""
    reachable: set[str] = set()
    try:
        host = get_node_hostname()
        if not host:
            raise RuntimeError("could not determine the node hostname")
        device_uuids = []
        for index in range(torch.cuda.device_count()):
            uuid = getattr(torch.cuda.get_device_properties(index), "uuid", None)
            if uuid is None or not str(uuid):
                raise RuntimeError(f"CUDA device {index} has no GPU UUID")
            device_uuids.append(str(uuid))
        assert device.index is not None
        active_uuid = device_uuids[device.index]
        reachable = {
            uuid
            for index, uuid in enumerate(device_uuids)
            if index == device.index or torch.cuda.can_device_access_peer(device.index, index)
        }
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    # Exchange inspection errors too, so no rank proceeds to NVSHMEM after a peer fails.
    gathered: list[tuple[str, str, set[str], str] | None] = [
        None for _ in range(dist.get_world_size(group))
    ]
    dist.all_gather_object(gathered, (host, active_uuid, reachable, error), group=group)
    records = [item for item in gathered if item is not None]
    errors = [f"group rank {rank}: {item[3]}" for rank, item in enumerate(records) if item[3]]
    uuids = [item[1] for item in records]
    if len(records) != len(gathered):
        reason = "missing peer topology information"
    elif errors:
        reason = "; ".join(errors)
    elif len({item[0] for item in records}) != 1:
        reason = "the process group spans multiple hosts"
    elif len(set(uuids)) != len(uuids):
        reason = "multiple ranks use the same GPU UUID"
    else:
        inaccessible = [
            (rank, peer)
            for rank, item in enumerate(records)
            for peer, uuid in enumerate(uuids)
            if uuid not in item[2]
        ]
        reason = f"directed CUDA P2P access is unavailable for group rank pairs {inaccessible}"
        if not inaccessible:
            return
    raise RuntimeError(
        "NVSHMEM_REMOTE_TRANSPORT=none requires a single-host process group with distinct "
        f"GPUs and full CUDA P2P access: {reason}"
    )


def init(group: dist.ProcessGroup, *, device: torch.device | str | int | None = None) -> None:
    """Initialize NVSHMEM once per process for the ranks in ``group``.

    Explicit ``NVSHMEM_REMOTE_TRANSPORT=none`` requires a single host with distinct,
    mutually peer-accessible GPUs. All group ranks must use the same transport setting.
    """
    global _BOOTSTRAP_GLOBAL_RANKS

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("OLMo symmetric memory requires torch.distributed to be initialized")

    resolved_device = _ensure_cuda_device(device)
    global_ranks = _group_global_ranks(group)
    if _BOOTSTRAP_GLOBAL_RANKS is not None:
        if _BOOTSTRAP_GLOBAL_RANKS != global_ranks:
            raise RuntimeError(
                "OLMo symmetric memory is already initialized for ranks "
                f"{_BOOTSTRAP_GLOBAL_RANKS}, cannot reinitialize for ranks {global_ranks}. "
                "Use one NVSHMEM bootstrap group per process."
            )
        return

    if os.getenv("NVSHMEM_REMOTE_TRANSPORT", "").strip().lower() == "none":
        _validate_local_only_group(group, resolved_device)

    unique_ids = _all_gather_unique_ids(group)
    ext = _load_cuda_extension()
    ext.olmo_symm_init(
        unique_ids,
        dist.get_rank(group),
        dist.get_world_size(group),
        resolved_device.index,
    )
    _BOOTSTRAP_GLOBAL_RANKS = global_ranks


def register_group(
    group: dist.ProcessGroup, *, device: torch.device | str | int | None = None
) -> None:
    """Register the group's rank-to-PE mapping with the symmetric-memory runtime."""
    init(group, device=device)
    assert _BOOTSTRAP_GLOBAL_RANKS is not None

    group_name = group.group_name
    if group_name in _REGISTERED_GROUPS:
        return

    bootstrap_pe_by_global_rank = {rank: pe for pe, rank in enumerate(_BOOTSTRAP_GLOBAL_RANKS)}
    rank_to_pe = []
    for global_rank in _group_global_ranks(group):
        try:
            rank_to_pe.append(bootstrap_pe_by_global_rank[global_rank])
        except KeyError as e:
            raise RuntimeError(
                f"Group {group_name!r} contains global rank {global_rank}, which is not in "
                f"the OLMo symmetric-memory bootstrap group {_BOOTSTRAP_GLOBAL_RANKS}"
            ) from e

    ext = _load_cuda_extension()
    ext.olmo_symm_register_group(group_name, rank_to_pe)
    _REGISTERED_GROUPS.add(group_name)


def _require_bootstrap_world_group(group: dist.ProcessGroup) -> None:
    assert _BOOTSTRAP_GLOBAL_RANKS is not None

    global_ranks = _group_global_ranks(group)
    if global_ranks != _BOOTSTRAP_GLOBAL_RANKS:
        raise RuntimeError(
            "OLMo symmetric-memory NVSHMEM barrier currently supports only the "
            f"bootstrap world ranks {_BOOTSTRAP_GLOBAL_RANKS}, got group ranks {global_ranks}. "
            "Use per-kernel group barriers for registered subgroups; exposing true NVSHMEM "
            "subgroup barriers requires creating and caching NVSHMEM teams."
        )


def empty(
    shape: Iterable[int],
    *,
    dtype: torch.dtype,
    device: torch.device | str | int | None = None,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Allocate a symmetric CUDA tensor after registering its process group."""
    resolved_device = _ensure_cuda_device(device)
    register_group(group, device=resolved_device)
    ext = _load_cuda_extension()
    return ext.olmo_symm_empty(tuple(int(dim) for dim in shape), dtype, resolved_device)


def peer_base_ptrs(
    tensor: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    barrier: bool = True,
) -> torch.Tensor:
    """Return device int64 direct peer base pointers for a symmetric tensor.

    This is a setup/prewarm primitive for CUDA-owned peer-window kernels. It
    registers the group, optionally waits until all ranks have allocated their
    matching symmetric tensor, then returns a device tensor indexed by group
    rank. The extension fails closed when a peer is not directly addressable.
    """

    register_group(group, device=tensor.device)
    if barrier:
        dist.barrier(group=group)
    ext = _load_cuda_extension()
    return ext.olmo_symm_peer_base_ptrs(tensor, group.group_name)


def barrier(group: dist.ProcessGroup, *, device: torch.device | str | int | None = None) -> None:
    """Enqueue an NVSHMEM barrier for the OLMo bootstrap world on the current CUDA stream."""
    if _BOOTSTRAP_GLOBAL_RANKS is None:
        init(group, device=device)
    _require_bootstrap_world_group(group)
    ext = _load_cuda_extension()
    ext.olmo_symm_world_barrier()


def rendezvous(
    tensor: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    barrier: bool = True,
) -> None:
    """Register a tensor's process group and optionally synchronize all group ranks."""
    register_group(group, device=tensor.device)
    if barrier:
        dist.barrier(group=group)
