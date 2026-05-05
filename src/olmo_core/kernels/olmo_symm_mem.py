from __future__ import annotations

import os
from typing import Iterable

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

from .symm_mem_vdev2d import _load_cuda_extension


_BOOTSTRAP_GLOBAL_RANKS: tuple[int, ...] | None = None
_REGISTERED_GROUPS: set[str] = set()


def is_enabled() -> bool:
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


def init(group: dist.ProcessGroup, *, device: torch.device | str | int | None = None) -> None:
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

    unique_ids = _all_gather_unique_ids(group)
    ext = _load_cuda_extension()
    ext.olmo_symm_init(
        unique_ids,
        dist.get_rank(group),
        dist.get_world_size(group),
        resolved_device.index,
    )
    _BOOTSTRAP_GLOBAL_RANKS = global_ranks


def register_group(group: dist.ProcessGroup, *, device: torch.device | str | int | None = None) -> None:
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
    resolved_device = _ensure_cuda_device(device)
    register_group(group, device=resolved_device)
    ext = _load_cuda_extension()
    return ext.olmo_symm_empty(tuple(int(dim) for dim in shape), dtype, resolved_device)


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
    register_group(group, device=tensor.device)
    if barrier:
        dist.barrier(group=group)
