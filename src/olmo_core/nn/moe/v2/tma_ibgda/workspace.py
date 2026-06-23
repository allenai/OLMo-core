from __future__ import annotations

from dataclasses import dataclass

import torch

from .metadata import TmaIbgdaRouteMetadata


TMA_IBGDA_ROUTE_RECORD_BYTES = 32
TMA_IBGDA_WORKSPACE_ALIGNMENT = 128
TMA_IBGDA_DOORBELL_BYTES = 8
TMA_IBGDA_COMPLETION_BYTES = 8


@dataclass(frozen=True)
class TmaIbgdaWorkspacePlan:
    """Host-side workspace sizing estimate for the rowwise TMA/IBGDA backend."""

    ep_world_size: int
    rank_capacity: int
    hidden_size: int
    dtype: torch.dtype
    route_records_bytes: int
    rank_counts_bytes: int
    rank_offsets_bytes: int
    payload_window_bytes: int
    total_bytes: int


@dataclass(frozen=True)
class TmaIbgdaPeerWindowPlan:
    """Registered peer-window layout for the future TMA/IBGDA transport.

    Each EP rank owns one registered window of `rank_stride_bytes`. Peers use
    out-of-band bootstrap data to learn base addresses/RDMA keys, then index
    rank-local route records, BF16 payload rows, doorbells, and completion
    counters through the offsets captured here.
    """

    ep_world_size: int
    rank_capacity: int
    hidden_size: int
    dtype: torch.dtype
    alignment: int
    route_records_offset: int
    routes_per_rank_offset: int
    rank_offsets_offset: int
    overflow_by_rank_offset: int
    payload_window_offset: int
    send_doorbells_offset: int
    recv_completions_offset: int
    rank_stride_bytes: int
    total_peer_window_bytes: int
    route_records_bytes: int
    routes_per_rank_bytes: int
    rank_offsets_bytes: int
    overflow_by_rank_bytes: int
    payload_window_bytes_per_rank: int
    send_doorbells_bytes: int
    recv_completions_bytes: int

    @property
    def peer_window_offsets(self) -> tuple[int, ...]:
        return tuple(rank * self.rank_stride_bytes for rank in range(self.ep_world_size))


def _dtype_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def plan_tma_ibgda_workspace(
    metadata: TmaIbgdaRouteMetadata,
    *,
    hidden_size: int,
    dtype: torch.dtype = torch.bfloat16,
    route_record_bytes: int = TMA_IBGDA_ROUTE_RECORD_BYTES,
) -> TmaIbgdaWorkspacePlan:
    """Compute a conservative host-side workspace plan for the future kernels."""

    if hidden_size <= 0:
        raise RuntimeError(f"hidden_size must be > 0, got {hidden_size}")
    if route_record_bytes <= 0:
        raise RuntimeError(f"route_record_bytes must be > 0, got {route_record_bytes}")
    dtype_bytes = _dtype_size(dtype)
    route_records_bytes = metadata.num_routes * route_record_bytes
    rank_counts_bytes = metadata.ep_world_size * torch.empty((), dtype=torch.long).element_size()
    rank_offsets_bytes = (metadata.ep_world_size + 1) * torch.empty((), dtype=torch.long).element_size()
    payload_window_bytes = (
        metadata.ep_world_size * metadata.rank_capacity * hidden_size * dtype_bytes
    )
    total_bytes = route_records_bytes + rank_counts_bytes + rank_offsets_bytes + payload_window_bytes
    return TmaIbgdaWorkspacePlan(
        ep_world_size=metadata.ep_world_size,
        rank_capacity=metadata.rank_capacity,
        hidden_size=hidden_size,
        dtype=dtype,
        route_records_bytes=route_records_bytes,
        rank_counts_bytes=rank_counts_bytes,
        rank_offsets_bytes=rank_offsets_bytes,
        payload_window_bytes=payload_window_bytes,
        total_bytes=total_bytes,
    )


def plan_tma_ibgda_peer_windows(
    metadata: TmaIbgdaRouteMetadata,
    *,
    hidden_size: int,
    dtype: torch.dtype = torch.bfloat16,
    route_record_bytes: int = TMA_IBGDA_ROUTE_RECORD_BYTES,
    alignment: int = TMA_IBGDA_WORKSPACE_ALIGNMENT,
) -> TmaIbgdaPeerWindowPlan:
    """Plan the registered per-rank windows used by an IBGDA/RDMA transport."""

    if hidden_size <= 0:
        raise RuntimeError(f"hidden_size must be > 0, got {hidden_size}")
    if route_record_bytes <= 0:
        raise RuntimeError(f"route_record_bytes must be > 0, got {route_record_bytes}")
    if alignment <= 0 or alignment & (alignment - 1):
        raise RuntimeError(f"alignment must be a positive power of two, got {alignment}")

    long_bytes = torch.empty((), dtype=torch.long).element_size()
    dtype_bytes = _dtype_size(dtype)
    route_records_bytes = metadata.num_routes * route_record_bytes
    routes_per_rank_bytes = metadata.ep_world_size * long_bytes
    rank_offsets_bytes = (metadata.ep_world_size + 1) * long_bytes
    overflow_by_rank_bytes = metadata.ep_world_size
    payload_window_bytes_per_rank = metadata.rank_capacity * hidden_size * dtype_bytes
    send_doorbells_bytes = metadata.ep_world_size * TMA_IBGDA_DOORBELL_BYTES
    recv_completions_bytes = metadata.ep_world_size * TMA_IBGDA_COMPLETION_BYTES

    cursor = 0
    route_records_offset = cursor
    cursor += route_records_bytes
    cursor = _align(cursor, alignment)

    routes_per_rank_offset = cursor
    cursor += routes_per_rank_bytes
    cursor = _align(cursor, alignment)

    rank_offsets_offset = cursor
    cursor += rank_offsets_bytes
    cursor = _align(cursor, alignment)

    overflow_by_rank_offset = cursor
    cursor += overflow_by_rank_bytes
    cursor = _align(cursor, alignment)

    payload_window_offset = cursor
    cursor += payload_window_bytes_per_rank
    cursor = _align(cursor, alignment)

    send_doorbells_offset = cursor
    cursor += send_doorbells_bytes
    cursor = _align(cursor, alignment)

    recv_completions_offset = cursor
    cursor += recv_completions_bytes
    rank_stride_bytes = _align(cursor, alignment)

    return TmaIbgdaPeerWindowPlan(
        ep_world_size=metadata.ep_world_size,
        rank_capacity=metadata.rank_capacity,
        hidden_size=hidden_size,
        dtype=dtype,
        alignment=alignment,
        route_records_offset=route_records_offset,
        routes_per_rank_offset=routes_per_rank_offset,
        rank_offsets_offset=rank_offsets_offset,
        overflow_by_rank_offset=overflow_by_rank_offset,
        payload_window_offset=payload_window_offset,
        send_doorbells_offset=send_doorbells_offset,
        recv_completions_offset=recv_completions_offset,
        rank_stride_bytes=rank_stride_bytes,
        total_peer_window_bytes=metadata.ep_world_size * rank_stride_bytes,
        route_records_bytes=route_records_bytes,
        routes_per_rank_bytes=routes_per_rank_bytes,
        rank_offsets_bytes=rank_offsets_bytes,
        overflow_by_rank_bytes=overflow_by_rank_bytes,
        payload_window_bytes_per_rank=payload_window_bytes_per_rank,
        send_doorbells_bytes=send_doorbells_bytes,
        recv_completions_bytes=recv_completions_bytes,
    )
