from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class TmaIbgdaRouteMetadata:
    """Route metadata derived from OLMo rowwise `[N, K]` rank/row maps."""

    dst_ranks: torch.Tensor
    dst_rows: torch.Tensor
    valid_mask: torch.Tensor
    source_rows: torch.Tensor
    topk_slots: torch.Tensor
    routes_per_rank: torch.Tensor
    rank_offsets: torch.Tensor
    route_ordinals: torch.Tensor
    overflow_by_rank: torch.Tensor
    num_tokens: int
    top_k: int
    ep_world_size: int
    rank_capacity: int
    static_route_budget: Optional[int]

    @property
    def num_routes(self) -> int:
        return self.num_tokens * self.top_k


def _validate_route_maps(
    dst_ranks: torch.Tensor,
    dst_rows: torch.Tensor,
    *,
    ep_world_size: int,
    rank_capacity: Optional[int],
) -> None:
    if ep_world_size <= 0:
        raise RuntimeError(f"ep_world_size must be > 0, got {ep_world_size}")
    if rank_capacity is not None and rank_capacity <= 0:
        raise RuntimeError(f"rank_capacity must be > 0 when provided, got {rank_capacity}")
    if dst_ranks.ndim != 2 or dst_rows.ndim != 2:
        raise RuntimeError(
            f"dst_ranks and dst_rows must be rank-2 [N, K], got "
            f"{tuple(dst_ranks.shape)} and {tuple(dst_rows.shape)}"
        )
    if tuple(dst_ranks.shape) != tuple(dst_rows.shape):
        raise RuntimeError(
            f"dst_ranks/dst_rows shape mismatch: {tuple(dst_ranks.shape)} "
            f"!= {tuple(dst_rows.shape)}"
        )
    if dst_ranks.device != dst_rows.device:
        raise RuntimeError("dst_ranks and dst_rows must be on the same device")
    if dst_ranks.dtype != torch.long or dst_rows.dtype != torch.long:
        raise RuntimeError("dst_ranks and dst_rows must be torch.long")
    if not dst_ranks.is_contiguous() or not dst_rows.is_contiguous():
        raise RuntimeError("dst_ranks and dst_rows must be contiguous")


def build_tma_ibgda_route_metadata(
    dst_ranks: torch.Tensor,
    dst_rows: torch.Tensor,
    *,
    ep_world_size: int,
    rank_capacity: Optional[int] = None,
    static_route_budget: Optional[int] = None,
) -> TmaIbgdaRouteMetadata:
    """Build the backend-neutral rowwise metadata for the TMA/IBGDA transport.

    Dropped routes must be encoded with both rank and row negative. Valid routes
    are checked against `ep_world_size` and `rank_capacity` when provided.
    """

    _validate_route_maps(
        dst_ranks,
        dst_rows,
        ep_world_size=ep_world_size,
        rank_capacity=rank_capacity,
    )
    if static_route_budget is not None and static_route_budget <= 0:
        raise RuntimeError(
            f"static_route_budget must be > 0 when provided, got {static_route_budget}"
        )

    num_tokens, top_k = dst_ranks.shape
    device = dst_ranks.device
    num_routes = num_tokens * top_k
    ranks_flat = dst_ranks.reshape(-1)
    rows_flat = dst_rows.reshape(-1)
    rank_dropped = ranks_flat < 0
    row_dropped = rows_flat < 0
    mismatched_drop = rank_dropped ^ row_dropped
    if bool(mismatched_drop.any().item()):
        raise RuntimeError("dropped routes must have both rank and row negative")

    valid_mask = ~(rank_dropped | row_dropped)
    if bool(((ranks_flat >= ep_world_size) & valid_mask).any().item()):
        raise RuntimeError("dst_ranks contains a valid route outside ep_world_size")

    inferred_capacity = (
        int(rows_flat[valid_mask].max().item()) + 1 if bool(valid_mask.any()) else 1
    )
    capacity = rank_capacity if rank_capacity is not None else inferred_capacity
    if bool(((rows_flat >= capacity) & valid_mask).any().item()):
        raise RuntimeError("dst_rows contains a valid route outside rank_capacity")

    route_ids = torch.arange(num_routes, device=device, dtype=torch.long)
    source_rows = torch.div(route_ids, top_k, rounding_mode="floor").view(num_tokens, top_k)
    topk_slots = (route_ids % top_k).view(num_tokens, top_k)

    safe_ranks = torch.where(valid_mask, ranks_flat, torch.zeros_like(ranks_flat))
    route_increments = valid_mask.to(dtype=torch.long)
    routes_per_rank = torch.zeros(ep_world_size, dtype=torch.long, device=device)
    if num_routes > 0:
        routes_per_rank.scatter_add_(0, safe_ranks, route_increments)

    rank_offsets = torch.zeros(ep_world_size + 1, dtype=torch.long, device=device)
    if ep_world_size > 0:
        rank_offsets[1:] = torch.cumsum(routes_per_rank, dim=0)

    route_ordinals = torch.full((num_routes,), -1, dtype=torch.long, device=device)
    if num_routes > 0:
        sort_keys = torch.where(
            valid_mask,
            safe_ranks,
            torch.full_like(safe_ranks, ep_world_size),
        )
        sort_order = torch.argsort(sort_keys, stable=True)
        sorted_valid = valid_mask[sort_order]
        sorted_ranks = safe_ranks[sort_order]
        sorted_ordinals = torch.arange(num_routes, device=device, dtype=torch.long)
        sorted_ordinals = sorted_ordinals - rank_offsets[sorted_ranks]
        sorted_ordinals = torch.where(
            sorted_valid,
            sorted_ordinals,
            torch.full_like(sorted_ordinals, -1),
        )
        route_ordinals[sort_order] = sorted_ordinals

    if static_route_budget is None:
        overflow_by_rank = torch.zeros(ep_world_size, dtype=torch.bool, device=device)
    else:
        overflow_by_rank = routes_per_rank > static_route_budget

    return TmaIbgdaRouteMetadata(
        dst_ranks=dst_ranks,
        dst_rows=dst_rows,
        valid_mask=valid_mask.view(num_tokens, top_k),
        source_rows=source_rows,
        topk_slots=topk_slots,
        routes_per_rank=routes_per_rank,
        rank_offsets=rank_offsets,
        route_ordinals=route_ordinals.view(num_tokens, top_k),
        overflow_by_rank=overflow_by_rank,
        num_tokens=num_tokens,
        top_k=top_k,
        ep_world_size=ep_world_size,
        rank_capacity=capacity,
        static_route_budget=static_route_budget,
    )
