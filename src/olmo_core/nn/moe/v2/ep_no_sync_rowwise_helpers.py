from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import nvtx
import torch

from olmo_core.distributed.utils import get_rank

if TYPE_CHECKING:
    from olmo_core.train.common import ReduceType

    from .block import MoEFusedV2TransformerBlock


def reset_ep_no_sync_rowwise_metrics(block: MoEFusedV2TransformerBlock) -> None:
    block._ep_no_sync_rowwise_drop_tokens_sum = None
    block._ep_no_sync_rowwise_total_tokens_sum = None
    block._ep_no_sync_rowwise_symm_util_max = None


def add_ep_no_sync_rowwise_metrics(
    block: MoEFusedV2TransformerBlock,
    out: Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]],
    reduce_type_cls,
) -> None:
    if (
        block._ep_no_sync_rowwise_drop_tokens_sum is not None
        and block._ep_no_sync_rowwise_total_tokens_sum is not None
    ):
        drop_ratio = (
            block._ep_no_sync_rowwise_drop_tokens_sum.to(dtype=torch.float32)
            / block._ep_no_sync_rowwise_total_tokens_sum.to(dtype=torch.float32).clamp_min(1.0)
        ).clamp(0.0, 1.0)
        out["token drop rate"] = (drop_ratio, reduce_type_cls.mean)

    if block._ep_no_sync_rowwise_symm_util_max is not None:
        out["symm buffer util"] = (
            block._ep_no_sync_rowwise_symm_util_max.to(dtype=torch.float32),
            reduce_type_cls.max,
        )


def accumulate_ep_no_sync_rowwise_metrics(
    block: MoEFusedV2TransformerBlock,
    *,
    drop_token_cnt: torch.Tensor,
    num_out_tokens: int,
    recv_splits_by_src_local: torch.Tensor,
    rank_capacity: int,
) -> None:
    if rank_capacity <= 0:
        return

    drop_sum = drop_token_cnt.to(dtype=torch.float32)
    total_sum = torch.tensor(float(num_out_tokens), device=drop_sum.device)
    util = recv_splits_by_src_local.sum(dtype=torch.float32) / torch.tensor(
        float(rank_capacity), device=drop_sum.device
    )

    if block._ep_no_sync_rowwise_drop_tokens_sum is None:
        block._ep_no_sync_rowwise_drop_tokens_sum = drop_sum
    else:
        block._ep_no_sync_rowwise_drop_tokens_sum = (
            block._ep_no_sync_rowwise_drop_tokens_sum + drop_sum
        )

    if block._ep_no_sync_rowwise_total_tokens_sum is None:
        block._ep_no_sync_rowwise_total_tokens_sum = total_sum
    else:
        block._ep_no_sync_rowwise_total_tokens_sum = (
            block._ep_no_sync_rowwise_total_tokens_sum + total_sum
        )

    if block._ep_no_sync_rowwise_symm_util_max is None:
        block._ep_no_sync_rowwise_symm_util_max = util
    else:
        block._ep_no_sync_rowwise_symm_util_max = torch.maximum(
            block._ep_no_sync_rowwise_symm_util_max,
            util,
        )


@nvtx.annotate("_build_rowwise_route_maps")
def build_rowwise_route_maps(
    block: MoEFusedV2TransformerBlock,
    *,
    routing_map: torch.Tensor,
    allowed_splits: torch.Tensor,
    keep_from_src_dest_local: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build per-route destination rank/row maps for row-wise dispatch.
    Routes dropped by tail-capacity are encoded as -1.
    """
    self = block
    assert self.ep_pg is not None
    assert self.num_local_routed_experts is not None
    if routing_map.ndim != 2:
        raise RuntimeError(
            f"routing_map must be rank-2 [N, K], got shape={tuple(routing_map.shape)}"
        )

    num_tokens, top_k = routing_map.shape
    num_routes = num_tokens * top_k
    expert_count = self.ep_world_size * self.num_local_routed_experts

    if allowed_splits.numel() != expert_count:
        raise RuntimeError(
            "allowed_splits size mismatch: "
            f"got {allowed_splits.numel()}, expected {expert_count}"
        )
    allowed_splits_i64 = allowed_splits.to(dtype=torch.long)

    expected_keep_shape = (
        self.ep_world_size,
        self.ep_world_size,
        self.num_local_routed_experts,
    )
    if tuple(keep_from_src_dest_local.shape) != expected_keep_shape:
        raise RuntimeError(
            "keep_from_src_dest_local shape mismatch: "
            f"got {tuple(keep_from_src_dest_local.shape)}, expected {expected_keep_shape}"
        )
    keep_matrix = keep_from_src_dest_local.to(dtype=torch.long)
    if num_routes == 0:
        dst_ranks_flat = torch.full(
            (0,),
            -1,
            dtype=torch.long,
            device=routing_map.device,
        )
        dst_rows_flat = torch.full(
            (0,),
            -1,
            dtype=torch.long,
            device=routing_map.device,
        )
        return (
            dst_ranks_flat.view(num_tokens, top_k),
            dst_rows_flat.view(num_tokens, top_k),
        )

    route_experts = routing_map.reshape(-1).to(dtype=torch.long)
    valid_mask = (route_experts >= 0) & (route_experts < expert_count)
    safe_experts = torch.where(
        valid_mask,
        route_experts,
        torch.zeros_like(route_experts),
    )

    # Compute stable in-expert position for each route without dynamic-shape
    # indexing (avoids host sync from nonzero/item on CUDA tensors).
    invalid_bucket = expert_count
    bucket_ids = torch.where(
        valid_mask,
        safe_experts,
        torch.full_like(safe_experts, invalid_bucket),
    )
    sort_order = torch.argsort(bucket_ids, stable=True)
    sorted_bucket_ids = bucket_ids.index_select(0, sort_order)
    counts_per_bucket = torch.zeros(
        (expert_count + 1,),
        device=routing_map.device,
        dtype=torch.long,
    )
    counts_per_bucket.scatter_add_(
        0,
        bucket_ids,
        torch.ones_like(bucket_ids, dtype=torch.long),
    )
    starts_per_bucket = torch.cumsum(counts_per_bucket, dim=0) - counts_per_bucket
    sorted_pos = torch.arange(
        num_routes,
        device=routing_map.device,
        dtype=torch.long,
    ) - starts_per_bucket.index_select(0, sorted_bucket_ids)

    pos_in_bucket = torch.empty_like(sorted_pos)
    pos_in_bucket.scatter_(0, sort_order, sorted_pos)

    keep_limits = allowed_splits_i64.index_select(0, safe_experts)
    kept_mask = valid_mask & (pos_in_bucket < keep_limits)

    dst_rank = torch.div(
        safe_experts,
        self.num_local_routed_experts,
        rounding_mode="floor",
    )
    dst_local_expert = torch.remainder(
        safe_experts,
        self.num_local_routed_experts,
    )

    prefix_by_source = torch.cumsum(keep_matrix, dim=0) - keep_matrix
    src_rank = get_rank(self.ep_pg)
    send_base_by_dest_local = prefix_by_source[src_rank]
    recv_total_by_dest_local = keep_matrix.sum(dim=0)
    local_expert_base_by_dest = (
        torch.cumsum(recv_total_by_dest_local, dim=1) - recv_total_by_dest_local
    )

    base_rows = local_expert_base_by_dest[dst_rank, dst_local_expert]
    base_rows = base_rows + send_base_by_dest_local[dst_rank, dst_local_expert]
    dst_rows_all = base_rows + pos_in_bucket

    neg_ones = torch.full_like(dst_rank, -1)
    dst_ranks_flat = torch.where(kept_mask, dst_rank, neg_ones)
    dst_rows_flat = torch.where(kept_mask, dst_rows_all, neg_ones)
    return (
        dst_ranks_flat.view(num_tokens, top_k),
        dst_rows_flat.view(num_tokens, top_k),
    )
