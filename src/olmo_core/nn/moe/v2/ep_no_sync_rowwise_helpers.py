from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch

from olmo_core._nvtx import nvtx
from olmo_core.distributed.utils import get_rank, hide_from_torch, unhide_from_torch

from .checkpointing import is_checkpoint_recomputing

if TYPE_CHECKING:
    from olmo_core.nn.ddp.block import OLMoDDPTransformerBlock
    from olmo_core.train.common import ReduceType

try:
    _torch_compile_disable = torch.compiler.disable
except AttributeError:

    def _torch_compile_disable(fn):
        return fn


def reset_ep_no_sync_rowwise_metrics(block: OLMoDDPTransformerBlock) -> None:
    block._ep_no_sync_rowwise_drop_tokens_sum = None
    block._ep_no_sync_rowwise_total_tokens_sum = None
    block._ep_no_sync_rowwise_symm_util_max = None


def add_ep_no_sync_rowwise_metrics(
    block: OLMoDDPTransformerBlock,
    out: Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]],
    reduce_type_cls,
) -> None:
    if (
        block._ep_no_sync_rowwise_drop_tokens_sum is not None
        and block._ep_no_sync_rowwise_total_tokens_sum is not None
    ):
        drop_tokens_sum = unhide_from_torch(block._ep_no_sync_rowwise_drop_tokens_sum)
        total_tokens_sum = unhide_from_torch(block._ep_no_sync_rowwise_total_tokens_sum)
        drop_ratio = (
            drop_tokens_sum.to(dtype=torch.float32)
            / total_tokens_sum.to(dtype=torch.float32).clamp_min(1.0)
        ).clamp(0.0, 1.0)
        out["token drop rate"] = (drop_ratio, reduce_type_cls.mean)

    if block._ep_no_sync_rowwise_symm_util_max is not None:
        symm_util_max = unhide_from_torch(block._ep_no_sync_rowwise_symm_util_max)
        out["symm buffer util"] = (
            symm_util_max.to(dtype=torch.float32),
            reduce_type_cls.max,
        )


@_torch_compile_disable
def should_accumulate_ep_no_sync_rowwise_metrics(
    accumulate_routed_aux_loss_metrics: Optional[bool],
) -> bool:
    # True means original checkpoint forward, False means recompute. None asks
    # us to use the same best-effort dynamic signal as router metric
    # accumulation for compiled checkpointed rowwise blocks.
    if accumulate_routed_aux_loss_metrics is None:
        return not is_checkpoint_recomputing()
    return accumulate_routed_aux_loss_metrics


@_torch_compile_disable
def accumulate_ep_no_sync_rowwise_metrics(
    block: OLMoDDPTransformerBlock,
    *,
    drop_token_cnt: torch.Tensor,
    num_out_tokens: int,
    recv_splits_by_src_local: torch.Tensor,
    rank_capacity: int,
) -> None:
    if rank_capacity <= 0:
        return

    drop_sum = drop_token_cnt.detach().to(dtype=torch.float32)
    total_sum = torch.empty_like(drop_sum).fill_(num_out_tokens)
    util = recv_splits_by_src_local.detach().sum(dtype=torch.float32) * (1.0 / rank_capacity)

    if block._ep_no_sync_rowwise_drop_tokens_sum is None:
        block._ep_no_sync_rowwise_drop_tokens_sum = hide_from_torch(drop_sum)
    else:
        prev = unhide_from_torch(block._ep_no_sync_rowwise_drop_tokens_sum)
        block._ep_no_sync_rowwise_drop_tokens_sum = hide_from_torch(prev + drop_sum)

    if block._ep_no_sync_rowwise_total_tokens_sum is None:
        block._ep_no_sync_rowwise_total_tokens_sum = hide_from_torch(total_sum)
    else:
        prev = unhide_from_torch(block._ep_no_sync_rowwise_total_tokens_sum)
        block._ep_no_sync_rowwise_total_tokens_sum = hide_from_torch(prev + total_sum)

    if block._ep_no_sync_rowwise_symm_util_max is None:
        block._ep_no_sync_rowwise_symm_util_max = hide_from_torch(util)
    else:
        prev = unhide_from_torch(block._ep_no_sync_rowwise_symm_util_max)
        block._ep_no_sync_rowwise_symm_util_max = hide_from_torch(torch.maximum(prev, util))


@nvtx.annotate("_build_rowwise_route_maps")
def build_rowwise_route_maps(
    block: OLMoDDPTransformerBlock,
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
    dst_rank = torch.div(
        safe_experts,
        self.num_local_routed_experts,
        rounding_mode="floor",
    )
    prefix_by_source = torch.cumsum(keep_matrix, dim=0) - keep_matrix
    src_rank = get_rank(self.ep_pg)
    send_base_by_dest_local = prefix_by_source[src_rank]
    recv_total_by_dest_local = keep_matrix.sum(dim=0)
    local_expert_base_by_dest = (
        torch.cumsum(recv_total_by_dest_local, dim=1) - recv_total_by_dest_local
    )

    base_rows_by_expert = (local_expert_base_by_dest + send_base_by_dest_local).reshape(-1)

    # Stable in-expert route position (rank within each expert bucket, in route
    # order). A stable sort groups equal buckets while preserving route order, so
    # each route's position is its offset from the start of its group. This keeps
    # O(routes) memory instead of a dense [expert, route] matrix, which would be
    # (expert_count + 1) * num_routes and blow up for large expert counts.
    seq = torch.arange(num_routes, device=routing_map.device, dtype=torch.long)
    order = torch.argsort(bucket_ids, stable=True)
    sorted_buckets = bucket_ids.index_select(0, order)
    is_group_start = torch.ones(num_routes, device=routing_map.device, dtype=torch.bool)
    is_group_start[1:] = sorted_buckets[1:] != sorted_buckets[:-1]
    group_start = torch.cummax(
        torch.where(is_group_start, seq, torch.zeros_like(seq)), dim=0
    ).values
    pos_in_bucket = torch.empty(num_routes, device=routing_map.device, dtype=torch.long)
    pos_in_bucket.scatter_(0, order, seq - group_start)

    keep_limits = allowed_splits_i64.index_select(0, safe_experts)
    base_rows = base_rows_by_expert.index_select(0, safe_experts)
    kept_mask = valid_mask & (pos_in_bucket < keep_limits)

    dst_rows_all = base_rows + pos_in_bucket

    neg_ones = torch.full_like(dst_rank, -1)
    dst_ranks_flat = torch.where(kept_mask, dst_rank, neg_ones)
    dst_rows_flat = torch.where(kept_mask, dst_rows_all, neg_ones)
    return (
        dst_ranks_flat.view(num_tokens, top_k),
        dst_rows_flat.view(num_tokens, top_k),
    )
