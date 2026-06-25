from __future__ import annotations

import os
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import nvtx
import torch
import torch.distributed as dist

from olmo_core.distributed.utils import get_rank

if TYPE_CHECKING:
    from olmo_core.nn.ddp.block import OLMoDDPTransformerBlock


def _rowwise_route_debug_enabled() -> bool:
    verbose = (
        os.getenv("OLMO_ROWWISE_VERBOSE_DEBUG_PRINT")
        or os.getenv("OLMO_TBO_VERBOSE_DEBUG_PRINT", "0")
    )
    if verbose.strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return False
    ranks = os.getenv("OLMO_ROWWISE_DEBUG_RANKS") or os.getenv("OLMO_TBO_DEBUG_RANKS")
    if not ranks or not dist.is_available() or not dist.is_initialized():
        return True
    rank = str(dist.get_rank())
    return rank in {part.strip() for part in ranks.split(",") if part.strip()}


def _rowwise_route_rank_tag() -> str:
    if not dist.is_available() or not dist.is_initialized():
        return "rank=? local_rank=?"
    return f"rank={dist.get_rank()} local_rank={os.getenv('LOCAL_RANK', '?')}"


def _rowwise_route_tensor_desc(name: str, tensor: Optional[torch.Tensor]) -> str:
    if tensor is None:
        return f"{name}=None"
    return f"{name}=tensor"


def _rowwise_route_debug_print(
    block: OLMoDDPTransformerBlock,
    label: str,
    **tensors: Optional[torch.Tensor],
) -> None:
    if not _rowwise_route_debug_enabled():
        return
    parts = [
        "[OLMO_ROWWISE_ROUTE_DEBUG]",
        _rowwise_route_rank_tag(),
        f"block={block.block_idx}",
        f"route_maps:{label}",
    ]
    parts.extend(_rowwise_route_tensor_desc(name, tensor) for name, tensor in tensors.items())
    print(" | ".join(str(part) for part in parts), flush=True)


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
    block: OLMoDDPTransformerBlock,
    *,
    drop_token_cnt: torch.Tensor,
    num_out_tokens: int,
    recv_splits_by_src_local: torch.Tensor,
    rank_capacity: int,
) -> None:
    if rank_capacity <= 0:
        return

    drop_sum = drop_token_cnt.to(dtype=torch.float32)
    total_sum = torch.empty_like(drop_sum).fill_(num_out_tokens)
    util = recv_splits_by_src_local.sum(dtype=torch.float32) * (1.0 / rank_capacity)

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

    # _rowwise_route_debug_print(
    #     self,
    #     "enter",
    #     routing_map=routing_map,
    #     allowed_splits=allowed_splits,
    #     keep_from_src_dest_local=keep_from_src_dest_local,
    # )
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

    # _rowwise_route_debug_print(self, "route_experts-enter", routing_map=routing_map)
    route_experts = routing_map.reshape(-1).to(dtype=torch.long)
    # _rowwise_route_debug_print(self, "route_experts-exit", route_experts=route_experts)
    # _rowwise_route_debug_print(self, "valid-mask-enter", route_experts=route_experts)
    valid_mask = (route_experts >= 0) & (route_experts < expert_count)
    # _rowwise_route_debug_print(self, "valid-mask-exit", valid_mask=valid_mask)
    # _rowwise_route_debug_print(self, "safe-experts-enter", route_experts=route_experts, valid_mask=valid_mask)
    safe_experts = torch.where(
        valid_mask,
        route_experts,
        torch.zeros_like(route_experts),
    )
    # _rowwise_route_debug_print(self, "safe-experts-exit", safe_experts=safe_experts)

    # Compute stable in-expert position for each route without dynamic-shape
    # indexing (avoids host sync from nonzero/item on CUDA tensors).
    invalid_bucket = expert_count
    # _rowwise_route_debug_print(self, "bucket-ids-enter", valid_mask=valid_mask, safe_experts=safe_experts)
    bucket_ids = torch.where(
        valid_mask,
        safe_experts,
        torch.full_like(safe_experts, invalid_bucket),
    )
    # _rowwise_route_debug_print(self, "bucket-ids-exit", bucket_ids=bucket_ids)
    # _rowwise_route_debug_print(self, "argsort-enter", bucket_ids=bucket_ids)
    sort_order = torch.argsort(bucket_ids, stable=True)
    # _rowwise_route_debug_print(self, "argsort-exit", sort_order=sort_order)
    # _rowwise_route_debug_print(self, "sorted-buckets-enter", bucket_ids=bucket_ids, sort_order=sort_order)
    sorted_bucket_ids = bucket_ids.index_select(0, sort_order)
    # _rowwise_route_debug_print(self, "sorted-buckets-exit", sorted_bucket_ids=sorted_bucket_ids)
    counts_per_bucket = torch.zeros(
        (expert_count + 1,),
        device=routing_map.device,
        dtype=torch.long,
    )
    # _rowwise_route_debug_print(self, "scatter-counts-enter", counts_per_bucket=counts_per_bucket, bucket_ids=bucket_ids)
    counts_per_bucket.scatter_add_(
        0,
        bucket_ids,
        torch.ones_like(bucket_ids, dtype=torch.long),
    )
    # _rowwise_route_debug_print(self, "scatter-counts-exit", counts_per_bucket=counts_per_bucket)
    # _rowwise_route_debug_print(self, "bucket-cumsum-enter", counts_per_bucket=counts_per_bucket)
    starts_per_bucket = torch.cumsum(counts_per_bucket, dim=0) - counts_per_bucket
    # _rowwise_route_debug_print(self, "bucket-cumsum-exit", starts_per_bucket=starts_per_bucket)
    # _rowwise_route_debug_print(self, "sorted-pos-enter", starts_per_bucket=starts_per_bucket, sorted_bucket_ids=sorted_bucket_ids)
    sorted_pos = torch.arange(
        num_routes,
        device=routing_map.device,
        dtype=torch.long,
    ) - starts_per_bucket.index_select(0, sorted_bucket_ids)
    # _rowwise_route_debug_print(self, "sorted-pos-exit", sorted_pos=sorted_pos)

    pos_in_bucket = torch.empty_like(sorted_pos)
    # _rowwise_route_debug_print(self, "pos-scatter-enter", pos_in_bucket=pos_in_bucket, sort_order=sort_order, sorted_pos=sorted_pos)
    pos_in_bucket.scatter_(0, sort_order, sorted_pos)
    # _rowwise_route_debug_print(self, "pos-scatter-exit", pos_in_bucket=pos_in_bucket)

    # _rowwise_route_debug_print(self, "keep-mask-enter", allowed_splits_i64=allowed_splits_i64, safe_experts=safe_experts, pos_in_bucket=pos_in_bucket)
    keep_limits = allowed_splits_i64.index_select(0, safe_experts)
    kept_mask = valid_mask & (pos_in_bucket < keep_limits)
    # _rowwise_route_debug_print(self, "keep-mask-exit", keep_limits=keep_limits, kept_mask=kept_mask)

    # _rowwise_route_debug_print(self, "dst-rank-enter", safe_experts=safe_experts)
    dst_rank = torch.div(
        safe_experts,
        self.num_local_routed_experts,
        rounding_mode="floor",
    )
    dst_local_expert = torch.remainder(
        safe_experts,
        self.num_local_routed_experts,
    )
    # _rowwise_route_debug_print(self, "dst-rank-exit", dst_rank=dst_rank, dst_local_expert=dst_local_expert)

    # _rowwise_route_debug_print(self, "prefix-enter", keep_matrix=keep_matrix)
    prefix_by_source = torch.cumsum(keep_matrix, dim=0) - keep_matrix
    src_rank = get_rank(self.ep_pg)
    send_base_by_dest_local = prefix_by_source[src_rank]
    recv_total_by_dest_local = keep_matrix.sum(dim=0)
    local_expert_base_by_dest = (
        torch.cumsum(recv_total_by_dest_local, dim=1) - recv_total_by_dest_local
    )
    # _rowwise_route_debug_print(
    #     self,
    #     "prefix-exit",
    #     prefix_by_source=prefix_by_source,
    #     send_base_by_dest_local=send_base_by_dest_local,
    #     recv_total_by_dest_local=recv_total_by_dest_local,
    #     local_expert_base_by_dest=local_expert_base_by_dest,
    # )

    # _rowwise_route_debug_print(
    #     self,
    #     "base-rows-enter",
    #     local_expert_base_by_dest=local_expert_base_by_dest,
    #     send_base_by_dest_local=send_base_by_dest_local,
    #     dst_rank=dst_rank,
    #     dst_local_expert=dst_local_expert,
    #     pos_in_bucket=pos_in_bucket,
    # )
    base_rows = local_expert_base_by_dest[dst_rank, dst_local_expert]
    base_rows = base_rows + send_base_by_dest_local[dst_rank, dst_local_expert]
    dst_rows_all = base_rows + pos_in_bucket
    # _rowwise_route_debug_print(self, "base-rows-exit", base_rows=base_rows, dst_rows_all=dst_rows_all)

    neg_ones = torch.full_like(dst_rank, -1)
    # _rowwise_route_debug_print(
    #     self,
    #     "final-where-enter",
    #     kept_mask=kept_mask,
    #     dst_rank=dst_rank,
    #     dst_rows_all=dst_rows_all,
    #     neg_ones=neg_ones,
    # )
    dst_ranks_flat = torch.where(kept_mask, dst_rank, neg_ones)
    dst_rows_flat = torch.where(kept_mask, dst_rows_all, neg_ones)
    # _rowwise_route_debug_print(self, "exit", dst_ranks_flat=dst_ranks_flat, dst_rows_flat=dst_rows_flat)
    return (
        dst_ranks_flat.view(num_tokens, top_k),
        dst_rows_flat.view(num_tokens, top_k),
    )
