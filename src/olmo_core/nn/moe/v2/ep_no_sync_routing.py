from typing import TYPE_CHECKING, Optional, Tuple, Union, cast

import nvtx
import torch
import torch.distributed as dist

from olmo_core.distributed.utils import get_rank
from ...moe.utils import moe_unpermute_1d_fused_drop_no_compile, moe_unpermute_no_compile

if TYPE_CHECKING:
    from .block import MoEFusedV2TransformerBlock


def build_padded_local_expert_batch_sizes_from_layout(
    block: "MoEFusedV2TransformerBlock",
    *,
    splits: torch.Tensor,
    offsets: torch.Tensor,
    total_rows: int,
) -> torch.Tensor:
    """
    Recover per-local-expert padded row spans from rank-split metadata.

    The vdev layout is local-expert-major, then source-rank. Capacity rows
    may extend past the last written segment; this helper stops each expert
    at the largest real segment end instead of charging tail capacity to the
    final expert.
    """
    assert block.num_local_routed_experts is not None
    expected = block.num_local_routed_experts * block.ep_world_size
    if splits.numel() != expected or offsets.numel() != expected:
        raise RuntimeError(
            "splits/offsets must have num_local_routed_experts * ep_world_size entries: "
            f"got splits={splits.numel()} offsets={offsets.numel()} expected={expected}"
        )
    ends = (offsets.to(dtype=torch.long) + splits.to(dtype=torch.long)).view(
        block.num_local_routed_experts,
        block.ep_world_size,
    )
    expert_ends = ends.max(dim=1).values.clamp_max(int(total_rows))
    starts = torch.cat(
        [
            torch.zeros(1, device=expert_ends.device, dtype=expert_ends.dtype),
            expert_ends[:-1],
        ],
        dim=0,
    )
    return (expert_ends - starts).clamp_min(0)


def build_tail_keep_quota(
    block: "MoEFusedV2TransformerBlock",
    recv_counts_per_src_local_expert: torch.Tensor,
    rank_capacity: int,
) -> torch.Tensor:
    """
    Build per-source keep quotas on destination rank.
    Order is local-expert-major then source-rank.
    """
    counts = recv_counts_per_src_local_expert.to(dtype=torch.long)
    counts_flat = counts.transpose(0, 1).reshape(-1)
    cumsum_counts = torch.cumsum(counts_flat, dim=0)
    kept_cumsum = torch.clamp(cumsum_counts, max=rank_capacity)
    prev = torch.cat([torch.zeros(1, device=counts.device, dtype=torch.long), kept_cumsum[:-1]])
    kept_flat = kept_cumsum - prev
    kept = kept_flat.view(block.num_local_routed_experts, block.ep_world_size).transpose(0, 1)
    return kept


@nvtx.annotate("SyncTokenCount", color="green")
def sync_tail_drop_allowed_splits_single_a2a(
    block: "MoEFusedV2TransformerBlock",
    requested_splits: torch.Tensor,
    *,
    rank_capacity: int,
    return_keep_matrix: bool = False,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """
    Tail-drop keep-split sync with a single all-gather.
    Each EP rank receives every rank's requested splits, then computes the
    same keep policy locally using the shared `rank_capacity`.
    """
    assert block.num_local_routed_experts is not None
    requested = requested_splits.to(dtype=torch.long)
    expected_splits = block.ep_world_size * block.num_local_routed_experts
    if requested.numel() != expected_splits:
        raise RuntimeError(
            "requested_splits size mismatch: "
            f"got {requested.numel()}, expected {expected_splits}"
        )

    gathered_payload = torch.empty(
        expected_splits * block.ep_world_size,
        device=requested.device,
        dtype=requested.dtype,
    )
    dist.all_gather_into_tensor(
        gathered_payload,
        requested,
        group=block.ep_pg,
    )

    gathered_payload_2d = gathered_payload.view(block.ep_world_size, expected_splits)
    global_requested = gathered_payload_2d.view(
        block.ep_world_size, block.ep_world_size, block.num_local_routed_experts
    )

    # Flatten in local-expert-major then source-rank order for each destination rank.
    counts_flat = global_requested.permute(2, 0, 1).reshape(-1, block.ep_world_size)
    cumsum_counts = torch.cumsum(counts_flat, dim=0)
    kept_cumsum = torch.clamp(cumsum_counts, max=rank_capacity)
    prev = torch.cat(
        [
            torch.zeros((1, block.ep_world_size), device=requested.device, dtype=torch.long),
            kept_cumsum[:-1],
        ],
        dim=0,
    )
    kept_flat = kept_cumsum - prev
    keep_from_src_dest_local = kept_flat.view(
        block.num_local_routed_experts, block.ep_world_size, block.ep_world_size
    ).permute(1, 2, 0)

    local_rank = get_rank(block.ep_pg)
    allowed_splits = keep_from_src_dest_local[local_rank].reshape(-1)
    allowed_splits = torch.minimum(allowed_splits, requested)

    # shape: (source_rank, local_expert)
    recv_splits_by_src_local = keep_from_src_dest_local[:, local_rank, :]
    send_side_drop_token_count = requested.sum() - allowed_splits.sum()
    if return_keep_matrix:
        return (
            allowed_splits,
            recv_splits_by_src_local,
            send_side_drop_token_count,
            keep_from_src_dest_local,
        )
    return allowed_splits, recv_splits_by_src_local, send_side_drop_token_count


@nvtx.annotate("_build_keep_reorder")
def build_keep_reorder(
    requested_splits: torch.Tensor,
    keep_splits: torch.Tensor,
    num_out_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a static-shape reorder map that moves kept tokens to the front while
    preserving within-group order. Returns:
      - reorder indices (original -> packed order),
      - inverse reorder indices (packed -> original order),
      - keep mask in packed order.
    """
    requested = requested_splits.to(dtype=torch.long)
    keep = keep_splits.to(dtype=torch.long)
    token_ids = torch.arange(num_out_tokens, device=keep.device, dtype=torch.long)

    requested_ends = torch.cumsum(requested, dim=0)

    max_expert_idx = requested.numel() - 1
    expert_ids = torch.searchsorted(requested_ends, token_ids, right=True).clamp_max(max_expert_idx)
    starts = requested_ends - requested
    pos_in_chunk = token_ids - starts.index_select(0, expert_ids)
    keep_mask = pos_in_chunk < keep.index_select(0, expert_ids)

    # Stable partition: keep rows first, then dropped rows.
    keep_i64 = keep_mask.to(dtype=torch.long)
    drop_i64 = (~keep_mask).to(dtype=torch.long)
    keep_rank = torch.cumsum(keep_i64, dim=0) - 1
    drop_rank = torch.cumsum(drop_i64, dim=0) - 1
    num_kept = keep_i64.sum(dtype=torch.long)
    packed_pos = torch.where(keep_mask, keep_rank, num_kept + drop_rank)

    reorder_indices = torch.empty_like(token_ids)
    reorder_indices.scatter_(0, packed_pos, token_ids)

    inverse_reorder_indices = packed_pos
    packed_keep_mask = keep_mask.index_select(0, reorder_indices)
    return reorder_indices, inverse_reorder_indices, packed_keep_mask


@nvtx.annotate("_restore_drop_unpermute_1d")
def restore_drop_unpermute_1d(
    block: "MoEFusedV2TransformerBlock",
    *,
    combine_out: torch.Tensor,
    local_inverse_reorder_indices: torch.Tensor,
    packed_keep_mask: torch.Tensor,
    num_kept: torch.Tensor,
    reversed_local_x_permutation_mapping: torch.Tensor,
    local_x_global_routed_expert_weights: torch.Tensor,
    hidden_shape_before_permute: torch.Size,
    row_id_map_is_packed: bool = False,
    backward_grad_input_buffer: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert block.routed_experts_router is not None
    merging_probs = local_x_global_routed_expert_weights.view(-1, block.routed_experts_router.top_k)
    backend = block.ep_no_sync_restore_unpermute_backend

    if backend == "te_fused":
        return cast(
            torch.Tensor,
            moe_unpermute_1d_fused_drop_no_compile(
                inp=combine_out,
                row_id_map=reversed_local_x_permutation_mapping,
                local_inverse_reorder_indices=local_inverse_reorder_indices,
                packed_keep_mask=packed_keep_mask,
                merging_probs=merging_probs,
                num_kept=num_kept,
                row_id_map_is_packed=row_id_map_is_packed,
                backward_grad_input_buffer=backward_grad_input_buffer,
                map_type="index",
            ),
        )
    if backend == "te_legacy":
        if row_id_map_is_packed:
            restored_local_x = combine_out
        else:
            with nvtx.annotate("RestoreDrop", color="green"):
                restored_local_x = combine_out.index_select(0, local_inverse_reorder_indices)
                restored_keep_mask = packed_keep_mask.index_select(0, local_inverse_reorder_indices)
                restored_local_x = torch.where(
                    restored_keep_mask.unsqueeze(-1),
                    restored_local_x,
                    torch.zeros_like(restored_local_x),
                )
        return cast(
            torch.Tensor,
            moe_unpermute_no_compile(
                inp=restored_local_x,
                row_id_map=reversed_local_x_permutation_mapping,
                merging_probs=merging_probs,
                restore_shape=hidden_shape_before_permute,
                map_type="index",
            ),
        )
    if backend == "cuda":
        raise RuntimeError(
            "ep_no_sync_restore_unpermute_backend='cuda' is not implemented yet. "
            "TODO: add a custom CUDA path mirroring TE _moe_unpermute_index_map semantics."
        )
    raise RuntimeError(f"Unhandled ep_no_sync_restore_unpermute_backend: {backend}")
