from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, Union, cast

import nvtx
import torch
import torch.distributed as dist

from olmo_core.distributed.utils import get_rank

from ...moe.utils import (
    moe_unpermute_1d_fused_drop_no_compile,
    moe_unpermute_no_compile,
)

if TYPE_CHECKING:
    from .block import MoEFusedV2TransformerBlock


@nvtx.annotate("_build_keep_reorder")
def build_keep_reorder(
    requested_splits: torch.Tensor,
    keep_splits: torch.Tensor,
    num_out_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    requested = requested_splits.to(dtype=torch.long)
    keep = keep_splits.to(dtype=torch.long)
    token_ids = torch.arange(num_out_tokens, device=keep.device, dtype=torch.long)

    requested_ends = torch.cumsum(requested, dim=0)

    max_expert_idx = requested.numel() - 1
    expert_ids = torch.searchsorted(
        requested_ends,
        token_ids,
        right=True,
    ).clamp_max(max_expert_idx)
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


@nvtx.annotate("SyncTokenCount", color="green")
def sync_tail_drop_allowed_splits_single_a2a(
    block: MoEFusedV2TransformerBlock,
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
    self = block
    assert self.num_local_routed_experts is not None
    requested = requested_splits.to(dtype=torch.long)
    expected_splits = self.ep_world_size * self.num_local_routed_experts
    if requested.numel() != expected_splits:
        raise RuntimeError(
            "requested_splits size mismatch: "
            f"got {requested.numel()}, expected {expected_splits}"
        )

    gathered_payload = torch.empty(
        expected_splits * self.ep_world_size,
        device=requested.device,
        dtype=requested.dtype,
    )
    dist.all_gather_into_tensor(
        gathered_payload,
        requested,
        group=self.ep_pg,
    )

    gathered_payload_2d = gathered_payload.view(self.ep_world_size, expected_splits)
    global_requested = gathered_payload_2d.view(
        self.ep_world_size,
        self.ep_world_size,
        self.num_local_routed_experts,
    )

    # Flatten in local-expert-major then source-rank order for each destination rank.
    counts_flat = global_requested.permute(2, 0, 1).reshape(-1, self.ep_world_size)
    cumsum_counts = torch.cumsum(counts_flat, dim=0)
    kept_cumsum = torch.clamp(cumsum_counts, max=rank_capacity)
    prev = torch.cat(
        [
            torch.zeros((1, self.ep_world_size), device=requested.device, dtype=torch.long),
            kept_cumsum[:-1],
        ],
        dim=0,
    )
    kept_flat = kept_cumsum - prev
    keep_from_src_dest_local = kept_flat.view(
        self.num_local_routed_experts,
        self.ep_world_size,
        self.ep_world_size,
    ).permute(1, 2, 0)

    local_rank = get_rank(self.ep_pg)
    allowed_splits = keep_from_src_dest_local[local_rank].reshape(-1)
    allowed_splits = torch.minimum(allowed_splits, requested)

    recv_splits_by_src_local = keep_from_src_dest_local[:, local_rank, :]
    send_side_drop_token_count = requested.sum() - allowed_splits.sum()
    if return_keep_matrix:
        return (
            allowed_splits,
            recv_splits_by_src_local,
            send_side_drop_token_count,
            keep_from_src_dest_local,
        )
    return (
        allowed_splits,
        recv_splits_by_src_local,
        send_side_drop_token_count,
    )


@nvtx.annotate("_restore_drop_unpermute_1d")
def restore_drop_unpermute_1d(
    block: MoEFusedV2TransformerBlock,
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
    self = block
    assert self.routed_experts_router is not None
    merging_probs = local_x_global_routed_expert_weights.view(
        -1,
        self.routed_experts_router.top_k,
    )
    backend = self.ep_no_sync_restore_unpermute_backend

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
    if backend == "te_unfused":
        if row_id_map_is_packed:
            restored_local_x = combine_out
        else:
            with nvtx.annotate("RestoreDrop", color="green"):
                restored_local_x = combine_out.index_select(
                    0,
                    local_inverse_reorder_indices,
                )
                restored_keep_mask = packed_keep_mask.index_select(
                    0,
                    local_inverse_reorder_indices,
                )
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
