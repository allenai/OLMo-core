from __future__ import annotations

from typing import Optional

import torch

from .metadata import build_tma_ibgda_route_metadata


def _check_bf16_matrix(name: str, tensor: torch.Tensor) -> None:
    if tensor.dtype != torch.bfloat16:
        raise RuntimeError(f"{name} must be torch.bfloat16")
    if tensor.ndim != 2:
        raise RuntimeError(f"{name} must be rank-2, got {tuple(tensor.shape)}")


def _check_optional_probs(probs: Optional[torch.Tensor], route_shape: torch.Size) -> None:
    if probs is None:
        return
    if probs.dtype != torch.float32:
        raise RuntimeError("probs must be torch.float32")
    if tuple(probs.shape) != tuple(route_shape):
        raise RuntimeError(
            f"probs shape mismatch: got {tuple(probs.shape)}, expected {tuple(route_shape)}"
        )


def reference_dispatch_bf16(
    input: torch.Tensor,
    dst_ranks: torch.Tensor,
    dst_rows: torch.Tensor,
    *,
    ep_world_size: int,
    rank_capacity: Optional[int] = None,
    probs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reference BF16 route dispatch for tests and CUDA-kernel parity.

    Returns all destination-rank buffers as `[ep_world_size, rank_capacity, hidden]`.
    This is intentionally a small semantic reference, not a production fallback.
    """

    _check_bf16_matrix("input", input)
    _check_optional_probs(probs, dst_ranks.shape)
    metadata = build_tma_ibgda_route_metadata(
        dst_ranks,
        dst_rows,
        ep_world_size=ep_world_size,
        rank_capacity=rank_capacity,
    )

    hidden = input.shape[1]
    output = torch.zeros(
        (ep_world_size, metadata.rank_capacity, hidden),
        dtype=input.dtype,
        device=input.device,
    )
    for token_idx in range(metadata.num_tokens):
        for topk_idx in range(metadata.top_k):
            if not bool(metadata.valid_mask[token_idx, topk_idx].item()):
                continue
            dst_rank = int(dst_ranks[token_idx, topk_idx].item())
            dst_row = int(dst_rows[token_idx, topk_idx].item())
            row = input[token_idx]
            if probs is not None:
                row = (row.float() * probs[token_idx, topk_idx]).to(dtype=input.dtype)
            output[dst_rank, dst_row] = row
    return output


def reference_combine_bf16(
    expert_out_by_rank: torch.Tensor,
    src_ranks: torch.Tensor,
    src_rows: torch.Tensor,
    *,
    probs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reference BF16 route gather/reduce for tests and CUDA-kernel parity."""

    if expert_out_by_rank.dtype != torch.bfloat16:
        raise RuntimeError("expert_out_by_rank must be torch.bfloat16")
    if expert_out_by_rank.ndim != 3:
        raise RuntimeError(
            "expert_out_by_rank must be rank-3 [ep_world_size, rank_capacity, hidden], "
            f"got {tuple(expert_out_by_rank.shape)}"
        )
    ep_world_size, rank_capacity, hidden = expert_out_by_rank.shape
    _check_optional_probs(probs, src_ranks.shape)
    metadata = build_tma_ibgda_route_metadata(
        src_ranks,
        src_rows,
        ep_world_size=ep_world_size,
        rank_capacity=rank_capacity,
    )

    output = torch.zeros(
        (metadata.num_tokens, hidden),
        dtype=torch.float32,
        device=expert_out_by_rank.device,
    )
    for token_idx in range(metadata.num_tokens):
        for topk_idx in range(metadata.top_k):
            if not bool(metadata.valid_mask[token_idx, topk_idx].item()):
                continue
            src_rank = int(src_ranks[token_idx, topk_idx].item())
            src_row = int(src_rows[token_idx, topk_idx].item())
            row = expert_out_by_rank[src_rank, src_row].float()
            if probs is not None:
                row = row * probs[token_idx, topk_idx]
            output[token_idx] += row
    return output.to(dtype=torch.bfloat16)
