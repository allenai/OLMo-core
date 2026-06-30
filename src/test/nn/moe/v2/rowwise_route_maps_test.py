from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from olmo_core.nn.moe.v2.ep_no_sync_rowwise_helpers import build_rowwise_route_maps


def _reference_rowwise_route_maps(
    routing_map: torch.Tensor,
    allowed_splits: torch.Tensor,
    keep_from_src_dest_local: torch.Tensor,
    *,
    ep_world_size: int,
    num_local_experts: int,
    src_rank: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, top_k = routing_map.shape
    expert_count = ep_world_size * num_local_experts

    routing_cpu = routing_map.cpu()
    allowed_cpu = allowed_splits.cpu().tolist()
    keep_cpu = keep_from_src_dest_local.cpu()

    recv_total_by_dest_local = keep_cpu.sum(dim=0)
    local_expert_base_by_dest = (
        torch.cumsum(recv_total_by_dest_local, dim=1) - recv_total_by_dest_local
    )
    prefix_by_source = torch.cumsum(keep_cpu, dim=0) - keep_cpu
    send_base_by_dest_local = prefix_by_source[src_rank]

    counts_by_expert = [0 for _ in range(expert_count)]
    dst_ranks = torch.full((num_tokens, top_k), -1, dtype=torch.long)
    dst_rows = torch.full((num_tokens, top_k), -1, dtype=torch.long)

    for token_idx in range(num_tokens):
        for topk_idx in range(top_k):
            expert = int(routing_cpu[token_idx, topk_idx].item())
            if expert < 0 or expert >= expert_count:
                continue

            pos_in_bucket = counts_by_expert[expert]
            counts_by_expert[expert] += 1
            if pos_in_bucket >= int(allowed_cpu[expert]):
                continue

            dst_rank = expert // num_local_experts
            dst_local_expert = expert % num_local_experts
            base_row = (
                int(local_expert_base_by_dest[dst_rank, dst_local_expert].item())
                + int(send_base_by_dest_local[dst_rank, dst_local_expert].item())
            )
            dst_ranks[token_idx, topk_idx] = dst_rank
            dst_rows[token_idx, topk_idx] = base_row + pos_in_bucket

    return dst_ranks.to(routing_map.device), dst_rows.to(routing_map.device)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_build_rowwise_route_maps_matches_reference(device: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")

    ep_world_size = 3
    num_local_experts = 2
    block = SimpleNamespace(
        ep_pg=object(),
        ep_world_size=ep_world_size,
        num_local_routed_experts=num_local_experts,
        block_idx=0,
    )

    routing_map = torch.tensor(
        [
            [0, 1, 5],
            [3, 0, -1],
            [6, 4, 1],
            [2, 3, 5],
            [0, 1, 4],
            [3, 2, 5],
        ],
        device=device,
        dtype=torch.long,
    )
    keep_from_src_dest_local = torch.tensor(
        [
            [[2, 3], [2, 2], [2, 3]],
            [[1, 0], [2, 1], [0, 1]],
            [[0, 2], [1, 0], [3, 0]],
        ],
        device=device,
        dtype=torch.long,
    )
    allowed_splits = keep_from_src_dest_local[0].reshape(-1)

    dst_ranks, dst_rows = build_rowwise_route_maps(
        block,
        routing_map=routing_map,
        allowed_splits=allowed_splits,
        keep_from_src_dest_local=keep_from_src_dest_local,
    )
    expected_ranks, expected_rows = _reference_rowwise_route_maps(
        routing_map,
        allowed_splits,
        keep_from_src_dest_local,
        ep_world_size=ep_world_size,
        num_local_experts=num_local_experts,
    )

    assert torch.equal(dst_ranks, expected_ranks)
    assert torch.equal(dst_rows, expected_rows)
