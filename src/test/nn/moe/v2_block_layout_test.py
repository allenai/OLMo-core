import types

import torch

from olmo_core.nn.moe.v2.block import MoEFusedV2TransformerBlock


def test_build_padded_local_expert_batch_sizes_excludes_tail_capacity_pad():
    dummy_block = types.SimpleNamespace(num_local_routed_experts=4, ep_world_size=2)
    # Layout is local-expert-major then source-rank.
    # Capacity rows are 24, but dispatch metadata only uses rows [0, 16).
    splits = torch.tensor([4, 1, 3, 1, 2, 1, 1, 1], dtype=torch.int64)
    offsets = torch.tensor([0, 5, 6, 9, 10, 13, 14, 15], dtype=torch.int64)

    padded_sizes = MoEFusedV2TransformerBlock._build_padded_local_expert_batch_sizes_from_layout(
        dummy_block,
        splits=splits,
        offsets=offsets,
        total_rows=24,
    )

    # Last expert should end at max(offset + split)=16, not at capacity=24.
    assert padded_sizes.tolist() == [6, 4, 4, 2]
