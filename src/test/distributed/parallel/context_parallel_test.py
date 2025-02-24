import torch

from olmo_core.distributed.parallel.context_parallel import (
    ContextParallelZigZagLoadBalancer,
)


def test_zig_zag_load_balancer():
    x = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).unsqueeze(0)
    assert ContextParallelZigZagLoadBalancer(cp_rank=0, cp_world_size=4).shard(x, 1).tolist() == [
        [
            0,
            7,
        ]
    ]
    assert ContextParallelZigZagLoadBalancer(cp_rank=3, cp_world_size=4).shard(x, 1).tolist() == [
        [
            3,
            4,
        ]
    ]


def test_zig_zag_load_balancer_with_cu_doc_lens():
    x = torch.tensor(list(range(12))).unsqueeze(0)
    cu_doc_lens = torch.tensor([0, 8, 12])
    assert ContextParallelZigZagLoadBalancer(cp_rank=0, cp_world_size=2).shard(
        x, 1, cu_doc_lens=cu_doc_lens
    ).tolist() == [
        [
            0,
            1,
            6,
            7,
            8,
            11,
        ]
    ]
    assert ContextParallelZigZagLoadBalancer(cp_rank=1, cp_world_size=2).shard(
        x, 1, cu_doc_lens=cu_doc_lens
    ).tolist() == [
        [
            2,
            3,
            4,
            5,
            9,
            10,
        ]
    ]
