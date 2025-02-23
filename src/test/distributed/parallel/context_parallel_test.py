import torch

from olmo_core.distributed.parallel.context_parallel import (
    ContextParallelZigZagLoadBalancer,
)


def test_zig_zag_load_balancer():
    x = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    assert ContextParallelZigZagLoadBalancer(cp_rank=0, cp_world_size=4).shard(x, 0).tolist() == [
        0,
        7,
    ]
    assert ContextParallelZigZagLoadBalancer(cp_rank=3, cp_world_size=4).shard(x, 0).tolist() == [
        3,
        4,
    ]
