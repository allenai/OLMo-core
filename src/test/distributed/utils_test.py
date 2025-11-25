from functools import partial

import pytest
import torch.distributed as dist

import olmo_core.distributed.utils as dist_utils
from olmo_core.testing import BACKENDS, run_distributed_test


def broadcast_object():
    if dist.get_rank() == 0:
        x = ("abc", "def")
    else:
        x = ("abc", "abc")
    x = dist_utils.broadcast_object(x)
    assert x == ("abc", "def")


@pytest.mark.parametrize("backend", BACKENDS)
def test_broadcast_object(backend: str):
    run_distributed_test(broadcast_object, backend=backend)


@pytest.mark.parametrize("n, world_size", [(2, 1), (8, 64)])
def test_do_n_at_a_time(n: int, world_size: int):
    times_called = 0
    calling_ranks = set()

    def func(rank: int):
        nonlocal times_called
        times_called += 1
        calling_ranks.add(rank)

    for rank in range(world_size):
        dist_utils.do_n_at_a_time(partial(func, rank), n=n, world_size=world_size, local_rank=rank)

    assert times_called == world_size
    assert calling_ranks == set(range(world_size))
