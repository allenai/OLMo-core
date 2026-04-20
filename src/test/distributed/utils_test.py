import datetime
import os
from functools import partial

import pytest
import torch
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_init_distributed_bootstraps_single_process_env(monkeypatch: pytest.MonkeyPatch):
    for env_var in (
        dist_utils.TORCH_RANK_ENV_VAR,
        dist_utils.TORCH_WORLD_SIZE_ENV_VAR,
        dist_utils.TORCH_MASTER_ADDR_ENV_VAR,
        dist_utils.TORCH_MASTER_PORT_ENV_VAR,
        dist_utils.OLMO_LOCAL_RANK_ENV_VAR,
        dist_utils.OLMO_LOCAL_WORLD_SIZE_ENV_VAR,
        dist_utils.OLMO_NUM_NODES_ENV_VAR,
        dist_utils.OLMO_SHARED_FS_ENV_VAR,
    ):
        monkeypatch.delenv(env_var, raising=False)

    dist_utils.init_distributed(
        backend="cpu:gloo,cuda:nccl",
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        assert dist.is_initialized()
        assert dist.get_rank() == 0
        assert dist.get_world_size() == 1
        assert os.environ[dist_utils.OLMO_LOCAL_RANK_ENV_VAR] == "0"
        assert os.environ[dist_utils.OLMO_LOCAL_WORLD_SIZE_ENV_VAR] == "1"
    finally:
        dist.destroy_process_group()
