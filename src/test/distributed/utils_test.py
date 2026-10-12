from functools import partial

import pytest
import torch
import torch.distributed as dist

import olmo_core.distributed.utils as dist_utils
from olmo_core.testing import BACKENDS, requires_multi_gpu, run_distributed_test


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


def all_gather_object_cpu():
    rank = dist.get_rank()
    world = dist.get_world_size()
    # Ragged shards, like the eval's per-rank example slices: rank r contributes r + 1 rows, so a
    # gather that mixed up rank order or dropped a shard cannot pass by symmetry.
    shard = [(rank * 100 + i, f"rank{rank}-row{i}") for i in range(rank + 1)]

    gathered = dist_utils.all_gather_object_cpu(shard)

    assert len(gathered) == world
    for r, part in enumerate(gathered):
        assert part == [
            (r * 100 + i, f"rank{r}-row{i}") for i in range(r + 1)
        ], f"rank {rank} got the wrong shard at index {r}: {part}"


@pytest.mark.parametrize("backend", BACKENDS)
def test_all_gather_object_cpu(backend: str):
    """
    The CPU-routed object gather must return every rank's shard, in rank order, on both backends.

    Under NCCL this also exercises the reason the helper exists: the gather has to run over a gloo
    side-group rather than serializing the payload onto the GPU.
    """
    run_distributed_test(all_gather_object_cpu, backend=backend)


def all_gather_object_cpu_stays_off_gpu():
    before = torch.cuda.memory_allocated()
    payload = [(i, "x" * 4096) for i in range(256)]  # ~1 MiB of text per rank
    gathered = dist_utils.all_gather_object_cpu(payload)
    after = torch.cuda.memory_allocated()

    assert len(gathered) == dist.get_world_size()
    assert all(part == payload for part in gathered)
    # The whole point: pickled bytes must not be staged in GPU memory. torch's default
    # all_gather_object would allocate max_pickled_size * world_size on the device here.
    assert after == before, f"gather allocated {after - before} bytes of GPU memory"


@requires_multi_gpu
def test_all_gather_object_cpu_stays_off_gpu():
    """
    Regression test for the OOM this helper was written for: on an ultra-long eval rung, gathering
    result text through NCCL asked for ~79 GiB of GPU memory with ~58 GiB free and killed the job
    after generation had already completed. Routing over gloo must leave GPU allocation untouched.
    """
    run_distributed_test(all_gather_object_cpu_stays_off_gpu, backend="nccl", world_size=2)


def all_gather_object_cpu_single_process():
    assert dist_utils.all_gather_object_cpu(("a", 1)) == [("a", 1)]


def test_all_gather_object_cpu_without_distributed():
    """Outside a process group the helper degrades to a one-element list rather than raising."""
    all_gather_object_cpu_single_process()


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
