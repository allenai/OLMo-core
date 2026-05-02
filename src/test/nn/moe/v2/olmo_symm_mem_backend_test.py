import os

import pytest
import torch
import torch.distributed as dist

from olmo_core.testing import requires_multi_gpu, run_distributed_test


def _require_olmo_symm_extension() -> None:
    try:
        from olmo_core.kernels.symm_mem_vdev2d import _load_cuda_extension

        ext = _load_cuda_extension()
        ext.olmo_symm_get_unique_id()
    except Exception as e:
        pytest.skip(f"OLMo NVSHMEM symmetric-memory extension is unavailable: {e}")


def _alloc_on_subgroup_only() -> None:
    from olmo_core.kernels import olmo_symm_mem

    os.environ["OLMO_USE_OWN_SYMM_MEM"] = "1"
    rank = dist.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    group = dist.new_group(ranks=[0, 1])

    if rank < 2:
        tensor = olmo_symm_mem.empty(
            (4, 4),
            dtype=torch.float32,
            device=device,
            group=group,
        )
        olmo_symm_mem.rendezvous(tensor, group=group)
        tensor.fill_(float(rank + 1))
        torch.cuda.synchronize()

    dist.barrier()


def _alloc_two_independent_subgroups() -> None:
    from olmo_core.kernels import olmo_symm_mem

    os.environ["OLMO_USE_OWN_SYMM_MEM"] = "1"
    rank = dist.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    group_a = dist.new_group(ranks=[0, 1])
    group_b = dist.new_group(ranks=[2, 3])
    group = group_a if rank < 2 else group_b
    subgroup_rank = dist.get_rank(group)

    tensor = olmo_symm_mem.empty(
        (2, 3),
        dtype=torch.float32,
        device=device,
        group=group,
    )
    olmo_symm_mem.rendezvous(tensor, group=group)
    tensor.fill_(float(subgroup_rank + 1))
    torch.cuda.synchronize()
    dist.barrier()


def _rowwise_dispatch_combine_roundtrip() -> None:
    from olmo_core.kernels import olmo_symm_mem
    from olmo_core.kernels.symm_mem_vdev2d import rowwise_combine_get, rowwise_dispatch_put

    os.environ["OLMO_USE_OWN_SYMM_MEM"] = "1"
    group = dist.group.WORLD
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    if world_size != 2:
        raise RuntimeError(f"Expected world_size=2, got {world_size}")

    device = torch.device("cuda", torch.cuda.current_device())
    rows = 2
    cols = 3
    x = (
        torch.arange(rows * cols, device=device, dtype=torch.float32)
        .reshape(rows, cols)
        .add_(rank * 100)
    )

    symm_out = olmo_symm_mem.empty(
        (rows, cols),
        dtype=torch.float32,
        device=device,
        group=group,
    )
    olmo_symm_mem.rendezvous(symm_out, group=group)
    symm_out.zero_()

    # Source row 0 goes to destination rank 0 at row=<source rank>.
    # Source row 1 goes to destination rank 1 at row=<source rank>.
    dst_ranks = torch.tensor([[0], [1]], device=device, dtype=torch.int64)
    dst_rows = torch.full((rows, 1), rank, device=device, dtype=torch.int64)

    rowwise_dispatch_put(
        x,
        symm_out,
        dst_ranks,
        dst_rows,
        group.group_name,
        nblocks=1,
    )
    torch.cuda.synchronize()

    combine_out = torch.empty_like(x)
    src_ranks = torch.tensor([[0], [1]], device=device, dtype=torch.int64)
    src_rows = torch.full((rows, 1), rank, device=device, dtype=torch.int64)
    rowwise_combine_get(
        symm_out,
        combine_out,
        src_ranks,
        src_rows,
        group.group_name,
        nblocks=1,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(combine_out, x)
    dist.barrier()


def _rowwise_dispatch_combine_subgroup_only() -> None:
    from olmo_core.kernels import olmo_symm_mem
    from olmo_core.kernels.symm_mem_vdev2d import rowwise_combine_get, rowwise_dispatch_put

    os.environ["OLMO_USE_OWN_SYMM_MEM"] = "1"
    rank = dist.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    group = dist.new_group(ranks=[0, 1])

    if rank < 2:
        subgroup_rank = dist.get_rank(group)
        rows = 2
        cols = 3
        x = (
            torch.arange(rows * cols, device=device, dtype=torch.float32)
            .reshape(rows, cols)
            .add_(subgroup_rank * 100)
        )

        symm_out = olmo_symm_mem.empty(
            (rows, cols),
            dtype=torch.float32,
            device=device,
            group=group,
        )
        olmo_symm_mem.rendezvous(symm_out, group=group)
        symm_out.zero_()

        dst_ranks = torch.tensor([[0], [1]], device=device, dtype=torch.int64)
        dst_rows = torch.full((rows, 1), subgroup_rank, device=device, dtype=torch.int64)
        rowwise_dispatch_put(
            x,
            symm_out,
            dst_ranks,
            dst_rows,
            group.group_name,
            nblocks=1,
        )
        torch.cuda.synchronize()

        combine_out = torch.empty_like(x)
        src_ranks = torch.tensor([[0], [1]], device=device, dtype=torch.int64)
        src_rows = torch.full((rows, 1), subgroup_rank, device=device, dtype=torch.int64)
        rowwise_combine_get(
            symm_out,
            combine_out,
            src_ranks,
            src_rows,
            group.group_name,
            nblocks=1,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(combine_out, x)

    dist.barrier()


@requires_multi_gpu
def test_olmo_symm_alloc_subgroup_without_world_participation():
    _require_olmo_symm_extension()
    if torch.cuda.device_count() < 4:
        pytest.skip("Requires at least 4 GPUs")
    run_distributed_test(
        _alloc_on_subgroup_only,
        world_size=4,
        backend="nccl",
        start_method="spawn",
    )


@requires_multi_gpu
def test_olmo_symm_alloc_two_independent_subgroups():
    _require_olmo_symm_extension()
    if torch.cuda.device_count() < 4:
        pytest.skip("Requires at least 4 GPUs")
    run_distributed_test(
        _alloc_two_independent_subgroups,
        world_size=4,
        backend="nccl",
        start_method="spawn",
    )


@requires_multi_gpu
def test_olmo_symm_rowwise_dispatch_combine_roundtrip():
    _require_olmo_symm_extension()
    run_distributed_test(
        _rowwise_dispatch_combine_roundtrip,
        world_size=2,
        backend="nccl",
        start_method="spawn",
    )


@requires_multi_gpu
def test_olmo_symm_rowwise_subgroup_without_world_participation():
    _require_olmo_symm_extension()
    if torch.cuda.device_count() < 4:
        pytest.skip("Requires at least 4 GPUs")
    run_distributed_test(
        _rowwise_dispatch_combine_subgroup_only,
        world_size=4,
        backend="nccl",
        start_method="spawn",
    )
