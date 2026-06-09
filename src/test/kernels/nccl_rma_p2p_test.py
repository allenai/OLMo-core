from __future__ import annotations

import torch
import torch.distributed as dist

from olmo_core.distributed.utils import get_local_rank
from olmo_core.kernels import nccl_rma_p2p
from olmo_core.testing import requires_multi_gpu, run_distributed_test


def _broadcast_unique_id(rank: int) -> bytes:
    # The RMA transport runs on its own NCCL communicator, built from this
    # unique id via ncclCommInitRank inside nccl_rma_p2p.init() -- it does not
    # piggyback on the torch process group. The torch PG (gloo, below) is used
    # only to broadcast the id and for barriers; it never touches the RMA data
    # path, so a CPU/gloo coordination backend is sufficient here.
    obj = [nccl_rma_p2p.get_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(obj, src=0)
    unique_id = obj[0]
    assert isinstance(unique_id, bytes), "Failed to broadcast NCCL unique ID"
    return unique_id


def _run_put_signal_wait() -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, f"Expected exactly 2 ranks, got {world_size}"
    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)

    # Build/load the extension on every rank before any collective. The first call
    # JIT-compiles it (slow); doing it here (with a barrier) avoids one rank stalling
    # inside a gloo collective while its peer is still compiling, and surfaces a real
    # build failure symmetrically instead of as a confusing "connection closed by peer".
    nccl_rma_p2p.runtime_version()
    dist.barrier()

    unique_id = _broadcast_unique_id(rank)
    ctx = nccl_rma_p2p.init(
        unique_id,
        rank=rank,
        world_size=world_size,
        device=local_rank,
    )

    n = 4096
    window_id, window = nccl_rma_p2p.alloc_window(ctx, (n,), dtype="float32")
    window.zero_()
    torch.cuda.synchronize()
    dist.barrier()

    peer = 1 - rank

    # One-way rank 0 -> rank 1.
    if rank == 0:
        window.copy_(torch.arange(n, device="cuda", dtype=torch.float32) + 17.0)
        nccl_rma_p2p.put_signal(ctx, window, peer=peer, window_id=window_id)
        torch.cuda.synchronize()
    else:
        nccl_rma_p2p.wait_signal(ctx, peer=peer, op_count=1)
        torch.cuda.synchronize()
        expected = torch.arange(n, device="cuda", dtype=torch.float32) + 17.0
        torch.testing.assert_close(window, expected)
    dist.barrier()

    window.zero_()
    torch.cuda.synchronize()
    dist.barrier()

    # One-way rank 1 -> rank 0. Signal counts are per peer, so each receiver
    # waits for the first signal from its sending peer.
    if rank == 1:
        window.copy_(torch.arange(n, device="cuda", dtype=torch.float32) + 31.0)
        nccl_rma_p2p.put_signal(ctx, window, peer=peer, window_id=window_id)
        torch.cuda.synchronize()
    else:
        nccl_rma_p2p.wait_signal(ctx, peer=peer, op_count=1)
        torch.cuda.synchronize()
        expected = torch.arange(n, device="cuda", dtype=torch.float32) + 31.0
        torch.testing.assert_close(window, expected)
    dist.barrier()

    nccl_rma_p2p.free_window(ctx, window_id)
    nccl_rma_p2p.destroy(ctx)
    dist.barrier()


@requires_multi_gpu
def test_nccl_rma_p2p_put_signal_wait():
    # gloo is only the coordination PG (broadcast + barriers); the RMA path uses its
    # own NCCL comm. Force "spawn" though -- the gloo default is "fork", which can't
    # re-initialize CUDA in the child processes.
    run_distributed_test(_run_put_signal_wait, world_size=2, backend="gloo", start_method="spawn")
