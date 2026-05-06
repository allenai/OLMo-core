from __future__ import annotations

import os

import torch
import torch.distributed as dist

from olmo_core.kernels import nccl_rma_p2p


def _init_dist() -> tuple[int, int, int]:
    dist.init_process_group(backend=os.getenv("OLMO_NCCL_RMA_TEST_PG_BACKEND", "gloo"))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError(f"nccl_rma_p2p_smoke.py expects exactly 2 ranks, got {world_size}")
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def _broadcast_unique_id(rank: int) -> bytes:
    obj = [nccl_rma_p2p.get_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(obj, src=0)
    unique_id = obj[0]
    if not isinstance(unique_id, bytes):
        raise RuntimeError("Failed to broadcast NCCL unique ID")
    return unique_id


def _check_equal(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.equal(actual, expected):
        max_abs = (actual - expected).abs().max().item()
        raise AssertionError(f"{name} mismatch: max_abs={max_abs}")


def main() -> None:
    rank, world_size, local_rank = _init_dist()
    unique_id = _broadcast_unique_id(rank)

    ctx = nccl_rma_p2p.init(
        unique_id,
        rank=rank,
        world_size=world_size,
        device=local_rank,
    )
    print(
        f"[rank {rank}] NCCL runtime version from extension: {nccl_rma_p2p.runtime_version()}",
        flush=True,
    )

    n = int(os.getenv("OLMO_NCCL_RMA_TEST_NUMEL", "4096"))
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
        _check_equal("rank0_to_rank1", window, expected)
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
        _check_equal("rank1_to_rank0", window, expected)
    dist.barrier()

    nccl_rma_p2p.free_window(ctx, window_id)
    nccl_rma_p2p.destroy(ctx)
    dist.barrier()
    if rank == 0:
        print("NCCL RMA P2P smoke test passed", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
