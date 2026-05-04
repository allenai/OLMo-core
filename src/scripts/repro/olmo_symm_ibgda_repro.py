from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as torch_symm

from olmo_core.kernels import olmo_symm_mem
from olmo_core.kernels import symm_mem_vdev2d as rowwise_kernels


def _dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _rank_print(rank: int, msg: str) -> None:
    print(f"[rank {rank}] {msg}", flush=True)


def _alloc_tensor(
    *,
    kind: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    group: dist.ProcessGroup,
    torch_symm_backend: str,
) -> torch.Tensor:
    if kind == "olmo":
        tensor = olmo_symm_mem.empty(shape, dtype=dtype, device=device, group=group)
        olmo_symm_mem.rendezvous(tensor, group=group)
        return tensor
    if kind == "torch":
        if torch_symm_backend != "default":
            torch_symm.set_backend(torch_symm_backend)
        tensor = torch_symm.empty(shape, dtype=dtype, device=device)
        torch_symm.rendezvous(tensor, group=group)
        return tensor
    if kind == "regular":
        return torch.empty(shape, dtype=dtype, device=device)
    raise ValueError(f"Unsupported tensor kind: {kind}")


def _resolve_peer(rank: int, world_size: int, peer_mode: str, peer_stride: int | None) -> int:
    if peer_stride is not None:
        return (rank + peer_stride) % world_size
    if peer_mode == "self":
        return rank
    if peer_mode == "intra-node":
        local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", "8"))
        local_base = (rank // local_world_size) * local_world_size
        peer = local_base + ((rank - local_base) ^ 1)
        if peer >= min(local_base + local_world_size, world_size):
            peer = rank
        return peer
    if world_size % 2 != 0:
        raise RuntimeError("cross-node peer mode expects an even world_size")
    return (rank + world_size // 2) % world_size


def _resolve_inverse_peer(rank: int, world_size: int, peer_mode: str, peer_stride: int | None) -> int:
    if peer_stride is not None:
        return (rank - peer_stride) % world_size
    if peer_mode == "self":
        return rank
    if peer_mode == "intra-node":
        return _resolve_peer(rank, world_size, peer_mode, peer_stride)
    return (rank - world_size // 2) % world_size


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Minimal OLMo-owned symmetric-memory rowwise dispatch repro. "
            "By default the NVSHMEM put source is a normal CUDA tensor, which "
            "is expected to trigger an IB local protection / IBGDA CQ failure "
            "on multi-node IBGDA runs."
        )
    )
    parser.add_argument("--op", choices=("dispatch", "combine"), default="dispatch")
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--dim", type=int, default=2880)
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--nblocks", type=int, default=256)
    parser.add_argument(
        "--peer-mode",
        choices=("cross-node", "intra-node", "self"),
        default="cross-node",
        help="Destination route pattern. cross-node is the hostfile2 IBGDA case.",
    )
    parser.add_argument(
        "--peer-stride",
        type=int,
        default=None,
        help="Override peer stride directly; destination peer is (rank + peer_stride) %% world_size.",
    )
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument(
        "--source-kind",
        choices=("regular", "torch", "olmo"),
        default="regular",
        help="Allocation kind for the local PUT source tensor.",
    )
    parser.add_argument(
        "--out-kind",
        choices=("regular", "torch", "olmo"),
        default="regular",
        help="Allocation kind for rowwise_combine_get output.",
    )
    parser.add_argument(
        "--gather-kind",
        choices=("none", "regular", "torch", "olmo"),
        default="none",
        help="Allocation kind for rowwise_combine_get gathered_out scratch.",
    )
    parser.add_argument(
        "--torch-symm-backend",
        choices=("NVSHMEM", "CUDA", "NCCL", "default"),
        default="NVSHMEM",
        help="Backend to set before torch symmetric source allocation.",
    )
    parser.add_argument(
        "--symm-source",
        action="store_true",
        help="Deprecated alias for --source-kind olmo.",
    )
    args = parser.parse_args()
    if args.symm_source:
        args.source_kind = "olmo"

    if os.getenv("OLMO_USE_OWN_SYMM_MEM", "1").strip().lower() in {"0", "false", "no", "off"}:
        raise RuntimeError("This repro must run with OLMO_USE_OWN_SYMM_MEM=1")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    group = dist.group.WORLD
    group_name = group.group_name

    if world_size < 2:
        raise RuntimeError("This repro needs at least 2 ranks so rowwise dispatch crosses peers")

    dtype = _dtype(args.dtype)
    _rank_print(
        rank,
        "env "
        f"OLMO_USE_OWN_SYMM_MEM={os.getenv('OLMO_USE_OWN_SYMM_MEM')} "
        f"NVSHMEM_IB_ENABLE_IBGDA={os.getenv('NVSHMEM_IB_ENABLE_IBGDA')} "
        f"NVSHMEM_IBGDA_NIC_HANDLER={os.getenv('NVSHMEM_IBGDA_NIC_HANDLER')} "
        f"group_name={group_name!r} source_kind={args.source_kind} "
        f"torch_symm_backend={args.torch_symm_backend}",
    )

    olmo_symm_mem.register_group(group, device=device)

    source_seed = torch.arange(args.rows * args.dim, device=device, dtype=torch.float32).reshape(args.rows, args.dim)
    source_seed = source_seed.mul_(0.001).add_(rank * 1000.0)
    source_seed = source_seed.to(dtype=dtype)

    peer = _resolve_peer(rank, world_size, args.peer_mode, args.peer_stride)
    if peer == rank:
        _rank_print(rank, "peer maps to self")
    route_ranks = torch.full((args.rows, 1), peer, dtype=torch.long, device=device)
    route_rows = torch.arange(args.rows, dtype=torch.long, device=device).view(args.rows, 1)

    dist.barrier(device_ids=[local_rank])
    _rank_print(rank, f"starting {args.op} to/from peer={peer}")

    try:
        if args.op == "dispatch":
            symm_out = olmo_symm_mem.empty((args.rows, args.dim), dtype=dtype, device=device, group=group)
            olmo_symm_mem.rendezvous(symm_out, group=group)
            symm_out.zero_()
            source = _alloc_tensor(
                kind=args.source_kind,
                shape=(args.rows, args.dim),
                dtype=dtype,
                device=device,
                group=group,
                torch_symm_backend=args.torch_symm_backend,
            )
            source.copy_(source_seed)
        else:
            expert_out = olmo_symm_mem.empty((args.rows, args.dim), dtype=dtype, device=device, group=group)
            olmo_symm_mem.rendezvous(expert_out, group=group)
            expert_out.copy_(source_seed)
            combine_out = _alloc_tensor(
                kind=args.out_kind,
                shape=(args.rows, args.dim),
                dtype=dtype,
                device=device,
                group=group,
                torch_symm_backend=args.torch_symm_backend,
            )
            gathered_out = None
            if args.gather_kind != "none":
                gathered_out = _alloc_tensor(
                    kind=args.gather_kind,
                    shape=(args.rows, 1, args.dim),
                    dtype=dtype,
                    device=device,
                    group=group,
                    torch_symm_backend=args.torch_symm_backend,
                )

        for i in range(args.iters):
            if args.op == "combine":
                combine_out.fill_(-123.0)
                if gathered_out is not None:
                    gathered_out.fill_(-123.0)
                rowwise_kernels.rowwise_combine_get(
                    expert_out,
                    combine_out,
                    route_ranks,
                    route_rows,
                    group_name,
                    nblocks=args.nblocks,
                    gathered_out=gathered_out,
                )
                result = combine_out
                expected_src_rank = peer
            else:
                assert symm_out is not None
                assert source is not None
                symm_out.fill_(-123.0)
                rowwise_kernels.rowwise_dispatch_put(
                    source,
                    symm_out,
                    route_ranks,
                    route_rows,
                    group_name,
                    nblocks=args.nblocks,
                )
                result = symm_out
                expected_src_rank = _resolve_inverse_peer(rank, world_size, args.peer_mode, args.peer_stride)
            torch.cuda.synchronize()
            dist.barrier(device_ids=[local_rank])
            expected = (
                torch.arange(args.rows * args.dim, device=device, dtype=torch.float32)
                .reshape(args.rows, args.dim)
                .mul_(0.001)
                .add_(expected_src_rank * 1000.0)
                .to(dtype=dtype)
            )
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
            if rank == 0:
                print(f"completed iter {i}", flush=True)
    finally:
        # Make elastic failures easier to read when one rank aborts first.
        sys.stdout.flush()
        sys.stderr.flush()

    _rank_print(rank, "completed without NVSHMEM/IBGDA failure")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
