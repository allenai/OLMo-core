import argparse
import os
from statistics import mean
from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare NCCL all_to_all_single vs NVSHMEM symmetric-memory "
            "all_to_all_vdev and all_to_all_vdev_2d."
        )
    )
    parser.add_argument("--rows", type=int, default=32 * 1024, help="Rows per rank.")
    parser.add_argument("--cols", type=int, default=2048, help="Columns per row.")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--major-align", type=int, default=16)
    return parser.parse_args()


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _event_timed_ms(fn, iters: int) -> List[float]:
    times_ms: List[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times_ms.append(float(start.elapsed_time(end)))
    return times_ms


def _defrag_by_splits_offsets(
    src: torch.Tensor, splits_offsets: torch.Tensor, out_rows: int
) -> torch.Tensor:
    out = torch.empty((out_rows, src.shape[1]), device=src.device, dtype=src.dtype)
    splits = splits_offsets[0]
    offsets = splits_offsets[1]
    cursor = 0
    for i in range(splits.numel()):
        chunk_rows = int(splits[i].item())
        if chunk_rows == 0:
            continue
        chunk_offset = int(offsets[i].item())
        out[cursor : cursor + chunk_rows].copy_(src[chunk_offset : chunk_offset + chunk_rows])
        cursor += chunk_rows
    if cursor != out_rows:
        raise RuntimeError(
            f"Defrag row mismatch: expected {out_rows}, recovered {cursor}. "
            f"splits={splits.tolist()} offsets={offsets.tolist()}"
        )
    return out


def _setup_nvshmem_backend(group: dist.ProcessGroup, device: torch.device) -> str:
    if not symm_mem.is_nvshmem_available():
        raise RuntimeError(
            "NVSHMEM is not available in this environment. "
            "Cannot run symmetric-memory all_to_all_vdev benchmarks."
        )

    backend = symm_mem.get_backend(device)
    if backend is None or backend.upper() != "NVSHMEM":
        symm_mem.set_backend("NVSHMEM")

    world_group_name = dist.group.WORLD.group_name
    group_name = group.group_name
    symm_mem.enable_symm_mem_for_group(world_group_name)
    symm_mem.enable_symm_mem_for_group(group_name)
    return group_name


def _alloc_rendezvous_symm_tensor(
    shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device, group: dist.ProcessGroup
) -> torch.Tensor:
    t = symm_mem.empty(shape, dtype=dtype, device=device)
    symm_mem.rendezvous(t, group=group)
    return t


def main() -> None:
    args = _parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
    group = dist.group.WORLD

    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    device = torch.device("cuda", local_rank)
    dtype = _dtype_from_name(args.dtype)

    if world_size != 4:
        raise RuntimeError(f"Expected world_size=4, got {world_size}.")
    if args.rows % world_size != 0:
        raise RuntimeError(f"rows ({args.rows}) must be divisible by world_size ({world_size}).")

    split_rows = args.rows // world_size
    splits = [split_rows] * world_size

    # Deterministic payload with rank-dependent offset.
    torch.manual_seed(1337 + rank)
    x = torch.randn((args.rows, args.cols), device=device, dtype=dtype)
    x.add_(rank)

    # NCCL all_to_all_single buffers.
    nccl_out = torch.empty_like(x)

    def run_nccl() -> None:
        dist.all_to_all_single(
            nccl_out,
            x,
            output_split_sizes=splits,
            input_split_sizes=splits,
            group=group,
        )

    # Symmetric-memory (NVSHMEM-backed) buffers.
    group_name = _setup_nvshmem_backend(group, device)
    out_cap = args.rows + (world_size - 1) * max(args.major_align - 1, 0)
    symm_in = _alloc_rendezvous_symm_tensor(x.shape, x.dtype, device, group)
    symm_in_splits = _alloc_rendezvous_symm_tensor((world_size,), torch.int64, device, group)
    symm_out_vdev = _alloc_rendezvous_symm_tensor((out_cap, args.cols), x.dtype, device, group)
    symm_out_vdev_splits_offsets = _alloc_rendezvous_symm_tensor((2, world_size), torch.int64, device, group)
    symm_out_vdev2d = _alloc_rendezvous_symm_tensor((out_cap, args.cols), x.dtype, device, group)
    symm_out_vdev2d_splits_offsets = _alloc_rendezvous_symm_tensor((2, world_size), torch.int64, device, group)
    symm_in.copy_(x)
    symm_in_splits.fill_(split_rows)

    def run_symm_vdev() -> None:
        torch.ops.symm_mem.all_to_all_vdev(
            symm_in,
            symm_out_vdev,
            symm_in_splits,
            symm_out_vdev_splits_offsets,
            group_name,
        )

    def run_symm_vdev2d() -> None:
        torch.ops.symm_mem.all_to_all_vdev_2d(
            symm_in,
            symm_out_vdev2d,
            symm_in_splits,
            symm_out_vdev2d_splits_offsets,
            group_name,
            major_align=args.major_align,
        )

    dist.barrier(group=group)
    for _ in range(args.warmup_iters):
        run_nccl()
    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    nccl_times_ms = _event_timed_ms(run_nccl, args.iters)

    dist.barrier(group=group)
    for _ in range(args.warmup_iters):
        run_symm_vdev()
    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    symm_vdev_times_ms = _event_timed_ms(run_symm_vdev, args.iters)

    dist.barrier(group=group)
    for _ in range(args.warmup_iters):
        run_symm_vdev2d()
    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    symm_vdev2d_times_ms = _event_timed_ms(run_symm_vdev2d, args.iters)

    # Correctness check.
    run_nccl()
    run_symm_vdev()
    run_symm_vdev2d()
    symm_vdev_defrag = _defrag_by_splits_offsets(
        symm_out_vdev, symm_out_vdev_splits_offsets, out_rows=args.rows
    )
    symm_vdev2d_defrag = _defrag_by_splits_offsets(
        symm_out_vdev2d, symm_out_vdev2d_splits_offsets, out_rows=args.rows
    )
    equal_vdev = torch.equal(nccl_out, symm_vdev_defrag)
    equal_vdev2d = torch.equal(nccl_out, symm_vdev2d_defrag)
    max_abs_diff_vdev = (
        float((nccl_out - symm_vdev_defrag).abs().max().item()) if not equal_vdev else 0.0
    )
    max_abs_diff_vdev2d = (
        float((nccl_out - symm_vdev2d_defrag).abs().max().item()) if not equal_vdev2d else 0.0
    )

    gathered = [None] * world_size
    result = {
        "rank": rank,
        "nccl_avg_ms": mean(nccl_times_ms),
        "symm_vdev_avg_ms": mean(symm_vdev_times_ms),
        "symm_vdev2d_avg_ms": mean(symm_vdev2d_times_ms),
        "equal_vdev": bool(equal_vdev),
        "equal_vdev2d": bool(equal_vdev2d),
        "max_abs_diff_vdev": max_abs_diff_vdev,
        "max_abs_diff_vdev2d": max_abs_diff_vdev2d,
    }
    dist.all_gather_object(gathered, result, group=group)

    if rank == 0:
        payload_gb = (args.rows * args.cols * torch.tensor([], dtype=dtype).element_size()) / 1e9
        print("=== AllToAll Comparison: NCCL vs NVSHMEM Symmetric Memory (vdev + vdev_2d) ===", flush=True)
        print(
            f"world_size={world_size} rows={args.rows} cols={args.cols} dtype={dtype} "
            f"warmup={args.warmup_iters} iters={args.iters} major_align={args.major_align}",
            flush=True,
        )
        print(f"payload_per_rank={payload_gb:.3f} GB", flush=True)
        for item in sorted(gathered, key=lambda z: z["rank"]):
            nccl_bw = payload_gb / (item["nccl_avg_ms"] / 1e3)
            symm_vdev_bw = payload_gb / (item["symm_vdev_avg_ms"] / 1e3)
            symm_vdev2d_bw = payload_gb / (item["symm_vdev2d_avg_ms"] / 1e3)
            speedup_vdev = item["nccl_avg_ms"] / item["symm_vdev_avg_ms"]
            speedup_vdev2d = item["nccl_avg_ms"] / item["symm_vdev2d_avg_ms"]
            print(
                f"rank={item['rank']} "
                f"nccl={item['nccl_avg_ms']:.3f} ms ({nccl_bw:.2f} GB/s) "
                f"symm_vdev={item['symm_vdev_avg_ms']:.3f} ms ({symm_vdev_bw:.2f} GB/s) "
                f"speedup_vdev={speedup_vdev:.3f}x "
                f"equal_vdev={item['equal_vdev']} max_abs_diff_vdev={item['max_abs_diff_vdev']:.6f} "
                f"symm_vdev2d={item['symm_vdev2d_avg_ms']:.3f} ms ({symm_vdev2d_bw:.2f} GB/s) "
                f"speedup_vdev2d={speedup_vdev2d:.3f}x "
                f"equal_vdev2d={item['equal_vdev2d']} max_abs_diff_vdev2d={item['max_abs_diff_vdev2d']:.6f}",
                flush=True,
            )

    dist.barrier(group=group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
