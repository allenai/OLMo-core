import argparse
import os
from statistics import mean
from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark rowwise BF16 vs FP8 dispatch+combine")
    p.add_argument("--rows", type=int, default=8192)
    p.add_argument("--cols", type=int, default=2048)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--experts-per-rank", type=int, default=4)
    p.add_argument("--drop-period", type=int, default=31)
    p.add_argument("--warmup-iters", type=int, default=10)
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--nblocks", type=int, default=0)
    return p.parse_args()


def _event_timed_ms(fn, iters: int) -> List[float]:
    times: List[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))
    return times


def _setup_nvshmem_backend(group: dist.ProcessGroup, device: torch.device) -> str:
    if not symm_mem.is_nvshmem_available():
        raise RuntimeError("NVSHMEM is not available")
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


def _build_global_row_maps(
    all_indices_cpu: torch.Tensor,
    world_size: int,
    experts_per_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, List[int], int]:
    total_experts = world_size * experts_per_rank
    _, rows, top_k = all_indices_cpu.shape

    dst_ranks = torch.full((world_size, rows, top_k), -1, dtype=torch.int64)
    dst_rows = torch.full((world_size, rows, top_k), -1, dtype=torch.int64)
    per_rank_counts = [0 for _ in range(world_size)]

    for src_rank in range(world_size):
        for n in range(rows):
            for k in range(top_k):
                expert = int(all_indices_cpu[src_rank, n, k].item())
                if expert < 0:
                    continue
                if expert >= total_experts:
                    raise RuntimeError(f"Found expert id {expert} >= total_experts={total_experts}")
                dst_rank = expert // experts_per_rank
                row = per_rank_counts[dst_rank]
                per_rank_counts[dst_rank] += 1
                dst_ranks[src_rank, n, k] = dst_rank
                dst_rows[src_rank, n, k] = row

    capacity = max(1, max(per_rank_counts) if per_rank_counts else 0)
    return dst_ranks, dst_rows, per_rank_counts, capacity


def main() -> None:
    args = _parse_args()
    if args.cols % 32 != 0:
        raise RuntimeError(f"cols must be divisible by 32 for MXFP8 scales, got {args.cols}")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))

    from olmo_core.kernels.symm_mem_vdev2d import (
        rowwise_combine_get,
        rowwise_combine_get_scaled,
        rowwise_dispatch_put,
        rowwise_dispatch_put_scaled,
    )

    group = dist.group.WORLD
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    device = torch.device("cuda", local_rank)

    torch.manual_seed(1337 + rank)
    x = torch.randn((args.rows, args.cols), device=device, dtype=torch.bfloat16)
    x.add_(rank)

    total_experts = world_size * args.experts_per_rank
    route_ids = torch.arange(args.rows * args.top_k, device=device, dtype=torch.int64).view(
        args.rows, args.top_k
    )
    indices = (route_ids + rank * 17) % total_experts
    if args.drop_period > 0:
        drop_mask = ((route_ids + rank * 3) % args.drop_period) == 0
        indices = torch.where(drop_mask, torch.full_like(indices, -1), indices)

    gathered_indices = [torch.empty_like(indices) for _ in range(world_size)]
    dist.all_gather(gathered_indices, indices, group=group)
    all_indices_cpu = torch.stack(gathered_indices, dim=0).cpu()

    dst_ranks_all, dst_rows_all, per_rank_counts, capacity = _build_global_row_maps(
        all_indices_cpu,
        world_size,
        args.experts_per_rank,
    )

    dst_ranks = dst_ranks_all[rank].to(device=device, dtype=torch.int64)
    dst_rows = dst_rows_all[rank].to(device=device, dtype=torch.int64)

    group_name = _setup_nvshmem_backend(group, device)

    dispatch_out_bf16 = _alloc_rendezvous_symm_tensor(
        (capacity, args.cols), torch.bfloat16, device, group
    )
    dispatch_out_fp8_q = _alloc_rendezvous_symm_tensor(
        (capacity, args.cols), torch.float8_e4m3fn, device, group
    )
    dispatch_out_fp8_scales = _alloc_rendezvous_symm_tensor(
        (capacity, args.cols // 32),
        torch.float8_e8m0fnu,
        device,
        group,
    )

    combine_out = torch.empty((args.rows, args.cols), device=device, dtype=torch.bfloat16)

    def _bf16_path():
        rowwise_dispatch_put(
            x,
            dispatch_out_bf16,
            dst_ranks,
            dst_rows,
            group_name,
            nblocks=args.nblocks,
        )
        rowwise_combine_get(
            dispatch_out_bf16,
            combine_out,
            dst_ranks,
            dst_rows,
            group_name,
            nblocks=args.nblocks,
        )

    def _fp8_path():
        rowwise_dispatch_put_scaled(
            x,
            dispatch_out_fp8_q,
            dispatch_out_fp8_scales,
            dst_ranks,
            dst_rows,
            group_name,
            block_size=32,
            nblocks=args.nblocks,
        )
        rowwise_combine_get_scaled(
            dispatch_out_fp8_q,
            dispatch_out_fp8_scales,
            combine_out,
            dst_ranks,
            dst_rows,
            group_name,
            block_size=32,
            nblocks=args.nblocks,
        )

    for _ in range(args.warmup_iters):
        _bf16_path()
        _fp8_path()
    torch.cuda.synchronize()

    bf16_ms = _event_timed_ms(_bf16_path, args.iters)
    fp8_ms = _event_timed_ms(_fp8_path, args.iters)

    bf16_mean = mean(bf16_ms)
    fp8_mean = mean(fp8_ms)

    if rank == 0:
        print("Rowwise dispatch+combine")
        print(f"  bf16 mean: {bf16_mean:.3f} ms")
        print(f"  fp8  mean: {fp8_mean:.3f} ms")
        print(f"  speedup:   {bf16_mean / max(fp8_mean, 1e-6):.3f}x")


if __name__ == "__main__":
    main()
