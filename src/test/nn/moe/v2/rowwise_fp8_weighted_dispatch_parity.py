import argparse
import os
import sys
from statistics import mean, median
from typing import List, Tuple

import torch
import torch.distributed as dist


# Developer-only artifact for the direct CUDA/NVSHMEM weighted MXFP8 dispatch
# experiment. The kernel is kept in the extension for future tuning, but the
# production MoE backward path intentionally does not call it because B300
# benchmarks showed exact parity without a speedup at normal rowwise settings.


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Parity/benchmark for fused weighted rowwise MXFP8 dispatch."
    )
    p.add_argument("--rows", type=int, default=16384)
    p.add_argument("--cols", type=int, default=4096)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--experts-per-rank", type=int, default=4)
    p.add_argument("--drop-period", type=int, default=0)
    p.add_argument("--warmup-iters", type=int, default=10)
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--nblocks", type=int, default=128)
    p.add_argument("--atol", type=float, default=0.0)
    p.add_argument("--rtol", type=float, default=0.0)
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
    from olmo_core.kernels import olmo_symm_mem

    os.environ["OLMO_USE_OWN_SYMM_MEM"] = "1"
    olmo_symm_mem.register_group(group, device=device)
    return group.group_name


def _alloc_rendezvous_symm_tensor(
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    from olmo_core.kernels import olmo_symm_mem

    t = olmo_symm_mem.empty(shape, dtype=dtype, device=device, group=group)
    olmo_symm_mem.rendezvous(t, group=group)
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
                    raise RuntimeError(
                        f"Found expert id {expert} >= total_experts={total_experts}"
                    )
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
        raise RuntimeError(f"cols must be divisible by 32, got {args.cols}")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))

    from olmo_core.kernels.mxfp8_utils import dequantize_rows_from_mxfp8
    from olmo_core.kernels.symm_mem_vdev2d import (
        rowwise_dispatch_put_scaled,
        rowwise_dispatch_put_scaled_weighted,
    )

    group = dist.group.WORLD
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    device = torch.device("cuda", local_rank)

    torch.manual_seed(9001 + rank)
    x = torch.randn((args.rows, args.cols), device=device, dtype=torch.bfloat16)
    probs = torch.rand((args.rows, args.top_k), device=device, dtype=torch.float32)

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
    local_count = int(per_rank_counts[rank])

    dst_ranks = dst_ranks_all[rank].to(device=device, dtype=torch.int64)
    dst_rows = dst_rows_all[rank].to(device=device, dtype=torch.int64)
    group_name = _setup_nvshmem_backend(group, device)

    old_q = _alloc_rendezvous_symm_tensor(
        (capacity, args.cols), torch.float8_e4m3fn, device, group
    )
    old_scales = _alloc_rendezvous_symm_tensor(
        (capacity, args.cols // 32), torch.float8_e8m0fnu, device, group
    )
    new_q = _alloc_rendezvous_symm_tensor(
        (capacity, args.cols), torch.float8_e4m3fn, device, group
    )
    new_scales = _alloc_rendezvous_symm_tensor(
        (capacity, args.cols // 32), torch.float8_e8m0fnu, device, group
    )

    def _old_path() -> None:
        weighted = (
            x.unsqueeze(1) * probs.to(dtype=x.dtype).unsqueeze(-1)
        ).reshape(args.rows * args.top_k, args.cols)
        if not weighted.is_contiguous():
            weighted = weighted.contiguous()
        rowwise_dispatch_put_scaled(
            weighted,
            old_q,
            old_scales,
            dst_ranks.reshape(-1, 1).contiguous(),
            dst_rows.reshape(-1, 1).contiguous(),
            group_name,
            block_size=32,
            nblocks=args.nblocks,
        )

    def _new_path() -> None:
        rowwise_dispatch_put_scaled_weighted(
            x,
            probs,
            new_q,
            new_scales,
            dst_ranks,
            dst_rows,
            group_name,
            block_size=32,
            nblocks=args.nblocks,
        )

    _old_path()
    _new_path()
    torch.cuda.synchronize()

    old_hp = dequantize_rows_from_mxfp8(
        old_q[:local_count],
        old_scales[:local_count],
        block_size=32,
        out_dtype=torch.bfloat16,
    )
    new_hp = dequantize_rows_from_mxfp8(
        new_q[:local_count],
        new_scales[:local_count],
        block_size=32,
        out_dtype=torch.bfloat16,
    )
    torch.testing.assert_close(new_hp, old_hp, atol=args.atol, rtol=args.rtol)

    q_mismatch = (
        new_q[:local_count].view(torch.uint8) != old_q[:local_count].view(torch.uint8)
    ).sum()
    scale_mismatch = (
        new_scales[:local_count].view(torch.uint8)
        != old_scales[:local_count].view(torch.uint8)
    ).sum()
    max_abs = (new_hp.float() - old_hp.float()).abs().max()
    stats = torch.tensor(
        [
            float(q_mismatch.item()),
            float(scale_mismatch.item()),
            float(max_abs.item()),
        ],
        device=device,
    )
    dist.all_reduce(stats, op=dist.ReduceOp.MAX, group=group)

    for _ in range(args.warmup_iters):
        _old_path()
        _new_path()
    torch.cuda.synchronize()

    old_ms = _event_timed_ms(_old_path, args.iters)
    new_ms = _event_timed_ms(_new_path, args.iters)
    local_timing = torch.tensor(
        [
            mean(old_ms),
            median(old_ms),
            min(old_ms),
            mean(new_ms),
            median(new_ms),
            min(new_ms),
        ],
        device=device,
    )
    dist.all_reduce(local_timing, op=dist.ReduceOp.MAX, group=group)

    if rank == 0:
        print("Weighted rowwise MXFP8 dispatch parity")
        print(f"  rows={args.rows} cols={args.cols} top_k={args.top_k} nblocks={args.nblocks}")
        print(f"  max q byte mismatches:     {int(stats[0].item())}")
        print(f"  max scale byte mismatches: {int(stats[1].item())}")
        print(f"  max dequant abs diff:      {stats[2].item():.6g}")
        print("Weighted rowwise MXFP8 dispatch timing, max across ranks")
        print(
            "  old materialized mean/median/min: "
            f"{local_timing[0].item():.3f} / {local_timing[1].item():.3f} / {local_timing[2].item():.3f} ms"
        )
        print(
            "  fused cuda      mean/median/min: "
            f"{local_timing[3].item():.3f} / {local_timing[4].item():.3f} / {local_timing[5].item():.3f} ms"
        )
        print(f"  speedup(mean): {local_timing[0].item() / max(local_timing[3].item(), 1e-6):.3f}x")

    dist.barrier(group=group)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
