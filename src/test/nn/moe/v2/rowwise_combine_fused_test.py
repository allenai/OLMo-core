import argparse
import os
import sys
from pathlib import Path
from statistics import mean
from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

# Example:
# torchrun --standalone --nproc-per-node=4 \
#   /workspace/OLMo-core/src/test/nn/moe/v2/rowwise_combine_fused_test.py


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare rowwise_combine_get_fused vs rowwise_combine_get for speed and accuracy."
    )
    parser.add_argument("--rows", type=int, default=4096, help="Output rows N per rank.")
    parser.add_argument("--cols", type=int, default=2048, help="Hidden dim D.")
    parser.add_argument("--top-k", type=int, default=4, help="Routes per row K.")
    parser.add_argument(
        "--expert-capacity",
        type=int,
        default=4096,
        help="Rows C in expert_out [C, D] per rank.",
    )
    parser.add_argument(
        "--drop-period",
        type=int,
        default=29,
        help="Set src route invalid when (route_id + rank*7) %% drop_period == 0. <=0 disables drops.",
    )
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--nblocks",
        type=int,
        default=int(os.environ.get("TORCH_SYMMMEM_NBLOCKS", "128")),
        help="Kernel launch blocks for combine kernels.",
    )
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


def _setup_nvshmem_backend(group: dist.ProcessGroup, device: torch.device) -> str:
    if not symm_mem.is_nvshmem_available():
        raise RuntimeError("NVSHMEM is not available in this environment.")

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


def _load_combine_callables():
    src_root = Path(__file__).resolve().parents[4]
    src_root_str = str(src_root)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)

    from olmo_core.kernels.symm_mem_vdev2d import (
        rowwise_combine_get,
        rowwise_combine_get_fused,
    )

    return rowwise_combine_get, rowwise_combine_get_fused


def _make_reference(
    all_expert_out: torch.Tensor,
    src_ranks: torch.Tensor,
    src_rows: torch.Tensor,
    probs: torch.Tensor,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, top_k = src_ranks.shape
    cols = all_expert_out.shape[2]
    routes = rows * top_k
    flat_ranks = src_ranks.reshape(routes)
    flat_rows = src_rows.reshape(routes)
    valid = (flat_ranks >= 0) & (flat_rows >= 0)

    gathered = torch.zeros((routes, cols), dtype=out_dtype, device=src_ranks.device)
    if valid.any():
        vr = flat_ranks[valid].to(dtype=torch.long)
        vw = flat_rows[valid].to(dtype=torch.long)
        gathered[valid] = all_expert_out[vr, vw]

    gathered_3d = gathered.view(rows, top_k, cols)
    gathered_f32 = gathered_3d.to(dtype=torch.float32)
    ref_unweighted = gathered_f32.sum(dim=1).to(dtype=out_dtype)
    ref_weighted = (gathered_f32 * probs.unsqueeze(-1)).sum(dim=1).to(dtype=out_dtype)
    return ref_unweighted, ref_weighted


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

    if args.rows <= 0 or args.cols <= 0 or args.top_k <= 0:
        raise RuntimeError("rows, cols, and top-k must all be > 0")
    if args.expert_capacity <= 0:
        raise RuntimeError("expert-capacity must be > 0")

    rowwise_combine_get, rowwise_combine_get_fused = _load_combine_callables()
    group_name = _setup_nvshmem_backend(group, device)

    torch.manual_seed(777 + rank)
    expert_out = _alloc_rendezvous_symm_tensor(
        (args.expert_capacity, args.cols), dtype, device, group
    )
    expert_out.copy_(torch.randn_like(expert_out))

    route_ids = torch.arange(args.rows * args.top_k, device=device, dtype=torch.int64).view(
        args.rows, args.top_k
    )
    src_ranks = (route_ids + rank * 11) % world_size
    src_rows = (route_ids * 17 + rank * 13) % args.expert_capacity
    if args.drop_period > 0:
        drop_mask = ((route_ids + rank * 7) % args.drop_period) == 0
        src_ranks = torch.where(drop_mask, torch.full_like(src_ranks, -1), src_ranks)
        src_rows = torch.where(drop_mask, torch.full_like(src_rows, -1), src_rows)

    probs = torch.rand((args.rows, args.top_k), device=device, dtype=torch.float32)
    probs = probs * (src_ranks >= 0).to(dtype=torch.float32)

    baseline_unweighted = torch.empty((args.rows, args.cols), device=device, dtype=dtype)
    fused_unweighted = torch.empty_like(baseline_unweighted)
    baseline_weighted = torch.empty_like(baseline_unweighted)
    fused_weighted = torch.empty_like(baseline_unweighted)

    def run_baseline_unweighted() -> None:
        rowwise_combine_get(
            expert_out,
            baseline_unweighted,
            src_ranks,
            src_rows,
            group_name,
            nblocks=args.nblocks,
        )

    def run_fused_unweighted() -> None:
        rowwise_combine_get_fused(
            expert_out,
            fused_unweighted,
            src_ranks,
            src_rows,
            group_name,
            nblocks=args.nblocks,
        )

    def run_baseline_weighted() -> None:
        rowwise_combine_get(
            expert_out,
            baseline_weighted,
            src_ranks,
            src_rows,
            group_name,
            probs=probs,
            nblocks=args.nblocks,
        )

    def run_fused_weighted() -> None:
        rowwise_combine_get_fused(
            expert_out,
            fused_weighted,
            src_ranks,
            src_rows,
            group_name,
            probs=probs,
            nblocks=args.nblocks,
        )

    dist.barrier(group=group)
    for _ in range(args.warmup_iters):
        run_baseline_unweighted()
    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    baseline_unweighted_ms = _event_timed_ms(run_baseline_unweighted, args.iters)

    dist.barrier(group=group)
    for _ in range(args.warmup_iters):
        run_fused_unweighted()
    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    fused_unweighted_ms = _event_timed_ms(run_fused_unweighted, args.iters)

    dist.barrier(group=group)
    for _ in range(args.warmup_iters):
        run_baseline_weighted()
    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    baseline_weighted_ms = _event_timed_ms(run_baseline_weighted, args.iters)

    dist.barrier(group=group)
    for _ in range(args.warmup_iters):
        run_fused_weighted()
    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    fused_weighted_ms = _event_timed_ms(run_fused_weighted, args.iters)

    # Correctness pass after timed loops.
    dist.barrier(group=group)
    run_baseline_unweighted()
    run_fused_unweighted()
    run_baseline_weighted()
    run_fused_weighted()
    torch.cuda.synchronize(device)
    dist.barrier(group=group)

    # Build direct reference from gathered expert_out to catch correlated bugs.
    gathered_expert_out = [torch.empty_like(expert_out) for _ in range(world_size)]
    dist.all_gather(gathered_expert_out, expert_out, group=group)
    all_expert_out = torch.stack(gathered_expert_out, dim=0)
    ref_unweighted, ref_weighted = _make_reference(
        all_expert_out,
        src_ranks,
        src_rows,
        probs,
        baseline_unweighted.dtype,
    )

    if dtype == torch.float32:
        atol = 1e-5
        rtol = 1e-5
    else:
        atol = 2e-2
        rtol = 2e-2

    close_baseline_unweighted = torch.allclose(
        baseline_unweighted, ref_unweighted, atol=atol, rtol=rtol
    )
    close_fused_unweighted = torch.allclose(fused_unweighted, ref_unweighted, atol=atol, rtol=rtol)
    close_baseline_weighted = torch.allclose(baseline_weighted, ref_weighted, atol=atol, rtol=rtol)
    close_fused_weighted = torch.allclose(fused_weighted, ref_weighted, atol=atol, rtol=rtol)
    close_baseline_vs_fused_unweighted = torch.allclose(
        baseline_unweighted, fused_unweighted, atol=atol, rtol=rtol
    )
    close_baseline_vs_fused_weighted = torch.allclose(
        baseline_weighted, fused_weighted, atol=atol, rtol=rtol
    )

    max_abs_diff_unweighted = float((baseline_unweighted - fused_unweighted).abs().max().item())
    max_abs_diff_weighted = float((baseline_weighted - fused_weighted).abs().max().item())
    max_abs_diff_ref_baseline = float((baseline_unweighted - ref_unweighted).abs().max().item())
    max_abs_diff_ref_fused = float((fused_unweighted - ref_unweighted).abs().max().item())
    max_abs_diff_ref_baseline_weighted = float(
        (baseline_weighted - ref_weighted).abs().max().item()
    )
    max_abs_diff_ref_fused_weighted = float((fused_weighted - ref_weighted).abs().max().item())

    local_ok = (
        close_baseline_unweighted
        and close_fused_unweighted
        and close_baseline_weighted
        and close_fused_weighted
        and close_baseline_vs_fused_unweighted
        and close_baseline_vs_fused_weighted
    )
    ok_tensor = torch.tensor([1 if local_ok else 0], device=device, dtype=torch.int32)
    dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN, group=group)
    global_ok = bool(ok_tensor.item() == 1)

    valid_routes = int(((src_ranks >= 0) & (src_rows >= 0)).sum().item())
    row_bytes = args.cols * torch.tensor([], dtype=dtype).element_size()
    traffic_gb = (valid_routes * row_bytes) / 1e9
    result = {
        "rank": rank,
        "valid_routes": valid_routes,
        "baseline_unweighted_ms": mean(baseline_unweighted_ms),
        "fused_unweighted_ms": mean(fused_unweighted_ms),
        "baseline_weighted_ms": mean(baseline_weighted_ms),
        "fused_weighted_ms": mean(fused_weighted_ms),
        "traffic_gb": traffic_gb,
        "close_baseline_unweighted": bool(close_baseline_unweighted),
        "close_fused_unweighted": bool(close_fused_unweighted),
        "close_baseline_weighted": bool(close_baseline_weighted),
        "close_fused_weighted": bool(close_fused_weighted),
        "close_baseline_vs_fused_unweighted": bool(close_baseline_vs_fused_unweighted),
        "close_baseline_vs_fused_weighted": bool(close_baseline_vs_fused_weighted),
        "max_abs_diff_unweighted": max_abs_diff_unweighted,
        "max_abs_diff_weighted": max_abs_diff_weighted,
        "max_abs_diff_ref_baseline": max_abs_diff_ref_baseline,
        "max_abs_diff_ref_fused": max_abs_diff_ref_fused,
        "max_abs_diff_ref_baseline_weighted": max_abs_diff_ref_baseline_weighted,
        "max_abs_diff_ref_fused_weighted": max_abs_diff_ref_fused_weighted,
    }
    gathered = [None] * world_size
    dist.all_gather_object(gathered, result, group=group)

    if rank == 0:
        print("=== Rowwise Combine Fused vs Baseline ===", flush=True)
        print(
            f"world_size={world_size} rows={args.rows} cols={args.cols} top_k={args.top_k} "
            f"expert_capacity={args.expert_capacity} dtype={dtype} warmup={args.warmup_iters} "
            f"iters={args.iters} nblocks={args.nblocks}",
            flush=True,
        )
        for item in sorted(gathered, key=lambda x: x["rank"]):
            bw_base_unweighted = item["traffic_gb"] / (item["baseline_unweighted_ms"] / 1e3)
            bw_fused_unweighted = item["traffic_gb"] / (item["fused_unweighted_ms"] / 1e3)
            bw_base_weighted = item["traffic_gb"] / (item["baseline_weighted_ms"] / 1e3)
            bw_fused_weighted = item["traffic_gb"] / (item["fused_weighted_ms"] / 1e3)
            speedup_unweighted = item["baseline_unweighted_ms"] / item["fused_unweighted_ms"]
            speedup_weighted = item["baseline_weighted_ms"] / item["fused_weighted_ms"]
            status = "PASS" if global_ok else "FAIL"
            print(
                f"rank={item['rank']} status={status} routes={item['valid_routes']} "
                f"base_unw={item['baseline_unweighted_ms']:.3f}ms "
                f"fused_unw={item['fused_unweighted_ms']:.3f}ms "
                f"speedup_unw={speedup_unweighted:.3f}x "
                f"bw_unw_base={bw_base_unweighted:.2f}GB/s bw_unw_fused={bw_fused_unweighted:.2f}GB/s "
                f"base_w={item['baseline_weighted_ms']:.3f}ms "
                f"fused_w={item['fused_weighted_ms']:.3f}ms "
                f"speedup_w={speedup_weighted:.3f}x "
                f"bw_w_base={bw_base_weighted:.2f}GB/s bw_w_fused={bw_fused_weighted:.2f}GB/s "
                f"max_diff_unw={item['max_abs_diff_unweighted']:.6f} "
                f"max_diff_w={item['max_abs_diff_weighted']:.6f}",
                flush=True,
            )
            if not global_ok:
                print(
                    f"  close(base_unw/ref)={item['close_baseline_unweighted']} "
                    f"close(fused_unw/ref)={item['close_fused_unweighted']} "
                    f"close(base_w/ref)={item['close_baseline_weighted']} "
                    f"close(fused_w/ref)={item['close_fused_weighted']} "
                    f"close(base_vs_fused_unw)={item['close_baseline_vs_fused_unweighted']} "
                    f"close(base_vs_fused_w)={item['close_baseline_vs_fused_weighted']} "
                    f"max_ref_diff_base_unw={item['max_abs_diff_ref_baseline']:.6f} "
                    f"max_ref_diff_fused_unw={item['max_abs_diff_ref_fused']:.6f} "
                    f"max_ref_diff_base_w={item['max_abs_diff_ref_baseline_weighted']:.6f} "
                    f"max_ref_diff_fused_w={item['max_abs_diff_ref_fused_weighted']:.6f}",
                    flush=True,
                )

    dist.barrier(group=group)
    if not global_ok:
        raise RuntimeError(
            "rowwise_combine_get_fused correctness check failed. See rank-0 summary for details."
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
