import argparse
import os
import sys
from pathlib import Path
from statistics import mean
from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test GPU-side row-wise NVSHMEM dispatch/combine extension."
    )
    parser.add_argument("--rows", type=int, default=8192, help="Local token rows N per rank.")
    parser.add_argument("--cols", type=int, default=2048, help="Hidden dim D.")
    parser.add_argument("--top-k", type=int, default=4, help="Routes per token K.")
    parser.add_argument(
        "--experts-per-rank",
        type=int,
        default=4,
        help="Experts per rank; total experts = world_size * experts_per_rank.",
    )
    parser.add_argument(
        "--drop-period",
        type=int,
        default=31,
        help="Route i is dropped when (i + rank*3) %% drop_period == 0. Set <=0 to disable drops.",
    )
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--nblocks",
        type=int,
        default=0,
        help="Grid blocks for row-wise kernels. 0 means auto.",
    )
    parser.add_argument(
        "--mode",
        choices=["direct_warp", "packed_peer"],
        default="direct_warp",
        help="Row-wise transport mode: direct warp-level GET/PUT or packed-by-peer all_to_all_vdev.",
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
        raise RuntimeError(
            "NVSHMEM is not available in this environment. Cannot run row-wise NVSHMEM test."
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


def _load_rowwise_callables():
    src_root = Path(__file__).resolve().parents[4]
    src_root_str = str(src_root)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)

    from olmo_core.kernels.symm_mem_vdev2d import (
        all_to_all_vdev_2d_offset_nblocks,
        rowwise_combine_get,
        rowwise_dispatch_put,
    )

    return rowwise_dispatch_put, rowwise_combine_get, all_to_all_vdev_2d_offset_nblocks


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


def _build_inverse_combine_maps(
    dst_ranks_all: torch.Tensor,
    dst_rows_all: torch.Tensor,
    *,
    world_size: int,
    rows: int,
    top_k: int,
    capacity: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # For each expert row on each rank, map it back to the source rank/route slot
    # (route slot = token_id * K + k).
    combine_dst_ranks = torch.full((world_size, capacity), -1, dtype=torch.int64)
    combine_dst_routes = torch.full((world_size, capacity), -1, dtype=torch.int64)
    for src_rank in range(world_size):
        for n in range(rows):
            base = n * top_k
            for k in range(top_k):
                d_rank = int(dst_ranks_all[src_rank, n, k].item())
                if d_rank < 0:
                    continue
                d_row = int(dst_rows_all[src_rank, n, k].item())
                combine_dst_ranks[d_rank, d_row] = src_rank
                combine_dst_routes[d_rank, d_row] = base + k
    return combine_dst_ranks, combine_dst_routes


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
    if args.experts_per_rank <= 0:
        raise RuntimeError("experts-per-rank must be > 0")

    (
        rowwise_dispatch_put,
        rowwise_combine_get,
        all_to_all_vdev_2d_offset_nblocks,
    ) = _load_rowwise_callables()

    torch.manual_seed(1337 + rank)
    x = torch.randn((args.rows, args.cols), device=device, dtype=dtype)
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
    gathered_inputs = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(gathered_indices, indices, group=group)
    dist.all_gather(gathered_inputs, x, group=group)
    all_indices_cpu = torch.stack(gathered_indices, dim=0).cpu()
    all_inputs_cpu = torch.stack(gathered_inputs, dim=0).cpu()

    dst_ranks_all, dst_rows_all, per_rank_counts, capacity = _build_global_row_maps(
        all_indices_cpu,
        world_size,
        args.experts_per_rank,
    )

    dst_ranks = dst_ranks_all[rank].to(device=device, dtype=torch.int64)
    dst_rows = dst_rows_all[rank].to(device=device, dtype=torch.int64)
    src_ranks = dst_ranks
    src_rows = dst_rows
    valid_route_mask = dst_ranks >= 0
    local_valid_routes = int(valid_route_mask.sum().item())

    group_name = _setup_nvshmem_backend(group, device)
    dispatch_out = _alloc_rendezvous_symm_tensor((capacity, args.cols), dtype, device, group)
    combine_out = torch.empty((args.rows, args.cols), device=device, dtype=dtype)
    combine_out_weighted = torch.empty((args.rows, args.cols), device=device, dtype=dtype)

    expected_dispatch_cpu = torch.zeros((capacity, args.cols), dtype=all_inputs_cpu.dtype)
    for src_rank in range(world_size):
        for n in range(args.rows):
            for k in range(args.top_k):
                d_rank = int(dst_ranks_all[src_rank, n, k].item())
                if d_rank != rank:
                    continue
                d_row = int(dst_rows_all[src_rank, n, k].item())
                expected_dispatch_cpu[d_row].copy_(all_inputs_cpu[src_rank, n])
    expected_dispatch = expected_dispatch_cpu.to(device=device, dtype=dtype)

    valid_counts = (dst_ranks >= 0).sum(dim=1).to(dtype=dtype)
    expected_combine = x * valid_counts.unsqueeze(1)

    torch.manual_seed(9001 + rank)
    probs = torch.rand((args.rows, args.top_k), device=device, dtype=torch.float32)
    probs = probs * valid_route_mask.to(dtype=torch.float32)
    probs_for_mul = probs.to(dtype=dtype)
    expected_weighted = x * probs.sum(dim=1, dtype=torch.float32).to(dtype).unsqueeze(1)

    # direct_warp mode uses custom CUDA extension kernels.
    # packed_peer mode groups transfers by peer rank and uses all_to_all_vdev for
    # payload + metadata (destination row / destination route id).
    if args.mode == "packed_peer":
        token_ids = (
            torch.arange(args.rows, device=device, dtype=torch.int64)
            .view(args.rows, 1)
            .expand(args.rows, args.top_k)
        )
        dispatch_send_token_ids = token_ids[valid_route_mask]
        dispatch_send_dst_ranks = dst_ranks[valid_route_mask]
        dispatch_send_dst_rows = dst_rows[valid_route_mask]
        if dispatch_send_dst_ranks.numel() > 0:
            dispatch_order = torch.argsort(dispatch_send_dst_ranks, stable=True)
            dispatch_send_token_ids = dispatch_send_token_ids.index_select(0, dispatch_order)
            dispatch_send_dst_ranks = dispatch_send_dst_ranks.index_select(0, dispatch_order)
            dispatch_send_dst_rows = dispatch_send_dst_rows.index_select(0, dispatch_order)
            dispatch_send_splits = torch.bincount(
                dispatch_send_dst_ranks, minlength=world_size
            ).to(dtype=torch.int64)
        else:
            dispatch_send_splits = torch.zeros((world_size,), device=device, dtype=torch.int64)
        dispatch_send_count = int(dispatch_send_token_ids.numel())

        combine_dst_ranks_all, combine_dst_routes_all = _build_inverse_combine_maps(
            dst_ranks_all,
            dst_rows_all,
            world_size=world_size,
            rows=args.rows,
            top_k=args.top_k,
            capacity=capacity,
        )
        local_recv_rows = per_rank_counts[rank]
        local_combine_dst_ranks = combine_dst_ranks_all[rank, :local_recv_rows].to(
            device=device, dtype=torch.int64
        )
        local_combine_dst_routes = combine_dst_routes_all[rank, :local_recv_rows].to(
            device=device, dtype=torch.int64
        )
        local_combine_src_rows = torch.arange(
            local_recv_rows, device=device, dtype=torch.int64
        )
        if local_recv_rows > 0:
            combine_order = torch.argsort(local_combine_dst_ranks, stable=True)
            local_combine_dst_ranks = local_combine_dst_ranks.index_select(0, combine_order)
            local_combine_dst_routes = local_combine_dst_routes.index_select(0, combine_order)
            local_combine_src_rows = local_combine_src_rows.index_select(0, combine_order)
            combine_send_splits = torch.bincount(
                local_combine_dst_ranks, minlength=world_size
            ).to(dtype=torch.int64)
        else:
            combine_send_splits = torch.zeros((world_size,), device=device, dtype=torch.int64)
        combine_send_count = int(local_recv_rows)

        dispatch_send_cap = max(1, args.rows * args.top_k)
        dispatch_recv_cap = max(1, capacity)
        combine_send_cap = max(1, capacity)
        combine_recv_cap = max(1, args.rows * args.top_k)

        # Dispatch packed buffers.
        packed_dispatch_send_payload = _alloc_rendezvous_symm_tensor(
            (dispatch_send_cap, args.cols), dtype, device, group
        )
        packed_dispatch_recv_payload = _alloc_rendezvous_symm_tensor(
            (dispatch_recv_cap, args.cols), dtype, device, group
        )
        packed_dispatch_send_rows = _alloc_rendezvous_symm_tensor(
            (dispatch_send_cap, 1), torch.int64, device, group
        )
        packed_dispatch_recv_rows = _alloc_rendezvous_symm_tensor(
            (dispatch_recv_cap, 1), torch.int64, device, group
        )
        packed_dispatch_in_splits_offsets = _alloc_rendezvous_symm_tensor(
            (2, world_size), torch.int64, device, group
        )
        packed_dispatch_out_splits_offsets_payload = _alloc_rendezvous_symm_tensor(
            (2, world_size), torch.int64, device, group
        )
        packed_dispatch_out_splits_offsets_rows = _alloc_rendezvous_symm_tensor(
            (2, world_size), torch.int64, device, group
        )

        dispatch_send_offsets = torch.zeros((world_size,), device=device, dtype=torch.int64)
        if world_size > 1:
            dispatch_send_offsets[1:] = torch.cumsum(dispatch_send_splits, dim=0)[:-1]
        packed_dispatch_in_splits_offsets[0].copy_(dispatch_send_splits)
        packed_dispatch_in_splits_offsets[1].copy_(dispatch_send_offsets)
        if dispatch_send_count > 0:
            packed_dispatch_send_rows[:dispatch_send_count, 0].copy_(dispatch_send_dst_rows)

        # Combine packed buffers.
        packed_combine_send_payload = _alloc_rendezvous_symm_tensor(
            (combine_send_cap, args.cols), dtype, device, group
        )
        packed_combine_recv_payload = _alloc_rendezvous_symm_tensor(
            (combine_recv_cap, args.cols), dtype, device, group
        )
        packed_combine_send_routes = _alloc_rendezvous_symm_tensor(
            (combine_send_cap, 1), torch.int64, device, group
        )
        packed_combine_recv_routes = _alloc_rendezvous_symm_tensor(
            (combine_recv_cap, 1), torch.int64, device, group
        )
        packed_combine_in_splits_offsets = _alloc_rendezvous_symm_tensor(
            (2, world_size), torch.int64, device, group
        )
        packed_combine_out_splits_offsets_payload = _alloc_rendezvous_symm_tensor(
            (2, world_size), torch.int64, device, group
        )
        packed_combine_out_splits_offsets_routes = _alloc_rendezvous_symm_tensor(
            (2, world_size), torch.int64, device, group
        )
        packed_gathered_flat = torch.zeros(
            (args.rows * args.top_k, args.cols), device=device, dtype=dtype
        )

        combine_send_offsets = torch.zeros((world_size,), device=device, dtype=torch.int64)
        if world_size > 1:
            combine_send_offsets[1:] = torch.cumsum(combine_send_splits, dim=0)[:-1]
        packed_combine_in_splits_offsets[0].copy_(combine_send_splits)
        packed_combine_in_splits_offsets[1].copy_(combine_send_offsets)
        if combine_send_count > 0:
            packed_combine_send_routes[:combine_send_count, 0].copy_(local_combine_dst_routes)

    def run_dispatch() -> None:
        dispatch_out.zero_()
        if args.mode == "direct_warp":
            rowwise_dispatch_put(
                x,
                dispatch_out,
                dst_ranks,
                dst_rows,
                group_name,
                nblocks=args.nblocks,
            )
            return

        # packed_peer dispatch
        if dispatch_send_count > 0:
            packed_dispatch_send_payload[:dispatch_send_count].copy_(
                x.index_select(0, dispatch_send_token_ids)
            )
        all_to_all_vdev_2d_offset_nblocks(
            packed_dispatch_send_payload,
            packed_dispatch_recv_payload,
            packed_dispatch_in_splits_offsets,
            packed_dispatch_out_splits_offsets_payload,
            group_name,
            nblocks=args.nblocks,
        )
        all_to_all_vdev_2d_offset_nblocks(
            packed_dispatch_send_rows,
            packed_dispatch_recv_rows,
            packed_dispatch_in_splits_offsets,
            packed_dispatch_out_splits_offsets_rows,
            group_name,
            nblocks=args.nblocks,
        )

        recv_splits = packed_dispatch_out_splits_offsets_payload[0]
        recv_offsets = packed_dispatch_out_splits_offsets_payload[1]
        for peer in range(world_size):
            chunk = int(recv_splits[peer].item())
            if chunk == 0:
                continue
            offset = int(recv_offsets[peer].item())
            dst_rows_chunk = packed_dispatch_recv_rows[offset : offset + chunk, 0]
            payload_chunk = packed_dispatch_recv_payload[offset : offset + chunk]
            dispatch_out.index_copy_(0, dst_rows_chunk, payload_chunk)

    def run_combine() -> None:
        if args.mode == "direct_warp":
            rowwise_combine_get(
                dispatch_out,
                combine_out,
                src_ranks,
                src_rows,
                group_name,
                nblocks=args.nblocks,
            )
            return

        # packed_peer combine
        if combine_send_count > 0:
            packed_combine_send_payload[:combine_send_count].copy_(
                dispatch_out.index_select(0, local_combine_src_rows)
            )
        all_to_all_vdev_2d_offset_nblocks(
            packed_combine_send_payload,
            packed_combine_recv_payload,
            packed_combine_in_splits_offsets,
            packed_combine_out_splits_offsets_payload,
            group_name,
            nblocks=args.nblocks,
        )
        all_to_all_vdev_2d_offset_nblocks(
            packed_combine_send_routes,
            packed_combine_recv_routes,
            packed_combine_in_splits_offsets,
            packed_combine_out_splits_offsets_routes,
            group_name,
            nblocks=args.nblocks,
        )

        packed_gathered_flat.zero_()
        recv_splits = packed_combine_out_splits_offsets_payload[0]
        recv_offsets = packed_combine_out_splits_offsets_payload[1]
        for peer in range(world_size):
            chunk = int(recv_splits[peer].item())
            if chunk == 0:
                continue
            offset = int(recv_offsets[peer].item())
            routes_chunk = packed_combine_recv_routes[offset : offset + chunk, 0]
            payload_chunk = packed_combine_recv_payload[offset : offset + chunk]
            packed_gathered_flat.index_copy_(0, routes_chunk, payload_chunk)

        gathered = packed_gathered_flat.view(args.rows, args.top_k, args.cols)
        combine_out.copy_(gathered.sum(dim=1))

    def run_combine_weighted() -> None:
        if args.mode == "direct_warp":
            rowwise_combine_get(
                dispatch_out,
                combine_out_weighted,
                src_ranks,
                src_rows,
                group_name,
                probs=probs,
                nblocks=args.nblocks,
            )
            return

        # packed_peer combine (weighted)
        if combine_send_count > 0:
            packed_combine_send_payload[:combine_send_count].copy_(
                dispatch_out.index_select(0, local_combine_src_rows)
            )
        all_to_all_vdev_2d_offset_nblocks(
            packed_combine_send_payload,
            packed_combine_recv_payload,
            packed_combine_in_splits_offsets,
            packed_combine_out_splits_offsets_payload,
            group_name,
            nblocks=args.nblocks,
        )
        all_to_all_vdev_2d_offset_nblocks(
            packed_combine_send_routes,
            packed_combine_recv_routes,
            packed_combine_in_splits_offsets,
            packed_combine_out_splits_offsets_routes,
            group_name,
            nblocks=args.nblocks,
        )

        packed_gathered_flat.zero_()
        recv_splits = packed_combine_out_splits_offsets_payload[0]
        recv_offsets = packed_combine_out_splits_offsets_payload[1]
        for peer in range(world_size):
            chunk = int(recv_splits[peer].item())
            if chunk == 0:
                continue
            offset = int(recv_offsets[peer].item())
            routes_chunk = packed_combine_recv_routes[offset : offset + chunk, 0]
            payload_chunk = packed_combine_recv_payload[offset : offset + chunk]
            packed_gathered_flat.index_copy_(0, routes_chunk, payload_chunk)

        gathered = packed_gathered_flat.view(args.rows, args.top_k, args.cols)
        combine_out_weighted.copy_((gathered * probs_for_mul.unsqueeze(-1)).sum(dim=1))

    dist.barrier(group=group)
    for _ in range(args.warmup_iters):
        run_dispatch()
    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    dispatch_times_ms = _event_timed_ms(run_dispatch, args.iters)

    dist.barrier(group=group)
    for _ in range(args.warmup_iters):
        run_combine()
    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    combine_times_ms = _event_timed_ms(run_combine, args.iters)

    dist.barrier(group=group)
    for _ in range(args.warmup_iters):
        run_combine_weighted()
    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    combine_weighted_times_ms = _event_timed_ms(run_combine_weighted, args.iters)

    # Keep final correctness collectives aligned across ranks.
    dist.barrier(group=group)
    run_dispatch()
    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    run_combine()
    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    run_combine_weighted()
    torch.cuda.synchronize(device)
    dist.barrier(group=group)

    equal_dispatch = torch.equal(dispatch_out, expected_dispatch)
    equal_combine = torch.equal(combine_out, expected_combine)
    max_abs_diff_combine = (
        float((combine_out - expected_combine).abs().max().item()) if not equal_combine else 0.0
    )
    close_combine_weighted = torch.allclose(combine_out_weighted, expected_weighted, atol=2e-2, rtol=2e-2)
    max_abs_diff_combine_weighted = (
        float((combine_out_weighted - expected_weighted).abs().max().item())
        if not close_combine_weighted
        else 0.0
    )

    gathered = [None] * world_size
    result = {
        "rank": rank,
        "dispatch_avg_ms": mean(dispatch_times_ms),
        "combine_avg_ms": mean(combine_times_ms),
        "combine_weighted_avg_ms": mean(combine_weighted_times_ms),
        "dispatch_routes": local_valid_routes,
        "combine_routes": local_valid_routes,
        "capacity": capacity,
        "recv_rows": per_rank_counts[rank],
        "equal_dispatch": bool(equal_dispatch),
        "equal_combine": bool(equal_combine),
        "close_combine_weighted": bool(close_combine_weighted),
        "max_abs_diff_combine": max_abs_diff_combine,
        "max_abs_diff_combine_weighted": max_abs_diff_combine_weighted,
    }
    dist.all_gather_object(gathered, result, group=group)

    if rank == 0:
        payload_gb = (args.rows * args.cols * torch.tensor([], dtype=dtype).element_size()) / 1e9
        row_gb = (args.cols * torch.tensor([], dtype=dtype).element_size()) / 1e9
        print("=== Row-wise NVSHMEM Dispatch/Combine Test ===", flush=True)
        print(
            f"world_size={world_size} rows={args.rows} cols={args.cols} top_k={args.top_k} "
            f"experts_per_rank={args.experts_per_rank} total_experts={total_experts} "
            f"dtype={dtype} warmup={args.warmup_iters} iters={args.iters} "
            f"mode={args.mode} nblocks={args.nblocks}",
            flush=True,
        )
        print(f"logical_payload_per_rank={payload_gb:.3f} GB", flush=True)
        for item in sorted(gathered, key=lambda z: z["rank"]):
            dispatch_logical_bw = payload_gb / (item["dispatch_avg_ms"] / 1e3)
            combine_logical_bw = payload_gb / (item["combine_avg_ms"] / 1e3)
            combine_weighted_logical_bw = payload_gb / (
                item["combine_weighted_avg_ms"] / 1e3
            )
            dispatch_traffic_gb = row_gb * item["dispatch_routes"]
            combine_traffic_gb = row_gb * item["combine_routes"]
            dispatch_traffic_bw = dispatch_traffic_gb / (item["dispatch_avg_ms"] / 1e3)
            combine_traffic_bw = combine_traffic_gb / (item["combine_avg_ms"] / 1e3)
            combine_weighted_traffic_bw = combine_traffic_gb / (
                item["combine_weighted_avg_ms"] / 1e3
            )
            dispatch_factor = (
                dispatch_traffic_gb / payload_gb if payload_gb > 0 else float("nan")
            )
            combine_factor = (
                combine_traffic_gb / payload_gb if payload_gb > 0 else float("nan")
            )
            pass_or_fail = (
                "PASS"
                if item["equal_dispatch"]
                and item["equal_combine"]
                and item["close_combine_weighted"]
                else "FAIL"
            )
            print(
                f"[{pass_or_fail}] rank={item['rank']} recv_rows={item['recv_rows']} cap={item['capacity']} "
                f"dispatch={item['dispatch_avg_ms']:.3f} ms "
                f"(logical {dispatch_logical_bw:.2f} GB/s, traffic {dispatch_traffic_bw:.2f} GB/s, x{dispatch_factor:.2f}) "
                f"combine={item['combine_avg_ms']:.3f} ms "
                f"(logical {combine_logical_bw:.2f} GB/s, traffic {combine_traffic_bw:.2f} GB/s, x{combine_factor:.2f}) "
                f"combine_w={item['combine_weighted_avg_ms']:.3f} ms "
                f"(logical {combine_weighted_logical_bw:.2f} GB/s, traffic {combine_weighted_traffic_bw:.2f} GB/s, x{combine_factor:.2f}) "
                f"equal_dispatch={item['equal_dispatch']} "
                f"equal_combine={item['equal_combine']} max_abs_diff_combine={item['max_abs_diff_combine']:.6f} "
                f"close_combine_weighted={item['close_combine_weighted']} "
                f"max_abs_diff_combine_weighted={item['max_abs_diff_combine_weighted']:.6f}",
                flush=True,
            )

    dist.barrier(group=group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
