from __future__ import annotations

import argparse
import os
import statistics
import time

import torch
import torch.distributed as dist

from olmo_core.kernels import olmo_symm_mem
from olmo_core.kernels import symm_mem_vdev2d as symm_mem_vdev2d_kernels


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone NVSHMEM rowwise GET/PUT smoke. This bypasses MoE "
            "routing/expert code and calls the rowwise kernels directly."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("gather_get", "combine_get", "combine_get_fused", "put_combine"),
        default="combine_get",
    )
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--expert-rows", type=int, default=5120)
    parser.add_argument("--dim", type=int, default=6144)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--nblocks", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument(
        "--peer-pattern",
        choices=("all_to_all", "ring", "self", "local_node", "remote_node"),
        default="all_to_all",
    )
    parser.add_argument(
        "--weighted",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass route probabilities to combine/reduce kernels.",
    )
    parser.add_argument(
        "--symmetric-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use symmetric output/gather scratch where the wrapper permits it.",
    )
    parser.add_argument(
        "--post-barrier",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Request a post NVSHMEM barrier in GET wrappers.",
    )
    parser.add_argument(
        "--check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Check the synthetic result against rank-coded expected values.",
    )
    parser.add_argument(
        "--scalar-meta-put",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use scalar nvshmem_int64_p writes for PUT-combine inverse metadata.",
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print rank-local progress markers around NVSHMEM/CUDA sync points.",
    )
    return parser.parse_args()


def _debug(args: argparse.Namespace, message: str) -> None:
    if not args.debug:
        return
    rank = dist.get_rank() if dist.is_initialized() else -1
    local_rank = os.environ.get("LOCAL_RANK", "?")
    print(f"[rank={rank} local={local_rank}] {message}", flush=True)


def _init_dist() -> tuple[int, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    return dist.get_rank(), local_rank, dist.get_world_size()


def _make_routes(
    *,
    rows: int,
    top_k: int,
    expert_rows: int,
    rank: int,
    world_size: int,
    pattern: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    row_ids = torch.arange(rows, device=device, dtype=torch.long).view(rows, 1)
    k_ids = torch.arange(top_k, device=device, dtype=torch.long).view(1, top_k)
    route_ids = row_ids * top_k + k_ids
    if pattern == "self":
        src_ranks = torch.full((rows, top_k), rank, device=device, dtype=torch.long)
        src_rows = route_ids
    elif pattern == "ring":
        src_ranks = (rank + k_ids + 1).expand(rows, top_k).remainder(world_size)
        src_rows = route_ids
    elif pattern in {"local_node", "remote_node"}:
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
        if world_size % local_world_size != 0:
            raise ValueError(f"{world_size=} must be divisible by {local_world_size=}")
        num_nodes = world_size // local_world_size
        node_id = rank // local_world_size
        local_rank = rank % local_world_size
        if pattern == "remote_node" and num_nodes < 2:
            raise ValueError("remote_node requires at least two nodes")
        peer_node = node_id if pattern == "local_node" else (node_id + 1) % num_nodes
        peer_base = peer_node * local_world_size
        src_ranks = peer_base + (local_rank + k_ids).expand(rows, top_k).remainder(local_world_size)
        src_rows = route_ids // local_world_size
    else:
        src_ranks = route_ids.remainder(world_size)
        src_rows = route_ids // world_size
    max_row = int(src_rows.max().item()) if src_rows.numel() else -1
    if max_row >= expert_rows:
        raise ValueError(
            f"expert_rows={expert_rows} is too small for {pattern=} with "
            f"rows={rows}, top_k={top_k}, world_size={world_size}; need > {max_row}"
        )
    return src_ranks.contiguous(), src_rows.contiguous()


def _expected(
    src_ranks: torch.Tensor,
    *,
    dim: int,
    probs: torch.Tensor | None,
    mode: str,
) -> torch.Tensor:
    values = (src_ranks.to(dtype=torch.float32) + 1.0)
    if mode == "gather_get":
        return values.reshape(-1, 1).expand(-1, dim)
    if probs is not None:
        values = values * probs
    return values.sum(dim=1, keepdim=True).expand(-1, dim)


def _alloc_symm(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    tensor = olmo_symm_mem.empty(shape, dtype=dtype, device=device, group=group)
    olmo_symm_mem.rendezvous(tensor, group=group)
    return tensor


def _run_once(
    *,
    args: argparse.Namespace,
    expert_out: torch.Tensor,
    src_ranks: torch.Tensor,
    src_rows: torch.Tensor,
    probs: torch.Tensor | None,
    group_name: str,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    device = expert_out.device
    dtype = expert_out.dtype
    rows = src_ranks.shape[0]
    top_k = src_ranks.shape[1]
    dim = expert_out.shape[1]

    if args.mode == "gather_get":
        out_shape = (rows * top_k, dim)
        if args.symmetric_output:
            _debug(args, f"alloc symmetric gather output shape={out_shape}")
            out = _alloc_symm(out_shape, dtype=dtype, device=device, group=group)
        else:
            out = torch.empty(out_shape, device=device, dtype=dtype)
        flat_ranks = src_ranks.reshape(rows * top_k, 1).contiguous()
        flat_rows = src_rows.reshape(rows * top_k, 1).contiguous()
        _debug(args, "before rowwise_gather_get")
        symm_mem_vdev2d_kernels.rowwise_gather_get(
            expert_out,
            out,
            flat_ranks,
            flat_rows,
            group_name,
            nblocks=args.nblocks,
            pre_barrier=True,
            post_barrier=args.post_barrier,
        )
        _debug(args, "after rowwise_gather_get launch")
        return out

    if args.mode in {"combine_get", "combine_get_fused"}:
        out = (
            _alloc_symm((rows, dim), dtype=dtype, device=device, group=group)
            if args.symmetric_output
            else torch.empty((rows, dim), device=device, dtype=dtype)
        )
        gathered_out = None
        if args.symmetric_output:
            gathered_out = _alloc_symm((rows, top_k, dim), dtype=dtype, device=device, group=group)
        fn = (
            symm_mem_vdev2d_kernels.rowwise_combine_get_fused
            if args.mode == "combine_get_fused"
            else symm_mem_vdev2d_kernels.rowwise_combine_get
        )
        _debug(args, f"before {args.mode}")
        fn(
            expert_out,
            out,
            src_ranks,
            src_rows,
            group_name,
            probs=probs,
            nblocks=args.nblocks,
            gathered_out=gathered_out,
            pre_barrier=True,
            post_barrier=args.post_barrier,
        )
        _debug(args, f"after {args.mode} launch")
        return out

    _debug(args, "alloc symmetric PUT-combine scratch/meta")
    gathered = _alloc_symm((rows, top_k, dim), dtype=dtype, device=device, group=group)
    inverse_meta = _alloc_symm(
        (expert_out.shape[0], 2),
        dtype=torch.long,
        device=device,
        group=group,
    )
    inverse_meta.fill_(-1)
    route_experts = torch.zeros_like(src_ranks, dtype=torch.int32)
    route_records, wave_offsets = symm_mem_vdev2d_kernels.rowwise_build_compact_route_records(
        src_ranks,
        src_rows,
        route_experts,
        num_local_experts=1,
        num_waves=1,
        nblocks=args.nblocks,
    )
    symm_mem_vdev2d_kernels.rowwise_inverse_route_meta_put_compact(
        inverse_meta,
        route_records,
        wave_offsets,
        src_rank=dist.get_rank(group),
        group_name=group_name,
        nblocks=args.nblocks,
        pre_barrier=True,
        post_barrier=True,
        scalar_put=args.scalar_meta_put,
    )
    _debug(args, "after inverse route meta PUT")
    row_start = torch.zeros((), device=device, dtype=torch.long)
    num_rows = torch.tensor(expert_out.shape[0], device=device, dtype=torch.long)
    _debug(args, "before rowwise_combine_put")
    symm_mem_vdev2d_kernels.rowwise_combine_put(
        expert_out,
        gathered,
        inverse_meta,
        row_start,
        num_rows,
        group_name,
        nblocks=args.nblocks,
        pre_barrier=True,
        post_barrier=True,
    )
    _debug(args, "after rowwise_combine_put launch")
    out = torch.empty((rows, dim), device=device, dtype=dtype)
    if probs is None:
        symm_mem_vdev2d_kernels.rowwise_reduce_gathered_routes_unweighted(
            gathered,
            out,
            route_ranks=src_ranks,
        )
    else:
        symm_mem_vdev2d_kernels.rowwise_reduce_gathered_routes(
            gathered,
            probs,
            out,
            route_ranks=src_ranks,
        )
    return out


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args = _parse_args()
    rank, local_rank, world_size = _init_dist()
    group = dist.group.WORLD
    group_name = group.group_name
    device = torch.device("cuda", local_rank)

    try:
        _debug(args, "initialized dist")
        _debug(args, f"alloc symmetric expert_out shape=({args.expert_rows}, {args.dim})")
        expert_out = _alloc_symm(
            (args.expert_rows, args.dim),
            dtype=torch.bfloat16,
            device=device,
            group=group,
        )
        _debug(args, "after expert_out rendezvous")
        expert_out.fill_(float(rank + 1))
        _debug(args, "after expert_out fill")
        src_ranks, src_rows = _make_routes(
            rows=args.rows,
            top_k=args.top_k,
            expert_rows=args.expert_rows,
            rank=rank,
            world_size=world_size,
            pattern=args.peer_pattern,
            device=device,
        )
        _debug(args, "after route tensors")
        probs = None
        if args.weighted:
            probs = torch.full(
                (args.rows, args.top_k),
                1.0 / float(args.top_k),
                device=device,
                dtype=torch.float32,
            )
        _debug(args, "after probs")

        for warmup_idx in range(args.warmup):
            _debug(args, f"warmup {warmup_idx} begin")
            _run_once(
                args=args,
                expert_out=expert_out,
                src_ranks=src_ranks,
                src_rows=src_rows,
                probs=probs,
                group_name=group_name,
                group=group,
            )
            _debug(args, f"warmup {warmup_idx} launched")
        _debug(args, "before post-warmup cuda synchronize")
        torch.cuda.synchronize()
        _debug(args, "after post-warmup cuda synchronize")
        _debug(args, "before post-warmup dist barrier")
        dist.barrier()
        _debug(args, "after post-warmup dist barrier")

        times: list[float] = []
        last_out = None
        for iter_idx in range(args.iters):
            _debug(args, f"iter {iter_idx} before pre-sync")
            torch.cuda.synchronize()
            _debug(args, f"iter {iter_idx} after pre-sync before barrier")
            dist.barrier()
            _debug(args, f"iter {iter_idx} after barrier")
            start = time.perf_counter()
            last_out = _run_once(
                args=args,
                expert_out=expert_out,
                src_ranks=src_ranks,
                src_rows=src_rows,
                probs=probs,
                group_name=group_name,
                group=group,
            )
            _debug(args, f"iter {iter_idx} launched before sync")
            torch.cuda.synchronize()
            _debug(args, f"iter {iter_idx} after sync")
            times.append((time.perf_counter() - start) * 1000.0)

        max_err = torch.tensor(0.0, device=device)
        if args.check:
            _debug(args, "before correctness check")
            assert last_out is not None
            expected = _expected(
                src_ranks,
                dim=args.dim,
                probs=probs,
                mode=args.mode,
            ).to(dtype=last_out.dtype)
            max_err = (last_out - expected).abs().max().to(dtype=torch.float32)

        local = torch.tensor(
            [statistics.median(times), float(max_err.item())],
            device=device,
            dtype=torch.float64,
        )
        _debug(args, "before all_gather result")
        gathered = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_gather(gathered, local)
        _debug(args, "after all_gather result")
        if rank == 0:
            max_ms = max(float(item[0].item()) for item in gathered)
            max_error = max(float(item[1].item()) for item in gathered)
            print(
                "NVSHMEM_GET_SMOKE "
                f"mode={args.mode} ranks={world_size} rows={args.rows} "
                f"expert_rows={args.expert_rows} dim={args.dim} top_k={args.top_k} "
                f"nblocks={args.nblocks} peer_pattern={args.peer_pattern} "
                f"weighted={args.weighted} symmetric_output={args.symmetric_output} "
                f"post_barrier={args.post_barrier} "
                f"host_ms/iter(max_rank)={max_ms:.3f} max_err={max_error:.6f}",
                flush=True,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
