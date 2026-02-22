import argparse
import os
from statistics import mean
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.distributed.distributed_c10d as c10d
from torch._C._distributed_c10d import _SymmetricMemory

# 16-rank real case (2 nodes x 8 ranks):
# TORCH_SYMMMEM_NBLOCKS=256 torchrun --nnodes=2 --nproc-per-node=8 \
#   /workspace/OLMo-core/src/test/nn/moe/v2/alltoall_team_test.py --num-teams=2 --team-size=8
#
# 8-GPU single-node simulation (2 teams x 4 ranks):
# TORCH_SYMMMEM_NBLOCKS=256 torchrun --nproc-per-node=8 \
#   /workspace/OLMo-core/src/test/nn/moe/v2/alltoall_team_test.py --num-teams=2
#
# Team-bootstrap experiment (uses private APIs):
# TORCH_SYMMMEM_NBLOCKS=128 torchrun --nnodes=2 --nproc-per-node=8 \
#   /workspace/OLMo-core/src/test/nn/moe/v2/alltoall_team_test.py \
#   --num-teams=2 --team-size=8 --bootstrap-group-as-world


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark NCCL all_to_all_single vs NVSHMEM symmetric-memory "
            "all_to_all_vdev and all_to_all_vdev_2d inside subgroup teams."
        )
    )
    parser.add_argument("--rows", type=int, default=2 * 8192 * 4, help="Rows per rank.")
    parser.add_argument("--cols", type=int, default=4096, help="Columns per row.")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--major-align", type=int, default=1)
    parser.add_argument("--num-teams", type=int, default=2, help="Number of disjoint all-to-all teams.")
    parser.add_argument(
        "--team-size",
        type=int,
        default=None,
        help=(
            "Ranks per team. If unset, use world_size // num_teams "
            "(lets 8 GPUs simulate 2 teams as 4+4)."
        ),
    )
    parser.add_argument(
        "--bootstrap-group-as-world",
        action="store_true",
        help=(
            "Experimental: bootstrap NVSHMEM allocator on the team group by "
            "aliasing team metadata as group '0' (world). This may help "
            "recover intra-node P2P behavior in multi-node jobs."
        ),
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


def _register_group0_alias_for_team(group: dist.ProcessGroup) -> None:
    if group.group_name == dist.group.WORLD.group_name:
        return

    # Keep symmetric-memory Python bookkeeping consistent with this manual
    # registration to avoid duplicate set_group_info() later.
    if symm_mem.is_symm_mem_enabled_for_group("0"):
        raise RuntimeError(
            "group '0' is already registered for symmetric memory. "
            "Cannot alias it to the team group. Restart the process and "
            "avoid enabling WORLD symmetric memory before this flag."
        )

    global_ranks = sorted(c10d._world.pg_group_ranks[group].keys())
    global_ranks_str = "_".join(map(str, global_ranks))
    store = c10d.PrefixStore(
        f"symmetric_memory-{global_ranks_str}",
        c10d._get_process_group_store(group),
    )
    _SymmetricMemory.set_group_info(
        "0",
        dist.get_rank(group),
        dist.get_world_size(group),
        store,
    )
    # pyright: ignore[reportPrivateUsage]
    symm_mem._group_name_to_store["0"] = store


def _setup_nvshmem_backend(
    group: dist.ProcessGroup, device: torch.device, bootstrap_group_as_world: bool
) -> str:
    if not symm_mem.is_nvshmem_available():
        raise RuntimeError(
            "NVSHMEM is not available in this environment. "
            "Cannot run symmetric-memory all_to_all_vdev benchmarks."
        )

    backend = symm_mem.get_backend(device)
    if backend is None or backend.upper() != "NVSHMEM":
        symm_mem.set_backend("NVSHMEM")

    group_name = group.group_name
    if bootstrap_group_as_world:
        # Register team group for rendezvous and alias it as group '0' for
        # NVSHMEM allocator bootstrap.
        symm_mem.enable_symm_mem_for_group(group_name)
        _register_group0_alias_for_team(group)
    else:
        world_group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(world_group_name)
        symm_mem.enable_symm_mem_for_group(group_name)
    return group_name


def _alloc_rendezvous_symm_tensor(
    shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device, group: dist.ProcessGroup
) -> torch.Tensor:
    t = symm_mem.empty(shape, dtype=dtype, device=device)
    symm_mem.rendezvous(t, group=group)
    return t


def _resolve_team_layout(
    *, rank: int, world_size: int, num_teams: int, requested_team_size: Optional[int]
) -> Tuple[int, int, List[List[int]]]:
    if num_teams <= 0:
        raise RuntimeError(f"num_teams must be > 0, got {num_teams}")

    if requested_team_size is None:
        if world_size % num_teams != 0:
            raise RuntimeError(
                f"world_size ({world_size}) must be divisible by num_teams ({num_teams}) when --team-size is unset"
            )
        team_size = world_size // num_teams
    else:
        if requested_team_size <= 0:
            raise RuntimeError(f"team_size must be > 0, got {requested_team_size}")
        team_size = requested_team_size
        expected_world = num_teams * team_size
        if world_size != expected_world:
            raise RuntimeError(
                f"world_size ({world_size}) does not match num_teams * team_size ({expected_world}). "
                "Set --team-size to match world size, or unset it to auto-split for simulation."
            )

    teams: List[List[int]] = []
    for team_id in range(num_teams):
        start = team_id * team_size
        teams.append(list(range(start, start + team_size)))

    team_id = rank // team_size
    if team_id >= num_teams:
        raise RuntimeError(
            f"rank ({rank}) is outside the computed team range: num_teams={num_teams}, team_size={team_size}"
        )
    return team_id, team_size, teams


def main() -> None:
    args = _parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))

    world = dist.group.WORLD
    rank = dist.get_rank(world)
    world_size = dist.get_world_size(world)
    device = torch.device("cuda", local_rank)
    dtype = _dtype_from_name(args.dtype)

    team_id, team_size, all_teams = _resolve_team_layout(
        rank=rank,
        world_size=world_size,
        num_teams=args.num_teams,
        requested_team_size=args.team_size,
    )

    team_group: Optional[dist.ProcessGroup] = None
    for ranks in all_teams:
        pg = dist.new_group(ranks=ranks, backend="nccl")
        if rank in ranks:
            team_group = pg

    if team_group is None:
        raise RuntimeError(f"Rank {rank} does not belong to any team")

    team_rank = dist.get_rank(team_group)
    team_world_size = dist.get_world_size(team_group)

    if args.rows % team_world_size != 0:
        raise RuntimeError(
            f"rows ({args.rows}) must be divisible by team_world_size ({team_world_size})"
        )

    split_rows = args.rows // team_world_size
    splits = [split_rows] * team_world_size

    # Deterministic payload with rank-dependent offset.
    torch.manual_seed(1337 + rank)
    x = torch.randn((args.rows, args.cols), device=device, dtype=dtype)
    x.add_(rank)

    # NCCL all_to_all_single buffers in team group.
    nccl_out = torch.empty_like(x)

    def run_nccl() -> None:
        dist.all_to_all_single(
            nccl_out,
            x,
            output_split_sizes=splits,
            input_split_sizes=splits,
            group=team_group,
        )

    # Symmetric-memory (NVSHMEM-backed) buffers in team group.
    group_name = _setup_nvshmem_backend(
        team_group, device, bootstrap_group_as_world=args.bootstrap_group_as_world
    )
    out_cap = args.rows + (team_world_size - 1) * max(args.major_align - 1, 0)
    symm_in = _alloc_rendezvous_symm_tensor(x.shape, x.dtype, device, team_group)
    symm_in_splits = _alloc_rendezvous_symm_tensor(
        (team_world_size,), torch.int64, device, team_group
    )
    symm_out_vdev = _alloc_rendezvous_symm_tensor((out_cap, args.cols), x.dtype, device, team_group)
    symm_out_vdev_splits_offsets = _alloc_rendezvous_symm_tensor(
        (2, team_world_size), torch.int64, device, team_group
    )
    symm_out_vdev2d = _alloc_rendezvous_symm_tensor((out_cap, args.cols), x.dtype, device, team_group)
    symm_out_vdev2d_splits_offsets = _alloc_rendezvous_symm_tensor(
        (2, team_world_size), torch.int64, device, team_group
    )
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

    dist.barrier(group=team_group)
    for _ in range(args.warmup_iters):
        run_nccl()
    torch.cuda.synchronize(device)
    dist.barrier(group=team_group)
    nccl_times_ms = _event_timed_ms(run_nccl, args.iters)

    dist.barrier(group=team_group)
    for _ in range(args.warmup_iters):
        run_symm_vdev()
    torch.cuda.synchronize(device)
    dist.barrier(group=team_group)
    symm_vdev_times_ms = _event_timed_ms(run_symm_vdev, args.iters)

    dist.barrier(group=team_group)
    for _ in range(args.warmup_iters):
        run_symm_vdev2d()
    torch.cuda.synchronize(device)
    dist.barrier(group=team_group)
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

    dist.barrier(group=world)
    gathered: List[Optional[Dict[str, object]]] = [None] * world_size
    result = {
        "rank": rank,
        "team_id": team_id,
        "team_rank": team_rank,
        "team_world_size": team_world_size,
        "nccl_avg_ms": mean(nccl_times_ms),
        "symm_vdev_avg_ms": mean(symm_vdev_times_ms),
        "symm_vdev2d_avg_ms": mean(symm_vdev2d_times_ms),
        "equal_vdev": bool(equal_vdev),
        "equal_vdev2d": bool(equal_vdev2d),
        "max_abs_diff_vdev": max_abs_diff_vdev,
        "max_abs_diff_vdev2d": max_abs_diff_vdev2d,
    }
    dist.all_gather_object(gathered, result, group=world)

    if rank == 0:
        payload_gb = (args.rows * args.cols * torch.tensor([], dtype=dtype).element_size()) / 1e9
        print("=== Team-Scoped AllToAll: NCCL vs NVSHMEM Symmetric Memory (vdev + vdev_2d) ===", flush=True)
        print(
            f"world_size={world_size} num_teams={args.num_teams} team_size={team_size} "
            f"rows={args.rows} cols={args.cols} dtype={dtype} "
            f"warmup={args.warmup_iters} iters={args.iters} major_align={args.major_align} "
            f"bootstrap_group_as_world={args.bootstrap_group_as_world}",
            flush=True,
        )
        if team_size != 8:
            print(
                f"NOTE: running in simulation mode for 2x8 by using team_size={team_size}. "
                "Use world_size=16 (or --team-size=8) for exact 2x8.",
                flush=True,
            )
        print(f"payload_per_rank={payload_gb:.3f} GB", flush=True)

        typed_gathered = sorted((item for item in gathered if item is not None), key=lambda z: z["rank"])  # type: ignore[index]

        for item in typed_gathered:
            nccl_bw = payload_gb / (item["nccl_avg_ms"] / 1e3)  # type: ignore[index]
            symm_vdev_bw = payload_gb / (item["symm_vdev_avg_ms"] / 1e3)  # type: ignore[index]
            symm_vdev2d_bw = payload_gb / (item["symm_vdev2d_avg_ms"] / 1e3)  # type: ignore[index]
            speedup_vdev = item["nccl_avg_ms"] / item["symm_vdev_avg_ms"]  # type: ignore[index]
            speedup_vdev2d = item["nccl_avg_ms"] / item["symm_vdev2d_avg_ms"]  # type: ignore[index]
            pass_or_fail = "PASS" if item["equal_vdev"] and item["equal_vdev2d"] else "FAIL"  # type: ignore[index]
            print(
                f"[{pass_or_fail}] "
                f"rank={item['rank']} team={item['team_id']} team_rank={item['team_rank']} "
                f"nccl={item['nccl_avg_ms']:.3f} ms ({nccl_bw:.2f} GB/s) "
                f"symm_vdev={item['symm_vdev_avg_ms']:.3f} ms ({symm_vdev_bw:.2f} GB/s) "
                f"speedup_vdev={speedup_vdev:.3f}x "
                f"equal_vdev={item['equal_vdev']} max_abs_diff_vdev={item['max_abs_diff_vdev']:.6f} "
                f"symm_vdev2d={item['symm_vdev2d_avg_ms']:.3f} ms ({symm_vdev2d_bw:.2f} GB/s) "
                f"speedup_vdev2d={speedup_vdev2d:.3f}x "
                f"equal_vdev2d={item['equal_vdev2d']} max_abs_diff_vdev2d={item['max_abs_diff_vdev2d']:.6f}",
                flush=True,
            )

        for t in range(args.num_teams):
            team_items = [item for item in typed_gathered if item["team_id"] == t]
            if not team_items:
                continue
            nccl_team = mean(float(item["nccl_avg_ms"]) for item in team_items)
            symm_team = mean(float(item["symm_vdev_avg_ms"]) for item in team_items)
            symm2d_team = mean(float(item["symm_vdev2d_avg_ms"]) for item in team_items)
            print(
                f"[TEAM {t}] avg nccl={nccl_team:.3f} ms symm_vdev={symm_team:.3f} ms "
                f"symm_vdev2d={symm2d_team:.3f} ms",
                flush=True,
            )

    dist.barrier(group=world)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
