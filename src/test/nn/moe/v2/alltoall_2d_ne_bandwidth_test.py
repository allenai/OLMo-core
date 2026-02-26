import argparse
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark NVSHMEM 2D all_to_all bandwidth scaling vs ne."
    )
    parser.add_argument(
        "--variant",
        choices=["plain", "offset"],
        default="offset",
        help="2D kernel variant to benchmark.",
    )
    parser.add_argument(
        "--impl",
        choices=["torch", "custom"],
        default="custom",
        help="Use torch op or local CUDA extension implementation.",
    )
    parser.add_argument(
        "--ne-list",
        type=str,
        default="1,2,4,8",
        help="Comma-separated local expert counts (ne) to benchmark.",
    )
    parser.add_argument(
        "--rows-per-split",
        type=int,
        default=2048,
        help="Rows in each (rank, expert) split.",
    )
    parser.add_argument("--cols", type=int, default=4096, help="Hidden dim D.")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--major-align", type=int, default=1, help="major_align for plain 2D variant.")
    parser.add_argument("--nblocks", type=int, default=256, help="nblocks for custom kernels.")
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    return parser.parse_args()


def _parse_ne_list(raw: str) -> List[int]:
    vals: List[int] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        v = int(p)
        if v <= 0:
            raise ValueError(f"ne must be > 0, got {v}")
        vals.append(v)
    if not vals:
        raise ValueError("ne-list cannot be empty")
    return vals


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _event_timed_ms(fn: Callable[[], None], iters: int) -> List[float]:
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


def _load_custom_vdev2d_callables() -> tuple[Callable[..., None], Callable[..., None]]:
    src_root = Path(__file__).resolve().parents[4]
    src_root_str = str(src_root)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)
    from olmo_core.kernels.symm_mem_vdev2d import (
        all_to_all_vdev_2d_nblocks,
        all_to_all_vdev_2d_offset_nblocks,
    )

    return all_to_all_vdev_2d_nblocks, all_to_all_vdev_2d_offset_nblocks


def _print_header(
    *,
    world_size: int,
    rows_per_split: int,
    cols: int,
    dtype: torch.dtype,
    ne_values: Sequence[int],
    variant: str,
    impl: str,
    nblocks: int,
) -> None:
    print("\n=== NVSHMEM 2D AllToAll ne-Scaling Benchmark ===", flush=True)
    print(
        f"world_size={world_size} rows_per_split={rows_per_split} cols={cols} dtype={dtype} "
        f"variant={variant} impl={impl} ne_list={list(ne_values)} nblocks={nblocks}",
        flush=True,
    )


def main() -> None:
    args = _parse_args()
    ne_values = _parse_ne_list(args.ne_list)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))

    group = dist.group.WORLD
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    device = torch.device("cuda", local_rank)
    dtype = _dtype_from_name(args.dtype)

    custom_vdev2d: Optional[Callable[..., None]] = None
    custom_vdev2d_offset: Optional[Callable[..., None]] = None
    if args.impl == "custom":
        custom_vdev2d, custom_vdev2d_offset = _load_custom_vdev2d_callables()

    group_name = _setup_nvshmem_backend(group, device)

    if rank == 0:
        _print_header(
            world_size=world_size,
            rows_per_split=args.rows_per_split,
            cols=args.cols,
            dtype=dtype,
            ne_values=ne_values,
            variant=args.variant,
            impl=args.impl,
            nblocks=args.nblocks,
        )

    results: List[dict] = []
    for ne in ne_values:
        nsplits = world_size * ne
        rows = args.rows_per_split * nsplits
        out_cap = rows + nsplits * max(args.major_align - 1, 0)

        x = torch.randn((rows, args.cols), device=device, dtype=dtype)
        x.add_(rank)

        symm_in = _alloc_rendezvous_symm_tensor(x.shape, x.dtype, device, group)
        symm_out = _alloc_rendezvous_symm_tensor((out_cap, args.cols), x.dtype, device, group)
        symm_out_splits_offsets = _alloc_rendezvous_symm_tensor((2, nsplits), torch.int64, device, group)
        symm_in.copy_(x)

        if args.variant == "plain":
            symm_in_splits = _alloc_rendezvous_symm_tensor((nsplits,), torch.int64, device, group)
            symm_in_splits.fill_(args.rows_per_split)

            def run_a2a2d() -> None:
                if custom_vdev2d is None:
                    torch.ops.symm_mem.all_to_all_vdev_2d(
                        symm_in,
                        symm_out,
                        symm_in_splits,
                        symm_out_splits_offsets,
                        group_name,
                        major_align=args.major_align,
                    )
                else:
                    custom_vdev2d(
                        symm_in,
                        symm_out,
                        symm_in_splits,
                        symm_out_splits_offsets,
                        group_name,
                        major_align=args.major_align,
                        nblocks=args.nblocks,
                    )
        else:
            symm_in_splits_offsets = _alloc_rendezvous_symm_tensor((2, nsplits), torch.int64, device, group)
            symm_in_splits_offsets[0].fill_(args.rows_per_split)
            symm_in_splits_offsets[1].copy_(
                torch.arange(nsplits, device=device, dtype=torch.int64) * args.rows_per_split
            )

            def run_a2a2d() -> None:
                if custom_vdev2d_offset is None:
                    torch.ops.symm_mem.all_to_all_vdev_2d_offset(
                        symm_in,
                        symm_out,
                        symm_in_splits_offsets,
                        symm_out_splits_offsets,
                        group_name,
                    )
                else:
                    custom_vdev2d_offset(
                        symm_in,
                        symm_out,
                        symm_in_splits_offsets,
                        symm_out_splits_offsets,
                        group_name,
                        nblocks=args.nblocks,
                    )

        dist.barrier(group=group)
        for _ in range(args.warmup_iters):
            run_a2a2d()
        torch.cuda.synchronize(device)
        dist.barrier(group=group)
        times_ms = _event_timed_ms(run_a2a2d, args.iters)

        avg_ms = mean(times_ms)
        payload_gb = (rows * args.cols * torch.tensor([], dtype=dtype).element_size()) / 1e9
        bw_gbs = payload_gb / (avg_ms / 1e3)
        est_blocks_per_split = max(args.nblocks // nsplits, 1) if args.nblocks > 0 else -1

        results.append(
            {
                "ne": ne,
                "rows": rows,
                "payload_gb": payload_gb,
                "avg_ms": avg_ms,
                "bw_gbs": bw_gbs,
                "nsplits": nsplits,
                "est_blocks_per_split": est_blocks_per_split,
            }
        )

    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, results, group=group)
    if rank == 0:
        per_rank = [r for r in gathered if r is not None]
        merged = []
        for idx in range(len(results)):
            avg_ms = mean([float(rr[idx]["avg_ms"]) for rr in per_rank])
            bw = mean([float(rr[idx]["bw_gbs"]) for rr in per_rank])
            merged.append(
                {
                    "ne": int(results[idx]["ne"]),
                    "nsplits": int(results[idx]["nsplits"]),
                    "rows": int(results[idx]["rows"]),
                    "payload_gb": float(results[idx]["payload_gb"]),
                    "avg_ms": avg_ms,
                    "bw_gbs": bw,
                    "est_blocks_per_split": int(results[idx]["est_blocks_per_split"]),
                }
            )

        print("[RESULTS]", flush=True)
        for item in merged:
            print(
                f"ne={item['ne']} nsplits={item['nsplits']} rows={item['rows']} "
                f"payload={item['payload_gb']:.3f} GB avg={item['avg_ms']:.3f} ms "
                f"bw={item['bw_gbs']:.2f} GB/s est_blocks_per_split={item['est_blocks_per_split']}",
                flush=True,
            )

    dist.barrier(group=group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
