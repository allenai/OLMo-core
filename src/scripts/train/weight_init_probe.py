"""Minimal PyTorch grouped_mm weight-initialization probe.

This intentionally avoids OLMo, DeepEP, NVSHMEM, rowwise EP, RoutedExperts,
SwiGLU, and the second expert GEMM. It measures one grouped GEMM and, by
default, one regular dense GEMM baseline:

    torch.nn.functional.grouped_mm(x, w, offs=offs)
    torch.mm(x, w)

The default shape matches the former routed-expert Linear1 GEMM:

    x:    [local_experts * rows_per_expert, d_model]
    w:    [local_experts, d_model, 2 * hidden_size]
    offs: cumsum([rows_per_expert] * local_experts)

Example:

    torchrun --standalone --nproc-per-node=8 \
      src/scripts/train/routed_experts_weight_init_probe.py \
      --d-model 8192 --hidden-size 8192 \
      --local-experts 8 --rows-per-expert 16384 \
      --init-modes empty normal uniform rand_sign zero fill \
      --ops grouped_mm mm
"""

from __future__ import annotations

import argparse
import os
import statistics

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=8192)
    parser.add_argument("--hidden-size", type=int, default=8192)
    parser.add_argument(
        "--out-features",
        type=int,
        default=0,
        help="Grouped GEMM N dimension. 0 means 2 * --hidden-size.",
    )
    parser.add_argument("--local-experts", type=int, default=8)
    parser.add_argument("--rows-per-expert", type=int, default=16384)
    parser.add_argument(
        "--init-modes",
        nargs="+",
        choices=("empty", "normal", "normal1", "uniform", "rand_sign", "zero", "fill"),
        default=["empty", "normal", "uniform", "rand_sign", "zero", "fill"],
    )
    parser.add_argument(
        "--ops",
        nargs="+",
        choices=("grouped_mm", "mm"),
        default=["grouped_mm", "mm"],
        help="Operations to benchmark. 'mm' is a regular dense GEMM baseline.",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--scale", type=float, default=0.02)
    parser.add_argument("--input-scale", type=float, default=0.2)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(name)


def _init_dist() -> tuple[int, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if "RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    return rank, local_rank, world_size


def _barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def _all_gather_max(values: list[float], *, world_size: int) -> list[float]:
    local = torch.tensor(values, device="cuda", dtype=torch.float32)
    if not dist.is_initialized():
        return [float(v) for v in local.detach().cpu().tolist()]
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    return torch.stack(gathered, dim=0).amax(dim=0).detach().cpu().tolist()


def _init_weight(weight: torch.Tensor, *, init_mode: str, scale: float) -> None:
    with torch.no_grad():
        if init_mode == "empty":
            return
        if init_mode == "normal":
            weight.normal_(mean=0.0, std=scale)
        elif init_mode == "normal1":
            weight.normal_(mean=0.0, std=1.0)
        elif init_mode == "uniform":
            weight.uniform_(-scale, scale)
        elif init_mode == "rand_sign":
            weight.bernoulli_(0.5).mul_(2.0).sub_(1.0).mul_(scale)
        elif init_mode == "zero":
            weight.zero_()
        elif init_mode == "fill":
            weight.fill_(scale)
        else:
            raise ValueError(init_mode)


def _run_mode(
    args: argparse.Namespace,
    *,
    rank: int,
    world_size: int,
    init_mode: str,
    op: str,
) -> None:
    dtype = _dtype(args.dtype)
    out_features = args.out_features if args.out_features > 0 else 2 * args.hidden_size
    rows = args.local_experts * args.rows_per_expert

    torch.manual_seed(args.seed + rank)
    x = (args.input_scale * torch.randn(rows, args.d_model, device="cuda")).to(dtype)
    if op == "grouped_mm":
        weight_shape = (args.local_experts, args.d_model, out_features)
    elif op == "mm":
        weight_shape = (args.d_model, out_features)
    else:
        raise ValueError(op)
    weight = torch.empty(*weight_shape, device="cuda", dtype=dtype)
    _init_weight(weight, init_mode=init_mode, scale=float(args.scale))
    offs = torch.arange(
        1,
        args.local_experts + 1,
        device="cuda",
        dtype=torch.int32,
    ) * int(args.rows_per_expert)

    _barrier()
    times: list[float] = []
    total = args.warmup + args.iters
    for idx in range(total):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if op == "grouped_mm":
            out = F.grouped_mm(x, weight, offs=offs)
        else:
            out = torch.mm(x, weight)
        end.record()
        end.synchronize()
        if idx >= args.warmup:
            times.append(start.elapsed_time(end))
        del out

    max_by_iter = _all_gather_max(times, world_size=world_size)
    if rank == 0:
        median_ms = statistics.median(max_by_iter) if max_by_iter else float("nan")
        print(
            "GEMM_PROBE "
            f"op={op} init={init_mode} ranks={world_size} dtype={args.dtype} "
            f"groups={args.local_experts} m_per_group={args.rows_per_expert} "
            f"k={args.d_model} n={out_features} "
            f"scale={args.scale} input_scale={args.input_scale} "
            f"max_rank_ms={[round(float(v), 3) for v in max_by_iter]} "
            f"median_max_rank_ms={median_ms:.3f}",
            flush=True,
        )

    del x, weight, offs
    torch.cuda.synchronize()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not hasattr(F, "grouped_mm"):
        raise RuntimeError("torch.nn.functional.grouped_mm is required")
    args = _parse_args()
    rank, _local_rank, world_size = _init_dist()
    try:
        for op in args.ops:
            for init_mode in args.init_modes:
                _run_mode(
                    args,
                    rank=rank,
                    world_size=world_size,
                    init_mode=init_mode,
                    op=op,
                )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
