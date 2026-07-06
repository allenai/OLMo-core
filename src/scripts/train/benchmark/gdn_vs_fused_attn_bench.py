"""
Benchmark a standalone GatedDeltaNet mixer against standalone FusedAttentionV2.

Defaults match the current s004 mixer shape:
    d_model=1536, n_heads=16, n_kv_heads=8, head_dim=128, seq_len=8192.

Example, single GPU:
    PYTHONPATH=src python -m scripts.train.benchmark.gdn_vs_fused_attn_bench

Example, all visible GPUs:
    PYTHONPATH=src torchrun --standalone --nproc-per-node=8 \
        -m scripts.train.benchmark.gdn_vs_fused_attn_bench

Small smoke test:
    PYTHONPATH=src python -m scripts.train.benchmark.gdn_vs_fused_attn_bench \
        --batch-size 1 --seq-len 128 --warmup-iters 1 --bench-iters 1 --mode forward
"""

from __future__ import annotations

import argparse
import os
import statistics
import warnings
from contextlib import nullcontext
from typing import Any, Optional

warnings.filterwarnings(
    "ignore",
    message=r"Use explicit .*scalar\.ptr.*",
    category=Warning,
)
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"nvidia_cutlass_dsl.*")

import torch
import torch.distributed as dist
from torch import nn
from torch.profiler import ProfilerActivity

from olmo_core.config import DType
from olmo_core.nn.attention import (
    AttentionBackendName,
    AttentionConfig,
    AttentionType,
    GateConfig,
    GateGranularity,
)
from olmo_core.nn.attention.base import SequenceMixer
from olmo_core.nn.attention.recurrent import GatedDeltaNetConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modules", choices=("both", "gdn", "attn"), default="both")
    parser.add_argument("--mode", choices=("forward", "fwd-bwd"), default="fwd-bwd")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--bench-iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)

    parser.add_argument("--d-model", type=int, default=1536)
    parser.add_argument("--d-attn", type=int, default=2048)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--n-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--gdn-n-v-heads", type=int, default=None)
    parser.add_argument("--expand-v", type=float, default=2.0)
    parser.add_argument("--allow-neg-eigval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--conv-size", type=int, default=4)

    parser.add_argument("--param-dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--compute-dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--autocast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--attn-backend", choices=("flash_2", "flash_3", "flash_4", "torch"), default="flash_4")
    parser.add_argument("--attn-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--profile-dir", type=str, default=None)
    parser.add_argument("--profile-iters", type=int, default=1)
    parser.add_argument("--profile-record-shapes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profile-with-stack", action="store_true")
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help="Run --profile-iters extra iterations inside cudaProfilerStart/Stop for nsys capture.",
    )

    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.seq_len <= 0:
        parser.error("--seq-len must be positive")
    if args.warmup_iters < 0:
        parser.error("--warmup-iters must be non-negative")
    if args.bench_iters <= 0:
        parser.error("--bench-iters must be positive")
    if args.d_attn != args.n_heads * args.head_dim:
        parser.error("--d-attn must equal --n-heads * --head-dim")
    if args.n_heads % args.n_kv_heads != 0:
        parser.error("--n-heads must be divisible by --n-kv-heads")
    if args.gdn_n_v_heads is not None and args.gdn_n_v_heads < args.n_heads:
        parser.error("--gdn-n-v-heads must be >= --n-heads")
    if args.gdn_n_v_heads is not None and args.gdn_n_v_heads % args.n_heads != 0:
        parser.error("--gdn-n-v-heads must be divisible by --n-heads")
    if args.profile_iters <= 0:
        parser.error("--profile-iters must be positive")
    return args


def init_distributed() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if world_size > 1 and not dist.is_initialized():
        try:
            dist.init_process_group(backend="nccl", device_id=device)
        except TypeError:
            dist.init_process_group(backend="nccl")

    return rank, local_rank, world_size, device


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def dtype_from_name(name: str) -> torch.dtype:
    return DType(name).as_pt()


def autocast_context(args: argparse.Namespace):
    if not args.autocast or args.compute_dtype == "float32":
        return nullcontext()
    return torch.autocast("cuda", dtype=dtype_from_name(args.compute_dtype))


def build_attn(args: argparse.Namespace, device: torch.device) -> SequenceMixer:
    layer_norm: Optional[LayerNormConfig] = None
    if args.qk_norm:
        layer_norm = LayerNormConfig(
            name=LayerNormType.rms,
            eps=1e-6,
            bias=False,
            dtype=DType.float32,
        )

    gate: Optional[GateConfig] = None
    if args.attn_gate:
        gate = GateConfig(granularity=GateGranularity.elementwise, full_precision=True)

    cfg = AttentionConfig(
        name=AttentionType.fused_v2,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        head_dim=args.head_dim,
        bias=False,
        rope=None,
        gate=gate,
        qk_norm=layer_norm,
        backend=AttentionBackendName(args.attn_backend),
        use_head_qk_norm=bool(layer_norm),
        dtype=DType(args.param_dtype),
        d_attn=args.d_attn,
    )
    module = cfg.build(args.d_model, layer_idx=0, n_layers=1, init_device=str(device))
    module.train()
    return module


def build_gdn(args: argparse.Namespace, device: torch.device) -> SequenceMixer:
    cfg = GatedDeltaNetConfig(
        n_heads=args.n_heads,
        n_v_heads=args.gdn_n_v_heads or args.n_heads,
        head_dim=args.head_dim,
        expand_v=args.expand_v,
        allow_neg_eigval=args.allow_neg_eigval,
        conv_size=args.conv_size,
        dtype=DType(args.param_dtype),
    )
    module = cfg.build(args.d_model, layer_idx=0, n_layers=1, init_device=str(device))
    module.train()
    return module


@torch.no_grad()
def make_input(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    dtype = dtype_from_name(args.compute_dtype)
    return torch.randn(args.batch_size, args.seq_len, args.d_model, dtype=dtype, device=device)


def run_one_iter(
    module: nn.Module,
    x_base: torch.Tensor,
    *,
    args: argparse.Namespace,
    label: Optional[str] = None,
) -> None:
    if args.mode == "forward":
        if label is not None:
            torch.cuda.nvtx.range_push(f"{label}.forward")
        try:
            with torch.no_grad(), autocast_context(args):
                module(x_base)
        finally:
            if label is not None:
                torch.cuda.nvtx.range_pop()
        return

    module.zero_grad(set_to_none=True)
    x = x_base.detach().requires_grad_(True)
    if label is not None:
        torch.cuda.nvtx.range_push(f"{label}.forward")
    try:
        with autocast_context(args):
            out = module(x)
            loss = out.float().sum() * (1.0 / out.numel())
    finally:
        if label is not None:
            torch.cuda.nvtx.range_pop()

    if label is not None:
        torch.cuda.nvtx.range_push(f"{label}.backward")
    try:
        loss.backward()
    finally:
        if label is not None:
            torch.cuda.nvtx.range_pop()


def event_time_ms(fn, *, warmup_iters: int, bench_iters: int) -> list[float]:
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(bench_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(float(start.elapsed_time(end)))
    return times_ms


def run_one_iter_split_timed(
    module: nn.Module,
    x_base: torch.Tensor,
    *,
    args: argparse.Namespace,
    label: str,
) -> dict[str, float]:
    total_start = torch.cuda.Event(enable_timing=True)
    fwd_start = torch.cuda.Event(enable_timing=True)
    fwd_end = torch.cuda.Event(enable_timing=True)
    bwd_start = torch.cuda.Event(enable_timing=True)
    bwd_end = torch.cuda.Event(enable_timing=True)
    total_end = torch.cuda.Event(enable_timing=True)

    total_start.record()
    module.zero_grad(set_to_none=True)
    x = x_base.detach().requires_grad_(True)

    torch.cuda.nvtx.range_push(f"{label}.forward")
    fwd_start.record()
    with autocast_context(args):
        out = module(x)
        loss = out.float().sum() * (1.0 / out.numel())
    fwd_end.record()
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push(f"{label}.backward")
    bwd_start.record()
    loss.backward()
    bwd_end.record()
    torch.cuda.nvtx.range_pop()

    total_end.record()
    torch.cuda.synchronize()

    fwd_ms = float(fwd_start.elapsed_time(fwd_end))
    bwd_ms = float(bwd_start.elapsed_time(bwd_end))
    total_ms = float(total_start.elapsed_time(total_end))
    return {
        "forward_ms": fwd_ms,
        "backward_ms": bwd_ms,
        "setup_ms": max(0.0, total_ms - fwd_ms - bwd_ms),
        "total_ms": total_ms,
    }


def event_time_ms_split(
    module: nn.Module,
    x_base: torch.Tensor,
    *,
    args: argparse.Namespace,
    label: str,
) -> dict[str, list[float]]:
    for _ in range(args.warmup_iters):
        run_one_iter(module, x_base, args=args, label=label)
    torch.cuda.synchronize()

    times = {
        "forward_ms": [],
        "backward_ms": [],
        "setup_ms": [],
        "total_ms": [],
    }
    for _ in range(args.bench_iters):
        result = run_one_iter_split_timed(module, x_base, args=args, label=label)
        for key, value in result.items():
            times[key].append(value)
    return times


def profile_module(
    name: str,
    module: nn.Module,
    x_base: torch.Tensor,
    *,
    args: argparse.Namespace,
    rank: int,
) -> Optional[str]:
    if args.profile_dir is None:
        return None

    profile_dir = os.path.abspath(args.profile_dir)
    os.makedirs(profile_dir, exist_ok=True)
    trace_path = os.path.join(
        profile_dir,
        (
            f"{name}_rank{rank:02d}_{args.mode}_"
            f"b{args.batch_size}_s{args.seq_len}_d{args.d_model}_"
            f"h{args.n_heads}_hd{args.head_dim}_ev{args.expand_v:g}_"
            f"compile{int(args.compile)}.json"
        ),
    )

    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=args.profile_record_shapes,
        with_stack=args.profile_with_stack,
    ) as prof:
        for _ in range(args.profile_iters):
            run_one_iter(module, x_base, args=args, label=name)
            prof.step()
    torch.cuda.synchronize()
    prof.export_chrome_trace(trace_path)
    return trace_path


def run_cuda_profiler_range(
    name: str,
    module: nn.Module,
    x_base: torch.Tensor,
    *,
    args: argparse.Namespace,
) -> None:
    if not args.cuda_profiler_range:
        return

    torch.cuda.synchronize()
    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()
    for _ in range(args.profile_iters):
        run_one_iter(module, x_base, args=args, label=name)
    torch.cuda.synchronize()
    cudart.cudaProfilerStop()
    torch.cuda.synchronize()


def describe_module(name: str, module: nn.Module, args: argparse.Namespace) -> dict[str, Any]:
    param_count = sum(p.numel() for p in module.parameters())
    result: dict[str, Any] = {
        "module": name,
        "params_m": param_count / 1_000_000,
    }
    if name == "gdn":
        n_v_heads = args.gdn_n_v_heads or args.n_heads
        head_v_dim = int(args.head_dim * args.expand_v)
        result.update(
            {
                "n_v_heads": n_v_heads,
                "head_v_dim": head_v_dim,
                "key_dim": args.n_heads * args.head_dim,
                "value_dim": n_v_heads * head_v_dim,
            }
        )
    return result


def benchmark_module(
    name: str,
    module: nn.Module,
    x_base: torch.Tensor,
    *,
    args: argparse.Namespace,
    device: torch.device,
    rank: int,
) -> dict[str, Any]:
    if args.compile:
        module = torch.compile(module)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    barrier()
    split_times: Optional[dict[str, list[float]]] = None
    if args.mode == "fwd-bwd":
        split_times = event_time_ms_split(module, x_base, args=args, label=name)
        times = split_times["total_ms"]
    else:
        times = event_time_ms(
            lambda: run_one_iter(module, x_base, args=args, label=name),
            warmup_iters=args.warmup_iters,
            bench_iters=args.bench_iters,
        )
    barrier()

    result = describe_module(name, module, args)
    result.update(
        {
            "rank": rank,
            "mean_ms": statistics.fmean(times),
            "median_ms": statistics.median(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "std_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
            "peak_mem_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
            "times_ms": times,
        }
    )
    if split_times is not None:
        result.update(
            {
                "forward_mean_ms": statistics.fmean(split_times["forward_ms"]),
                "backward_mean_ms": statistics.fmean(split_times["backward_ms"]),
                "setup_mean_ms": statistics.fmean(split_times["setup_ms"]),
                "forward_times_ms": split_times["forward_ms"],
                "backward_times_ms": split_times["backward_ms"],
                "setup_times_ms": split_times["setup_ms"],
            }
        )
    run_cuda_profiler_range(name, module, x_base, args=args)
    if args.cuda_profiler_range:
        result["cuda_profiler_range_iters"] = args.profile_iters
    trace_path = profile_module(name, module, x_base, args=args, rank=rank)
    if trace_path is not None:
        result["trace_path"] = trace_path
    return result


def gather_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    if not is_distributed():
        return [result]
    gathered: list[Optional[dict[str, Any]]] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, result)
    return [item for item in gathered if item is not None]


def print_results(results: list[dict[str, Any]], *, world_size: int) -> None:
    results = sorted(results, key=lambda item: int(item["rank"]))
    for result in results:
        extra = ""
        if result["module"] == "gdn":
            extra = (
                f" n_v_heads={result['n_v_heads']} head_v_dim={result['head_v_dim']} "
                f"key_dim={result['key_dim']} value_dim={result['value_dim']}"
            )
        print(
            "rank={rank:02d} module={module} mean_ms={mean_ms:.3f} median_ms={median_ms:.3f} "
            "min_ms={min_ms:.3f} max_ms={max_ms:.3f} std_ms={std_ms:.3f} "
            "params_m={params_m:.2f} peak_mem_gib={peak_mem_gib:.2f}{extra}".format(
                **result,
                extra=extra,
            ),
            flush=True,
        )
        if "forward_mean_ms" in result:
            print(
                "rank={rank:02d} module={module} split forward_ms={forward_mean_ms:.3f} "
                "backward_ms={backward_mean_ms:.3f} setup_ms={setup_mean_ms:.3f} "
                "combined_ms={mean_ms:.3f}".format(**result),
                flush=True,
            )
        if "trace_path" in result:
            print(f"rank={result['rank']:02d} module={result['module']} trace={result['trace_path']}", flush=True)
        if "cuda_profiler_range_iters" in result:
            print(
                f"rank={result['rank']:02d} module={result['module']} "
                f"cuda_profiler_range_iters={result['cuda_profiler_range_iters']}",
                flush=True,
            )

    if world_size > 1:
        mean_ms = [float(result["mean_ms"]) for result in results]
        fastest = min(results, key=lambda item: float(item["mean_ms"]))
        slowest = max(results, key=lambda item: float(item["mean_ms"]))
        print(
            "summary module={module} avg_ms={avg_ms:.3f} fastest_rank={fastest_rank} "
            "fastest_ms={fastest_ms:.3f} slowest_rank={slowest_rank} slowest_ms={slowest_ms:.3f} "
            "rank_time_ratio={ratio:.3f}".format(
                module=results[0]["module"],
                avg_ms=statistics.fmean(mean_ms),
                fastest_rank=fastest["rank"],
                fastest_ms=float(fastest["mean_ms"]),
                slowest_rank=slowest["rank"],
                slowest_ms=float(slowest["mean_ms"]),
                ratio=float(slowest["mean_ms"]) / float(fastest["mean_ms"]),
            ),
            flush=True,
        )


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size, device = init_distributed()

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)

    if rank == 0:
        print(
            "GDN vs FusedAttentionV2 benchmark: "
            f"modules={args.modules} mode={args.mode} batch={args.batch_size} seq={args.seq_len} "
            f"d_model={args.d_model} d_attn={args.d_attn} n_heads={args.n_heads} "
            f"n_kv_heads={args.n_kv_heads} head_dim={args.head_dim} expand_v={args.expand_v} "
            f"param_dtype={args.param_dtype} compute_dtype={args.compute_dtype} autocast={args.autocast} "
            f"attn_backend={args.attn_backend} attn_gate={args.attn_gate} qk_norm={args.qk_norm} "
            f"warmup={args.warmup_iters} iters={args.bench_iters} world_size={world_size}",
            flush=True,
        )

    x_base = make_input(args, device)
    cases: list[tuple[str, nn.Module]] = []
    if args.modules in ("both", "gdn"):
        cases.append(("gdn", build_gdn(args, device)))
    if args.modules in ("both", "attn"):
        cases.append(("attn", build_attn(args, device)))

    barrier()
    for name, module in cases:
        result = benchmark_module(name, module, x_base, args=args, device=device, rank=rank)
        gathered = gather_result(result)
        if rank == 0:
            print_results(gathered, world_size=world_size)
            print("", flush=True)

        del module
        torch.cuda.empty_cache()

    barrier()
    if is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
