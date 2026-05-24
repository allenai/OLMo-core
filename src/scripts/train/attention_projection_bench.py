"""
Microbenchmark regular attention vs FusedAttentionV2 with bf16 and MXFP8 projections.

The benchmark times:
- end-to-end forward
- end-to-end forward + backward
- forward QKV prep, core attention, output projection
- isolated backward for QKV prep, core attention, output projection

Example:
    PYTHONPATH=src python src/scripts/train/attention_projection_bench.py \
        --backend flash_4 --batch-size 2 --seq-len 8192 --d-model 4096 \
        --d-attn 5120 --n-heads 40 --n-kv-heads 10
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from typing import Callable, Iterable

import torch

from olmo_core.nn.attention import Attention, AttentionBackendName, FusedAttentionV2


@dataclass(frozen=True)
class BenchResult:
    name: str
    total_fwd_ms: float
    total_fwd_bwd_ms: float
    qkv_fwd_ms: float
    core_fwd_ms: float
    out_fwd_ms: float
    qkv_bwd_ms: float
    core_bwd_ms: float
    out_bwd_ms: float

    @property
    def estimated_bwd_ms(self) -> float:
        return self.qkv_bwd_ms + self.core_bwd_ms + self.out_bwd_ms


def run_with_nvtx(label: str, fn: Callable[[], None]) -> None:
    torch.cuda.nvtx.range_push(label)
    try:
        fn()
    finally:
        torch.cuda.nvtx.range_pop()


def cuda_time(
    label: str,
    fn: Callable[[], None],
    *,
    warmup: int,
    iters: int,
) -> float:
    for _ in range(warmup):
        run_with_nvtx(f"{label}/warmup", fn)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run_with_nvtx(label, fn)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def cuda_time_backward(
    label: str,
    prepare: Callable[[], tuple[Iterable[torch.Tensor], Iterable[torch.Tensor]]],
    *,
    warmup: int,
    iters: int,
) -> float:
    def run_once() -> None:
        outputs, grad_outputs = prepare()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.cuda.nvtx.range_push(label)
        try:
            torch.autograd.backward(tuple(outputs), tuple(grad_outputs))
        finally:
            torch.cuda.nvtx.range_pop()
        end.record()
        torch.cuda.synchronize()
        if not hasattr(run_once, "times"):
            run_once.times = []  # type: ignore[attr-defined]
        run_once.times.append(start.elapsed_time(end))  # type: ignore[attr-defined]

    for _ in range(warmup):
        outputs, grad_outputs = prepare()
        torch.autograd.backward(tuple(outputs), tuple(grad_outputs))
    torch.cuda.synchronize()

    run_once.times = []  # type: ignore[attr-defined]
    for _ in range(iters):
        run_once()
    return statistics.median(run_once.times)  # type: ignore[attr-defined]


def zero_module_grads(module: torch.nn.Module) -> None:
    module.zero_grad(set_to_none=True)


def refresh_mxfp8_attention_cache(module: torch.nn.Module) -> None:
    refresh = getattr(module, "refresh_mxfp8_attention_cache", None)
    if refresh is not None:
        refresh()


def make_attention(
    kind: str,
    *,
    d_model: int,
    d_attn: int,
    n_heads: int,
    n_kv_heads: int,
    backend: AttentionBackendName,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.nn.Module:
    common_kwargs = dict(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        d_attn=d_attn,
        bias=False,
        backend=backend,
        dtype=dtype,
        init_device=str(device),
    )
    if kind == "regular":
        module = Attention(**common_kwargs)
    elif kind == "fused_v2_bf16":
        module = FusedAttentionV2(**common_kwargs)
    elif kind == "fused_v2_mxfp8":
        module = FusedAttentionV2(**common_kwargs, mxfp8_projections=True)
    else:
        raise NotImplementedError(kind)

    module.train()
    refresh_mxfp8_attention_cache(module)
    return module


def benchmark_module(
    name: str,
    module: torch.nn.Module,
    x_base: torch.Tensor,
    grad_y: torch.Tensor,
    *,
    warmup: int,
    iters: int,
) -> BenchResult:
    B, T, _ = x_base.shape
    n_heads = module.n_heads  # type: ignore[attr-defined]
    head_dim = module.head_dim  # type: ignore[attr-defined]
    q_dim = n_heads * head_dim

    with torch.no_grad():
        q_static, k_static, v_static = module._prepare_qkv(x_base)  # type: ignore[attr-defined]
        att_static = module.sdpa(q_static, k_static, v_static)  # type: ignore[attr-defined]
        att_flat_static = att_static.reshape(B, T, q_dim).contiguous()

    q_grad = torch.randn_like(q_static)
    k_grad = torch.randn_like(k_static)
    v_grad = torch.randn_like(v_static)
    att_grad = torch.randn_like(att_static)

    def total_fwd() -> None:
        with torch.no_grad():
            module(x_base)

    def total_fwd_bwd() -> None:
        zero_module_grads(module)
        x = x_base.detach().requires_grad_(True)
        y = module(x)
        y.backward(grad_y)

    def qkv_fwd() -> None:
        with torch.no_grad():
            module._prepare_qkv(x_base)  # type: ignore[attr-defined]

    def core_fwd() -> None:
        with torch.no_grad():
            module.sdpa(q_static, k_static, v_static)  # type: ignore[attr-defined]

    def out_fwd() -> None:
        with torch.no_grad():
            module.w_out(att_flat_static)  # type: ignore[attr-defined]

    def prepare_qkv_bwd() -> tuple[Iterable[torch.Tensor], Iterable[torch.Tensor]]:
        zero_module_grads(module)
        x = x_base.detach().requires_grad_(True)
        q, k, v = module._prepare_qkv(x)  # type: ignore[attr-defined]
        return (q, k, v), (q_grad, k_grad, v_grad)

    def prepare_core_bwd() -> tuple[Iterable[torch.Tensor], Iterable[torch.Tensor]]:
        q = q_static.detach().requires_grad_(True)
        k = k_static.detach().requires_grad_(True)
        v = v_static.detach().requires_grad_(True)
        att = module.sdpa(q, k, v)  # type: ignore[attr-defined]
        return (att,), (att_grad,)

    def prepare_out_bwd() -> tuple[Iterable[torch.Tensor], Iterable[torch.Tensor]]:
        zero_module_grads(module)
        att_flat = att_flat_static.detach().requires_grad_(True)
        y = module.w_out(att_flat)  # type: ignore[attr-defined]
        return (y,), (grad_y,)

    return BenchResult(
        name=name,
        total_fwd_ms=cuda_time(f"{name}/total_fwd", total_fwd, warmup=warmup, iters=iters),
        total_fwd_bwd_ms=cuda_time(f"{name}/total_fwd_bwd", total_fwd_bwd, warmup=warmup, iters=iters),
        qkv_fwd_ms=cuda_time(f"{name}/qkv_fwd", qkv_fwd, warmup=warmup, iters=iters),
        core_fwd_ms=cuda_time(f"{name}/core_fwd", core_fwd, warmup=warmup, iters=iters),
        out_fwd_ms=cuda_time(f"{name}/out_fwd", out_fwd, warmup=warmup, iters=iters),
        qkv_bwd_ms=cuda_time_backward(f"{name}/qkv_bwd", prepare_qkv_bwd, warmup=warmup, iters=iters),
        core_bwd_ms=cuda_time_backward(f"{name}/core_bwd", prepare_core_bwd, warmup=warmup, iters=iters),
        out_bwd_ms=cuda_time_backward(f"{name}/out_bwd", prepare_out_bwd, warmup=warmup, iters=iters),
    )


def print_results(results: list[BenchResult]) -> None:
    headers = [
        "case",
        "fwd",
        "fwd+bwd",
        "qkv_f",
        "core_f",
        "out_f",
        "qkv_b",
        "core_b",
        "out_b",
        "bwd_parts",
    ]
    rows = [
        [
            result.name,
            result.total_fwd_ms,
            result.total_fwd_bwd_ms,
            result.qkv_fwd_ms,
            result.core_fwd_ms,
            result.out_fwd_ms,
            result.qkv_bwd_ms,
            result.core_bwd_ms,
            result.out_bwd_ms,
            result.estimated_bwd_ms,
        ]
        for result in results
    ]

    widths = [
        max(len(headers[i]), *(len(format_cell(row[i])) for row in rows))
        for i in range(len(headers))
    ]
    print(" ".join(headers[i].rjust(widths[i]) for i in range(len(headers))))
    print(" ".join("-" * width for width in widths))
    for row in rows:
        print(" ".join(format_cell(row[i]).rjust(widths[i]) for i in range(len(headers))))


def format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="flash_4", choices=[x.value for x in AttentionBackendName])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--d-model", type=int, default=4096)
    parser.add_argument("--d-attn", type=int, default=5120)
    parser.add_argument("--n-heads", type=int, default=40)
    parser.add_argument("--n-kv-heads", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["regular", "fused_v2_bf16", "fused_v2_mxfp8"],
        choices=["regular", "fused_v2_bf16", "fused_v2_mxfp8"],
        help="Benchmark only the selected cases.",
    )
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help="Wrap the benchmark in cudaProfilerStart/Stop for nsys capture-range=cudaProfilerApi.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")

    backend = AttentionBackendName(args.backend)
    backend.assert_supported()

    if (args.batch_size * args.seq_len) % 32 != 0:
        raise ValueError("batch_size * seq_len must be divisible by 32 for MXFP8 wgrad")

    torch.manual_seed(args.seed)
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    dtype = torch.bfloat16

    x_base = torch.randn(
        args.batch_size,
        args.seq_len,
        args.d_model,
        device=device,
        dtype=dtype,
    )
    grad_y = torch.randn_like(x_base)

    cases = [(case, case) for case in args.cases]

    if args.cuda_profiler_range:
        for label, kind in cases:
            print(f"Prewarming {label} before cudaProfilerStart...", flush=True)
            module = make_attention(
                kind,
                d_model=args.d_model,
                d_attn=args.d_attn,
                n_heads=args.n_heads,
                n_kv_heads=args.n_kv_heads,
                backend=backend,
                dtype=dtype,
                device=device,
            )
            benchmark_module(
                label,
                module,
                x_base,
                grad_y,
                warmup=1,
                iters=1,
            )
            del module
            torch.cuda.empty_cache()
        torch.cuda.synchronize()

    results = []
    if args.cuda_profiler_range:
        torch.cuda.cudart().cudaProfilerStart()
    try:
        for label, kind in cases:
            print(f"Running {label}...", flush=True)
            module = make_attention(
                kind,
                d_model=args.d_model,
                d_attn=args.d_attn,
                n_heads=args.n_heads,
                n_kv_heads=args.n_kv_heads,
                backend=backend,
                dtype=dtype,
                device=device,
            )
            results.append(
                benchmark_module(
                    label,
                    module,
                    x_base,
                    grad_y,
                    warmup=args.warmup,
                    iters=args.iters,
                )
            )
            del module
            torch.cuda.empty_cache()
        torch.cuda.synchronize()
    finally:
        if args.cuda_profiler_range:
            torch.cuda.cudart().cudaProfilerStop()

    print()
    print("Times are median milliseconds. Backward component rows exclude their forward setup.")
    print_results(results)


if __name__ == "__main__":
    main()
