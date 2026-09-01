import argparse
from typing import Sequence

import grouped_gemm.ops as grouped_gemm_ops
import torch
import torch.nn.functional as F
from transformer_engine.pytorch import GroupedLinear

DEFAULT_EXPERTS = (2, 4, 8, 16, 32, 64, 128)


def _parse_expert_counts(text: str) -> list[int]:
    counts = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not counts:
        raise ValueError("Expected at least one expert count")
    if any(count <= 0 for count in counts):
        raise ValueError(f"Expert counts must be positive, got: {counts}")
    return counts


def _parse_dtype(dtype_name: str) -> torch.dtype:
    name = dtype_name.strip().lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16"):
        return torch.float16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Benchmark transformer_engine GroupedLinear vs torch.grouped_mm "
            "vs grouped_gemm.ops.gmm across different expert counts."
        )
    )
    p.add_argument(
        "--expert-counts",
        type=str,
        default=",".join(str(x) for x in DEFAULT_EXPERTS),
        help="Comma-separated expert counts (default: 2,4,8,16,32,64).",
    )
    p.add_argument(
        "--total-tokens",
        type=int,
        default=128 * 1024,
        help="Total tokens fixed across all expert-count cases.",
    )
    p.add_argument("--k", type=int, default=2560, help="Input feature size.")
    p.add_argument("--n", type=int, default=2560, help="Output feature size.")
    p.add_argument("--dtype", type=str, default="bf16", help="bf16 or fp16.")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument(
        "--skip-correctness-check",
        action="store_true",
        help="Skip one-time numerical agreement check for each expert count.",
    )
    return p.parse_args()


def _require_runtime_support() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if not hasattr(F, "grouped_mm"):
        raise RuntimeError("torch.nn.functional.grouped_mm is required")


def _build_unique_token_splits(num_experts: int, total_tokens: int) -> list[int]:
    min_required = num_experts * (num_experts + 1) // 2
    if total_tokens < min_required:
        raise ValueError(
            "Cannot assign unique positive token counts with fixed total tokens. "
            f"Need at least {min_required} tokens for {num_experts} experts, got {total_tokens}."
        )

    tri = num_experts * (num_experts - 1) // 2

    # Make splits moderately more uneven than unit-stride while staying feasible.
    target_step = 2
    if tri > 0:
        max_step = (total_tokens - num_experts) // tri
        step = max(1, min(target_step, max_step))
    else:
        step = 1

    start = (total_tokens - step * tri) // num_experts
    splits = [start + step * idx for idx in range(num_experts)]

    remaining = total_tokens - sum(splits)
    if remaining > 0:
        q, r = divmod(remaining, num_experts)
        if q:
            splits = [v + q for v in splits]
        for i in range(r):
            splits[num_experts - r + i] += 1

    assert sum(splits) == total_tokens
    assert len(set(splits)) == num_experts
    return splits


def _timed_ms(fn, *, warmup: int, iters: int) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters)


def _build_te_module(
    *,
    num_experts: int,
    k: int,
    n: int,
    dtype: torch.dtype,
    device: torch.device,
    weights_kn: torch.Tensor,
) -> GroupedLinear:
    module = GroupedLinear(
        num_gemms=num_experts,
        in_features=k,
        out_features=n,
        bias=False,
        params_dtype=dtype,
        device=device,
    )
    module.eval()
    with torch.no_grad():
        for expert_idx in range(num_experts):
            te_weight = getattr(module, f"weight{expert_idx}")
            te_weight.copy_(weights_kn[expert_idx].transpose(0, 1).contiguous())
    return module


def _run_case(
    *,
    num_experts: int,
    total_tokens: int,
    k: int,
    n: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    run_correctness_check: bool,
) -> dict[str, float | int]:
    device = torch.device("cuda")
    token_splits_cpu = _build_unique_token_splits(num_experts, total_tokens)

    x_mk = torch.randn(total_tokens, k, device=device, dtype=dtype)
    w_kn = torch.randn(num_experts, k, n, device=device, dtype=dtype)

    # torch.grouped_mm path: GPU-side offsets.
    offs_gpu = torch.tensor(token_splits_cpu, device=device, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    assert offs_gpu.is_cuda

    # TE + grouped_gemm paths: CPU-side split sizes.
    batch_sizes_cpu = torch.tensor(token_splits_cpu, device="cpu", dtype=torch.int64)
    assert not batch_sizes_cpu.is_cuda

    te_module = _build_te_module(
        num_experts=num_experts,
        k=k,
        n=n,
        dtype=dtype,
        device=device,
        weights_kn=w_kn,
    )

    if run_correctness_check:
        with torch.no_grad():
            out_torch = F.grouped_mm(x_mk, w_kn, offs=offs_gpu, out_dtype=dtype)
            out_te = te_module(x_mk, token_splits_cpu)
            out_grouped_gemm = grouped_gemm_ops.gmm(x_mk, w_kn, batch_sizes_cpu, trans_b=False)
        # Kernels differ internally; tolerate expected low-precision drift.
        torch.testing.assert_close(out_te, out_torch, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(out_grouped_gemm, out_torch, atol=2e-2, rtol=2e-2)

    te_ms = _timed_ms(lambda: te_module(x_mk, token_splits_cpu), warmup=warmup, iters=iters)
    torch_ms = _timed_ms(
        lambda: F.grouped_mm(x_mk, w_kn, offs=offs_gpu, out_dtype=dtype),
        warmup=warmup,
        iters=iters,
    )
    grouped_gemm_ms = _timed_ms(
        lambda: grouped_gemm_ops.gmm(x_mk, w_kn, batch_sizes_cpu, trans_b=False),
        warmup=warmup,
        iters=iters,
    )

    return {
        "experts": num_experts,
        "total_tokens": total_tokens,
        "min_tokens": min(token_splits_cpu),
        "max_tokens": max(token_splits_cpu),
        "te_ms": te_ms,
        "torch_ms": torch_ms,
        "grouped_gemm_ms": grouped_gemm_ms,
    }


def _print_results(results: Sequence[dict[str, float | int]]) -> None:
    cols = (
        ("experts", 8),
        ("tokens", 10),
        ("min/max", 15),
        ("torch.grouped_mm", 18),
        ("TE.GroupedLinear", 18),
        ("grouped_gemm.gmm", 18),
        ("best", 12),
    )
    header = " ".join(f"{name:>{width}}" for name, width in cols)
    print(header)
    print("-" * len(header))
    for row in results:
        torch_ms = float(row["torch_ms"])
        te_ms = float(row["te_ms"])
        gmm_ms = float(row["grouped_gemm_ms"])
        best_name, best_ms = min(
            (
                ("torch", torch_ms),
                ("TE", te_ms),
                ("gmm", gmm_ms),
            ),
            key=lambda item: item[1],
        )
        min_max = f"{int(row['min_tokens'])}/{int(row['max_tokens'])}"
        best = f"{best_name}:{best_ms:.3f}"
        print(
            " ".join(
                (
                    f"{int(row['experts']):>8d}",
                    f"{int(row['total_tokens']):>10d}",
                    f"{min_max:>15}",
                    f"{torch_ms:>18.3f}",
                    f"{te_ms:>18.3f}",
                    f"{gmm_ms:>18.3f}",
                    f"{best:>12}",
                )
            )
        )


def main() -> None:
    args = _parse_args()
    _require_runtime_support()
    dtype = _parse_dtype(args.dtype)
    expert_counts = _parse_expert_counts(args.expert_counts)

    torch.manual_seed(1234)

    print(
        "Running benchmark with GPU-side offsets for torch.grouped_mm and "
        "CPU-side split sizes for TE/grouped_gemm."
    )
    print(
        f"dtype={dtype}, k={args.k}, n={args.n}, "
        f"total_tokens={args.total_tokens}, warmup={args.warmup}, iters={args.iters}"
    )

    results: list[dict[str, float | int]] = []
    for num_experts in expert_counts:
        torch.cuda.empty_cache()
        result = _run_case(
            num_experts=num_experts,
            total_tokens=args.total_tokens,
            k=args.k,
            n=args.n,
            dtype=dtype,
            warmup=args.warmup,
            iters=args.iters,
            run_correctness_check=not args.skip_correctness_check,
        )
        results.append(result)

    _print_results(results)


if __name__ == "__main__":
    main()
