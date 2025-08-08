# pyright: reportMissingImports=false
import argparse
from typing import Any, Optional

import torch

try:
    import triton.testing as ttesting
except Exception as e:  # pragma: no cover - benchmark utility
    raise RuntimeError(
        "This benchmark requires the 'triton' package. Install it and run on a CUDA GPU."
    ) from e

from olmo_core.nn.rope import RotaryEmbedding, RotaryEmbeddingPerExampleStart


def _parse_dtype(value: str) -> torch.dtype:
    value = value.lower()
    if value in {"fp32", "float32"}:
        return torch.float32
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    raise argparse.ArgumentTypeError(f"Unsupported dtype: {value}")


def _tokens_per_second(batch: int, seqlen: int, elapsed_ms: float) -> float:
    # Measure throughput as tokens/s (per position across the batch)
    return (batch * seqlen) / (elapsed_ms / 1e3)


def _elements_per_second(
    batch: int, heads: int, seqlen: int, head_size: int, elapsed_ms: float
) -> float:
    # Elements rotated per second
    total = batch * heads * seqlen * head_size
    return total / (elapsed_ms / 1e3)


@torch.inference_mode()
def _bench_forward(
    module: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    head_first: bool,
    start_pos: Optional[torch.Tensor | int],
    dtype: torch.dtype,
    use_autocast: bool,
):
    def _run():
        if use_autocast:
            with torch.autocast(device_type=q.device.type, dtype=dtype, enabled=True):
                module(q, k, head_first=head_first, start_pos=start_pos)
        else:
            module(q, k, head_first=head_first, start_pos=start_pos)

    return ttesting.do_bench(_run)


def _bench_forward_backward(
    module: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    head_first: bool,
    start_pos: Optional[torch.Tensor | int],
    dtype: torch.dtype,
    use_autocast: bool,
):
    def _run():
        q1 = q.detach().clone().requires_grad_(True)
        k1 = k.detach().clone().requires_grad_(True)
        if use_autocast:
            with torch.autocast(device_type=q.device.type, dtype=dtype, enabled=True):
                qo, ko = module(q1, k1, head_first=head_first, start_pos=start_pos)
                loss = qo.sum() + ko.sum()
        else:
            qo, ko = module(q1, k1, head_first=head_first, start_pos=start_pos)
            loss = qo.sum() + ko.sum()
        loss.backward()

    return ttesting.do_bench(_run)


def main():  # pragma: no cover - benchmark utility
    parser = argparse.ArgumentParser(description="Benchmark RoPE implementations (Triton)")
    parser.add_argument("--device", default="cuda", choices=["cuda"], help="Device (GPU only)")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--seq", type=int, default=2048, help="Sequence length")
    parser.add_argument("--heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--head-size", type=int, default=128, help="Head size (per head)")
    parser.add_argument(
        "--head-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Benchmark with head-first layout (B, H, T, D)",
    )
    parser.add_argument(
        "--dtype",
        type=_parse_dtype,
        default="bf16",
        help="Computation dtype for inputs (bf16/fp16/fp32)",
    )
    parser.add_argument(
        "--full-precision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, RoPE computes in fp32 regardless of input dtype",
    )
    parser.add_argument(
        "--start-mode",
        choices=["default", "scalar", "vector"],
        default="default",
        help="How to set start_pos: none, single int, or per-example vector",
    )
    parser.add_argument("--scalar-start", type=int, default=64, help="Scalar start_pos when used")
    parser.add_argument(
        "--bench-mode",
        choices=["forward", "fwbw", "both"],
        default="both",
        help="Which pass to benchmark: forward-only, forward+backward, or both",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compile modules with torch.compile before benchmarking",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this benchmark")

    device = torch.device(args.device)
    torch.manual_seed(1337)

    B, T, H, HS = args.batch, args.seq, args.heads, args.head_size
    shape = (B, H, T, HS) if args.head_first else (B, T, H, HS)

    # Inputs
    q = torch.randn(*shape, device=device, dtype=args.dtype)
    k = torch.randn(*shape, device=device, dtype=args.dtype)

    # Modules
    rope_default = RotaryEmbedding(head_size=HS, full_precision=args.full_precision).to(device)
    rope_batched = RotaryEmbeddingPerExampleStart(
        head_size=HS, full_precision=args.full_precision
    ).to(device)
    if args.compile and hasattr(torch, "compile"):
        # Use inductor; allow dynamic shapes since seq/start_pos can vary
        rope_default = torch.compile(rope_default, mode="max-autotune")
        rope_batched = torch.compile(rope_batched, mode="max-autotune")

    # Warm up cache and CUDA context
    rope_default.warmup_cache(args.seq + args.scalar_start, device)
    rope_batched.warmup_cache(args.seq + args.scalar_start, device)
    torch.cuda.synchronize()

    # Determine start_pos argument(s)
    start_default: Optional[int] = None
    start_batched: Optional[int | torch.Tensor] = None
    if args.start_mode == "default":
        start_default = None
        start_batched = None
    elif args.start_mode == "scalar":
        start_default = int(args.scalar_start)
        start_batched = int(args.scalar_start)
    else:  # vector
        start_default = None  # not supported for RotaryEmbedding
        start_batched = torch.randint(
            low=0, high=max(args.scalar_start, 1), size=(B,), device=device
        )

    # Use autocast for bf16/fp16 only if not forcing full precision
    use_autocast = (args.dtype in {torch.bfloat16, torch.float16}) and (not args.full_precision)

    print("RoPE benchmark (Triton)")
    print(f" - device         : {device}")
    print(f" - shape          : B={B}, T={T}, H={H}, HS={HS}, head_first={args.head_first}")
    print(f" - dtype          : {str(args.dtype).replace('torch.', '')}")
    print(f" - full_precision : {args.full_precision}")
    print(f" - start_mode     : {args.start_mode}")

    def _print_row(label: str, ms: float):
        tps = _tokens_per_second(B, T, ms)
        eps = _elements_per_second(B, H, T, HS, ms)
        print(f"{label:<12}{ms:8.3f} ms | {tps / 1e6:6.3f} Mtok/s | {eps / 1e9:6.3f} Gelem/s")

    # Optional warmup runs to trigger torch.compile graphs
    if args.compile:
        _ = rope_default(q, k, head_first=args.head_first, start_pos=start_default)
        _ = rope_batched(q, k, head_first=args.head_first, start_pos=start_batched)
        if args.bench_mode in {"fwbw", "both"}:
            q_w = q.detach().clone().requires_grad_(True)
            k_w = k.detach().clone().requires_grad_(True)
            out = rope_default(q_w, k_w, head_first=args.head_first, start_pos=start_default)
            (out[0].sum() + out[1].sum()).backward()
            q_w = q.detach().clone().requires_grad_(True)
            k_w = k.detach().clone().requires_grad_(True)
            out = rope_batched(q_w, k_w, head_first=args.head_first, start_pos=start_batched)
            (out[0].sum() + out[1].sum()).backward()
        torch.cuda.synchronize()

    # Benchmark default implementation
    if args.start_mode != "vector":
        if args.bench_mode in {"forward", "both"}:
            ms_default_fwd = _bench_forward(
                rope_default,
                q,
                k,
                head_first=args.head_first,
                start_pos=start_default,
                dtype=args.dtype,
                use_autocast=use_autocast,
            )
            _print_row("default fwd", ms_default_fwd)
        if args.bench_mode in {"fwbw", "both"}:
            ms_default_fb = _bench_forward_backward(
                rope_default,
                q,
                k,
                head_first=args.head_first,
                start_pos=start_default,
                dtype=args.dtype,
                use_autocast=use_autocast,
            )
            _print_row("default fb", ms_default_fb)
    else:
        print("default     N/A (vector start_pos unsupported)")

    # Benchmark per-example implementation
    if args.bench_mode in {"forward", "both"}:
        ms_batched_fwd = _bench_forward(
            rope_batched,
            q,
            k,
            head_first=args.head_first,
            start_pos=start_batched,
            dtype=args.dtype,
            use_autocast=use_autocast,
        )
        _print_row("batched fwd", ms_batched_fwd)
    if args.bench_mode in {"fwbw", "both"}:
        ms_batched_fb = _bench_forward_backward(
            rope_batched,
            q,
            k,
            head_first=args.head_first,
            start_pos=start_batched,
            dtype=args.dtype,
            use_autocast=use_autocast,
        )
        _print_row("batched fb", ms_batched_fb)


if __name__ == "__main__":  # pragma: no cover - benchmark utility
    main()
