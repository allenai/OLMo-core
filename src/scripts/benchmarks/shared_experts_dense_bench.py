"""
Quick forward-time comparison of the two dense-MLP implementations: MoE-v2's single-expert
``SharedExperts`` vs the standard ``FeedForward``. See ``shared_experts_v2_test.py`` for the
correctness/parity checks.

Run as a script (any device; auto-selects CUDA + bf16 when available). The size sweep used to
compare the two implementations (run on a GPU with the torch 2.10 image):

    python src/scripts/benchmarks/shared_experts_dense_bench.py \
        --sizes 1024:4096 2048:8192 4096:11008 4096:14336 \
        --seq-len 2048 --batch-size 4

Or a single size:
    python src/scripts/benchmarks/shared_experts_dense_bench.py --d-model 4096 --hidden-size 11008
"""

import argparse
import time
from typing import Callable, Tuple

import torch

from olmo_core.config import DType
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig


def _timed_ms(fn: Callable[[], object], *, warmup: int, iters: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end) / iters)

    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) * 1e3 / iters


def benchmark(
    *,
    batch_size: int,
    seq_len: int,
    d_model: int,
    hidden_size: int,
    warmup: int,
    iters: int,
    device: torch.device,
    dtype: DType,
    compile: bool = False,
    compile_mode: str = "default",
) -> Tuple[float, float]:
    """Time ``FeedForward`` and single-expert ``SharedExperts`` forwards; return (ff_ms, se_ms)."""
    torch.manual_seed(0)

    feed_forward = FeedForwardConfig(hidden_size=hidden_size, bias=False, dtype=dtype).build(
        d_model=d_model, init_device=device.type
    )
    shared = SharedExpertsConfig(
        d_model=d_model, hidden_size=hidden_size, num_experts=1, bias=False, dtype=dtype
    ).build(init_device=device.type)

    ff_fn: Callable[[torch.Tensor], object] = feed_forward
    se_fn: Callable[[torch.Tensor], object] = shared
    if compile:
        # `SharedExperts` pays for a permute + strided SwiGLU + batch-1 bmm that inductor can fuse
        # away, so compile both to compare on equal footing. The warmup iters absorb the JIT cost.
        ff_fn = torch.compile(feed_forward, mode=compile_mode)
        se_fn = torch.compile(shared, mode=compile_mode)

    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype.as_pt())

    with torch.no_grad():
        ff_ms = _timed_ms(lambda: ff_fn(x), warmup=warmup, iters=iters, device=device)
        se_ms = _timed_ms(lambda: se_fn(x), warmup=warmup, iters=iters, device=device)

    compile_note = f", compile={compile_mode}" if compile else ""
    print(f"Dense MLP forward ({device.type}, {dtype.value}, {tuple(x.shape)}{compile_note})")
    print(f"  FeedForward (standard):          {ff_ms:.3f} ms")
    print(f"  SharedExperts (dense, E=1):      {se_ms:.3f} ms")
    print(f"  SharedExperts / FeedForward:     {se_ms / max(ff_ms, 1e-6):.3f}x")
    return ff_ms, se_ms


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--d-model", type=int, default=2048)
    p.add_argument("--hidden-size", type=int, default=8192)
    p.add_argument(
        "--sizes",
        nargs="+",
        default=None,
        metavar="D_MODEL:HIDDEN_SIZE",
        help="Sweep several sizes in one run, e.g. --sizes 1024:4096 4096:11008. "
        "Overrides --d-model/--hidden-size.",
    )
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=[d.value for d in DType], default=None)
    p.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile both implementations before timing.",
    )
    p.add_argument(
        "--compile-mode",
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode (only used with --compile).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    dtype = (
        DType(args.dtype)
        if args.dtype is not None
        else (DType.bfloat16 if device.type == "cuda" else DType.float32)
    )

    if args.sizes:
        sizes = [tuple(int(v) for v in s.split(":", 1)) for s in args.sizes]
    else:
        sizes = [(args.d_model, args.hidden_size)]

    for d_model, hidden_size in sizes:
        benchmark(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            d_model=d_model,
            hidden_size=hidden_size,
            warmup=args.warmup,
            iters=args.iters,
            device=device,
            dtype=dtype,
            compile=args.compile,
            compile_mode=args.compile_mode,
        )


if __name__ == "__main__":
    main()
