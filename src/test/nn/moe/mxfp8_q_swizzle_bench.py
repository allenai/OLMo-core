import argparse
from typing import Optional

import torch

from olmo_core.kernels.mxfp8_utils import (
    grouped_scales_to_mxfp8_blocked,
    quantize_grouped_2d_to_mxfp8_blocked_fused,
    quantize_rows_to_mxfp8,
)

_TRITON_IMPORT_ERROR: Optional[Exception]
try:
    import triton
except Exception as e:  # pragma: no cover
    triton = None
    _TRITON_IMPORT_ERROR = e
else:
    _TRITON_IMPORT_ERROR = None


def _balanced_offs(rows: int, groups: int, device: torch.device) -> torch.Tensor:
    base = rows // groups
    rem = rows % groups
    sizes = [base + (1 if i < rem else 0) for i in range(groups)]
    return torch.tensor(sizes, device=device, dtype=torch.int32).cumsum(0)


def _bench(label: str, fn, *, warmup: int, rep: int) -> float:
    assert triton is not None
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    print(f"{label:24s} {ms:8.4f} ms")
    return float(ms)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark MXFP8 Q, Q+swizzle, and fused Q+swizzle"
    )
    parser.add_argument("--rows", type=int, default=40960)
    parser.add_argument("--cols", type=int, default=2048)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if triton is None:
        raise RuntimeError("Triton is required for this benchmark") from _TRITON_IMPORT_ERROR
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    device = torch.device("cuda")
    torch.manual_seed(123)
    x = torch.randn((args.rows, args.cols), device=device, dtype=dtype)
    offs = _balanced_offs(args.rows, args.groups, device)

    if args.check:
        q_ref, row_scales = quantize_rows_to_mxfp8(x, block_size=32)
        blocked_ref = grouped_scales_to_mxfp8_blocked(row_scales, offs)
        q_fused, blocked_fused = quantize_grouped_2d_to_mxfp8_blocked_fused(x, offs)
        torch.cuda.synchronize()
        q_ok = torch.equal(q_ref.view(torch.uint8), q_fused.view(torch.uint8))
        scales_ok = torch.equal(blocked_ref.view(torch.uint8), blocked_fused.view(torch.uint8))
        print(f"correctness q={q_ok} scales={scales_ok}")
        if not q_ok or not scales_ok:
            raise RuntimeError("fused Q+swizzle does not match Q followed by swizzle")

    print(
        f"rows={args.rows} cols={args.cols} groups={args.groups} dtype={dtype} "
        f"warmup={args.warmup} rep={args.rep}"
    )
    q_ms = _bench(
        "Q",
        lambda: quantize_rows_to_mxfp8(x, block_size=32),
        warmup=args.warmup,
        rep=args.rep,
    )
    q_swizzle_ms = _bench(
        "Q + swizzle",
        lambda: grouped_scales_to_mxfp8_blocked(
            quantize_rows_to_mxfp8(x, block_size=32)[1],
            offs,
            zero_unwritten_tail=False,
        ),
        warmup=args.warmup,
        rep=args.rep,
    )
    fused_ms = _bench(
        "fused Q+swizzle",
        lambda: quantize_grouped_2d_to_mxfp8_blocked_fused(
            x,
            offs,
            block_size=32,
            zero_unwritten_tail=False,
        ),
        warmup=args.warmup,
        rep=args.rep,
    )

    print(f"fused / Q           {fused_ms / q_ms:8.3f}x")
    print(f"fused / Q+swizzle   {fused_ms / q_swizzle_ms:8.3f}x")


if __name__ == "__main__":
    main()
