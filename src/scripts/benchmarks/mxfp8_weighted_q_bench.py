import argparse

import torch

try:
    import triton
except Exception as e:  # pragma: no cover
    triton = None
    _TRITON_IMPORT_ERROR = e
else:
    _TRITON_IMPORT_ERROR = None  # type: ignore[assignment]

from olmo_core.kernels.mxfp8_utils import (
    quantize_rows_to_mxfp8,
    weighted_quantize_rows_to_mxfp8,
)


def _bench(label: str, fn, *, warmup: int, rep: int) -> float:
    assert triton is not None
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    print(f"{label:28s} {ms:8.4f} ms")
    return float(ms)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark materialized route weighting plus MXFP8 Q vs fused weighted MXFP8 Q."
    )
    parser.add_argument("--rows", type=int, default=16384)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=40)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if triton is None:
        raise RuntimeError("Triton is required for this benchmark") from _TRITON_IMPORT_ERROR
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if args.cols % 32 != 0:
        raise RuntimeError(f"cols must be divisible by 32, got {args.cols}")

    device = torch.device("cuda")
    torch.manual_seed(1234)
    x = torch.randn(args.rows, args.cols, device=device, dtype=torch.bfloat16)
    weights = torch.rand(args.rows, args.top_k, device=device, dtype=torch.float32)
    weighted = torch.empty(
        args.rows * args.top_k,
        args.cols,
        device=device,
        dtype=torch.bfloat16,
    )
    q_out = torch.empty_like(weighted, dtype=torch.float8_e4m3fn)
    scales_out = torch.empty(
        args.rows * args.top_k,
        args.cols // 32,
        device=device,
        dtype=torch.float8_e8m0fnu,
    )

    def old_materialized_q() -> None:
        torch.mul(
            x.unsqueeze(1),
            weights.to(dtype=x.dtype).unsqueeze(-1),
            out=weighted.view(args.rows, args.top_k, args.cols),
        )
        quantize_rows_to_mxfp8(
            weighted,
            block_size=32,
            out=q_out,
            scales_out=scales_out,
        )

    def fused_weighted_q() -> None:
        weighted_quantize_rows_to_mxfp8(
            x,
            weights,
            block_size=32,
            out=q_out,
            scales_out=scales_out,
        )

    if args.check:
        old_materialized_q()
        q_ref = q_out.clone()
        scales_ref = scales_out.clone()
        fused_weighted_q()
        q_ok = torch.equal(q_out.view(torch.uint8), q_ref.view(torch.uint8))
        scales_ok = torch.equal(scales_out.view(torch.uint8), scales_ref.view(torch.uint8))
        print(f"correctness q={q_ok} scales={scales_ok}")
        if not q_ok or not scales_ok:
            raise RuntimeError("fused weighted Q does not match materialized weighted Q")

    print(
        f"rows={args.rows} cols={args.cols} top_k={args.top_k} "
        f"warmup={args.warmup} rep={args.rep}"
    )
    old_ms = _bench(
        "materialized mul + Q",
        old_materialized_q,
        warmup=args.warmup,
        rep=args.rep,
    )
    fused_ms = _bench(
        "fused weighted Q",
        fused_weighted_q,
        warmup=args.warmup,
        rep=args.rep,
    )
    print(f"speedup             {old_ms / fused_ms:8.3f}x")


if __name__ == "__main__":
    main()
