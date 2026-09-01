import argparse

import torch
import torch.nn.functional as F

from olmo_core.kernels import scaled_grouped_mm_q


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark BF16 grouped_mm vs MXFP8 scaled_grouped_mm_q"
    )
    p.add_argument("--num-groups", type=int, default=16)
    p.add_argument("--tokens-per-group", type=int, default=256)
    p.add_argument("--k", type=int, default=1024)
    p.add_argument("--n", type=int, default=2048)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--run-backward", action="store_true")
    return p.parse_args()


def _timed_ms(fn, *, warmup: int, iters: int) -> float:
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


def _require_runtime_support() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not hasattr(F, "grouped_mm"):
        raise RuntimeError("torch.nn.functional.grouped_mm is required")
    if not hasattr(F, "scaled_grouped_mm"):
        raise RuntimeError("torch.nn.functional.scaled_grouped_mm is required")

    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    if major < 10:
        raise RuntimeError(
            "MXFP8 benchmark requires SM100+ for this implementation; "
            f"detected capability {major}.{minor}"
        )


def main() -> None:
    args = _parse_args()
    _require_runtime_support()

    device = torch.device("cuda")
    dtype = torch.bfloat16

    total_tokens = args.num_groups * args.tokens_per_group
    offs = torch.arange(
        args.tokens_per_group,
        total_tokens + 1,
        args.tokens_per_group,
        device=device,
        dtype=torch.int32,
    )

    torch.manual_seed(123)
    a = torch.randn(total_tokens, args.k, device=device, dtype=dtype)
    b = torch.randn(args.num_groups, args.k, args.n, device=device, dtype=dtype)

    fwd_bf16_ms = _timed_ms(
        lambda: F.grouped_mm(a, b, offs=offs, out_dtype=torch.bfloat16),
        warmup=args.warmup,
        iters=args.iters,
    )
    fwd_fp8_ms = _timed_ms(
        lambda: scaled_grouped_mm_q(a, b, offs=offs),
        warmup=args.warmup,
        iters=args.iters,
    )

    print("Forward")
    print(f"  grouped_mm bf16:           {fwd_bf16_ms:.3f} ms")
    print(f"  scaled_grouped_mm_q mxfp8: {fwd_fp8_ms:.3f} ms")
    print(f"  speedup:                   {fwd_bf16_ms / max(fwd_fp8_ms, 1e-6):.3f}x")

    if not args.run_backward:
        return

    a_bwd = a.detach().clone().requires_grad_(True)
    b_bwd = b.detach().clone().requires_grad_(True)
    a_bwd_fp8 = a.detach().clone().requires_grad_(True)
    b_bwd_fp8 = b.detach().clone().requires_grad_(True)
    input_grad_out = torch.empty_like(a_bwd_fp8)

    def _bwd_bf16():
        a_bwd.grad = None
        b_bwd.grad = None
        out = F.grouped_mm(a_bwd, b_bwd, offs=offs, out_dtype=torch.bfloat16)
        loss = out.float().square().mean()
        loss.backward()

    def _bwd_fp8():
        a_bwd_fp8.grad = None
        b_bwd_fp8.grad = None
        out = scaled_grouped_mm_q(
            a_bwd_fp8,
            b_bwd_fp8,
            offs=offs,
            input_grad_out=input_grad_out,
        )
        loss = out.float().square().mean()
        loss.backward()

    bwd_bf16_ms = _timed_ms(_bwd_bf16, warmup=args.warmup, iters=args.iters)
    bwd_fp8_ms = _timed_ms(_bwd_fp8, warmup=args.warmup, iters=args.iters)

    print("Backward")
    print(f"  grouped_mm bf16:           {bwd_bf16_ms:.3f} ms")
    print(f"  scaled_grouped_mm_q mxfp8: {bwd_fp8_ms:.3f} ms")
    print(f"  speedup:                   {bwd_bf16_ms / max(bwd_fp8_ms, 1e-6):.3f}x")


if __name__ == "__main__":
    main()
