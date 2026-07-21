import argparse

import torch

from olmo_core.config import DType
from olmo_core.kernels.mxfp8_utils import (
    quantize_grouped_2d_to_mxfp8_blocked,
    quantize_grouped_weight_3d_to_mxfp8_blocked,
)
from olmo_core.nn.moe.v2.fp8 import MoERowwiseFP8Config
from olmo_core.nn.moe.v2.routed_experts import RoutedExperts


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark RoutedExperts BF16 vs FP8 rowwise path")
    p.add_argument("--num-experts", type=int, default=16)
    p.add_argument("--tokens-per-expert", type=int, default=256)
    p.add_argument("--d-model", type=int, default=1024)
    p.add_argument("--hidden-size", type=int, default=2048)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--run-backward", action="store_true")
    p.add_argument("--run-breakdown", action="store_true")
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


def _build_module(
    *,
    d_model: int,
    hidden_size: int,
    num_experts: int,
) -> RoutedExperts:
    module = RoutedExperts(
        d_model=d_model,
        hidden_size=hidden_size,
        num_experts=num_experts,
        bias=False,
        dtype=DType.bfloat16,
        rowwise_fp8=MoERowwiseFP8Config(enabled=True, block_size=32, use_fast_accum=True),
        init_device="cuda",
    )
    module.train()
    return module


def _require_runtime_support() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    if major < 10:
        raise RuntimeError(
            "Rowwise FP8 benchmark requires SM100+ for this implementation; "
            f"detected capability {major}.{minor}"
        )


def main() -> None:
    args = _parse_args()
    _require_runtime_support()

    device = torch.device("cuda")
    dtype = torch.bfloat16

    torch.manual_seed(0)

    total_tokens = args.num_experts * args.tokens_per_expert
    x = torch.randn(total_tokens, args.d_model, device=device, dtype=dtype)
    batch_size_per_expert = torch.full(
        (args.num_experts,),
        args.tokens_per_expert,
        device=device,
        dtype=torch.int32,
    )

    module = _build_module(
        d_model=args.d_model,
        hidden_size=args.hidden_size,
        num_experts=args.num_experts,
    )

    down_out = torch.empty((total_tokens, args.d_model), device=device, dtype=dtype)
    up_proj_input_grad_out = torch.empty_like(x)

    fwd_bf16_ms = _timed_ms(
        lambda: module(
            x,
            batch_size_per_expert,
            down_proj_out=down_out,
            up_proj_input_grad_out=up_proj_input_grad_out,
            use_rowwise_fp8=False,
        ),
        warmup=args.warmup,
        iters=args.iters,
    )

    fwd_fp8_ms = _timed_ms(
        lambda: module(
            x,
            batch_size_per_expert,
            down_proj_out=down_out,
            up_proj_input_grad_out=up_proj_input_grad_out,
            use_rowwise_fp8=True,
        ),
        warmup=args.warmup,
        iters=args.iters,
    )

    print("RoutedExperts Forward")
    print(f"  bf16 grouped_mm: {fwd_bf16_ms:.3f} ms")
    print(f"  fp8  path:       {fwd_fp8_ms:.3f} ms")
    print(f"  speedup:         {fwd_bf16_ms / max(fwd_fp8_ms, 1e-6):.3f}x")

    if args.run_breakdown:
        offs = torch.cumsum(batch_size_per_expert, dim=0, dtype=torch.int32)
        w_up_gate_t = module.w_up_gate.transpose(1, 2)
        w_down = module.w_down
        quant_a_ms = _timed_ms(
            lambda: quantize_grouped_2d_to_mxfp8_blocked(x, offs),
            warmup=args.warmup,
            iters=args.iters,
        )
        quant_w_up_ms = _timed_ms(
            lambda: quantize_grouped_weight_3d_to_mxfp8_blocked(w_up_gate_t),
            warmup=args.warmup,
            iters=args.iters,
        )
        quant_w_down_ms = _timed_ms(
            lambda: quantize_grouped_weight_3d_to_mxfp8_blocked(w_down),
            warmup=args.warmup,
            iters=args.iters,
        )
        print("FP8 Quantization Breakdown")
        print(f"  activation quant + scale pack: {quant_a_ms:.3f} ms")
        print(f"  up/gate weight quant + pack:   {quant_w_up_ms:.3f} ms")
        print(f"  down weight quant + pack:      {quant_w_down_ms:.3f} ms")

    if not args.run_backward:
        return

    x_bf16 = x.detach().clone().requires_grad_(True)
    x_fp8 = x.detach().clone().requires_grad_(True)
    grad_out = torch.randn((total_tokens, args.d_model), device=device, dtype=dtype)

    def _bwd_bf16():
        module.zero_grad(set_to_none=True)
        x_bf16.grad = None
        out = module(
            x_bf16,
            batch_size_per_expert,
            down_proj_out=None,
            up_proj_input_grad_out=None,
            use_rowwise_fp8=False,
        )
        out.backward(grad_out)

    def _bwd_fp8():
        module.zero_grad(set_to_none=True)
        x_fp8.grad = None
        out = module(
            x_fp8,
            batch_size_per_expert,
            down_proj_out=None,
            up_proj_input_grad_out=None,
            use_rowwise_fp8=True,
        )
        out.backward(grad_out)

    bwd_bf16_ms = _timed_ms(_bwd_bf16, warmup=args.warmup, iters=args.iters)
    bwd_fp8_ms = _timed_ms(_bwd_fp8, warmup=args.warmup, iters=args.iters)

    print("RoutedExperts Backward")
    print(f"  bf16 grouped_mm: {bwd_bf16_ms:.3f} ms")
    print(f"  fp8  path:       {bwd_fp8_ms:.3f} ms")
    print(f"  speedup:         {bwd_bf16_ms / max(bwd_fp8_ms, 1e-6):.3f}x")


if __name__ == "__main__":
    main()
