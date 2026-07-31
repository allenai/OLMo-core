"""
Microbenchmark for the fused gated RMS norm kernel used by GatedDeltaNet2
(:mod:`olmo_core.kernels.fused_rms_norm_gated`) against the original fla implementation and a
``torch.compile`` reference.

This script lives outside of ``src/test`` so it's not run as part of the CI test suite.
It requires a CUDA device. Example usage on a GPU node:

    python src/scripts/benchmark_fused_rms_norm_gated.py
    python src/scripts/benchmark_fused_rms_norm_gated.py --T 1048576 --sweep
    ncu -k "rms_norm_gated" --set full python src/scripts/benchmark_fused_rms_norm_gated.py \\
        --T 1048576 --impl vendored

The default row counts correspond to tokens-per-rank-microbatch x n_v_heads across the mainline
ladder sizes (D = head_v_dim = 128 everywhere).
"""

import argparse
import time
from typing import Callable, Dict, List, Optional

import torch

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}


def fwd_bytes(T: int, D: int, esize: int) -> int:
    # read x + g, write y, write rstd (fp32).
    return 3 * T * D * esize + 4 * T


def bwd_bytes(T: int, D: int, esize: int) -> int:
    # read x + g + dy, write dx + dg, read rstd (fp32). dw partials are noise.
    return 5 * T * D * esize + 4 * T


def make_inputs(T: int, D: int, dtype: torch.dtype, requires_grad: bool = False):
    x = torch.randn(T, D, dtype=dtype, device="cuda", requires_grad=requires_grad)
    g = torch.randn(T, D, dtype=dtype, device="cuda", requires_grad=requires_grad)
    w = torch.randn(D, dtype=dtype, device="cuda", requires_grad=requires_grad)
    dy = torch.randn(T, D, dtype=dtype, device="cuda")
    return x, g, w, dy


def eager_ref(x, g, w, activation: str, eps: float):
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if w is not None:
        y = y * w.float()
    gf = g.float()
    if activation in ("swish", "silu"):
        y = y * gf * torch.sigmoid(gf)
    else:
        y = y * torch.sigmoid(gf)
    return y.to(x.dtype)


def get_impls(activation: str, eps: float) -> Dict[str, Dict[str, Optional[Callable]]]:
    """
    Returns, per implementation, callables for 'fwd' (kernel wrapper only, where available),
    'bwd' (kernel wrapper only, where available), and 'autograd' (full fwd for use with
    ``.backward()``).
    """
    from fla.modules.fused_norm_gate import layer_norm_gated_bwd as fla_bwd
    from fla.modules.fused_norm_gate import layer_norm_gated_fwd as fla_fwd
    from fla.modules.fused_norm_gate import rms_norm_gated as fla_rms_norm_gated

    from olmo_core.kernels.fused_rms_norm_gated import (
        rms_norm_gated,
        rms_norm_gated_bwd,
        rms_norm_gated_fwd,
    )

    compiled_ref = torch.compile(eager_ref, mode="max-autotune-no-cudagraphs", dynamic=False)

    return {
        "fla": {
            "fwd": lambda x, g, w, dy: fla_fwd(
                x, g, w, None, activation=activation, eps=eps, is_rms_norm=True
            ),
            "bwd": lambda x, g, w, dy, rstd: fla_bwd(
                dy, x, g, w, None, activation=activation, eps=eps, rstd=rstd, is_rms_norm=True
            ),
            "autograd": lambda x, g, w: fla_rms_norm_gated(
                x, g, w, None, activation=activation, eps=eps
            ),
        },
        "vendored": {
            "fwd": lambda x, g, w, dy: rms_norm_gated_fwd(x, g, w, activation, eps),
            "bwd": lambda x, g, w, dy, rstd: rms_norm_gated_bwd(dy, x, g, w, rstd, activation),
            "autograd": lambda x, g, w: rms_norm_gated(x, g, w, activation=activation, eps=eps),
        },
        "compile": {
            "fwd": lambda x, g, w, dy: compiled_ref(x, g, w, activation, eps),
            "bwd": None,  # no direct-bwd entrypoint; covered by 'fwdbwd'
            "autograd": lambda x, g, w: compiled_ref(x, g, w, activation, eps),
        },
    }


def bench_first_call(fn: Callable) -> float:
    """Wall-clock of the very first call (captures triton compile + autotune cost)."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3  # ms


def bench(fn: Callable, grad_to_none: Optional[List[torch.Tensor]] = None) -> float:
    import triton.testing

    return triton.testing.do_bench(fn, grad_to_none=grad_to_none, return_mode="median")  # ms


def run_benchmarks(args) -> None:
    dtype = DTYPES[args.dtype]
    esize = torch.tensor([], dtype=dtype).element_size()
    impls = get_impls(args.activation, args.eps)
    if args.impl != "all":
        impls = {args.impl: impls[args.impl]}

    header = (
        f"{'T':>9} {'impl':>9} | {'fwd µs':>9} {'GB/s':>7} {'%pk':>5} | "
        f"{'bwd µs':>9} {'GB/s':>7} {'%pk':>5} | {'f+b µs':>9} {'GB/s':>7} {'%pk':>5} | "
        f"{'1st call ms':>11}"
    )
    print(f"\ndtype={args.dtype} D={args.D} activation={args.activation} eps={args.eps}")
    print(header)
    print("-" * len(header))

    for T in args.T:
        fb, bb = fwd_bytes(T, args.D, esize), bwd_bytes(T, args.D, esize)
        for name, impl in impls.items():
            x, g, w, dy = make_inputs(T, args.D, dtype, requires_grad=True)
            xd, gd, wd = x.detach(), g.detach(), w.detach()
            fwd_fn, bwd_fn, autograd_fn = impl["fwd"], impl["bwd"], impl["autograd"]
            assert autograd_fn is not None

            # First call (compile + autotune cost), on the autograd path.
            y = autograd_fn(x, g, w)
            first_ms = bench_first_call(lambda: y.backward(dy, retain_graph=True))
            first_ms += 0.0  # bwd first call only; fwd's was consumed building `y`

            # fwd-only (kernel wrapper, no autograd).
            fwd_us = fwd_gbps = fwd_pct = float("nan")
            if fwd_fn is not None:
                fwd_ms = bench(lambda: fwd_fn(xd, gd, wd, dy))
                fwd_us = fwd_ms * 1e3
                fwd_gbps = fb / (fwd_ms * 1e-3) / 1e9
                fwd_pct = 100 * fwd_gbps / args.peak_gbps

            # bwd-only (kernel wrapper, no autograd), vendored/fla only.
            bwd_us = bwd_gbps = bwd_pct = float("nan")
            if bwd_fn is not None and fwd_fn is not None:
                if name == "vendored":
                    _, rstd = fwd_fn(xd, gd, wd, dy)
                else:
                    _, _, rstd, _ = fwd_fn(xd, gd, wd, dy)
                bwd_ms = bench(lambda: bwd_fn(xd, gd, wd, dy, rstd))
                bwd_us = bwd_ms * 1e3
                bwd_gbps = bb / (bwd_ms * 1e-3) / 1e9
                bwd_pct = 100 * bwd_gbps / args.peak_gbps

            # fwd+bwd through autograd (includes python wrapper overhead).
            def fwdbwd():
                out = autograd_fn(x, g, w)
                out.backward(dy)

            fbd_ms = bench(fwdbwd, grad_to_none=[x, g, w])
            fbd_us = fbd_ms * 1e3
            fbd_gbps = (fb + bb) / (fbd_ms * 1e-3) / 1e9
            fbd_pct = 100 * fbd_gbps / args.peak_gbps

            print(
                f"{T:>9} {name:>9} | {fwd_us:>9.1f} {fwd_gbps:>7.0f} {fwd_pct:>5.1f} | "
                f"{bwd_us:>9.1f} {bwd_gbps:>7.0f} {bwd_pct:>5.1f} | "
                f"{fbd_us:>9.1f} {fbd_gbps:>7.0f} {fbd_pct:>5.1f} | {first_ms:>11.1f}"
            )
        print()


def _raw_jit_fn(kernel):
    """Unwrap Autotuner/Heuristics decorators down to the raw JITFunction."""
    while hasattr(kernel, "fn"):
        kernel = kernel.fn
    return kernel


def run_sweep(args) -> None:
    """
    Sweep launch configs for the vendored kernels directly (bypassing @triton.autotune) to pick
    a fixed config for this GPU.
    """
    import math

    import triton

    from olmo_core.kernels import fused_rms_norm_gated as k

    dtype = DTYPES[args.dtype]
    esize = torch.tensor([], dtype=dtype).element_size()
    fwd_fn = _raw_jit_fn(k.rms_norm_gated_fwd_kernel)
    bwd_fn = _raw_jit_fn(k.rms_norm_gated_bwd_kernel)
    D = args.D
    BD = triton.next_power_of_2(D)

    print(
        f"\nsweep: dtype={args.dtype} D={D} activation={args.activation} "
        f"device={torch.cuda.get_device_name()}"
    )
    for T in args.T:
        x, g, w, dy = make_inputs(T, D, dtype)
        y, rstd = torch.empty_like(x), torch.empty(T, dtype=torch.float, device="cuda")
        dx, dg = torch.empty_like(x), torch.empty_like(g)
        NS = min(torch.cuda.get_device_properties(x.device.index).multi_processor_count, T)
        BS = math.ceil(T / NS)
        dw = torch.empty(NS, D, dtype=torch.float, device="cuda")
        fb, bb = fwd_bytes(T, D, esize), bwd_bytes(T, D, esize)

        results = []
        for BT in args.sweep_bt:
            for num_warps in args.sweep_warps:
                for num_stages in args.sweep_stages:

                    def run_fwd():
                        fwd_fn[(triton.cdiv(T, BT),)](
                            x=x, g=g, y=y, w=w, rstd=rstd, eps=args.eps, T=T,
                            D=D, BT=BT, BD=BD, NB=0, ACTIVATION=args.activation,
                            HAS_WEIGHT=True, num_warps=num_warps, num_stages=num_stages,
                        )  # fmt: skip

                    def run_bwd():
                        bwd_fn[(NS,)](
                            x=x, g=g, w=w, dy=dy, dx=dx, dg=dg, dw=dw, rstd=rstd, T=T, BS=BS,
                            D=D, BT=BT, BD=BD, NB=0, ACTIVATION=args.activation,
                            HAS_WEIGHT=True, num_warps=num_warps, num_stages=num_stages,
                        )  # fmt: skip

                    try:
                        fwd_ms, bwd_ms = bench(run_fwd), bench(run_bwd)
                    except Exception as e:  # e.g. resource limits
                        print(f"  BT={BT} warps={num_warps} stages={num_stages}: FAILED ({e})")
                        continue
                    results.append((fwd_ms, bwd_ms, BT, num_warps, num_stages))

        print(f"\nT={T}: (top 5 by fwd, then by bwd)")
        for fwd_ms, bwd_ms, BT, nw, ns in sorted(results)[:5]:
            print(
                f"  fwd BT={BT:>3} warps={nw} stages={ns}: {fwd_ms * 1e3:>8.1f} µs "
                f"({fb / (fwd_ms * 1e-3) / 1e9:>6.0f} GB/s)"
            )
        for fwd_ms, bwd_ms, BT, nw, ns in sorted(results, key=lambda r: r[1])[:5]:
            print(
                f"  bwd BT={BT:>3} warps={nw} stages={ns}: {bwd_ms * 1e3:>8.1f} µs "
                f"({bb / (bwd_ms * 1e-3) / 1e9:>6.0f} GB/s)"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--T", type=int, nargs="+", default=[262144, 524288, 1048576, 2097152],
        help="row counts (tokens x n_v_heads)",
    )  # fmt: skip
    parser.add_argument("--D", type=int, default=128, help="feature dim (head_v_dim)")
    parser.add_argument("--dtype", choices=list(DTYPES), default="bf16")
    parser.add_argument("--activation", choices=["sigmoid", "swish"], default="sigmoid")
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument(
        "--peak-gbps", type=float, default=8000, help="peak memory bandwidth (B300 ~8000)"
    )
    parser.add_argument("--impl", choices=["all", "fla", "vendored", "compile"], default="all")
    parser.add_argument("--sweep", action="store_true", help="sweep vendored kernel launch configs")
    parser.add_argument("--sweep-bt", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--sweep-warps", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--sweep-stages", type=int, nargs="+", default=[2, 3, 4])
    args = parser.parse_args()

    assert torch.cuda.is_available(), "this benchmark requires a CUDA device"
    torch.manual_seed(0)
    print(f"device: {torch.cuda.get_device_name()}")

    if args.sweep:
        run_sweep(args)
    else:
        run_benchmarks(args)


if __name__ == "__main__":
    main()
