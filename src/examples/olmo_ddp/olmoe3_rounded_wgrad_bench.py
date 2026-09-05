"""Isolated BF16-rounded GEMM epilogue probe, never imported by training.

Preserve GEMM -> BF16 store rounding -> FP32 gradient add. Patch QuACK's class
only in this private process before its first FP32-output/C GEMM compilation.
No parameter/gradient buffer from a training run is read or modified.
"""

import argparse
import importlib
import importlib.metadata
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


def worker(projection, routing, output):
    """Check every FP32 accumulator element across eight updates, then time it."""
    import cutlass
    import cutlass.cute as cute
    import torch
    import torch.nn.functional as F
    from quack.gemm_default_epi import GemmDefaultEpiMixin, GemmDefaultSm100

    from olmo_core.ops.grad_accum import gradient_add

    assert importlib.metadata.version("quack-kernels") == "0.5.0"
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(482)

    class RoundedGradientGemm(GemmDefaultSm100):
        @cute.jit
        def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
            # The BF16 round-trip is intentional and must precede the C addition.
            tRS_rD.store(tRS_rD.load().to(cutlass.BFloat16).to(cutlass.Float32))
            return GemmDefaultEpiMixin.epi_visit_subtile(
                self, params, epi_loop_tensors, tRS_rD, tRS_rC
            )

    gemm_module = importlib.import_module("quack.gemm")
    gemm_module.GemmDefaultSm100 = RoundedGradientGemm
    counts = [1024] * 512 if routing == "uniform" else [0, 80, 160, 3856] * 128
    edges = [0]
    for count in counts:
        edges.append(edges[-1] + count)
    cu = torch.tensor(edges, device="cuda", dtype=torch.int32)
    width_in, width_out = (512, 2048) if projection == "up" else (1024, 512)
    a = torch.randn(524288, width_out, device="cuda", dtype=torch.bfloat16).mul_(0.1).T
    b = torch.randn(524288, width_in, device="cuda", dtype=torch.bfloat16).mul_(0.1)
    shape = (512, width_out, width_in)
    native_gradient = torch.zeros(shape, device="cuda", dtype=torch.float32)
    fused_gradient = torch.zeros_like(native_gradient)

    def native():
        gradient = F.grouped_mm(a, b, offs=cu[1:])
        gradient_add(native_gradient, gradient)

    def fused():
        gemm_module.gemm(
            a,
            b.T,
            fused_gradient,
            fused_gradient,
            None,
            256,
            256,
            2,
            1,
            pingpong=False,
            persistent=True,
            is_dynamic_persistent=True,
            cu_seqlens_k=cu,
        )

    def exact():
        left, right = native_gradient.reshape(-1), fused_gradient.reshape(-1)
        mismatch = 0
        for start in range(0, left.numel(), 4 * 1024 * 1024):
            x, y = left[start : start + 4 * 1024 * 1024], right[start : start + 4 * 1024 * 1024]
            mismatch += int((x != y).sum())
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        return mismatch

    for step in range(8):
        native()
        fused()
        print("ROUNDED_WGRAD_PARITY", projection, routing, step, exact(), flush=True)

    def timing(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        events = []
        for _ in range(25):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            events.append((start, end))
        torch.cuda.synchronize()
        values = [start.elapsed_time(end) for start, end in events]
        return {"median_ms": statistics.median(values), "mean_ms": statistics.mean(values)}

    summary = {
        "source_commit": os.environ.get("GIT_REF"),
        "projection": projection,
        "routing": routing,
        "shape": shape,
        "exact_accumulation_updates": 8,
        "torch_plus_vectorized_add_before": timing(native),
        "bf16_rounded_fused": timing(fused),
        "torch_plus_vectorized_add_after": timing(native),
        "caveat": "Synthetic inputs/routing and standalone accumulation only; no autograd/DDP lifetime qualification or training gain implied.",
    }
    output.write_text(json.dumps(summary, indent=2))
    print("ROUNDED_WGRAD_RESULT", json.dumps(summary), flush=True)


def main():
    """Each case uses a fresh process and compilation cache directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--projection", choices=("up", "down"))
    parser.add_argument("--routing", choices=("uniform", "skew"))
    parser.add_argument("--output-dir", type=Path, default=Path("/results/rounded-wgrad"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.projection:
        worker(
            args.projection,
            args.routing,
            args.output_dir / f"{args.projection}-{args.routing}.json",
        )
        return
    cases = []
    for projection in ("up", "down"):
        for routing in ("uniform", "skew"):
            command = [
                sys.executable,
                "-u",
                __file__,
                "--projection",
                projection,
                "--routing",
                routing,
            ]
            try:
                code = subprocess.run(command, timeout=600, check=False).returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
            cases.append({"projection": projection, "routing": routing, "exit_code": code})
            (args.output_dir / "cases.json").write_text(json.dumps(cases, indent=2))
    if any(case["exit_code"] != 0 for case in cases):
        raise RuntimeError("Rounded-gradient probe failed; do not integrate")


if __name__ == "__main__":
    main()
