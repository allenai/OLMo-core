"""Qualify the two remaining KDA scan cutoffs at the exact production shape.

Benchmark-only process-local overrides. No package or training defaults change.
The current optimized chain remains the reference, including FP32 gates, expand_v=2,
negative eigenvalues, and L2-normalized Q/K. Report all input gradients; full-layer
optimizer qualification is a separate gate if any candidate improves latency.
"""

import json
import os
import statistics
from pathlib import Path

import torch


def main():
    """Compare B1, B2a, and combined overrides against bracketed reference timings."""
    from kernel_fun._common import support
    from kernel_fun.kda import chunk_kda
    from kernel_fun.kda._kernels import bwd_dhu, bwd_scan

    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(20260905)
    support.MIN_CTAS = 128
    path = Path("/results/kda-scan")
    path.mkdir(parents=True, exist_ok=True)
    report = {
        "source": os.environ.get("GIT_REF"),
        "gpu": torch.cuda.get_device_name(),
        "q_shape": [4, 8192, 8, 128],
        "v_shape": [4, 8192, 8, 256],
        "cases": [],
    }

    def set_mode(mode):
        bwd_scan._MIN_CTAS = 128 if mode in ("scan", "both") else 256
        bwd_dhu._MIN_CTAS = 128 if mode in ("dhu", "both") else 256

    for strength in (0.1, 16.0):
        shape = (4, 8192, 8, 128)
        q = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn(4, 8192, 8, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        g = (-torch.rand(shape, device="cuda") * strength).requires_grad_()
        beta = (2 * torch.rand(4, 8192, 8, device="cuda")).requires_grad_()
        inputs = (q, k, v, g, beta)
        dy = torch.randn_like(v)

        def execute(q=q, k=k, v=v, g=g, beta=beta, inputs=inputs, dy=dy):
            y, _ = chunk_kda(q=q, k=k, v=v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
            return (y, *torch.autograd.grad(y, inputs, dy))

        def measure(execute=execute):
            for _ in range(4):
                execute()
            torch.cuda.synchronize()
            times = []
            for _ in range(20):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                execute()
                b.record()
                b.synchronize()
                times.append(a.elapsed_time(b))
            return {"median_ms": statistics.median(times), "mean_ms": statistics.mean(times)}

        set_mode("baseline")
        expected = tuple(x.detach().clone() for x in execute())
        for mode in ("baseline", "scan", "dhu", "both", "baseline"):
            set_mode(mode)
            actual = execute()
            errors = []
            for name, ref, got in zip(("y", "dq", "dk", "dv", "dg", "dbeta"), expected, actual):
                # Chunk to avoid unnecessary full-tensor FP64 diagnostic memory.
                delta = got.float() - ref.float()
                errors.append(
                    {
                        "name": name,
                        "finite": bool(torch.isfinite(got).all()),
                        "max_abs": float(delta.abs().max()),
                        "relative_l2": float(delta.norm() / ref.float().norm().clamp_min(1e-20)),
                        "mismatches": int((ref != got).sum()),
                    }
                )
            row = {"mode": mode, "decay_strength": strength, "errors": errors, **measure()}
            report["cases"].append(row)
            (path / "summary.json").write_text(json.dumps(report, indent=2))
            print("KDA_SCAN_RESULT", json.dumps(row), flush=True)
        del expected, actual, inputs, q, k, v, g, beta, dy
    set_mode("baseline")


if __name__ == "__main__":
    main()
