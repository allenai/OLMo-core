"""Bounded launch-tiling screen for the already-qualified SwiGLU backward kernel.

No training flags are changed here. Arithmetic, libdevice exp and FP fusion stay
exactly as in the existing kernel; full BF16 output equality is a mandatory gate.
"""

import json
import os
import statistics
from pathlib import Path

import torch

from olmo_core.ops.swiglu_pairwise import swiglu_backward_pair


def measure(fn):
    """Measure the complete allocating wrapper, bracketing candidates with baseline."""
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(40):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return {"median_ms": statistics.median(samples), "mean_ms": statistics.mean(samples)}


def main():
    """Screen the production 32768*top16 rows and hidden1024 on one B300."""
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(20260905)
    assert torch.cuda.get_device_capability(0) == (10, 3)
    output = Path("/results/swiglu-tile")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "source": os.environ.get("GIT_REF"),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "rows": 32768 * 16,
        "hidden": 1024,
        "cases": [],
        "caveat": "Standalone launch-tiling screen, not end-to-end training qualification",
    }
    configurations = [
        (256, 4),
        (512, 4),
        (1024, 4),
        (2048, 4),
        (4096, 4),
        (2048, 8),
        (4096, 8),
        (8192, 8),
        (8192, 16),
        (16384, 16),
    ]
    for scale in (1.0, 4.0):
        x = torch.randn(32768 * 16, 2048, device="cuda", dtype=torch.bfloat16)
        x.mul_(scale)
        dy = torch.randn(32768 * 16, 1024, device="cuda", dtype=torch.bfloat16)
        expected = swiglu_backward_pair(x, dy)
        assert torch.isfinite(expected).all()
        row = {
            "input_scale": scale,
            "baseline_before": measure(lambda: swiglu_backward_pair(x, dy)),
            "candidates": [],
        }
        for block, warps in configurations:
            found = swiglu_backward_pair(x, dy, block=block, warps=warps)
            exact = torch.equal(expected, found)
            del found
            result = {"block": block, "warps": warps, "exact_output": exact}
            if exact:
                result.update(
                    measure(lambda: swiglu_backward_pair(x, dy, block=block, warps=warps))
                )
            row["candidates"].append(result)
            print("SWIGLU_TILE", json.dumps({"input_scale": scale, **result}), flush=True)
        row["baseline_after"] = measure(lambda: swiglu_backward_pair(x, dy))
        report["cases"].append(row)
        (output / "summary.json").write_text(json.dumps(report, indent=2))
        del expected, dy, x
    if not all(c["exact_output"] for row in report["cases"] for c in row["candidates"]):
        raise RuntimeError("Some launch settings changed BF16 output; do not promote them")


if __name__ == "__main__":
    main()
