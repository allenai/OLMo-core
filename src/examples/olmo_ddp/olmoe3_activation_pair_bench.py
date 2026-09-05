"""Qualify paired activation gradients against the actual compiled BF16 autograd result."""

import json
import os
import statistics
from pathlib import Path

import torch
from olmoe3_activation_bench import activation

from olmo_core.ops.swiglu_pairwise import swiglu_backward_pair


def compare(actual, expected):
    """Report mismatch counts before enforcing zero numerical tolerance."""
    equal = (actual == expected) | (torch.isnan(actual) & torch.isnan(expected))
    count = int((~equal).sum().item())
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    diff = torch.where(finite, (actual.float() - expected.float()).abs(), 0)
    summary = {
        "mismatched_elements": count,
        "elements": actual.numel(),
        "max_abs_finite_difference": float(diff.max().item()) if diff.numel() else 0,
    }
    print("ACTIVATION_PAIR_PARITY", json.dumps(summary), flush=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    return summary


def time_native_backward(compiled, x, dy, label):
    """Bracket candidate timings with the unchanged compiled backward on the same GPU."""
    events = []
    for index in range(25):
        x.grad = None
        y = compiled(x)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        y.backward(dy)
        end.record()
        if index >= 5:
            events.append((start, end))
    torch.cuda.synchronize()
    durations = [a.elapsed_time(b) for a, b in events]
    row = {
        "implementation": label,
        "shape": list(x.shape),
        "median_ms": statistics.median(durations),
        "mean_ms": statistics.mean(durations),
    }
    print("ACTIVATION_PAIR_BENCHMARK", json.dumps(row), flush=True)
    return row


def main():
    """One-GPU exact tests then a real-shape backward benchmark; no training changes."""
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(913)
    results = Path(os.environ.get("RESULTS_DIR", "/results")) / "activation-pair"
    results.mkdir(parents=True, exist_ok=True)
    compiled = torch.compile(activation, fullgraph=True, dynamic=False)
    cases = []
    for rows, hidden in ((0, 1024), (1, 128), (7, 129), (33, 512), (65, 1024), (524288, 1024)):
        x = torch.randn(rows, 2 * hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        dy = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            x[::3].mul_(5)
            x[1::3].mul_(0.1)
            if rows and rows < 100:
                x[0, hidden : hidden + 8] = torch.tensor(
                    [0.0, -0.0, 80.0, -80.0, float("inf"), -float("inf"), float("nan"), 1e-30],
                    dtype=x.dtype,
                    device=x.device,
                )
        y = compiled(x)
        y.backward(dy)
        expected = x.grad
        timings = []
        if rows == 524288:
            timings.append(time_native_backward(compiled, x, dy, "native-before"))
        for block, warps in ((1024, 4), (2048, 4), (4096, 4), (4096, 8)):
            actual = swiglu_backward_pair(x, dy, block=block, warps=warps)
            parity = compare(actual, expected)
            row = {"shape": list(x.shape), "block": block, "warps": warps, **parity}
            if rows == 524288:
                for _ in range(5):
                    actual = swiglu_backward_pair(x, dy, block=block, warps=warps)
                torch.cuda.synchronize()
                events = []
                for _ in range(20):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                        enable_timing=True
                    )
                    start.record()
                    actual = swiglu_backward_pair(x, dy, block=block, warps=warps)
                    end.record()
                    events.append((start, end))
                torch.cuda.synchronize()
                durations = [a.elapsed_time(b) for a, b in events]
                row.update(
                    median_ms=statistics.median(durations), mean_ms=statistics.mean(durations)
                )
                compare(actual, expected)
            timings.append(row)
            print("ACTIVATION_PAIR_BENCHMARK", json.dumps(row), flush=True)
        if rows == 524288:
            timings.append(time_native_backward(compiled, x, dy, "native-after"))
            compare(x.grad, expected)
        cases.extend(timings)
        (results / "summary.json").write_text(
            json.dumps(
                {
                    "source_commit": os.environ.get("GIT_REF"),
                    "torch": torch.__version__,
                    "gpu": torch.cuda.get_device_name(0),
                    "cases": cases,
                    "caveat": "Synthetic inputs and exact comparisons with compiled, not eager, BF16 autograd. Not model-integrated.",
                },
                indent=2,
            )
        )
        del x, dy, y, expected, actual


if __name__ == "__main__":
    main()
