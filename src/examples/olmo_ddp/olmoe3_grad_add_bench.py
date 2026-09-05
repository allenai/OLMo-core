"""Qualify vectorized BF16-to-FP32 gradient addition; benchmark-only, not in training."""

import json
import os
import statistics
from functools import partial
from pathlib import Path

import torch

from olmo_core.ops.grad_accum import gradient_add


def qualify():
    """Exact comparisons include tails, empty tensors, repeated accumulation, and nonfinites."""
    torch.manual_seed(731)
    for elements in (0, 1, 129, 70001):
        source = torch.randn(elements, device="cuda", dtype=torch.bfloat16)
        initial = torch.randn(elements, device="cuda", dtype=torch.float32)
        if elements > 5:
            source[:5] = torch.tensor(
                [0.0, float("inf"), -float("inf"), float("nan"), 1e30],
                device="cuda",
                dtype=torch.bfloat16,
            )
            initial[:5] = torch.tensor(
                [-0.0, -float("inf"), float("inf"), 0.0, -1e30], device="cuda"
            )
        for block, warps in ((2048, 4), (4096, 4), (8192, 8)):
            actual, expected = initial.clone(), initial.clone()
            for _ in range(8):
                gradient_add(actual, source, block, warps)
                expected.add_(source)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    print("GRAD_ADD_QUALIFICATION passed", flush=True)


def benchmark():
    """Isolated real expert-weight sizes, old/new/old, CUDA-event timing only."""
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    qualify()
    summary = {
        "source_commit": os.environ.get("GIT_REF"),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "qualified": True,
        "arms": [],
        "caveat": "Isolated contiguous gradient additions, not a whole-training speedup. Model integration and actual runtime layout remain unqualified.",
    }
    for shape in ((512, 1024, 512), (512, 2048, 512)):
        source = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        destination = torch.zeros(shape, device="cuda", dtype=torch.float32)
        expected = torch.zeros_like(destination)
        for _ in range(25):
            expected.add_(source)
        for label, settings in (
            ("old-before", None),
            ("triton-2048-w4", (2048, 4)),
            ("triton-4096-w4", (4096, 4)),
            ("triton-8192-w8", (8192, 8)),
            ("old-after", None),
        ):
            destination.zero_()

            execute = (
                partial(destination.add_, source)
                if settings is None
                else partial(gradient_add, destination, source, *settings)
            )

            for _ in range(5):
                execute()
            torch.cuda.synchronize()
            timings = []
            for _ in range(20):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                start.record()
                execute()
                end.record()
                timings.append((start, end))
            torch.cuda.synchronize()
            durations = [start.elapsed_time(end) for start, end in timings]
            torch.testing.assert_close(destination, expected, rtol=0, atol=0)
            median = statistics.median(durations)
            row = {
                "implementation": label,
                "shape": shape,
                "median_ms": median,
                "mean_ms": statistics.mean(durations),
                "min_ms": min(durations),
                "max_ms": max(durations),
                "logical_TB_per_second": source.numel() * 10 / (median / 1000) / 1e12,
                "exact_after_25_additions": True,
            }
            summary["arms"].append(row)
            print("GRAD_ADD_BENCHMARK", json.dumps(row), flush=True)
        del source, destination, expected, execute
    Path("/results/grad-add-benchmark.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    benchmark()
