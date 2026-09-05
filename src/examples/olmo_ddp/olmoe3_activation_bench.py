"""Benchmark the existing expert activation and compiler tuning without changing its math."""

import json
import os
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from torch._inductor.utils import run_and_get_code


def activation(up_gate):
    """Match the BF16 autograd training branch of RoutedExperts.chunk_and_activate."""
    up, gate = up_gate.chunk(2, dim=-1)
    return up * F.silu(gate)


def summarize(values):
    """Keep the complete distribution summary, not just the fastest iteration."""
    return {
        "median_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "samples": len(values),
    }


def main():
    """Use real activation dimensions with synthetic inputs; no model integration or training."""
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(731)
    output = Path(os.environ.get("RESULTS_DIR", "/results")) / "activation-probe"
    output.mkdir(parents=True, exist_ok=True)
    rows, hidden = 524288, 1024  # 4 x 8192 tokens x top16, expert hidden=1024.
    x = torch.randn(rows, 2 * hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    # Include a moderate activation range without overflowing the synthetic BF16 gradients.
    with torch.no_grad():
        x[::3].mul_(5)
        x[1::3].mul_(0.1)
    reference_y = reference_dx = None
    results = []
    for label, options in (
        ("default-before", {}),
        (
            "pointwise-coordinate-tuning",
            {"max_autotune_pointwise": True, "coordinate_descent_tuning": True},
        ),
        ("default-after", {}),
    ):
        compiled = torch.compile(activation, fullgraph=True, dynamic=False, options=options)

        def execute():
            x.grad = None
            y = compiled(x)
            y.backward(dy)
            return y

        y, source_codes = run_and_get_code(execute)
        torch.cuda.synchronize()
        for index, source in enumerate(source_codes):
            (output / f"{label}-generated-{index}.py").write_text(source)
        if reference_y is None:
            reference_y = y.detach().clone()
            reference_dx = x.grad.detach().clone()
        else:
            torch.testing.assert_close(y, reference_y, rtol=0, atol=0)
            torch.testing.assert_close(x.grad, reference_dx, rtol=0, atol=0)
        for _ in range(5):
            execute()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        events = []
        for _ in range(20):
            x.grad = None
            start = torch.cuda.Event(enable_timing=True)
            middle = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            y = compiled(x)
            middle.record()
            y.backward(dy)
            end.record()
            events.append((start, middle, end))
        torch.cuda.synchronize()
        torch.testing.assert_close(y, reference_y, rtol=0, atol=0)
        torch.testing.assert_close(x.grad, reference_dx, rtol=0, atol=0)
        row = {
            "arm": label,
            "options": options,
            "exact_outputs_and_gradients": True,
            "forward": summarize([a.elapsed_time(b) for a, b, c in events]),
            "backward": summarize([b.elapsed_time(c) for a, b, c in events]),
            "forward_backward": summarize([a.elapsed_time(c) for a, b, c in events]),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "generated_source_files": len(source_codes),
        }
        results.append(row)
        print("ACTIVATION_BENCHMARK", json.dumps(row), flush=True)
        (output / "benchmark.json").write_text(
            json.dumps(
                {
                    "git_commit": os.environ.get("GIT_REF"),
                    "gpu": torch.cuda.get_device_name(0),
                    "torch": torch.__version__,
                    "input_shape": list(x.shape),
                    "input_stride": list(x.stride()),
                    "dtype": str(x.dtype),
                    "arms": results,
                    "caveat": "Isolated synthetic activation/gradient inputs at the real expert shape. Not trained-routing or full-model throughput. Exact comparison is against default compiled BF16 autograd, not a different FP32 formula.",
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
