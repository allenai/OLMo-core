"""Bounded, one-GPU BF16 expert GEMM probe; never imported by training.

Compare Torch with three explicit QuACK 0.5.0 tiles, without autotuning or
epilogue/precision changes. Synthetic routing is NOT a trained routing replay.
Forward, input gradients, and weight gradients are separate cases. Each case
has its own process so a rejected configuration cannot poison another case.
"""

import argparse
import importlib.metadata
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

OPERATIONS = ("up-forward", "up-dgrad", "up-wgrad", "down-forward", "down-dgrad", "down-wgrad")
TILES = ((256, 256, 2, 1), (128, 128, 1, 1), (128, 256, 1, 1))


def run_case(operation, routing, output):
    import torch
    import torch.nn.functional as F
    from quack.gemm import gemm

    assert importlib.metadata.version("quack-kernels") == "0.5.0"
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(20260905)
    assert torch.cuda.get_device_capability(0)[0] == 10

    # Same total rows, expert count, and projection widths as MB4 / EP1.
    # Skew includes empty experts and 3.77x mean load; offsets stay 16-aligned.
    counts = [1024] * 512 if routing == "uniform" else [0, 80, 160, 3856] * 128
    assert sum(counts) == 524288
    edges = [0]
    for count in counts:
        edges.append(edges[-1] + count)
    cu = torch.tensor(edges, device="cuda", dtype=torch.int32)
    offs = cu[1:]
    width_in, width_out = (512, 2048) if operation.startswith("up") else (1024, 512)

    def rand(*shape):
        return torch.randn(shape, device="cuda", dtype=torch.bfloat16).mul_(0.1)

    # Public QuACK GEMM takes B transposed relative to torch.grouped_mm.
    if operation.endswith("wgrad"):
        a = rand(524288, width_out).T
        b = rand(524288, width_in)
        out_shape = (512, width_out, width_in)
        sequence_arg = {"cu_seqlens_k": cu}
    else:
        rows_in, rows_out = (
            (width_out, width_in) if operation.endswith("dgrad") else (width_in, width_out)
        )
        a = rand(524288, rows_in)
        # Dgrad reuses row-major original weights, without an extra packing copy.
        b = (
            rand(512, width_out, width_in)
            if operation.endswith("dgrad")
            else rand(512, rows_out, rows_in).mT
        )
        out_shape = (524288, rows_out)
        sequence_arg = {"cu_seqlens_m": cu}

    def native():
        return F.grouped_mm(a, b, offs=offs)

    # Allocate output on every call for both backends, matching the unbuffered
    # training path. Compilation, input construction, and checks are not timed.
    def candidate(tile):
        out = torch.empty(out_shape, device="cuda", dtype=torch.bfloat16)
        gemm(
            a,
            b.mT,
            out,
            None,
            None,
            *tile,
            pingpong=False,
            persistent=True,
            is_dynamic_persistent=True,
            **sequence_arg,
        )
        return out

    expected = native()
    assert bool(torch.isfinite(expected).all())

    # Independent FP64 oracle on slices from empty, light, heavy, and last experts.
    # Wgrad slices retain the FULL reduction dimension for each sampled expert.
    def oracle_error(actual):
        summaries = []
        for expert in (0, 1, 2, 3, 255, 511):
            start, end = edges[expert : expert + 2]
            if operation.endswith("wgrad"):
                oracle = a[:8, start:end].double() @ b[start:end, :16].double()
                found = actual[expert, :8, :16].double()
            else:
                stop = min(start + 8, end)
                oracle = a[start:stop].double() @ b[expert, :, :16].double()
                found = actual[start:stop, :16].double()
            if found.numel():
                torch.testing.assert_close(found, oracle, rtol=0.01, atol=0.001)
                summaries.append(float((found - oracle).abs().max()))
        return max(summaries)

    native_oracle = oracle_error(expected)

    def parity(actual):
        # Initial screening bounds, not a promotion gate for model training.
        # Chunk checking avoids multi-GiB FP32 diagnostic temporaries.
        ref, got = expected.reshape(-1), actual.reshape(-1)
        mismatches, max_abs, squared_error, squared_ref = 0, 0.0, 0.0, 0.0
        for start in range(0, ref.numel(), 4 * 1024 * 1024):
            r = ref[start : start + 4 * 1024 * 1024].float()
            g = got[start : start + 4 * 1024 * 1024].float()
            torch.testing.assert_close(g, r, rtol=0.01, atol=0.001)
            difference = g - r
            mismatches += int((g != r).sum())
            max_abs = max(max_abs, float(difference.abs().max()))
            squared_error += float(difference.double().square().sum())
            squared_ref += float(r.double().square().sum())
        return {
            "mismatched_elements": mismatches,
            "elements": ref.numel(),
            "max_abs_difference": max_abs,
            "relative_rms_difference": (squared_error / squared_ref) ** 0.5,
            "fp64_sample_max_abs_difference": oracle_error(actual),
        }

    def benchmark(fn):
        for _ in range(5):
            _result = fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(25):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            _result = fn()
            end.record()
            samples.append((start, end))
        torch.cuda.synchronize()
        times = [start.elapsed_time(end) for start, end in samples]
        return {"median_ms": statistics.median(times), "mean_ms": statistics.mean(times)}

    summary = {
        "source_commit": os.environ.get("GIT_REF"),
        "quack_version": importlib.metadata.version("quack-kernels"),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "operation": operation,
        "routing": routing,
        "counts": counts,
        "a_shape": list(a.shape),
        "a_stride": list(a.stride()),
        "b_shape": list(b.shape),
        "b_stride": list(b.stride()),
        "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "native_fp64_sample_max_abs_difference": native_oracle,
        "cases": [],
        "caveat": "Synthetic routing, isolated kernels, L2-warm repeat. No end-to-end gain or training qualification implied.",
    }
    summary["cases"].append({"implementation": "torch-before", **benchmark(native)})
    for tile in TILES:
        actual = candidate(tile)
        correctness = parity(actual)
        del actual
        row = {"implementation": "quack", "tile": tile, **correctness}
        row.update(benchmark(lambda: candidate(tile)))
        summary["cases"].append(row)
        print("GROUPED_GEMM_RESULT", json.dumps(row), flush=True)
        output.write_text(json.dumps(summary, indent=2))
    summary["cases"].append({"implementation": "torch-after", **benchmark(native)})
    output.write_text(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", choices=OPERATIONS)
    parser.add_argument("--routing", choices=("uniform", "skew"))
    parser.add_argument("--output-dir", type=Path, default=Path("/results/grouped-gemm"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.operation:
        assert args.routing is not None
        run_case(
            args.operation, args.routing, args.output_dir / f"{args.operation}-{args.routing}.json"
        )
        return
    cases = []
    for operation in OPERATIONS:
        for routing in ("uniform", "skew"):
            print("GROUPED_GEMM_CASE", operation, routing, flush=True)
            command = [
                sys.executable,
                "-u",
                __file__,
                "--operation",
                operation,
                "--routing",
                routing,
                "--output-dir",
                str(args.output_dir),
            ]
            try:
                result = subprocess.run(command, timeout=900, check=False)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
            cases.append({"operation": operation, "routing": routing, "exit_code": code})
            (args.output_dir / "cases.json").write_text(json.dumps(cases, indent=2))
    if any(case["exit_code"] != 0 for case in cases):
        raise RuntimeError(
            "At least one GEMM case failed; inspect logs and cases.json before promotion"
        )


if __name__ == "__main__":
    main()
