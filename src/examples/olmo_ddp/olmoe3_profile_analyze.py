"""Summarize clean throughput windows and GPU interval unions (not additive kernel sums)."""

import argparse
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path


def union_duration(intervals):
    """Return the duration of merged intervals, in their original time unit."""
    end = float("-inf")
    total = 0.0
    for start, stop in sorted(intervals):
        if stop > end:
            total += stop - max(start, end)
            end = stop
    return total


def family(name):
    """Conservative name-based attribution; retain raw top kernels alongside categories."""
    name = name.lower()
    if "nccl" in name:
        return "collective"
    if any(key in name for key in ("kda", "wy_dqkg", "intra_bwd", "fwd_state", "bwd_dhu")):
        return "KDA"
    if any(key in name for key in ("conv1d", "cconv", "causal_conv")):
        return "short convolution"
    if any(key in name for key in ("grouped", "group_gemm", "groupgemm")):
        return "grouped GEMM"
    if any(key in name for key in ("gemm", "matmul", "cutlass")):
        return "GEMM / unresolved cutlass"
    if any(key in name for key in ("permute", "sort", "histogram", "router", "scatter", "gather")):
        return "routing / packing / indexing"
    if any(key in name for key in ("adam", "multi_tensor", "foreach")):
        return "optimizer / tensor-list operations"
    if "flash" in name or "attention" in name:
        return "full attention"
    if "norm" in name:
        return "normalization"
    return "other / unresolved"


def summarize_trace(path):
    """Report union time and sums separately; communication-only time is not causal attribution."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        events = json.load(handle)["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("dur", 0) > 0]
    grouped = defaultdict(float)
    names = defaultdict(float)
    compute, collective = [], []
    for event in kernels:
        kind = family(event["name"])
        grouped[kind] += event["dur"]
        names[event["name"]] += event["dur"]
        interval = (event["ts"], event["ts"] + event["dur"])
        (collective if kind == "collective" else compute).append(interval)
    busy = union_duration(compute + collective)
    compute_busy = union_duration(compute)
    return {
        "trace": str(path),
        "kernel_count": len(kernels),
        "gpu_kernel_busy_union_ms": busy / 1000,
        "collective_union_ms": union_duration(collective) / 1000,
        "collective_without_other_kernel_ms": (busy - compute_busy) / 1000,
        "kernel_duration_sums_ms_nonadditive": {
            k: v / 1000 for k, v in sorted(grouped.items(), key=lambda p: -p[1])
        },
        "top_kernels_by_duration_sum_ms": [
            (name, us / 1000) for name, us in sorted(names.items(), key=lambda p: -p[1])[:35]
        ],
        "caveat": "Kernel durations overlap. Communication-only union is not a measured counterfactual speedup or a complete critical-path analysis.",
    }


def main():
    """Analyze a completed pass directory without importing the training stack."""
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--traces", action="store_true")
    args = parser.parse_args()
    provenance = json.loads((args.run_dir / "provenance.json").read_text())
    # Exact step keys also make accidental duplicate metrics visible rather than averaging retries.
    rows = [json.loads(line) for line in (args.run_dir / "metrics.jsonl").read_text().splitlines()]
    steps = [row["step"] for row in rows]
    if len(steps) != len(set(steps)):
        raise ValueError("Duplicate logged steps: separate attempts before comparing throughput")
    summary = {"provenance": provenance, "windows": []}
    for first, last in provenance["clean_windows_relative_steps"]:
        window = [row for row in rows if first <= row["step"] - provenance["source_step"] <= last]
        metrics = {}
        for key in (
            "throughput/device/TPS",
            "throughput/device/TFLOPs_per_GPU",
            "throughput/device/MFU (%)",
            "throughput/device/data loading (s)",
            "train/CE loss",
            "optim/grad_norm",
        ):
            values = [row[key] for row in window if key in row]
            if values:
                metrics[key] = {
                    "n": len(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "stdev": statistics.stdev(values) if len(values) > 1 else 0,
                }
        tps = [row["throughput/device/TPS"] for row in window if "throughput/device/TPS" in row]
        if tps:
            latencies = [
                provenance["global_batch_tokens"] / provenance["gpus"] / value for value in tps
            ]
            metrics["step_seconds_from_tps"] = {
                "mean": statistics.mean(latencies),
                "median": statistics.median(latencies),
            }
        summary["windows"].append({"relative_steps": [first, last], "metrics": metrics})
    if args.traces:
        summary["traces"] = [
            summarize_trace(path)
            for path in sorted((args.run_dir / "profiler").glob("*.chrome_trace.json.gz"))
        ]
    output = args.run_dir / "analysis.json"
    output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
