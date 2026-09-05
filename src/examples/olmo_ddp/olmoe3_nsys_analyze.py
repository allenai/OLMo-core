"""Read-only GPU interval and CPU API summaries from an existing Nsight SQLite export."""

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path


def interval_union(intervals):
    """Merge same-device intervals without summing overlapping streams."""
    previous = None
    total = 0
    for start, end in sorted(intervals):
        if end <= start:
            continue
        total += max(0, end - max(start, previous if previous is not None else start))
        previous = max(end, previous if previous is not None else end)
    return total


def summarize_timeline(path):
    """Keep CPU-inclusive sums distinct from per-device GPU interval unions."""
    devices = defaultdict(lambda: {"compute": [], "collective": [], "memory": []})
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL"):
            if table not in tables:
                continue
            columns = {r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')}
            name = next((c for c in ("demangledName", "shortName", "name") if c in columns), None)
            if name is None or "StringIds" not in tables:
                raise ValueError("Kernel names are required to classify NCCL activity")
            device = "k.deviceId" if "deviceId" in columns else "0"
            for dev, start, end, label in conn.execute(
                f'SELECT {device}, k.start, k.end, s.value FROM "{table}" k '
                f'JOIN StringIds s ON k."{name}"=s.id'
            ):
                kind = "collective" if "nccl" in label.lower() else "compute"
                devices[dev][kind].append((start, end))
        for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
            if table not in tables:
                continue
            columns = {r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')}
            device = "deviceId" if "deviceId" in columns else "0"
            for dev, start, end in conn.execute(f'SELECT {device}, start, end FROM "{table}"'):
                devices[dev]["memory"].append((start, end))
        apis = []
        for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
            if table not in tables or "StringIds" not in tables:
                continue
            columns = {r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')}
            if "nameId" not in columns:
                continue
            for name, count, duration, maximum in conn.execute(
                f"SELECT s.value, COUNT(*), SUM(a.end-a.start), MAX(a.end-a.start) "
                f'FROM "{table}" a JOIN StringIds s ON a.nameId=s.id GROUP BY s.value'
            ):
                apis.append(
                    {
                        "table": table,
                        "name": name,
                        "calls": count,
                        "inclusive_sum_ms": duration / 1e6,
                        "max_call_ms": maximum / 1e6,
                    }
                )
    timelines = []
    for dev, groups in sorted(devices.items()):
        compute, collective, memory = (groups[k] for k in ("compute", "collective", "memory"))
        kernels = compute + collective
        if not kernels:
            continue
        first, last = min(a for a, _ in kernels), max(b for _, b in kernels)
        # Memory operations outside the first/last kernel are not counted as filling its gaps.
        clipped_memory = [
            (max(first, a), min(last, b)) for a, b in memory if a < last and b > first
        ]
        kernel_busy = interval_union(kernels)
        compute_busy = interval_union(compute)
        all_busy = interval_union(kernels + clipped_memory)
        comm = interval_union(collective)
        timelines.append(
            {
                "device": dev,
                "kernel_count": len(kernels),
                "kernel_span_ms": (last - first) / 1e6,
                "kernel_busy_union_ms": kernel_busy / 1e6,
                "collective_union_ms": comm / 1e6,
                "collective_without_other_kernel_ms": (kernel_busy - compute_busy) / 1e6,
                "collective_overlapping_other_kernel_ms": (compute_busy + comm - kernel_busy) / 1e6,
                "kernel_and_memory_busy_union_ms": all_busy / 1e6,
                "no_recorded_gpu_operation_in_kernel_span_ms": (last - first - all_busy) / 1e6,
            }
        )
    return {
        "database": str(path),
        "devices": timelines,
        "cpu_api_inclusive_nonadditive": sorted(apis, key=lambda r: -r["inclusive_sum_ms"]),
        "caveat": "Instrumented timing only. CPU scopes and GPU kernels overlap. Collective-only time is not a removable critical-path budget; no recorded GPU operation is not proof of hardware idleness. Compare with unprofiled timings and the separate PyTorch trace.",
    }


def main():
    """Print compact summaries; raw captures and exports remain on the mounted filesystem."""
    parser = argparse.ArgumentParser()
    parser.add_argument("databases", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in args.databases:
        summary = summarize_timeline(path)
        if args.output_dir is not None:
            (args.output_dir / f"{path.stem}-timeline.json").write_text(
                json.dumps(summary, indent=2)
            )
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
