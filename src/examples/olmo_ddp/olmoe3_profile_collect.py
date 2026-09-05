"""CPU-only profile analysis; publish only small summaries to Beaker results."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from olmoe3_nsys_tools import install_nsys

ROOT = Path("/weka/olmo-3p5-checkpoints/production-profiling")


def main():
    """Wait for completed passes and analyze artifacts without consuming any GPUs."""
    results = Path(os.environ.get("RESULTS_DIR", "/results")) / "profile-summaries"
    results.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("names", nargs="+")
    args = parser.parse_args()
    pending = list(args.names)
    deadline = time.monotonic() + 7200
    standalone_nsys = None
    for name in pending:
        if Path(name).name != name:
            raise ValueError("Expected a run directory name, not a path")
    while pending:
        ready = []
        for name in pending:
            run = ROOT / name
            if not (run / "provenance.json").is_file() or not (run / "metrics.jsonl").is_file():
                continue
            if not args.allow_partial and len(list(run.glob("memory-rank-*.json"))) != 64:
                continue
            provenance = json.loads((run / "provenance.json").read_text())
            if not args.allow_partial and provenance["pass"] == "nsys":
                ranks = provenance.get("nsys_profiled_ranks") or list(range(64))
                if not all((run / f"nsys-rank-{rank}.nsys-rep").is_file() for rank in ranks):
                    continue
                if provenance.get("nsys_version") not in (None, "installed"):
                    markers = [run / f"nsys-rank-{rank}-validation.json" for rank in ranks]
                    if not all(path.is_file() for path in markers):
                        continue
                    if not all(
                        json.loads(path.read_text())["valid_cuda_trace"] for path in markers
                    ):
                        raise RuntimeError(f"Invalid CUDA capture in {name}")
            ready.append(name)
        if not ready:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Runs did not finish in two hours: {pending}")
            print(f"Waiting for completed passes: {pending}", flush=True)
            time.sleep(30)
            continue
        name = ready[0]
        pending.remove(name)
        run = ROOT / name
        destination = results / name
        destination.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                "src/examples/olmo_ddp/olmoe3_profile_analyze.py",
                str(run),
                "--traces",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        for filename in ("analysis.json", "metrics.jsonl", "provenance.json"):
            shutil.copy2(run / filename, destination / filename)
        summary = json.loads((run / "analysis.json").read_text())
        provenance = summary["provenance"]
        summary["partial_collection"] = args.allow_partial
        (destination / "analysis.json").write_text(json.dumps(summary, indent=2))
        print("PROFILE_PARTIAL_COLLECTION", name, args.allow_partial, flush=True)
        for window in summary["windows"]:
            print("PROFILE_WINDOW", name, json.dumps(window), flush=True)
        print("PROFILE_FIRST_UPDATES", name, json.dumps(summary["first_updates"]), flush=True)
        print("PROFILE_MEMORY", name, json.dumps(summary["memory_by_rank"]), flush=True)
        for trace in summary.get("traces", []):
            print("PROFILE_TRACE", name, json.dumps(trace), flush=True)
        inventory = [
            {"name": str(p.relative_to(run)), "bytes": p.stat().st_size}
            for p in run.rglob("*")
            if p.is_file()
        ]
        (destination / "artifact-inventory.json").write_text(json.dumps(inventory, indent=2))
        # Read-only CUDA/NVTX reports. Large SQLite exports and raw captures stay on Weka.
        nsys = (
            shutil.which("nsys") or "/opt/nvidia/nsight-compute/2025.3.1/host/target-linux-x64/nsys"
        )
        if provenance.get("nsys_version") not in (None, "installed"):
            if standalone_nsys is None:
                standalone_nsys = install_nsys()
            nsys = str(standalone_nsys)
        for rank in provenance.get("nsys_profiled_ranks") or range(0, 64, 8):
            report = run / f"nsys-rank-{rank}.nsys-rep"
            if not report.is_file():
                continue
            validation = run / f"nsys-rank-{rank}-validation.json"
            if validation.is_file():
                shutil.copy2(validation, destination / validation.name)
            # Autograd emits unique op IDs: the full NVTX projection is ~17MiB/rank.
            # Keep it with raw Weka artifacts, not in result datasets or output logs.
            output = run / f"nsys-rank-{rank}-stats.txt"
            with output.open("w") as handle:
                result = subprocess.run(
                    [
                        nsys,
                        "stats",
                        "--report=cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum,nvtx_gpu_proj_sum",
                        "--format=csv",
                        str(report),
                    ],
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    timeout=600,
                )
            if result.returncode:
                print(f"Nsight stats failed for rank{rank}; retained diagnostic output", flush=True)
            compact = []
            with output.open() as handle:
                for line in handle:
                    if "/nvtx_gpu_proj_sum.py" in line:
                        break
                    compact.append(line)
            compact.append(f"\nFull per-op NVTX GPU projection retained on Weka: {output}\n")
            (destination / output.name).write_text("".join(compact))
            if rank == 0:
                print("NSYS_RANK0_STATS_START", name, flush=True)
                print("".join(line[:240] + "\n" for line in compact[:30]), flush=True)
                print("Full CUDA summaries in results; full NVTX projection on Weka.", flush=True)
                print("NSYS_RANK0_STATS_END", name, flush=True)
        print(f"Collected small summaries for {name} in {destination}", flush=True)


if __name__ == "__main__":
    main()
