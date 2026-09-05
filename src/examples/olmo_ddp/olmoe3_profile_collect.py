"""CPU-only profile analysis; publish only small summaries to Beaker results."""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/weka/olmo-3p5-checkpoints/production-profiling")


def main():
    results = Path(os.environ.get("RESULTS_DIR", "/results")) / "profile-summaries"
    results.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[1:]:
        if Path(name).name != name:
            raise ValueError("Expected a run directory name, not a path")
        run = ROOT / name
        deadline = time.monotonic() + 7200
        while len(list(run.glob("memory-rank-*.json"))) != 64:
            if time.monotonic() > deadline:
                raise TimeoutError(f"{name}: all 64 ranks did not finish in two hours")
            print(f"Waiting for all ranks of {name}", flush=True)
            time.sleep(30)
        provenance = json.loads((run / "provenance.json").read_text())
        if provenance["pass"] == "nsys":
            while len(list(run.glob("nsys-rank-*.nsys-rep"))) != 64:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{name}: Nsight did not finish exporting all ranks")
                print(f"Waiting for Nsight report exports for {name}", flush=True)
                time.sleep(30)
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
        for rank in range(0, 64, 8):
            report = run / f"nsys-rank-{rank}.nsys-rep"
            if not report.is_file():
                continue
            output = destination / f"nsys-rank-{rank}-stats.txt"
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
        print(f"Collected small summaries for {name} in {destination}", flush=True)


if __name__ == "__main__":
    main()
