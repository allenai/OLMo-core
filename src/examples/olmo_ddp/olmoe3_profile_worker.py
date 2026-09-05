"""One torchrun worker; run separate profiler processes without overlapping CUPTI consumers."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from olmoe3_nsys_tools import NsysSettings, validate_report


def main():
    """Run all selected passes; a failed pass fails the whole synchronized allocation."""
    run_name, cluster = sys.argv[1:]
    rank = int(os.environ["RANK"])
    root = Path("/weka/olmo-3p5-checkpoints/production-profiling")
    for mode in os.environ.get("OLMOE3_DEEP_PROFILE_PASSES", "nsys,torch").split(","):
        name = f"{run_name}-{mode}"
        output = root / name
        output.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "src/examples/olmo_ddp/olmoe3_small_deep_profile.py",
            "train",
            name,
            cluster,
        ]
        settings = NsysSettings.from_env() if mode == "nsys" else None
        if settings is not None and rank in settings.ranks:
            nsys = (
                os.environ.get("OLMOE3_NSYS_BINARY")
                or shutil.which("nsys")
                or "/opt/nvidia/nsight-compute/2025.3.1/host/target-linux-x64/nsys"
            )
            if settings.version != "installed" and not os.environ.get("OLMOE3_NSYS_BINARY"):
                raise RuntimeError("Pinned standalone Nsight was not installed on this node")
            if not Path(nsys).is_file():
                raise RuntimeError(
                    "Nsight Systems is required; refusing an unrecorded profiling pass"
                )
            command = [
                nsys,
                "profile",
                "--trace=" + settings.trace,
                "--sample=none",
                "--cpuctxsw=none",
                "--capture-range=cudaProfilerApi",
                "--capture-range-end=stop",
                "--kill=none",
                "--output",
                str(output / f"nsys-rank-{rank}"),
                *command,
            ]
        env = dict(os.environ, OLMOE3_DEEP_PROFILE_PASS=mode)
        print(f"Launching profiling pass {mode}, rank={rank}, artifacts={output}", flush=True)
        subprocess.run(command, env=env, check=True)
        if settings is not None and rank in settings.ranks:
            validation = validate_report(nsys, output / f"nsys-rank-{rank}.nsys-rep")
            validation.update(rank=rank, version=settings.version, trace=settings.trace)
            (output / f"nsys-rank-{rank}-validation.json").write_text(
                json.dumps(validation, indent=2)
            )
            print("NSYS_TRACE_VALIDATION", json.dumps(validation), flush=True)
            if not validation["valid_cuda_trace"] or not validation["nccl_kernel_count"]:
                raise RuntimeError("Nsight report is missing actual CUDA/copy/NCCL activity")


if __name__ == "__main__":
    main()
