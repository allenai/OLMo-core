"""One torchrun worker; run separate profiler processes without overlapping CUPTI consumers."""

import os
import shutil
import subprocess
import sys
from pathlib import Path


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
        if mode == "nsys":
            nsys = (
                shutil.which("nsys")
                or "/opt/nvidia/nsight-compute/2025.3.1/host/target-linux-x64/nsys"
            )
            if not Path(nsys).is_file():
                raise RuntimeError(
                    "Nsight Systems is required; refusing an unrecorded profiling pass"
                )
            command = [
                nsys,
                "profile",
                "--trace=cuda,nvtx,osrt",
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


if __name__ == "__main__":
    main()
