"""Fresh distributed agents for production training and the save/restore gate."""

import hashlib
import json
import os
import subprocess
import sys
import time

from olmoe3_lr_sweep_plan import ROOT, find_run, smoke_runs
from olmoe3_profile_node import resolve_ready_leader


def main():
    """Resolve current replica addresses; do not reuse agents across checkpoint restores."""
    from beaker import Beaker

    name, cluster = sys.argv[1:]
    experiment, job = os.environ["BEAKER_EXPERIMENT_ID"], os.environ["BEAKER_JOB_ID"]
    rank = int(os.environ["BEAKER_REPLICA_RANK"])
    assert (
        int(os.environ["BEAKER_REPLICA_COUNT"]),
        int(os.environ["BEAKER_ASSIGNED_GPU_COUNT"]),
    ) == (8, 8)
    subprocess.run(
        [
            sys.executable,
            "src/examples/olmo_ddp/olmoe3_profile_topology.py",
            str(ROOT / "topology" / experiment / job),
        ],
        check=True,
    )
    ready = ROOT / "rendezvous" / experiment
    ready.mkdir(parents=True, exist_ok=True)
    (ready / f"{job}.json").write_text(json.dumps({"job": job, "rank": rank}))
    with Beaker.from_env() as beaker:
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            leader = resolve_ready_leader(beaker, beaker.workload.get(experiment), ready, 8)
            if leader:
                break
            print(f"Node {rank}: waiting for current replica assignments", flush=True)
            time.sleep(10)
        else:
            raise TimeoutError("Sweep rendezvous did not become ready within 15 minutes")
    _, host = leader
    smoke = os.environ.get("OLMOE3_LR_SWEEP_SMOKE") == "1"
    parent, child = smoke_runs()
    passes = [(parent, 2), (child, 4), (child, 6)] if smoke else [(find_run(name), None)]
    port = 29000 + int(hashlib.sha256(experiment.encode()).hexdigest()[:8], 16) % 1000
    for index, (r, stop) in enumerate(passes):
        env = dict(os.environ)
        if stop is not None:
            env["OLMOE3_LR_SMOKE_STOP"] = str(stop)
        print(f"SWEEP_AGENT node={rank} run={r.run_id} stop={stop}", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nnodes=8",
                "--nproc-per-node=8",
                f"--node-rank={rank}",
                "--rdzv-backend=static",
                f"--rdzv-endpoint={host}:{port+index}",
                f"--rdzv-id={experiment}-{index}",
                "--rdzv-conf=read_timeout=900",
                "--max-restarts=0",
                "src/examples/olmo_ddp/olmoe3_small_lr_sweep.py",
                "train",
                r.run_id,
                cluster,
            ],
            env=env,
            check=True,
        )
    print(f"LR_SWEEP_NODE_COMPLETE node={rank}", flush=True)


if __name__ == "__main__":
    main()
