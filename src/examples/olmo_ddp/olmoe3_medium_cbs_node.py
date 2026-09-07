"""Retain sixteen nodes across fresh, same-batch resume, fork, and child resume."""

import hashlib
import json
import os
import subprocess
import sys
import time

from olmoe3_medium_cbs_plan import ROOT, SMOKE_BASELINE, SMOKE_BRANCH, find_run
from olmoe3_profile_node import resolve_ready_leader


def main():
    """Use current Beaker assignments, with a fresh distributed agent for every restore."""
    from beaker import Beaker

    name, cluster = sys.argv[1:]
    experiment, job = os.environ["BEAKER_EXPERIMENT_ID"], os.environ["BEAKER_JOB_ID"]
    rank = int(os.environ["BEAKER_REPLICA_RANK"])
    if (int(os.environ["BEAKER_REPLICA_COUNT"]), int(os.environ["BEAKER_ASSIGNED_GPU_COUNT"])) != (
        16,
        8,
    ):
        raise RuntimeError("Medium CBS requires sixteen eight-GPU nodes")
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
    with Beaker.from_env(check_for_upgrades=False) as beaker:
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            leader = resolve_ready_leader(beaker, beaker.workload.get(experiment), ready, 16)
            if leader:
                break
            print(f"Node {rank}: waiting for current CBS replicas", flush=True)
            time.sleep(10)
        else:
            raise TimeoutError("CBS rendezvous did not become ready within fifteen minutes")
    _, host = leader
    smoke = os.environ.get("OLMOE3_MEDIUM_CBS_SMOKE", "0") == "1"
    run = find_run(name)
    if smoke and run != SMOKE_BASELINE:
        raise ValueError("Smoke must use its explicit parent identity")
    passes = (
        [
            (SMOKE_BASELINE, 0, 2),
            (SMOKE_BASELINE, 2, 4),
            (SMOKE_BRANCH, 2, 3),
            (SMOKE_BRANCH, 3, 4),
        ]
        if smoke
        else [(run, run.start, run.end)]
    )
    port = 29000 + int(hashlib.sha256(experiment.encode()).hexdigest()[:8], 16) % 900
    for index, (phase, start, stop) in enumerate(passes):
        env = dict(
            os.environ,
            OLMOE3_MEDIUM_CBS_RUN=phase.run_id,
            OLMOE3_MEDIUM_CBS_EXPECTED_START=str(start),
            OLMOE3_MEDIUM_CBS_STOP=str(stop),
        )
        print(
            f"MEDIUM_CBS_AGENT node={rank} run={phase.run_id} start={start} stop={stop}", flush=True
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nnodes=16",
                "--nproc-per-node=8",
                f"--node-rank={rank}",
                "--rdzv-backend=static",
                f"--rdzv-endpoint={host}:{port+index}",
                f"--rdzv-id={experiment}-{index}",
                "--rdzv-conf=read_timeout=900",
                "--max-restarts=0",
                "src/examples/olmo_ddp/olmoe3_medium_cbs.py",
                "train",
                phase.run_id,
                cluster,
            ],
            env=env,
            check=True,
        )
    print(f"MEDIUM_CBS_NODE_COMPLETE node={rank}", flush=True)


if __name__ == "__main__":
    main()
