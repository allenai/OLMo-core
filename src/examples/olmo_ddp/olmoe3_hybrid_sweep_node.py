"""Pinned runtime/topology, fresh distributed agents, and matched-arm restore smoke."""

import hashlib
import json
import os
import subprocess
import sys
import time

from olmoe3_hybrid_sweep_plan import ROOT, find_run, smoke_runs
from olmoe3_profile_node import resolve_ready_leader
from olmoe3_small_hero_runtime import verify_runtime


def main():
    """Run one full curve point, or both isolated save/restore smoke arms."""
    from beaker import Beaker

    verify_runtime()
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
    (ready / f"{job}.json").write_text(json.dumps(dict(job=job, rank=rank)))
    with Beaker.from_env(check_for_upgrades=False) as beaker:
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            leader = resolve_ready_leader(beaker, beaker.workload.get(experiment), ready, 8)
            if leader:
                break
            print(f"Node{rank}: waiting for current replicas", flush=True)
            time.sleep(10)
        else:
            raise TimeoutError("Hybrid sweep rendezvous timed out")
    _, host = leader
    smoke = find_run(name).smoke
    passes = (
        [(r, start, stop) for r in smoke_runs() for start, stop in [(0, 4), (4, 6)]]
        if smoke
        else [(find_run(name), None, None)]
    )
    port = 29000 + int(hashlib.sha256(experiment.encode()).hexdigest()[:8], 16) % 1000
    for index, (r, start, stop) in enumerate(passes):
        env = dict(os.environ)
        if smoke:
            env.update(HYBRID_SMOKE_EXPECTED_START=str(start), HYBRID_SMOKE_STOP=str(stop))
        print(f"HYBRID_AGENT rank={rank} run={r.run_id} start={start} stop={stop}", flush=True)
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
                "src/examples/olmo_ddp/olmoe3_hybrid_sweep.py",
                "train",
                r.run_id,
                cluster,
            ],
            env=env,
            check=True,
        )
    print(f"HYBRID_NODE_COMPLETE rank={rank}", flush=True)


if __name__ == "__main__":
    main()
