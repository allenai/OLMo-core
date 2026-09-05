"""Run matched save/restore smokes or one 100B integration arm on 8x8 B300s."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from olmoe3_profile_node import resolve_ready_leader


def main():
    """Use current Beaker replica assignments and separate agents for every restore."""
    from beaker import Beaker

    name, cluster = sys.argv[1:]
    experiment = os.environ["BEAKER_EXPERIMENT_ID"]
    job = os.environ["BEAKER_JOB_ID"]
    rank = int(os.environ["BEAKER_REPLICA_RANK"])
    if (int(os.environ["BEAKER_REPLICA_COUNT"]), int(os.environ["BEAKER_ASSIGNED_GPU_COUNT"])) != (
        8,
        8,
    ):
        raise RuntimeError("Integration requires 8 nodes with 8 GPUs each")
    root = Path("/weka/olmo-3p5-checkpoints/production-integration")
    topology = root / "topology" / experiment / job
    subprocess.run(
        [sys.executable, "src/examples/olmo_ddp/olmoe3_profile_topology.py", str(topology)],
        check=True,
    )
    ready = root / "rendezvous" / experiment
    ready.mkdir(parents=True, exist_ok=True)
    (ready / f"{job}.json").write_text(json.dumps({"job": job, "rank": rank}))
    with Beaker.from_env() as beaker:
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            leader = resolve_ready_leader(beaker, beaker.workload.get(experiment), ready, 8)
            if leader:
                break
            print(f"Node {rank}: waiting for all current jobs to be ready", flush=True)
            time.sleep(10)
        else:
            raise TimeoutError("Integration rendezvous did not become ready in 15 minutes")
    _, host = leader
    smoke = os.environ.get("OLMOE3_INTEGRATION_SMOKE", "0") == "1"
    passes = (
        [
            (f"{name}-{arm}", arm, start, stop)
            for arm in ("reference", "optimized")
            for start, stop in ((0, 4), (4, 8))
        ]
        if smoke
        else [(name, os.environ["OLMOE3_INTEGRATION_ARM"], 0, 6000)]
    )
    port = 29000 + int(hashlib.sha256(experiment.encode()).hexdigest()[:8], 16) % 1000
    for index, (run_name, arm, start, stop) in enumerate(passes):
        env = dict(
            os.environ,
            OLMOE3_INTEGRATION_ARM=arm,
            OLMOE3_INTEGRATION_EXPECTED_START=str(start),
            OLMOE3_INTEGRATION_STOP=str(stop),
        )
        print(
            f"INTEGRATION_AGENT node={rank} run={run_name} arm={arm} start={start} stop={stop}",
            flush=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nnodes=8",
                "--nproc-per-node=8",
                f"--node-rank={rank}",
                "--rdzv-backend=static",
                f"--rdzv-endpoint={host}:{port + index}",
                f"--rdzv-id={experiment}-{index}",
                "--rdzv-conf=read_timeout=900",
                "--max-restarts=0",
                "src/examples/olmo_ddp/olmoe3_small_integration.py",
                "train",
                run_name,
                cluster,
            ],
            env=env,
            check=True,
        )
    print(f"INTEGRATION_NODE_COMPLETE node={rank}", flush=True)


if __name__ == "__main__":
    main()
