"""Hold one allocation for bounded speed/restore passes, never for dependency waits."""

import hashlib
import json
import os
import subprocess
import sys
import time

from olmoe3_medium_cbs64_plan import AUTOMATION, PHASES, ROOT, WAVES, phase_environment
from olmoe3_profile_node import resolve_ready_leader


def main():
    from beaker import Beaker

    wave, cluster = sys.argv[1:]
    phases = [PHASES[key] for key in WAVES[wave]]
    nodes = phases[0].gpus // 8
    assert all(p.gpus == nodes * 8 for p in phases)
    rank = int(os.environ["BEAKER_REPLICA_RANK"])
    assert (
        int(os.environ["BEAKER_REPLICA_COUNT"]),
        int(os.environ["BEAKER_ASSIGNED_GPU_COUNT"]),
    ) == (nodes, 8)
    experiment, job = os.environ["BEAKER_EXPERIMENT_ID"], os.environ["BEAKER_JOB_ID"]
    subprocess.run(
        [
            sys.executable,
            "src/examples/olmo_ddp/olmoe3_profile_topology.py",
            str(ROOT / "topology" / experiment / job),
        ],
        check=True,
    )
    ready = AUTOMATION / "rendezvous" / experiment
    ready.mkdir(parents=True, exist_ok=True)
    (ready / f"{job}.json").write_text(json.dumps({"job": job, "rank": rank}))
    with Beaker.from_env(check_for_upgrades=False) as beaker:
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            leader = resolve_ready_leader(beaker, beaker.workload.get(experiment), ready, nodes)
            if leader:
                break
            print(f"MEDIUM_CBS64_WAIT node={rank} wave={wave}", flush=True)
            time.sleep(10)
        else:
            raise TimeoutError("Current replicas did not rendezvous")
    _, host = leader
    port = 29000 + int(hashlib.sha256(experiment.encode()).hexdigest()[:8], 16) % 900
    for index, phase in enumerate(phases):
        # Fresh process groups/model/optimizer for every pass, including restore smoke.
        env = phase_environment(os.environ, phase)
        print(f"MEDIUM_CBS64_AGENT node={rank} phase={phase.name}", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nnodes={nodes}",
                "--nproc-per-node=8",
                f"--node-rank={rank}",
                "--rdzv-backend=static",
                f"--rdzv-endpoint={host}:{port + index}",
                f"--rdzv-id={experiment}-{index}",
                "--rdzv-conf=read_timeout=900",
                "--max-restarts=0",
                "src/examples/olmo_ddp/olmoe3_medium_cbs64.py",
                "train",
                phase.run_id,
                cluster,
            ],
            env=env,
            check=True,
        )
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if all(
                (phase.root / "audit" / f"complete-rank{r}.json").is_file()
                for r in range(phase.gpus)
            ):
                break
            time.sleep(5)
        else:
            raise TimeoutError(f"Missing complete-rank gate for {phase.name}")
    print(f"MEDIUM_CBS64_NODE_COMPLETE node={rank} wave={wave}", flush=True)


if __name__ == "__main__":
    main()
