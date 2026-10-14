"""Run both data segments inside one 64-GPU allocation, using current replica membership."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import olmoe3_combined_decay_plan as p


def main():
    """Resolve this attempt's leader, then run resumable PT and MT workers in sequence."""
    from beaker import Beaker
    from olmoe3_profile_node import resolve_ready_leader

    plan_path = Path(sys.argv[1]).resolve(strict=True)
    plan = p.fixed.read_json(plan_path)
    experiment, job = os.environ["BEAKER_EXPERIMENT_ID"], os.environ["BEAKER_JOB_ID"]
    rank = int(os.environ["BEAKER_REPLICA_RANK"])
    nodes = plan["gpus"] // 8
    assert (
        int(os.environ["BEAKER_REPLICA_COUNT"]),
        int(os.environ["BEAKER_ASSIGNED_GPU_COUNT"]),
    ) == (nodes, 8)
    root = Path(plan["save_folder"])
    for flag in ("STOP.json", "STORAGE_PAUSED.json"):
        assert not (root / flag).exists(), f"Operator attention required: {root / flag}"
    registration = Path("/weka/olmo-3p5-checkpoints/uploader/control/registrations") / (
        plan["run_name"] + ".json"
    )
    reg = p.fixed.read_json(registration)
    assert reg["enabled"] and all(reg.get(k) == v for k, v in plan["registration"].items())
    # Runtime bookkeeping is outside the checkpoint root, which has its own ownership marker.
    auto = Path("/weka/olmo-3p5-checkpoints/uploader/automation") / plan["run_name"]
    with Beaker.from_env(check_for_upgrades=False) as b:
        from google.protobuf.json_format import MessageToDict

        uploader = b.workload.get("01M1YTRJV16A5YC300SEZAW6D1")
        assert MessageToDict(uploader)["status"] == "STATUS_RUNNING"
        fs = os.statvfs("/weka/olmo-3p5-checkpoints")
        assert fs.f_bavail * fs.f_frsize >= 8_000_000_000_000
        subprocess.run(
            [
                sys.executable,
                "src/examples/olmo_ddp/olmoe3_profile_topology.py",
                str(auto / "topology" / experiment / job),
            ],
            check=True,
        )
        ready = auto / "rendezvous" / experiment
        p.write_json(ready / f"{job}.json", dict(job=job, rank=rank, plan_sha256=p.digest(plan)))
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            leader = resolve_ready_leader(b, b.workload.get(experiment), ready, nodes)
            if leader:
                break
            time.sleep(10)
        else:
            raise TimeoutError("Current replica generation not ready after 15 minutes")
    leader_job, host = leader
    port = 29000 + int(hashlib.sha256(experiment.encode()).hexdigest()[:8], 16) % 1000
    for index, phase in enumerate(("pt", "mt")):
        print(
            "COMBINED_DECAY_PHASE",
            json.dumps(dict(phase=phase, node=rank, leader=leader_job)),
            flush=True,
        )
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
                f"--rdzv-id={experiment}-{leader_job}-{phase}",
                "--rdzv-conf=read_timeout=900",
                "--max-restarts=0",
                "src/examples/olmo_ddp/olmoe3_combined_decay.py",
                "--plan",
                str(plan_path),
                "--phase",
                phase,
                "--cluster",
                plan["cluster"],
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
