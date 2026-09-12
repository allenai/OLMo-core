"""Eight-node MT runner: weights-only fork, two steps, full-state reload, final horizon."""

import hashlib
import json
import os
import subprocess
import sys
import time

from olmoe3_hero_decay_plan import validate_checkpoint
from olmoe3_hero_decay_runtime import verify_runtime
from olmoe3_hero_mt_plan import AUTOMATION, END, PT_STEP, SMOKE_END, find_run
from olmoe3_lr_sweep_plan import checkpoint_complete
from olmoe3_lr_sweep_watch import atomic_json, log
from olmoe3_profile_node import resolve_ready_leader


def main():
    from beaker import Beaker

    verify_runtime()
    r = find_run(sys.argv[1])
    exp, job = os.environ["BEAKER_EXPERIMENT_ID"], os.environ["BEAKER_JOB_ID"]
    rank = int(os.environ["BEAKER_REPLICA_RANK"])
    assert (
        int(os.environ["BEAKER_REPLICA_COUNT"]),
        int(os.environ["BEAKER_ASSIGNED_GPU_COUNT"]),
    ) == (8, 8)
    subprocess.run(
        [sys.executable, "src/examples/olmo_ddp/olmoe3_hero_mt.py", "--validate-only"], check=True
    )
    rendezvous = AUTOMATION / "rendezvous" / exp
    atomic_json(rendezvous / f"{job}.json", dict(job=job, rank=rank))
    with Beaker.from_env(check_for_upgrades=False) as b:
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            leader = resolve_ready_leader(b, b.workload.get(exp), rendezvous, 8)
            if leader:
                break
            time.sleep(10)
        else:
            raise TimeoutError("MT replica rendezvous timed out")
    leader_job, host = leader
    selection = rendezvous / f"selection-{leader_job}.json"
    if rank == 0:
        choices = [(0, r.source)]
        for p in r.root.glob("step*"):
            if p.name[4:].isdigit() and checkpoint_complete(p):
                choices.append((int(p.name[4:]), p))
        start, source = max(choices, key=lambda x: x[0])
        assert 0 <= start <= END
        validate_checkpoint(source, PT_STEP if source == r.source else start)
        atomic_json(selection, dict(step=start, source=str(source)))
    deadline = time.monotonic() + 900
    while not selection.is_file():
        assert time.monotonic() < deadline
        time.sleep(2)
    initial = json.loads(selection.read_text())
    start, source = initial["step"], initial["source"]
    smoke = r.root / "audit/mt-smoke-saved.json"
    if rank == 0 and start >= SMOKE_END and not smoke.exists():
        validate_checkpoint(r.root / f"step{SMOKE_END}", SMOKE_END)
        for gpu in range(64):
            row = json.loads((r.root / "audit" / f"initial-mt-transfer-rank{gpu}.json").read_text())
            assert all(
                row[k]
                for k in ("weights_and_buffers_sampled_exact", "optimizer_reset", "data_reset")
            )
        atomic_json(smoke, dict(step=SMOKE_END, reconciled=True, all_64_ranks_verified=True))
    stages = ([SMOKE_END] if start < SMOKE_END else []) + ([END] if start < END else [])
    port = 29000 + int(hashlib.sha256(exp.encode()).hexdigest()[:8], 16) % 900
    for i, stop in enumerate(stages):
        if stop == END:
            deadline = time.monotonic() + 300
            while not smoke.is_file():
                assert time.monotonic() < deadline
                time.sleep(2)
        env = {
            **os.environ,
            "OLMO35_HERO_EXPECTED_START": str(start),
            "OLMO35_HERO_STOP": str(stop),
            "OLMO35_HERO_ALLOW_CONTINUATION": "1",
            "OLMO35_MT_LOAD": source,
            "WANDB_RUN_ID": hashlib.sha256(r.run_id.encode()).hexdigest()[:8],
            "WANDB_RESUME": "allow",
        }
        log("MT_PASS_START", run=r.run_id, node=rank, start=start, stop=stop, source=source)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nnodes=8",
                "--nproc-per-node=8",
                f"--node-rank={rank}",
                "--rdzv-backend=static",
                f"--rdzv-endpoint={host}:{port+i}",
                f"--rdzv-id={exp}-{i}",
                "--rdzv-conf=read_timeout=900",
                "--max-restarts=0",
                "src/examples/olmo_ddp/olmoe3_hero_mt.py",
                "train",
                r.run_id,
                "ai2/holmes",
            ],
            env=env,
            check=True,
        )
        marker = smoke if stop == SMOKE_END else r.root / "audit/mt-success.json"
        if rank == 0:
            validate_checkpoint(r.root / f"step{stop}", stop)
            for gpu in range(64):
                row = json.loads(
                    (r.root / "audit" / f"mt-restore-step{start}-rank{gpu}.json").read_text()
                )
                fields = (
                    ("weights_and_buffers_sampled_exact", "optimizer_reset", "data_reset")
                    if source == str(r.source)
                    else ("sampled_state_exact", "rng_exact", "data_state_exact")
                )
                assert all(row[k] for k in fields)
            atomic_json(
                marker,
                dict(
                    step=stop,
                    experiment=exp,
                    source_commit=os.environ["GIT_REF"],
                    restore_start=start,
                    all_64_ranks_verified=True,
                ),
            )
        deadline = time.monotonic() + 300
        while not marker.exists():
            assert time.monotonic() < deadline
            time.sleep(2)
        start, source = stop, str(r.root / f"step{stop}")
    log("MT_NODE_COMPLETE", run=r.run_id, node=rank)


if __name__ == "__main__":
    main()
