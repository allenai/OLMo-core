"""Production allocation starts with a two-step full-state save/reload gate."""

import hashlib
import json
import os
import subprocess
import sys
import time

from olmoe3_hero_decay_plan import AUTOMATION, END, SMOKE_END, START, find_run, validate_checkpoint
from olmoe3_lr_sweep_plan import checkpoint_complete
from olmoe3_lr_sweep_watch import atomic_json, log
from olmoe3_profile_node import resolve_ready_leader


def main():
    from beaker import Beaker

    run = find_run(sys.argv[1])
    exp, job = os.environ["BEAKER_EXPERIMENT_ID"], os.environ["BEAKER_JOB_ID"]
    rank = int(os.environ["BEAKER_REPLICA_RANK"])
    assert (
        int(os.environ["BEAKER_REPLICA_COUNT"]),
        int(os.environ["BEAKER_ASSIGNED_GPU_COUNT"]),
    ) == (8, 8)
    subprocess.run(
        [sys.executable, "src/examples/olmo_ddp/olmoe3_hero_decay.py", "--validate-only"],
        check=True,
    )
    ready_dir = AUTOMATION / "rendezvous" / exp
    ready_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(ready_dir / f"{job}.json", dict(job=job, rank=rank))
    with Beaker.from_env(check_for_upgrades=False) as b:
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            leader = resolve_ready_leader(b, b.workload.get(exp), ready_dir, 8)
            if leader:
                break
            time.sleep(10)
        else:
            raise TimeoutError("Decay replica rendezvous timed out")
    leader_job, host = leader
    selection = ready_dir / f"selection-{leader_job}.json"
    if rank == 0:
        choices = [(START, run.source)]
        for p in run.root.glob("step*"):
            if p.name[4:].isdigit() and checkpoint_complete(p):
                choices.append((int(p.name[4:]), p))
        step, source = max(choices, key=lambda x: x[0])
        validate_checkpoint(source, step)
        assert START <= step <= END
        atomic_json(selection, dict(step=step, source=str(source)))
    deadline = time.monotonic() + 900
    while not selection.is_file():
        assert time.monotonic() < deadline
        time.sleep(2)
    initial = json.loads(selection.read_text())
    start, source = initial["step"], initial["source"]
    # Recover publication-before-receipt interruption without restarting the decay from its parent.
    smoke_marker = run.root / "audit/resume-smoke-saved.json"
    if rank == 0 and start >= SMOKE_END and not smoke_marker.exists():
        validate_checkpoint(run.root / f"step{SMOKE_END}", SMOKE_END)
        for gpu in range(64):
            row = json.loads(
                (run.root / "audit" / f"full-restore-step{START}-rank{gpu}.json").read_text()
            )
            assert all(
                row[k] is True for k in ("sampled_state_exact", "rng_exact", "data_state_exact")
            )
        atomic_json(smoke_marker, dict(step=SMOKE_END, reconciled=True, all_64_ranks_verified=True))
    if start >= SMOKE_END:
        deadline = time.monotonic() + 300
        while not smoke_marker.exists():
            assert time.monotonic() < deadline
            time.sleep(2)
    stages = ([SMOKE_END] if start < SMOKE_END else []) + ([END] if start < END else [])
    port = 29000 + int(hashlib.sha256(exp.encode()).hexdigest()[:8], 16) % 900
    for i, stop in enumerate(stages):
        if stop == END:
            (
                validate_checkpoint(run.root / f"step{SMOKE_END}", SMOKE_END)
                if start == SMOKE_END
                else None
            )
            # The two-step gate is mandatory even if this allocation resumed later.
            assert (run.root / "audit/resume-smoke-saved.json").is_file()
        env = {
            **os.environ,
            "OLMO35_HERO_EXPECTED_START": str(start),
            "OLMO35_HERO_STOP": str(stop),
            "OLMO35_HERO_ALLOW_CONTINUATION": "1",
            "OLMO35_DECAY_LOAD": source,
            "WANDB_RUN_ID": hashlib.sha256(run.run_id.encode()).hexdigest()[:8],
            "WANDB_RESUME": "allow",
        }
        log("DECAY_PASS_START", run=run.run_id, node=rank, start=start, stop=stop, source=source)
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
                "src/examples/olmo_ddp/olmoe3_hero_decay.py",
                "train",
                run.run_id,
                "ai2/holmes",
            ],
            env=env,
            check=True,
        )
        if rank == 0:
            validate_checkpoint(run.root / f"step{stop}", stop)
            for gpu in range(64):
                row = json.loads(
                    (run.root / "audit" / f"full-restore-step{start}-rank{gpu}.json").read_text()
                )
                assert all(
                    row[k] is True for k in ("sampled_state_exact", "rng_exact", "data_state_exact")
                )
            marker = "resume-smoke-saved.json" if stop == SMOKE_END else "decay-success.json"
            atomic_json(
                run.root / "audit" / marker,
                dict(
                    step=stop,
                    experiment=exp,
                    source_commit=os.environ["GIT_REF"],
                    restore_start=start,
                    all_64_ranks_verified=True,
                ),
            )
        marker = (
            run.root
            / "audit"
            / ("resume-smoke-saved.json" if stop == SMOKE_END else "decay-success.json")
        )
        deadline = time.monotonic() + 300
        while not marker.is_file():
            assert time.monotonic() < deadline
            time.sleep(2)
        start, source = stop, str(run.root / f"step{stop}")
    log("DECAY_NODE_COMPLETE", run=run.run_id, node=rank)


if __name__ == "__main__":
    main()
