"""Eight-node LC runner: weights-only fork, two steps, full-state reload, final horizon."""

import hashlib
import json
import math
import os
import subprocess
import sys
import time

from olmoe3_hero_decay_plan import validate_checkpoint
from olmoe3_hero_decay_runtime import verify_runtime
from olmoe3_hero_lc_cache import validate_cache
from olmoe3_hero_lc_plan import (
    AUTOMATION,
    END,
    GATE_END,
    SMOKE_END,
    SOURCE_STEP,
    find_run,
)
from olmoe3_lr_sweep_plan import checkpoint_complete
from olmoe3_lr_sweep_watch import atomic_json, log
from olmoe3_profile_node import resolve_ready_leader


def verify_gpu_gate(run):
    """Require both startup passes, all-rank full-state restore and finite measured updates."""
    validate_checkpoint(run.root / f"step{GATE_END}", GATE_END)
    for gpu in range(64):
        transfer = json.loads(
            (run.root / "audit" / f"initial-lc-transfer-rank{gpu}.json").read_text()
        )
        restore = json.loads(
            (run.root / "audit" / f"lc-restore-step{SMOKE_END}-rank{gpu}.json").read_text()
        )
        assert all(
            transfer[k]
            for k in ("weights_and_buffers_sampled_exact", "optimizer_reset", "data_reset")
        )
        assert all(restore[k] for k in ("sampled_state_exact", "rng_exact", "data_state_exact"))
    metrics = [
        json.loads(line) for line in (run.root / "audit/metrics.jsonl").read_text().splitlines()
    ]
    for step in range(1, GATE_END + 1):
        rows = [v for v in metrics if v["step"] == step and "train/CE loss" in v]
        assert rows, f"No measured loss for LC gate step{step}"
        for row in rows:
            assert math.isfinite(row["train/CE loss"])
            assert math.isfinite(row["optim/total grad norm"]) and row["optim/total grad norm"] > 0
    return dict(finite_updates=GATE_END, weights_reset_and_resume_verified=True)


def main():
    from beaker import Beaker

    validate_cache()
    verify_runtime()
    r = find_run(sys.argv[1])
    phase = os.environ["OLMO35_LC_PHASE"]
    assert phase in ("smoke", "train")
    target = GATE_END if phase == "smoke" else END
    exp, job = os.environ["BEAKER_EXPERIMENT_ID"], os.environ["BEAKER_JOB_ID"]
    rank = int(os.environ["BEAKER_REPLICA_RANK"])
    assert (
        int(os.environ["BEAKER_REPLICA_COUNT"]),
        int(os.environ["BEAKER_ASSIGNED_GPU_COUNT"]),
    ) == (8, 8)
    subprocess.run(
        [sys.executable, "src/examples/olmo_ddp/olmoe3_hero_lc.py", "--validate-only"], check=True
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
            raise TimeoutError("LC replica rendezvous timed out")
    leader_job, host = leader
    selection = rendezvous / f"selection-{leader_job}.json"
    if rank == 0:
        choices = [(0, r.source)]
        for p in r.root.glob("step*"):
            if p.name[4:].isdigit() and checkpoint_complete(p):
                choices.append((int(p.name[4:]), p))
        start, source = max(choices, key=lambda x: x[0])
        assert 0 <= start <= target
        if phase == "train":
            gate = json.loads((r.root / "audit/lc-gate-success.json").read_text())
            assert gate["step"] == GATE_END and gate["all_64_ranks_verified"]
            assert start >= GATE_END
        validate_checkpoint(source, SOURCE_STEP if source == r.source else start)
        atomic_json(selection, dict(step=start, source=str(source)))
    deadline = time.monotonic() + 900
    while not selection.is_file():
        assert time.monotonic() < deadline
        time.sleep(2)
    initial = json.loads(selection.read_text())
    start, source = initial["step"], initial["source"]
    if phase == "smoke" and start == GATE_END:
        proof = verify_gpu_gate(r)
        if rank == 0:
            atomic_json(
                r.root / "audit/lc-gate-success.json",
                dict(
                    step=GATE_END,
                    experiment=exp,
                    source_commit=os.environ["GIT_REF"],
                    restore_start=SMOKE_END,
                    all_64_ranks_verified=True,
                    reconciled=True,
                    **proof,
                ),
            )
    smoke = r.root / "audit/lc-smoke-saved.json"
    if rank == 0 and start >= SMOKE_END and not smoke.exists():
        validate_checkpoint(r.root / f"step{SMOKE_END}", SMOKE_END)
        for gpu in range(64):
            row = json.loads((r.root / "audit" / f"initial-lc-transfer-rank{gpu}.json").read_text())
            assert all(
                row[k]
                for k in ("weights_and_buffers_sampled_exact", "optimizer_reset", "data_reset")
            )
        atomic_json(smoke, dict(step=SMOKE_END, reconciled=True, all_64_ranks_verified=True))
    stages = ([SMOKE_END] if start < SMOKE_END else []) + ([target] if start < target else [])
    port = 29000 + int(hashlib.sha256(exp.encode()).hexdigest()[:8], 16) % 900
    for i, stop in enumerate(stages):
        if stop > SMOKE_END:
            deadline = time.monotonic() + 300
            while not smoke.is_file():
                assert time.monotonic() < deadline
                time.sleep(2)
        env = {
            **os.environ,
            "OLMO35_HERO_EXPECTED_START": str(start),
            "OLMO35_HERO_STOP": str(stop),
            "OLMO35_HERO_ALLOW_CONTINUATION": "1",
            "OLMO35_LC_LOAD": source,
            "WANDB_RUN_ID": hashlib.sha256(r.run_id.encode()).hexdigest()[:8],
            "WANDB_RESUME": "allow",
        }
        log("LC_PASS_START", run=r.run_id, node=rank, start=start, stop=stop, source=source)
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
                "src/examples/olmo_ddp/olmoe3_hero_lc.py",
                "train",
                r.run_id,
                "ai2/holmes",
            ],
            env=env,
            check=True,
        )
        marker = (
            smoke
            if stop == SMOKE_END
            else r.root
            / ("audit/lc-gate-success.json" if phase == "smoke" else "audit/lc-success.json")
        )
        if rank == 0:
            validate_checkpoint(r.root / f"step{stop}", stop)
            for gpu in range(64):
                row = json.loads(
                    (r.root / "audit" / f"lc-restore-step{start}-rank{gpu}.json").read_text()
                )
                fields = (
                    ("weights_and_buffers_sampled_exact", "optimizer_reset", "data_reset")
                    if source == str(r.source)
                    else ("sampled_state_exact", "rng_exact", "data_state_exact")
                )
                assert all(row[k] for k in fields)
            proof = verify_gpu_gate(r) if phase == "smoke" and stop == GATE_END else {}
            atomic_json(
                marker,
                dict(
                    step=stop,
                    experiment=exp,
                    source_commit=os.environ["GIT_REF"],
                    restore_start=start,
                    all_64_ranks_verified=True,
                    **proof,
                ),
            )
        deadline = time.monotonic() + 300
        while not marker.exists():
            assert time.monotonic() < deadline
            time.sleep(2)
        start, source = stop, str(r.root / f"step{stop}")
    log("LC_NODE_COMPLETE", run=r.run_id, node=rank)


if __name__ == "__main__":
    main()
