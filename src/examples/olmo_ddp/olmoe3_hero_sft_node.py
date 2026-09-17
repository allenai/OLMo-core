"""Single-node SFT; first qualify both LC transfers and a full-state restart."""

import hashlib
import json
import math
import os
import subprocess
import sys

from olmoe3_hero_decay_plan import inventory
from olmoe3_hero_decay_plan import validate_checkpoint as validate_parent
from olmoe3_hero_decay_runtime import verify_runtime
from olmoe3_hero_sft_plan import BATCH, GPUS, MOUNT, find_run, runs
from olmoe3_lr_sweep_plan import checkpoint_complete
from olmoe3_lr_sweep_watch import atomic_json, log


def validate_checkpoint(root, step):
    """SFT checkpoints have eight ranks and their own global token counter."""
    inventory(root)
    assert checkpoint_complete(root)
    for rank in range(GPUS):
        row = json.loads((root / "resume_audit" / f"rank{rank}.json").read_text())
        assert (row["step"], row["tokens"], row["rank"], row["gpus"]) == (
            step,
            step * BATCH,
            rank,
            GPUS,
        )
        assert (root / "train" / f"rank{rank}.pt").is_file()


def check_restore(run, start, fresh):
    fields = (
        ("weights_and_buffers_sampled_exact", "optimizer_reset", "data_reset")
        if fresh
        else ("sampled_state_exact", "rng_exact", "data_state_exact")
    )
    for rank in range(GPUS):
        proof = json.loads(
            (run.root / "audit" / f"restore-step{start}-rank{rank}.json").read_text()
        )
        assert all(proof[key] for key in fields)


def verify_smoke(run):
    validate_checkpoint(run.root / "step4", 4)
    check_restore(run, 0, True)
    check_restore(run, 2, False)
    metrics = [
        json.loads(line) for line in (run.root / "audit/metrics.jsonl").read_text().splitlines()
    ]
    for step in range(1, 5):
        rows = [row for row in metrics if row["step"] == step and "train/CE loss" in row]
        assert rows, ("missing measured SFT update", step)
        for row in rows:
            assert math.isfinite(row["train/CE loss"])
            assert math.isfinite(row["optim/total grad norm"]) and row["optim/total grad norm"] > 0
    validation = [
        json.loads(line) for line in (run.root / "audit/validation.jsonl").read_text().splitlines()
    ]
    assert validation and all(math.isfinite(row["ce_loss"]) for row in validation)


def main():
    verify_runtime()
    assert MOUNT.is_mount()
    r = find_run(sys.argv[1])
    assert int(os.environ["BEAKER_ASSIGNED_GPU_COUNT"]) == GPUS
    assert int(os.environ.get("BEAKER_REPLICA_COUNT", "1")) == 1
    if r.smoke and os.environ.get("HERO_SFT_DIAGNOSTIC") != "1":
        subprocess.run(
            [sys.executable, "src/examples/olmo_ddp/olmoe3_hero_sft.py", "--attention-smoke"],
            check=True,
        )
    target = 4 if r.smoke else r.total_steps
    if not r.smoke:
        for smoke in (s for s in runs(True) if s.arm == r.arm):
            gate = json.loads((smoke.root / "audit/sft-gate-success.json").read_text())
            assert gate["all_8_ranks_verified"] and gate["source_commit"] == os.environ["GIT_REF"]
    choices = [(0, r.source)] + [
        (int(p.name[4:]), p)
        for p in r.root.glob("step*")
        if p.name[4:].isdigit() and checkpoint_complete(p)
    ]
    start, source = max(choices, key=lambda item: item[0])
    assert 0 <= start <= target
    (validate_parent(source, 5961) if start == 0 else validate_checkpoint(source, start))
    stops = ([2] if r.smoke and start < 2 else []) + ([target] if start < target else [])
    for stop in stops:
        env = {
            **os.environ,
            "OLMO35_HERO_STOP": "4",
            "OLMO35_HERO_ALLOW_CONTINUATION": "1",
            "HERO_SFT_LOAD": str(source),
            "HERO_SFT_STOP": str(stop),
            "WANDB_RUN_ID": hashlib.sha256(r.run_id.encode()).hexdigest()[:8],
            "WANDB_RESUME": "allow",
        }
        log("SFT_PASS_START", run=r.run_id, start=start, stop=stop, source=str(source))
        subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                "--nproc-per-node=8",
                "--max-restarts=0",
                "src/examples/olmo_ddp/olmoe3_hero_sft.py",
                "train",
                r.run_id,
                "ai2/holmes",
            ],
            env=env,
            check=True,
        )
        if os.environ.get("HERO_SFT_DIAGNOSTIC") == "1":
            log("SFT_DIAGNOSTIC_COMPLETE", run=r.run_id)
            return
        validate_checkpoint(r.root / f"step{stop}", stop)
        check_restore(r, start, start == 0)
        start, source = stop, r.root / f"step{stop}"
    if r.smoke:
        verify_smoke(r)
    atomic_json(
        r.root / "audit" / ("sft-gate-success.json" if r.smoke else "sft-success.json"),
        dict(
            step=target,
            all_8_ranks_verified=True,
            source_commit=os.environ["GIT_REF"],
            experiment=os.environ["BEAKER_EXPERIMENT_ID"],
            **r.as_dict(),
        ),
    )
    log("SFT_COMPLETE", run=r.run_id, step=target)


if __name__ == "__main__":
    main()
