"""One approved graceful restart of both heroes with the reduced checkpoint cadence.

Validate first, cancel through the existing W&B callback, verify each final full-state
checkpoint, then resume the same training/upload/W&B lineage exactly once. No deletion.
"""

import copy
import fcntl
import json
import os
import time

from olmoe3_lr_sweep_plan import checkpoint_complete
from olmoe3_lr_sweep_watch import Controller, atomic_json, log, replace_env, status
from olmoe3_small_hero_control import preflight, validation_spec
from olmoe3_small_hero_plan import AUTOMATION, BATCH, BRANCH, CAMPAIGN, runs

PREVIOUS_COMMIT = "171be9bef9517c12efbb517c0c4504e99d56e4e9"
PREVIOUS = {
    "emo": ("01M1ZB12N71WARP833J5M1MX00", "lasc1m2x"),
    "non-emo": ("01M1ZB164S5WB148A8ZNE9YM7Q", "aqb1droj"),
}


def resume_spec(original, run, commit, step, wandb_id):
    """Preserve the actual worker spec except source pin and explicit resume identity."""
    spec = copy.deepcopy(original)
    assert len(spec["tasks"]) == 8
    for task in spec["tasks"]:
        env = {v["name"]: v.get("value") for v in task["envVars"]}
        assert env["GIT_REF"] == PREVIOUS_COMMIT
        assert task["arguments"] == [
            "python",
            "src/examples/olmo_ddp/olmoe3_small_hero_node.py",
            run.run_id,
            "ai2/holmes",
        ]
        assert task["resources"]["gpuCount"] == 8
        assert task["context"]["priority"] == "urgent"
        assert task["context"]["minRuntime"] in ("1h", "1h0m0s", "3600s")
    task = copy.deepcopy(spec["tasks"][0])
    task.update(name="train", replicas=8, leaderSelection=True)
    replace_env(
        task,
        {
            "GIT_REF": commit,
            "GIT_BRANCH": BRANCH,
            "OLMO35_HERO_EXPECTED_START": str(step),
            "WANDB_RUN_ID": wandb_id,
            "WANDB_RESUME": "must",
        },
    )
    spec["tasks"] = [task]
    spec["retry"] = {"allowedTaskRetries": 0}
    spec["description"] = json.dumps(
        {
            **run.as_dict(),
            "cadence_update": "20260908",
            "resume_step": step,
            "replaces": PREVIOUS[run.arm][0],
            "wandb_run_id": wandb_id,
        }
    )
    return spec


def latest_complete(run):
    """Resolve the greatest complete step; never adopt a temp directory or stale audit."""
    candidates = [
        (int(path.name[4:]), path)
        for path in run.root.iterdir()
        if path.name.startswith("step") and path.name[4:].isdigit() and checkpoint_complete(path)
    ]
    assert candidates, f"No complete checkpoint for {run.run_id}"
    step, path = max(candidates)
    complete = json.loads((run.root / "audit" / f"complete-step{step}.json").read_text())
    assert complete["step"] == step and complete["tokens"] == step * BATCH
    assert not complete["storage_paused"] and not (run.root / "STORAGE_PAUSED.json").exists()
    for rank in range(64):
        audit = json.loads((path / "resume_audit" / f"rank{rank}.json").read_text())
        assert (audit["step"], audit["tokens"], audit["rank"], audit["gpus"]) == (
            step,
            step * BATCH,
            rank,
            64,
        )
        assert (path / "train" / f"rank{rank}.pt").is_file()
    return step, path


def main():
    """Perform one durable, fail-closed handoff of the two already-running heroes."""
    import wandb
    from beaker import Beaker

    commit = os.environ["GIT_REF"]
    receipt_path = AUTOMATION / "heroes-cadence-20260908.json"
    with (AUTOMATION / "LOCK").open("a") as lock, Beaker.from_env(
        check_for_upgrades=False
    ) as beaker:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = (
            json.loads(receipt_path.read_text())
            if receipt_path.exists()
            else {
                "source_commit": commit,
                "runs": {},
            }
        )
        assert receipt["source_commit"] == commit
        control = Controller(beaker, commit, automation=AUTOMATION)
        gate_name = f"{CAMPAIGN}-cadence-config-{commit[:8]}"
        gate_spec = validation_spec(control.template, commit)
        deadline = time.monotonic() + 1800
        while True:
            gate = control.ensure(gate_name, gate_spec)
            state = control.report(gate)
            if state == "STATUS_SUCCEEDED":
                break
            if state in ("STATUS_FAILED", "STATUS_CANCELED", "STATUS_STOPPED", "AMBIGUOUS"):
                raise RuntimeError(f"Cadence validation blocked: {state}")
            assert time.monotonic() < deadline, "Config validation timed out; heroes unchanged"
            time.sleep(20)
        assert (AUTOMATION / "fresh-bucket.json").is_file()
        preflight(beaker)
        originals = {}
        for run in runs():
            experiment, wandb_id = PREVIOUS[run.arm]
            workload = beaker.workload.get(experiment)
            originals[run.arm] = beaker.experiment.get_spec(workload).to_json()
            # Validate the spec and W&B identity BEFORE asking either trainer to stop.
            resume_spec(originals[run.arm], run, commit, 1, wandb_id)
            wb = wandb.Api(timeout=30).run(f"ai2-llm/olmo3p5-hero/{wandb_id}")
            assert wb.name == run.run_id, (wb.name, run.run_id)
            if run.arm not in receipt["runs"]:
                assert status(workload) == "STATUS_RUNNING"
                assert not {"cancel", "canceled", "cancelled"}.intersection(wb.tags)
                receipt["runs"][run.arm] = {
                    "previous_experiment": experiment,
                    "wandb_id": wandb_id,
                    "phase": "validated",
                    "initial_wandb_step": wb.summary.get("_step", 0),
                }
        atomic_json(receipt_path, receipt)
        for run in runs():
            entry = receipt["runs"][run.arm]
            if entry["phase"] in ("validated", "cancel_requested"):
                entry["phase"] = "cancel_requested"
                atomic_json(receipt_path, receipt)
                wb = wandb.Api(timeout=30).run(f"ai2-llm/olmo3p5-hero/{entry['wandb_id']}")
                if "cancel" not in wb.tags:
                    wb.tags = [*wb.tags, "cancel"]
                    wb.update()
                log("HERO_GRACEFUL_STOP_REQUESTED", run=run.run_id, wandb_id=entry["wandb_id"])
        deadline = time.monotonic() + 1800
        while True:
            states = {arm: status(beaker.workload.get(ids[0])) for arm, ids in PREVIOUS.items()}
            log("HERO_HANDOFF_WAIT", states=states)
            if all(value == "STATUS_SUCCEEDED" for value in states.values()):
                break
            assert not any(
                value in ("STATUS_FAILED", "STATUS_CANCELED") for value in states.values()
            ), states
            assert time.monotonic() < deadline, "Graceful stop timed out; no replacement submitted"
            time.sleep(20)
        # All old writers have exited. Require final checkpoint + all-rank audit before resume.
        for run in runs():
            entry = receipt["runs"][run.arm]
            if entry["phase"] == "submitted":
                continue
            step, path = latest_complete(run)
            assert step >= entry["initial_wandb_step"]
            entry.update(phase="checkpoint_verified", resume_step=step, resume_path=str(path))
            atomic_json(receipt_path, receipt)
            log("HERO_FINAL_CHECKPOINT_VERIFIED", run=run.run_id, step=step, path=str(path))
            wb = wandb.Api(timeout=30).run(f"ai2-llm/olmo3p5-hero/{entry['wandb_id']}")
            wb.tags = [tag for tag in wb.tags if tag != "cancel"]
            wb.update()
        preflight(beaker)
        for run in runs():
            entry = receipt["runs"][run.arm]
            if entry["phase"] == "submitted":
                continue
            workload = control.ensure(
                f"{run.run_id}-train-r3-cadence250",
                resume_spec(
                    originals[run.arm], run, commit, entry["resume_step"], entry["wandb_id"]
                ),
            )
            assert workload is not None, "Ambiguous resume submission; reconcile manually"
            entry.update(phase="submitted", experiment_id=workload.experiment.id)
            atomic_json(receipt_path, receipt)
            log("HERO_CADENCE_RESUME_SUBMITTED", run=run.run_id, **entry)
        log("BOTH_HERO_CADENCE_RESUMES_SUBMITTED", receipt=receipt)


if __name__ == "__main__":
    main()
