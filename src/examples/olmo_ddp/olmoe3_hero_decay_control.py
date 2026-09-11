"""CPU-only, durable two-hero fork watcher with immutable staging and bounded fan-out."""

import copy
import fcntl
import hashlib
import json
import os
import time

from olmoe3_hero_decay_plan import (
    AUTOMATION,
    BASE_COMMIT,
    BRANCH,
    END,
    PARENTS,
    START,
    ready,
    runs,
    verified_copy,
)
from olmoe3_lr_sweep_watch import Controller, atomic_json, log, replace_env, status
from olmoe3_small_hero_plan import CONTROL, MOUNT, STATE, UPLOADER


def training_name(run):
    """One explicitly authorized EMO repair; retain the failed attempt's durable receipt."""
    return run.run_id + ("-train-r2" if run.emo else "-train")


def training_spec(original, run, commit):
    """Clone the live arm's exact worker environment and topology; isolate all identities."""
    spec = copy.deepcopy(original)
    assert len(spec["tasks"]) == 8
    task = spec["tasks"][0]
    env = {v["name"]: v.get("value") for v in task["envVars"]}
    assert env["GIT_REF"] == BASE_COMMIT
    task.update(name="train", replicas=8, leaderSelection=True, timeout="720h")
    task["arguments"] = ["python", "src/examples/olmo_ddp/olmoe3_hero_decay_node.py", run.run_id]
    task["context"].update(priority="urgent", minRuntime="1h", autoResume=True)
    assert task["resources"]["gpuCount"] == 8
    assert task.get("result", {}).get("path") in (None, "/noop-results")
    replace_env(
        task,
        {
            "GIT_REF": commit,
            "GIT_BRANCH": BRANCH,
            "OLMO35_HERO_EXPECTED_START": str(START),
            "OLMO35_HERO_STOP": str(END),
            "OLMO35_HERO_ALLOW_CONTINUATION": "1",
            "OLMO35_HERO_SMOKE": "0",
            "WANDB_RUN_ID": None,
            "WANDB_RESUME": None,
            "OLMO35_DECAY_LOAD": None,
            "GANTRY_INSTALL_CMD": "true",
            "GANTRY_POST_SETUP_CMD": "bash src/examples/olmo_ddp/olmoe3_hero_decay_setup.sh",
            "OLMO35_DECAY_CPU_VALIDATE": None,
        },
    )
    spec["tasks"] = [task]
    spec["retry"] = {"allowedTaskRetries": 0}
    spec["description"] = json.dumps(
        {**run.as_dict(), "gate": "two-step save/full-state reload before continuation"}
    )
    return spec


def protect(store, run):
    """Temporarily prevent parent cleanup; restore its original policy only after verified copy."""
    path = AUTOMATION / "parent-policies" / f"{run.arm}.json"
    current = json.loads(store.registration_path(run.parent.run_id).read_text())
    assert current["lineage_id"] == run.parent.run_id
    assert current["checkpoint_root"] == str(run.parent.root)
    assert current["bucket_id"] == run.bucket and current["remote_prefix"] == run.arm
    if not path.exists():
        assert current["deletion_mode"] == "apply"
        atomic_json(path, current)
    original = json.loads(path.read_text())
    copied = run.source.with_name(run.source.name + "-copy.json").is_file()
    policy = {
        k: original[k] for k in ("deletion_mode", "delete_grace_seconds", "min_local_checkpoints")
    }
    if not copied:
        # Keep apply mode: the unchanged live hero validates it on an infrastructure resume.
        # This exceeds every scheduled checkpoint in the whole 14T run; effectively a hold.
        policy["min_local_checkpoints"] = 10_000
    if any(current[k] != value for k, value in policy.items()):
        store.set_lineage_deletion_policy(run.parent.run_id, **policy)
        log("PARENT_RETENTION_UPDATED", arm=run.arm, **policy, independent_copy_verified=copied)


def main():
    from beaker import Beaker, BeakerExperimentSpec
    from olmo_checkpoint_uploader.backend import HuggingFaceBucketBackend
    from olmo_checkpoint_uploader.models import Registration
    from olmo_checkpoint_uploader.state import StateStore
    from olmoe3_hero_decay_eval import advance_evals, eval_specs

    commit = os.environ["GIT_REF"]
    assert MOUNT.is_mount(), "No overlay writes"
    AUTOMATION.mkdir(parents=True, exist_ok=True)
    with (AUTOMATION / "LOCK").open("a") as lock, Beaker.from_env(check_for_upgrades=False) as b:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        store = StateStore(CONTROL, STATE)
        backend = HuggingFaceBucketBackend()
        backend.assert_private(runs()[0].bucket)
        control = Controller(b, commit, automation=AUTOMATION)
        originals = {
            arm: b.experiment.get_spec(b.workload.get(eid)).to_json()
            for arm, eid in PARENTS.items()
        }
        planned = {r.arm: training_spec(originals[r.arm], r, commit) for r in runs()}
        followups = {r.arm: eval_specs(b, r, commit) for r in runs()}
        for spec in [*planned.values(), *(s for plan in followups.values() for s in plan.values())]:
            BeakerExperimentSpec.from_json(copy.deepcopy(spec))
        assert status(b.workload.get(UPLOADER)) == "STATUS_RUNNING"
        for r in runs():
            protect(store, r)
            r.root.mkdir(parents=True, exist_ok=True)
            store.register(
                Registration(
                    run_id=r.run_id,
                    lineage_id=r.run_id,
                    checkpoint_root=str(r.root),
                    bucket_id=r.bucket,
                    remote_prefix=r.prefix,
                    deletion_mode="apply",
                    min_local_checkpoints=2,
                    delete_grace_seconds=3600,
                )
            )
        atomic_json(
            AUTOMATION / "plan.json", dict(source_commit=commit, runs=[r.as_dict() for r in runs()])
        )
        log(
            "DECAY_WATCHER_ARMED",
            source_commit=commit,
            fork_step=START,
            end_step=END,
            protected=[r.parent.run_id for r in runs()],
            gpu_resources=0,
        )
        previous = None
        while True:
            snapshot = {}
            for run in runs():
                # Copying is independent of training allocation and only ever targets the exact fork.
                if not run.source.with_name(run.source.name + "-copy.json").is_file():
                    if not ready(run.parent, START):
                        snapshot[run.arm] = dict(state="waiting_for_checkpoint", step=START)
                        continue
                    verified_copy(run.parent.root / f"step{START}", run.source, START)
                    protect(store, run)
                if status(b.workload.get(UPLOADER)) != "STATUS_RUNNING":
                    snapshot[run.arm] = dict(state="waiting_for_uploader")
                    continue
                if os.statvfs(MOUNT).f_bavail * os.statvfs(MOUNT).f_frsize < 12_000_000_000_000:
                    snapshot[run.arm] = dict(state="waiting_for_storage")
                    continue
                if run.emo:
                    assert status(b.workload.get("01M28S8ZSMXTQWG8N4PQYE2AH6")) == "STATUS_FAILED"
                work = control.ensure(training_name(run), planned[run.arm])
                state = control.report(work)
                snapshot[run.arm] = dict(
                    state=state, experiment=work.experiment.id if work else None
                )
                if state == "STATUS_SUCCEEDED":
                    assert ready(run, END)
                    gate = json.loads((run.root / "audit/decay-success.json").read_text())
                    assert gate["step"] == END and gate["all_64_ranks_verified"]
                    snapshot[run.arm]["evals"] = advance_evals(b, control, run, followups[run.arm])
                    uploaded = store.load_checkpoint(run.run_id, END)
                    snapshot[run.arm]["endpoint_upload_verified"] = bool(
                        uploaded and uploaded.remote_verified
                    )
                # Failed jobs are reported but never auto-copied or allowed to block the other arm.
            atomic_json(AUTOMATION / "status.json", dict(updated_at=time.time(), runs=snapshot))
            if snapshot != previous:
                log("DECAY_WATCHER_STATUS", runs=snapshot)
                previous = snapshot
            if (
                all(
                    row.get("evals", {}).get("complete") and row.get("endpoint_upload_verified")
                    for row in snapshot.values()
                )
                and len(snapshot) == 2
            ):
                log("BOTH_DECAYS_AND_EVALS_COMPLETE")
                return
            time.sleep(30)


if __name__ == "__main__":
    main()
