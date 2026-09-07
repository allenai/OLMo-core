"""Explicitly gated, duplicate-safe medium CBS smoke and shared-prefix controller.

Smoke mode never launches production. Production requires a successful save/restore
smoke at the exact source pin and settings, and waits for the audited step4000 fork.
"""

import argparse
import copy
import fcntl
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from olmoe3_medium_cbs_plan import (
    AUTOMATION,
    BASELINE,
    BRANCH,
    CAMPAIGN,
    CONTROL,
    MOUNT,
    SMOKE_BASELINE,
    SMOKE_BRANCH,
    WORKSPACE,
    validate,
)
from olmoe3_medium_followup_plan import VARIANTS

BUCKET = "allenai/olmo-checkpoint-uploader-pilot-20260902-jm01"
UPLOADER = "01M1SDYA8S877VN54CJRN4RJZA"
BRANCH_NAME = "codex/medium-optimization-20260907"


def log(message, **fields):
    """Emit compact controller events directly to Beaker logs."""
    print(json.dumps({"time": time.time(), "event": message, **fields}), flush=True)


def atomic_json(path, value):
    """Persist an intent before any externally visible submission."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f".{os.getpid()}.tmp")
    with temp.open("w") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def status(workload):
    """Use Beaker's named status enum, never guessed integer values."""
    return (
        workload.DESCRIPTOR.fields_by_name["status"]
        .enum_type.values_by_number[workload.status]
        .name
    )


def replace_env(task, values):
    """Retain secret references without materializing their values."""
    env = {v["name"]: v for v in task["envVars"]}
    for key, value in values.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = {"name": key, "value": str(value)}
    task["envVars"] = list(env.values())


def checkpoint_complete(run, step):
    """Require finalized full state and every rank's post-save audit."""
    path = run.root / f"step{step}"
    if path.is_symlink() or not path.is_dir():
        return False
    if not all(
        (path / item).is_file()
        for item in (".metadata.json", "model_and_optim/.metadata", "train/rank0.pt")
    ):
        return False
    if json.loads((path / ".metadata.json").read_text()).get("ephemeral") is True:
        return False
    return all(
        (run.root / "audit" / f"state-step{step}-rank{r}.json").is_file() for r in range(128)
    )


def preflight(beaker, runs):
    """Check actual mounted capacity/private bucket/daemon before registering this campaign."""
    from olmo_checkpoint_uploader.backend import HuggingFaceBucketBackend
    from olmo_checkpoint_uploader.models import Registration
    from olmo_checkpoint_uploader.state import StateStore

    validate()
    assert MOUNT.is_mount(), "Checkpoint mount missing; refuse container overlay writes"
    fs = os.statvfs(MOUNT)
    free = fs.f_bavail * fs.f_frsize
    # Full no-deletion baseline+child+smoke footprint is approximately 12 TB.
    # On controller restart, already-written campaign checkpoints are part of
    # that budget, not an additional 12 TB requirement. Count only ordinary
    # files beneath this campaign's exact roots (never unrelated mount data).
    from olmoe3_medium_cbs_plan import RUNS

    occupied = 0
    for candidate in RUNS:
        if candidate.root.is_symlink():
            raise RuntimeError(f"Refuse symlink checkpoint root: {candidate.root}")
        if not candidate.root.is_dir():
            continue
        for directory in candidate.root.iterdir():
            if not directory.name.startswith("step") or not directory.name[4:].isdigit():
                continue
            if directory.is_symlink() or not directory.is_dir():
                continue
            for current, dirs, files in os.walk(directory, followlinks=False):
                dirs[:] = [d for d in dirs if not (Path(current) / d).is_symlink()]
                occupied += sum(
                    (Path(current) / f).stat().st_size
                    for f in files
                    if not (Path(current) / f).is_symlink()
                )
    required_free = max(1_000_000_000_000, 12_000_000_000_000 - occupied)
    assert free > required_free, f"Insufficient medium staging capacity: {free} < {required_free}"
    assert status(beaker.workload.get(UPLOADER)) == "STATUS_RUNNING", "Uploader is not running"
    HuggingFaceBucketBackend().assert_private(BUCKET)
    cache = MOUNT / "production-cbs/work/olmoe3-small-cbs-8mi-100b-lr1p3em3-uploader-r1"
    assert (cache / "global_indices_dataset_size1708983195_epoch1_seed928543231_v1.npy").is_file()
    store = StateStore(CONTROL, MOUNT / "uploader/state")
    for run in runs:
        run.root.mkdir(parents=True, exist_ok=True)
        registration = Registration(
            run_id=run.run_id,
            lineage_id=run.run_id,
            checkpoint_root=str(run.root),
            bucket_id=BUCKET,
            remote_prefix=f"runs/{run.run_id}",
            deletion_mode="apply",
            min_local_checkpoints=run.keep,
            delete_grace_seconds=3600,
        )
        created = store.register(registration)
        log("registration", run=run.run_id, created=created, keep=run.keep)
    log("capacity_gate", free_bytes=free, campaign_bytes=occupied, required_free=required_free)


def training_spec(template, run, *, commit, variant, mb, smoke=False):
    """Keep the qualified sixteen-node hardware/environment and replace orchestration only."""
    spec = copy.deepcopy(template)
    if len(spec["tasks"]) not in (1, 16):
        raise ValueError("Expected one replica spec or sixteen exported tasks")
    spec["tasks"] = [spec["tasks"][0]]
    task = spec["tasks"][0]
    task.update(name="train", replicas=16, leaderSelection=True)
    assert task["resources"]["gpuCount"] == 8
    task["arguments"] = [
        "python",
        "src/examples/olmo_ddp/olmoe3_medium_cbs_node.py",
        run.run_id,
        "ai2/holmes",
    ]
    replace_env(
        task,
        {
            "GIT_REF": commit,
            "GIT_BRANCH": BRANCH_NAME,
            "RESULTS_DIR": "/noop-results",
            "OLMOE3_MEDIUM_GPUS": 128,
            "OLMOE3_MEDIUM_MB": mb,
            "OLMOE3_MEDIUM_BATCH": run.batch,
            "OLMOE3_MEDIUM_DIAGNOSTIC": 0,
            "OLMOE3_DEEP_PROFILE_TEST": variant,
            "OLMOE3_MEDIUM_CBS_SMOKE": int(smoke),
            "OLMOE3_MEDIUM_CBS_RUN": run.run_id,
            "OLMOE3_MEDIUM_CBS_STOP": None,
            "OLMOE3_MEDIUM_CBS_EXPECTED_START": None,
            "OLMOE3_MEDIUM_CAPTURE": 0,
            "OLMOE3_DEEP_PROFILE_PLAN": None,
            "OLMOE3_MEDIUM_FOLLOWUP": None,
            "OLMOE3_BEAKER_WORKSPACE": WORKSPACE,
            "OLMOE3_WANDB_PROJECT": "olmoe3-production-cbs",
            "NCCL_PROTO": "Simple" if variant == "optimized-simple" else None,
        },
    )
    task["context"] = {"priority": "urgent", "minRuntime": "1h", "autoResume": False}
    task["timeout"] = "6h" if smoke else "48h"
    task["result"] = {"path": "/noop-results"}
    spec["retry"] = {"allowedTaskRetries": 0}
    spec["description"] = json.dumps(
        {"campaign": CAMPAIGN, "run": asdict(run), "variant": variant, "mb": mb, "smoke": smoke}
    )
    return spec


def ensure(beaker, workspace, name, spec):
    """Reconcile durable exact-name submission; ambiguity never triggers a new create."""
    from beaker import BeakerExperimentSpec

    path = AUTOMATION / "submissions" / f"{name}.json"
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    entry = json.loads(path.read_text()) if path.exists() else None
    if entry:
        assert entry["spec_sha256"] == digest, f"Spec changed for {name}"
        if entry.get("experiment_id"):
            return beaker.workload.get(entry["experiment_id"])
    matches = [
        w
        for w in beaker.workload.list(workspace=workspace, name_or_description=name, limit=100)
        if w.HasField("experiment") and w.experiment.name == name
    ]
    if matches:
        assert len(matches) == 1
        workload = matches[0]
        # No adoption without a prior matching durable intent.
        assert entry, f"Unexpected existing exact-name workload: {name}"
    elif entry:
        raise RuntimeError(f"Ambiguous submission must be reconciled, not retried: {name}")
    else:
        parsed = BeakerExperimentSpec.from_json(copy.deepcopy(spec))
        atomic_json(AUTOMATION / "specs" / f"{name}.json", spec)
        entry = {"phase": "submitting", "spec_sha256": digest, "name": name}
        atomic_json(path, entry)
        workload = beaker.experiment.create(spec=parsed, name=name, workspace=workspace)
    atomic_json(path, {**entry, "phase": "submitted", "experiment_id": workload.experiment.id})
    log("submitted_or_reconciled", name=name, experiment=workload.experiment.id)
    return workload


def main():
    """Run only the explicit smoke or production stage, with a single controller lock."""
    from beaker import Beaker

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "production"], required=True)
    parser.add_argument("--qualified-experiment", required=True)
    parser.add_argument("--config-gate", required=True)
    parser.add_argument("--smoke-experiment")
    parser.add_argument(
        "--variant",
        choices=[v for v in VARIANTS if v != "baseline"],
        required=True,
    )
    parser.add_argument("--mb", choices=[2, 4], type=int, required=True)
    args = parser.parse_args()
    commit = os.environ["GIT_REF"]
    assert MOUNT.is_mount()
    AUTOMATION.mkdir(parents=True, exist_ok=True)
    with (
        (AUTOMATION / "controller.lock").open("a") as lock,
        Beaker.from_env(check_for_upgrades=False) as b,
    ):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert status(b.workload.get(args.config_gate)) == "STATUS_SUCCEEDED"
        gate_spec = b.experiment.get_spec(b.workload.get(args.config_gate)).to_json()
        assert {
            v.get("value")
            for t in gate_spec["tasks"]
            for v in t["envVars"]
            if v["name"] == "GIT_REF"
        } == {commit}, "CBS config gate must match the exact source pin"
        assert status(b.workload.get(args.qualified_experiment)) == "STATUS_SUCCEEDED"
        template = b.experiment.get_spec(b.workload.get(args.qualified_experiment)).to_json()
        settings = {"commit": commit, "variant": args.variant, "mb": args.mb}
        if args.mode == "production":
            assert args.smoke_experiment
            assert status(b.workload.get(args.smoke_experiment)) == "STATUS_SUCCEEDED"
            gate = json.loads((AUTOMATION / "smoke-qualified.json").read_text())
            assert gate == {**settings, "experiment": args.smoke_experiment}
            assert checkpoint_complete(SMOKE_BRANCH, 4)
        runs = [SMOKE_BASELINE, SMOKE_BRANCH] if args.mode == "smoke" else [BASELINE, BRANCH]
        preflight(b, runs)
        workspace = b.workspace.get(WORKSPACE)
        atomic_json(
            AUTOMATION / f"plan-{args.mode}.json", {**settings, "runs": [asdict(r) for r in runs]}
        )
        parent = runs[0]
        workload = ensure(
            b,
            workspace,
            f"{CAMPAIGN}-save-restore-smoke" if args.mode == "smoke" else f"{parent.run_id}-train",
            training_spec(template, parent, **settings, smoke=args.mode == "smoke"),
        )
        child = None
        previous = None
        while True:
            workload = b.workload.get(workload.experiment.id)
            states = {"parent": status(workload)}
            if args.mode == "smoke" and states["parent"] == "STATUS_SUCCEEDED":
                assert checkpoint_complete(SMOKE_BRANCH, 4)
                assert all(
                    (SMOKE_BRANCH.root / "audit" / f"complete-step4-rank{r}.json").is_file()
                    for r in range(128)
                )
                atomic_json(
                    AUTOMATION / "smoke-qualified.json",
                    {**settings, "experiment": workload.experiment.id},
                )
                log("SMOKE_QUALIFIED", experiment=workload.experiment.id)
                return
            if (
                args.mode == "production"
                and child is None
                and checkpoint_complete(BASELINE, BRANCH.start)
            ):
                child = ensure(
                    b,
                    workspace,
                    f"{BRANCH.run_id}-train",
                    training_spec(template, BRANCH, **settings),
                )
            if child is not None:
                child = b.workload.get(child.experiment.id)
                states["child"] = status(child)
            if states != previous:
                log("STATUS", **states)
                atomic_json(AUTOMATION / "status.json", states)
                previous = states
            if any(s in ("STATUS_FAILED", "STATUS_CANCELED") for s in states.values()):
                raise RuntimeError(f"CBS job requires review; no automatic copies: {states}")
            if len(states) == 2 and all(s == "STATUS_SUCCEEDED" for s in states.values()):
                log("CBS_COMPLETE")
                return
            time.sleep(30)


if __name__ == "__main__":
    main()
