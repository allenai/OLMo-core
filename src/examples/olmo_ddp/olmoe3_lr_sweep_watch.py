"""Restartable Beaker controller: register -> config gate -> smoke -> 5 trunks -> 20 decays.

Only this controller submits sweep jobs. An fsync'd intent precedes each submission.
An ambiguous submission is reconciled by exact name, never blindly retried. Failed
training experiments are reported, not automatically copied into a retry storm.
"""

import argparse
import copy
import fcntl
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from olmoe3_lr_sweep_plan import (
    AUTOMATION,
    BUCKET,
    CONTROL,
    DEPLOYMENT,
    EXTENSION_LABEL,
    MOUNT,
    QUALIFIED_EXPERIMENT,
    QUALIFIED_SMOKE,
    QUALIFIED_SMOKE_COMMIT,
    STATE,
    SWEEP,
    UPLOADER_EXPERIMENT,
    WORKSPACE,
    checkpoint_complete,
    extension_runs,
    runs,
    smoke_runs,
    validate_plan,
)


def log(message, **fields):
    """Write structured, immediately visible Beaker logs without credentials."""
    print(
        json.dumps(
            {"time": datetime.now(timezone.utc).isoformat(), "message": message, **fields},
            sort_keys=True,
        ),
        flush=True,
    )


def atomic_json(path, payload):
    """Durably publish controller state on the shared filesystem."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def status(workload):
    """Read the SDK's enum name, not a guessed integer status value."""
    return (
        workload.DESCRIPTOR.fields_by_name["status"]
        .enum_type.values_by_number[workload.status]
        .name
    )


def replace_env(task, values):
    """Replace named environment entries while preserving secret references."""
    env = {v["name"]: v for v in task["envVars"]}
    for name, value in values.items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = {"name": name, "value": str(value)}
    task["envVars"] = list(env.values())


def training_spec(template, run, commit, *, smoke=False):
    """Clone the actual qualified 64-GPU spec, changing only run orchestration."""
    spec = copy.deepcopy(template)
    # The Beaker export expands a replica group into eight independent-looking tasks.
    # Rebuild its original single synchronized replica group before resubmission.
    assert len(spec["tasks"]) == 8
    spec["tasks"] = [spec["tasks"][0]]
    spec["tasks"][0].update(name="train", replicas=8, leaderSelection=True)
    spec["description"] = json.dumps({"sweep": SWEEP, **run.as_dict()})
    spec["retry"] = {"allowedTaskRetries": 0}
    for task in spec["tasks"]:
        task["arguments"] = [
            "python",
            "src/examples/olmo_ddp/olmoe3_lr_sweep_node.py",
            run.run_id,
            "ai2/holmes",
        ]
        replace_env(
            task,
            {
                "GIT_REF": commit,
                "GIT_BRANCH": "codex/small-lr100b-sweep",
                "OLMOE3_LR_SWEEP_SMOKE": "1" if smoke else "0",
                "OLMOE3_INTEGRATION_BASELINE": "optimized100b",
                "OLMOE3_INTEGRATION_COMMUNICATION": "none",
                "OLMOE3_BEAKER_WORKSPACE": WORKSPACE,
                "RESULTS_DIR": "/noop-results",
            },
        )
        task["result"] = {"path": "/noop-results"}
        task["context"]["priority"] = "urgent"
        task["context"]["minRuntime"] = "1h"
    assert len(spec["tasks"]) == 1
    assert spec["tasks"][0]["replicas"] * spec["tasks"][0]["resources"]["gpuCount"] == 64
    return spec


def validation_spec(template, commit):
    """Validate all built configs in the actual production image, without GPUs."""
    t = copy.deepcopy(template["tasks"][0])
    t["name"] = "validate"
    t["arguments"] = ["python", "src/examples/olmo_ddp/olmoe3_small_lr_sweep.py", "--validate-only"]
    t["resources"] = {"cpuCount": 4, "memory": "16 GiB", "sharedMemory": "2 GiB"}
    t["context"] = {"priority": "urgent", "minRuntime": "0s", "autoResume": False}
    t["constraints"] = {"cluster": ["ai2/rhea"]}
    t["hostNetworking"] = False
    t["propagateFailure"] = False
    t["propagatePreemption"] = False
    t.pop("synchronizedStartTimeout", None)
    t["timeout"] = "30m"
    t["result"] = {"path": "/noop-results"}
    replace_env(
        t,
        {
            "GIT_REF": commit,
            "GIT_BRANCH": "codex/small-lr100b-sweep",
            "GANTRY_TASK_NAME": "validate",
            "RESULTS_DIR": "/noop-results",
            "NUM_NODES": "1",
            "GANTRY_INSTALL_CMD": "true",
            "GANTRY_POST_SETUP_CMD": "gh auth setup-git && uv pip install --python "
            '"$(command -v python)" --no-deps \'kernel-fun @ git+https://github.com/'
            "allenai/kernel-fun.git@7a6983baf2beb4ec4d7fe914ec9f6670438af99b"
            "#subdirectory=packages/kernel-fun'",
        },
    )
    return {"version": "v2", "tasks": [t], "retry": {"allowedTaskRetries": 0}}


def preflight_and_register(beaker, *, items=None, automation=None, minimum_free=28_000_000_000_000):
    """Audit live capacity/bucket/daemon and explicitly register only this sweep."""
    from olmo_checkpoint_uploader.backend import HuggingFaceBucketBackend
    from olmo_checkpoint_uploader.models import Registration
    from olmo_checkpoint_uploader.state import StateStore

    validate_plan()
    assert MOUNT.is_mount(), "Refuse to write checkpoints on a container overlay"
    fs = os.statvfs(MOUNT)
    free = fs.f_bavail * fs.f_frsize
    assert free > minimum_free, f"Insufficient worst-case staging capacity: {free}"
    assert status(beaker.workload.get(UPLOADER_EXPERIMENT)) == "STATUS_RUNNING"
    backend = HuggingFaceBucketBackend()
    backend.assert_private(BUCKET)
    store = StateStore(CONTROL, STATE)
    cache = (
        MOUNT
        / "production-cbs/work/olmoe3-small-cbs-8mi-100b-lr1p3em3-uploader-r1"
        / "global_indices_dataset_size1708983195_epoch1_seed928543231_v1.npy"
    )
    assert cache.is_file(), f"Missing qualified shared data order: {cache}"
    items = runs() + smoke_runs() if items is None else items
    automation = AUTOMATION if automation is None else automation
    for r in items:
        r.root.mkdir(parents=True, exist_ok=True)
        registration = Registration(
            run_id=r.run_id,
            lineage_id=r.run_id,
            checkpoint_root=str(r.root),
            bucket_id=BUCKET,
            remote_prefix=f"runs/{r.run_id}",
            deletion_mode="apply",
            min_local_checkpoints=r.keep,
            delete_grace_seconds=3600,
        )
        created = store.register(registration)
        log("registration ready", run_id=r.run_id, created=created, keep=r.keep)
    atomic_json(automation / "plan.json", [r.as_dict() for r in items])
    log("PREFLIGHT_PASSED", free_bytes=free, bucket=BUCKET, uploader=UPLOADER_EXPERIMENT)


class Controller:
    """A single-lock, durable, exact-name reconciled controller."""

    def __init__(self, beaker, commit, *, automation=None, items=None, reuse_smoke=False):
        self.beaker = beaker
        self.workspace = beaker.workspace.get(WORKSPACE)
        self.commit = commit
        self.last_status = {}
        self.automation = AUTOMATION if automation is None else automation
        self.items = runs() if items is None else items
        self.reuse_smoke = reuse_smoke
        self.smoke_verified = False
        template_path = self.automation / "qualified-template.json"
        if not template_path.exists():
            original = beaker.experiment.get_spec(beaker.workload.get(QUALIFIED_EXPERIMENT))
            atomic_json(template_path, original.to_json())
        self.template = json.loads(template_path.read_text())

    def ensure(self, name, spec):
        """Submit once; a lost create response is never grounds for another submission."""
        from beaker import BeakerExperimentSpec

        path = self.automation / "submissions" / f"{name}.json"
        digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        entry = json.loads(path.read_text()) if path.exists() else None
        if entry:
            assert entry["spec_sha256"] == digest, f"Spec drift for {name}"
            if entry.get("experiment_id"):
                return self.beaker.workload.get(entry["experiment_id"])
        matches = [
            w
            for w in self.beaker.workload.list(
                workspace=self.workspace, name_or_description=name, limit=100
            )
            if w.HasField("experiment") and w.experiment.name == name
        ]
        if len(matches) > 1:
            raise RuntimeError(f"Duplicate exact-name experiments require review: {name}")
        if matches:
            w = matches[0]
            # Adopt only this source/run, never an unrelated same-name workload.
            actual = self.beaker.experiment.get_spec(w).to_json()
            refs = [
                v.get("value")
                for t in actual["tasks"]
                for v in t["envVars"]
                if v["name"] == "GIT_REF"
            ]
            assert refs and all(v == self.commit for v in refs)
        elif entry:
            log("AMBIGUOUS_SUBMISSION_REQUIRES_RECONCILIATION", name=name)
            return None
        else:
            # The SDK consumes dictionary entries while parsing; preserve the durable spec.
            parsed = BeakerExperimentSpec.from_json(copy.deepcopy(spec))
            atomic_json(self.automation / "specs" / f"{name}.json", spec)
            atomic_json(
                path, {"phase": "submitting", "spec_sha256": digest, "source_commit": self.commit}
            )
            w = self.beaker.experiment.create(spec=parsed, name=name, workspace=self.workspace)
            log("EXPERIMENT_SUBMITTED", name=name, experiment_id=w.experiment.id)
        atomic_json(
            path,
            {
                "phase": "submitted",
                "spec_sha256": digest,
                "source_commit": self.commit,
                "experiment_id": w.experiment.id,
            },
        )
        return w

    def report(self, workload):
        if workload is None:
            return "AMBIGUOUS"
        s = status(workload)
        name = workload.experiment.name
        if self.last_status.get(name) != s:
            log("experiment status", name=name, experiment_id=workload.experiment.id, status=s)
            self.last_status[name] = s
        return s

    def tick(self):
        """Gate production fan-out on config validation and save/restore smoke success."""
        gate_suffix = (
            f"{DEPLOYMENT}-extension-{EXTENSION_LABEL}" if self.reuse_smoke else DEPLOYMENT
        )
        gate = self.ensure(
            f"{SWEEP}-config-validation-{gate_suffix}", validation_spec(self.template, self.commit)
        )
        if self.report(gate) != "STATUS_SUCCEEDED":
            return
        parent, child = smoke_runs()
        if not self.smoke_verified:
            if self.reuse_smoke:
                smoke = self.beaker.workload.get(QUALIFIED_SMOKE)
                actual = self.beaker.experiment.get_spec(smoke).to_json()
                refs = [
                    v.get("value")
                    for t in actual["tasks"]
                    for v in t["envVars"]
                    if v["name"] == "GIT_REF"
                ]
                assert len(refs) == 8 and set(refs) == {QUALIFIED_SMOKE_COMMIT}
            else:
                smoke = self.ensure(
                    f"{SWEEP}-save-restore-smoke-{DEPLOYMENT}",
                    training_spec(self.template, parent, self.commit, smoke=True),
                )
            if self.report(smoke) != "STATUS_SUCCEEDED":
                return
            assert checkpoint_complete(child.root / "step6")
            sessions = [
                json.loads(p.read_text()) for p in (child.root / "audit").glob("session-*.json")
            ]
            assert {2, 4}.issubset({p["resumed_step"] for p in sessions})
            self.smoke_verified = True
            log("SAVE_RESTORE_GATE_PASSED", reused=self.reuse_smoke)
        states = {}
        for r in [r for r in self.items if not r.parent]:
            w = self.ensure(f"{r.run_id}-train", training_spec(self.template, r, self.commit))
            states[r.run_id] = self.report(w)
        for r in [r for r in self.items if r.parent]:
            if states[r.parent] != "STATUS_SUCCEEDED":
                continue
            trunk = next(p for p in self.items if p.run_id == r.parent)
            # Validate all four forks before launching any child of this trunk.
            assert (trunk.root / "audit/completed-step6000.json").is_file()
            for sibling in [c for c in self.items if c.parent == r.parent]:
                assert checkpoint_complete(sibling.parent_path), sibling.as_dict()
            w = self.ensure(f"{r.run_id}-train", training_spec(self.template, r, self.commit))
            states[r.run_id] = self.report(w)
        atomic_json(self.automation / "status.json", {"source_commit": self.commit, "runs": states})
        if len(states) == len(self.items) and all(s == "STATUS_SUCCEEDED" for s in states.values()):
            log("SWEEP_COMPLETE", runs=len(self.items))
            return True
        return False


def main():
    """Stay alive independently of trainers and the interactive session."""
    from beaker import Beaker

    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--extension", choices=[EXTENSION_LABEL])
    args = parser.parse_args()
    assert MOUNT.is_mount()
    automation = AUTOMATION / f"extension-{args.extension}" if args.extension else AUTOMATION
    items = extension_runs() if args.extension else runs()
    automation.mkdir(parents=True, exist_ok=True)
    with (automation / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with Beaker.from_env(default_workspace=WORKSPACE, check_for_upgrades=False) as beaker:
            preflight_and_register(
                beaker,
                items=items if args.extension else items + smoke_runs(),
                automation=automation,
                # One extra LR's full no-deletion footprint is <6 TB; allow 7 TB.
                minimum_free=7_000_000_000_000 if args.extension else 28_000_000_000_000,
            )
            controller = Controller(
                beaker,
                os.environ["GIT_REF"],
                automation=automation,
                items=items,
                reuse_smoke=bool(args.extension),
            )
            heartbeat = 0.0
            while True:
                try:
                    if controller.tick():
                        return
                except Exception as exc:  # noqa: BLE001 - log service errors; never blind-resubmit
                    log("WATCHER_ERROR_NO_BLIND_RELAUNCH", error=repr(exc))
                if args.once:
                    return
                if time.monotonic() - heartbeat >= 300:
                    log("watcher heartbeat", sweep=SWEEP)
                    heartbeat = time.monotonic()
                time.sleep(60)


if __name__ == "__main__":
    main()
