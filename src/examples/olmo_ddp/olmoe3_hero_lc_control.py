"""CPU LC controller: finished MT -> copied source -> 4-step GPU gate -> 100B -> evals."""

import copy
import fcntl
import json
import os
import time

from olmoe3_hero_decay_plan import ready, validate_checkpoint, verified_copy
from olmoe3_hero_lc_eval import advance_ruler, eval_specs, ruler_spec, worker
from olmoe3_hero_lc_plan import (
    AUTOMATION,
    BRANCH,
    CAMPAIGN,
    CONTROL,
    DEPLOYMENT,
    DEPLOYMENT_AUTOMATION,
    END,
    GATE_END,
    METADATA_CACHE,
    MOUNT,
    MT_JOBS,
    REPLACED_SMOKES,
    SOURCE_STEP,
    STATE,
    TRAINING_MOUNT,
    UPLOADER,
    WORKSPACE,
    runs,
)
from olmoe3_lr_sweep_watch import Controller, atomic_json, log, replace_env, status


class LCController(Controller):
    """Reuse submit-once intents without importing another stage's mutable eval adapter."""

    def __init__(self, beaker, commit):
        self.beaker = beaker
        self.workspace = beaker.workspace.get(WORKSPACE)
        self.commit = commit
        self.last_status = {}
        self.automation = DEPLOYMENT_AUTOMATION


def mount_lc(task):
    """Mount the existing shared LC data; worker code must never write this volume."""
    path = str(TRAINING_MOUNT)
    matches = [d for d in task["datasets"] if d["mountPath"] == path]
    if not matches:
        task["datasets"].append(dict(mountPath=path, source=dict(weka="oe-training-default")))
    else:
        assert len(matches) == 1 and matches[0]["source"] == dict(weka="oe-training-default")


def training_spec(original, run, commit, phase):
    """Retain the working MT image/environment/topology, changing only the LC wrapper."""
    assert phase in ("smoke", "train")
    spec = copy.deepcopy(original)
    assert len(spec["tasks"]) == 8
    task = spec["tasks"][0]
    env = {v["name"]: v.get("value") for v in task["envVars"]}
    assert env["GANTRY_INSTALL_CMD"] == "true"
    assert env["GANTRY_POST_SETUP_CMD"] == "bash src/examples/olmo_ddp/olmoe3_hero_decay_setup.sh"
    assert env["GIT_REF"] in (
        "f9ffa9b636fa51654892eb94c11f919d89dd7753",
        "05d8adba0e543b6e06d7592eb0b70171a2dd89bf",
    )
    task.update(
        name="lc-" + phase,
        replicas=8,
        leaderSelection=True,
        timeout="3h" if phase == "smoke" else "720h",
    )
    task["arguments"] = ["python", "src/examples/olmo_ddp/olmoe3_hero_lc_node.py", run.run_id]
    task["context"].update(priority="urgent", minRuntime="1h", autoResume=True)
    assert task["resources"]["gpuCount"] == 8
    assert task.get("result", {}).get("path") in (None, "/noop-results")
    mount_lc(task)
    replace_env(
        task,
        {
            "GIT_REF": commit,
            "GIT_BRANCH": BRANCH,
            "OLMO35_LC_PHASE": phase,
            "OLMO35_HERO_EXPECTED_START": "0",
            "OLMO35_HERO_STOP": str(GATE_END if phase == "smoke" else END),
            "OLMO35_HERO_ALLOW_CONTINUATION": "1",
            "OLMO35_HERO_SMOKE": "0",
            "OLMO35_MT_LOAD": None,
            "OLMO35_LC_LOAD": None,
            "OLMO35_DECAY_LOAD": None,
            "OLMO35_DECAY_CPU_VALIDATE": None,
            "WANDB_RUN_ID": None,
            "WANDB_RESUME": None,
            "CACHED_PATH_CACHE_ROOT": str(METADATA_CACHE),
        },
    )
    spec["tasks"] = [task]
    spec["retry"] = dict(allowedTaskRetries=0)
    spec["description"] = json.dumps(dict(campaign=CAMPAIGN, phase=phase, **run.as_dict()))
    return spec


def main():
    from beaker import Beaker, BeakerExperimentSpec
    from olmo_checkpoint_uploader.backend import HuggingFaceBucketBackend
    from olmo_checkpoint_uploader.models import Registration
    from olmo_checkpoint_uploader.state import StateStore

    assert MOUNT.is_mount()
    commit = os.environ["GIT_REF"]
    AUTOMATION.mkdir(parents=True, exist_ok=True)
    DEPLOYMENT_AUTOMATION.mkdir(parents=True, exist_ok=True)
    with (AUTOMATION / "LOCK").open("a") as lock, Beaker.from_env(check_for_upgrades=False) as b:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert status(b.workload.get(os.environ["OLMO35_LC_GATE"])) == "STATUS_SUCCEEDED"
        kernel_gate = b.workload.get(os.environ["OLMO35_LC_KERNEL_GATE"])
        assert status(kernel_gate) == "STATUS_SUCCEEDED"
        for gate_id in (os.environ["OLMO35_LC_GATE"], os.environ["OLMO35_LC_KERNEL_GATE"]):
            gate_spec = b.experiment.get_spec(b.workload.get(gate_id)).to_json()
            assert all(
                any(v["name"] == "GIT_REF" and v.get("value") == commit for v in t["envVars"])
                for t in gate_spec["tasks"]
            )
        control = LCController(b, commit)
        training = {
            r.arm: {
                phase: training_spec(
                    b.experiment.get_spec(b.workload.get(MT_JOBS[r.arm])).to_json(),
                    r,
                    commit,
                    phase,
                )
                for phase in ("smoke", "train")
            }
            for r in runs()
        }
        followups = {r.arm: eval_specs(b, r, commit) for r in runs()}
        rulers = {r.arm: ruler_spec(b, r, commit) for r in runs()}
        for spec in [
            *(s for group in training.values() for s in group.values()),
            *(s for group in followups.values() for s in group.values()),
            *rulers.values(),
        ]:
            BeakerExperimentSpec.from_json(copy.deepcopy(spec))
        store = StateStore(CONTROL, STATE)
        HuggingFaceBucketBackend().assert_private(runs()[0].bucket)
        assert status(b.workload.get(UPLOADER)) == "STATUS_RUNNING"
        for r in runs():
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
        plan = dict(
            source_commit=commit,
            deployment=DEPLOYMENT,
            replaced_smokes=REPLACED_SMOKES,
            kernel_gate=kernel_gate.experiment.id,
            runs=[r.as_dict() for r in runs()],
        )
        # Normalize tuples to JSON arrays before comparing a durable plan.
        plan = json.loads(json.dumps(plan))
        path = DEPLOYMENT_AUTOMATION / "plan.json"
        if path.exists():
            assert json.loads(path.read_text()) == plan
        else:
            assert all(status(b.workload.get(w)) == "STATUS_FAILED" for w in REPLACED_SMOKES)
            assert not any(
                p.is_dir() and p.name[4:].isdigit() for r in runs() for p in r.root.glob("step*")
            ), "Repair expected both original LC smokes to fail before saving any step"
            atomic_json(path, plan)
        previous = None
        log("LC_WATCHER_ARMED", **plan, gpu_resources=0)
        while True:
            snapshot = {}
            for r in runs():
                try:
                    parent_state = status(b.workload.get(MT_JOBS[r.arm]))
                    if parent_state != "STATUS_SUCCEEDED":
                        snapshot[r.arm] = dict(
                            state="waiting_for_finished_mt", parent_status=parent_state
                        )
                        continue
                    receipt = r.source.with_name(r.source.name + "-copy.json")
                    if not receipt.exists():
                        assert ready(r.parent, SOURCE_STEP)
                        parent_gate = json.loads(
                            (r.parent.root / "audit/mt-success.json").read_text()
                        )
                        assert (
                            parent_gate["step"] == SOURCE_STEP
                            and parent_gate["all_64_ranks_verified"]
                        )
                        verified_copy(r.parent.root / f"step{SOURCE_STEP}", r.source, SOURCE_STEP)
                    proof = json.loads(receipt.read_text())
                    assert proof["source"] == str(r.parent.root / f"step{SOURCE_STEP}")
                    assert (
                        proof["destination"] == str(r.source) and proof["all_file_hashes_verified"]
                    )
                    validate_checkpoint(r.source, SOURCE_STEP)
                    if status(b.workload.get(UPLOADER)) != "STATUS_RUNNING":
                        snapshot[r.arm] = dict(state="waiting_for_uploader")
                        continue
                    fs = os.statvfs(MOUNT)
                    if fs.f_bavail * fs.f_frsize < 12_000_000_000_000:
                        snapshot[r.arm] = dict(state="waiting_for_storage")
                        continue
                    smoke = control.ensure(
                        r.run_id + "-smoke-" + DEPLOYMENT, training[r.arm]["smoke"]
                    )
                    smoke_state = control.report(smoke)
                    row = dict(
                        state="gpu_gate_" + smoke_state,
                        smoke_experiment=smoke.experiment.id if smoke else None,
                    )
                    if smoke_state != "STATUS_SUCCEEDED":
                        snapshot[r.arm] = row
                        continue
                    gate = json.loads((r.root / "audit/lc-gate-success.json").read_text())
                    assert gate["step"] == GATE_END and gate["all_64_ranks_verified"]
                    assert gate["source_commit"] == commit
                    w = control.ensure(r.run_id + "-train-" + DEPLOYMENT, training[r.arm]["train"])
                    state = control.report(w)
                    row.update(state=state, experiment=w.experiment.id if w else None)
                    if state == "STATUS_SUCCEEDED":
                        assert ready(r, END)
                        success = json.loads((r.root / "audit/lc-success.json").read_text())
                        assert success["step"] == END and success["all_64_ranks_verified"]
                        row["evals"] = worker.advance_evals(b, control, r, followups[r.arm])
                        if row["evals"].get("qualify", {}).get("status") == "STATUS_SUCCEEDED":
                            row["ruler"] = advance_ruler(b, control, r, rulers[r.arm])
                        uploaded = store.load_checkpoint(r.run_id, END)
                        row["endpoint_upload_verified"] = bool(
                            uploaded and uploaded.remote_verified
                        )
                    snapshot[r.arm] = row
                except Exception as error:
                    snapshot[r.arm] = dict(
                        state="needs_attention", error=f"{type(error).__name__}: {error}"
                    )
            atomic_json(
                DEPLOYMENT_AUTOMATION / "status.json", dict(updated_at=time.time(), runs=snapshot)
            )
            if snapshot != previous:
                log("LC_WATCHER_STATUS", runs=snapshot)
                previous = snapshot
            if all(
                row.get("evals", {}).get("complete")
                and row.get("ruler", {}).get("complete")
                and row.get("endpoint_upload_verified")
                for row in snapshot.values()
            ):
                log("BOTH_LC_RUNS_UPLOADED_AND_EVALUATED")
                return
            time.sleep(30)


if __name__ == "__main__":
    main()
