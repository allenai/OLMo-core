"""CPU-only, restartable MT controller; exact endpoint gates and bounded eval fan-out."""

import copy
import fcntl
import json
import os
import time

from olmoe3_hero_decay_plan import ready, validate_checkpoint, verified_copy
from olmoe3_hero_mt_eval import eval_specs, worker
from olmoe3_hero_mt_plan import (
    AUTOMATION,
    BRANCH,
    CAMPAIGN,
    CONTROL,
    DECAY_JOBS,
    END,
    MOUNT,
    PT_STEP,
    STATE,
    UPLOADER,
    WORKSPACE,
    runs,
)
from olmoe3_lr_sweep_watch import Controller, atomic_json, log, replace_env, status


def training_spec(original, run, commit):
    """Retain the repaired decay image, packages, env flags, topology and node exclusions."""
    spec = copy.deepcopy(original)
    assert len(spec["tasks"]) == 8
    task = spec["tasks"][0]
    env = {v["name"]: v.get("value") for v in task["envVars"]}
    assert env["GIT_REF"] == "36170c2d14272f62670235d176d7424ac85bebf8"
    assert env["GANTRY_INSTALL_CMD"] == "true"
    assert env["GANTRY_POST_SETUP_CMD"] == "bash src/examples/olmo_ddp/olmoe3_hero_decay_setup.sh"
    task.update(name="train", replicas=8, leaderSelection=True, timeout="720h")
    task["arguments"] = ["python", "src/examples/olmo_ddp/olmoe3_hero_mt_node.py", run.run_id]
    task["context"].update(priority="urgent", minRuntime="1h", autoResume=True)
    assert task["resources"]["gpuCount"] == 8
    assert task.get("result", {}).get("path") in (None, "/noop-results")
    replace_env(
        task,
        {
            "GIT_REF": commit,
            "GIT_BRANCH": BRANCH,
            "OLMO35_HERO_EXPECTED_START": "0",
            "OLMO35_HERO_STOP": str(END),
            "OLMO35_HERO_ALLOW_CONTINUATION": "1",
            "OLMO35_HERO_SMOKE": "0",
            "OLMO35_MT_LOAD": None,
            "OLMO35_DECAY_LOAD": None,
            "OLMO35_DECAY_CPU_VALIDATE": None,
            "WANDB_RUN_ID": None,
            "WANDB_RESUME": None,
        },
    )
    spec["tasks"] = [task]
    spec["retry"] = {"allowedTaskRetries": 0}
    spec["description"] = json.dumps(
        dict(
            campaign=CAMPAIGN,
            **run.as_dict(),
            runtime="identical repaired PT/decay",
            gate="2-step save/full-state reload",
        )
    )
    return spec


class MTController(Controller):
    def __init__(self, beaker, commit):
        self.beaker = beaker
        self.workspace = beaker.workspace.get(WORKSPACE)
        self.commit = commit
        self.last_status = {}
        self.automation = AUTOMATION


def main():
    from beaker import Beaker, BeakerExperimentSpec
    from olmo_checkpoint_uploader.backend import HuggingFaceBucketBackend
    from olmo_checkpoint_uploader.models import Registration
    from olmo_checkpoint_uploader.state import StateStore

    assert MOUNT.is_mount(), "Refuse overlay storage"
    commit = os.environ["GIT_REF"]
    AUTOMATION.mkdir(parents=True, exist_ok=True)
    with (AUTOMATION / "LOCK").open("a") as lock, Beaker.from_env(check_for_upgrades=False) as b:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert status(b.workload.get(os.environ["OLMO35_MT_GATE"])) == "STATUS_SUCCEEDED"
        control = MTController(b, commit)
        planned = {
            r.arm: training_spec(
                b.experiment.get_spec(b.workload.get(DECAY_JOBS[r.arm])).to_json(), r, commit
            )
            for r in runs()
        }
        followups = {r.arm: eval_specs(b, r, commit) for r in runs()}
        for spec in [
            *planned.values(),
            *(s for group in followups.values() for s in group.values()),
        ]:
            BeakerExperimentSpec.from_json(copy.deepcopy(spec))
        store = StateStore(CONTROL, STATE)
        backend = HuggingFaceBucketBackend()
        backend.assert_private(runs()[0].bucket)
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
        plan = dict(source_commit=commit, runs=[r.as_dict() for r in runs()])
        plan_path = AUTOMATION / "plan.json"
        if plan_path.exists():
            assert json.loads(plan_path.read_text()) == plan
        else:
            atomic_json(plan_path, plan)
        log("MT_WATCHER_ARMED", **plan, gpu_resources=0)
        previous = None
        while True:
            snapshot = {}
            for r in runs():
                try:
                    receipt = r.source.with_name(r.source.name + "-copy.json")
                    if not receipt.exists():
                        if not ready(r.parent, PT_STEP):
                            snapshot[r.arm] = dict(
                                state="waiting_for_decay_endpoint", parent=DECAY_JOBS[r.arm]
                            )
                            continue
                        verified_copy(r.parent.root / f"step{PT_STEP}", r.source, PT_STEP)
                    proof = json.loads(receipt.read_text())
                    assert proof["source"] == str(r.parent.root / f"step{PT_STEP}")
                    assert (
                        proof["destination"] == str(r.source) and proof["all_file_hashes_verified"]
                    )
                    validate_checkpoint(r.source, PT_STEP)
                    if status(b.workload.get(UPLOADER)) != "STATUS_RUNNING":
                        snapshot[r.arm] = dict(state="waiting_for_uploader")
                        continue
                    fs = os.statvfs(MOUNT)
                    if fs.f_bavail * fs.f_frsize < 12_000_000_000_000:
                        snapshot[r.arm] = dict(state="waiting_for_storage")
                        continue
                    work = control.ensure(r.run_id + "-train", planned[r.arm])
                    state = control.report(work)
                    row = dict(
                        state=state,
                        experiment=work.experiment.id if work else None,
                        source_staged=True,
                    )
                    if state == "STATUS_SUCCEEDED":
                        assert ready(r, END)
                        gate = json.loads((r.root / "audit/mt-success.json").read_text())
                        assert gate["step"] == END and gate["all_64_ranks_verified"]
                        row["evals"] = worker.advance_evals(b, control, r, followups[r.arm])
                        uploaded = store.load_checkpoint(r.run_id, END)
                        row["endpoint_upload_verified"] = bool(
                            uploaded and uploaded.remote_verified
                        )
                    snapshot[r.arm] = row
                except Exception as error:
                    snapshot[r.arm] = dict(
                        state="needs_attention", error=f"{type(error).__name__}: {error}"
                    )
            atomic_json(AUTOMATION / "status.json", dict(updated_at=time.time(), runs=snapshot))
            if snapshot != previous:
                log("MT_WATCHER_STATUS", runs=snapshot)
                previous = snapshot
            if all(
                row.get("evals", {}).get("complete") and row.get("endpoint_upload_verified")
                for row in snapshot.values()
            ):
                log("BOTH_MT_RUNS_UPLOADED_AND_EVALUATED")
                return
            time.sleep(30)


if __name__ == "__main__":
    main()
