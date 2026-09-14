"""Zero-GPU submit-once controller: data gate -> two restart smokes -> six SFT trials."""

import copy
import fcntl
import json
import os
import time

from olmoe3_hero_sft_plan import (
    AUTOMATION,
    BRANCH,
    CAMPAIGN,
    CONTROL,
    LC_JOBS,
    MOUNT,
    STATE,
    UPLOADER,
    WORKSPACE,
    data_plan,
    runs,
)
from olmoe3_lr_sweep_watch import Controller, atomic_json, log, replace_env, status


class SFTController(Controller):
    def __init__(self, beaker, commit):
        self.beaker = beaker
        self.workspace = beaker.workspace.get(WORKSPACE)
        self.commit = commit
        self.last_status = {}
        self.automation = AUTOMATION


def training_spec(original, run, commit, prepare=False):
    """Keep the successful LC runtime/allowed hosts, one node, no result payload."""
    spec = copy.deepcopy(original)
    assert len(spec["tasks"]) == 8
    task = spec["tasks"][0]
    env = {v["name"]: v.get("value") for v in task["envVars"]}
    assert env["GANTRY_INSTALL_CMD"] == "true"
    assert env["GANTRY_POST_SETUP_CMD"] == "bash src/examples/olmo_ddp/olmoe3_hero_decay_setup.sh"
    assert task["resources"]["gpuCount"] == 8
    task.update(
        name="sft-prepare" if prepare else "sft",
        replicas=1,
        leaderSelection=False,
        timeout="3h" if prepare or run.smoke else "24h",
    )
    task.pop("synchronizedStartTimeout", None)
    task["propagateFailure"] = False
    task["propagatePreemption"] = False
    # Removed from Beaker after the LC jobs were submitted; preserve all other exclusions.
    task["constraints"]["hostname"] = [
        host for host in task["constraints"]["hostname"] if host != "holmes-cs-aus-520.reviz.ai2.in"
    ]
    task["arguments"] = (
        ["python", "src/examples/olmo_ddp/olmoe3_hero_sft.py", "--prepare"]
        if prepare
        else ["python", "src/examples/olmo_ddp/olmoe3_hero_sft_node.py", run.run_id]
    )
    task["context"].update(priority="urgent", minRuntime="1h", autoResume=True)
    # No source data writes: the only writable training artifacts use the dedicated mount.
    if not any(d["mountPath"] == "/weka/oe-adapt-default" for d in task["datasets"]):
        task["datasets"].append(
            {"mountPath": "/weka/oe-adapt-default", "source": {"weka": "oe-adapt-default"}}
        )
    if prepare:
        task.pop("resources", None)
        task["constraints"] = {"cluster": ["ai2/phobos"]}
        task["context"].update(minRuntime="0s", autoResume=False)
        task.pop("hostNetworking", None)
    replace_env(
        task,
        {
            "GIT_REF": commit,
            "GIT_BRANCH": BRANCH,
            "OLMO35_HERO_STOP": "4",
            "OLMO35_HERO_ALLOW_CONTINUATION": "1",
            "OLMO35_HERO_EXPECTED_START": "0",
            "OLMO35_HERO_SMOKE": "0",
            "OLMO35_LC_PHASE": None,
            "OLMO35_MT_LOAD": None,
            "OLMO35_LC_LOAD": None,
            "OLMO35_DECAY_LOAD": None,
            "HERO_SFT_LOAD": None,
            "HERO_SFT_STOP": None,
            "OLMO35_DECAY_CPU_VALIDATE": "1" if prepare else None,
            "WANDB_RUN_ID": None,
            "WANDB_RESUME": None,
            "NUM_NODES": "1",
            "GANTRY_TASK_NAME": "sft-prepare" if prepare else "sft",
        },
    )
    assert task.get("result", {}).get("path") in (None, "/noop-results")
    spec["tasks"] = [task]
    spec["retry"] = {"allowedTaskRetries": 0}
    spec["description"] = json.dumps(dict(campaign=CAMPAIGN, prepare=prepare, **run.as_dict()))
    return spec


def main():
    from beaker import Beaker, BeakerExperimentSpec
    from olmo_checkpoint_uploader.backend import HuggingFaceBucketBackend
    from olmo_checkpoint_uploader.models import Registration
    from olmo_checkpoint_uploader.state import StateStore

    assert MOUNT.is_mount()
    commit = os.environ["GIT_REF"]
    AUTOMATION.mkdir(parents=True, exist_ok=True)
    with (AUTOMATION / "LOCK").open("a") as lock, Beaker.from_env(check_for_upgrades=False) as b:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        control = SFTController(b, commit)
        gate_id = os.environ["HERO_SFT_CONFIG_GATE"]
        while status(b.workload.get(gate_id)) != "STATUS_SUCCEEDED":
            state = status(b.workload.get(gate_id))
            assert state not in ("STATUS_FAILED", "STATUS_CANCELED", "STATUS_STOPPED"), state
            log("SFT_WAITING_CONFIG_GATE", experiment=gate_id, status=state)
            time.sleep(60)
        assert data_plan()["source_commit"] == commit
        proof = json.loads((AUTOMATION / "config-success.json").read_text())
        assert proof["passed"] and proof["source_commit"] == commit
        for eid in LC_JOBS.values():
            assert status(b.workload.get(eid)) == "STATUS_SUCCEEDED"
        specs = {
            r.run_id: training_spec(
                b.experiment.get_spec(b.workload.get(LC_JOBS[r.arm])).to_json(), r, commit
            )
            for r in runs() + runs(True)
        }
        for spec in specs.values():
            BeakerExperimentSpec.from_json(copy.deepcopy(spec))
        store = StateStore(CONTROL, STATE)
        for bucket in {r.bucket for r in runs() + runs(True)}:
            HuggingFaceBucketBackend().assert_private(bucket)
        assert status(b.workload.get(UPLOADER)) == "STATUS_RUNNING"
        for r in runs() + runs(True):
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
            AUTOMATION / "plan.json",
            {
                "source_commit": commit,
                "data_plan": data_plan(),
                "runs": [r.as_dict() for r in runs()],
            },
        )
        previous = None
        while True:
            snapshot = {}
            fs = os.statvfs(MOUNT)
            storage_ok = fs.f_bavail * fs.f_frsize > 12_000_000_000_000
            uploader_ok = status(b.workload.get(UPLOADER)) == "STATUS_RUNNING"
            gates = []
            for r in runs(True):
                w = (
                    control.ensure(r.run_id + "-" + commit[:8], specs[r.run_id])
                    if storage_ok and uploader_ok
                    else None
                )
                state = control.report(w) if w else "waiting_for_storage_or_uploader"
                snapshot[r.run_id] = {"status": state, "experiment": w.experiment.id if w else None}
                passed = state == "STATUS_SUCCEEDED"
                if passed:
                    gate = json.loads((r.root / "audit/sft-gate-success.json").read_text())
                    assert gate["source_commit"] == commit and gate["all_8_ranks_verified"]
                gates.append(passed)
            for r in runs():
                if not all(gates) or not storage_ok or not uploader_ok:
                    snapshot[r.run_id] = {"status": "waiting_for_both_gpu_gates_or_storage"}
                    continue
                w = control.ensure(r.run_id + "-" + commit[:8], specs[r.run_id])
                state = control.report(w)
                row = {"status": state, "experiment": w.experiment.id if w else None}
                if state == "STATUS_SUCCEEDED":
                    proof = json.loads((r.root / "audit/sft-success.json").read_text())
                    assert (
                        proof["step"] == data_plan()["total_steps"]
                        and proof["all_8_ranks_verified"]
                    )
                    checkpoints = [data_plan()["steps_per_epoch"], data_plan()["total_steps"]]
                    row["uploads"] = {
                        str(step): bool(
                            (saved := store.load_checkpoint(r.run_id, step))
                            and saved.remote_verified
                        )
                        for step in checkpoints
                    }
                    row["validation"] = [
                        json.loads(line)
                        for line in (r.root / "audit/validation.jsonl").read_text().splitlines()
                    ]
                    from olmoe3_hero_sft_convert import advance_conversions

                    row["conversions"] = advance_conversions(b, control, r, commit)
                snapshot[r.run_id] = row
            atomic_json(AUTOMATION / "status.json", {"updated_at": time.time(), "runs": snapshot})
            if snapshot != previous:
                log("SFT_SWEEP_STATUS", runs=snapshot)
                previous = snapshot
            if all(
                snapshot[r.run_id].get("conversions", {}).get("complete")
                and all(snapshot[r.run_id].get("uploads", {}).values())
                for r in runs()
            ):
                log("SFT_SWEEP_COMPLETE", campaign=CAMPAIGN)
                return
            time.sleep(60)


if __name__ == "__main__":
    main()
