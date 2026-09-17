"""Durable native-only 4T decay -> MT -> LC -> SFT controller, independent of evals."""

import argparse
import copy
import fcntl
import json
import os
import subprocess
import sys
import time

from olmoe3_hero_decay_plan import (
    AUTOMATION as DECAY_AUTOMATION,
    ready,
    validate_checkpoint,
    verified_copy,
)
from olmoe3_hero_mt_plan import MTRun
from olmoe3_hero_lc_plan import LCRun
from olmoe3_hero_sft_plan import SFTRun, BRANCH, runs as sft_runs
from olmoe3_lr_sweep_watch import Controller, atomic_json, log, replace_env, status
from olmoe3_small_hero_plan import CONTROL, STATE, MOUNT, UPLOADER, WORKSPACE

AUTOMATION = DECAY_AUTOMATION / "native-pipeline"
# Preserve already-submitted MT/LC specs, receipts and their qualified runtime.
# Only the SFT recipe and controller change in this deployment.
MT_LC_COMMIT = "d031ab975c0dd11e986b3f017e0aac2e3608b521"


def controller(beaker, commit):
    c = object.__new__(Controller)
    c.beaker, c.workspace, c.commit = beaker, beaker.workspace.get(WORKSPACE), commit
    c.last_status, c.automation = {}, AUTOMATION
    return c


def recipe(stage, arm):
    return MTRun(arm) if stage == "mt" else LCRun(arm)


def native_spec(beaker, stage, run, commit, parent_id, phase="train"):
    """Build from the actual successful parent, preserving the pinned runtime."""
    parent = beaker.workload.get(parent_id)
    original = beaker.experiment.get_spec(parent).to_json()
    if stage == "mt":
        from olmoe3_hero_mt_control import training_spec

        return training_spec(original, run, MT_LC_COMMIT)
    if stage == "lc":
        from olmoe3_hero_lc_control import training_spec

        # Trainer progress updates replace the display description with prose.
        # Bind provenance to immutable task arguments/source, not that mutable text.
        assert parent.experiment.name == run.parent.run_id + "-train"
        assert all(
            t["arguments"]
            == ["python", "src/examples/olmo_ddp/olmoe3_hero_mt_node.py", run.parent.run_id]
            for t in original["tasks"]
        )
        original["description"] = json.dumps(dict(run_id=run.parent.run_id, posttrain_emo=False))
        return training_spec(original, run, MT_LC_COMMIT, phase)
    from olmoe3_hero_sft_control import training_spec

    return training_spec(original, run, commit)


def prep_spec(beaker, stage, run, commit, parent_id):
    s = native_spec(beaker, stage, run, commit, parent_id)
    t = s["tasks"][0]
    t.update(
        name=stage + "-prepare",
        replicas=1,
        leaderSelection=False,
        propagateFailure=False,
        propagatePreemption=False,
        timeout="8h",
    )
    t.pop("synchronizedStartTimeout", None)
    t["resources"] = dict(gpuCount=1, sharedMemory="16 GiB")
    t["arguments"] = [
        "python",
        "src/examples/olmo_ddp/olmoe3_hero_4t_pipeline.py",
        "--prepare",
        stage,
        "--arm",
        run.arm,
    ]
    replace_env(t, {"OLMO35_DECAY_CPU_VALIDATE": "1", "NUM_NODES": "1"})
    return s


def prepare(stage, arm):
    """Do large source copying and dataset validation on a separate allocated I/O worker."""
    from olmoe3_hero_decay_runtime import verify_runtime

    verify_runtime()
    assert MOUNT.is_mount()
    if stage in ("mt", "lc"):
        r = recipe(stage, arm)
        step = 240000 if stage == "mt" else 5961
        marker = "decay-success.json" if stage == "mt" else "mt-success.json"
        proof = json.loads((r.parent.root / "audit" / marker).read_text())
        assert proof["step"] == step and proof["all_64_ranks_verified"]
        assert ready(r.parent, step)
        verified_copy(r.parent.root / f"step{step}", r.source, step)
        validate_checkpoint(r.source, step)
        script = f"src/examples/olmo_ddp/olmoe3_hero_{stage}.py"
        subprocess.run([sys.executable, script, "--validate-only"], check=True)
        subprocess.run([sys.executable, script, "--data-probe"], check=True)
    else:
        with (AUTOMATION / "sft-prepare.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            subprocess.run(
                [sys.executable, "src/examples/olmo_ddp/olmoe3_hero_sft.py", "--prepare"],
                check=True,
                env=dict(os.environ, OLMO35_HERO_STOP="4"),
            )
    atomic_json(
        AUTOMATION / f"prepared-{stage}-{arm}.json",
        dict(passed=True, stage=stage, arm=arm, source_commit=os.environ["GIT_REF"]),
    )
    log("FOUR_T_STAGE_PREPARED", stage=stage, arm=arm)


def submitted(c, name):
    path = c.automation / "submissions" / (name + ".json")
    if not path.is_file():
        return None
    entry = json.loads(path.read_text())
    return c.beaker.workload.get(entry["experiment_id"]) if entry.get("experiment_id") else None


def complete(c, name):
    w = submitted(c, name)
    return w if w is not None and status(w) == "STATUS_SUCCEEDED" else None


def ensure(c, name, spec, snapshot):
    from beaker import BeakerExperimentSpec

    BeakerExperimentSpec.from_json(copy.deepcopy(spec))
    refs = {v["value"] for task in spec["tasks"] for v in task["envVars"] if v["name"] == "GIT_REF"}
    assert len(refs) == 1
    previous_commit = c.commit
    try:
        c.commit = refs.pop()
        assert c.commit in (previous_commit, MT_LC_COMMIT)
        w = c.ensure(name, spec)
    finally:
        c.commit = previous_commit
    snapshot[name] = dict(status=c.report(w), experiment=w.experiment.id if w else None)
    return w


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", choices=("mt", "lc", "sft"))
    p.add_argument("--arm", choices=("emo", "non-emo"))
    p.add_argument("--validate-configs", action="store_true")
    a = p.parse_args()
    if a.validate_configs:
        from olmoe3_hero_decay_runtime import verify_runtime

        verify_runtime()
        for stage in ("mt", "lc"):
            subprocess.run(
                [
                    sys.executable,
                    f"src/examples/olmo_ddp/olmoe3_hero_{stage}.py",
                    "--validate-only",
                ],
                check=True,
            )
        subprocess.run(
            [sys.executable, "src/examples/olmo_ddp/olmoe3_hero_sft.py", "--prepare"],
            check=True,
            env=dict(os.environ, OLMO35_HERO_STOP="4"),
        )
        for stage in ("decay", "mt", "lc", "sft"):
            subprocess.run(
                [
                    sys.executable,
                    "src/examples/olmo_ddp/olmoe3_hero_4t_evals.py",
                    "--tick",
                    stage,
                    "--validate-only",
                ],
                check=True,
            )
        atomic_json(
            AUTOMATION / ("config-success-" + os.environ["GIT_REF"] + ".json"),
            dict(passed=True, source_commit=os.environ["GIT_REF"]),
        )
        log("FOUR_T_ALL_CONFIGS_VALIDATED")
        return
    if a.prepare:
        assert a.arm
        return prepare(a.prepare, a.arm)
    from beaker import Beaker
    from olmo_checkpoint_uploader.models import Registration
    from olmo_checkpoint_uploader.state import StateStore

    assert MOUNT.is_mount()
    AUTOMATION.mkdir(parents=True, exist_ok=True)
    commit = os.environ["GIT_REF"]
    with (AUTOMATION / "LOCK").open("a") as lock, Beaker.from_env(check_for_upgrades=False) as b:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        c = controller(b, commit)
        store = StateStore(CONTROL, STATE)
        all_runs = (
            [r for arm in ("emo", "non-emo") for r in (MTRun(arm), LCRun(arm))]
            + sft_runs()
            + sft_runs(True)
        )
        assert len(sft_runs()) == 2 and all(not r.emo for r in all_runs)
        assert all(r.epochs == 2 and r.lr == 5e-5 for r in sft_runs())
        for r in all_runs:
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
            dict(
                commit=commit,
                branch=BRANCH,
                inference_numerical_gate="disabled_by_user_20260916",
                runs=[r.as_dict() for r in all_runs],
            ),
        )
        previous = None
        log(
            "FOUR_T_NATIVE_PIPELINE_ARMED",
            commit=commit,
            gpus=0,
            training_runs=6,
            sft_lrs=[5e-5],
            sft_epochs=2,
            sft_dataset="jacobmorrison/length-investigation-gptoss-120b-high",
            mt_lc_commit=MT_LC_COMMIT,
        )
        while True:
            snapshot = {}
            if (
                status(b.workload.get(UPLOADER)) != "STATUS_RUNNING"
                or os.statvfs(MOUNT).f_bavail * os.statvfs(MOUNT).f_frsize < 12_000_000_000_000
            ):
                log("FOUR_T_PIPELINE_WAIT_STORAGE_OR_UPLOADER")
                time.sleep(60)
                continue
            config_name = "olmo35-small-4t-config-check-cpu-20260916-" + commit[:8]
            entry = DECAY_AUTOMATION / "launched-emo.json"
            if entry.is_file():
                config_spec = prep_spec(
                    b, "mt", MTRun("emo"), commit, json.loads(entry.read_text())["experiment"]
                )
                from olmoe3_hero_lc_control import mount_lc

                mount_lc(config_spec["tasks"][0])
                t = config_spec["tasks"][0]
                replace_env(t, {"GIT_REF": commit, "GIT_BRANCH": BRANCH})
                # Config construction / packing validation needs no GPU. Avoid
                # consuming even one GPU from the gang-scheduled training pool.
                t.pop("resources", None)
                t.pop("hostNetworking", None)
                t["constraints"] = {"cluster": ["ai2/phobos"]}
                t["context"].update(minRuntime="0s", autoResume=True)
                if not any(d["mountPath"] == "/weka/oe-adapt-default" for d in t["datasets"]):
                    t["datasets"].append(
                        {
                            "mountPath": "/weka/oe-adapt-default",
                            "source": {"weka": "oe-adapt-default"},
                        }
                    )
                config_spec["tasks"][0]["arguments"] = [
                    "python",
                    "src/examples/olmo_ddp/olmoe3_hero_4t_pipeline.py",
                    "--validate-configs",
                ]
                ensure(c, config_name, config_spec, snapshot)
            if complete(c, config_name) is None:
                atomic_json(AUTOMATION / "status.json", dict(updated_at=time.time(), runs=snapshot))
                time.sleep(60)
                continue
            config_proof = json.loads(
                (AUTOMATION / ("config-success-" + commit + ".json")).read_text()
            )
            assert config_proof["passed"] and config_proof["source_commit"] == commit
            for arm in ("emo", "non-emo"):
                try:
                    entry = DECAY_AUTOMATION / f"launched-{arm}.json"
                    if not entry.exists():
                        snapshot[arm] = {"waiting": "decay_submission"}
                        continue
                    d = b.workload.get(json.loads(entry.read_text())["experiment"])
                    snapshot[arm] = {"decay": status(d), "experiment": d.experiment.id}
                    if status(d) != "STATUS_SUCCEEDED":
                        continue
                    parent = d
                    for stage in ("mt", "lc"):
                        r = recipe(stage, arm)
                        prep = r.run_id + "-prepare"
                        ensure(
                            c, prep, prep_spec(b, stage, r, commit, parent.experiment.id), snapshot
                        )
                        if complete(c, prep) is None:
                            break
                        proof = json.loads(
                            (AUTOMATION / f"prepared-{stage}-{arm}.json").read_text()
                        )
                        assert proof["passed"] and proof["source_commit"] == MT_LC_COMMIT
                        if stage == "lc":
                            name = r.run_id + "-smoke"
                            ensure(
                                c,
                                name,
                                native_spec(b, stage, r, commit, parent.experiment.id, "smoke"),
                                snapshot,
                            )
                            if complete(c, name) is None:
                                break
                            gate = json.loads((r.root / "audit/lc-gate-success.json").read_text())
                            assert gate["all_64_ranks_verified"] and gate["step"] == 4
                        name = r.run_id + "-train"
                        ensure(
                            c,
                            name,
                            native_spec(b, stage, r, commit, parent.experiment.id),
                            snapshot,
                        )
                        parent = complete(c, name)
                        if parent is None:
                            break
                        gate = json.loads((r.root / "audit" / f"{stage}-success.json").read_text())
                        assert gate["all_64_ranks_verified"] and gate["step"] == 5961
                    else:
                        # Per-arm progression: one slow lineage never blocks the other's SFTs.
                        smoke = SFTRun(arm, "5em5", True)
                        name = smoke.run_id + "-prepare"
                        ensure(
                            c,
                            name,
                            prep_spec(b, "sft", smoke, commit, parent.experiment.id),
                            snapshot,
                        )
                        if complete(c, name) is None:
                            continue
                        name = smoke.run_id + "-train"
                        ensure(
                            c,
                            name,
                            native_spec(b, "sft", smoke, commit, parent.experiment.id),
                            snapshot,
                        )
                        if complete(c, name) is None:
                            continue
                        gate = json.loads((smoke.root / "audit/sft-gate-success.json").read_text())
                        assert gate["all_8_ranks_verified"] and gate["source_commit"] == commit
                        for r in (x for x in sft_runs() if x.arm == arm):
                            ensure(
                                c,
                                r.run_id + "-train",
                                native_spec(b, "sft", r, commit, parent.experiment.id),
                                snapshot,
                            )
                except Exception as e:
                    snapshot[arm] = {"needs_attention": f"{type(e).__name__}: {e}"}
            atomic_json(AUTOMATION / "status.json", dict(updated_at=time.time(), runs=snapshot))
            if snapshot != previous:
                log("FOUR_T_NATIVE_STATUS", runs=snapshot)
                previous = snapshot
            if all(complete(c, r.run_id + "-train") is not None for r in sft_runs()):
                log("FOUR_T_NATIVE_TRAINING_COMPLETE")
                return
            time.sleep(60)


if __name__ == "__main__":
    main()
