"""4T LR-only expansion: 14 new trials, with final conversion and canonical evals."""

import argparse
import copy
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time

from olmoe3_hero_sft_plan import (
    AUTOMATION,
    BATCH,
    BRANCH,
    CONTROL,
    DATA,
    DATA_PLAN,
    GPUS,
    MOUNT,
    SEQUENCE,
    SMOKE_GATE_COMMIT,
    STATE,
    UPLOADER,
    WORKSPACE,
    SFTRun,
    data_plan,
    new_lr_runs,
    runs,
)
from olmoe3_lr_sweep_watch import Controller, atomic_json, log, replace_env, status

DEPLOYMENT = AUTOMATION / "lr-sweep-20260918"
EVAL_WORKSPACE = "ai2/OLMo-3-moe-experiments"


def train_spec(original, run, commit):
    """Clone a successful 4T SFT job, changing only source and LR-run identity."""
    spec = copy.deepcopy(original)
    assert len(spec["tasks"]) == 1 and not run.smoke and run.lr_label != "5em5"
    task = spec["tasks"][0]
    env = {x["name"]: x.get("value") for x in task["envVars"]}
    assert env["GIT_REF"] == SMOKE_GATE_COMMIT
    assert task["arguments"] == [
        "python",
        "src/examples/olmo_ddp/olmoe3_hero_sft_node.py",
        SFTRun(run.arm, "5em5").run_id,
    ]
    assert task["resources"]["gpuCount"] == GPUS
    assert task["context"]["priority"] == "urgent"
    assert task["context"]["minRuntime"] == 3600000000000
    assert task["context"]["autoResume"] and task["result"]["path"] == "/noop-results"
    task["arguments"][-1] = run.run_id
    replace_env(task, {"GIT_REF": commit, "GIT_BRANCH": BRANCH})
    spec["description"] = json.dumps(dict(stage="4t-two-epoch-sft-lr-sweep", **run.as_dict()))
    return spec


def config_gate():
    """Construct real configs without allocating models or repacking the dataset."""
    from olmoe3_hero_decay_runtime import verify_runtime
    from olmoe3_hero_sft import config_builder
    from olmo_core.internal.experiment import CliContext, SubCmd

    verify_runtime()
    assert MOUNT.is_mount() and data_plan()["steps_per_epoch"] == 1680
    manifest = json.loads((DATA / "manifest.json").read_text())
    assert manifest["dataset"] == "jacobmorrison/length-investigation-gptoss-120b-high"
    assert manifest["dataset_revision"] == "2fa53f4df6e4e41f9202c31cc8e26b2a04bce027"
    assert (
        hashlib.sha256((DATA / "manifest.json").read_bytes()).hexdigest()
        == data_plan()["manifest_sha256"]
    )
    checked = []
    for run in new_lr_runs():
        baseline = SFTRun(run.arm, "5em5")
        cfg = config_builder()(CliContext(__file__, SubCmd.dry_run, run.run_id, "ai2/holmes", []))
        current = json.loads(json.dumps(cfg.as_dict(json_safe=True)))
        old = json.loads((AUTOMATION / "configs" / (baseline.run_id + ".json")).read_text())
        assert current["model"] == old["model"]
        module = copy.deepcopy(current["train_module"])
        assert module["optim"]["lr"] == run.lr
        module["optim"]["lr"] = baseline.lr
        assert module == old["train_module"], "Non-LR training change"
        for section in ("dataset", "data_loader"):
            normalized = json.loads(
                json.dumps(current[section]).replace(run.run_id, baseline.run_id)
            )
            assert normalized == old[section], section
        assert cfg.model.num_active_params == 794233472 and cfg.model.num_params == 12496341632
        assert cfg.data_loader.global_batch_size == BATCH
        assert cfg.train_module.rank_microbatch_size == SEQUENCE
        assert not cfg.trainer.load_optim_state and not cfg.trainer.load_trainer_state
        assert cfg.trainer.load_path == str(run.source)
        assert cfg.trainer.hard_stop.value == run.total_steps == 3360
        assert cfg.trainer.callbacks["checkpointer"].fixed_steps == [1680, 3360]
        assert cfg.trainer.max_duration.value == 2
        smoke = next(r for r in runs(True) if r.arm == run.arm)
        proof = json.loads((smoke.root / "audit/sft-gate-success.json").read_text())
        assert proof["all_8_ranks_verified"] and proof["source_commit"] == SMOKE_GATE_COMMIT
        assert proof["source"] == str(run.source)
        atomic_json(DEPLOYMENT / "configs" / (run.run_id + ".json"), current)
        checked.append(run.run_id)
    atomic_json(
        DEPLOYMENT / "config-gate.json",
        dict(
            passed=True,
            commit=os.environ["GIT_REF"],
            runs=checked,
            reused_gpu_gate=SMOKE_GATE_COMMIT,
            data_plan_sha256=hashlib.sha256(DATA_PLAN.read_bytes()).hexdigest(),
        ),
    )
    log("SFT_LR_CONFIG_GATE_PASSED", runs=checked)


def control(beaker, commit, workspace, directory):
    """Use durable exact-name submission intents in an isolated namespace."""
    c = object.__new__(Controller)
    c.beaker, c.commit, c.workspace = beaker, commit, beaker.workspace.get(workspace)
    c.automation, c.last_status = directory, {}
    return c


def tick(beaker, commit):
    """Advance each new trial independently; never retry failed jobs under new names."""
    from beaker import BeakerExperimentSpec
    from olmo_checkpoint_uploader.backend import HuggingFaceBucketBackend
    from olmo_checkpoint_uploader.models import Registration
    from olmo_checkpoint_uploader.state import StateStore
    from olmoe3_hero_sft_convert import export_root, spec_for as convert_spec
    from olmoe3_hero_sft_eval_control import BUNDLES, spec_for as eval_spec
    from olmoe3_hero_decay_eval import TEMPLATES
    from olmoe3_hero_4t_eval_policy import validate_export

    gate_id = os.environ["HERO_SFT_LR_CONFIG_GATE"]
    gate_status = status(beaker.workload.get(gate_id))
    if gate_status != "STATUS_SUCCEEDED":
        log("SFT_LR_WAITING_GATE", experiment=gate_id, status=gate_status)
        return
    gate = json.loads((DEPLOYMENT / "config-gate.json").read_text())
    assert gate["passed"] and gate["commit"] == commit
    assert set(gate["runs"]) == {r.run_id for r in new_lr_runs()}
    train = control(beaker, commit, WORKSPACE, DEPLOYMENT / "training")
    evaluate = control(beaker, commit, EVAL_WORKSPACE, DEPLOYMENT / "evals")
    store = StateStore(CONTROL, STATE)
    fs = os.statvfs(MOUNT)
    free = fs.f_bavail * fs.f_frsize
    uploader_ok = status(beaker.workload.get(UPLOADER)) == "STATUS_RUNNING"
    eval_template = beaker.experiment.get_spec(beaker.workload.get(TEMPLATES["gen_mc"])).to_json()
    parents = {}
    for baseline in runs():
        proof = json.loads((baseline.root / "audit/sft-success.json").read_text())
        assert proof["step"] == 3360 and proof["all_8_ranks_verified"]
        assert proof["source_commit"] == SMOKE_GATE_COMMIT
        w = beaker.workload.get(proof["experiment"])
        assert status(w) == "STATUS_SUCCEEDED"
        parents[baseline.arm] = beaker.experiment.get_spec(w).to_json()
    snapshot = {}
    if not (DEPLOYMENT / "bucket-checked.json").exists():
        HuggingFaceBucketBackend().assert_private(new_lr_runs()[0].bucket)
        atomic_json(DEPLOYMENT / "bucket-checked.json", dict(passed=True))
    for run in new_lr_runs():
        row = snapshot[run.run_id] = {}
        spec = train_spec(parents[run.arm], run, commit)
        BeakerExperimentSpec.from_json(copy.deepcopy(spec))
        receipt = train.automation / "submissions" / (run.run_id + "-train.json")
        if not receipt.exists() and (free < 12_000_000_000_000 or not uploader_ok):
            row["waiting"] = "storage_or_uploader"
            continue
        run.root.mkdir(parents=True, exist_ok=True)
        store.register(
            Registration(
                run_id=run.run_id,
                lineage_id=run.run_id,
                checkpoint_root=str(run.root),
                bucket_id=run.bucket,
                remote_prefix=run.prefix,
                deletion_mode="apply",
                min_local_checkpoints=2,
                delete_grace_seconds=3600,
            )
        )
        w = train.ensure(run.run_id + "-train", spec)
        row["train"] = dict(status=train.report(w), experiment=w.experiment.id if w else None)
        if w is None or status(w) != "STATUS_SUCCEEDED":
            continue
        proof = json.loads((run.root / "audit/sft-success.json").read_text())
        assert proof["experiment"] == w.experiment.id and proof["step"] == 3360
        assert proof["source_commit"] == commit and proof["all_8_ranks_verified"]
        if free < 10_000_000_000_000:
            row["waiting"] = "conversion_free_space"
            continue
        conversion = convert_spec(beaker, run, run.total_steps, commit)
        conv = evaluate.ensure(run.run_id + "-step3360-convert", conversion)
        row["convert"] = dict(
            status=evaluate.report(conv), experiment=conv.experiment.id if conv else None
        )
        if conv is None or status(conv) != "STATUS_SUCCEEDED":
            continue
        model = export_root(run) / run.arm / "step3360/hf"
        validate_export(model)
        assert json.loads((model / "sft-metadata-audit.json").read_text())["passed"]
        for bundle in BUNDLES:
            spec = eval_spec(eval_template, bundle, run, commit)
            replace_env(spec["tasks"][0], {"HERO_SFT_TEMPERATURE": "0.6"})
            ew = evaluate.ensure(run.run_id + "-epoch2-" + bundle, spec)
            row[bundle] = dict(
                status=evaluate.report(ew), experiment=ew.experiment.id if ew else None
            )
            if ew is not None and status(ew) == "STATUS_SUCCEEDED":
                done = json.loads(
                    (model.parent / "posttrain-evals-r1" / bundle / "success.json").read_text()
                )
                assert done["passed"] and done["bundle"] == bundle
    atomic_json(DEPLOYMENT / "status.json", dict(at=time.time(), free_bytes=free, runs=snapshot))
    log("SFT_LR_STATUS", runs=snapshot)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-gate", action="store_true")
    p.add_argument("--tick", action="store_true")
    args = p.parse_args()
    if args.config_gate:
        return config_gate()
    from beaker import Beaker

    assert MOUNT.is_mount()
    DEPLOYMENT.mkdir(parents=True, exist_ok=True)
    lock_name = "TICK.lock" if args.tick else "LOCK"
    with (DEPLOYMENT / lock_name).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.tick:
            with Beaker.from_env(check_for_upgrades=False) as b:
                tick(b, os.environ["GIT_REF"])
            return
        while True:
            try:
                result = subprocess.run(
                    [sys.executable, __file__, "--tick"],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                print(result.stdout, end="", flush=True)
                if result.returncode:
                    log("SFT_LR_TICK_FAILED", error=result.stderr[-3000:])
            except subprocess.TimeoutExpired:
                log("SFT_LR_TICK_TIMEOUT")
            time.sleep(60)


if __name__ == "__main__":
    main()
