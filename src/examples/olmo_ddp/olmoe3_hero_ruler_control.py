"""Four-target RULER companion: existing 1.267T exports now, qualified 2T decays later."""

import argparse
import copy
import fcntl
import hashlib
import json
import os
import subprocess
import time

from olmoe3_hero_decay_plan import BRANCH, HELPER_REF, MOUNT
from olmoe3_hero_ruler import AUTOMATION, CAMPAIGN, model_path, verify_success
from olmoe3_lr_sweep_watch import Controller, atomic_json, log, replace_env, status

WORKSPACE = "ai2/OLMo-3-moe-experiments"
WORKER_TEMPLATE = "01M26P7QE8SA6584507NBQK9DA"
WATCHER_TEMPLATE = "01M28EM0MYCK6CNEKKCWM5F91M"
TARGETS = tuple((m, a) for m in ("1267b", "decay2t") for a in ("emo", "non-emo"))


def qualified(model):
    """Return False for absent prerequisites; fail closed on contradictory source evidence."""
    files = (
        model / "_HERO_CONVERSION_SUCCESS.json",
        model.parent / "conversion-success.json",
        model.parent / "vllm-parity-success.json",
        model.parent / "eval-smoke-success.json",
    )
    if not all(p.is_file() for p in files):
        return False
    assert model.resolve() == model
    digest = hashlib.sha256(files[0].read_bytes()).hexdigest()
    for p in files[1:]:
        row = json.loads(p.read_text())
        assert row.get("passed") is True and not row.get("diagnostic_only"), str(p)
        if p.name != "conversion-success.json":
            assert row["precise"] and row["conversion_sha256"] == digest, str(p)
    return True


def worker_spec(template, milestone, arm, commit):
    """Preserve the exact working package/kernel pins and allocate four evaluation GPUs."""
    model_path(milestone, arm)
    assert len(commit) == 40 and set(commit) <= set("0123456789abcdef")
    spec = copy.deepcopy(template)
    assert len(spec["tasks"]) == 1
    task = spec["tasks"][0]
    command = task["arguments"][0]
    old = (
        "python ladders/olmoe3/workloads/hero_full_eval.py gen_mc "
        "/weka/olmo-3p5-checkpoints/scratch/hero-hf-20260909/emo/step75500/hf "
        "--instances 4 --fast-pilot"
    )
    assert command.count(old) == 1 and command.count(HELPER_REF) == 2
    replacement = (
        "git init --quiet /tmp/hero-ruler-wrapper\n"
        "git -C /tmp/hero-ruler-wrapper remote add origin https://github.com/allenai/OLMo-core.git\n"
        f"git -C /tmp/hero-ruler-wrapper fetch --quiet --depth=1 origin {commit}\n"
        f"git -C /tmp/hero-ruler-wrapper checkout --quiet {commit}\n"
        "python /tmp/hero-ruler-wrapper/src/examples/olmo_ddp/olmoe3_hero_ruler.py "
        f"--milestone {milestone} --arm {arm} --instances 4"
    )
    task["arguments"] = [command.replace(old, replacement)]
    task["name"] = "ruler"
    task["context"] = dict(priority="urgent", minRuntime="1h", autoResume=True)
    task["timeout"] = "6h"
    replace_env(task, {"GIT_REF": commit, "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"})
    assert task["resources"]["gpuCount"] == 4
    assert task["constraints"]["cluster"] == ["ai2/jupiter", "ai2/ceres"]
    assert not task.get("result", {}).get("path")
    spec["retry"] = {"allowedTaskRetries": 0}
    spec["description"] = json.dumps(
        dict(
            campaign=CAMPAIGN,
            milestone=milestone,
            arm=arm,
            model=str(model_path(milestone, arm)),
            suite="standard RULER 4K/8K/16K/32K/64K/128K; 100 examples x 13 tasks x 6 lengths",
            source_commit=commit,
            profile="bf16-grouped-fla-pilot-v1",
            numerically_qualified=False,
        )
    )
    return spec


def watch():
    """Submit each target at most once, independently of the existing decay/training watcher."""
    from beaker import Beaker, BeakerExperimentSpec

    assert MOUNT.is_mount(), "No writes to an unmounted overlay"
    AUTOMATION.mkdir(parents=True, exist_ok=True)
    commit = os.environ["GIT_REF"]
    with (AUTOMATION / "LOCK").open("a") as lock, Beaker.from_env(check_for_upgrades=False) as b:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        control = Controller(b, commit, automation=AUTOMATION)
        control.workspace = b.workspace.get(WORKSPACE)
        template = b.experiment.get_spec(b.workload.get(WORKER_TEMPLATE)).to_json()
        specs = {(m, a): worker_spec(template, m, a, commit) for m, a in TARGETS}
        for spec in specs.values():
            BeakerExperimentSpec.from_json(copy.deepcopy(spec))
        atomic_json(
            AUTOMATION / "plan.json",
            dict(
                commit=commit,
                targets=[dict(milestone=m, arm=a, model=str(model_path(m, a))) for m, a in TARGETS],
            ),
        )
        log("RULER_WATCHER_ARMED", targets=TARGETS, gpu_resources=0)
        previous = None
        while True:
            snapshot = {}
            for milestone, arm in TARGETS:
                key = f"{milestone}-{arm}"
                model = model_path(milestone, arm)
                if not qualified(model):
                    snapshot[key] = dict(status="waiting_for_conversion_and_qualification")
                    continue
                w = control.ensure(f"{CAMPAIGN}-{key}", specs[(milestone, arm)])
                state = control.report(w)
                snapshot[key] = dict(status=state, experiment=w.experiment.id if w else None)
                if state == "STATUS_SUCCEEDED":
                    snapshot[key]["results"] = verify_success(model)
            atomic_json(AUTOMATION / "status.json", dict(updated_at=time.time(), runs=snapshot))
            if snapshot != previous:
                log("RULER_STATUS", runs=snapshot)
                previous = snapshot
            if len(snapshot) == 4 and all(
                r["status"] == "STATUS_SUCCEEDED" for r in snapshot.values()
            ):
                log("ALL_HERO_RULER_COMPLETE")
                return
            time.sleep(30)


def launch(apply):
    """Validate and submit the clean-pinned CPU-only watcher, with no resource requests."""
    from beaker import Beaker, BeakerExperimentSpec
    from beaker.exceptions import BeakerNotFoundError

    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    remote = subprocess.check_output(
        ["git", "ls-remote", "origin", f"refs/heads/{BRANCH}"], text=True
    ).split()[0]
    assert commit == remote
    with Beaker.from_env(check_for_upgrades=False) as b:
        template = b.experiment.get_spec(b.workload.get(WORKER_TEMPLATE)).to_json()
        for milestone, arm in TARGETS:
            BeakerExperimentSpec.from_json(worker_spec(template, milestone, arm, commit))
        spec = b.experiment.get_spec(b.workload.get(WATCHER_TEMPLATE)).to_json()
        task = spec["tasks"][0]
        task.pop("resources", None)
        task.pop("result", None)
        task["name"] = "ruler-watcher"
        task["context"] = dict(priority="urgent", minRuntime="0s", autoResume=True)
        task["constraints"] = {"cluster": ["ai2/phobos"]}
        task["timeout"] = "720h"
        task["arguments"] = [
            "bash",
            "-c",
            "gh auth setup-git && exec uv run --no-project "
            "--with 'beaker-py==2.7.2' python -u "
            "src/examples/olmo_ddp/olmoe3_hero_ruler_control.py watch",
        ]
        replace_env(
            task, {"GIT_REF": commit, "GIT_BRANCH": BRANCH, "GANTRY_TASK_NAME": "ruler-watcher"}
        )
        assert not task.get("result", {}).get("path")
        spec["retry"] = {"allowedTaskRetries": 0}
        spec["description"] = (
            CAMPAIGN
            + ": CPU-only four-target standard RULER companion; no training/cleanup changes"
        )
        parsed = BeakerExperimentSpec.from_json(copy.deepcopy(spec))
        name = f"{CAMPAIGN}-watcher-{commit[:8]}"
        log("RULER_LAUNCH_PLAN", name=name, source_commit=commit, targets=TARGETS, gpu_resources=0)
        if not apply:
            return
        try:
            w = b.workload.get("jacobm/" + name)
            actual = b.experiment.get_spec(w).to_json()["tasks"][0]
            for key in ("command", "arguments", "envVars", "image", "datasets", "constraints"):
                assert actual.get(key) == parsed.to_json()["tasks"][0].get(key), key
        except BeakerNotFoundError:
            w = b.experiment.create(name=name, spec=parsed, workspace=b.workspace.get(WORKSPACE))
        log("RULER_WATCHER_SUBMITTED", experiment=w.experiment.id, status=status(w))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "watch"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    watch() if args.mode == "watch" else launch(args.apply)
