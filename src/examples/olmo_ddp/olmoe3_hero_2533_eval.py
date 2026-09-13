"""Bounded step151000 stable-PT evaluation: frozen workers, HF fallback, and RULER.

This campaign does not modify training, checkpoint retention, or inference kernels.
Each arm has one conversion, one qualification, and four evaluation jobs. Failed
jobs require explicit intervention; durable intents reconcile scheduler retries.
"""

import argparse
import copy
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import olmoe3_hero_ruler as ruler
import olmoe3_hero_stable_eval as stable
from olmoe3_hero_decay_plan import CORE_REF, HELPER_REF, ready, validate_checkpoint
from olmoe3_hero_ruler_control import WORKER_TEMPLATE, worker_spec
from olmoe3_hero_stable_control import identity
from olmoe3_hero_stable_launch import TEMPLATE
from olmoe3_hero_stable_launch import controller_spec as old_controller_spec
from olmoe3_lr_sweep_watch import Controller, atomic_json, log, replace_env, status
from olmoe3_small_hero_plan import BATCH, BUCKET, MOUNT, STATE

CAMPAIGN = "olmo35-small-stable2533t-20260913"
BRANCH = "codex/hero-2533t-evals-20260913"
STEP = 151000
ROOT = MOUNT / "scratch" / CAMPAIGN
AUTOMATION = MOUNT / "uploader/automation" / CAMPAIGN
ARMS = ("emo", "non-emo")
STAGES = ("convert", "qualify", "gen_mc", "math", "code", "ruler")
WORKSPACE = "ai2/OLMo-3-moe-experiments"
FILE = Path(__file__).name

# Reuse the qualified campaign's validators and spec transformations unchanged.
stable.CAMPAIGN, stable.END, stable.ROOT = CAMPAIGN, STEP, ROOT
stable.WRAPPER = "/tmp/hero-decay-wrapper/src/examples/olmo_ddp/" + FILE
ruler.CAMPAIGN = CAMPAIGN + "-ruler"


def model_path(milestone, arm):
    """Resolve only this campaign's two stable checkpoints."""
    if milestone not in ("1267b", "decay2t") or arm not in ARMS:
        raise ValueError((milestone, arm))
    return stable.output_root(arm) / "hf"


ruler.model_path = model_path


def available(arm):
    """Require a completed local checkpoint or a verified upload of the exact lineage."""
    run = stable.parent(arm)
    if ready(run, STEP):
        return True
    path = STATE / "checkpoints" / run.run_id / f"step-{STEP:012d}.json"
    if not path.is_file():
        return False
    row = json.loads(path.read_text())
    expected = dict(
        step=STEP,
        run_id=run.run_id,
        lineage_id=run.run_id,
        checkpoint_path=str(run.root / f"step{STEP}"),
        bucket_id=BUCKET,
    )
    assert all(row.get(k) == v for k, v in expected.items()), "Uploader identity mismatch"
    return row.get("remote_verified") is True and row.get("source_complete") is True


def build_specs(beaker, arm, commit):
    """Preserve frozen package/kernel/recipe pins and allocated GPU worker resources."""
    specs = stable.build_specs(beaker, arm, commit)
    for spec in specs.values():
        task = spec["tasks"][0]
        task["arguments"] = [task["arguments"][0].replace("step120000", f"step{STEP}")]
        assert "step120000" not in task["arguments"][0]
        replace_env(task, {"HF_XET_HIGH_PERFORMANCE": "1"})
    template = beaker.experiment.get_spec(beaker.workload.get(WORKER_TEMPLATE)).to_json()
    spec = worker_spec(template, "1267b", arm, commit)
    task = spec["tasks"][0]
    old = "olmoe3_hero_ruler.py --milestone 1267b"
    assert task["arguments"][0].count(old) == 1
    task["arguments"] = [task["arguments"][0].replace(old, FILE + " ruler")]
    spec["description"] = json.dumps(
        dict(
            campaign=CAMPAIGN,
            arm=arm,
            step=STEP,
            tokens=STEP * BATCH,
            stage="ruler",
            model=str(model_path("1267b", arm)),
            source_commit=commit,
            helper_ref=HELPER_REF,
            profile="bf16-grouped-fla-pilot-v1",
        )
    )
    specs["ruler"] = spec
    return specs


def run_worker(args):
    """Invoke the existing download/conversion or evaluation implementation."""
    stable.prepare_scratch()
    if args.mode == "ruler":
        stable.qualified(args.arm)
        sys.argv = [ruler.__file__, "--milestone", "1267b", "--arm", args.arm, "--instances", "4"]
        ruler.main()
        return
    expected = CORE_REF if args.mode == "convert" else HELPER_REF
    assert args.source is not None
    assert (
        subprocess.check_output(
            ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True
        ).strip()
        == expected
    )
    assert not subprocess.check_output(
        ["git", "-C", str(args.source), "status", "--porcelain"], text=True
    ).strip()
    if args.mode == "convert":
        sys.path.insert(0, str(args.source / "src/examples/olmo_ddp"))
        import hero_hf_convert as convert
        import hero_hf_download as download
        import hero_hf_stage as stage

        for module in (download, stage, convert):
            module.SCRATCH, module.TARGETS = ROOT, {2533: STEP}
            module.prepare_scratch = stable.prepare_scratch
        sys.argv = [stage.__file__, "--arm", args.arm, "--step", str(STEP)]
        stage.main()
        validate_checkpoint(stable.output_root(args.arm) / "olmo-core", STEP)
        sys.argv = [
            convert.__file__,
            "--arm",
            args.arm,
            "--step",
            str(STEP),
            "--full",
            "--portable-reference",
            "--precise",
        ]
        convert.main()
        stable.converted(args.arm)
    else:
        # stable.main uses the same source pin, validators and scoped FAST_MODELS.
        sys.argv = [stable.__file__, args.mode, "--arm", args.arm, "--source", str(args.source)]
        stable.main()


class EvalController(Controller):
    """Keep exact-name submission reconciliation without initializing training."""

    def __init__(self, beaker, commit):
        self.beaker, self.commit = beaker, commit
        self.workspace = beaker.workspace.get(WORKSPACE)
        self.automation, self.last_status = AUTOMATION, {}

    def ensure(self, name, spec):
        work = super().ensure(name, spec)
        if work is not None:
            assert identity(self.beaker.experiment.get_spec(work).to_json()) == identity(spec)
        return work


def advance(control, arm, specs):
    """Only successful preparation releases this arm's four evaluation bundles."""
    if not available(arm) and not (stable.output_root(arm) / "download-success.json").is_file():
        return dict(state="waiting_for_checkpoint", step=STEP)
    fs = os.statvfs(MOUNT)
    if fs.f_bavail * fs.f_frsize < 12_000_000_000_000:
        return dict(state="waiting_for_storage")
    result = dict(state="in_progress", jobs={})
    for stage in STAGES:
        if stage == "qualify":
            stable.converted(arm)
        elif stage not in ("convert", "qualify"):
            stable.qualified(arm)
        work = control.ensure(f"{CAMPAIGN}-{arm}-{stage}", specs[stage])
        state = control.report(work)
        result["jobs"][stage] = dict(status=state, experiment=work.experiment.id if work else None)
        if stage in ("convert", "qualify") and state != "STATUS_SUCCEEDED":
            if state in ("STATUS_FAILED", "STATUS_CANCELED", "STATUS_CANCELLED", "AMBIGUOUS"):
                result["state"] = "needs_attention"
            return result
        if state == "STATUS_SUCCEEDED" and stage not in ("convert", "qualify"):
            if stage == "ruler":
                ruler.verify_success(model_path("1267b", arm))
            else:
                stable.completed_bundle(arm, stage)
    states = [row["status"] for row in result["jobs"].values()]
    if all(s == "STATUS_SUCCEEDED" for s in states):
        result["state"] = "complete"
    elif any(
        s in ("STATUS_FAILED", "STATUS_CANCELED", "STATUS_CANCELLED", "AMBIGUOUS") for s in states
    ):
        result["state"] = "needs_attention"
    return result


def watch(check=False):
    """Run a resource-free CPU monitor with independent arms and immutable target plan."""
    from beaker import Beaker, BeakerExperimentSpec

    assert MOUNT.is_mount()
    commit = os.environ["GIT_REF"]
    with Beaker.from_env(check_for_upgrades=False) as b:
        specs = {arm: build_specs(b, arm, commit) for arm in ARMS}
        for stages in specs.values():
            for spec in stages.values():
                assert identity(
                    BeakerExperimentSpec.from_json(copy.deepcopy(spec)).to_json()
                ) == identity(spec)
        log(
            "MATCHED_2533_PREFLIGHT_PASSED",
            ready={arm: available(arm) for arm in ARMS},
            source_commit=commit,
            workers=12,
            tokens=STEP * BATCH,
        )
        if check:
            return
        AUTOMATION.mkdir(parents=True, exist_ok=True)
        with (AUTOMATION / "LOCK").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            plan = dict(
                campaign=CAMPAIGN,
                source_commit=commit,
                step=STEP,
                arms=list(ARMS),
                stages=list(STAGES),
                root=str(ROOT),
                workspace=WORKSPACE,
            )
            if (AUTOMATION / "plan.json").exists():
                assert json.loads((AUTOMATION / "plan.json").read_text()) == plan
            else:
                atomic_json(AUTOMATION / "plan.json", plan)
            stable.prepare_scratch()
            control, previous = EvalController(b, commit), None
            log("MATCHED_2533_WATCHER_ARMED", **plan)
            while True:
                snapshot = {}
                for arm in ARMS:
                    try:
                        snapshot[arm] = advance(control, arm, specs[arm])
                    except Exception as error:
                        snapshot[arm] = dict(
                            state="needs_attention", error=f"{type(error).__name__}: {error}"
                        )
                atomic_json(AUTOMATION / "status.json", dict(updated_at=time.time(), arms=snapshot))
                if snapshot != previous:
                    log("MATCHED_2533_STATUS", arms=snapshot)
                    previous = snapshot
                if all(row["state"] == "complete" for row in snapshot.values()):
                    return
                time.sleep(30)


def controller_spec(template, commit, check=False):
    """Omit all resources on Phobos; Gantry's result directory stays empty."""
    spec = old_controller_spec(template, commit, check=check)
    task = spec["tasks"][0]
    task["arguments"][-1] = (
        "exec uv run --no-project --with 'beaker-py==2.7.2' python -u "
        "src/examples/olmo_ddp/" + FILE + (" preflight" if check else "watch")
    )
    replace_env(task, {"GIT_BRANCH": BRANCH})
    spec["description"] = (
        f"{CAMPAIGN}: two stable PT151000 arms -> conversion/qualification -> OLMoBase + ordinary RULER. No training, retention or deletion changes."
    )
    assert "resources" not in task
    return spec


def launch(check, apply):
    """Submit one pinned preflight/controller; never submit duplicate experiments."""
    from beaker import Beaker, BeakerExperimentSpec
    from beaker.exceptions import BeakerNotFoundError

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if apply:
        assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        assert (
            subprocess.check_output(
                ["git", "ls-remote", "origin", f"refs/heads/{BRANCH}"], text=True
            ).split()[0]
            == commit
        )
    with Beaker.from_env(check_for_upgrades=False) as b:
        for arm in ARMS:
            for stage, spec in build_specs(b, arm, commit).items():
                assert identity(
                    BeakerExperimentSpec.from_json(copy.deepcopy(spec)).to_json()
                ) == identity(spec)
                log(
                    "MATCHED_2533_SPEC_VALID",
                    arm=arm,
                    stage=stage,
                    gpus=spec["tasks"][0]["resources"]["gpuCount"],
                )
        template = b.experiment.get_spec(b.workload.get(TEMPLATE)).to_json()
        spec = controller_spec(template, commit, check=check)
        parsed = BeakerExperimentSpec.from_json(copy.deepcopy(spec))
        name = f"{CAMPAIGN}-{'preflight' if check else 'watcher'}-{commit[:8]}"
        if not apply:
            log("MATCHED_2533_PLAN", name=name, source_commit=commit)
            return
        if not check:
            assert (
                status(b.workload.get(f"jacobm/{CAMPAIGN}-preflight-{commit[:8]}"))
                == "STATUS_SUCCEEDED"
            )
        try:
            work = b.workload.get("jacobm/" + name)
            actual = b.experiment.get_spec(work).to_json()["tasks"][0]
            wanted = parsed.to_json()["tasks"][0]
            for key in ("command", "arguments", "envVars", "image", "datasets", "constraints"):
                assert actual.get(key) == wanted.get(key), key
            assert not actual.get("resources")
        except BeakerNotFoundError:
            work = b.experiment.create(name=name, spec=parsed, workspace=b.workspace.get(WORKSPACE))
        log(
            "MATCHED_2533_SUBMITTED",
            name=name,
            experiment=work.experiment.id,
            source_commit=commit,
            status=status(work),
        )


def main():
    """Dispatch explicit worker or controller actions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "launch-preflight",
            "launch-watch",
            "preflight",
            "watch",
            "convert",
            "gen_mc",
            "math",
            "code",
            "ruler",
        ),
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--instances", type=int, choices=(4,), default=4)
    args = parser.parse_args()
    if args.mode.startswith("launch-"):
        launch(args.mode == "launch-preflight", args.apply)
    elif args.mode in ("preflight", "watch"):
        watch(args.mode == "preflight")
    else:
        assert args.arm in ARMS
        run_worker(args)


if __name__ == "__main__":
    main()
