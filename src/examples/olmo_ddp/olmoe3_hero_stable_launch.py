"""Submit only a resource-free Phobos qualification or durable stable-eval watcher."""

import argparse
import copy
import json
import subprocess

from olmoe3_hero_stable_control import identity
from olmoe3_hero_stable_eval import ARMS, BRANCH, CAMPAIGN, WORKSPACE, build_specs
from olmoe3_lr_sweep_watch import log, replace_env, status

TEMPLATE = "01M2938WCZRD02WRGKBP158RSE"


def controller_spec(template, commit, *, check=False):
    """No resources means no CPU, memory or GPU scheduling request on Phobos."""
    spec = copy.deepcopy(template)
    assert len(spec["tasks"]) == 1
    task = spec["tasks"][0]
    task.pop("resources", None)
    task.pop("result", None)
    task["name"] = "stable-eval-preflight" if check else "stable-eval-watcher"
    task["constraints"] = {"cluster": ["ai2/phobos"]}
    task["context"] = dict(priority="urgent", minRuntime="0s", autoResume=not check)
    task["timeout"] = "30m" if check else "720h"
    task["arguments"] = [
        "bash",
        "-euc",
        "exec uv run --no-project --with 'beaker-py==2.7.2' python -u "
        "src/examples/olmo_ddp/olmoe3_hero_stable_control.py" + (" --check-only" if check else ""),
    ]
    replace_env(
        task,
        {
            "GIT_REF": commit,
            "GIT_BRANCH": BRANCH,
            "GANTRY_TASK_NAME": task["name"],
            "GANTRY_INSTALL_CMD": "true",
        },
    )
    spec["retry"] = {"allowedTaskRetries": 0}
    spec["description"] = (
        f"{CAMPAIGN}: {'read-only preflight' if check else 'CPU-only watcher'} for stable "
        "EMO/non-EMO step120000 -> verified scratch copy -> conversion -> qualification "
        "-> allocated frozen OLMoBase evals; no training/retention changes or deletions."
    )
    return spec


def main():
    from beaker import Beaker, BeakerExperimentSpec
    from beaker.exceptions import BeakerNotFoundError

    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("validate", "watch"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if args.apply:
        assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        assert (
            subprocess.check_output(
                ["git", "ls-remote", "origin", f"refs/heads/{BRANCH}"], text=True
            ).split()[0]
            == commit
        ), "Push clean source first"
    with Beaker.from_env(check_for_upgrades=False) as b:
        for arm in ARMS:
            for stage, raw in build_specs(b, arm, commit).items():
                parsed = BeakerExperimentSpec.from_json(copy.deepcopy(raw)).to_json()
                assert identity(parsed) == identity(raw)
                log(
                    "STABLE_EVAL_WORKER_SPEC_VALID",
                    arm=arm,
                    stage=stage,
                    gpus=raw["tasks"][0]["resources"]["gpuCount"],
                )
        check = args.stage == "validate"
        template = b.experiment.get_spec(b.workload.get(TEMPLATE)).to_json()
        spec = controller_spec(template, commit, check=check)
        parsed = BeakerExperimentSpec.from_json(copy.deepcopy(spec))
        assert not parsed.to_json()["tasks"][0].get("resources")
        gate_name = f"{CAMPAIGN}-preflight-{commit[:8]}"
        name = gate_name if check else f"{CAMPAIGN}-watcher-{commit[:8]}"
        log(
            "STABLE_EVAL_CONTROLLER_PLAN",
            name=name,
            source_commit=commit,
            resources="omitted",
            cluster="ai2/phobos",
            worker_jobs=10,
        )
        if not args.apply:
            return
        if not check:
            assert status(b.workload.get("jacobm/" + gate_name)) == "STATUS_SUCCEEDED"
        try:
            work = b.workload.get("jacobm/" + name)
            actual = b.experiment.get_spec(work).to_json()["tasks"][0]
            wanted = parsed.to_json()["tasks"][0]
            for key in ("command", "arguments", "image", "datasets", "envVars", "constraints"):
                assert actual.get(key) == wanted.get(key), key
            assert not actual.get("resources")
        except BeakerNotFoundError:
            work = b.experiment.create(name=name, spec=parsed, workspace=b.workspace.get(WORKSPACE))
        print(
            json.dumps(
                dict(
                    name=name,
                    experiment=work.experiment.id,
                    source_commit=commit,
                    status=status(work),
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
