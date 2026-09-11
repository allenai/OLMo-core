"""Clean-pin launch of CPU validation and then the CPU-only two-hero decay watcher."""

import argparse
import copy
import json
import subprocess

from olmoe3_hero_decay_control import training_spec
from olmoe3_hero_decay_eval import eval_specs
from olmoe3_hero_decay_plan import BRANCH, CAMPAIGN, PARENTS, UPLOADER_REF, runs
from olmoe3_lr_sweep_watch import replace_env, status
from olmoe3_small_hero_control import validation_spec
from olmoe3_small_hero_plan import WORKSPACE


def identity(spec):
    task = spec["tasks"][0]
    return {
        k: task.get(k)
        for k in ("command", "arguments", "envVars", "image", "datasets", "constraints")
    }


def main():
    from beaker import Beaker, BeakerExperimentSpec
    from beaker.exceptions import BeakerNotFoundError

    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("validate", "watch"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    remote = subprocess.check_output(
        ["git", "ls-remote", "origin", f"refs/heads/{BRANCH}"], text=True
    ).split()[0]
    assert remote == commit, "Push the exact clean source before launch"
    with Beaker.from_env(check_for_upgrades=False) as b:
        # Validate all actual training/eval specs in the Beaker schema without submitting them.
        for r in runs():
            original = b.experiment.get_spec(b.workload.get(PARENTS[r.arm])).to_json()
            plan = {"train": training_spec(original, r, commit), **eval_specs(b, r, commit)}
            for raw in plan.values():
                BeakerExperimentSpec.from_json(copy.deepcopy(raw))
        gate_name = f"{CAMPAIGN}-config-{commit[:8]}"
        if args.stage == "validate":
            original = b.experiment.get_spec(b.workload.get(PARENTS["emo"])).to_json()
            spec = validation_spec(original, commit)
            task = spec["tasks"][0]
            task.pop("resources", None)
            task.pop("replicas", None)
            task.pop("leaderSelection", None)
            task["constraints"] = {"cluster": ["ai2/phobos"]}
            task["arguments"] = [
                "python",
                "src/examples/olmo_ddp/olmoe3_hero_decay.py",
                "--validate-only",
            ]
            task["context"] = dict(priority="urgent", minRuntime="0s", autoResume=False)
            for item in task["envVars"]:
                if item["name"] == "GANTRY_POST_SETUP_CMD":
                    item["value"] = item["value"].replace(
                        "github.com/allenai/kernel-fun.git@",
                        "github.com/allenai/kernel-fun-dev.git@",
                    )
            replace_env(
                task,
                {
                    "GIT_BRANCH": BRANCH,
                    "OLMO35_HERO_EXPECTED_START": None,
                    "OLMO35_HERO_STOP": None,
                    "WANDB_RUN_ID": None,
                    "WANDB_RESUME": None,
                },
            )
            name = gate_name
        else:
            gate = b.workload.get("jacobm/" + gate_name)
            assert (
                status(gate) == "STATUS_SUCCEEDED"
            ), "Real-image configuration gate must succeed first"
            spec = b.experiment.get_spec(b.workload.get("01M27JJ57YJCZ5FVMJDZWHZZ6Y")).to_json()
            task = spec["tasks"][0]
            task.pop("resources", None)
            task["constraints"] = {"cluster": ["ai2/phobos"]}
            task["timeout"] = "720h"
            task["context"] = dict(priority="urgent", minRuntime="0s", autoResume=True)
            task["name"] = "decay-watcher"
            task["arguments"] = [
                "bash",
                "-c",
                "gh auth setup-git && exec uv run --no-project "
                "--with 'beaker-py==2.7.2' --with 'olmo-checkpoint-uploader @ "
                f"git+https://github.com/jacob-morrison/olmo-checkpoint-uploader.git@{UPLOADER_REF}' "
                "python -u src/examples/olmo_ddp/olmoe3_hero_decay_control.py",
            ]
            replace_env(
                task,
                {
                    "GIT_REF": commit,
                    "GIT_BRANCH": BRANCH,
                    "GANTRY_TASK_NAME": "decay-watcher",
                    "GANTRY_CHECK_FOR_UPGRADES": "0",
                },
            )
            spec["retry"] = {"allowedTaskRetries": 0}
            name = f"{CAMPAIGN}-watcher-{commit[:8]}"
        spec["description"] = (
            f"{CAMPAIGN}: {args.stage}, no resource requests; independent hero WSD decays108000->120000, full state/upload/eval gates."
        )
        assert "resources" not in task
        parsed = BeakerExperimentSpec.from_json(copy.deepcopy(spec))
        assert not parsed.to_json()["tasks"][0].get("resources")
        print(
            json.dumps(
                dict(
                    stage=args.stage,
                    name=name,
                    commit=commit,
                    resources="omitted",
                    cluster="ai2/phobos",
                )
            ),
            flush=True,
        )
        if not args.apply:
            return
        try:
            w = b.workload.get("jacobm/" + name)
            assert identity(b.experiment.get_spec(w).to_json()) == identity(parsed.to_json())
        except BeakerNotFoundError:
            w = b.experiment.create(name=name, spec=parsed, workspace=b.workspace.get(WORKSPACE))
        print(json.dumps(dict(name=name, experiment=w.experiment.id, status=status(w))), flush=True)


if __name__ == "__main__":
    main()
