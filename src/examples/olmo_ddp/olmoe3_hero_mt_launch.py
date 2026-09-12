"""Clean-pin deployment: real-image CPU gate, then a no-resource Phobos MT watcher."""

import argparse
import copy
import json
import subprocess

from olmoe3_hero_mt_control import training_spec
from olmoe3_hero_mt_eval import eval_specs
from olmoe3_hero_mt_plan import BRANCH, CAMPAIGN, DECAY_JOBS, END, WORKSPACE, runs
from olmoe3_lr_sweep_watch import replace_env, status

UPLOADER_REF = "50069318bd7b6bcfed655a8a01d2892e56b7abff"


def build_spec(beaker, commit, stage, gate=None):
    template = "01M29363936Y2BSJYTPPZFMV9X" if stage == "validate" else "01M2938WCZRD02WRGKBP158RSE"
    spec = copy.deepcopy(beaker.experiment.get_spec(beaker.workload.get(template)).to_json())
    assert len(spec["tasks"]) == 1
    t = spec["tasks"][0]
    t.pop("resources", None)
    t.pop("replicas", None)
    t.pop("leaderSelection", None)
    t["constraints"] = {"cluster": ["ai2/phobos"]}
    t["context"] = dict(priority="urgent", minRuntime="0s", autoResume=stage == "watch")
    t["timeout"] = "45m" if stage == "validate" else "720h"
    t["name"] = "mt-" + stage
    if stage == "validate":
        t["arguments"] = [
            "bash",
            "-euc",
            "python src/examples/olmo_ddp/olmoe3_hero_decay_runtime.py && "
            "python src/examples/olmo_ddp/olmoe3_hero_mt.py --validate-only && "
            "python src/examples/olmo_ddp/olmoe3_hero_mt.py --data-probe",
        ]
    else:
        assert gate
        t["arguments"] = [
            "bash",
            "-euc",
            "gh auth setup-git; exec uv run --no-project "
            "--with 'beaker-py==2.7.2' --with 'olmo-checkpoint-uploader @ "
            f"git+https://github.com/jacob-morrison/olmo-checkpoint-uploader.git@{UPLOADER_REF}' "
            "python -u src/examples/olmo_ddp/olmoe3_hero_mt_control.py",
        ]
    replace_env(
        t,
        {
            "GIT_REF": commit,
            "GIT_BRANCH": BRANCH,
            "GANTRY_TASK_NAME": t["name"],
            "GANTRY_CHECK_FOR_UPGRADES": "0",
            "OLMO35_HERO_EXPECTED_START": "0",
            "OLMO35_HERO_STOP": str(END),
            "OLMO35_HERO_ALLOW_CONTINUATION": "1",
            "OLMO35_DECAY_CPU_VALIDATE": "1" if stage == "validate" else None,
            "OLMO35_MT_GATE": gate,
            "OLMO35_MT_LOAD": None,
            "OLMO35_DECAY_LOAD": None,
            "WANDB_RUN_ID": None,
            "WANDB_RESUME": None,
        },
    )
    spec["retry"] = {"allowedTaskRetries": 0}
    spec["description"] = (
        f"{CAMPAIGN}: {stage}; resource-free CPU controller; two independent 64GPU cosine MT jobs with upload/conversion/evals."
    )
    assert "resources" not in t
    return spec


def main():
    from beaker import Beaker, BeakerExperimentSpec
    from beaker.exceptions import BeakerNotFoundError

    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("validate", "watch"))
    parser.add_argument("--gate")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    remote = subprocess.check_output(
        ["git", "ls-remote", "origin", f"refs/heads/{BRANCH}"], text=True
    ).split()[0]
    assert remote == commit
    with Beaker.from_env(check_for_upgrades=False) as b:
        for run in runs():
            original = b.experiment.get_spec(b.workload.get(DECAY_JOBS[run.arm])).to_json()
            plans = dict(train=training_spec(original, run, commit), **eval_specs(b, run, commit))
            for spec in plans.values():
                BeakerExperimentSpec.from_json(copy.deepcopy(spec))
        if args.stage == "watch":
            assert args.gate and status(b.workload.get(args.gate)) == "STATUS_SUCCEEDED"
        spec = build_spec(b, commit, args.stage, args.gate)
        parsed = BeakerExperimentSpec.from_json(copy.deepcopy(spec))
        assert not parsed.to_json()["tasks"][0].get("resources")
        name = f"{CAMPAIGN}-{args.stage}-{commit[:8]}"
        print(
            json.dumps(
                dict(
                    name=name,
                    commit=commit,
                    worker_specs_validated=12,
                    resources="omitted",
                    stage=args.stage,
                )
            ),
            flush=True,
        )
        if not args.apply:
            return
        try:
            work = b.workload.get("jacobm/" + name)
            old = b.experiment.get_spec(work).to_json()["tasks"][0]
            assert old["arguments"] == spec["tasks"][0]["arguments"]
            assert any(
                v.get("name") == "GIT_REF" and v.get("value") == commit for v in old["envVars"]
            )
        except BeakerNotFoundError:
            work = b.experiment.create(name=name, spec=parsed, workspace=b.workspace.get(WORKSPACE))
        print(
            json.dumps(dict(name=name, experiment=work.experiment.id, status=status(work))),
            flush=True,
        )


if __name__ == "__main__":
    main()
