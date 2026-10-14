"""Render an inert Beaker spec; no submission or uploader registration occurs here.

Example (after creating and reviewing PLAN.json):
    python src/examples/olmo_ddp/olmoe3_combined_decay_spec.py \
        --plan /shared/path/PLAN.json --output /shared/path/beaker-spec.json

The plan path must be on a mounted Weka volume. This is a preparation-only tool.
After explicit launch authorization, register the plan's new uploader lineage,
check the current Holmes allowlist, then submit the saved spec to the workspace
ai2/olmo3p5-training. Do not attach an automatic SFT-completion launch trigger.
"""

import argparse
import copy
import json
import subprocess
from pathlib import Path

import olmoe3_combined_decay_plan as p

TEMPLATE = "01M4XMGH8GJTYN8MFE3DX49DBG"
IB_DOWN = {496, 511, 523, 549, 550}


def build_spec(template, plan, plan_path, commit):
    """Keep the qualified image, mounts and secret references; isolate the new run."""
    from olmoe3_lr_sweep_watch import replace_env

    spec = copy.deepcopy(template)
    task = spec["tasks"][0]
    spec["tasks"] = [task]
    task.update(name="train", replicas=plan["gpus"] // 8, leaderSelection=True, timeout="720h")
    task["resources"]["gpuCount"] = 8
    task["constraints"] = dict(
        hostname=[
            h
            for h in task["constraints"]["hostname"]
            if int(h.split(".")[0].rsplit("-", 1)[-1]) not in IB_DOWN
        ]
    )
    assert len(task["constraints"]["hostname"]) >= plan["gpus"] // 8
    task["arguments"] = [
        "python",
        "src/examples/olmo_ddp/olmoe3_combined_decay_node.py",
        str(plan_path),
    ]
    task["context"].update(priority="urgent", minRuntime="8h", autoResume=True)
    task["result"] = dict(path="/noop-results")
    task["envVars"] = [
        e
        for e in task["envVars"]
        if not e["name"].startswith(("QKGAIN_", "OLMO35_", "HYBRID_", "COMBINED_"))
    ]
    replace_env(
        task,
        dict(
            GIT_REF=commit,
            GIT_BRANCH=p.BRANCH,
            NUM_NODES=str(plan["gpus"] // 8),
            GANTRY_TASK_NAME="train",
            GANTRY_INSTALL_CMD="true",
            GANTRY_POST_SETUP_CMD="bash src/examples/olmo_ddp/olmoe3_profile_setup.sh",
            WANDB_RUN_ID=None,
            WANDB_RESUME=None,
            WANDB_MODE=None,
            OLMO35_HERO_GPUS=str(plan["gpus"]),
            OLMOE3_BEAKER_WORKSPACE=p.WORKSPACE,
            OLMO_PROFILE_LB_COUNT_BATCHED="0",
            RESULTS_DIR="/noop-results",
            CACHED_PATH_CACHE_ROOT=str(Path(plan["save_folder"]).parent / "cached-path-metadata"),
        ),
    )
    for volume in ["olmo-3p5-checkpoints", "dolma-3p5", "oe-adapt-default"]:
        mount = "/weka/" + volume
        if not any(d["mountPath"] == mount for d in task["datasets"]):
            task["datasets"].append(dict(mountPath=mount, source=dict(weka=volume)))
    assert any(str(plan_path).startswith(d["mountPath"] + "/") for d in task["datasets"])
    spec["retry"] = dict(allowedTaskRetries=10)
    spec["description"] = json.dumps(
        dict(
            recipe=plan["recipe"],
            plan_sha256=p.digest(plan),
            source=plan["source"],
            run_name=plan["run_name"],
        )
    )
    return spec


if __name__ == "__main__":
    from beaker import Beaker, BeakerExperimentSpec

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = p.fixed.read_json(args.plan)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    b = Beaker.from_env(check_for_upgrades=False)
    template = b.experiment.get_spec(b.workload.get(TEMPLATE)).to_json()
    spec = build_spec(template, plan, args.plan.resolve(), commit)
    BeakerExperimentSpec.from_json(copy.deepcopy(spec))
    p.write_json(args.output, spec)
    print("PREPARED_ONLY", args.output, "workspace=" + p.WORKSPACE)
