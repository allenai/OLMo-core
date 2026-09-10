"""Launch the explicit gated medium campaign from a clean, pushed checkout."""

import argparse
import copy
import json
import os
import shlex
import subprocess
from pathlib import Path

from beaker import Beaker, BeakerExperimentSpec

from olmoe3_medium_cbs64_plan import BRANCH_NAME, CAMPAIGN, WORKSPACE
from olmoe3_medium_cbs_control import atomic_json, replace_env


def main():
    """Submit a CPU-only controller; it gates all training submissions itself."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    core = Path(__file__).resolve().parents[3]
    assert not subprocess.check_output(["git", "-C", str(core), "status", "--porcelain"]).strip()
    commit = subprocess.check_output(
        ["git", "-C", str(core), "rev-parse", "HEAD"], text=True
    ).strip()
    remote = subprocess.check_output(
        ["git", "-C", str(core), "ls-remote", "origin", f"refs/heads/{BRANCH_NAME}"], text=True
    )
    assert remote.split()[0] == commit
    spec = json.loads(args.template.read_text())
    assert len(spec["tasks"]) == 1
    task = spec["tasks"][0]
    assert task["constraints"]["cluster"] == ["ai2/rhea"]
    assert not task["resources"].get("gpuCount")
    task["context"] = {"priority": "urgent", "minRuntime": "0s", "autoResume": True}
    task["timeout"] = "168h"
    task["result"] = {"path": "/noop-results"}
    replace_env(
        task, {"GIT_REF": commit, "GIT_BRANCH": BRANCH_NAME, "RESULTS_DIR": "/noop-results"}
    )
    task["arguments"] = [
        "bash",
        "-c",
        "gh auth setup-git && exec uv run --no-project --with 'beaker-py==2.7.2' "
        "--with 'olmo-checkpoint-uploader @ git+https://github.com/jacob-morrison/"
        "olmo-checkpoint-uploader.git@3b3a102b106956ae8ca1ea8e92a0ad7ec8fbacf0' "
        + shlex.join(["python", "-u", "src/examples/olmo_ddp/olmoe3_medium_cbs64_control.py"]),
    ]
    spec["retry"] = {"allowedTaskRetries": 0}
    spec["description"] = (
        f"{CAMPAIGN}: gated64GPU resume/speed,128GPU comparison,64GPU CBS32/64 to100B"
    )
    name = f"{CAMPAIGN}-controller"
    args.output.mkdir(parents=True, exist_ok=True)
    with Beaker.from_env(check_for_upgrades=False) as beaker:
        workspace = beaker.workspace.get(WORKSPACE)
        matches = [
            w
            for w in beaker.workload.list(workspace=workspace, name_or_description=name, limit=100)
            if w.HasField("experiment") and w.experiment.name == name
        ]
        if matches:
            assert len(matches) == 1
            print(json.dumps({"existing": matches[0].experiment.id, "name": name}), flush=True)
            return
        parsed = BeakerExperimentSpec.from_json(copy.deepcopy(spec))
        atomic_json(args.output / "spec.json", spec)
        if not args.submit:
            print("MEDIUM_CBS64_CONTROLLER_SPEC_VALID", name, commit, flush=True)
            return
        path = args.output / "submission.json"
        assert not path.exists(), "Ambiguous prior submission; reconcile rather than duplicate"
        record = {"name": name, "source_commit": commit, "phase": "submitting"}
        atomic_json(path, record)
        workload = beaker.experiment.create(spec=parsed, name=name, workspace=workspace)
        record.update(phase="submitted", experiment_id=workload.experiment.id)
        atomic_json(path, record)
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
