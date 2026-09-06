"""Submit the standalone controller from a clean, pushed branch (not training directly)."""

import argparse
import copy
import json
import subprocess
from pathlib import Path

from beaker import Beaker, BeakerExperimentSpec
from olmoe3_lr_sweep_plan import (
    DEPLOYMENT,
    EXTENSION_LABEL,
    SWEEP,
    UPLOADER_COMMIT,
    UPLOADER_EXPERIMENT,
    WORKSPACE,
)
from olmoe3_lr_sweep_watch import atomic_json, replace_env


def main():
    """Record a local intent, submit exactly once, and save the Beaker receipt."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--extension", choices=[EXTENSION_LABEL])
    args = parser.parse_args()
    assert not subprocess.check_output(["git", "status", "--porcelain"]).strip()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    remote = subprocess.check_output(
        ["git", "ls-remote", "origin", "refs/heads/codex/small-lr100b-sweep"], text=True
    )
    assert remote.split()[0] == commit, "Push the exact clean commit before launch"
    name = f"{SWEEP}-controller-{DEPLOYMENT}"
    if args.extension:
        name += f"-extension-{args.extension}"
    with Beaker.from_env(default_workspace=WORKSPACE, check_for_upgrades=False) as b:
        workspace = b.workspace.get(WORKSPACE)
        existing = [
            w
            for w in b.workload.list(workspace=workspace, name_or_description=name)
            if w.HasField("experiment") and w.experiment.name == name
        ]
        if existing:
            assert len(existing) == 1
            print(json.dumps({"existing_controller": existing[0].experiment.id}))
            return
        spec = b.experiment.get_spec(b.workload.get(UPLOADER_EXPERIMENT)).to_json()
        t = spec["tasks"][0]
        t["arguments"] = [
            "bash",
            "-c",
            (
                "gh auth setup-git && exec uv run --no-project "
                "--with 'beaker-py==2.7.2' --with 'olmo-checkpoint-uploader @ "
                f"git+https://github.com/jacob-morrison/olmo-checkpoint-uploader.git@{UPLOADER_COMMIT}' "
                "python -u src/examples/olmo_ddp/olmoe3_lr_sweep_watch.py"
                + (f" --extension {args.extension}" if args.extension else "")
            ),
        ]
        replace_env(
            t,
            {
                "GITHUB_REPO": "allenai/OLMo-core",
                "GIT_REF": commit,
                "GIT_BRANCH": "codex/small-lr100b-sweep",
                "GANTRY_INSTALL_CMD": "true",
                "GANTRY_TASK_NAME": "main",
                "HF_XET_CACHE": None,
                "HF_XET_SHARD_CACHE_SIZE_LIMIT": None,
                "HF_XET_CHUNK_CACHE_SIZE_BYTES": None,
                "HF_XET_HIGH_PERFORMANCE": None,
            },
        )
        t["envVars"].append({"name": "BEAKER_TOKEN", "secret": "jacobm_BEAKER_TOKEN"})
        t["resources"] = {"cpuCount": 2, "memory": "8 GiB"}
        t["context"] = {"priority": "urgent", "minRuntime": "0s", "autoResume": True}
        t["timeout"] = "720h"
        spec["description"] = "Small 100B LR sweep controller; gated five trunks and twenty decays"
        if args.extension:
            spec["description"] = (
                f"Small 100B LR sweep extension {args.extension}; one trunk and four decays; "
                "isolated ledger, original five LR trajectories untouched"
            )
        spec["retry"] = {"allowedTaskRetries": 3}
        parsed = BeakerExperimentSpec.from_json(copy.deepcopy(spec))
        atomic_json(args.output / "controller-spec.json", spec)
        if not args.submit:
            print("Controller spec validated; --submit required")
            return
        receipt = args.output / "controller-submission.json"
        assert not receipt.exists(), "Existing intent requires reconciliation, not another submit"
        atomic_json(receipt, {"phase": "submitting", "source_commit": commit, "name": name})
        w = b.experiment.create(spec=parsed, name=name, workspace=workspace)
        result = {
            "phase": "submitted",
            "source_commit": commit,
            "name": name,
            "experiment_id": w.experiment.id,
            "workspace": WORKSPACE,
        }
        atomic_json(receipt, result)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
