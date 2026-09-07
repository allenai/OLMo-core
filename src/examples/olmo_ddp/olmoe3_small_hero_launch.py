"""Launch the approved gated hero controller from a clean, pushed checkout."""

import json
import os
import subprocess

import yaml
from olmoe3_lr_sweep_watch import replace_env
from olmoe3_small_hero_control import with_mounts
from olmoe3_small_hero_plan import BRANCH, CAMPAIGN, UPLOADER_COMMIT, WORKSPACE


def main():
    """Submit one urgent unallocated CPU controller; it alone owns training fan-out."""
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    remote = subprocess.check_output(
        ["git", "ls-remote", "origin", f"refs/heads/{BRANCH}"], text=True
    ).split()[0]
    assert remote == commit, "Push the exact clean source before launch"
    spec = yaml.safe_load(
        subprocess.check_output(
            ["beaker", "experiment", "spec", "01M1YTV5221E787H67DMKTM255"], text=True
        )
    )
    spec["description"] = (
        "Small hero EMO/non-EMO: fresh private HF bucket, local Dolma preflight, real-image "
        "validation and 64GPU save/restore smoke gates, then two urgent 64GPU ~3T trunks. "
        "LR1.1e-3 warmup2000 batch16Mi MB4; continuable14T. No ungated training fan-out."
    )
    task = spec["tasks"][0]
    task["timeout"] = "720h"
    task["arguments"] = [
        "bash",
        "-c",
        (
            "gh auth setup-git && exec uv run --no-project "
            "--with beaker-py==2.7.2 --with 'olmo-checkpoint-uploader @ git+https://github.com/"
            f"jacob-morrison/olmo-checkpoint-uploader.git@{UPLOADER_COMMIT}' "
            "python -u src/examples/olmo_ddp/olmoe3_small_hero_control.py"
        ),
    ]
    replace_env(task, {"GIT_REF": commit, "GIT_BRANCH": BRANCH, "GANTRY_CHECK_FOR_UPGRADES": "0"})
    with_mounts(task)
    name = f"{CAMPAIGN}-controller"
    # This local launcher is intentionally one-shot. The controller itself has durable
    # intent/reconciliation. Refuse any existing same-name experiment before submitting.
    listing = json.loads(
        subprocess.check_output(
            ["beaker", "workspace", "experiments", WORKSPACE, "--format", "json"], text=True
        )
    )
    assert not any(x["name"] == name for x in listing), "Controller already exists; inspect it"
    result = subprocess.run(
        [
            "beaker",
            "experiment",
            "create",
            "-",
            "--name",
            name,
            "--workspace",
            WORKSPACE,
            "--format",
            "json",
        ],
        input=json.dumps(spec),
        text=True,
        capture_output=True,
        check=True,
        env={**os.environ, "GANTRY_CHECK_FOR_UPGRADES": "0"},
    )
    payload = json.loads(result.stdout)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
