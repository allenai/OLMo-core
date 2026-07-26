#!/usr/bin/env python3
"""Compare a localized production GDN1/GDN2 failure with a recurrent reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from gantry.api import GitRepoState, Recipe

from scripts.train.jacobm_olmoe_ladder.v2.launchers.pretraining.launch_geometry_matched_scale_nope_smokes import (
    GDN2_FLA_OVERLAY,
    GDN2_FLA_SPEC,
    load_manifest,
    validate_remote_commit,
)

SCRIPT_DIR = Path(__file__).resolve().parent
MANIFEST = SCRIPT_DIR / "manifests" / "geometry_matched_scale_gdn2_ev1_noneg_nope_gated_full.yaml"
RECORD = SCRIPT_DIR / "generated" / "gdn2_capture_reference_submissions.json"
ANALYZER = "src/scripts/train/jacobm_olmoe_ladder/v2/diagnostics/analyze_gdn2_nonfinite_capture.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    capture = args.capture.resolve()
    if not capture.is_file():
        raise FileNotFoundError(capture)
    payload = torch.load(capture, map_location="cpu", weights_only=False)
    module_type = str(payload.get("module_type"))
    if module_type not in {"GatedDeltaNet", "GatedDeltaNet2"}:
        raise ValueError(f"capture is from unsupported boundary {module_type}")
    output = (args.output or capture.with_name(capture.stem + "_reference.json")).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    manifest = load_manifest(MANIFEST)
    source = manifest["source"]
    beaker = manifest["beaker"]
    print(f"Capture: {capture}")
    print(f"Output:  {output}")
    print("Resources: 1 B300, urgent, unallocated, no checkpoint or W&B writes")
    if not args.submit:
        print("Dry run only; pass --submit to launch.")
        return

    commit = validate_remote_commit(str(source["remote"]), str(source["branch"]))
    if module_type == "GatedDeltaNet2":
        pre_setup = (
            "unset S3_PROFILE"
            f"\nrm -rf {GDN2_FLA_OVERLAY}"
            f"\npython -m pip install --target {GDN2_FLA_OVERLAY} --no-deps "
            f"--no-build-isolation '{GDN2_FLA_SPEC}'"
            f'\nPYTHONPATH={GDN2_FLA_OVERLAY} python -c "import fla; '
            "from fla.ops.gdn2 import chunk_gdn2; assert fla.__version__ == '0.5.2'\""
        )
        pythonpath = f"{GDN2_FLA_OVERLAY}:src"
    else:
        pre_setup = (
            "unset S3_PROFILE"
            '\npython -c "import fla; from fla.ops.gated_delta_rule import '
            "chunk_gated_delta_rule; print(fla.__version__)\""
        )
        pythonpath = "src"
    recipe = Recipe(
        args=[ANALYZER, str(capture), "--output", str(output)],
        name=f"gdn-reference-{capture.parent.name}-{capture.stem}",
        description=(
            f"Exact production {module_type} activation/state comparison: chunk kernel versus "
            "sequential recurrent reference"
        ),
        workspace=str(beaker["workspace"]),
        task_name="gdn2-captured-reference",
        git_repo=GitRepoState.from_env(ref=commit, branch=str(source["branch"])),
        allow_dirty=False,
        yes=True,
        clusters=[str(beaker["cluster"])],
        gpus=1,
        shared_memory="128GiB",
        beaker_image=str(source["image"]),
        env_vars=[
            ("PYTHONPATH", pythonpath),
            ("PYTHONUNBUFFERED", "1"),
            ("OLMO_SHARED_FS", "1"),
            ("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
        ],
        weka=[(str(item["bucket"]), str(item["mount"])) for item in manifest.get("weka", [])],
        priority=str(beaker["priority"]),
        min_runtime=str(beaker["min_runtime"]),
        preemptible=False,
        auto_resume=False,
        task_timeout="6h",
        host_networking=True,
        no_python=True,
        pre_setup=pre_setup,
    )
    workload = recipe.launch(show_logs=False)
    experiment = workload.experiment
    record = {
        "commit": commit,
        "module_type": module_type,
        "capture": str(capture),
        "output": str(output),
        "experiment_id": experiment.id,
        "task_ids": [task.id for task in experiment.tasks],
        "url": (
            f"https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/{experiment.id}"
        ),
    }
    existing: list[dict[str, object]] = []
    if RECORD.is_file():
        existing = json.loads(RECORD.read_text())
    existing.append(record)
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"Submitted: {record['url']}")
    print(f"Recorded submission in {RECORD}")


if __name__ == "__main__":
    main()
