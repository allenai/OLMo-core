#!/usr/bin/env python3
"""Launch clean original-GDN2 retrains with the actual FLA v0.5.2 release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.train.jacobm_olmoe_ladder.v2.launchers.pretraining.launch_geometry_matched_scale_full import (
    recipe_for,
    validate,
)
from scripts.train.jacobm_olmoe_ladder.v2.launchers.pretraining.launch_geometry_matched_scale_nope_smokes import (
    load_manifest,
    validate_remote_commit,
)

SCRIPT_DIR = Path(__file__).resolve().parent
MANIFEST = SCRIPT_DIR / "manifests" / "geometry_matched_scale_gdn2_nope_gated_full_candidate.yaml"
RECORD = SCRIPT_DIR / "generated" / "gdn2_v052_release_fresh_original_submissions.json"
RUN_SUFFIX = "-fla-v052-release-fresh-r1"
FLA_OVERLAY = "/tmp/fla-gdn2-v0.5.2-release"
FLA_SPEC = (
    "flash-linear-attention[cuda] @ git+https://github.com/fla-org/"
    "flash-linear-attention.git@v0.5.2"
)
FLA_COMMIT = "9c8e42e762fce087c27b673af4922795d9edb85e"
SELECTED_TASKS = ("810m-cx1", "810m-cx2", "1p2b-cx1")


def _prepare() -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    manifest = load_manifest(MANIFEST)
    manifest["experiment"]["description"] = (
        "Clean original-GDN2 retrains using the actual FLA v0.5.2 release"
    )
    rows = validate(manifest)
    rows_by_task = {str(row["task_name"]): row for row in rows}
    selected = []
    for task_name in SELECTED_TASKS:
        row = rows_by_task[task_name]
        selected.append(
            {
                **row,
                "task_name": f"{task_name}{RUN_SUFFIX}",
                "run_name": f"{row['run_name']}{RUN_SUFFIX}",
            }
        )

    checkpoint_root = Path(str(manifest["experiment"]["checkpoint_root"]))
    existing = [
        checkpoint_root / str(row["run_name"])
        for row in selected
        if (checkpoint_root / str(row["run_name"])).exists()
    ]
    if existing:
        raise RuntimeError(
            "Fresh-release launch refuses existing checkpoint directories:\n"
            + "\n".join(f"  - {path}" for path in existing)
        )

    source = manifest["source"]
    commit = validate_remote_commit(str(source["remote"]), str(source["branch"]))
    return manifest, selected, commit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    manifest, rows, commit = _prepare()
    print("\nsize  Cx nodes GPU/node world EP MB accum LR run")
    for row in rows:
        print(
            f"{row['model_size']:<5} {row['cx']:>2} {row['num_nodes']:>5} "
            f"{row['gpus_per_node']:>8} {row['world_size']:>5} "
            f"{row['expert_parallel_size']:>2} {row['rank_microbatch_sequences']:>2} "
            f"{row['accumulation_steps']:>5} {row['learning_rate']:<7} "
            f"{row['run_name']}"
        )
    print(f"\nSelected {len(rows)} fresh runs using {sum(row['world_size'] for row in rows)} GPUs.")
    print(f"Source commit: {commit}")
    print(f"FLA release: v0.5.2 ({FLA_COMMIT})")
    if not args.submit:
        print("Dry run only; pass --submit to launch.")
        return

    records: list[dict[str, Any]] = []
    for row in rows:
        workload = recipe_for(
            manifest,
            row,
            commit=commit,
            debug_gradients=True,
            recipe_suffix_override="",
            description_suffix=(
                "clean from-scratch trajectory; actual FLA v0.5.2 release; "
                "non-finite gradient diagnostics enabled"
            ),
            gdn2_fla_overlay=FLA_OVERLAY,
            gdn2_fla_spec=FLA_SPEC,
            gdn2_fla_expected_commit=FLA_COMMIT,
        ).launch(show_logs=False)
        experiment = workload.experiment
        record = {
            "task_name": row["task_name"],
            "run_name": row["run_name"],
            "model_size": row["model_size"],
            "cx": row["cx"],
            "learning_rate": row["learning_rate"],
            "global_batch_size": row["global_batch_size"],
            "world_size": row["world_size"],
            "expert_parallel_size": row["expert_parallel_size"],
            "rank_microbatch_sequences": row["rank_microbatch_sequences"],
            "accumulation_steps": row["accumulation_steps"],
            "source_commit": commit,
            "fla_tag": "v0.5.2",
            "fla_commit": FLA_COMMIT,
            "fresh_checkpoint_directory": True,
            "experiment_id": experiment.id,
            "task_ids": [task.id for task in experiment.tasks],
            "url": (
                "https://beaker.org/orgs/ai2/workspaces/"
                f"OLMo-3-moe-experiments/work/{experiment.id}"
            ),
        }
        records.append(record)
        print(f"Submitted {row['task_name']}: {record['url']}")

    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(records, indent=2) + "\n")
    print(f"Recorded {len(records)} submissions in {RECORD}")


if __name__ == "__main__":
    main()
