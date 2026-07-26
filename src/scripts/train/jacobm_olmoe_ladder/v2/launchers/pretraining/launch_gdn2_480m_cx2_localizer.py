#!/usr/bin/env python3
"""Replay and localize the deterministic canonical-GDN2 480M Cx2 failure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.train.jacobm_olmoe_ladder.v2.launchers.pretraining.launch_geometry_matched_scale_full import (
    recipe_for,
    validate,
)
from scripts.train.jacobm_olmoe_ladder.v2.launchers.pretraining.launch_geometry_matched_scale_nope_smokes import (
    load_manifest,
    validate_remote_commit,
)

SCRIPT_DIR = Path(__file__).resolve().parent
MANIFEST = SCRIPT_DIR / "manifests" / "geometry_matched_scale_gdn2_ev1_noneg_nope_gated_full.yaml"
RECORD = SCRIPT_DIR / "generated" / "gdn2_480m_cx2_localizer_submissions.json"
TARGET_TASK = "480m-cx2"
EXPECTED_CHECKPOINT = "step24500"
EXPECTED_FAILURE_STEP = 24_668
LOCALIZE_START_STEP = 24_640
HARD_STOP_STEP = 24_750
DUMP_ROOT = Path(
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/"
    "olmo-ddp/debug/gdn2-480m-cx2-localizer"
)
RUN_ID = "canonical-gdn2-480m-cx2-step24668-r1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(MANIFEST)
    rows = validate(manifest)
    row = next(row for row in rows if str(row["task_name"]) == TARGET_TASK)

    checkpoint_dir = Path(manifest["experiment"]["checkpoint_root"]) / str(row["run_name"])
    checkpoints = sorted(path for path in checkpoint_dir.glob("step*") if path.is_dir())
    if [path.name for path in checkpoints] != [EXPECTED_CHECKPOINT]:
        raise RuntimeError(
            f"Expected only {EXPECTED_CHECKPOINT} in {checkpoint_dir}, found "
            f"{[path.name for path in checkpoints]}"
        )
    output_dir = DUMP_ROOT / RUN_ID
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing diagnostic output {output_dir}")

    print(
        f"Replay {row['run_name']} from {EXPECTED_CHECKPOINT}: world={row['world_size']} "
        f"EP={row['expert_parallel_size']} global_batch={row['global_batch_size']} "
        f"MB={row['rank_microbatch_sequences']} accum={row['accumulation_steps']} "
        f"localize={LOCALIZE_START_STEP}..{HARD_STOP_STEP} "
        f"expected_failure={EXPECTED_FAILURE_STEP} checkpoint_writes=False wandb=False "
        f"dump={output_dir}"
    )
    if not args.submit:
        print("Dry run only; pass --submit to launch.")
        return

    source = manifest["source"]
    commit = validate_remote_commit(str(source["remote"]), str(source["branch"]))
    debug_steps = ",".join(str(step) for step in range(LOCALIZE_START_STEP, HARD_STOP_STEP + 1))
    workload = recipe_for(
        manifest,
        row,
        commit=commit,
        debug_gradients=True,
        runtime_env_overrides={
            "OLMOE3_HYBRID_HARD_STOP_STEPS": str(HARD_STOP_STEP),
            # Retain checkpoint discovery/loading while protecting the source
            # trajectory from saves, pruning, W&B updates, or eval callbacks.
            "OLMOE3_HYBRID_CHECKPOINTS": "1",
            "OLMOE3_HYBRID_CHECKPOINT_WRITES": "0",
            "OLMOE3_HYBRID_WANDB": "0",
            "OLMOE3_HYBRID_GDN2_LOCALIZE_NONFINITE": "1",
            "OLMOE3_HYBRID_GDN2_LOCALIZE_START_STEP": str(LOCALIZE_START_STEP),
            "OLMOE3_HYBRID_GDN2_LOCALIZE_END_STEP": str(HARD_STOP_STEP),
            "OLMOE3_HYBRID_GDN2_LOCALIZE_DUMP_ROOT": str(DUMP_ROOT),
            "OLMOE3_HYBRID_GDN2_LOCALIZE_RUN_ID": RUN_ID,
            "OLMO_DEBUG_CHECK_LOCAL_LOSS": "1",
            "OLMO_DEBUG_DUMP_DIR": str(DUMP_ROOT),
            "OLMO_DEBUG_RUN_ID": RUN_ID,
            "OLMO_DEBUG_DUMP_STEPS": debug_steps,
        },
        recipe_suffix_override="-step24668-localizer-r1",
        description_suffix=(
            "read-only canonical-GDN2 480M Cx2 deterministic failure replay with "
            "pre-reduction loss and per-rank module localization"
        ),
    ).launch(show_logs=False)

    experiment = workload.experiment
    record = {
        "task_name": TARGET_TASK,
        "run_name": row["run_name"],
        "commit": commit,
        "source_checkpoint": str(checkpoint_dir / EXPECTED_CHECKPOINT),
        "expected_failure_step": EXPECTED_FAILURE_STEP,
        "localize_start_step": LOCALIZE_START_STEP,
        "hard_stop_step": HARD_STOP_STEP,
        "rank_microbatch_sequences": row["rank_microbatch_sequences"],
        "global_batch_size": row["global_batch_size"],
        "world_size": row["world_size"],
        "expert_parallel_size": row["expert_parallel_size"],
        "checkpoint_writes": False,
        "wandb": False,
        "dump_dir": str(output_dir),
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
