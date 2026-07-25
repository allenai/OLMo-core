#!/usr/bin/env python3
"""Replay the deterministic 1.2B Cx4 GDN2 failure without backward recomputation."""

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
MANIFEST = SCRIPT_DIR / "manifests" / "geometry_matched_scale_gdn2_nope_gated_full_candidate.yaml"
RECORD = SCRIPT_DIR / "generated" / "gdn2_no_recompute_replays.json"
TARGET_TASK = "1p2b-cx4"
EXPECTED_CHECKPOINT = "step9000"
EXPECTED_FAILURE_STEP = 9_059
HARD_STOP_STEP = 9_075
RANK_MICROBATCH_SEQUENCES = 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(MANIFEST)
    rows = validate(manifest)
    row = next(row for row in rows if str(row["task_name"]) == TARGET_TASK)

    checkpoint_dir = Path(manifest["experiment"]["checkpoint_root"]) / str(row["run_name"])
    checkpoints = sorted(
        path for path in checkpoint_dir.glob("step*") if path.is_dir()
    )
    if [path.name for path in checkpoints] != [EXPECTED_CHECKPOINT]:
        raise RuntimeError(
            f"Expected only {EXPECTED_CHECKPOINT} in {checkpoint_dir}, found "
            f"{[path.name for path in checkpoints]}"
        )

    rank_sequences = int(row["rank_sequences"])
    if rank_sequences % RANK_MICROBATCH_SEQUENCES:
        raise ValueError(
            f"Rank batch {rank_sequences} is not divisible by MB={RANK_MICROBATCH_SEQUENCES}"
        )

    print(
        f"Replay {row['run_name']} from {EXPECTED_CHECKPOINT}: world={row['world_size']} "
        f"EP={row['expert_parallel_size']} global_batch={row['global_batch_size']} "
        f"MB={RANK_MICROBATCH_SEQUENCES} accum={rank_sequences} "
        f"disable_recompute=True expected_failure={EXPECTED_FAILURE_STEP} "
        f"hard_stop={HARD_STOP_STEP} checkpoint_writes=False wandb=False"
    )
    if not args.submit:
        print("Dry run only; pass --submit to launch.")
        return

    source = manifest["source"]
    commit = validate_remote_commit(str(source["remote"]), str(source["branch"]))
    workload = recipe_for(
        manifest,
        row,
        commit=commit,
        debug_gradients=True,
        runtime_env_overrides={
            "OLMOE3_HYBRID_RANK_MICROBATCH_SEQUENCES": str(
                RANK_MICROBATCH_SEQUENCES
            ),
            "OLMOE3_HYBRID_HARD_STOP_STEPS": str(HARD_STOP_STEP),
            "OLMOE3_HYBRID_GDN2_DISABLE_RECOMPUTE": "1",
            # Keep checkpoint discovery/loading enabled while preventing the
            # diagnostic from modifying or pruning the canonical run directory.
            "OLMOE3_HYBRID_CHECKPOINTS": "1",
            "OLMOE3_HYBRID_CHECKPOINT_WRITES": "0",
            "OLMOE3_HYBRID_WANDB": "0",
        },
        recipe_suffix_override="-gdn2-no-recompute-mb1-r1",
        description_suffix=(
            "deterministic step9059 replay with FLA backward recomputation disabled, "
            "MB1, read-only checkpoint state, and all-rank gradient diagnostics"
        ),
    ).launch(show_logs=False)

    experiment = workload.experiment
    record = {
        "task_name": TARGET_TASK,
        "run_name": row["run_name"],
        "commit": commit,
        "source_checkpoint": str(checkpoint_dir / EXPECTED_CHECKPOINT),
        "expected_failure_step": EXPECTED_FAILURE_STEP,
        "hard_stop_step": HARD_STOP_STEP,
        "rank_microbatch_sequences": RANK_MICROBATCH_SEQUENCES,
        "global_batch_size": row["global_batch_size"],
        "world_size": row["world_size"],
        "expert_parallel_size": row["expert_parallel_size"],
        "disable_recompute": True,
        "checkpoint_writes": False,
        "wandb": False,
        "experiment_id": experiment.id,
        "task_ids": [task.id for task in experiment.tasks],
        "url": (
            "https://beaker.org/orgs/ai2/workspaces/"
            f"OLMo-3-moe-experiments/work/{experiment.id}"
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
