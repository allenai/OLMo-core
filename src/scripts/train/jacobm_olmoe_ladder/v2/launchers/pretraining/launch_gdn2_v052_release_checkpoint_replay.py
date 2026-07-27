#!/usr/bin/env python3
"""Replay the deterministic 275M GDN2 failure with the FLA v0.5.2 release tag."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.train.jacobm_olmoe_ladder.v2.launchers.pretraining.launch_geometry_matched_scale_full import (
    recipe_for,
)
from scripts.train.jacobm_olmoe_ladder.v2.launchers.pretraining.launch_geometry_matched_scale_nope_smokes import (
    load_manifest,
    validate_remote_commit,
)

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_MANIFEST = (
    SCRIPT_DIR / "manifests" / "geometry_matched_scale_gdn2_ev1_noneg_nope_gated_full.yaml"
)
RECORD = SCRIPT_DIR / "generated" / "gdn2_v052_release_checkpoint_replay_submissions.json"

RUN_NAME = "pt-275m-geometry-hybrid-gdn2-ev2-nope-gated-cx8-lr1p6e-3-r1"
EXPECTED_CHECKPOINT = "step36500"
EXPECTED_FAILURE_STEP = 36_768
LOCALIZE_START_STEP = 36_740
HARD_STOP_STEP = 37_000
FLA_OVERLAY = "/tmp/fla-gdn2-v0.5.2-release"
FLA_SPEC = (
    "flash-linear-attention[cuda] @ git+https://github.com/fla-org/"
    "flash-linear-attention.git@v0.5.2"
)
FLA_COMMIT = "9c8e42e762fce087c27b673af4922795d9edb85e"
DUMP_ROOT = Path(
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/"
    "olmo-ddp/debug/gdn2-v052-release-replay"
)
RUN_ID = "275m-gdn2-ev2-neg-cx8-lr1p6e3-step36768-v052-release-r1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(BASE_MANIFEST)
    manifest["experiment"]["description"] = (
        "Read-only deterministic 275M GDN2 expand_v=2 negative-eigenvalue "
        "checkpoint replay with the FLA v0.5.2 release tag"
    )
    manifest["training"]["model_variant"] = "geometry_275m_gdn2_ev2_nope_gated"
    manifest["beaker"]["auto_resume"] = False

    row = {
        "task_name": "275m-cx8-lr1p6e-3-v052-release-replay",
        "run_name": RUN_NAME,
        "model_size": "275m",
        "cx": 8,
        "learning_rate": "1.6e-3",
        "global_batch_size": 786_432,
        "num_nodes": 1,
        "gpus_per_node": 8,
        "world_size": 8,
        "expert_parallel_size": 1,
        "expert_parallel_path": "rowwise_nvshmem",
        "rank_microbatch_sequences": 12,
    }

    checkpoint_dir = Path(manifest["experiment"]["checkpoint_root"]) / RUN_NAME
    checkpoints = sorted(path for path in checkpoint_dir.glob("step*") if path.is_dir())
    if [path.name for path in checkpoints] != [EXPECTED_CHECKPOINT]:
        raise RuntimeError(
            f"Expected only {EXPECTED_CHECKPOINT} in {checkpoint_dir}, found "
            f"{[path.name for path in checkpoints]}"
        )
    output_dir = DUMP_ROOT / RUN_ID
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing diagnostic output {output_dir}")

    source = manifest["source"]
    commit = validate_remote_commit(str(source["remote"]), str(source["branch"]))
    print(
        f"Replay {RUN_NAME} from {EXPECTED_CHECKPOINT}: FLA tag=v0.5.2 "
        f"commit={FLA_COMMIT} world=8 EP=1 MB=12 global_batch=786432 "
        f"expected_failure={EXPECTED_FAILURE_STEP} hard_stop={HARD_STOP_STEP} "
        "checkpoint_writes=False wandb=False"
    )
    if not args.submit:
        print("Dry run only; pass --submit to launch.")
        return

    workload = recipe_for(
        manifest,
        row,
        commit=commit,
        debug_gradients=True,
        runtime_env_overrides={
            "OLMOE3_HYBRID_HARD_STOP_STEPS": str(HARD_STOP_STEP),
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
        },
        recipe_suffix_override="-fla-v0p5p2-release-replay-r1",
        description_suffix=(
            "same checkpoint, optimizer, data position, LR, batch, and architecture; "
            "only FLA changes from cbb0a72 to release v0.5.2"
        ),
        gdn2_fla_overlay=FLA_OVERLAY,
        gdn2_fla_spec=FLA_SPEC,
        gdn2_fla_expected_commit=FLA_COMMIT,
    ).launch(show_logs=False)

    experiment = workload.experiment
    record = {
        "run_name": RUN_NAME,
        "source_checkpoint": str(checkpoint_dir / EXPECTED_CHECKPOINT),
        "source_commit": commit,
        "fla_tag": "v0.5.2",
        "fla_commit": FLA_COMMIT,
        "previous_fla_commit": "cbb0a72efb55c18ca0ef4f298298317573ad2cb3",
        "expected_failure_step": EXPECTED_FAILURE_STEP,
        "localize_start_step": LOCALIZE_START_STEP,
        "hard_stop_step": HARD_STOP_STEP,
        "checkpoint_writes": False,
        "wandb": False,
        "dump_dir": str(output_dir),
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
