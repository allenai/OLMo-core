#!/usr/bin/env python3
"""Build the versioned Jacob OLMoDDP checkpoint publication manifest."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GCS_ROOT = "gs://ai2-llm/checkpoints/jacobm/olmoe3/olmo-ddp-converted/v1"
LOCAL_ROOT = Path(
    "/weka/oe-adapt-default/jacobm/olmoe3/olmo-ddp-migration/converted-checkpoints"
)
MODEL_ORDER = {"275m": 0, "480m": 1, "810m": 2, "1p2b": 3}
STAGE_ORDER = {"pretraining": 0, "midtraining": 1}
FAMILY_DESCRIPTIONS = {
    "baseline": "Baseline MoE ladder architecture",
    "expert_coarse_24e_top2": (
        "Expert-granularity intervention with 24 routed experts and top-2 routing"
    ),
    "expert_fine_96e_top8": (
        "Expert-granularity intervention with 96 routed experts and top-8 routing"
    ),
    "sparsity_high_96e_top4": (
        "Higher-total-sparsity intervention with 96 routed experts and top-4 routing"
    ),
    "sparsity_huge_192e_top4": (
        "Highest-total-sparsity intervention with 192 routed experts and top-4 routing"
    ),
    "shared_no_shared": "No-shared-expert intervention with active parameters matched",
    "dense0_shared": "Dense-schedule intervention with dense layer 0 and a shared expert",
    "dense2_shared": (
        "Dense-schedule intervention with dense layers through layer 2 and a shared expert"
    ),
    "dense4_shared": (
        "Dense-schedule intervention with dense layers through layer 4 and a shared expert"
    ),
    "qwen_active_4p5d": "Qwen-like active-parameter-matched 4.5d intervention",
    "qwen_true_3d": "Qwen-like true-3d plus depth intervention",
    "integration_wide": (
        "Wide integration architecture with 256 routed experts and top-8 routing"
    ),
    "integration_deep": (
        "Deep integration architecture with 256 routed experts and top-8 routing"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--pretrain",
        type=Path,
        default=repo_root / "JACOBM_DDP_PRETRAIN_CANDIDATES.tsv",
    )
    parser.add_argument(
        "--midtrain",
        type=Path,
        default=repo_root / "JACOBM_DDP_MIDTRAIN_CANDIDATES.tsv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "JACOBM_DDP_PUBLICATION_MANIFEST.json",
    )
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def midtrain_lr(run_name: str) -> str:
    match = re.search(r"-lr(.+)-r\d+$", run_name)
    if match is None:
        raise ValueError(f"Could not parse learning rate from {run_name!r}")
    return match.group(1).replace("p", ".")


def destination(stage: str, family: str, model: str, cx: int) -> str:
    return f"{GCS_ROOT}/{stage}/{family}/{model}/cx{cx}/"


def local_output(stage: str, run_name: str, step: int) -> Path:
    return LOCAL_ROOT / stage / run_name / f"step{step}"


def base_entry(
    *,
    stage: str,
    family: str,
    model: str,
    cx: int,
    run_name: str,
    step: int,
    source: str,
) -> dict[str, Any]:
    gcs_uri = destination(stage, family, model, cx)
    source_path = Path(source)
    source_config = source_path / "config.json"
    output = local_output(stage, run_name, step)
    existing_output = output if output.exists() else None
    return {
        "id": f"{stage}/{family}/{model}/cx{cx}",
        "stage": stage,
        "family": family,
        "family_description": FAMILY_DESCRIPTIONS[family],
        "model_size": model,
        "data_multiple": cx,
        "data_multiple_name": f"Cx{cx}",
        "source_run_name": run_name,
        "source_step": step,
        "source_checkpoint": source,
        "source_config": str(source_config) if source_config.is_file() else None,
        "source_config_status": (
            "colocated" if source_config.is_file() else "reconstruction_required"
        ),
        "local_output": str(output),
        "existing_local_output": (
            str(existing_output) if existing_output is not None else None
        ),
        "gcs_uri": gcs_uri,
        "readme_uri": f"{gcs_uri}README.md",
        "optimizer_state_included": False,
        "trainer_state_included": False,
        "required_verification_protocol": "strict_tensor_and_exact_logits_v1",
        "publication_state": "pending",
    }


def build_entries(
    pretrain_path: Path, midtrain_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entries: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    for row in read_tsv(pretrain_path):
        if row["exists"] != "True":
            excluded.append(
                {
                    "stage": "pretraining",
                    "family": row["family"],
                    "model_size": row["model"],
                    "data_multiple": int(row["cx"]),
                    "reason": "checkpoint_not_run",
                }
            )
            continue
        entry = base_entry(
            stage="pretraining",
            family=row["family"],
            model=row["model"],
            cx=int(row["cx"]),
            run_name=row["run_name"],
            step=int(row["step"]),
            source=row["source"],
        )
        entry.update(
            learning_rate=row["lr"],
            final_training_loss=float(row["loss"]),
            wandb_run_id=row["run_id"],
        )
        entries.append(entry)

    for row in read_tsv(midtrain_path):
        if row["status"] not in {"ready", "converted_verified"}:
            excluded.append(
                {
                    "stage": "midtraining",
                    "family": row["family"],
                    "model_size": row["model"],
                    "data_multiple": int(row["cx"]),
                    "reason": f"status_{row['status']}",
                }
            )
            continue
        entry = base_entry(
            stage="midtraining",
            family=row["family"],
            model=row["model"],
            cx=int(row["cx"]),
            run_name=row["run_name"],
            step=int(row["step"]),
            source=row["source"],
        )
        entry.update(
            learning_rate=midtrain_lr(row["run_name"]),
            prior_conversion_status=(
                "exact_logits_verified_but_requires_strict_tensor_v1"
                if row["status"] == "converted_verified"
                else None
            ),
        )
        entries.append(entry)

    entries.sort(
        key=lambda item: (
            STAGE_ORDER[item["stage"]],
            item["family"],
            MODEL_ORDER[item["model_size"]],
            item["data_multiple"],
        )
    )
    excluded.sort(
        key=lambda item: (
            STAGE_ORDER[item["stage"]],
            item["family"],
            MODEL_ORDER[item["model_size"]],
            item["data_multiple"],
        )
    )
    return entries, excluded


def validate(entries: list[dict[str, Any]]) -> None:
    for field in ("id", "source_checkpoint", "local_output", "gcs_uri"):
        values = [entry[field] for entry in entries]
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate manifest {field}")
    for entry in entries:
        source = Path(entry["source_checkpoint"])
        if not (source / "model_and_optim" / ".metadata").is_file():
            raise FileNotFoundError(
                f"Missing model_and_optim/.metadata: {source}"
            )
        expected_suffix = (
            f"/{entry['stage']}/{entry['family']}/{entry['model_size']}/"
            f"cx{entry['data_multiple']}/"
        )
        if not entry["gcs_uri"].endswith(expected_suffix):
            raise ValueError(f"Unexpected destination layout: {entry['gcs_uri']}")


def main() -> None:
    args = parse_args()
    entries, excluded = build_entries(args.pretrain, args.midtrain)
    validate(entries)
    manifest = {
        "schema_version": 1,
        "publication_version": "v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gcs_root": f"{GCS_ROOT}/",
        "selection_rule": (
            "Canonical final checkpoint for every completed selected pretraining and "
            "midtraining family/model-size/data-multiple cell"
        ),
        "model_count": len(entries),
        "pretraining_model_count": sum(
            entry["stage"] == "pretraining" for entry in entries
        ),
        "midtraining_model_count": sum(
            entry["stage"] == "midtraining" for entry in entries
        ),
        "colocated_source_config_count": sum(
            entry["source_config_status"] == "colocated" for entry in entries
        ),
        "config_reconstruction_required_count": sum(
            entry["source_config_status"] == "reconstruction_required"
            for entry in entries
        ),
        "excluded_count": len(excluded),
        "excluded": excluded,
        "models": entries,
    }
    args.output.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(entries)} models to {args.output}")


if __name__ == "__main__":
    main()
