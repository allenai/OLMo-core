#!/usr/bin/env python3
"""Plot C4 validation CE for the checkpoints selected by a pretraining wave."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

V2_DIR = Path(__file__).resolve().parent
V1_DIR = V2_DIR.parent / "v1"
DEFAULT_TRAIN_RESULTS = (
    V2_DIR
    / "results"
    / "pretraining"
    / "geometry_gdn_ev2_rope_gated"
    / "results.json"
)
DEFAULT_VALIDATION_RESULTS = V2_DIR / "results" / "validation" / "hybrid_full.json"
DEFAULT_V1_EVAL_RESULTS = V1_DIR / "results" / "in_loop_evals.json"
DEFAULT_OUTPUT = (
    V2_DIR
    / "plots"
    / "pretraining"
    / "geometry_gdn_ev2_rope_gated"
    / "c4_validation_fixed_lr_scale_comparison.png"
)
DEFAULT_RESULT_BASE = (
    V2_DIR
    / "results"
    / "pretraining"
    / "geometry_gdn_ev2_rope_gated"
    / "c4_validation_results"
)
C4_METRIC = "eval/lm/c4_en-validation/CE loss"
MODELS = ("275m", "480m", "810m", "1p2b")
CXS = (1, 2, 4, 8)
INTERVENTION = "geometry_gdn_ev2_rope_gated"
VARIANTS = (
    "wide_integration",
    "hybrid_gdn_ev1",
    "geometry_gdn_ev2",
    "geometry_gdn_ev2_nope",
    "geometry_gdn_ev2_nope_gated",
    INTERVENTION,
)
STYLES = {
    "wide_integration": {
        "label": "wide integration (SWA)",
        "color": "#111827",
        "linestyle": "--",
    },
    "hybrid_gdn_ev1": {
        "label": "hybrid (GDN, expand_v=1)",
        "color": "#2563eb",
        "linestyle": ":",
    },
    "geometry_gdn_ev2": {
        "label": "geometry-matched GDN (expand_v=2)",
        "color": "#dc2626",
        "linestyle": "-.",
    },
    "geometry_gdn_ev2_nope": {
        "label": "geometry-matched GDN + NoPE",
        "color": "#7c3aed",
        "linestyle": (0, (3, 1, 1, 1)),
    },
    "geometry_gdn_ev2_nope_gated": {
        "label": "geometry-matched GDN + NoPE + gated attention",
        "color": "#059669",
        "linestyle": "--",
    },
    INTERVENTION: {
        "label": "geometry-matched GDN + RoPE + gated attention",
        "color": "#d97706",
        "linestyle": "-",
    },
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def selected_training_points(payload: dict[str, Any]) -> dict[tuple[str, int, str], dict]:
    if payload["wave"] != INTERVENTION:
        raise ValueError(f"expected training wave {INTERVENTION!r}, got {payload['wave']!r}")
    selected: dict[tuple[str, int, str], dict] = {}
    for result in payload["results"]:
        model = str(result["model"])
        cx = int(result["cx"])
        for variant, point in result["references"].items():
            if point is not None and variant in VARIANTS:
                selected[(model, cx, variant)] = point
        point = result["intervention_result"]
        if point is not None:
            selected[(model, cx, INTERVENTION)] = point
    return selected


def collect_c4_records(
    selected: dict[tuple[str, int, str], dict],
    validation_payload: dict[str, Any],
    v1_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    validation_by_source = {
        record["source_run"]: record for record in validation_payload["records"]
    }
    v1_by_run_id = {record["run_id"]: record for record in v1_payload["records"]}
    records: list[dict[str, Any]] = []

    for model in MODELS:
        for cx in CXS:
            for variant in VARIANTS:
                point = selected.get((model, cx, variant))
                record: dict[str, Any] = {
                    "model": model,
                    "cx": cx,
                    "variant": variant,
                    "variant_label": STYLES[variant]["label"],
                    "training_selection_mode": (
                        "observed_train_loss_optimum" if model == "275m" else "wide_lr_transfer"
                    ),
                    "source_training_run_id": point["run_id"] if point else None,
                    "source_training_run_name": point["name"] if point else None,
                    "learning_rate": point["lr"] if point else None,
                    "c4_validation_ce": None,
                    "validation_state": "training_result_missing",
                    "validation_run_id": None,
                    "validation_url": None,
                }
                if point is None:
                    records.append(record)
                    continue

                if variant == "wide_integration":
                    eval_record = v1_by_run_id.get(point["run_id"])
                    if eval_record is None:
                        record["validation_state"] = "not_registered"
                    else:
                        value = (
                            eval_record["metrics"].get(C4_METRIC)
                            if eval_record["state"] == "finished"
                            else None
                        )
                        record.update(
                            validation_state=eval_record["state"],
                            validation_run_id=eval_record["run_id"],
                            validation_url=eval_record["url"],
                            c4_validation_ce=float(value) if value is not None else None,
                        )
                else:
                    eval_record = validation_by_source.get(point["name"])
                    if eval_record is None:
                        record["validation_state"] = "not_registered"
                    else:
                        value = (
                            eval_record["eval_metrics"].get(C4_METRIC)
                            if eval_record["state"] == "finished"
                            else None
                        )
                        record.update(
                            validation_state=eval_record["state"],
                            validation_run_id=eval_record["wandb_id"],
                            validation_url=eval_record["wandb_url"],
                            c4_validation_ce=float(value) if value is not None else None,
                        )
                records.append(record)
    return records


def plot(records: list[dict[str, Any]], output_path: Path) -> Path:
    by_key = {(record["model"], record["cx"], record["variant"]): record for record in records}
    fig, axes = plt.subplots(1, len(MODELS), figsize=(14.4, 4.9), squeeze=False)
    fig.patch.set_facecolor("white")

    for axis, model in zip(axes[0], MODELS, strict=True):
        axis.set_facecolor("white")
        for variant in VARIANTS:
            style = STYLES[variant]
            values = [by_key[(model, cx, variant)]["c4_validation_ce"] for cx in CXS]
            if not any(value is not None for value in values):
                continue
            y = np.array([float(value) if value is not None else np.nan for value in values])
            axis.plot(
                CXS,
                y,
                marker="o",
                markersize=5,
                linewidth=2.0,
                color=style["color"],
                linestyle=style["linestyle"],
                label=style["label"],
            )

        pending_intervention = []
        for cx in CXS:
            record = by_key[(model, cx, INTERVENTION)]
            if record["c4_validation_ce"] is None:
                pending_intervention.append(cx)
                continue
            axis.annotate(
                f"LR {record['learning_rate']:.2g}",
                (cx, record["c4_validation_ce"]),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=7,
                color=STYLES[INTERVENTION]["color"],
            )
        if pending_intervention:
            pending = ", ".join(f"Cx{cx}" for cx in pending_intervention)
            axis.text(
                0.04,
                0.04,
                f"gated-RoPE validation pending: {pending}",
                transform=axis.transAxes,
                fontsize=7,
                color="#6b7280",
                ha="left",
                va="bottom",
            )

        mode = "observed train-loss optimum" if model == "275m" else "wide-LR transfer"
        axis.set_xscale("log", base=2)
        axis.set_xticks(CXS)
        axis.set_xticklabels(tuple(f"Cx{cx}" for cx in CXS))
        axis.set_xlabel("data multiple")
        axis.set_title(f"{model}\n{mode}", fontsize=10)
        axis.grid(True, which="both", alpha=0.25)
    axes[0][0].set_ylabel("C4 validation CE loss")

    handles, labels = axes[0][0].get_legend_handles_labels()
    legend_columns = math.ceil(len(handles) / 2)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=legend_columns,
        bbox_to_anchor=(0.5, -0.05),
        frameon=False,
    )
    fig.suptitle("Geometry-matched gated-RoPE comparison: C4 validation loss")
    fig.text(
        0.5,
        0.925,
        "Same selected checkpoints as the train-loss plot • finished evaluations only • lower is better",
        ha="center",
        color="#4b5563",
        fontsize=9,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.13, 1, 0.90))
    fig.savefig(output_path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_results(records: list[dict[str, Any]], output_base: Path) -> tuple[Path, Path]:
    generated_at = datetime.now(UTC).isoformat()
    payload = {
        "generated_at": generated_at,
        "metric": C4_METRIC,
        "selection_rule": (
            "Use the same checkpoints selected by the final-250M training-loss comparison: "
            "observed-best learning rates at 275M and wide-optimal transferred learning rates "
            "at larger sizes. Never substitute a partial or running validation result."
        ),
        "variants": list(VARIANTS),
        "records": records,
    }
    json_path = output_base.with_suffix(".json")
    md_path = output_base.with_suffix(".md")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# C4 validation comparison for selected gated-RoPE checkpoints",
        "",
        f"Generated: `{generated_at}`",
        "",
        f"Metric: `{C4_METRIC}` (lower is better).",
        "",
        "The checkpoint selection is identical to the final-250M training-loss comparison: ",
        "observed-best learning rates at 275M and wide-optimal LR transfers at larger sizes.",
        "Running, missing, and unregistered validation cells remain pending.",
        "",
        "| Model | Cx | Variant | Selection | LR | C4 validation CE | Validation state | W&B |",
        "|---|---:|---|---|---:|---:|---|---|",
    ]
    for record in records:
        value = record["c4_validation_ce"]
        value_text = f"{value:.6f}" if value is not None else "—"
        lr = record["learning_rate"]
        lr_text = f"{lr:.2g}" if lr is not None else "—"
        link = (
            f"[{record['validation_run_id']}]({record['validation_url']})"
            if record["validation_url"]
            else "—"
        )
        lines.append(
            f"| {record['model']} | Cx{record['cx']} | {record['variant_label']} | "
            f"{record['training_selection_mode']} | {lr_text} | {value_text} | "
            f"{record['validation_state']} | {link} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-results", type=Path, default=DEFAULT_TRAIN_RESULTS)
    parser.add_argument("--validation-results", type=Path, default=DEFAULT_VALIDATION_RESULTS)
    parser.add_argument("--v1-eval-results", type=Path, default=DEFAULT_V1_EVAL_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--result-base", type=Path, default=DEFAULT_RESULT_BASE)
    args = parser.parse_args()

    selected = selected_training_points(load_json(args.train_results))
    records = collect_c4_records(
        selected,
        load_json(args.validation_results),
        load_json(args.v1_eval_results),
    )
    output = plot(records, args.output)
    result_paths = write_results(records, args.result_base)
    print(output)
    for path in result_paths:
        print(path)


if __name__ == "__main__":
    main()
