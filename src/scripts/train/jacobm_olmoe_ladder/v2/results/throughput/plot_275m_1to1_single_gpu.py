#!/usr/bin/env python3
"""Plot the active-matched 275M 1:1 single-GPU throughput comparison."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from scripts.train.jacobm_olmoe_ladder.v2.models.geometry_matched_275m import (
    build_geometry_matched_one_to_one_model_config,
)


SCRIPT_DIR = Path(__file__).resolve().parent
V2_DIR = SCRIPT_DIR.parents[1]
DEFAULT_INPUT = SCRIPT_DIR / "275m_gdn_gdn2_swa_1to1_single_gpu.csv"
DEFAULT_OUTPUT = V2_DIR / "plots" / "throughput" / "275m_1to1_2m_comparison.png"
BATCH_TOKENS = 2_097_152

FAMILIES: tuple[dict[str, Any], ...] = (
    {
        "csv_family": "SWA-1to1",
        "mixer": "swa",
        "label": "SWA",
        "mixer_label": "SWA (local)",
        "color": "#4C78A8",
    },
    {
        "csv_family": "GDN1-1to1",
        "mixer": "gdn1",
        "label": "GDN1",
        "mixer_label": "GDN1 (recurrent)",
        "color": "#F58518",
    },
    {
        "csv_family": "GDN2-1to1",
        "mixer": "gdn2",
        "label": "GDN2",
        "mixer_label": "GDN2 (recurrent)",
        "color": "#8F63B8",
    },
)
FULL_ATTENTION_COLOR = "#34495E"


def load_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    selected = {
        row["family"]: row
        for row in rows
        if int(row["batch_tokens"]) == BATCH_TOKENS
        and int(row["gpus"]) == 1
        and row["path"] == "EP1/all-reduce"
    }
    expected = {str(family["csv_family"]) for family in FAMILIES}
    if set(selected) != expected:
        raise ValueError(
            f"expected exactly the 2 Mi single-GPU rows {sorted(expected)}, "
            f"found {sorted(selected)}"
        )
    if any(float(row["max_skipped_steps"]) != 0 for row in selected.values()):
        raise ValueError("cannot plot a run with skipped optimizer steps")
    return selected


def label_bars(axis: plt.Axes, bars: Any, labels: list[str], padding: int = 5) -> None:
    axis.bar_label(bars, labels=labels, padding=padding, fontsize=10, fontweight="bold")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = load_rows(args.input)
    records: list[dict[str, Any]] = []
    for family in FAMILIES:
        model = build_geometry_matched_one_to_one_model_config(str(family["mixer"]))
        row = rows[str(family["csv_family"])]
        hybrid_layers = tuple(range(0, model.n_layers, 2))
        full_layers = tuple(range(1, model.n_layers, 2))
        if len(hybrid_layers) != len(full_layers):
            raise ValueError(f"{family['label']} is not a strict 1:1 model")
        records.append(
            {
                **family,
                "layers": model.n_layers,
                "hybrid_layers": len(hybrid_layers),
                "full_layers": len(full_layers),
                "total_params_b": model.num_params / 1e9,
                "tflops": float(row["final10_median_tflops_gpu"]),
                "tps_k": float(row["final10_median_tps_gpu"]) / 1e3,
            }
        )

    labels = [str(record["label"]) for record in records]
    colors = [str(record["color"]) for record in records]
    positions = list(range(len(records)))

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    fig.patch.set_facecolor("#FAFAFA")
    for axis in axes.flat:
        axis.set_facecolor("#FAFAFA")

    architecture_axis = axes[0, 0]
    architecture_axis.barh(
        positions,
        [record["hybrid_layers"] for record in records],
        color=colors,
        edgecolor="white",
        height=0.58,
    )
    architecture_axis.barh(
        positions,
        [record["full_layers"] for record in records],
        left=[record["hybrid_layers"] for record in records],
        color=FULL_ATTENTION_COLOR,
        edgecolor="white",
        height=0.58,
    )
    architecture_axis.set_yticks(positions, labels)
    architecture_axis.invert_yaxis()
    architecture_axis.set_xlim(0, 13.2)
    architecture_axis.set_xlabel("Transformer layers")
    architecture_axis.set_title("Architecture: strict 1:1 layer mix", loc="left")
    architecture_axis.xaxis.grid(True, alpha=0.18)
    architecture_axis.set_axisbelow(True)
    for index, record in enumerate(records):
        architecture_axis.text(
            record["layers"] + 0.18,
            index,
            f"{record['layers']} total",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
        architecture_axis.text(
            record["hybrid_layers"] / 2,
            index,
            str(record["mixer_label"]),
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            fontweight="bold",
        )
        architecture_axis.text(
            record["hybrid_layers"] + record["full_layers"] / 2,
            index,
            "Full attention",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            fontweight="bold",
        )
    params_axis = axes[0, 1]
    params_bars = params_axis.bar(
        labels,
        [record["total_params_b"] for record in records],
        color=colors,
        width=0.62,
    )
    params_axis.set_ylim(0, 4.2)
    params_axis.set_ylabel("Total parameters (billions)")
    params_axis.set_title("Stored model parameters", loc="left")
    params_axis.yaxis.grid(True, alpha=0.18)
    params_axis.set_axisbelow(True)
    label_bars(
        params_axis,
        params_bars,
        [f"{record['total_params_b']:.3f}B" for record in records],
    )

    tflops_axis = axes[1, 0]
    tflops_bars = tflops_axis.bar(
        labels,
        [record["tflops"] for record in records],
        color=colors,
        width=0.62,
    )
    tflops_axis.set_ylim(0, 660)
    tflops_axis.set_ylabel("TFLOPs/GPU")
    tflops_axis.set_title("Compute throughput", loc="left")
    tflops_axis.yaxis.grid(True, alpha=0.18)
    tflops_axis.set_axisbelow(True)
    label_bars(
        tflops_axis,
        tflops_bars,
        [f"{record['tflops']:.1f}" for record in records],
    )

    tps_axis = axes[1, 1]
    tps_bars = tps_axis.bar(
        labels,
        [record["tps_k"] for record in records],
        color=colors,
        width=0.62,
    )
    tps_axis.set_ylim(0, 365)
    tps_axis.set_ylabel("Tokens/s/GPU (thousands)")
    tps_axis.set_title("Raw token throughput", loc="left")
    tps_axis.yaxis.grid(True, alpha=0.18)
    tps_axis.set_axisbelow(True)
    label_bars(
        tps_axis,
        tps_bars,
        [f"{record['tps_k']:.1f}k" for record in records],
    )

    fig.suptitle(
        "275M 1:1 attention-ratio models — single B300, 2 Mi-token batch",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.955,
        "MB16 • 8,192-token context • EP1/all-reduce • final-10-step medians",
        ha="center",
        fontsize=11,
        color="#4B5563",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
