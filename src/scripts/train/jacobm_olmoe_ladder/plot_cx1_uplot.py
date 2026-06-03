#!/usr/bin/env python
"""Render a Cx1 LR U-plot from W&B train/CE loss histories."""

from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import wandb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="ai2-llm/jacobm-olmoe-ladder")
    parser.add_argument("--output", default="/tmp/olmoe3-plots/olmoe3_275m_cx1_lr_uplot.png")
    args = parser.parse_args()

    api = wandb.Api()
    runs = list(api.runs(args.project))
    fields = ["_step", "train/CE loss", "throughput/total tokens"]
    points = []

    for run in runs:
        name = run.display_name
        if "olmoe3-tiny-275m-cx1" not in name or "b256k" not in name:
            continue

        match = re.search(
            r"lr(5e-3|3e-3|2e-3|1\.5e-3|1\.2e-3|1e-3|8e-4|7e-4|6e-4|5e-4|4e-4|3e-4)",
            name,
        )
        if match is None:
            continue

        lr_tag = match.group(1)
        history: list[tuple[int, float]] = []
        for row in run.scan_history(keys=fields, page_size=1000):
            loss = row.get("train/CE loss")
            step = row.get("_step")
            tokens = row.get("throughput/total tokens")
            if loss is None or step is None:
                continue
            if tokens is None:
                tokens = int(step) * 262_144
            history.append((int(tokens), float(loss)))

        if not history:
            continue

        history.sort()
        final_tokens = history[-1][0]
        vals_100 = [loss for tokens, loss in history if tokens >= final_tokens - 100_000_000]
        vals_250 = [loss for tokens, loss in history if tokens >= final_tokens - 250_000_000]
        vals_500 = [loss for tokens, loss in history if tokens >= final_tokens - 500_000_000]
        points.append(
            {
                "lr": float(lr_tag),
                "lr_tag": lr_tag,
                "avg100": statistics.mean(vals_100),
                "avg250": statistics.mean(vals_250),
                "avg500": statistics.mean(vals_500),
            }
        )

    points.sort(key=lambda point: point["lr"])
    xs = [point["lr"] for point in points]

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=180)
    ax.plot(
        xs,
        [point["avg100"] for point in points],
        color="#F97316",
        marker="s",
        linestyle="--",
        linewidth=1.8,
        markersize=5,
        label="avg last 100M tokens",
    )
    ax.plot(
        xs,
        [point["avg250"] for point in points],
        color="#2563EB",
        marker="o",
        linestyle="-",
        linewidth=2.5,
        markersize=6,
        label="avg last 250M tokens",
    )
    ax.plot(
        xs,
        [point["avg500"] for point in points],
        color="#16A34A",
        marker="^",
        linestyle=":",
        linewidth=2.0,
        markersize=6,
        label="avg last 500M tokens",
    )

    best = min(points, key=lambda point: point["avg250"])
    ax.scatter([best["lr"]], [best["avg250"]], s=95, color="#DC2626", zorder=5, label=f"best avg250M: {best['lr_tag']}")

    for point in points:
        ax.annotate(
            point["lr_tag"],
            (point["lr"], point["avg250"]),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8,
            color="#1F2937",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Learning rate (log scale)")
    ax.set_ylabel("train/CE loss")
    ax.set_title("OLMoE3 tiny 275M active Cx1 LR U-plot")
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", axis="x", alpha=0.12)
    ax.legend(frameon=False, loc="best")
    ax.text(
        0.01,
        0.02,
        "Batch: 256k tokens/step. Metric: W&B train/CE loss. Runs: Cx1, ~4.03B tokens.",
        transform=ax.transAxes,
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
