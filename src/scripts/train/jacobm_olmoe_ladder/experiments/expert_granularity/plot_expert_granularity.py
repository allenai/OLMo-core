#!/usr/bin/env python
"""Plot expert-granularity U-curves from W&B history."""

from __future__ import annotations

import argparse
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb


PROJECT = "ai2-llm/jacobm-olmoe-ladder"
LOSS_KEY = "train/CE loss"
TOKENS_KEY = "throughput/total tokens"
FIELDS = ["_step", TOKENS_KEY, LOSS_KEY]
LR_RE = re.compile(r"lr([0-9]+(?:\.[0-9]+)?e-[0-9]+)")


@dataclass(frozen=True)
class Point:
    variant: str
    cx: int
    lr: float
    lr_tag: str
    loss: float
    state: str
    tokens_b: float
    run_id: str
    name: str


def mean_loss(points: list[tuple[float, float]], end_tokens: float, window_tokens: int) -> float:
    vals = [loss for tokens, loss in points if end_tokens - window_tokens <= tokens <= end_tokens]
    return statistics.mean(vals) if vals else math.nan


def history_loss(run, window_m: int) -> tuple[float, float] | None:
    points: list[tuple[float, float]] = []
    for row in run.scan_history(keys=FIELDS, page_size=10000):
        tokens = row.get(TOKENS_KEY)
        loss = row.get(LOSS_KEY)
        if tokens is None or loss is None:
            continue
        points.append((float(tokens), float(loss)))
    if not points:
        return None
    points.sort()
    end_tokens = points[-1][0]
    return mean_loss(points, end_tokens, window_m * 1_000_000), end_tokens / 1e9


def lr_from_name(name: str) -> tuple[str, float] | None:
    match = LR_RE.search(name)
    if match is None:
        return None
    tag = match.group(1)
    return tag, float(tag)


def cx_from_name(name: str) -> int | None:
    match = re.search(r"cx([0-9]+)", name)
    return int(match.group(1)) if match else None


def baseline_variant_name(name: str) -> str | None:
    if "olmoe3-tiny-275m-cx" not in name:
        return None
    if "gpu2-ep1mb16" in name and "cx1" in name:
        return "baseline_48e_top4"
    if "gpu2-ep1mb16" in name and "cx2" in name:
        return "baseline_48e_top4"
    if "gpu4-ep1mb16" in name and "cx4" in name:
        return "baseline_48e_top4"
    if "gpu4-ep1mb8" in name and "cx8" in name:
        return "baseline_48e_top4"
    return None


def expert_variant_name(name: str) -> str | None:
    if "eg24e2k" in name:
        return "coarse_24e_top2"
    if "eg96e8k" in name:
        return "fine_96e_top8"
    if "eg192e16k" in name:
        return "extreme_192e_top16"
    if "eg384e32k" in name:
        return "ultra_384e_top32"
    return None


def load_points(project: str, window_m: int) -> list[Point]:
    api = wandb.Api(timeout=90)
    points: list[Point] = []
    runs = list(
        api.runs(
            project,
            filters={
                "$or": [
                    {"tags": {"$all": ["exp_expert_granularity"]}},
                    {
                        "display_name": {
                            "$regex": "olmoe3-tiny-275m-cx(1|2|4|8).*gpu(2|4)-ep1mb(8|16)"
                        }
                    },
                ]
            },
        )
    )
    for run in runs:
        name = run.display_name or run.name
        if any(marker in name.lower() for marker in ("smoke", "sanity", "evaltest")):
            continue

        variant = expert_variant_name(name) or baseline_variant_name(name)
        if variant is None:
            continue
        cx = cx_from_name(name)
        lr_info = lr_from_name(name)
        if cx not in {1, 2, 4, 8} or lr_info is None:
            continue

        loss_info = history_loss(run, window_m)
        if loss_info is None or math.isnan(loss_info[0]):
            continue
        lr_tag, lr = lr_info
        loss, tokens_b = loss_info
        points.append(
            Point(
                variant=variant,
                cx=cx,
                lr=lr,
                lr_tag=lr_tag,
                loss=loss,
                state=run.state,
                tokens_b=tokens_b,
                run_id=run.id,
                name=name,
            )
        )

    points.sort(key=lambda p: (p.cx, p.variant, p.lr, p.name))
    return points


def fit_lr(group: list[Point]) -> tuple[float, float] | None:
    finished = sorted([p for p in group if p.state == "finished"], key=lambda p: p.lr)
    if len(finished) < 3:
        return None
    best_idx = min(range(len(finished)), key=lambda idx: finished[idx].loss)
    if best_idx == 0 or best_idx == len(finished) - 1:
        return None
    fit_group = finished[best_idx - 1 : best_idx + 2]
    x = np.array([math.log10(p.lr) for p in fit_group])
    y = np.array([p.loss for p in fit_group])
    a, b, c = np.polyfit(x, y, 2)
    if a <= 0:
        return None
    opt_log_lr = -b / (2 * a)
    if not min(x) <= opt_log_lr <= max(x):
        return None
    opt_lr = 10**opt_log_lr
    opt_loss = float(a * opt_log_lr**2 + b * opt_log_lr + c)
    return opt_lr, opt_loss


def plot_cx(points: list[Point], cx: int, out_path: Path, window_m: int) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    labels = {
        "baseline_48e_top4": "baseline 48E/top4",
        "coarse_24e_top2": "coarse 24E/top2",
        "fine_96e_top8": "fine 96E/top8",
        "extreme_192e_top16": "extreme 192E/top16",
        "ultra_384e_top32": "ultra 384E/top32",
    }
    variants = [
        "baseline_48e_top4",
        "coarse_24e_top2",
        "fine_96e_top8",
        "extreme_192e_top16",
        "ultra_384e_top32",
    ]
    for variant in variants:
        group = sorted([p for p in points if p.cx == cx and p.variant == variant], key=lambda p: p.lr)
        if not group:
            continue
        finished = [p for p in group if p.state == "finished"]
        running = [p for p in group if p.state != "finished"]
        color = None
        if finished:
            (line,) = ax.plot(
                [p.lr for p in finished],
                [p.loss for p in finished],
                marker="o",
                linewidth=1.8,
                label=labels[variant],
            )
            color = line.get_color()
        if running:
            ax.scatter(
                [p.lr for p in running],
                [p.loss for p in running],
                marker="x",
                alpha=0.5,
                color=color,
                label=f"{labels[variant]} running",
            )
        fit = fit_lr(group)
        if fit is not None:
            lr, loss = fit
            ax.axvline(lr, color=color, linestyle=":", linewidth=1.2, alpha=0.75)
            ax.scatter([lr], [loss], marker="*", s=85, color=color, edgecolor="black", linewidth=0.4)
            ax.annotate(
                f"{lr:.2g}",
                (lr, loss),
                textcoords="offset points",
                xytext=(6, -14),
                ha="left",
                fontsize=8,
                color=color,
            )
        for point in group:
            ax.annotate(
                point.lr_tag,
                (point.lr, point.loss),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
                alpha=0.85 if point.state == "finished" else 0.55,
            )
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel(f"train CE avg{window_m}M")
    ax.set_title(f"Expert granularity 275M Cx{cx}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--window-m", type=int, default=250)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parents[2] / "plots" / "expert_granularity",
    )
    args = parser.parse_args()

    points = load_points(args.project, args.window_m)
    for cx in sorted({point.cx for point in points}):
        plot_cx(points, cx, args.output_dir / f"275m_cx{cx}_uplot.png", args.window_m)


if __name__ == "__main__":
    main()
