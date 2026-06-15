#!/usr/bin/env python
"""Plot dense-layer schedule U-curves from W&B history."""

from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb

LADDER_DIR = Path(__file__).parents[2]
if str(LADDER_DIR) not in sys.path:
    sys.path.insert(0, str(LADDER_DIR))

from experiment_summary_plots import SummaryVariant, plot_observed_best_summary
from wandb_cache import DEFAULT_CACHE_DIR, scan_history_cached


PROJECT = "ai2-llm/jacobm-olmoe-ladder"
LOSS_KEY = "train/CE loss"
TOKENS_KEY = "throughput/total tokens"
FIELDS = ["_step", TOKENS_KEY, LOSS_KEY]
LR_RE = re.compile(r"lr([0-9]+(?:\.[0-9]+)?e-[0-9]+)")

EXPERIMENT_TAG = "exp_dense_schedule"
OUTPUT_SUBDIR = "dense_schedule"
PLOT_TITLE = "Dense-layer schedule"
VARIANT_LABELS = {
    "baseline_dense1_shared": "baseline dense1 + shared",
    "baseline_dense1_shared_b384k": "baseline dense1 + shared (b384k)",
    "dense0_shared": "dense0 + shared",
    "dense2_shared": "dense2 + shared",
    "dense4_shared": "dense4 + shared",
}
VARIANT_ORDER = [
    "baseline_dense1_shared",
    "baseline_dense1_shared_b384k",
    "dense0_shared",
    "dense2_shared",
    "dense4_shared",
]
SUMMARY_VARIANTS = [
    SummaryVariant(
        "baseline",
        ("baseline_dense1_shared", "baseline_dense1_shared_b384k"),
        "baseline dense1 + shared",
        color="black",
        linestyle="--",
    ),
    SummaryVariant("dense0_shared", ("dense0_shared",), "dense0 + shared"),
    SummaryVariant("dense2_shared", ("dense2_shared",), "dense2 + shared"),
    SummaryVariant("dense4_shared", ("dense4_shared",), "dense4 + shared"),
]


@dataclass(frozen=True)
class Point:
    model: str
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


def history_loss(
    run,
    *,
    project: str,
    cache_dir: Path,
    window_m: int,
    refresh_cache: bool = False,
    refresh_stale_cache: bool = False,
) -> tuple[float, float] | None:
    points: list[tuple[float, float]] = []
    for row in scan_history_cached(
        run,
        project=project,
        keys=FIELDS,
        cache_dir=cache_dir,
        refresh_cache=refresh_cache,
        refresh_stale_cache=refresh_stale_cache,
        page_size=10000,
    ):
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


def model_from_name(name: str) -> str | None:
    if "tiny-275m" in name or "sp-275m" in name or "se-275m" in name or "eg-275m" in name or "ds-275m" in name:
        return "275m"
    if "mid_480m" in name or "mid-480m" in name or "m480-cx" in name or "eg-480m" in name or "480m" in name:
        return "480m"
    if "810m" in name:
        return "810m"
    if "1p2b" in name:
        return "1p2b"
    return None


def baseline_variant_name(name: str) -> str | None:
    if "olmoe3-tiny-275m-cx" in name:
        if "b384k" in name and "cx2" in name:
            return "baseline_dense1_shared_b384k"
        if "gpu2-ep1mb16" in name and "cx2" in name:
            return "baseline_dense1_shared"
        if "cx" in name:
            return "baseline_dense1_shared"
        return None

    if "m480-cx" in name or "olmoe3-moe-a0-480m-cx" in name:
        if "cx2" in name and "b384k" in name:
            return "baseline_dense1_shared_b384k"
        return "baseline_dense1_shared"

    if "olmoe3-moe-a0-810m-cx" in name:
        if "cx2" in name and "b384k" in name:
            return "baseline_dense1_shared_b384k"
        return "baseline_dense1_shared"

    if "olmoe3-moe-a0-1p2b-cx" in name:
        if "cx2" in name and "b384k" in name:
            return "baseline_dense1_shared_b384k"
        return "baseline_dense1_shared"

    return None


def experiment_variant_name(name: str) -> str | None:
    if "ds0-sh" in name or "dense0_shared" in name:
        return "dense0_shared"
    if "ds2-sh" in name or "dense2_shared" in name:
        return "dense2_shared"
    if "ds4-sh" in name or "dense4_shared" in name:
        return "dense4_shared"
    if "ds1-sh" in name or "dense1_shared" in name:
        return "baseline_dense1_shared"
    return None


def load_points(
    project: str,
    cache_dir: Path,
    window_m: int,
    include_running: bool = False,
    refresh_cache: bool = False,
    refresh_stale_cache: bool = False,
) -> list[Point]:
    api = wandb.Api(timeout=90)
    points: list[Point] = []
    runs = list(
        api.runs(
            project,
            filters={
                "$or": [
                    {"tags": {"$all": [EXPERIMENT_TAG]}},
                    {"display_name": {"$regex": "olmoe3-tiny-275m-cx(1|2|4|8).*gpu(2|4)-ep1mb(8|16)"}},
                    {"display_name": {"$regex": "m480-cx(1|2|4|8)"}},
                    {"display_name": {"$regex": "olmoe3-moe-a0-480m-cx(1|2|4|8)"}},
                    {"display_name": {"$regex": "olmoe3-moe-a0-810m-cx(1|2|4|8)"}},
                    {"display_name": {"$regex": "olmoe3-moe-a0-1p2b-cx(1|2|4|8)"}},
                    {"display_name": {"$regex": "olmoe3-moe-a0-1p2b-cx2-b384k"}},
                ]
            },
        )
    )
    for run in runs:
        if not include_running and run.state != "finished":
            continue
        name = run.display_name or run.name
        lowered = name.lower()
        if any(marker in lowered for marker in ("smoke", "sanity", "evaltest")):
            continue

        model = model_from_name(name)
        variant = experiment_variant_name(name) or baseline_variant_name(name)
        if model is None or variant is None:
            continue
        cx = cx_from_name(name)
        lr_info = lr_from_name(name)
        if cx not in {1, 2, 4, 8} or lr_info is None:
            continue

        loss_info = history_loss(
            run,
            project=project,
            cache_dir=cache_dir,
            window_m=window_m,
            refresh_cache=refresh_cache,
            refresh_stale_cache=refresh_stale_cache,
        )
        if loss_info is None or math.isnan(loss_info[0]):
            continue
        lr_tag, lr = lr_info
        loss, tokens_b = loss_info
        points.append(
            Point(
                model=model,
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

    model_order = {"275m": 0, "480m": 1, "810m": 2, "1p2b": 3}
    points.sort(key=lambda p: (model_order.get(p.model, 99), p.cx, p.variant, p.lr, p.name))
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


def plot_cx(points: list[Point], model: str, cx: int, out_path: Path, window_m: int) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for variant in VARIANT_ORDER:
        group = sorted(
            [p for p in points if p.model == model and p.cx == cx and p.variant == variant],
            key=lambda p: p.lr,
        )
        if not group:
            continue
        finished = [p for p in group if p.state == "finished"]
        running = [p for p in group if p.state != "finished"]
        color = None
        label = VARIANT_LABELS[variant]
        if finished:
            (line,) = ax.plot(
                [p.lr for p in finished],
                [p.loss for p in finished],
                marker="o",
                linewidth=1.8,
                label=label,
            )
            color = line.get_color()
        if running:
            ax.scatter(
                [p.lr for p in running],
                [p.loss for p in running],
                marker="x",
                alpha=0.55,
                color=color,
                label=f"{label} running",
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
    ax.set_title(f"{PLOT_TITLE} {model} Cx{cx}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--window-m", type=int, default=250)
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Force a full W&B history re-download for every selected finished run. Use sparingly.",
    )
    parser.add_argument(
        "--refresh-stale-cache",
        action="store_true",
        help="Refresh only missing/stale/short finished-run histories before plotting.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parents[2] / "plots" / OUTPUT_SUBDIR,
    )
    parser.add_argument(
        "--include-running",
        action="store_true",
        help="Include running/incomplete runs. Default is completed runs only to keep U-plot axes stable.",
    )
    args = parser.parse_args()

    points = load_points(
        args.project,
        args.cache_dir,
        args.window_m,
        include_running=args.include_running,
        refresh_cache=args.refresh_cache,
        refresh_stale_cache=args.refresh_stale_cache,
    )
    experiment_keys = {
        (point.model, point.cx)
        for point in points
        if not point.variant.startswith("baseline_")
    }
    points = [point for point in points if (point.model, point.cx) in experiment_keys]
    model_order = {"275m": 0, "480m": 1, "810m": 2, "1p2b": 3}
    for model, cx in sorted(experiment_keys, key=lambda key: (model_order.get(key[0], 99), key[1])):
        plot_cx(points, model, cx, args.output_dir / f"{model}_cx{cx}_uplot.png", args.window_m)
    plot_observed_best_summary(
        points,
        out_path=args.output_dir / "summary_observed_best.png",
        title="Dense-layer schedule observed best",
        variants=SUMMARY_VARIANTS,
        window_m=args.window_m,
    )


if __name__ == "__main__":
    main()
