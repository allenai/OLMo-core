#!/usr/bin/env python
"""Plot shared-expert U-curves from W&B history."""

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

from wandb_cache import DEFAULT_CACHE_DIR, scan_history_cached
from experiment_batch_filters import has_canonical_batch_for_cx
from experiment_summary_plots import SummaryVariant, plot_observed_best_summary


PROJECT = "ai2-llm/jacobm-olmoe-ladder"
LOSS_KEY = "train/CE loss"
TOKENS_KEY = "throughput/total tokens"
FIELDS = ["_step", TOKENS_KEY, LOSS_KEY]
LR_RE = re.compile(r"lr([0-9]+(?:\.[0-9]+)?e-[0-9]+)")

EXPERIMENT_TAG = "exp_shared_expert"
OUTPUT_SUBDIR = "shared_expert"
PLOT_TITLE = "Shared expert"
VARIANT_LABELS = {'baseline_48e_top4': 'baseline shared d/2', 'baseline_48e_top4_b256k': 'baseline shared d/2 (b256k)', 'baseline_48e_top4_b384k': 'baseline shared d/2 (b384k)', 'baseline_48e_top4_b512k': 'baseline shared d/2 (b512k)', 'no_shared_matched_active': 'no shared, routed 9/8 d'}
PLOTTABLE_STATES = {"finished", "running"}

VARIANT_ORDER = ['baseline_48e_top4', 'baseline_48e_top4_b256k', 'baseline_48e_top4_b384k', 'baseline_48e_top4_b512k', 'no_shared_matched_active']
SUMMARY_VARIANTS = [
    SummaryVariant(
        "baseline",
        ("baseline_48e_top4", "baseline_48e_top4_b384k"),
        "baseline shared d/2",
        color="black",
        linestyle="--",
    ),
    SummaryVariant("no_shared_matched_active", ("no_shared_matched_active",), "no shared, routed 9/8 d"),
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
        tail_window_tokens=window_m * 1_000_000,
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
    if (
        "mid_480m" in name
        or "mid-480m" in name
        or "m480-cx" in name
        or "eg-480m" in name
        or "se-480m" in name
        or "ds-480m" in name
    ):
        return "480m"
    if "810m" in name:
        return "810m"
    if "1p2b" in name:
        return "1p2b"
    return None


def baseline_variant_name(name: str) -> str | None:
    if "olmoe3-tiny-275m-cx" in name:
        if "b384k" in name and "cx2" in name:
            return "baseline_48e_top4_b384k"
        if "gpu2-ep1mb16" in name and "cx1" in name:
            return "baseline_48e_top4"
        if "gpu2-ep1mb16" in name and "cx2" in name:
            return "baseline_48e_top4_b256k"
        if "gpu4-ep1mb16" in name and "cx4" in name:
            return "baseline_48e_top4"
        if "gpu4-ep1mb8" in name and "cx8" in name:
            return "baseline_48e_top4"
        return None

    if "m480-cx" in name:
        if "cx2" in name and "b384k" in name:
            return "baseline_48e_top4_b384k"
        if "cx2" in name and "b512k" in name:
            return "baseline_48e_top4_b512k"
        return "baseline_48e_top4"

    if "olmoe3-moe-a0-810m-cx" in name:
        if "cx2" in name and "b384k" in name:
            return "baseline_48e_top4_b384k"
        return "baseline_48e_top4"

    if "olmoe3-moe-a0-1p2b-cx" in name or "olmoe3-moe-a0-1p2b-cx2-b384k" in name:
        if "cx2" in name and "b384k" in name:
            return "baseline_48e_top4_b384k"
        return "baseline_48e_top4"

    return None


def experiment_variant_name(name: str) -> str | None:
    if "se0m9" in name or "no_shared_matched_active" in name:
        return "no_shared_matched_active"
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
        allowed_states = PLOTTABLE_STATES if include_running else {"finished"}
        if run.state not in allowed_states:
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
        if not has_canonical_batch_for_cx(name, cx):
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
        running = [p for p in group if p.state == "running"]
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
        title="Shared expert observed best",
        variants=SUMMARY_VARIANTS,
        window_m=args.window_m,
    )


if __name__ == "__main__":
    main()
