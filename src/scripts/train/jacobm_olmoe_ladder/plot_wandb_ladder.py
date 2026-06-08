#!/usr/bin/env python
"""Plot MoE ladder U-curves from W&B history."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from analyze_wandb_ladder import load_rows, mean_loss_in_window
from wandb_cache import DEFAULT_CACHE_DIR


CANONICAL_BATCH_BY_CX = {
    1: "256k",
    2: "256k",
    4: "512k",
    8: "768k",
    16: "1M",
}

CANONICAL_FAMILY_BY_CX = {
    1: "gpu2-ep1mb16",
    2: "gpu2-ep1mb16",
    4: "gpu4-ep1mb16",
    8: "gpu4-ep1mb8",
    16: "gpu8-ep1mb16",
}

CANONICAL_FAMILY_BY_MODEL_CX = {
    "275m": CANONICAL_FAMILY_BY_CX,
    "mid_480m": {
        1: "gpu4-ep1mb8",
        2: "gpu4-ep1mb8",
        4: "gpu4-ep1mb8",
        8: "gpu4-ep1mb8",
        16: "gpu4-ep1mb8",
    },
    "810m": {
        1: ("gpu4-ep1mb4", "gpu8-ep1mb4"),
        2: "gpu8-ep1mb4",
        4: "gpu8-ep1mb4",
        8: "gpu8-ep1mb4",
    },
    "1p2b": {
        1: "gpu8-ep1mb2",
        4: "gpu8-ep1mb2",
    },
}

MODEL_SORT_ORDER = {
    "275m": 0,
    "mid_480m": 1,
    "810m": 2,
    "1p2b": 3,
    "unknown": 99,
}


def is_analysis_run(name: str) -> bool:
    lowered = name.lower()
    ignored_markers = ("smoke", "smoketest", "sanity", "pilot")
    return not any(marker in lowered for marker in ignored_markers)


def model_label_from_name(name: str) -> str:
    if "tiny-275m" in name:
        return "275m"
    if "mid-480m" in name or "mid_480m" in name or "480m" in name:
        return "mid_480m"
    if "810m" in name:
        return "810m"
    if "1p2b" in name:
        return "1p2b"
    return "unknown"


def family_label_from_name(name: str) -> str:
    if "gpu8-ep1mb16" in name:
        return "gpu8-ep1mb16"
    if "gpu8-ep1mb4" in name:
        return "gpu8-ep1mb4"
    if "gpu8-ep1mb2" in name:
        return "gpu8-ep1mb2"
    if "gpu4-ep1mb16" in name:
        return "gpu4-ep1mb16"
    if "gpu4-ep1mb8" in name:
        return "gpu4-ep1mb8"
    if "gpu4-ep1mb4" in name:
        return "gpu4-ep1mb4"
    if "gpu2-ep1mb16" in name:
        return "gpu2-ep1mb16"
    if "-n2-" in name or "-n2_" in name:
        return "n2"
    return "original"


def summarize_rows(rows, window_m: int, finished_only: bool, canonical_only: bool):
    points = []
    for row in rows:
        if not row.history:
            continue
        if not is_analysis_run(row.name):
            continue
        if finished_only and row.state != "finished":
            continue
        if canonical_only and row.spec.batch_label != CANONICAL_BATCH_BY_CX.get(row.spec.cx):
            continue
        final = row.history[-1]
        avg, count = mean_loss_in_window(row.history, final.tokens, window_m * 1_000_000)
        if math.isnan(avg):
            continue
        points.append(
            {
                "model": model_label_from_name(row.name),
                "cx": row.spec.cx,
                "batch": row.spec.batch_label,
                "lr": row.spec.lr,
                "lr_tag": row.spec.lr_tag,
                "state": row.state,
                "family": family_label_from_name(row.name),
                "tokens_b": final.tokens / 1e9,
                "loss": avg,
                "n": count,
                "name": row.name,
            }
        )
    return points


def style_for_state(state: str) -> dict:
    if state == "finished":
        return {"alpha": 1.0, "marker": "o", "linewidth": 1.8}
    return {"alpha": 0.45, "marker": "x", "linewidth": 1.0}


def fitted_lr(points) -> tuple[float, float, int] | None:
    finished = sorted([p for p in points if p["state"] == "finished"], key=lambda p: p["lr"])
    if len(finished) < 3:
        return None

    best_idx = min(range(len(finished)), key=lambda i: finished[i]["loss"])
    if best_idx == 0 or best_idx == len(finished) - 1:
        return None

    if len(finished) >= 5:
        start = min(max(best_idx - 2, 0), len(finished) - 5)
        fit_points = finished[start : start + 5]
    else:
        fit_points = finished[best_idx - 1 : best_idx + 2]

    x = np.array([math.log10(p["lr"]) for p in fit_points])
    y = np.array([p["loss"] for p in fit_points])
    a, b, c = np.polyfit(x, y, 2)
    if a <= 0:
        return None

    optimum_log_lr = -b / (2 * a)
    min_log_lr = min(math.log10(p["lr"]) for p in finished)
    max_log_lr = max(math.log10(p["lr"]) for p in finished)
    if not min_log_lr <= optimum_log_lr <= max_log_lr:
        return None

    optimum_lr = 10**optimum_log_lr
    optimum_loss = float(a * optimum_log_lr**2 + b * optimum_log_lr + c)
    return optimum_lr, optimum_loss, len(fit_points)


def annotate_fitted_lr(ax, group, label_prefix: str, color=None) -> None:
    fit = fitted_lr(group)
    if fit is None:
        return
    lr, loss, n_points = fit
    ax.axvline(lr, color=color, linestyle=":", linewidth=1.2, alpha=0.75)
    ax.scatter([lr], [loss], marker="*", s=80, color=color, edgecolor="black", linewidth=0.4, zorder=5)
    ax.annotate(
        f"{label_prefix} fit{n_points}: {lr:.2g}",
        (lr, loss),
        textcoords="offset points",
        xytext=(6, -14),
        ha="left",
        fontsize=8,
        color=color,
        alpha=0.9,
    )


def canonical_family_for(point) -> str:
    by_cx = CANONICAL_FAMILY_BY_MODEL_CX.get(point["model"], CANONICAL_FAMILY_BY_CX)
    family = by_cx.get(point["cx"], point["family"])
    if isinstance(family, tuple):
        return family[0]
    return family


def is_canonical_family(point) -> bool:
    by_cx = CANONICAL_FAMILY_BY_MODEL_CX.get(point["model"], CANONICAL_FAMILY_BY_CX)
    family = by_cx.get(point["cx"], point["family"])
    if isinstance(family, tuple):
        return point["family"] in family
    return point["family"] == family


def plot_cx(points, model: str, cx: int, out_path: Path, window_m: int) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    cx_points = sorted([p for p in points if p["model"] == model and p["cx"] == cx], key=lambda p: p["lr"])
    colors_by_family = {}
    for state, family in sorted({(p["state"], p["family"]) for p in cx_points}):
        group = [p for p in cx_points if p["state"] == state and p["family"] == family]
        style = style_for_state(state)
        (line,) = ax.plot(
            [p["lr"] for p in group],
            [p["loss"] for p in group],
            label=f"{state} ({family})",
            **style,
        )
        colors_by_family.setdefault(family, line.get_color())
    for family in sorted({p["family"] for p in cx_points}):
        group = [p for p in cx_points if p["family"] == family]
        annotate_fitted_lr(ax, group, family, colors_by_family.get(family))
    for point in cx_points:
        ax.annotate(
            point["lr_tag"],
            (point["lr"], point["loss"]),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
            alpha=0.85 if point["state"] == "finished" else 0.55,
        )
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel(f"train CE avg{window_m}M")
    ax.set_title(f"{model} Cx{cx} LR sweep")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_model(points, model: str, out_path: Path, window_m: int) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    model_points = [
        p
        for p in points
        if p["model"] == model and is_canonical_family(p)
    ]
    families_by_cx = {
        cx: sorted({p["family"] for p in model_points if p["cx"] == cx and p["state"] == "finished"})
        for cx in sorted({p["cx"] for p in model_points})
    }
    for cx in sorted(families_by_cx):
        for family in families_by_cx[cx]:
            group = sorted(
                [
                    p
                    for p in model_points
                    if p["cx"] == cx and p["family"] == family and p["state"] == "finished"
                ],
                key=lambda p: p["lr"],
            )
            if not group:
                continue
            label = f"Cx{cx}" if len(families_by_cx[cx]) == 1 else f"Cx{cx} ({family})"
            (line,) = ax.plot(
                [p["lr"] for p in group],
                [p["loss"] for p in group],
                marker="o",
                linewidth=1.8,
                label=label,
            )
            annotate_fitted_lr(ax, group, f"Cx{cx}", line.get_color())
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel(f"train CE avg{window_m}M")
    ax.set_title(f"{model} LR sweeps")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_cx_across_models(points, cx: int, out_path: Path, window_m: int) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    cx_points = [
        p
        for p in points
        if p["cx"] == cx and p["state"] == "finished" and is_canonical_family(p)
    ]
    for model in sorted({p["model"] for p in cx_points}, key=lambda m: MODEL_SORT_ORDER.get(m, 98)):
        model_points = sorted([p for p in cx_points if p["model"] == model], key=lambda p: p["lr"])
        if not model_points:
            continue
        (line,) = ax.plot(
            [p["lr"] for p in model_points],
            [p["loss"] for p in model_points],
            marker="o",
            linewidth=1.8,
            label=model,
        )
        annotate_fitted_lr(ax, model_points, model, line.get_color())
        for point in model_points:
            ax.annotate(
                point["lr_tag"],
                (point["lr"], point["loss"]),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
                alpha=0.85,
            )
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel(f"train CE avg{window_m}M")
    ax.set_title(f"Cx{cx} LR sweeps by model size")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="ai2-llm/jacobm-olmoe-ladder")
    parser.add_argument("--name-regex", default="olmoe3-tiny-275m-cx")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--window-m", type=int, default=250)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "plots")
    parser.add_argument("--finished-only", action="store_true")
    parser.add_argument(
        "--include-noncanonical",
        action="store_true",
        help="Include historical batch-size probes instead of only the ladder batch for each Cx.",
    )
    args = parser.parse_args()

    loader_args = SimpleNamespace(
        project=args.project,
        name_regex=args.name_regex,
        cache_dir=args.cache_dir,
        refresh_cache=args.refresh_cache,
        current_family=False,
        exclude_current_family=False,
        states=["finished"] if args.finished_only else None,
    )
    rows = load_rows(loader_args)
    points = summarize_rows(rows, args.window_m, args.finished_only, not args.include_noncanonical)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for model in sorted({p["model"] for p in points}):
        for cx in sorted({p["cx"] for p in points if p["model"] == model}):
            plot_cx(points, model, cx, args.output_dir / f"{model}_cx{cx}_uplot.png", args.window_m)
    for model in sorted({p["model"] for p in points}):
        plot_model(points, model, args.output_dir / f"{model}_all_cx_uplot.png", args.window_m)
    for cx in sorted({p["cx"] for p in points}):
        plot_cx_across_models(points, cx, args.output_dir / f"cx{cx}_all_models_uplot.png", args.window_m)


if __name__ == "__main__":
    main()
