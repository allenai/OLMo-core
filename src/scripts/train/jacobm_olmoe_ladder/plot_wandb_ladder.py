#!/usr/bin/env python
"""Plot MoE ladder U-curves from W&B history."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt

from analyze_wandb_ladder import load_rows, mean_loss_in_window
from wandb_cache import DEFAULT_CACHE_DIR


CANONICAL_BATCH_BY_CX = {
    1: "256k",
    2: "256k",
    4: "512k",
    8: "768k",
    16: "1M",
}


def is_analysis_run(name: str) -> bool:
    lowered = name.lower()
    return "smoke" not in lowered and "smoketest" not in lowered


def model_label_from_name(name: str) -> str:
    if "tiny-275m" in name:
        return "275m"
    if "810m" in name:
        return "810m"
    if "1p2b" in name:
        return "1p2b"
    return "unknown"


def family_label_from_name(name: str) -> str:
    if "gpu8-ep1mb16" in name:
        return "gpu8-ep1mb16"
    if "gpu4-ep1mb16" in name:
        return "gpu4-ep1mb16"
    if "gpu4-ep1mb8" in name:
        return "gpu4-ep1mb8"
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


def plot_cx(points, cx: int, out_path: Path, window_m: int) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    cx_points = sorted([p for p in points if p["cx"] == cx], key=lambda p: p["lr"])
    for state, family in sorted({(p["state"], p["family"]) for p in cx_points}):
        group = [p for p in cx_points if p["state"] == state and p["family"] == family]
        style = style_for_state(state)
        ax.plot(
            [p["lr"] for p in group],
            [p["loss"] for p in group],
            label=f"{state} ({family})",
            **style,
        )
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
    ax.set_title(f"275M Cx{cx} LR sweep")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_model(points, model: str, out_path: Path, window_m: int) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    model_points = [p for p in points if p["model"] == model]
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
            ax.plot(
                [p["lr"] for p in group],
                [p["loss"] for p in group],
                marker="o",
                linewidth=1.8,
                label=label,
            )
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel(f"train CE avg{window_m}M")
    ax.set_title(f"{model} LR sweeps")
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
        states=None,
    )
    rows = load_rows(loader_args)
    points = summarize_rows(rows, args.window_m, args.finished_only, not args.include_noncanonical)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for cx in sorted({p["cx"] for p in points}):
        plot_cx(points, cx, args.output_dir / f"275m_cx{cx}_uplot.png", args.window_m)
    for model in sorted({p["model"] for p in points}):
        plot_model(points, model, args.output_dir / f"{model}_all_cx_uplot.png", args.window_m)


if __name__ == "__main__":
    main()
