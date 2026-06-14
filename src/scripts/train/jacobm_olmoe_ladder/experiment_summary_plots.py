#!/usr/bin/env python
"""Observed-best summary plots for ladder ablation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt


MODEL_ORDER = {"275m": 0, "480m": 1, "810m": 2, "1p2b": 3}
CX_ORDER = [1, 2, 4, 8]
BASELINE_KEY = "baseline"


@dataclass(frozen=True)
class SummaryVariant:
    key: str
    aliases: tuple[str, ...]
    label: str
    color: str | None = None
    linestyle: str = "-"


def _variant_key(variant: str, variants: Sequence[SummaryVariant]) -> str | None:
    for summary_variant in variants:
        if variant in summary_variant.aliases:
            return summary_variant.key
    return None


def _best_observed(points: Iterable[object], variants: Sequence[SummaryVariant]) -> dict[tuple[str, int, str], object]:
    best: dict[tuple[str, int, str], object] = {}
    for point in points:
        if point.state != "finished":
            continue
        variant = _variant_key(point.variant, variants)
        if variant is None:
            continue
        key = (point.model, point.cx, variant)
        if key not in best or point.loss < best[key].loss:
            best[key] = point
    return best


def _write_placeholder(out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 3.0))
    ax.axis("off")
    ax.text(
        0.5,
        0.55,
        "No completed non-baseline variant points yet",
        ha="center",
        va="center",
        fontsize=12,
    )
    ax.set_title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_observed_best_summary(
    points: Sequence[object],
    *,
    out_path: Path,
    title: str,
    variants: Sequence[SummaryVariant],
    window_m: int,
) -> bool:
    """Plot best observed finished loss by model/Cx/variant.

    Each point is the completed run with the lowest avg-window train CE for a
    single (model size, data multiple, canonical variant). Running runs and
    diagnostic/non-canonical aliases are ignored by the caller-provided variant
    list.
    """

    best = _best_observed(points, variants)
    experiment_keys = {
        (model, cx)
        for (model, cx, variant) in best
        if variant != BASELINE_KEY
    }
    if not experiment_keys:
        _write_placeholder(out_path, title)
        return False

    models = sorted({model for model, _ in experiment_keys}, key=lambda model: MODEL_ORDER.get(model, 99))
    fig_width = max(6.0, 3.8 * len(models))
    fig, axes = plt.subplots(1, len(models), figsize=(fig_width, 4.4), squeeze=False, sharey=False)

    for ax, model in zip(axes[0], models):
        model_keys = {cx for model_name, cx in experiment_keys if model_name == model}
        handles = []
        labels = []
        for summary_variant in variants:
            xs: list[int] = []
            ys: list[float] = []
            point_labels: list[str] = []
            for cx in CX_ORDER:
                if cx not in model_keys:
                    continue
                point = best.get((model, cx, summary_variant.key))
                if point is None:
                    continue
                xs.append(cx)
                ys.append(point.loss)
                point_labels.append(point.lr_tag)
            if not xs:
                continue
            (line,) = ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2.0 if summary_variant.key == BASELINE_KEY else 1.8,
                markersize=5.5,
                color=summary_variant.color,
                linestyle=summary_variant.linestyle,
                label=summary_variant.label,
            )
            handles.append(line)
            labels.append(summary_variant.label)
            for x, y, lr_tag in zip(xs, ys, point_labels):
                ax.annotate(
                    lr_tag,
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    fontsize=7,
                    color=line.get_color(),
                    alpha=0.9,
                )
        ax.set_xscale("log", base=2)
        ax.set_xticks(CX_ORDER)
        ax.set_xticklabels([f"Cx{cx}" for cx in CX_ORDER])
        ax.set_xlabel("data multiple")
        ax.set_title(model)
        ax.grid(True, which="both", alpha=0.25)
        if not any(cx in model_keys for cx in CX_ORDER):
            ax.axis("off")
    axes[0][0].set_ylabel(f"best observed train CE avg{window_m}M")
    fig.suptitle(title)

    unique: dict[str, object] = {}
    for ax in axes[0]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            unique.setdefault(label, handle)
    if unique:
        fig.legend(
            unique.values(),
            unique.keys(),
            loc="lower center",
            ncol=min(len(unique), 4),
            bbox_to_anchor=(0.5, -0.02),
            frameon=False,
        )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True
