#!/usr/bin/env python
"""Write a Markdown snapshot of the rows currently used by ladder plots."""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

LADDER_DIR = Path(__file__).parent
EXPERT_DIR = LADDER_DIR / "experiments" / "expert_granularity"
TOTAL_SPARSITY_DIR = LADDER_DIR / "experiments" / "total_sparsity"
SHARED_EXPERT_DIR = LADDER_DIR / "experiments" / "shared_expert"
DENSE_SCHEDULE_DIR = LADDER_DIR / "experiments" / "dense_schedule"
QWEN3_LIKE_DIR = LADDER_DIR / "experiments" / "qwen3_like"
INTEGRATION_DIR = LADDER_DIR / "experiments" / "integration"
for path in (LADDER_DIR, EXPERT_DIR, TOTAL_SPARSITY_DIR, SHARED_EXPERT_DIR, DENSE_SCHEDULE_DIR, QWEN3_LIKE_DIR, INTEGRATION_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_wandb_ladder import load_rows
from plot_wandb_ladder import (
    MODEL_SORT_ORDER,
    fitted_lr,
    is_canonical_family,
    summarize_rows,
)
from wandb_cache import DEFAULT_CACHE_DIR

import plot_expert_granularity as eg
import plot_dense_schedule as ds
import plot_qwen3_like as q3
import plot_integration as integ
import plot_shared_expert as se
import plot_total_sparsity as ts


BASELINE_PROJECT = "ai2-llm/jacobm-olmoe-ladder"
BASELINE_NAME_REGEX = (
    "olmoe3-tiny-275m-cx|olmoe3-moe-a0-810m-cx|olmoe3-moe-a0-1p2b-cx|"
    "olmoe3-810m-cx|m480-cx"
)
MODEL_LABELS = {
    "275m": "275M",
    "480m": "480M",
    "810m": "810M",
    "1p2b": "1.2B",
}
EXPERIMENT_MODULES = {
    "Total Sparsity": ts,
    "Shared Expert": se,
    "Dense Schedule": ds,
    "Qwen3-Like": q3,
    "Integration Candidates": integ,
}
VARIANT_LABELS = {
    "baseline_48e_top4": "baseline 48E/top4",
    "baseline_48e_top4_b256k": "baseline 48E/top4 (b256k)",
    "baseline_48e_top4_b384k": "baseline 48E/top4 (b384k)",
    "baseline_48e_top4_b512k": "baseline 48E/top4 (b512k)",
    "coarse_24e_top2": "coarse 24E/top2",
    "coarse_24e_top2_b512k": "coarse 24E/top2 (b512k)",
    "coarse_24e_top2_b384k": "coarse 24E/top2 (b384k)",
    "fine_96e_top8": "fine 96E/top8",
    "fine_96e_top8_b512k": "fine 96E/top8 (b512k)",
    "fine_96e_top8_b384k": "fine 96E/top8 (b384k)",
    "extreme_192e_top16": "extreme 192E/top16",
    "ultra_384e_top32": "ultra 384E/top32",
    "low_total_24e_top4": "low total 24E/top4",
    "high_total_96e_top4": "high total 96E/top4",
    "huge_total_192e_top4": "huge total 192E/top4",
    "baseline_48e_top4_sparsity_tag": "sparsity baseline 48E/top4",
    "no_shared_matched_active": "no shared, routed 9/8 d",
    "baseline_dense1_shared": "baseline dense1 + shared",
    "baseline_dense1_shared_b384k": "baseline dense1 + shared (b384k)",
    "dense0_shared": "dense0 + shared",
    "dense2_shared": "dense2 + shared",
    "dense4_shared": "dense4 + shared",
    "active_matched": "Qwen-like active matched 4.5d",
    "true_3d_depth_matched": "Qwen-like true 3.0d + depth",
    "integration_wide_256e8k": "integration wide 256E/top8",
    "integration_deep_256e8k": "integration deep 256E/top8",
}


def fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def fmt_lr(value: float | None) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.3g}"


def fmt_tokens(value: float | None) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.3f}"


def md_link(label: str, url: str | None) -> str:
    if not url:
        return label
    return f"[{label}]({url})"


def model_sort_key(model: str) -> tuple[int, str]:
    return MODEL_SORT_ORDER.get(model, 98), model


def row_sort_key(point: dict) -> tuple[int, int, str, float, str]:
    return (
        MODEL_SORT_ORDER.get(point["model"], 98),
        point["cx"],
        point["family"],
        point["lr"],
        point["name"],
    )


def experiment_sort_key(point) -> tuple[int, int, str, float, str]:
    return (
        MODEL_SORT_ORDER.get(point.model, 98),
        point.cx,
        point.variant,
        point.lr,
        point.name,
    )


def markdown_table(headers: list[str], rows: Iterable[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def baseline_points(args: argparse.Namespace) -> tuple[list[dict], dict[str, tuple[str, str]]]:
    loader_args = SimpleNamespace(
        project=args.project,
        name_regex=args.name_regex,
        cache_dir=args.cache_dir,
        refresh_cache=args.refresh_cache,
        refresh_stale_cache=args.refresh_stale_cache,
        current_family=False,
        exclude_current_family=False,
        states=["finished"],
    )
    rows = load_rows(loader_args)
    run_links = {row.name: (row.run_id, row.url) for row in rows}
    points = summarize_rows(rows, args.window_m, finished_only=True, canonical_only=True)
    points = [point for point in points if is_canonical_family(point)]
    points.sort(key=row_sort_key)
    return points, run_links


def expert_points(args: argparse.Namespace) -> list[eg.Point]:
    points = eg.load_points(
        args.project,
        args.cache_dir,
        args.window_m,
        include_running=False,
        refresh_cache=args.refresh_cache,
        refresh_stale_cache=args.refresh_stale_cache,
    )
    models_with_expert_runs = {
        point.model for point in points if not point.variant.startswith("baseline_")
    }
    points = [point for point in points if point.model in models_with_expert_runs]
    points.sort(key=experiment_sort_key)
    return points


def ablation_points(args: argparse.Namespace, module) -> list:
    points = module.load_points(
        args.project,
        args.cache_dir,
        args.window_m,
        include_running=False,
        refresh_cache=args.refresh_cache,
        refresh_stale_cache=args.refresh_stale_cache,
    )
    experiment_keys = {
        (point.model, point.cx)
        for point in points
        if not point.variant.startswith("baseline_")
    }
    points = [point for point in points if (point.model, point.cx) in experiment_keys]
    points.sort(key=experiment_sort_key)
    return points


def write_baseline_section(lines: list[str], points: list[dict], run_links: dict[str, tuple[str, str]]) -> None:
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for point in points:
        groups[(point["model"], point["cx"])].append(point)

    summary_rows: list[list[str]] = []
    for (model, cx), group in sorted(groups.items(), key=lambda item: (model_sort_key(item[0][0]), item[0][1])):
        best = min(group, key=lambda point: point["loss"])
        fit = fitted_lr(group)
        fit_lr = fit[0] if fit is not None else None
        fit_loss = fit[1] if fit is not None else None
        summary_rows.append(
            [
                MODEL_LABELS.get(model, model),
                f"Cx{cx}",
                best["family"],
                best["batch"],
                best["lr_tag"],
                fmt_float(best["loss"]),
                fmt_lr(fit_lr),
                fmt_float(fit_loss),
                str(len(group)),
            ]
        )

    lines.extend(
        [
            "## Baseline Ladder",
            "",
            "These are the completed canonical-family points used by the baseline per-model and per-Cx plots.",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["model", "Cx", "family", "batch", "best observed LR", "best avg250M", "fit LR", "fit avg250M", "points"],
            summary_rows,
        )
    )
    lines.append("")

    detail_rows: list[list[str]] = []
    for point in points:
        run_id, url = run_links.get(point["name"], ("", ""))
        detail_rows.append(
            [
                MODEL_LABELS.get(point["model"], point["model"]),
                f"Cx{point['cx']}",
                point["family"],
                point["batch"],
                point["lr_tag"],
                fmt_tokens(point["tokens_b"]),
                fmt_float(point["loss"]),
                str(point["n"]),
                md_link(run_id or point["name"], url),
            ]
        )
    lines.extend(
        markdown_table(
            ["model", "Cx", "family", "batch", "LR", "tokensB", "avg250M", "n", "W&B"],
            detail_rows,
        )
    )
    lines.append("")


def write_expert_section(lines: list[str], points: list[eg.Point]) -> None:
    if not points:
        return

    groups: dict[tuple[str, int, str], list[eg.Point]] = defaultdict(list)
    for point in points:
        groups[(point.model, point.cx, point.variant)].append(point)

    summary_rows: list[list[str]] = []
    for (model, cx, variant), group in sorted(
        groups.items(), key=lambda item: (model_sort_key(item[0][0]), item[0][1], item[0][2])
    ):
        best = min(group, key=lambda point: point.loss)
        fit = eg.fit_lr(group)
        fit_lr = fit[0] if fit is not None else None
        fit_loss = fit[1] if fit is not None else None
        summary_rows.append(
            [
                MODEL_LABELS.get(model, model),
                f"Cx{cx}",
                VARIANT_LABELS.get(variant, variant),
                best.lr_tag,
                fmt_float(best.loss),
                fmt_lr(fit_lr),
                fmt_float(fit_loss),
                str(len(group)),
            ]
        )

    lines.extend(
        [
            "## Expert Granularity",
            "",
            "These are the completed points used by the expert-granularity plots. Baseline rows are included only for model sizes that also have expert-granularity variants.",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["model", "Cx", "variant", "best observed LR", "best avg250M", "fit LR", "fit avg250M", "points"],
            summary_rows,
        )
    )
    lines.append("")

    detail_rows: list[list[str]] = []
    for point in points:
        url = f"https://wandb.ai/{BASELINE_PROJECT}/runs/{point.run_id}"
        detail_rows.append(
            [
                MODEL_LABELS.get(point.model, point.model),
                f"Cx{point.cx}",
                VARIANT_LABELS.get(point.variant, point.variant),
                point.lr_tag,
                fmt_tokens(point.tokens_b),
                fmt_float(point.loss),
                md_link(point.run_id, url),
            ]
        )
    lines.extend(
        markdown_table(
            ["model", "Cx", "variant", "LR", "tokensB", "avg250M", "W&B"],
            detail_rows,
        )
    )
    lines.append("")


def write_experiment_section(lines: list[str], title: str, points: list, fit_fn) -> None:
    if not points:
        return

    groups: dict[tuple[str, int, str], list] = defaultdict(list)
    for point in points:
        groups[(point.model, point.cx, point.variant)].append(point)

    summary_rows: list[list[str]] = []
    for (model, cx, variant), group in sorted(
        groups.items(), key=lambda item: (model_sort_key(item[0][0]), item[0][1], item[0][2])
    ):
        best = min(group, key=lambda point: point.loss)
        fit = fit_fn(group)
        fit_lr = fit[0] if fit is not None else None
        fit_loss = fit[1] if fit is not None else None
        summary_rows.append(
            [
                MODEL_LABELS.get(model, model),
                f"Cx{cx}",
                VARIANT_LABELS.get(variant, variant),
                best.lr_tag,
                fmt_float(best.loss),
                fmt_lr(fit_lr),
                fmt_float(fit_loss),
                str(len(group)),
            ]
        )

    lines.extend(
        [
            f"## {title}",
            "",
            f"These are the completed points used by the {title.lower()} plots. Baseline rows are included only for model/Cx settings that also have completed experiment variants.",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["model", "Cx", "variant", "best observed LR", "best avg250M", "fit LR", "fit avg250M", "points"],
            summary_rows,
        )
    )
    lines.append("")

    detail_rows: list[list[str]] = []
    for point in points:
        url = f"https://wandb.ai/{BASELINE_PROJECT}/runs/{point.run_id}"
        detail_rows.append(
            [
                MODEL_LABELS.get(point.model, point.model),
                f"Cx{point.cx}",
                VARIANT_LABELS.get(point.variant, point.variant),
                point.lr_tag,
                fmt_tokens(point.tokens_b),
                fmt_float(point.loss),
                md_link(point.run_id, url),
            ]
        )
    lines.extend(
        markdown_table(
            ["model", "Cx", "variant", "LR", "tokensB", "avg250M", "W&B"],
            detail_rows,
        )
    )
    lines.append("")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=BASELINE_PROJECT)
    parser.add_argument("--name-regex", default=BASELINE_NAME_REGEX)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--window-m", type=int, default=250)
    parser.add_argument("--output", type=Path, default=LADDER_DIR / "PLOTTED_RESULTS.md")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--refresh-stale-cache", action="store_true")
    args = parser.parse_args()

    baseline, run_links = baseline_points(args)
    expert = expert_points(args)
    ablations = {title: ablation_points(args, module) for title, module in EXPERIMENT_MODULES.items()}
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Plotted Results",
        "",
        f"Generated: {generated_at}",
        "",
        "This file is generated by `write_plotted_results_md.py` from the same cached W&B histories used by the plotters.",
        "It mirrors the default plotting policy: completed runs only, training CE averaged over the final 250M tokens, and running jobs excluded so axes and tables do not shift around live partial runs.",
        "",
        "Regenerate with:",
        "",
        "```bash",
        "uv run --with wandb python src/scripts/train/jacobm_olmoe_ladder/write_plotted_results_md.py",
        "```",
        "",
    ]
    write_baseline_section(lines, baseline, run_links)
    write_expert_section(lines, expert)
    for title, points in ablations.items():
        module = EXPERIMENT_MODULES[title]
        write_experiment_section(lines, title, points, module.fit_lr)
    args.output.write_text("\n".join(lines).rstrip() + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
