#!/usr/bin/env python
"""Plot observed-best comparisons across all ladder interventions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

LADDER_DIR = Path(__file__).parents[1]
EXPERIMENTS_DIR = Path(__file__).parent
EXPERT_DIR = EXPERIMENTS_DIR / "expert_granularity"
TOTAL_SPARSITY_DIR = EXPERIMENTS_DIR / "total_sparsity"
SHARED_EXPERT_DIR = EXPERIMENTS_DIR / "shared_expert"
DENSE_SCHEDULE_DIR = EXPERIMENTS_DIR / "dense_schedule"
QWEN3_LIKE_DIR = EXPERIMENTS_DIR / "qwen3_like"
INTEGRATION_DIR = EXPERIMENTS_DIR / "integration"
for path in (
    LADDER_DIR,
    EXPERT_DIR,
    TOTAL_SPARSITY_DIR,
    SHARED_EXPERT_DIR,
    DENSE_SCHEDULE_DIR,
    QWEN3_LIKE_DIR,
    INTEGRATION_DIR,
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment_summary_plots import SummaryVariant, plot_observed_best_summary
from wandb_cache import DEFAULT_CACHE_DIR

import plot_dense_schedule as dense_schedule
import plot_expert_granularity as expert_granularity
import plot_integration as integration
import plot_qwen3_like as qwen3_like
import plot_shared_expert as shared_expert
import plot_total_sparsity as total_sparsity

PROJECT = "ai2-llm/jacobm-olmoe-ladder"
OUTPUT_DIR = LADDER_DIR / "plots" / "interventions"

SUMMARY_VARIANTS = [
    SummaryVariant(
        "baseline",
        (
            "baseline_48e_top4",
            "baseline_48e_top4_b256k",
            "baseline_48e_top4_b384k",
            "baseline_48e_top4_b512k",
            "baseline_dense1_shared",
            "baseline_dense1_shared_b384k",
            "baseline_48e_top4_sparsity_tag",
        ),
        "baseline 48E/top4",
        color="black",
        linestyle="--",
    ),
    SummaryVariant("eg_coarse_24e_top2", ("coarse_24e_top2", "coarse_24e_top2_b384k", "coarse_24e_top2_b512k"), "EG coarse 24E/top2"),
    SummaryVariant("eg_fine_96e_top8", ("fine_96e_top8", "fine_96e_top8_b384k", "fine_96e_top8_b512k"), "EG fine 96E/top8"),
    SummaryVariant("sp_high_total_96e_top4", ("high_total_96e_top4",), "sparsity 96E/top4"),
    SummaryVariant("sp_huge_total_192e_top4", ("huge_total_192e_top4",), "sparsity 192E/top4"),
    SummaryVariant("shared_no_shared", ("no_shared_matched_active",), "no shared, active matched"),
    SummaryVariant("dense0_shared", ("dense0_shared",), "dense0 + shared"),
    SummaryVariant("dense2_shared", ("dense2_shared",), "dense2 + shared"),
    SummaryVariant("dense4_shared", ("dense4_shared",), "dense4 + shared"),
    SummaryVariant("qwen_active_matched", ("active_matched",), "Qwen-like 4.5d"),
    SummaryVariant("qwen_true_3d", ("true_3d_depth_matched",), "Qwen-like 3.0d + depth"),
    SummaryVariant("integration_wide", ("integration_wide_256e8k",), "integration wide"),
    SummaryVariant("integration_deep", ("integration_deep_256e8k",), "integration deep"),
]

MODULES = (
    expert_granularity,
    total_sparsity,
    shared_expert,
    dense_schedule,
    qwen3_like,
    integration,
)


def load_all_points(args: argparse.Namespace) -> list[object]:
    points: list[object] = []
    seen: set[tuple[str, str, int, str, str]] = set()
    for module in MODULES:
        module_points = module.load_points(
            args.project,
            args.cache_dir,
            args.window_m,
            include_running=False,
            refresh_cache=args.refresh_cache,
            refresh_stale_cache=args.refresh_stale_cache,
        )
        for point in module_points:
            key = (point.name, point.variant, point.cx, point.lr_tag, point.run_id)
            if key in seen:
                continue
            seen.add(key)
            points.append(point)
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--window-m", type=int, default=250)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--refresh-stale-cache", action="store_true")
    args = parser.parse_args()

    points = load_all_points(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_observed_best_summary(
        points,
        out_path=args.output_dir / "summary_observed_best.png",
        title="All interventions observed best",
        variants=SUMMARY_VARIANTS,
        window_m=args.window_m,
    )


if __name__ == "__main__":
    main()
