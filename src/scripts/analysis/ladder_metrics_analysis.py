#!/usr/bin/env python3
"""
Ladder Metrics Analysis Script

Aggregates evaluation metrics from model ladder runs into OLMo 3-style
evaluation clusters (Math, Code, MC_STEM, MC_NonSTEM, GenQA, BasicSkills).

Separates accuracy-based metrics from BPB (bits-per-byte) metrics for proper
interpretation.

Usage:
    python ladder_metrics_analysis.py --ladder-dir ~/Downloads/<ladder-name>
    python ladder_metrics_analysis.py --compare ladder1:~/Downloads/hybrid-gdn ladder2:~/Downloads/pure-gdn
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Optional plotting imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


# Task clusters for ACCURACY metrics (higher is better)
ACCURACY_CLUSTERS = {
    "MC_STEM": [
        "arc_challenge",
        "arc_easy",
        "mmlu_stem",
    ],
    "MC_NonSTEM": [
        "mmlu_humanities",
        "mmlu_social_sciences",
        "mmlu_other",
        "csqa",
        "piqa",
        "socialiqa",
    ],
    "GenQA": [
        "hellaswag",
        "winogrande",
    ],
    "BasicSkills": [
        "basic_skills_arithmetic",
        "basic_skills_coding",
        "basic_skills_common_knowledge",
        "basic_skills_logical_reasoning",
        "basic_skills_pattern",
        "basic_skills_string_operations",
    ],
}

# Task clusters for BPB metrics (lower is better)
BPB_CLUSTERS = {
    "Math": [
        "gsm8k",
        "minerva_math_algebra",
        "minerva_math_counting",
        "minerva_math_geometry",
        "minerva_math_intermediate",
        "minerva_math_number",
        "minerva_math_prealgebra",
        "minerva_math_precalculus",
        "minerva_math_500",
    ],
    "Code": [
        "codex_humaneval",
        "codex_mbpp",
        "mt_mbpp_rust",
        "mt_mbpp_java",
        "mt_mbpp_cpp",
    ],
}

# LM eval tasks (perplexity/loss based)
LM_TASKS = [
    "c4_en",
    "dolma_books",
    "dolma_common-crawl",
    "dolma_pes2o",
    "dolma_reddit",
    "dolma_stack",
    "dolma_wiki",
    "ice",
    "m2d2_s2orc",
    "pile",
    "wikitext_103",
]


def find_accuracy_columns(df: pd.DataFrame, task_patterns: List[str]) -> Dict[str, str]:
    """
    Find accuracy/length-normalized accuracy columns for tasks.
    Returns dict mapping task name to column name.
    """
    task_to_col = {}

    for col in df.columns:
        col_lower = col.lower()

        # Skip non-accuracy metrics
        if "bpb" in col_lower or "ce loss" in col_lower or "soft loss" in col_lower or "log soft" in col_lower:
            continue

        # Only want accuracy or length-normalized accuracy
        if "accuracy" not in col_lower and "acc" not in col_lower:
            continue

        # Prefer "v2" versions and "rc" (reading comprehension) over "mc" (multiple choice fast)
        for pattern in task_patterns:
            if pattern.lower() in col_lower:
                # Extract a task key
                task_key = pattern.lower()

                # Prefer rc_5shot over mc_5shot_fast, and v2 versions
                is_rc = "_rc_" in col_lower
                is_v2 = "v2" in col_lower
                is_len_norm = "length-normalized" in col_lower or "len_norm" in col_lower

                # Score this column (higher = better preference)
                score = 0
                if is_rc:
                    score += 10
                if is_v2:
                    score += 5
                if is_len_norm:
                    score += 2

                if task_key not in task_to_col or score > task_to_col[task_key][1]:
                    task_to_col[task_key] = (col, score)
                break

    # Return just the column names
    return {k: v[0] for k, v in task_to_col.items()}


def find_bpb_columns(df: pd.DataFrame, task_patterns: List[str]) -> Dict[str, str]:
    """
    Find BPB columns for tasks.
    Returns dict mapping task name to column name.
    """
    task_to_col = {}

    for col in df.columns:
        col_lower = col.lower()

        # Only want BPB metrics
        if "bpb" not in col_lower:
            continue

        for pattern in task_patterns:
            if pattern.lower() in col_lower:
                task_key = pattern.lower()

                # Prefer v2 versions
                is_v2 = "v2" in col_lower
                score = 5 if is_v2 else 0

                if task_key not in task_to_col or score > task_to_col[task_key][1]:
                    task_to_col[task_key] = (col, score)
                break

    return {k: v[0] for k, v in task_to_col.items()}


def find_lm_columns(df: pd.DataFrame) -> Dict[str, Tuple[str, str]]:
    """
    Find LM eval columns (CE loss and PPL).
    Returns dict mapping task name to (ce_loss_col, ppl_col).
    """
    task_to_cols = {}

    for col in df.columns:
        col_lower = col.lower()

        if "eval/lm/" not in col_lower:
            continue

        for task in LM_TASKS:
            if task.lower() in col_lower:
                if task not in task_to_cols:
                    task_to_cols[task] = [None, None]

                if "ce loss" in col_lower:
                    task_to_cols[task][0] = col
                elif "ppl" in col_lower:
                    task_to_cols[task][1] = col
                break

    return {k: tuple(v) for k, v in task_to_cols.items() if v[0] or v[1]}


def aggregate_accuracy_cluster(
    df: pd.DataFrame,
    task_patterns: List[str],
) -> Tuple[Optional[float], Dict[str, float]]:
    """
    Aggregate accuracy metrics for a cluster.
    Returns (average, {task: value}).
    """
    task_to_col = find_accuracy_columns(df, task_patterns)
    if not task_to_col:
        return None, {}

    final_row = df[df["step"] == df["step"].max()].iloc[0]

    values = {}
    for task, col in task_to_col.items():
        if col in final_row and pd.notna(final_row[col]):
            values[task] = final_row[col]

    if values:
        return sum(values.values()) / len(values), values
    return None, {}


def aggregate_bpb_cluster(
    df: pd.DataFrame,
    task_patterns: List[str],
) -> Tuple[Optional[float], Dict[str, float]]:
    """
    Aggregate BPB metrics for a cluster.
    Returns (average, {task: value}).
    """
    task_to_col = find_bpb_columns(df, task_patterns)
    if not task_to_col:
        return None, {}

    final_row = df[df["step"] == df["step"].max()].iloc[0]

    values = {}
    for task, col in task_to_col.items():
        if col in final_row and pd.notna(final_row[col]):
            values[task] = final_row[col]

    if values:
        return sum(values.values()) / len(values), values
    return None, {}


def analyze_ladder(ladder_dir: Path) -> Dict[str, Dict]:
    """
    Analyze all metrics files in a ladder directory.

    Returns:
        Dict mapping size -> {
            "accuracy_clusters": {...},
            "bpb_clusters": {...},
            "lm_metrics": {...},
            "raw": {...},
            "step": int,
            "tokens": int
        }
    """
    results = {}

    pkl_files = list(ladder_dir.glob("metrics_*.pkl"))

    for pkl_path in sorted(pkl_files):
        size = pkl_path.stem.replace("metrics_", "")

        try:
            df = pd.read_pickle(pkl_path)
        except Exception as e:
            print(f"Warning: Could not load {pkl_path}: {e}")
            continue

        if df.empty:
            continue

        final_row = df[df["step"] == df["step"].max()].iloc[0]
        final_step = int(final_row.get("step", 0))
        final_tokens = int(final_row.get("tokens", 0))

        # Aggregate accuracy clusters
        accuracy_clusters = {}
        accuracy_details = {}
        for cluster_name, task_patterns in ACCURACY_CLUSTERS.items():
            avg, details = aggregate_accuracy_cluster(df, task_patterns)
            if avg is not None:
                accuracy_clusters[cluster_name] = avg
                accuracy_details[cluster_name] = details

        # Aggregate BPB clusters
        bpb_clusters = {}
        bpb_details = {}
        for cluster_name, task_patterns in BPB_CLUSTERS.items():
            avg, details = aggregate_bpb_cluster(df, task_patterns)
            if avg is not None:
                bpb_clusters[cluster_name] = avg
                bpb_details[cluster_name] = details

        # Get LM metrics
        lm_cols = find_lm_columns(df)
        lm_metrics = {}
        for task, (ce_col, ppl_col) in lm_cols.items():
            lm_metrics[task] = {}
            if ce_col and pd.notna(final_row.get(ce_col)):
                lm_metrics[task]["CE loss"] = final_row[ce_col]
            if ppl_col and pd.notna(final_row.get(ppl_col)):
                lm_metrics[task]["PPL"] = final_row[ppl_col]

        # Store raw metrics
        metric_cols = [
            c for c in df.columns if c not in ["name", "step", "tokens", "size", "num_params"]
        ]
        raw_metrics = {col: final_row[col] for col in metric_cols if pd.notna(final_row.get(col))}

        results[size] = {
            "accuracy_clusters": accuracy_clusters,
            "accuracy_details": accuracy_details,
            "bpb_clusters": bpb_clusters,
            "bpb_details": bpb_details,
            "lm_metrics": lm_metrics,
            "raw": raw_metrics,
            "step": final_step,
            "tokens": final_tokens,
            "num_params": final_row.get("num_params", 0),
        }

    return results


def create_accuracy_table(
    ladder_results: Dict[str, Dict[str, Dict]],
    sizes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Create accuracy comparison table."""
    rows = []

    for ladder_name, size_results in ladder_results.items():
        for size, data in size_results.items():
            if sizes and size not in sizes:
                continue

            row = {
                "Ladder": ladder_name,
                "Size": size,
            }
            row.update(data.get("accuracy_clusters", {}))
            rows.append(row)

    df = pd.DataFrame(rows)
    return _sort_by_size(df) if not df.empty else df


def create_bpb_table(
    ladder_results: Dict[str, Dict[str, Dict]],
    sizes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Create BPB comparison table."""
    rows = []

    for ladder_name, size_results in ladder_results.items():
        for size, data in size_results.items():
            if sizes and size not in sizes:
                continue

            row = {
                "Ladder": ladder_name,
                "Size": size,
            }
            row.update(data.get("bpb_clusters", {}))
            rows.append(row)

    df = pd.DataFrame(rows)
    return _sort_by_size(df) if not df.empty else df


def create_lm_table(
    ladder_results: Dict[str, Dict[str, Dict]],
    sizes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Create LM eval comparison table."""
    rows = []

    for ladder_name, size_results in ladder_results.items():
        for size, data in size_results.items():
            if sizes and size not in sizes:
                continue

            row = {
                "Ladder": ladder_name,
                "Size": size,
            }
            # Add average PPL across domains
            lm = data.get("lm_metrics", {})
            ppls = [v.get("PPL") for v in lm.values() if v.get("PPL") is not None]
            if ppls:
                row["Avg PPL"] = sum(ppls) / len(ppls)

            # Add individual domain PPLs
            for task in ["c4_en", "pile", "wikitext_103"]:
                if task in lm and "PPL" in lm[task]:
                    row[f"{task} PPL"] = lm[task]["PPL"]

            rows.append(row)

    df = pd.DataFrame(rows)
    return _sort_by_size(df) if not df.empty else df


SIZE_ORDER = {
    "60M": 0, "100M": 1, "190M": 2, "370M": 3, "600M": 4,
    "760M": 5, "1B": 6, "3B": 7, "7B": 8, "13B": 9, "32B": 10,
}


def _sort_by_size(df: pd.DataFrame, group_by_size: bool = True) -> pd.DataFrame:
    """
    Sort dataframe by model size.

    Args:
        df: DataFrame with 'Ladder' and 'Size' columns
        group_by_size: If True, group same-sized models together (size first, then ladder).
                      If False, group by ladder first, then size.
    """
    df["_size_order"] = df["Size"].map(lambda x: SIZE_ORDER.get(x, 99))
    if group_by_size:
        # Group same-sized models together: sort by size first, then ladder
        df = df.sort_values(["_size_order", "Ladder"]).drop("_size_order", axis=1)
    else:
        # Group by ladder first, then size
        df = df.sort_values(["Ladder", "_size_order"]).drop("_size_order", axis=1)
    return df


def print_accuracy_table(df: pd.DataFrame, group_by_size: bool = True):
    """Print accuracy table (higher is better)."""
    if df.empty:
        print("\nNo accuracy metrics found!")
        return

    # Re-sort based on grouping preference
    df = _sort_by_size(df.copy(), group_by_size=group_by_size)

    cluster_cols = [c for c in ACCURACY_CLUSTERS.keys() if c in df.columns]

    print("\n" + "=" * 100)
    print("ACCURACY METRICS (higher is better)")
    print("=" * 100)

    header = f"{'Size':<8} {'Ladder':<25}"
    for col in cluster_cols:
        header += f" {col:>12}"
    print(header)
    print("-" * 100)

    prev_size = None
    for _, row in df.iterrows():
        # Add separator between different sizes when grouping by size
        if group_by_size and prev_size is not None and row['Size'] != prev_size:
            print()
        prev_size = row['Size']

        line = f"{row['Size']:<8} {row['Ladder']:<25}"
        for col in cluster_cols:
            val = row.get(col)
            if pd.notna(val):
                line += f" {val*100:>11.1f}%"
            else:
                line += f" {'-':>12}"
        print(line)

    print("=" * 100)


def print_bpb_table(df: pd.DataFrame, group_by_size: bool = True):
    """Print BPB table (lower is better)."""
    if df.empty:
        print("\nNo BPB metrics found!")
        return

    # Re-sort based on grouping preference
    df = _sort_by_size(df.copy(), group_by_size=group_by_size)

    cluster_cols = [c for c in BPB_CLUSTERS.keys() if c in df.columns]

    print("\n" + "=" * 100)
    print("BPB METRICS (lower is better)")
    print("=" * 100)

    header = f"{'Size':<8} {'Ladder':<25}"
    for col in cluster_cols:
        header += f" {col:>12}"
    print(header)
    print("-" * 100)

    prev_size = None
    for _, row in df.iterrows():
        # Add separator between different sizes when grouping by size
        if group_by_size and prev_size is not None and row['Size'] != prev_size:
            print()
        prev_size = row['Size']

        line = f"{row['Size']:<8} {row['Ladder']:<25}"
        for col in cluster_cols:
            val = row.get(col)
            if pd.notna(val):
                line += f" {val:>12.4f}"
            else:
                line += f" {'-':>12}"
        print(line)

    print("=" * 100)


def print_lm_table(df: pd.DataFrame, group_by_size: bool = True):
    """Print LM eval table (lower is better)."""
    if df.empty:
        print("\nNo LM metrics found!")
        return

    # Re-sort based on grouping preference
    df = _sort_by_size(df.copy(), group_by_size=group_by_size)

    ppl_cols = [c for c in df.columns if "PPL" in c]
    if not ppl_cols:
        return

    print("\n" + "=" * 100)
    print("LANGUAGE MODELING (Perplexity, lower is better)")
    print("=" * 100)

    header = f"{'Size':<8} {'Ladder':<25}"
    for col in ppl_cols:
        header += f" {col:>14}"
    print(header)
    print("-" * 100)

    prev_size = None
    for _, row in df.iterrows():
        # Add separator between different sizes when grouping by size
        if group_by_size and prev_size is not None and row['Size'] != prev_size:
            print()
        prev_size = row['Size']

        line = f"{row['Size']:<8} {row['Ladder']:<25}"
        for col in ppl_cols:
            val = row.get(col)
            if pd.notna(val):
                line += f" {val:>14.2f}"
            else:
                line += f" {'-':>14}"
        print(line)

    print("=" * 100)


def get_ladder_colors(ladder_names: List[str]) -> Dict[str, str]:
    """Get distinct colors for each ladder."""
    # Use a colorblind-friendly palette
    colors = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # olive
        "#17becf",  # cyan
    ]
    return {name: colors[i % len(colors)] for i, name in enumerate(ladder_names)}


def get_size_numeric(size: str) -> float:
    """Convert size string to numeric value for x-axis positioning."""
    size_to_num = {
        "60M": 60, "100M": 100, "190M": 190, "370M": 370, "600M": 600,
        "760M": 760, "1B": 1000, "3B": 3000, "7B": 7000, "13B": 13000, "32B": 32000,
    }
    return size_to_num.get(size, 0)


def plot_accuracy_metrics(
    ladder_results: Dict[str, Dict[str, Dict]],
    sizes: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    show: bool = True,
):
    """
    Plot accuracy metrics comparison across ladders.

    Creates a multi-panel figure with one subplot per accuracy cluster,
    showing how each ladder performs across model sizes.
    """
    if not PLOTTING_AVAILABLE:
        print("Warning: matplotlib not available, skipping plots")
        return

    clusters = list(ACCURACY_CLUSTERS.keys())
    n_clusters = len(clusters)
    n_cols = 2
    n_rows = (n_clusters + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten() if n_clusters > 1 else [axes]

    ladder_names = list(ladder_results.keys())
    colors = get_ladder_colors(ladder_names)

    for idx, cluster in enumerate(clusters):
        ax = axes[idx]

        for ladder_name, size_results in ladder_results.items():
            points = []

            for size, data in size_results.items():
                if sizes and size not in sizes:
                    continue
                acc_clusters = data.get("accuracy_clusters", {})
                if cluster in acc_clusters:
                    points.append((get_size_numeric(size), acc_clusters[cluster] * 100))

            # Sort by size (x value) to ensure proper line connections
            if points:
                points.sort(key=lambda p: p[0])
                x_vals, y_vals = zip(*points)
                ax.plot(x_vals, y_vals, 'o-', label=ladder_name,
                       color=colors[ladder_name], linewidth=2, markersize=8)

        ax.set_xlabel("Model Size (params)", fontsize=10)
        ax.set_ylabel("Accuracy (%)", fontsize=10)
        ax.set_title(f"{cluster}", fontsize=12, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)

        # Format x-axis with size labels
        all_sizes = sorted(set(
            size for size_results in ladder_results.values()
            for size in size_results.keys()
            if not sizes or size in sizes
        ), key=get_size_numeric)
        ax.set_xticks([get_size_numeric(s) for s in all_sizes])
        ax.set_xticklabels(all_sizes, fontsize=9)

    # Hide unused subplots
    for idx in range(n_clusters, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Accuracy Metrics by Cluster (higher is better)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path / "accuracy_comparison.png", dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path / 'accuracy_comparison.png'}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_bpb_metrics(
    ladder_results: Dict[str, Dict[str, Dict]],
    sizes: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    show: bool = True,
):
    """
    Plot BPB metrics comparison across ladders.
    """
    if not PLOTTING_AVAILABLE:
        print("Warning: matplotlib not available, skipping plots")
        return

    clusters = list(BPB_CLUSTERS.keys())
    n_clusters = len(clusters)

    fig, axes = plt.subplots(1, n_clusters, figsize=(7 * n_clusters, 5))
    axes = axes if n_clusters > 1 else [axes]

    ladder_names = list(ladder_results.keys())
    colors = get_ladder_colors(ladder_names)

    for idx, cluster in enumerate(clusters):
        ax = axes[idx]

        for ladder_name, size_results in ladder_results.items():
            points = []

            for size, data in size_results.items():
                if sizes and size not in sizes:
                    continue
                bpb_clusters = data.get("bpb_clusters", {})
                if cluster in bpb_clusters:
                    points.append((get_size_numeric(size), bpb_clusters[cluster]))

            # Sort by size (x value) to ensure proper line connections
            if points:
                points.sort(key=lambda p: p[0])
                x_vals, y_vals = zip(*points)
                ax.plot(x_vals, y_vals, 'o-', label=ladder_name,
                       color=colors[ladder_name], linewidth=2, markersize=8)

        ax.set_xlabel("Model Size (params)", fontsize=10)
        ax.set_ylabel("BPB (bits per byte)", fontsize=10)
        ax.set_title(f"{cluster}", fontsize=12, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)

        # Format x-axis with size labels
        all_sizes = sorted(set(
            size for size_results in ladder_results.values()
            for size in size_results.keys()
            if not sizes or size in sizes
        ), key=get_size_numeric)
        ax.set_xticks([get_size_numeric(s) for s in all_sizes])
        ax.set_xticklabels(all_sizes, fontsize=9)

    fig.suptitle("BPB Metrics (lower is better)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path / "bpb_comparison.png", dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path / 'bpb_comparison.png'}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_lm_metrics(
    ladder_results: Dict[str, Dict[str, Dict]],
    sizes: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    show: bool = True,
):
    """
    Plot language modeling metrics (perplexity) comparison across ladders.
    """
    if not PLOTTING_AVAILABLE:
        print("Warning: matplotlib not available, skipping plots")
        return

    # Collect all LM tasks that have data
    all_tasks = set()
    for size_results in ladder_results.values():
        for data in size_results.values():
            all_tasks.update(data.get("lm_metrics", {}).keys())

    # Select key tasks for visualization
    key_tasks = ["c4_en", "pile", "wikitext_103", "dolma_common-crawl", "dolma_wiki"]
    tasks_to_plot = [t for t in key_tasks if t in all_tasks]

    if not tasks_to_plot:
        print("No LM metrics to plot")
        return

    n_tasks = len(tasks_to_plot) + 1  # +1 for average PPL
    n_cols = min(3, n_tasks)
    n_rows = (n_tasks + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_tasks > 1 else [axes]

    ladder_names = list(ladder_results.keys())
    colors = get_ladder_colors(ladder_names)

    # Plot average PPL first
    ax = axes[0]
    for ladder_name, size_results in ladder_results.items():
        points = []

        for size, data in size_results.items():
            if sizes and size not in sizes:
                continue
            lm = data.get("lm_metrics", {})
            ppls = [v.get("PPL") for v in lm.values() if v.get("PPL") is not None]
            if ppls:
                points.append((get_size_numeric(size), sum(ppls) / len(ppls)))

        # Sort by size (x value) to ensure proper line connections
        if points:
            points.sort(key=lambda p: p[0])
            x_vals, y_vals = zip(*points)
            ax.plot(x_vals, y_vals, 'o-', label=ladder_name,
                   color=colors[ladder_name], linewidth=2, markersize=8)

    ax.set_xlabel("Model Size (params)", fontsize=10)
    ax.set_ylabel("Perplexity", fontsize=10)
    ax.set_title("Average PPL", fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)

    all_sizes = sorted(set(
        size for size_results in ladder_results.values()
        for size in size_results.keys()
        if not sizes or size in sizes
    ), key=get_size_numeric)
    ax.set_xticks([get_size_numeric(s) for s in all_sizes])
    ax.set_xticklabels(all_sizes, fontsize=9)

    # Plot individual task PPLs
    for idx, task in enumerate(tasks_to_plot, start=1):
        if idx >= len(axes):
            break
        ax = axes[idx]

        for ladder_name, size_results in ladder_results.items():
            points = []

            for size, data in size_results.items():
                if sizes and size not in sizes:
                    continue
                lm = data.get("lm_metrics", {})
                if task in lm and "PPL" in lm[task]:
                    points.append((get_size_numeric(size), lm[task]["PPL"]))

            # Sort by size (x value) to ensure proper line connections
            if points:
                points.sort(key=lambda p: p[0])
                x_vals, y_vals = zip(*points)
                ax.plot(x_vals, y_vals, 'o-', label=ladder_name,
                       color=colors[ladder_name], linewidth=2, markersize=8)

        ax.set_xlabel("Model Size (params)", fontsize=10)
        ax.set_ylabel("Perplexity", fontsize=10)
        ax.set_title(f"{task}", fontsize=12, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        ax.set_xticks([get_size_numeric(s) for s in all_sizes])
        ax.set_xticklabels(all_sizes, fontsize=9)

    # Hide unused subplots
    for idx in range(n_tasks, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Language Modeling - Perplexity (lower is better)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path / "lm_comparison.png", dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path / 'lm_comparison.png'}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_summary_comparison(
    ladder_results: Dict[str, Dict[str, Dict]],
    sizes: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    show: bool = True,
):
    """
    Create a summary bar chart comparing ladders at each model size.

    Shows grouped bars for each metric cluster, making it easy to compare
    architectures at the same scale.
    """
    if not PLOTTING_AVAILABLE:
        print("Warning: matplotlib not available, skipping plots")
        return

    ladder_names = list(ladder_results.keys())
    colors = get_ladder_colors(ladder_names)

    # Get all available sizes
    all_sizes = sorted(set(
        size for size_results in ladder_results.values()
        for size in size_results.keys()
        if not sizes or size in sizes
    ), key=get_size_numeric)

    if not all_sizes:
        print("No sizes to plot")
        return

    # Create a figure with subplots for each metric type
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Accuracy clusters average
    ax = axes[0, 0]
    x = range(len(all_sizes))
    width = 0.8 / len(ladder_names)

    for i, ladder_name in enumerate(ladder_names):
        size_results = ladder_results[ladder_name]
        y_vals = []
        for size in all_sizes:
            if size in size_results:
                acc = size_results[size].get("accuracy_clusters", {})
                if acc:
                    y_vals.append(sum(acc.values()) / len(acc) * 100)
                else:
                    y_vals.append(0)
            else:
                y_vals.append(0)

        offset = (i - len(ladder_names)/2 + 0.5) * width
        bars = ax.bar([xi + offset for xi in x], y_vals, width,
                     label=ladder_name, color=colors[ladder_name])

    ax.set_xlabel("Model Size", fontsize=11)
    ax.set_ylabel("Average Accuracy (%)", fontsize=11)
    ax.set_title("Average Accuracy Across Clusters", fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_sizes, fontsize=10)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # 2. Individual accuracy clusters as grouped bar
    ax = axes[0, 1]
    acc_clusters = list(ACCURACY_CLUSTERS.keys())

    # For each size, show all clusters for all ladders
    # Use the largest available size for this comparison
    largest_size = all_sizes[-1] if all_sizes else None

    if largest_size:
        x_clusters = range(len(acc_clusters))
        for i, ladder_name in enumerate(ladder_names):
            size_results = ladder_results[ladder_name]
            y_vals = []
            if largest_size in size_results:
                acc = size_results[largest_size].get("accuracy_clusters", {})
                for cluster in acc_clusters:
                    y_vals.append(acc.get(cluster, 0) * 100)
            else:
                y_vals = [0] * len(acc_clusters)

            offset = (i - len(ladder_names)/2 + 0.5) * width
            ax.bar([xi + offset for xi in x_clusters], y_vals, width,
                  label=ladder_name, color=colors[ladder_name])

        ax.set_xlabel("Metric Cluster", fontsize=11)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_title(f"Accuracy Breakdown at {largest_size}", fontsize=12, fontweight='bold')
        ax.set_xticks(list(x_clusters))
        ax.set_xticklabels(acc_clusters, fontsize=9, rotation=15, ha='right')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    # 3. BPB comparison
    ax = axes[1, 0]
    bpb_clusters = list(BPB_CLUSTERS.keys())

    for i, ladder_name in enumerate(ladder_names):
        size_results = ladder_results[ladder_name]
        y_vals = []
        for size in all_sizes:
            if size in size_results:
                bpb = size_results[size].get("bpb_clusters", {})
                if bpb:
                    y_vals.append(sum(bpb.values()) / len(bpb))
                else:
                    y_vals.append(0)
            else:
                y_vals.append(0)

        offset = (i - len(ladder_names)/2 + 0.5) * width
        ax.bar([xi + offset for xi in x], y_vals, width,
              label=ladder_name, color=colors[ladder_name])

    ax.set_xlabel("Model Size", fontsize=11)
    ax.set_ylabel("Average BPB", fontsize=11)
    ax.set_title("Average BPB Across Clusters (lower is better)", fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(all_sizes)))
    ax.set_xticklabels(all_sizes, fontsize=10)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Relative performance (percentage difference from first ladder)
    ax = axes[1, 1]
    if len(ladder_names) > 1:
        baseline_ladder = ladder_names[0]
        x = range(len(all_sizes))

        for i, ladder_name in enumerate(ladder_names[1:], start=1):
            y_vals = []
            for size in all_sizes:
                baseline_data = ladder_results[baseline_ladder].get(size, {})
                compare_data = ladder_results[ladder_name].get(size, {})

                baseline_acc = baseline_data.get("accuracy_clusters", {})
                compare_acc = compare_data.get("accuracy_clusters", {})

                if baseline_acc and compare_acc:
                    baseline_avg = sum(baseline_acc.values()) / len(baseline_acc)
                    compare_avg = sum(compare_acc.values()) / len(compare_acc)
                    if baseline_avg > 0:
                        diff = ((compare_avg - baseline_avg) / baseline_avg) * 100
                        y_vals.append(diff)
                    else:
                        y_vals.append(0)
                else:
                    y_vals.append(0)

            ax.bar([xi + (i-1) * width for xi in x], y_vals, width,
                  label=f"{ladder_name} vs {baseline_ladder}", color=colors[ladder_name])

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel("Model Size", fontsize=11)
        ax.set_ylabel("Relative Accuracy Difference (%)", fontsize=11)
        ax.set_title(f"Accuracy Relative to {baseline_ladder}", fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(all_sizes)))
        ax.set_xticklabels(all_sizes, fontsize=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, "Need multiple ladders\nfor comparison",
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title("Relative Performance", fontsize=12, fontweight='bold')

    fig.suptitle("Ladder Comparison Summary", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path / "summary_comparison.png", dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path / 'summary_comparison.png'}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_all(
    ladder_results: Dict[str, Dict[str, Dict]],
    sizes: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    show: bool = True,
):
    """Generate all plots."""
    if not PLOTTING_AVAILABLE:
        print("\nError: matplotlib is not installed. Install it with: pip install matplotlib")
        return

    print("\nGenerating plots...")
    plot_accuracy_metrics(ladder_results, sizes, output_path, show)
    plot_bpb_metrics(ladder_results, sizes, output_path, show)
    plot_lm_metrics(ladder_results, sizes, output_path, show)
    plot_summary_comparison(ladder_results, sizes, output_path, show)
    print("Done generating plots.")


def export_results(
    ladder_results: Dict[str, Dict[str, Dict]],
    output_path: Path,
    sizes: Optional[List[str]] = None,
):
    """Export results to various formats."""
    acc_df = create_accuracy_table(ladder_results, sizes)
    bpb_df = create_bpb_table(ladder_results, sizes)
    lm_df = create_lm_table(ladder_results, sizes)

    # Save as CSV
    if not acc_df.empty:
        acc_df.to_csv(output_path / "accuracy_comparison.csv", index=False)
    if not bpb_df.empty:
        bpb_df.to_csv(output_path / "bpb_comparison.csv", index=False)
    if not lm_df.empty:
        lm_df.to_csv(output_path / "lm_comparison.csv", index=False)

    # Save full data as pickle
    pkl_path = output_path / "ladder_comparison_full.pkl"
    pd.to_pickle({
        "accuracy": acc_df,
        "bpb": bpb_df,
        "lm": lm_df,
        "raw": ladder_results,
    }, pkl_path)

    # Generate markdown
    md_path = output_path / "ladder_comparison.md"
    with open(md_path, "w") as f:
        f.write("# Ladder Comparison Results\n\n")

        f.write("## Accuracy Metrics (higher is better)\n\n")
        if not acc_df.empty:
            f.write(acc_df.to_markdown(index=False))
        f.write("\n\n")

        f.write("## BPB Metrics (lower is better)\n\n")
        if not bpb_df.empty:
            f.write(bpb_df.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Language Modeling (lower is better)\n\n")
        if not lm_df.empty:
            f.write(lm_df.to_markdown(index=False))
        f.write("\n")

    print(f"\nExported results to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and compare model ladder evaluation metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze a single ladder
    python ladder_metrics_analysis.py --ladder-dir ~/Downloads/hybrid-gdn-ladder

    # Compare multiple ladders
    python ladder_metrics_analysis.py --compare \\
        hybrid-gdn:~/Downloads/hybrid-gdn-ladder \\
        pure-gdn:~/Downloads/pure-gdn-ladder \\
        transformer:~/Downloads/olmo3-ladder

    # Compare with plots
    python ladder_metrics_analysis.py --compare \\
        hybrid:~/Downloads/hybrid-ladder \\
        transformer:~/Downloads/transformer-ladder \\
        --plot

    # Export results with plots (saves to directory without displaying)
    python ladder_metrics_analysis.py --compare ... --export ~/results --plot

    # Group results by ladder instead of by size
    python ladder_metrics_analysis.py --compare ... --group-by-ladder
        """,
    )

    parser.add_argument(
        "--ladder-dir",
        type=Path,
        help="Directory containing metrics_*.pkl files for a single ladder",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        help="Compare multiple ladders. Format: name:path name:path ...",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        help="Only include these sizes (e.g., --sizes 1B 3B 7B)",
    )
    parser.add_argument(
        "--export",
        type=Path,
        help="Export results to this directory",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Also print raw metric values",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate visualization plots (requires matplotlib)",
    )
    parser.add_argument(
        "--group-by-ladder",
        action="store_true",
        help="Group results by ladder instead of by model size (default groups by size)",
    )

    args = parser.parse_args()

    if not args.ladder_dir and not args.compare:
        parser.error("Either --ladder-dir or --compare must be specified")

    ladder_results = {}

    if args.ladder_dir:
        ladder_dir = Path(args.ladder_dir).expanduser()
        ladder_name = ladder_dir.name
        print(f"\nAnalyzing ladder: {ladder_name}")
        print(f"Directory: {ladder_dir}")
        ladder_results[ladder_name] = analyze_ladder(ladder_dir)

    if args.compare:
        for spec in args.compare:
            if ":" in spec:
                name, path = spec.split(":", 1)
            else:
                path = spec
                name = Path(path).name

            ladder_dir = Path(path).expanduser()
            print(f"\nAnalyzing ladder: {name}")
            print(f"Directory: {ladder_dir}")
            ladder_results[name] = analyze_ladder(ladder_dir)

    # Determine grouping preference (default: group by size)
    group_by_size = not args.group_by_ladder

    # Create and display tables
    acc_df = create_accuracy_table(ladder_results, args.sizes)
    bpb_df = create_bpb_table(ladder_results, args.sizes)
    lm_df = create_lm_table(ladder_results, args.sizes)

    print_accuracy_table(acc_df, group_by_size=group_by_size)
    print_bpb_table(bpb_df, group_by_size=group_by_size)
    print_lm_table(lm_df, group_by_size=group_by_size)

    # Print raw metrics if requested
    if args.raw:
        print("\n\nRAW METRICS:")
        print("-" * 50)
        for ladder_name, sizes in ladder_results.items():
            print(f"\n{ladder_name}:")
            for size, data in sizes.items():
                if args.sizes and size not in args.sizes:
                    continue
                print(f"\n  {size} (step {data['step']}, {data['tokens']:,} tokens):")
                for metric, value in sorted(data["raw"].items()):
                    if isinstance(value, float):
                        print(f"    {metric}: {value:.4f}")
                    else:
                        print(f"    {metric}: {value}")

    # Generate plots if requested
    if args.plot:
        # If exporting, save plots to export dir without showing
        # Otherwise show interactively
        output_path = Path(args.export).expanduser() if args.export else None
        show_plots = args.export is None
        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)
        plot_all(ladder_results, args.sizes, output_path, show=show_plots)

    # Export if requested
    if args.export:
        export_path = Path(args.export).expanduser()
        export_path.mkdir(parents=True, exist_ok=True)
        export_results(ladder_results, export_path, args.sizes)


if __name__ == "__main__":
    main()
