#!/usr/bin/env python3
"""
Ladder Metrics Analysis Script

Aggregates evaluation metrics from model ladder runs into OLMo 3-style
evaluation clusters (Math, Code, MC_STEM, MC_NonSTEM, GenQA, BasicSkills).

Usage:
    python ladder_metrics_analysis.py --ladder-dir ~/Downloads/<ladder-name>
    python ladder_metrics_analysis.py --compare ladder1:~/Downloads/hybrid-gdn ladder2:~/Downloads/pure-gdn
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# Task cluster definitions matching OLMo 3 evaluation groups
TASK_CLUSTERS = {
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
        "humaneval",
        "codex_humaneval",
        "codex_mbpp",
        "mbpp",
        "mt_mbpp_rust",
        "mt_mbpp_java",
        "mt_mbpp_cpp",
    ],
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

# Preferred metric types (in order of preference)
PREFERRED_METRICS = ["acc", "accuracy", "len_norm", "bpb", "ce_loss"]


def find_metric_columns(df: pd.DataFrame, task_patterns: List[str]) -> List[str]:
    """Find columns matching any of the task patterns."""
    matched = []
    for col in df.columns:
        col_lower = col.lower()
        for pattern in task_patterns:
            if pattern.lower() in col_lower:
                matched.append(col)
                break
    return matched


def get_best_metric_for_task(df: pd.DataFrame, task_cols: List[str]) -> Dict[str, str]:
    """
    For each task, select the best metric type available.
    Returns dict mapping task base name to the column with best metric.
    """
    task_to_col = {}
    for col in task_cols:
        # Extract task base name (before metric type)
        col_lower = col.lower()
        for metric in PREFERRED_METRICS:
            if metric in col_lower:
                # Find the task name part
                task_base = col_lower.split(metric)[0].rstrip("_/ ")
                if task_base not in task_to_col:
                    task_to_col[task_base] = col
                break
        else:
            # No known metric type, use as-is
            task_to_col[col] = col
    return task_to_col


def aggregate_cluster_metrics(
    df: pd.DataFrame,
    cluster_name: str,
    task_patterns: List[str],
) -> Optional[float]:
    """Aggregate metrics for a single cluster."""
    matched_cols = find_metric_columns(df, task_patterns)
    if not matched_cols:
        return None

    # Get one column per task (preferring accuracy metrics)
    task_to_col = get_best_metric_for_task(df, matched_cols)

    # Get values from final checkpoint
    final_row = df[df["step"] == df["step"].max()].iloc[0]

    values = []
    for task, col in task_to_col.items():
        if col in final_row and pd.notna(final_row[col]):
            values.append(final_row[col])

    if values:
        return sum(values) / len(values)
    return None


def analyze_ladder(ladder_dir: Path) -> Dict[str, Dict[str, Dict]]:
    """
    Analyze all metrics files in a ladder directory.

    Returns:
        Dict mapping size -> {"clusters": {...}, "raw": {...}, "step": int, "tokens": int}
    """
    results = {}

    # Find all metrics pickle files
    pkl_files = list(ladder_dir.glob("metrics_*.pkl"))

    for pkl_path in sorted(pkl_files):
        # Extract size from filename (e.g., "metrics_7B.pkl" -> "7B")
        size = pkl_path.stem.replace("metrics_", "")

        try:
            df = pd.read_pickle(pkl_path)
        except Exception as e:
            print(f"Warning: Could not load {pkl_path}: {e}")
            continue

        if df.empty:
            continue

        # Get final checkpoint info
        final_row = df[df["step"] == df["step"].max()].iloc[0]
        final_step = int(final_row.get("step", 0))
        final_tokens = int(final_row.get("tokens", 0))

        # Aggregate clusters
        clusters = {}
        for cluster_name, task_patterns in TASK_CLUSTERS.items():
            value = aggregate_cluster_metrics(df, cluster_name, task_patterns)
            if value is not None:
                clusters[cluster_name] = value

        # Store all raw metric columns for reference
        metric_cols = [
            c for c in df.columns if c not in ["name", "step", "tokens", "size", "num_params"]
        ]
        raw_metrics = {col: final_row[col] for col in metric_cols if pd.notna(final_row.get(col))}

        results[size] = {
            "clusters": clusters,
            "raw": raw_metrics,
            "step": final_step,
            "tokens": final_tokens,
            "num_params": final_row.get("num_params", 0),
        }

    return results


def create_comparison_table(
    ladder_results: Dict[str, Dict[str, Dict]],
    sizes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Create a comparison table across multiple ladders.

    Args:
        ladder_results: Dict mapping ladder_name -> size -> results
        sizes: Optional list of sizes to include (default: all)

    Returns:
        DataFrame with comparison
    """
    rows = []

    for ladder_name, size_results in ladder_results.items():
        for size, data in size_results.items():
            if sizes and size not in sizes:
                continue

            row = {
                "Ladder": ladder_name,
                "Size": size,
                "Params": data.get("num_params", 0),
                "Tokens": data.get("tokens", 0),
            }
            row.update(data["clusters"])
            rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Sort by size (convert to numeric for sorting)
    size_order = {
        "60M": 0,
        "100M": 1,
        "190M": 2,
        "370M": 3,
        "600M": 4,
        "760M": 5,
        "1B": 6,
        "3B": 7,
        "7B": 8,
        "13B": 9,
        "32B": 10,
    }
    df["_size_order"] = df["Size"].map(lambda x: size_order.get(x, 99))
    df = df.sort_values(["Ladder", "_size_order"]).drop("_size_order", axis=1)

    return df


def print_olmo3_style_table(df: pd.DataFrame, cluster_cols: Optional[List[str]] = None):
    """Print a nicely formatted table similar to OLMo 3 paper."""
    if cluster_cols is None:
        cluster_cols = list(TASK_CLUSTERS.keys())

    # Filter to available columns
    cluster_cols = [c for c in cluster_cols if c in df.columns]

    print("\n" + "=" * 100)
    print("LADDER COMPARISON (OLMo 3 Style)")
    print("=" * 100)

    # Header
    header = f"{'Ladder':<20} {'Size':<8}"
    for col in cluster_cols:
        header += f" {col:>12}"
    print(header)
    print("-" * 100)

    # Rows
    for _, row in df.iterrows():
        line = f"{row['Ladder']:<20} {row['Size']:<8}"
        for col in cluster_cols:
            val = row.get(col)
            if pd.notna(val):
                # Format as percentage if < 1, otherwise as-is
                if val < 1:
                    line += f" {val*100:>11.1f}%"
                else:
                    line += f" {val:>12.2f}"
            else:
                line += f" {'-':>12}"
        print(line)

    print("=" * 100)


def export_results(
    ladder_results: Dict[str, Dict[str, Dict]],
    output_path: Path,
):
    """Export results to various formats."""
    comparison_df = create_comparison_table(ladder_results)

    # Save as CSV
    csv_path = output_path / "ladder_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"Saved CSV to: {csv_path}")

    # Save as pickle for full data
    pkl_path = output_path / "ladder_comparison_full.pkl"
    pd.to_pickle({"comparison": comparison_df, "raw": ladder_results}, pkl_path)
    print(f"Saved full data to: {pkl_path}")

    # Generate markdown table
    md_path = output_path / "ladder_comparison.md"
    with open(md_path, "w") as f:
        f.write("# Ladder Comparison Results\n\n")
        f.write(comparison_df.to_markdown(index=False))
        f.write("\n")
    print(f"Saved markdown to: {md_path}")


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

    # Export results
    python ladder_metrics_analysis.py --ladder-dir ~/Downloads/hybrid-gdn-ladder --export ~/results
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

    # Create and display comparison
    comparison_df = create_comparison_table(ladder_results, args.sizes)

    if comparison_df.empty:
        print("\nNo metrics found!")
        return

    print_olmo3_style_table(comparison_df)

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

    # Export if requested
    if args.export:
        export_path = Path(args.export).expanduser()
        export_path.mkdir(parents=True, exist_ok=True)
        export_results(ladder_results, export_path)


if __name__ == "__main__":
    main()
