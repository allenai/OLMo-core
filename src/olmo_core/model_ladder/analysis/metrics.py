"""
Shared metrics extraction utilities for OlmoBaseEval evaluation suites.

Provides BPB column finding and aggregation used by both scaling law fitting
and ladder metrics analysis scripts.
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# =============================================================================
# OlmoBaseEval Base Easy Suite (Table 45) - BPB metrics for small-scale decisions
# BPB = Bits Per Byte (lower is better)
# =============================================================================

BASE_EASY_SUITE: Dict[str, Dict[str, Any]] = {
    "Math_BPB": {
        "tasks": [
            "minerva_math_algebra",
            "minerva_math_counting",
            "minerva_math_geometry",
            "minerva_math_intermediate_algebra",
            "minerva_math_number_theory",
            "minerva_math_prealgebra",
            "minerva_math_precalculus",
        ],
        "metric_type": "bpb",
        "icl": 4,
    },
    "Code_BPB": {
        "tasks": [
            "humaneval",
            "codex_humaneval",
            "mbpp",
            "codex_mbpp",
            "mt_mbpp_rust",
            "mt_mbpp_java",
            "mt_mbpp_cpp",
            "mt_mbpp_python",
            "mt_mbpp_javascript",
            "mt_mbpp_typescript",
            "mt_mbpp_go",
            "mt_mbpp_ruby",
            "mt_mbpp_swift",
            "mt_mbpp_kotlin",
            "mt_mbpp_scala",
            "mt_mbpp_php",
            "mt_mbpp_perl",
            "mt_mbpp_r",
            "mt_mbpp_julia",
            "mt_mbpp_lua",
            "mt_mbpp_d",
        ],
        "metric_type": "bpb",
        "icl": 3,
    },
    "QA_BPB": {
        "tasks": [
            "arc_challenge",
            "arc_easy",
            "mmlu",
            "csqa",
            "hellaswag",
            "winogrande",
            "socialiqa",
            "piqa",
            "coqa",
            "drop",
            "jeopardy",
            "naturalqs",
            "squad",
            "sciq",
            "qasper",
            "basic_skills",
            "dbqa",
            "protocolqa",
            "lambada",
            "medmcqa",
            "medqa",
            "sciriff",
        ],
        "metric_type": "bpb",
        "icl": 5,
    },
}


# =============================================================================
# Column Matching Utilities
# =============================================================================


def normalize_task_name(name: str) -> str:
    """Normalize task name for matching."""
    return name.lower().replace("-", "_").replace(" ", "_")


def find_metric_columns(
    df: pd.DataFrame,
    task_patterns: List[str],
    metric_types: List[str],
    prefer_v2: bool = True,
) -> Dict[str, str]:
    """
    Find columns matching task patterns and metric types.

    Args:
        df: DataFrame with evaluation results
        task_patterns: List of task name patterns to search for
        metric_types: List of metric types (e.g., ["acc", "accuracy", "pass@1"])
        prefer_v2: Prefer v2 versions of tasks

    Returns:
        Dict mapping normalized task name to column name
    """
    task_to_col: Dict[str, Tuple[str, int]] = {}

    for col in df.columns:
        col_lower = col.lower()
        col_normalized = normalize_task_name(col)

        has_metric = any(m.lower() in col_lower for m in metric_types)
        if not has_metric:
            continue

        for pattern in task_patterns:
            pattern_normalized = normalize_task_name(pattern)

            if pattern_normalized in col_normalized:
                score = 0
                if prefer_v2 and "v2" in col_lower:
                    score += 10
                if "_rc_" in col_lower:
                    score += 5
                if "length" in col_lower and "norm" in col_lower:
                    score += 3
                if "pass@1" in col_lower:
                    score += 2
                if "5shot" in col_lower:
                    score += 1

                if (
                    pattern_normalized not in task_to_col
                    or score > task_to_col[pattern_normalized][1]
                ):
                    task_to_col[pattern_normalized] = (col, score)
                break

    return {k: v[0] for k, v in task_to_col.items()}


def find_bpb_columns(df: pd.DataFrame, task_patterns: List[str]) -> Dict[str, str]:
    """Find BPB (bits per byte) columns for tasks."""
    return find_metric_columns(
        df,
        task_patterns,
        metric_types=["bpb", "bits_per_byte", "bits-per-byte"],
    )


# =============================================================================
# Cluster Aggregation
# =============================================================================


def aggregate_base_easy_cluster(
    df: pd.DataFrame,
    cluster_name: str,
    cluster_config: Dict[str, Any],
) -> Tuple[Optional[float], Dict[str, float]]:
    """
    Aggregate BPB metrics for a Base Easy cluster using the final checkpoint row.

    Returns (average_bpb, {task: bpb_value}).
    """
    task_patterns = cluster_config["tasks"]
    task_to_col = find_bpb_columns(df, task_patterns)

    if not task_to_col:
        return None, {}

    final_row = df[df["step"] == df["step"].max()].iloc[0]

    values: Dict[str, float] = {}
    for task, col in task_to_col.items():
        if col in final_row and pd.notna(final_row[col]):
            values[task] = float(final_row[col])

    if values:
        return sum(values.values()) / len(values), values
    return None, {}


def aggregate_base_easy_cluster_for_row(
    row: pd.Series,
    all_columns: pd.Index,
    cluster_name: str,
    cluster_config: Dict[str, Any],
    precomputed_col_map: Optional[Dict[str, str]] = None,
) -> Optional[float]:
    """
    Aggregate BPB metrics for a Base Easy cluster from a single row.

    This is the per-checkpoint variant used for scaling law fitting,
    where we need BPB for each checkpoint individually.

    Args:
        row: A single row from the DataFrame.
        all_columns: The full set of columns (for column name matching).
        cluster_name: Name of the cluster (e.g., "Math_BPB").
        cluster_config: Config dict from BASE_EASY_SUITE.
        precomputed_col_map: Optional pre-computed task-to-column mapping
            to avoid re-matching on every row.

    Returns:
        Average BPB for this cluster at this checkpoint, or None if no data.
    """
    if precomputed_col_map is None:
        # Build a minimal DataFrame with just column names for matching
        dummy_df = pd.DataFrame(columns=all_columns)
        task_patterns = cluster_config["tasks"]
        precomputed_col_map = find_bpb_columns(dummy_df, task_patterns)

    if not precomputed_col_map:
        return None

    values = []
    for _task, col in precomputed_col_map.items():
        val = row.get(col)
        if val is not None and pd.notna(val):
            values.append(float(val))

    if values:
        return sum(values) / len(values)
    return None
