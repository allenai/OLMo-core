#!/usr/bin/env python3
"""
Ladder Metrics Analysis Script - OlmoBaseEval Implementation

Aggregates evaluation metrics from model ladder runs into the exact OlmoBaseEval
evaluation suite as defined in the OLMo 3 paper (Tables 45, 46).

The suite consists of:
- Base Easy Suite: BPB metrics for small-scale data decisions (Table 45)
- Base Main Suite: Accuracy/pass@k metrics for in-loop evaluation (Table 46)
- Base Held-out Suite: Held-out benchmarks to prevent overfitting

Clusters follow the paper's task clustering (Section 3.3.1):
- Math, Code, FIM, MC_STEM, MC_NonSTEM, GenQA

Usage:
    python ladder_metrics_analysis.py --ladder-dir ~/Downloads/<ladder-name>
    python ladder_metrics_analysis.py --compare ladder1:~/Downloads/hybrid-gdn ladder2:~/Downloads/pure-gdn
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Optional plotting imports
try:
    import matplotlib.pyplot as plt

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


# =============================================================================
# OlmoBaseEval Base Easy Suite (Table 45) - BPB metrics for small-scale decisions
# BPB = Bits Per Byte (lower is better)
# =============================================================================

BASE_EASY_SUITE = {
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
        "icl": 4,  # 4-shot with human-written examples
    },
    "Code_BPB": {
        "tasks": [
            "humaneval",
            "codex_humaneval",
            "mbpp",
            "codex_mbpp",
            # MT MBPP (17 languages)
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
# OlmoBaseEval Base Main Suite (Table 46) - Accuracy/pass@k for in-loop evaluation
# =============================================================================

BASE_MAIN_SUITE = {
    # -------------------------------------------------------------------------
    # Math Cluster - CoT EM, pass@k (temp=0.6, top_p=0.6)
    # -------------------------------------------------------------------------
    "Math": {
        "tasks": {
            "gsm8k": {"format": "cot_em", "metric": "pass@k", "icl": 8, "k": [1, 4]},
            "gsm_symbolic": {"format": "cot_em", "metric": "pass@k", "icl": 8, "k": [1, 4], "subtasks": 3},
            "minerva_math": {"format": "cot_em", "metric": "pass@k", "icl": 4, "k": [1, 4], "subtasks": 7},
            "math_500": {"format": "cot_em", "metric": "pass@k", "icl": 4, "k": [1, 16]},
        },
        "higher_is_better": True,
    },
    # -------------------------------------------------------------------------
    # Code Cluster - Code Exec, pass@k (temp=0.6, top_p=0.6)
    # -------------------------------------------------------------------------
    "Code": {
        "tasks": {
            "humaneval": {"format": "code_exec", "metric": "pass@k", "icl": 3, "k": [1, 16]},
            "codex_humaneval": {"format": "code_exec", "metric": "pass@k", "icl": 3, "k": [1, 16]},
            "mbpp": {"format": "code_exec", "metric": "pass@k", "icl": 3, "k": [1, 16]},
            "codex_mbpp": {"format": "code_exec", "metric": "pass@k", "icl": 3, "k": [1, 16]},
            "bigcodebench": {"format": "code_exec", "metric": "pass@k", "icl": 3, "k": [1]},
            "ds_1000": {"format": "code_exec", "metric": "pass@k", "icl": 3, "k": [1]},
            "deepseek_leetcode": {"format": "code_exec", "metric": "pass@k", "icl": 0, "k": [1, 16]},
            "multipl_e_humaneval": {"format": "code_exec", "metric": "pass@k", "icl": 0, "k": [1, 16], "subtasks": 6},
            "multipl_e_mbpp": {"format": "code_exec", "metric": "pass@k", "icl": 0, "k": [1, 16], "subtasks": 6},
        },
        "higher_is_better": True,
    },
    # -------------------------------------------------------------------------
    # FIM Cluster - Fill-in-the-Middle, pass@1 (temp=0.8, top_p=0.95)
    # -------------------------------------------------------------------------
    "FIM": {
        "tasks": {
            "humaneval_fim_single": {"format": "fim", "metric": "pass@1", "icl": 0},
            "humaneval_fim_random": {"format": "fim", "metric": "pass@1", "icl": 0},
            "humaneval_fim_multi": {"format": "fim", "metric": "pass@1", "icl": 0},
        },
        "higher_is_better": True,
    },
    # -------------------------------------------------------------------------
    # MC_STEM Cluster - Multiple Choice, Accuracy
    # -------------------------------------------------------------------------
    "MC_STEM": {
        "tasks": {
            "arc_challenge": {"format": "mc", "metric": "acc", "icl": 5, "subtasks": 2},
            "arc_easy": {"format": "mc", "metric": "acc", "icl": 5},
            "mmlu_stem": {"format": "mc", "metric": "acc", "icl": 5, "subtasks": 19},
            "medmcqa": {"format": "mc", "metric": "acc", "icl": 5},
            "medqa": {"format": "mc", "metric": "acc", "icl": 5},
            "sciq": {"format": "mc", "metric": "acc", "icl": 5},
        },
        "higher_is_better": True,
    },
    # -------------------------------------------------------------------------
    # MC_NonSTEM Cluster - Multiple Choice / Rank Choice, Accuracy
    # -------------------------------------------------------------------------
    "MC_NonSTEM": {
        "tasks": {
            "mmlu_humanities": {"format": "mc", "metric": "acc", "icl": 5, "subtasks": 13},
            "mmlu_social_sciences": {"format": "mc", "metric": "acc", "icl": 5, "subtasks": 12},
            "mmlu_other": {"format": "mc", "metric": "acc", "icl": 5, "subtasks": 14},
            "csqa": {"format": "mc", "metric": "acc", "icl": 5},
            "piqa": {"format": "mc", "metric": "acc", "icl": 5},
            "socialiqa": {"format": "mc", "metric": "acc", "icl": 5},
            # Gen2MC tasks (new in OLMo 3)
            "drop_gen2mc": {"format": "mc", "metric": "acc", "icl": 5},
            "jeopardy_gen2mc": {"format": "mc", "metric": "acc", "icl": 5},
            "naturalqs_gen2mc": {"format": "mc", "metric": "acc", "icl": 5},
            "squad_gen2mc": {"format": "mc", "metric": "acc", "icl": 5},
            "coqa_gen2mc": {"format": "mc", "metric": "acc", "icl": 0},
            "basic_skills_mc": {"format": "mc", "metric": "acc", "icl": 5, "subtasks": 6},
            "basic_skills": {"format": "mc", "metric": "acc", "icl": 5, "subtasks": 6},
            # Rank Choice tasks
            "hellaswag": {"format": "rc_per_char", "metric": "acc", "icl": 5},
            "winogrande": {"format": "rc_none", "metric": "acc", "icl": 5},
            "lambada": {"format": "rc_per_char", "metric": "acc", "icl": 0},
        },
        "higher_is_better": True,
    },
    # -------------------------------------------------------------------------
    # GenQA Cluster - Generative QA, F1
    # -------------------------------------------------------------------------
    "GenQA": {
        "tasks": {
            "drop": {"format": "genqa", "metric": "f1", "icl": 5},
            "jeopardy": {"format": "genqa", "metric": "f1", "icl": 5},
            "naturalqs": {"format": "genqa", "metric": "f1", "icl": 5},
            "squad": {"format": "genqa", "metric": "f1", "icl": 5},
            "coqa": {"format": "genqa", "metric": "f1", "icl": 0},
        },
        "higher_is_better": True,
    },
}


# =============================================================================
# OlmoBaseEval Base Held-out Suite - To prevent overfitting
# =============================================================================

BASE_HELDOUT_SUITE = {
    "HeldOut": {
        "tasks": {
            "mmlu_pro": {"format": "mc", "metric": "acc", "icl": 5, "subtasks": 13},
            "lbpp": {"format": "code_exec", "metric": "pass@k", "icl": 0, "k": [1]},
            "deepmind_math": {"format": "cot_em", "metric": "pass@k", "icl": 5, "k": [1]},
            "bigbench_hard": {"format": "cot_em", "metric": "acc", "icl": 3, "subtasks": 55},
            "bbh": {"format": "cot_em", "metric": "acc", "icl": 3, "subtasks": 55},
        },
        "higher_is_better": True,
    },
}


# =============================================================================
# LM Evaluation Tasks (perplexity/loss based)
# =============================================================================

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
        Dict mapping task name to column name
    """
    task_to_col: Dict[str, Tuple[str, int]] = {}

    for col in df.columns:
        col_lower = col.lower()
        col_normalized = normalize_task_name(col)

        # Check if column contains any of the metric types
        has_metric = any(m.lower() in col_lower for m in metric_types)
        if not has_metric:
            continue

        for pattern in task_patterns:
            pattern_normalized = normalize_task_name(pattern)

            if pattern_normalized in col_normalized:
                # Score this column for preference
                score = 0

                # Prefer v2 versions
                if prefer_v2 and "v2" in col_lower:
                    score += 10

                # Prefer rc over mc for certain tasks
                if "_rc_" in col_lower:
                    score += 5

                # Prefer length-normalized accuracy
                if "length" in col_lower and "norm" in col_lower:
                    score += 3

                # Prefer pass@1 over other pass@k
                if "pass@1" in col_lower:
                    score += 2

                # Prefer 5shot over other shot counts for consistency
                if "5shot" in col_lower:
                    score += 1

                if pattern_normalized not in task_to_col or score > task_to_col[pattern_normalized][1]:
                    task_to_col[pattern_normalized] = (col, score)
                break

    return {k: v[0] for k, v in task_to_col.items()}


def find_accuracy_columns(df: pd.DataFrame, task_patterns: List[str]) -> Dict[str, str]:
    """Find accuracy/acc columns for tasks."""
    return find_metric_columns(
        df,
        task_patterns,
        metric_types=["accuracy", "acc", "len_norm", "length-normalized"],
    )


def find_passk_columns(df: pd.DataFrame, task_patterns: List[str], k: int = 1) -> Dict[str, str]:
    """Find pass@k columns for tasks."""
    return find_metric_columns(
        df,
        task_patterns,
        metric_types=[f"pass@{k}", f"pass_{k}", "pass@"],
    )


def find_f1_columns(df: pd.DataFrame, task_patterns: List[str]) -> Dict[str, str]:
    """Find F1 score columns for tasks."""
    return find_metric_columns(
        df,
        task_patterns,
        metric_types=["f1", "f1_score"],
    )


def find_bpb_columns(df: pd.DataFrame, task_patterns: List[str]) -> Dict[str, str]:
    """Find BPB (bits per byte) columns for tasks."""
    return find_metric_columns(
        df,
        task_patterns,
        metric_types=["bpb", "bits_per_byte", "bits-per-byte"],
    )


def find_lm_columns(df: pd.DataFrame) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """
    Find LM eval columns (CE loss and PPL).
    Returns dict mapping task name to (ce_loss_col, ppl_col).
    """
    task_to_cols: Dict[str, List[Optional[str]]] = {}

    for col in df.columns:
        col_lower = col.lower()

        if "eval/lm/" not in col_lower and "lm_eval" not in col_lower:
            continue

        for task in LM_TASKS:
            if task.lower() in col_lower:
                if task not in task_to_cols:
                    task_to_cols[task] = [None, None]

                if "ce loss" in col_lower or "ce_loss" in col_lower:
                    task_to_cols[task][0] = col
                elif "ppl" in col_lower:
                    task_to_cols[task][1] = col
                break

    return {k: (v[0], v[1]) for k, v in task_to_cols.items() if v[0] or v[1]}


# =============================================================================
# Cluster Aggregation
# =============================================================================

def aggregate_base_easy_cluster(
    df: pd.DataFrame,
    cluster_name: str,
    cluster_config: Dict[str, Any],
) -> Tuple[Optional[float], Dict[str, float]]:
    """
    Aggregate BPB metrics for a Base Easy cluster.
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


def aggregate_base_main_cluster(
    df: pd.DataFrame,
    cluster_name: str,
    cluster_config: Dict[str, Any],
) -> Tuple[Optional[float], Dict[str, float]]:
    """
    Aggregate metrics for a Base Main cluster based on task format.
    Returns (average_metric, {task: metric_value}).
    """
    tasks = cluster_config["tasks"]
    final_row = df[df["step"] == df["step"].max()].iloc[0]

    values: Dict[str, float] = {}

    for task_name, task_config in tasks.items():
        metric_type = task_config.get("metric", "acc")

        # Find appropriate column based on metric type
        if metric_type == "acc":
            task_to_col = find_accuracy_columns(df, [task_name])
        elif metric_type == "pass@k":
            k_values = task_config.get("k", [1])
            # Use pass@1 for cluster average by default
            task_to_col = find_passk_columns(df, [task_name], k=k_values[0])
        elif metric_type == "pass@1":
            task_to_col = find_passk_columns(df, [task_name], k=1)
        elif metric_type == "f1":
            task_to_col = find_f1_columns(df, [task_name])
        else:
            task_to_col = find_accuracy_columns(df, [task_name])

        task_normalized = normalize_task_name(task_name)
        if task_normalized in task_to_col:
            col = task_to_col[task_normalized]
            if col in final_row and pd.notna(final_row[col]):
                values[task_name] = float(final_row[col])

    if values:
        return sum(values.values()) / len(values), values
    return None, {}


def aggregate_all_clusters(
    df: pd.DataFrame,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Aggregate all OlmoBaseEval clusters.

    Returns:
        {
            "base_easy": {cluster_name: {"avg": float, "tasks": {task: value}}},
            "base_main": {cluster_name: {"avg": float, "tasks": {task: value}}},
            "heldout": {cluster_name: {"avg": float, "tasks": {task: value}}},
        }
    """
    results: Dict[str, Dict[str, Dict[str, Any]]] = {
        "base_easy": {},
        "base_main": {},
        "heldout": {},
    }

    # Base Easy Suite (BPB)
    for cluster_name, cluster_config in BASE_EASY_SUITE.items():
        avg, tasks = aggregate_base_easy_cluster(df, cluster_name, cluster_config)
        if avg is not None:
            results["base_easy"][cluster_name] = {"avg": avg, "tasks": tasks}

    # Base Main Suite (Accuracy/pass@k)
    for cluster_name, cluster_config in BASE_MAIN_SUITE.items():
        avg, tasks = aggregate_base_main_cluster(df, cluster_name, cluster_config)
        if avg is not None:
            results["base_main"][cluster_name] = {"avg": avg, "tasks": tasks}

    # Held-out Suite
    for cluster_name, cluster_config in BASE_HELDOUT_SUITE.items():
        avg, tasks = aggregate_base_main_cluster(df, cluster_name, cluster_config)
        if avg is not None:
            results["heldout"][cluster_name] = {"avg": avg, "tasks": tasks}

    return results


# =============================================================================
# Ladder Analysis
# =============================================================================

def analyze_ladder(ladder_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Analyze all metrics files in a ladder directory.

    Returns:
        Dict mapping size -> {
            "base_easy": {...},
            "base_main": {...},
            "heldout": {...},
            "lm_metrics": {...},
            "raw": {...},
            "step": int,
            "tokens": int,
            "num_params": int,
        }
    """
    results: Dict[str, Dict[str, Any]] = {}

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

        # Aggregate all OlmoBaseEval clusters
        cluster_results = aggregate_all_clusters(df)

        # Get LM metrics
        lm_cols = find_lm_columns(df)
        lm_metrics: Dict[str, Dict[str, float]] = {}
        for task, (ce_col, ppl_col) in lm_cols.items():
            lm_metrics[task] = {}
            if ce_col and pd.notna(final_row.get(ce_col)):
                lm_metrics[task]["CE loss"] = float(final_row[ce_col])
            if ppl_col and pd.notna(final_row.get(ppl_col)):
                lm_metrics[task]["PPL"] = float(final_row[ppl_col])

        # Store raw metrics
        metric_cols = [
            c for c in df.columns if c not in ["name", "step", "tokens", "size", "num_params"]
        ]
        raw_metrics = {col: final_row[col] for col in metric_cols if pd.notna(final_row.get(col))}

        results[size] = {
            "base_easy": cluster_results["base_easy"],
            "base_main": cluster_results["base_main"],
            "heldout": cluster_results["heldout"],
            "lm_metrics": lm_metrics,
            "raw": raw_metrics,
            "step": final_step,
            "tokens": final_tokens,
            "num_params": final_row.get("num_params", 0),
        }

    return results


# =============================================================================
# Table Creation
# =============================================================================

SIZE_ORDER = {
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


def _sort_by_size(df: pd.DataFrame, group_by_size: bool = True) -> pd.DataFrame:
    """Sort dataframe by model size."""
    df = df.copy()
    df["_size_order"] = df["Size"].map(lambda x: SIZE_ORDER.get(x, 99))
    if group_by_size:
        df = df.sort_values(["_size_order", "Ladder"]).drop("_size_order", axis=1)
    else:
        df = df.sort_values(["Ladder", "_size_order"]).drop("_size_order", axis=1)
    return df


def create_base_main_table(
    ladder_results: Dict[str, Dict[str, Dict[str, Any]]],
    sizes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Create Base Main Suite comparison table (accuracy-based metrics)."""
    rows = []

    for ladder_name, size_results in ladder_results.items():
        for size, data in size_results.items():
            if sizes and size not in sizes:
                continue

            row: Dict[str, Any] = {
                "Ladder": ladder_name,
                "Size": size,
            }

            # Add cluster averages
            base_main = data.get("base_main", {})
            for cluster_name in BASE_MAIN_SUITE.keys():
                if cluster_name in base_main:
                    row[cluster_name] = base_main[cluster_name]["avg"]

            rows.append(row)

    df = pd.DataFrame(rows)
    return _sort_by_size(df) if not df.empty else df


def create_base_easy_table(
    ladder_results: Dict[str, Dict[str, Dict[str, Any]]],
    sizes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Create Base Easy Suite comparison table (BPB metrics)."""
    rows = []

    for ladder_name, size_results in ladder_results.items():
        for size, data in size_results.items():
            if sizes and size not in sizes:
                continue

            row: Dict[str, Any] = {
                "Ladder": ladder_name,
                "Size": size,
            }

            # Add cluster averages
            base_easy = data.get("base_easy", {})
            for cluster_name in BASE_EASY_SUITE.keys():
                if cluster_name in base_easy:
                    row[cluster_name] = base_easy[cluster_name]["avg"]

            rows.append(row)

    df = pd.DataFrame(rows)
    return _sort_by_size(df) if not df.empty else df


def create_heldout_table(
    ladder_results: Dict[str, Dict[str, Dict[str, Any]]],
    sizes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Create Held-out Suite comparison table."""
    rows = []

    for ladder_name, size_results in ladder_results.items():
        for size, data in size_results.items():
            if sizes and size not in sizes:
                continue

            row: Dict[str, Any] = {
                "Ladder": ladder_name,
                "Size": size,
            }

            # Add individual task scores from heldout
            heldout = data.get("heldout", {})
            for cluster_name, cluster_data in heldout.items():
                for task_name, task_value in cluster_data.get("tasks", {}).items():
                    row[task_name] = task_value

            rows.append(row)

    df = pd.DataFrame(rows)
    return _sort_by_size(df) if not df.empty else df


def create_lm_table(
    ladder_results: Dict[str, Dict[str, Dict[str, Any]]],
    sizes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Create LM eval comparison table."""
    rows = []

    for ladder_name, size_results in ladder_results.items():
        for size, data in size_results.items():
            if sizes and size not in sizes:
                continue

            row: Dict[str, Any] = {
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


# =============================================================================
# Printing Functions
# =============================================================================

def print_base_main_table(df: pd.DataFrame, group_by_size: bool = True) -> None:
    """Print Base Main Suite table (OlmoBaseEval Main - accuracy metrics)."""
    if df.empty:
        print("\nNo Base Main Suite metrics found!")
        return

    df = _sort_by_size(df.copy(), group_by_size=group_by_size)

    cluster_cols = [c for c in BASE_MAIN_SUITE.keys() if c in df.columns]

    print("\n" + "=" * 120)
    print("OlmoBaseEval BASE MAIN SUITE (accuracy/pass@k, higher is better)")
    print("=" * 120)

    header = f"{'Size':<8} {'Ladder':<25}"
    for col in cluster_cols:
        header += f" {col:>12}"
    print(header)
    print("-" * 120)

    prev_size = None
    for _, row in df.iterrows():
        if group_by_size and prev_size is not None and row["Size"] != prev_size:
            print()
        prev_size = row["Size"]

        line = f"{row['Size']:<8} {row['Ladder']:<25}"
        for col in cluster_cols:
            val = row.get(col)
            if pd.notna(val):
                line += f" {val*100:>11.1f}%"
            else:
                line += f" {'-':>12}"
        print(line)

    print("=" * 120)


def print_base_easy_table(df: pd.DataFrame, group_by_size: bool = True) -> None:
    """Print Base Easy Suite table (OlmoBaseEval Easy - BPB metrics)."""
    if df.empty:
        print("\nNo Base Easy Suite metrics found!")
        return

    df = _sort_by_size(df.copy(), group_by_size=group_by_size)

    cluster_cols = [c for c in BASE_EASY_SUITE.keys() if c in df.columns]

    print("\n" + "=" * 100)
    print("OlmoBaseEval BASE EASY SUITE (BPB, lower is better)")
    print("=" * 100)

    header = f"{'Size':<8} {'Ladder':<25}"
    for col in cluster_cols:
        header += f" {col:>14}"
    print(header)
    print("-" * 100)

    prev_size = None
    for _, row in df.iterrows():
        if group_by_size and prev_size is not None and row["Size"] != prev_size:
            print()
        prev_size = row["Size"]

        line = f"{row['Size']:<8} {row['Ladder']:<25}"
        for col in cluster_cols:
            val = row.get(col)
            if pd.notna(val):
                line += f" {val:>14.4f}"
            else:
                line += f" {'-':>14}"
        print(line)

    print("=" * 100)


def print_heldout_table(df: pd.DataFrame, group_by_size: bool = True) -> None:
    """Print Held-out Suite table."""
    if df.empty:
        print("\nNo Held-out Suite metrics found!")
        return

    df = _sort_by_size(df.copy(), group_by_size=group_by_size)

    # Get all task columns
    task_cols = [c for c in df.columns if c not in ["Ladder", "Size"]]
    if not task_cols:
        return

    print("\n" + "=" * 100)
    print("OlmoBaseEval HELD-OUT SUITE (accuracy, higher is better)")
    print("=" * 100)

    header = f"{'Size':<8} {'Ladder':<25}"
    for col in task_cols:
        header += f" {col:>14}"
    print(header)
    print("-" * 100)

    prev_size = None
    for _, row in df.iterrows():
        if group_by_size and prev_size is not None and row["Size"] != prev_size:
            print()
        prev_size = row["Size"]

        line = f"{row['Size']:<8} {row['Ladder']:<25}"
        for col in task_cols:
            val = row.get(col)
            if pd.notna(val):
                line += f" {val*100:>13.1f}%"
            else:
                line += f" {'-':>14}"
        print(line)

    print("=" * 100)


def print_lm_table(df: pd.DataFrame, group_by_size: bool = True) -> None:
    """Print LM eval table (lower is better)."""
    if df.empty:
        print("\nNo LM metrics found!")
        return

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
        if group_by_size and prev_size is not None and row["Size"] != prev_size:
            print()
        prev_size = row["Size"]

        line = f"{row['Size']:<8} {row['Ladder']:<25}"
        for col in ppl_cols:
            val = row.get(col)
            if pd.notna(val):
                line += f" {val:>14.2f}"
            else:
                line += f" {'-':>14}"
        print(line)

    print("=" * 100)


# =============================================================================
# Visualization Functions (matching OLMo 3 paper style)
# =============================================================================

def get_ladder_colors(ladder_names: List[str]) -> Dict[str, str]:
    """Get distinct colors for each ladder (OLMo 3 paper color scheme)."""
    colors = [
        "#00838F",  # Teal (OLMo 3)
        "#E65100",  # Orange
        "#1565C0",  # Blue
        "#2E7D32",  # Green
        "#6A1B9A",  # Purple
        "#C62828",  # Red
        "#4527A0",  # Deep purple
        "#00695C",  # Dark teal
        "#EF6C00",  # Dark orange
        "#1976D2",  # Lighter blue
    ]
    return {name: colors[i % len(colors)] for i, name in enumerate(ladder_names)}


def get_size_numeric(size: str) -> float:
    """Convert size string to numeric value for x-axis positioning."""
    size_to_num = {
        "60M": 60,
        "100M": 100,
        "190M": 190,
        "370M": 370,
        "600M": 600,
        "760M": 760,
        "1B": 1000,
        "3B": 3000,
        "7B": 7000,
        "13B": 13000,
        "32B": 32000,
    }
    return size_to_num.get(size, 0)


def plot_base_main_metrics(
    ladder_results: Dict[str, Dict[str, Dict[str, Any]]],
    sizes: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Plot Base Main Suite metrics comparison across ladders.
    Matches OLMo 3 paper Figure 29 style.
    """
    if not PLOTTING_AVAILABLE:
        print("Warning: matplotlib not available, skipping plots")
        return

    clusters = list(BASE_MAIN_SUITE.keys())
    n_clusters = len(clusters)
    n_cols = 3
    n_rows = (n_clusters + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
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
                base_main = data.get("base_main", {})
                if cluster in base_main:
                    points.append((get_size_numeric(size), base_main[cluster]["avg"] * 100))

            if points:
                points.sort(key=lambda p: p[0])
                x_vals, y_vals = zip(*points)
                ax.plot(
                    x_vals,
                    y_vals,
                    "o-",
                    label=ladder_name,
                    color=colors[ladder_name],
                    linewidth=2,
                    markersize=6,
                )

        ax.set_xlabel("Model Size", fontsize=10)
        ax.set_ylabel("Accuracy (%)" if cluster != "FIM" else "pass@1 (%)", fontsize=10)
        ax.set_title(f"Base Main {cluster}", fontsize=11, fontweight="bold")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

        # Format x-axis
        all_sizes = sorted(
            set(
                size
                for size_results in ladder_results.values()
                for size in size_results.keys()
                if not sizes or size in sizes
            ),
            key=get_size_numeric,
        )
        ax.set_xticks([get_size_numeric(s) for s in all_sizes])
        ax.set_xticklabels(all_sizes, fontsize=8, rotation=45)

    # Hide unused subplots
    for idx in range(n_clusters, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("OlmoBaseEval Base Main Suite", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path / "olmobaseeval_main.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path / 'olmobaseeval_main.png'}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_base_easy_metrics(
    ladder_results: Dict[str, Dict[str, Dict[str, Any]]],
    sizes: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Plot Base Easy Suite metrics (BPB) comparison across ladders.
    Matches OLMo 3 paper Figure 30 style (bottom row).
    """
    if not PLOTTING_AVAILABLE:
        print("Warning: matplotlib not available, skipping plots")
        return

    clusters = list(BASE_EASY_SUITE.keys())
    n_clusters = len(clusters)

    fig, axes = plt.subplots(1, n_clusters, figsize=(5 * n_clusters, 4))
    axes_list = axes if n_clusters > 1 else [axes]

    ladder_names = list(ladder_results.keys())
    colors = get_ladder_colors(ladder_names)

    for idx, cluster in enumerate(clusters):
        ax = axes_list[idx]

        for ladder_name, size_results in ladder_results.items():
            points = []

            for size, data in size_results.items():
                if sizes and size not in sizes:
                    continue
                base_easy = data.get("base_easy", {})
                if cluster in base_easy:
                    points.append((get_size_numeric(size), base_easy[cluster]["avg"]))

            if points:
                points.sort(key=lambda p: p[0])
                x_vals, y_vals = zip(*points)
                ax.plot(
                    x_vals,
                    y_vals,
                    "o-",
                    label=ladder_name,
                    color=colors[ladder_name],
                    linewidth=2,
                    markersize=6,
                )

        ax.set_xlabel("Model Size", fontsize=10)
        ax.set_ylabel("Bits-per-byte", fontsize=10)
        ax.set_title(f"Base Easy {cluster.replace('_BPB', '')}", fontsize=11, fontweight="bold")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

        # Format x-axis
        all_sizes = sorted(
            set(
                size
                for size_results in ladder_results.values()
                for size in size_results.keys()
                if not sizes or size in sizes
            ),
            key=get_size_numeric,
        )
        ax.set_xticks([get_size_numeric(s) for s in all_sizes])
        ax.set_xticklabels(all_sizes, fontsize=8, rotation=45)

    fig.suptitle(
        "OlmoBaseEval Base Easy Suite (BPB - lower is better)", fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path / "olmobaseeval_easy.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path / 'olmobaseeval_easy.png'}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_scaling_analysis(
    ladder_results: Dict[str, Dict[str, Dict[str, Any]]],
    sizes: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Plot scaling analysis showing Easy vs Main suite relationship.
    Matches OLMo 3 paper Figure 31 style.
    """
    if not PLOTTING_AVAILABLE:
        print("Warning: matplotlib not available, skipping plots")
        return

    # Create 3-panel figure: QA, Math, Code
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ladder_names = list(ladder_results.keys())
    colors = get_ladder_colors(ladder_names)

    # Define mappings between Easy and Main clusters
    cluster_pairs = [
        ("QA_BPB", ["MC_STEM", "MC_NonSTEM", "GenQA"], "Base Main QA"),
        ("Math_BPB", ["Math"], "Base Main Math"),
        ("Code_BPB", ["Code"], "Base Main Code"),
    ]

    for ax_idx, (easy_cluster, main_clusters, title) in enumerate(cluster_pairs):
        ax = axes[ax_idx]

        for ladder_name, size_results in ladder_results.items():
            easy_vals = []
            main_vals = []

            for size, data in size_results.items():
                if sizes and size not in sizes:
                    continue

                base_easy = data.get("base_easy", {})
                base_main = data.get("base_main", {})

                if easy_cluster in base_easy:
                    easy_bpb = base_easy[easy_cluster]["avg"]

                    # Average main clusters
                    main_avgs = []
                    for mc in main_clusters:
                        if mc in base_main:
                            main_avgs.append(base_main[mc]["avg"])

                    if main_avgs:
                        main_acc = sum(main_avgs) / len(main_avgs)
                        easy_vals.append(easy_bpb)
                        main_vals.append(main_acc * 100)

            if easy_vals and main_vals:
                ax.scatter(
                    easy_vals,
                    main_vals,
                    label=ladder_name,
                    color=colors[ladder_name],
                    s=80,
                    marker="x",
                    linewidths=2,
                )

        ax.set_xlabel("Bits-per-byte", fontsize=10)
        ax.set_ylabel("Accuracy (%)" if "Math" not in title else "pass@1 (%)", fontsize=10)
        ax.set_title(f"{title}\n(Easy Suite → Main Suite)", fontsize=11, fontweight="bold")
        ax.invert_xaxis()  # Lower BPB is better, so invert
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        "Scaling Analysis: Easy Suite BPB vs Main Suite Accuracy", fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path / "scaling_analysis.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path / 'scaling_analysis.png'}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_summary_comparison(
    ladder_results: Dict[str, Dict[str, Dict[str, Any]]],
    sizes: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Create a comprehensive summary comparison figure.
    Matches OLMo 3 paper Table 2/3 style visualization.
    """
    if not PLOTTING_AVAILABLE:
        print("Warning: matplotlib not available, skipping plots")
        return

    ladder_names = list(ladder_results.keys())
    colors = get_ladder_colors(ladder_names)

    all_sizes = sorted(
        set(
            size
            for size_results in ladder_results.values()
            for size in size_results.keys()
            if not sizes or size in sizes
        ),
        key=get_size_numeric,
    )

    if not all_sizes:
        print("No sizes to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Base Main clusters average across sizes
    ax = axes[0, 0]
    x = list(range(len(all_sizes)))
    width = 0.8 / len(ladder_names)

    for i, ladder_name in enumerate(ladder_names):
        size_results = ladder_results[ladder_name]
        y_vals = []
        for size in all_sizes:
            if size in size_results:
                base_main = size_results[size].get("base_main", {})
                if base_main:
                    y_vals.append(sum(c["avg"] for c in base_main.values()) / len(base_main) * 100)
                else:
                    y_vals.append(0)
            else:
                y_vals.append(0)

        offset = (i - len(ladder_names) / 2 + 0.5) * width
        ax.bar([xi + offset for xi in x], y_vals, width, label=ladder_name, color=colors[ladder_name])

    ax.set_xlabel("Model Size", fontsize=11)
    ax.set_ylabel("Average Accuracy (%)", fontsize=11)
    ax.set_title("OlmoBaseEval Main Suite Average", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(all_sizes, fontsize=10)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # 2. Cluster breakdown at largest size
    ax = axes[0, 1]
    main_clusters = list(BASE_MAIN_SUITE.keys())
    largest_size = all_sizes[-1] if all_sizes else None

    if largest_size:
        x_clusters = list(range(len(main_clusters)))
        for i, ladder_name in enumerate(ladder_names):
            size_results = ladder_results[ladder_name]
            y_vals = []
            if largest_size in size_results:
                base_main = size_results[largest_size].get("base_main", {})
                for cluster in main_clusters:
                    if cluster in base_main:
                        y_vals.append(base_main[cluster]["avg"] * 100)
                    else:
                        y_vals.append(0)
            else:
                y_vals = [0] * len(main_clusters)

            offset = (i - len(ladder_names) / 2 + 0.5) * width
            ax.bar(
                [xi + offset for xi in x_clusters],
                y_vals,
                width,
                label=ladder_name,
                color=colors[ladder_name],
            )

        ax.set_xlabel("Metric Cluster", fontsize=11)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_title(f"Main Suite Breakdown at {largest_size}", fontsize=12, fontweight="bold")
        ax.set_xticks(x_clusters)
        ax.set_xticklabels(main_clusters, fontsize=9, rotation=30, ha="right")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    # 3. Base Easy BPB comparison
    ax = axes[1, 0]

    for i, ladder_name in enumerate(ladder_names):
        size_results = ladder_results[ladder_name]
        y_vals = []
        for size in all_sizes:
            if size in size_results:
                base_easy = size_results[size].get("base_easy", {})
                if base_easy:
                    y_vals.append(sum(c["avg"] for c in base_easy.values()) / len(base_easy))
                else:
                    y_vals.append(0)
            else:
                y_vals.append(0)

        offset = (i - len(ladder_names) / 2 + 0.5) * width
        ax.bar([xi + offset for xi in x], y_vals, width, label=ladder_name, color=colors[ladder_name])

    ax.set_xlabel("Model Size", fontsize=11)
    ax.set_ylabel("Average BPB", fontsize=11)
    ax.set_title("OlmoBaseEval Easy Suite Average (lower is better)", fontsize=12, fontweight="bold")
    ax.set_xticks(list(range(len(all_sizes))))
    ax.set_xticklabels(all_sizes, fontsize=10)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # 4. Relative performance comparison
    ax = axes[1, 1]
    if len(ladder_names) > 1:
        baseline_ladder = ladder_names[0]
        x = list(range(len(all_sizes)))

        for i, ladder_name in enumerate(ladder_names[1:], start=1):
            y_vals = []
            for size in all_sizes:
                baseline_data = ladder_results[baseline_ladder].get(size, {})
                compare_data = ladder_results[ladder_name].get(size, {})

                baseline_main = baseline_data.get("base_main", {})
                compare_main = compare_data.get("base_main", {})

                if baseline_main and compare_main:
                    baseline_avg = sum(c["avg"] for c in baseline_main.values()) / len(baseline_main)
                    compare_avg = sum(c["avg"] for c in compare_main.values()) / len(compare_main)
                    if baseline_avg > 0:
                        diff = ((compare_avg - baseline_avg) / baseline_avg) * 100
                        y_vals.append(diff)
                    else:
                        y_vals.append(0)
                else:
                    y_vals.append(0)

            ax.bar(
                [xi + (i - 1) * width for xi in x],
                y_vals,
                width,
                label=f"{ladder_name} vs {baseline_ladder}",
                color=colors[ladder_name],
            )

        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xlabel("Model Size", fontsize=11)
        ax.set_ylabel("Relative Accuracy Difference (%)", fontsize=11)
        ax.set_title(f"Main Suite Accuracy Relative to {baseline_ladder}", fontsize=12, fontweight="bold")
        ax.set_xticks(list(range(len(all_sizes))))
        ax.set_xticklabels(all_sizes, fontsize=10)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
    else:
        ax.text(
            0.5,
            0.5,
            "Need multiple ladders\nfor comparison",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_title("Relative Performance", fontsize=12, fontweight="bold")

    fig.suptitle("OlmoBaseEval Summary Comparison", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path / "summary_comparison.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path / 'summary_comparison.png'}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_lm_metrics(
    ladder_results: Dict[str, Dict[str, Dict[str, Any]]],
    sizes: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """Plot language modeling metrics (perplexity) comparison."""
    if not PLOTTING_AVAILABLE:
        print("Warning: matplotlib not available, skipping plots")
        return

    all_tasks: set = set()
    for size_results in ladder_results.values():
        for data in size_results.values():
            all_tasks.update(data.get("lm_metrics", {}).keys())

    key_tasks = ["c4_en", "pile", "wikitext_103", "dolma_common-crawl", "dolma_wiki"]
    tasks_to_plot = [t for t in key_tasks if t in all_tasks]

    if not tasks_to_plot:
        print("No LM metrics to plot")
        return

    n_tasks = len(tasks_to_plot) + 1
    n_cols = min(3, n_tasks)
    n_rows = (n_tasks + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes_list = axes.flatten() if n_tasks > 1 else [axes]

    ladder_names = list(ladder_results.keys())
    colors = get_ladder_colors(ladder_names)

    # Plot average PPL first
    ax = axes_list[0]
    for ladder_name, size_results in ladder_results.items():
        points = []

        for size, data in size_results.items():
            if sizes and size not in sizes:
                continue
            lm = data.get("lm_metrics", {})
            ppls = [v.get("PPL") for v in lm.values() if v.get("PPL") is not None]
            if ppls:
                points.append((get_size_numeric(size), sum(ppls) / len(ppls)))

        if points:
            points.sort(key=lambda p: p[0])
            x_vals, y_vals = zip(*points)
            ax.plot(
                x_vals, y_vals, "o-", label=ladder_name, color=colors[ladder_name], linewidth=2, markersize=8
            )

    ax.set_xlabel("Model Size", fontsize=10)
    ax.set_ylabel("Perplexity", fontsize=10)
    ax.set_title("Average PPL", fontsize=12, fontweight="bold")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    all_sizes = sorted(
        set(
            size
            for size_results in ladder_results.values()
            for size in size_results.keys()
            if not sizes or size in sizes
        ),
        key=get_size_numeric,
    )
    ax.set_xticks([get_size_numeric(s) for s in all_sizes])
    ax.set_xticklabels(all_sizes, fontsize=9, rotation=45)

    # Plot individual task PPLs
    for idx, task in enumerate(tasks_to_plot, start=1):
        if idx >= len(axes_list):
            break
        ax = axes_list[idx]

        for ladder_name, size_results in ladder_results.items():
            points = []

            for size, data in size_results.items():
                if sizes and size not in sizes:
                    continue
                lm = data.get("lm_metrics", {})
                if task in lm and "PPL" in lm[task]:
                    points.append((get_size_numeric(size), lm[task]["PPL"]))

            if points:
                points.sort(key=lambda p: p[0])
                x_vals, y_vals = zip(*points)
                ax.plot(
                    x_vals,
                    y_vals,
                    "o-",
                    label=ladder_name,
                    color=colors[ladder_name],
                    linewidth=2,
                    markersize=8,
                )

        ax.set_xlabel("Model Size", fontsize=10)
        ax.set_ylabel("Perplexity", fontsize=10)
        ax.set_title(f"{task}", fontsize=12, fontweight="bold")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        ax.set_xticks([get_size_numeric(s) for s in all_sizes])
        ax.set_xticklabels(all_sizes, fontsize=9, rotation=45)

    for idx in range(n_tasks, len(axes_list)):
        axes_list[idx].set_visible(False)

    fig.suptitle("Language Modeling - Perplexity (lower is better)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path / "lm_comparison.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path / 'lm_comparison.png'}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_all(
    ladder_results: Dict[str, Dict[str, Dict[str, Any]]],
    sizes: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """Generate all OlmoBaseEval-style plots."""
    if not PLOTTING_AVAILABLE:
        print("\nError: matplotlib is not installed. Install it with: pip install matplotlib")
        return

    print("\nGenerating OlmoBaseEval plots...")
    plot_base_main_metrics(ladder_results, sizes, output_path, show)
    plot_base_easy_metrics(ladder_results, sizes, output_path, show)
    plot_scaling_analysis(ladder_results, sizes, output_path, show)
    plot_summary_comparison(ladder_results, sizes, output_path, show)
    plot_lm_metrics(ladder_results, sizes, output_path, show)
    print("Done generating plots.")


# =============================================================================
# LaTeX Table Generation
# =============================================================================

def generate_latex_table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    metric_cols: List[str],
    format_pct: bool = True,
    higher_is_better: bool = True,
    group_by_size: bool = True,
) -> str:
    """Generate a LaTeX table with booktabs formatting."""
    if df.empty:
        return ""

    df = _sort_by_size(df.copy(), group_by_size=group_by_size)

    def escape_latex(s: str) -> str:
        return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")

    n_metrics = len(metric_cols)
    col_spec = "ll" + "r" * n_metrics

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"    \centering")
    lines.append(r"    \caption{" + escape_latex(caption) + "}")
    lines.append(r"    \label{" + label + "}")
    lines.append(r"    \begin{tabular}{" + col_spec + "}")
    lines.append(r"        \toprule")

    header_parts = ["Size", "Architecture"]
    for col in metric_cols:
        header_parts.append(escape_latex(col))
    lines.append("        " + " & ".join(header_parts) + r" \\")
    lines.append(r"        \midrule")

    # Find best values per size
    best_per_size: Dict[str, Dict[str, float]] = {}
    for col in metric_cols:
        best_per_size[col] = {}
        for size in df["Size"].unique():
            size_df = df[df["Size"] == size]
            valid_vals = size_df[col].dropna()
            if not valid_vals.empty:
                if higher_is_better:
                    best_per_size[col][size] = valid_vals.max()
                else:
                    best_per_size[col][size] = valid_vals.min()

    prev_size = None
    for _, row in df.iterrows():
        if group_by_size and prev_size is not None and row["Size"] != prev_size:
            lines.append(r"        \midrule")
        prev_size = row["Size"]

        row_parts = [row["Size"], escape_latex(row["Ladder"])]

        for col in metric_cols:
            val = row.get(col)
            if pd.notna(val):
                if format_pct:
                    formatted = f"{val * 100:.1f}"
                else:
                    formatted = f"{val:.4f}"

                if col in best_per_size and row["Size"] in best_per_size[col]:
                    if abs(val - best_per_size[col][row["Size"]]) < 1e-6:
                        formatted = r"\textbf{" + formatted + "}"
                row_parts.append(formatted)
            else:
                row_parts.append("--")

        lines.append("        " + " & ".join(row_parts) + r" \\")

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_comparison_latex(
    ladder_results: Dict[str, Dict[str, Dict[str, Any]]],
    sizes: Optional[List[str]] = None,
) -> str:
    """Generate LaTeX tables comparing all ladders in OlmoBaseEval format."""
    main_df = create_base_main_table(ladder_results, sizes)
    easy_df = create_base_easy_table(ladder_results, sizes)
    heldout_df = create_heldout_table(ladder_results, sizes)
    lm_df = create_lm_table(ladder_results, sizes)

    latex_parts = []
    latex_parts.append("% OlmoBaseEval LaTeX tables generated by ladder_metrics_analysis.py")
    latex_parts.append("% Requires: \\usepackage{booktabs}")
    latex_parts.append("")

    # Base Main Suite
    if not main_df.empty:
        main_cols = [c for c in BASE_MAIN_SUITE.keys() if c in main_df.columns]
        if main_cols:
            latex_parts.append(
                generate_latex_table(
                    main_df,
                    caption="OlmoBaseEval Base Main Suite comparison (higher is better)",
                    label="tab:olmobaseeval_main",
                    metric_cols=main_cols,
                    format_pct=True,
                    higher_is_better=True,
                    group_by_size=True,
                )
            )

    # Base Easy Suite
    if not easy_df.empty:
        easy_cols = [c for c in BASE_EASY_SUITE.keys() if c in easy_df.columns]
        if easy_cols:
            latex_parts.append(
                generate_latex_table(
                    easy_df,
                    caption="OlmoBaseEval Base Easy Suite comparison (BPB, lower is better)",
                    label="tab:olmobaseeval_easy",
                    metric_cols=easy_cols,
                    format_pct=False,
                    higher_is_better=False,
                    group_by_size=True,
                )
            )

    # Held-out Suite
    if not heldout_df.empty:
        heldout_cols = [c for c in heldout_df.columns if c not in ["Ladder", "Size"]]
        if heldout_cols:
            latex_parts.append(
                generate_latex_table(
                    heldout_df,
                    caption="OlmoBaseEval Held-out Suite comparison (higher is better)",
                    label="tab:olmobaseeval_heldout",
                    metric_cols=heldout_cols,
                    format_pct=True,
                    higher_is_better=True,
                    group_by_size=True,
                )
            )

    # LM comparison
    if not lm_df.empty:
        ppl_cols = [c for c in lm_df.columns if "PPL" in c]
        if ppl_cols:
            latex_parts.append(
                generate_latex_table(
                    lm_df,
                    caption="Language modeling perplexity comparison (lower is better)",
                    label="tab:lm_comparison",
                    metric_cols=ppl_cols,
                    format_pct=False,
                    higher_is_better=False,
                    group_by_size=True,
                )
            )

    return "\n\n".join(latex_parts)


# =============================================================================
# Export Functions
# =============================================================================

def export_results(
    ladder_results: Dict[str, Dict[str, Dict[str, Any]]],
    output_path: Path,
    sizes: Optional[List[str]] = None,
) -> None:
    """Export results to various formats."""
    main_df = create_base_main_table(ladder_results, sizes)
    easy_df = create_base_easy_table(ladder_results, sizes)
    heldout_df = create_heldout_table(ladder_results, sizes)
    lm_df = create_lm_table(ladder_results, sizes)

    # Save as CSV
    if not main_df.empty:
        main_df.to_csv(output_path / "olmobaseeval_main.csv", index=False)
    if not easy_df.empty:
        easy_df.to_csv(output_path / "olmobaseeval_easy.csv", index=False)
    if not heldout_df.empty:
        heldout_df.to_csv(output_path / "olmobaseeval_heldout.csv", index=False)
    if not lm_df.empty:
        lm_df.to_csv(output_path / "lm_comparison.csv", index=False)

    # Save full data as pickle
    pkl_path = output_path / "olmobaseeval_full.pkl"
    pd.to_pickle(
        {
            "base_main": main_df,
            "base_easy": easy_df,
            "heldout": heldout_df,
            "lm": lm_df,
            "raw": ladder_results,
        },
        pkl_path,
    )

    # Generate markdown
    md_path = output_path / "olmobaseeval_results.md"
    with open(md_path, "w") as f:
        f.write("# OlmoBaseEval Results\n\n")

        f.write("## Base Main Suite (accuracy/pass@k, higher is better)\n\n")
        if not main_df.empty:
            f.write(main_df.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Base Easy Suite (BPB, lower is better)\n\n")
        if not easy_df.empty:
            f.write(easy_df.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Held-out Suite (accuracy, higher is better)\n\n")
        if not heldout_df.empty:
            f.write(heldout_df.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Language Modeling (perplexity, lower is better)\n\n")
        if not lm_df.empty:
            f.write(lm_df.to_markdown(index=False))
        f.write("\n")

    # Generate LaTeX
    latex_output = generate_comparison_latex(ladder_results, sizes)
    latex_path = output_path / "olmobaseeval_tables.tex"
    with open(latex_path, "w") as f:
        f.write(latex_output)

    print(f"\nExported OlmoBaseEval results to: {output_path}")
    print(
        f"  CSV files: olmobaseeval_main.csv, olmobaseeval_easy.csv, olmobaseeval_heldout.csv, lm_comparison.csv"
    )
    print(f"  LaTeX tables: {latex_path}")
    print(f"  Markdown: {md_path}")
    print(f"  Full data: {pkl_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="OlmoBaseEval: Analyze and compare model ladder evaluation metrics",
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

    # Compare with plots (OLMo 3 paper style)
    python ladder_metrics_analysis.py --compare \\
        hybrid:~/Downloads/hybrid-ladder \\
        transformer:~/Downloads/transformer-ladder \\
        --plot

    # Export results with plots
    python ladder_metrics_analysis.py --compare ... --export ~/results --plot

    # Print LaTeX tables
    python ladder_metrics_analysis.py --compare ... --latex
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
        help="Generate OlmoBaseEval-style visualization plots",
    )
    parser.add_argument(
        "--group-by-ladder",
        action="store_true",
        help="Group results by ladder instead of by model size",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Print LaTeX tables (booktabs format) to stdout",
    )

    args = parser.parse_args()

    if not args.ladder_dir and not args.compare:
        parser.error("Either --ladder-dir or --compare must be specified")

    ladder_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

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

    group_by_size = not args.group_by_ladder

    # Create and display tables
    main_df = create_base_main_table(ladder_results, args.sizes)
    easy_df = create_base_easy_table(ladder_results, args.sizes)
    heldout_df = create_heldout_table(ladder_results, args.sizes)
    lm_df = create_lm_table(ladder_results, args.sizes)

    print_base_main_table(main_df, group_by_size=group_by_size)
    print_base_easy_table(easy_df, group_by_size=group_by_size)
    print_heldout_table(heldout_df, group_by_size=group_by_size)
    print_lm_table(lm_df, group_by_size=group_by_size)

    # Print raw metrics if requested
    if args.raw:
        print("\n\nRAW METRICS:")
        print("-" * 50)
        for ladder_name, sizes_data in ladder_results.items():
            print(f"\n{ladder_name}:")
            for size, data in sizes_data.items():
                if args.sizes and size not in args.sizes:
                    continue
                print(f"\n  {size} (step {data['step']}, {data['tokens']:,} tokens):")

                print("    Base Main Suite:")
                for cluster, cluster_data in data.get("base_main", {}).items():
                    print(f"      {cluster}: {cluster_data['avg']*100:.1f}%")
                    for task, val in cluster_data.get("tasks", {}).items():
                        print(f"        - {task}: {val*100:.1f}%")

                print("    Base Easy Suite (BPB):")
                for cluster, cluster_data in data.get("base_easy", {}).items():
                    print(f"      {cluster}: {cluster_data['avg']:.4f}")

    # Print LaTeX tables if requested
    if args.latex:
        print("\n" + "=" * 80)
        print("LATEX TABLES (booktabs format)")
        print("=" * 80)
        print("\n% Add to preamble: \\usepackage{booktabs}\n")

        latex_output = generate_comparison_latex(ladder_results, args.sizes)
        print(latex_output)

    # Generate plots if requested
    if args.plot:
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
