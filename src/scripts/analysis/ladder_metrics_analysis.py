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


# Corrected non-embedding parameter counts for each model type and size.
# These override the values from pickle files which may include embedding params.
# Imported from fit_chinchilla_scaling_laws.py conventions.
# Human-readable display names for each ladder.
# Used in plots, tables, and printed output.
DISPLAY_NAMES: Dict[str, str] = {
    "olmo3": "Olmo 3",
    "olmo3-1": "Olmo 3 v1",
    "olmo3-2": "Olmo 3 v2",
    "olmo3-3": "Olmo 3 v3",
    "pure-gdn": "Pure GDN",
    "hybrid-gdn": "Hybrid GDN (1/4)",
    "hybrid-gdn-half": "Hybrid GDN (1/2)",
    "hybrid-gdn-eight": "Hybrid GDN (1/8)",
    "hybrid-gdn-middle": "Hybrid GDN (Middle)",
    "pure-mamba": "Pure Mamba",
    "hybrid-mamba": "Hybrid Mamba",
}


def get_display_name(ladder_name: str) -> str:
    """Return a human-readable display name for a ladder, falling back to the raw name."""
    return DISPLAY_NAMES.get(ladder_name.lower(), ladder_name)


def _starred_display_name(ladder_name: str, star_ladder: Optional[str] = None) -> str:
    """Return display name with a star appended if this is the selected architecture."""
    display = get_display_name(ladder_name)
    if star_ladder and ladder_name.lower() == star_ladder.lower():
        display += " \u2605"  # ★
    return display


CORRECTED_PARAM_COUNTS: Dict[str, Dict[str, int]] = {
    "olmo3-1": {
        "60M": 57_422_208,
        "100M": 101_736_960,
        "190M": 190_354_176,
        "370M": 371_262_464,
        "600M": 547_964_160,
        "760M": 758_220_288,
    },
    "olmo3-2": {
        "60M": 57_422_208,
        "100M": 101_736_960,
        "190M": 190_354_176,
        "370M": 371_262_464,
        "600M": 547_964_160,
        "760M": 758_220_288,
    },
    "olmo3-3": {
        "60M": 57_422_208,
        "100M": 101_736_960,
        "190M": 190_354_176,
        "370M": 371_262_464,
        "600M": 547_964_160,
        "760M": 758_220_288,
    },
    "pure-gdn": {
        "60M": 78_045_696,
        "100M": 139_771_584,
        "190M": 275_789_856,
        "370M": 573_609_472,
        "600M": 779_794_176,
    },
    "hybrid-gdn": {
        "60M": 72_889_824,
        "100M": 130_262_928,
        "190M": 254_430_936,
        "370M": 523_022_720,
        "600M": 721_836_672,
        "760M": 947_913_600,
        "1B": 1_481_856_384,
    },
    "hybrid-gdn-middle": {
        "60M": 70_311_888,
        "100M": 127_093_376,
        "190M": 247_311_296,
        "370M": 510_376_032,
        "600M": 707_347_296,
    },
    "hybrid-gdn-half": {
        "60M": 67_733_952,
        "100M": 120_754_272,
        "190M": 233_072_016,
        "370M": 472_435_968,
        "600M": 663_879_168,
    },
    "hybrid-gdn-eight": {
        "60M": 75_467_760,
        "100M": 133_432_480,
        "190M": 261_550_576,
        "370M": 548_316_096,
        "600M": 750_815_424,
        "760M": 979_529_152,
    },
    "pure-mamba": {
        "60M": 59_837_760,
        "100M": 109_746_048,
        "190M": 207_115_584,
    },
    "hybrid-mamba": {
        "60M": 59_837_760,
        "100M": 107_743_776,
        "190M": 202_925_232,
    },
}


def get_corrected_param_count(ladder_name: str, size: str) -> Optional[int]:
    """
    Look up the corrected non-embedding parameter count for a given ladder and size.
    """
    ladder_lower = ladder_name.lower()

    # Try exact match first
    if ladder_lower in CORRECTED_PARAM_COUNTS:
        size_map = CORRECTED_PARAM_COUNTS[ladder_lower]
        if size in size_map:
            return size_map[size]

    # Try substring matches
    for pattern, size_map in CORRECTED_PARAM_COUNTS.items():
        if pattern in ladder_lower:
            if size in size_map:
                return size_map[size]

    return None


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
            "gsm_symbolic": {
                "format": "cot_em",
                "metric": "pass@k",
                "icl": 8,
                "k": [1, 4],
                "subtasks": 3,
            },
            "minerva_math": {
                "format": "cot_em",
                "metric": "pass@k",
                "icl": 4,
                "k": [1, 4],
                "subtasks": 7,
            },
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
            "deepseek_leetcode": {
                "format": "code_exec",
                "metric": "pass@k",
                "icl": 0,
                "k": [1, 16],
            },
            "multipl_e_humaneval": {
                "format": "code_exec",
                "metric": "pass@k",
                "icl": 0,
                "k": [1, 16],
                "subtasks": 6,
            },
            "multipl_e_mbpp": {
                "format": "code_exec",
                "metric": "pass@k",
                "icl": 0,
                "k": [1, 16],
                "subtasks": 6,
            },
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

                if (
                    pattern_normalized not in task_to_col
                    or score > task_to_col[pattern_normalized][1]
                ):
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


def analyze_ladder(
    ladder_dir: Path,
    ladder_name: str = "",
    use_all_checkpoints: bool = False,
    post_decay_only: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze all metrics files in a ladder directory.

    Args:
        ladder_dir: Directory containing metrics_*.pkl files.
        ladder_name: Name of the ladder (used for corrected parameter count lookup).
        use_all_checkpoints: If True, include all checkpoints from each model size
            (keyed as "size@D/N_ratio"). If False, only use the final checkpoint.
        post_decay_only: If True (default), only include post-decay checkpoints
            (D/N = 10, 20, 40, 80, 160, ...). If False, also include pre-decay.

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

        # Get corrected parameter count
        raw_num_params = df["num_params"].iloc[0] if "num_params" in df.columns else 0
        corrected_params = get_corrected_param_count(ladder_name, size)
        num_params = corrected_params if corrected_params is not None else raw_num_params
        if corrected_params is not None and raw_num_params:
            print(
                f"  {size}: Using corrected non-embedding params: "
                f"{corrected_params/1e6:.1f}M (was {raw_num_params/1e6:.1f}M)"
            )

        # Canonical D/N values from WSDS Chinchilla schedule with tokens_per_param=20.
        # Post-decay: end of cooldown at each period (0.5xC, 1xC, 2xC, ...).
        # Pre-decay: ~90% through each period (decay_fraction=0.1).
        # Always snap against the full set so pre-decay checkpoints don't get
        # mis-snapped to post-decay values.  Then filter by post_decay_only.
        _POST_DECAY_DN = [10, 20, 40, 80, 160]
        _PRE_DECAY_DN = [9, 19, 38, 76, 152]
        _ALL_DN = sorted(_POST_DECAY_DN + _PRE_DECAY_DN)
        _KEEP_DN = set(_POST_DECAY_DN) if post_decay_only else set(_ALL_DN)

        if use_all_checkpoints:
            # Process each checkpoint row
            for _, row in df.iterrows():
                tokens = row.get("tokens")
                step = row.get("step")
                if tokens is None or pd.isna(tokens):
                    continue

                # D/N = tokens / original params (token schedule is based on original count).
                raw_dn = tokens / raw_num_params if raw_num_params else 0
                dn_ratio = min(_ALL_DN, key=lambda c: abs(c - raw_dn)) if raw_dn > 0 else 0
                if dn_ratio not in _KEEP_DN:
                    print(
                        f"    SKIP {size}: step={step}, "
                        f"tokens={tokens:.3e}, raw_dn={raw_dn:.2f}, "
                        f"snapped={dn_ratio} (not in _KEEP_DN)"
                    )
                    continue
                print(
                    f"    KEEP {size}: step={step}, "
                    f"tokens={tokens:.3e}, raw_dn={raw_dn:.2f}, "
                    f"snapped={dn_ratio}"
                )
                checkpoint_key = f"{size}@{dn_ratio}"

                # Build a single-row DataFrame for aggregation
                row_df = pd.DataFrame([row])
                row_df["step"] = row.get("step", 0)

                cluster_results = aggregate_all_clusters(row_df)

                # Get LM metrics
                lm_cols = find_lm_columns(row_df)
                lm_metrics: Dict[str, Dict[str, float]] = {}
                for task, (ce_col, ppl_col) in lm_cols.items():
                    lm_metrics[task] = {}
                    if ce_col and pd.notna(row.get(ce_col)):
                        lm_metrics[task]["CE loss"] = float(row[ce_col])
                    if ppl_col and pd.notna(row.get(ppl_col)):
                        lm_metrics[task]["PPL"] = float(row[ppl_col])

                results[checkpoint_key] = {
                    "base_easy": cluster_results["base_easy"],
                    "base_main": cluster_results["base_main"],
                    "heldout": cluster_results["heldout"],
                    "lm_metrics": lm_metrics,
                    "raw": {},
                    "step": int(step) if step is not None and not pd.isna(step) else 0,
                    "tokens": int(tokens),
                    "num_params": num_params,
                    "size_label": size,
                }
        else:
            final_row = df[df["step"] == df["step"].max()].iloc[0]
            final_step = int(final_row.get("step", 0))
            final_tokens = int(final_row.get("tokens", 0))

            # Aggregate all OlmoBaseEval clusters
            cluster_results = aggregate_all_clusters(df)

            # Get LM metrics
            lm_cols = find_lm_columns(df)
            lm_metrics_final: Dict[str, Dict[str, float]] = {}
            for task, (ce_col, ppl_col) in lm_cols.items():
                lm_metrics_final[task] = {}
                if ce_col and pd.notna(final_row.get(ce_col)):
                    lm_metrics_final[task]["CE loss"] = float(final_row[ce_col])
                if ppl_col and pd.notna(final_row.get(ppl_col)):
                    lm_metrics_final[task]["PPL"] = float(final_row[ppl_col])

            # Store raw metrics
            metric_cols = [
                c for c in df.columns if c not in ["name", "step", "tokens", "size", "num_params"]
            ]
            raw_metrics = {
                col: final_row[col] for col in metric_cols if pd.notna(final_row.get(col))
            }

            results[size] = {
                "base_easy": cluster_results["base_easy"],
                "base_main": cluster_results["base_main"],
                "heldout": cluster_results["heldout"],
                "lm_metrics": lm_metrics_final,
                "raw": raw_metrics,
                "step": final_step,
                "tokens": final_tokens,
                "num_params": num_params,
                "size_label": size,
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
    """Sort dataframe by model size. Handles checkpoint keys like '190M@20'."""
    df = df.copy()
    df["_size_order"] = df["Size"].map(lambda x: SIZE_ORDER.get(get_size_label(x), 99))
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
            if sizes and get_size_label(size) not in sizes:
                continue

            row: Dict[str, Any] = {
                "Ladder": get_display_name(ladder_name),
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
            if sizes and get_size_label(size) not in sizes:
                continue

            row: Dict[str, Any] = {
                "Ladder": get_display_name(ladder_name),
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
            if sizes and get_size_label(size) not in sizes:
                continue

            row: Dict[str, Any] = {
                "Ladder": get_display_name(ladder_name),
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
            if sizes and get_size_label(size) not in sizes:
                continue

            row: Dict[str, Any] = {
                "Ladder": get_display_name(ladder_name),
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


def get_size_label(size_key: str) -> str:
    """Extract the human-readable size label from a key like '190M' or '190M@20'."""
    return size_key.split("@")[0]


def get_size_numeric(size: str) -> float:
    """Convert size string to numeric value for x-axis positioning.

    Handles both plain sizes ('190M') and checkpoint keys ('190M@20').
    """
    base_size = size.split("@")[0]
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
    return size_to_num.get(base_size, 0)


def shade_color(hex_color: str, fraction: float) -> Tuple[float, float, float]:
    """Blend a hex color with white. fraction=0 -> white, fraction=1 -> original color."""
    r = int(hex_color[1:3], 16) / 255.0
    g = int(hex_color[3:5], 16) / 255.0
    b = int(hex_color[5:7], 16) / 255.0
    return (1 - fraction + fraction * r, 1 - fraction + fraction * g, 1 - fraction + fraction * b)


def _get_dn_ratio(size_key: str) -> Optional[int]:
    """Extract D/N ratio from a checkpoint key like '190M@20'. Returns None for plain keys."""
    if "@" in size_key:
        try:
            return int(size_key.split("@")[1])
        except (ValueError, IndexError):
            pass
    return None


def _has_checkpoints(size_results: Dict[str, Dict[str, Any]]) -> bool:
    """Check if any key in the results uses the checkpoint format (size@ratio)."""
    return any("@" in k for k in size_results.keys())


def _get_x_value(size_key: str, data: Dict[str, Any]) -> float:
    """Get x-axis value: use corrected num_params if available, else size bucket."""
    num_params = data.get("num_params", 0)
    if num_params and num_params > 0:
        return float(num_params) / 1e6  # In millions
    return get_size_numeric(size_key)


def _get_x_label(size_key: str, data: Dict[str, Any]) -> str:
    """Get human-readable x-axis label."""
    return data.get("size_label", get_size_label(size_key))


def _plot_ladder_on_axis(
    ax,
    size_results: Dict[str, Dict[str, Any]],
    extract_y,
    ladder_name: str,
    color: str,
    sizes: Optional[List[str]] = None,
    added_dn_labels: Optional[set] = None,
    use_final_checkpoint: bool = False,
) -> None:
    """
    Plot a single ladder's data on an axis.

    When the data contains checkpoint keys (size@ratio), draws:
      - Scatter points gradient-colored by Chinchilla multiple (D/N ratio),
        with lighter shades for smaller D/N and full color for the largest.
      - A single curve (line) through the mean of each model size's checkpoints.

    When the data is final-checkpoint-only, draws the classic "o-" line plot.

    Args:
        ax: Matplotlib axis.
        size_results: The ladder's {size_key: data} dict.
        extract_y: Callable(data) -> Optional[float] that pulls the y value.
        ladder_name: Label for the legend.
        color: Hex colour string for this ladder.
        sizes: Optional filter on base size labels.
        added_dn_labels: Mutable set tracking which D/N legend entries have
            already been added (shared across ladders on the same axis).
        use_final_checkpoint: If True, draw the curve through the final
            (highest D/N) checkpoint per model size instead of the mean.
    """
    from matplotlib.lines import Line2D

    multi = _has_checkpoints(size_results)

    if not multi:
        # Simple case: one point per model size -> line plot
        points = []
        for size_key, data in size_results.items():
            if sizes and get_size_label(size_key) not in sizes:
                continue
            y = extract_y(data)
            if y is not None:
                points.append((_get_x_value(size_key, data), y))
        if points:
            points.sort(key=lambda p: p[0])
            x_vals, y_vals = zip(*points)
            ax.plot(
                x_vals,
                y_vals,
                "o-",
                label=ladder_name,
                color=color,
                linewidth=2,
                markersize=6,
            )
        return

    # ------------------------------------------------------------------
    # Multi-checkpoint case
    # ------------------------------------------------------------------

    # Collect all points with their D/N ratio
    # points_by_size: {base_size_label: [(x, y, dn_ratio), ...]}
    points_by_size: Dict[str, List[Tuple[float, float, int]]] = {}
    all_dn_ratios: set = set()

    for size_key, data in size_results.items():
        if sizes and get_size_label(size_key) not in sizes:
            continue
        y = extract_y(data)
        if y is None:
            continue
        dn = _get_dn_ratio(size_key)
        if dn is None:
            dn = 0
        x_val = _get_x_value(size_key, data)
        label = _get_x_label(size_key, data)
        points_by_size.setdefault(label, []).append((x_val, y, dn))
        all_dn_ratios.add(dn)

    if not points_by_size:
        return

    sorted_dn = sorted(all_dn_ratios)
    dn_min, dn_max = sorted_dn[0], sorted_dn[-1]
    dn_span = max(dn_max - dn_min, 1)

    # Scatter all checkpoints, gradient-coloured by D/N
    for _label, pts in points_by_size.items():
        for x_val, y_val, dn in pts:
            frac = 0.25 + 0.75 * (dn - dn_min) / dn_span
            pt_color = shade_color(color, frac)
            ax.scatter(
                x_val,
                y_val,
                color=pt_color,
                s=30,
                alpha=0.7,
                edgecolors="white",
                linewidth=0.3,
                zorder=4,
            )

    # Curve through each model size's checkpoints (mean or final)
    curve_points = []
    for _label, pts in points_by_size.items():
        if use_final_checkpoint:
            # Use the checkpoint with the highest D/N ratio (final)
            final_pt = max(pts, key=lambda p: p[2])
            curve_points.append((final_pt[0], final_pt[1]))
        else:
            mean_x = sum(p[0] for p in pts) / len(pts)
            mean_y = sum(p[1] for p in pts) / len(pts)
            curve_points.append((mean_x, mean_y))
    curve_points.sort(key=lambda p: p[0])

    if curve_points:
        cx, cy = zip(*curve_points)
        ax.plot(
            cx,
            cy,
            color=color,
            linewidth=2.5,
            alpha=0.8,
            zorder=5,
            label=ladder_name,
        )

    # Add gradient legend entries (shared across ladders on the same axis)
    if added_dn_labels is None:
        added_dn_labels = set()

    # Pick a few representative D/N values for the legend
    if len(sorted_dn) <= 5:
        legend_dns = sorted_dn
    else:
        step = max(1, len(sorted_dn) // 4)
        legend_dns = sorted_dn[::step]
        if sorted_dn[-1] not in legend_dns:
            legend_dns.append(sorted_dn[-1])

    dn_handles = []
    for dn in legend_dns:
        if dn in added_dn_labels:
            continue
        added_dn_labels.add(dn)
        frac = 0.25 + 0.75 * (dn - dn_min) / dn_span
        dn_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=shade_color("#555555", frac),
                markersize=7,
                label=f"D/N={dn}",
            )
        )

    if dn_handles:
        # Add a secondary legend for D/N gradient
        existing_legend = ax.get_legend()
        if existing_legend is None:
            # Will be added after all ladders are plotted
            pass
        ax._dn_legend_handles = getattr(ax, "_dn_legend_handles", []) + dn_handles


def _finalize_legends(ax, has_checkpoints: bool) -> None:
    """Add primary + D/N gradient legends to an axis if checkpoint data was plotted."""
    dn_handles = getattr(ax, "_dn_legend_handles", [])
    if has_checkpoints and dn_handles:
        # Primary legend (ladder names) is already set; add secondary for D/N
        first_legend = ax.legend(loc="best", fontsize=8)
        ax.add_artist(first_legend)
        ax.legend(
            handles=dn_handles,
            loc="lower left",
            fontsize=7,
            framealpha=0.9,
            title="Chinchilla mult.",
            title_fontsize=7,
        )
    else:
        ax.legend(loc="best", fontsize=8)


def plot_base_main_metrics(
    ladder_results: Dict[str, Dict[str, Dict[str, Any]]],
    sizes: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    show: bool = True,
    use_final_checkpoint: bool = False,
    star_ladder: Optional[str] = None,
) -> None:
    """
    Plot Base Main Suite metrics comparison across ladders.
    Matches OLMo 3 paper Figure 29 style.
    Uses corrected num_params for x-axis positioning when available.
    With --all-checkpoints: scatter + gradient by D/N ratio, curve through final checkpoints.
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

    any_checkpoints = any(_has_checkpoints(sr) for sr in ladder_results.values())

    for idx, cluster in enumerate(clusters):
        ax = axes[idx]
        dn_labels: set = set()

        def _extract(data, _c=cluster):
            bm = data.get("base_main", {})
            if _c in bm:
                return bm[_c]["avg"] * 100
            return None

        for ladder_name in ladder_names:
            _plot_ladder_on_axis(
                ax,
                ladder_results[ladder_name],
                _extract,
                _starred_display_name(ladder_name, star_ladder),
                colors[ladder_name],
                sizes,
                added_dn_labels=dn_labels,
                use_final_checkpoint=use_final_checkpoint,
            )

        ax.set_xlabel("Non-embedding Parameters (M)", fontsize=10)
        ax.set_ylabel("Accuracy (%)" if cluster != "FIM" else "pass@1 (%)", fontsize=10)
        ax.set_title(f"Base Main {cluster}", fontsize=11, fontweight="bold")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            _finalize_legends(ax, any_checkpoints)

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
    use_final_checkpoint: bool = False,
    star_ladder: Optional[str] = None,
) -> None:
    """
    Plot Base Easy Suite metrics (BPB) comparison across ladders.
    Matches OLMo 3 paper Figure 30 style (bottom row).
    Uses corrected num_params for x-axis positioning when available.
    With --all-checkpoints: scatter + gradient by D/N ratio, curve through final checkpoints.
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

    any_checkpoints = any(_has_checkpoints(sr) for sr in ladder_results.values())

    for idx, cluster in enumerate(clusters):
        ax = axes_list[idx]
        dn_labels: set = set()

        def _extract(data, _c=cluster):
            be = data.get("base_easy", {})
            if _c in be:
                return be[_c]["avg"]
            return None

        for ladder_name in ladder_names:
            _plot_ladder_on_axis(
                ax,
                ladder_results[ladder_name],
                _extract,
                _starred_display_name(ladder_name, star_ladder),
                colors[ladder_name],
                sizes,
                added_dn_labels=dn_labels,
                use_final_checkpoint=use_final_checkpoint,
            )

        ax.set_xlabel("Non-embedding Parameters (M)", fontsize=10)
        ax.set_ylabel("Bits-per-byte", fontsize=10)
        ax.set_title(f"Base Easy {cluster.replace('_BPB', '')}", fontsize=11, fontweight="bold")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            _finalize_legends(ax, any_checkpoints)

    fig.suptitle(
        "OlmoBaseEval Base Easy Suite (BPB - lower is better)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
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
    star_ladder: Optional[str] = None,
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

            for size_key, data in size_results.items():
                if sizes and get_size_label(size_key) not in sizes:
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
                    label=_starred_display_name(ladder_name, star_ladder),
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
        if ax_idx == 0:
            ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        "Scaling Analysis: Easy Suite BPB vs Main Suite Accuracy",
        fontsize=14,
        fontweight="bold",
        y=1.02,
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
    include_main: bool = False,
    use_final_checkpoint: bool = False,
    star_ladder: Optional[str] = None,
) -> None:
    """
    Create a comprehensive summary comparison figure.
    Matches OLMo 3 paper Table 2/3 style visualization.

    When include_main=False, panels 1 & 2 show Easy Suite metrics instead of Main Suite,
    and the relative comparison (panel 4) uses Easy Suite BPB.
    """
    if not PLOTTING_AVAILABLE:
        print("Warning: matplotlib not available, skipping plots")
        return

    ladder_names = list(ladder_results.keys())
    colors = get_ladder_colors(ladder_names)

    # Collect unique size labels across all ladders (use human-readable labels for bar charts)
    all_size_labels = sorted(
        set(
            _get_x_label(size_key, data)
            for size_results in ladder_results.values()
            for size_key, data in size_results.items()
            if not sizes or get_size_label(size_key) in sizes
        ),
        key=get_size_numeric,
    )

    if not all_size_labels:
        print("No sizes to plot")
        return

    n_rows = 2 if include_main else 1
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Helper: find data for a given size label within a ladder's results
    def _find_by_label(
        size_results: Dict[str, Dict[str, Any]], label: str
    ) -> Optional[Dict[str, Any]]:
        """Find the data entry matching a human-readable size label.

        When use_final_checkpoint=True, picks the checkpoint with the most tokens.
        Otherwise, averages the suite metrics across all matching checkpoints.
        """
        matches = [
            (size_key, data)
            for size_key, data in size_results.items()
            if _get_x_label(size_key, data) == label
        ]
        if not matches:
            return None
        if len(matches) == 1 or use_final_checkpoint:
            # Pick the one with most tokens (final)
            return max(matches, key=lambda m: m[1].get("tokens", 0))[1]
        # Average suite metrics across all checkpoints for this size
        averaged: Dict[str, Any] = dict(matches[0][1])  # shallow copy from first match
        for suite_key in ("base_easy", "base_main", "heldout"):
            # Collect cluster averages across all checkpoints
            all_cluster_names: set = set()
            for _, data in matches:
                all_cluster_names.update(data.get(suite_key, {}).keys())
            if not all_cluster_names:
                continue
            merged_suite: Dict[str, Dict[str, Any]] = {}
            for cluster in all_cluster_names:
                vals = [
                    data[suite_key][cluster]["avg"]
                    for _, data in matches
                    if cluster in data.get(suite_key, {})
                ]
                if vals:
                    merged_suite[cluster] = {"avg": sum(vals) / len(vals)}
            averaged[suite_key] = merged_suite
        return averaged

    x = list(range(len(all_size_labels)))
    width = 0.8 / len(ladder_names)

    if include_main:
        # Panel 1: Base Main clusters average across sizes
        ax = axes[0, 0]

        for i, ladder_name in enumerate(ladder_names):
            size_results = ladder_results[ladder_name]
            y_vals = []
            for label in all_size_labels:
                data = _find_by_label(size_results, label)
                if data:
                    base_main = data.get("base_main", {})
                    if base_main:
                        y_vals.append(
                            sum(c["avg"] for c in base_main.values()) / len(base_main) * 100
                        )
                    else:
                        y_vals.append(0)
                else:
                    y_vals.append(0)

            offset = (i - len(ladder_names) / 2 + 0.5) * width
            ax.bar(
                [xi + offset for xi in x],
                y_vals,
                width,
                label=_starred_display_name(ladder_name, star_ladder),
                color=colors[ladder_name],
            )

        ax.set_xlabel("Model Size", fontsize=11)
        ax.set_ylabel("Average Accuracy (%)", fontsize=11)
        ax.set_title("OlmoBaseEval Main Suite Average", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(all_size_labels, fontsize=10)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

        # Panel 2: Cluster breakdown at largest size
        ax = axes[0, 1]
        main_clusters = list(BASE_MAIN_SUITE.keys())
        largest_label = all_size_labels[-1] if all_size_labels else None

        if largest_label:
            x_clusters = list(range(len(main_clusters)))
            for i, ladder_name in enumerate(ladder_names):
                size_results = ladder_results[ladder_name]
                data = _find_by_label(size_results, largest_label)
                y_vals = []
                if data:
                    base_main = data.get("base_main", {})
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
                    label=_starred_display_name(ladder_name, star_ladder),
                    color=colors[ladder_name],
                )

            ax.set_xlabel("Metric Cluster", fontsize=11)
            ax.set_ylabel("Accuracy (%)", fontsize=11)
            ax.set_title(f"Main Suite Breakdown at {largest_label}", fontsize=12, fontweight="bold")
            ax.set_xticks(x_clusters)
            ax.set_xticklabels(main_clusters, fontsize=9, rotation=30, ha="right")
            ax.grid(True, alpha=0.3, axis="y")

        # Easy suite and relative comparison go in row 1
        easy_row = 1
    else:
        easy_row = 0

    # Base Easy BPB comparison
    ax = axes[easy_row, 0]

    for i, ladder_name in enumerate(ladder_names):
        size_results = ladder_results[ladder_name]
        y_vals = []
        for label in all_size_labels:
            data = _find_by_label(size_results, label)
            if data:
                base_easy = data.get("base_easy", {})
                if base_easy:
                    y_vals.append(sum(c["avg"] for c in base_easy.values()) / len(base_easy))
                else:
                    y_vals.append(0)
            else:
                y_vals.append(0)

        offset = (i - len(ladder_names) / 2 + 0.5) * width
        ax.bar(
            [xi + offset for xi in x],
            y_vals,
            width,
            label=_starred_display_name(ladder_name, star_ladder),
            color=colors[ladder_name],
        )

    ax.set_xlabel("Model Size", fontsize=11)
    ax.set_ylabel("Average BPB", fontsize=11)
    ax.set_title(
        "OlmoBaseEval Easy Suite Average (lower is better)", fontsize=12, fontweight="bold"
    )
    ax.set_xticks(list(range(len(all_size_labels))))
    ax.set_xticklabels(all_size_labels, fontsize=10)
    if not include_main:
        ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Relative performance comparison
    ax = axes[easy_row, 1]
    if len(ladder_names) > 1:
        baseline_ladder = ladder_names[0]
        x = list(range(len(all_size_labels)))

        # Choose suite for relative comparison based on include_main flag
        suite_key = "base_main" if include_main else "base_easy"

        for i, ladder_name in enumerate(ladder_names[1:], start=1):
            y_vals = []
            for label in all_size_labels:
                baseline_data = _find_by_label(ladder_results[baseline_ladder], label)
                compare_data = _find_by_label(ladder_results[ladder_name], label)

                baseline_suite = baseline_data.get(suite_key, {}) if baseline_data else {}
                compare_suite = compare_data.get(suite_key, {}) if compare_data else {}

                if baseline_suite and compare_suite:
                    baseline_avg = sum(c["avg"] for c in baseline_suite.values()) / len(
                        baseline_suite
                    )
                    compare_avg = sum(c["avg"] for c in compare_suite.values()) / len(compare_suite)
                    if include_main:
                        # For accuracy: relative difference as percentage
                        if baseline_avg > 0:
                            diff = ((compare_avg - baseline_avg) / baseline_avg) * 100
                        else:
                            diff = 0
                    else:
                        # For BPB: percentage difference (compare - baseline) / baseline * 100
                        # More negative = better (lower BPB is better)
                        if baseline_avg > 0:
                            diff = ((compare_avg - baseline_avg) / baseline_avg) * 100
                        else:
                            diff = 0
                    y_vals.append(diff)
                else:
                    y_vals.append(0)

            ax.bar(
                [xi + (i - 1) * width for xi in x],
                y_vals,
                width,
                label=f"{_starred_display_name(ladder_name, star_ladder)} vs {_starred_display_name(baseline_ladder, star_ladder)}",
                color=colors[ladder_name],
            )

        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xlabel("Model Size", fontsize=11)
        if include_main:
            ax.set_ylabel("Relative Accuracy Difference (%)", fontsize=11)
            ax.set_title(
                f"Main Suite Accuracy Relative to {get_display_name(baseline_ladder)}",
                fontsize=12,
                fontweight="bold",
            )
        else:
            ax.set_ylabel("BPB Difference (%, more negative = better)", fontsize=11)
            ax.set_title(
                f"Easy Suite BPB % Difference vs {get_display_name(baseline_ladder)}",
                fontsize=12,
                fontweight="bold",
            )
        ax.set_xticks(list(range(len(all_size_labels))))
        ax.set_xticklabels(all_size_labels, fontsize=10)
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
    use_final_checkpoint: bool = False,
    star_ladder: Optional[str] = None,
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

    any_checkpoints = any(_has_checkpoints(sr) for sr in ladder_results.values())

    # Plot average PPL first
    ax = axes_list[0]
    dn_labels: set = set()

    def _extract_avg_ppl(data):
        lm = data.get("lm_metrics", {})
        ppls = [v.get("PPL") for v in lm.values() if v.get("PPL") is not None]
        if ppls:
            return sum(ppls) / len(ppls)
        return None

    for ladder_name in ladder_names:
        _plot_ladder_on_axis(
            ax,
            ladder_results[ladder_name],
            _extract_avg_ppl,
            _starred_display_name(ladder_name, star_ladder),
            colors[ladder_name],
            sizes,
            added_dn_labels=dn_labels,
            use_final_checkpoint=use_final_checkpoint,
        )

    ax.set_xlabel("Non-embedding Parameters (M)", fontsize=10)
    ax.set_ylabel("Perplexity", fontsize=10)
    ax.set_title("Average PPL", fontsize=12, fontweight="bold")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    _finalize_legends(ax, any_checkpoints)

    # Plot individual task PPLs
    for idx, task in enumerate(tasks_to_plot, start=1):
        if idx >= len(axes_list):
            break
        ax = axes_list[idx]
        dn_labels_task: set = set()

        def _extract_task_ppl(data, _t=task):
            lm = data.get("lm_metrics", {})
            if _t in lm and "PPL" in lm[_t]:
                return lm[_t]["PPL"]
            return None

        for ladder_name in ladder_names:
            _plot_ladder_on_axis(
                ax,
                ladder_results[ladder_name],
                _extract_task_ppl,
                _starred_display_name(ladder_name, star_ladder),
                colors[ladder_name],
                sizes,
                added_dn_labels=dn_labels_task,
                use_final_checkpoint=use_final_checkpoint,
            )

        ax.set_xlabel("Non-embedding Parameters (M)", fontsize=10)
        ax.set_ylabel("Perplexity", fontsize=10)
        ax.set_title(f"{task}", fontsize=12, fontweight="bold")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

    for idx in range(n_tasks, len(axes_list)):
        axes_list[idx].set_visible(False)

    fig.suptitle(
        "Language Modeling - Perplexity (lower is better)", fontsize=14, fontweight="bold", y=1.02
    )
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
    include_main: bool = False,
    use_final_checkpoint: bool = False,
    star_ladder: Optional[str] = None,
) -> None:
    """Generate all OlmoBaseEval-style plots."""
    if not PLOTTING_AVAILABLE:
        print("\nError: matplotlib is not installed. Install it with: pip install matplotlib")
        return

    print("\nGenerating OlmoBaseEval plots...")
    if include_main:
        plot_base_main_metrics(
            ladder_results, sizes, output_path, show, use_final_checkpoint=use_final_checkpoint,
            star_ladder=star_ladder,
        )
    plot_base_easy_metrics(
        ladder_results, sizes, output_path, show, use_final_checkpoint=use_final_checkpoint,
        star_ladder=star_ladder,
    )
    if include_main:
        plot_scaling_analysis(ladder_results, sizes, output_path, show, star_ladder=star_ladder)
    plot_summary_comparison(
        ladder_results,
        sizes,
        output_path,
        show,
        include_main=include_main,
        use_final_checkpoint=use_final_checkpoint,
        star_ladder=star_ladder,
    )
    plot_lm_metrics(
        ladder_results, sizes, output_path, show, use_final_checkpoint=use_final_checkpoint,
        star_ladder=star_ladder,
    )
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
    include_main: bool = False,
) -> str:
    """Generate LaTeX tables comparing all ladders in OlmoBaseEval format."""
    easy_df = create_base_easy_table(ladder_results, sizes)
    lm_df = create_lm_table(ladder_results, sizes)

    latex_parts = []
    latex_parts.append("% OlmoBaseEval LaTeX tables generated by ladder_metrics_analysis.py")
    latex_parts.append("% Requires: \\usepackage{booktabs}")
    latex_parts.append("")

    # Base Main Suite
    if include_main:
        main_df = create_base_main_table(ladder_results, sizes)
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
    if include_main:
        heldout_df = create_heldout_table(ladder_results, sizes)
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
# Ablation-style LaTeX Table (Architecture Comparison)
# =============================================================================

# Metadata for the ablation table: group, display name, attention %, row order.
# The keys must match the ladder names used in --compare.
ABLATION_ARCHITECTURES: List[Dict[str, Any]] = [
    # --- Pure Architectures ---
    {
        "key": "olmo3",
        "aliases": ["olmo3-1", "olmo3-2", "olmo3-3"],
        "display": "Olmo 3 (Transformer)",
        "attn_pct": "100\\%",
        "group": "Pure Architectures",
    },
    {
        "key": "pure-gdn",
        "aliases": [],
        "display": "Pure GatedDeltaNet",
        "attn_pct": "0\\%",
        "group": "Pure Architectures",
    },
    {
        "key": "pure-mamba",
        "aliases": [],
        "display": "Pure Mamba2",
        "attn_pct": "0\\%",
        "group": "Pure Architectures",
    },
    # --- Hybrid: Interleaved Attention ---
    {
        "key": "hybrid-gdn-half",
        "aliases": [],
        "display": "GatedDeltaNet + 1/2 Attn",
        "attn_pct": "50\\%",
        "group": "Hybrid: Interleaved Attention",
    },
    {
        "key": "hybrid-gdn",
        "aliases": [],
        "display": "GatedDeltaNet + 1/4 Attn",
        "attn_pct": "25\\%",
        "group": "Hybrid: Interleaved Attention",
        "star": True,
    },
    {
        "key": "hybrid-gdn-eight",
        "aliases": [],
        "display": "GatedDeltaNet + 1/8 Attn",
        "attn_pct": "12.5\\%",
        "group": "Hybrid: Interleaved Attention",
    },
    {
        "key": "hybrid-mamba",
        "aliases": [],
        "display": "Mamba2 + 1/4 Attn",
        "attn_pct": "25\\%",
        "group": "Hybrid: Interleaved Attention",
    },
    # --- Hybrid: Middle Placement ---
    {
        "key": "hybrid-gdn-middle",
        "aliases": [],
        "display": "GatedDeltaNet + Middle Attn",
        "attn_pct": "25\\%",
        "group": "Hybrid: Middle Placement",
    },
    {
        "key": "hybrid-gdn-middle-no-final",
        "aliases": [],
        "display": "GatedDeltaNet + Middle Attn (no final A)",
        "attn_pct": "25\\%",
        "group": "Hybrid: Middle Placement",
    },
]


def generate_ablation_latex_table(
    ladder_results: Dict[str, Dict[str, Dict[str, Any]]],
    sizes: Optional[List[str]] = None,
    star_ladder: Optional[str] = None,
    metric_clusters: Optional[List[str]] = None,
) -> str:
    """
    Generate a publication-ready ablation table comparing architectures across scales.

    Layout: architectures as rows (grouped), model sizes as column groups,
    with sub-columns for each metric cluster plus an average.

    Args:
        ladder_results: {ladder_name: {size_key: {base_easy/base_main/...}}}
        sizes: Which sizes to include as column groups (e.g. ["190M", "370M", "760M"]).
            If None, auto-detects from data.
        star_ladder: Ladder name to mark with a star (e.g. "hybrid-gdn").
        metric_clusters: Which Base Easy clusters to show (default: all three BPB clusters).
    """
    if metric_clusters is None:
        metric_clusters = ["Math_BPB", "Code_BPB"]  # Math, Code (QA excluded to match paper table)

    # Short display names for sub-columns
    cluster_short = {
        "Math_BPB": "Math",
        "Code_BPB": "Code",
        "QA_BPB": "QA",
        "Math": "Math",
        "Code": "Code",
        "FIM": "FIM",
        "MC_STEM": "MC\\_ST",
        "MC_NonSTEM": "MC\\_NS",
        "GenQA": "GenQA",
    }

    # Determine which sizes to show
    if sizes:
        show_sizes = sizes
    else:
        all_sizes_seen: set = set()
        for size_results in ladder_results.values():
            for sk in size_results:
                all_sizes_seen.add(get_size_label(sk))
        show_sizes = sorted(all_sizes_seen, key=lambda s: SIZE_ORDER.get(s, 99))

    n_metrics = len(metric_clusters)
    n_sub = n_metrics + 1  # +1 for Avg column
    n_sizes = len(show_sizes)

    # Build architecture rows in order, matching available ladder_results
    arch_rows: List[Dict[str, Any]] = []
    for arch in ABLATION_ARCHITECTURES:
        # Find matching ladder name in results
        matched_key = None
        all_keys = [arch["key"]] + arch.get("aliases", [])
        for k in all_keys:
            if k in ladder_results:
                matched_key = k
                break
        arch_rows.append(
            {
                **arch,
                "matched_key": matched_key,
                "star": arch.get("star", False) or (star_ladder and arch["key"] == star_ladder),
            }
        )

    # Filter out architectures with no data
    arch_rows = [a for a in arch_rows if a["matched_key"] is not None]

    if not arch_rows:
        return "% No matching architectures found in results"

    # Determine if we're using base_easy (BPB, lower is better) or base_main
    suite_key = "base_easy"
    higher_is_better = False
    for mc in metric_clusters:
        if mc in BASE_MAIN_SUITE:
            suite_key = "base_main"
            higher_is_better = True
            break

    format_pct = higher_is_better

    # Collect all values for best/second-best highlighting per size
    # values_by_size_metric[size][metric_idx] = [(arch_idx, value), ...]
    values_by_size_metric: Dict[str, Dict[int, List[Tuple[int, float]]]] = {
        s: {i: [] for i in range(n_sub)} for s in show_sizes
    }
    # Also collect the actual values for rendering
    # cell_values[arch_idx][size][metric_idx] = float | None
    cell_values: List[Dict[str, Dict[int, Optional[float]]]] = []

    for arch_idx, arch in enumerate(arch_rows):
        size_data: Dict[str, Dict[int, Optional[float]]] = {}
        mk = arch["matched_key"]
        results = ladder_results[mk] if mk else {}

        for size_label in show_sizes:
            metrics: Dict[int, Optional[float]] = {}
            # Find data for this size
            data = None
            for sk, sd in results.items():
                if get_size_label(sk) == size_label:
                    data = sd
                    break

            vals = []
            for mi, cluster in enumerate(metric_clusters):
                val = None
                if data:
                    suite = data.get(suite_key, {})
                    if cluster in suite:
                        val = suite[cluster]["avg"]
                metrics[mi] = val
                if val is not None:
                    vals.append(val)
                    values_by_size_metric[size_label][mi].append((arch_idx, val))

            # Average
            if vals:
                avg = sum(vals) / len(vals)
                metrics[n_metrics] = avg
                values_by_size_metric[size_label][n_metrics].append((arch_idx, avg))
            else:
                metrics[n_metrics] = None

            size_data[size_label] = metrics
        cell_values.append(size_data)

    # Find best and second-best per size per metric
    best_idx: Dict[str, Dict[int, Optional[int]]] = {s: {} for s in show_sizes}
    second_idx: Dict[str, Dict[int, Optional[int]]] = {s: {} for s in show_sizes}
    for size_label in show_sizes:
        for mi in range(n_sub):
            entries = values_by_size_metric[size_label][mi]
            if not entries:
                best_idx[size_label][mi] = None
                second_idx[size_label][mi] = None
                continue
            if higher_is_better:
                sorted_entries = sorted(entries, key=lambda e: e[1], reverse=True)
            else:
                sorted_entries = sorted(entries, key=lambda e: e[1])
            best_idx[size_label][mi] = sorted_entries[0][0]
            if len(sorted_entries) > 1:
                second_idx[size_label][mi] = sorted_entries[1][0]
            else:
                second_idx[size_label][mi] = None

    # --- Build LaTeX ---
    def _escape(s: str) -> str:
        return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")

    lines: List[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")

    suite_label = "\\olmothreeeval" if suite_key == "base_easy" else "Base Main"
    lines.append(r"\caption{")
    lines.append(r"    \textbf{Architecture ablation results.}")
    lines.append(
        f"    Average {suite_label} score for pure and hybrid architectures at {'three' if n_sizes == 3 else str(n_sizes)} representative scales."
    )
    lines.append(
        r"    All models are trained with identical data and hyperparameters at each scale "
        r"(see \Cref{sec:scaling-laws} for methodology)."
    )
    lines.append(
        r"    \textbf{Bold} indicates best performance per scale; "
        r"\underline{underline} indicates second best."
    )
    lines.append(r"    $\bigstar$ marks our selected architecture.")
    lines.append(r"}")
    lines.append(r"\label{tab:ablation-evals}")
    lines.append(r"\small")
    lines.append(r"\resizebox{\textwidth}{!}{%")

    # Column spec: l c | (n_sub columns) per size
    col_spec = "@{}l c" + (" " + "c" * n_sub) * n_sizes + "@{}"
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header row 1: size names spanning sub-columns
    h1_parts = ["", ""]
    for size_label in show_sizes:
        h1_parts.append(r"\multicolumn{" + str(n_sub) + r"}{c}{\textbf{" + size_label + r"}}")
    lines.append(" & ".join(h1_parts) + r" \\")

    # cmidrule for each size group
    cmidrules = []
    col_start = 3  # first two are Architecture and Attn %
    for _ in show_sizes:
        col_end = col_start + n_sub - 1
        cmidrules.append(f"\\cmidrule(lr){{{col_start}-{col_end}}}")
        col_start = col_end + 1
    lines.append(" ".join(cmidrules))

    # Header row 2: sub-column names
    h2_parts = [r"\textbf{Architecture}", r"\textbf{Attn \%}"]
    for _ in show_sizes:
        for cluster in metric_clusters:
            h2_parts.append(cluster_short.get(cluster, _escape(cluster)))
        h2_parts.append("Avg")
    lines.append(" & ".join(h2_parts) + r" \\")
    lines.append(r"\midrule")

    # Body rows
    prev_group = None
    for arch_idx, arch in enumerate(arch_rows):
        group = arch["group"]
        if group != prev_group:
            if prev_group is not None:
                lines.append(r"\midrule")
            lines.append(
                r"\multicolumn{" + str(2 + n_sub * n_sizes) + r"}{l}{\textit{" + group + r"}} \\"
            )
            prev_group = group

        # Architecture name
        display = arch["display"]
        if arch["star"]:
            display += r"$^\bigstar$"
        row_parts = [r"\quad " + display, arch["attn_pct"]]

        # Values per size
        for size_label in show_sizes:
            for mi in range(n_sub):
                val = cell_values[arch_idx].get(size_label, {}).get(mi)
                if val is not None:
                    if format_pct:
                        formatted = f"{val * 100:.1f}"
                    else:
                        formatted = f"{val:.4f}"
                    if best_idx[size_label].get(mi) == arch_idx:
                        formatted = r"\textbf{" + formatted + "}"
                    elif second_idx[size_label].get(mi) == arch_idx:
                        formatted = r"\underline{" + formatted + "}"
                    row_parts.append(formatted)
                else:
                    row_parts.append("--")

        lines.append(" & ".join(row_parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append("")
    lines.append(r"\vspace{0.5em}")
    lines.append(r"{\footnotesize")
    lines.append(
        r"\textit{Note:} ``Interleaved'' places attention layers at regular intervals "
        r"(e.g., 1/4 = every 4th layer)."
    )
    lines.append(
        r"``Middle'' concentrates attention layers in the middle of the network; ``no final A'' omits the final attention layer."
    )
    lines.append(
        r"All models at each scale share the same number of total layers, "
        r"with layer types substituted according to the configuration."
    )
    lines.append(r"}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


# =============================================================================
# LaTeX/TikZ Plot Generation (pgfplots)
# =============================================================================

# Color palette for LaTeX plots - semantically grouped, colorblind-friendly.
# Each entry: (pgfplots color definition string, mark style, line style)
LATEX_PLOT_STYLES: Dict[str, Dict[str, str]] = {
    # Pure architectures — solid lines, filled marks
    "olmo3": {
        "color_def": "\\definecolor{clrTransformer}{HTML}{2563EB}",
        "color_name": "clrTransformer",
        "mark": "square*",
        "mark_options": "fill=clrTransformer",
        "line_style": "",
    },
    "olmo3-1": {
        "color_def": "\\definecolor{clrTransformer}{HTML}{2563EB}",
        "color_name": "clrTransformer",
        "mark": "square*",
        "mark_options": "fill=clrTransformer",
        "line_style": "",
    },
    "olmo3-2": {
        "color_def": "\\definecolor{clrTransformerV2}{HTML}{3B82F6}",
        "color_name": "clrTransformerV2",
        "mark": "square*",
        "mark_options": "fill=clrTransformerV2",
        "line_style": "",
    },
    "olmo3-3": {
        "color_def": "\\definecolor{clrTransformerV3}{HTML}{60A5FA}",
        "color_name": "clrTransformerV3",
        "mark": "square*",
        "mark_options": "fill=clrTransformerV3",
        "line_style": "",
    },
    "pure-gdn": {
        "color_def": "\\definecolor{clrGDN}{HTML}{DC2626}",
        "color_name": "clrGDN",
        "mark": "triangle*",
        "mark_options": "fill=clrGDN",
        "line_style": "",
    },
    "pure-mamba": {
        "color_def": "\\definecolor{clrMamba}{HTML}{D97706}",
        "color_name": "clrMamba",
        "mark": "diamond*",
        "mark_options": "fill=clrMamba",
        "line_style": "",
    },
    # Hybrids — dashed/dotted lines, open or half-filled marks
    "hybrid-gdn": {
        "color_def": "\\definecolor{clrHybGDN}{HTML}{7C3AED}",
        "color_name": "clrHybGDN",
        "mark": "*",
        "mark_options": "fill=clrHybGDN",
        "line_style": "",
    },
    "hybrid-gdn-half": {
        "color_def": "\\definecolor{clrHybGDNHalf}{HTML}{A78BFA}",
        "color_name": "clrHybGDNHalf",
        "mark": "o",
        "mark_options": "",
        "line_style": "dashed",
    },
    "hybrid-gdn-eight": {
        "color_def": "\\definecolor{clrHybGDNEight}{HTML}{5B21B6}",
        "color_name": "clrHybGDNEight",
        "mark": "triangle",
        "mark_options": "",
        "line_style": "dashed",
    },
    "hybrid-gdn-middle": {
        "color_def": "\\definecolor{clrHybGDNMid}{HTML}{059669}",
        "color_name": "clrHybGDNMid",
        "mark": "pentagon*",
        "mark_options": "fill=clrHybGDNMid",
        "line_style": "densely dotted",
    },
    "hybrid-mamba": {
        "color_def": "\\definecolor{clrHybMamba}{HTML}{0891B2}",
        "color_name": "clrHybMamba",
        "mark": "pentagon*",
        "mark_options": "fill=clrHybMamba",
        "line_style": "densely dotted",
    },
    "hybrid-gdn-middle-no-final": {
        "color_def": "\\definecolor{clrHybGDNMidNoFinal}{HTML}{059669}",
        "color_name": "clrHybGDNMidNoFinal",
        "mark": "pentagon",
        "mark_options": "fill=white",
        "line_style": "densely dotted",
    },
}

# Fallback styles when a ladder name is not in LATEX_PLOT_STYLES
_FALLBACK_COLORS = [
    ("clrFallbackA", "9333EA"),
    ("clrFallbackB", "0EA5E9"),
    ("clrFallbackC", "10B981"),
    ("clrFallbackD", "F59E0B"),
    ("clrFallbackE", "EF4444"),
    ("clrFallbackF", "6366F1"),
]
_FALLBACK_MARKS = ["*", "square*", "triangle*", "diamond*", "pentagon*", "o"]


def _get_latex_style(ladder_name: str, idx: int) -> Dict[str, str]:
    """Get LaTeX plot style for a ladder, falling back to auto-generated style."""
    key = ladder_name.lower()
    if key in LATEX_PLOT_STYLES:
        return LATEX_PLOT_STYLES[key]

    # Fallback: generate a style from the index
    fb_color_name, fb_hex = _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)]
    fb_mark = _FALLBACK_MARKS[idx % len(_FALLBACK_MARKS)]
    filled = fb_mark.endswith("*")
    return {
        "color_def": f"\\definecolor{{{fb_color_name}}}{{{{{fb_hex}}}}}",
        "color_name": fb_color_name,
        "mark": fb_mark,
        "mark_options": f"fill={fb_color_name}" if filled else "",
        "line_style": "dashed" if idx % 2 else "",
    }


def _escape_latex_text(s: str) -> str:
    """Escape special LaTeX characters in text strings."""
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")


def generate_latex_plot(
    ladder_results: Dict[str, Dict[str, Dict[str, Any]]],
    metric_name: str,
    extract_fn,
    ylabel: str,
    title: str,
    sizes: Optional[List[str]] = None,
    higher_is_better: bool = True,
    width: str = "0.95\\linewidth",
    height: str = "0.55\\linewidth",
    use_final_checkpoint: bool = False,
    legend_columns: int = 3,
    star_ladder: Optional[str] = None,
    emit_color_defs: bool = False,
) -> str:
    """
    Generate a single TikZ/pgfplots figure showing metric vs parameter count.

    Args:
        ladder_results: The standard {ladder_name: {size_key: data}} dict.
        metric_name: Short identifier for comments (e.g. "base_easy_avg").
        extract_fn: Callable(data) -> Optional[float] to pull the y-value from a size entry.
        ylabel: Y-axis label for the plot.
        title: Plot title.
        sizes: Optional filter on size labels.
        higher_is_better: Controls y-axis direction hint in comments.
        width: TikZ width specification.
        height: TikZ height specification.
        use_final_checkpoint: If True, pick the final (highest D/N) checkpoint per size.
        legend_columns: Number of columns in the legend.
        star_ladder: If set, append a star symbol to this ladder's legend entry.
        emit_color_defs: If True, include \\definecolor commands before the tikzpicture.
            Set to True when using this function standalone; generate_latex_plots()
            emits color definitions once at the top of the file instead.

    Returns:
        TikZ code string. Color definitions are only included if emit_color_defs=True.
    """
    ladder_names = list(ladder_results.keys())

    # Collect all data points per ladder
    ladder_data: Dict[str, List[Tuple[float, float]]] = {}
    all_x: List[float] = []
    all_y: List[float] = []

    for ladder_name in ladder_names:
        points = []
        size_results = ladder_results[ladder_name]

        # Group by base size label to handle multi-checkpoint data
        by_size: Dict[str, List[Tuple[float, float, int]]] = {}
        for size_key, data in size_results.items():
            if sizes and get_size_label(size_key) not in sizes:
                continue
            y = extract_fn(data)
            if y is None:
                continue
            x = _get_x_value(size_key, data) * 1e6  # Convert back to raw param count
            dn = _get_dn_ratio(size_key)
            base_label = get_size_label(size_key)
            by_size.setdefault(base_label, []).append((x, y, dn if dn is not None else 0))

        # Reduce to one point per size
        for _label, pts in by_size.items():
            if use_final_checkpoint or not _has_checkpoints(size_results):
                # Use the checkpoint with highest D/N (or the only one)
                best = max(pts, key=lambda p: p[2])
                points.append((best[0], best[1]))
            else:
                # Average across checkpoints
                avg_x = sum(p[0] for p in pts) / len(pts)
                avg_y = sum(p[1] for p in pts) / len(pts)
                points.append((avg_x, avg_y))

        points.sort(key=lambda p: p[0])
        if points:
            ladder_data[ladder_name] = points
            all_x.extend(p[0] for p in points)
            all_y.extend(p[1] for p in points)

    if not all_x:
        return f"% No data available for {metric_name}\n"

    # Compute axis limits with padding
    x_min = min(all_x) * 0.8
    x_max = max(all_x) * 1.2
    y_range = max(all_y) - min(all_y) if len(all_y) > 1 else 1.0
    y_pad = y_range * 0.1
    y_min = min(all_y) - y_pad
    y_max = max(all_y) + y_pad

    # Build meaningful tick labels
    # For the typical range (60M-760M), use round values
    import math

    # Determine appropriate tick spacing based on data range
    data_range_millions = (max(all_x) - min(all_x)) / 1e6

    if data_range_millions < 20:
        # Small range: use 10M increments
        tick_spacing = 10e6
    elif data_range_millions < 200:
        # Medium range: use 100M increments
        tick_spacing = 100e6
    else:
        # Large range: use 200M increments
        tick_spacing = 200e6

    # Generate ticks starting from a round number
    first_tick = math.ceil(min(all_x) / tick_spacing) * tick_spacing
    major_ticks = []
    current = first_tick
    while current <= max(all_x):
        major_ticks.append(current)
        current += tick_spacing

    # Ensure we have at least 3 ticks for readability
    if len(major_ticks) < 3:
        tick_spacing = tick_spacing / 2
        first_tick = math.ceil(min(all_x) / tick_spacing) * tick_spacing
        major_ticks = []
        current = first_tick
        while current <= max(all_x):
            major_ticks.append(current)
            current += tick_spacing

    def _fmt_xtick(v: float) -> str:
        if v >= 1e9:
            return f"{v/1e9:.1f}B".rstrip("0").rstrip(".")
        return f"{v/1e6:.0f}M"

    xtick_str = ", ".join(f"{v:.0f}" for v in major_ticks)
    xticklabels_str = ", ".join(_fmt_xtick(v) for v in major_ticks)

    # Collect styles; color definitions are emitted by generate_latex_plots() unless
    # emit_color_defs=True (for standalone usage).
    styles: Dict[str, Dict[str, str]] = {}
    seen_colors: set = set()
    color_defs: List[str] = []
    for idx, ladder_name in enumerate(ladder_names):
        if ladder_name not in ladder_data:
            continue
        style = _get_latex_style(ladder_name, idx)
        styles[ladder_name] = style
        if emit_color_defs and style["color_name"] not in seen_colors:
            seen_colors.add(style["color_name"])
            color_defs.append(style["color_def"])

    # Build the TikZ code
    lines: List[str] = []
    direction = "higher is better" if higher_is_better else "lower is better"
    lines.append(f"% {_escape_latex_text(title)} ({direction})")
    if color_defs:
        for cdef in color_defs:
            lines.append(cdef)
        lines.append("")
    lines.append(r"\begin{tikzpicture}")
    lines.append(r"\begin{axis}[")
    lines.append(f"    width={width},")
    lines.append(f"    height={height},")
    lines.append("    xmode=log,")
    lines.append("    log basis x=10,")
    lines.append("    xlabel={Parameters},")
    lines.append(f"    ylabel={{{_escape_latex_text(ylabel)}}},")
    lines.append(f"    xmin={x_min:.0f}, xmax={x_max:.0f},")
    lines.append(f"    ymin={y_min:.2f}, ymax={y_max:.2f},")
    lines.append(f"    xtick={{{xtick_str}}},")
    lines.append(f"    xticklabels={{{xticklabels_str}}},")
    lines.append(r"    legend style={")
    lines.append(r"        at={(0.5,-0.22)},")
    lines.append(r"        anchor=north,")
    lines.append(r"        font=\scriptsize,")
    lines.append(r"        cells={anchor=west},")
    lines.append(r"        draw=none,")
    lines.append(f"        legend columns={legend_columns},")
    lines.append(r"        column sep=6pt,")
    lines.append(r"        row sep=1pt,")
    lines.append(r"    },")
    lines.append(r"    grid=major,")
    lines.append(r"    grid style={gray!15},")
    lines.append(r"    every axis plot/.append style={thick, mark size=2.5pt, smooth},")
    lines.append(r"]")
    lines.append("")

    # Add plot lines for each ladder
    for ladder_name in ladder_names:
        if ladder_name not in ladder_data:
            continue
        points = ladder_data[ladder_name]
        style = styles[ladder_name]

        # Build addplot options (all lines use solid style with different colors)
        plot_opts = [f"color={style['color_name']}"]
        if style["mark_options"]:
            plot_opts.append(f"mark={style['mark']}")
            plot_opts.append(f"mark options={{{style['mark_options']}}}")
        else:
            plot_opts.append(f"mark={style['mark']}")

        # Skip line_style to make all lines solid (same type)

        plot_opts_str = ", ".join(plot_opts)

        coords = " ".join(f"({p[0]:.0f},{p[1]:.4f})" for p in points)
        lines.append(f"\\addplot[{plot_opts_str}]")
        lines.append(f"    coordinates {{{coords}}};")

        display = get_display_name(ladder_name)
        if star_ladder and ladder_name.lower() == star_ladder.lower():
            display += r" $\bigstar$"
        lines.append(f"\\addlegendentry{{{_escape_latex_text(display)}}}")
        lines.append("")

    lines.append(r"\end{axis}")
    lines.append(r"\end{tikzpicture}")

    return "\n".join(lines)


def generate_latex_bar_chart(
    ladder_results: Dict[str, Dict[str, Dict[str, Any]]],
    metric_name: str,
    extract_fn,
    ylabel: str,
    title: str,
    sizes: Optional[List[str]] = None,
    higher_is_better: bool = True,
    width: str = "0.95\\linewidth",
    height: str = "0.55\\linewidth",
    use_final_checkpoint: bool = False,
    legend_columns: int = 3,
    star_ladder: Optional[str] = None,
    emit_color_defs: bool = False,
) -> str:
    """
    Generate a TikZ/pgfplots bar chart showing metric comparison across model sizes.

    Args:
        ladder_results: The standard {ladder_name: {size_key: data}} dict.
        metric_name: Short identifier for comments.
        extract_fn: Callable(data) -> Optional[float] to pull the y-value from a size entry.
        ylabel: Y-axis label for the plot.
        title: Plot title.
        sizes: Optional filter on size labels.
        higher_is_better: Controls y-axis direction hint in comments.
        width: TikZ width specification.
        height: TikZ height specification.
        use_final_checkpoint: If True, pick the final (highest D/N) checkpoint per size.
        legend_columns: Number of columns in the legend.
        star_ladder: If set, append a star symbol to this ladder's legend entry.
        emit_color_defs: If True, include \\definecolor commands before the tikzpicture.

    Returns:
        TikZ code string for a grouped bar chart.
    """
    ladder_names = list(ladder_results.keys())

    # Collect unique size labels across all ladders
    all_size_labels = sorted(
        set(
            _get_x_label(size_key, data)
            for size_results in ladder_results.values()
            for size_key, data in size_results.items()
            if not sizes or get_size_label(size_key) in sizes
        ),
        key=get_size_numeric,
    )

    if not all_size_labels:
        return f"% No data available for {metric_name}\n"

    # Helper: find data for a given size label within a ladder's results
    def _find_by_label(
        size_results: Dict[str, Dict[str, Any]], label: str
    ) -> Optional[Dict[str, Any]]:
        matches = [
            (size_key, data)
            for size_key, data in size_results.items()
            if _get_x_label(size_key, data) == label
        ]
        if not matches:
            return None
        if len(matches) == 1 or use_final_checkpoint:
            return max(matches, key=lambda m: m[1].get("tokens", 0))[1]
        # Average across checkpoints
        averaged: Dict[str, Any] = dict(matches[0][1])
        for suite_key in ("base_easy", "base_main", "heldout"):
            all_cluster_names: set = set()
            for _, data in matches:
                all_cluster_names.update(data.get(suite_key, {}).keys())
            if not all_cluster_names:
                continue
            merged_suite: Dict[str, Dict[str, Any]] = {}
            for cluster in all_cluster_names:
                vals = [
                    data[suite_key][cluster]["avg"]
                    for _, data in matches
                    if cluster in data.get(suite_key, {})
                ]
                if vals:
                    merged_suite[cluster] = {"avg": sum(vals) / len(vals)}
            averaged[suite_key] = merged_suite
        return averaged

    # Collect data for each ladder and size
    bar_data: Dict[str, List[float]] = {}
    all_y: List[float] = []

    for ladder_name in ladder_names:
        size_results = ladder_results[ladder_name]
        y_vals = []
        for label in all_size_labels:
            data = _find_by_label(size_results, label)
            y_val = extract_fn(data) if data else None
            if y_val is not None:
                y_vals.append(y_val)
                all_y.append(y_val)
            else:
                y_vals.append(0)
        bar_data[ladder_name] = y_vals

    if not all_y:
        return f"% No data available for {metric_name}\n"

    # Compute axis limits
    y_range = max(all_y) - min(all_y) if len(all_y) > 1 else 1.0
    y_pad = y_range * 0.1
    y_min = min(all_y) - y_pad
    y_max = max(all_y) + y_pad

    # Collect styles
    styles: Dict[str, Dict[str, str]] = {}
    seen_colors: set = set()
    color_defs: List[str] = []
    for idx, ladder_name in enumerate(ladder_names):
        style = _get_latex_style(ladder_name, idx)
        styles[ladder_name] = style
        if emit_color_defs and style["color_name"] not in seen_colors:
            seen_colors.add(style["color_name"])
            color_defs.append(style["color_def"])

    # Build the TikZ code
    lines: List[str] = []
    direction = "higher is better" if higher_is_better else "lower is better"
    lines.append(f"% {_escape_latex_text(title)} ({direction})")
    if color_defs:
        for cdef in color_defs:
            lines.append(cdef)
        lines.append("")
    lines.append(r"\begin{tikzpicture}")
    lines.append(r"\begin{axis}[")
    lines.append(f"    width={width},")
    lines.append(f"    height={height},")
    lines.append("    ybar,")
    lines.append(f"    bar width={0.8 / len(ladder_names):.3f}cm,")
    lines.append("    xlabel={Model Size},")
    lines.append(f"    ylabel={{{_escape_latex_text(ylabel)}}},")
    lines.append(f"    ymin={y_min:.2f}, ymax={y_max:.2f},")
    lines.append("    symbolic x coords={" + ", ".join(all_size_labels) + "},")
    lines.append("    xtick=data,")
    lines.append(r"    legend style={")
    lines.append(r"        at={(0.5,-0.22)},")
    lines.append(r"        anchor=north,")
    lines.append(r"        font=\scriptsize,")
    lines.append(r"        cells={anchor=west},")
    lines.append(r"        draw=none,")
    lines.append(f"        legend columns={legend_columns},")
    lines.append(r"        column sep=6pt,")
    lines.append(r"        row sep=1pt,")
    lines.append(r"    },")
    lines.append(r"    grid=major,")
    lines.append(r"    grid style={gray!15},")
    lines.append(r"    enlarge x limits=0.15,")
    lines.append(r"]")
    lines.append("")

    # Add plot for each ladder
    for ladder_name in ladder_names:
        y_vals = bar_data[ladder_name]
        style = styles[ladder_name]

        coords = " ".join(f"({label},{y:.4f})" for label, y in zip(all_size_labels, y_vals))
        lines.append(f"\\addplot[fill={style['color_name']}, draw={style['color_name']}]")
        lines.append(f"    coordinates {{{coords}}};")

        display = get_display_name(ladder_name)
        if star_ladder and ladder_name.lower() == star_ladder.lower():
            display += r" $\bigstar$"
        lines.append(f"\\addlegendentry{{{_escape_latex_text(display)}}}")
        lines.append("")

    lines.append(r"\end{axis}")
    lines.append(r"\end{tikzpicture}")

    return "\n".join(lines)


def generate_latex_plots(
    ladder_results: Dict[str, Dict[str, Dict[str, Any]]],
    sizes: Optional[List[str]] = None,
    include_main: bool = False,
    use_final_checkpoint: bool = False,
    star_ladder: Optional[str] = None,
) -> str:
    """
    Generate publication-ready LaTeX/TikZ plots for all metric suites.

    Produces pgfplots code for:
      - Base Easy Suite average BPB bar chart (grouped by size)
      - Base Easy Suite average BPB vs parameters (line plot)
      - Each Base Easy cluster (Math_BPB, Code_BPB, QA_BPB) vs parameters
      - Base Main Suite average accuracy vs parameters (if include_main)
      - Each Base Main cluster vs parameters (if include_main)
      - Average LM perplexity vs parameters

    Args:
        ladder_results: The standard {ladder_name: {size_key: data}} dict.
        sizes: Optional filter on size labels.
        include_main: Include Base Main Suite plots.
        use_final_checkpoint: Use final checkpoint per size instead of mean.
        star_ladder: Ladder name to highlight with a star in the legend.

    Returns:
        Complete LaTeX string with all plots as separate figures.
    """
    parts: List[str] = []
    parts.append("% ===================================================================")
    parts.append("% LaTeX/TikZ plots generated by ladder_metrics_analysis.py")
    parts.append("% Requires: \\usepackage{pgfplots}, \\pgfplotsset{compat=1.18}")
    parts.append("% ===================================================================")
    parts.append("")

    # Emit all color definitions once, at the top (outside any figure/tikzpicture)
    ladder_names = list(ladder_results.keys())
    seen_colors: set = set()
    parts.append("% --- Color definitions ---")
    for idx, ladder_name in enumerate(ladder_names):
        style = _get_latex_style(ladder_name, idx)
        if style["color_name"] not in seen_colors:
            seen_colors.add(style["color_name"])
            parts.append(style["color_def"])
    parts.append("")

    # --- Base Easy Suite: Average BPB ---
    def _extract_easy_avg(data):
        be = data.get("base_easy", {})
        if be:
            return sum(c["avg"] for c in be.values()) / len(be)
        return None

    # Bar chart version (grouped by size)
    parts.append("% --- Base Easy Suite: Average BPB Bar Chart ---")
    parts.append(r"\begin{figure}[htbp]")
    parts.append(r"    \centering")
    parts.append(
        generate_latex_bar_chart(
            ladder_results,
            "base_easy_avg_bar",
            _extract_easy_avg,
            ylabel="Average BPB",
            title="Base Easy Suite Average",
            sizes=sizes,
            higher_is_better=False,
            use_final_checkpoint=use_final_checkpoint,
            star_ladder=star_ladder,
        )
    )
    parts.append(
        r"    \caption{Base Easy Suite average BPB comparison across model sizes (lower is better).}"
    )
    parts.append(r"    \label{fig:base_easy_avg_bar}")
    parts.append(r"\end{figure}")
    parts.append("")

    # Line plot version (vs parameters)
    parts.append("% --- Base Easy Suite: Average BPB vs Parameters ---")
    parts.append(r"\begin{figure}[htbp]")
    parts.append(r"    \centering")
    parts.append(
        generate_latex_plot(
            ladder_results,
            "base_easy_avg",
            _extract_easy_avg,
            ylabel="Average BPB",
            title="Base Easy Suite Average",
            sizes=sizes,
            higher_is_better=False,
            use_final_checkpoint=use_final_checkpoint,
            star_ladder=star_ladder,
        )
    )
    parts.append(
        r"    \caption{Base Easy Suite average BPB vs.\ parameter count (lower is better).}"
    )
    parts.append(r"    \label{fig:base_easy_avg}")
    parts.append(r"\end{figure}")
    parts.append("")

    # --- Base Easy Suite: Per-cluster ---
    for cluster_name in BASE_EASY_SUITE.keys():

        def _extract_cluster(data, _c=cluster_name):
            be = data.get("base_easy", {})
            if _c in be:
                return be[_c]["avg"]
            return None

        clean_name = cluster_name.replace("_BPB", "")
        parts.append(f"% --- Base Easy: {clean_name} BPB vs Parameters ---")
        parts.append(r"\begin{figure}[htbp]")
        parts.append(r"    \centering")
        parts.append(
            generate_latex_plot(
                ladder_results,
                f"base_easy_{cluster_name}",
                _extract_cluster,
                ylabel=f"{clean_name} BPB",
                title=f"Base Easy {clean_name}",
                sizes=sizes,
                higher_is_better=False,
                use_final_checkpoint=use_final_checkpoint,
                star_ladder=star_ladder,
            )
        )
        parts.append(
            f"    \\caption{{Base Easy {_escape_latex_text(clean_name)} BPB vs.\\ parameter count.}}"
        )
        parts.append(f"    \\label{{fig:base_easy_{cluster_name.lower()}}}")
        parts.append(r"\end{figure}")
        parts.append("")

    # --- Base Main Suite (if requested) ---
    if include_main:

        def _extract_main_avg(data):
            bm = data.get("base_main", {})
            if bm:
                return sum(c["avg"] for c in bm.values()) / len(bm) * 100
            return None

        parts.append("% --- Base Main Suite: Average Accuracy vs Parameters ---")
        parts.append(r"\begin{figure}[htbp]")
        parts.append(r"    \centering")
        parts.append(
            generate_latex_plot(
                ladder_results,
                "base_main_avg",
                _extract_main_avg,
                ylabel="Average Accuracy (\\%)",
                title="Base Main Suite Average",
                sizes=sizes,
                higher_is_better=True,
                use_final_checkpoint=use_final_checkpoint,
                star_ladder=star_ladder,
            )
        )
        parts.append(
            r"    \caption{Base Main Suite average accuracy vs.\ parameter count (higher is better).}"
        )
        parts.append(r"    \label{fig:base_main_avg}")
        parts.append(r"\end{figure}")
        parts.append("")

        for cluster_name in BASE_MAIN_SUITE.keys():

            def _extract_main_cluster(data, _c=cluster_name):
                bm = data.get("base_main", {})
                if _c in bm:
                    return bm[_c]["avg"] * 100
                return None

            parts.append(f"% --- Base Main: {cluster_name} vs Parameters ---")
            parts.append(r"\begin{figure}[htbp]")
            parts.append(r"    \centering")
            parts.append(
                generate_latex_plot(
                    ladder_results,
                    f"base_main_{cluster_name}",
                    _extract_main_cluster,
                    ylabel="Accuracy (\\%)" if cluster_name != "FIM" else "pass@1 (\\%)",
                    title=f"Base Main {cluster_name}",
                    sizes=sizes,
                    higher_is_better=True,
                    use_final_checkpoint=use_final_checkpoint,
                    star_ladder=star_ladder,
                )
            )
            parts.append(
                f"    \\caption{{Base Main {_escape_latex_text(cluster_name)} vs.\\ parameter count.}}"
            )
            parts.append(f"    \\label{{fig:base_main_{cluster_name.lower()}}}")
            parts.append(r"\end{figure}")
            parts.append("")

    # --- Relative Performance Comparison ---
    if len(ladder_names) > 1:
        baseline_ladder = ladder_names[0]
        suite_key = "base_main" if include_main else "base_easy"

        # Collect unique size labels
        all_size_labels = sorted(
            set(
                _get_x_label(size_key, data)
                for size_results in ladder_results.values()
                for size_key, data in size_results.items()
                if not sizes or get_size_label(size_key) in sizes
            ),
            key=get_size_numeric,
        )

        # Helper function to find data by label
        def _find_by_label(
            size_results: Dict[str, Dict[str, Any]], label: str
        ) -> Optional[Dict[str, Any]]:
            matches = [
                (size_key, data)
                for size_key, data in size_results.items()
                if _get_x_label(size_key, data) == label
            ]
            if not matches:
                return None
            if len(matches) == 1 or use_final_checkpoint:
                return max(matches, key=lambda m: m[1].get("tokens", 0))[1]
            averaged: Dict[str, Any] = dict(matches[0][1])
            for sk in ("base_easy", "base_main", "heldout"):
                all_cluster_names: set = set()
                for _, data in matches:
                    all_cluster_names.update(data.get(sk, {}).keys())
                if not all_cluster_names:
                    continue
                merged_suite: Dict[str, Dict[str, Any]] = {}
                for cluster in all_cluster_names:
                    vals = [
                        data[sk][cluster]["avg"]
                        for _, data in matches
                        if cluster in data.get(sk, {})
                    ]
                    if vals:
                        merged_suite[cluster] = {"avg": sum(vals) / len(vals)}
                averaged[sk] = merged_suite
            return averaged

        # Build relative comparison data for each non-baseline ladder
        relative_data: Dict[str, List[float]] = {}
        all_diffs: List[float] = []

        for ladder_name in ladder_names[1:]:
            diffs = []
            for label in all_size_labels:
                baseline_data = _find_by_label(ladder_results[baseline_ladder], label)
                compare_data = _find_by_label(ladder_results[ladder_name], label)

                baseline_suite = baseline_data.get(suite_key, {}) if baseline_data else {}
                compare_suite = compare_data.get(suite_key, {}) if compare_data else {}

                if baseline_suite and compare_suite:
                    baseline_avg = sum(c["avg"] for c in baseline_suite.values()) / len(
                        baseline_suite
                    )
                    compare_avg = sum(c["avg"] for c in compare_suite.values()) / len(compare_suite)
                    if baseline_avg > 0:
                        diff = ((compare_avg - baseline_avg) / baseline_avg) * 100
                    else:
                        diff = 0
                    diffs.append(diff)
                    all_diffs.append(diff)
                else:
                    diffs.append(0)
            relative_data[ladder_name] = diffs

        if all_diffs:
            # Generate bar chart for relative comparison
            parts.append("% --- Relative Performance Comparison ---")
            parts.append(r"\begin{figure}[htbp]")
            parts.append(r"    \centering")

            # Build the bar chart manually
            y_range = max(all_diffs) - min(all_diffs) if len(all_diffs) > 1 else 1.0
            y_pad = y_range * 0.1
            y_min = min(all_diffs) - y_pad
            y_max = max(all_diffs) + y_pad

            # Get styles
            compare_ladders = ladder_names[1:]
            bar_lines: List[str] = []
            bar_lines.append(r"\begin{tikzpicture}")
            bar_lines.append(r"\begin{axis}[")
            bar_lines.append(r"    width=0.95\linewidth,")
            bar_lines.append(r"    height=0.55\linewidth,")
            bar_lines.append(r"    ybar,")
            bar_lines.append(f"    bar width={0.8 / len(compare_ladders):.3f}cm,")
            bar_lines.append(r"    xlabel={Model Size},")
            if include_main:
                bar_lines.append(r"    ylabel={Relative Accuracy Difference (\%)},")
            else:
                bar_lines.append(r"    ylabel={BPB Difference (\%, more negative = better)},")
            bar_lines.append(f"    ymin={y_min:.2f}, ymax={y_max:.2f},")
            bar_lines.append("    symbolic x coords={" + ", ".join(all_size_labels) + "},")
            bar_lines.append(r"    xtick=data,")
            bar_lines.append(r"    legend style={")
            bar_lines.append(r"        at={(0.5,-0.22)},")
            bar_lines.append(r"        anchor=north,")
            bar_lines.append(r"        font=\scriptsize,")
            bar_lines.append(r"        cells={anchor=west},")
            bar_lines.append(r"        draw=none,")
            bar_lines.append(r"        legend columns=2,")
            bar_lines.append(r"        column sep=6pt,")
            bar_lines.append(r"        row sep=1pt,")
            bar_lines.append(r"    },")
            bar_lines.append(r"    grid=major,")
            bar_lines.append(r"    grid style={gray!15},")
            bar_lines.append(r"    enlarge x limits=0.15,")
            bar_lines.append(r"    extra y ticks={0},")
            bar_lines.append(r"    extra y tick style={grid=major, grid style={black, thick}},")
            bar_lines.append(r"]")
            bar_lines.append("")

            for idx, ladder_name in enumerate(compare_ladders, start=1):
                diffs = relative_data[ladder_name]
                style = _get_latex_style(ladder_name, idx)

                coords = " ".join(
                    f"({label},{diff:.4f})" for label, diff in zip(all_size_labels, diffs)
                )
                bar_lines.append(
                    f"\\addplot[fill={style['color_name']}, draw={style['color_name']}]"
                )
                bar_lines.append(f"    coordinates {{{coords}}};")

                display = get_display_name(ladder_name)
                if star_ladder and ladder_name.lower() == star_ladder.lower():
                    display += r" $\bigstar$"
                bar_lines.append(
                    f"\\addlegendentry{{{_escape_latex_text(display)} vs {_escape_latex_text(get_display_name(baseline_ladder))}}}"
                )
                bar_lines.append("")

            bar_lines.append(r"\end{axis}")
            bar_lines.append(r"\end{tikzpicture}")

            parts.append("\n".join(bar_lines))

            if include_main:
                parts.append(
                    f"    \\caption{{Main Suite accuracy relative to {_escape_latex_text(get_display_name(baseline_ladder))} (percentage difference).}}"
                )
            else:
                parts.append(
                    f"    \\caption{{Easy Suite BPB percentage difference vs {_escape_latex_text(get_display_name(baseline_ladder))} (more negative = better).}}"
                )
            parts.append(r"    \label{fig:relative_comparison}")
            parts.append(r"\end{figure}")
            parts.append("")

    # --- LM Perplexity ---
    def _extract_avg_ppl(data):
        lm = data.get("lm_metrics", {})
        ppls = [v.get("PPL") for v in lm.values() if v.get("PPL") is not None]
        if ppls:
            return sum(ppls) / len(ppls)
        return None

    parts.append("% --- Average LM Perplexity vs Parameters ---")
    parts.append(r"\begin{figure}[htbp]")
    parts.append(r"    \centering")
    parts.append(
        generate_latex_plot(
            ladder_results,
            "lm_avg_ppl",
            _extract_avg_ppl,
            ylabel="Perplexity",
            title="Average LM Perplexity",
            sizes=sizes,
            higher_is_better=False,
            use_final_checkpoint=use_final_checkpoint,
            star_ladder=star_ladder,
        )
    )
    parts.append(
        r"    \caption{Average language modeling perplexity vs.\ parameter count (lower is better).}"
    )
    parts.append(r"    \label{fig:lm_avg_ppl}")
    parts.append(r"\end{figure}")
    parts.append("")

    return "\n".join(parts)


# =============================================================================
# Export Functions
# =============================================================================


def export_results(
    ladder_results: Dict[str, Dict[str, Dict[str, Any]]],
    output_path: Path,
    sizes: Optional[List[str]] = None,
    include_main: bool = False,
) -> None:
    """Export results to various formats."""
    easy_df = create_base_easy_table(ladder_results, sizes)
    lm_df = create_lm_table(ladder_results, sizes)

    pkl_data: Dict[str, Any] = {
        "base_easy": easy_df,
        "lm": lm_df,
        "raw": ladder_results,
    }

    main_df = create_base_main_table(ladder_results, sizes) if include_main else pd.DataFrame()
    heldout_df = create_heldout_table(ladder_results, sizes) if include_main else pd.DataFrame()

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
    pkl_data["base_main"] = main_df
    pkl_data["heldout"] = heldout_df
    pd.to_pickle(pkl_data, pkl_path)

    # Generate markdown
    md_path = output_path / "olmobaseeval_results.md"
    with open(md_path, "w") as f:
        f.write("# OlmoBaseEval Results\n\n")

        if not main_df.empty:
            f.write("## Base Main Suite (accuracy/pass@k, higher is better)\n\n")
            f.write(main_df.to_markdown(index=False))
            f.write("\n\n")

        f.write("## Base Easy Suite (BPB, lower is better)\n\n")
        if not easy_df.empty:
            f.write(easy_df.to_markdown(index=False))
        f.write("\n\n")

        if not heldout_df.empty:
            f.write("## Held-out Suite (accuracy, higher is better)\n\n")
            f.write(heldout_df.to_markdown(index=False))
            f.write("\n\n")

        f.write("## Language Modeling (perplexity, lower is better)\n\n")
        if not lm_df.empty:
            f.write(lm_df.to_markdown(index=False))
        f.write("\n")

    # Generate LaTeX tables
    latex_output = generate_comparison_latex(ladder_results, sizes, include_main=include_main)
    latex_path = output_path / "olmobaseeval_tables.tex"
    with open(latex_path, "w") as f:
        f.write(latex_output)

    # Generate ablation table
    ablation_output = generate_ablation_latex_table(ladder_results, sizes)
    ablation_path = output_path / "olmobaseeval_ablation.tex"
    with open(ablation_path, "w") as f:
        f.write(ablation_output)

    # Generate LaTeX/TikZ plots
    latex_plots = generate_latex_plots(ladder_results, sizes, include_main=include_main)
    plots_path = output_path / "olmobaseeval_plots.tex"
    with open(plots_path, "w") as f:
        f.write(latex_plots)

    print(f"\nExported OlmoBaseEval results to: {output_path}")
    csv_files = "olmobaseeval_easy.csv, lm_comparison.csv"
    if include_main:
        csv_files = "olmobaseeval_main.csv, " + csv_files + ", olmobaseeval_heldout.csv"
    print(f"  CSV files: {csv_files}")
    print(f"  LaTeX tables: {latex_path}")
    print(f"  LaTeX ablation: {ablation_path}")
    print(f"  LaTeX plots: {plots_path}")
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

    # Generate LaTeX/TikZ plots (pgfplots) for publication
    python ladder_metrics_analysis.py --compare ... --latex-plots

    # Highlight a specific ladder with a star in LaTeX plots
    python ladder_metrics_analysis.py --compare ... --latex-plots --star-ladder hybrid-gdn
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
    parser.add_argument(
        "--latex-ablation",
        action="store_true",
        help="Print a publication-ready ablation table (architectures as rows, "
        "sizes as column groups) in LaTeX booktabs format to stdout.",
    )
    parser.add_argument(
        "--latex-plots",
        action="store_true",
        help="Generate LaTeX/TikZ (pgfplots) code for publication-ready plots "
        "showing metrics vs parameter count. Output to stdout or to file with --export/--output.",
    )
    parser.add_argument(
        "--star-ladder",
        type=str,
        default=None,
        help="Ladder name to highlight with a star in plot legends (e.g., hybrid-gdn).",
    )
    parser.add_argument(
        "--all-checkpoints",
        action="store_true",
        help="Use all checkpoints from each model size (not just the final one). "
        "This gives more data points for scaling analysis.",
    )
    parser.add_argument(
        "--include-pre-decay",
        action="store_true",
        help="Include pre-decay checkpoints in addition to post-decay. "
        "By default, only post-decay checkpoints (D/N = 10, 20, 40, 80, 160, ...) are used.",
    )
    parser.add_argument(
        "--curve-final",
        action="store_true",
        help="Draw the scaling curve through the final (highest D/N) checkpoint per model size "
        "instead of the mean of all checkpoints. Only affects --all-checkpoints mode.",
    )
    parser.add_argument(
        "--main",
        action="store_true",
        help="Include Base Main Suite results (accuracy/pass@k metrics). "
        "By default only the Easy Suite (BPB) is shown since most main metrics "
        "are unavailable for ladder runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Directory to save figures to (creates it if needed)",
    )

    args = parser.parse_args()

    if not args.ladder_dir and not args.compare:
        parser.error("Either --ladder-dir or --compare must be specified")

    # Determine figure output directory
    figure_output = None
    if args.output:
        figure_output = Path(args.output).expanduser()
        figure_output.mkdir(parents=True, exist_ok=True)
    elif args.export:
        figure_output = Path(args.export).expanduser()
        figure_output.mkdir(parents=True, exist_ok=True)

    ladder_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    if args.ladder_dir:
        ladder_dir = Path(args.ladder_dir).expanduser()
        ladder_name = ladder_dir.name
        print(f"\nAnalyzing ladder: {get_display_name(ladder_name)}")
        print(f"Directory: {ladder_dir}")
        ladder_results[ladder_name] = analyze_ladder(
            ladder_dir,
            ladder_name=ladder_name,
            use_all_checkpoints=args.all_checkpoints,
            post_decay_only=not args.include_pre_decay,
        )

    if args.compare:
        for spec in args.compare:
            if ":" in spec:
                name, path = spec.split(":", 1)
            else:
                path = spec
                name = Path(path).name

            ladder_dir = Path(path).expanduser()
            print(f"\nAnalyzing ladder: {get_display_name(name)}")
            print(f"Directory: {ladder_dir}")
            ladder_results[name] = analyze_ladder(
                ladder_dir,
                ladder_name=name,
                use_all_checkpoints=args.all_checkpoints,
                post_decay_only=not args.include_pre_decay,
            )

    group_by_size = not args.group_by_ladder

    # Create and display tables
    include_main = args.main
    easy_df = create_base_easy_table(ladder_results, args.sizes)
    lm_df = create_lm_table(ladder_results, args.sizes)

    if include_main:
        main_df = create_base_main_table(ladder_results, args.sizes)
        heldout_df = create_heldout_table(ladder_results, args.sizes)
        print_base_main_table(main_df, group_by_size=group_by_size)

    print_base_easy_table(easy_df, group_by_size=group_by_size)

    if include_main:
        print_heldout_table(heldout_df, group_by_size=group_by_size)  # defined above in same guard

    print_lm_table(lm_df, group_by_size=group_by_size)

    # Print relative Easy Suite improvement summary when comparing multiple ladders
    ladder_names = list(ladder_results.keys())
    if len(ladder_names) > 1:
        baseline_name = ladder_names[0]
        baseline_results = ladder_results[baseline_name]

        # Collect all size labels across ladders
        all_size_labels = sorted(
            set(
                get_size_label(sk)
                for sr in ladder_results.values()
                for sk in sr.keys()
                if not args.sizes or get_size_label(sk) in args.sizes
            ),
            key=get_size_numeric,
        )

        print("\n" + "=" * 100)
        print(f"EASY SUITE RELATIVE IMPROVEMENT vs baseline ({get_display_name(baseline_name)})")
        print("  (positive = better, i.e. lower BPB)")
        print("=" * 100)

        easy_clusters = list(BASE_EASY_SUITE.keys())
        header = f"{'Size':<8} {'Ladder':<25}"
        for c in easy_clusters:
            header += f" {c:>14}"
        header += f" {'Avg':>14}"
        print(header)
        print("-" * 100)

        for size_label in all_size_labels:
            # Find baseline data for this size
            baseline_data = None
            for sk, sd in baseline_results.items():
                if get_size_label(sk) == size_label:
                    baseline_data = sd
                    break

            if baseline_data is None:
                continue

            baseline_easy = baseline_data.get("base_easy", {})
            if not baseline_easy:
                continue

            for ladder_name in ladder_names[1:]:
                compare_data = None
                for sk, sd in ladder_results[ladder_name].items():
                    if get_size_label(sk) == size_label:
                        compare_data = sd
                        break

                if compare_data is None:
                    continue

                compare_easy = compare_data.get("base_easy", {})
                line = f"{size_label:<8} {get_display_name(ladder_name):<25}"
                diffs = []
                for cluster in easy_clusters:
                    b_val = baseline_easy.get(cluster, {}).get("avg")
                    c_val = compare_easy.get(cluster, {}).get("avg")
                    if b_val is not None and c_val is not None and b_val > 0:
                        # Lower BPB is better, so positive diff = improvement
                        diff_pct = ((b_val - c_val) / b_val) * 100
                        diffs.append(diff_pct)
                        sign = "+" if diff_pct >= 0 else ""
                        line += f" {sign}{diff_pct:>12.2f}%"
                    else:
                        line += f" {'-':>14}"
                if diffs:
                    avg_diff = sum(diffs) / len(diffs)
                    sign = "+" if avg_diff >= 0 else ""
                    line += f" {sign}{avg_diff:>12.2f}%"
                else:
                    line += f" {'-':>14}"
                print(line)
            print()

        print("=" * 100)

    # Print raw metrics if requested
    if args.raw:
        print("\n\nRAW METRICS:")
        print("-" * 50)
        for ladder_name, sizes_data in ladder_results.items():
            print(f"\n{get_display_name(ladder_name)}:")
            for size_key, data in sizes_data.items():
                size_label = get_size_label(size_key)
                if args.sizes and size_label not in args.sizes:
                    continue
                print(f"\n  {size_key} (step {data['step']}, {data['tokens']:,} tokens):")

                if include_main:
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

        latex_output = generate_comparison_latex(
            ladder_results, args.sizes, include_main=include_main
        )
        print(latex_output)

    # Print ablation-style LaTeX table if requested
    if args.latex_ablation:
        print("\n" + "=" * 80)
        print("LATEX ABLATION TABLE")
        print("=" * 80)
        print("\n% Add to preamble: \\usepackage{booktabs}\n")

        ablation_output = generate_ablation_latex_table(
            ladder_results,
            sizes=args.sizes,
            star_ladder=args.star_ladder,
        )
        print(ablation_output)

    # Generate LaTeX/TikZ plots if requested
    if args.latex_plots:
        latex_plots = generate_latex_plots(
            ladder_results,
            args.sizes,
            include_main=include_main,
            use_final_checkpoint=args.curve_final,
            star_ladder=args.star_ladder,
        )
        if figure_output:
            plots_path = figure_output / "olmobaseeval_plots.tex"
            with open(plots_path, "w") as f:
                f.write(latex_plots)
            print(f"\nSaved LaTeX plots to: {plots_path}")
        else:
            print("\n" + "=" * 80)
            print("LATEX/TIKZ PLOTS (pgfplots)")
            print("=" * 80)
            print()
            print(latex_plots)

    # Generate plots if requested
    if args.plot:
        show_plots = figure_output is None
        plot_all(
            ladder_results,
            args.sizes,
            figure_output,
            show=show_plots,
            include_main=include_main,
            use_final_checkpoint=args.curve_final,
            star_ladder=args.star_ladder,
        )

    # Export if requested
    if args.export:
        export_path = Path(args.export).expanduser()
        export_path.mkdir(parents=True, exist_ok=True)
        export_results(ladder_results, export_path, args.sizes, include_main=include_main)


if __name__ == "__main__":
    main()
