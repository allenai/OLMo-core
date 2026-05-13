#!/usr/bin/env python3
"""Compare paper-reported numbers vs our eval numbers for 7B models.

Paper numbers are in percentage (0-100 scale).
Our eval numbers are in 0-1 scale (converted to percentage for display).

Usage:
    python3 src/scripts/train/hybrid-small-suite/compare_paper.py
"""

import json
import glob
import re
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

# ── Paper-reported numbers (percentage scale) ──────────────────────────
# Columns: Math, Code, MC STEM, MC Non-STEM, GenQA, LBPP, BBH, MMLU Pro, DM Math
PAPER_TASKS = [
    "Math", "MC STEM", "MC Non-STEM", "GenQA",
    "RULER 4k", "RULER 8k", "RULER 16k", "RULER 32k", "RULER 64k", "RULER 128k",
]

PAPER_NUMBERS: dict[str, dict[str, float]] = {
    "OLMo 3 7B LC": {
        "Math": 54.6, "MC STEM": 66.2, "MC Non-STEM": 78.2,
        "GenQA": 72.5,
        "RULER 4k": 95.8, "RULER 8k": 89.3, "RULER 16k": 83.2, "RULER 32k": 78.9, "RULER 64k": 70.9,
    },
    "OLMo Hybrid 7B LC": {
        "Math": 55.1, "MC STEM": 70.0, "MC Non-STEM": 80.4,
        "GenQA": 72.9,
        "RULER 4k": 92.2, "RULER 8k": 89.8, "RULER 16k": 88.4, "RULER 32k": 86.2, "RULER 64k": 85.0,
    },
}

# ── Mapping from paper task names to olmo-eval task names ──────────────
# Each entry: paper_task -> eval_task_name (score will be ×100 for comparison)
TASK_MAPPING: dict[str, str] = {
    "Math": "olmobase:math",
    "MC STEM": "olmobase:mcqa_stem",
    "MC Non-STEM": "olmobase:mcqa_non_stem",
    "GenQA": "olmobase:gen",
    "RULER 4k": "ruler_all__4096",
    "RULER 8k": "ruler_all__8192",
    "RULER 16k": "ruler_all__16384",
    "RULER 32k": "ruler_all__32768",
    "RULER 64k": "ruler_all__65536",
    "RULER 128k": "ruler_all__131072",
    # These are not in our eval set yet:
    # "LBPP": ???,
    # "BBH": ???,
    # "MMLU Pro": ???,
    # "DM Math": ???,
}

# ── Map paper model names to eval model identifiers ────────────────────
# The eval results use the HF model name or checkpoint dir name.
# We need to figure out which results files correspond to which paper row.
# For now, we only have one checkpoint per 7B model (the released ones).
MODEL_MAPPING: dict[str, str] = {
    # paper name -> substring to match in eval model path
    "OLMo 3 7B": "Olmo-3-1025-7B",
    "OLMo Hybrid 7B": "Olmo-Hybrid-7B",
}


def collect_7b_results() -> dict[str, dict[str, float]]:
    """Collect eval results for 7B models, keyed by model short name -> task -> score."""
    model_tasks: dict[str, dict[str, float]] = defaultdict(dict)
    files = glob.glob(str(RESULTS_DIR / "**" / "metrics.json"), recursive=True)

    for f in files:
        with open(f) as fh:
            m = json.load(fh)
        model_path = m.get("config", {}).get("provider", {}).get("model", "?")
        if "7b" not in model_path.lower() and "7B" not in model_path:
            continue
        # Use last path component as name
        name = model_path.rstrip("/").split("/")[-1]
        if name.endswith("-hf"):
            name = model_path.rstrip("/").split("/")[-2]
        summary = m.get("summary", {})
        for task, info in summary.items():
            score = info.get("score")
            if score is not None:
                model_tasks[name][task] = score

    return dict(model_tasks)


def _plain_width(s: str) -> int:
    return len(re.sub(r"\033\[[^m]*m", "", s))


def print_comparison():
    eval_results = collect_7b_results()

    # Figure out which eval model matches each paper model family
    eval_model_name: dict[str, str] = {}  # paper family -> eval key
    for paper_family, eval_substr in MODEL_MAPPING.items():
        for name in eval_results:
            if eval_substr in name:
                eval_model_name[paper_family] = name
                break

    print("7B Paper vs Eval Comparison")
    print("=" * 80)
    print(f"Eval models found: {list(eval_model_name.values())}")
    print()

    # For each paper model row, show paper vs eval side by side
    task_width = max(len(t) for t in PAPER_TASKS)
    paper_col = 8
    eval_col = 8
    delta_col = 10

    header = f"{'Task':<{task_width}}  {'Paper':>{paper_col}}  {'Eval':>{eval_col}}  {'Delta':>{delta_col}}"
    sep = "-" * len(header)

    for paper_model, paper_scores in PAPER_NUMBERS.items():
        # Find the eval model family
        family = None
        for fam in MODEL_MAPPING:
            if fam in paper_model:
                family = fam
                break
        if family is None or family not in eval_model_name:
            print(f"\n{paper_model}  (no eval data)")
            continue

        eval_name = eval_model_name[family]
        eval_data = eval_results.get(eval_name, {})

        print(f"\n\033[1m{paper_model}\033[0m  (eval: {eval_name})")
        print(header)
        print(sep)

        for task in PAPER_TASKS:
            paper_val = paper_scores.get(task)
            eval_task = TASK_MAPPING.get(task)

            row = f"{task:<{task_width}}"

            # Paper value
            if paper_val is not None:
                row += f"  {paper_val:>{paper_col}.1f}"
            else:
                row += f"  {'—':>{paper_col}}"

            # Eval value (convert 0-1 to percentage)
            if eval_task and eval_task in eval_data:
                eval_val = eval_data[eval_task] * 100
                row += f"  {eval_val:>{eval_col}.1f}"

                # Delta
                if paper_val is not None:
                    delta = eval_val - paper_val
                    sign = "+" if delta >= 0 else ""
                    delta_str = f"({sign}{delta:.1f})"
                    if abs(delta) > 2:
                        # Large gap — red
                        delta_str = f"\033[31m{delta_str}\033[0m"
                    elif abs(delta) > 0.5:
                        # Small gap — yellow
                        delta_str = f"\033[33m{delta_str}\033[0m"
                    else:
                        # Match — green
                        delta_str = f"\033[32m{delta_str}\033[0m"
                    pad = delta_col - _plain_width(delta_str)
                    row += "  " + " " * max(0, pad) + delta_str
                else:
                    row += f"  {'':>{delta_col}}"
            else:
                row += f"  {'N/A':>{eval_col}}"
                row += f"  {'—':>{delta_col}}"

            print(row)


if __name__ == "__main__":
    print_comparison()
