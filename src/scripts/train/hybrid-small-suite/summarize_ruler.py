#!/usr/bin/env python3
"""Summarize RULER subtask results for LC models in a table format.

Shows per-subtask scores across context lengths for each model,
filtered to only LC-stage models.

Usage:
    uv run src/scripts/train/hybrid-small-suite/summarize_ruler.py --group yashasbls-hybrid-small-evals-v2
    uv run src/scripts/train/hybrid-small-suite/summarize_ruler.py --group yashasbls-hybrid-small-evals-v2 --no-color
    uv run src/scripts/train/hybrid-small-suite/summarize_ruler.py --group yashasbls-hybrid-small-evals-v2 --include-midtrain
"""

from __future__ import annotations

import argparse
import json
import glob
import re
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

# RULER subtask types (without context length suffix)
RULER_SUBTASKS = [
    "ruler_all",
    "ruler_niah_s_1",
    "ruler_niah_s_2",
    "ruler_niah_s_3",
    "ruler_niah_mk_1",
    "ruler_niah_mk_2",
    "ruler_niah_mk_3",
    "ruler_niah_mv",
    "ruler_niah_mq",
    "ruler_cwe",
    "ruler_fwe",
    "ruler_vt",
    "ruler_qa_1",
    "ruler_qa_2",
]

CONTEXT_LENGTHS = [4096, 8192, 16384, 32768, 65536, 131072]

SCALE_ORDER = {"275M": 0, "810M": 1, "1.4B": 2, "7B": 3}


def get_model_short_name(model_path: str) -> str:
    model_path = model_path.rstrip("/")
    if model_path.count("/") <= 2:
        return model_path.split("/")[-1]
    return model_path.split("/")[-2]


def parse_model_info(name: str) -> tuple[str, str]:
    """Return (scale, stage) from model name."""
    n = name.lower()
    if "275m" in n:
        scale = "275M"
    elif "810m" in n:
        scale = "810M"
    elif "1.4b" in n:
        scale = "1.4B"
    elif "7b" in n:
        scale = "7B"
    else:
        scale = "?"

    if "long-context" in n or "lc-v" in n:
        stage = "LC"
    elif "midtraining" in n or "midtrain" in n:
        stage = "mid"
    else:
        stage = "?"
    return scale, stage


def collect_results(group: str) -> dict[str, dict[str, float]]:
    """Collect results: model_name -> task -> score."""
    model_tasks: dict[str, dict[str, float]] = defaultdict(dict)
    search_dir = RESULTS_DIR / group
    if not search_dir.exists():
        print(f"Results directory not found: {search_dir}")
        return {}

    for f in glob.glob(str(search_dir / "**" / "metrics.json"), recursive=True):
        with open(f) as fh:
            m = json.load(fh)
        if m.get("experiment_group", "") != group:
            continue
        model_path = m.get("config", {}).get("provider", {}).get("model", "?")
        basename = get_model_short_name(model_path)
        summary = m.get("summary", {})
        for task, info in summary.items():
            score = info.get("score")
            if score is not None:
                model_tasks[basename][task] = score

    return dict(model_tasks)


def _plain_width(formatted: str) -> int:
    return len(re.sub(r"\033\[[^m]*m", "", formatted))


def fmt_score(score: float, is_best: bool, is_worst: bool, no_color: bool) -> str:
    s = f"{score:.3f}"
    if no_color or (not is_best and not is_worst):
        return s
    if is_best:
        return f"\033[1;32m{s}\033[0m"
    if is_worst:
        return f"\033[1;31m{s}\033[0m"
    return s


def main():
    parser = argparse.ArgumentParser(description="Summarize RULER subtask results for LC models.")
    parser.add_argument("--group", default="yashasbls-hybrid-small-evals-v2", help="Eval group name")
    parser.add_argument("--no-color", action="store_true", help="Disable color output")
    parser.add_argument("--include-midtrain", action="store_true", help="Also show midtraining models")
    parser.add_argument("--length", type=int, default=None, help="Show only one context length")
    args = parser.parse_args()

    results = collect_results(args.group)
    if not results:
        print("No results found.")
        return

    # Filter to LC models (and optionally midtrain)
    filtered = {}
    for name, tasks in results.items():
        scale, stage = parse_model_info(name)
        if stage == "LC" or (args.include_midtrain and stage == "mid"):
            filtered[name] = tasks

    if not filtered:
        print("No LC models found in results.")
        return

    # Sort models by scale then stage (LC after mid)
    def sort_key(name):
        scale, stage = parse_model_info(name)
        return (SCALE_ORDER.get(scale, 99), 0 if stage == "mid" else 1, name)

    models = sorted(filtered.keys(), key=sort_key)

    # Determine context lengths to show
    lengths = [args.length] if args.length else CONTEXT_LENGTHS

    # For each context length, print a table
    for ctx_len in lengths:
        # Check if any model has data for this length
        has_data = False
        for m in models:
            for st in RULER_SUBTASKS:
                task_key = f"{st}__{ctx_len}"
                if task_key in filtered[m]:
                    has_data = True
                    break
            if has_data:
                break
        if not has_data:
            continue

        # Build short display names for columns
        col_names = []
        for m in models:
            scale, stage = parse_model_info(m)
            # Extract LR if present
            lr_match = re.search(r"-lr([\d.e\-]+)", m)
            lr_str = f" lr={lr_match.group(1)}" if lr_match else ""
            col_names.append(f"{scale} {stage}{lr_str}")

        col_width = max(14, max(len(c) for c in col_names) + 2)
        task_col_width = 20

        print(f"\n{'=' * 70}")
        print(f"  RULER subtasks @ {ctx_len} tokens ({ctx_len // 1024}K)")
        print(f"{'=' * 70}")

        # Header
        header = " " * task_col_width
        for cn in col_names:
            header += "  " + cn.center(col_width)
        print(header)
        print("-" * (task_col_width + len(models) * (col_width + 2)))

        for subtask in RULER_SUBTASKS:
            task_key = f"{subtask}__{ctx_len}"
            # Get short display name
            display = subtask.replace("ruler_", "")

            row_scores: list[float | None] = []
            for m in models:
                row_scores.append(filtered[m].get(task_key))

            # Skip if no data
            if all(s is None for s in row_scores):
                continue

            # Find best/worst
            valid = [s for s in row_scores if s is not None]
            best = max(valid) if len(valid) > 1 else None
            worst = min(valid) if len(valid) > 1 else None

            row = display.ljust(task_col_width)
            for score in row_scores:
                if score is None:
                    cell = "—".center(col_width)
                else:
                    is_best = best is not None and score == best
                    is_worst = worst is not None and score == worst
                    cell_str = fmt_score(score, is_best, is_worst, args.no_color)
                    pad = col_width - _plain_width(cell_str)
                    cell = " " * (pad // 2) + cell_str + " " * (pad - pad // 2)
                row += "  " + cell
            print(row)

        # Bold the "all" row
        print()


if __name__ == "__main__":
    main()
