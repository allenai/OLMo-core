#!/usr/bin/env python3
"""Summarize LR sweep results in a table format (like summarize_results.py).

Columns are grouped by model size, with one column per LR value.
Rows are tasks (same grouping as summarize_results.py).

Usage:
    uv run src/scripts/train/hybrid-small-suite/visualize_lr_sweep.py --group yashasbls-hybrid-small-evals-debug
    uv run src/scripts/train/hybrid-small-suite/visualize_lr_sweep.py --group yashasbls-hybrid-small-evals-debug --no-color
"""

from __future__ import annotations

import argparse
import json
import glob
import re
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

# Same task groups as summarize_results.py
KEY_TASK_GROUPS = [
    ("OLMoBase aggregate", [
        "olmobase:mcqa_stem",
        "olmobase:mcqa_non_stem",
        "olmobase:gen",
        "olmobase:math",
        "olmobase:easy:code:bpb",
        "olmobase:easy:qa:rc",
    ]),
    ("Common sense / NLU", [
        "arc:rc:olmo3base",
        "hellaswag:rc:olmo3base",
        "mmlu:rc:olmo3base",
        "winogrande:rc:olmo3base",
        "piqa:rc:olmo3base",
        "csqa:rc:olmo3base",
        "socialiqa:rc:olmo3base",
        "sciq:rc:olmo3base",
    ]),
    ("Reading comprehension", [
        "drop:rc:olmo3base",
        "naturalqs:rc:olmo3base",
        "squad:rc:olmo3base",
        "coqa:rc:olmo3base",
        "jeopardy:rc:olmo3base",
    ]),
    ("Math / Code", [
        "gsm8k:olmo3base",
        "minerva_math:olmo3base",
        "codex_humaneval:bpb:olmo3base",
        "mbpp:bpb:olmo3base",
    ]),
    ("PPL", [
        "c4_100k:ppl",
    ]),
    ("Long context (RULER)", [
        "ruler_all__4096",
        "ruler_all__8192",
        "ruler_all__16384",
        "ruler_all__32768",
        "ruler_all__65536",
        "ruler_all__131072",
    ]),
]

KEY_TASKS = [t for _, tasks in KEY_TASK_GROUPS for t in tasks]

LOWER_IS_BETTER = {
    "c4_100k:ppl",
    "olmobase:easy:code:bpb",
    "codex_humaneval:bpb:olmo3base",
    "mbpp:bpb:olmo3base",
}

SCALE_ORDER = {"275M": 0, "810M": 1, "1.4B": 2}


def get_model_short_name(model_path: str) -> str:
    model_path = model_path.rstrip("/")
    if model_path.count("/") <= 2:
        return model_path.split("/")[-1]
    return model_path.split("/")[-2]


def parse_size_and_lr(name: str) -> tuple[str, str] | None:
    """Extract (size, lr) from model name like 'hybrid-small-lc-v3-275m-lr2e-4'."""
    m = re.search(r"(\d+\.?\d*[mb])-lr([\d.e\-]+)", name, re.IGNORECASE)
    if m:
        size = m.group(1).upper()
        return size, m.group(2)
    return None


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


def fmt_score(score: float, is_best: bool, is_worst: bool, gap: float | None, no_color: bool) -> str:
    if score > 1:
        s = f"{score:.2f}"
    else:
        s = f"{score:.3f}"
    # Add gap from best in brackets (skip for best itself)
    if gap is not None and not is_best:
        if score > 1:
            s += f" ({gap:+.2f})"
        else:
            s += f" ({gap:+.3f})"
    if no_color or (not is_best and not is_worst):
        return s
    if is_best:
        return f"\033[1;32m{s}\033[0m"  # bold green
    if is_worst:
        return f"\033[1;31m{s}\033[0m"  # bold red
    return s


def _plain_width(formatted: str) -> int:
    return len(re.sub(r"\033\[[^m]*m", "", formatted))


def print_table(
    by_size: dict[str, dict[str, dict[str, float]]],
    no_color: bool = False,
):
    """Print a table with columns = (size, lr) pairs, rows = tasks."""
    # Build column list: sorted by size then LR
    columns: list[tuple[str, str]] = []  # (size, lr)
    sizes = sorted(by_size.keys(), key=lambda s: SCALE_ORDER.get(s, 99))
    for size in sizes:
        lrs = sorted(by_size[size].keys(), key=lambda x: float(x))
        for lr in lrs:
            columns.append((size, lr))

    if not columns:
        print("No data to display.")
        return

    col_width = 18
    task_col_width = 30

    # Header row 1: size
    header1 = " " * task_col_width
    prev_size = None
    for size, lr in columns:
        sep = "||" if prev_size and size != prev_size else "  "
        header1 += sep + size.center(col_width)
        prev_size = size
    print(header1)

    # Header row 2: lr
    header2 = " " * task_col_width
    prev_size = None
    for size, lr in columns:
        sep = "||" if prev_size and size != prev_size else "  "
        header2 += sep + f"lr={lr}".center(col_width)
        prev_size = size
    print(header2)

    # Separator
    total_width = task_col_width + len(columns) * (col_width + 2)
    print("-" * total_width)

    # Rows grouped by task group
    for group_label, tasks in KEY_TASK_GROUPS:
        # Check if any task in this group has data
        has_data = False
        for task in tasks:
            for size, lr in columns:
                if task in by_size[size].get(lr, {}):
                    has_data = True
                    break
            if has_data:
                break
        if not has_data:
            continue

        # Group header
        print(f"\n{group_label}")
        print("-" * total_width)

        for task in tasks:
            # Collect scores for this row
            row_scores: list[float | None] = []
            for size, lr in columns:
                row_scores.append(by_size[size].get(lr, {}).get(task))

            # Skip if no data at all for this task
            if all(s is None for s in row_scores):
                continue

            # Find best and worst per size group
            best_per_size: dict[str, float | None] = {}
            worst_per_size: dict[str, float | None] = {}
            for size in sizes:
                size_scores = [
                    by_size[size].get(lr, {}).get(task)
                    for lr in sorted(by_size[size].keys(), key=lambda x: float(x))
                ]
                valid = [s for s in size_scores if s is not None]
                if len(valid) > 1:
                    if task in LOWER_IS_BETTER:
                        best_per_size[size] = min(valid)
                        worst_per_size[size] = max(valid)
                    else:
                        best_per_size[size] = max(valid)
                        worst_per_size[size] = min(valid)

            # Format row
            task_label = task[:task_col_width].ljust(task_col_width)
            row = task_label
            prev_size = None
            for idx, (size, lr) in enumerate(columns):
                sep = "||" if prev_size and size != prev_size else "  "
                score = row_scores[idx]
                if score is None:
                    cell = "—".center(col_width)
                else:
                    is_best = (best_per_size.get(size) is not None and score == best_per_size[size])
                    is_worst = (worst_per_size.get(size) is not None and score == worst_per_size[size])
                    # Gap from best: positive means worse for higher-is-better, negative for lower-is-better
                    gap = None
                    if best_per_size.get(size) is not None and not is_best:
                        if task in LOWER_IS_BETTER:
                            gap = score - best_per_size[size]  # positive = worse (higher)
                        else:
                            gap = score - best_per_size[size]  # negative = worse (lower)
                    cell_str = fmt_score(score, is_best, is_worst, gap, no_color)
                    pad = col_width - _plain_width(cell_str)
                    cell = " " * (pad // 2) + cell_str + " " * (pad - pad // 2)
                row += sep + cell
                prev_size = size
            print(row)


def main():
    parser = argparse.ArgumentParser(description="Summarize LR sweep results in a table.")
    parser.add_argument("--group", default="yashasbls-hybrid-small-evals-debug", help="Eval group name")
    parser.add_argument("--no-color", action="store_true", help="Disable color output")
    args = parser.parse_args()

    results = collect_results(args.group)
    if not results:
        print("No results found.")
        return

    print(f"Loaded results for {len(results)} models\n")

    # Organize: size -> lr -> {task: score}
    by_size: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))

    for model_name, tasks in results.items():
        parsed = parse_size_and_lr(model_name)
        if not parsed:
            print(f"  [skip] Could not parse: {model_name}")
            continue
        size, lr = parsed
        for task, score in tasks.items():
            by_size[size][lr][task] = score

    print_table(dict(by_size), no_color=args.no_color)


if __name__ == "__main__":
    main()
