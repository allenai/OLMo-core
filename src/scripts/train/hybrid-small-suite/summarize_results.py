#!/usr/bin/env python3
"""Generate summary tables from downloaded olmo-eval results.

Usage:
    python3 src/scripts/train/hybrid-small-suite/summarize_results.py
"""

import json
import glob
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

# Key aggregate tasks grouped for display: (group_label, [task_names])
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

# Flat list for non-grouped uses
KEY_TASKS = [t for _, tasks in KEY_TASK_GROUPS for t in tasks]

# Tasks where lower scores are better. Everything else is higher-is-better.
LOWER_IS_BETTER = {
    "c4_100k:ppl",
    "olmobase:easy:code:bpb",
    "codex_humaneval:bpb:olmo3base",
    "mbpp:bpb:olmo3base",
}

# Model name -> (scale, stage) for sorting
SCALE_ORDER = {"275M": 0, "275m": 0, "810M": 1, "810m": 1, "1B": 2, "1b": 2, "1.4B": 3, "1.4b": 3, "7B": 4, "7b": 4}
STAGE_ORDER = {"pretrain": 0, "midtrain": 1, "long-context": 2, "baseline": 3}


def parse_model_info(model_name: str) -> tuple[str, str, str]:
    """Parse model short name into (scale, stage, display_name)."""
    name = model_name.lower()
    # Determine scale
    if "275m" in name:
        scale = "275M"
    elif "810m" in name:
        scale = "810M"
    elif "1.4b" in name:
        scale = "1.4B"
    elif "-1b" in name:
        scale = "1B"
    elif "7b" in name:
        scale = "7B"
    else:
        scale = "?"

    # Determine stage
    if "long-context" in name:
        stage = "long-context"
    elif "midtraining" in name:
        stage = "midtrain"
    elif "cx100" in name:
        stage = "pretrain"
    elif "olmo-2" in name or "olmo-3" in name or "olmo-hybrid" in name:
        stage = "baseline"
    else:
        stage = "?"

    return scale, stage, model_name


def get_model_short_name(model_path: str) -> str:
    """Extract a short model name from a full path or HF model ID."""
    model_path = model_path.rstrip("/")
    # HF-style: org/model-name (no deep path)
    if model_path.count("/") == 1:
        return model_path.split("/")[-1]
    # Weka-style: /weka/.../model-name/step-hf
    return model_path.split("/")[-2]


def collect_results(group: str | None = None) -> dict[str, dict[str, float]]:
    """Collect all results keyed by model short name -> task -> score."""
    model_tasks = defaultdict(dict)
    search_dir = RESULTS_DIR / group if group else RESULTS_DIR
    files = glob.glob(str(search_dir / "**" / "metrics.json"), recursive=True)

    for f in files:
        with open(f) as fh:
            m = json.load(fh)
        # Skip results that belong to a different experiment group
        if group and m.get("experiment_group", "") != group:
            continue
        model_path = m.get("config", {}).get("provider", {}).get("model", "?")
        basename = get_model_short_name(model_path)
        summary = m.get("summary", {})
        for task, info in summary.items():
            score = info.get("score")
            if score is not None:
                model_tasks[basename][task] = score

    return dict(model_tasks)


def fmt_score(score: float, is_best_global: bool, is_best_group: bool, no_color: bool) -> str:
    """Format a score with optional color.
    Green (bold) = best across entire row (takes priority).
    Yellow (bold) = best within scale group on this row.
    """
    if score > 1:  # PPL-style: lower is better
        s = f"{score:.2f}"
    else:
        s = f"{score:.3f}"
    if no_color:
        return s
    if is_best_global:
        return f"\033[1;32m{s}\033[0m"   # bold green
    if is_best_group:
        return f"\033[1;33m{s}\033[0m"   # bold yellow
    return s


def fmt_bold(s: str, no_color: bool) -> str:
    if no_color:
        return s
    return f"\033[1m{s}\033[0m"


def _plain_width(formatted: str) -> int:
    """Length of string after stripping ANSI escape codes."""
    import re
    return len(re.sub(r"\033\[[^m]*m", "", formatted))


def print_table(results: dict[str, dict[str, float]], tasks: list[str], no_color: bool = False, all_tasks: bool = False):
    """Print a formatted table with two-row headers grouped by scale."""
    sort_key = lambda m: (SCALE_ORDER.get(parse_model_info(m)[0], 99), STAGE_ORDER.get(parse_model_info(m)[1], 99))
    # Put non-7B first (sorted by scale/stage), then 7B at the end
    non_7b = sorted([m for m in results if parse_model_info(m)[0] != "7B"], key=sort_key)
    is_7b = sorted([m for m in results if parse_model_info(m)[0] == "7B"], key=sort_key)
    models = non_7b + is_7b
    baselines = [m for m in models if parse_model_info(m)[1] == "baseline"]

    active_tasks = [t for t in tasks if any(t in results[m] for m in models)]
    if not active_tasks:
        print("No matching tasks found.")
        return

    scales = {}  # scale -> list of model indices
    for i, m in enumerate(models):
        scale = parse_model_info(m)[0]
        scales.setdefault(scale, []).append(i)

    stage_names = {}
    for m in models:
        scale, stage, _ = parse_model_info(m)
        if scale == "7B" and stage != "baseline":
            stage = "hybrid-" + stage
        stage_names[m] = stage

    task_width = max(len(t) for t in active_tasks)
    col_width = max(10, max(len(stage_names[m]) for m in models))
    vsep = " | "
    thick_vsep = " || "

    # Scale groups for vertical separators: after each scale group
    # Use thick separator before 7B section
    def needs_vsep(i: int) -> str | None:
        """Return separator type or None."""
        if i == len(models) - 1:
            return None
        scale_i = parse_model_info(models[i])[0]
        scale_next = parse_model_info(models[i + 1])[0]
        if scale_i != scale_next:
            if scale_next == "7B":
                return "thick"
            return "thin"
        return None

    # Build header rows: suite, scale, arch, stage
    header_suite = f"{'':<{task_width}}"
    header1 = f"{'':<{task_width}}"
    header_arch = f"{'':<{task_width}}"
    header2 = f"{'Task':<{task_width}}"

    for i, m in enumerate(models):
        scale = parse_model_info(m)[0]
        stage = stage_names[m]
        name_lower = m.lower()
        is_hybrid = "hybrid" in name_lower or m not in baselines
        arch_label = "hybrid" if is_hybrid else "trans."

        # Suite label
        if m not in baselines:
            suite_label = "hybrid-small"
        elif "olmo-2" in name_lower:
            suite_label = "OLMo-2"
        elif "olmo-3" in name_lower:
            suite_label = "OLMo-3"
        elif "olmo-hybrid" in name_lower:
            suite_label = "OLMo-3"
        else:
            suite_label = ""

        header_suite += f"  {suite_label:>{col_width}}"
        header1 += f"  {scale:>{col_width}}"
        header_arch += f"  {arch_label:>{col_width}}"
        header2 += f"  {stage:>{col_width}}"

        sep_type = needs_vsep(i)
        if sep_type == "thick":
            header_suite += thick_vsep
            header1 += thick_vsep
            header_arch += thick_vsep
            header2 += thick_vsep
        elif sep_type == "thin":
            header_suite += vsep
            header1 += vsep
            header_arch += vsep
            header2 += vsep

    sep_line = "-" * _plain_width(header2)
    print(header_suite)
    print(header1)
    print(header_arch)
    print(header2)
    print(sep_line)

    active_task_set = set(active_tasks)

    if all_tasks:
        groups = [("All tasks", active_tasks)]
    else:
        groups = [(label, [t for t in task_list if t in active_task_set])
                  for label, task_list in KEY_TASK_GROUPS]
        groups = [(label, tasks) for label, tasks in groups if tasks]

    for group_label, group_tasks in groups:
        if not all_tasks:
            print(f"\n{group_label}")
            print(sep_line)
        for task in group_tasks:
            scores = {m: results[m].get(task) for m in models}
            valid = {m: s for m, s in scores.items() if s is not None}

            # Global best (green) — exclude 7B baselines from consideration
            lower_is_better = task in LOWER_IS_BETTER
            valid_no_7b = {m: s for m, s in valid.items() if parse_model_info(m)[0] != "7B"}
            if valid_no_7b:
                best_global = min(valid_no_7b, key=valid_no_7b.__getitem__) if lower_is_better else max(valid_no_7b, key=valid_no_7b.__getitem__)
            else:
                best_global = None

            # Per-scale best (yellow) — only among non-baseline models
            best_in_scale: dict[str, str] = {}  # scale -> best model
            for scale, idxs in scales.items():
                scale_models = [models[j] for j in idxs if models[j] not in baselines]
                scale_valid = {m: scores[m] for m in scale_models if scores[m] is not None}
                if scale_valid:
                    best_in_scale[scale] = min(scale_valid, key=scale_valid.__getitem__) if lower_is_better else max(scale_valid, key=scale_valid.__getitem__)

            row = f"{task:<{task_width}}"
            for i, m in enumerate(models):
                is_baseline = m in baselines
                score = scores[m]
                scale = parse_model_info(m)[0]
                is_best_global = (m == best_global)
                is_best_group = (not is_baseline) and (m == best_in_scale.get(scale)) and not is_best_global

                if score is None:
                    cell = f"{'—':>{col_width}}"
                    if is_baseline and not no_color:
                        cell = fmt_bold(cell, no_color)
                    row += f"  {cell}"
                else:
                    plain = f"{score:.2f}" if score > 1 else f"{score:.3f}"
                    formatted = fmt_score(score, is_best_global, is_best_group, no_color)
                    if is_baseline and not is_best_global and not is_best_group:
                        formatted = fmt_bold(plain, no_color)
                    ansi_extra = len(formatted) - len(plain)
                    row += f"  {formatted:>{col_width + ansi_extra}}"
                sep_type = needs_vsep(i)
                if sep_type == "thick":
                    row += thick_vsep
                elif sep_type == "thin":
                    row += vsep
            print(row)


def print_csv(results: dict[str, dict[str, float]], tasks: list[str]):
    """Print CSV for easy pasting into spreadsheets."""
    models = sorted(results.keys(), key=lambda m: (
        SCALE_ORDER.get(parse_model_info(m)[0], 99),
        STAGE_ORDER.get(parse_model_info(m)[1], 99),
    ))

    active_tasks = [t for t in tasks if any(t in results[m] for m in models)]

    display_names = {}
    for m in models:
        scale, stage, _ = parse_model_info(m)
        display_names[m] = f"{scale} {stage}"

    # Header
    print("Task," + ",".join(display_names[m] for m in models))

    # Rows
    for task in active_tasks:
        row = [task]
        for m in models:
            score = results[m].get(task)
            if score is None:
                row.append("")
            elif isinstance(score, float):
                row.append(f"{score:.2f}" if score > 1 else f"{score:.3f}")
            else:
                row.append(str(score))
        print(",".join(row))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", action="store_true", help="Output as CSV")
    parser.add_argument("--no-color", action="store_true", help="Disable color output")
    parser.add_argument("--all-tasks", action="store_true", help="Show all tasks, not just key ones")
    parser.add_argument("--group", type=str, default=None, help="Only summarize results from this group subdirectory")
    parser.add_argument("--baseline-group", type=str, default=None, help="Separate group to use as baselines")
    args = parser.parse_args()

    results = collect_results(group=args.group)
    if args.baseline_group:
        baseline_results = collect_results(group=args.baseline_group)
        # Merge baseline results (don't overwrite test group models)
        for model, tasks in baseline_results.items():
            if model not in results:
                results[model] = tasks
            else:
                # Only fill in missing tasks from baseline
                for task, score in tasks.items():
                    results[model].setdefault(task, score)
    print(f"Loaded results for {len(results)} models\n")

    if args.all_tasks:
        all_tasks = sorted(set(t for tasks in results.values() for t in tasks))
        tasks = all_tasks
    else:
        tasks = KEY_TASKS

    if args.csv:
        print_csv(results, tasks)
    else:
        print_table(results, tasks, no_color=args.no_color, all_tasks=args.all_tasks)


if __name__ == "__main__":
    main()
