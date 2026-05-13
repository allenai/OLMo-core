#!/usr/bin/env python3
"""Generate summary tables from downloaded olmo-eval results.

Usage:
    python3 src/scripts/train/hybrid-small-suite/summarize_results.py
"""

import json
import glob
import re
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

# Model name -> (scale, stage) for sorting
SCALE_ORDER = {"275M": 0, "275m": 0, "810M": 1, "810m": 1, "1B": 2, "1b": 2, "1.4B": 3, "1.4b": 3, "7B": 4, "7b": 4}
STAGE_ORDER = {"pretrain": 0, "midtrain": 1, "long-context": 2, "baseline": 3, "hybrid-baseline": 4}
BASELINE_STAGES = {"baseline", "hybrid-baseline"}

# Tasks where lower scores are better (perplexity, bits-per-byte).
LOWER_IS_BETTER_PATTERNS = ("ppl", "bpb")


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
    elif "olmo-hybrid" in name:
        stage = "hybrid-baseline"
    elif "olmo-2" in name or "olmo-3" in name or "olmo3" in name:
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


def collect_results() -> dict[str, dict[str, float]]:
    """Collect all results keyed by model short name -> task -> score."""
    model_tasks = defaultdict(dict)
    files = glob.glob(str(RESULTS_DIR / "**" / "metrics.json"), recursive=True)

    for f in files:
        with open(f) as fh:
            m = json.load(fh)
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


def fmt_underline(s: str, no_color: bool) -> str:
    if no_color:
        return s
    return f"\033[4m{s}\033[0m"


def _plain_width(formatted: str) -> int:
    """Length of string after stripping ANSI escape codes."""
    return len(re.sub(r"\033\[[^m]*m", "", formatted))


def print_table(results: dict[str, dict[str, float]], tasks: list[str], no_color: bool = False, all_tasks: bool = False):
    """Print a formatted table with two-row headers grouped by scale."""
    sort_key = lambda m: (SCALE_ORDER.get(parse_model_info(m)[0], 99), STAGE_ORDER.get(parse_model_info(m)[1], 99))
    baselines = sorted([m for m in results if parse_model_info(m)[1] in BASELINE_STAGES], key=sort_key)
    non_baselines = sorted([m for m in results if parse_model_info(m)[1] not in BASELINE_STAGES], key=sort_key)
    models = baselines + non_baselines
    baseline_set = set(baselines)

    active_tasks = [t for t in tasks if any(t in results[m] for m in models)]
    if not active_tasks:
        print("No matching tasks found.")
        return

    scales = {}  # scale -> list of model indices
    for i, m in enumerate(models):
        scale = parse_model_info(m)[0]
        scales.setdefault(scale, []).append(i)

    stage_names = {m: parse_model_info(m)[1] for m in models}

    task_width = max(len(t) for t in active_tasks)
    # Per-column width: baselines just need room for score, non-baselines need score + delta
    col_widths = [max(16, len(stage_names[m])) if stage_names[m] not in BASELINE_STAGES else max(6, len(stage_names[m])) for m in models]
    vsep = " | "

    # Scale groups for vertical separators: after each scale group (and after baselines)
    # A separator goes after baselines and after each scale boundary
    def needs_vsep(i: int) -> bool:
        if i == len(models) - 1:
            return False
        if baselines and i == len(baselines) - 1:
            return True
        scale_i = parse_model_info(models[i])[0]
        scale_next = parse_model_info(models[i + 1])[0]
        return scale_i != scale_next

    # Build header row 1: scale labels, centered over their column group
    header1 = " " * task_width
    # row 2: stage labels
    header2 = f"{'Task':<{task_width}}"
    for i, m in enumerate(models):
        scale = parse_model_info(m)[0]
        stage = stage_names[m]
        # For row1, only print scale at start of each group
        idxs = scales[scale]
        if i == idxs[0]:
            span_width = sum(col_widths[j] + 2 for j in idxs) + len(vsep) * sum(1 for j in idxs[:-1] if needs_vsep(j))
            label1 = scale[:span_width]
            centered = label1.center(span_width)
            if scale == "1B" and not no_color:
                centered = fmt_underline(centered.strip(), no_color).center(span_width + 8)  # 8 = ANSI codes len
            header1 += centered
        stage_hdr = f"{stage:>{col_widths[i]}}"
        if scale == "1B" and not no_color:
            stage_hdr = fmt_underline(stage_hdr.strip(), no_color)
            stage_hdr = f"{stage_hdr:>{col_widths[i] + 8}}"
        header2 += f"  {stage_hdr}"
        if needs_vsep(i):
            header1 += vsep
            header2 += vsep

    sep_line = "-" * _plain_width(header2)
    print(header1)
    print(header2)
    print(sep_line)

    active_task_set = set(active_tasks)

    if all_tasks:
        groups = [("All tasks", active_tasks)]
    else:
        groups = [(label, [t for t in task_list if t in active_task_set])
                  for label, task_list in KEY_TASK_GROUPS]
        groups = [(label, grp_tasks) for label, grp_tasks in groups if grp_tasks]

    for group_label, group_tasks in groups:
        if not all_tasks:
            print(f"\n{group_label}")
            print(sep_line)
        for task in group_tasks:
            scores = {m: results[m].get(task) for m in models}
            valid = {m: s for m, s in scores.items() if s is not None}

            lower_is_better = any(p in task for p in LOWER_IS_BETTER_PATTERNS)

            # 1B baseline score (reference for deltas)
            baseline_1b_score = None
            for m in baselines:
                if parse_model_info(m)[0] == "1B" and scores[m] is not None:
                    baseline_1b_score = scores[m]
                    break

            # Global best among test models (1B baseline + hybrids, excluding 7B baselines)
            test_valid = {m: s for m, s in valid.items() if parse_model_info(m)[0] != "7B" or m not in baseline_set}
            # Include 1B baseline in the competition
            if baseline_1b_score is not None:
                for m in baselines:
                    if parse_model_info(m)[0] == "1B":
                        test_valid[m] = baseline_1b_score
            if test_valid:
                best_global = min(test_valid, key=test_valid.__getitem__) if lower_is_better else max(test_valid, key=test_valid.__getitem__)
            else:
                best_global = None

            # Per-scale best (yellow) — only among non-baseline models
            best_in_scale: dict[str, str] = {}  # scale -> best model
            for scale, idxs in scales.items():
                scale_models = [models[j] for j in idxs if models[j] not in baseline_set]
                scale_valid = {m: scores[m] for m in scale_models if scores[m] is not None}
                if scale_valid:
                    best_in_scale[scale] = min(scale_valid, key=scale_valid.__getitem__) if lower_is_better else max(scale_valid, key=scale_valid.__getitem__)

            row = f"{task:<{task_width}}"
            for i, m in enumerate(models):
                is_baseline = m in baseline_set
                score = scores[m]
                scale = parse_model_info(m)[0]
                is_best_global = (m == best_global)
                is_best_group = (not is_baseline) and (m == best_in_scale.get(scale)) and not is_best_global

                if score is None:
                    cell = f"{'—':>{col_widths[i]}}"
                    if is_baseline and not no_color:
                        cell = fmt_bold(cell, no_color)
                    if scale == "1B" and is_baseline and not no_color:
                        cell = fmt_underline(f"{'—':>{col_widths[i]}}", no_color)
                    row += f"  {cell}"
                else:
                    plain = f"{score:.2f}" if score > 1 else f"{score:.3f}"
                    formatted = fmt_score(score, is_best_global, is_best_group, no_color)
                    if is_baseline and not is_best_global and not is_best_group:
                        formatted = fmt_bold(plain, no_color)
                    # Underline the 1B reference baseline
                    if scale == "1B" and is_baseline and not no_color:
                        if is_best_global:
                            formatted = f"\033[4;1;32m{plain}\033[0m"
                        else:
                            formatted = f"\033[4;1m{plain}\033[0m"
                    # Build delta suffix for non-baseline models
                    delta_plain = ""
                    delta_formatted = ""
                    if not is_baseline and baseline_1b_score is not None:
                        delta = score - baseline_1b_score
                        if lower_is_better:
                            delta = -delta  # flip so positive = good
                        sign = "+" if delta >= 0 else ""
                        delta_plain = f" ({sign}{delta:.3f})" if abs(score - baseline_1b_score) <= 1 else f" ({sign}{delta:.2f})"
                        if not no_color:
                            # color just the parens part
                            delta_inner = delta_plain[1:]  # strip leading space
                            if delta > 0:
                                delta_formatted = f" \033[32m{delta_inner}\033[0m"
                            elif delta < 0:
                                delta_formatted = f" \033[31m{delta_inner}\033[0m"
                            else:
                                delta_formatted = delta_plain
                        else:
                            delta_formatted = delta_plain
                    # Right-align the full cell (score + delta) within col_width
                    full_plain = plain + delta_plain
                    full_formatted = formatted + delta_formatted
                    pad = col_widths[i] - len(full_plain)
                    if pad > 0:
                        cell = " " * pad + full_formatted
                    else:
                        cell = full_formatted
                    row += f"  {cell}"
                if needs_vsep(i):
                    row += vsep
            print(row)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-color", action="store_true", help="Disable color output")
    parser.add_argument("--all-tasks", action="store_true", help="Show all tasks, not just key ones")
    args = parser.parse_args()

    results = collect_results()
    print(f"Loaded results for {len(results)} models\n")

    if args.all_tasks:
        all_tasks = sorted(set(t for tasks in results.values() for t in tasks))
        tasks = all_tasks
    else:
        tasks = KEY_TASKS

    print_table(results, tasks, no_color=args.no_color, all_tasks=args.all_tasks)


if __name__ == "__main__":
    main()
