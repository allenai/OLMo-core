#!/usr/bin/env python3
"""Show RULER subtask breakdown side-by-side for OLMo 3 7B vs Hybrid 7B.

Usage:
    python3 src/scripts/train/hybrid-small-suite/ruler_breakdown.py
"""

from __future__ import annotations

import json
import glob
import re
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

MODEL_MAPPING = {
    "OLMo 3 7B": "Olmo-3-1025-7B",
    "Hybrid 7B": "Olmo-Hybrid-7B",
}

CONTEXT_LENGTHS = ["4096", "8192", "16384", "32768", "65536", "131072"]
CONTEXT_LABELS = ["4k", "8k", "16k", "32k", "64k", "128k"]

# Subtask display order
SUBTASKS = [
    "niah_s_1", "niah_s_2", "niah_s_3",
    "niah_mk_1", "niah_mk_2", "niah_mk_3",
    "niah_mq", "niah_mv",
    "cwe", "fwe", "vt",
    "qa_1", "qa_2",
    "all",
]


def collect_ruler_results() -> dict[str, dict[str, float]]:
    """Collect RULER results keyed by model short name -> task -> score."""
    model_tasks: dict[str, dict[str, float]] = defaultdict(dict)
    files = glob.glob(str(RESULTS_DIR / "**" / "metrics.json"), recursive=True)

    for f in files:
        with open(f) as fh:
            m = json.load(fh)
        model_path = m.get("config", {}).get("provider", {}).get("model", "?")
        if "7b" not in model_path.lower() and "7B" not in model_path:
            continue
        name = model_path.rstrip("/").split("/")[-1]
        if name.endswith("-hf"):
            name = model_path.rstrip("/").split("/")[-2]
        summary = m.get("summary", {})
        for task, info in summary.items():
            if "ruler" in task:
                score = info.get("score")
                if score is not None:
                    model_tasks[name][task] = score

    return dict(model_tasks)


def _plain_width(s: str) -> int:
    return len(re.sub(r"\033\[[^m]*m", "", s))


def color_score(score: float, ref_score: float | None) -> str:
    """Color a score based on comparison to reference. Green if close/better, red if much worse."""
    s = f"{score * 100:5.1f}"
    if ref_score is None:
        return s
    diff = score - ref_score
    if abs(diff) < 0.02:
        return f"\033[32m{s}\033[0m"  # green — match
    elif diff > 0:
        return f"\033[1;32m{s}\033[0m"  # bold green — better
    elif abs(diff) < 0.1:
        return f"\033[33m{s}\033[0m"  # yellow — small gap
    else:
        return f"\033[31m{s}\033[0m"  # red — big gap


def main():
    results = collect_ruler_results()

    # Resolve model names
    model_names: dict[str, str] = {}
    for label, substr in MODEL_MAPPING.items():
        for name in results:
            if substr in name:
                model_names[label] = name
                break

    print("RULER Subtask Breakdown: OLMo 3 7B vs Hybrid 7B")
    print("=" * 90)
    print(f"Models: {model_names}")
    print()

    olmo3_name = model_names.get("OLMo 3 7B")
    hybrid_name = model_names.get("Hybrid 7B")
    olmo3 = results.get(olmo3_name, {}) if olmo3_name else {}
    hybrid = results.get(hybrid_name, {}) if hybrid_name else {}

    subtask_width = 12
    model_col = 7
    hybrid_col = 16  # score + " (+xx.x)"
    gap = "    "  # gap between side-by-side tables

    def build_table(ctx: str, ctx_label: str) -> list[tuple[str, int]]:
        """Build a table for one context length, returned as list of (line, plain_width) tuples."""
        lines: list[tuple[str, int]] = []
        title = f"Context: {ctx_label}"
        header = f"{'Subtask':<{subtask_width}}  {'OLMo3':>{model_col}}  {'Hybrid':>{hybrid_col}}"
        col_width = len(header)
        lines.append((f"\033[1m{title}\033[0m", len(title)))
        lines.append((header, col_width))
        lines.append(("-" * col_width, col_width))

        for subtask in SUBTASKS:
            task_key = f"ruler_{subtask}__{ctx}"
            o3_score = olmo3.get(task_key)
            hy_score = hybrid.get(task_key)

            label = subtask if subtask != "all" else "\033[1mall\033[0m"
            row = f"{label:<{subtask_width + (len(label) - _plain_width(label))}}"

            if o3_score is not None:
                row += f"  {o3_score * 100:>{model_col}.1f}"
            else:
                row += f"  {'—':>{model_col}}"

            if hy_score is not None:
                score_str = f"{hy_score * 100:.1f}"
                # Build delta in brackets
                if o3_score is not None:
                    delta = (hy_score - o3_score) * 100
                    sign = "+" if delta >= 0 else ""
                    delta_plain = f"({sign}{delta:.1f})"
                    if delta < -5:
                        delta_colored = f"\033[31m{delta_plain}\033[0m"
                    elif delta < -1:
                        delta_colored = f"\033[33m{delta_plain}\033[0m"
                    elif delta > 1:
                        delta_colored = f"\033[32m{delta_plain}\033[0m"
                    else:
                        delta_colored = delta_plain
                    full_plain = f"{score_str} {delta_plain}"
                    colored = color_score(hy_score, o3_score)
                    full_formatted = f"{colored} {delta_colored}"
                else:
                    full_plain = score_str
                    full_formatted = score_str
                pad = hybrid_col - len(full_plain)
                row += "  " + " " * max(0, pad) + full_formatted
            else:
                row += f"  {'—':>{hybrid_col}}"

            lines.append((row, _plain_width(row)))
        return lines

    # Build all tables
    tables = []
    for ctx, ctx_label in zip(CONTEXT_LENGTHS, CONTEXT_LABELS):
        tables.append(build_table(ctx, ctx_label))

    # Determine fixed column width (max plain width across all lines in all tables)
    table_width = max(pw for table in tables for _, pw in table)

    # Print in 2 rows × 3 cols
    for row_idx in range(2):
        row_tables = tables[row_idx * 3 : row_idx * 3 + 3]
        max_lines = max(len(t) for t in row_tables)
        # Pad shorter tables
        for t in row_tables:
            while len(t) < max_lines:
                t.append(("", 0))
        # Print side by side
        for line_idx in range(max_lines):
            parts = []
            for t in row_tables:
                line, pw = t[line_idx]
                parts.append(line + " " * (table_width - pw))
            print(gap.join(parts))
        print()


if __name__ == "__main__":
    main()
