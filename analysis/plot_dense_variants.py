#!/usr/bin/env python3
"""
Grouped bar charts comparing the q4b-base-dense variants against
Qwen3-4B-Base-yarn2-olmocore on RULER and HELMET, by context length.

Reads ruler_and_helmet.csv (RULER cols 4k..64k; HELMET cols "<L> avg").
Outputs: dense_variants_ruler.png, dense_variants_helmet.png
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "ruler_and_helmet.csv"

# label -> modelname in the CSV
ROWS = {
    "Qwen3-4B-Base-yarn2": "Qwen3-4B-Base-yarn2-olmocore",
    "dense-lr1.1e-4": "q4b-base-dense-lr1.1e-4_step2385",
    "dense-lr3.2e-4": "q4b-base-dense-8node_step2385",
    "dense-lr9.6e-4": "q4b-base-dense-lr9.6e-4_step2385",
}

COLORS = {
    "Qwen3-4B-Base-yarn2": "#d62728",
    "dense-lr1.1e-4": "#2ca02c",
    "dense-lr3.2e-4": "#1f77b4",
    "dense-lr9.6e-4": "#ff7f0e",
}

RULER_COLS = [("4k", "4k"), ("8k", "8k"), ("16k", "16k"), ("32k", "32k"), ("64k", "64k")]
HELMET_COLS = [("8K avg", "8k"), ("16K avg", "16k"), ("32k avg", "32k"), ("64k avg", "64k")]


def load():
    by_name = {}
    with open(CSV, newline="") as f:
        for row in csv.DictReader(f):
            by_name[row["modelname"]] = row
    return by_name


def val(row, col):
    v = (row.get(col) or "").strip()
    return float(v) if v not in ("", "-") else np.nan


def plot(by_name, cols, title, ylabel, outfile):
    xlabels = [x for _, x in cols]
    x = np.arange(len(cols))
    n = len(ROWS)
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (label, name) in enumerate(ROWS.items()):
        row = by_name[name]
        heights = [val(row, col) for col, _ in cols]
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(x + offset, heights, width, label=label, color=COLORS[label])
        ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Evaluation context length (tokens)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    # headroom so the legend (upper right) clears the tallest bars + value labels
    ymax = np.nanmax([val(by_name[n], c) for n in ROWS.values() for c, _ in cols])
    ax.set_ylim(top=ymax * 1.18)
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(ROOT / outfile, dpi=150)
    print(f"wrote {outfile}")


def main():
    by_name = load()
    plot(
        by_name,
        RULER_COLS,
        "Dense variants vs. Qwen3-4B-Base-yarn2: RULER",
        "RULER score",
        "dense_variants_ruler.png",
    )
    plot(
        by_name,
        HELMET_COLS,
        "Dense variants vs. Qwen3-4B-Base-yarn2: HELMET (avg)",
        "HELMET average",
        "dense_variants_helmet.png",
    )


if __name__ == "__main__":
    main()
