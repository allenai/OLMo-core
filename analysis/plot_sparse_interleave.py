#!/usr/bin/env python3
"""
Plot the interleaved sparse-landmark runs (1/2/3/4 sparse layers per regular layer)
against the dense and full-sparse-landmark baselines, on RULER and HELMET.

Series ("Nsparse" = N sparse-landmark layers per 1 regular/dense layer):
    1sparse  -> qwen3-4b-interleaved-alt-test_step2385   (strict alternating)
    2sparse  -> q4b-interleaved-2sparse-1reg_step2385
    3sparse  -> q4b-interleaved-3sparse-1reg_step2385
    4sparse  -> q4b-interleaved-4sparse-1reg_step2385

Baselines:
    dense    -> q4b-base-dense-8node_step2385
    landmark -> q4b-base-sparse-landmark-8node_step2385   (fully sparse landmark)

Reads ruler_and_helmet.csv (RULER cols 4k..64k; HELMET cols "<L> avg").
Outputs: sparse_interleave_ruler.png, sparse_interleave_helmet.png
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "ruler_and_helmet.csv"

# label -> modelname in the CSV
ROWS = {
    "1sparse": "qwen3-4b-interleaved-alt-test_step2385",
    "2sparse": "q4b-interleaved-2sparse-1reg_step2385",
    "3sparse": "q4b-interleaved-3sparse-1reg_step2385",
    "4sparse": "q4b-interleaved-4sparse-1reg_step2385",
    "dense baseline": "q4b-base-dense-8node_step2385",
    "sparse landmark baseline": "q4b-base-sparse-landmark-8node_step2385",
}

STYLE = {
    "1sparse": dict(color="#2ca02c", ls="-", marker="o"),
    "2sparse": dict(color="#ff7f0e", ls="-", marker="o"),
    "3sparse": dict(color="#d62728", ls="-", marker="o"),
    "4sparse": dict(color="#9467bd", ls="-", marker="o"),
    "dense baseline": dict(color="#1f77b4", ls="--", marker="s", zorder=5),
    "sparse landmark baseline": dict(color="black", ls=":", marker="x", zorder=5),
}

RULER_COLS = [("4k", 4), ("8k", 8), ("16k", 16), ("32k", 32), ("64k", 64)]
HELMET_COLS = [("8K avg", 8), ("16K avg", 16), ("32k avg", 32), ("64k avg", 64)]


def load():
    by_name = {}
    with open(CSV, newline="") as f:
        for row in csv.DictReader(f):
            by_name[row["modelname"]] = row
    return by_name


def series(by_name, cols):
    data = {}
    for label, name in ROWS.items():
        row = by_name[name]
        pts = []
        for col, length in cols:
            v = (row.get(col) or "").strip()
            if v not in ("", "-"):
                pts.append((length, float(v)))
        data[label] = pts
    return data


def plot(data, title, ylabel, outfile):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for label in ROWS:
        pts = data[label]
        if not pts:
            continue
        ax.plot([x for x, _ in pts], [y for _, y in pts], label=label, **STYLE[label])
    ax.set_xscale("log", base=2)
    all_x = sorted({x for pts in data.values() for x, _ in pts})
    ax.set_xticks(all_x)
    ax.set_xticklabels([f"{x}k" for x in all_x])
    ax.set_xlabel("Evaluation context length (tokens)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT / outfile, dpi=150)
    print(f"wrote {outfile}")


def main():
    by_name = load()
    plot(
        series(by_name, RULER_COLS),
        "Interleaved sparse-landmark: RULER",
        "RULER score",
        "sparse_interleave_ruler.png",
    )
    plot(
        series(by_name, HELMET_COLS),
        "Interleaved sparse-landmark: HELMET (avg)",
        "HELMET average",
        "sparse_interleave_helmet.png",
    )


if __name__ == "__main__":
    main()
