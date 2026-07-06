#!/usr/bin/env python3
"""
Plot landmark top-k retrieval evals for q4b-base-fast-landmark.

The landmark inference procedure decodes by attending to only the top-k landmark
*blocks* (block_size = 64 tokens). We summarize a run by its **effective context
window %** -- the fraction of the eval length the model can actually attend to:

    eff% = 64 * top_k / eval_len_tokens   (== top_k / n_blocks)

Two experiment designs are plotted together:

* **Constant-fraction families** (10 / 25 / 50 / 75%): top_k scales with length so
  the effective window stays fixed. 100% is the same checkpoint with no top-k cap
  (full landmark attention), read from the local baseline result CSVs.
* **Constant-budget series**: a fixed top_k regardless of length --
  ``top_k=1`` (the degenerate ~0.1-2% extreme), the low-budget family
  ``top_k=2,3,4,5`` (RULER only), and ``top_k=64`` (one block-set; effective
  window shrinks 100%->6.25% as length grows from 4k to 64k).

Inputs (repo root):
    ruler_topk.csv      RULER top-k runs; names carry "_<len>k_tk<topk>[_tag]".
    ruler_results.csv   baseline RULER (no-suffix q4b-base-fast-landmark row is 100%)
                        plus constant-budget flat rows "<baseline>_tk<topk>" whose
                        single row holds the score at every length.
    helmet_topk.csv     HELMET top-k runs (helmet_average per experiment).
    helmet_results.csv  baseline HELMET; the no-suffix row is 100%.

Outputs: landmark_ruler_eff_ctx.png, landmark_helmet_eff_ctx.png
"""
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
BLOCK = 64
BASELINE = "q4b-base-fast-landmark-8node_step2385"  # no-suffix => 100% effective ctx
SUFFIX_RE = re.compile(r"_(\d+)k_tk(\d+)(?:_\w+)?$")  # optional trailing tag, e.g. _fixsat
FLAT_RE = re.compile(re.escape(BASELINE) + r"_tk(\d+)$")  # constant-budget flat rows
FAMILIES = [10, 25, 50, 75]  # constant-fraction eff% families (top_k runs snapped to these)
FLAT_BUDGETS = [2, 3, 4, 5, 10, 16, 32]  # constant-budget top_k series (from flat ruler_results rows)

# Series draw order + style. eff% families are solid; 100% dashed; constant-budget dotted/dash-dot.
SERIES = {
    "top_k=1": dict(color="black", ls=":", marker="x", zorder=5),
    # low-budget flat family: RdPu ramp (light->dark = increasing budget), distinct from families.
    "top_k=2": dict(color="#fa9fb5", ls=":", marker="x", zorder=5),
    "top_k=3": dict(color="#f768a1", ls=":", marker="x", zorder=5),
    "top_k=4": dict(color="#dd3497", ls=":", marker="x", zorder=5),
    "top_k=5": dict(color="#ae017e", ls=":", marker="x", zorder=5),
    # mid-budget flat family: teal ramp (light->dark = increasing budget), distinct from families.
    "top_k=10": dict(color="#80cdc1", ls=":", marker="x", zorder=5),
    "top_k=16": dict(color="#35978f", ls=":", marker="x", zorder=5),
    "top_k=32": dict(color="#01665e", ls=":", marker="x", zorder=5),
    "10%": dict(color="#9467bd", ls="-", marker="o"),
    "25%": dict(color="#d62728", ls="-", marker="o"),
    "50%": dict(color="#ff7f0e", ls="-", marker="o"),
    "75%": dict(color="#2ca02c", ls="-", marker="o"),
    "100% (full attn)": dict(color="#1f77b4", ls="--", marker="o"),
    "top_k=64 (flat)": dict(color="#8c564b", ls="-.", marker="s", zorder=5),
}


def classify(len_k: int, top_k: int) -> "list[str]":
    """Series labels a (len, top_k) run belongs to (a run can be both an eff% family and flat)."""
    labels = []
    frac = top_k / (len_k * 1024 / BLOCK)  # effective fraction = top_k / n_blocks
    for fam in FAMILIES:
        if abs(frac - fam / 100) <= 0.015:
            labels.append(f"{fam}%")
    if top_k == 1:
        labels.append("top_k=1")
    if top_k == 64:
        labels.append("top_k=64 (flat)")
    return labels


def add(data, labels, len_k, score):
    if score in ("", "-", None):
        return
    for label in labels:
        data.setdefault(label, {})[len_k] = float(score)


def load_ruler():
    """RULER: {series -> {len_k -> score}}."""
    data: dict = {}
    with open(ROOT / "ruler_topk.csv", newline="") as f:
        for row in csv.DictReader(f):
            m = SUFFIX_RE.search(row["modelname"])
            if not m:
                continue
            len_k, top_k = int(m.group(1)), int(m.group(2))
            add(data, classify(len_k, top_k), len_k, row.get(f"{len_k}k", ""))
    with open(ROOT / "ruler_results.csv", newline="") as f:
        reader = csv.DictReader(f)
        len_cols = [c for c in (reader.fieldnames or []) if c.endswith("k") and c != "modelname"]
        for row in reader:
            name = row["modelname"]
            if name == BASELINE:
                for c in len_cols:
                    add(data, ["100% (full attn)"], int(c[:-1]), row[c])
            elif (m := FLAT_RE.match(name)) and int(m.group(1)) in FLAT_BUDGETS:
                for c in len_cols:
                    add(data, [f"top_k={int(m.group(1))}"], int(c[:-1]), row[c])
    return data


def load_helmet():
    """HELMET: {series -> {len_k -> helmet_average}}."""
    data: dict = {}
    with open(ROOT / "helmet_topk.csv", newline="") as f:
        for row in csv.DictReader(f):
            m = SUFFIX_RE.search(row["modelname"])
            if not m:
                continue
            len_k, top_k = int(m.group(1)), int(m.group(2))
            add(data, classify(len_k, top_k), len_k, row["helmet_average"])
    label_len = {"8K avg": 8, "16K avg": 16, "32k avg": 32, "64k avg": 64}
    with open(ROOT / "helmet_results.csv", newline="") as f:
        for row in csv.DictReader(f):
            name = row["modelname"]
            if name == BASELINE:
                for label, len_k in label_len.items():
                    add(data, ["100% (full attn)"], len_k, row.get(label, ""))
            elif (m := FLAT_RE.match(name)) and int(m.group(1)) in FLAT_BUDGETS:
                for label, len_k in label_len.items():
                    add(data, [f"top_k={int(m.group(1))}"], len_k, row.get(label, ""))
    return data


def plot(data, title, outfile):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for label, style in SERIES.items():
        if label not in data:
            continue
        pts = sorted(data[label].items())
        ax.plot([k for k, _ in pts], [v for _, v in pts], label=label, **style)
    ax.set_xscale("log", base=2)
    all_x = sorted({k for d in data.values() for k in d})
    ax.set_xticks(all_x)
    ax.set_xticklabels([f"{k}k" for k in all_x])
    ax.set_xlabel("Evaluation context length (tokens)")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Effective context window", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT / outfile, dpi=150)
    print(f"wrote {outfile}")


def main():
    plot(
        load_ruler(),
        "q4b-base-fast-landmark: RULER vs. effective context window",
        "landmark_ruler_eff_ctx.png",
    )
    plot(
        load_helmet(),
        "q4b-base-fast-landmark: HELMET vs. effective context window",
        "landmark_helmet_eff_ctx.png",
    )


if __name__ == "__main__":
    main()
