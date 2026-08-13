"""Paper-ready cut of the model-scale figure: the four task panels in a single row.

Drops reordering and the aggregate gap panel from `make_fig_scale.py`, leaving two low-CTC tasks
(HotpotQA, BEIR FiQA) and two high-CTC ones (contradiction, qdmatch NQ) side by side.

Reordering is the honest one to drop: its ladder stops at 16k and its full arm is already under the
0.10 floor there (0.0473 at 4B), so it had exactly one usable scale in the aggregate anyway.

Reads `data/ctc_scale_data.csv`; writes `ctc_scale_row.{pdf,png}`.

Usage:  python3 paperdraft/figures/make_fig_scale_row.py
"""
import os

import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))

# Stable label for this figure, so it can be referred to by name across sessions.
# The index of every label lives in paperdraft/figures/README.md. It is stamped into the
# PDF/PNG metadata too, so `pdfinfo <out>.pdf | grep Title` names the script that made it.
FIGURE_LABEL = "CTC-SCALE-ROW"
META = {"Title": f"{FIGURE_LABEL} — model scaling, 4-panel row (CTC-SCALE minus reorder + summary)", "Creator": "make_fig_scale_row.py"}

raw = pd.read_csv(os.path.join(HERE, "data", "ctc_scale_data.csv"))
df = raw[raw["plotted_in_figure"] == "yes"].copy()
df["context_tokens"] = df["context_tokens"].astype(float)

LADDER = np.array([2048, 4096, 8192, 16384, 32768], dtype=float)
SCALES = ["0.8b", "2b", "4b"]
SCALE_LABEL = {"0.8b": "0.8B", "2b": "2B", "4b": "4B"}
SCALE_COLOR = {"0.8b": "#c3bade", "2b": "#8e7cc3", "4b": "#4b3d8f"}

LOW = "#2f6f9f"
HIGH = "#c1553a"
FULL_LS = "-"
CHUNK_LS = (0, (1.6, 1.7))
TXT = "#3f3f3f"
MUTED = "#8a8a8a"

PANELS = [
    ("hpqa", "HotpotQA", "low CTC $O_T(N)$ · gold-ID F1", LOW),
    ("fiqa", "BEIR FiQA", "low CTC $O_T(N)$ · set-F1", LOW),
    ("contra_real", "contradiction", "high CTC $O_T(N^2)$ · set-F1", HIGH),
    ("qdmatch_nq", "qdmatch (NQ)", "high CTC $O_T(N^2)$ · pair-F1", HIGH),
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": "#c9c9c9",
    "axes.labelcolor": TXT,
    "xtick.color": TXT,
    "ytick.color": TXT,
    "mathtext.fontset": "dejavuserif",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def short_reason(reason):
    r = str(reason)
    if "OOM" in r:
        return "chunked run OOMed"
    if "not launched" in r or "never launched" in r:
        return "pair never launched"
    return r[:38] + "…"


fig, axes = plt.subplots(1, 4, figsize=(14.2, 3.9))
fig.subplots_adjust(left=0.052, right=0.992, top=0.695, bottom=0.265, wspace=0.20)

fig.suptitle("Scaling 0.8B → 4B does not close the chunked-attention gap on high-CTC tasks",
             fontsize=13, color="#2b2b2b", x=0.052, ha="left", y=1.005)
fig.text(0.052, 0.885,
         "Qwen3.5 at three parameter counts, identical training shards and identical evaluation "
         "ladders; solid = full attention, dotted = chunked",
         fontsize=8.2, color=MUTED, ha="left", va="bottom")

for i, (task, title, sub, cls_color) in enumerate(PANELS):
    ax = axes[i]
    notes = []
    for sc in SCALES:
        s = df[(df["task"] == task) & (df["model_scale"] == sc)].sort_values("context_tokens")
        s = s[s["full_attention"].notna() & s["chunked_attention"].notna()]
        if len(s) == 0:
            # Named, not silently absent: the export's drop_reason for the 0.8B chunked runs says
            # explicitly not to read the missing arm as a gap.
            miss = raw[(raw["task"] == task) & (raw["model_scale"] == sc)]
            why = short_reason(miss["drop_reason"].iloc[0]) if len(miss) else "no runs"
            notes.append(f"{SCALE_LABEL[sc]}: {why}")
            continue
        c = SCALE_COLOR[sc]
        ax.plot(s["context_tokens"], s["chunked_attention"], ls=CHUNK_LS, color=c, lw=1.8,
                marker="o", ms=3.8, mew=0, zorder=2)
        ax.plot(s["context_tokens"], s["full_attention"], ls=FULL_LS, color=c, lw=1.8,
                marker="o", ms=3.8, mew=0, zorder=3)

    ax.set_xscale("log", base=2)
    ax.set_xticks(LADDER)
    ax.set_xticklabels(["2k", "4k", "8k", "16k", "32k"])
    ax.set_xlim(1750, 40000)
    ax.set_ylim(0, 1.02)
    ax.set_yticks(np.arange(0, 1.001, 0.2))
    ax.yaxis.grid(True, color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)
    for s_ in ["top", "right"]:
        ax.spines[s_].set_visible(False)
    ax.spines["left"].set_color("#c9c9c9")
    ax.spines["bottom"].set_color("#c9c9c9")
    ax.tick_params(length=3, width=0.8)
    ax.set_xlabel("context tokens", color=TXT, labelpad=5)
    if i == 0:
        ax.set_ylabel("score (task metric)", color=TXT, labelpad=6)

    ax.set_title(f"({'abcd'[i]})  {title}", fontsize=11.5, color="#2b2b2b", pad=15, loc="left")
    ax.text(0.0, 1.012, sub, transform=ax.transAxes, fontsize=7.4, color=cls_color,
            ha="left", va="bottom", alpha=0.85)
    if task == "hpqa":
        ax.text(0.03, 0.10, "all three scales overlap:\nthe gap never opens", fontsize=7.2,
                color=MUTED, transform=ax.transAxes, ha="left", va="bottom", linespacing=1.4)
    if notes:
        ax.text(0.03, 0.06, "not shown — " + "; ".join(notes), transform=ax.transAxes,
                fontsize=7.0, color=MUTED, ha="left", va="bottom")

handles = [Patch(facecolor=SCALE_COLOR[sc], label=f"Qwen3.5-{SCALE_LABEL[sc]}") for sc in SCALES]
handles += [Line2D([0], [0], color="#6b6b6b", lw=2.0, ls=FULL_LS, label="full attention"),
            Line2D([0], [0], color="#6b6b6b", lw=2.0, ls=CHUNK_LS, label="chunked attention")]
fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9.2,
           bbox_to_anchor=(0.5, 0.005), columnspacing=2.2)

fig.savefig(os.path.join(HERE, "ctc_scale_row.pdf"), bbox_inches="tight", facecolor="white", metadata=META)
fig.savefig(os.path.join(HERE, "ctc_scale_row.png"), dpi=220, bbox_inches="tight",
            facecolor="white", metadata=META)

for task, title, _, _ in PANELS:
    r = df[(df["task"] == task) & (df["context_tokens"] == 32768)]
    cells = "  ".join(
        f"{SCALE_LABEL[sc]}=" +
        (f"{r[r['model_scale'] == sc]['relative_gap'].iloc[0]:+.3f}"
         if len(r[r["model_scale"] == sc]) else "–")
        for sc in SCALES)
    print(f"{task:14s} gap@32k  {cells}")
print(f"[{FIGURE_LABEL}] wrote {HERE}")
