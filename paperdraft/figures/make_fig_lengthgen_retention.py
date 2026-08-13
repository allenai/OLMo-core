"""Paper-ready cut of the length-generalization figure: panel (b) alone.

Same data and same computation as `make_fig_lengthgen.py`'s second panel -- each task's score past
the 32k training ceiling, divided by its own 32k score, so absolute task difficulty drops out and
only the shape of the decay is left. Sized as a standalone figure rather than a quadrant.

Reads `data/ctc_length_generalization_data.csv`; writes `ctc_lengthgen_retention.{pdf,png}`.

Usage:  python3 paperdraft/figures/make_fig_lengthgen_retention.py
"""
import os

import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))

# Stable label for this figure, so it can be referred to by name across sessions.
# The index of every label lives in paperdraft/figures/README.md. It is stamped into the
# PDF/PNG metadata too, so `pdfinfo <out>.pdf | grep Title` names the script that made it.
FIGURE_LABEL = "CTC-RETENTION"
META = {"Title": f"{FIGURE_LABEL} — length-gen retention only (CTC-LENGTHGEN panel b, standalone)", "Creator": "make_fig_lengthgen_retention.py"}

raw = pd.read_csv(os.path.join(HERE, "data", "ctc_length_generalization_data.csv"))
df = raw[raw["plotted_in_figure"] == "yes"].copy()

LOW = "#2f6f9f"
HIGH = "#c1553a"
TXT = "#3f3f3f"
MUTED = "#8a8a8a"

RUNGS = np.array([32768, 65536, 131072], dtype=float)
RUNG_LABELS = ["32k", "64k", "128k"]

LABEL = {"rerank": "MS MARCO rerank", "msmarco": "MS MARCO", "oolong": "Oolong",
         "hpqa": "HotpotQA", "nq": "NQ", "contra_real": "contradiction",
         "qdmatch_nq": "qdmatch (NQ)"}

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


def ladder(task):
    s = df[(df["task"] == task) & df["context_tokens"].notna()].sort_values("context_tokens")
    s = s[s["full_attention"].notna()]
    return s["context_tokens"].values.astype(float), s["full_attention"].values.astype(float)


def label_right(ax, items, ylim, min_gap_frac=0.058):
    """Nudge end-of-line labels apart so two tasks landing on the same value stay readable."""
    items = sorted(items, key=lambda it: it[1])
    span = ylim[1] - ylim[0]
    placed = []
    for x, y, text, c in items:
        yy = y
        if placed and yy - placed[-1] < span * min_gap_frac:
            yy = placed[-1] + span * min_gap_frac
        placed.append(yy)
        if abs(yy - y) > 1e-9:
            ax.annotate("", xy=(x, y), xytext=(x * 1.045, yy),
                        arrowprops=dict(arrowstyle="-", color=c, lw=0.7, alpha=0.55,
                                        shrinkA=0, shrinkB=0),
                        annotation_clip=False, zorder=1)
        ax.annotate(text, xy=(x * 1.05, yy), color=c, fontsize=8.4,
                    va="center", ha="left", annotation_clip=False)


# Only tasks with at least two rungs; a single 32k point says nothing about generalization.
tok = df[df["context_tokens"].notna()]
TASKS = [t for t in tok["task"].unique() if (tok["task"] == t).sum() >= 2]
TASKS.sort(key=lambda t: -ladder(t)[1][-1])
CLASS = {t: df[df["task"] == t]["ctc_class"].iloc[0] for t in TASKS}

fig, ax = plt.subplots(figsize=(6.9, 4.9))
fig.subplots_adjust(left=0.115, right=0.755, top=0.795, bottom=0.185)

ends, at128 = [], {}
for t in TASKS:
    x, y = ladder(t)
    if abs(x[0] - 32768) > 1:
        raise SystemExit(f"{t}: first rung is {x[0]:.0f}, not 32768 -- cannot normalise")
    r = y / y[0]
    c = LOW if CLASS[t] == "low" else HIGH
    ax.plot(x, r, color=c, lw=2.0, marker="o", ms=4.4, mew=0,
            zorder=3 if CLASS[t] == "high" else 2)
    ends.append((x[-1], r[-1], LABEL[t], c))
    if x[-1] > 100000:
        at128[t] = r[-1]

ax.axhline(1.0, color="#b8b8b8", lw=0.9, ls=(0, (3, 3)), zorder=1)
label_right(ax, ends, (0, 1.12))

ax.set_xscale("log", base=2)
ax.set_xticks(RUNGS)
ax.set_xticklabels(RUNG_LABELS)
ax.set_xlim(29000, 150000)
ax.set_ylim(0, 1.12)
ax.set_yticks(np.arange(0, 1.001, 0.2))
ax.yaxis.grid(True, color="#e6e6e6", lw=0.8)
ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color("#c9c9c9")
ax.spines["bottom"].set_color("#c9c9c9")
ax.tick_params(length=3, width=0.8)
ax.set_xlabel("context tokens", color=TXT, labelpad=6)
ax.set_ylabel("score relative to the same task at 32k", color=TXT, labelpad=6)

ax.set_title("Past the 32k training ceiling, low-CTC tasks decay\nand high-CTC tasks fall off a "
             "cliff", fontsize=12.5, color="#2b2b2b", pad=16, loc="left")

lo = [v for t, v in at128.items() if CLASS[t] == "low"]
hi = [v for t, v in at128.items() if CLASS[t] == "high"]
ax.annotate(f"low CTC keeps {min(lo):.0%}–{max(lo):.0%}", xy=(0.035, 0.30),
            xycoords="axes fraction", color=LOW, fontsize=8.8, fontweight="bold")
ax.annotate(f"high CTC keeps {min(hi):.0%}–{max(hi):.0%}", xy=(0.035, 0.20),
            xycoords="axes fraction", color=HIGH, fontsize=8.8, fontweight="bold")

ax.legend(handles=[Line2D([0], [0], color=LOW, lw=2.4, label="low CTC  $O_T(N)$"),
                   Line2D([0], [0], color=HIGH, lw=2.4, label="high CTC  $O_T(NM)$ and above")],
          loc="lower center", ncol=2, frameon=False, fontsize=8.8,
          bbox_to_anchor=(0.5, -0.30), columnspacing=1.8, handlelength=1.8)

fig.savefig(os.path.join(HERE, "ctc_lengthgen_retention.pdf"), bbox_inches="tight",
            facecolor="white", metadata=META)
fig.savefig(os.path.join(HERE, "ctc_lengthgen_retention.png"), dpi=220, bbox_inches="tight",
            facecolor="white", metadata=META)

for t in TASKS:
    x, y = ladder(t)
    print(f"  {t:14s} {CLASS[t]:4s} " +
          "  ".join(f"{xx/1024:.0f}k={yy/y[0]:.3f}" for xx, yy in zip(x, y)))
print(f"[{FIGURE_LABEL}] wrote {HERE}")
