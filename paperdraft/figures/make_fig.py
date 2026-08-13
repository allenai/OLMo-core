"""CTC figure 1: low-vs-high-CTC aggregate over the ladder, plus six per-task panels.

Prasann's plotting code, repointed from the hand-made `ctc_data.csv` at
`/home/claude/ctc_data.csv` onto `data/ctc_suite_grid_data.csv`, which
`export_ctc_suite_data.py` builds from the harvested grade JSONs. Same columns, so the plotting
below is unchanged apart from the task keys and the three notes marked ⚠.

Usage:  python3 paperdraft/figures/make_fig.py     (writes ctc_figure.{png,pdf})
"""
import os

import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))

# Stable label for this figure, so it can be referred to by name across sessions.
# The index of every label lives in paperdraft/figures/README.md. It is stamped into the
# PDF/PNG metadata too, so `pdfinfo <out>.pdf | grep Title` names the script that made it.
FIGURE_LABEL = "CTC-GRID"
META = {"Title": f"{FIGURE_LABEL} — low-vs-high CTC aggregate + 6 high-CTC task panels", "Creator": "make_fig.py"}

# ⚠ 1/3. Only rows the export marks as currently quotable. Without this filter the superseded
# xabsence row (training CE never left the 0.78 flatline) and the never-built rungs would be
# averaged in as if they were measurements. See data/README.md, "What is dropped, and why".
df = pd.read_csv(os.path.join(HERE, "data", "ctc_suite_grid_data.csv"))
df = df[df["plotted_in_figure"] == "yes"].copy()
df["context_tokens"] = df["context_tokens"].astype(float)

LADDER = np.array([2048, 4096, 8192, 16384, 32768], dtype=float)

# colour = CTC class ; linestyle = attention type
LOW = "#2f6f9f"
HIGH = "#c1553a"
LOW_FILL = "#d8e6f2"
HIGH_FILL = "#f4ddd4"
FULL_LS = "-"
CHUNK_LS = (0, (1.6, 1.7))
TXT = "#3f3f3f"


def bin_floor(x):
    idx = np.searchsorted(LADDER, x, side="right") - 1
    return LADDER[max(idx, 0)]


df["bin"] = df["context_tokens"].apply(bin_floor)


def chain_aggregate(mask, cols=("full_attention", "chunked_attention")):
    """Chained mean: chain step-to-step growth using only tasks present in both
    adjacent buckets, so task drop-out/entry can't create spurious non-monotonic
    jumps in the aggregate line (same method as the 'Chained mean' figure)."""
    sub = df[mask]
    tasks = sub["task"].unique()
    full = df[df["task"].isin(tasks)]
    buckets = sorted(full["bin"].unique())
    pivots = {c: full.pivot_table(index="task", columns="bin", values=c)[buckets]
              for c in cols}
    vals = {c: [pivots[c][buckets[0]].dropna().mean()] for c in cols}
    for i in range(1, len(buckets)):
        b0, b1 = buckets[i - 1], buckets[i]
        common_idx = pivots[cols[0]][[b0, b1]].dropna().index
        for c in cols:
            step = pivots[c].loc[common_idx, [b0, b1]]
            factor = step[b1].mean() / step[b0].mean()
            vals[c].append(vals[c][-1] * factor)
    return np.array(buckets), np.array(vals[cols[0]]), np.array(vals[cols[1]])


def get_task(name):
    s = df[df["task"] == name].sort_values("context_tokens")
    return (s["context_tokens"].values,
            s["full_attention"].values,
            s["chunked_attention"].values)


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


def style_axes(ax, ylab=False):
    ax.set_xscale("log", base=2)
    ax.set_xticks(LADDER)
    ax.set_xticklabels(["2k", "4k", "8k", "16k", "32k"])
    ax.set_xlim(1750, 40000)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.001, 0.2))
    ax.yaxis.grid(True, color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#c9c9c9")
    ax.spines["bottom"].set_color("#c9c9c9")
    ax.tick_params(length=3, width=0.8)
    ax.set_xlabel("context tokens", color=TXT, labelpad=6)
    if ylab:
        ax.set_ylabel("score (task metric)", color=TXT, labelpad=6)


def draw_pair(ax, x, yf, yc, color, fill, ms=4.0, lw=1.8, z=2):
    ax.plot(x, yc, ls=CHUNK_LS, color=color, lw=lw, marker="o", ms=ms, mew=0, zorder=z)
    ax.plot(x, yf, ls=FULL_LS, color=color, lw=lw, marker="o", ms=ms, mew=0, zorder=z + 1)


fig = plt.figure(figsize=(13.6, 11.4))
gs = gridspec.GridSpec(3, 6, figure=fig, hspace=0.42, wspace=0.42,
                       left=0.065, right=0.94, top=0.925, bottom=0.085,
                       height_ratios=[1.18, 1.0, 1.0])

# ---------- top: low and high CTC aggregates in one panel ----------
ax = fig.add_subplot(gs[0, 1:5])
xl, ylf, ylc = chain_aggregate(df["ctc_class"] == "low")
xh, yhf, yhc = chain_aggregate(df["ctc_class"] == "high")
draw_pair(ax, xh, yhf, yhc, HIGH, HIGH_FILL, ms=5.2, lw=2.2, z=2)
draw_pair(ax, xl, ylf, ylc, LOW, LOW_FILL, ms=5.2, lw=2.2, z=4)
style_axes(ax, ylab=True)
ax.set_title("chunked attention matches full attention on low-CTC tasks "
             "and falls steadily behind on high-CTC tasks",
             fontsize=13.5, color="#2b2b2b", pad=14)
n_low = df[df["ctc_class"] == "low"]["task"].nunique()
n_high = df[df["ctc_class"] == "high"]["task"].nunique()
ax.text(0.0, 1.012,
        f"chained mean over {n_low} low-CTC and {n_high} high-CTC tasks (isolates "
        "within-task change from task drop-out)",
        transform=ax.transAxes, fontsize=8, color="#8a8a8a", ha="left", va="bottom")
ax.annotate(f"low CTC  $O_T(N)$\n({n_low} tasks)", xy=(xl[-1] * 1.06, ylf[-1]),
            color=LOW, fontsize=9, va="center", ha="left", annotation_clip=False)
ax.annotate(f"high CTC  $O_T(NM)$+\n({n_high} tasks)", xy=(xh[-1] * 1.06, 0.34),
            color=HIGH, fontsize=9, va="center", ha="left", annotation_clip=False)


def gap_arrow(ax, x, y_top, y_bot, color, side, dx_frac=0.13):
    ax.annotate("", xy=(x, y_top), xytext=(x, y_bot),
               arrowprops=dict(arrowstyle="<->", color=color, lw=1.5,
                              shrinkA=0, shrinkB=0), zorder=6)
    gap = y_top - y_bot
    mid = (y_top + y_bot) / 2
    lx = x * (1 + dx_frac) if side == "right" else x * (1 - dx_frac)
    ha = "left" if side == "right" else "right"
    ax.annotate(f"+{gap:.2f}", xy=(lx, mid), color=color, fontsize=8.2,
               ha=ha, va="center", fontweight="bold", zorder=7,
               bbox=dict(facecolor="white", edgecolor="none", pad=1.3, alpha=0.9))


gap_arrow(ax, xl[0], ylf[0], ylc[0], LOW, side="right")
gap_arrow(ax, xl[-1], ylf[-1], ylc[-1], LOW, side="left")
gap_arrow(ax, xh[0], yhf[0], yhc[0], HIGH, side="right")
gap_arrow(ax, xh[-1], yhf[-1], yhc[-1], HIGH, side="left")

# ---------- task panels ----------
# ⚠ 2/3. Task keys are the suite row names used by the export (outlier_amzn -> outlier_amazon,
# outlier_wiki -> outlier_scalek, Contradiction -> contra_real, niah -> niah_contra).
#
# ⚠ 3/3. Every panel is high-CTC only: the low-CTC comparison series (outlier_amzn in panel 1,
# niah in panel 4) are drawn in the TOP aggregate and were dropped from the panels on request, so
# the six small plots are now purely "what the gap looks like on a high-CTC task". The bottom
# legend still carries the blue entry because the top panel still uses it.
#
# The fifth panel was "cross-document absence"/xabsence, which is SUPERSEDED --- its numbers never
# came from a model that learned the task (training CE sat on the 0.78 flatline). It is now
# `strmatch`: same O_T(N^2), same set-F1, and the widest collapse in the suite (full 0.999 -> 0.994
# against chunked 0.003 -> 0.000, i.e. the chunked arm is at the floor from the FIRST rung).
panels = [
    dict(title="outlier detection", sub="outlier_wiki · high CTC $O_T(NM)$ · set-F1",
         legend=None,
         series=[("outlier_scalek", "outlier_wiki", HIGH, HIGH_FILL)]),
    dict(title="query-document matching", sub="qdmatch_hpqa · high CTC $O_T(N^2)+$ · pair-F1",
         legend=None,
         series=[("qdmatch_hpqa", "qdmatch_hpqa", HIGH, HIGH_FILL)]),
    dict(title="reordering", sub="reorder · high CTC $O_T(N^2)+$ · Kendall tau",
         legend=None,
         series=[("reorder", "reorder", HIGH, HIGH_FILL)]),
    dict(title="contradiction detection", sub="contradiction · high CTC $O_T(N^2)+$ · set-F1",
         legend=None,
         series=[("contra_real", "contradiction", HIGH, HIGH_FILL)]),
    dict(title="string matching", sub="strmatch · high CTC $O_T(N^2)+$ · set-F1",
         legend=None,
         series=[("strmatch", "strmatch", HIGH, HIGH_FILL)]),
    dict(title="text grouping", sub="textgroups · high CTC $O_T(N^3)+$ · group-F1",
         legend=None,
         series=[("textgroups", "textgroups", HIGH, HIGH_FILL)]),
]
positions = [gs[1, 0:2], gs[1, 2:4], gs[1, 4:6], gs[2, 0:2], gs[2, 2:4], gs[2, 4:6]]

for i, (p, pos) in enumerate(zip(panels, positions)):
    ax = fig.add_subplot(pos)
    handles = []
    for task, label, color, fill in p["series"]:
        x, yf, yc = get_task(task)
        if len(x) == 0:
            raise SystemExit(f"no plottable rows for task {task!r} -- check the export")
        draw_pair(ax, x, yf, yc, color, fill)
        handles.append(Line2D([0], [0], color=color, lw=1.8, label=label))
    style_axes(ax, ylab=(i in (0, 3)))
    ax.set_title(p["title"], fontsize=11.5, color="#2b2b2b", pad=16)
    ax.text(0.0, 1.012, p["sub"], transform=ax.transAxes, fontsize=7.6,
            color="#8a8a8a", ha="left", va="bottom")
    if p["legend"] is not None:
        leg = ax.legend(handles=handles, loc="lower left", fontsize=7.4, frameon=False,
                        handlelength=1.8, borderpad=0.1, labelspacing=0.3,
                        bbox_to_anchor=p["legend"], bbox_transform=ax.transAxes)
        for t, (_, _, c, _) in zip(leg.get_texts(), p["series"]):
            t.set_color(c)

fig.legend(handles=[Line2D([0], [0], color=LOW, lw=2.4, label="low CTC  $O_T(N)$"),
                    Line2D([0], [0], color=HIGH, lw=2.4, label="high CTC  $O_T(NM)$ and above"),
                    Line2D([0], [0], color="#6b6b6b", lw=2.0, ls=FULL_LS,
                           label="full attention"),
                    Line2D([0], [0], color="#6b6b6b", lw=2.0, ls=CHUNK_LS,
                           label="chunked attention")],
           loc="lower center", ncol=4, frameon=False, fontsize=9.5,
           bbox_to_anchor=(0.5, 0.005), columnspacing=2.4)

fig.savefig(os.path.join(HERE, "ctc_figure.png"), dpi=220, bbox_inches="tight",
            facecolor="white", metadata=META)
fig.savefig(os.path.join(HERE, "ctc_figure.pdf"), bbox_inches="tight", facecolor="white", metadata=META)
print("low ", [f"{a:.0f}:{b:.3f}/{c:.3f}" for a, b, c in zip(xl, ylf, ylc)])
print("high", [f"{a:.0f}:{b:.3f}/{c:.3f}" for a, b, c in zip(xh, yhf, yhc)])
print(f"[{FIGURE_LABEL}] wrote {HERE}")
