"""CTC figure 2: chained-mean full-attention curve, and the chunked-attention penalty by bucket.

Prasann's plotting code, repointed from the hand-made `ctc_data.csv` at
`/home/claude/ctc_data.csv` onto `data/ctc_suite_grid_data.csv`, which
`export_ctc_suite_data.py` builds from the harvested grade JSONs. Same columns, so the plotting
below is unchanged apart from the two notes marked ⚠.

Usage:  python3 paperdraft/figures/make_fig2.py    (writes ctc_figure2.{png,pdf})
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
FIGURE_LABEL = "CTC-PENALTY"
META = {"Title": f"{FIGURE_LABEL} — chained mean over the ladder + per-bucket penalty bars", "Creator": "make_fig2.py"}

# ⚠ 1/2. Only rows the export marks as currently quotable --- see the note in make_fig.py.
df = pd.read_csv(os.path.join(HERE, "data", "ctc_suite_grid_data.csv"))
df = df[df["plotted_in_figure"] == "yes"].copy()
df["context_tokens"] = df["context_tokens"].astype(float)
LADDER = np.array([2048, 4096, 8192, 16384, 32768], dtype=float)
LADDER_LABELS = ["2k", "4k", "8k", "16k", "32k"]
RANGE_LABELS = ["<4k", "4-8k", "8-16k", "16-32k", "≥32k"]

BLUE = "#3f7dc4"
ORANGE = "#e2622f"
BLUE_FAINT = "#a9c8e8"
ORANGE_FAINT = "#f0a687"
TXT = "#3f3f3f"

def bin_floor(x):
    idx = np.searchsorted(LADDER, x, side="right") - 1
    return LADDER[max(idx, 0)]

df["bin"] = df["context_tokens"].apply(bin_floor)

low_tasks = df[df.ctc_class == "low"]["task"].unique().tolist()
high_tasks = df[df.ctc_class == "high"]["task"].unique().tolist()

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.edgecolor": "#c9c9c9",
    "axes.labelcolor": TXT,
    "xtick.color": TXT,
    "ytick.color": TXT,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 6.0))
fig.subplots_adjust(left=0.06, right=0.98, top=0.85, bottom=0.14, wspace=0.28)

# ============ LEFT: chained mean ============
def chain(tasks, col="full_attention"):
    sub = df[df.task.isin(tasks)]
    buckets = sorted(sub["bin"].unique())
    pivot = sub.pivot_table(index="task", columns="bin", values=col)[buckets]
    vals = [pivot[buckets[0]].dropna().mean()]
    for i in range(1, len(buckets)):
        b0, b1 = buckets[i - 1], buckets[i]
        common = pivot[[b0, b1]].dropna()
        factor = common[b1].mean() / common[b0].mean()
        vals.append(vals[-1] * factor)
    return buckets, vals, pivot

rng = np.random.default_rng(7)

def scatter_and_line(ax, tasks, color, faint, label, jitter_dir):
    buckets, vals, pivot = chain(tasks)
    xs = np.arange(len(buckets))
    # individual task runs, jittered
    for bi, b in enumerate(buckets):
        col = pivot[b].dropna()
        n = len(col)
        if n == 0:
            continue
        jitter = (rng.uniform(-0.16, 0.16, size=n))
        ax.scatter(xs[bi] + jitter, col.values, s=13, color=faint, alpha=0.75,
                   zorder=2, linewidths=0)
    ax.plot(xs, vals, color=color, lw=2.6, marker="o", ms=6.5, zorder=4, label=label)
    return xs, vals

xs_low, vals_low = scatter_and_line(ax1, low_tasks, BLUE, BLUE_FAINT, "Low CTC", -1)
xs_high, vals_high = scatter_and_line(ax1, high_tasks, ORANGE, ORANGE_FAINT, "High CTC", 1)

ax1.set_xticks(np.arange(5))
ax1.set_xticklabels(LADDER_LABELS)
ax1.set_xlim(-0.4, 4.9)
ax1.set_ylim(0, 1.03)
ax1.set_yticks(np.arange(0, 1.01, 0.2))
ax1.yaxis.grid(True, color="#e6e6e6", lw=0.8)
ax1.set_axisbelow(True)
for s in ["top", "right"]:
    ax1.spines[s].set_visible(False)
ax1.spines["left"].set_color("#c9c9c9")
ax1.spines["bottom"].set_color("#c9c9c9")
ax1.set_xlabel("Context length (tokens)", labelpad=8)
ax1.set_ylabel("Full-attention performance", labelpad=8)
ax1.set_title("Chained mean", fontsize=14, color="#1f1f1f", loc="left", pad=28,
              fontweight="bold")
# ⚠ 2/2. Task count and per-step range were hardcoded ("all 17 tasks (5-8 per step)"); the export
# now carries 21 usable tasks, so both are computed from the data rather than restated.
n_tasks = df["task"].nunique()
per_step = df.groupby("bin")["task"].nunique()
ax1.text(0.0, 1.06,
         f"cumulated within-task changes, all {n_tasks} tasks "
         f"({per_step.min()}-{per_step.max()} per step)",
         transform=ax1.transAxes, fontsize=9, color="#8a8a8a", ha="left", va="bottom")

ax1.annotate("Low CTC", xy=(xs_low[-1] + 0.1, vals_low[-1]), color=BLUE,
            fontsize=10.5, va="center", ha="left")
ax1.annotate("High CTC", xy=(xs_high[-1] + 0.1, vals_high[-1]), color=ORANGE,
            fontsize=10.5, va="center", ha="left")

legend_handles = [
    Line2D([0], [0], color=BLUE, lw=2.6, marker="o", ms=6, label="Low CTC"),
    Line2D([0], [0], color=ORANGE, lw=2.6, marker="o", ms=6, label="High CTC"),
    Line2D([0], [0], color="#9a9a9a", lw=0, marker="o", ms=5, label="individual task run"),
]
ax1.legend(handles=legend_handles, loc="lower left", frameon=False, fontsize=9,
          handletextpad=0.6, labelspacing=0.5)

# ============ RIGHT: penalty bar chart ============
# ⚠ The floor rule, which this panel needs and the left one does not. `relative_gap` is a RATIO,
# so once the full arm is itself near zero the ratio is noise: at 32k grouping reads full 0.0066
# vs chunked 0.0095, i.e. relative_gap = -0.44, which enters a mean as a spurious 44% chunked WIN.
# That single cell plus reorder dropping out at 32k made the high-CTC bar read -61% at >=32k after
# -70% at 16-32k --- the penalty appearing to IMPROVE at the deepest bucket, which is the one thing
# it does not do. With the floor the series is monotonic: -32, -37, -48, -63, -70.
# fig1_gap_vs_context.py uses 0.15 for the same reason (documented in data/README.md); this uses
# 0.10, one notch lower, because 0.15 also swallowed qdmatch_fiqa@32k --- a full arm of
# 0.1244 +/- 0.0148, i.e. 8.4 SE above zero. That is a real measurement, not a ratio between two
# near-zero numbers, and excluding it dropped the newest high-CTC task out of the deepest bucket
# exactly where its gap is widest. 0.10 keeps it (>=32k goes -69.5% -> -72.4%) and still excludes
# every genuinely-noise cell, grouping@32k included (0.0066 +/- 0.0036, under 2 SE from zero).
# It bites only on high-CTC rows; no low-CTC row is near the floor, so the blue bars are
# identical at any of these thresholds.
FLOOR = 0.10
floored = df[df.full_attention >= FLOOR]
_dropped = df[(df.full_attention < FLOOR) & df.relative_gap.notna()]
for _, r in _dropped.iterrows():
    print(f"  floor: dropped {r['task']}@{int(r['context_tokens'])} from the ratio "
          f"(full={r['full_attention']:.4f} < {FLOOR}, relative_gap={r['relative_gap']:+.3f})")


def bucket_avg_gap(tasks):
    sub = floored[floored.task.isin(tasks)]
    g = sub.groupby("bin")["relative_gap"].mean()
    return g.reindex(LADDER)

low_gap = bucket_avg_gap(low_tasks) * 100
high_gap = bucket_avg_gap(high_tasks) * 100
# Counts describe the pool the bars are actually averaged over, so they use the same floored set.
low_count = floored[floored.task.isin(low_tasks)].groupby("bin")["task"].nunique().reindex(LADDER)
high_count = floored[floored.task.isin(high_tasks)].groupby("bin")["task"].nunique().reindex(LADDER)

x = np.arange(5)
blue_h = -low_gap.values
orange_h = -high_gap.values
bw = 0.32
gap = 0.05

# soft alternating background bands, one per bucket, for quiet visual grouping
for xi in x:
    if xi % 2 == 0:
        ax2.axvspan(xi - 0.5, xi + 0.5, color="#f7f7f7", zorder=0, lw=0)

bars_low = ax2.bar(x - bw / 2 - gap / 2, blue_h, width=bw, color=BLUE, zorder=3,
                   edgecolor="white", linewidth=1.1)
bars_high = ax2.bar(x + bw / 2 + gap / 2, orange_h, width=bw, color=ORANGE, zorder=3,
                    edgecolor="white", linewidth=1.1)

# rounded-tip illusion: a small matching dot capping each bar's far end
for xi, h in zip(x - bw / 2 - gap / 2, blue_h):
    ax2.plot(xi, h, marker="o", ms=6.2, color=BLUE, zorder=4, mec="none")
for xi, h in zip(x + bw / 2 + gap / 2, orange_h):
    ax2.plot(xi, h, marker="o", ms=6.2, color=ORANGE, zorder=4, mec="none")

# value labels just beyond each bar's tip
for xi, h in zip(x - bw / 2 - gap / 2, blue_h):
    ax2.annotate(f"{h:.0f}%", (xi, h), xytext=(0, -8), textcoords="offset points",
                ha="center", va="top", fontsize=7.8, color=BLUE, fontweight="medium")
for xi, h in zip(x + bw / 2 + gap / 2, orange_h):
    ax2.annotate(f"{h:.0f}%", (xi, h), xytext=(0, -8), textcoords="offset points",
                ha="center", va="top", fontsize=7.8, color=ORANGE, fontweight="medium")

ax2.set_ylim(-100, 4)
ax2.set_xlim(-0.62, 4.62)
ax2.set_yticks([0, -25, -50, -75, -100])
ax2.set_yticklabels(["0", "-25%", "-50%", "-75%", "-100%"])
ax2.yaxis.grid(True, color="#e9e9e9", lw=0.8, zorder=0)
ax2.set_axisbelow(True)
ax2.axhline(0, color="#b8b8b8", lw=1.1, zorder=2)
for s in ["top", "right", "left"]:
    ax2.spines[s].set_visible(False)
ax2.spines["bottom"].set_color("#c9c9c9")
ax2.tick_params(length=0)

ax2.set_xticks(x)
ax2.set_xticklabels(RANGE_LABELS, fontsize=9.5)
sub_labels = [f"{int(high_count[b])} · {int(low_count[b])}" for b in LADDER]
for xi, lbl in zip(x, sub_labels):
    ax2.text(xi, -110, lbl, ha="center", va="top", fontsize=8, color="#a3a3a3")
ax2.set_xlabel("Context length (tokens) · tasks pooled, high · low",
              labelpad=24, fontsize=9.5, color="#6b6b6b")
ax2.set_ylabel("Performance penalty for efficient attention", labelpad=8)
ax2.set_title("Chunked attention costs high-CTC tasks more as context grows",
              fontsize=13.3, color="#1f1f1f", loc="left", pad=32, fontweight="bold")
ax2.text(0.0, 1.06, "Runs pooled by the bucket they fit inside; pool composition varies",
         transform=ax2.transAxes, fontsize=9, color="#8a8a8a", ha="left", va="bottom")

legend_handles2 = [
    Line2D([0], [0], marker="o", ms=7, color=BLUE, lw=0, label="Low CTC tasks, avg"),
    Line2D([0], [0], marker="o", ms=7, color=ORANGE, lw=0, label="High CTC tasks, avg"),
]
ax2.legend(handles=legend_handles2, loc="lower left", bbox_to_anchor=(0.0, 0.0),
          frameon=False, fontsize=9.3, handletextpad=0.5, labelspacing=0.55,
          borderaxespad=0.3)

fig.savefig(os.path.join(HERE, "ctc_figure2.png"), dpi=220, bbox_inches="tight",
            facecolor="white", metadata=META)
fig.savefig(os.path.join(HERE, "ctc_figure2.pdf"), bbox_inches="tight", facecolor="white", metadata=META)

print("low chain", [round(v,3) for v in vals_low])
print("high chain", [round(v,3) for v in vals_high])
print("low gap%", (-low_gap).round(1).tolist())
print("high gap%", (-high_gap).round(1).tolist())
print("counts", list(zip(RANGE_LABELS, sub_labels)))
print(f"[{FIGURE_LABEL}] wrote {HERE}")
