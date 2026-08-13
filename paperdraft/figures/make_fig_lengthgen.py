"""CTC length-generalization figure: what happens past the 32k training ceiling.

Reads `data/ctc_length_generalization_data.csv` (built by `export_ctc_suite_data.py` from the
harvested grade JSONs) and writes `ctc_lengthgen_figure.{png,pdf}`.

Same visual language as `make_fig.py`: colour = CTC class, solid = full attention, dotted =
chunked attention, log-2 x-axis.

Four panels:
  (a) dense score vs context on the 32k / 64k / 128k rungs
  (b) the same lines normalised to their own 32k value -- retention past the ceiling
  (c) qdmatch (NQ), the only task with a chunked arm run past the ceiling
  (d) outlier (wiki, fixed M) on a DOCUMENT-count axis, not a token axis

Usage:  python3 paperdraft/figures/make_fig_lengthgen.py
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
FIGURE_LABEL = "CTC-LENGTHGEN"
META = {"Title": f"{FIGURE_LABEL} — length generalization past 32k, 4 panels", "Creator": "make_fig_lengthgen.py"}

# Only rows the export marks as currently quotable. The dropped rungs are real evals but not
# trustworthy measurements -- see the footnote block at the bottom of the figure, which is built
# from the `drop_reason` column rather than hand-written.
raw = pd.read_csv(os.path.join(HERE, "data", "ctc_length_generalization_data.csv"))
df = raw[raw["plotted_in_figure"] == "yes"].copy()

LOW = "#2f6f9f"
HIGH = "#c1553a"
FULL_LS = "-"
CHUNK_LS = (0, (1.6, 1.7))
TXT = "#3f3f3f"
MUTED = "#8a8a8a"

RUNGS = np.array([32768, 65536, 131072], dtype=float)
RUNG_LABELS = ["32k", "64k", "128k"]

# Pretty names for the task keys the export uses.
LABEL = {
    "rerank": "MS MARCO rerank",
    "msmarco": "MS MARCO",
    "oolong": "Oolong",
    "hpqa": "HotpotQA",
    "nq": "NQ",
    "contra_real": "contradiction",
    "qdmatch_nq": "qdmatch (NQ)",
    "niah_contra": "NIAH-contra",
    "outlier_amazon": "outlier (Amazon)",
    "outlier_scalek": "outlier (wiki, scale-$k$)",
    "outlier_fixk": "outlier (wiki, fixed $M$)",
    "textgroups": "textgroups",
    "absence": "absence",
    "cycle": "cycle",
}

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


def style_axes(ax, ylab=None, xlab="context tokens"):
    ax.set_xscale("log", base=2)
    ax.set_xticks(RUNGS)
    ax.set_xticklabels(RUNG_LABELS)
    ax.set_xlim(28000, 155000)
    ax.yaxis.grid(True, color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#c9c9c9")
    ax.spines["bottom"].set_color("#c9c9c9")
    ax.tick_params(length=3, width=0.8)
    ax.set_xlabel(xlab, color=TXT, labelpad=6)
    if ylab:
        ax.set_ylabel(ylab, color=TXT, labelpad=6)


def ladder(task, col="full_attention"):
    """The (x, y) points for one task on the token ladder, ordered, NaNs dropped."""
    s = df[(df["task"] == task) & df["context_tokens"].notna()].sort_values("context_tokens")
    s = s[s[col].notna()]
    return s["context_tokens"].values.astype(float), s[col].values.astype(float)


# Tasks that have at least two rungs on the token ladder -- a single 32k point says nothing about
# generalisation, so those tasks sit in the footnote instead of the plot.
tok = df[df["context_tokens"].notna()]
LADDER_TASKS = [t for t in tok["task"].unique() if (tok["task"] == t).sum() >= 2]
LADDER_TASKS.sort(key=lambda t: -ladder(t)[1][-1])
CLASS = {t: df[df["task"] == t]["ctc_class"].iloc[0] for t in LADDER_TASKS}
ORDER = {t: df[df["task"] == t]["ctc_order"].iloc[0] for t in LADDER_TASKS}

def label_right(ax, items, ylim, min_gap_frac=0.052):
    """Annotate at the right edge of the line, nudging labels apart so two tasks that land on
    nearly the same value stay readable."""
    items = sorted(items, key=lambda it: it[1])          # (x, y, text, colour), bottom-up
    span = ylim[1] - ylim[0]
    placed = []
    for x, y, text, c in items:
        yy = y
        if placed and yy - placed[-1] < span * min_gap_frac:
            yy = placed[-1] + span * min_gap_frac
        placed.append(yy)
        if abs(yy - y) > 1e-9:                            # leader line back to the true value
            ax.annotate("", xy=(x, y), xytext=(x * 1.045, yy),
                        arrowprops=dict(arrowstyle="-", color=c, lw=0.7, alpha=0.55,
                                        shrinkA=0, shrinkB=0),
                        annotation_clip=False, zorder=1)
        ax.annotate(text, xy=(x * 1.05, yy), color=c, fontsize=8,
                    va="center", ha="left", annotation_clip=False)


fig = plt.figure(figsize=(13.6, 9.2))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.52, wspace=0.30,
                       left=0.065, right=0.90, top=0.895, bottom=0.215)

fig.suptitle("Past the 32k training ceiling, low-CTC tasks decay and high-CTC tasks fall off a "
             "cliff", fontsize=14, color="#2b2b2b", x=0.065, ha="left", y=0.975)
fig.text(0.065, 0.938,
         "full (dense) attention, Qwen3.5-4B, evaluated on 64k and 128k rungs of tasks it was "
         "only ever trained on up to 32k",
         fontsize=8.6, color=MUTED, ha="left", va="bottom")

# ---------------- (a) absolute dense score ----------------
ax = fig.add_subplot(gs[0, 0])
ends = []
for t in LADDER_TASKS:
    x, y = ladder(t)
    c = LOW if CLASS[t] == "low" else HIGH
    ax.plot(x, y, ls=FULL_LS, color=c, lw=1.9, marker="o", ms=4.2, mew=0,
            alpha=0.95, zorder=3 if CLASS[t] == "high" else 2)
    ends.append((x[-1], y[-1], LABEL[t], c))
label_right(ax, ends, (0, 1.02))
style_axes(ax, ylab="score (task metric)")
ax.set_ylim(0, 1.02)
ax.set_yticks(np.arange(0, 1.001, 0.2))
ax.set_title("(a)  absolute score on the long rungs", fontsize=11.5, color="#2b2b2b",
             pad=16, loc="left")
ax.text(0.0, 1.012, f"{len(LADDER_TASKS)} tasks with two or more rungs · full attention only",
        transform=ax.transAxes, fontsize=7.6, color=MUTED, ha="left", va="bottom")

# ---------------- (b) retention relative to 32k ----------------
ax = fig.add_subplot(gs[0, 1])
ret, ends = {}, []
for t in LADDER_TASKS:
    x, y = ladder(t)
    if abs(x[0] - 32768) > 1:                     # every ladder task starts at the 32k rung
        raise SystemExit(f"{t}: first rung is {x[0]:.0f}, not 32768 -- cannot normalise")
    r = y / y[0]
    ret[t] = (x, r)
    c = LOW if CLASS[t] == "low" else HIGH
    ax.plot(x, r, ls=FULL_LS, color=c, lw=1.9, marker="o", ms=4.2, mew=0,
            alpha=0.95, zorder=3 if CLASS[t] == "high" else 2)
    ends.append((x[-1], r[-1], LABEL[t], c))
label_right(ax, ends, (0, 1.15))
ax.axhline(1.0, color="#b8b8b8", lw=0.9, ls=(0, (3, 3)), zorder=1)
style_axes(ax, ylab="score relative to the same task at 32k")
ax.set_ylim(0, 1.15)
ax.set_yticks(np.arange(0, 1.001, 0.2))
ax.set_title("(b)  retention past the ceiling", fontsize=11.5, color="#2b2b2b", pad=16, loc="left")
ax.text(0.0, 1.012, "each task divided by its own 32k score, so absolute difficulty drops out",
        transform=ax.transAxes, fontsize=7.6, color=MUTED, ha="left", va="bottom")

# Band annotation: where each class lands at 128k.
at128 = {t: r[-1] for t, (x, r) in ret.items() if x[-1] > 100000}
lo = [v for t, v in at128.items() if CLASS[t] == "low"]
hi = [v for t, v in at128.items() if CLASS[t] == "high"]
if lo and hi:
    ax.annotate(f"low CTC keeps {min(lo):.0%}–{max(lo):.0%}", xy=(0.03, 0.30),
                xycoords="axes fraction", color=LOW, fontsize=8.4, fontweight="bold")
    ax.annotate(f"high CTC keeps {min(hi):.0%}–{max(hi):.0%}", xy=(0.03, 0.20),
                xycoords="axes fraction", color=HIGH, fontsize=8.4, fontweight="bold")

# ---------------- (c) the one task with a chunked arm past the ceiling ----------------
ax = fig.add_subplot(gs[1, 0])
xq, yqf = ladder("qdmatch_nq", "full_attention")
_, yqc = ladder("qdmatch_nq", "chunked_attention")
ax.plot(xq, yqc, ls=CHUNK_LS, color=HIGH, lw=2.0, marker="o", ms=4.6, mew=0, zorder=2)
ax.plot(xq, yqf, ls=FULL_LS, color=HIGH, lw=2.0, marker="o", ms=4.6, mew=0, zorder=3)
for i, (xx, yf, yc) in enumerate(zip(xq, yqf, yqc)):
    ax.annotate("", xy=(xx, yf), xytext=(xx, yc),
                arrowprops=dict(arrowstyle="<->", color=HIGH, lw=1.2, shrinkA=0, shrinkB=0),
                zorder=4)
    last = i == len(xq) - 1                      # keep the 128k label inside the axes
    ax.annotate(f"−{1 - yc / yf:.0%}" if yf > 0 else "",
                xy=(xx * (0.96 if last else 1.04), (yf + yc) / 2 + (0.045 if last else 0)),
                color=HIGH, fontsize=8, fontweight="bold",
                va="center", ha="right" if last else "left",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.9), zorder=5)
style_axes(ax, ylab="pair-F1")
ax.set_ylim(0, 0.78)
ax.set_title("(c)  qdmatch (NQ): the chunked arm dies first", fontsize=11.5, color="#2b2b2b",
             pad=16, loc="left")
ax.text(0.0, 1.012, "high CTC $O_T(N^2)$ · the only task with a chunked run past 32k",
        transform=ax.transAxes, fontsize=7.6, color=MUTED, ha="left", va="bottom")

# ---------------- (d) fixed-M outlier on the document axis ----------------
ax = fig.add_subplot(gs[1, 1])
fx = df[(df["task"] == "outlier_fixk") & df["n_docs"].notna()].copy()
fx["n_docs"] = fx["n_docs"].astype(float)
fx = fx.sort_values("n_docs")
ax.plot(fx["n_docs"], fx["full_attention"], ls=FULL_LS, color=LOW, lw=2.0,
        marker="o", ms=4.6, mew=0, zorder=3)
ax.set_xscale("log", base=2)
ax.set_xticks(fx["n_docs"].values)
ax.set_xticklabels([f"{int(v)}" for v in fx["n_docs"].values], fontsize=8)
ax.set_xlim(fx["n_docs"].min() * 0.9, fx["n_docs"].max() * 1.1)
ax.set_ylim(0, 1.02)
ax.set_yticks(np.arange(0, 1.001, 0.2))
ax.yaxis.grid(True, color="#e6e6e6", lw=0.8)
ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color("#c9c9c9")
ax.spines["bottom"].set_color("#c9c9c9")
ax.tick_params(length=3, width=0.8)
ax.set_xlabel("documents in the corpus ($N$), $M$ held fixed", color=TXT, labelpad=6)
ax.set_ylabel("set-F1", color=TXT, labelpad=6)
ax.set_title("(d)  outlier (wiki, fixed $M$): collapse tracks $N$", fontsize=11.5,
             color="#2b2b2b", pad=16, loc="left")
ax.text(0.0, 1.012,
        "⚠ different x-axis: document count, not tokens · full attention only",
        transform=ax.transAxes, fontsize=7.6, color=MUTED, ha="left", va="bottom")
ax.annotate(f"{fx['full_attention'].iloc[0]:.2f} → {fx['full_attention'].iloc[-1]:.2f}\n"
            f"over {int(fx['n_docs'].iloc[0])}→{int(fx['n_docs'].iloc[-1])} documents",
            xy=(0.40, 0.72), xycoords="axes fraction", color=LOW, fontsize=8.4,
            fontweight="bold", ha="left", va="top")

# ---------------- legend + provenance footnote ----------------
fig.legend(handles=[Line2D([0], [0], color=LOW, lw=2.4, label="low CTC  $O_T(N)$"),
                    Line2D([0], [0], color=HIGH, lw=2.4,
                           label="high CTC  $O_T(NM)$ and above"),
                    Line2D([0], [0], color="#6b6b6b", lw=2.0, ls=FULL_LS,
                           label="full attention"),
                    Line2D([0], [0], color="#6b6b6b", lw=2.0, ls=CHUNK_LS,
                           label="chunked attention")],
           loc="lower center", ncol=4, frameon=False, fontsize=9.5,
           bbox_to_anchor=(0.5, 0.135), columnspacing=2.4)

# Build the "what is not here" note straight from the export, so it can never drift from the data.
dropped = raw[raw["plotted_in_figure"] == "no"]
lines = []
for reason, grp in dropped.groupby("drop_reason"):
    names = sorted({LABEL.get(t, t) for t in grp["task"]})
    short = reason if len(reason) <= 165 else reason[:162].rsplit(" ", 1)[0] + "…"
    lines.append(f"• {', '.join(names)} — {short}")
only32 = sorted({LABEL.get(t, t) for t in tok["task"].unique()} - {LABEL[t] for t in LADDER_TASKS})
if only32:
    lines.append(f"• {', '.join(only32)} — 32k rung only once the exclusions above are applied, "
                 "so no length-generalisation slope can be drawn")
fig.text(0.065, 0.098, f"Excluded long rungs ({len(dropped)} of {len(raw)} exported rows):",
         fontsize=8.2, color="#5f5f5f", ha="left", va="top", fontweight="bold")
fig.text(0.065, 0.078, "\n".join(lines), fontsize=7.4, color=MUTED, ha="left", va="top",
         linespacing=1.6)

fig.savefig(os.path.join(HERE, "ctc_lengthgen_figure.png"), dpi=220, bbox_inches="tight",
            facecolor="white", metadata=META)
fig.savefig(os.path.join(HERE, "ctc_lengthgen_figure.pdf"), bbox_inches="tight",
            facecolor="white", metadata=META)

print(f"ladder tasks ({len(LADDER_TASKS)}):")
for t in LADDER_TASKS:
    x, y = ladder(t)
    _, r = ret[t]
    print(f"  {t:16s} {CLASS[t]:4s} " +
          "  ".join(f"{xx/1024:.0f}k={yy:.4f}({rr:.2f})" for xx, yy, rr in zip(x, y, r)))
print(f"[{FIGURE_LABEL}] wrote {HERE}")
