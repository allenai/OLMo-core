"""CTC model-scale figure: does the chunked-attention penalty shrink as the model grows?

Reads `data/ctc_scale_data.csv` (built by `export_ctc_suite_data.py` from the harvested grade
JSONs) and writes `ctc_scale_figure.{png,pdf}`.

Qwen3.5 at 0.8B / 2B / 4B, trained on identical shards and evaluated on identical ladders, so the
only thing varying across a panel's three colours is parameter count.

Colour = model scale (a purple ramp, kept distinct from the blue/red CTC-class colours used in
`make_fig.py`), solid = full attention, dotted = chunked attention.

Panels (a)-(e) are one task each; panel (f) summarises the relative gap at a fixed rung.

Usage:  python3 paperdraft/figures/make_fig_scale.py
"""
import os

import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))

# Stable label for this figure, so it can be referred to by name across sessions.
# The index of every label lives in paperdraft/figures/README.md. It is stamped into the
# PDF/PNG metadata too, so `pdfinfo <out>.pdf | grep Title` names the script that made it.
FIGURE_LABEL = "CTC-SCALE"
META = {"Title": f"{FIGURE_LABEL} — model scaling 0.8B/2B/4B, 5 task panels + gap summary", "Creator": "make_fig_scale.py"}

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

# A relative gap is 1 - chunked/full, so it is meaningless once the full arm is itself at the
# floor -- it becomes a ratio of noise. Same rule and threshold as make_fig2.py.
FLOOR = 0.10

# The rung each task is summarised at in panel (f). reorder's ladder stops at 16k and its full arm
# is already under the floor there, so it is summarised one rung down; the tick label says so.
REF_RUNG = {"hpqa": 32768, "fiqa": 32768, "contra_real": 32768, "qdmatch_nq": 32768,
            "reorder": 8192}

PANELS = [
    ("hpqa", "HotpotQA", "HotpotQA", "hpqa · low CTC $O_T(N)$ · gold-ID F1", LOW),
    ("fiqa", "BEIR FiQA", "FiQA", "fiqa · low CTC $O_T(N)$ · set-F1", LOW),
    ("contra_real", "contradiction", "contra.", "contradiction · high CTC $O_T(N^2)$ · set-F1",
     HIGH),
    ("qdmatch_nq", "qdmatch (NQ)", "qdmatch", "qdmatch_nq · high CTC $O_T(N^2)$ · pair-F1", HIGH),
    ("reorder", "reordering", "reorder", "reorder · high CTC $O_T(N^2)$ · Kendall tau", HIGH),
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


def style_axes(ax, ylab=False):
    ax.set_xscale("log", base=2)
    ax.set_xticks(LADDER)
    ax.set_xticklabels(["2k", "4k", "8k", "16k", "32k"])
    ax.set_xlim(1750, 40000)
    ax.set_ylim(0, 1.02)
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


def short_reason(reason):
    """A few words for an in-panel note; the full text goes in the figure footnote."""
    r = str(reason)
    if "OOM" in r:
        return "chunked run OOMed"
    if "not launched" in r or "never launched" in r:
        return "pair never launched"
    return r[:38] + "…"


fig = plt.figure(figsize=(13.6, 8.6))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.28,
                       left=0.058, right=0.985, top=0.875, bottom=0.20)

fig.suptitle("Scaling 0.8B → 4B does not close the chunked-attention gap on high-CTC tasks",
             fontsize=14, color="#2b2b2b", x=0.058, ha="left", y=0.975)
fig.text(0.058, 0.935,
         "Qwen3.5 at three parameter counts, identical training shards and identical evaluation "
         "ladders; solid = full attention, dotted = chunked. The gap moves in both directions "
         "with scale (contradiction 35%→26%, qdmatch 73%→85%) but stays large either way.",
         fontsize=8.6, color=MUTED, ha="left", va="bottom")

# ---------------- (a)-(e) per-task ladders ----------------
for i, (task, title, _short, sub, cls_color) in enumerate(PANELS):
    ax = fig.add_subplot(gs[i // 3, i % 3])
    notes = []
    for sc in SCALES:
        s = df[(df["task"] == task) & (df["model_scale"] == sc)].sort_values("context_tokens")
        s = s[s["full_attention"].notna() & s["chunked_attention"].notna()]
        if len(s) == 0:
            miss = raw[(raw["task"] == task) & (raw["model_scale"] == sc)]
            why = short_reason(miss["drop_reason"].iloc[0]) if len(miss) else "no runs"
            notes.append(f"{SCALE_LABEL[sc]}: {why}")
            continue
        c = SCALE_COLOR[sc]
        ax.plot(s["context_tokens"], s["chunked_attention"], ls=CHUNK_LS, color=c, lw=1.8,
                marker="o", ms=3.8, mew=0, zorder=2)
        ax.plot(s["context_tokens"], s["full_attention"], ls=FULL_LS, color=c, lw=1.8,
                marker="o", ms=3.8, mew=0, zorder=3)
    style_axes(ax, ylab=(i % 3 == 0))
    ax.set_title(f"({'abcde'[i]})  {title}", fontsize=11.5, color="#2b2b2b", pad=16, loc="left")
    ax.text(0.0, 1.012, sub, transform=ax.transAxes, fontsize=7.6, color=cls_color,
            ha="left", va="bottom", alpha=0.85)
    if task == "hpqa":
        ax.text(0.03, 0.14, "all three scales overlap: the gap never opens", fontsize=7.2,
                color=MUTED, transform=ax.transAxes, ha="left", va="bottom")
    if notes:
        # Spelled out rather than left as a silent absence: the export's drop_reason for the
        # 0.8B chunked runs says explicitly not to read the missing arm as a gap.
        ax.text(0.03, 0.06, "not shown — " + "; ".join(notes), transform=ax.transAxes,
                fontsize=7.2, color=MUTED, ha="left", va="bottom")

# ---------------- (f) relative gap at a fixed rung ----------------
ax = fig.add_subplot(gs[1, 2])
groups, bar_w = [], 0.26
for task, _title, short, _sub, cls_color in PANELS:
    rung = REF_RUNG[task]
    vals = {}
    for sc in SCALES:
        r = df[(df["task"] == task) & (df["model_scale"] == sc) &
               (df["context_tokens"] == rung)]
        if len(r) == 0:
            vals[sc] = (None, "no pair")
        elif r["full_attention"].iloc[0] < FLOOR:
            vals[sc] = (None, "floor")     # full arm at the floor -> the ratio is noise/noise
        elif pd.isna(r["relative_gap"].iloc[0]):
            vals[sc] = (None, "no pair")
        else:
            vals[sc] = (float(r["relative_gap"].iloc[0]), None)
    groups.append((task, short, rung, vals, cls_color))

for gi, (task, short, rung, vals, cls_color) in enumerate(groups):
    for si, sc in enumerate(SCALES):
        x = gi + (si - 1) * bar_w
        v, why = vals[sc]
        if v is None:
            ax.text(x, 0.015, "–", ha="center", va="bottom", fontsize=9, color="#bcbcbc")
            continue
        ax.bar(x, v, width=bar_w * 0.88, color=SCALE_COLOR[sc], lw=0, zorder=3)
        txt = f"{v * 100:+.1f}%" if abs(v) < 0.1 else f"{v:.0%}"
        ax.text(x, max(v, 0) + 0.02, txt, ha="center", va="bottom", fontsize=6.8,
                color=SCALE_COLOR[sc], fontweight="bold")
ax.axhline(0, color="#c9c9c9", lw=0.9, zorder=2)

ax.set_xticks(range(len(groups)))
ax.set_xticklabels([f"{s}\n@{r // 1024}k" for _, s, r, _, _ in groups], fontsize=8.0)
for lbl, (_, _, _, _, c) in zip(ax.get_xticklabels(), groups):
    lbl.set_color(c)
ax.set_ylim(-0.10, 1.0)
ax.set_yticks(np.arange(0, 1.001, 0.2))
ax.set_yticklabels([f"{v:.0%}" for v in np.arange(0, 1.001, 0.2)])
ax.set_ylabel("relative gap  $1-$ chunked$/$full", color=TXT, labelpad=6)
ax.yaxis.grid(True, color="#e6e6e6", lw=0.8)
ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color("#c9c9c9")
ax.spines["bottom"].set_color("#c9c9c9")
ax.tick_params(length=3, width=0.8)
ax.set_title("(f)  the gap at the top rung", fontsize=11.5, color="#2b2b2b", pad=16, loc="left")
ax.text(0.0, 1.012, "“–” = no complete pair, or the full arm is under the "
        f"{FLOOR:.2f} floor where the ratio is noise",
        transform=ax.transAxes, fontsize=7.2, color=MUTED, ha="left", va="bottom")

# ---------------- legend + provenance footnote ----------------
handles = [Patch(facecolor=SCALE_COLOR[sc], label=f"Qwen3.5-{SCALE_LABEL[sc]}") for sc in SCALES]
handles += [Line2D([0], [0], color="#6b6b6b", lw=2.0, ls=FULL_LS, label="full attention"),
            Line2D([0], [0], color="#6b6b6b", lw=2.0, ls=CHUNK_LS, label="chunked attention")]
fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9.5,
           bbox_to_anchor=(0.5, 0.115), columnspacing=2.2)

dropped = raw[raw["plotted_in_figure"] == "no"]
lines = []
for reason, grp in dropped.groupby("drop_reason"):
    who = sorted({f"{SCALE_LABEL[s]} {t}" for s, t in zip(grp["model_scale"], grp["task"])})
    short = reason if len(reason) <= 170 else reason[:167].rsplit(" ", 1)[0] + "…"
    lines.append(f"• {', '.join(who)} — {short}")
fig.text(0.058, 0.082, f"Excluded runs ({len(dropped)} of {len(raw)} exported rows):",
         fontsize=8.2, color="#5f5f5f", ha="left", va="top", fontweight="bold")
fig.text(0.058, 0.062, "\n".join(lines), fontsize=7.4, color=MUTED, ha="left", va="top",
         linespacing=1.6)

fig.savefig(os.path.join(HERE, "ctc_scale_figure.png"), dpi=220, bbox_inches="tight",
            facecolor="white", metadata=META)
fig.savefig(os.path.join(HERE, "ctc_scale_figure.pdf"), bbox_inches="tight", facecolor="white", metadata=META)

for task, _short, rung, vals, _ in groups:
    cells = "  ".join(f"{SCALE_LABEL[s]}=" +
                      (f"{vals[s][0]:+.3f}" if vals[s][0] is not None else f"–({vals[s][1]})")
                      for s in SCALES)
    print(f"{task:14s} @{rung//1024:>2}k  {cells}")
print(f"[{FIGURE_LABEL}] wrote {HERE}")
