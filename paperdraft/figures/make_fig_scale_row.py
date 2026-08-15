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

# ⚠ BLUE/ORANGE shared verbatim with make_fig2.py and the other CTC figures. Here they colour the
# CTC-CLASS GROUP HEADERS only -- inside the panels, colour encodes model scale (the purple ramp
# above), so the two encodings never appear on the same mark.
LOW = "#3f7dc4"
HIGH = "#e2622f"
FULL_LS = "-"
CHUNK_LS = (0, (1.6, 1.7))
# Paper figure: all TEXT is near-black ink -- no grey text. Everything that was a grey sub-caption
# or in-panel note is printed for the LaTeX caption instead; see the CAPTION block at the end.
TXT = "#111111"

# (task key, panel title, metric for the caption, CTC class)
PANELS = [
    ("hpqa", "HotpotQA", "gold-ID F1", "low"),
    ("fiqa", "BEIR FiQA", "set-F1", "low"),
    ("contra_real", "Contradiction", "set-F1", "high"),
    ("qdmatch_nq", "QDmatch (NQ)", "pair-F1", "high"),
]
# Contiguous panel spans that share a CTC class, drawn as a header + rule above the row.
GROUPS = [("Low CTC", 0, 1, LOW), ("High CTC", 2, 3, HIGH)]

# Conference format, identical to the other CTC figures so they sit together on a page.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Liberation Serif", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 14,
    "axes.labelsize": 14.5,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.edgecolor": "#c9c9c9",
    "axes.labelcolor": TXT,
    "xtick.color": TXT,
    "ytick.color": TXT,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


PANEL_NOTES = {}


def short_reason(reason):
    r = str(reason)
    if "OOM" in r:
        return "chunked run OOMed"
    if "not launched" in r or "never launched" in r:
        return "pair never launched"
    return r[:38] + "…"


# The headline claim and the method subtitle used to be drawn here as grey text. They go in the
# LaTeX caption now (printed at the end of the run); `top` instead leaves room for the CTC-class
# group headers, which are drawn after the panels once the axes rectangles are known.
# Vertical stack above the panels, top down: legend, CTC-class group headers, panel titles. `top`
# has to leave room for all three; `bottom` only needs the x label now that the legend moved up.
fig, axes = plt.subplots(1, 4, figsize=(14.6, 5.0))
fig.subplots_adjust(left=0.055, right=0.992, top=0.700, bottom=0.135, wspace=0.22)

for i, (task, title, metric, cls) in enumerate(PANELS):
    ax = axes[i]
    notes = []
    for sc in SCALES:
        # ⚠ Sourced from `raw`, not `df`, and each ARM is plotted independently.
        #
        # The export marks a whole (task, scale) row plotted_in_figure="no" when EITHER arm is
        # missing, because a missing arm makes `relative_gap` undefined. That is right for a gap
        # figure and wrong for this one: these panels plot absolute scores, so a measured arm is
        # still a measurement even when its partner never ran. Requiring the pair silently threw
        # away contradiction 0.8B DENSE (0.962 -> 0.861 across the ladder), which exists and is
        # fine -- only its chunked partner OOMed.
        #
        # Every value plotted here is a real graded run; the only thing being overridden is the
        # export's pair-completeness rule. An arm with no data simply draws no line, and the
        # missing side is named in the caption so the absence is never read as a measured zero.
        s = raw[(raw["task"] == task) & (raw["model_scale"] == sc)].copy()
        s["context_tokens"] = s["context_tokens"].astype(float)
        s = s.sort_values("context_tokens")
        s_full = s[s["full_attention"].notna()]
        s_chunk = s[s["chunked_attention"].notna()]
        if len(s_full) == 0 and len(s_chunk) == 0:
            why = short_reason(s["drop_reason"].iloc[0]) if len(s) else "no runs"
            notes.append(f"{SCALE_LABEL[sc]}: {why}")
            continue
        c = SCALE_COLOR[sc]
        if len(s_chunk):
            ax.plot(s_chunk["context_tokens"], s_chunk["chunked_attention"], ls=CHUNK_LS, color=c,
                    lw=2.3, marker="o", ms=5.0, mew=0, zorder=2)
        if len(s_full):
            ax.plot(s_full["context_tokens"], s_full["full_attention"], ls=FULL_LS, color=c,
                    lw=2.3, marker="o", ms=5.0, mew=0, zorder=3)
        if len(s_full) and not len(s_chunk):
            why = short_reason(s["drop_reason"].iloc[0]) if len(s) else "no chunked run"
            notes.append(f"{SCALE_LABEL[sc]}: full arm only ({why})")
        elif len(s_chunk) and not len(s_full):
            why = short_reason(s["drop_reason"].iloc[0]) if len(s) else "no full run"
            notes.append(f"{SCALE_LABEL[sc]}: chunked arm only ({why})")

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
    ax.set_xlabel("Context Length", color=TXT, labelpad=6)
    if i == 0:
        ax.set_ylabel("Score (Task Metric)", color=TXT, labelpad=7)

    # Centred, no (a)/(b) prefix -- the CTC-class group header above the row does the grouping that
    # the letters used to imply, and the per-panel metric moved to the caption.
    ax.set_title(title, fontsize=16, color=TXT, pad=12, loc="center")
    PANEL_NOTES[task] = notes

# ---------------- CTC-class group headers ----------------
# Drawn after the panels because they are centred on axes rectangles and need the laid-out
# positions. Label only -- the coloured rule that used to span each group was removed; the class
# colour on the text carries it, and the rules competed with the panel titles underneath.
for name, i0, i1, colour in GROUPS:
    x0 = axes[i0].get_position().x0
    x1 = axes[i1].get_position().x1
    # y sits BELOW the legend band and ABOVE the panel titles. The legend is anchored at 1.0 and is
    # roughly 0.06 tall, so anything above ~0.86 collides with it.
    fig.text((x0 + x1) / 2, 0.825, name, transform=fig.transFigure, fontsize=17,
             color=colour, ha="center", va="bottom")

# Legend on top, above the group headers. Scale is a filled swatch (it encodes the purple ramp
# inside the panels); attention type is a line style.
handles = [Patch(facecolor=SCALE_COLOR[sc], label=f"Qwen3.5-{SCALE_LABEL[sc]}") for sc in SCALES]
handles += [Line2D([0], [0], color=TXT, lw=2.2, ls=FULL_LS, label="Full Attention"),
            Line2D([0], [0], color=TXT, lw=2.2, ls=CHUNK_LS, label="Chunked Attention")]
fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=14,
           bbox_to_anchor=(0.5, 1.0), columnspacing=2.4, handletextpad=0.7)

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

# ---------------- CAPTION material ----------------
# Previously drawn on the figure as grey text. The "not shown" notes especially must survive: a
# missing arm in a panel would otherwise read as a measured gap rather than a run that never landed.
print("\n--- caption ---")
# ⚠ THE CLAIM IS "DOES NOT CLOSE", NOT "DOES NOT NARROW", AND THE DIFFERENCE IS LOAD-BEARING.
# Once the 0.8B contradiction chunked arm landed (2026-08-15) that panel has all three scales, and
# its relative gap narrows monotonically with size: 0.396 -> 0.353 -> 0.261. qdmatch (NQ) moves the
# other way, 0.729 -> 0.848. So scale helps on one high-CTC task and hurts on the other, and a
# blanket "scaling does not help" would be contradicted by the contradiction panel itself.
print("  Scaling 0.8B to 4B does not close the chunked-attention gap on high-CTC tasks. Qwen3.5 at "
      "three parameter counts, identical training shards and identical evaluation ladders; solid = "
      "full attention, dotted = chunked. Low CTC = $O_T(N)$; high CTC = $O_T(N^2)$. On contradiction "
      "the gap narrows steadily with scale (0.40/0.35/0.26 relative at 32k) but is still 0.26 at 4B; "
      "on QDmatch (NQ) it widens (0.73 at 2B to 0.85 at 4B). Neither trend reaches parity.")
for task, title, metric, cls in PANELS:
    print(f"  {title} -- {cls} CTC, {metric}.")
print("  HotpotQA: all three scales overlap; the gap never opens.")
for task, notes in PANEL_NOTES.items():
    if notes:
        print(f"  NOT SHOWN in {task}: {'; '.join(notes)} -- absent runs, not measured zeros.")
print(f"[{FIGURE_LABEL}] wrote {HERE}")
