"""CTC-FAMILY: does the chunked-attention penalty reproduce across model families?

Two panels, one per CTC class, four families on each: an $O_T(N^2)$ pair-finding task
(contradiction) and an $O_T(N)$ retrieval task (HotpotQA). Solid = full attention, dashed =
chunked. The point of the pairing is that the SAME models, mask and ladders lose almost nothing on
retrieval and a great deal on pair-finding -- so the penalty is priced by the task's traversal
complexity, not by chunking being generally harmful.

    python3 paperdraft/figures/make_fig_family.py     (writes ctc_family_figure.{png,pdf})

⚠ THE NUMBERS ARE IMPORTED, NOT RETYPED. `SERIES` comes from visualizations/make_family_figure.py,
which is also what the HTML artifact renders, so the paper figure and the artifact cannot drift.
That module reads most series from result JSONs on disk; the few literal dicts in it are the ones
whose source dirs are not in the repo.

⚠ CONTRADICTION AND HOTPOTQA SIT ON DIFFERENT BOTTOM RUNGS (2.5k vs 2k) and that is deliberate:
the realistic contradiction ladder starts at n=56 because n=44 falls below the training minimum of
52. Each panel therefore draws its own x ladder rather than a shared one.

⚠ TWO SERIES ARE DELIBERATELY ABSENT AND ARE ANNOTATED ON THE FIGURE, because an unexplained gap
in a 4-family panel reads as a measured collapse:
  * Olmo-Hybrid-7B chunked on contradiction -- trained, but its final CE was 0.958 against
    OLMo-3's 0.171 on identical task/data/steps, i.e. an optimization failure. Its own qdmatch
    chunked arm converged (CE 0.156) and scores 0.931 at 2k, so the backbone is fine. Publishing
    the 0.113/0.062/0.021/0.006 ladder as a hybrid-vs-standard finding would be wrong.
  * Olmo-Hybrid-7B on HotpotQA, both arms -- never trained on it; its retrieval-family run was
    qdmatch_hpqa, a different ($O_T(N^2)$) task.
"""
import importlib.util
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

FIGURE_LABEL = "CTC-FAMILY"
META = {"Title": f"{FIGURE_LABEL} — chunked-attention penalty across four model families, "
                 f"one O(N^2) task and one O(N) task",
        "Creator": "make_fig_family.py"}

# Import the artifact's data module by path -- visualizations/ is not a package.
_spec = importlib.util.spec_from_file_location(
    "_famdata", os.path.join(REPO, "visualizations", "make_family_figure.py"))
_fam = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fam)
SERIES, RUNG_LABEL = _fam.SERIES, _fam.RUNG_LABEL

PANELS = [
    ("contradiction", "Contradiction", r"set-F1", r"(high CTC, $O_T(N^2)$)", _fam.RUNGS),
    ("hotpotqa", "HotpotQA", r"gold-ID F1", r"(low CTC, $O_T(N)$)", _fam.HPQA_RUNGS),
]

# One hue per family, held across both panels so a reader tracks a model by colour alone. Blue and
# orange are reserved for the low/high CTC class split in CTC-GRID and CTC-PENALTY, so a family
# palette that reused them would collide with the class meaning those figures establish.
COLOR = {
    "Qwen3.5-4B":     "#2f6f4f",
    "OLMo-3-7B":      "#8a4fa8",
    "Olmo-Hybrid-7B": "#b08a1e",
    "Llama-3.2-3B":   "#b03a48",
}
TXT = "#111111"
CLASS_BLUE, CLASS_ORANGE = "#3f7dc4", "#e2622f"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Liberation Serif", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 15,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.edgecolor": "#c9c9c9",
    "axes.labelcolor": TXT,
    "xtick.color": TXT,
    "ytick.color": TXT,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# ⚠ THE Y-AXIS IS CUT, AND THE LIMITS ARE SHARED ACROSS BOTH PANELS ON PURPOSE.
# Starting at 0 wasted the bottom half of both panels -- nothing plots below 0.53. But letting each
# panel autoscale to its own data would be worse than the wasted space: HotpotQA spans roughly
# 0.87-1.00 and contradiction 0.54-0.99, so independent limits would stretch a 0.09 retrieval drop
# to look exactly as steep as a 0.34 pair-finding collapse, and the figure's entire argument is
# that those two are different in magnitude. One shared window keeps the slopes comparable by eye.
# Computed from the data rather than hardcoded so the floor cannot silently clip a future series.
_vals = [v for task, *_ in PANELS for d in _fam.SERIES[task].values() for v in d.values()]
YMIN = min(0.95, (int(min(_vals) * 20) / 20) - 0.05)   # round down to a 0.05 tick, then pad
YMAX = 1.02

fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.3))
# top leaves room for three stacked bands above the axes: the figure legend, the panel title, and
# the CTC-class line under it. The title pad is what separates the latter two -- at the default the
# class label lands on top of the title.
fig.subplots_adjust(left=0.06, right=0.985, top=0.735, bottom=0.175, wspace=0.17)

summary = []
for ax, (task, title, metric, cls, rungs) in zip(axes, PANELS):
    xs = list(range(len(rungs)))
    for fam in _fam.FAMILIES:
        for arm, style, marker in (("dense", "-", "o"), ("chunked", "--", "s")):
            d = SERIES[task].get((fam, arm), {})
            pts = [(i, d[r]) for i, r in enumerate(rungs) if r in d]
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    style, marker=marker, color=COLOR[fam], linewidth=2.1,
                    markersize=5.6, markeredgewidth=0,
                    alpha=1.0 if arm == "dense" else 0.92, zorder=3)
        dd, cc = SERIES[task].get((fam, "dense"), {}), SERIES[task].get((fam, "chunked"), {})
        deep = [r for r in rungs if r in dd and r in cc]
        if deep:
            summary.append((task, fam, rungs[-1] if deep[-1] == rungs[-1] else deep[-1],
                            dd[deep[-1]], cc[deep[-1]], dd[deep[-1]] - cc[deep[-1]]))

    ax.set_xticks(xs)
    ax.set_xticklabels([RUNG_LABEL[r] for r in rungs])
    ax.set_xlim(-0.16, len(rungs) - 0.84)
    ax.set_ylim(YMIN, YMAX)
    ax.set_xlabel("Context Length")
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, fontsize=17, color=TXT, pad=34)
    ax.text(0.5, 1.028, cls, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=14, color=CLASS_ORANGE if task == "contradiction" else CLASS_BLUE)
    ax.set_ylabel(f"Score ({metric})")

# ⚠ THE TWO OLMO-HYBRID ABSENCE NOTES USED TO BE DRAWN IN-PANEL HERE AND ARE NOW CAPTION-ONLY.
# They still print in the CAPTION block below, and the reasons are in this module's docstring --
# the explanation moved, it was not retracted. Both absences remain deliberate: the contradiction
# chunked arm is an optimization failure (CE 0.958 vs OLMo-3's 0.171 on identical task/data/steps)
# and HotpotQA was never trained for that family. Anyone re-adding an in-panel note should put it
# back in ink at the bottom-left, where no series runs.

handles = [Line2D([], [], color=COLOR[f], linewidth=2.4, label=f) for f in _fam.FAMILIES]
handles += [Line2D([], [], color=TXT, linewidth=2.0, linestyle="-", marker="o",
                   markersize=5.4, label="Full attention"),
            Line2D([], [], color=TXT, linewidth=2.0, linestyle="--", marker="s",
                   markersize=5.4, label="Chunked attention")]
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.055, 1.005),
           ncol=6, frameon=False, handlelength=2.5, columnspacing=1.5,
           fontsize=13.5, labelcolor=TXT)

for ext, kw in (("png", {"dpi": 220}), ("pdf", {})):
    fig.savefig(os.path.join(HERE, f"ctc_family_figure.{ext}"),
                bbox_inches="tight", facecolor="white", metadata=META, **kw)

# ---------------- CAPTION material ----------------
print(f"[{FIGURE_LABEL}] wrote ctc_family_figure.png / .pdf in {HERE}\n")
print(f"{'task':14s} {'family':16s} {'rung':>7s} {'full':>7s} {'chunked':>8s} {'gap':>8s}")
print("-" * 64)
for task, fam, rung, d, c, g in summary:
    print(f"{task:14s} {fam:16s} {RUNG_LABEL[rung]:>7s} {d:7.4f} {c:8.4f} {g:+8.4f}")

print("\n--- caption ---")
print("  The chunked-attention penalty is a property of the task, not of the model family. Four "
      "families fine-tuned on identical shards and scored on identical ladders; solid = full "
      "attention, dashed = document-chunked (with curriculum mask-mixing). Every family loses "
      "little on $O_T(N)$ retrieval and a great deal on $O_T(N^2)$ pair-finding.")
print("  Contradiction is scored on contradiction_iid, the ladder matching the training shard for "
      "all four families; HotpotQA on the shared 2k-aligned ladder (verified identical by md5 "
      "across the two staging paths the families used).")
print("  Llama-3.2-3B contradiction is the ']]'-truncated re-score: it emits no EOS and rambles to "
      "the token cap on 87-100% of examples, which halves raw precision.")
# These two lines carry the absences that used to be drawn in-panel. They MUST reach the paper
# caption: a family with no chunked line reads as a measured collapse to zero otherwise.
print("  NOT SHOWN, Olmo-Hybrid-7B chunked on contradiction: the arm trained, but its final CE was "
      "0.958 against OLMo-3's 0.171 on identical task, data and steps -- an optimization failure. "
      "Its own qdmatch chunked arm converged (CE 0.156, 0.931 at 2k), so the backbone is sound.")
print("  NOT SHOWN, Olmo-Hybrid-7B on HotpotQA: never trained on it -- that family's "
      "retrieval-family run was qdmatch_hpqa, an $O_T(N^2)$ task.")
print("  eval_size = 500 per rung except the Llama contradiction ladder; a difference under ~0.04 "
      "is within noise.")
