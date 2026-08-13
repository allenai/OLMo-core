"""Figure 4 (fig:sec5-cross-task) — cross-task scaling of full vs. chunked attention.

Replaces the 6-panel `section5_cross_task.pdf` from the corpus-reasoning repo. 3x3 grid,
columns = CTC class (O(N) | O(NM) | O(N^2) and above), 9 panels, 17 task settings.

KEPT UNCHANGED from the previous version (same runs, same numbers, same conventions) --
these are the per-N sweeps from `corpus-reasoning/paperdraft/iclr2026/figures/section5_cross_task.py`,
read out of /scratch/users/prasann/corpus-reasoning/outputs/eval_results on 2026-08-03:
  * Outlier (wiki), N=20/50/100, scale-k (solid) vs fixed-k (dashed)   [Qwen3.5-4B]
  * Grouping (OpenAlex), N=20/50/100                                   [Qwen3.5-4B]
  * Contradiction, N=20/50/100/500                                     [Qwen3.5-4B]
  * Reorder (Gutenberg), N=5/20/50, + the one pure-chunked (no mask mixing) X point
                                                                       [Qwen3.5-9B]

NEW, from the CTC-suite artifact ("CTC-suite grid - Qwen3.5-4B dense vs chunked", the
assembled table in results/ctc_suite/dense_vs_chunked_table.md, updated 2026-07-27). All
Qwen3.5-4B; ladder = token-budget rungs 2k/4k/8k/16k/32k, which scale N per task. dense =
the `-full` checkpoint under full attention; chunked = the `-cmix` checkpoint (curriculum
mask mixing) under document-chunked attention, i.e. best-case chunked, the same convention
as the kept panels. eval_size=500 per rung except where noted.
  * Retrieval:         NIAH-contra, NQ, MS MARCO           (gold-ID F1)
  * Ranking/subjective: MS MARCO rerank (MRR@10), OBLIQ    (gold-ID F1)
  * Absence + GovReport: absence (set-F1), GovReport (ROUGE-1)
  * Outlier (wiki) again, on the token ladder -- the same task as the kept per-N panel,
    carried out to 32k, where the gap the per-N panel opens keeps widening (0.02 -> 0.30).
  * Cross-document matching: strmatch, qdmatch (HPQA), xabsence, textgroups

Blank rungs are runs that do not exist, not zeros. Excluded settings, with reasons:
  * hotpotqa, oolong  -- no `-full` checkpoint (chunked-only), so no gap to plot. This is
    why the old NQ/HotpotQA 0.8B-LoRA panels are gone: the 4B suite supersedes them for
    retrieval, but it cannot supply HotpotQA. Body text that cites the HPQA panel needs a
    rewrite.
  * scifact, groups4  -- the `-full` checkpoint const-collapsed during training (under FULL
    attention), so the dense arm measures a broken checkpoint, not full attention.
  * mathmatch, NarrativeQA (helmet_qa) -- near-floor in both arms (<=0.08); a ratio between
    two floor numbers is noise.
  * outlier (Amazon), grouping (unlabeled), absence (official), redundancy -- removed from
    the suite on request 2026-08-03.
  * cycle -- O(N^3) and saturated in BOTH arms (dense 1.000/1.000/0.998/1.000 vs chunked
    0.996/0.994/0.994/0.992). It is a genuine no-gap high-CTC case; dropped here for the
    same reason it was dropped from Figure 1, so it is a counterexample the figure does not
    show. Set SHOW_CYCLE=True to put it back in the cross-doc panel.
  * grouping_labeled -- its two suite arms are near-identical at every rung
    (0.439/0.439, 0.365/0.370, 0.231/0.226, 0.054/0.051), which is not what two
    independently trained arms look like; worth checking the mask was applied before use.

eval-size caveat: OBLIQ is 126 examples per rung (SE +-0.045, under the 500 minimum) and
contradiction is 488 (the whole held-out file). Everything else is 500.

Encoding: colour = attention (blue = full, grey = chunked + mask mixing). Marker = task
within a multi-task panel. Linestyle is reserved for the data variant (outlier scale-k vs
fixed-k). Never marker = arm, so the two encodings do not collide.

Usage:  python3 figures/fig4_cross_task.py    (writes fig4_cross_task.{pdf,png})
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SHOW_CYCLE = False

K = 1024
SUITE = [2 * K, 4 * K, 8 * K, 16 * K, 32 * K]

# Colour = attention type, matching the other paper figures. The chunked grey is stepped
# from the previous #b0b0b0 to #8a8a8a: at this figure's print size (\textwidth, ~1.7in
# panels) the lighter grey drops out against the grid on thin multi-task lines.
STD_BLUE, CHUNK_GREY = "#1f77b4", "#8a8a8a"
INK, MUTED = "#161a20", "#5c6673"
MARKERS = ["o", "s", "^", "D", "v"]


def pairs(xs, ys):
    """Drop the rungs where a run does not exist (None), keeping x and y aligned."""
    return ([x for x, y in zip(xs, ys) if y is not None],
            [y for y in ys if y is not None])


# ── panel data ────────────────────────────────────────────────────────────────────────
# Each task: (display name, xs, dense ys, chunked ys). None = no run at that rung.

RETRIEVAL = [
    ("NIAH",     SUITE, [.988, .988, .976, .984, .966], [.984, .994, .976, .978, .970]),
    ("NQ",       SUITE, [.904, .930, .864, None, None], [.964, .942, .896, None, None]),
    ("MS MARCO", SUITE, [.961, .939, .917, .900, None], [.897, .902, .868, .871, None]),
]
RANKING = [
    ("rerank",   SUITE, [.989, .971, .960, .952, None], [.980, .966, .950, .949, None]),
    ("OBLIQ",    SUITE, [.740, .659, .608, .277, .327], [.689, .605, .563, .262, .293]),
]
LABELING = [
    ("absence",   SUITE, [.964, .976, .986, .932, None], [.949, .961, .981, .945, None]),
    ("GovReport", SUITE, [.329, .338, .341, None, None], [.335, .341, .346, .347, .349]),
]
OUTLIER_TOK = [
    ("outlier",  SUITE, [.982, .956, .877, .679, .428], [.962, .869, .641, .343, .125]),
]
CROSSDOC = [
    ("strmatch",   SUITE, [.999, .997, .997, .995, .994], [.003, .003, .000, .001, .000]),
    ("qdmatch",    SUITE, [.999, .997, .998, .992, .981], [.650, .760, .668, .547, .333]),
    ("xabsence",   SUITE, [.609, .545, .528, .505, .518], [.135, .057, .037, .012, .008]),
    ("textgroups", SUITE, [.196, .090, .054, .051, .047], [.087, .021, .019, .007, .001]),
]
if SHOW_CYCLE:
    CROSSDOC.append(
        ("cycle", SUITE, [1.000, 1.000, .998, 1.000, None], [.996, .994, .994, .992, None]))

# Kept per-N panels (values read from the old eval JSONs; see docstring).
OUTLIER_N = [20, 50, 100]
OUTLIER_SCALEK = ([.9225, .88625, .68125], [.8625, .80917, .5175])
OUTLIER_FIXEDK = ([.9475, .92167, .9325], [.9525, .955, .97625])
GROUPING_N = [20, 50, 100]
GROUPING = ([.71120, .51704, .19626], [.65758, .38883, .03102])
CONTRA_N = [20, 50, 100, 500]
CONTRA = ([.96311, .94877, .94467, .88525], [.90779, .84221, .83811, .53074])
REORDER_N = [5, 20, 50]
REORDER = ([.76520, .44284, .07349], [.70240, .19305, .02143])
REORDER_NOMIX = (20, .04872)   # pure chunked, no mask mixing


def multi(ax, tasks, title, ylabel, xlabel, xs_ticks, legend_loc="lower left",
          legend_kw=None, ytop=1.06):
    """A panel holding one or more tasks: colour = arm, marker = task."""
    handles = []
    for i, (name, xs, dense, chunked) in enumerate(tasks):
        m = MARKERS[i % len(MARKERS)]
        for ys, color in ((dense, STD_BLUE), (chunked, CHUNK_GREY)):
            px, py = pairs(xs, ys)
            if px:
                ax.plot(px, py, marker=m, color=color, linewidth=1.3, markersize=3.2,
                        markeredgewidth=0)
        if len(tasks) > 1:
            handles.append(Line2D([], [], marker=m, color=MUTED, lw=0, ms=3.4, label=name))
    _style(ax, title, ylabel, xlabel, xs_ticks, ytop=ytop)
    if handles:
        # Opaque-ish white patch, no border: in the crowded cross-document panel the only
        # clear band still has a line running through it, and a frameless legend there is
        # unreadable. Invisible in the panels that have real whitespace.
        ax.legend(handles=handles, loc=legend_loc, fontsize=5.2,
                  handletextpad=0.2, labelspacing=0.25, borderaxespad=0.2,
                  frameon=True, facecolor="white", edgecolor="none", framealpha=0.85,
                  **(legend_kw or {}))


def _style(ax, title, ylabel, xlabel, xs_ticks, ytop=1.06):
    ax.set_xscale("log")
    ax.set_xticks(xs_ticks)
    ax.set_xticklabels([f"{x // K}k" if x >= K and x % K == 0 else str(x)
                        for x in xs_ticks])
    ax.minorticks_off()
    ax.set_ylim(-0.04, ytop)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "1"])
    ax.set_xlabel(xlabel, fontsize=6, labelpad=1.5)
    ax.set_ylabel(ylabel, fontsize=6, labelpad=1.5)
    ax.set_title(title, fontsize=7, pad=3.5)
    ax.tick_params(axis="both", labelsize=5.5, pad=1.5)
    ax.grid(True, linestyle=":", alpha=0.4, linewidth=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.edgecolor": MUTED,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.size": 7,
    })
    # Sized for \textwidth (5.5in) at scale 1, so the point sizes above are the printed
    # ones. Height is capped near 4.3in on purpose: with a caption this long, a taller
    # float exceeds \topfraction and LaTeX defers the figure to the end of the document.
    fig, ax = plt.subplots(3, 3, figsize=(5.6, 4.35))

    tok = "context tokens"
    ndoc = "documents $N$"

    # ── column 1: O(N) ──
    multi(ax[0, 0], RETRIEVAL, "Retrieval", "gold-ID F1", tok, SUITE)
    multi(ax[1, 0], RANKING, "Ranking / subjective", "MRR@10, F1", tok, SUITE)
    multi(ax[2, 0], LABELING, "Absence, summarization", "F1, ROUGE-1", tok, SUITE)

    # ── column 2: O(NM) ──
    a = ax[0, 1]
    for (dense, chunked), ls, tag in ((OUTLIER_SCALEK, "-", "scale-$k$"),
                                      (OUTLIER_FIXEDK, "--", "fixed $k$")):
        a.plot(OUTLIER_N, dense, marker="o", linestyle=ls, color=STD_BLUE,
               linewidth=1.3, markersize=3.2, markeredgewidth=0)
        a.plot(OUTLIER_N, chunked, marker="o", linestyle=ls, color=CHUNK_GREY,
               linewidth=1.3, markersize=3.2, markeredgewidth=0)
    _style(a, "Outlier (wiki), $N$ sweep", "set-F1", ndoc, OUTLIER_N)
    a.legend(handles=[Line2D([], [], color=MUTED, ls="-", lw=1.1, label="scale-$k$"),
                      Line2D([], [], color=MUTED, ls="--", lw=1.1, label="fixed $k$")],
             frameon=False, loc="lower left", fontsize=5.2, handlelength=2.0,
             handletextpad=0.3, labelspacing=0.25, borderaxespad=0.2)

    a = ax[1, 1]
    a.plot(GROUPING_N, GROUPING[0], marker="o", color=STD_BLUE, linewidth=1.3,
           markersize=3.2, markeredgewidth=0)
    a.plot(GROUPING_N, GROUPING[1], marker="o", color=CHUNK_GREY, linewidth=1.3,
           markersize=3.2, markeredgewidth=0)
    _style(a, "Grouping (OpenAlex)", "pairwise F1", ndoc, GROUPING_N)

    multi(ax[2, 1], OUTLIER_TOK, "Outlier (wiki), token sweep", "set-F1", tok, SUITE)

    # ── column 3: O(N^2) and above ──
    a = ax[0, 2]
    a.plot(CONTRA_N, CONTRA[0], marker="o", color=STD_BLUE, linewidth=1.3,
           markersize=3.2, markeredgewidth=0)
    a.plot(CONTRA_N, CONTRA[1], marker="o", color=CHUNK_GREY, linewidth=1.3,
           markersize=3.2, markeredgewidth=0)
    _style(a, "Contradiction", "EM", ndoc, CONTRA_N)

    a = ax[1, 2]
    a.axhline(0.0, color=INK, linestyle=":", linewidth=1.0)
    a.plot(REORDER_N, REORDER[0], marker="o", color=STD_BLUE, linewidth=1.3,
           markersize=3.2, markeredgewidth=0)
    a.plot(REORDER_N, REORDER[1], marker="o", color=CHUNK_GREY, linewidth=1.3,
           markersize=3.2, markeredgewidth=0)
    a.plot([REORDER_NOMIX[0]], [REORDER_NOMIX[1]], marker="x", linestyle="None",
           color=INK, markersize=4.5, markeredgewidth=1.2)
    _style(a, "Reorder (Gutenberg)", r"Kendall $\tau$", ndoc, REORDER_N)
    a.annotate("random", (REORDER_N[0], 0.0), xytext=(2, 3),
               textcoords="offset points", ha="left", va="bottom",
               fontsize=5.2, color=MUTED)

    # Four tasks in one small panel and no clear band anywhere inside it: two dense lines
    # are pinned at 1.0, the chunked ones pile up near 0, and qdmatch sweeps through the
    # middle. Rather than occlude data, this panel alone gets headroom above 1.0 for the
    # legend; its ticks are still 0/0.5/1, and it is the panel whose vertical scale means
    # least anyway (four different metrics share the axis).
    multi(ax[2, 2], CROSSDOC, "Cross-document matching", "task metric", tok, SUITE,
          ytop=1.52, legend_loc="upper center",
          legend_kw={"bbox_to_anchor": (0.5, 1.03), "ncol": 2, "columnspacing": 0.6})

    # Column headers = CTC class.
    for col, header in enumerate([r"$O_T(N)$", r"$O_T(N\,M)$",
                                  r"$O_T(N^2)$ and above"]):
        ax[0, col].annotate(header, xy=(0.5, 1.30), xycoords="axes fraction",
                            ha="center", va="bottom", fontsize=8.5)

    fig.legend(handles=[
        Line2D([], [], color=STD_BLUE, marker="o", ms=3.4, lw=1.4,
               label="Full attention"),
        Line2D([], [], color=CHUNK_GREY, marker="o", ms=3.4, lw=1.4,
               label="Chunked + mask mixing"),
        Line2D([], [], color=INK, marker="x", ms=4.5, lw=0, markeredgewidth=1.2,
               label="Chunked, no mask mixing"),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.015), ncol=3, frameon=False,
        fontsize=6.5, handletextpad=0.4, columnspacing=1.8)

    fig.tight_layout(rect=(0, 0.022, 1, 0.965), h_pad=1.4, w_pad=1.2)
    out = Path(__file__).resolve().parent / "fig4_cross_task.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    n = (len(RETRIEVAL) + len(RANKING) + len(LABELING) + len(OUTLIER_TOK)
         + len(CROSSDOC) + 4)
    print(f"wrote {out}  (9 panels, {n} task settings)")


if __name__ == "__main__":
    main()
