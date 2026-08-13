"""Figure 1 (fig:motivatingtask) — full-vs-efficient attention gap across context length.

y = RELATIVE performance gap, 1 - chunked/full: the fraction of full attention's score that
document-chunked attention fails to keep. One line per task, one marker per (task, rung).

Why relative and not the raw difference: the raw gap is bounded above by the full-attention
score, so on any task where full attention itself decays the difference must shrink even as
chunked keeps less and less of it. That bound hid the trend on four high-complexity tasks --
reorder (full 0.747 -> 0.047, raw gap peaks at 4k then falls) and textgroups (full 0.196 ->
0.047) are capped from above, while strmatch (chunked 0.003 at the FIRST rung) and xabsence
are saturated from below. On this axis reorder reads 0.37 -> 0.84 and textgroups 0.56.

The cost is a floor rule: a ratio between two near-zero numbers is noise, so a rung is dropped
when the FULL arm is under FLOOR. That removes helmet_qa (full 0.033-0.039) and mathmatch
(0.003) from the figure entirely -- both are tasks that never trained, not tasks with no gap --
and leaves textgroups a single point at 2k. Watch grouping's last point: its full arm is 0.196,
only just above the floor.

TWO DATA SOURCES, deliberately:

1. Most tasks — the CTC-suite grid, Qwen3.5-4B full fine-tune, full attention vs
   document-chunked + mask mixing (the `-cmix` arm = best-case chunked). Assembled from
   the S3 grade JSONs 2026-07-22, updated 2026-07-27; same numbers as the artifact
   "CTC-suite grid - Qwen3.5-4B dense vs chunked". Rungs are 2k/4k/8k/16k/32k tokens.

2. contradiction, HotpotQA and grouping — the earlier per-N corpus-reasoning runs, whose
   suite ladders are unusable or suspect (see below). x for these is docs x per-doc tokens
   from Table 5 (contradiction ~45 tok/doc, HotpotQA ~130, grouping ~200), which is how
   those runs were denominated in the previous version of this figure.
     * contradiction — the suite ladder is suspect (Prasann, 2026-08-03) and its gap in
       fact SHRINKS there (0.376 -> 0.302) because full attention itself decays
       0.849 -> 0.335. Per-N: Qwen3.5-4B, std vs mask-mix p50, 3ep, eval_size 488.
       n=20 (~0.9k tokens) is dropped so the axis can start at 2k.
     * HotpotQA — the suite has no dense checkpoint at all (S3 distcp incomplete, 80/128
       shards). Per-N: Qwen3.5-0.8B LoRA, std vs chunked, cot 4ep, eval_size 500.
     * grouping — per-N: Qwen3.5-4B, pairwise-F1, std vs the best chunked arm available at
       each rung (n20 pure 0.658 > mix-p10 0.640; n50 mix-p50 0.462 > pure 0.389; n100 only
       pure 0.031), matching the cloud's best-case-chunked convention.
   ⚠ These are NOT the same model as the cloud, and HotpotQA is not the same model as the
   other two (0.8B LoRA vs 4B full-FT).

Excluded, with reasons: oolong + hotpotqa-suite (no dense checkpoint), groups4 (chunked
0.000 is a constant-output eval artifact, 489/500 identical generations), obliq_retrieval
(ladder voided 2026-07-26 — a random-k guess beat both arms at every rung).

CTC class drives colour. Two classifications were corrected by Prasann 2026-08-03:
  * absence is O(N) — checking whether item i survived into the second copy is one operation
    per item. xabsence is the O(N^2) one: every A claim must be compared against every B claim.
  * outlier (Amazon) is O(N) — the majority attribute is a star rating or a product category
    from a small fixed set, so it does not carry the open-ended category discovery that makes
    outlier (wiki) O(N.M).

Removed on request 2026-08-03: cycle and grouping_labeled. Note for grouping_labeled, in case
it comes back: its suite arms are near-identical at every rung (0.439/0.439, 0.365/0.370,
0.231/0.226, 0.054/0.051) and plain grouping's suite row behaves the same way, which is not
what two independently trained arms look like — worth checking whether the chunked mask was
actually applied there before trusting either. The old-run grouping numbers used here do not
have that property.

Usage:  python3 figures/fig1_gap_vs_context.py            (writes fig1_gap_vs_context.pdf)
        python3 figures/fig1_gap_vs_context.py --labels   (also writes a TEMPORARY
            fig1_gap_vs_context_labeled.{pdf,png} with every task line named, for
            eyeballing which tasks agree with the CTC expectation. Inspection aid only --
            not the paper figure.)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator


def place_labels(ax, fig, items, min_gap_pt=10.5):
    """Annotate (x, y, text, color) at the end of each line, pushed apart vertically.

    Lines terminate at only a handful of distinct x values, so labels stack. We convert
    each anchor to display pixels, spread the ones sharing an x until they are at least
    `min_gap_pt` apart, and re-express the correction as an offset in points."""
    fig.canvas.draw()
    px_per_pt = fig.dpi / 72.0
    # Cluster anchors whose x are close in display space -- labels sit to the RIGHT of the
    # line end, so two lines ending at 26k and 32k still collide and must be spread together.
    xs_sorted = sorted({x for x, _, _, _ in items},
                       key=lambda x: ax.transData.transform((x, 0))[0])
    cluster_of, cluster_x, prev_px = {}, [], None
    for x in xs_sorted:
        px = ax.transData.transform((x, 0))[0]
        if prev_px is None or px - prev_px > 70:
            cluster_x.append(x)
        cluster_of[x] = cluster_x[-1]
        prev_px = px
    by_x = {}
    for x, y, text, color in items:
        by_x.setdefault(cluster_of[x], []).append((y, text, color, x))
    for _, group in by_x.items():
        group.sort()
        ys_px = [ax.transData.transform((gx, y))[1] for y, _, _, gx in group]
        for i in range(1, len(ys_px)):
            need = ys_px[i - 1] + min_gap_pt * px_per_pt
            if ys_px[i] < need:
                ys_px[i] = need
        for (y, text, color, gx), y_px in zip(group, ys_px):
            base_px = ax.transData.transform((gx, y))[1]
            ax.annotate(text, (gx, y),
                        xytext=(5, (y_px - base_px) / px_per_pt),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=7, color=color, zorder=7,
                        annotation_clip=False)

K = 1024
SUITE = [2 * K, 4 * K, 8 * K, 16 * K, 32 * K]
FLOOR = 0.15   # full-attention score below this -> the ratio is noise, drop the rung


def curve(pts):
    """(xs, ys) of relative gap for the rungs whose full arm clears FLOOR."""
    keep = [(x, f, c) for x, f, c in pts if f >= FLOOR]
    return [x for x, _, _ in keep], [1.0 - c / f for _, f, c in keep]


def suite(full, chunked):
    """CTC-suite rows: pair the five standard rungs with the arms that have evals."""
    return [(x, f, c) for x, f, c in zip(SUITE, full, chunked)
            if f is not None and c is not None]


# task -> ([(tokens, full, chunked), ...], ctc_class)
TASKS = {
    # ── O_T(N) ──
    "niah":             (suite([.988, .988, .976, .984, .966], [.984, .994, .976, .978, .970]), "N"),
    "nq":               (suite([.904, .930, .864, None, None], [.964, .942, .896, None, None]), "N"),
    "msmarco":          (suite([.961, .939, .917, .900, None], [.897, .902, .868, .871, None]), "N"),
    "rerank":           (suite([.989, .971, .960, .952, None], [.980, .966, .950, .949, None]), "N"),
    "scifact":          (suite([.963, .948, .926, .899, .879], [.976, .956, .933, .886, .840]), "N"),
    "obliq_twitter":    (suite([.740, .659, .608, .277, .327], [.689, .605, .563, .262, .293]), "N"),
    "outlier_amzn":     (suite([.921, .891, .896, .870, .864], [.912, .894, .887, .875, .858]), "N"),
    "absence":          (suite([.964, .976, .986, .932, None], [.949, .961, .981, .945, None]), "N"),
    # ── O_T(NM) and above ──
    "outlier_wiki":     (suite([.982, .956, .877, .679, .428], [.962, .869, .641, .343, .125]), "hi"),
    "qdmatch_hpqa":     (suite([.999, .997, .998, .992, .981], [.650, .760, .668, .547, .333]), "hi"),
    "strmatch":         (suite([.999, .997, .997, .995, .994], [.003, .003, .000, .001, .000]), "hi"),
    "xabsence":         (suite([.609, .545, .528, .505, .518], [.135, .057, .037, .012, .008]), "hi"),
    "reorder":          (suite([.747, .600, .262, .047, None], [.471, .214, .043, .000, None]), "hi"),
    "mathmatch":        (suite([.003, .001, .000, None, None], [.001, .000, .000, None, None]), "hi"),
    "textgroups":       (suite([.196, .090, .054, .051, .047], [.087, .021, .019, .007, .001]), "hi"),
    # per-N source (see docstring)
    "grouping":         ([(4000, .7112, .6576), (10000, .5170, .4623), (20000, .1963, .0310)], "hi"),
}

# Highlighted per-N ladders, drawn as stars.
HPQA = [(2600, .852, .800), (13000, .763, .705), (26000, .769, .728)]
CONTRA = [(2250, .982, .948), (4500, .982, .945), (22500, .962, .825)]

# Validated with the dataviz palette validator (light surface, categorical):
# CVD dE 10.9 (protan), normal-vision dE 31.5, both >= 3:1 on the surface. The palette's
# stock green (#008300) FAILS against this orange at dE 3.2 — re-run the validator before
# substituting any hue here.
ORANGE, GREEN = "#eb6834", "#0f6b45"
INK, MUTED, GRID = "#161a20", "#5c6673", "#dfe3e9"


def main(labeled=False):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": MUTED,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.size": 9,
    })
    fig, ax = plt.subplots(figsize=(8.4, 5.4) if labeled else (5.8, 4.3))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.axhline(0, color=MUTED, lw=0.8)

    n_dots = 0
    anchors = []
    for name, (pts, ctc) in TASKS.items():
        color = ORANGE if ctc == "N" else GREEN
        xs, ys = curve(pts)
        if not xs:
            continue
        n_dots += len(xs)
        ax.plot(xs, ys, color=color, lw=0.9, alpha=0.55 if labeled else 0.30, zorder=2)
        ax.scatter(xs, ys, s=22, color=color, alpha=0.65 if labeled else 0.42,
                   linewidths=0, zorder=3)
        if labeled:
            anchors.append((xs[-1], ys[-1], name, color))

    for pts, color, name, dy in ((HPQA, ORANGE, "HotpotQA", -15),
                                 (CONTRA, GREEN, "Contradiction", 11)):
        xs, ys = curve(pts)
        ax.plot(xs, ys, color=color, lw=1.7, zorder=4)
        ax.scatter(xs, ys, marker="*", s=210, color=color,
                   edgecolors="white", linewidths=0.7, zorder=5)
        if labeled:
            anchors.append((xs[-1], ys[-1], name + " *", color))
        else:
            ax.annotate(name, (xs[-1], ys[-1]), xytext=(2, dy),
                        textcoords="offset points", ha="right", va="center",
                        fontsize=9, fontweight="bold", color=INK, zorder=6)

    ax.set_xscale("log")
    ax.set_xticks(SUITE)
    ax.set_xticklabels(["2k", "4k", "8k", "16k", "32k"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(1750, 150000 if labeled else 42000)

    # Plain linear y (Prasann 2026-08-03): on the relative axis the tasks spread across the
    # full 0-1 range on their own, so no rescaling is needed to separate the two groups.
    ax.set_ylim(-0.14, 1.06)
    ax.yaxis.set_major_locator(FixedLocator([0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("relative gap: $1-$ efficient $/$ full")

    handles = [
        plt.Line2D([], [], marker="o", ms=6, lw=1.4, color=ORANGE, alpha=.6,
                   label="low complexity, $O_T(N)$"),
        plt.Line2D([], [], marker="o", ms=6, lw=1.4, color=GREEN, alpha=.6,
                   label="high complexity, $O_T(NM)$–$O_T(N^3)$"),
    ]
    # Above the panel: on a linear axis the low-complexity band fills the bottom strip and
    # the high-complexity lines fill the rest, so there is no in-panel gap left for a legend.
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              frameon=False, fontsize=8.5, handletextpad=0.5, columnspacing=2.0,
              borderaxespad=0.0)

    ax.text(0.015, -0.055, "efficient attention matches full",
            transform=ax.get_yaxis_transform(), fontsize=7.5, color=MUTED,
            ha="left", va="center", style="italic")

    if labeled:
        place_labels(ax, fig, anchors)

    fig.tight_layout()
    stem = "fig1_gap_vs_context_labeled" if labeled else "fig1_gap_vs_context"
    out = Path(__file__).resolve().parent / f"{stem}.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {out}  ({n_dots} points over {len(TASKS)} tasks "
          f"+ {len(HPQA) + len(CONTRA)} highlighted)")


if __name__ == "__main__":
    main()
    if "--labels" in sys.argv:
        main(labeled=True)
