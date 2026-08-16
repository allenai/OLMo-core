"""CTC-SUITE-HIGH / CTC-SUITE-LOW: the per-task dense-vs-chunked ladder for every suite task.

Two appendix figures, one per CTC class, so that the class-level averages in the main paper
(CTC-GRID, CTC-PENALTY) can be checked against the individual tasks they average over:

    CTC-SUITE-HIGH  10 high-CTC tasks   (O_T(NM), O_T(N^2), O_T(N^3))
    CTC-SUITE-LOW   12 low-CTC tasks    (O_T(N))

Solid = full attention, dashed = document-chunked (with mask mixing). Qwen3.5-4B throughout.

    python3 paperdraft/figures/make_fig_suite_ladders.py

⚠ THE PANEL COUNTS ARE THE PAPER'S OWN 12/10 SPLIT and are read from the data, not hardcoded --
CTC-GRID's Average panel is labelled "Low CTC - 12 Tasks / High CTC - 10 Tasks". If a task is added
or reclassified, these figures pick it up and the layout re-flows; the assertion below fails loudly
if the grid ever stops fitting the panels.

⚠ EVERY AXIS IS SHARED WITHIN A FIGURE, ON PURPOSE. The per-task metrics differ (gold-ID F1,
set-F1, pair-F1, Kendall tau, MRR@10, partial credit), so the panels are NOT commensurable as
absolute numbers -- what is comparable is the SHAPE of each pair of lines and the gap between them.
A shared 0-1 axis keeps those gaps readable against each other; per-panel autoscaling would make a
0.02 gap and a 0.6 gap look identical, which is the one thing these figures exist to distinguish.

⚠ SOME LADDERS STOP AT 16k. outlier fix-k, absence and re-order have no 32k grade JSON -- the cell
was never evaluated. Those points are absent rather than zero, and the script prints them so the
absence stays visible in the caption instead of reading as a collapse.
"""
import csv
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
# The active draft is paperdraft2/; figure CODE stays here with the rest of the figure scripts and
# the label index, so outputs are written to both trees rather than the script being forked.
ALSO = os.path.join(REPO, "paperdraft2", "figures")

LADDER = [2048, 4096, 8192, 16384, 32768]
LADDER_LABELS = ["2k", "4k", "8k", "16k", "32k"]
BLUE, ORANGE, TXT = "#3f7dc4", "#e2622f", "#111111"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Liberation Serif", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 13, "axes.labelsize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.edgecolor": "#c9c9c9", "axes.labelcolor": TXT,
    "xtick.color": TXT, "ytick.color": TXT,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def load():
    """Group the exported grid CSV into {ctc_class: [(task_label, metric, {rung: (full, chunk)})]}."""
    tasks, order = {}, []
    with open(os.path.join(HERE, "data", "ctc_suite_grid_data.csv")) as fh:
        for r in csv.DictReader(fh):
            if r["plotted_in_figure"] != "yes":
                continue
            key = r["task"]
            if key not in tasks:
                tasks[key] = {"label": r["task_label"], "metric": r["metric"],
                              "cls": r["ctc_class"], "order": r["ctc_order"], "pts": {}}
                order.append(key)
            f = r["full_attention"]
            c = r["chunked_attention"]
            tasks[key]["pts"][int(float(r["context_tokens"]))] = (
                float(f) if f else None, float(c) if c else None)
    return [tasks[k] for k in order]


def draw(entries, ncols, nrows, color, title, out, label):
    assert len(entries) <= ncols * nrows, \
        f"{label}: {len(entries)} tasks will not fit a {nrows}x{ncols} grid -- widen the layout"
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.05 * ncols, 2.5 * nrows),
                             sharex=True, sharey=True)
    axes = axes.ravel()
    missing = []
    for ax, e in zip(axes, entries):
        xs = [i for i, r in enumerate(LADDER) if r in e["pts"]]
        full = [e["pts"][LADDER[i]][0] for i in xs]
        chunk = [e["pts"][LADDER[i]][1] for i in xs]
        ax.plot(xs, full, "-", marker="o", color=color, linewidth=2.0, markersize=4.4,
                markeredgewidth=0, zorder=3)
        ax.plot(xs, chunk, "--", marker="s", color=color, linewidth=2.0, markersize=4.4,
                markeredgewidth=0, alpha=0.95, zorder=3)
        if len(xs) < len(LADDER):
            missing.append(f"{e['label']} (no {'/'.join(LADDER_LABELS[i] for i in range(len(LADDER)) if i not in xs)})")
        ax.set_title(e["label"], fontsize=12.5, color=TXT, pad=6)
        ax.grid(axis="y", color="#ececec", linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[len(entries):]:
        ax.axis("off")

    axes[0].set_xlim(-0.2, len(LADDER) - 0.8)
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_xticks(range(len(LADDER)))
    axes[0].set_xticklabels(LADDER_LABELS)
    # Axis labels go on the outer panels. ⚠ When the grid is ragged (10 tasks in 12 slots) the
    # bottom of a column is NOT always the bottom ROW, and sharex hides tick labels everywhere but
    # the last row -- which leaves "Context Length" sitting under a panel with no tick numbers.
    # A panel is the bottom of its column iff nothing occupies the slot ncols further on, so turn
    # its tick labels back on explicitly.
    for i, ax in enumerate(axes[:len(entries)]):
        if i % ncols == 0:
            ax.set_ylabel("Score")
        if i + ncols >= len(entries):
            ax.set_xlabel("Context Length")
            ax.tick_params(labelbottom=True)

    handles = [Line2D([], [], color=TXT, linestyle="-", marker="o", markersize=4.6,
                      linewidth=2.0, label="Full attention"),
               Line2D([], [], color=TXT, linestyle="--", marker="s", markersize=4.6,
                      linewidth=2.0, label="Chunked attention")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2,
               frameon=False, fontsize=12.5, labelcolor=TXT, handlelength=2.6, columnspacing=2.0)
    fig.suptitle(title, fontsize=15, color=color, y=1.035)
    fig.tight_layout(rect=(0, 0, 1, 0.965))

    meta = {"Title": f"{label} — per-task dense vs chunked ladders", "Creator": "make_fig_suite_ladders.py"}
    for d in (HERE, ALSO):
        if not os.path.isdir(d):
            continue
        for ext, kw in (("png", {"dpi": 200}), ("pdf", {})):
            fig.savefig(os.path.join(d, f"{out}.{ext}"), bbox_inches="tight",
                        facecolor="white", metadata=meta, **kw)
    plt.close(fig)
    return missing


def main():
    entries = load()
    high = [e for e in entries if e["cls"] == "high"]
    low = [e for e in entries if e["cls"] == "low"]

    # BOTH figures use a 4-wide grid on purpose. 10 panels fit 5x2, but at \textwidth that leaves
    # each panel ~1.1in across and the tick labels stop being readable in print; 4x3 (two slots
    # blank) gives ~1.6in. It also makes the two figures the same shape and panel size, so the
    # reader can compare high against low directly instead of across two different scales.
    m_hi = draw(high, 4, 3, ORANGE, r"High CTC  ($O_T(NM)$, $O_T(N^2)$, $O_T(N^3)$)",
                "ctc_suite_high", "CTC-SUITE-HIGH")
    m_lo = draw(low, 4, 3, BLUE, r"Low CTC  ($O_T(N)$)",
                "ctc_suite_low", "CTC-SUITE-LOW")

    print(f"[CTC-SUITE-HIGH] {len(high)} tasks -> ctc_suite_high.{{pdf,png}} in {HERE} and {ALSO}")
    print(f"[CTC-SUITE-LOW]  {len(low)} tasks -> ctc_suite_low.{{pdf,png}}  in {HERE} and {ALSO}")
    print()
    for name, group in (("HIGH", high), ("LOW", low)):
        print(f"--- {name} CTC panels (task : CTC : metric) ---")
        for e in group:
            print(f"  {e['label']:26s} {e['order']:10s} {e['metric']}")
        print()
    for name, miss in (("HIGH", m_hi), ("LOW", m_lo)):
        if miss:
            print(f"⚠ {name}: ladders that stop early (cell never evaluated, NOT a measured zero): "
                  + "; ".join(miss))


if __name__ == "__main__":
    main()
