"""Merge the harvested per-rung grades into the FINAL 21-task suite table.

Emits `suite_table.json`: one entry per (suite row, arm, rung) with the metric, eval_size and a
binomial error bar, plus a per-row coverage/blocker record. Rows whose only surviving numbers live
in the 2026-07-27 markdown table (no grade JSON reachable on any node) are carried in explicitly and
tagged so they are never mistaken for a fresh grade.
"""
import json, math, collections

RUNGS = [2048, 4096, 8192, 16384, 32768]

# suite row -> (task dir in the grade trees, metric label, complexity class)
ROWS = [
    ("fiqa",            "fiqa",            "gold_id_f1",     "N"),
    ("nq",              "nq",              "gold_id_f1",     "N"),
    ("hpqa",            "hotpotqa",        "gold_id_f1",     "N"),
    ("qdmatch_fiqa",    "qdmatch_fiqa",    "pair_f1",        "N2"),
    ("qdmatch_nq",      "qdmatch_nq",      "pair_f1",        "N2"),
    ("qdmatch_hpqa",    "qdmatch_hpqa",    "pair_f1",        "N2"),
    # O(N), not O(NM): the majority attribute is a star rating or a product category from a small
    # FIXED set, so there is no open-ended category discovery -- the same thing that makes Oolong
    # O(N). Only outlier (wiki) scale-k has a category set that grows with N. Confirmed by prasann
    # 2026-08-13; it had been NM here, which propagated a wrong class into the artifact and from
    # there into the paper's task table.
    ("outlier_amazon",  "outlier_amzn",    "set_f1",         "N"),
    ("outlier_scalek",  "outlier",         "set_f1",         "NM"),
    ("outlier_fixk",    "outlier_fixedM",  "set_f1",         "N"),
    ("oolong",          "oolong",          "partial_credit", "N"),
    ("grouping",        "grouping",        "pairwise_f1",    "NM"),
    # O(N), not O(N^2): checking whether item i survived into the second copy is one operation per
    # item, not an all-pairs search. Confirmed by prasann 2026-08-13; fixed here at the source so
    # the artifact, both paper tables and the exported CSVs stop disagreeing. `xabsence` -- two
    # corpora of paraphrase twins, where finding an orphan does need all pairs -- stays N2.
    ("absence",         "absence_gutenberg","set_f1",        "N"),
    ("xabsence",        "xabsence",        "set_f1",         "N2"),
    ("rerank",          "rerank",          "mrr@10",         "N"),
    ("msmarco",         "msmarco",         "gold_id_f1",     "N"),
    ("reorder",         "reorder",         "kendall_tau",    "N2"),
    ("obliq",           "obliq_twitter",   "gold_id_f1",     "N"),
    ("niah_contra",     "niah",            "gold_id_f1",     "N"),
    ("contra_real",     "contradiction_iid","set_f1",        "N2"),
    ("strmatch",        "strmatch",        "set_f1",         "N2"),
    ("textgroups",      "textgroups",      "textgroups_f1",  "NM"),
    ("scifact",         "scifact",         "gold_id_f1",     "N"),
]

# Cells with no grade JSON reachable on any node, carried over from
# results/ctc_suite/dense_vs_chunked_table.md (2026-07-27) so the row is not silently empty.
CARRIED = {
    "xabsence": {
        "dense":   {2048: 0.609, 4096: 0.545, 8192: 0.528, 16384: 0.505, 32768: 0.518},
        "chunked": {2048: 0.135, 4096: 0.057, 8192: 0.037, 16384: 0.012, 32768: 0.008},
        "eval_size": 520,
    },
    "obliq": {
        "dense":   {2048: 0.740, 4096: 0.659, 8192: 0.608, 16384: 0.277, 32768: 0.327},
        "chunked": {2048: 0.689, 4096: 0.605, 8192: 0.563, 16384: 0.262, 32768: 0.293},
        "eval_size": 126,
    },
}

# A ladder whose rung file is named for a different nominal length than the suite column it belongs
# in. The realistic contradiction ladder's bottom rung is 2560, not 2048: n=44 (the 2048 rung of
# `contradiction_clean`) sits below the training minimum of n=52, so the IID rebuild starts at n=56.
# Every other rung is n-identical to `contradiction_clean`.
RUNG_ALIAS = {"contradiction_iid": {2048: 2560}}

# Rows whose on-disk numbers are superseded by a rebuild that is still in flight. The cells are kept
# so the row is not silently blank, but they are tagged and must never be quoted as current.
#
# oolong CAME OFF this list on 2026-08-13: the retrain it was waiting for landed, both arms finished
# their 2k-32k ladders, and the row is now pinned by run name to `ctcms-oolong-{full,cmix}-4b-vsl`
# (see the OVERRIDES block in harvest_grades.py). Its numbers are current.
#
# xabsence WENT ON it the same day, and for a different reason: its cells are not superseded by a
# rebuild of the *eval*, they are superseded by the discovery that the task was not being learned at
# all. Four learnability probes moved training CE off the 0.78 flatline to 0.23-0.33, so whatever the
# 2026-07-27 numbers measured, it was not a model that had learned xabsence. They stay visible and
# struck through until the probe evals land.
VOID = {"xabsence"}

# Individual cells to drop rather than render. This is NOT for cells that are merely low.
#
# outlier fix-k @ 32k: that rung is n=220, the top of the training range, and the fill-in run
# (n=140/170/190/300/440) shows the row is a smooth monotone decline in n, not a ladder with a cliff
# at the end. Rendered as the last column of a 2k-32k ladder it reads as "collapses at 32k", which
# is the one thing it does not do. The curve lives in the "Beyond the grid" section, where the
# x-axis is n and the shape is legible. It is also the cell the stop-token fix moves most
# (0.3191 -> 0.3913), so the on-disk number is stale in a second, independent way.
DROP_CELLS = {("outlier_fixk", 32768)}


def se(p, n):
    """Binomial standard error -- an UPPER bound for a [0,1]-valued per-example metric."""
    if p is None or not n:
        return None
    return math.sqrt(max(p * (1 - p), 0.0) / n)

grades = json.load(open("debug/ctc_final_suite/harvested_grades.json"))
by = collections.defaultdict(dict)
for g in grades:
    by[(g["task"], g["arm"])][g["rung"]] = g

out = []
for row, task_dir, metric, klass in ROWS:
    entry = {"row": row, "task_dir": task_dir, "metric": metric, "class": klass, "cells": {}}
    for arm in ("dense", "chunked"):
        cells = {}
        for rung in RUNGS:
            if (row, rung) in DROP_CELLS:
                continue
            src_rung = RUNG_ALIAS.get(task_dir, {}).get(rung, rung)
            g = by.get((task_dir, arm), {}).get(src_rung)
            if g:
                cells[rung] = {
                    "value": round(g["metric_value"], 4),
                    "eval_size": g["eval_size"],
                    "se": round(se(g["metric_value"], g["eval_size"]), 4),
                    "source": "grade-json",
                    "graded": g["mtime"],
                    "ladder": (g["eval_data"] or "").split("/")[-3] if g["eval_data"] else None,
                }
            elif row in CARRIED and rung in CARRIED[row][arm]:
                v, n = CARRIED[row][arm][rung], CARRIED[row]["eval_size"]
                cells[rung] = {"value": v, "eval_size": n, "se": round(se(v, n), 4),
                               "source": "table-2026-07-27", "graded": None, "ladder": None}
        if row in VOID:
            for c in cells.values():
                c["source"] = "superseded"
        entry["cells"][arm] = cells
    live = row not in VOID
    entry["n_dense"] = len(entry["cells"]["dense"]) if live else 0
    entry["n_chunked"] = len(entry["cells"]["chunked"]) if live else 0
    entry["n_void"] = 0 if live else len(entry["cells"]["dense"]) + len(entry["cells"]["chunked"])
    out.append(entry)

json.dump(out, open("debug/ctc_final_suite/suite_table.json", "w"), indent=1)

print(f"{'row':17s} {'dense':>28s}   {'chunked':>28s}")
for e in out:
    def fmt(arm):
        return " ".join(f"{e['cells'][arm][r]['value']:.3f}" if r in e["cells"][arm] else "  --  "
                        for r in RUNGS)
    print(f"{e['row']:17s} {fmt('dense')}   {fmt('chunked')}")
per_arm = len(ROWS) * len(RUNGS)
print(f"\ncells: dense={sum(e['n_dense'] for e in out)}/{per_arm}  chunked={sum(e['n_chunked'] for e in out)}/{per_arm}"
      f"  superseded={sum(e['n_void'] for e in out)}")
