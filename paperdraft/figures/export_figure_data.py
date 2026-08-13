"""Export the raw data behind Figure 1 and the cross-task grid as tidy CSVs.

Both figures hardcode their numbers in the plotting scripts, so this imports those modules
rather than re-transcribing --- the CSVs cannot drift from what the paper plots.

Outputs (into figures/data/):
  fig1_gap_vs_context_data.csv   -- one row per (task, rung), full + chunked + relative gap
  fig4_cross_task_data.csv       -- one row per (panel, task, rung), full + chunked

Every row carries a context length in tokens and a `ctc_class` of low/high, as requested.

Usage:  python3 figures/export_figure_data.py
"""
import csv
from pathlib import Path

import fig1_gap_vs_context as f1
import fig4_cross_task as f4

OUT = Path(__file__).resolve().parent / "data"

# ── CTC classification ────────────────────────────────────────────────────────────────
# Column assignment comes from the cross-task grid (fig4_cross_task.py): column 1 is
# O_T(N), column 2 O_T(N M), column 3 O_T(N^2) and above. Figure 1 collapses columns 2-3
# into a single "hi" bucket, so the finer order is filled in from the grid where the task
# appears there. mathmatch is in neither grid column, so its order is left unspecified.
CTC_ORDER = {
    "niah": "O_T(N)", "nq": "O_T(N)", "msmarco": "O_T(N)", "rerank": "O_T(N)",
    "scifact": "O_T(N)", "obliq_twitter": "O_T(N)", "helmet_qa": "O_T(N)",
    "helmet_summ": "O_T(N)", "outlier_amzn": "O_T(N)", "absence": "O_T(N)",
    "outlier_wiki": "O_T(NM)", "grouping": "O_T(NM)",
    "qdmatch_hpqa": "O_T(N^2)+", "strmatch": "O_T(N^2)+", "xabsence": "O_T(N^2)+",
    "reorder": "O_T(N^2)+", "textgroups": "O_T(N^2)+", "contradiction": "O_T(N^2)+",
    "mathmatch": "unspecified",
}

# Per-task metric from Table 1 (table_suite.tex), which is finer than the figure panel
# labels -- a panel like "Cross-document matching" shares one y-axis across four different
# metrics. Where the panel label disagrees with Table 1 (it calls set-F1 "gold-ID F1" for
# NIAH/MS MARCO/OBLIQ), Table 1 wins here; see data/README.md.
METRIC = {
    "niah": "set-F1", "NIAH": "set-F1",
    "nq": "gold-ID F1", "NQ": "gold-ID F1",
    "msmarco": "set-F1", "MS MARCO": "set-F1",
    "rerank": "MRR@10",
    "scifact": "set-F1",
    "obliq_twitter": "set-F1", "OBLIQ": "set-F1",
    "helmet_qa": "token-F1 (NarrativeQA)",
    "helmet_summ": "ROUGE-1 (GovReport)", "GovReport": "ROUGE-1",
    "outlier_amzn": "set-F1", "outlier_wiki": "set-F1", "outlier": "set-F1",
    "outlier (wiki)": "set-F1",
    "absence": "set-F1",
    "grouping": "pairwise-F1",
    "qdmatch_hpqa": "pair-F1", "qdmatch": "pair-F1",
    "strmatch": "pair-F1",
    "xabsence": "set-F1",
    "reorder": "Kendall tau",
    "textgroups": "group-F1",
    "contradiction": "pair-F1 / EM", "Contradiction": "pair-F1 / EM",
    "HotpotQA": "gold-ID F1",
    "mathmatch": "unspecified",
}

# Per-doc token estimates from Table 5 (appendix_data_stats.tex), used by fig1 to put the
# per-N ladders on a token axis. Reused here to give the per-N panels of the cross-task
# grid an approximate context length too.
TOK_PER_DOC = {"contradiction": 45, "hotpotqa": 130, "grouping": 200,
               "outlier_wiki": 130, "reorder": 130}

# fig1 denominates its three per-N ladders in tokens; invert that back to N.
F1_PER_N_DOCS = {
    "grouping": {4000: 20, 10000: 50, 20000: 100},
    "HotpotQA": {2600: 20, 13000: 100, 26000: 200},
    "Contradiction": {2250: 50, 4500: 100, 22500: 500},
}

# Only what the two figure scripts actually state. Suite rungs are 500 per the fig4
# docstring, with two documented exceptions. The per-N ladders are NOT covered by that
# statement, so they are exported as "unknown" rather than assumed -- Table 6 in the
# appendix flags several of them (outlier-wiki 56-100, grouping 200 at N=100) and its
# outlier-wiki row already disagrees with Table 5. See data/README.md.
EVAL_SIZE_SUITE = {"obliq_twitter": 126, "OBLIQ": 126}
EVAL_SIZE_PER_N = {"contradiction": 488, "Contradiction": 488, "HotpotQA": 500}


def _eval_size(task, per_n=False):
    if per_n:
        return EVAL_SIZE_PER_N.get(task, "unknown (see README)")
    return EVAL_SIZE_SUITE.get(task, 500)


def _rel_gap(full, chunked):
    """1 - chunked/full, blank where it is undefined (missing arm or a zero full arm)."""
    if full in (None, 0) or chunked is None:
        return ""
    return round(1.0 - chunked / full, 6)


def export_fig1():
    rows = []
    for task, (pts, ctc) in f1.TASKS.items():
        per_n = task in F1_PER_N_DOCS
        for tokens, full, chunked in pts:
            plotted = full >= f1.FLOOR
            rows.append({
                "figure": "fig1",
                "series": "background",
                "task": task,
                "ctc_class": "low" if ctc == "N" else "high",
                "ctc_order": CTC_ORDER.get(task, ""),
                "metric": METRIC.get(task, ""),
                "context_tokens": tokens,
                "n_docs": F1_PER_N_DOCS[task][tokens] if per_n else "",
                "source": "per_N runs" if per_n else "CTC suite (token ladder)",
                "full_attention": full,
                "chunked_attention": chunked,
                "relative_gap": _rel_gap(full, chunked),
                "eval_size": _eval_size(task, per_n),
                "plotted_in_figure": "yes" if plotted else "no",
                "drop_reason": "" if plotted
                               else f"full arm {full} below FLOOR={f1.FLOOR}; ratio is noise",
            })
    for task, pts in (("HotpotQA", f1.HPQA), ("Contradiction", f1.CONTRA)):
        ctc = "low" if task == "HotpotQA" else "high"
        for tokens, full, chunked in pts:
            rows.append({
                "figure": "fig1",
                "series": "highlighted (star line)",
                "task": task,
                "ctc_class": ctc,
                "ctc_order": CTC_ORDER.get(task.lower(), "O_T(N)" if ctc == "low" else ""),
                "metric": METRIC.get(task, ""),
                "context_tokens": tokens,
                "n_docs": F1_PER_N_DOCS[task][tokens],
                "source": "per_N runs",
                "full_attention": full,
                "chunked_attention": chunked,
                "relative_gap": _rel_gap(full, chunked),
                "eval_size": _eval_size(task, per_n=True),
                "plotted_in_figure": "yes" if full >= f1.FLOOR else "no",
                "drop_reason": "",
            })
    _write("fig1_gap_vs_context_data.csv", rows)
    return rows


# panel -> (column CTC class, big-O of the column, x unit, metric)
F4_PANELS = {
    "Retrieval": ("low", "O_T(N)", "tokens", "gold-ID F1"),
    "Ranking / subjective": ("low", "O_T(N)", "tokens", "MRR@10 (rerank), gold-ID F1 (OBLIQ)"),
    "Absence, summarization": ("low", "O_T(N)", "tokens", "set-F1 (absence), ROUGE-1 (GovReport)"),
    "Outlier (wiki), N sweep": ("high", "O_T(NM)", "documents", "set-F1"),
    "Grouping (OpenAlex)": ("high", "O_T(NM)", "documents", "pairwise F1"),
    "Outlier (wiki), token sweep": ("high", "O_T(NM)", "tokens", "set-F1"),
    "Contradiction": ("high", "O_T(N^2)+", "documents", "EM"),
    "Reorder (Gutenberg)": ("high", "O_T(N^2)+", "documents", "Kendall tau"),
    "Cross-document matching": ("high", "O_T(N^2)+", "tokens", "task metric (see notes)"),
}


def _f4_row(panel, task, x, unit, full, chunked, variant="", tok_key=None, note=""):
    ctc, order, _, panel_ylabel = F4_PANELS[panel]
    if unit == "tokens":
        tokens, n_docs = x, ""
    else:
        n_docs, tokens = x, x * TOK_PER_DOC[tok_key]
    gap = _rel_gap(full, chunked)
    return {
        "figure": "fig4_cross_task",
        "panel": panel,
        "task": task,
        "variant": variant,
        "ctc_class": ctc,
        "ctc_order": order,
        "x_axis": unit,
        "context_tokens": tokens,
        "context_tokens_exact": "yes" if unit == "tokens" else "no (N x per-doc tokens)",
        "n_docs": n_docs,
        "metric": METRIC.get(task, ""),
        "panel_y_label": panel_ylabel,
        "full_attention": "" if full is None else full,
        "chunked_attention": "" if chunked is None else chunked,
        "relative_gap": gap,
        "eval_size": _eval_size(task, per_n=(unit != "tokens")),
        "notes": note,
    }


def export_fig4():
    rows = []
    suite_panels = [
        ("Retrieval", f4.RETRIEVAL),
        ("Ranking / subjective", f4.RANKING),
        ("Absence, summarization", f4.LABELING),
        ("Outlier (wiki), token sweep", f4.OUTLIER_TOK),
        ("Cross-document matching", f4.CROSSDOC),
    ]
    for panel, tasks in suite_panels:
        for name, xs, dense, chunked in tasks:
            for x, d, c in zip(xs, dense, chunked):
                rows.append(_f4_row(panel, name, x, "tokens", d, c))

    p = "Outlier (wiki), N sweep"
    for (dense, chunked), tag in ((f4.OUTLIER_SCALEK, "scale-k"),
                                  (f4.OUTLIER_FIXEDK, "fixed-k")):
        for n, d, c in zip(f4.OUTLIER_N, dense, chunked):
            rows.append(_f4_row(p, "outlier (wiki)", n, "documents", d, c,
                                variant=tag, tok_key="outlier_wiki"))

    for n, d, c in zip(f4.GROUPING_N, *f4.GROUPING):
        rows.append(_f4_row("Grouping (OpenAlex)", "grouping", n, "documents", d, c,
                            tok_key="grouping"))
    for n, d, c in zip(f4.CONTRA_N, *f4.CONTRA):
        rows.append(_f4_row("Contradiction", "contradiction", n, "documents", d, c,
                            tok_key="contradiction"))
    for n, d, c in zip(f4.REORDER_N, *f4.REORDER):
        rows.append(_f4_row("Reorder (Gutenberg)", "reorder", n, "documents", d, c,
                            tok_key="reorder"))
    n, y = f4.REORDER_NOMIX
    rows.append(_f4_row("Reorder (Gutenberg)", "reorder", n, "documents", None, y,
                        variant="chunked, NO mask mixing", tok_key="reorder",
                        note="single X point; chunked arm only, no matching full run"))
    _write("fig4_cross_task_data.csv", rows)
    return rows


def _write(name, rows):
    OUT.mkdir(exist_ok=True)
    path = OUT / name
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}  ({len(rows)} rows)")


if __name__ == "__main__":
    export_fig1()
    export_fig4()
