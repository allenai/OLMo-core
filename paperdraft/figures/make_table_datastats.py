"""TABLE-DATASTATS: the per-rung corpus statistics for every task in the CTC suite.

This REPLACES the old hand-written `tab:data-stats`, which described a per-$N$ dataset regime
(NQ k=20..200, contradiction N=20..500, eval sizes 400/488/200) that no experiment in the paper
uses any more. Every result in the paper comes from a five-rung TOKEN ladder (2k/4k/8k/16k/32k),
so the table has to be indexed the same way the experiments are.

    python3 paperdraft/figures/make_table_datastats.py

⚠ THE DOCUMENT COUNTS ARE MEASURED, NOT DECLARED. `data/ctc_suite_rung_stats.json` is the median
over a sample of real examples from the exact rung file each result was graded on (resolved through
the `ladder` + `task_dir` columns of `data/ctc_suite_grid_data.csv`, because several tasks are
served from a non-default ladder tree -- nq from `eval_rungs_ext32k`, msmarco from
`eval_rungs_tokenaccurate`, outlier fix-$M$ from `eval_rungs_fixedM`). Do not "tidy" these to round
numbers: N=296 for HotpotQA at 32k is what the model actually saw.

⚠ CORPUS SIZE IS REPORTED IN CHARACTERS, NOT TOKENS. Rung files are named for a build-time token
TARGET that is individually wrong in both directions (niah/rung_32768 measures p50 17198; absence
16k measures 48801), and measuring true prefill tokens needs the tokenizer plus each task's
`build_eval_prefill` -- that is what `debug/ctc_modelscale/measure_all_rungs.py` is for. Characters
are substrate-independent and are what we can state without a GPU, so they are what we state.

⚠ OOLONG AND THE HELMET PORTS ARE SINGLE-DOCUMENT (N=1) BY CONSTRUCTION. That is the point of
including them -- they anchor "long context" against "many documents" -- not a measurement failure.
"""
import csv
import json
import os

FIGURE_LABEL = "TABLE-DATASTATS"

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
ALSO = os.path.join(REPO, "paperdraft2", "figures")

LADDER = ["2048", "4096", "8192", "16384", "32768"]

#: task key -> (display name, source corpus). Sources are taken verbatim from the task write-ups in
#: `appendix_amazon_reviews.tex` (\label{app:data-details}); if a description there changes, change
#: it here too so the table and the prose cannot drift.
TASKS = [
    ("nq",             "NQ",                      "Wikipedia 100w (DPR)"),
    ("hpqa",           "HotpotQA (bridge)",       "Wikipedia 100w (DPR)"),
    ("niah_contra",    "NIAH-contra",             "PubMed claims"),
    ("scifact",        "BEIR SciFact",            "SciFact abstracts"),
    ("fiqa",           "BEIR FiQA",               "FiQA-2018 posts"),
    ("msmarco",        "MS MARCO",                "msmarco-v1-passage"),
    ("rerank",         "MS MARCO rerank",         "msmarco-v1-passage"),
    ("obliq",          "OBLIQ",                   "OBLIQ-Bench posts"),
    ("oolong",         "Oolong",                  "labelled item stream"),
    ("outlier_amazon", "Outlier (Amazon)",        "Amazon Reviews 2023"),
    ("outlier_fixk",   "Outlier (wiki, fix-$M$)", "Wikipedia 100w"),
    ("absence",        "Absence",                 "Project Gutenberg"),
    ("outlier_scalek", "Outlier (wiki, scale-$k$)", "Wikipedia 100w"),
    ("grouping",       "Grouping",                "OpenAlex abstracts"),
    ("textgroups",     "Textgroups",              "synthetic passages"),
    ("contra_real",    "Contradiction",           "PubMed claims"),
    ("xabsence",       "Cross-corpus absence",    "PubMed claims"),
    ("strmatch",       "String matching",         "synthetic strings"),
    ("qdmatch_nq",     "QDmatch (NQ)",            "Wikipedia 100w (DPR)"),
    ("qdmatch_hpqa",   "QDmatch (HotpotQA)",      "Wikipedia 100w (DPR)"),
    ("qdmatch_fiqa",   "QDmatch (FiQA)",          "FiQA-2018 posts"),
    ("reorder",        "Reordering",              "Project Gutenberg"),
]

CTC_TEX = {"O_T(N)": r"\ctc{N}", "O_T(NM)": r"\ctc{NM}", "O_T(N^2)+": r"\ctc{N^2}"}

#: Rows whose ladder does NOT grow monotonically with the rung label, marked in the table so the
#: number is not read as a clean length sweep. Both were found by measuring; neither is visible
#: from the file names, which is exactly why they survived this long.
CAVEAT = {
    # rung_8192 is 71.8 MB and rung_32768 is 66.5 MB: the "8k" corpus is the LARGER of the two, and
    # the 16k rung is smaller than both. The score curve tracks the corpus, not the label
    # (0.608 at 8k -> 0.277 at 16k -> 0.327 at 32k), so OBLIQ's ladder is not a length sweep.
    "obliq": r"$\dagger$",
    # 20/40/70/100 documents at 2k/4k/8k/16k -- the pool saturates, so the "16k" rung holds ~34k
    # characters (roughly an 8k-token prompt). MS MARCO itself was rebuilt on the token-accurate
    # ladder (23/46/93/187/374) but rerank was not, and no `eval_data` path was recorded for the
    # graded rerank runs, so its 32k rung's provenance is unconfirmed.
    "rerank": r"$\ddagger$",
}


def load():
    stats = json.load(open(os.path.join(HERE, "data", "ctc_suite_rung_stats.json")))
    meta, evalsz = {}, {}
    for r in csv.DictReader(open(os.path.join(HERE, "data", "ctc_suite_grid_data.csv"))):
        meta.setdefault(r["task"], (r["ctc_class"], r["ctc_order"], r["metric"]))
        # eval_size is per (task, rung, arm); the suite is 500 nearly everywhere, so record the set
        # and let the table print the exception rather than a single number that is sometimes wrong.
        v = r["eval_size"]
        if v and v.isdigit():
            evalsz.setdefault(r["task"], set()).add(int(v))
    return stats, meta, evalsz


def main():
    stats, meta, evalsz = load()
    lines, notes = [], []
    prev_cls = None
    for key, name, source in TASKS:
        cls, order, metric = meta[key]
        if prev_cls is not None and cls != prev_cls:
            lines.append(r"\midrule")
        prev_cls = cls
        cells = []
        for rung in LADDER:
            s = stats.get(key, {}).get(rung)
            cells.append("--" if not s else f"{s['n_docs']:,}")
        sizes = sorted(evalsz.get(key, {500}))
        ev = str(sizes[0]) if len(sizes) == 1 else "/".join(str(x) for x in sizes)
        if sizes and sizes[0] < 500:
            notes.append(f"{name} eval_size={sizes[0]}")
        chars = stats.get(key, {}).get("32768") or stats.get(key, {}).get("16384") or {}
        cpd = chars.get("chars_per_doc")
        lines.append(f"{name}{CAVEAT.get(key, '')} & {source} & {CTC_TEX[order]} & {metric} & "
                     + " & ".join(cells) + f" & {cpd if cpd else '--'} & {ev} \\\\")

    body = "\n".join(lines)
    tex = r"""% GENERATED by paperdraft/figures/make_table_datastats.py -- do not hand-edit.
% Document counts are MEASURED from the rung files each result was graded on; see the script header.
\begin{table}[htbp]
\centering
\scriptsize
\setlength{\tabcolsep}{3.2pt}
\begin{tabular}{l l c l r r r r r r r}
\toprule
& & & & \multicolumn{5}{c}{\textbf{Documents per example ($N$)}} & & \\
\cmidrule(lr){5-9}
\textbf{Task} & \textbf{Source corpus} & \textbf{\cname} & \textbf{Metric}
 & 2k & 4k & 8k & 16k & 32k & \textbf{Chars/doc} & \textbf{Eval} \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\caption{\textbf{Corpus statistics for every task in the suite, at every rung of the length
ladder.} Column headings 2k--32k are the ladder's token budgets; the entries are the median number
of documents actually present in an example of that rung, measured from the evaluation files the
reported results were graded on. Chars/doc is the median document length in characters at the
deepest available rung. Every task trains on 20{,}000 examples spanning the same range of corpus
sizes, and evaluates on the number of examples in the last column. Reordering and Absence stop at
16k by ladder policy; Oolong is single-document by construction, which is what makes it a control
for ``long context'' against ``many documents''. $\dagger$~OBLIQ's ladder does not grow
monotonically --- its 8k rung is the largest corpus of the five and its 16k rung is smaller than
its 8k one --- so its curve should not be read as a length sweep. $\ddagger$~The reranking pool
saturates at 100 documents by 16k, so the deeper rungs of that row grow far less than the label
suggests.}
\label{tab:data-stats}
\end{table}
"""
    for d in (os.path.join(HERE), ALSO):
        if os.path.isdir(d):
            with open(os.path.join(d, "table_datastats.tex"), "w") as fh:
                fh.write(tex)
    print(f"[{FIGURE_LABEL}] wrote table_datastats.tex ({len(TASKS)} tasks) to {HERE} and {ALSO}")
    if notes:
        print("⚠ eval sets below the 500-example floor -- flag these inline wherever quoted:")
        for n in notes:
            print(f"    {n}")
    missing = [f"{n} @{r}" for k, n, _ in TASKS for r in LADDER
               if not stats.get(k, {}).get(r)]
    if missing:
        print("⚠ rungs with no file (unevaluated cells, NOT measured zeros): " + "; ".join(missing))


if __name__ == "__main__":
    main()
