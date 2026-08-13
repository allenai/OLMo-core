"""Dump one real example per CTC-suite task, trimmed, to source the paper appendix.

Reads the unified-format eval JSONL for every task in the suite and prints a
compact rendering: source tag, corpus size, the per-example query (if any), the
gold field, and a few documents (gold ones prioritized) truncated for display.
Content is verbatim from the eval files — the appendix examples in
`paperdraft/iclr2026/sections/appendix_examples.tex` are hand-formatted from
this output so every quoted string is real data.

Usage:
    python3 debug/paper_examples/dump_for_paper.py [DATA_DIR] > debug/paper_examples/dump.txt
"""
import json
import os
import sys

DATA = sys.argv[1] if len(sys.argv) > 1 else \
    "/accounts/projects/berkeleynlp/prasann/projects/corpus-reasoning/data"

# Re-synced 2026-08-13 to the frozen 22-task suite (blocks 12 / 2 / 7 / 1), so this list is now
# exactly the roster in debug/ctc_final_suite/build_suite_table.py.
#   Cut with their tasks: HELMET NarrativeQA, HELMET GovReport, qdmatch (ObliQ), cycle, groups-of-4,
#   redundancy, absence (official), mathmatch.
#   Added: outlier (wiki, fixed M), qdmatch (FiQA).
#   Re-pointed: absence now reads the Gutenberg text-diff build the suite actually grades, not the
#   PubMed build -- the appendix had been showing an example of a task nothing scores.
#
# Entries whose filename is an absolute path read a staged CTC ladder rung over /net instead of
# DATA. Those files are node-local, so the node in the path is the one that had it when this was
# written; if it moves, `harvested_grades.json` records the eval_data path for every graded rung.
STAGED = "/net/{node}/data/prasann/ctc_suite_staged"

# (display name, filename or absolute path, n docs to show, note)
TASKS = [
    # ── O(N) ──
    ("NQ retrieval",         "nq_validation_k20_hn19_500.jsonl", 3, "single-gold, BM25 hard negs"),
    ("HotpotQA bridge",      "hotpotqa_eval_k20_bridge_unified_hn18_500_cot.jsonl", 4, "2 golds + short CoT"),
    ("NIAH-contradiction",   "niah_contradiction_eval_n99_p1.jsonl", 3, "needle = the contradicting claim"),
    ("BEIR SciFact",         "beir_scifact_ladder_k22_299.jsonl", 3, "claim -> abstract"),
    ("BEIR FiQA (CE)",       "beir_fiqa_ce_ladder_k20_648.jsonl", 3, "CE-cleaned negs"),
    ("MS MARCO retrieval",   "msmarco_trainhn_eval_k20_500.jsonl", 3, "10% CE-filtered hard negs"),
    ("MS MARCO rerank",      "msmarco_trainhn_eval_k20_500.jsonl", 3, "same pool, ranked output"),
    ("OBLIQ retrieval",      "obliq_writing_test_k30_461.jsonl", 3, "long-query BEIR-style"),
    ("OOLONG",               "oolong_test_synth_ctx1024_spliteval.jsonl", 1, "aggregate over labeled items"),
    ("Absence (Gutenberg)",  STAGED.format(node="cubbins") + "/eval_rungs/absence_gutenberg/rung_2048.jsonl", 4,
                             "text-diff; answer is sentence prefixes, NOT doc IDs"),
    ("Outlier (Amazon)",     STAGED.format(node="cubbins") + "/eval_rungs/outlier_amzn/rung_2048.jsonl", 4,
                             "closed M: star rating / product category"),
    ("Outlier (wiki fixedM)", STAGED.format(node="sneetches") + "/eval_rungs_fixedM/outlier_fixedM/rung_8192.jsonl", 4,
                             "article count pinned as N grows"),
    # ── O(NM) ──
    ("Outlier (wiki scale-k)", "outlier_wiki100w_n55_k3_eval.jsonl", 4, "M grows with N"),
    ("Grouping (OpenAlex)",  "openalex_grouping_n20_levels_eval_500.jsonl", 4, "partition into k groups"),
    # ── O(N^2) ──
    ("Contradiction",        "contradiction_eval_pubmed_both_n20_k3.jsonl", 5, "PubMed perturbation pairs"),
    ("xabsence",             "xabsence_eval_pubmed_p8_k3.jsonl", 5, "cross-corpus orphans"),
    ("strmatch",             "strmatch_eval_n20_k3_sub3_l10_h3.jsonl", 4, "shared word runs"),
    ("qdmatch (NQ)",         "qdmatch_eval_nq_q20_n20_k3_separate.jsonl", 5, "sparse query-doc pairs"),
    ("qdmatch (HPQA)",       "qdmatch_eval_hpqa_q20_n20_k3_separate.jsonl", 5, "2 golds per query"),
    ("qdmatch (FiQA)",       STAGED.format(node="sneetches") + "/eval_rungs/qdmatch_fiqa/rung_8192.jsonl", 6,
                             "finance forum answers; matching is not entity overlap"),
    ("Reorder (Gutenberg)",  "reorder_gutenberg_n50_eval.jsonl", 3, "recover permutation"),
    # ── O(N^3+) ──
    ("Textgroups",           "textgroups_eval_n20_g3_k2_t70_w0_mixed.jsonl", 3, "feature-count triples"),
]

TRUNC = 190


def show(txt, n=TRUNC):
    txt = " ".join(str(txt).split())
    return txt if len(txt) <= n else txt[:n] + " …"


def gold_positions(ex):
    """1-indexed positions mentioned by whatever gold field this task uses."""
    g = ex.get("gold_doc_indices")
    out = []
    if isinstance(g, list):
        for item in g:
            if isinstance(item, list):
                out += [int(x) for x in item]
            elif isinstance(item, int):
                out.append(item + 1)  # flat lists are 0-indexed in this schema
    for pair in ex.get("gold_pairs", []) or []:
        out += [int(x) for x in pair]
    return out


def resolve(fname):
    if os.path.isabs(fname):                      # a staged CTC ladder rung, read over /net
        return fname if os.path.exists(fname) else None
    p = os.path.join(DATA, fname)
    if os.path.exists(p):
        return p
    stem = fname.split(".jsonl")[0]
    hits = sorted(f for f in os.listdir(DATA) if f.startswith(stem) and f.endswith(".jsonl"))
    return os.path.join(DATA, hits[0]) if hits else None


for name, fname, n_show, note in TASKS:
    print("=" * 100)
    print(f"### {name}   [{note}]")
    path = resolve(fname)
    if path is None:
        print(f"    !! no file matching {fname} in {DATA}")
        continue
    print(f"    file: {os.path.basename(path)}")
    with open(path) as fh:
        ex = json.loads(fh.readline())

    docs = ex.get("documents", [])
    print(f"    n_documents: {len(docs)}   keys: {sorted(ex.keys())}")
    for key in ("source", "queries", "answers", "gold_doc_indices", "gold_pairs",
                "gold_order", "num_pairs", "num_unmatched", "num_queries",
                "num_docs", "num_relevant", "layout", "level", "k",
                "cluster_labels", "meta", "_meta"):
        if key in ex and ex[key] not in ([], None, ""):
            print(f"    {key}: {show(ex[key], 420)}")

    golds = set(gold_positions(ex))
    order = [i for i in range(1, len(docs) + 1) if i in golds]
    order += [i for i in range(1, len(docs) + 1) if i not in golds]
    for i in order[:n_show]:
        d = docs[i - 1]
        tag = " (GOLD)" if i in golds else ""
        extra = "".join(f" [{k}={d[k]}]" for k in ("type", "corpus", "title")
                        if k in d and d[k] is not None)
        print(f"      [{i}]{tag}{extra} {show(d.get('text', d))}")
    print()
