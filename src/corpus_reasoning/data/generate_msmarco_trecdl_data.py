"""Generate MS MARCO TREC-DL in-context retrieval data with DENSE judgments.

Unlike `generate_msmarco_data.py` (dev qrels, ~1 sparse positive/query → BM25
distractors are riddled with unlabeled positives / false negatives), this uses
the **TREC-DL 2019/2020 passage track**, where NIST judged hundreds of passages
per query with GRADED relevance (0-3). The candidate pool is built ONLY from
judged passages:

  * gold        = judged grade >= --rel-threshold (default 2 = "highly/perfectly
                  relevant", the standard TREC-DL passage binarization)
  * distractors = judged grade <  --rel-threshold (grades 0 "irrelevant" and 1
                  "related but does not answer") — GUARANTEED-true negatives.

No BM25 guessing → no false negatives. Grade-1 "related" passages are kept ahead
of grade-0 ones, so the distractors are still topically hard, just human-verified
non-answers. Mirrors the BEIR pipeline: reuses OBLIQ's `_build_examples` /
`_assemble_example`, so the output schema is identical to beir_*/msmarco_rerank.

Passage text + docids come from the prebuilt pyserini `msmarco-v1-passage` index
(docids align with TREC-DL qrels — no 8.8M-passage download). Topics/qrels come
from pyserini's bundled TREC-DL resources.

Needs pyserini → run via sbatch (jobs/gen_msmarco_trecdl.sh).

Usage:
    python scripts/data/generate_msmarco_trecdl_data.py --year 2019 \\
        --num-docs 50 --output-dir data
"""

import argparse
import math
import random
from pathlib import Path

from corpus_reasoning.lib.io import save_jsonl, print_dataset_stats
from corpus_reasoning.data.generate_obliq_data import _build_examples, _print_obliq_stats
from corpus_reasoning.data.generate_msmarco_data import _passage_text

# pyserini resource-name candidates per year (first that resolves wins).
_TOPIC_CANDIDATES = {2019: ["dl19-passage"], 2020: ["dl20-passage", "dl20"]}
_QREL_CANDIDATES = {2019: ["dl19-passage"], 2020: ["dl20-passage", "dl20"]}


def _resolve(getter, candidates, kind):
    """Return the first candidate name that yields a non-empty pyserini resource."""
    for name in candidates:
        try:
            res = getter(name)
        except Exception:
            continue
        if res:
            return name, res
    from pyserini.search import Topics, Qrels  # for a helpful error
    avail = sorted(t.value for t in (Topics if kind == "topics" else Qrels))
    raise SystemExit(f"None of {candidates} resolved a TREC-DL {kind}. "
                     f"Available {kind}: {avail}")


def load_trec_dl(year, rel_threshold):
    """Return (queries, qrels, judged_neg, corpus_by_id) from TREC-DL judgments.

    queries: list[{_id, text}]   qrels: {qid: [gold docids]}
    judged_neg: {qid: [non-relevant docids, hardest (grade 1) first]}
    corpus_by_id: {docid: {_id, text}} for every judged docid with retrievable text.
    """
    from pyserini.search.lucene import LuceneSearcher
    from pyserini.search import get_topics, get_qrels

    searcher = LuceneSearcher.from_prebuilt_index("msmarco-v1-passage")
    tname, topics = _resolve(get_topics, _TOPIC_CANDIDATES[year], "topics")
    qname, qrels_raw = _resolve(get_qrels, _QREL_CANDIDATES[year], "qrels")
    print(f"  topics='{tname}' ({len(topics)})  qrels='{qname}' ({len(qrels_raw)})")
    topic_text = {str(qid): t.get("title", "") for qid, t in topics.items()}

    queries, qrels, judged_neg, all_docids = [], {}, {}, set()
    skipped_no_text = 0
    for qid, doc_grades in qrels_raw.items():
        sqid = str(qid)
        qtext = topic_text.get(sqid)
        if not qtext:
            skipped_no_text += 1
            continue
        golds = [str(d) for d, g in doc_grades.items() if int(g) >= rel_threshold]
        # Non-relevant, hardest first (grade 1 "related" ahead of grade 0).
        negs = sorted((d for d, g in doc_grades.items() if int(g) < rel_threshold),
                      key=lambda d: -int(doc_grades[d]))
        negs = [str(d) for d in negs]
        if not golds:
            continue
        queries.append({"_id": sqid, "text": qtext})
        qrels[sqid] = golds
        judged_neg[sqid] = negs
        all_docids.update(golds)
        all_docids.update(negs)
    if skipped_no_text:
        print(f"  {skipped_no_text} judged queries had no topic text (skipped)")

    corpus_by_id = {}
    for did in all_docids:
        text = _passage_text(searcher, did)
        if text:
            corpus_by_id[did] = {"_id": did, "text": text}
    missing = len(all_docids) - len(corpus_by_id)
    if missing:
        print(f"  {missing}/{len(all_docids)} judged docids had no text in the index")
    return queries, qrels, judged_neg, corpus_by_id


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, choices=[2019, 2020], default=2019)
    ap.add_argument("--num-docs", type=int, default=50,
                    help="documents per example (judged golds + judged negatives)")
    ap.add_argument("--rel-threshold", type=int, default=2,
                    help="min graded judgment counted as gold (TREC-DL passage: 2)")
    ap.add_argument("--min-neg-frac", type=float, default=0.25,
                    help="reserve >= this fraction of each pool for verified "
                         "negatives; golds beyond the remainder are sampled down "
                         "(TREC-DL is densely judged → pools would otherwise be "
                         "all-gold and degenerate)")
    ap.add_argument("--output-dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true",
                    help="bypass the deprecation guard (you almost certainly want "
                         "generate_msmarco_trainhn_data.py instead)")
    args = ap.parse_args()

    if not args.force:
        raise NotImplementedError(
            "generate_msmarco_trecdl_data.py is DEPRECATED. The canonical MS MARCO "
            "in-context retrieval task is now generated by "
            "generate_msmarco_trainhn_data.py (2k train / 500 eval, CE-cleaned hard "
            "negatives + format-identical random fill, no false negatives). This "
            "TREC-DL judged-pool generator is eval-only (~97 queries) and kept for "
            "reference. If you really need it, re-run with --force. See results/beir.md.")

    rng = random.Random(args.seed)
    print(f"Loading TREC-DL {args.year} passage judgments "
          f"(rel-threshold={args.rel_threshold}) ...")
    queries, qrels, judged_neg, corpus_by_id = load_trec_dl(args.year, args.rel_threshold)
    print(f"  queries(with gold)={len(queries)}  judged docs={len(corpus_by_id):,}")

    # TREC-DL is densely judged → many queries have >num_docs relevant passages,
    # which would leave a pool with zero verified negatives (degenerate retrieval).
    # Reserve a floor of negatives by sampling golds down to leave room.
    max_gold = args.num_docs - max(1, math.ceil(args.num_docs * args.min_neg_frac))
    capped = 0
    for q in queries:
        g = qrels[q["_id"]]
        if len(g) > max_gold:
            qrels[q["_id"]] = rng.sample(g, max_gold)
            capped += 1
    if capped:
        print(f"  Capped golds to {max_gold} for {capped} queries "
              f"(reserving >={args.num_docs - max_gold} verified negatives/pool)")

    # Feed judged NON-relevant passages as the distractor candidate list — same
    # contract as the BM25 `hits` in the BEIR pipeline, but every candidate is a
    # human-verified true negative.
    hits_per_query = [[{"docid": d} for d in judged_neg[q["_id"]] if d in corpus_by_id]
                      for q in queries]

    examples = _build_examples(
        queries, hits_per_query, qrels, {}, corpus_by_id, args.num_docs,
        f"msmarco_trecdl{args.year}", [(0, args.num_docs)], rng,
        desc=f"TREC-DL {args.year} passage")

    path = (f"{args.output_dir}/msmarco_trecdl{args.year}"
            f"_k{args.num_docs}_{len(examples)}.jsonl")
    save_jsonl(path, examples)
    print_dataset_stats(examples, f"msmarco_trecdl{args.year}", path)
    _print_obliq_stats(examples, f"TREC-DL {args.year}", path)


if __name__ == "__main__":
    main()
