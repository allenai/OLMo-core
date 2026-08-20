"""
BEIR (SciFact, FiQA, ...) -> a :class:`~ctc.data.sources.retrieval.RetrievalPool`.

Loads a BEIR corpus and its qrels, BM25-mines candidates from the dataset's **own** corpus, and
cross-encoder-filters them so the hard negatives are hard rather than merely unlabelled.

.. warning::
   **The CE filter is not optional on the sparsely-judged sets.** FiQA averages ~2.6 gold documents
   per query over a ~57k corpus, so BM25's top hits are full of relevant-but-unjudged passages.
   Shipping those as negatives does not make retrieval harder, it makes the labels wrong. The
   pre-migration tree learned this the expensive way: ``generate_beir_data.py`` (plain BM25
   distractors) was superseded by ``generate_beir_ce_data.py`` for exactly this reason, and the
   shipped ``beir_fiqa_ce_*`` ladders are the CE ones.

SciFact's shipped ladder (``beir_scifact_ladder_k*``) predates the CE path and was built with plain
BM25 distractors -- it is single-gold and densely judged, so the false-negative rate is much lower.
``ce_filter`` therefore defaults on for both, and turning it off for SciFact reproduces the shipped
build rather than being a mistake.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .retrieval import Candidate, QueryPool, RetrievalPool, margin_filter

__all__ = ["load_beir", "load_pool", "BEIR_DATASETS"]

#: Held-out BEIR sets the suite uses, with the number of gold documents per query that decides how
#: badly an unfiltered BM25 pool leaks.
BEIR_DATASETS: Dict[str, str] = {
    "scifact": "single-gold, densely judged (the NQ analogue)",
    "fiqa": "~2.6 gold/query, sparsely judged -- CE filtering is what makes its negatives valid",
}


def load_beir(name: str, split: str = "test") -> Tuple[List[Candidate], List[Dict], Dict]:
    """
    Read one BEIR dataset off the Hub.

    :param name: BEIR name, e.g. ``"scifact"`` or ``"fiqa"``.
    :param split: qrels split; falls back to the dataset's only split when absent.

    :returns: ``(corpus, queries, qrels)``. ``title`` is folded into the document text -- BEIR
        titles are part of the passage, and emitting them as a separate ``title`` field is how the
        docchunk wrapping leak got in. Only queries with at least one positive qrel are returned.

    :raises ImportError: If ``datasets`` is missing; install ``ctc[sources]``.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - depends on the install
        raise ImportError("BEIR loading needs `datasets`: pip install './ctc[sources]'") from e

    corpus_ds = load_dataset(f"BeIR/{name}", "corpus")["corpus"]
    queries_ds = load_dataset(f"BeIR/{name}", "queries")["queries"]
    qrels_ds = load_dataset(f"BeIR/{name}-qrels")
    qsplit = split if split in qrels_ds else list(qrels_ds.keys())[0]

    corpus = []
    for row in corpus_ds:
        title = (row.get("title") or "").strip()
        text = row["text"]
        # FiQA ships 38 corpus rows whose text is empty. They are undrawable -- the schema
        # refuses an empty document -- and at the 2k-32k rungs (~80 of 57k docs per example)
        # they were never sampled, so the failure surfaced only at the 10M rung, where one
        # example draws half the corpus. A pool must not contain documents no example can hold.
        if not text or not text.strip():
            continue
        corpus.append(Candidate(str(row["_id"]), f"{title}. {text}" if title else text))

    qrels: Dict[str, List[str]] = {}
    for row in qrels_ds[qsplit]:
        if int(row["score"]) > 0:
            qrels.setdefault(str(row["query-id"]), []).append(str(row["corpus-id"]))

    queries = [
        {"_id": str(r["_id"]), "text": r["text"]} for r in queries_ds if str(r["_id"]) in qrels
    ]
    return corpus, queries, qrels


def load_pool(
    *,
    dataset: str = "fiqa",
    split: str = "test",
    seed: int = 42,
    bm25_topk: int = 100,
    ce_filter: bool = True,
    ce_model: Optional[str] = None,
    ce_margin: float = 3.0,
    index_dir: str = "data/.cache/beir/_indices",
    bm25_threads: int = 8,
) -> RetrievalPool:
    """
    Build a BEIR retrieval pool: gold from qrels, hard negatives BM25-mined and CE-filtered.

    :param dataset: BEIR name; see :data:`BEIR_DATASETS`.
    :param split: qrels split.
    :param seed: Shuffles the query order, which fixes which question example *i* uses.
    :param bm25_topk: BM25 candidates per query to consider as hard negatives.
    :param ce_filter: Run the cross-encoder margin filter. On by default; see the module warning.
    :param ce_model: Cross-encoder id; defaults to the MS MARCO MiniLM used by every shipped build.
    :param ce_margin: Required gap below the gold score. 3.0 is SBERT's recipe.
    :param index_dir: Where the per-dataset Lucene index is cached.
    :param bm25_threads: Lucene threads.

    :returns: The pool, queries in a seeded shuffled order.

    :raises ImportError: If ``datasets``/``pyserini`` (or ``transformers`` with ``ce_filter``) are
        missing.
    """
    import random

    from .._bm25 import LocalBM25Searcher

    corpus, queries, qrels = load_beir(dataset, split)
    by_id = {c.id: c for c in corpus}

    searcher = LocalBM25Searcher(corpus, f"{index_dir}/{dataset}", threads=bm25_threads)
    hits = searcher.batch_search(
        [q["text"] for q in queries], k=bm25_topk + 50, threads=bm25_threads
    )
    scorer = None
    if ce_filter:
        from .retrieval import cross_encoder

        scorer = cross_encoder(ce_model)

    prepared: List[QueryPool] = []
    for query, query_hits in zip(queries, hits):
        gold_ids = [g for g in qrels.get(str(query["_id"]), []) if g in by_id]
        if not gold_ids:
            continue
        gold = tuple(by_id[g] for g in gold_ids)
        gold_set = set(gold_ids)
        candidates = [
            by_id[h["docid"]]
            for h in query_hits
            if h["docid"] not in gold_set and h["docid"] in by_id
        ][:bm25_topk]
        if scorer is None or not candidates:
            # BM25 rank order already puts the hardest first.
            hard = tuple(candidates)
        else:
            gold_scores = scorer.score([(query["text"], d.text) for d in gold])
            cand_scores = scorer.score([(query["text"], d.text) for d in candidates])
            hard = tuple(margin_filter(gold_scores, candidates, cand_scores, ce_margin))
        prepared.append(QueryPool(query=query["text"], gold=gold, hard=hard))

    rng = random.Random(seed)
    rng.shuffle(prepared)
    # `beir_<name>`, not `beir_<name>_ce`: the tag names the corpus, and the retrieval spec's
    # declared `sources` is the list a provenance audit checks against. Whether the negatives were
    # CE-filtered is a property of the build, recorded per example in `_meta`.
    return RetrievalPool(queries=tuple(prepared), corpus=tuple(corpus), source=f"beir_{dataset}")
