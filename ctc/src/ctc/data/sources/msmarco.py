"""
MS MARCO -> a :class:`~ctc.data.sources.retrieval.RetrievalPool` carrying per-document CE scores.

One build feeds two ladders. ``rerank`` grades the *ordering* of the whole pool and needs a
relevance score for every document; ``retrieval`` grades which ids were named. Both read the same
file, which is why the loader lives here rather than in either task.

Hard-negative *selection* needs no model: SentenceTransformers already mined the MS MARCO train
split with BM25 plus ~13 dense retrievers, and shipped the cross-encoder scores alongside. Those
negatives are, by construction, the passages most likely to be unlabelled positives, so they are
margin-filtered against the gold's own score (SBERT's recipe, margin 3).

.. note::
   **Scoring the random fill is not optional for ``rerank``.** The shipped score pickle covers only
   gold and mined negatives, so without ``score_all`` every random-fill document has ``ce_scores =
   None``. The rerank grader reads ``None`` as gain 0 and excludes it from the Kendall-tau set, so
   the reference order tails off into display order and the metric quietly grades a shorter list
   than the one the model ranked.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .retrieval import Candidate, QueryPool, RetrievalPool, margin_filter

__all__ = ["load_pool", "HF_REPO", "MAX_PID"]

HF_REPO = "sentence-transformers/msmarco-hard-negatives"
HARD_NEG_FILE = "msmarco-hard-negatives.jsonl.gz"
CE_SCORE_FILE = "cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz"

#: MS MARCO v1 passage ids are contiguous 0..8841822, so a random integer id is a real passage --
#: byte-identical in format to gold and hard, with no syntactic giveaway. The ~1e-6 chance of
#: drawing a true positive is negligible against a 8.8M collection.
MAX_PID = 8841822


def load_pool(
    *,
    num_queries: int = 5_000,
    seed: int = 42,
    index: str = "msmarco-v1-passage",
    ce_margin: float = 3.0,
    max_docs: int = 250,
    score_all: bool = True,
    ce_model: Optional[str] = None,
    candidate_multiplier: float = 2.0,
) -> RetrievalPool:
    """
    Prepare MS MARCO queries with CE-filtered hard negatives and per-document CE scores.

    :param num_queries: Queries wanted. More are examined (see ``candidate_multiplier``) because a
        query whose gold passage has no text in the index is skipped.
    :param seed: Shuffles the query order, fixing which query example *i* uses.
    :param index: Prebuilt pyserini index supplying passage text and the random fill.
    :param ce_margin: Keep a mined negative only if its CE score is this far below the gold's.
    :param max_docs: Documents the largest rung will need. The random fill is drawn **per query, up
        to this size, at load time** rather than per example, for two reasons: it must be
        CE-scored (see the module note), and pre-drawing is what lets a rung be an exact prefix of
        the next one.
    :param score_all: Cross-encoder-score the random fill too, so every document has a relevance
        value. See the module note -- ``rerank`` needs this.
    :param ce_model: Cross-encoder id; must match the pickle's model so scores share one scale.
        The default does.
    :param candidate_multiplier: Examine this multiple of ``num_queries`` candidates.

    :returns: The pool, queries in a seeded shuffled order, each carrying ``scores``.

    :raises ImportError: If ``datasets``/``huggingface_hub``/pyserini are missing.
    """
    import gzip
    import json
    import pickle
    import random

    try:
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover - depends on the install
        raise ImportError(
            "MS MARCO loading needs `datasets` + `huggingface_hub`: pip install './ctc[sources]'"
        ) from e

    from .._bm25 import PrebuiltBM25Searcher

    searcher = PrebuiltBM25Searcher(index)
    path = hf_hub_download(HF_REPO, HARD_NEG_FILE, repo_type="dataset")

    rows: Dict[int, Tuple[List[int], List[int]]] = {}
    with gzip.open(path, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            negatives, seen = [], set()
            for mined in row.get("neg", {}).values():
                for pid in mined:
                    if pid not in seen:
                        seen.add(pid)
                        negatives.append(int(pid))
            rows[int(row["qid"])] = ([int(p) for p in row.get("pos", [])], negatives)

    rng = random.Random(seed)
    qids = list(rows)
    rng.shuffle(qids)
    qids = qids[: int(num_queries * candidate_multiplier)]

    wanted = {str(q) for q in qids}
    query_text = {
        str(r["_id"]): r["text"]
        for r in load_dataset("BeIR/msmarco", "queries")["queries"]
        if str(r["_id"]) in wanted
    }

    with gzip.open(hf_hub_download(HF_REPO, CE_SCORE_FILE, repo_type="dataset"), "rb") as handle:
        all_scores = pickle.load(handle)
    ce = {q: all_scores[q] for q in qids if q in all_scores}
    del all_scores

    scorer = None
    if score_all:
        from .retrieval import cross_encoder

        scorer = cross_encoder(ce_model)

    prepared: List[QueryPool] = []
    for qid in qids:
        if len(prepared) >= num_queries:
            break
        text = query_text.get(str(qid))
        if text is None:
            continue
        positives, negatives = rows[qid]
        gold = [
            Candidate(str(p), passage) for p in positives if (passage := searcher.passage(str(p)))
        ]
        if not gold:
            continue
        score_map = ce.get(qid, {})
        candidates, candidate_scores = [], []
        for pid in negatives:
            if pid not in score_map:
                continue
            passage = searcher.passage(str(pid))
            if passage:
                candidates.append(Candidate(str(pid), passage))
                candidate_scores.append(score_map[pid])
        gold_scores = [score_map[p] for p in positives if p in score_map]
        hard = margin_filter(gold_scores, candidates, candidate_scores, ce_margin)
        used = {c.id for c in gold} | {c.id for c in hard}
        fill = _random_fill(searcher, max_docs - len(used), used, rng)
        prepared.append(
            QueryPool(
                query=text,
                gold=tuple(gold),
                hard=tuple(hard),
                fill=tuple(fill),
                scores=_pool_scores(gold, hard, fill, score_map, text, scorer),
            )
        )

    return RetrievalPool(queries=tuple(prepared), corpus=(), source="msmarco_trainhn")


def _random_fill(searcher, count: int, used, rng) -> List[Candidate]:
    """
    Draw distinct real passages by random integer pid.

    :param searcher: Passage-text source.
    :param count: How many to draw.
    :param used: Ids already in this query's pool.
    :param rng: Seeded RNG.

    :returns: Up to ``count`` candidates; fewer only if the attempt cap is hit.
    """
    out: List[Candidate] = []
    attempts, cap = 0, max(0, count) * 50 + 200
    while len(out) < max(0, count) and attempts < cap:
        attempts += 1
        pid = str(rng.randint(0, MAX_PID))
        if pid in used:
            continue
        text = searcher.passage(pid)
        if not text:
            continue
        used.add(pid)
        out.append(Candidate(pid, text))
    return out


def _pool_scores(gold, hard, fill, score_map, query, scorer) -> Dict[str, float]:
    """
    Relevance score per document id, for the graded ordering ``rerank`` is scored against.

    A gold with no mined CE score is a **coverage gap in the pickle, not evidence of
    irrelevance**, so it is pinned above every hard negative rather than dropped to the bottom --
    the pre-migration behaviour, and the difference between grading a ranking and grading a bug.

    :param gold: Gold candidates.
    :param hard: Surviving hard negatives.
    :param fill: Random-fill candidates; scored only when ``scorer`` is given.
    :param score_map: ``pid -> score`` from the shipped pickle.
    :param query: Query text, for on-the-fly scoring.
    :param scorer: Cross-encoder, or ``None`` to use the pickle alone.

    :returns: ``doc id -> score``.
    """
    scores: Dict[str, float] = {}
    hard_values = []
    for candidate in hard:
        value = score_map.get(int(candidate.id))
        if value is not None:
            scores[candidate.id] = float(value)
            hard_values.append(float(value))
    known = [float(score_map[int(c.id)]) for c in gold if int(c.id) in score_map]
    if known:
        fill = max(known)
    elif hard_values:
        fill = max(hard_values) + 1.0
    else:
        fill = 0.0
    for candidate in gold:
        scores[candidate.id] = float(score_map.get(int(candidate.id), fill))
    if scorer is not None:
        missing = [c for c in list(gold) + list(hard) + list(fill) if c.id not in scores]
        if missing:
            for candidate, value in zip(missing, scorer.score([(query, c.text) for c in missing])):
                scores[candidate.id] = float(value)
    return scores
