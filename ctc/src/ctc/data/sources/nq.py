"""
Natural Questions (``nq_open``) -> a :class:`~ctc.data.sources.retrieval.RetrievalPool`.

Every document -- gold, hard negative and filler -- comes out of pyserini's ``wikipedia-dpr-100w``,
so the whole context shares one surface format and nothing about a passage's provenance is visible
from its shape. Gold is the first BM25 hit that *contains* an answer string; questions with no such
hit are dropped.

.. warning::
   **The banned 98%-hard regime is one flag away, and nothing warned you.** The pre-migration
   generator defaulted ``--hard-neg-frac`` to **1.0** and left ``--ce-filter`` **off**, so a plain
   run reproduced the retired all-hard-negative pipeline that produced the ``hn49``/``hn99``/
   ``hn199``/``ladder64k`` files nobody may use. The defaults here are inverted --
   ``hard_neg_frac=0.1`` and ``ce_filter=True`` -- which is the "p10 + CE-filter" pipeline every
   current NQ result was measured on. Raising ``hard_neg_frac`` is a deliberate act, not an
   omission.

   The CE filter is what makes the *gold* trustworthy, separately from the negatives: an
   answer-string match is sometimes incidental, and the model is then marked wrong for not
   retrieving a passage a hard negative out-ranks. A gold is kept only when it beats every
   reference hard negative by ``ce_margin``.
"""

from __future__ import annotations

from typing import List, Optional

from .retrieval import Candidate, QueryPool, RetrievalPool

__all__ = ["load_pool"]

#: Compare the gold against only this many top hard negatives. Fixed rather than "all of them" so
#: the keep/drop decision does not move with the pool size -- the same questions must survive at
#: every rung, or the ladder's rungs stop grading the same eval set.
CE_REFERENCE_NEGATIVES = 20


def load_pool(
    *,
    split: str = "train",
    num_questions: int = 25_000,
    seed: int = 42,
    max_hard: int = 40,
    max_pair_overlap: float = 0.5,
    filler_pool: int = 60_000,
    ce_filter: bool = True,
    ce_model: Optional[str] = None,
    ce_gold_min: float = 0.0,
    ce_margin: float = 0.0,
    batch_size: int = 64,
    bm25_threads: int = 8,
) -> RetrievalPool:
    """
    Prepare NQ questions with BM25 gold, verified hard negatives and a random-filler pool.

    :param split: ``nq_open`` split. ``"train"`` has ~88k questions; ``"validation"`` is the
        canonical 3.6k NQ-open test set.
    :param num_questions: Questions to *consider*; the yield is lower because a question with no
        answer-bearing BM25 hit is dropped, and the CE filter drops more.
    :param seed: Shuffles the question order, fixing which question example *i* uses.
    :param max_hard: Hard negatives mined per question. A rung takes a prefix of these.
    :param max_pair_overlap: Near-duplicate rejection between hard negatives.
    :param filler_pool: Random passages pre-fetched once, to draw ordinary distractors from.
    :param ce_filter: Keep only questions whose gold is genuinely the most relevant passage. On by
        default -- see the module warning.
    :param ce_model: Cross-encoder id.
    :param ce_gold_min: Absolute floor on the gold's relevance logit.
    :param ce_margin: Required gap between the gold and the best reference hard negative.
    :param batch_size: Questions per fused BM25 batch.
    :param bm25_threads: Lucene thread pool size.

    :returns: The pool, questions in a seeded shuffled order.

    :raises ImportError: If ``datasets``/pyserini (or ``transformers`` with ``ce_filter``) are
        missing.
    """
    import random

    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - depends on the install
        raise ImportError("NQ loading needs `datasets`: pip install './ctc[sources]'") from e

    from .._bm25 import PrebuiltBM25Searcher, pick_hard_negatives

    dataset = load_dataset("nq_open", split=split)
    order = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(order)
    order = order[:num_questions]

    searcher = PrebuiltBM25Searcher()
    scorer = None
    if ce_filter:
        from .retrieval import cross_encoder

        scorer = cross_encoder(ce_model)

    prepared: List[QueryPool] = []
    for start in range(0, len(order), batch_size):
        rows = [dataset[i] for i in order[start : start + batch_size]]
        questions = [r["question"] for r in rows]
        hit_lists = searcher.batch_search(questions, k=max(200, max_hard * 5), threads=bm25_threads)

        batch: List[QueryPool] = []
        for row, hits in zip(rows, hit_lists):
            answers = list(row["answer"])
            gold_hit = next(
                (h for h in hits if any(a.lower() in h["text"].lower() for a in answers)), None
            )
            if gold_hit is None:
                continue
            hard = pick_hard_negatives(
                hits,
                max_hard,
                exclude_answers=answers,
                exclude_docids={gold_hit["docid"]},
                max_pair_overlap=max_pair_overlap,
            )
            batch.append(
                QueryPool(
                    query=row["question"],
                    gold=(Candidate(gold_hit["docid"], gold_hit["text"]),),
                    hard=tuple(Candidate(h["docid"], h["text"]) for h in hard),
                    # The FULL alias list, not just answers[0]: gold selection and distractor
                    # exclusion both run over every alias, so storing one left ~18% of golds not
                    # containing their own recorded answer.
                    answers=tuple(answers),
                )
            )
        prepared.extend(
            batch if scorer is None else _keep_trusted_gold(batch, scorer, ce_gold_min, ce_margin)
        )

    fillers = searcher.sample_pool(filler_pool, seed=seed)
    return RetrievalPool(
        queries=tuple(prepared),
        corpus=tuple(Candidate(f"filler:{i}", text) for i, text in enumerate(fillers)),
        source="nq",
    )


def _keep_trusted_gold(
    batch: List[QueryPool], scorer, gold_min: float, margin: float
) -> List[QueryPool]:
    """
    Drop questions whose answer-bearing gold is not the most query-relevant passage in the pool.

    :param batch: Prepared questions.
    :param scorer: Cross-encoder.
    :param gold_min: Absolute floor on the gold's score.
    :param margin: Required gap over the best reference hard negative.

    :returns: The survivors. All pairs are scored in one call.
    """
    pairs = []
    spans = []
    for query in batch:
        gold_index = len(pairs)
        pairs.append((query.query, query.gold[0].text))
        hard_indices = []
        for candidate in query.hard[:CE_REFERENCE_NEGATIVES]:
            hard_indices.append(len(pairs))
            pairs.append((query.query, candidate.text))
        spans.append((gold_index, hard_indices))
    if not pairs:
        return []
    scores = scorer.score(pairs)
    kept = []
    for query, (gold_index, hard_indices) in zip(batch, spans):
        gold_score = scores[gold_index]
        best_hard = max((scores[i] for i in hard_indices), default=float("-inf"))
        if gold_score >= gold_min and gold_score >= best_hard + margin:
            kept.append(query)
    return kept
