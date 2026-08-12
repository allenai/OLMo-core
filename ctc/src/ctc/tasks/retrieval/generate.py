"""
The ``retrieval`` data contract: one query, a pool of documents, name the relevant ones.

Six corpora alias to this one task -- ``nq``, ``hotpotqa``, ``niah_contradiction``,
``beir_scifact``, ``beir_fiqa`` and ``msmarco`` -- and they differ only in how gold and hard
negatives are found. That mining lives in :mod:`ctc.data.sources`; this file is the assembly every
one of them shares, and it is shared because the pre-migration tree wrote it eight times in seven
files and the copies drifted.

**Gold is 0-based here**, unlike the pair/group family. That is a per-task property, declared once
on the spec and checked against the generated file, because an index-base disagreement is silent:
correct answers score zero, uniformly, and read as a modelling result.
"""

from __future__ import annotations

import random
from typing import Dict, List, Mapping, Optional, Sequence

from ...data.schema import make_document, make_example
from ...data.sources.retrieval import Candidate, QueryPool, RetrievalPool

__all__ = ["assemble", "draw_pool", "hard_count"]


def hard_count(num_negatives: int, hard_frac: float, available: int) -> int:
    """
    How many of an example's negatives are *hard* negatives.

    A constant fraction rather than a constant count, so the hard/random mix is the same at every
    rung. A fixed count would make the short rungs almost entirely hard and the long ones almost
    entirely random, and the ladder would then be measuring two different tasks at its two ends.

    :param num_negatives: Non-gold slots in this example.
    :param hard_frac: Fraction of them that should be hard negatives.
    :param available: Hard negatives this query actually has.

    :returns: The count, at least 1 whenever ``hard_frac > 0`` and any are available.
    """
    if hard_frac <= 0 or available <= 0 or num_negatives <= 0:
        return 0
    return min(num_negatives, available, max(1, round(hard_frac * num_negatives)))


def draw_pool(
    query: QueryPool,
    corpus: Sequence[Candidate],
    num_docs: int,
    hard_frac: float,
    rng: random.Random,
) -> Optional[Dict[str, List[Candidate]]]:
    """
    Choose this example's documents: every gold, a prefix of the hard negatives, then fillers.

    A *prefix* of the hard negatives, not a sample: they are sorted hardest-first, so a short rung
    holds the hardest ones and a long rung holds those plus easier ones. Sampling instead would
    change which negatives a question faces at each rung, confounding "the context got longer" with
    "the distractors got different".

    :param query: The prepared query.
    :param corpus: Filler supply, used only when the query has no pre-drawn ``fill``.
    :param num_docs: Total documents wanted.
    :param hard_frac: Fraction of negatives that should be hard.
    :param rng: Seeded RNG.

    :returns: ``{"gold": [...], "hard": [...], "random": [...]}``, or ``None`` when the pool cannot
        be filled to ``num_docs``.
    """
    gold = list(query.gold[:num_docs])
    need = num_docs - len(gold)
    if need <= 0:
        return {"gold": gold, "hard": [], "random": []}

    n_hard = hard_count(need, hard_frac, len(query.hard))
    hard = list(query.hard[:n_hard])
    used = {c.id for c in gold} | {c.id for c in hard}

    n_random = need - len(hard)
    if query.fill:
        # Pre-drawn per-query fill: taking a prefix keeps a rung an exact subset of the next one.
        fillers = [c for c in query.fill if c.id not in used][:n_random]
    else:
        indices = list(range(len(corpus)))
        rng.shuffle(indices)
        fillers = []
        for i in indices:
            if len(fillers) >= n_random:
                break
            if corpus[i].id in used:
                continue
            used.add(corpus[i].id)
            fillers.append(corpus[i])
    if len(fillers) < n_random:
        return None
    return {"gold": gold, "hard": hard, "random": fillers}


def assemble(
    query: QueryPool,
    drawn: Mapping[str, Sequence[Candidate]],
    *,
    source: str,
    rng: random.Random,
    with_scores: bool = False,
    meta: Optional[Mapping[str, object]] = None,
) -> Dict:
    """
    Tag, shuffle and emit a unified-format retrieval example.

    :param query: The prepared query.
    :param drawn: Output of :func:`draw_pool`.
    :param source: Corpus tag.
    :param rng: Seeded RNG. Consumes exactly one shuffle over the tagged list.
    :param with_scores: Emit ``ce_scores`` parallel to ``documents``. Required by ``rerank``, whose
        target is a graded ordering; a missing score is ``None``, which the grader reads as gain 0
        and drops from the Kendall-tau set.
    :param meta: Per-example metadata.

    :returns: The example. ``gold_doc_indices`` and ``hard_neg_indices`` are **0-based**, and
        ``hard_neg_indices`` tags only the verified hard negatives -- the random fill is left
        untagged, because calling an ordinary distractor a mined one destroys the only record of
        how the pool was built.
    """
    tagged = (
        [(c, "gold") for c in drawn["gold"]]
        + [(c, "hard") for c in drawn["hard"]]
        + [(c, "random") for c in drawn["random"]]
    )
    rng.shuffle(tagged)
    example = make_example(
        documents=[make_document(c.text) for c, _ in tagged],
        queries=[query.query],
        # NQ keeps its full alias list; the corpora with no textual answer carry [""] rather than
        # [], matching every shipped retrieval file.
        answers=list(query.answers) if query.answers else [""],
        source=source,
        gold=[i for i, (_, tag) in enumerate(tagged) if tag == "gold"],
        meta=dict(meta or {}),
        hard_neg_indices=[i for i, (_, tag) in enumerate(tagged) if tag == "hard"],
    )
    if with_scores:
        example["ce_scores"] = [query.scores.get(c.id) for c, _ in tagged]
    return example


def build_from_pool(
    rng: random.Random,
    *,
    index: int,
    corpus: RetrievalPool,
    num_docs: int,
    hard_frac: float,
    with_scores: bool = False,
    meta: Optional[Mapping[str, object]] = None,
) -> Optional[Dict]:
    """
    The whole per-example path, shared by every retrieval-flavoured ladder.

    :param rng: Seeded RNG.
    :param index: Example counter; query *index* is used, so no question is reused inside a split
        while another goes unused.
    :param corpus: The prepared pool.
    :param num_docs: Corpus size N.
    :param hard_frac: Fraction of negatives that are hard.
    :param with_scores: Emit ``ce_scores``.
    :param meta: Extra metadata.

    :returns: The example, or ``None`` when the pool is exhausted or too thin for ``num_docs``.
    """
    if index >= len(corpus.queries):
        return None
    query = corpus.queries[index]
    drawn = draw_pool(query, corpus.corpus, num_docs, hard_frac, rng)
    if drawn is None:
        return None
    return assemble(
        query,
        drawn,
        source=corpus.source,
        rng=rng,
        with_scores=with_scores,
        meta={"hard_frac": hard_frac, **dict(meta or {})},
    )
