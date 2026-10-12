"""
``rerank`` -- MS MARCO passage re-ranking, one of the five in-distribution ladders.

Structurally a retrieval example, graded differently: the answer is the *ordering* of the whole
pool, not the subset that is relevant. It therefore reuses :mod:`ctc.tasks.retrieval.generate`'s
assembly, with one addition that is not optional.

.. warning::
   **Every document must carry a ``ce_scores`` entry.** The graded target is built from those
   scores; a ``None`` is read as gain 0 and dropped from the Kendall-tau set, so an unscored random
   fill makes the reference order tail off into display order and the metric quietly grades a
   shorter list than the model ranked. That is why
   :func:`ctc.data.sources.msmarco.load_pool` draws and scores each query's random fill at *load*
   time rather than leaving it to the example builder -- and why ``score_all`` defaults on.

   The pre-migration *binary* rerank format -- gold-first target, no per-document scores -- is
   disabled in the spec rather than deprecated, so stale MRR-only numbers cannot be read as
   current ones.

``rerank`` and the ``msmarco`` source of ``retrieval`` are one build, not two: the same file feeds
both, which is why the loader lives in :mod:`ctc.data.sources.msmarco` rather than in either task.
"""

from __future__ import annotations

import random
from typing import Dict, Optional

from ...data.generators.base import Generator
from ...data.sources import msmarco as source
from ...data.sources.retrieval import RetrievalPool
from ..retrieval.generate import build_from_pool

__all__ = ["build_example", "GENERATOR"]


def build_example(
    rng: random.Random,
    *,
    index: int,
    corpus: RetrievalPool,
    num_docs: int,
    hard_frac: float = 0.1,
) -> Optional[Dict]:
    """
    Build one CE-graded rerank example.

    :param rng: Seeded RNG.
    :param index: Example counter; query *index* is used.
    :param corpus: Prepared MS MARCO pool, carrying per-document CE scores.
    :param num_docs: Pool size N.
    :param hard_frac: Fraction of the non-gold slots filled from the CE-verified mined negatives.

    :returns: The example with ``ce_scores`` parallel to ``documents``, or ``None`` when the pool
        is exhausted or has too few pre-drawn fillers for ``num_docs``.
    """
    return build_from_pool(
        rng,
        index=index,
        corpus=corpus,
        num_docs=num_docs,
        hard_frac=hard_frac,
        with_scores=True,
    )


GENERATOR = Generator(
    name="rerank",
    task="rerank",
    source="msmarco_trainhn",
    build_example=build_example,
    defaults={"num_docs": 20, "hard_frac": 0.1},
    corpus=source.load_pool,
    corpus_defaults={
        "num_queries": 5_000,
        "seed": 42,
        "index": "msmarco-v1-passage",
        "ce_margin": 3.0,
        "max_docs": 250,
        "score_all": True,
        "ce_model": None,
        "candidate_multiplier": 2.0,
    },
    indexed=True,
    notes="MS MARCO, CE-graded: every document carries a relevance score",
)
