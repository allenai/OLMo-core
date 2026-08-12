"""
``nq`` -- Natural Questions in-context retrieval, one of the five in-distribution ladders.

One question, one answer-bearing Wikipedia passage, and N-1 distractors of which a **tenth** are
BM25 hard negatives. Graded on which document ids were named, not on the answer text.

.. warning::
   **``hard_frac`` defaults to 0.1 and the gold filter defaults on, and both are inversions of the
   pre-migration defaults.** That generator shipped ``--hard-neg-frac 1.0`` with ``--ce-filter``
   off, so a plain run silently rebuilt the retired all-hard-negative regime -- the one that
   produced the ``hn49``/``hn99``/``hn199``/``ladder64k`` files that must never be used. Every
   current NQ number was measured on the 10%-hard, CE-filtered pipeline. Raising ``hard_frac`` is a
   deliberate act; leaving it alone must not be a mistake.
"""

from __future__ import annotations

import random
from typing import Dict, Optional

from ....data.generators.base import Generator
from ....data.sources import nq as source
from ....data.sources.retrieval import RetrievalPool
from ..generate import build_from_pool

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
    Build one NQ retrieval example.

    :param rng: Seeded RNG.
    :param index: Example counter; question *index* is used.
    :param corpus: Prepared NQ pool.
    :param num_docs: Corpus size N (1 gold + N-1 distractors).
    :param hard_frac: Fraction of the distractors that are BM25 hard negatives. See the module
        warning before changing it.

    :returns: The example, or ``None`` when the pool is exhausted.
    """
    return build_from_pool(rng, index=index, corpus=corpus, num_docs=num_docs, hard_frac=hard_frac)


GENERATOR = Generator(
    name="nq",
    task="retrieval",
    source="nq",
    build_example=build_example,
    defaults={"num_docs": 20, "hard_frac": 0.1},
    corpus=source.load_pool,
    corpus_defaults={
        "split": "train",
        "num_questions": 25_000,
        "seed": 42,
        "max_hard": 40,
        "max_pair_overlap": 0.5,
        "filler_pool": 60_000,
        "ce_filter": True,
        "ce_model": None,
        "ce_gold_min": 0.0,
        "ce_margin": 0.0,
        "batch_size": 64,
        "bm25_threads": 8,
    },
    indexed=True,
    notes="NQ-open over wikipedia-dpr-100w; 10% hard negatives, CE-filtered gold",
)
