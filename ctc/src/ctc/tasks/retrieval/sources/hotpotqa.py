"""
``hotpotqa`` -- multi-hop in-context retrieval, the multi-gold member of the ``retrieval`` family.

One question, **two** Wikipedia paragraphs that jointly answer it, and N-2 distractors of which a
tenth are the benchmark's own TF-IDF-retrieved ones. Graded on which document ids were named, not on
the answer text -- the same contract and the same grader as ``nq``, so a gap against ``nq`` is a
multi-hop effect rather than a scoring one.

Two-gold is not a detail of this ladder, it is the point of it. ``nq``, ``fiqa`` and ``scifact`` are
single-gold in practice, so ``hotpotqa`` is the only in-distribution ladder that exercises the
multi-gold instruction wording (:func:`ctc.tasks._retrieval.retrieval_instruction` switches on it)
and the multi-gold F1 path. Both gold documents survive every shrink, so the shorter rungs ask the
same two-hop question over a smaller corpus.

.. note::
   **``hard_frac`` defaults to 0.1 and the cross-encoder pass defaults on, matching ``nq``.** The
   retired all-hard-negative NQ regime is banned suite-wide and no ladder may reintroduce it by
   default. The supply here is different in kind, though: HotpotQA ships exactly 8 distractors per
   question, so past ~80 documents per example the realised hard fraction falls below 0.1 because
   the corpus runs out, not because the policy changed. See
   :mod:`ctc.data.sources.hotpotqa` for why topping it up from an external index is the wrong fix.
"""

from __future__ import annotations

import random
from typing import Dict, Optional

from ....data.generators.base import Generator
from ....data.sources import hotpotqa as source
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
    Build one HotpotQA multi-hop retrieval example.

    :param rng: Seeded RNG.
    :param index: Example counter; question *index* is used.
    :param corpus: Prepared HotpotQA pool.
    :param num_docs: Corpus size N (2 gold + N-2 distractors). Must be at least 2, or the example
        cannot hold its own gold.
    :param hard_frac: Fraction of the distractors that are the benchmark's own distractors, capped
        by the 8 a question actually has. See the module note before changing it.

    :returns: The example, with **0-based** ``gold_doc_indices`` holding both supporting paragraphs,
        or ``None`` when the pool is exhausted.
    """
    return build_from_pool(rng, index=index, corpus=corpus, num_docs=num_docs, hard_frac=hard_frac)


GENERATOR = Generator(
    name="hotpotqa",
    task="retrieval",
    source="hotpotqa",
    build_example=build_example,
    defaults={"num_docs": 20, "hard_frac": 0.1},
    corpus=source.load_pool,
    corpus_defaults={
        "split": "train",
        "num_questions": 25_000,
        "seed": 42,
        # Bridge only: every shipped HotpotQA file is bridge-only, and a comparison question names
        # both of its entities outright, so the ladder would measure lookup rather than hops.
        "question_type": "bridge",
        "level": "all",
        "filler_pool": 60_000,
        "ce_filter": True,
        "ce_model": None,
        "ce_margin": 0.0,
        "batch_size": 256,
    },
    indexed=True,
    notes="HotpotQA distractor, bridge questions; 2 gold/question, 10% benchmark hard negatives",
)
