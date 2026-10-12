"""
``fiqa`` and ``scifact`` -- the two held-out BEIR retrieval ladders.

Same contract and same grader as ``nq``, different corpora: finance forum posts and scientific
abstracts instead of Wikipedia passages, with gold from BEIR's own qrels rather than an
answer-string match. A gap against ``nq`` is therefore a *source* effect, which is the only thing
these ladders exist to measure.

Two generators, one module, because they differ only in which dataset they name -- and in how
badly an unfiltered BM25 pool would leak. SciFact is single-gold and densely judged; FiQA averages
~2.6 gold per query over a ~57k corpus, so its top BM25 hits are full of relevant-but-unjudged
passages and the cross-encoder margin filter is what makes its negatives valid at all.

.. note::
   ``eval_size`` for the shipped SciFact ladder is **299**, below the suite's 500 floor. Any number
   from it must be quoted with its size and error bar inline (about +/-0.027 at f1 0.7). Building
   a bigger one is not possible: 299 is every SciFact test query with a qrel.
"""

from __future__ import annotations

import random
from typing import Dict, Optional

from ....data.generators.base import Generator
from ....data.sources import beir as source
from ....data.sources.retrieval import RetrievalPool
from ..generate import build_from_pool

__all__ = ["build_example", "FIQA", "SCIFACT"]


def build_example(
    rng: random.Random,
    *,
    index: int,
    corpus: RetrievalPool,
    num_docs: int,
    hard_frac: float = 0.1,
) -> Optional[Dict]:
    """
    Build one BEIR retrieval example.

    :param rng: Seeded RNG.
    :param index: Example counter; query *index* is used.
    :param corpus: Prepared BEIR pool.
    :param num_docs: Corpus size N.
    :param hard_frac: Fraction of the distractors drawn from the CE-surviving BM25 candidates; the
        rest are random real corpus documents, which are format-identical so provenance is not
        visible from a document's shape.

    :returns: The example, or ``None`` when the pool is exhausted.
    """
    return build_from_pool(rng, index=index, corpus=corpus, num_docs=num_docs, hard_frac=hard_frac)


def _generator(name: str, dataset: str, notes: str) -> Generator:
    return Generator(
        name=name,
        task="retrieval",
        source=f"beir_{dataset}",
        build_example=build_example,
        defaults={"num_docs": 20, "hard_frac": 0.1},
        corpus=source.load_pool,
        corpus_defaults={
            "dataset": dataset,
            "split": "test",
            "seed": 42,
            "bm25_topk": 100,
            "ce_filter": True,
            "ce_model": None,
            "ce_margin": 3.0,
            "index_dir": "data/.cache/beir/_indices",
            "bm25_threads": 8,
        },
        indexed=True,
        # NOT eval_only: in the 22-task suite every row trains in-domain -- one model per
        # (task, arm), fiqa and scifact included (suite coverage record, 2026-08-20). Their
        # older role as OOD probes of the 5-task MIXED models still holds by convention:
        # never feed these shards to a multi-task mix you plan to score them against.
        notes=notes,
    )


FIQA = _generator(
    "fiqa", "fiqa", "held-out BEIR retrieval; ~2.6 gold/query, CE filtering is load-bearing"
)
SCIFACT = _generator(
    "scifact",
    "scifact",
    "held-out BEIR retrieval; single-gold, eval_size 299 (below the 500 floor)",
)
