"""
``outlier`` over Wikipedia 100-word passages -- one of the five in-distribution ladders.

Most passages come from a handful of articles; a few come from one nobody else shares. Article
titles are **never rendered**: a Wikipedia title states the topic outright, and the task would
reduce to reading K+1 titles.

This is the one ladder with its own :func:`nested_ladder`, and the reason is correctness rather
than convenience -- see the warning in :mod:`ctc.tasks.outlier.generate`.
"""

from __future__ import annotations

import random
from typing import Dict, List, Mapping, Optional, Sequence, Set

from ....data.generators.base import Generator
from ....data.sources import wiki100w as source
from ..generate import TOPIC_QUERY, assemble, partition_with_min, prefix_to_size

__all__ = ["build_example", "nested_ladder", "GENERATOR"]

SOURCE_TAG = "wiki_outlier_topic"


def _titles(run: Sequence[source.Chunk]) -> Set[str]:
    """
    :param run: A sampled run.

    :returns: **Every** article title the run touched. A composed run spans several articles, so
        excluding only ``run[0]``'s title lets a later "distinct topic" quietly reuse one of them.
    """
    return {chunk.title for chunk in run}


def build_example(
    rng: random.Random,
    *,
    corpus: source.ArticlePool,
    num_docs: int,
    num_outliers: int = 3,
    min_run: int = 4,
    max_run: int = 14,
) -> Optional[Dict]:
    """
    Build one Wikipedia outlier example: several majority topics plus one rarer outlier topic.

    :param rng: Seeded RNG.
    :param corpus: The article pool. **Drawn from, not consumed**: unlike a question pool, an
        article can back many examples, so this generator takes no example index.
    :param num_docs: Corpus size N.
    :param num_outliers: Documents in the outlier topic, K.
    :param min_run: Smallest majority topic. Raised to ``num_outliers + 1`` internally, because a
        majority topic that is no larger than the outlier makes the answer ambiguous.
    :param max_run: Largest majority topic.

    :returns: The example, or ``None`` when the pool could not supply distinct enough articles.

    :raises ValueError: If ``num_docs`` cannot hold the outlier plus one majority topic.
    """
    if num_docs <= num_outliers:
        raise ValueError(f"num_docs={num_docs} must exceed num_outliers={num_outliers}")
    outlier = corpus.sample_run(num_outliers, rng, single_only=True)
    if outlier is None:
        return None

    floor = num_outliers + 1
    used = _titles(outlier)
    remaining = num_docs - num_outliers
    topics = max(1, min(remaining // floor, max(1, remaining // max(min_run, floor))))
    counts = partition_with_min(remaining, topics, floor, rng)
    if not counts:
        return None

    majority: List[source.Chunk] = []
    for count in counts:
        run = corpus.sample_run(count, rng, exclude_titles=used)
        if run is None:
            return None
        used |= _titles(run)
        majority.extend(run)

    return assemble(
        [(c.title, c.body) for c in majority],
        [(c.title, c.body) for c in outlier],
        source=SOURCE_TAG,
        query=TOPIC_QUERY,
        rng=rng,
        titles=False,
        meta={
            "minority_label": outlier[0].title,
            "num_outliers": num_outliers,
            "num_categories": len({c.title for c in majority}) + 1,
            "mode": "mixed",
        },
    )


def nested_ladder(
    rng: random.Random,
    *,
    index: int,
    corpus: source.ArticlePool,
    rungs: Mapping[str, int],
    num_docs: int,
    num_outliers: int = 3,
    min_run: int = 4,
    max_run: int = 14,
) -> Optional[Dict[str, Dict]]:
    """
    Build one example at every rung at once, keeping the outlier fixed and growing the majority.

    The generic nested shrink drops *random* distractors, which can shrink a majority topic below
    the outlier count and destroy the "outlier is the uniquely rarest topic" invariant. Here
    instead:

    * one outlier article is sampled and **fixed across every rung**, so the question and its
      answer text are identical rung to rung;
    * an ordered list of single-article majority runs is sampled up to the largest rung;
    * each shorter rung is a **chunk-prefix** of that list, backed off so no partial trailing topic
      is left smaller than the outlier.

    :param rng: Seeded RNG for this row.
    :param index: Example counter, for the caller's bookkeeping.
    :param corpus: The article pool.
    :param rungs: ``label -> documents``, any order.
    :param num_docs: Ignored; the rung table supplies the sizes. Present so the ladder builder can
        pass one resolved config to both entry points.
    :param num_outliers: Documents in the outlier topic.
    :param min_run: Smallest majority topic; raised to ``num_outliers + 1``.
    :param max_run: Largest majority topic.

    :returns: ``{rung label: example}``, or ``None`` when the pool could not supply the largest
        rung.
    """
    del num_docs  # the rung table is the authority here
    largest = max(rungs.values())
    floor = num_outliers + 1
    outlier = corpus.sample_run(num_outliers, rng, single_only=True)
    if outlier is None:
        return None

    runs: List[List[source.Chunk]] = []
    used = _titles(outlier)
    placed, attempts = 0, 0
    while placed < largest - num_outliers and attempts < 400:
        attempts += 1
        run = corpus.sample_run(
            rng.randint(max(min_run, floor), max(max_run, floor)),
            rng,
            exclude_titles=used,
            single_only=True,
        )
        if run is None:
            continue
        used |= _titles(run)
        runs.append(run)
        placed += len(run)
    if placed < largest - num_outliers:
        return None

    out: Dict[str, Dict] = {}
    for label, size in rungs.items():
        majority = prefix_to_size(runs, size - num_outliers, floor)
        # One stream per rung, so adding a rung never perturbs another rung's ordering.
        rung_rng = random.Random(f"{index}:outlier:{label}")
        out[label] = assemble(
            [(c.title, c.body) for c in majority],
            [(c.title, c.body) for c in outlier],
            source=SOURCE_TAG,
            query=TOPIC_QUERY,
            rng=rung_rng,
            titles=False,
            meta={
                "minority_label": outlier[0].title,
                "num_outliers": num_outliers,
                "num_categories": len({c.title for c in majority}) + 1,
                "mode": "mixed",
                "rung": label,
            },
        )
    return out


GENERATOR = Generator(
    name="outlier",
    task="outlier",
    source=SOURCE_TAG,
    build_example=build_example,
    defaults={"num_docs": 22, "num_outliers": 3, "min_run": 4, "max_run": 14},
    corpus=source.load_pool,
    corpus_defaults={
        "cache": "data/wiki100w_article_pool.pkl",
        "build": True,
        "min_article_chunks": 4,
        "shards": 8,
    },
    shrink_safe=False,  # see nested_ladder: random shrink can break the scale-K invariant
    build_ladder=nested_ladder,
    notes="wiki100w passages; scale-K ladder keeps the outlier the uniquely rarest topic",
)
