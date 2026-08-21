"""
``outlier_review`` -- the held-out outlier ladder. Amazon reviews instead of Wikipedia passages.

Same contract, same grader, difficulty-matched to the in-distribution ladder so the gap it measures
is a *source* effect rather than a difficulty effect. Two example kinds:

* **category** -- the majority come from one product category, the outliers from another. The build
  the shipped ``outlier_review_matched_*`` ladder used, and the default here.
* **rating** -- the majority share one star rating and the outliers have a different one. Harder
  and noisier (a 4-star and a 5-star review read almost alike), and not what the grid measured.

Ground truth is structured metadata only -- the HF category and the ``rating`` field -- so there is
no labelling model and no label noise.

.. note::
   Two documented departures from the shipped files, both improvements, both worth knowing before
   diffing against one:

   * the shipped ladder's four rungs were generated **independently**, so its rungs grade different
     questions; this generator uses the shared nested shrink, which they do not.
   * ``meta`` is normalised to ``_meta`` by :mod:`ctc.data.schema` -- the outlier generators were
     among the four files that wrote the bare spelling.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from ....data.generators.base import Generator
from ....data.sources import amazon as source
from ..generate import assemble, partition_with_min

__all__ = ["CATEGORY_QUERY", "RATING_QUERY", "build_example", "GENERATOR"]

CATEGORY_QUERY = "Can you find outliers from the least common category?"
RATING_QUERY = "Can you find outliers with the least common rating in this data?"


def _build_category(
    rng: random.Random,
    pool: source.ReviewPool,
    num_docs: int,
    num_outliers: int,
    max_categories: int,
) -> Optional[Dict]:
    floor = num_outliers + 1
    remaining = num_docs - num_outliers
    eligible = [c for c in pool.categories if len(pool.by_category[c]) >= max(num_outliers, floor)]
    if len(eligible) < 2:
        return None
    topics = max(1, min(max_categories - 1, remaining // floor))
    if topics > len(eligible) - 1:
        topics = len(eligible) - 1
    if topics < 1:
        return None

    outlier_category = rng.choice(eligible)
    majority_categories = rng.sample([c for c in eligible if c != outlier_category], topics)
    counts = partition_with_min(remaining, topics, floor, rng)
    if not counts:
        return None

    majority: List[source.Review] = []
    for category, count in zip(majority_categories, counts):
        if len(pool.by_category[category]) < count:
            return None
        majority.extend(rng.sample(pool.by_category[category], count))
    outliers = rng.sample(pool.by_category[outlier_category], num_outliers)

    distribution = dict(zip(majority_categories, counts))
    distribution[outlier_category] = num_outliers
    return assemble(
        [(r.title, r.text) for r in majority],
        [(r.title, r.text) for r in outliers],
        source="review_outlier_category",
        query=CATEGORY_QUERY,
        rng=rng,
        titles=True,
        meta={
            "minority_label": outlier_category,
            "num_outliers": num_outliers,
            "num_categories": len(majority_categories) + 1,
            "category_distribution": distribution,
        },
    )


def _build_rating(
    rng: random.Random, pool: source.ReviewPool, num_docs: int, num_outliers: int
) -> Optional[Dict]:
    category = rng.choice(pool.categories)
    by_rating: Dict[int, List[source.Review]] = {}
    for review in pool.by_category[category]:
        by_rating.setdefault(review.rating, []).append(review)

    remaining = num_docs - num_outliers
    majority_options = [r for r in source.VALID_RATINGS if len(by_rating.get(r, [])) >= remaining]
    if not majority_options:
        return None
    majority_rating = rng.choice(majority_options)
    minority_options = [
        r
        for r in source.VALID_RATINGS
        if r != majority_rating and len(by_rating.get(r, [])) >= num_outliers
    ]
    if not minority_options:
        return None
    minority_rating = rng.choice(minority_options)

    majority = rng.sample(by_rating[majority_rating], remaining)
    outliers = rng.sample(by_rating[minority_rating], num_outliers)
    return assemble(
        [(r.title, r.text) for r in majority],
        [(r.title, r.text) for r in outliers],
        source="review_outlier_rating",
        query=RATING_QUERY,
        rng=rng,
        titles=True,
        meta={
            "majority_label": majority_rating,
            "minority_label": minority_rating,
            "num_outliers": num_outliers,
            "category": category,
        },
    )


def build_example(
    rng: random.Random,
    *,
    corpus: source.ReviewPool,
    num_docs: int,
    num_outliers: int = 3,
    rating_ratio: float = 0.0,
    max_categories: int = 2,
) -> Optional[Dict]:
    """
    Build one review-outlier example.

    :param rng: Seeded RNG.
    :param corpus: The review pool. Drawn from, not consumed, so this generator takes no example
        index.
    :param num_docs: Corpus size N.
    :param num_outliers: Outlier documents, K.
    :param rating_ratio: Fraction of examples built as rating-skew rather than category-skew.
        **Defaults to 0.0**, which is the shipped difficulty-matched build; the pre-migration
        default of 0.5 mixes in a second, harder task.
    :param max_categories: Distinct categories per example. 2 -- one majority plus the outlier --
        is the shipped build; higher uses the uneven multi-majority partition.

    :returns: The example, or ``None`` when the pools were too thin.

    :raises ValueError: If ``num_docs`` cannot hold the outliers.
    """
    if num_docs <= num_outliers:
        raise ValueError(f"num_docs={num_docs} must exceed num_outliers={num_outliers}")
    if rating_ratio > 0 and rng.random() < rating_ratio:
        return _build_rating(rng, corpus, num_docs, num_outliers)
    return _build_category(rng, corpus, num_docs, num_outliers, max_categories)


GENERATOR = Generator(
    name="outlier_review",
    task="outlier",
    source="review_outlier_category",
    build_example=build_example,
    defaults={
        "num_docs": 30,
        "num_outliers": 3,
        "rating_ratio": 0.0,
        "max_categories": 2,
    },
    corpus=source.load_pool,
    corpus_defaults={
        "categories": source.DEFAULT_CATEGORIES,
        "pool_size": 20_000,
        "min_chars": 120,
        "max_chars": 1500,
        "seed": 42,
        "hf_dataset": "McAuley-Lab/Amazon-Reviews-2023",
    },
    # NOT eval_only: a 22-task-suite row trains in-domain (one model per task x arm). The OOD
    # convention -- never train the 5-task MIXED models on this -- is a protocol rule for that
    # family, not a property of the ladder.
    notes="held-out: Amazon reviews graded by the outlier spec; difficulty-matched `matched` build",
)
