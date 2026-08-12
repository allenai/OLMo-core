"""
The ``outlier`` data contract: mostly-one-topic corpora with a few items that are not.

Two corpora, one contract -- Wikipedia passages (in-distribution) and Amazon reviews (held out) --
so the placement, the gold shape and the answer string live here and only the sampling differs.

.. warning::
   **The outlier must stay the uniquely rarest group at every rung, and the generic shrink breaks
   that.** Dropping random distractors to make a shorter rung can leave a majority topic with fewer
   documents than the outlier, at which point "the odd one out" has two answers and only one is
   labelled. This is not hypothetical -- it is why the pre-migration suite grew a second, outlier-
   specific ladder builder after the generic one produced rungs whose gold was ambiguous.

   The fix, carried over here as :func:`nested_ladder`: fix each example's outlier article across
   every rung and grow only the *majority*, as an ordered list of single-topic runs whose prefixes
   are the shorter rungs -- backing off so the last, partial topic never ends up smaller than the
   outlier. Every rung then grades the same question over a nested corpus **and** keeps the
   invariant.

**Gold is 0-based** for this task, and the ``answers`` string is 1-based metadata (the prompt
renders 1-based ids). Both are the pre-migration convention and both are checked against the spec.
"""

from __future__ import annotations

import random
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ...data.schema import make_document, make_example

__all__ = ["TOPIC_QUERY", "partition_with_min", "assemble", "prefix_to_size"]

#: The query every outlier example carries. Fixed text: it is part of the trained prompt format.
TOPIC_QUERY = (
    "Can you find passages that are about a different topic than the rest of these passages?"
)


def partition_with_min(total: int, parts: int, minimum: int, rng: random.Random) -> List[int]:
    """
    Split ``total`` into ``parts`` counts, each at least ``minimum``, with uneven proportions.

    Exponential weights (approximately Dirichlet with alpha=1) rather than an even split, so the
    majority distribution varies markedly across examples. An even split would make "the smallest
    group" recoverable from the *shape* of the counts alone, without reading any passage.

    :param total: Documents to distribute.
    :param parts: How many groups.
    :param minimum: Floor per group -- ``num_outliers + 1``, which is what keeps the outlier the
        uniquely smallest group.
    :param rng: Seeded RNG.

    :returns: The counts, or ``[]`` when ``parts * minimum`` exceeds ``total``.
    """
    remainder = total - parts * minimum
    if remainder < 0 or parts <= 0:
        return []
    weights = [rng.expovariate(1.0) for _ in range(parts)]
    total_weight = sum(weights) or 1.0
    counts = [int(remainder * w / total_weight) for w in weights]
    for i in range(remainder - sum(counts)):
        counts[i % parts] += 1
    return [c + minimum for c in counts]


def assemble(
    majority: Sequence[Tuple[str, str]],
    outliers: Sequence[Tuple[str, str]],
    *,
    source: str,
    query: str,
    rng: random.Random,
    meta: Optional[Mapping[str, object]] = None,
    titles: bool = False,
) -> Dict:
    """
    Shuffle majority and outlier items together and record where the outliers landed.

    :param majority: ``(title, text)`` items sharing the majority attribute.
    :param outliers: ``(title, text)`` items that do not.
    :param source: Corpus tag.
    :param query: The task query.
    :param rng: Seeded RNG. Consumes exactly one shuffle.
    :param meta: Per-example metadata.
    :param titles: Render the item title as the document ``title``. **Off for Wikipedia**, where
        the article title names the topic and would hand over the answer without reading anything;
        on for reviews, whose headline is part of the review body a human would read.

    :returns: A unified-format example with **0-based** ``gold_doc_indices`` and a 1-based
        ``answers`` string.
    """
    items = [(item, False) for item in majority] + [(item, True) for item in outliers]
    rng.shuffle(items)
    gold = [i for i, (_, is_outlier) in enumerate(items) if is_outlier]
    documents = [make_document(text, title=title if titles else None) for (title, text), _ in items]
    return make_example(
        documents=documents,
        queries=[query],
        # Metadata only -- the graded target is rebuilt from gold_doc_indices by the spec. Kept
        # because every shipped outlier file has it and a diff against one would otherwise show
        # a spurious difference.
        answers=["; ".join(str(g + 1) for g in gold)],
        source=source,
        gold=gold,
        meta=dict(meta or {}),
    )


def prefix_to_size(runs: Sequence[Sequence], target: int, minimum: int) -> List:
    """
    Take a nested prefix of ordered single-topic runs, never ending on an undersized topic.

    :param runs: Majority runs, in a fixed order. Each is one topic.
    :param target: Documents wanted.
    :param minimum: Smallest a topic may be and still outrank the outlier
        (``num_outliers + 1``).

    :returns: The prefix. When it would end part-way through a topic with fewer than ``minimum``
        of that topic's documents, it backs off to that topic's start and drops the undersized
        tail -- otherwise the shorter rung would contain a group *smaller* than the outlier, which
        is the ambiguity this whole construction exists to prevent.
    """
    flat: List = []
    owner: List[int] = []
    for run_index, run in enumerate(runs):
        for item in run:
            flat.append(item)
            owner.append(run_index)

    size = min(target, len(flat))
    if size == 0:
        return []
    last = owner[size - 1]
    start = owner.index(last)
    if 0 < size - start < minimum:
        size = start
    return flat[:size]
