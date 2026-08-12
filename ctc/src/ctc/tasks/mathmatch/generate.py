"""
``mathmatch`` data: N arithmetic expressions, find every pair whose answers differ by at most X.

The pair-valued sibling of :mod:`ctc.tasks.groups4`, and structurally the synthetic stand-in for
contradiction: same gold shape (``[[a, b], ...]``, 1-based), same prompt builder, same parser, same
metrics -- but with a corpus that costs nothing to build and a gold set that is exact by
construction rather than by an LLM's judgement.

**Exactly K pairs, guaranteed.** K base values are drawn more than 3X apart, giving each a safe
halo; each gets a partner within X; every remaining value is placed more than X from everything.
So the K planted pairs are the only within-X pairs.

Note this is the *strict* construction that :mod:`ctc.tasks.groups4` deliberately abandoned:
distractors here really are isolated, so "the two closest values" is a reliable gold detector for
K=1. That is acceptable because mathmatch's difficulty is meant to come from N and from evaluating
the arithmetic, not from decoy density -- but it means the closest-pair shortcut probe is expected
to fire on this task, and :mod:`ctc.data.audit.shortcuts` records it as a known property rather
than a finding.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from ...data.generators.base import Generator
from ...data.gold import remap_groups, shuffle_with_remap
from ...data.schema import make_document, make_example
from ...data.synthetic import auto_value_range, build_expression

__all__ = ["build_example", "sample_answers", "GENERATOR"]


def sample_answers(
    n_docs: int,
    n_pairs: int,
    tol: int,
    ans_min: int,
    ans_max: int,
    rng: random.Random,
    max_attempts: int = 200,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Draw the answer values: exactly ``n_pairs`` within-``tol`` pairs and nothing else.

    :param n_docs: Total values N.
    :param n_pairs: Gold pairs K.
    :param tol: Closeness threshold X.
    :param ans_min: Smallest answer value.
    :param ans_max: Largest answer value.
    :param rng: Seeded RNG.
    :param max_attempts: Restarts before giving up.

    :returns: ``(values, gold_pairs)`` as construction-order index pairs.

    :raises RuntimeError: If no assignment was found -- widen the value range or lower
        ``n_docs``/``n_pairs``.
    """
    for _ in range(max_attempts):
        spacing = max(1, 3 * tol + 1)
        candidates = list(range(ans_min, ans_max + 1))
        rng.shuffle(candidates)
        bases: List[int] = []
        for v in candidates:
            if all(abs(v - b) >= spacing for b in bases):
                bases.append(v)
                if len(bases) == n_pairs:
                    break
        if len(bases) < n_pairs:
            continue

        partners: List[int] = []
        for b in bases:
            offsets = [d for d in range(-tol, tol + 1) if d != 0]
            rng.shuffle(offsets)
            partner = next((b + d for d in offsets if ans_min <= b + d <= ans_max), None)
            if partner is None:
                break
            partners.append(partner)
        if len(partners) < n_pairs:
            continue

        paired = bases + partners
        distractors: List[int] = []
        pool = list(range(ans_min, ans_max + 1))
        rng.shuffle(pool)
        for v in pool:
            if all(abs(v - other) > tol for other in paired + distractors):
                distractors.append(v)
                if len(distractors) == n_docs - 2 * n_pairs:
                    break
        if len(distractors) < n_docs - 2 * n_pairs:
            continue

        return paired + distractors, [(i, n_pairs + i) for i in range(n_pairs)]

    raise RuntimeError(
        f"could not assemble answers after {max_attempts} attempts; widen [{ans_min}, {ans_max}] "
        f"or lower num_docs/num_pairs"
    )


def build_example(
    rng: random.Random,
    *,
    num_docs: int,
    num_pairs: int = 3,
    tolerance: int = 2,
    ans_min: Optional[int] = None,
    ans_max: Optional[int] = None,
    numbers_only: bool = False,
) -> Dict:
    """
    Build one mathmatch example.

    :param rng: Seeded RNG.
    :param num_docs: Corpus size N.
    :param num_pairs: Gold pairs K.
    :param tolerance: Closeness threshold X.
    :param ans_min: Smallest answer value; defaults to a range scaled to ``num_docs`` (see
        :func:`~ctc.data.synthetic.auto_value_range`). A fixed range cannot hold a ladder's worth
        of mutually-distant values.
    :param ans_max: Largest answer value; same default.
    :param numbers_only: Render bare values instead of arithmetic, isolating matching from
        per-document arithmetic. The query text is unchanged either way.

    :returns: A unified-format example whose ``gold_doc_indices`` holds 1-based pairs.
    """
    if ans_min is None or ans_max is None:
        span = auto_value_range(num_docs, tolerance)
        ans_min, ans_max = -span, span
    values, gold_pairs = sample_answers(num_docs, num_pairs, tolerance, ans_min, ans_max, rng)
    texts = [str(v) for v in values] if numbers_only else [build_expression(v, rng) for v in values]
    items, old_to_new = shuffle_with_remap(list(zip(texts, values)), rng=rng, base=1)
    return make_example(
        documents=[make_document(text) for text, _ in items],
        queries=[f"Find all pairs of expressions whose answers differ by at most {tolerance}."],
        source="synth_mathmatch",
        gold=remap_groups(gold_pairs, old_to_new),
        meta={"tolerance": tolerance, "answer_values": [v for _, v in items]},
    )


GENERATOR = Generator(
    name="mathmatch",
    task="mathmatch",
    source="synth_mathmatch",
    build_example=build_example,
    defaults={
        "num_docs": 20,
        "num_pairs": 3,
        "tolerance": 2,
        "ans_min": None,
        "ans_max": None,
        "numbers_only": False,
    },
    notes="pure synthetic; the no-LLM stand-in for contradiction's gold shape",
)
