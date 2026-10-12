"""
``groups4`` data: N arithmetic expressions, find every tight cluster of G of them.

A group is gold when all G answers lie within X of each other. This is the G-tuple generalisation of
:mod:`ctc.tasks.mathmatch` (which finds pairs). A literal subset-*sum* property cannot be made
provably unique without astronomical numbers or a brute-force scan of all C(N,G) subsets, so the
task uses the all-pairwise-within-X property, which is provably unique with small values.

**Exactly K groups, guaranteed.** K cluster centres are spaced more than 2X apart so clusters cannot
merge or touch, and each gold group is G distinct values inside a span of X around its centre.

.. warning::
   **Distractors are in-distribution noise, not isolated singletons.** An earlier version placed
   every distractor more than X from everything, so a single close pair was a 100%-reliable,
   trivially cheap gold detector at any N -- the task was solvable without ever looking for a
   G-clique (``records/ctc-setting-verification-2026-07-23.md``). Now a distractor may land within X
   of up to G-2 existing values, so near-duplicate pairs and triples occur among distractors too.
   That stays exact: for a 1-D set "all pairwise distances <= X" is equivalent to "fits in one span
   of width X", so capping each point's within-X neighbour count at G-2 before insertion caps every
   span at G-1 after it, and no accidental G-clique can form.
   :func:`ctc.data.audit.closest_pair_is_gold` guards this on every build.
"""

from __future__ import annotations

import bisect
import random
from typing import Dict, List, Optional, Tuple

from ...data.generators.base import Generator
from ...data.gold import remap_groups, shuffle_with_remap
from ...data.schema import make_document, make_example
from ...data.synthetic import auto_value_range, build_expression

__all__ = ["build_example", "sample_values", "GENERATOR"]


def sample_values(
    n_docs: int,
    n_groups: int,
    group_size: int,
    tol: int,
    ans_min: int,
    ans_max: int,
    rng: random.Random,
    max_attempts: int = 400,
) -> Tuple[List[int], List[List[int]]]:
    """
    Draw the answer values: exactly ``n_groups`` all-within-``tol`` clusters and nothing else.

    :param n_docs: Total values N.
    :param n_groups: Gold clusters K.
    :param group_size: Values per cluster G.
    :param tol: Cluster width X.
    :param ans_min: Smallest answer value.
    :param ans_max: Largest answer value.
    :param rng: Seeded RNG.
    :param max_attempts: Restarts before giving up.

    :returns: ``(values, gold_index_groups)`` in construction order.

    :raises RuntimeError: If no assignment was found -- widen the value range, or lower
        ``n_docs``/``tol``/``group_size``.
    """
    G, K = group_size, n_groups
    for _ in range(max_attempts):
        spacing = 2 * tol + 1
        candidates = list(range(ans_min + tol, ans_max - tol + 1))
        rng.shuffle(candidates)
        centers: List[int] = []
        for c in candidates:
            if all(abs(c - other) > spacing for other in centers):
                centers.append(c)
                if len(centers) == K:
                    break
        if len(centers) < K:
            continue

        groups = []
        for c in centers:
            window = list(range(c, c + tol + 1))
            if len(window) < G:
                break
            groups.append(rng.sample(window, G))
        if len(groups) < K:
            continue
        gold_values = [v for g in groups for v in g]
        if len(set(gold_values)) != len(gold_values):
            continue  # windows collided; retry

        # `chosen` stays sorted so the within-tol neighbour count is a bisect, not a scan.
        chosen = sorted(gold_values)
        distractors: List[int] = []
        for _slot in range(n_docs - K * G):
            for _try in range(300):
                if chosen and rng.random() < 0.5:
                    # Biased draw near an existing value, so local density is realistic rather
                    # than only appearing by pigeonhole at small N.
                    anchor = rng.choice(chosen)
                    offset = rng.randint(1, 3 * tol)
                    v = anchor + offset if rng.random() < 0.5 else anchor - offset
                else:
                    v = rng.randint(ans_min, ans_max)
                if not ans_min <= v <= ans_max:
                    continue
                at = bisect.bisect_left(chosen, v)
                if at < len(chosen) and chosen[at] == v:
                    continue  # every value must be distinct
                lo = bisect.bisect_left(chosen, v - tol)
                hi = bisect.bisect_right(chosen, v + tol)
                if hi - lo <= G - 2:
                    bisect.insort(chosen, v)
                    distractors.append(v)
                    break
            else:
                break
        if len(distractors) < n_docs - K * G:
            continue

        return gold_values + distractors, [list(range(i * G, i * G + G)) for i in range(K)]

    raise RuntimeError(
        f"could not assemble values after {max_attempts} attempts; widen [{ans_min}, {ans_max}] "
        f"or lower num_docs/tolerance/group_size"
    )


def build_example(
    rng: random.Random,
    *,
    num_docs: int,
    num_groups: int = 1,
    group_size: int = 4,
    tolerance: int = 5,
    ans_min: Optional[int] = None,
    ans_max: Optional[int] = None,
    numbers_only: bool = False,
) -> Dict:
    """
    Build one groups4 example.

    :param rng: Seeded RNG.
    :param num_docs: Corpus size N.
    :param num_groups: Gold clusters K.
    :param group_size: Expressions per cluster G; larger is harder (N^G).
    :param tolerance: Max pairwise difference within a cluster X.
    :param ans_min: Smallest answer value; defaults to a range scaled to ``num_docs`` and to how
        densely distractors may pack (see :func:`~ctc.data.synthetic.auto_value_range`).
    :param ans_max: Largest answer value; same default.
    :param numbers_only: Render bare numbers instead of arithmetic, isolating the clustering task
        from per-document arithmetic.

    :returns: A unified-format example whose ``gold_doc_indices`` holds one 1-based id list per
        cluster.
    """
    if ans_min is None or ans_max is None:
        span = auto_value_range(num_docs, tolerance, per_window=group_size - 1)
        ans_min, ans_max = -span, span
    values, gold_groups = sample_values(
        num_docs, num_groups, group_size, tolerance, ans_min, ans_max, rng
    )
    texts = [str(v) for v in values] if numbers_only else [build_expression(v, rng) for v in values]
    shuffled, old_to_new = shuffle_with_remap(texts, rng=rng, base=1)
    query = (
        f"Find every group of {group_size} expressions whose answers are all within "
        f"{tolerance} of each other (every pair in the group differs by at most {tolerance})."
    )
    return make_example(
        documents=[make_document(t) for t in shuffled],
        queries=[query],
        source="groups4",
        gold=remap_groups(gold_groups, old_to_new),
        meta={
            "group_size": group_size,
            "num_groups": num_groups,
            "tolerance": tolerance,
            "numbers_only": numbers_only,
        },
    )


GENERATOR = Generator(
    name="groups4",
    task="groups4",
    source="groups4",
    build_example=build_example,
    defaults={
        "num_docs": 100,
        "num_groups": 1,
        "group_size": 4,
        "tolerance": 5,
        "ans_min": None,
        "ans_max": None,
        "numbers_only": False,
    },
    notes="pure synthetic; hardest task in the suite (F1 ~0.14 at n=20 even with sort-CoT)",
)
