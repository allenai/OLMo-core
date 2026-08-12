"""
The ``grouping_labeled`` data contract: N abstracts, a requested K, one gold partition.

Granularity is set by which OpenAlex concept **level** supplies the gold: L0 is ~19 top-level
fields (coarse, few large groups), L3 is thousands of narrow concepts (fine, many small ones). An
example picks a level, samples K distinct concept values at it, and draws abstracts from each.

**Gold is ``gold_doc_indices``, 0-based, and it is a list of groups** -- one sub-list per cluster,
covering every document exactly once. ``cluster_labels`` runs parallel to it and is *not* scored;
it exists so the labeled target can name each group.

.. warning::
   **The level/k-density confound (port record trap 14) is fixed here, and the fix has three
   parts.** Picking K as a plain fraction of ``n_docs`` ignores how many distinct concept values a
   level even has. OpenAlex L0 has 19 -- a ceiling **more data cannot raise** -- so as the ladder
   grew, coarse levels started asking for infeasible partitions, ``build_example`` returned
   ``None``, and the outer retry loop quietly refilled the quota from whichever level was still
   easy. The realised level mix then drifted with N, conflating "more documents" with "finer
   grouping" along the exact axis the ladder varies.

   1. :func:`sample_k_for_level` clamps the fraction-derived range to what is *provably buildable*
      against the actual index: K cannot exceed the number of distinct values, and cannot fall
      below the smallest K whose biggest pools jointly cover ``n_docs``.
   2. :func:`build_example` chooses values and partition sizes capacity-aware -- biased toward
      larger pools, then redistributing overflow -- instead of drawing a random partition and
      rejecting it whenever a large slice landed on a small pool.
   3. The level is a deterministic function of the **example index**, not a per-attempt RNG draw.
      That is this port's form of the pre-migration fixed per-level quota, and it is stronger: with
      the level pinned by the index, a level's accept rate cannot influence how often it appears.

.. warning::
   **No grouping ladder can be nested.** The gold partition covers every document, so the generic
   shrink -- which keeps gold and drops distractors -- has nothing to drop, and a partition over
   fewer abstracts is a different partition. ``BUILD_MATRIX.md`` row 13 filed the nested mode as
   ACTION A13 and it was never built; this generator declares ``shrink_safe=False``, which routes
   it to independent per-rung draws, and the build report says so. Rung-to-rung deltas carry
   eval-set resampling noise on top of the length effect.
"""

from __future__ import annotations

import random
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ...data.gold import remap, shuffle_with_remap
from ...data.schema import make_document, make_example
from ...data.sources.openalex import OpenAlexPool, Paper
from .._grouping import build_labeled_target

__all__ = [
    "SOURCE_TAG",
    "K_FRACTION_PER_LEVEL",
    "QUERY_PHRASINGS",
    "sample_partition_sizes",
    "min_k_by_capacity",
    "sample_k_for_level",
    "choose_group_values",
    "partition_with_capacity",
    "assemble",
]

#: Corpus tag. Constant across levels on purpose: the pre-migration files wrote
#: ``openalex_grouping_train`` / ``openalex_grouping_eval``, i.e. they encoded the *split* in the
#: one field that is supposed to name the corpus, so a mixed file could not be separated by source
#: at all. The level is carried in ``level`` and the split by which file a row is in.
SOURCE_TAG = "openalex_grouping"

#: Per-level K range as ``(min, max)`` fractions of ``n_docs``, verbatim from the pre-migration
#: generator. Coarser level -> fewer, larger groups. These are a *target*; what is buildable is
#: decided against the real index by :func:`sample_k_for_level`.
K_FRACTION_PER_LEVEL: Dict[int, Tuple[float, float]] = {
    0: (0.05, 0.15),
    1: (0.10, 0.30),
    2: (0.10, 0.85),
    3: (0.25, 0.95),
}

#: The three query phrasings, verbatim. Varied so a model cannot key on one exact sentence; the
#: requested K is the only part that carries information.
QUERY_PHRASINGS: Tuple[str, ...] = (
    "Group these documents into {k} groups.",
    "Partition the following abstracts into {k} categories.",
    "Cluster these {n} papers into {k} groups.",
)


def sample_partition_sizes(
    total: int, k: int, rng: random.Random, *, min_per: int = 1
) -> List[int]:
    """
    Random partition of ``total`` into ``k`` parts, each at least ``min_per``.

    :param total: Documents to hand out.
    :param k: Groups.
    :param rng: Seeded RNG.
    :param min_per: Floor per group.

    :returns: The sizes, shuffled so the floor's placement carries no order information.

    :raises ValueError: If ``total`` cannot give every group its floor.
    """
    if total < k * min_per:
        raise ValueError(f"total={total} cannot give {k} group(s) {min_per} document(s) each")
    sizes = [min_per] * k
    for _ in range(total - k * min_per):
        sizes[rng.randrange(k)] += 1
    rng.shuffle(sizes)
    return sizes


def min_k_by_capacity(pool_sizes_desc: Sequence[int], n_docs: int) -> int:
    """
    Smallest K whose ``K`` biggest concept pools can jointly hold ``n_docs`` documents.

    A **lower** bound on K, not an upper one, and the direction is the counter-intuitive part:
    ``sum(top-K)`` only grows with K, so once K is large enough every larger K stays feasible. The
    binding failure at coarse levels is therefore *too few groups* -- their pools, even the biggest
    ones, cannot jointly supply ``n_docs`` -- rather than too many.

    :param pool_sizes_desc: Concept-pool sizes, descending.
    :param n_docs: Documents the example needs.

    :returns: That K, or ``0`` when every eligible value combined still falls short.
    """
    cumulative = 0
    for k, size in enumerate(pool_sizes_desc, start=1):
        cumulative += size
        if cumulative >= n_docs:
            return k
    return 0


def sample_k_for_level(
    level: int,
    n_docs: int,
    rng: random.Random,
    level_index: Mapping[str, Sequence[Paper]],
    *,
    k_fraction: Mapping[int, Tuple[float, float]] = K_FRACTION_PER_LEVEL,
    min_per_group: int = 1,
) -> Optional[int]:
    """
    Draw the number of groups for one level, clamped to what its index can actually build.

    :param level: Concept level.
    :param n_docs: Documents per example.
    :param rng: Seeded RNG.
    :param level_index: This level's concept value -> papers.
    :param k_fraction: Per-level ``(min, max)`` fractions of ``n_docs``.
    :param min_per_group: Smallest allowed group.

    :returns: K, or ``None`` when the level is infeasible at this ``n_docs`` -- which callers must
        treat as "skip this draw", never as "retry until some other level works". See the module
        warning.
    """
    low_fraction, high_fraction = k_fraction[level]
    low = max(2, round(low_fraction * n_docs))
    high = max(low, round(high_fraction * n_docs))
    high = min(high, n_docs // min_per_group)

    pool_sizes = sorted((len(papers) for papers in level_index.values()), reverse=True)
    eligible = len(pool_sizes)
    required = min_k_by_capacity(pool_sizes, n_docs)
    if required == 0 or required > eligible:
        return None
    feasible_low, feasible_high = max(low, required), min(high, eligible)
    if feasible_low > feasible_high:
        # The fraction target and feasibility do not overlap, which only happens on a very thin
        # index. Prefer the smallest feasible K over dropping the level from the mix entirely --
        # dropping it is the drift this whole function exists to stop.
        feasible_low = feasible_high = required
    return rng.randint(feasible_low, feasible_high)


def choose_group_values(
    level_index: Mapping[str, Sequence[Paper]],
    k: int,
    n_docs: int,
    rng: random.Random,
    *,
    diversity_multiplier: int = 3,
) -> Optional[List[str]]:
    """
    Pick ``k`` distinct concept values whose pools jointly hold ``n_docs`` documents.

    Weighted toward larger pools so the draw is very likely feasible first time -- matching
    :func:`sample_k_for_level`'s clamp -- while still varying example to example.

    :param level_index: Concept value -> papers.
    :param k: Groups wanted.
    :param n_docs: Documents the example needs.
    :param rng: Seeded RNG.
    :param diversity_multiplier: Candidates considered, as a multiple of ``k``.

    :returns: The chosen values, or ``None`` when the level has fewer than ``k`` values.
    """
    by_size = sorted(level_index, key=lambda value: -len(level_index[value]))
    if len(by_size) < k:
        return None
    candidates = by_size[: min(len(by_size), max(k * diversity_multiplier, k + 10))]
    # `chosen` is an order-preserving LIST, not a set. `pool.pop(i)` already prevents duplicates,
    # and a set's iteration order depends on PYTHONHASHSEED, which would make the output
    # non-deterministic across processes despite a fixed seed -- the group order feeds back into
    # how many further rng calls happen and in what order.
    chosen: List[str] = []
    pool = [(value, len(level_index[value])) for value in candidates]
    for _ in range(k):
        if not pool:
            break
        threshold = rng.uniform(0, sum(weight for _, weight in pool))
        accumulated = 0.0
        for i, (value, weight) in enumerate(pool):
            accumulated += weight
            if accumulated >= threshold:
                chosen.append(value)
                pool.pop(i)
                break
        else:
            chosen.append(pool.pop()[0])
    if sum(len(level_index[value]) for value in chosen) >= n_docs:
        return chosen
    # The weighted draw came up short near the feasibility boundary; fall back to the
    # deterministic top-k, which sample_k_for_level's clamp guarantees is feasible.
    return by_size[:k]


def partition_with_capacity(
    total: int, caps: Sequence[int], rng: random.Random, *, min_per: int = 1
) -> Optional[List[int]]:
    """
    Random partition of ``total`` into ``len(caps)`` parts, each within ``[min_per, caps[i]]``.

    Starts from an uncapped random partition and moves overflow onto groups with spare room, in an
    RNG-chosen order. That is the second half of the trap-14 fix: the pre-migration code drew a
    plain random partition and *rejected the whole example* whenever a large slice happened to land
    on a small concept pool, so the rejection rate -- and hence the realised level mix -- depended
    on the pool-size distribution.

    :param total: Documents to hand out.
    :param caps: Per-group capacity.
    :param rng: Seeded RNG.
    :param min_per: Floor per group.

    :returns: The sizes, or ``None`` when no assignment exists.
    """
    k = len(caps)
    if total < k * min_per or sum(caps) < total:
        return None
    sizes = sample_partition_sizes(total, k, rng, min_per=min_per)
    for _ in range(4 * k + 20):
        over = [i for i in range(k) if sizes[i] > caps[i]]
        if not over:
            return sizes
        i = over[0]
        excess = sizes[i] - caps[i]
        sizes[i] = caps[i]
        under = [j for j in range(k) if sizes[j] < caps[j]]
        rng.shuffle(under)
        for j in under:
            if not excess:
                break
            moved = min(caps[j] - sizes[j], excess)
            sizes[j] += moved
            excess -= moved
        if excess:  # pragma: no cover - unreachable while sum(caps) >= total
            sizes[i] += excess
    return None


def assemble(
    groups: Sequence[Tuple[str, Sequence[Paper]]],
    rng: random.Random,
    *,
    source: str,
    level: int,
    meta: Optional[Mapping[str, object]] = None,
) -> Dict:
    """
    Shuffle the drawn groups into display order and record the gold partition.

    :param groups: ``(concept value, its papers)`` per group, in construction order.
    :param rng: Seeded RNG. Consumes exactly one shuffle over document positions.
    :param source: Corpus tag.
    :param level: The concept level the gold came from, carried for analysis.
    :param meta: Per-example metadata.

    :returns: A unified-format example with 0-based ``gold_doc_indices``, ``cluster_labels``
        parallel to it, and ``answers[0]`` built by the task's own target builder.
    """
    papers: List[Paper] = []
    clusters: List[List[int]] = []
    labels: List[str] = []
    for value, members in groups:
        clusters.append([len(papers) + i for i in range(len(members))])
        papers.extend(members)
        labels.append(value)

    shuffled, old_to_new = shuffle_with_remap(papers, rng=rng, base=0)
    # `remap` per group, NOT `remap_groups`: the latter sorts the OUTER list, which would break the
    # positional correspondence between gold_doc_indices[i] and cluster_labels[i] that
    # `_grouping.build_labeled_target` relies on -- every group would get some other group's name.
    clusters = [remap(cluster, old_to_new) for cluster in clusters]

    k = len(clusters)
    query = rng.choice(QUERY_PHRASINGS).format(k=k, n=len(shuffled))
    example = make_example(
        documents=[make_document(p.abstract, title=p.title) for p in shuffled],
        queries=[query],
        answers=[],
        source=source,
        gold=clusters,
        meta=dict(meta or {}),
        cluster_labels=labels,
        level=level,
        k=k,
    )
    # Built from the finished example rather than assembled inline, so the stored answer cannot
    # drift from what the spec's target builder emits at training time.
    example["answers"] = [build_labeled_target(example)]
    return example


def build_example(
    rng: random.Random,
    *,
    corpus: OpenAlexPool,
    index: int,
    num_docs: int,
    levels: str = "0,1,2,3",
    min_per_group: int = 1,
) -> Optional[Dict]:
    """
    Build one grouping example, at a level fixed by the example index.

    :param rng: Seeded RNG.
    :param corpus: The OpenAlex pool, already sliced to this split.
    :param index: Running example counter. **The level is ``levels[index % len(levels)]``** -- see
        the module warning; drawing it from the RNG is what let the mix drift with N.
    :param num_docs: Abstracts per example -- the scaling axis.
    :param levels: Comma-separated concept levels to rotate through. A string rather than a tuple
        so ``-C levels=0,1`` works from the CLI. Non-uniform mixes are not ported: the
        pre-migration ``--level-mix`` defaulted to uniform and no shipped file used anything else.
    :param min_per_group: Smallest allowed cluster. ``1`` allows singletons, as shipped.

    :returns: A unified-format example, or ``None`` when this level cannot supply a buildable
        partition at this ``num_docs``. A run of these means the pool is too thin at that level,
        and :class:`~ctc.data.build.BuildReport`'s ``skipped`` count is where it shows.

    :raises ValueError: If ``levels`` is empty or holds a level OpenAlex does not define.
    """
    wanted = [int(part) for part in str(levels).split(",") if part.strip() != ""]
    if not wanted or any(level not in K_FRACTION_PER_LEVEL for level in wanted):
        raise ValueError(f"levels={levels!r} must be a non-empty subset of 0,1,2,3")
    level = wanted[index % len(wanted)]

    level_index = corpus.level_index(level, min_per_value=min_per_group)
    if not level_index:
        return None
    k = sample_k_for_level(level, num_docs, rng, level_index, min_per_group=min_per_group)
    if k is None:
        return None
    chosen = choose_group_values(level_index, k, num_docs, rng)
    if chosen is None:
        return None
    caps = [len(level_index[value]) for value in chosen]
    sizes = partition_with_capacity(num_docs, caps, rng, min_per=min_per_group)
    if sizes is None:
        return None
    # Pair the largest slice with the largest pool. Both already respect each other's cap, so this
    # only makes the resulting label sizes readable.
    order = sorted(range(k), key=lambda i: -caps[i])
    groups = [(chosen[i], rng.sample(list(level_index[chosen[i]]), sizes[i])) for i in order]

    return assemble(
        groups,
        rng,
        source=SOURCE_TAG,
        level=level,
        meta={
            "n": num_docs,
            "group_sizes": [len(members) for _, members in groups],
            "provenance": corpus.provenance,
        },
    )
