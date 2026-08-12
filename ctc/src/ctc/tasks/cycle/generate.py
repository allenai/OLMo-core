"""
``cycle`` data: N comparative claims over named entities, find the impossible loops.

Every claim in one example uses the same predicate ("A eats more chicken than B"), so the corpus is
a single directed graph where an edge means A > B. The relation is a strict order, so a directed
cycle is logically impossible; the task is to find each one.

**Exactly K cycles, guaranteed.** All entities share one random total order. Every background edge
runs strictly forward in that order, so the background alone is a DAG and any cycle must use a
planted backward edge. Each cycle's L entities occupy *consecutive* ranks (a "block") and the cycle
is the forward chain through the block plus one backward closing edge. Given one rule -- no
background edge has both endpoints in the same block -- a forward path from the block's bottom to
its top is trapped inside the block (rank only increases, and no outside entity ranks between the
endpoints), so the planted chain is the unique completion of each closing edge.

.. warning::
   **Cycle entities are full participants in background-edge sampling.** An earlier version drew
   background edges from a disjoint pool, which pinned every cycle entity's claim frequency at
   exactly 2 (one in-edge, one out-edge) regardless of N, while background frequency grew with N.
   Gold was then "the three rarest names", and the shortcut got *stronger* as N grew -- the
   opposite of the intended O(N^L) search, and it was measured, not theorised
   (``records/ctc-setting-verification-2026-07-23.md``). The fix is the ``in_same_block`` exclusion
   below, which is the *only* restriction on background edges. A rewrite that reintroduces a
   separate distractor pool silently restores the shortcut, which is why
   :func:`ctc.data.audit.cycle_frequency_gap` runs on every build.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

from ...data.generators.base import Generator
from ...data.gold import remap_groups, shuffle_with_remap
from ...data.schema import make_document, make_example

__all__ = ["build_example", "GENERATOR"]

#: Distinct first names. Examples needing more draw readable ``Name_<k>`` fallbacks. Names must be
#: unique: two entities sharing one would merge in the graph and break the exactly-K invariant.
NAMES: Tuple[str, ...] = tuple(dict.fromkeys("""
Bob Jane Dan Alice Carlos Mei Omar Priya Liam Nina Hugo Sara Kofi Yuki Ivan Lena
Pablo Aisha Tom Greta Raj Elsa Noah Fatima Erik Zoe Sven Maya Paolo Rosa Ahmed
Clara Diego Hana Felix Ingrid Jamal Kira Luca Mira Nadia Oscar Petra Quinn Rosa
Theo Uma Viktor Wen Xena Yara Zane Anya Bruno Cleo Dora Enzo Faye Gus Halle Igor
Jada Kai Lara Milo Nora Otto Pia Remy Suki Tariq Vera Will Xavi Yusuf Zara Aldo
Bea Cyrus Dina Esme Finn Gita Hiro Iris Juno Kane Lily Marco Nia Olen Posy Rune
Said Tess Ugo Val Wade Xia Yael Zeb
""".split()))

PREDICATES: Tuple[str, ...] = (
    "{a} eats more chicken than {b}",
    "{a} is taller than {b}",
    "{a} runs faster than {b}",
    "{a} is older than {b}",
    "{a} owns more books than {b}",
    "{a} scored higher on the exam than {b}",
    "{a} has more money than {b}",
    "{a} lifts heavier weights than {b}",
    "{a} sleeps longer than {b}",
    "{a} drinks more coffee than {b}",
    "{a} lives closer to the office than {b}",
    "{a} types faster than {b}",
)

QUERY = (
    "Each claim asserts a strict comparison (the first person ranks strictly above the second). "
    "Find every set of claims that forms a cycle — an impossible loop where the ranking comes back "
    "to where it started."
)


def _entity_names(n: int, rng: random.Random) -> List[str]:
    """
    :param n: How many distinct names are needed.
    :param rng: Seeded RNG.

    :returns: ``n`` distinct names. The full pool is shuffled *before* truncating, so which names
        form the cycle is stable across different ``num_docs`` at one seed -- that is what keeps an
        eval ladder row-aligned, with the same gold claims at every rung and only the background
        growing.
    """
    names = list(NAMES)
    rng.shuffle(names)
    return names if n <= len(names) else names + [f"Name_{k}" for k in range(n - len(names))]


def _entity_budget(n_cycles: int, cycle_len: int, n_distract: int) -> int:
    """
    :param n_cycles: K.
    :param cycle_len: L.
    :param n_distract: Background edges required.

    :returns: Total entity count: the smallest pool admitting ``n_distract`` distinct forward edges
        once same-block pairs are excluded. Grows as ``sqrt(2 * n_distract)``, so background-entity
        degree is unchanged by the anti-shortcut fix -- only cycle-entity degree moves.
    """
    d = 2
    while True:
        total = n_cycles * cycle_len + d
        blocked = n_cycles * (cycle_len * (cycle_len - 1) // 2)
        if total * (total - 1) // 2 - blocked >= max(1, n_distract):
            return total
        d += 1


def build_example(
    rng: random.Random, *, num_docs: int, cycle_len: int = 3, num_cycles: int = 1
) -> Dict:
    """
    Build one cycle example.

    :param rng: Seeded RNG.
    :param num_docs: Corpus size N -- the scaling axis.
    :param cycle_len: Cycle length L. Longer is harder: naive search is O(N^L).
    :param num_cycles: Number of planted cycles K.

    :returns: A unified-format example whose ``gold_doc_indices`` holds one 1-based claim-id list
        per cycle.

    :raises ValueError: If ``num_docs`` cannot hold K cycles of length L.
    """
    L, K = cycle_len, num_cycles
    n_distract = num_docs - K * L
    if n_distract < 0:
        raise ValueError(f"num_docs={num_docs} too small for {K} cycles of length {L}")

    predicate = rng.choice(PREDICATES)
    total = _entity_budget(K, L, n_distract)
    names = _entity_names(total, rng)[:total]
    cycle_entities, background = names[: K * L], names[K * L :]

    # One global rank order, with each cycle's entities spliced in as a consecutive block.
    rng.shuffle(background)
    insert_points = sorted(rng.sample(range(len(background) + 1), K))
    order: List[str] = []
    blocks: List[Tuple[int, int]] = []
    prev = 0
    for c, point in enumerate(insert_points):
        order.extend(background[prev:point])
        start = len(order)
        order.extend(cycle_entities[c * L : (c + 1) * L])
        blocks.append((start, start + L - 1))
        prev = point
    order.extend(background[prev:])

    # Planted cycles first, so their edge positions are known before the background is appended.
    edges: List[Tuple[str, str]] = []
    gold_groups: List[List[int]] = []
    for start, end in blocks:
        ring = order[start : end + 1]
        gold_groups.append(list(range(len(edges), len(edges) + L)))
        edges.extend((ring[i], ring[(i + 1) % L]) for i in range(L))

    eligible = [
        (order[i], order[j])
        for i in range(total)
        for j in range(i + 1, total)
        if not any(s <= i <= e and s <= j <= e for s, e in blocks)
    ]
    rng.shuffle(eligible)
    edges.extend(eligible[:n_distract])

    claims, old_to_new = shuffle_with_remap(edges, rng=rng, base=1)
    return make_example(
        documents=[make_document(predicate.format(a=a, b=b)) for a, b in claims],
        queries=[QUERY],
        source="cycle",
        gold=remap_groups(gold_groups, old_to_new),
        meta={"cycle_len": L, "num_cycles": K, "predicate": predicate},
    )


GENERATOR = Generator(
    name="cycle",
    task="cycle",
    source="cycle",
    build_example=build_example,
    defaults={"num_docs": 100, "cycle_len": 3, "num_cycles": 1},
    notes="pure synthetic; no corpus, no network",
)
