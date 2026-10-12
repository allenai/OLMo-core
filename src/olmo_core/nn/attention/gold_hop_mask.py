"""
Gold-aware **hop-controlled** document graph -- the mask side of the multi-hop gold-routing
experiment (``records/multihop-gold-routing-experiment.md`` §"Approach A").

The question
------------
Cross-document information reaches the answer through three channels: **(a)** the FREE query/answer
tokens, which attend everything under *every* mask; **(b)** a direct doc->doc edge; **(c)** a multi-hop
path traversed across layers (doc ``b`` reads doc ``m`` at layer L, and ``m`` already read doc ``a`` at
layer L-1). If (c) works, cross-document attention need not be quadratic -- it suffices that the gold
documents be *connected*, not *adjacent*.

This module **forbids (b) for the gold pair** and forces a path of a controlled length, which is the
literal form of the question. Every other mask in the tree is gold-agnostic and therefore cannot
answer it: they never forbid the gold edge.

The arms (``gold_hops``)
------------------------

===============  ==========================  =====================  ==================
``gold_hops``    gold edge                   gold path              channels
===============  ==========================  =====================  ==================
``1``            **forced present**          1 hop                  a + b + c
``2``            **deleted**                 forced shortest 2      a + c  (the test)
``3``            **deleted**                 forced shortest 3      a + c
``GOLD_HOPS_INF``  **deleted**               **none**               a only (the control)
===============  ==========================  =====================  ==================

**Read the ladder as ``hop2`` vs ``hop_inf``, never ``hop2`` vs ``chunked``.** A gold-aware mask is a
train-time signal that encodes the answer -- "the doc I am *not* allowed to see" is "my contradiction
partner" -- so every arm here carries some leak. ``hop2`` and ``hop_inf`` have *identical*
gold-edge-absent structure and differ only in whether a path exists, so the leak inflates both equally
and cancels in the contrast. ``chunked`` has no doc edges at all, hence no hole to read, and is a scale
reference only.

Two structural choices, both load-bearing
-----------------------------------------
1. **The base graph is gold-AGNOSTIC** ``random_doc`` at ``doc_keep_prob`` (built here with the *same*
   seeded hash as :func:`~olmo_core.nn.attention.chunked_mask._random_doc_keep`, so
   ``gold_hops``-off is bit-identical to the ``random_doc`` pattern). At ``p=0.25`` three quarters of
   doc pairs are already non-adjacent by coin flip, so deleting the gold edge looks like an ordinary
   miss -- camouflage. ``doc_keep_prob`` is a **knob**: if the base graph turns out too sparse to route
   at all (the measured ``hier-K10`` dead zone, where 100% reachability and max 3 hops still died at
   the chunked floor), the ladder moves to ``p=0.5``, whose larger residual leak ``hop_inf`` cancels
   anyway.
2. **Per-example resampling is MANDATORY** (``random_doc_per_example`` semantics: the base graph mixes
   a nonce derived from the example's own chunk layout). A graph held fixed across examples is
   memorizable, and against a memorized graph a *missing* edge announces which document is gold --
   amplifying the exact leak the camouflage exists to suppress.

⚠ Causality: the intermediary must sit BETWEEN the gold docs
------------------------------------------------------------
The mask is causal (doc ``b`` attends doc ``a`` only if ``a < b``), so a 2-hop path ``b -> m -> a``
requires ``a < m < b``. An **adjacent** gold pair (``b == a + 1``) has no possible intermediary and is
*silently* ``hop_inf`` no matter what the arm says; a pair at distance 2 additionally admits no 3-hop
path. Measured on ``contra_n50_v2_orig`` (6000 pairs): **4.03% are adjacent**, 7.82% are at distance
<= 2.

This is **not** fixed by constraining gold placement -- that would make placement itself gold-aware
("the partner is never nearby"), re-introducing the leak the design exists to avoid, and the eval file
cannot be rebuilt anyway. Instead it is handled explicitly: such pairs are counted
(``n_unroutable``), and every pair's **realized** hop distance is recorded per example
(:class:`GoldHopStats`) so eval can stratify and report the routable subset as the mixture-free number.
The dilution biases ``hop2`` *toward* ``hop_inf``, i.e. against the hypothesis, which is the safe
direction.

Edge-count matching
-------------------
Arms must not differ in density, or the hop contrast becomes a density contrast. The base graph is a
deterministic function of ``(seed, doc_keep_prob, per-example nonce)`` and therefore **identical across
arms for the same example**, so the edit is compensated back to the base graph's exact edge count with
safe random non-gold edges (:func:`edit_doc_graph_for_hops`). Realized drift is recorded rather than
assumed.

Plumbing
--------
Gold identity must never enter the token stream (a gold marker would trivially leak the answer), so
this reuses :mod:`~olmo_core.nn.attention.gold_grad_mask`'s proven mechanism verbatim: a
``forward_pre_hook`` fingerprints the live ``input_ids`` (SHA1 to the first EOS) and looks the example
up in a ``gold_pairs.json`` sidecar. NB that module is a *backward*-graph pruner; only its lookup/hook
pattern is reused here, not its detach logic. A fingerprint **miss** leaves the row's graph all-True
(plain causal over context -- graceful degradation, as the gold-grad hook does), never all-False.

Scope: eager only. The per-forward Python fingerprint is not ``torch.compile``-capturable (exactly like
the mask-mix curriculum), so this family runs with ``--no-compile`` and
:func:`~olmo_core.nn.attention.chunked_mask.build_chunked_mask_mod` deliberately declines this pattern,
sending it down the dense materialized-mask path.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .chunked_mask import (
    _random_doc_keep,
    build_chunk_ids_from_tokens,
    random_doc_nonce,
)
from .gold_grad_mask import content_fingerprint_from_row

__all__ = [
    "GOLD_HOPS_INF",
    "GOLD_HOP_VALUES",
    "GoldHopStats",
    "GoldHopMaskHolder",
    "build_base_doc_graph",
    "shortest_gold_hops",
    "edit_doc_graph_for_hops",
    "gold_hop_graph_for_row",
    "make_fingerprint_gold_hop_fn",
    "install_gold_hop_mask",
    "dump_realized_hops",
]

#: ``gold_hops`` sentinel for the ``hop_inf`` arm: the gold edge is deleted **and every path between
#: the pair is cut**, so the only surviving channel is the FREE bridge. An ``int`` sentinel (rather
#: than the string ``"inf"``) so it threads through the ``Optional[int]`` config fields unchanged.
GOLD_HOPS_INF = -1

#: The valid ``gold_hops`` settings.
GOLD_HOP_VALUES = (1, 2, 3, GOLD_HOPS_INF)

# Forcing pair P's path can delete an edge that pair Q's forced path depends on (the "kill all shorter
# paths" step of hop3 in particular). Rather than assume independence, re-force whichever pairs came
# out wrong and iterate. Convergence is recorded, never assumed -- see GoldHopStats.repair_passes.
_MAX_REPAIR_PASSES = 8


# ---------------------------------------------------------------------------
# The gold-agnostic base graph
# ---------------------------------------------------------------------------


def build_base_doc_graph(
    n_docs: int,
    *,
    doc_keep_prob: float,
    seed: int,
    nonce: Optional[int] = None,
) -> np.ndarray:
    """
    The gold-AGNOSTIC ``random_doc`` base graph over document indices: ``adj[q, k]`` is ``True`` iff
    document ``q`` may attend the strictly-earlier document ``k``.

    Built with :func:`~olmo_core.nn.attention.chunked_mask._random_doc_keep`, i.e. the *same* seeded
    multiplicative hash the ``"random_doc"`` pattern uses, so an unedited graph is bit-identical to what
    ``random_doc`` would produce for the same ``(seed, doc_keep_prob, nonce)``. That equivalence is what
    lets the hop ladder claim a gold-agnostic base rather than merely a similar one, and it is asserted
    in the tests.

    :param n_docs: Number of context documents (chunk ids ``0 .. n_docs-1``).
    :param doc_keep_prob: Bernoulli keep-probability per ordered document pair.
    :param seed: The graph seed.
    :param nonce: Per-example nonce (see
        :func:`~olmo_core.nn.attention.chunked_mask.random_doc_nonce`). **Required in practice**: a
        graph shared across examples is memorizable, and then the deleted gold edge announces itself.

    :returns: A ``(n_docs, n_docs)`` boolean array, strictly lower-triangular (``q > k``).
    """
    docs = torch.arange(int(n_docs), dtype=torch.int64)
    qc = docs.view(-1, 1)
    kc = docs.view(1, -1)
    nonce_t = None if nonce is None else torch.tensor([int(nonce)], dtype=torch.int64)
    keep = _random_doc_keep(qc, kc, float(doc_keep_prob), int(seed), nonce=nonce_t)
    return (keep & (qc > kc)).numpy()


# ---------------------------------------------------------------------------
# Graph queries
# ---------------------------------------------------------------------------


def _out_neighbors(adj: np.ndarray, x: int) -> List[int]:
    return np.flatnonzero(adj[x]).tolist()


def shortest_gold_hops(adj: np.ndarray, a: int, b: int) -> int:
    """
    Shortest path length from the later document ``b`` to the earlier document ``a`` under ``adj``
    (BFS over ``adj[q, k] == True`` meaning "``q`` attends ``k``").

    :returns: The number of hops, or :data:`GOLD_HOPS_INF` if ``a`` is unreachable from ``b``. ``1``
        means the direct gold edge survives.
    """
    if a == b:
        return 0
    dist = {int(b): 0}
    queue = deque([int(b)])
    while queue:
        x = queue.popleft()
        for y in _out_neighbors(adj, x):
            if y == a:
                return dist[x] + 1
            if y not in dist:
                dist[y] = dist[x] + 1
                queue.append(y)
    return GOLD_HOPS_INF


def _find_path(adj: np.ndarray, b: int, a: int) -> Optional[List[int]]:
    """One shortest path ``[b, ..., a]``, or ``None`` if ``a`` is unreachable from ``b``."""
    if a == b:
        return [int(a)]
    prev: Dict[int, int] = {int(b): -1}
    queue = deque([int(b)])
    while queue:
        x = queue.popleft()
        for y in _out_neighbors(adj, x):
            if y in prev:
                continue
            prev[y] = x
            if y == a:
                path = [int(a)]
                while prev[path[-1]] != -1:
                    path.append(prev[path[-1]])
                return path[::-1]
            queue.append(y)
    return None


def _target_is_routable(a: int, b: int, hops: int) -> bool:
    """
    Whether a path of exactly ``hops`` is *constructible* for this pair at all, given causality.

    A path ``b -> m_1 -> ... -> a`` of length ``h`` needs ``h - 1`` distinct intermediaries strictly
    between ``a`` and ``b``, so it exists only if ``b - a >= h``. Adjacent pairs (``b - a == 1``) are
    therefore unroutable at every ``h >= 2`` -- 4.03% of the n=50 gold pairs, measured.
    """
    if hops == GOLD_HOPS_INF:
        return True
    return (b - a) >= hops


def _pair_target(a: int, b: int, hops: int) -> int:
    """
    The hop count this pair will actually be held to.

    Equal to the arm, except where causality forbids it: an unroutable pair is honestly cut to
    :data:`GOLD_HOPS_INF` rather than mislabelled as hop-``h``. This is the single definition of the
    target -- both the edit and the check read it, so they cannot drift apart.
    """
    return hops if _target_is_routable(a, b, hops) else GOLD_HOPS_INF


def _pair_ok(adj: np.ndarray, a: int, b: int, hops: int) -> bool:
    """Whether this pair's realized structure already matches its target."""
    return shortest_gold_hops(adj, a, b) == _pair_target(a, b, hops)


# ---------------------------------------------------------------------------
# The edit
# ---------------------------------------------------------------------------


class _ForcedEdges:
    """
    The edges each gold pair's forced path currently depends on, tracked **per pair**.

    ⚠ Why per pair rather than one accumulating set: re-forcing a pair picks fresh intermediaries, so a
    flat set keeps every edge from every *abandoned* attempt marked "load-bearing". Measured on the real
    n=50 shard, that union grew to 21 edges over the repair passes and the protection logic inverted --
    ``_kill_two_paths`` started deleting the path it had just built, and the pair could never converge
    (it sat at hop_inf inside the hop3 arm). Replacing a pair's entry keeps the set true.
    """

    def __init__(self) -> None:
        self._by_pair: Dict[Tuple[int, int], set] = {}

    def set(self, pair: Tuple[int, int], edges: Iterable[Tuple[int, int]]) -> None:
        self._by_pair[pair] = set(edges)

    def of(self, pair: Tuple[int, int]) -> set:
        return self._by_pair.get(pair, set())

    def union(self) -> set:
        out: set = set()
        for s in self._by_pair.values():
            out |= s
        return out

    def __contains__(self, edge) -> bool:
        return any(edge in s for s in self._by_pair.values())


def _kill_two_paths(
    adj: np.ndarray, a: int, b: int, forced: "_ForcedEdges", rng: random.Random
) -> None:
    """
    Delete every 2-hop path ``b -> m -> a`` -- the step that makes ``hop3`` mean *shortest* 3, not
    "3 among others".

    Cuts are chosen to protect load-bearing edges, in strict priority: this pair's own forced path
    first (so the kill can never sever the path it exists to protect), then any other pair's, then an
    arbitrary but seeded choice.
    """
    own = forced.of((a, b))
    for m in range(a + 1, b):
        if not (adj[m, a] and adj[b, m]):
            continue
        if (b, m) in own:  # m is this pair's second intermediary -> keep b->m, drop m->a
            adj[m, a] = False
        elif (m, a) in own:  # m is this pair's first intermediary -> keep m->a, drop b->m
            adj[b, m] = False
        elif (m, a) in forced:
            adj[b, m] = False
        elif (b, m) in forced:
            adj[m, a] = False
        else:
            # Neither edge is load-bearing; cut one deterministically given the seeded rng.
            if rng.random() < 0.5:
                adj[b, m] = False
            else:
                adj[m, a] = False


def _cut_pair(adj: np.ndarray, a: int, b: int, forced: "_ForcedEdges") -> None:
    """
    Cut **every** path from ``b`` to ``a``, preferring to sacrifice edges no forced path needs.

    ⚠ The obvious implementation -- delete ``a``'s in-edges from everything reachable from ``b`` -- is
    correct in isolation and **wrong with multiple gold pairs**, which is the real case (3 pairs per
    example). Measured on the real n=50 shard: pairs ``(1, 6)`` and ``(5, 7)`` interleave, so cutting
    ``7 -> ... -> 5`` deletes edge ``6 -> 5``, which is exactly the edge pair ``(1, 6)``'s forced 3-hop
    path routes through; re-forcing it re-creates the path, and the two edits ping-pong forever.

    So instead: repeatedly find a surviving path and cut ONE edge on it, taking the last unforced edge
    (nearest ``a``, keeping the cut local) and only breaking a forced edge if every edge on the path is
    forced -- a genuine conflict, which the caller's repair pass then re-forces. Each iteration removes
    an edge, so this terminates.
    """
    adj[b, a] = False
    while True:
        path = _find_path(adj, b, a)
        if path is None:
            return
        edges = list(zip(path[:-1], path[1:]))
        victim = next((e for e in reversed(edges) if e not in forced), edges[-1])
        adj[victim[0], victim[1]] = False


def _force_path_pair(
    adj: np.ndarray,
    a: int,
    b: int,
    hops: int,
    rng: random.Random,
    forced: "_ForcedEdges",
    treated: Optional[set] = None,
) -> None:
    """
    Force a pair to a shortest path of exactly ``hops``. Caller guarantees it is routable.

    :param treated: Every pair under edit in this example (gold + decoys), as ``(a, b)`` with ``a < b``.
        Intermediaries are chosen to **avoid creating another treated pair's direct edge**: a 2-hop path
        ``b -> m -> a`` adds the edge ``b -> m``, which *is* the direct edge of the pair ``(m, b)`` --
        and if that pair is also under edit, it forbids exactly that edge, so the two edits fight.
        ⚠ Measured: without this, the decoy fix stops converging at 12 decoys/pair.

        This is a **preference, not a constraint**: if no collision-free intermediary exists the
        original candidate set is used unchanged, so the routability contract (``b - a >= hops`` =>
        a path is constructible) that :func:`_pair_target` relies on still holds exactly. With ~39
        treated pairs out of 1225 the fallback is essentially never taken.
    """
    if hops == 1:
        adj[b, a] = True
        forced.set((a, b), {(b, a)})
        return

    # Every other arm deletes the direct gold edge. That shared structure is what makes hop_inf a
    # leak-matched control for hop2 rather than merely a sparser arm.
    adj[b, a] = False
    treated = treated or set()

    def _free(x: int, y: int) -> bool:
        """Is the edge between docs x, y usable -- i.e. not some treated pair's forbidden direct edge?"""
        return (min(x, y), max(x, y)) not in treated

    span = list(range(a + 1, b))

    if hops == 2:
        candidates = [m for m in span if _free(a, m) and _free(m, b)] or span
        m = candidates[rng.randrange(len(candidates))]
        adj[m, a] = True
        adj[b, m] = True
        forced.set((a, b), {(m, a), (b, m)})
    elif hops == 3:
        legal = [
            (x, y)
            for i, x in enumerate(span)
            for y in span[i + 1 :]
            if _free(a, x) and _free(x, y) and _free(y, b)
        ]
        if legal:
            m1, m2 = legal[rng.randrange(len(legal))]
        else:
            m1, m2 = sorted(rng.sample(span, 2))
        adj[m1, a] = True
        adj[m2, m1] = True
        adj[b, m2] = True
        forced.set((a, b), {(m1, a), (m2, m1), (b, m2)})
        # Shortest == 3 requires deleting every 2-path too. The forced edges are protected, so this
        # cannot cut the path it just built: for m1 it drops (b, m1), for m2 it drops (m2, a).
        _kill_two_paths(adj, a, b, forced, rng)
    else:  # pragma: no cover - guarded by the caller
        raise ValueError(f"unsupported gold_hops {hops!r}; expected one of {GOLD_HOP_VALUES}")


def _apply_edits(
    adj: np.ndarray,
    pairs: Sequence[Tuple[int, int]],
    hops: int,
    rng: random.Random,
    forced: "_ForcedEdges",
    treated: Optional[set] = None,
) -> None:
    """
    Apply the arm's edit to a set of pairs -- gold, plus (when the decoy fix is on) their
    distance-matched decoys, treated IDENTICALLY because anything that distinguished them would
    re-open the leak the decoys exist to close.

    ⚠ **Order matters and is load-bearing.** Every path is forced *before* any cut, so the cuts know
    which edges are load-bearing and can route around them. Cutting first would delete edges the later
    forcing then re-adds, and the two would fight (measured -- see :func:`_cut_pair`).
    """
    targets = [(a, b, _pair_target(a, b, hops)) for a, b in pairs]
    for a, b, target in targets:
        if target != GOLD_HOPS_INF:
            _force_path_pair(adj, a, b, hops, rng, forced, treated=treated)
    for a, b, target in targets:
        if target == GOLD_HOPS_INF:
            # Either the hop_inf arm, or a pair causality leaves unroutable (an adjacent pair at h=2,
            # distance <= 2 at h=3). Both must genuinely have NO path: deleting only the direct edge
            # would leave a distance-2 pair at a realized 2 hops *inside the hop3 arm*, i.e. a rung
            # leaking into the one below it. Measured: 7.82% of n=50 gold pairs sit at distance <= 2.
            _cut_pair(adj, a, b, forced)


def _compensate_edge_count(
    adj: np.ndarray,
    pairs: Sequence[Tuple[int, int]],
    hops: int,
    rng: random.Random,
    *,
    target_edges: int,
    forced: "_ForcedEdges",
) -> int:
    """
    Restore the edge count to ``target_edges`` (the untouched base graph's) using **safe** random
    non-gold edges, so the arms differ in gold-path structure and not in density.

    Every candidate is validated against the arm's invariants before being kept, which matters: at
    ``hop3`` an added edge can create a 2-path, and at ``hop_inf`` it can reconnect the pair. Gold
    direct edges are never candidates.

    :returns: The realized edge count (may fall short of ``target_edges`` if too few safe edges exist;
        the shortfall is recorded rather than hidden).
    """
    d = adj.shape[0]
    gold_direct = {(b, a) for a, b in pairs}
    cur = int(adj.sum())

    def _all_ok() -> bool:
        return all(_pair_ok(adj, a, b, hops) for a, b in pairs)

    if cur < target_edges:
        cands = [
            (q, k)
            for q in range(d)
            for k in range(q)
            if not adj[q, k] and (q, k) not in gold_direct
        ]
        rng.shuffle(cands)
        for q, k in cands:
            if cur >= target_edges:
                break
            adj[q, k] = True
            if _all_ok():
                cur += 1
            else:
                adj[q, k] = False
    elif cur > target_edges:
        keep = forced.union()
        cands = [(q, k) for q in range(d) for k in range(q) if adj[q, k] and (q, k) not in keep]
        rng.shuffle(cands)
        for q, k in cands:
            if cur <= target_edges:
                break
            adj[q, k] = False
            if _all_ok():
                cur -= 1
            else:
                adj[q, k] = True
    return cur


@dataclass
class GoldHopStats:
    """
    What the edit actually produced for one example -- **measured, not assumed**.

    Every field exists because the corresponding property is one the experiment's reading depends on
    and which a silent failure would otherwise fake.

    :param pairs: The gold pairs as ``(a, b)`` with ``a < b``.
    :param distances: ``b - a`` per pair (the causal room available for intermediaries).
    :param realized_hops: The **measured** shortest path per pair after the edit
        (:data:`GOLD_HOPS_INF` = unreachable). Eval stratifies on this; ``hop2``'s headline f1 is a
        mixture over it, and the routable subset is the mixture-free number.
    :param unroutable: Per pair, whether the target hop count was impossible by causality (an adjacent
        pair at ``hop2``). These are honestly ``hop_inf``, never mislabelled.
    :param base_edges: Edge count of the untouched gold-agnostic base graph.
    :param final_edges: Edge count after edit + compensation. ``final_edges - base_edges`` is the
        per-example density drift between this arm and the base -- and, because the base graph is a
        pure function of ``(seed, keep_prob, nonce)``, between this arm and every other arm.
    :param repair_passes: How many force/verify passes the multi-pair interference needed.
    :param converged: Whether every **gold** pair reached its target (or an honest ``hop_inf``) within
        :data:`_MAX_REPAIR_PASSES`. Decoys are excluded on purpose -- see :attr:`decoys_ok`.
    :param n_decoys: How many distance-matched decoy pairs were given the arm's edit.
    :param decoys_ok: How many of them actually realized it. This is **camouflage quality**, not
        correctness: a decoy that missed is simply not a lookalike. ``decoys_ok / n_decoys`` well below
        1 means the disguise is thinner than configured, and the leak probe should be re-run.
    """

    pairs: List[Tuple[int, int]] = field(default_factory=list)
    distances: List[int] = field(default_factory=list)
    realized_hops: List[int] = field(default_factory=list)
    unroutable: List[bool] = field(default_factory=list)
    base_edges: int = 0
    final_edges: int = 0
    repair_passes: int = 0
    converged: bool = True
    n_decoys: int = 0
    decoys_ok: int = 0

    @property
    def edge_drift(self) -> int:
        """Signed per-example edge-count drift from the gold-agnostic base graph (0 = fully matched)."""
        return self.final_edges - self.base_edges


#: Gold pairs closer than this get **no decoys**. Two measured reasons, both decisive:
#:
#: * **They do not need them.** The leak is a long-distance phenomenon. Measured lift of
#:   ``P(unreachable | gold) / P(unreachable | non-gold)`` under ``hop_inf``: **1.32** at distance 1,
#:   1.51 at 2-4, 2.54 at 5-12, 9.29 at 13-25, **45.23** at 26-49. A short unreachable pair is utterly
#:   unremarkable -- three quarters of distance-1 pairs are unreachable by coin flip.
#: * **They break the edit.** 12 decoys at distance 1 forbid 12 direct edges inside one small
#:   neighbourhood, and then a nearby pair has no intermediary that is not some other treated pair's
#:   forbidden edge. Measured: gold pairs stop converging at hop2.
_MIN_DECOY_DISTANCE = 5


def sample_distance_matched_decoys(
    n_docs: int,
    gold_pairs: Sequence[Tuple[int, int]],
    n_decoys: int,
    rng: random.Random,
    *,
    hops: Optional[int] = None,
    min_distance: int = _MIN_DECOY_DISTANCE,
) -> List[Tuple[int, int]]:
    """
    Non-gold pairs at **exactly the same distances** as the gold pairs, to be given the same edit.

    ⚠ This exists because the camouflage the design *assumed* does not hold, and the gap was measured
    (``debug/gold_hop/leak_probe.py``): an arm's edit applied ONLY to gold pairs makes the gold pair
    identifiable from the graph alone. Worst case ``hop_inf`` -- a graph-only logistic regression, no
    text, out-of-sample, reaches **precision@3 = 16.2% against a 0.245% chance rate (66x)**.

    The reason is distance. ``P(unreachable | non-gold)`` is 0.756 at distance 1 but **0.022** at
    distance 26-49, so the record's aggregate "31.2% of pairs are unreachable" is the wrong null: it is
    dominated by short pairs, while gold sits at mean distance ~17. An unreachable *distant* pair is a
    beacon.

    Distance matching is what defuses it: if ``n_decoys`` non-gold pairs **at the gold pair's own
    distance** get the identical treatment, "unreachable at distance 30" stops naming the gold pair and
    names a set of ``1 + n_decoys`` equally-treated candidates instead.

    :param n_docs: Number of context documents.
    :param gold_pairs: The real gold pairs as ``(a, b)``, ``a < b``.
    :param n_decoys: Decoys **per gold pair**.
    :param rng: The example's seeded RNG.
    :param hops: The arm. Pairs the arm cannot route anyway (``b - a < hops``) get no decoys -- they are
        honest ``hop_inf`` whatever the arm says, and at those distances unreachability is the norm.
    :param min_distance: Skip decoys for gold pairs closer than this (see :data:`_MIN_DECOY_DISTANCE`).

    :returns: The decoy pairs (disjoint from ``gold_pairs``, and from each other).
    """
    gold = {tuple(p) for p in gold_pairs}
    chosen: List[Tuple[int, int]] = []
    taken = set(gold)
    for a, b in gold_pairs:
        d = b - a
        if d < min_distance:
            continue
        if hops is not None and hops != GOLD_HOPS_INF and d < hops:
            continue
        cands = [(x, x + d) for x in range(0, n_docs - d) if (x, x + d) not in taken]
        rng.shuffle(cands)
        for c in cands[: max(0, n_decoys)]:
            chosen.append(c)
            taken.add(c)
    return chosen


def edit_doc_graph_for_hops(
    adj: np.ndarray,
    pairs: Sequence[Sequence[int]],
    hops: int,
    rng: random.Random,
    *,
    compensate: bool = True,
    n_decoys: int = 0,
) -> Tuple[np.ndarray, GoldHopStats]:
    """
    Edit a gold-agnostic base graph into the arm's hop-controlled graph.

    For each gold pair ``(a, b)``: ``hops=1`` forces the direct edge; ``hops in (2, 3)`` **deletes** it
    and forces a shortest path of exactly that length; :data:`GOLD_HOPS_INF` deletes it and cuts every
    path. The edit is then compensated back to the base graph's edge count with safe random non-gold
    edges, and every pair's realized structure is measured.

    :param adj: The base graph from :func:`build_base_doc_graph` (not modified).
    :param pairs: Gold pairs as 0-based chunk indices, e.g. ``[[9, 28], [10, 31]]`` (order within a
        pair is irrelevant; they are sorted here).
    :param hops: One of :data:`GOLD_HOP_VALUES`.
    :param rng: A seeded :class:`random.Random` (seed per example so the graph is stable across epochs
        and reproducible at eval).
    :param compensate: Restore the base edge count after the edit. Leave on -- otherwise the arms
        differ in density and the hop contrast is confounded.
    :param n_decoys: Non-gold pairs **per gold pair**, at matched distances, given the identical edit
        (:func:`sample_distance_matched_decoys`). ``0`` reproduces the original gold-only design, whose
        structural leak is measured and large (``hop_inf``: graph-only precision@3 16.2% vs 0.245%
        chance). Raise it to dilute the arm's signature across ``1 + n_decoys`` candidates.

    :returns: ``(edited_graph, stats)``. ``stats`` describes the GOLD pairs only -- decoys are
        deliberately indistinguishable and are not tracked per pair.
    """
    if hops not in GOLD_HOP_VALUES:
        raise ValueError(f"gold_hops must be one of {GOLD_HOP_VALUES} (got {hops!r})")
    adj = np.array(adj, dtype=bool, copy=True)
    norm_pairs: List[Tuple[int, int]] = []
    for p in pairs:
        if len(p) != 2:
            raise ValueError(f"gold pair {list(p)!r} is not a pair; the sidecar must be gold_pairs")
        a, b = sorted(int(x) for x in p)
        if a == b:
            raise ValueError(
                f"degenerate gold pair {list(p)!r} (a document cannot contradict itself)"
            )
        norm_pairs.append((a, b))

    base_edges = int(adj.sum())
    forced = _ForcedEdges()
    # Decoys get the IDENTICAL edit at the IDENTICAL distance, so the arm's structural signature stops
    # naming the gold pair. They are treated exactly like gold from here on -- that is the whole point;
    # anything that distinguished them would re-open the leak.
    decoys = (
        sample_distance_matched_decoys(adj.shape[0], norm_pairs, n_decoys, rng, hops=hops)
        if n_decoys > 0
        else []
    )
    treated = list(norm_pairs) + decoys
    treated_set = {(a, b) for a, b in treated}
    _apply_edits(adj, treated, hops, rng, forced, treated=treated_set)

    # Multi-pair interference: pair Q's edit can still invalidate pair P (a cut whose every path edge
    # is forced has to break one). Re-apply to whatever came out wrong and re-check, bounded.
    passes = 0
    for passes in range(1, _MAX_REPAIR_PASSES + 1):
        wrong = [(a, b) for a, b in treated if not _pair_ok(adj, a, b, hops)]
        if not wrong:
            break
        _apply_edits(adj, wrong, hops, rng, forced, treated=treated_set)
    # ⚠ Convergence is judged on the GOLD pairs only -- they are the contract. A decoy that misses its
    # target is a camouflage shortfall, not a correctness bug: it simply is not one of the lookalikes,
    # which weakens the disguise slightly and is reported as `decoys_ok`. (Decoys CAN be unsatisfiable:
    # 12 decoys at distance 1 saturate a neighbourhood, and a distance-3 pair then has no intermediary
    # that is not some other treated pair's forbidden direct edge. Gold is unaffected -- measured.)
    converged = not [(a, b) for a, b in norm_pairs if not _pair_ok(adj, a, b, hops)]
    decoys_ok = sum(1 for a, b in decoys if _pair_ok(adj, a, b, hops))

    final_edges = int(adj.sum())
    if compensate:
        final_edges = _compensate_edge_count(
            adj, treated, hops, rng, target_edges=base_edges, forced=forced
        )

    realized = [shortest_gold_hops(adj, a, b) for a, b in norm_pairs]
    stats = GoldHopStats(
        pairs=norm_pairs,
        distances=[b - a for a, b in norm_pairs],
        realized_hops=realized,
        unroutable=[not _target_is_routable(a, b, hops) for a, b in norm_pairs],
        base_edges=base_edges,
        final_edges=final_edges,
        repair_passes=passes,
        converged=converged,
        n_decoys=len(decoys),
        decoys_ok=decoys_ok,
    )
    return adj, stats


def gold_hop_graph_for_row(
    n_docs: int,
    pairs: Sequence[Sequence[int]],
    *,
    hops: int,
    doc_keep_prob: float,
    seed: int,
    nonce: Optional[int],
    fingerprint: str = "",
    compensate: bool = True,
    n_decoys: int = 0,
) -> Tuple[np.ndarray, GoldHopStats]:
    """
    One example's hop-controlled document graph: build the gold-agnostic base, then edit it.

    The per-example RNG is seeded from ``(seed, fingerprint)`` so the same example always yields the
    same graph -- across epochs, across a resumed run, and at eval.
    """
    base = build_base_doc_graph(n_docs, doc_keep_prob=doc_keep_prob, seed=seed, nonce=nonce)
    rng = random.Random(f"{seed}:{fingerprint}:hop{hops}")
    return edit_doc_graph_for_hops(base, pairs, hops, rng, compensate=compensate, n_decoys=n_decoys)


# ---------------------------------------------------------------------------
# Fingerprint -> per-example adjacency (the non-leaky lookup)
# ---------------------------------------------------------------------------

#: A function mapping a batch's ``input_ids`` ``(B, S)`` to a ``(B, D, D)`` bool doc->doc adjacency.
GoldHopFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class GoldHopMaskHolder:
    """
    Shared per-forward holder: the model's ``forward_pre_hook`` sets :attr:`adjacency` each forward and
    every :class:`~olmo_core.nn.attention.document_chunked.DocumentChunkedAttention` layer reads it.

    :param adjacency: ``(B, D, D)`` bool -- ``adjacency[b, q, k]`` = example ``b``'s document ``q`` may
        attend document ``k``.
    :param n_attached: How many attention modules read this holder (0 means the mask is silently
        inert -- worth asserting at launch).
    :param stats: ``{fingerprint: GoldHopStats}`` for every example seen. This is the realized-hop
        record eval stratifies on; it is measured from the built graph, not assumed from the arm.
    :param counters: Lookup accounting -- ``graph_rows`` = forwards' rows that actually carry documents
        and therefore NEED a graph; ``hits`` = how many of those found their fingerprint. See
        :meth:`hit_rate` / :meth:`require_full_hit_rate`.
    """

    adjacency: Optional[torch.Tensor] = None
    n_attached: int = 0
    stats: Dict[str, GoldHopStats] = field(default_factory=dict)
    counters: Dict[str, int] = field(default_factory=dict)
    misses: List[str] = field(default_factory=list)
    _hook_handle: object = field(default=None, repr=False)

    @property
    def hit_rate(self) -> float:
        """
        Fraction of **document-bearing** rows whose gold pairs were found.

        ⚠ The denominator is deliberately ``graph_rows``, not every row the hook ever saw. Under a KV
        cache, decode feeds **one token at a time**, so a decode row carries no documents and its
        fingerprint cannot match by construction -- counting those would drive the rate to ~1/31 on a
        perfectly healthy contradiction eval and fire a false alarm. A row with no documents needs no
        graph, so it is neither a hit nor a miss.
        """
        n = self.counters.get("graph_rows", 0)
        return 1.0 if n == 0 else self.counters.get("hits", 0) / n

    def require_full_hit_rate(self, context: str = "") -> None:
        """
        Raise unless **every** document-bearing row found its gold pairs.

        ⚠ **This is mandatory at EVAL and must never be relaxed.** A fingerprint miss degrades to an
        all-True graph = plain causal over the context (the right call while *training*, where the
        warm-up mock batch always misses). At eval that same kindness is catastrophic: every arm would
        quietly score as unrestricted `standard` -- near the 0.943 ceiling -- and the ladder would
        report a triumphant result while measuring nothing. The failure looks exactly like success,
        which is why it is an exception and not a warning.

        It also catches the ``use_cache=False`` trap: without a KV cache, decode re-runs the FULL
        sequence (prompt + generated tokens so far), whose fingerprint differs from the prompt-only
        prefill the sidecar is keyed on -- so every decode row would carry documents AND miss.

        :raises SystemExit: if any document-bearing row missed.
        """
        n = self.counters.get("graph_rows", 0)
        hits = self.counters.get("hits", 0)
        if n == 0:
            raise SystemExit(
                f"[gold-hop] FATAL{': ' + context if context else ''}: the gold-hop hook never saw a "
                "single document-bearing row, so the mask was never applied. Nothing was measured."
            )
        if hits < n:
            sample = "\n  ".join(self.misses[:3])
            raise SystemExit(
                f"[gold-hop] FATAL{': ' + context if context else ''}: {n - hits}/{n} document-bearing "
                f"rows MISSED the gold-pairs sidecar (hit rate {self.hit_rate:.1%}).\n"
                "A miss degrades to an ALL-TRUE graph = plain causal over the context, so those rows "
                "were evaluated with NO gold edit at all -- i.e. as unrestricted `standard`, near the "
                "ceiling. The arm would look like a triumphant result while measuring nothing.\n"
                "Likely causes: (1) the sidecar is keyed on the TRAINING shard's rows, whose "
                "fingerprint (SHA1 to the first EOS, answer included) cannot match a prompt-only eval "
                "prefill -- build one with build_gold_pairs_for_eval.py; (2) a prefill-layout flag "
                "(--cot-mode / --free-pad-repeat / --repeat-doc-text / --summary-every-k / boundary "
                "ids) differs from the sidecar build; (3) generation ran with use_cache=False, so "
                "decode re-forwards the full sequence and its fingerprint is not the prefill's.\n"
                f"example missing fingerprints:\n  {sample}"
            )

    def hop_histogram(self) -> Dict[int, int]:
        """Realized hop counts over every pair seen so far (:data:`GOLD_HOPS_INF` = no path)."""
        hist: Dict[int, int] = {}
        for s in self.stats.values():
            for h in s.realized_hops:
                hist[h] = hist.get(h, 0) + 1
        return hist

    def summary(self) -> str:
        """One-line realized-structure summary -- log it, do not trust the config."""
        hist = self.hop_histogram()
        n_pairs = sum(hist.values()) or 1
        drift = [s.edge_drift for s in self.stats.values()] or [0]
        n_unroutable = sum(sum(s.unroutable) for s in self.stats.values())
        parts = " ".join(
            f"{'inf' if h == GOLD_HOPS_INF else h}={hist[h]}({hist[h] / n_pairs:.1%})"
            for h in sorted(hist, key=lambda x: (x == GOLD_HOPS_INF, x))
        )
        return (
            f"[gold-hop] examples={len(self.stats)} pairs={n_pairs} realized_hops: {parts} "
            f"unroutable={n_unroutable}({n_unroutable / n_pairs:.2%}) "
            f"edge_drift mean={float(np.mean(drift)):+.3f} max_abs={int(np.max(np.abs(drift)))}"
        )


def make_fingerprint_gold_hop_fn(
    gold_pairs_table: Dict[str, Iterable[Iterable[int]]],
    *,
    doc_start_id: int,
    doc_end_id: int,
    eos_id: int,
    hops: int,
    doc_keep_prob: float,
    seed: int = 42,
    per_example: bool = True,
    compensate: bool = True,
    n_decoys: int = 0,
    stats: Optional[Dict[str, GoldHopStats]] = None,
    debug_calls: int = 12,
) -> GoldHopFn:
    """
    Build ``fn(input_ids) -> (B, D, D)`` bool adjacency, looking each row's gold **pairs** up by content
    fingerprint (SHA1 of the ids to the first EOS) -- so gold identity never enters the token stream.

    A row whose fingerprint is **absent** degrades gracefully to an **all-True** graph (plain causal
    over the context, i.e. no restriction), matching
    :func:`~olmo_core.nn.attention.gold_grad_mask.make_fingerprint_gold_mask_fn`. It must never degrade
    to all-False: that would silently isolate every document and quietly turn the row into a `chunked`
    example. The trainer's synthetic warm-up mock batch always misses, which is exactly why the
    permissive direction is the right one.

    Graphs are **cached by fingerprint**: an example's graph is a pure function of its content, the arm,
    and the seed, so it is built once and reused on every epoch.

    :param gold_pairs_table: ``{fingerprint: [[a, b], ...]}`` from ``gold_pairs.json`` (0-based chunk
        indices). The **flat** ``gold_fingerprints.json`` is rejected: it cannot say which document
        contradicts which, and "delete the gold edge" is meaningless without the partner.
    :param hops: One of :data:`GOLD_HOP_VALUES`.
    :param doc_keep_prob: Base-graph keep probability (the camouflage density).
    :param per_example: Mix the per-example nonce into the base graph. **Leave on** -- see the module
        docstring; a graph fixed across examples makes the deleted edge a gold beacon.
    :param n_decoys: Distance-matched non-gold decoy pairs per gold pair (the leak fix). ⚠ **Must match
        the value the model trained under** -- it changes the graph, so a mismatch evaluates the model
        on a mask it never saw. At eval, read it from the checkpoint's ``config.json``.
    :param stats: Optional dict to record :class:`GoldHopStats` into (the holder passes its own).
    """
    if hops not in GOLD_HOP_VALUES:
        raise ValueError(f"gold_hops must be one of {GOLD_HOP_VALUES} (got {hops!r})")

    table: Dict[str, List[List[int]]] = {}
    for k, v in gold_pairs_table.items():
        items = list(v)
        # The flat sidecar's values are bare ints ([6, 18, 19, ...]) and would otherwise blow up with
        # an opaque TypeError three frames down. Name the actual problem instead.
        if items and not all(isinstance(p, (list, tuple)) and len(p) == 2 for p in items):
            raise ValueError(
                "gold_hop needs a PAIR-preserving gold sidecar (values like [[6, 19], [18, 48]]); "
                f"got {items[:3]!r}. The flat gold_fingerprints.json is an unordered SET and cannot "
                "express which doc contradicts which -- and 'delete the gold edge' is meaningless "
                "without the partner. Build one with `build_gold_sidecar_from_shard.py --emit pairs`."
            )
        table[k] = [[int(x) for x in p] for p in items]

    cache: Dict[str, np.ndarray] = {}
    record = stats if stats is not None else {}
    # ``graph_rows`` counts only rows that CARRY DOCUMENTS, i.e. that actually need a graph. A KV-cached
    # decode step feeds a single token, which carries none and whose fingerprint cannot match by
    # construction -- counting those as misses would make a healthy contradiction eval read ~1/31.
    state = {"calls": 0, "rows": 0, "hits": 0, "graph_rows": 0}
    misses: List[str] = []

    def fn(input_ids: torch.Tensor) -> torch.Tensor:
        ids_cpu = input_ids.detach().to("cpu")
        if ids_cpu.dim() == 1:
            ids_cpu = ids_cpu.unsqueeze(0)
        roles = build_chunk_ids_from_tokens(
            ids_cpu, doc_start_id=doc_start_id, doc_end_id=doc_end_id, eos_id=eos_id, mode="chunked"
        )
        nonces = random_doc_nonce(roles)
        b_size = roles.shape[0]
        ids2d = ids_cpu.tolist()

        n_docs_per_row: List[int] = []
        for b in range(b_size):
            ctx = roles[b][roles[b] >= 0]
            n_docs_per_row.append(int(ctx.max().item()) + 1 if ctx.numel() else 0)
        d_max = max([1] + n_docs_per_row)

        out = torch.zeros((b_size, d_max, d_max), dtype=torch.bool)
        n_found = 0
        for b in range(b_size):
            n_docs = n_docs_per_row[b]
            if n_docs == 0:
                # No documents -> no graph needed, and nothing to restrict. This is the KV-cached
                # decode step (a single generated token): it is a FREE query, which every arm lets
                # attend the whole cache causally, so plain causal IS the gold-hop mask for this row.
                # Not a hit and NOT a miss -- see GoldHopMaskHolder.hit_rate.
                out[b] = True
                continue
            state["graph_rows"] += 1
            fp = content_fingerprint_from_row(ids2d[b], eos_id)
            pairs = table.get(fp)
            if pairs is None:
                # MISS -> unrestricted (plain causal over context). Never all-False. Correct while
                # TRAINING (the warm-up mock batch always misses); CATASTROPHIC at eval, where it
                # silently scores the arm as `standard` -- hence require_full_hit_rate().
                out[b] = True
                if len(misses) < 32:
                    misses.append(fp)
                continue
            n_found += 1
            graph = cache.get(fp)
            if graph is None:
                graph, st = gold_hop_graph_for_row(
                    n_docs,
                    pairs,
                    hops=hops,
                    doc_keep_prob=doc_keep_prob,
                    seed=seed,
                    nonce=int(nonces[b].item()) if per_example else None,
                    fingerprint=fp,
                    compensate=compensate,
                    n_decoys=n_decoys,
                )
                cache[fp] = graph
                record[fp] = st
            out[b, :n_docs, :n_docs] = torch.from_numpy(graph)

        state["calls"] += 1
        state["rows"] += b_size
        state["hits"] += n_found
        if state["calls"] <= debug_calls:
            tag = " (warmup mock)" if n_found == 0 and state["calls"] == 1 else ""
            print(
                f"[gold-hop] call#{state['calls']}{tag}: B={b_size} D={d_max} hops={hops} "
                f"keep_prob={doc_keep_prob} fp_hits={n_found}/{b_size} "
                f"cum_hits={state['hits']}/{state['graph_rows']} doc_bearing_rows",
                flush=True,
            )
        return out

    # Tagged so ``install_gold_hop_mask`` can prove the hook's arm matches the arm recorded in the
    # model config, instead of letting a launcher typo train hop_inf under a run named hop2 -- and so
    # it can adopt this fn's realized-structure record onto the holder (otherwise ``holder.summary()``
    # silently reports nothing, which is worse than not offering it).
    fn.hops = hops  # type: ignore[attr-defined]
    fn.stats = record  # type: ignore[attr-defined]
    fn.counters = state  # type: ignore[attr-defined]
    fn.misses = misses  # type: ignore[attr-defined]
    return fn


def dump_realized_hops(stats: Dict[str, GoldHopStats], path: str) -> int:
    """
    Write the per-example **realized** structure to JSON, keyed by content fingerprint.

    This is what turns the headline into a dose-response curve: eval joins on the fingerprint and
    stratifies per-example f1 by ``realized_hops``, which matters because ``hop2``'s f1 is a 96/4
    mixture (~4% of gold pairs are adjacent and therefore silently ``hop_inf``). The routable subset is
    the mixture-free number.

    :param stats: A :attr:`GoldHopMaskHolder.stats` mapping.
    :param path: Destination JSON path.

    :returns: The number of examples written.
    """
    import json

    payload = {
        fp: {
            "pairs": [list(p) for p in s.pairs],
            "distances": s.distances,
            "realized_hops": s.realized_hops,  # -1 == GOLD_HOPS_INF (no path)
            "unroutable": s.unroutable,
            "base_edges": s.base_edges,
            "final_edges": s.final_edges,
            "edge_drift": s.edge_drift,
            "converged": s.converged,
        }
        for fp, s in stats.items()
    }
    with open(path, "w") as f:
        json.dump(payload, f)
    return len(payload)


def install_gold_hop_mask(model: torch.nn.Module, gold_hop_fn: GoldHopFn) -> GoldHopMaskHolder:
    """
    Install the gold-aware hop-controlled graph on ``model`` in place.

    Registers a ``forward_pre_hook`` that computes the per-example doc->doc adjacency from the batch's
    live ``input_ids`` (via ``gold_hop_fn``) and stashes it in the returned holder, then points every
    :class:`~olmo_core.nn.attention.document_chunked.DocumentChunkedAttention` module at that holder.
    Layers read the adjacency when their pattern is ``"gold_hop_controlled"``.

    This mirrors :func:`~olmo_core.nn.attention.gold_grad_mask.install_gold_grad_mask`'s hook, and for
    the same reason: the lookup must key off token *content*, so gold identity never rides in the token
    stream. Unlike that function it patches no ``sdpa`` -- the graph is consumed by the mask builder,
    not by a detach.

    ⚠ Not ``torch.compile``-capturable (per-forward Python + a hash). Run with ``--no-compile``.

    :returns: The :class:`GoldHopMaskHolder`. Check ``n_attached > 0``: zero means no layer reads the
        graph and the arm is silently inert.

    :raises ValueError: If a layer's configured ``gold_hops`` disagrees with the hook's arm, or no
        ``"gold_hop_controlled"`` layer exists to read the graph.
    """
    from .document_chunked import DocumentChunkedAttention

    holder = GoldHopMaskHolder()
    # Adopt the fn's realized-structure record, so ``holder.summary()`` reports what was actually built
    # rather than an empty dict (the launcher logs it at the end of the run, and eval stratifies on it).
    fn_stats = getattr(gold_hop_fn, "stats", None)
    if isinstance(fn_stats, dict):
        holder.stats = fn_stats
    fn_counters = getattr(gold_hop_fn, "counters", None)
    if isinstance(fn_counters, dict):
        holder.counters = fn_counters
    fn_misses = getattr(gold_hop_fn, "misses", None)
    if isinstance(fn_misses, list):
        holder.misses = fn_misses

    def pre_hook(module, args, kwargs):
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        if input_ids is None:
            return
        adj = gold_hop_fn(input_ids)
        holder.adjacency = adj.to(device=input_ids.device, dtype=torch.bool)

    holder._hook_handle = model.register_forward_pre_hook(pre_hook, with_kwargs=True)

    fn_hops = getattr(gold_hop_fn, "hops", None)
    n = 0
    n_gold_layers = 0
    for m in model.modules():
        if isinstance(m, DocumentChunkedAttention):
            if m.cross_doc_mode == "gold_hop_controlled":
                n_gold_layers += 1
                if fn_hops is not None and m._pattern.gold_hops != fn_hops:
                    raise ValueError(
                        f"gold-hop arm mismatch: layer {m.layer_idx} is configured with "
                        f"gold_hops={m._pattern.gold_hops} but the installed hook builds "
                        f"gold_hops={fn_hops}. The run would be named after one arm and trained on "
                        "another."
                    )
            m._gold_hop_holder = holder
            n += 1
    if n_gold_layers == 0:
        raise ValueError(
            "install_gold_hop_mask found no attention layer with "
            "cross_doc_mode='gold_hop_controlled'; the hook would compute a graph nothing reads. "
            "Build the model with cross_doc_mode='gold_hop_controlled' first."
        )
    holder.n_attached = n
    return holder
