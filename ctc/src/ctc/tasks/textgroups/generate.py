"""
``textgroups`` data: N short passages, find every triple whose *textual* feature values sum to T.

The textual analogue of :mod:`ctc.tasks.groups4`. Each passage carries a feature value the model
must read off the prose rather than a number printed in it: how many nouns it contains, how many
verbs, how many adjectives, or how many times a chosen common word appears.

The features are chosen to be **dense** (spread across every sentence, not concentrated in one) and
**exactly defined** -- "noun" means "a token in our closed noun lexicon". The lexicons are pairwise
disjoint, so a passage's counts are independent, and per-sentence counts are random, so neither
word count nor sentence count is a proxy for the feature.

**Exactly K triples, guaranteed**, and here the sum-to-target property groups4 avoided *is*
tractable: at G=3, C(N,3) is small enough to both construct and brute-force verify. K disjoint
triples summing to T are planted; a distractor is admitted only if it completes no near-target
triple with any pair already placed, maintained incrementally as a set of forbidden completing
values -- O(1) per candidate, so exactness holds at any N with no global C(N,3) scan.

Scaling is what the ``filler_max`` knob buys: values above roughly T/3 cannot form a target triple
among themselves, so they are placeable without bound and N grows freely while exactness holds.
**T, not N, sets the combinatorial search breadth** -- and it costs passage length.
"""

from __future__ import annotations

import random
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ...data.generators.base import Generator
from ...data.gold import remap_groups, shuffle_with_remap
from ...data.schema import make_document, make_example

__all__ = ["build_example", "sample_values", "FEATURES", "GENERATOR"]

NOUNS: Tuple[str, ...] = tuple(
    """river mountain teacher engine harbor garden lantern soldier merchant valley bridge castle
farmer anchor orchard painter compass meadow captain tavern forest blacksmith fountain carriage
shepherd windmill courtyard violin telescope harvest glacier cathedral scholar musician voyage
saddle beacon quarry chapel wagon sailor tower island desert canyon miner weaver potter hunter
baker library market palace cottage harvester watchman gardener fisherman innkeeper drummer""".split()
)
VERBS: Tuple[str, ...] = tuple(
    """wandered shouted gathered vanished trembled lingered stumbled whispered marched collapsed
drifted arrived departed shattered glimmered echoed faltered soared crumbled scattered returned
circled hesitated rushed paused rested waited knelt listened watched rebuilt explored guarded
studied celebrated argued labored traveled sketched harvested""".split()
)
ADJECTIVES: Tuple[str, ...] = tuple(
    """ancient weary golden silent restless fragile stormy distant crimson hollow gentle ragged
luminous narrow frozen endless bitter radiant crooked solemn vivid humble fierce tranquil scarlet
rugged mournful brisk lavish somber""".split()
)
ADVERBS: Tuple[str, ...] = tuple(
    "quietly suddenly slowly wearily eagerly calmly abruptly softly bravely restlessly".split()
)
PRONOUNS: Tuple[str, ...] = ("They", "She", "He", "It", "We")
#: Common words usable as the ``connector`` feature. Disjoint from every other lexicon, so a
#: connector count is exact.
CONNECTORS: Tuple[str, ...] = ("and", "with", "near", "beyond", "beside", "amid")

POS_LEX = {"nouns": NOUNS, "verbs": VERBS, "adjectives": ADJECTIVES}
FEATURES: Tuple[str, ...] = ("nouns", "verbs", "adjectives", "connector")

_ALL = [w for lex in (NOUNS, VERBS, ADJECTIVES, ADVERBS, CONNECTORS) for w in lex]
_ALL += [p.lower() for p in PRONOUNS]
assert len(_ALL) == len(set(_ALL)), "lexicons must be pairwise disjoint"


# ── passage realisation ─────────────────────────────────────────────────────────────────────────


def _clause(scored: str, k: int, word: Optional[str], rng: random.Random) -> str:
    """
    A clause containing exactly ``k`` of the scored token, with filler drawn only from the other
    lexicons so the count is precise.

    :param scored: Feature name.
    :param k: Exact occurrences of the scored token.
    :param word: The connector string, when ``scored == "connector"``.
    :param rng: Seeded RNG.

    :returns: The clause.

    :raises ValueError: On an unknown feature.
    """
    if scored == "nouns":
        if k == 0:
            return f"{rng.choice(PRONOUNS)} {rng.choice(VERBS)} {rng.choice(ADVERBS)}"
        joined = " and ".join(f"the {rng.choice(NOUNS)}" for _ in range(k))
        return f"{joined.capitalize()} {rng.choice(VERBS)} {rng.choice(ADVERBS)}"
    if scored == "verbs":
        if k == 0:
            return f"The {rng.choice(NOUNS)} stood {rng.choice(ADVERBS)}"
        verbs = " and ".join(rng.choice(VERBS) for _ in range(k))  # drawn before the pronoun
        return f"{rng.choice(PRONOUNS)} {verbs}"
    if scored == "adjectives":
        if k == 0:
            return f"{rng.choice(PRONOUNS)} {rng.choice(VERBS)} {rng.choice(ADVERBS)}"
        adjs = ", ".join(rng.choice(ADJECTIVES) for _ in range(k))
        return f"The {adjs} {rng.choice(NOUNS)} {rng.choice(VERBS)}"
    if scored == "connector":
        nouns = [rng.choice(NOUNS) for _ in range(k + 1)]
        parts = [f"the {nouns[0]}"]
        for i in range(k):
            parts += [word or "", f"the {nouns[i + 1]}"]
        return f"{' '.join(parts).capitalize()} {rng.choice(VERBS)}"
    raise ValueError(f"unknown feature {scored!r}")


def _passage(
    scored: str, value: int, word: Optional[str], rng: random.Random, max_sents: int = 80
) -> str:
    """
    A passage whose scored feature count is exactly ``value``, spread across several sentences so
    no single sentence carries it all.

    :param scored: Feature name.
    :param value: The exact total count.
    :param word: Connector string, when applicable.
    :param rng: Seeded RNG.
    :param max_sents: Sentence cap; the tail is merged into earlier sentences beyond it.

    :returns: The passage.
    """
    parts: List[int] = []
    remaining = value
    while remaining > 0:
        take = min(remaining, rng.randint(1, 3))
        parts.append(take)
        remaining -= take
    parts += [0] * rng.randint(0, 2)  # zero-count sentences decouple length from the count
    rng.shuffle(parts)
    if len(parts) < 3:
        parts += [0] * (3 - len(parts))
    if len(parts) > max_sents:
        head, tail = parts[:max_sents], parts[max_sents:]
        for i, extra in enumerate(tail):
            head[i % max_sents] += extra
        parts = head
    return " ".join(_clause(scored, k, word, rng).strip().rstrip(".") + "." for k in parts)


def count_feature(text: str, scored: str, word: Optional[str]) -> int:
    """
    Recompute a passage's feature count from its text.

    Used as a generation-time assertion, and by the audit stage to confirm shipped data still says
    what its metadata claims.

    :param text: The passage.
    :param scored: Feature name.
    :param word: Connector string, when applicable.

    :returns: The count.
    """
    tokens = [t.strip(".,").lower() for t in text.split()]
    if scored == "connector":
        return sum(1 for t in tokens if t == word)
    return sum(1 for t in tokens if t in POS_LEX[scored])


# ── value construction ──────────────────────────────────────────────────────────────────────────


def _near_target_triples(values: Sequence[int], target: int, margin: int) -> int:
    return sum(1 for c in combinations(values, 3) if abs(sum(c) - target) <= margin)


def _partition(total: int, parts: int, lo: int, hi: int, rng: random.Random) -> Optional[List[int]]:
    """
    :param total: Sum to hit.
    :param parts: How many integers.
    :param lo: Minimum per part.
    :param hi: Maximum per part.
    :param rng: Seeded RNG.

    :returns: ``parts`` integers in ``[lo, hi]`` summing to ``total``, or ``None`` if infeasible.
    """
    if not parts * lo <= total <= parts * hi:
        return None
    vals: List[int] = []
    remaining = total
    for i in range(parts):
        slots_left = parts - i - 1
        v_lo = max(lo, remaining - slots_left * hi)
        v_hi = min(hi, remaining - slots_left * lo)
        if v_lo > v_hi:
            return None
        v = rng.randint(v_lo, v_hi)
        vals.append(v)
        remaining -= v
    rng.shuffle(vals)
    return vals


def sample_values(
    n_docs: int,
    n_groups: int,
    group_size: int,
    target: int,
    tol: int,
    cmin: int,
    cmax: int,
    filler_max: int,
    rng: random.Random,
    separation: int = 0,
    max_attempts: int = 2000,
) -> Tuple[List[int], List[List[int]]]:
    """
    Draw feature values: exactly ``n_groups`` disjoint triples summing to ``target``, and no others.

    :param n_docs: Total passages N.
    :param n_groups: Gold triples K.
    :param group_size: Passages per group G.
    :param target: The sum each gold group hits, T.
    :param tol: Slack W; a group is gold if its sum is within W of T.
    :param cmin: Minimum feature count, and the floor for gold members.
    :param cmax: Maximum count for gold members; below T so gold reads as ordinary.
    :param filler_max: Maximum count for distractors; may far exceed ``cmax``.
    :param rng: Seeded RNG.
    :param separation: Isolation margin M. Guarantees no non-gold triple sums within M of the
        target, so gold sits in a clean gap. **This makes the task easier** -- the model need only
        find triples summing to roughly T rather than solve exact subset-sum against near-miss
        decoys. M=0 keeps the hard construction.
    :param max_attempts: Restarts before giving up.

    :returns: ``(values, gold_index_groups)`` in construction order.

    :raises ValueError: If ``n_docs`` is too small for K groups of G.
    :raises RuntimeError: If no assignment was found -- try a more central target, a wider
        ``[cmin, cmax]``, a smaller ``n_docs``, or fewer groups.
    """
    G, K, T = group_size, n_groups, target
    margin = max(tol, separation)
    n_distract = n_docs - K * G
    if n_distract < 0:
        raise ValueError(f"num_docs={n_docs} too small for {K} groups of {G}")

    def forbid(chosen: Sequence[int], new_v: int, forbidden: Set[int]) -> None:
        """Block every value completing a near-target triple with ``(new_v, y)``."""
        for y in chosen:
            center = T - (new_v + y)
            forbidden.update(range(center - margin, center + margin + 1))

    for _ in range(max_attempts):
        # K disjoint planted groups. Every value in a new group is more than M from every value
        # already placed, so a mixed triple (2 from group A, 1 from B) sums to T - a_i + b_k, which
        # is within M of T only if |a_i - b_k| <= M -- excluded. The brute check below then always
        # passes and is just a cheap confirmation.
        gold_groups: List[List[int]] = []
        used: List[int] = []
        for _g in range(K):
            placed = None
            for _try in range(400):
                parts = _partition(T, G, cmin, cmax, rng)
                if parts is None or any(abs(v - u) <= margin for v in parts for u in used):
                    continue
                placed = parts
                break
            if placed is None:
                break
            gold_groups.append(placed)
            used.extend(placed)
        if len(gold_groups) < K:
            continue
        gold_values = [v for g in gold_groups for v in g]
        if _near_target_triples(gold_values, T, margin) != K:
            continue

        chosen: List[int] = []
        forbidden: Set[int] = set()
        for v in gold_values:
            forbid(chosen, v, forbidden)
            chosen.append(v)

        distractors: List[int] = []
        for _attempt in range(n_distract * 300 + 5000):
            if len(distractors) == n_distract:
                break
            c = rng.randint(cmin, filler_max)
            if c in forbidden:
                continue
            forbid(chosen, c, forbidden)
            chosen.append(c)
            distractors.append(c)
        if len(distractors) < n_distract:
            continue

        values = gold_values + distractors
        # Safety net for small corpora; for large N the incremental argument above already holds.
        if n_docs <= 150:
            assert _near_target_triples(values, T, margin) == K, "exactness check failed"
        return values, [list(range(i * G, i * G + G)) for i in range(K)]

    raise RuntimeError(
        f"could not assemble values after {max_attempts} attempts; try a more central target, a "
        f"wider [{cmin}, {cmax}], a smaller num_docs, or fewer groups"
    )


def build_example(
    rng: random.Random,
    *,
    num_docs: int,
    num_groups: int = 2,
    group_size: int = 3,
    target: int = 70,
    tolerance: int = 0,
    separation: int = 0,
    feature: str = "mixed",
    cmin: int = 4,
    cmax: int = 40,
    filler_max: Optional[int] = None,
) -> Dict:
    """
    Build one textgroups example.

    :param rng: Seeded RNG.
    :param num_docs: Corpus size N.
    :param num_groups: Gold triples K.
    :param group_size: Passages per group G.
    :param target: Sum each gold group hits, T. Larger T widens the combinatorial search (more
        small counts can combine) at the cost of longer passages.
    :param tolerance: Slack W; 0 means exact.
    :param separation: Isolation margin M; see :func:`sample_values`.
    :param feature: One of :data:`FEATURES`, or ``"mixed"`` to pick per example.
    :param cmin: Minimum feature count.
    :param cmax: Maximum count for gold members.
    :param filler_max: Maximum count for distractors; defaults to ``max(cmax, target)``, widened
        when a separation margin is set so an always-safe band above ``T - 2*cmin + M`` stays
        populated.

    :returns: A unified-format example whose ``gold_doc_indices`` holds 1-based triples, and whose
        ``_meta.counts`` holds every passage's feature value in display order so the CoT builders
        can narrate without re-deriving the lexicon.

    :raises RuntimeError: If a passage could not be realised at its required count.
    """
    if feature == "mixed":
        feature = rng.choice(FEATURES)
    word = rng.choice(CONNECTORS) if feature == "connector" else None

    if filler_max is None:
        filler_max = max(cmax, target)
        if separation > 0:
            # A margin widens the forbidden band, so the distractor range could be fully excluded.
            # Any value above T - 2*cmin + M can never complete a near-target triple; +30 gives
            # that always-safe band enough width for randint to hit it reliably.
            filler_max = max(filler_max, target - 2 * cmin + separation + 30)

    values, gold_groups = sample_values(
        num_docs,
        num_groups,
        group_size,
        target,
        tolerance,
        cmin,
        cmax,
        filler_max,
        rng,
        separation=separation,
    )

    texts: List[str] = []
    for v in values:
        for _ in range(20):
            text = _passage(feature, v, word, rng)
            if count_feature(text, feature, word) == v:
                break
        else:
            raise RuntimeError(f"could not realise value {v} for feature {feature!r}")
        texts.append(text)

    items, old_to_new = shuffle_with_remap(list(zip(texts, values)), rng=rng, base=1)
    phrase = (
        f'the number of times the word "{word}" appears'
        if feature == "connector"
        else f"the number of {feature}"
    )
    tol_clause = f" (within {tolerance} of {target})" if tolerance else f" (exactly {target})"
    return make_example(
        documents=[make_document(text) for text, _ in items],
        queries=[
            f"Each passage has a value: {phrase}. Find every group of {group_size} passages "
            f"whose values add up to {target}{tol_clause}."
        ],
        source="textgroups",
        gold=remap_groups(gold_groups, old_to_new),
        meta={
            "feature": feature,
            "word": word,
            "target": target,
            "tolerance": tolerance,
            "group_size": group_size,
            "num_groups": num_groups,
            "filler_max": filler_max,
            "separation": separation,
            "counts": [v for _, v in items],
        },
    )


GENERATOR = Generator(
    name="textgroups",
    task="textgroups",
    source="textgroups",
    build_example=build_example,
    defaults={
        "num_docs": 40,
        "num_groups": 2,
        "group_size": 3,
        "target": 70,
        "tolerance": 0,
        "separation": 0,
        "feature": "mixed",
        "cmin": 4,
        "cmax": 40,
        "filler_max": None,
    },
    notes="pure synthetic; the feature must be read from prose, not from a printed number",
)
