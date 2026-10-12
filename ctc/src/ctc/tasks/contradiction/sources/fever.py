"""
``contra_fever`` -- the held-out contradiction ladder. FEVER claims instead of PubMed.

Same contract, same grader, different corpus: gold is a REFUTES row's ``(false claim, refuting
Wikipedia sentence)``, which contradicts by construction rather than by a model's writing. That is
what makes it a clean out-of-distribution probe for a task whose in-distribution gold is
model-written -- a gap here is a *source* effect, not a labelling-process effect.

Three distractor layers are available, and the shipped ladder uses none of them:

* **hard NEI** claims about the gold's own Wikipedia page. NEI claims are unverifiable against
  Wikipedia by definition, so unlike SUPPORTS evidence they rarely state the truth a REFUTES claim
  denies. Not risk-free: a broad REFUTES claim ("X does not have an acting career") can still be
  contradicted by any NEI claim that mentions X acting.
* **decoy clusters** on non-gold pages, which may safely contain SUPPORTS ``(claim, evidence)``
  tuples -- pair-shaped and consistent, so "find the dense topical island" stops working.
* **random fillers** to top up.

.. note::
   **``plain`` is the default here, and that is the difficulty-matched build.** The pre-migration
   generator defaulted the hard distractors ON, but every shipped ``contradiction_eval_fever_plain_*``
   file -- the only FEVER variant the results grid has ever used -- was built with them off, so
   that its difficulty matches the PubMed ladder it is compared against. Turning them on builds a
   *harder* ladder whose numbers are not comparable with the published OOD column.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Set

from ....data.generators.base import Generator
from ....data.sources import fever as source
from ..generate import assemble, place_pairs

__all__ = ["build_example", "GENERATOR"]


def _take(candidates: Sequence[str], count: int, seen: Set[str], budget: int, rng) -> List[str]:
    """
    :param candidates: Claim texts to draw from.
    :param count: How many to take.
    :param seen: Texts already placed; mutated.
    :param budget: Remaining document slots.
    :param rng: Seeded RNG.

    :returns: The drawn claims.
    """
    if count <= 0 or budget <= 0:
        return []
    pool = [c for c in candidates if c not in seen]
    rng.shuffle(pool)
    out: List[str] = []
    for claim in pool[:count]:
        if len(out) >= budget:
            break
        out.append(claim)
        seen.add(claim)
    return out


def build_example(
    rng: random.Random,
    *,
    index: int,
    corpus: source.FeverPool,
    num_docs: int,
    num_pairs: int = 3,
    hard_nei_per_pair: int = 0,
    decoy_support_pairs: int = 0,
    use_decoys: bool = False,
) -> Optional[Dict]:
    """
    Build one FEVER contradiction example.

    :param rng: Seeded RNG.
    :param index: Example counter; gold pairs are consumed in pool order.
    :param corpus: The FEVER pool.
    :param num_docs: Corpus size N.
    :param num_pairs: Gold pairs K.
    :param hard_nei_per_pair: Topic-matched NEI claims per cluster. 0 reproduces the shipped
        ``plain`` build -- see the module note before raising it.
    :param decoy_support_pairs: SUPPORTS tuples per decoy cluster. **Only ever in decoy clusters**:
        a SUPPORTS sentence from a gold page often states the very truth the gold REFUTES claim
        denies, which is an unlabelled contradiction pair and therefore a false negative.
    :param use_decoys: Fill the remainder with topical decoy clusters instead of flat random
        claims.

    :returns: A unified-format example with 1-based gold pairs, or ``None`` when the pool is
        exhausted.

    :raises ValueError: If ``num_docs`` cannot hold the gold claims.
    :raises RuntimeError: If the gold clusters alone overflow ``num_docs`` -- lower
        ``hard_nei_per_pair`` rather than letting the example come out over-long.
    """
    if num_docs < 2 * num_pairs:
        raise ValueError(f"num_docs={num_docs} cannot hold {num_pairs} gold pairs")
    start = index * num_pairs
    if start + num_pairs > len(corpus.pairs):
        return None
    picked = corpus.pairs[start : start + num_pairs]

    statements, positions = place_pairs([(claim, evidence) for claim, evidence, _ in picked])
    seen: Set[str] = set(statements)
    gold_pages = {page for _, _, page in picked}
    for _, _, page in picked:
        statements.extend(
            _take(
                corpus.nei_by_page.get(page, ()),
                hard_nei_per_pair,
                seen,
                num_docs - len(statements),
                rng,
            )
        )
    if len(statements) > num_docs:
        raise RuntimeError(
            f"gold clusters total {len(statements)} > num_docs={num_docs}; lower "
            "hard_nei_per_pair or raise num_docs"
        )

    if use_decoys and len(statements) < num_docs:
        pages = [p for p in corpus.pages if p not in gold_pages]
        rng.shuffle(pages)
        for page in pages:
            budget = num_docs - len(statements)
            if budget <= 0:
                break
            statements.extend(
                _take(corpus.nei_by_page.get(page, ()), hard_nei_per_pair, seen, budget, rng)
            )
            budget = num_docs - len(statements)
            for claim, evidence in corpus.support_pairs_by_page.get(page, ())[:decoy_support_pairs]:
                if budget < 2 or claim in seen or evidence in seen:
                    continue
                statements.extend([claim, evidence])
                seen.update([claim, evidence])
                budget -= 2

    if len(statements) < num_docs:
        need = num_docs - len(statements)
        candidates = rng.sample(corpus.fillers, min(need * 3, len(corpus.fillers)))
        picked_fillers = [c for c in candidates if c not in seen][:need]
        if len(picked_fillers) < need:
            return None
        statements.extend(picked_fillers)

    return assemble(
        statements,
        positions,
        source="fever",
        rng=rng,
        meta={
            "num_pairs": num_pairs,
            "variant": "plain" if not (hard_nei_per_pair or use_decoys) else "hard",
        },
    )


GENERATOR = Generator(
    name="contra_fever",
    task="contradiction",
    source="fever",
    build_example=build_example,
    defaults={
        "num_docs": 100,
        "num_pairs": 3,
        "hard_nei_per_pair": 0,
        "decoy_support_pairs": 0,
        "use_decoys": False,
    },
    corpus=source.load_pool,
    corpus_defaults={"split": "train", "seed": 42, "dataset": "copenlu/fever_gold_evidence"},
    indexed=True,
    eval_only=True,
    notes="held-out: FEVER claims graded by the contradiction spec; `plain` build",
)
