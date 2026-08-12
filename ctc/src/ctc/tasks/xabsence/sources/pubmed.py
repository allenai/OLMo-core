"""
``xabsence`` over PubMed claim sentences -- the shipped suite row (BUILD_MATRIX row 22).

Each example draws ``P`` claim/paraphrase twins plus ``k`` orphans from one pool. The twins are
split across corpora A and B; the orphans go into one corpus only, and the model has to find them
by comparing meaning across the boundary rather than by reading any single document.

.. note::
   **BUILD_MATRIX blocker B3 is still open.** The only staged pool,
   ``/scratch/users/prasann/corpus-reasoning/data/xabsence_pool_pubmed.jsonl``, holds **659
   entries**, and the 32k rung needs ``P + k = 492`` distinct entries *per example*. Every example
   at that rung would therefore be three quarters of the same pool, which is a corpus with far
   fewer distinct questions than rows. The build report's ``duplicates`` count is where this shows
   up; mine a larger pool (``-C base_url=...``) before building the long rungs for anything graded.

.. note::
   **The ladder is measured, and it is not BUILD_MATRIX's.** Row 22 estimated ~95 tokens per pair,
   i.e. ~47.5 per document, giving ``P18/P39/P81/P165/P333`` -> 39/81/165/333/669 documents. The
   shipped ``xabsence_eval_pubmed_p{8,18,48}_k3.jsonl`` files measure 772/1394/3424 median Qwen3
   tokens at 19/39/99 documents, i.e. ``tokens ~= 120 + 33.3 * n_docs`` -- so the estimate
   overshoots by ~1.4x and the shipped ``P333`` file is a ~22k-token file labelled 32k. The row in
   :data:`ctc.data.ladders.LADDERS` is the inverted fit.
"""

from __future__ import annotations

import random
from typing import Dict, List, Mapping, Optional, Tuple

from ....data.generators.base import Generator
from ....data.sources import paraphrase as source
from ..generate import assemble, split_for_docs

__all__ = ["build_example", "nested_ladder", "GENERATOR"]

#: Corpus tag, as shipped in ``xabsence_{train,eval}_pubmed_p{P}_k3.jsonl``.
SOURCE_TAG = "xabsence_pubmed"


def _draw(
    corpus: source.ParaphrasePool,
    pairs: int,
    num_unmatched: int,
    decoys_per_unmatched: int,
    rng: random.Random,
) -> Optional[Tuple[List[source.ParaphrasePair], List[source.ParaphrasePair], List[str]]]:
    """
    Draw one example's entries: the orphans, their sides, and the matched pairs around them.

    The matched list is ordered **decoys first**, which is what lets the nested ladder take a
    prefix: every rung, however short, keeps the lexical counterparts that stop the orphan being
    findable by word overlap.

    :param corpus: The twin pool.
    :param pairs: Matched pairs wanted, ``P``.
    :param num_unmatched: Orphans wanted, ``k``.
    :param decoys_per_unmatched: Lexical counterparts forced in per orphan; see the module warning
        in :mod:`ctc.data.sources.paraphrase`.
    :param rng: Seeded RNG.

    :returns: ``(matched, unmatched, sides)``, or ``None`` when the pool cannot supply ``P + k``
        distinct entries. A skip rather than a short example: an example built from fewer pairs
        than its rung asks for would sit under its own label.
    """
    if len(corpus) < pairs + num_unmatched:
        return None
    orphans = rng.sample(range(len(corpus)), num_unmatched)
    sides = [rng.choice("AB") for _ in orphans]

    taken = set(orphans)
    forced: List[int] = []
    for index, side in zip(orphans, sides):
        for candidate in corpus.decoys(side, index, decoys_per_unmatched):
            if candidate not in taken:
                taken.add(candidate)
                forced.append(candidate)
    forced = forced[:pairs]

    rest = [i for i in range(len(corpus)) if i not in taken]
    matched = forced + rng.sample(rest, pairs - len(forced))
    return (
        [corpus.pairs[i] for i in matched],
        [corpus.pairs[i] for i in orphans],
        sides,
    )


def build_example(
    rng: random.Random,
    *,
    corpus: source.ParaphrasePool,
    num_docs: int,
    num_unmatched: int = 3,
    decoys_per_unmatched: int = 2,
) -> Optional[Dict]:
    """
    Build one xabsence example.

    :param rng: Seeded RNG.
    :param corpus: The twin pool. **Drawn from, not consumed**: one entry can back many examples,
        so this generator takes no example index.
    :param num_docs: Corpus size N; converted to ``P`` pairs by
        :func:`ctc.tasks.xabsence.generate.split_for_docs`.
    :param num_unmatched: Orphans per example, ``k``. Fixed rather than Bernoulli because the model
        is told nothing about how many there are and the pre-migration ladder held it at 3.
    :param decoys_per_unmatched: Matched pairs forced in per orphan because their cross-corpus form
        is lexically nearest to it. **Not tuning**: without them the orphan is the one document
        with no lexically close counterpart, and a word-overlap ranking finds it (0.60 at this
        rung, against a 0.05 chance baseline). Two is the measured optimum on the staged pool --
        more decoys crowd the example with high-overlap documents and the signal comes back.

    :returns: A unified-format example with 0-based gold positions, or ``None`` when the pool was
        too small for this rung.

    :raises ValueError: If ``num_docs`` cannot hold ``num_unmatched`` orphans and a matched pair.
    """
    pairs, unmatched_count = split_for_docs(num_docs, num_unmatched)
    drawn = _draw(corpus, pairs, unmatched_count, decoys_per_unmatched, rng)
    if drawn is None:
        return None
    matched, unmatched, sides = drawn
    return assemble(
        matched,
        unmatched,
        rng=rng,
        source=SOURCE_TAG,
        sides=sides,
        meta={
            "num_pairs": pairs,
            "num_unmatched": unmatched_count,
            "decoys_per_unmatched": decoys_per_unmatched,
            "provenance": corpus.provenance,
        },
    )


def nested_ladder(
    rng: random.Random,
    *,
    index: int,
    corpus: source.ParaphrasePool,
    rungs: Mapping[str, int],
    num_docs: int,
    num_unmatched: int = 3,
    decoys_per_unmatched: int = 2,
) -> Optional[Dict[str, Dict]]:
    """
    Build one row at every rung at once, holding the orphans fixed and dropping matched pairs.

    The generic nested shrink drops *random* documents, and dropping one half of a matched pair
    leaves its partner genuinely unmatched -- a correct answer the label does not list. Growing and
    shrinking by whole pairs is safe, so:

    * ``P_max + k`` entries are drawn once, and the ``k`` orphans and their A/B sides are **fixed
      across every rung**, so the question and its answer text are identical rung to rung;
    * each shorter rung keeps a *prefix* of the same matched list, which ``_draw`` orders decoys
      first -- so its documents are a subset of every longer rung's *and* the shortest rung, where
      the lexical shortcut is strongest, still holds every orphan's counterpart.

    :param rng: Seeded RNG for this row.
    :param index: Example counter, for the caller's bookkeeping.
    :param corpus: The twin pool.
    :param rungs: ``label -> documents``, any order.
    :param num_docs: Ignored; the rung table supplies the sizes. Present so the ladder builder can
        pass one resolved config to both entry points.
    :param num_unmatched: Orphans per example.
    :param decoys_per_unmatched: See :func:`build_example`.

    :returns: ``{rung label: example}``, or ``None`` when the pool could not supply the largest
        rung.

    :raises ValueError: If the shortest rung cannot hold every orphan's decoys, which would leave
        the rungs grading the same question under different amounts of shortcut.
    """
    del num_docs  # the rung table is the authority here
    split = {label: split_for_docs(size, num_unmatched) for label, size in rungs.items()}
    orphans = {count for _, count in split.values()}
    if len(orphans) > 1:
        # A rung with a different orphan count grades a different question, and the audit
        # would report it as "the rungs are not measuring the same thing" with no clue why.
        raise ValueError(
            f"rung sizes {sorted(rungs.values())} disagree in parity, so they would carry "
            f"{sorted(orphans)} unmatched claims respectively rather than one shared answer"
        )
    sizes = {label: pairs for label, (pairs, _) in split.items()}
    unmatched_count = orphans.pop()
    largest = max(sizes.values())
    needed = unmatched_count * decoys_per_unmatched
    if min(sizes.values()) < needed:
        raise ValueError(
            f"the shortest rung holds {min(sizes.values())} matched pairs but "
            f"{unmatched_count} orphans need {needed} lexical decoys; a rung without them is "
            "solvable by word overlap while the longer rungs are not"
        )
    drawn = _draw(corpus, largest, unmatched_count, decoys_per_unmatched, rng)
    if drawn is None:
        return None
    matched, unmatched, sides = drawn

    out: Dict[str, Dict] = {}
    for label, pairs in sizes.items():
        # One stream per rung, so adding a rung never perturbs another rung's ordering.
        rung_rng = random.Random(f"{index}:xabsence:{label}")
        out[label] = assemble(
            matched[:pairs],
            unmatched,
            rng=rung_rng,
            source=SOURCE_TAG,
            sides=sides,
            meta={
                "num_pairs": pairs,
                "num_unmatched": unmatched_count,
                "decoys_per_unmatched": decoys_per_unmatched,
                "rung": label,
                "provenance": corpus.provenance,
            },
        )
    return out


GENERATOR = Generator(
    name="xabsence",
    task="xabsence",
    source=SOURCE_TAG,
    build_example=build_example,
    defaults={"num_docs": 59, "num_unmatched": 3, "decoys_per_unmatched": 2},
    corpus=source.load_pool,
    corpus_defaults={
        "pool_path": None,
        "mode": "paraphrase",
        "num_abstracts": 8_000,
        "num_claims": 30_000,
        "seed": 42,
        "model": "Qwen2.5-14B-Instruct",
        "base_url": None,
        "max_concurrent": 25,
        "max_overlap": 0.22,
        "cache_dir": "data/.cache/ctc_llm",
    },
    # Dropping a random document orphans its partner and invents gold: see the note in
    # ctc.tasks.xabsence.generate. Whole pairs can go, which is what nested_ladder does.
    shrink_safe=False,
    build_ladder=nested_ladder,
    notes="PubMed claim/paraphrase twins across two corpora; the ladder drops whole matched pairs",
)
