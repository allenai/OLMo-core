"""
``redundancy`` data: N PubMed claims, find every pair stating the *same* fact.

Contradiction's mirror image, and deliberately its structural twin: same corpus, same document
shape, same 1-based pair gold, same parser and scorer. Only the relation differs -- ``S'`` restates
``S`` instead of denying it -- which is what makes the pair a controlled comparison rather than two
unrelated tasks.

The one structural difference from :mod:`ctc.tasks.strmatch`, the other member of the pair family
ported alongside it, is where the criterion lives. Redundancy's is **fixed**: the instruction is
the whole ask and ``queries`` is empty, so every example asks the same question. Strmatch's varies
per example and is carried in ``queries[0]``. Collapsing the two -- giving redundancy a query, or
freezing strmatch's -- produces data that still validates and still scores, and silently turns one
task into the other; ``ctc/tests/data/test_pair_criteria.py`` is what stops that.

Mining and the pool live in :mod:`ctc.data.sources.pubmed_redundancy` (they need a model); this
module is the assembly, and the assembly is where the false-negative control lives.

.. note::
   This ladder was **dropped from the pre-migration suite** on 2026-07-19 as "LLM-serving-bound"
   (``BUILD_MATRIX.md`` row 17), so unlike contradiction it has no calibrated rung ladder from that
   era and no shipped rung files -- only ``redundancy_{train,eval}_pubmed_both_n{20,100}_*.jsonl``
   at a single size. Its ladder row here is derived from contradiction's measured one, which is the
   same corpus at the same document shape; see :data:`ctc.data.ladders.CALIBRATION`.

.. warning::
   **The generic shrink cannot make this task's shorter rungs.**
   :func:`ctc.data.build.shrink` protects gold and drops random non-gold documents -- but a planted
   hard negative is a *pair*, and half of one is an ordinary filler. Shrinking a 784-document
   example down to 46 keeps all K gold pairs whole and leaves essentially none of the H hard
   negatives intact, so every rung below the longest would grade a corpus with its decoys removed:
   the task gets *easier* as the context gets shorter, for a reason that is not context length.
   :func:`nested_ladder` builds the rungs by prefix instead, which is why this generator declares
   ``shrink_safe=False``.

.. note::
   The pre-migration example carried ``_hardneg_pairs``, the planted non-redundant pairs' positions.
   That field cannot survive :func:`ctc.data.build.shrink`, which drops distractors and remaps only
   the gold structure and ``hard_neg_indices`` -- a carried copy would be silently wrong at every
   rung but the longest. The positions are emitted as ``hard_neg_indices`` instead, flat and in the
   spec's own 1-based numbering, which shrink does remap; a hard-negative pair that loses one half
   to a shrink leaves the other in the list, which is the honest record of what the shorter rung
   actually contains.
"""

from __future__ import annotations

import random
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ...data.generators.base import Generator
from ...data.gold import remap, remap_groups, shuffle_with_remap
from ...data.schema import make_document, make_example
from ...data.sources import pubmed_redundancy as source
from ...data.sources.pubmed import sample_fillers

__all__ = ["build_example", "nested_ladder", "GENERATOR"]


def _place(
    pairs: Sequence[source.RedundantPair], hardnegs: Sequence[source.HardNegativePair]
) -> Tuple[List[str], List[Tuple[int, int]], List[int]]:
    """
    Lay the planted sentences out at the front of a corpus, before the shuffle.

    :param pairs: Gold redundant pairs.
    :param hardnegs: Planted same-abstract non-redundant pairs.

    :returns: ``(statements, gold_positions, hardneg_positions)`` in construction order.
    """
    statements: List[str] = []
    gold_positions: List[Tuple[int, int]] = []
    hardneg_positions: List[int] = []
    for pair in pairs:
        gold_positions.append((len(statements), len(statements) + 1))
        statements.extend([pair.claim, pair.paraphrase])
    for hard in hardnegs:
        hardneg_positions.extend([len(statements), len(statements) + 1])
        statements.extend([hard.first, hard.second])
    return statements, gold_positions, hardneg_positions


def _emit(
    statements: Sequence[str],
    gold_positions: Sequence[Tuple[int, int]],
    hardneg_positions: Sequence[int],
    *,
    rng: random.Random,
    meta: Mapping[str, object],
) -> Dict:
    """
    Shuffle a corpus of claims and record where the planted pairs landed.

    :param statements: Claim texts in construction order -- gold, then hard negatives, then filler.
    :param gold_positions: Construction-order gold pair positions.
    :param hardneg_positions: Construction-order hard-negative positions, flat.
    :param rng: Seeded RNG; consumes exactly one shuffle.
    :param meta: Per-example metadata.

    :returns: A unified-format example whose ``gold_doc_indices`` holds **1-based** pairs.
    """
    items, old_to_new = shuffle_with_remap(list(statements), rng=rng, base=1)
    return make_example(
        documents=[make_document(text) for text in items],
        queries=[],
        answers=[],
        source="pubmed_redundancy",
        gold=remap_groups([list(pair) for pair in gold_positions], old_to_new),
        meta=dict(meta),
        hard_neg_indices=remap(hardneg_positions, old_to_new),
    )


def _slice_pool(
    corpus: source.RedundancyPool, index: int, num_pairs: int, num_hardneg: int
) -> Optional[Tuple[Sequence[source.RedundantPair], Sequence[source.HardNegativePair]]]:
    """
    :param corpus: The pool.
    :param index: Example counter.
    :param num_pairs: Gold pairs K.
    :param num_hardneg: Hard-negative pairs H.

    :returns: This example's slice of each pool, or ``None`` when either is exhausted.
    """
    gold_start, hard_start = index * num_pairs, index * num_hardneg
    if gold_start + num_pairs > len(corpus.pairs):
        return None
    if hard_start + num_hardneg > len(corpus.hardnegs):
        return None
    return (
        corpus.pairs[gold_start : gold_start + num_pairs],
        corpus.hardnegs[hard_start : hard_start + num_hardneg],
    )


def build_example(
    rng: random.Random,
    *,
    index: int,
    corpus: source.RedundancyPool,
    num_docs: int,
    num_pairs: int = 3,
    num_hardneg: int = 6,
) -> Optional[Dict]:
    """
    Build one redundancy example.

    :param rng: Seeded RNG.
    :param index: Example counter. Gold and hard-negative pairs are consumed in pool order rather
        than sampled, so no pair is silently reused inside a split while another goes unused.
    :param corpus: The redundancy pool.
    :param num_docs: Corpus size N -- the scaling axis.
    :param num_pairs: Gold redundant pairs K.
    :param num_hardneg: Planted same-abstract non-redundant pairs H. Twice K pre-migration, and the
        ratio is the task: fewer hard negatives makes "which two sentences are about one topic?"
        a sufficient answer.

    :returns: A unified-format example with 1-based gold pairs, or ``None`` when either pool is
        exhausted or the fillers cannot fill the corpus.

    :raises ValueError: If ``num_docs`` cannot hold the planted pairs.
    """
    if num_docs < 2 * (num_pairs + num_hardneg):
        raise ValueError(
            f"num_docs={num_docs} cannot hold {num_pairs} gold + {num_hardneg} hard-negative pairs"
        )
    sliced = _slice_pool(corpus, index, num_pairs, num_hardneg)
    if sliced is None:
        return None
    pairs, hardnegs = sliced
    statements, gold_positions, hardneg_positions = _place(pairs, hardnegs)

    # THE false-negative control, and it covers the hard negatives as well as gold: a filler drawn
    # from an abstract that already contributed a sentence to this example may restate that
    # sentence's finding, which would be an unlabelled second redundant pair -- scored as a model
    # error every time it is found. PubMed's scale means two unrelated abstracts almost never state
    # the same fact; the same abstract very well might.
    fillers = sample_fillers(
        corpus.fillers,
        exclude_abstracts=[p.abstract_id for p in pairs] + [h.abstract_id for h in hardnegs],
        exclude_texts=statements,
        count=num_docs - len(statements),
        rng=rng,
    )
    if len(statements) + len(fillers) < num_docs:
        return None

    return _emit(
        statements + fillers,
        gold_positions,
        hardneg_positions,
        rng=rng,
        meta={
            "num_pairs": num_pairs,
            "num_hardneg": num_hardneg,
            "modes": sorted({p.mode for p in pairs}),
            "provenance": corpus.provenance,
        },
    )


def nested_ladder(
    rng: random.Random,
    *,
    index: int,
    rungs: Mapping[str, int],
    corpus: source.RedundancyPool,
    num_docs: int = 0,
    num_pairs: int = 3,
    num_hardneg: int = 6,
) -> Optional[Dict[str, Dict]]:
    """
    Build every rung of one row at once, keeping the planted hard-negative pairs whole.

    Fillers are drawn once for the longest rung and each rung takes a prefix of them, on top of the
    same gold pairs and the same hard negatives. A shorter rung's documents are then a subset of the
    longer one's, every rung grades the same K questions, and -- unlike the generic shrink -- every
    rung still contains all H decoy pairs. The hard negatives do not thin out with the corpus, which
    means the *ratio* of decoys to fillers falls as the context grows; that is the same direction as
    a real long-context corpus and the opposite of the shrink, which removed them fastest at the
    short rungs where they matter most.

    :param rng: Seeded RNG for the filler draw.
    :param index: Example counter; selects this row's pairs and keys the per-rung shuffles.
    :param rungs: rung label -> document count.
    :param corpus: The redundancy pool.
    :param num_docs: Ignored -- the ladder sets the size per rung.
    :param num_pairs: Gold redundant pairs K.
    :param num_hardneg: Planted non-redundant pairs H.

    :returns: rung label -> example, or ``None`` when a pool is exhausted or the fillers run out.

    :raises ValueError: If the shortest rung cannot hold the planted pairs.
    """
    smallest = min(rungs.values())
    if smallest < 2 * (num_pairs + num_hardneg):
        raise ValueError(
            f"the {smallest}-document rung cannot hold {num_pairs} gold + {num_hardneg} "
            "hard-negative pairs"
        )
    sliced = _slice_pool(corpus, index, num_pairs, num_hardneg)
    if sliced is None:
        return None
    pairs, hardnegs = sliced
    statements, gold_positions, hardneg_positions = _place(pairs, hardnegs)

    longest = max(rungs.values())
    fillers = sample_fillers(
        corpus.fillers,
        exclude_abstracts=[p.abstract_id for p in pairs] + [h.abstract_id for h in hardnegs],
        exclude_texts=statements,
        count=longest - len(statements),
        rng=rng,
    )
    if len(statements) + len(fillers) < longest:
        return None

    meta = {
        "num_pairs": num_pairs,
        "num_hardneg": num_hardneg,
        "modes": sorted({p.mode for p in pairs}),
        "provenance": corpus.provenance,
    }
    return {
        label: _emit(
            statements + fillers[: size - len(statements)],
            gold_positions,
            hardneg_positions,
            # One stream per rung, so adding a rung never perturbs another rung's ordering.
            rng=random.Random(f"{index}:redundancy:{label}"),
            meta=meta,
        )
        for label, size in rungs.items()
    }


GENERATOR = Generator(
    name="redundancy",
    task="redundancy",
    source="pubmed_redundancy",
    build_example=build_example,
    defaults={"num_docs": 100, "num_pairs": 3, "num_hardneg": 6},
    corpus=source.load_pool,
    corpus_defaults={
        "pairs_path": None,
        "num_abstracts": 20_000,
        "num_mined_pairs": 30_000,
        "seed": 42,
        "mode": "subtle",
        "model": "gemini-2.5-flash",
        "base_url": None,
        "max_concurrent": 32,
        "max_overlap": 0.5,
        "validity_filter": True,
        "hardneg_order": "overlap",
        "cache_dir": "data/.cache/ctc_llm",
    },
    indexed=True,
    shrink_safe=False,  # see nested_ladder: a random shrink halves the planted hard negatives
    build_ladder=nested_ladder,
    notes="PubMed paraphrase pairs; needs a model to mine once, then reuse via pairs_path",
)
