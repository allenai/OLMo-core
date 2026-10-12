"""
The ``contradiction`` data contract: source corpus -> task JSONL.

Separate from ``spec.py`` because the two change for different reasons. Editing the spec reprices
every result that already exists; editing a generator only affects data built afterwards. When both
lived in one file it was easy to touch one while meaning to touch the other, and the format
fingerprint could tell you *that* something moved but not which.

Contradiction has two live source corpora, each with its own way of finding claim pairs that
genuinely conflict, so each gets a module under ``sources/``. They share this file's assembly --
place the gold pairs, fill with distractors, shuffle, record 1-based pair positions -- and only the
claim mining differs.

.. warning::
   **Never harvest fillers with a domain-agnostic glob.** The pre-migration builder globbed a
   mutable directory with ``contradiction_*_k3.jsonl``, which also matched the FEVER and wiki
   corpora, so PubMed contradiction evals shipped Wikipedia claims as distractors -- 92.2% of
   fillers at 2k rising to 99.6% at 32k, against gold that was 100% PubMed. "Find the contradicting
   pair among n docs" then collapses to "find the biomedical sentences".

   The leak is **closed** for both the xlong and CTC-suite ladders (rebuilt and re-verified at
   0.00% on 2026-08-04/05), and it is closed *structurally* here: a pool holds one corpus, and an
   example's fillers can only come from the pool its gold came from. Two results of that rebuild
   are worth carrying forward, because each overturned a headline:

   * The contamination **depressed** the scores rather than flattering them -- it was a train/eval
     domain shift, not a shortcut -- and worst at the long end: 32k went 0.335 -> 0.559 on the
     clean ladder, roughly 10 SE at eval_size 500.
   * The published "dense collapses at 32k" was therefore an artifact of the ladder. The real dense
     curve declines gracefully, and the dense-vs-chunked absolute gap **narrows** with context
     (0.441 at 2k -> 0.369 at 32k) rather than widening.

   The clean rung counts live in :data:`ctc.data.ladders.LADDERS`. They are *not* the pre-migration
   ``contra`` ladder, which was fit against the contaminated filler pool and overshoots every rung
   by ~1.8x.
"""

from __future__ import annotations

import random
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ...data.gold import remap_groups, shuffle_with_remap
from ...data.schema import make_document, make_example

__all__ = ["SOURCES", "assemble", "place_pairs"]

#: Source name -> the ladder it builds, resolvable through
#: :func:`ctc.data.generators.base.get`. ``pubmed`` is the in-distribution anchor; ``fever`` is the
#: held-out probe. ``wiki`` is deliberately absent: the pre-migration
#: ``generate_wiki_contradiction_data.py`` has zero references anywhere in the old tree and nothing
#: was ever built from it.
SOURCES: Dict[str, str] = {"pubmed": "contradiction", "fever": "contra_fever"}


def place_pairs(pairs: Sequence[Tuple[str, str]]) -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    Lay gold pairs out at the front of a corpus, before the shuffle.

    :param pairs: ``(claim, contradicting claim)``.

    :returns: ``(statements, pair_indices)`` in construction order.
    """
    statements: List[str] = []
    positions: List[Tuple[int, int]] = []
    for first, second in pairs:
        positions.append((len(statements), len(statements) + 1))
        statements.extend([first, second])
    return statements, positions


def assemble(
    statements: Sequence[str],
    pair_indices: Sequence[Tuple[int, int]],
    *,
    source: str,
    rng: random.Random,
    meta: Optional[Mapping[str, object]] = None,
) -> Dict:
    """
    Shuffle a corpus of claims and record where the contradicting pairs landed.

    :param statements: Claim texts in construction order -- gold pairs first, then distractors.
    :param pair_indices: ``(a, b)`` construction-order positions of each gold pair.
    :param source: Corpus tag.
    :param rng: Seeded RNG. Consumes exactly one shuffle.
    :param meta: Per-example metadata.

    :returns: A unified-format example whose ``gold_doc_indices`` holds **1-based** pairs, sorted
        within each pair and across pairs. Contradiction is one of the eight tasks whose gold is
        1-based; reading it as 0-based scores every correct answer zero, uniformly, and looks like
        a modelling result.
    """
    items, old_to_new = shuffle_with_remap(list(statements), rng=rng, base=1)
    return make_example(
        documents=[make_document(text) for text in items],
        queries=[],
        answers=[],
        source=source,
        gold=remap_groups([list(pair) for pair in pair_indices], old_to_new),
        meta=dict(meta or {}),
    )
