"""
The ``absence`` data contract: a numbered corpus, a second copy with items removed, find the gap.

Three corpora share it -- Gutenberg prose sentences, PubMed claim sentences and synthetic number
sequences -- so the deletion, the second-version rendering and the gold shape live here and only
the element supply differs.

**Gold is 0-based** (``ctc.tasks._absence`` converts to the 1-based ids the prompt renders, and
:mod:`ctc.data.gold` makes the base a required argument for exactly this reason).

.. warning::
   **No absence ladder can be nested, and the reason is structural.** The second version is carried
   in ``queries[0]`` as *rendered text*, so it is a function of the whole corpus: drop a distractor
   to make a shorter rung and the query still lists it, which is not a shorter version of the
   question but a broken one. The generic shrink does not touch ``queries``, and a task-owned
   ``build_ladder`` fares no better -- the audit's nesting check keys a rung's identity on the gold
   text *plus the query*, so rebuilding the query per rung reads as "the rungs grade different
   questions". Both generators therefore declare ``shrink_safe=False`` with no ``build_ladder``,
   which routes them to independent per-rung draws exactly like ``oolong``, and the build report
   says so. Rung-to-rung deltas on these ladders carry eval-set resampling noise.

.. note::
   **Low deletion rates are the HARD regime, not the easy one.** AbsenceBench (arXiv:2506.11440)
   found scattered single deletions harder than one big contiguous gap, which is why ``p`` defaults
   to 0.1 rather than to something that leaves an obvious hole.
"""

from __future__ import annotations

import random
from typing import Dict, List, Mapping, Optional, Sequence

from ...data.schema import make_document, make_example

__all__ = ["choose_removed", "assemble", "SECOND_VERSION_PREFIX"]

#: The pre-migration second-version header, verbatim. It is part of the trained prompt format:
#: every shipped ``absence_*_{pubmed,numerical,official_*}`` file carries it, and the instruction
#: refers to "the text below".
SECOND_VERSION_PREFIX = "Second version:\n\n"


def choose_removed(
    n: int, p: float, rng: random.Random, *, min_remove: int = 1, max_remove_frac: float = 0.5
) -> List[int]:
    """
    Pick which positions the second version drops.

    Bernoulli per element rather than a fixed count, so the *number* missing is itself unknown --
    a fixed K would let a model stop after K guesses, and "how many are missing" is half the task.
    The floor and the cap are the pre-migration guards: zero removals gives an example with no
    answer, and more than half removed turns "what is missing" into "what is left".

    :param n: Elements in the corpus.
    :param p: Per-element deletion probability.
    :param rng: Seeded RNG.
    :param min_remove: Fewest removals; a resample when the draw came up short.
    :param max_remove_frac: Cap, as a fraction of ``n``.

    :returns: The removed positions, ascending.
    """
    cap = max(min_remove, int(n * max_remove_frac))
    removed = [i for i in range(n) if rng.random() < p]
    if len(removed) < min_remove:
        removed = rng.sample(range(n), min_remove)
    if len(removed) > cap:
        removed = rng.sample(removed, cap)
    return sorted(set(removed))


def assemble(
    elements: Sequence[str],
    removed: Sequence[int],
    *,
    source: str,
    separator: str = "\n",
    prefix: str = SECOND_VERSION_PREFIX,
    answers: Sequence[str] = (),
    meta: Optional[Mapping[str, object]] = None,
) -> Dict:
    """
    Render one absence example: the full corpus as documents, the survivors as the query.

    :param elements: The original corpus, in display order. Must already be de-duplicated -- two
        identical elements make the gold ambiguous, since the model cannot tell which of the two
        the second version dropped.
    :param removed: 0-based positions the second version omits.
    :param source: Corpus tag.
    :param separator: How the second version joins its survivors. ``"\\n"`` everywhere except the
        numeric variant, which uses ``", "`` -- both are the pre-migration renderings and both are
        part of the trained prompt format.
    :param prefix: Header for the second version.
    :param answers: Answer strings for the Gutenberg text-diff format; empty for the id formats,
        whose answer *is* the gold index set.
    :param meta: Per-example metadata.

    :returns: A unified-format example with **0-based** ``gold_doc_indices``.

    :raises ValueError: If ``elements`` holds a duplicate, which would make gold ambiguous.
    """
    if len(set(elements)) != len(elements):
        raise ValueError(
            "absence elements must be unique: a repeated element makes the gold ambiguous, since "
            "'which copy was deleted' has no answer the model could give"
        )
    dropped = set(removed)
    kept = [e for i, e in enumerate(elements) if i not in dropped]
    return make_example(
        documents=[make_document(e) for e in elements],
        queries=[prefix + separator.join(kept)],
        answers=list(answers),
        source=source,
        gold=sorted(dropped),
        meta=dict(meta or {}),
    )
