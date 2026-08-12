"""
The ``xabsence`` data contract: two corpora under one index, and the claims with no twin.

Corpus A holds original claims, corpus B holds their paraphrases, and the documents are laid out as
the whole A block followed by the whole B block sharing a single 1-based index -- which is exactly
what the ``xabsence`` serializer renders and what the instruction's "the OTHER corpus" refers to.

**Gold is 0-based** (:mod:`ctc.tasks._absence` converts to the 1-based ids the prompt renders, and
:mod:`ctc.data.gold` makes the base a required argument for exactly this reason).

.. warning::
   **An orphan must be rendered in the form its own corpus uses.** The pre-migration generator
   always inserted the orphan as an *original* claim, so a B-side orphan was the one
   non-paraphrase sitting in a block of paraphrases. A trained 4B read it straight off that single
   document: recall 0.98 on B-side orphans against 0.08 on A-side ones, which pinned ``set_f1`` at
   ~0.5 and made it **flat in n from 39 to 669 documents** -- i.e. it had stopped being an all-pairs
   task and nobody could tell from the score alone. :func:`assemble` takes ``original`` for an
   A-side orphan and ``paraphrase`` for a B-side one, so a matched document and an orphan are drawn
   from the same distribution and are individually indistinguishable.

.. note::
   **Dropping a random document creates gold that is not in the label.** Remove one half of a
   matched pair and its partner becomes unmatched -- a real, correct answer the example does not
   list. So ``xabsence`` cannot use the generic nested shrink and declares ``shrink_safe=False``.
   It does get a nested ladder anyway, because dropping whole *pairs* is safe:
   :func:`ctc.tasks.xabsence.sources.pubmed.nested_ladder`.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ...data.schema import make_document, make_example

__all__ = ["split_for_docs", "assemble"]


def split_for_docs(num_docs: int, num_unmatched: int) -> Tuple[int, int]:
    """
    Convert a rung's document count into a pair count and an orphan count.

    A ladder row is quoted in documents, like every other task's, but an example is built from
    pairs: ``2P + k`` documents for ``P`` matched pairs and ``k`` orphans. Only an odd remainder can
    be left over, and it becomes **one more orphan** rather than one fewer document -- the rung
    label is a context length and must be met exactly, while ``k`` is something the model is never
    told and the instruction deliberately does not fix. The shipped ladder is odd at every rung so
    the default ``k=3`` is exact anyway.

    :param num_docs: The rung's document count.
    :param num_unmatched: Orphans per example, ``k``; a floor, raised by one on a parity mismatch.

    :returns: ``(matched pairs, orphans)``, together exactly ``num_docs`` documents.

    :raises ValueError: If the rung cannot hold the orphans and at least one matched pair. A corpus
        with no matched pair at all makes every document a correct answer.
    """
    pairs = (num_docs - num_unmatched) // 2
    if pairs < 1:
        raise ValueError(
            f"num_docs={num_docs} cannot hold {num_unmatched} unmatched claims and a matched pair; "
            "an example with no matched pair has every document as a correct answer"
        )
    return pairs, num_docs - 2 * pairs


def assemble(
    matched: Sequence,
    unmatched: Sequence,
    *,
    rng: random.Random,
    source: str,
    sides: Optional[Sequence[str]] = None,
    meta: Optional[Mapping[str, object]] = None,
) -> Dict:
    """
    Render one xabsence example: an A block, then a B block, then the orphan positions.

    :param matched: :class:`~ctc.data.sources.paraphrase.ParaphrasePair` entries placed in both
        corpora -- the original into A, the rewrite into B.
    :param unmatched: Entries placed in exactly one corpus. Each contributes **one** document.
    :param rng: Seeded RNG; shuffles each block, and picks the orphans' sides when ``sides`` is not
        given.
    :param source: Corpus tag.
    :param sides: One of ``"A"``/``"B"`` per orphan. Passed in by the ladder builder so a row's
        orphans sit on the same side at every rung; drawn here otherwise.
    :param meta: Per-example metadata.

    :returns: A unified-format example with **0-based** ``gold_doc_indices``.

    :raises ValueError: If an orphan's text appears twice in the context. It then does have a
        counterpart in the other corpus, so the label is wrong. Two *matched* documents may share a
        text, because ``mode="exact"`` twins are byte-identical by design.
    """
    chosen = list(sides) if sides is not None else [rng.choice("AB") for _ in unmatched]

    def _doc(text: str, corpus: str, orphan: bool) -> Dict:
        # `make_document` for the empty-text guard; `corpus` is the extra key the xabsence
        # serializer renders and `_orphan` is popped once the display order is fixed.
        return {**make_document(text), "corpus": corpus, "_orphan": orphan}

    a_items: List[Dict] = [_doc(pair.original, "A", False) for pair in matched]
    b_items: List[Dict] = [_doc(pair.paraphrase, "B", False) for pair in matched]
    for pair, side in zip(unmatched, chosen):
        # The form has to match the block, not the entry: see the module warning.
        text = pair.original if side == "A" else pair.paraphrase
        (a_items if side == "A" else b_items).append(_doc(text, side, True))

    rng.shuffle(a_items)
    rng.shuffle(b_items)
    items = a_items + b_items
    gold = [i for i, doc in enumerate(items) if doc.pop("_orphan")]
    counts = Counter(doc["text"] for doc in items)
    orphaned = [items[i]["text"] for i in gold if counts[items[i]["text"]] > 1]
    if orphaned:
        raise ValueError(
            f"{len(orphaned)} 'unmatched' claim(s) appear more than once in the context, so they "
            "do have a counterpart and the label is wrong. A matched pair may legitimately repeat "
            "a text -- `mode='exact'` twins are byte-identical -- but an orphan may not."
        )
    return make_example(
        documents=items,
        queries=[],
        answers=[],
        source=source,
        gold=sorted(gold),
        num_pairs=len(matched),
        num_unmatched=len(unmatched),
        meta=dict(meta or {}),
    )
