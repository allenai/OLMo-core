"""
The ``reorder`` data contract: consecutive passages of one narrative, shown shuffled.

An example is N passages that were originally consecutive, presented under shuffled display ids;
the target is the permutation restoring source order. Corpus-free by design -- everything here is a
function of a list of sentences, so the construction can be tested without loading 11 GB of
Gutenberg. The corpus half lives in :mod:`ctc.tasks.reorder.sources.gutenberg`.

**Gold is ``gold_order``, 1-based, and it is not a list of document indices.** ``gold_order[i]`` is
the *display id* of the passage that originally sat at source position ``i``, so it is a
permutation of ``1..N`` rather than a subset of positions. The spec declares
``extra["gold_field"] = "gold_order"``; reading ``gold_doc_indices`` finds nothing at all.

.. warning::
   **No reorder ladder can be nested, and the reason is structural.** Every passage is part of the
   answer, so the generic shrink -- which keeps gold and drops distractors -- has nothing to drop.
   A task-owned ``build_ladder`` does not help either: the audit keys a rung's identity on the gold
   documents' text, and a shorter reorder rung is by definition a *different* permutation over
   *fewer* passages, which reads as "the rungs grade different questions". So this generator
   declares ``shrink_safe=False`` with no ``build_ladder``, which routes it to independent per-rung
   draws, and the build report says so. Rung-to-rung deltas on this ladder carry eval-set
   resampling noise on top of the length effect. ``BUILD_MATRIX.md`` row 24 filed the nested mode
   as ACTION A24 and it was never built.

.. warning::
   **The trailing partial passage is dropped, and that is an anti-shortcut measure.** Grouping
   consecutive sentences up to a word target leaves a short remainder at the end of every run. Emit
   it and the shortest passage is systematically the *last* one in source order, which pins one
   position of the permutation without reading anything -- and the pin gets cheaper as N grows,
   because a single free position is worth more Kendall tau on a short ladder rung. The
   pre-migration chunker dropped it too (``generate_reorder_data.py:142-144``, the
   ``cur_words >= min_words`` flush plus the closing filter); the reason was not recorded there, so
   it is recorded here. ``ctc.data.audit.reorder_length_position_bias`` is the standing check.
"""

from __future__ import annotations

import json
import random
from typing import Dict, List, Mapping, Optional, Sequence

from ...data.gold import check_indices
from ...data.schema import make_document, make_example

__all__ = ["passage_runs", "assemble", "GOLD_FIELD"]

#: Where this task's gold lives. Mirrors ``SPEC.extra["gold_field"]``; named here so a generator
#: cannot disagree with the spec by copying the default.
GOLD_FIELD = "gold_order"


def passage_runs(
    sentences: Sequence[str],
    *,
    target_words: int = 100,
    max_words: int = 160,
) -> List[List[str]]:
    """
    Group consecutive sentences into ~``target_words`` passages, in runs of consecutive passages.

    A *run* is what an example may be drawn from: within one run the passages really were adjacent
    in the book, so "restore the original order" has an answer a reader could find. Anything that
    breaks adjacency breaks the run rather than being silently skipped over, which is the same
    discipline :func:`ctc.data.sources.gutenberg.prose_runs` applies to sentences.

    :param sentences: Consecutive clean prose sentences, from one
        :class:`~ctc.data.sources.gutenberg.ProseRun`.
    :param target_words: Words to accumulate before closing a passage. 100 is the pre-migration
        value (``--target-words 100``, the ``reorder_gutenberg100w_*`` files) and the one the rung
        ladder is calibrated against.
    :param max_words: A closed passage longer than this is discarded *and breaks its run*. Only
        reachable when the sentence that closed it was itself enormous -- a wall of run-on dialogue
        or an un-split table -- and such a passage would be several times the length of every other
        one in the example, which is both a length cue and a rung-label error.

    :returns: Runs of passage texts, each run in source order. Runs shorter than two passages are
        dropped, since they can back no example.

    :raises ValueError: If ``max_words`` is below ``target_words``, which would discard every
        passage and silently return no runs at all.
    """
    if max_words < target_words:
        raise ValueError(f"max_words={max_words} is below target_words={target_words}")
    runs: List[List[str]] = []
    run: List[str] = []
    current: List[str] = []
    words = 0

    def close_run() -> None:
        nonlocal run
        if len(run) >= 2:
            runs.append(run)
        run = []

    for sentence in sentences:
        current.append(sentence)
        words += len(sentence.split())
        if words < target_words:
            continue
        if words <= max_words:
            run.append(" ".join(current))
        else:
            close_run()
        current, words = [], 0
    # The remainder is deliberately NOT emitted: it is the one passage whose length is a function
    # of its source position. See the module warning.
    close_run()
    return runs


def assemble(
    passages: Sequence[str],
    rng: random.Random,
    *,
    source: str,
    meta: Optional[Mapping[str, object]] = None,
) -> Dict:
    """
    Shuffle passages into display order and record the permutation that undoes it.

    :param passages: Consecutive passages **in source order**.
    :param rng: Seeded RNG. Consumes exactly one ``shuffle`` over a list of integers -- the
        pre-migration idiom (``generate_reorder_data.py:163-171``), kept identical.
    :param source: Corpus tag.
    :param meta: Per-example metadata.

    :returns: A unified-format example carrying ``gold_order`` (1-based display ids in source
        order) and ``answers[0]`` as that list rendered to JSON, which is what the target builder
        emits.

    :raises ValueError: If fewer than two passages are supplied -- a one-passage "permutation" has
        no wrong answer and would inflate every mean it entered.
    """
    n = len(passages)
    if n < 2:
        raise ValueError(f"a reorder example needs at least 2 passages, got {n}")
    # order[display_pos] = the source position shown at that display position.
    order = list(range(n))
    rng.shuffle(order)
    documents = [make_document(passages[order[display]]) for display in range(n)]
    gold_order = [0] * n
    for display, source_pos in enumerate(order):
        gold_order[source_pos] = display + 1

    check_indices(gold_order, n, base=1, field=GOLD_FIELD)
    return make_example(
        documents=documents,
        queries=[],
        answers=[json.dumps(gold_order)],
        source=source,
        gold=gold_order,
        gold_field=GOLD_FIELD,
        meta=dict(meta or {}),
    )
