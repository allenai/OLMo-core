"""
``absence`` from Project Gutenberg -- the suite's absence row, in its text-diff form.

A window of N consecutive sentences from one book is Version A; Version B is the same window with
K whole sentences removed, rendered as running prose.

Chosen over the PubMed variant as the suite row on 2026-07-19 (``BUILD_MATRIX.md`` row 18) for two
reasons worth keeping in view: each sentence is naturally its own document, so the corpus has real
chunk structure for the full-vs-chunked contrast; and it shares no substrate with ``contradiction``,
which removes the PubMed-overlap confound between two rows of the same table.

.. warning::
   **The graded field is ``gold_doc_indices``, not ``answers``.** The pre-migration Gutenberg files
   were answered with the first four words of each removed sentence, as a JSON list of strings; the
   registered spec grades the id set (``ABSENCE_INSTRUCTION``, ``Missing: [i], [j]``) and does not
   branch on ``meta.format``, even though :func:`ctc.tasks.absence.spec.score_textdiff` and
   ``ABSENCE_GUTENBERG_INSTRUCTION`` both still exist unwired. So the ids are what a build must get
   right, and ``answers`` is carried as provenance for the snippet path -- and because the
   prefix-uniqueness filter below is what makes *that* path unambiguous, keeping it costs nothing
   and losing it would quietly close the option.

.. warning::
   **Only prefix-unique sentences may be removed.** In the snippet answer format two sentences that
   open the same way give an answer matching both, i.e. two correct answers under one label. The
   window is rejected when fewer than K sentences have a unique four-word opening.
"""

from __future__ import annotations

import random
from typing import Dict, Optional

from ....data.generators.base import Generator
from ....data.sources import gutenberg as source
from ..generate import assemble

__all__ = ["first_four", "build_example", "GENERATOR"]

#: Corpus tag, as shipped in ``absence_eval_gutenberg_n{10,50,200}_k3.jsonl``.
SOURCE_TAG = "absence_gutenberg"


def first_four(sentence: str) -> str:
    """
    :param sentence: A window sentence.

    :returns: Its first four words -- the graded answer form for this variant.
    """
    return " ".join(sentence.split()[:4])


def build_example(
    rng: random.Random,
    *,
    corpus: source.BookPool,
    num_docs: int,
    num_removed: int = 3,
) -> Optional[Dict]:
    """
    Build one Gutenberg text-diff absence example.

    :param rng: Seeded RNG.
    :param corpus: The prose-run pool.
    :param num_docs: Sentences in the window -- the scaling axis.
    :param num_removed: Sentences removed in Version B. A fixed count here rather than the
        Bernoulli rate the id variants use, because that is what the shipped files did and what the
        prefix-uniqueness filter is sized against.

    :returns: A unified-format example with 0-based gold positions **and** the first-four-words
        answers, or ``None`` when no run is long enough or the window cannot supply ``num_removed``
        prefix-unique sentences.

    :raises ValueError: If ``num_docs`` cannot hold ``num_removed`` deletions.
    """
    if num_docs <= num_removed:
        raise ValueError(f"num_docs={num_docs} cannot hold {num_removed} removed sentences")
    window = corpus.sample_window(num_docs, rng)
    if window is None:
        return None
    sentences = list(window.sentences)
    if len(set(sentences)) != len(sentences):
        # A repeated sentence makes gold ambiguous whichever copy is dropped; skip the window
        # rather than emit an example with two correct answers.
        return None

    prefixes: Dict[str, int] = {}
    for sentence in sentences:
        key = first_four(sentence).lower()
        prefixes[key] = prefixes.get(key, 0) + 1
    candidates = [i for i, s in enumerate(sentences) if prefixes[first_four(s).lower()] == 1]
    if len(candidates) < num_removed:
        return None
    removed = sorted(rng.sample(candidates, num_removed))

    return assemble(
        sentences,
        removed,
        source=SOURCE_TAG,
        # Version B is running prose, not a numbered list: joined by spaces, with no header. This
        # is the shipped rendering, and the answer format depends on it -- a numbered Version B
        # would hand back the ids the text-diff form deliberately withholds.
        separator=" ",
        prefix="",
        answers=[first_four(sentences[i]) for i in removed],
        meta={
            "format": "textdiff",
            "book": window.book,
            "n": num_docs,
            "k": num_removed,
            "removed_sentences": [sentences[i] for i in removed],
            "provenance": corpus.provenance,
        },
    )


GENERATOR = Generator(
    name="absence",
    task="absence",
    source=SOURCE_TAG,
    build_example=build_example,
    defaults={"num_docs": 50, "num_removed": 3},
    corpus=source.load_pool,
    corpus_defaults={
        "hf_dataset": "sedthh/gutenberg_english",
        "text_column": "TEXT",
        "id_column": None,
        "local_dir": None,
        "max_books": 2000,
        "max_sentences_per_book": 4000,
        "min_run": 12,
        "min_sentence_words": 4,
        "splitter": "punkt",
        "seed": 42,
    },
    # The second version is rendered text in `queries[0]`, so it cannot survive a resize: see the
    # module docstring of ctc.tasks.absence.generate.
    shrink_safe=False,
    notes="Gutenberg prose; Version A/B text diff, answer is the first four words of each removal",
)
