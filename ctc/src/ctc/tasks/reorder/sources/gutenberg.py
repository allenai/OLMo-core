"""
``reorder`` from Project Gutenberg -- narrative passages shuffled, ``BUILD_MATRIX.md`` row 24.

N consecutive ~100-word passages from one book, shown in a random order; the model outputs the
permutation that restores the narrative. The only corpus in the suite where the *ordering* of
ordinary prose is the whole signal, so it is the one row a document-chunked mask can fail on
without failing to retrieve anything.

The book substrate is :mod:`ctc.data.sources.gutenberg`, shared with ``absence`` -- one loader, one
sentence-splitting decision, one train/eval split by book. This module turns its prose runs into a
per-book stream of passages once, at load time, so a draw is a slice rather than a scan.

.. warning::
   **A passage never straddles a prose-run boundary; an EXAMPLE may, and that is a measured
   trade-off rather than an oversight.** ``ctc.data.sources.gutenberg.prose_runs`` breaks a run on
   any heading, table-of-contents line, fragment or sub-four-word sentence, and Gutenberg prose is
   full of all four: measured over 3,000 books, the median run is **22 sentences** and only 7 runs
   in 173,275 reach 800. Confining an example to one run therefore caps the ladder at roughly its
   4k rung -- the 32k rung wants ~1,400 consecutive sentences and **no run in 3,000 books has
   them**, which is not a thin-pool warning but a hard stop (the build raises after 50 consecutive
   rejections).

   So a block is drawn from the book's passage stream, which is every run's passages concatenated
   **in source order**. What that costs is the seam: two passages either side of a chapter heading
   were consecutive in the book but a reader has a weaker cue for their order. What it keeps is the
   thing prose runs were adopted for -- no heading, index line or scrape fragment ever lands
   *inside* a passage. It is also what the pre-migration generator did, by splitting on a chapter
   regex and concatenating every chapter's chunks, so the rung ladder is calibrated for it.
   ``_meta.run_seams`` records how many seams each example crosses, so the cost is measurable
   rather than assumed.

.. warning::
   **Streaming HF is not seed-reproducible; this loader does not stream.** The pre-migration
   generator took its books off ``load_dataset(..., streaming=True)``
   (``generate_reorder_data.py:192``), whose order depends on stored shard order and the
   ``datasets`` version, so the same seed gave different books on different machines. The shared
   loader takes a *seeded random sample of indices* from a materialised dataset, and its
   ``local_dir`` path (sorted glob) is deterministic outright. Any reorder fixture must use
   ``local_dir``.

.. note::
   **Passages carry no internal newlines.** The pre-migration passages kept Gutenberg's own hard
   line wraps, which is why the ``reorder`` serializer collapses ``\\n\\n`` to ``\\n``: without the
   collapse a passage split across two chunks under the chunked mask and leaked its tail into the
   unmasked region (``ctc.format.documents._reorder``). Whitespace-normalised passages make that
   collapse a no-op rather than a repair, so the hazard is closed at the source as well.

.. note::
   **The corpus is already on disk.** ``sedthh/gutenberg_english`` (~11 GB, 29 arrow shards) is
   cached at ``/data/prasann/hf/datasets/sedthh___gutenberg_english`` on **horton**. Point
   ``HF_HOME=/data/prasann/hf`` at it from a job on that node; reading it over ``/net/horton/...``
   is the slow NFS layer and an 11 GB memory-map will not finish.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

from ....data.generators.base import Generator
from ....data.sources import gutenberg as source
from ..generate import assemble, passage_runs

__all__ = ["BookPassages", "PassagePool", "passages_from_books", "load_pool", "build_example"]

#: Corpus tag. The pre-migration files wrote this under ``source_type`` and left ``source`` unset,
#: so every ``source``-keyed consumer -- including the filler-provenance audit -- skipped reorder
#: entirely. Settled here: the unified ``source`` field carries it, and ``source_type`` is gone.
SOURCE_TAG = "reorder_gutenberg"


@dataclass(frozen=True)
class BookPassages:
    """
    One book's passages, in source order.

    :param book: Book id, used for the train/eval split and carried into ``_meta``.
    :param passages: ~``target_words``-word passages, every run's concatenated in source order.
    :param seams: Indices ``i`` such that a prose-run boundary sits between passage ``i - 1`` and
        passage ``i``. Recorded rather than avoided; see the module warning.
    """

    book: str
    passages: Tuple[str, ...]
    seams: FrozenSet[int]

    def __len__(self) -> int:
        return len(self.passages)


@dataclass(frozen=True)
class PassagePool:
    """
    Every scanned book's passage stream.

    :param books: The books, in load order.
    :param provenance: What produced the pool; copied into every example's ``_meta``.
    """

    books: Tuple[BookPassages, ...]
    provenance: Dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.books)

    def for_split(self, split: str) -> "PassagePool":
        """
        Slice by **book**, matching :meth:`ctc.data.sources.gutenberg.BookPool.for_split`.

        Two blocks from one book share vocabulary, characters and narrator, so splitting anywhere
        finer would put near-neighbours of an eval block into training.

        :param split: ``"train"`` or ``"eval"``.

        :returns: A pool holding only that split's books.

        :raises ValueError: On an unknown split name.
        """
        if split not in ("train", "eval"):
            raise ValueError(f"split must be 'train' or 'eval', got {split!r}")
        names = sorted(book.book for book in self.books)
        cut = max(1, len(names) // 10)
        kept = set(names[:cut]) if split == "eval" else set(names[cut:])
        return PassagePool(
            books=tuple(b for b in self.books if b.book in kept), provenance=self.provenance
        )

    def sample_block(self, length: int, rng: random.Random) -> Optional[Tuple[BookPassages, int]]:
        """
        Draw one block of consecutive passages.

        :param length: Passages wanted.
        :param rng: Seeded RNG.

        :returns: ``(book, start index)``, or ``None`` when no book has that many passages -- the
            builder counts that as a skip rather than emitting a short example, which would put a
            rung under its label.
        """
        candidates = [book for book in self.books if len(book) >= length]
        if not candidates:
            return None
        book = candidates[rng.randrange(len(candidates))]
        return book, rng.randint(0, len(book) - length)


def passages_from_books(
    pool: source.BookPool, *, target_words: int = 100, max_words: int = 160
) -> PassagePool:
    """
    Reduce a prose-run pool to a per-book passage stream. Pure, so a fixture uses the same path.

    :param pool: The Gutenberg prose-run pool.
    :param target_words: Words per passage; see :func:`~ctc.tasks.reorder.generate.passage_runs`.
    :param max_words: Passage length cap; see :func:`~ctc.tasks.reorder.generate.passage_runs`.

    :returns: The passage pool, books in first-seen order.
    """
    order: List[str] = []
    by_book: Dict[str, Tuple[List[str], List[int]]] = {}
    for run in pool.runs:
        if run.book not in by_book:
            by_book[run.book] = ([], [])
            order.append(run.book)
        passages, seams = by_book[run.book]
        for group in passage_runs(run.sentences, target_words=target_words, max_words=max_words):
            if passages:
                # A seam is where THIS group starts, i.e. the join between two runs (or between two
                # in-run groups split by an over-long passage).
                seams.append(len(passages))
            passages.extend(group)
    books = tuple(
        BookPassages(book=name, passages=tuple(by_book[name][0]), seams=frozenset(by_book[name][1]))
        for name in order
        if by_book[name][0]
    )
    provenance = dict(pool.provenance)
    provenance.update(
        {
            "books_with_passages": len(books),
            "passages": sum(len(b) for b in books),
            "target_words": target_words,
        }
    )
    return PassagePool(books=books, provenance=provenance)


def load_pool(*, target_words: int = 100, max_words: int = 160, **book_options) -> PassagePool:
    """
    Load Gutenberg books and reduce them to passage streams.

    :param target_words: Words per passage. 100 is the pre-migration value
        (``--target-words 100``, the ``reorder_gutenberg100w_*`` files) and what the rung ladder is
        calibrated against.
    :param max_words: Passage length cap.
    :param book_options: Forwarded to :func:`ctc.data.sources.gutenberg.load_pool`.

    :returns: The passage pool.
    """
    return passages_from_books(
        source.load_pool(**book_options), target_words=target_words, max_words=max_words
    )


def build_example(rng: random.Random, *, corpus: PassagePool, num_docs: int) -> Optional[Dict]:
    """
    Build one reorder example from a block of one book's passages.

    :param rng: Seeded RNG.
    :param corpus: The passage pool.
    :param num_docs: Passages per example -- the scaling axis.

    :returns: A unified-format example with a 1-based ``gold_order``, or ``None`` when no book has
        ``num_docs`` passages or the block repeats one.

    :raises ValueError: If ``num_docs`` is below 2.
    """
    if num_docs < 2:
        raise ValueError(f"num_docs={num_docs} cannot make a permutation with a wrong answer")
    drawn = corpus.sample_block(num_docs, rng)
    if drawn is None:
        return None
    book, start = drawn
    passages = list(book.passages[start : start + num_docs])
    if len(set(passages)) != len(passages):
        # A repeated passage makes the permutation ambiguous: two display ids are interchangeable,
        # so a correct answer can be marked wrong. Rare (refrains, chapter epigraphs) but real.
        return None

    return assemble(
        passages,
        rng,
        source=SOURCE_TAG,
        meta={
            "book": book.book,
            "n": num_docs,
            "start": start,
            "passage_word_lens": [len(p.split()) for p in passages],
            # How many prose-run boundaries this block crosses. See the module warning: zero means
            # the block is one uninterrupted run; a positive count means some adjacent pair is
            # separated by a heading or a fragment in the original book.
            "run_seams": sum(1 for i in book.seams if start < i < start + num_docs),
            "provenance": corpus.provenance,
        },
    )


GENERATOR = Generator(
    name="reorder",
    task="reorder",
    source=SOURCE_TAG,
    build_example=build_example,
    defaults={"num_docs": 29},
    corpus=load_pool,
    corpus_defaults={
        "hf_dataset": "sedthh/gutenberg_english",
        "text_column": "TEXT",
        "id_column": None,
        "local_dir": None,
        # Higher than absence's 300 because the ladder needs whole long books: the 32k rung is 233
        # passages, i.e. ~23k words, so a book that is short or mostly front matter contributes
        # nothing to the top of the ladder even though it serves absence fine.
        "max_books": 2000,
        "max_sentences_per_book": 8000,
        "min_run": 12,
        "min_sentence_words": 4,
        "splitter": "punkt",
        "seed": 42,
        "target_words": 100,
        "max_words": 160,
    },
    # Every passage is part of the answer, so there is nothing for the generic shrink to drop, and
    # a shorter rung is a different permutation. See the module docstring of
    # ctc.tasks.reorder.generate.
    shrink_safe=False,
    notes="Gutenberg narrative passages shuffled; target is the restoring permutation (gold_order)",
)
