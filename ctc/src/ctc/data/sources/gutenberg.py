"""
Project Gutenberg prose, split into runs of clean consecutive sentences.

The substrate for ``absence`` (a window of N sentences, K of them deleted in a second copy) and,
when it is ported, for ``reorder``. What a task needs from this corpus is not "a book" but **a run
of consecutive, complete, ordinary prose sentences**: a window that straddles a chapter heading, a
table of contents or a fragment left by a bad split produces an example whose gold is unreadable
and whose difficulty comes from the scrape rather than from the task. So the pool stores runs, not
books, and the filtering happens once at load time.

.. warning::
   **Sentence splitting is not incidental here -- it defines the documents.** The pre-migration
   generator loaded NLTK punkt and *silently* fell back to a regex splitter when NLTK or its model
   was missing, so the same seed on two machines produced different data. The fallback is kept
   because it is what the shipped ``absence_eval_gutenberg_*`` files may have been built with, but
   it is now selected explicitly by ``splitter=``: ``"punkt"`` raises if the model is absent
   instead of quietly changing the corpus.

.. note::
   **The corpus is already on disk.** ``sedthh/gutenberg_english`` (~11 GB) is cached on **horton**
   at ``/data/prasann/hf`` (both layouts: ``datasets/sedthh___gutenberg_english`` and
   ``hub/datasets--sedthh--gutenberg_english``) and on **cubbins** at
   ``/data/prasann/hf_cache/hub``. Point ``HF_HOME`` at it from a job on that node rather than
   re-downloading, and read it over the node-local disk -- ``/net/horton/...`` is the slow NFS
   layer and an 11 GB memory-map over it will not finish. See
   ``debug/absence_port/build_absence_horton.sh`` for the working invocation.

.. warning::
   **What the long rungs need is a long unbroken RUN, not a long book, and runs that long are
   rare.** A run ends at every heading, contents line or bad split, so book length says almost
   nothing about it. Measured with ``punkt`` over this corpus (``debug/absence_port/
   check_long_runs.py``), at ``max_books=300`` the train split holds **one** run of >= 438
   sentences -- the 32k rung -- so that rung would emit near-copies of a single passage until the
   builder's duplicate guard gave up. At ``max_books=2000`` it is 23 train / 5 eval runs, which is
   why 2000 is the default here and matches the pre-migration ``--max-books-to-scan``. Even then a
   500-row eval set at 32k is drawn from five passages: quote that alongside any 32k absence
   number, or raise ``max_books`` further.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "ProseRun",
    "BookPool",
    "strip_boilerplate",
    "split_sentences",
    "is_prose",
    "prose_runs",
    "load_pool",
]

#: The license header/footer wrapped around every Project Gutenberg text. Left in, it becomes a
#: block of near-identical "sentences" shared by every book -- a deleted one would be findable from
#: the boilerplate alone.
_START = re.compile(r"\*\*\*\s*START OF (?:THE|THIS)?\s*PROJECT GUTENBERG[^\n]*\*\*\*", re.I)
_END = re.compile(r"\*\*\*\s*END OF (?:THE|THIS)?\s*PROJECT GUTENBERG[^\n]*\*\*\*", re.I)

#: Honorific, academic and bibliographic abbreviations punkt's English model does not know, so it
#: splits "the Rev. John ..." or "M.A., and ..." into two "sentences". Folded into the loaded
#: tokenizer, exactly as the pre-migration generator did.
EXTRA_ABBREV = frozenset(
    {
        "rev", "hon", "esq", "jr", "jun", "sr", "sen", "gov", "gen", "col", "capt",
        "lt", "maj", "sgt", "pres", "prof", "dr", "mr", "mrs", "ms", "st", "mt",
        "messrs", "vol", "no", "pp", "p", "vs", "viz", "fig", "ed", "trans", "etc",
        "ca", "cf", "m.a", "ph.d", "ll.d", "d.d", "b.a", "m.d", "f.r.s", "f.s.a",
    }
)  # fmt: skip

#: The regex fallback's abbreviation set: shorter, because it can only merge on the last token.
_ABBREV = frozenset(
    {
        "mr", "mrs", "ms", "dr", "st", "jr", "sr", "vs", "etc", "no", "mt",
        "capt", "gen", "col", "sgt", "rev", "hon", "prof", "messrs", "esq",
    }
)  # fmt: skip

#: Terminal punctuation a real sentence may end on, closing quotes included.
_SENT_END = tuple(".!?\"')”’")
#: Opening characters a real sentence may start with, besides a letter or digit.
_SENT_OPEN = "\"'“‘"

_REGEX_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE = re.compile(r"\s+")

_PUNKT = None


def strip_boilerplate(text: str) -> str:
    """
    :param text: A raw Gutenberg book.

    :returns: The body between the START and END markers, or the whole text when neither is found.
    """
    match = _START.search(text)
    if match:
        text = text[match.end() :]
    match = _END.search(text)
    if match:
        text = text[: match.start()]
    return text.strip()


def _punkt_tokenizer():
    """
    :returns: A punkt tokenizer with :data:`EXTRA_ABBREV` folded in, loaded once.

    :raises ImportError: If NLTK or its punkt model is missing. Deliberately *not* a silent
        fallback: the splitter decides what a document is, so swapping it changes the corpus.
    """
    global _PUNKT
    if _PUNKT is not None:
        return _PUNKT
    try:
        import nltk

        try:  # NLTK >= 3.9 ships punkt as `punkt_tab` and refuses to unpickle the old model.
            tokenizer = nltk.tokenize.PunktTokenizer("english")
        except AttributeError:  # pragma: no cover - older NLTK
            tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
        tokenizer._params.abbrev_types.update(EXTRA_ABBREV)
    except Exception as e:  # pragma: no cover - depends on the install
        raise ImportError(
            "punkt sentence splitting needs nltk and its punkt model: "
            "pip install nltk && python -m nltk.downloader punkt punkt_tab. "
            "Pass splitter='regex' to use the documented fallback instead -- but note it "
            "produces a DIFFERENT corpus, so do not mix the two in one build."
        ) from e
    _PUNKT = tokenizer
    return tokenizer


def _split_regex(text: str) -> List[str]:
    """
    :param text: Whitespace-normalised prose.

    :returns: Sentences, merging a fragment back onto its predecessor when that predecessor ends in
        a known abbreviation, so "Mr. Bob" does not become two sentences.
    """
    out: List[str] = []
    for part in _REGEX_SPLIT.split(text):
        if out:
            words = out[-1].split()
            last = words[-1].rstrip(".").lower() if words else ""
            if last in _ABBREV:
                out[-1] = out[-1] + " " + part
                continue
        out.append(part)
    return out


def split_sentences(text: str, splitter: str = "punkt") -> List[str]:
    """
    :param text: Book text, already stripped of boilerplate.
    :param splitter: ``"punkt"`` (NLTK, with our abbreviations) or ``"regex"`` (the stdlib
        fallback). Explicit because the choice changes the documents themselves.

    :returns: Its sentences, on whitespace-normalised text.

    :raises ValueError: On an unknown splitter name.
    """
    text = _WHITESPACE.sub(" ", text).strip()
    if not text:
        return []
    if splitter == "regex":
        return _split_regex(text)
    if splitter != "punkt":
        raise ValueError(f"splitter must be 'punkt' or 'regex', got {splitter!r}")
    return list(_punkt_tokenizer().tokenize(text))


def is_prose(sentence: str, min_words: int) -> bool:
    """
    Whether a sentence is a clean, complete prose sentence.

    :param sentence: Candidate.
    :param min_words: Word floor. Must be at least 4 where the answer is the first four words.

    :returns: True when it has enough words, contains a lowercase letter (which drops ALL-CAPS
        headings like ``FOLK LORE.``), starts like text and ends on terminal punctuation (which
        drops cut fragments like ``(Vol.``).
    """
    sentence = sentence.strip()
    if len(sentence.split()) < min_words:
        return False
    if not any(c.islower() for c in sentence):
        return False
    if not sentence.endswith(_SENT_END):
        return False
    first = sentence[:1]
    return bool(first) and (first.isalpha() or first in _SENT_OPEN)


def prose_runs(sentences: Sequence[str], min_run: int, min_words: int) -> List[List[str]]:
    """
    Split a sentence list into maximal runs of consecutive clean prose.

    A heading or a fragment *breaks* a run rather than being dropped from it, so a window sampled
    inside a run is guaranteed to be consecutive text with no scrape junk spliced into the middle.

    :param sentences: A book's sentences, in order.
    :param min_run: Shortest run worth keeping.
    :param min_words: Word floor per sentence.

    :returns: The runs.
    """
    runs: List[List[str]] = []
    current: List[str] = []
    for sentence in sentences:
        if is_prose(sentence, min_words):
            current.append(sentence.strip())
            continue
        if len(current) >= min_run:
            runs.append(current)
        current = []
    if len(current) >= min_run:
        runs.append(current)
    return runs


@dataclass(frozen=True)
class ProseRun:
    """
    :param book: Source book id, carried into ``_meta`` and used for the train/eval split.
    :param sentences: Consecutive clean prose sentences.
    """

    book: str
    sentences: Tuple[str, ...]

    def __len__(self) -> int:
        return len(self.sentences)


@dataclass(frozen=True)
class BookPool:
    """
    :param runs: Every prose run, in load order.
    :param provenance: What produced the pool, copied into every example's ``_meta``.
    """

    runs: Tuple[ProseRun, ...]
    provenance: Dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.runs)

    def for_split(self, split: str) -> "BookPool":
        """
        Slice by **book**, not by run.

        Two windows from one book share vocabulary, characters and narrator, so splitting runs
        would put near-neighbours of an eval window into training.

        :param split: ``"train"`` or ``"eval"``.

        :returns: A pool holding only that split's books.

        :raises ValueError: On an unknown split name.
        """
        books = sorted({run.book for run in self.runs})
        cut = max(1, len(books) // 10)
        if split == "eval":
            kept = set(books[:cut])
        elif split == "train":
            kept = set(books[cut:])
        else:
            raise ValueError(f"split must be 'train' or 'eval', got {split!r}")
        return BookPool(
            runs=tuple(run for run in self.runs if run.book in kept), provenance=self.provenance
        )

    def sample_window(self, length: int, rng: random.Random) -> Optional[ProseRun]:
        """
        Draw one window of consecutive sentences.

        :param length: Sentences wanted.
        :param rng: Seeded RNG.

        :returns: The window as a :class:`ProseRun`, or ``None`` when no run is long enough -- the
            builder counts that as a skip rather than emitting a short example, which would put a
            rung under its label.
        """
        candidates = [run for run in self.runs if len(run) >= length]
        if not candidates:
            return None
        run = candidates[rng.randrange(len(candidates))]
        start = rng.randint(0, len(run) - length)
        return ProseRun(book=run.book, sentences=run.sentences[start : start + length])


def _iter_local_books(local_dir: str):
    """
    :param local_dir: Directory of ``.txt`` books.

    :returns: ``(book id, text)`` in sorted filename order -- deterministic, unlike a streaming HF
        read, whose order depends on stored shard order and the ``datasets`` version.
    """
    from pathlib import Path

    for path in sorted(Path(local_dir).glob("*.txt")):
        yield path.stem, path.read_text(encoding="utf-8", errors="ignore")


def _iter_hf_books(dataset: str, text_column: str, id_column: Optional[str], max_books: int, seed):
    """
    :param dataset: HF dataset id.
    :param text_column: Column holding the book text.
    :param id_column: Column holding a book id, or ``None`` to hash the text.
    :param max_books: Books to visit.
    :param seed: Which books, and in what order.

    :returns: ``(book id, text)`` for a seeded random sample of books -- random rather than
        first-N, because the dataset is roughly alphabetical by author and the first N books share
        a period and a style.

    :raises ImportError: If ``datasets`` is missing; install ``ctc[sources]``.
    """
    import hashlib

    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - depends on the install
        raise ImportError("Gutenberg loading needs `datasets`: pip install './ctc[sources]'") from e

    data = load_dataset(dataset, split="train")
    order = list(range(len(data)))
    random.Random(seed).shuffle(order)
    for i in order[:max_books]:
        row = data[int(i)]
        text = row.get(text_column)
        if not text:
            continue
        if id_column and row.get(id_column) is not None:
            yield f"{dataset}/{row[id_column]}", text
        else:
            digest = hashlib.md5(text[:2000].encode("utf-8")).hexdigest()[:12]
            yield f"{dataset}/{digest}", text


def load_pool(
    *,
    hf_dataset: str = "sedthh/gutenberg_english",
    text_column: str = "TEXT",
    id_column: Optional[str] = None,
    local_dir: Optional[str] = None,
    max_books: int = 2000,
    max_sentences_per_book: int = 4000,
    min_run: int = 12,
    min_sentence_words: int = 4,
    splitter: str = "punkt",
    seed: int = 42,
) -> BookPool:
    """
    Load books and reduce them to runs of clean consecutive prose.

    :param hf_dataset: HF dataset id. The cached copy on horton is the one to use.
    :param text_column: Book-text column.
    :param id_column: Book-id column; ``None`` hashes the text instead.
    :param local_dir: Read ``*.txt`` from here instead of HF. Deterministic, and the only
        reproducible path if the dataset version moves under you.
    :param max_books: Books to scan. Not the memory knob it looks like: short windows are
        plentiful at any setting, but **long unbroken runs are not**, and the long rungs are
        rationed by them -- see the module warning for the measured counts. 2000 is the
        pre-migration ``--max-books-to-scan``, and 300 leaves the 32k rung with a single usable run.
    :param max_sentences_per_book: Truncate a book to this many sentences. A bound on the pool's
        size in memory; a book is far longer than any rung needs.
    :param min_run: Shortest run kept. Below the largest rung's window this run can never be used,
        but a small floor keeps the pool honest about how much clean prose a book actually has.
    :param min_sentence_words: Word floor per sentence; **must be >= 4** where the answer format is
        the first four words of a removed sentence.
    :param splitter: ``"punkt"`` or ``"regex"``; see :func:`split_sentences`.
    :param seed: Which books are scanned.

    :returns: The pool.

    :raises ValueError: If ``min_sentence_words`` is below 4.
    """
    if min_sentence_words < 4:
        raise ValueError(
            f"min_sentence_words={min_sentence_words} is below 4, so a removed sentence may have "
            "no first-four-words answer to grade against"
        )
    books = (
        _iter_local_books(local_dir)
        if local_dir
        else _iter_hf_books(hf_dataset, text_column, id_column, max_books, seed)
    )
    runs: List[ProseRun] = []
    scanned = 0
    for book_id, text in books:
        scanned += 1
        sentences = split_sentences(strip_boilerplate(text), splitter)[:max_sentences_per_book]
        runs.extend(
            ProseRun(book=book_id, sentences=tuple(run))
            for run in prose_runs(sentences, min_run, min_sentence_words)
        )
    return BookPool(
        runs=tuple(runs),
        provenance={
            "corpus": local_dir or hf_dataset,
            "books": scanned,
            "runs": len(runs),
            "splitter": splitter,
            "seed": seed,
        },
    )
