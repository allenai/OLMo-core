"""
Wikipedia 100-word passages, grouped into articles -- the substrate for ``outlier``.

The outlier task needs *runs of consecutive passages from one article*, because "a different
topic" is only well-defined if the majority genuinely shares one. Sampling those directly out of
the Lucene index means rejection-sampling random ordinals and walking neighbours one
``searcher.doc()`` call at a time (~0.2-0.4 ms each), and long single-article runs barely exist
(mean ~2.2 chunks, max ~87), so the retry count explodes. Scanning the index once and pickling a
``(title, [bodies...])`` pool makes sampling a memory operation and the generators ~50-100x faster.

The pickle is the artifact: build it once, point every build at it. :func:`build_article_pool` is
the only function here that needs pyserini.
"""

from __future__ import annotations

import bisect
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

__all__ = ["Chunk", "ArticlePool", "extract_title_and_body", "build_article_pool", "load_pool"]

_QUOTED_TITLE = re.compile(r'^"(.+)"\s*$')


def extract_title_and_body(raw_text: str) -> Tuple[str, str]:
    """
    Split a ``wikipedia-dpr-100w`` ``contents`` field into title and body.

    Line 1 is the title, usually quoted; line 2+ is the body, which often re-prefixes the title
    verbatim. Stripping that prefix matters here more than it looks: an un-stripped body starts
    with its own article title, which is the one string that identifies the topic without reading
    the passage -- and the outlier task hides titles for exactly that reason.

    :param raw_text: The raw ``contents`` field.

    :returns: ``(title, body)``; ``("", body)`` when there is no title line.
    """
    if "\n" not in raw_text:
        return "", raw_text.strip()
    line, rest = raw_text.split("\n", 1)
    match = _QUOTED_TITLE.match(line.strip())
    title = match.group(1) if match else line.strip()
    body = rest.strip()
    if title and body.startswith(title):
        body = body[len(title) :].lstrip(" .,:;-")
    return title, body


@dataclass(frozen=True)
class Chunk:
    """
    :param title: The article the passage belongs to. Carried so a caller can exclude **every**
        title a run touched, not just the first -- a composed run spans several articles, and
        excluding only ``run[0]`` is how a "distinct topic" quietly shares one with the majority.
    :param body: The passage text. What the model sees; the title is never rendered, because a
        Wikipedia title gives the outlier away without reading anything.
    """

    title: str
    body: str


class ArticlePool:
    """In-memory article pool with length-bucketed sampling. No Lucene calls."""

    def __init__(self, articles: Sequence[Tuple[str, Sequence[str]]], min_chunks: int = 4) -> None:
        """
        :param articles: ``(title, [bodies...])``.
        :param min_chunks: The filter the pool was built with, recorded for provenance.
        """
        self.articles: List[Tuple[str, Tuple[str, ...]]] = [
            (title, tuple(bodies)) for title, bodies in articles
        ]
        self.min_chunks = min_chunks
        order = sorted(range(len(self.articles)), key=lambda i: len(self.articles[i][1]))
        self._sorted_idx = order
        self._sorted_lens = [len(self.articles[i][1]) for i in order]
        self.max_chunks = self._sorted_lens[-1] if self._sorted_lens else 0

    def __len__(self) -> int:
        return len(self.articles)

    @classmethod
    def from_cache(cls, path: str) -> "ArticlePool":
        """
        :param path: Pickle written by :func:`build_article_pool`.

        :returns: The pool.
        """
        with open(path, "rb") as handle:
            blob = pickle.load(handle)
        return cls(blob["articles"], min_chunks=blob.get("min_article_chunks", 4))

    def for_split(self, split: str) -> "ArticlePool":
        """
        Split the *articles* between train and eval.

        Article-level, not passage-level: two passages from one article share a topic, so putting
        one in train and one in eval leaks the very thing the task asks about.

        :param split: ``"train"`` or ``"eval"``.

        :returns: A pool over that split's articles.

        :raises ValueError: On an unknown split name.
        """
        cut = max(1, len(self.articles) // 10)
        if split == "eval":
            kept = self.articles[:cut]
        elif split == "train":
            kept = self.articles[cut:]
        else:
            raise ValueError(f"split must be 'train' or 'eval', got {split!r}")
        return ArticlePool(kept, min_chunks=self.min_chunks)

    def _candidates(self, n_chunks: int) -> List[int]:
        return self._sorted_idx[bisect.bisect_left(self._sorted_lens, n_chunks) :]

    def _compose(self, n_chunks: int, rng, exclude: Set[str], attempts: int = 400):
        out: List[Chunk] = []
        used = set(exclude)
        remaining = n_chunks
        guard = 0
        if not self.articles:
            return None
        while remaining > 0 and guard < attempts:
            guard += 1
            title, bodies = self.articles[rng.randrange(len(self.articles))]
            if title in used:
                continue
            take = min(len(bodies), remaining)
            start = rng.randint(0, len(bodies) - take)
            out.extend(Chunk(title, b) for b in bodies[start : start + take])
            used.add(title)
            remaining -= take
        return None if remaining > 0 else out

    def sample_run(
        self,
        n_chunks: int,
        rng,
        *,
        exclude_titles: Optional[Set[str]] = None,
        single_only: bool = False,
        max_attempts: int = 64,
    ) -> Optional[List[Chunk]]:
        """
        Sample ``n_chunks`` consecutive passages, preferring a window inside one article.

        :param n_chunks: Run length.
        :param rng: Seeded RNG.
        :param exclude_titles: Articles already used in this example.
        :param single_only: Refuse to compose from several articles. Required for the *outlier*
            run, which must be one coherent topic; optional for majority runs, where the fallback
            is what lets a rung exceed the longest article (~87 chunks).
        :param max_attempts: Draws before falling back to composition.

        :returns: The run, or ``None`` when nothing long enough is available.
        """
        exclude = set(exclude_titles or ())
        candidates = self._candidates(n_chunks)
        for _ in range(max_attempts if candidates else 0):
            index = candidates[rng.randrange(len(candidates))]
            title, bodies = self.articles[index]
            if title in exclude:
                continue
            start = rng.randint(0, len(bodies) - n_chunks)
            return [Chunk(title, b) for b in bodies[start : start + n_chunks]]
        if single_only:
            return None
        return self._compose(n_chunks, rng, exclude)


def build_article_pool(
    cache: str = "data/wiki100w_article_pool.pkl",
    *,
    min_article_chunks: int = 4,
    shards: int = 8,
    max_lids: Optional[int] = None,
) -> str:
    """
    Scan the ``wikipedia-dpr-100w`` index once and pickle the article pool.

    :param cache: Output path. The scan is skipped when it already exists.
    :param min_article_chunks: Drop articles shorter than this.
    :param shards: Parallel scan **processes**. Threads plateau at ~5.6k docs/s, so processes are
        required -- and they must be *spawned*: pyserini starts a JVM, and forking a process that
        has already initialised one deadlocks. The parent therefore never opens the index either.
    :param max_lids: Scan only the first N Lucene ids, for a partial pool.

    :returns: The cache path.

    :raises ImportError: If pyserini is missing.
    """
    import multiprocessing as mp
    from collections import defaultdict
    from concurrent.futures import ProcessPoolExecutor

    if Path(cache).exists():
        return cache

    context = mp.get_context("spawn")
    if max_lids is None:
        with ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
            total = list(pool.map(_index_size, [0]))[0]
    else:
        total = max_lids

    bounds = [
        (i * total // shards, (i + 1) * total // shards, min_article_chunks) for i in range(shards)
    ]
    pieces: Dict[str, List[Tuple[int, List[str]]]] = defaultdict(list)
    with ProcessPoolExecutor(max_workers=shards, mp_context=context) as pool:
        for part in pool.map(_scan_range, bounds):
            for start, title, bodies in part:
                pieces[title].append((start, bodies))

    articles = []
    for title, parts in pieces.items():
        parts.sort(key=lambda p: p[0])
        bodies = [b for _, chunk in parts for b in chunk]
        if len(bodies) >= min_article_chunks:
            articles.append((title, bodies))

    Path(cache).parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as handle:
        pickle.dump(
            {"min_article_chunks": min_article_chunks, "articles": articles},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return cache


def _index_size(_=None) -> int:
    """:returns: The passage count, read in a throwaway child so the parent never opens the JVM."""
    from pyserini.search.lucene import LuceneSearcher

    from .._bm25 import WIKI_INDEX

    return LuceneSearcher.from_prebuilt_index(WIKI_INDEX).num_docs


def _scan_range(args) -> List[Tuple[int, str, List[str]]]:
    """
    Scan Lucene ids ``[lo, hi)`` and group consecutive same-title passages into articles.

    The first and last run of a range are always kept even when short: they may be halves of an
    article split across a shard boundary, re-joined by title in the parent.

    :param args: ``(lo, hi, min_chunks)``.

    :returns: ``(start_lid, title, bodies)`` runs.
    """
    import json

    from pyserini.search.lucene import LuceneSearcher

    from .._bm25 import WIKI_INDEX

    lo, hi, min_chunks = args
    searcher = LuceneSearcher.from_prebuilt_index(WIKI_INDEX)
    kept: List[Tuple[int, str, List[str]]] = []
    title: Optional[str] = None
    bodies: List[str] = []
    start = -1
    seen_first = False

    def emit(is_last: bool) -> None:
        nonlocal seen_first
        if title is None:
            return
        is_first, seen_first = not seen_first, True
        if is_first or is_last or len(bodies) >= min_chunks:
            kept.append((start, title, list(bodies)))

    for lid in range(lo, hi):
        doc = searcher.doc(lid)
        if doc is None:
            emit(False)
            title, bodies = None, []
            continue
        raw = json.loads(doc.raw())
        head, body = extract_title_and_body(raw.get("contents", raw.get("body", "")))
        if not head or not body:
            emit(False)
            title, bodies = None, []
            continue
        if head == title:
            bodies.append(body)
        else:
            emit(False)
            title, bodies, start = head, [body], lid
    emit(True)
    return kept


def load_pool(
    *,
    cache: str = "data/wiki100w_article_pool.pkl",
    build: bool = True,
    min_article_chunks: int = 4,
    shards: int = 8,
) -> ArticlePool:
    """
    Load the article pool, building the pickle on first use.

    :param cache: Pickle path.
    :param build: Scan the index when the pickle is absent. Off means "fail rather than start a
        multi-hour scan by accident".
    :param min_article_chunks: Passed to the builder.
    :param shards: Passed to the builder.

    :returns: The pool.

    :raises FileNotFoundError: If the cache is absent and ``build`` is off.
    """
    if not Path(cache).exists():
        if not build:
            raise FileNotFoundError(
                f"no article pool at {cache}; build one with "
                "ctc.data.sources.wiki100w.build_article_pool (a full index scan)"
            )
        build_article_pool(cache, min_article_chunks=min_article_chunks, shards=shards)
    return ArticlePool.from_cache(cache)
