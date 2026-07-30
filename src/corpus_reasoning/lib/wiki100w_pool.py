"""Precomputed article pool over the wikipedia-dpr-100w Lucene index.

`sample_article_run` (scripts/lib/wiki100w_sample.py) assembles a contiguous
same-article chunk run by rejection-sampling random Lucene ids and walking
neighbours via per-id `searcher.doc(lid)` lookups (~0.2-0.4ms each). For long
runs this wastes huge numbers of retries because long single-article runs
barely exist (max ~87 chunks/article; mean ~2.2). This module scans the index
ONCE (parallel across processes), groups consecutive same-title chunks into
articles, and pickles a `(title, [bodies...])` pool. `ArticlePool.sample_run`
then reproduces the same article-run sampling from memory — no Lucene calls —
giving a ~50-100x speedup for the generators.

The pool returns chunk dicts with the SAME shape `sample_article_run` consumers
rely on: each chunk is `{"title": <article title>, "body": <chunk body>}`.
"""

import bisect
import json
import multiprocessing as mp
import os
import pickle
import random
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

from corpus_reasoning.lib.bm25 import INDEX_NAME
from corpus_reasoning.lib.wiki100w_sample import extract_title_and_body


# ───────────────────────────── index scan (build) ──────────────────────────

def _scan_range(args) -> list:
    """Scan Lucene ids [lo, hi) and return article runs that could survive the
    `min_chunks` filter after cross-shard merge.

    Returns a list of `(start_lid, title, [bodies...])`. Interior runs are kept
    only when they already reach `min_chunks`; the very first and very last run
    of the range are ALWAYS kept (they may be partial halves of an article split
    across a shard boundary, to be re-joined by title in the parent).
    """
    lo, hi, min_chunks = args
    from pyserini.search.lucene import LuceneSearcher
    s = LuceneSearcher.from_prebuilt_index(INDEX_NAME)

    kept: list = []
    cur_title = None
    cur_bodies: list = []
    cur_start = -1
    seen_first = False

    def _emit(is_last: bool):
        nonlocal seen_first
        if cur_title is None:
            return
        is_first = not seen_first
        seen_first = True
        if is_first or is_last or len(cur_bodies) >= min_chunks:
            kept.append((cur_start, cur_title, cur_bodies))

    for lid in range(lo, hi):
        d = s.doc(lid)
        if d is None:
            # gap acts as an article boundary
            _emit(is_last=False)
            cur_title = None
            cur_bodies = []
            continue
        raw = json.loads(d.raw())
        raw_text = raw.get("contents", raw.get("body", ""))
        title, body = extract_title_and_body(raw_text)
        if not title or not body:
            _emit(is_last=False)
            cur_title = None
            cur_bodies = []
            continue
        if title == cur_title:
            cur_bodies.append(body)
        else:
            _emit(is_last=False)
            cur_title = title
            cur_bodies = [body]
            cur_start = lid
    _emit(is_last=True)
    return kept


def _num_docs(_=None) -> int:
    """Open the index in a fresh process and return its passage count."""
    from pyserini.search.lucene import LuceneSearcher
    return LuceneSearcher.from_prebuilt_index(INDEX_NAME).num_docs


def build_article_pool(
    min_article_chunks: int = 4,
    shards: int = 8,
    cache: str = "data/wiki100w_article_pool.pkl",
    max_lids: int | None = None,
    verbose: bool = True,
) -> str:
    """Scan the wiki100w index and pickle a `[(title, [bodies...]), ...]` pool.

    :param min_article_chunks: keep only articles with >= this many chunks.
    :param shards: number of parallel scan PROCESSES (each opens its own
        LuceneSearcher; threads plateau ~5.6k docs/s so processes are required).
    :param cache: pickle output path. Scan is skipped if it already exists.
    :param max_lids: if set, only scan Lucene ids [0, max_lids) — for fast
        partial pools / smoke tests.

    :returns: the cache path.
    """
    if os.path.exists(cache):
        if verbose:
            print(f"[pool] cache exists, skipping scan: {cache}")
        return cache

    # Use a SPAWN context: pyserini starts a JVM, and forking a process that has
    # already initialised a JVM deadlocks. Spawned children are fresh interpreters
    # that each start their own JVM cleanly. We also never load the index in the
    # parent — num_docs is fetched from a throwaway child.
    ctx = mp.get_context("spawn")
    if max_lids is None:
        with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as ex:
            n_total = list(ex.map(_num_docs, [0]))[0]
        n = n_total
    else:
        n = max_lids
        n_total = max_lids
    if verbose:
        print(f"[pool] scanning {n:,} lids across {shards} processes "
              f"(min_chunks={min_article_chunks})...", flush=True)

    bounds = [(i * n // shards, (i + 1) * n // shards, min_article_chunks)
              for i in range(shards)]
    t0 = time.time()
    results: list = []
    with ProcessPoolExecutor(max_workers=shards, mp_context=ctx) as ex:
        for part in ex.map(_scan_range, bounds):
            results.append(part)
    scan_s = time.time() - t0

    # Merge boundary-split pieces by title, then re-filter by min_chunks.
    pieces: dict = defaultdict(list)
    for part in results:
        for (start, title, bodies) in part:
            pieces[title].append((start, bodies))
    articles: list = []
    for title, plist in pieces.items():
        plist.sort(key=lambda x: x[0])
        bodies = [b for _, bb in plist for b in bb]
        if len(bodies) >= min_article_chunks:
            articles.append((title, bodies))

    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump({"min_article_chunks": min_article_chunks,
                     "n_lids_scanned": n,
                     "articles": articles}, f, protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        n_chunks = sum(len(b) for _, b in articles)
        rate = n / max(scan_s, 1e-9)
        print(f"[pool] scanned {n:,} lids in {scan_s:.1f}s ({rate:,.0f} lids/s); "
              f"kept {len(articles):,} articles / {n_chunks:,} chunks; "
              f"wrote {cache} ({os.path.getsize(cache)/1e6:.1f} MB)")
    return cache


# ───────────────────────────── sampling (use) ──────────────────────────────

class ArticlePool:
    """In-memory article pool with length-bucketed sampling.

    Reproduces `sample_article_run`'s output shape (a list of
    `{"title", "body"}` chunk dicts) without any Lucene lookups.
    """

    def __init__(self, cache: str = "data/wiki100w_article_pool.pkl"):
        with open(cache, "rb") as f:
            blob = pickle.load(f)
        self.min_article_chunks = blob["min_article_chunks"]
        self.articles: list = blob["articles"]  # [(title, [bodies...]), ...]
        # Length-bucketing index: article indices sorted ascending by chunk count,
        # with a parallel sorted length array for bisect-based ">= n" queries.
        order = sorted(range(len(self.articles)),
                       key=lambda i: len(self.articles[i][1]))
        self._sorted_idx = order
        self._sorted_lens = [len(self.articles[i][1]) for i in order]
        self.max_chunks = self._sorted_lens[-1] if self._sorted_lens else 0
        self.cache = cache

    def __len__(self) -> int:
        return len(self.articles)

    def _candidates_with_min_len(self, n: int) -> list:
        """Article indices whose chunk count is >= n (a slice of the sorted index)."""
        pos = bisect.bisect_left(self._sorted_lens, n)
        return self._sorted_idx[pos:]

    def _compose(self, n_chunks: int, rng: random.Random,
                 exclude_titles: set, max_compose_attempts: int = 400) -> list | None:
        """Assemble n_chunks by concatenating consecutive windows from several
        distinct articles (used when no single article is long enough)."""
        out: list = []
        used = set(exclude_titles)
        remaining = n_chunks
        guard = 0
        n_art = len(self.articles)
        while remaining > 0 and guard < max_compose_attempts:
            guard += 1
            title, bodies = self.articles[rng.randrange(n_art)]
            if title in used:
                continue
            take = min(len(bodies), remaining)
            start = rng.randint(0, len(bodies) - take)
            out.extend({"title": title, "body": b}
                       for b in bodies[start:start + take])
            used.add(title)
            remaining -= take
        if remaining > 0:
            return None
        return out

    def sample_run(
        self,
        n_chunks: int,
        rng: random.Random | None = None,
        exclude_titles: set | None = None,
        single_only: bool = False,
        max_attempts: int = 64,
    ) -> list | None:
        """Sample a run of `n_chunks` chunks.

        Prefers a single-article window of `n_chunks` consecutive chunks (the
        original `sample_article_run` semantics). If `single_only` is False and
        no single article is long enough (or all are excluded), it falls back to
        composing the run from multiple distinct articles so callers don't fail
        on lengths that exceed the longest article (~87 chunks).

        Every returned chunk carries its true article title, so callers can
        exclude ALL titles a run touched (not just the first).

        :returns: `[{"title", "body"}, ...]` of length `n_chunks`, or None.
        """
        if rng is None:
            rng = random.Random()
        exclude_titles = exclude_titles or set()
        cand = self._candidates_with_min_len(n_chunks)
        if cand:
            for _ in range(max_attempts):
                i = cand[rng.randrange(len(cand))]
                title, bodies = self.articles[i]
                if title in exclude_titles:
                    continue
                start = rng.randint(0, len(bodies) - n_chunks)
                window = bodies[start:start + n_chunks]
                return [{"title": title, "body": b} for b in window]
            # every sampled candidate was excluded; fall through to compose
        if single_only:
            return None
        return self._compose(n_chunks, rng, exclude_titles)


__all__ = ["build_article_pool", "ArticlePool"]
