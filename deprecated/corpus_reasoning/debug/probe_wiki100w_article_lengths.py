"""Estimate the distribution of Wikipedia-100w article lengths.

Adjacent Lucene ids belong to the same article (verified in
probe_wiki100w_docids.py), so we can scan a contiguous slice and group
consecutive same-title chunks to recover per-article chunk counts.

Scans a 1M-chunk slice (representative of the full 21M corpus, since the
index has no obvious length-based ordering) and reports how many articles
exceed various chunk-count thresholds.

Run from repo root:
    python -m scripts.debug.probe_wiki100w_article_lengths
"""

import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import groupby

from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm

INDEX_NAME = "wikipedia-dpr-100w"
SLICE_START = 5_000_000   # mid-index, away from edge effects
SLICE_LEN = 1_000_000
N_THREADS = 32

_QUOTED = re.compile(r'^"(.+)"\s*$')


def main():
    print(f"Loading {INDEX_NAME}...")
    s = LuceneSearcher.from_prebuilt_index(INDEX_NAME)
    n_total = s.num_docs
    print(f"  {n_total:,} total chunks; scanning lids "
          f"[{SLICE_START}, {SLICE_START + SLICE_LEN})")

    def title_of(lid):
        doc = s.doc(lid)
        if doc is None:
            return None
        try:
            raw = json.loads(doc.raw())
        except Exception:
            return None
        text = raw.get("contents", raw.get("body", ""))
        line1 = text.split("\n", 1)[0].strip()
        m = _QUOTED.match(line1)
        return m.group(1) if m else line1

    lids = range(SLICE_START, SLICE_START + SLICE_LEN)
    with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
        titles = list(tqdm(ex.map(title_of, lids), total=SLICE_LEN, desc="fetch titles"))

    # Group consecutive same-title chunks. Drop boundary articles (the very
    # first and last runs may be truncated by the slice).
    runs = [(t, sum(1 for _ in g)) for t, g in groupby(titles)]
    runs = [r for r in runs if r[0]]  # drop None titles
    if len(runs) >= 3:
        runs = runs[1:-1]  # drop boundary partial articles
    lengths = [L for _, L in runs]

    print(f"\nArticles in slice (excl. partial-at-boundary): {len(lengths):,}")
    print(f"Chunks accounted for: {sum(lengths):,}")
    print(f"Mean / median / max chunks per article: "
          f"{sum(lengths)/len(lengths):.2f} / "
          f"{sorted(lengths)[len(lengths)//2]} / {max(lengths)}")

    # Bucketed histogram
    buckets = [(1, 1), (2, 4), (5, 9), (10, 19), (20, 49),
               (50, 99), (100, 199), (200, 499), (500, 10**9)]
    hist = Counter()
    for L in lengths:
        for lo, hi in buckets:
            if lo <= L <= hi:
                hist[(lo, hi)] += 1
                break
    print("\nArticle-length histogram (slice):")
    for lo, hi in buckets:
        bucket = f"≥{lo}" if hi == 10**9 else f"{lo}-{hi}"
        n = hist[(lo, hi)]
        pct = 100 * n / len(lengths)
        print(f"  {bucket:>10}: {n:>7,} ({pct:5.2f}%)")

    # Threshold counts + extrapolation to the full corpus
    scale = n_total / SLICE_LEN
    print(f"\nThreshold counts (slice → extrapolated to full corpus, ×{scale:.1f}):")
    for T in [10, 20, 30, 50, 100, 150, 200, 300, 500]:
        n = sum(1 for L in lengths if L >= T)
        print(f"  ≥{T:>3} chunks: {n:>7,} articles in slice  →  "
              f"~{int(n * scale):>9,} articles in full corpus")


if __name__ == "__main__":
    main()
