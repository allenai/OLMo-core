"""Smoke test for scripts.lib.wiki100w_sample.

Exercises each primitive once with small parameters and prints what it
returned, so we can eyeball whether the outputs look sensible before
plugging them into the data generators.

Run from repo root:
    python -m scripts.debug.probe_wiki100w_helper
"""

import random

from corpus_reasoning.lib.bm25 import BM25Searcher
from corpus_reasoning.lib.wiki100w_sample import (
    sample_article_run,
    sample_two_articles,
    pick_chunk_with_claim,
    find_related_chunks,
    sample_random_chunks,
)


def _truncate(s, n=120):
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[:n] + "..."


def main():
    rng = random.Random(0)
    s = BM25Searcher()

    print("\n=== sample_article_run(n_chunks=5) ===")
    run = sample_article_run(s, 5, rng=rng)
    if run is None:
        print("  FAILED to find a 5-chunk article run")
    else:
        for i, d in enumerate(run):
            print(f"  [{i}] lid={d['lid']:>9} title={d['title']!r}")
            print(f"       body: {_truncate(d['body'])}")

    print("\n=== sample_two_articles(n_majority=4, n_outlier=1) ===")
    pair = sample_two_articles(s, 4, 1, rng=rng)
    if pair is None:
        print("  FAILED")
    else:
        maj, out = pair
        print(f"  majority title: {maj[0]['title']!r}  ({len(maj)} chunks)")
        print(f"  outlier title:  {out[0]['title']!r}  ({len(out)} chunks)")
        print(f"  outlier body:   {_truncate(out[0]['body'])}")

    print("\n=== pick_chunk_with_claim() ===")
    a = pick_chunk_with_claim(s, rng=rng)
    if a is None:
        print("  FAILED")
    else:
        print(f"  title: {a['title']!r}")
        print(f"  body:  {_truncate(a['body'])}")
        print(f"  {len(a['claims'])} claim sentence(s):")
        for c in a['claims'][:3]:
            print(f"    - {_truncate(c)}")

    print("\n=== find_related_chunks(query=A's first claim, k=10) ===")
    if a is not None:
        related = find_related_chunks(
            s, a['claims'][0], k=10,
            exclude_titles={a['title']},
            require_claim=True,
        )
        print(f"  {len(related)} usable related chunks")
        for r in related[:5]:
            print(f"  - title={r['title']!r}")
            print(f"    body: {_truncate(r['body'])}")
            print(f"    {len(r['claims'])} claims; first: {_truncate(r['claims'][0])}")

    print("\n=== sample_random_chunks(n=3) ===")
    fillers = sample_random_chunks(s, 3, rng=rng)
    for d in fillers:
        print(f"  - title={d['title']!r}  body: {_truncate(d['body'])}")


if __name__ == "__main__":
    main()
