"""
How long are Gutenberg prose runs, and how many sentences does a 100-word passage take?

This is the measurement that decided ``reorder``'s corpus layer. A first version drew each example
from ONE prose run; the numbers below say why that cannot work -- the median run is 22 sentences
and only 7 runs in 173,275 reach 800, so the 32k rung (233 passages, ~1,400 sentences) has no
eligible run at all in 3,000 books. Hence the per-book passage stream in
``ctc.tasks.reorder.sources.gutenberg``, whose blocks may cross a run seam.

Run on horton (the node holding the 11 GB arrow cache):
    srun -p berkeleynlp -w horton -c 8 --mem 96G bash -c '...'
"""

from __future__ import annotations

import statistics
import sys

from ctc.data.sources import gutenberg
from ctc.tasks.reorder.generate import passage_runs

MAX_BOOKS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000


def main() -> int:
    pool = gutenberg.load_pool(max_books=MAX_BOOKS, max_sentences_per_book=20000)
    lengths = sorted((len(run) for run in pool.runs), reverse=True)
    print(f"books scanned {pool.provenance['books']}, runs {len(pool.runs)}")
    print(
        f"run length: max {lengths[0]}, p99 {lengths[len(lengths) // 100]}, median {statistics.median(lengths)}"
    )
    for threshold in (100, 200, 400, 800, 1600, 2800):
        print(
            f"  runs with >= {threshold:5d} sentences: {sum(1 for n in lengths if n >= threshold)}"
        )

    per_passage = []
    for run in pool.runs[:400]:
        passages = [p for group in passage_runs(run.sentences) for p in group]
        if passages:
            per_passage.append(len(run.sentences) / len(passages))
    print(f"sentences per 100-word passage: median {statistics.median(per_passage):.2f}")
    words = [
        len(p.split())
        for run in pool.runs[:400]
        for group in passage_runs(run.sentences)
        for p in group
    ]
    print(
        f"passage words: median {statistics.median(words):.0f}, p95 {sorted(words)[int(0.95 * len(words))]}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
