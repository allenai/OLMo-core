"""Can the Gutenberg pool supply the long absence rungs? Reports the prose-run length spectrum.

The 32k rung samples a window of 438 consecutive clean prose sentences, and a run breaks at every
heading, table-of-contents line or bad split -- so "the book is long enough" is not the question,
"is there an unbroken stretch that long" is. Throwaway measurement.
"""

import sys

from ctc.data.sources import gutenberg

max_books = int(sys.argv[1]) if len(sys.argv) > 1 else 300
pool = gutenberg.load_pool(max_books=max_books)
print("provenance:", pool.provenance)
for split in ("train", "eval"):
    sub = pool.for_split(split)
    lengths = sorted((len(r) for r in sub.runs), reverse=True)
    print(f"{split}: {len(sub.runs)} runs, longest {lengths[:5]}")
    for rung, n in (("2k", 32), ("4k", 60), ("8k", 114), ("16k", 222), ("32k", 438)):
        print(f"   rung {rung:>3} n={n:>4}: {sum(1 for x in lengths if x >= n)} usable run(s)")
