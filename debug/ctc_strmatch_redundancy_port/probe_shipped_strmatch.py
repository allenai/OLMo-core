"""Measure the bag-of-words shortcut on the SHIPPED (pre-migration) strmatch eval data.

The criterion is a run of >= k CONTIGUOUS words. If the pair with the largest order-free word
overlap is always gold, the contiguity half of the criterion is decorative and the task is
solvable by set intersection.
"""

import json
import sys
from itertools import combinations

path = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 50

hits = total = 0
run_hits = 0
vocab = set()
gold_share, hn_share = [], []
ndocs = None
with open(path, encoding="utf-8") as fh:
    for i, line in enumerate(fh):
        if i >= limit:
            break
        row = json.loads(line)
        docs = [d["text"].split() for d in row["documents"]]
        for d in docs:
            vocab.update(d)
        ndocs = len(docs)
        gold = {tuple(sorted(p)) for p in row["gold_doc_indices"]}
        sets = [set(d) for d in docs]
        best = max(combinations(range(len(docs)), 2), key=lambda p: len(sets[p[0]] & sets[p[1]]))
        hits += tuple(sorted((best[0] + 1, best[1] + 1))) in gold
        total += 1
        # distribution of overlap sizes
        overlaps = sorted(
            (len(sets[a] & sets[b]), (a + 1, b + 1)) for a, b in combinations(range(len(docs)), 2)
        )
        top = [o for o, _ in overlaps if o > 0]
        gold_share.append(sorted(len(sets[a - 1] & sets[b - 1]) for a, b in gold))
        hn_share.append(sorted(t for t in top)[-8:])
print(f"file={path}  ndocs={ndocs}  examples={total}")
print(f"max-overlap pair is gold: {hits}/{total} = {hits/total:.3f}")
print(f"chance (random pair is gold): {len(gold_share[0])/ (ndocs*(ndocs-1)/2):.5f}")
print("gold pair overlaps (first 5):", gold_share[:5])
print("largest non-zero overlaps (first 3):", hn_share[:3])
print(f"distinct words seen over {total} examples: {len(vocab)}")
