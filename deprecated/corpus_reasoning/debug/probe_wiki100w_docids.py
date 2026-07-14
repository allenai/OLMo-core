"""Probe the wikipedia-dpr-100w Lucene index to confirm docid format.

We need to know:
1. What does a docid look like? (string format)
2. Is article identity recoverable from docid alone, or do we need the title prefix in `contents`?
3. Are chunks of the same article assigned consecutive integer Lucene ids? (lets us walk neighbors cheaply)
4. What are the field names in the raw JSON?

Run from repo root:
    python -m scripts.debug.probe_wiki100w_docids
"""

import json
import random
from collections import Counter

from pyserini.search.lucene import LuceneSearcher

INDEX_NAME = "wikipedia-dpr-100w"

def main():
    print(f"Loading {INDEX_NAME}...")
    s = LuceneSearcher.from_prebuilt_index(INDEX_NAME)
    n = s.num_docs
    print(f"  {n:,} passages\n")

    rng = random.Random(0)
    sample_lids = rng.sample(range(n), 12)

    print("=== Sample of 12 random passages ===")
    for lid in sample_lids:
        doc = s.doc(lid)
        if doc is None:
            print(f"  lid={lid}: doc is None")
            continue
        docid = doc.docid()
        raw = json.loads(doc.raw())
        keys = list(raw.keys())
        text = raw.get("contents", raw.get("body", ""))
        first_line = text.split("\n", 1)[0][:120]
        rest_preview = text.split("\n", 1)[1][:80] if "\n" in text else "(no newline)"
        print(f"  lid={lid:>9}  docid={docid!r}  keys={keys}")
        print(f"    line1: {first_line!r}")
        print(f"    line2: {rest_preview!r}")
    print()

    print("=== docid pattern analysis ===")
    docid_samples = []
    for lid in sample_lids:
        doc = s.doc(lid)
        if doc:
            docid_samples.append(doc.docid())
    has_separator = Counter()
    for d in docid_samples:
        for sep in [":", "-", "_", "::", "#"]:
            if sep in d:
                has_separator[sep] += 1
    print(f"  separators seen: {dict(has_separator)}")
    print(f"  all numeric? {all(d.lstrip('-').isdigit() for d in docid_samples)}")
    print()

    print("=== Are consecutive lucene ids the same article? ===")
    base_lid = sample_lids[0]
    for offset in range(-2, 3):
        lid = base_lid + offset
        if lid < 0 or lid >= n:
            continue
        doc = s.doc(lid)
        if not doc:
            continue
        raw = json.loads(doc.raw())
        text = raw.get("contents", raw.get("body", ""))
        first_line = text.split("\n", 1)[0][:80]
        print(f"  lid={lid:>9}  docid={doc.docid()!r}  line1={first_line!r}")
    print()

    print("=== Try fetching by string docid round-trip ===")
    for lid in sample_lids[:3]:
        doc = s.doc(lid)
        if not doc:
            continue
        docid_str = doc.docid()
        doc2 = s.doc(docid_str)
        ok = doc2 is not None and doc2.docid() == docid_str
        print(f"  lid={lid} docid={docid_str!r}  round-trip ok? {ok}")

if __name__ == "__main__":
    main()
