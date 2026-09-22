#!/usr/bin/env python
"""Sort unordered gold pairs low-high in an existing rung file.

`expand_ctc_rung.py` shuffles documents and remaps indices but did not re-sort the pair, so a
contradiction pair that was (low, high) in the source came out in arbitrary order. Contradiction is
scored as a SET INTERSECTION over sorted pairs, so an unsorted pair can never be matched: it costs
recall on every example containing it, and the loss looks exactly like a long-context collapse
rather than a data defect. The vendored spec's `_check_gold` now refuses such a file outright, which
is how this was found.

Sorting is meaning-preserving here -- a contradiction pair is an unordered pair of documents. Do NOT
point this at a task whose pairs are ordered by construction (qdmatch's (query, doc)); those are
already consistent and sorting them would corrupt them, so the task is named explicitly rather than
inferred.
"""
from __future__ import annotations

import argparse
import json
import os


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--field", default="gold_doc_indices")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for path in args.files:
        rows = [json.loads(line) for line in open(path)]
        fixed = pairs = 0
        for row in rows:
            for pair in row.get(args.field) or []:
                if isinstance(pair, list) and len(pair) == 2:
                    pairs += 1
                    if pair[0] > pair[1]:
                        pair.sort()
                        fixed += 1
        print(f"{os.path.basename(path)}: sorted {fixed}/{pairs} pairs", flush=True)
        if args.dry_run or not fixed:
            continue
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        os.replace(tmp, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
