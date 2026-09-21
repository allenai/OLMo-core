#!/usr/bin/env python
"""Replace documents an expanded rung repeated inside one example.

`expand_ctc_rung.py` draws fillers from "documents that are non-gold in EVERY example", which
includes documents already sitting in the example being grown -- so a filler can land on top of a
text the example already had. The shipped suite ladders carry 0.07% (contradiction) to 0.5% (nq)
duplicated documents this way; on a source whose pool is small relative to the docs a long rung
needs, the rate is several times higher (contra_fever: 1.8%).

Duplicates never touch gold (gold texts are excluded from the pool by construction), so they do not
corrupt a label. What they do is quietly shrink the effective corpus: a 4,411-document example with
1.8% repeats is really ~4,330 distinct documents, and the repeats make the retrieval haystack
slightly easier than its label claims.

The fix is positional, which is what keeps it safe: each duplicate is REPLACED IN PLACE by a pool
document the example does not already contain. Document count, every index field and therefore
every gold index are untouched -- only the text at a duplicated slot changes.
"""
from __future__ import annotations

import argparse
import json
import os
import random


def doc_text(doc) -> str:
    return doc["text"] if isinstance(doc, dict) else str(doc)


def build_pool(rows, base: int, pairs: bool) -> list:
    """Documents that are non-gold in every example -- expand_ctc_rung's own pool definition."""
    gold_texts, seen = set(), {}
    for row in rows:
        docs = row["documents"]
        for g in row.get("gold_doc_indices") or []:
            for idx in (g if pairs else [g]):
                gold_texts.add(doc_text(docs[idx - base]))
    for row in rows:
        for doc in row["documents"]:
            seen.setdefault(doc_text(doc), doc)
    return [doc for text, doc in seen.items() if text not in gold_texts]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--files", nargs="+", required=True, help="rung JSONLs to rewrite in place")
    ap.add_argument("--base", type=int, default=1, help="gold index base (contradiction = 1)")
    ap.add_argument("--pairs", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for path in args.files:
        rows = [json.loads(line) for line in open(path)]
        pool = build_pool(rows, args.base, args.pairs)
        pool_texts = [doc_text(d) for d in pool]
        replaced = total = 0
        for i, row in enumerate(rows):
            rng = random.Random(args.seed + i)
            docs = row["documents"]
            total += len(docs)
            present = set()
            order = list(range(len(pool)))
            rng.shuffle(order)
            cursor = 0
            for pos, doc in enumerate(docs):
                text = doc_text(doc)
                if text not in present:
                    present.add(text)
                    continue
                while cursor < len(order) and pool_texts[order[cursor]] in present:
                    cursor += 1
                if cursor >= len(order):
                    break  # pool exhausted; leave the remaining duplicates rather than fail
                docs[pos] = pool[order[cursor]]
                present.add(pool_texts[order[cursor]])
                cursor += 1
                replaced += 1
        rate = replaced / max(1, total)
        print(f"{os.path.basename(path)}: replaced {replaced}/{total} docs ({rate:.4%})", flush=True)
        if args.dry_run:
            continue
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        os.replace(tmp, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
