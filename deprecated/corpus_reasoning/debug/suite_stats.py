"""Compute per-file context-length + count stats for the task-suite data.
Streams each JSONL so huge files (oolong 4M-token, etc.) don't blow memory.
Context length = sum of document texts + query chars; approx tokens = chars/4.
"""
import glob
import json
import os
import statistics
import sys


def ctx_chars(ex):
    c = sum(len(d.get("text", "")) for d in ex.get("documents", []))
    c += sum(len(q) for q in ex.get("queries", []) or [])
    return c


def file_stats(path):
    n = 0
    chars = []
    ndocs = []
    golds = []
    for line in open(path):
        if not line.strip():
            continue
        ex = json.loads(line)
        n += 1
        chars.append(ctx_chars(ex))
        ndocs.append(len(ex.get("documents", [])))
        g = ex.get("gold_doc_indices", [])
        golds.append(len(g) if g else 0)
    if not chars:
        return None
    return {
        "n": n,
        "med_chars": int(statistics.median(chars)),
        "max_chars": max(chars),
        "med_tok": int(statistics.median(chars) / 4),
        "max_tok": int(max(chars) / 4),
        "med_ndocs": int(statistics.median(ndocs)),
        "max_ndocs": max(ndocs),
        "med_gold": int(statistics.median(golds)) if golds else 0,
    }


def main():
    patterns = sys.argv[1:] or ["data/*.jsonl"]
    files = []
    for p in patterns:
        files.extend(sorted(glob.glob(p)))
    print(f"{'file':<58} {'n':>6} {'med_tok':>8} {'max_tok':>9} "
          f"{'ndocs':>7} {'gold':>5}")
    for f in files:
        s = file_stats(f)
        if not s:
            continue
        name = os.path.basename(f)
        print(f"{name:<58} {s['n']:>6} {s['med_tok']:>8,} {s['max_tok']:>9,} "
              f"{s['med_ndocs']:>4}/{s['max_ndocs']:<3} {s['med_gold']:>5}")


if __name__ == "__main__":
    main()
