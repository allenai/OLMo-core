"""Audit the IID contradiction rungs before anyone believes a number measured on them.

Checks, per rung:
  * row count (the >=500 floor) and n_docs (must sit inside the training support n in [52, 950])
  * gold-pair word-Jaccard -- the whole point of the rebuild. Train measures ~0.306; the `both`
    ladder these replace measures ~0.375. If the new rungs do not land near the train value the
    perturbation mode did not actually change and the rebuild is pointless.
  * measured prompt tokens, so the rung labels are honest (the old ladder's labels were fit on a
    contaminated pool and overshot ~1.8x)
  * gold indices in range, and identical example ids across rungs (fixed-eval-set property)
"""

import argparse
import json
import re
import statistics
from pathlib import Path

TRAIN_SUPPORT = (52, 950)


def toks(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def jaccard(a, b):
    A, B = toks(a), toks(b)
    return len(A & B) / max(1, len(A | B))


def load(path, limit=None):
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def gold_stats(records):
    ovl, oob = [], 0
    for r in records:
        docs = [d["text"] if isinstance(d, dict) else d for d in r["documents"]]
        for pair in r["gold_doc_indices"]:
            a, b = pair[0] - 1, pair[1] - 1
            if 0 <= a < len(docs) and 0 <= b < len(docs):
                ovl.append(jaccard(docs[a], docs[b]))
            else:
                oob += 1
    return ovl, oob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung-dir", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--tokenizer", default="/data/prasann/hf_models/Qwen3.5-4B-Base")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
    except Exception as e:                                    # noqa: BLE001
        print(f"[audit] no tokenizer ({e}); falling back to the fitted 170 + 42.8*n", flush=True)

    train_ovl, _ = gold_stats(load(args.train, limit=200))
    print(f"[audit] train gold-pair jaccard median={statistics.median(train_ovl):.3f}", flush=True)

    rungs = {}
    first_gold = None
    for p in sorted(Path(args.rung_dir).glob("rung_*.jsonl"),
                    key=lambda q: int(q.stem.split("_")[1])):
        label = int(p.stem.split("_")[1])
        recs = load(p)
        ovl, oob = gold_stats(recs)
        ndocs = [len(r["documents"]) for r in recs]

        if tok is not None:
            sample = recs[:25]
            lens = [len(tok("\n".join(
                d["text"] if isinstance(d, dict) else d for d in r["documents"]
            )).input_ids) for r in sample]
            tok_p50 = int(statistics.median(lens))
            tok_p95 = int(sorted(lens)[max(0, int(0.95 * len(lens)) - 1)])
        else:
            tok_p50 = tok_p95 = int(170 + 42.8 * statistics.median(ndocs))

        # Identify an example by its gold SENTENCE pairs, never by gold_doc_indices: expansion adds
        # fillers and reshuffles, so the indices legitimately move between rungs even though the
        # example is the same one. Comparing indices reports a false mismatch at every rung.
        gold_sig = []
        for r in recs:
            docs = [d["text"] if isinstance(d, dict) else d for d in r["documents"]]
            gold_sig.append(frozenset(
                frozenset((docs[a - 1].strip(), docs[b - 1].strip()))
                for a, b in r["gold_doc_indices"]
                if 0 <= a - 1 < len(docs) and 0 <= b - 1 < len(docs)
            ))
        if first_gold is None:
            first_gold = gold_sig
            same_examples = True
        else:
            same_examples = gold_sig == first_gold

        rungs[label] = {
            "rows": len(recs),
            "n_docs_min": min(ndocs),
            "n_docs_max": max(ndocs),
            "n_docs_in_train_support": TRAIN_SUPPORT[0] <= min(ndocs) and max(ndocs) <= TRAIN_SUPPORT[1],
            "gold_jaccard_median": round(statistics.median(ovl), 4),
            "gold_jaccard_mean": round(statistics.mean(ovl), 4),
            "gold_out_of_range": oob,
            "tokens_p50": tok_p50,
            "tokens_p95": tok_p95,
            "same_examples_as_smallest_rung": same_examples,
            "eval_size_ok": len(recs) >= 500,
        }
        print(f"  rung_{label}: {json.dumps(rungs[label])}", flush=True)

    result = {
        "train_gold_jaccard_median": round(statistics.median(train_ovl), 4),
        "both_ladder_reference_jaccard_median": 0.375,
        "rungs": rungs,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
