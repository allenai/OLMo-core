"""
Per-shard sanity + in-loop-signal constants for the fast2k screening loop.

Two things the collector cannot compute from Beaker logs alone:

1. **The chance floor of the training CE.** Outlier's label is a list of ``k`` document ids drawn
   from ``n``; a model that has learned nothing but the OUTPUT FORMAT still has to pick a subset,
   so its cross-entropy bottoms out at ``ln C(n, k) / (answer tokens)``. A soft-token arm that
   collapses to "guess ids from the visible list" parks EXACTLY there
   (``debug/ds64/xhdr_collapse_diagnosis.md``), which is what makes this number the cheapest
   possible early kill signal -- readable at step 10, not after an eval.
   Reported as ``ce_floor`` (mean over examples of ``ln C(n_i, k_i)``, divided by the shard's mean
   labelled-token count from ``metadata.json``).
2. **Train/eval overlap.** The training rows are sliced from the ds64 2k POOL; the eval rung is a
   separate file from ``outlier_lengthmix/eval_rungs``. Fingerprint = sha1 over the example's
   document texts (+ gold indices), so an identical example is caught even if the rows differ in
   whitespace or field order. Also reports the DOCUMENT-level overlap, which is expected to be
   nonzero (both draw passages from the same wiki100w corpus) and is not a leak by itself.

    python debug/ds64_fast2k/fast2k_stats.py --arm-jsonl ARM.jsonl --shard-dir SHARD \
        --eval-jsonl RUNG_2048.jsonl --out SHARD/fast2k_stats.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os


def _docs(ex):
    d = ex.get("documents") or []
    out = []
    for x in d:
        if isinstance(x, dict):
            out.append(str(x.get("text") or x.get("content") or x.get("passage") or x))
        else:
            out.append(str(x))
    return out


def _gold(ex):
    g = ex.get("gold_doc_indices")
    if g is None:
        g = ex.get("gold_indices") or []
    if g and isinstance(g[0], list):  # per-query lists
        g = [i for sub in g for i in sub]
    return sorted(int(i) for i in g)


def _fp(ex):
    h = hashlib.sha1()
    for t in _docs(ex):
        h.update(t.strip().encode("utf-8", "replace"))
        h.update(b"\x00")
    h.update(str(_gold(ex)).encode())
    return h.hexdigest()


def _doc_fps(ex):
    return {hashlib.sha1(t.strip().encode("utf-8", "replace")).hexdigest() for t in _docs(ex)}


def read(path, limit=0):
    out = []
    with open(path) as f:
        for i, ln in enumerate(f):
            if not ln.strip():
                continue
            if limit and i >= limit:
                break
            out.append(json.loads(ln))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm-jsonl", required=True)
    ap.add_argument("--shard-dir", required=True, help="tokenized shard (for metadata.json)")
    ap.add_argument("--eval-jsonl", default="", help="the 2k eval rung to check overlap against")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    train = read(args.arm_jsonl)
    meta = json.load(open(os.path.join(args.shard_dir, "metadata.json")))
    ans_tok = meta["num_loss_tokens"] / max(1, meta["num_instances"])

    ns, ks, logs = [], [], []
    for ex in train:
        n, k = len(_docs(ex)), len(_gold(ex))
        if n and 0 < k <= n:
            ns.append(n)
            ks.append(k)
            logs.append(math.log(math.comb(n, k)))
    mean_logc = sum(logs) / max(1, len(logs))
    stats = {
        "arm_jsonl": args.arm_jsonl,
        "shard_dir": args.shard_dir,
        "n_examples": len(train),
        "shard_num_instances": meta["num_instances"],
        "shard_num_dropped": meta["num_dropped"],
        "shard_max_example_len": meta["max_example_len"],
        "shard_min_example_len": meta["min_example_len"],
        "shard_mean_example_len": meta["num_tokens"] / max(1, meta["num_instances"]),
        "mean_answer_tokens": ans_tok,
        "n_docs_min": min(ns) if ns else None,
        "n_docs_mean": (sum(ns) / len(ns)) if ns else None,
        "n_docs_max": max(ns) if ns else None,
        "k_gold_mean": (sum(ks) / len(ks)) if ks else None,
        "mean_log_choose": mean_logc,
        # CE a format-only guesser cannot beat: ln C(n,k) spread over the answer tokens.
        "ce_floor": mean_logc / max(1e-9, ans_tok),
        # f1 a uniform guesser gets when it names k ids out of n (the xhdr collapse signature)
        "f1_uniform_guess": (sum(k / n for n, k in zip(ns, ks)) / len(ns)) if ns else None,
    }
    if args.eval_jsonl and os.path.exists(args.eval_jsonl):
        ev = read(args.eval_jsonl)
        tfp = {_fp(e) for e in train}
        efp = {_fp(e) for e in ev}
        tdoc, edoc = set(), set()
        for e in train:
            tdoc |= _doc_fps(e)
        for e in ev:
            edoc |= _doc_fps(e)
        stats.update({
            "eval_jsonl": args.eval_jsonl, "eval_size": len(ev),
            "example_overlap": len(tfp & efp),
            "example_overlap_frac": len(tfp & efp) / max(1, len(efp)),
            "document_overlap": len(tdoc & edoc),
            "document_overlap_frac": len(tdoc & edoc) / max(1, len(edoc)),
            "eval_n_docs_mean": (sum(len(_docs(e)) for e in ev) / max(1, len(ev))),
        })
    print(json.dumps(stats, indent=2))
    if stats.get("example_overlap"):
        print(f"!!! TRAIN/EVAL EXAMPLE OVERLAP: {stats['example_overlap']} examples -- DO NOT USE THIS SHARD")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
