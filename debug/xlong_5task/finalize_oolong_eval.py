"""Finalize the generated oolong eval rungs into the v2 eval layout, disjoint from OUR train pool.

``corpus_reasoning/data/finalize_oolong_v2.py`` asserts disjointness only against the **v1**
``oolong_test_synth_ctx*_splittrain.jsonl`` files. The 2k->256k build generates a brand-new oolong
training pool, so the contamination that matters here is against *that*, not against v1. Same
signature scheme (query + first 200 chars of the packed document body), same output naming
(``oolong_test_synth_ctx{ctx}_spliteval.jsonl``) so the existing eval wiring picks the files up.

Emits at least ``--min-per-rung`` examples per rung and fails loudly if a rung falls short after
contamination filtering, rather than silently writing a small eval set.
"""

import argparse
import glob
import json
import os
import sys


def load(p: str) -> list:
    return [json.loads(line) for line in open(p) if line.strip()]


def sig(ex: dict) -> str:
    q = (ex.get("queries") or [""])[0]
    body = ex["documents"][0]["text"] if ex.get("documents") else ""
    return q + "||" + body[:200]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-root", default="/data/prasann/xlong5/oolong_eval_gen")
    ap.add_argument("--train-glob",
                    default="/data/prasann/xlong5/pools/oolong/*/*.jsonl")
    ap.add_argument("--out-dir", default="/data/prasann/xlong5/eval/oolong")
    ap.add_argument("--rungs", default="2k:2048,4k:4096,8k:8192,16k:16384,32k:32768,64k:65536,128k:131072,256k:262144")
    ap.add_argument("--min-per-rung", type=int, default=500)
    args = ap.parse_args()

    print("building train signature set ...", flush=True)
    train_keys = set()
    for p in glob.glob(args.train_glob):
        for ex in load(p):
            train_keys.add(sig(ex))
    print(f"  {len(train_keys)} train signatures from {len(glob.glob(args.train_glob))} files",
          flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    failed = []
    for spec in args.rungs.split(","):
        label, ctx = spec.split(":")
        files = sorted(glob.glob(os.path.join(args.gen_root, label, "*.jsonl")))
        rows = [ex for f in files for ex in load(f)]
        n_raw = len(rows)

        seen = set()
        kept = []
        n_contam = n_dup = 0
        for ex in rows:
            s = sig(ex)
            if s in train_keys:
                n_contam += 1
                continue
            if s in seen:
                n_dup += 1
                continue
            seen.add(s)
            kept.append(ex)

        out = os.path.join(args.out_dir, f"oolong_test_synth_ctx{ctx}_spliteval.jsonl")
        with open(out, "w") as f:
            for ex in kept:
                f.write(json.dumps(ex) + "\n")
        status = "OK" if len(kept) >= args.min_per_rung else "SHORT"
        if status == "SHORT":
            failed.append(label)
        print(f"[{label:>4}] raw={n_raw} train_contaminated={n_contam} dup={n_dup} "
              f"-> kept={len(kept)}  {status}  {out}", flush=True)

    if failed:
        print(f"\nRUNGS BELOW {args.min_per_rung}: {failed}", flush=True)
        sys.exit(1)
    print(f"\nall rungs >= {args.min_per_rung}", flush=True)


if __name__ == "__main__":
    main()
