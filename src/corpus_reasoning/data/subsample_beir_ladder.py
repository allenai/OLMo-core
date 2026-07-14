"""Carve fixed-k ladder rungs out of an existing rich BEIR retrieval pool.

The BEIR CE/plain generators emit one file per pool size (k = #docs/example).
To build a length-ladder cheaply (no GPU/BM25/CE rerun) we subsample a large
pool (e.g. k100) down to smaller fixed k: keep ALL gold docs, then fill the
remaining slots with a random subset of the pool's non-gold docs (which are
already CE-cleaned for fiqa / BM25-mined for scifact), shuffle, and remap
gold_doc_indices / hard_neg_indices to the new positions.

Usage:
    python scripts/data/subsample_beir_ladder.py \
        --src data/beir_fiqa_ce_test_k100_648.jsonl \
        --ks 10 20 40 80 --out-prefix data/beir_fiqa_ce_ladder
"""
import argparse
import json
import random


def load(path):
    return [json.loads(l) for l in open(path)]


def subsample_example(ex, k, rng):
    docs = ex["documents"]
    gold = set(ex.get("gold_doc_indices", []))
    hard = set(ex.get("hard_neg_indices", []))
    n = len(docs)
    gold_list = [i for i in range(n) if i in gold]
    nongold = [i for i in range(n) if i not in gold]
    if len(gold_list) >= k:
        # pathological: more golds than budget -> keep first k golds only
        keep = gold_list[:k]
    else:
        need = k - len(gold_list)
        rng.shuffle(nongold)
        keep = gold_list + nongold[:need]
    rng.shuffle(keep)
    old2new = {old: new for new, old in enumerate(keep)}
    new_docs = [docs[i] for i in keep]
    new_gold = sorted(old2new[i] for i in keep if i in gold)
    new_hard = sorted(old2new[i] for i in keep if i in hard)
    out = dict(ex)
    out["documents"] = new_docs
    out["gold_doc_indices"] = new_gold
    out["hard_neg_indices"] = new_hard
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--ks", type=int, nargs="+", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = load(args.src)
    print(f"loaded {len(src)} examples from {args.src} "
          f"(pool k={len(src[0]['documents'])})")
    for k in args.ks:
        rng = random.Random(args.seed + k)
        out = []
        dropped = 0
        for ex in src:
            if len(ex["documents"]) < k:
                dropped += 1
                continue
            out.append(subsample_example(ex, k, rng))
        n_gold = [len(e["gold_doc_indices"]) for e in out]
        # every example must retain >=1 gold or retrieval f1 is undefined
        with_gold = sum(1 for g in n_gold if g >= 1)
        path = f"{args.out_prefix}_k{k}_{len(out)}.jsonl"
        with open(path, "w") as f:
            for e in out:
                f.write(json.dumps(e) + "\n")
        med_gold = sorted(n_gold)[len(n_gold) // 2] if n_gold else 0
        print(f"  k{k}: wrote {len(out)} (dropped {dropped} short) "
              f"golds/ex median={med_gold} with_gold={with_gold}/{len(out)} -> {path}")


if __name__ == "__main__":
    main()
