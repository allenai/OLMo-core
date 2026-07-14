"""Build contradiction pair JSONLs with HARD negatives (same-abstract pairs).

The default negatives in prepare_contradiction_pair_classifier.py are random
sentence pairs from one example — unrelated topics, so a classifier can win by
detecting topical relatedness instead of judging contradiction.

This builder instead draws negatives as pairs of two distinct claim sentences
from the SAME PubMed abstract: topically related and (since an abstract is
internally consistent) non-contradictory. The classifier can no longer use
topic mismatch as a shortcut.

  positive (label 1): a gold contradiction pair from the task file
  negative (label 0): two claim sentences from the same PubMed abstract

Loads the PubMed pool once and emits all four mode/split files together, with
DISJOINT abstracts for the train vs eval negatives (no sentence leakage).

Usage:
    python scripts/data/build_contradiction_hardneg_pairs.py \
        --obvious-train data/contradiction_modecmp_obvious_train.jsonl \
        --obvious-eval  data/contradiction_modecmp_obvious_eval.jsonl \
        --subtle-train  data/contradiction_modecmp_subtle_train.jsonl \
        --subtle-eval   data/contradiction_modecmp_subtle_eval.jsonl \
        --out-prefix data/contradiction_modecmp_hardneg \
        --pool-abstracts 20000 --neg-ratio 3
"""

import argparse
import random
from collections import defaultdict

from corpus_reasoning.lib.io import load_jsonl, save_jsonl
from corpus_reasoning.data.generate_pubmed_contradiction_data import load_pubmed_pool


def extract_positives(task_file):
    """Gold contradiction pairs from a contradiction task JSONL (1-indexed)."""
    out = []
    for ex_id, ex in enumerate(load_jsonl(task_file)):
        docs = ex["documents"]
        n = len(docs)
        for a, b in ex["gold_doc_indices"]:
            if a == b or not (1 <= a <= n and 1 <= b <= n):
                continue
            out.append({
                "text_a": docs[a - 1]["text"],
                "text_b": docs[b - 1]["text"],
                "label": 1,
                "_meta": {"source_example_id": ex_id, "pair": [a - 1, b - 1]},
            })
    return out


def same_abstract_pairs(abstract_ids, claims_by_aid, per_abstract_cap, rng):
    """All same-abstract sentence pairs (capped per abstract for diversity)."""
    pairs = []
    for aid in abstract_ids:
        sents = claims_by_aid[aid]
        idx = list(range(len(sents)))
        cand = [(i, j) for i in idx for j in idx if i < j]
        rng.shuffle(cand)
        for i, j in cand[:per_abstract_cap]:
            a, b = sents[i], sents[j]
            if rng.random() < 0.5:
                a, b = b, a
            pairs.append({"text_a": a, "text_b": b, "label": 0,
                          "_meta": {"neg_type": "same_abstract", "abstract_id": aid}})
    rng.shuffle(pairs)
    return pairs


def take(pool, n, label):
    if len(pool) < n:
        raise RuntimeError(
            f"need {n} {label} negatives but pool only has {len(pool)}; "
            f"raise --pool-abstracts or --per-abstract-cap"
        )
    out, pool[:] = pool[:n], pool[n:]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--obvious-train", required=True)
    p.add_argument("--obvious-eval", required=True)
    p.add_argument("--subtle-train", required=True)
    p.add_argument("--subtle-eval", required=True)
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--pool-abstracts", type=int, default=20000)
    p.add_argument("--neg-ratio", type=float, default=3.0)
    p.add_argument("--per-abstract-cap", type=int, default=4,
                   help="max negative pairs drawn from a single abstract")
    p.add_argument("--eval-abstract-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)

    # ── negative pool: claim sentences grouped by abstract ──
    claim_pool, _ = load_pubmed_pool(args.pool_abstracts, args.seed)
    claims_by_aid = defaultdict(list)
    for sent, aid in claim_pool:
        claims_by_aid[aid].append(sent)
    multi = [aid for aid, s in claims_by_aid.items() if len(s) >= 2]
    rng.shuffle(multi)
    n_eval = int(round(len(multi) * args.eval_abstract_frac))
    eval_aids, train_aids = multi[:n_eval], multi[n_eval:]
    print(f"abstracts with >=2 claims: {len(multi)} "
          f"(train={len(train_aids)}, eval={len(eval_aids)})")

    train_negs = same_abstract_pairs(train_aids, claims_by_aid, args.per_abstract_cap, rng)
    eval_negs = same_abstract_pairs(eval_aids, claims_by_aid, args.per_abstract_cap, rng)
    print(f"negative pool: train={len(train_negs)}  eval={len(eval_negs)}")

    # ── positives + matched negatives per mode/split ──
    # obvious and subtle of the same split each draw from the full negative
    # pool for that split (negatives are mode-independent; a fresh shuffled
    # copy per draw means obvious and subtle may share negative pairs, which
    # is fine and keeps the only cross-mode difference the positives).
    specs = [
        ("obvious_train", args.obvious_train),
        ("obvious_eval", args.obvious_eval),
        ("subtle_train", args.subtle_train),
        ("subtle_eval", args.subtle_eval),
    ]
    pools = {"train": train_negs, "eval": eval_negs}

    for name, task_file in specs:
        split = "train" if name.endswith("train") else "eval"
        pos = extract_positives(task_file)
        n_neg = int(round(args.neg_ratio * len(pos)))
        # fresh shuffled copy so obvious and subtle each get the full pool
        pool = list(pools[split])
        rng.shuffle(pool)
        neg = take(pool, n_neg, split)
        rows = pos + neg
        rng.shuffle(rows)
        out = f"{args.out_prefix}_{name}_pairs.jsonl"
        save_jsonl(out, rows)
        print(f"  {name}: {len(pos)} pos + {len(neg)} neg -> {out}")


if __name__ == "__main__":
    main()
