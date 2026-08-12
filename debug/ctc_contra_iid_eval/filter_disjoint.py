"""Drop eval examples whose gold sentences were consumed by the CTC training file.

The generator gives a disjoint train/eval split for free when both splits come out of ONE run
(a single cursor walks the shuffled pair pool, train first, eval second). The CTC training run
emitted no eval split and exhausted its 40k-abstract pool, so the holdout has to be generated
separately -- and a separate run re-shuffles, so disjointness must be enforced rather than
assumed. This is the same check that caught contradiction_eval_pubmed_realistic_n100_k3.jsonl
sharing gold sentences with training.

Filters on train GOLD sentences (the ones the model was taught to pair). Overlap with train
*filler* sentences is reported but not filtered: seeing a sentence as a distractor does not
teach the contradiction relation, and filtering on it would reject most of the pool.
"""

import argparse
import json


def gold_sentences(record):
    """Yield the gold sentence strings of one record (gold_doc_indices is 1-indexed)."""
    docs = [d["text"] if isinstance(d, dict) else d for d in record["documents"]]
    for pair in record["gold_doc_indices"]:
        for idx in pair:
            i = idx - 1
            if 0 <= i < len(docs):
                yield docs[i].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval-in", required=True)
    ap.add_argument("--eval-out", required=True)
    ap.add_argument("--keep", type=int, default=500)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    print(f"[filter] scanning train golds: {args.train}", flush=True)
    train_gold = set()
    train_any = set()          # 64-bit hashes; 10M+ sentences, too big to hold as strings
    n_train = 0
    with open(args.train) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_train += 1
            docs = [d["text"] if isinstance(d, dict) else d for d in r["documents"]]
            for d in docs:
                train_any.add(hash(d.strip()))
            train_gold.update(gold_sentences(r))
            if n_train % 2000 == 0:
                print(f"  {n_train} train examples, {len(train_gold)} gold sentences", flush=True)
    print(f"[filter] train: {n_train} examples, {len(train_gold)} gold sentences, "
          f"{len(train_any)} distinct doc sentences", flush=True)

    kept, dropped_gold, filler_hits, n_in = [], 0, 0, 0
    with open(args.eval_in) as f:
        for line in f:
            r = json.loads(line)
            n_in += 1
            golds = list(gold_sentences(r))
            if any(g in train_gold for g in golds):
                dropped_gold += 1
                continue
            if any(hash(g) in train_any for g in golds):
                filler_hits += 1          # seen only as a distractor in training -- kept
            kept.append(r)
            if len(kept) >= args.keep:
                break

    with open(args.eval_out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    report = {
        "train_file": args.train,
        "train_examples": n_train,
        "train_gold_sentences": len(train_gold),
        "eval_candidates_read": n_in,
        "dropped_gold_overlap": dropped_gold,
        "kept": len(kept),
        "kept_with_filler_only_overlap": filler_hits,
        "drop_rate": round(dropped_gold / max(1, n_in), 4),
        "eval_out": args.eval_out,
    }
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)

    if len(kept) < args.keep:
        print(f"!!! only {len(kept)} examples survived, wanted {args.keep} -- "
              f"regenerate with a larger --num-eval", flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
