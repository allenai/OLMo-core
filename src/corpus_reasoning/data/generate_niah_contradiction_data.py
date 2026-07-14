"""Generate N-ified ("NIAH") contradiction data by deriving it from existing
O(N^2) contradiction examples.

The O(N^2) contradiction task asks the model to find ALL contradicting pairs
among N claims (every pair must be compared). The N-ified variant is the O(N)
analogue: a single *query* claim is given, and the model must find the one
document in the corpus that contradicts it — a needle-in-a-haystack retrieval.

This generator does NOT call any LLM. It reuses the gold pairs already mined in
`contradiction_{train,eval}_pubmed_*.jsonl`: for a chosen gold pair (a, b) we
promote one element to the query and leave the other as the needle, with every
other claim (including the other gold pairs, which do NOT contradict the query)
acting as distractors.

Output schema matches the unified single-doc *retrieval* format consumed by
`scripts/eval/evaluate.py --task niah_contradiction` (or --task retrieval):

    {
      "documents": [{"title": null, "text": ...}, ...],   # N distractors + 1 needle
      "queries":   ["<claim being contradicted>"],
      "answers":   [],
      "gold_doc_indices": [needle_pos],                    # 0-indexed, single doc
      "source": "niah_contradiction"
    }

Usage:
    python scripts/data/generate_niah_contradiction_data.py \\
        --from-train data/contradiction_train_pubmed_both_n100_k3.jsonl \\
        --from-eval  data/contradiction_eval_pubmed_both_n100_k3.jsonl \\
        --pairs-per-example 1 --output-dir data
"""

import argparse
import json
import random


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def derive_niah_examples(src, pairs_per_example, num_docs, rng):
    """Turn one O(N^2) contradiction example into >=1 NIAH examples.

    For each selected gold pair (a, b) (1-indexed claim IDs), randomly pick
    which side is the query and which is the needle, drop the query claim from
    the corpus, shuffle the rest, and record the needle's new 0-indexed slot.

    If `num_docs` is set and smaller than the available distractor count, drop
    random non-needle distractors down to that size (the needle is always kept).
    """
    docs = src["documents"]
    golds = src["gold_doc_indices"]
    k = min(pairs_per_example, len(golds))
    chosen = rng.sample(golds, k)

    out = []
    for a, b in chosen:
        pair = [a, b]
        rng.shuffle(pair)
        query_id, needle_id = pair  # 1-indexed into src docs
        claim = docs[query_id - 1]["text"]
        needle_text = docs[needle_id - 1]["text"]
        # Self-describing query so the generic single-doc retrieval instruction
        # ("identify which document is most relevant to answering the
        # question") reads correctly without any eval-side code change.
        query_text = (
            "Find the document that directly contradicts the following claim: "
            + claim
        )

        # Everything except the query claim becomes a candidate document.
        remaining = [d for i, d in enumerate(docs, start=1) if i != query_id]

        if num_docs is not None and num_docs < len(remaining):
            non_needle = [d for d in remaining if d["text"] != needle_text]
            needle_doc = next(d for d in remaining if d["text"] == needle_text)
            keep = rng.sample(non_needle, num_docs - 1)
            remaining = keep + [needle_doc]

        rng.shuffle(remaining)
        needle_pos = next(i for i, d in enumerate(remaining)
                          if d["text"] == needle_text)

        out.append({
            "documents": [{"title": None, "text": d["text"]} for d in remaining],
            "queries": [query_text],
            "answers": [],
            "gold_doc_indices": [needle_pos],
            "source": "niah_contradiction",
        })
    return out


def run(src_path, split_label, args, rng):
    src = load_jsonl(src_path)
    out = []
    for ex in src:
        out.extend(derive_niah_examples(
            ex, args.pairs_per_example, args.num_docs, rng))

    n = len(out[0]["documents"]) if out else 0
    tag = f"n{n}_p{args.pairs_per_example}"
    path = f"{args.output_dir}/niah_contradiction_{split_label}_{tag}.jsonl"
    save_jsonl(path, out)
    print(f"{split_label}: {len(src)} source -> {len(out)} NIAH examples, "
          f"n_docs={n}  ->  {path}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-train", type=str, default="")
    ap.add_argument("--from-eval", type=str, default="")
    ap.add_argument("--pairs-per-example", type=int, default=1,
                    help="NIAH examples emitted per source example (<=K).")
    ap.add_argument("--num-docs", type=int, default=None,
                    help="Optional corpus size (shrink only). Default: source "
                         "size minus the removed query claim.")
    ap.add_argument("--output-dir", type=str, default="data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    for label, path in [("train", args.from_train), ("eval", args.from_eval)]:
        if path:
            run(path, label, args, rng)


if __name__ == "__main__":
    main()
