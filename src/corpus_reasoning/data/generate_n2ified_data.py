"""N²-ify any single-query retrieval task (NQ / HotpotQA / ObliQ).

DEPRECATED — raises NotImplementedError. This is the EXHAUSTIVE bipartite
variant (every query matches a document → N matching pairs). The task now uses
the sparse k=3 design; use scripts/data/generate_qdmatch_data.py instead.

The O(N) version asks one query over a doc pool. The N²-ified version mixes N
queries AND their N pooled gold documents into one context, and asks the model
to match every query to its document(s) — O(N²), since each of N queries must
be compared against each of N docs. This is the apples-to-apples quadratic
counterpart of the linear retrieval tasks.

Built by DERIVING from existing single-query unified JSONL (no re-fetch, no
LLM): take N source examples, keep each one's query + its gold doc(s), pool all
gold docs into one shuffled document set, and record per-query gold positions.
By default the documents ARE the union of every query's gold docs, so each doc
is the answer to exactly one query (a clean bipartite matching). Extra
distractors per query can be added with --distractors-per-query (drawn from the
source examples' hard negatives / non-gold docs).

Output is the multi-query retrieval schema already consumed by
`scripts/eval/evaluate.py --task retrieval` (auto-detected as multi-query):

    {
      "documents": [{"title": ..., "text": ...}, ...],
      "queries":   ["q1", ..., "qN"],
      "answers":   ["a1", ..., "aN"],
      "gold_doc_indices": [[i], [j, k], ...],   # 0-indexed, per-query
      "source": "n2ified_<src>"
    }

Usage:
    python scripts/data/generate_n2ified_data.py \\
        --from-train data/nq_train_k20_hn19_2500.jsonl \\
        --from-eval  data/nq_validation_k20_hn19_500.jsonl \\
        --num-queries 20 --distractors-per-query 0 --output-dir data
"""

import argparse
import json
import random


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _query_unit(ex, max_gold=0):
    """Pull (query, answer, gold_doc_dicts, distractor_doc_dicts) from one
    single-query source example. `max_gold` (>0) caps gold docs per query —
    useful for ObliQ, which has ~8 large gold docs per query (keeps the pooled
    context tractable and the matching clean)."""
    docs = ex["documents"]
    gold_idx = ex["gold_doc_indices"]
    # Source gold may be flat (single-query) or already a 1-list-of-lists.
    if gold_idx and isinstance(gold_idx[0], list):
        gold_idx = gold_idx[0]
    if max_gold and len(gold_idx) > max_gold:
        gold_idx = gold_idx[:max_gold]
    gold_docs = [docs[i] for i in gold_idx]
    hn = ex.get("hard_neg_indices") or []
    gold_set = set(gold_idx)
    distractors = [docs[i] for i in hn if i not in gold_set]
    if not distractors:  # fall back to any non-gold doc
        distractors = [d for i, d in enumerate(docs) if i not in gold_set]
    q = ex["queries"][0]
    a = (ex.get("answers") or [""])[0]
    return q, a, gold_docs, distractors


def build_example(units, distractors_per_query, rng):
    """Pool a list of N (query, answer, gold_docs, distractors) units into one
    multi-query matching example."""
    documents = []          # list of doc dicts
    per_query_gold_docs = []  # list of lists of doc dicts (identity by object)
    for q, a, gold_docs, distractors in units:
        per_query_gold_docs.append(gold_docs)
        documents.extend(gold_docs)
        if distractors_per_query > 0:
            extra = distractors[:distractors_per_query]
            documents.extend(extra)

    # Shuffle the pooled documents; map each gold doc (by identity) to its slot.
    order = list(range(len(documents)))
    rng.shuffle(order)
    shuffled = [documents[i] for i in order]
    id_of = {id(documents[i]): new for new, i in enumerate(order)}

    queries, answers, gold = [], [], []
    for (q, a, _, _), gdocs in zip(units, per_query_gold_docs):
        queries.append(q)
        answers.append(a)
        gold.append(sorted(id_of[id(d)] for d in gdocs))

    src = units[0][0]  # placeholder, overwritten by caller tag
    return {
        "documents": shuffled,
        "queries": queries,
        "answers": answers,
        "gold_doc_indices": gold,   # per-query, 0-indexed
        "source": "n2ified",
    }


def run(src_path, split_label, count, args, rng, src_tag):
    src = load_jsonl(src_path)
    units = [_query_unit(ex, args.max_gold_per_query) for ex in src]
    N = args.num_queries
    if len(units) < N:
        raise RuntimeError(f"source has {len(units)} queries < N={N}")

    # Each example samples N DISTINCT source units; units may recur across
    # examples (queries are independent), so we can emit any --num-* count
    # regardless of source size, even at large N.
    examples = []
    for _ in range(count):
        units_i = rng.sample(units, N)
        ex = build_example(units_i, args.distractors_per_query, rng)
        ex["source"] = f"n2ified_{src_tag}"
        examples.append(ex)

    n_docs = len(examples[0]["documents"]) if examples else 0
    dtag = f"_d{args.distractors_per_query}" if args.distractors_per_query else ""
    tag = f"{src_tag}_q{N}{dtag}"
    path = f"{args.output_dir}/n2ified_{split_label}_{tag}.jsonl"
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"{split_label}: {len(src)} source units -> {len(examples)} N²-ified "
          f"examples, {N} queries x {n_docs} docs  ->  {path}")


def main():
    raise NotImplementedError(
        "generate_n2ified_data.py is deprecated. It builds the EXHAUSTIVE "
        "bipartite variant where every one of the N queries matches a document "
        "(N matching pairs). The task now uses the SPARSE query-document "
        "matching design (only k=3 relevant queries, the rest match nothing) — "
        "use scripts/data/generate_qdmatch_data.py (default --num-relevant 3) "
        "for NQ / HotpotQA / ObliQ instead."
    )
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-train", default="")
    ap.add_argument("--from-eval", default="")
    ap.add_argument("--num-queries", type=int, default=20,
                    help="N: queries (and gold docs) pooled per example — the "
                         "O(N²) scaling axis")
    ap.add_argument("--distractors-per-query", type=int, default=0,
                    help="extra non-gold docs added per query (0 = pure "
                         "bipartite matching; docs are exactly the N gold sets)")
    ap.add_argument("--num-train", type=int, default=2000)
    ap.add_argument("--num-eval", type=int, default=300)
    ap.add_argument("--max-gold-per-query", type=int, default=0,
                    help="cap gold docs kept per query (0 = keep all). Set 1 "
                         "for ObliQ to keep pooled context tractable.")
    ap.add_argument("--src-tag", default="",
                    help="short name for the source task (nq/hpqa/obliq); "
                         "inferred from filename if omitted")
    ap.add_argument("--output-dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    def infer_tag(path):
        if args.src_tag:
            return args.src_tag
        name = path.split("/")[-1]
        for t in ("hotpotqa", "hpqa", "nq", "obliq", "musique"):
            if name.startswith(t):
                return "hpqa" if t == "hotpotqa" else t
        return "src"

    rng = random.Random(args.seed)
    for label, path, count in [("train", args.from_train, args.num_train),
                               ("eval", args.from_eval, args.num_eval)]:
        if path and count > 0:
            run(path, label, count, args, rng, infer_tag(path))


if __name__ == "__main__":
    main()
