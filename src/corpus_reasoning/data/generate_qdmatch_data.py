"""Sparse query-document matching (qdmatch) — the N²-ified retrieval task.

REPLACES the old bipartite `generate_n2ified_data.py`. Instead of N queries each
matched 1:1 to N docs, an example mixes **M queries and N documents** of which only
a few queries are relevant — each relevant query is matched to ALL of its gold
documents (1 for NQ, 2 for HotpotQA bridge questions). The model must find those
sparse query→doc pairs among the M×N possible combinations — an O(M·N) all-pairs
search with a sparse gold set, the retrieval analogue of contradiction.

Built by DERIVING from existing single-query unified retrieval JSONL (nq / hpqa /
obliq) — no re-fetch, no LLM. Per example we draw DISJOINT pools of source units:
  - k  "relevant" queries -> contribute their query AND all its gold docs (the gold pairs)
  - distractor queries     -> contribute their query only (gold docs withheld -> unanswerable)
  - distractor docs         -> contribute one gold doc only (query withheld -> matches nothing)
Disjointness guarantees the only true matches are the relevant queries' gold pairs.

Multi-gold note: `--num-relevant k` counts relevant QUERIES, not pairs. With NQ
(1 gold/query) that is k pairs; with HotpotQA bridge (2 golds/query) it is 2k
pairs (each relevant query maps to both bridge docs). Set `--max-gold-per-query`
to cap (0 = keep every gold the source provides).

Indexing: a SINGLE shared 1-based index over all M+N items; each item is tagged
Query:/Document: at render time (see QDMATCH_INSTRUCTION). Gold is a list of
ordered [query_id, doc_id] pairs (1-based). Output schema:

    {
      "documents": [ {"type": "query"|"document", "text": ...}, ... ],  # presentation order
      "queries": [],                       # unused (instruction is fixed)
      "gold_pairs": [[q_id, d_id], ...],   # 1-based, ordered (query first)
      "layout": "separate"|"shuffled",
      "num_queries": M, "num_docs": N, "num_relevant": k,
      "source": "qdmatch_<src>"
    }

Layout (ablation): `separate` (default) lists all M queries, then all N docs (each
block internally shuffled); `shuffled` interleaves all M+N items in one shuffle.
Either way the index is shared and contiguous — only the order differs.

Usage (HotpotQA, 2 golds/query -> 2k pairs):
    python scripts/data/generate_qdmatch_data.py \\
        --from-train data/hotpotqa_train_k100_bridge_2000.jsonl \\
        --from-eval  data/hotpotqa_eval_k100_bridge_500.jsonl \\
        --num-docs 20 --num-relevant 3 --layout separate \\
        --num-train 2000 --num-eval 300 --output-dir data
"""

import argparse
import json
import random
from collections import defaultdict


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _query_unit(ex, max_gold=0):
    """(query, [gold_doc_item, ...]) from one single-query source example.
    Keeps ALL gold docs (capped to max_gold if >0). Returns None if no gold."""
    docs = ex["documents"]
    gold_idx = ex["gold_doc_indices"]
    if gold_idx and isinstance(gold_idx[0], list):
        gold_idx = gold_idx[0]
    if not gold_idx:
        return None
    if max_gold and len(gold_idx) > max_gold:
        gold_idx = gold_idx[:max_gold]
    q = ex["queries"][0]
    gold_docs = [{"type": "document", "text": docs[i]["text"]} for i in gold_idx]
    return q, gold_docs


def build_example(units, M, N, k, layout, rng):
    """Pool disjoint source units into one qdmatch example.

    `units` supplies a prefix of distinct units: the first k are relevant
    queries (each contributes all its gold docs), the next M-k are distractor
    queries, and the rest supply distractor docs (one gold each) up to N docs."""
    relevant = units[:k]                                  # k relevant queries
    G = sum(len(gd) for (_, gd) in relevant)              # total relevant gold docs
    if G > N:
        raise ValueError(f"relevant gold docs G={G} exceeds N={N}; raise --num-docs")
    dquery = units[k:M]                                   # M-k distractor queries
    ddoc_units = units[M:M + (N - G)]                     # N-G distractor docs

    # Query items (k relevant + M-k distractor); tag the relevant ones.
    query_items = [{"type": "query", "text": q} for (q, _) in relevant + dquery]
    for ri in range(k):
        query_items[ri]["_relq"] = ri

    # Doc items: relevant golds (tagged by their query) + distractor golds.
    doc_items = []
    for ri, (_, gd) in enumerate(relevant):
        for d in gd:
            item = dict(d)
            item["_relq"] = ri
            doc_items.append(item)
    for (_, gd) in ddoc_units:
        doc_items.append(dict(gd[0]))

    if layout == "separate":
        rng.shuffle(query_items)
        rng.shuffle(doc_items)
        items = query_items + doc_items
    elif layout == "shuffled":
        items = query_items + doc_items
        rng.shuffle(items)
    else:
        raise ValueError(f"unknown layout {layout!r}")

    # Global 1-based index; recover gold pairs via the _relq tag.
    q_idx = {}                       # relevant-query id -> global query index
    d_idxs = defaultdict(list)       # relevant-query id -> [global doc indices]
    for gi, item in enumerate(items, start=1):
        ri = item.pop("_relq", None)
        if ri is None:
            continue
        if item["type"] == "query":
            q_idx[ri] = gi
        else:
            d_idxs[ri].append(gi)
    gold_pairs = sorted([[q_idx[ri], di] for ri in range(k) for di in d_idxs[ri]])

    return {
        "documents": items,
        "queries": [],
        "gold_pairs": gold_pairs,
        "layout": layout,
        "num_queries": M,
        "num_docs": N,
        "num_relevant": k,
        "source": "qdmatch",
    }


def run(src_path, split_label, count, args, rng, src_tag):
    src = load_jsonl(src_path)
    units = [u for u in (_query_unit(ex, args.max_gold_per_query) for ex in src)
             if u is not None]
    M, N, k = args.num_queries, args.num_docs, args.num_relevant
    if k > M:
        raise ValueError(f"--num-relevant {k} cannot exceed M={M} queries")
    # Upper bound on distinct units consumed: k + (M-k) + (N-G) with G>=k, so
    # at most M+N-k. Sample that many distinct units; build_example uses a prefix.
    need = M + N - k
    if len(units) < need:
        raise RuntimeError(
            f"source has {len(units)} usable units < M+N-k={need} "
            f"(M={M}, N={N}, k={k}); use a larger source or smaller M/N")

    examples = []
    for _ in range(count):
        units_i = rng.sample(units, need)
        examples.append(build_example(units_i, M, N, k, args.layout, rng))
        examples[-1]["source"] = f"qdmatch_{src_tag}"

    npairs = len(examples[0]["gold_pairs"]) if examples else 0
    tag = f"{src_tag}_q{M}_n{N}_k{k}_{args.layout}"
    path = f"{args.output_dir}/qdmatch_{split_label}_{tag}.jsonl"
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"{split_label}: {len(units)} usable units -> {len(examples)} examples, "
          f"M={M} queries x N={N} docs, k={k} relevant queries ({npairs} gold pairs), "
          f"layout={args.layout}  ->  {path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-train", default="")
    ap.add_argument("--from-eval", default="")
    ap.add_argument("--num-docs", type=int, default=20,
                    help="N: number of documents per example")
    ap.add_argument("--num-queries", type=int, default=-1,
                    help="M: number of queries per example (default: = N)")
    ap.add_argument("--num-relevant", type=int, default=3,
                    help="k: number of relevant QUERIES (each maps to all its "
                         "gold docs; pairs = k for NQ, 2k for HotpotQA bridge)")
    ap.add_argument("--layout", choices=["separate", "shuffled"], default="separate",
                    help="separate (default): queries block then docs block; "
                         "shuffled: all items interleaved (ablation)")
    ap.add_argument("--num-train", type=int, default=2000)
    ap.add_argument("--num-eval", type=int, default=300)
    ap.add_argument("--max-gold-per-query", type=int, default=0,
                    help="cap gold docs kept per relevant query (0 = keep all; "
                         "HotpotQA bridge has 2, NQ has 1)")
    ap.add_argument("--src-tag", default="",
                    help="short source name (nq/hpqa/obliq); inferred if omitted")
    ap.add_argument("--output-dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.num_queries < 0:
        args.num_queries = args.num_docs   # M defaults to N

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
