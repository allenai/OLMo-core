"""Convert unified-format retrieval JSONL into triplets for dense retrieval.

Takes the same training data used by the long-context models (e.g.,
hotpotqa_train_k20_shuffled_bridge_5000.jsonl) and extracts
(query, positive_doc, negative_doc) triplets, ensuring the retrieval model
sees the exact same query-document pairs.

For each example with Q gold docs and D-Q distractors, generates Q*(D-Q) triplets
(one per gold × distractor combination), so the model sees every positive-negative
pair from the original context.

Use --max-triplets or adjust epochs to control total training tokens.

Usage:
    python scripts/data/convert_to_retrieval_triplets.py \
        data/hotpotqa_train_k20_shuffled_bridge_5000.jsonl \
        data/hotpotqa_train_k20_retrieval_triplets.jsonl

    python scripts/data/convert_to_retrieval_triplets.py \
        data/hotpotqa_train_k20_shuffled_bridge_5000.jsonl \
        data/hotpotqa_train_k20_retrieval_triplets_50k.jsonl \
        --max-triplets 50000
"""

import argparse
import random
from corpus_reasoning.lib.io import load_jsonl, save_jsonl


def doc_to_text(doc):
    """Format a document dict as 'Title: text' for the retrieval model."""
    title = doc.get("title")
    if title:
        return f"{title}: {doc['text']}"
    return doc["text"]


def convert_to_triplets(examples, rng, max_triplets=None):
    """Convert unified-format examples to exhaustive (query, pos, neg) triplets.

    Each example has documents, queries, and gold_doc_indices.
    For single-query examples, generates one triplet per (gold_doc, distractor) pair.
    For multi-query examples, generates triplets per query.
    """
    triplets = []

    for ex in examples:
        documents = ex["documents"]
        queries = ex["queries"]
        gold_indices = ex["gold_doc_indices"]
        
        # Normalize gold_indices to per-query lists
        if queries and gold_indices:
            if isinstance(gold_indices[0], list):
                # Multi-query: gold_indices is already per-query
                per_query_gold = gold_indices
            else:
                # Single-query: one flat list of gold indices
                per_query_gold = [gold_indices]

        for qi, query in enumerate(queries):
            if qi >= len(per_query_gold):
                continue
            gold_set = set(per_query_gold[qi])
            gold_docs = [(i, doc_to_text(documents[i])) for i in sorted(gold_set)
                         if i < len(documents)]
            distractor_docs = [(i, doc_to_text(documents[i]))
                               for i in range(len(documents)) if i not in gold_set]

            for _gid, gold_text in gold_docs:
                for _did, dist_text in distractor_docs:
                    triplets.append({
                        "query": query,
                        "positive": gold_text,
                        "negative": dist_text,
                    })

    rng.shuffle(triplets)
    if max_triplets and len(triplets) > max_triplets:
        triplets = triplets[:max_triplets]

    return triplets


def main():
    parser = argparse.ArgumentParser(
        description="Convert unified-format JSONL to dense retrieval triplets")
    parser.add_argument("input_file", type=str,
                        help="Input unified-format JSONL")
    parser.add_argument("output_file", type=str,
                        help="Output triplet JSONL")
    parser.add_argument("--max-triplets", type=int, default=None,
                        help="Maximum number of triplets to output (subsample if more)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print(f"Loading {args.input_file}...")
    examples = load_jsonl(args.input_file)
    print(f"  {len(examples)} examples")

    triplets = convert_to_triplets(examples, rng, args.max_triplets)

    # Stats
    avg_query_chars = sum(len(t["query"]) for t in triplets) / len(triplets)
    avg_pos_chars = sum(len(t["positive"]) for t in triplets) / len(triplets)
    avg_neg_chars = sum(len(t["negative"]) for t in triplets) / len(triplets)
    avg_total_chars = avg_query_chars + avg_pos_chars + avg_neg_chars
    approx_tokens = sum(len(t["query"]) + len(t["positive"]) + len(t["negative"])
                        for t in triplets) / 4

    print(f"\n  Generated {len(triplets)} triplets")
    print(f"  Avg chars: query={avg_query_chars:.0f}, pos={avg_pos_chars:.0f}, neg={avg_neg_chars:.0f}, total={avg_total_chars:.0f}")
    print(f"  Approx total tokens: {approx_tokens/1e6:.1f}M")

    save_jsonl(args.output_file, triplets)
    print(f"\n  Saved to {args.output_file}")

    # Show samples
    print("\n=== Samples ===")
    for t in triplets[:3]:
        print(f"  Q: {t['query'][:80]}")
        print(f"  +: {t['positive'][:80]}...")
        print(f"  -: {t['negative'][:80]}...")
        print()


if __name__ == "__main__":
    main()
