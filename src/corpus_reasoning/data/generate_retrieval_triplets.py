"""Generate triplet training data for dense retrieval model baselines.

Produces (query, positive_doc, negative_doc) triplets from HotpotQA and NQ,
using the same source datasets and document formatting as the long-context
training data generators (generate_hotpotqa_data.py, generate_nq_training_data.py).

For HotpotQA (2 gold docs per question), each question produces 2 triplets
(one per gold doc), with negatives sampled from the distractor pool.

Output format: JSONL with fields {query, positive, negative} — compatible with
sentence-transformers and pylate training pipelines.

Usage:
    # HotpotQA triplets (bridge questions, k=20 distractor pool)
    python scripts/generate_retrieval_triplets.py --dataset hotpotqa --num-examples 5000 --num-docs 20

    # NQ triplets
    python scripts/generate_retrieval_triplets.py --dataset nq --num-examples 5000 --num-docs 20

    # HotpotQA with multiple hard negatives per positive
    python scripts/generate_retrieval_triplets.py --dataset hotpotqa --num-examples 5000 --negatives-per-positive 3
"""

import argparse
import random
from datasets import load_dataset
from corpus_reasoning.lib.io import save_jsonl, print_dataset_stats


# --- HotpotQA helpers (mirrored from generate_hotpotqa_data.py) ---

def hotpotqa_paragraphs(context):
    """Convert HotpotQA context to list of {title, text} dicts."""
    titles = context["title"]
    sentences_list = context["sentences"]
    docs = []
    for title, sentences in zip(titles, sentences_list):
        text = " ".join(s.strip() for s in sentences)
        docs.append({"title": title, "text": text})
    return docs


def get_supporting_titles(example):
    return set(example["supporting_facts"]["title"])


def format_doc_text(doc, use_titles=True):
    """Format a document as plain text (title: text) for retrieval model input."""
    if use_titles and doc.get("title"):
        return f"{doc['title']}: {doc['text']}"
    return doc["text"]


# --- NQ helpers (mirrored from generate_nq_training_data.py) ---

import re

def make_title(context: str) -> str:
    for sep in [" - Wikipedia", " -- Wikipedia", "\n", ". "]:
        if sep in context[:200]:
            candidate = context[:context.index(sep)].strip()
            if 3 < len(candidate) < 100:
                return candidate
    snippet = context[:80]
    if " " in snippet[40:]:
        snippet = snippet[:snippet.rindex(" ", 0, 80)]
    return snippet.strip().rstrip(".,;:-")


def split_into_chunks(text, target_len=450):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    for sent in sentences:
        if current and len(current) + len(sent) + 1 > target_len:
            chunks.append(current.strip())
            current = sent
        else:
            current = current + " " + sent if current else sent
    if current.strip():
        chunks.append(current.strip())
    return chunks


# --- Triplet generation ---

def generate_hotpotqa_triplets(args):
    """Generate triplets from HotpotQA dataset."""
    rng = random.Random(args.seed)

    print(f"Loading hotpotqa distractor ({args.split})...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=args.split)
    print(f"  Loaded {len(ds)} examples")

    if args.question_type != "all":
        ds = ds.filter(lambda ex: ex["type"] == args.question_type)
        print(f"  After filtering to type={args.question_type}: {len(ds)} examples")

    indices = list(range(len(ds)))
    rng.shuffle(indices)
    n = min(args.num_examples, len(ds))
    selected = indices[:n]

    # Build external distractor pool
    pool_indices = indices[n:n + n * 3] if n < len(ds) else indices
    distractor_pool = []
    for idx in pool_indices:
        ex = ds[idx]
        sup_titles = get_supporting_titles(ex)
        for doc in hotpotqa_paragraphs(ex["context"]):
            if doc["title"] not in sup_titles:
                distractor_pool.append(doc)
    rng.shuffle(distractor_pool)
    print(f"  Distractor pool: {len(distractor_pool)} paragraphs")

    triplets = []
    for i in selected:
        ex = ds[i]
        all_docs = hotpotqa_paragraphs(ex["context"])
        sup_titles = get_supporting_titles(ex)

        supporting = [d for d in all_docs if d["title"] in sup_titles]
        local_distractors = [d for d in all_docs if d["title"] not in sup_titles]

        query = ex["question"]

        for gold_doc in supporting:
            # Sample negatives: prefer local distractors, fall back to pool
            neg_candidates = list(local_distractors)
            if len(neg_candidates) < args.negatives_per_positive:
                extra = rng.sample(distractor_pool,
                                   min(args.negatives_per_positive - len(neg_candidates),
                                       len(distractor_pool)))
                neg_candidates.extend(extra)
            rng.shuffle(neg_candidates)

            for neg_doc in neg_candidates[:args.negatives_per_positive]:
                triplets.append({
                    "query": query,
                    "positive": format_doc_text(gold_doc, use_titles=not args.no_titles),
                    "negative": format_doc_text(neg_doc, use_titles=not args.no_titles),
                })

    rng.shuffle(triplets)
    return triplets


def generate_nq_triplets(args):
    """Generate triplets from NQ dataset."""
    rng = random.Random(args.seed)

    print(f"Loading NQ dataset ({args.split})...")
    ds = load_dataset("tilyupo/nq_cqa", split=args.split)
    print(f"  Loaded {len(ds)} examples")

    indices = list(range(len(ds)))
    rng.shuffle(indices)
    n = min(args.num_examples, len(ds))
    selected = indices[:n]

    # Build distractor pool from non-selected examples
    pool_indices = indices[n:n + n * 3] if n < len(ds) else indices
    distractor_pool = [ds[idx] for idx in pool_indices]

    if args.split_docs:
        # Pre-compute distractor chunks
        dist_chunks = []
        for d in distractor_pool:
            chunks = split_into_chunks(d["context"])
            title = make_title(d["context"]) if not args.no_titles else None
            for chunk in chunks:
                dist_chunks.append({"title": title, "text": chunk})
        rng.shuffle(dist_chunks)
        print(f"  Distractor chunk pool: {len(dist_chunks)} chunks")
    else:
        dist_docs = []
        for d in distractor_pool:
            title = make_title(d["context"]) if not args.no_titles else None
            dist_docs.append({"title": title, "text": d["context"]})
        rng.shuffle(dist_docs)
        print(f"  Distractor pool: {len(dist_docs)} documents")

    triplets = []
    for i in selected:
        sample = ds[i]
        query = sample["question"]

        # Build gold doc
        gold_text = sample["context"]
        gold_title = make_title(gold_text) if not args.no_titles else None

        if args.split_docs:
            gold_chunks = split_into_chunks(gold_text)
            answer = sample["answer"].lower()
            gold_chunk = gold_chunks[0]
            for chunk in gold_chunks:
                if answer in chunk.lower():
                    gold_chunk = chunk
                    break
            gold_doc = {"title": gold_title, "text": gold_chunk}
            neg_pool = dist_chunks
        else:
            gold_doc = {"title": gold_title, "text": gold_text}
            neg_pool = dist_docs

        neg_candidates = rng.sample(neg_pool, min(args.negatives_per_positive, len(neg_pool)))
        for neg_doc in neg_candidates:
            triplets.append({
                "query": query,
                "positive": format_doc_text(gold_doc, use_titles=not args.no_titles),
                "negative": format_doc_text(neg_doc, use_titles=not args.no_titles),
            })

    rng.shuffle(triplets)
    return triplets


def main():
    parser = argparse.ArgumentParser(description="Generate retrieval triplet training data")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["hotpotqa", "nq"],
                        help="Source dataset")
    parser.add_argument("--num-examples", type=int, default=5000,
                        help="Number of source examples to sample (triplets may be more)")
    parser.add_argument("--negatives-per-positive", type=int, default=1,
                        help="Number of hard negatives per positive document")
    parser.add_argument("--question-type", type=str, default="bridge",
                        choices=["bridge", "comparison", "all"],
                        help="HotpotQA question type filter")
    parser.add_argument("--num-docs", type=int, default=20,
                        help="(Unused for triplets, kept for consistency with other scripts)")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "validation"])
    parser.add_argument("--no-titles", action="store_true",
                        help="Omit document titles")
    parser.add_argument("--split-docs", action="store_true",
                        help="(NQ only) Split documents into ~450-char chunks")
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.dataset == "hotpotqa":
        triplets = generate_hotpotqa_triplets(args)
    else:
        triplets = generate_nq_triplets(args)

    # Build filename
    label = "train" if args.split == "train" else "eval"
    flags = [args.dataset, label, "triplets"]
    if args.dataset == "hotpotqa" and args.question_type != "all":
        flags.append(args.question_type)
    if args.negatives_per_positive > 1:
        flags.append(f"neg{args.negatives_per_positive}")
    if args.split_docs:
        flags.append("split")
    tag = "_".join(flags)
    path = f"{args.output_dir}/{tag}_{len(triplets)}.jsonl"

    save_jsonl(path, triplets)
    print_dataset_stats(triplets, f"{args.dataset} triplets", path)

    # Show samples
    if triplets:
        print("\n=== Samples ===")
        for t in triplets[:3]:
            print(f"  Q: {t['query'][:80]}")
            print(f"  +: {t['positive'][:80]}...")
            print(f"  -: {t['negative'][:80]}...")
            print()


if __name__ == "__main__":
    main()
