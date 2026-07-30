"""Evaluate dense retrieval baselines on the same eval data as LLM retrieval.

Parses the pre-generated retrieval JSONL (same files used by evaluate_retrieval.py),
extracts individual documents and questions, encodes them with the retrieval model,
ranks documents by similarity, and computes the same metrics (EM, recall, precision, F1).

Supports both single-vector (DPR/SentenceTransformer) and multi-vector (ColBERT) models.

Usage:
    # Evaluate DPR model
    python scripts/evaluate_retrieval_baseline.py --mode dpr \
        --model-path outputs/ModernBERT-base-dpr-hotpotqa-lr3e-5/final \
        --eval-data data/hotpotqa_eval_k20_shuffled_retrieval_bridge_500.jsonl

    # Evaluate ColBERT model
    python scripts/evaluate_retrieval_baseline.py --mode colbert \
        --model-path outputs/ModernBERT-base-colbert-hotpotqa-lr3e-5-emb128/final \
        --eval-data data/hotpotqa_eval_k20_shuffled_retrieval_bridge_500.jsonl

    # Evaluate base (untrained) model
    python scripts/evaluate_retrieval_baseline.py --mode dpr \
        --model-path answerdotai/ModernBERT-base \
        --eval-data data/hotpotqa_eval_k20_shuffled_retrieval_bridge_500.jsonl

    # NQ eval (auto-detects single gold doc)
    python scripts/evaluate_retrieval_baseline.py --mode dpr \
        --model-path outputs/ModernBERT-base-dpr-nq/final \
        --eval-data data/nq_train_k20_random_500_retrieval.jsonl
"""

import argparse
import json
import re
import random
import numpy as np
from pathlib import Path

from corpus_reasoning.lib.io import load_jsonl, save_results
from corpus_reasoning.lib.metrics import (
    parse_doc_ids, retrieval_exact_match, retrieval_recall,
    retrieval_precision, retrieval_f1, aggregate,
)


def parse_documents_from_input(input_text):
    """Extract individual documents from the eval JSONL input field.

    Parses text like:
        Document [1] (Title: Foo): Some text...

        Document [2] (Title: Bar): Other text...

        Question: What is ...

    Returns:
        docs: list of {"id": int, "title": str|None, "text": str, "full_text": str}
        question: str
    """
    # Split off the question(s) at the end
    # Handle single-query format
    question_match = re.search(r'\n\nQuestion:\s*(.+)$', input_text, re.DOTALL)
    # Handle multi-query format
    multi_q_match = re.search(r'\nQuestion 1:\s*(.+)$', input_text, re.DOTALL)

    if multi_q_match and (not question_match or multi_q_match.start() < question_match.start()):
        doc_text = input_text[:multi_q_match.start()]
        question = multi_q_match.group(0).strip()
    elif question_match:
        doc_text = input_text[:question_match.start()]
        question = question_match.group(1).strip()
    else:
        doc_text = input_text
        question = ""

    # Split on document boundaries using findall to locate each doc start
    # This is more robust than a single regex with lookahead
    doc_starts = [m.start() for m in re.finditer(r'Document\s+\[', doc_text)]
    doc_chunks = []
    for i, start in enumerate(doc_starts):
        end = doc_starts[i + 1] if i + 1 < len(doc_starts) else len(doc_text)
        doc_chunks.append(doc_text[start:end].strip())

    header_pattern = re.compile(
        r'Document\s+\[(\d+)\]\s*'
        r'(?:\(Title:\s*(.*?)\)\s*:\s*|\:\s*)'
        r'(.*)',
        re.DOTALL
    )

    docs = []
    for chunk in doc_chunks:
        match = header_pattern.match(chunk)
        if not match:
            continue
        doc_id = int(match.group(1))
        title = match.group(2).strip() if match.group(2) else None
        text = match.group(3).strip()
        if title:
            full_text = f"{title}: {text}"
        else:
            full_text = text
        docs.append({
            "id": doc_id,
            "title": title,
            "text": text,
            "full_text": full_text,
        })

    return docs, question


def parse_multi_questions(question_text):
    """Parse multiple questions from multi-query format.

    Input like: "Question 1: What...\nQuestion 2: How..."
    Returns list of question strings.
    """
    parts = re.split(r'Question\s+\d+:\s*', question_text)
    return [p.strip() for p in parts if p.strip()]


def detect_task(examples):
    """Auto-detect task type from data format (same logic as evaluate_retrieval.py)."""
    ex = examples[0]
    output = ex["output"]
    if "Q1:" in output or "Q2:" in output:
        return "multi-hotpotqa"
    ids = parse_doc_ids(output)
    return "nq" if len(ids) <= 1 else "hotpotqa"


def encode_dpr(model, texts, batch_size=64, is_query=False):
    """Encode texts with a SentenceTransformer model."""
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=len(texts) > 100,
        convert_to_numpy=True,
    )
    return embeddings


def score_dpr(query_emb, doc_embs):
    """Compute cosine similarity between query and each document."""
    # Normalize
    query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    norms = np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-8
    doc_embs_normed = doc_embs / norms
    return doc_embs_normed @ query_emb


def encode_colbert(model, texts, batch_size=64, is_query=False):
    """Encode texts with a ColBERT model. Returns list of 2D arrays."""
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=len(texts) > 100,
        is_query=is_query,
    )
    return embeddings


def score_colbert(query_embs, doc_embs_list):
    """Compute ColBERT MaxSim score between query and each document.

    query_embs: (num_query_tokens, dim)
    doc_embs_list: list of (num_doc_tokens, dim) arrays

    For each query token, find max similarity across all doc tokens.
    Sum these max similarities.
    """
    query_embs = np.array(query_embs, dtype=np.float32)
    scores = []
    for doc_embs in doc_embs_list:
        doc_embs = np.array(doc_embs, dtype=np.float32)
        # (num_query_tokens, num_doc_tokens)
        sim = query_embs @ doc_embs.T
        # MaxSim: for each query token, take max over doc tokens, then sum
        score = sim.max(axis=1).sum()
        scores.append(float(score))
    return np.array(scores)


def evaluate_single_query(query, docs, gold_ids, model, args, is_colbert=False):
    """Evaluate a single query against its document set.

    Returns dict with predicted_ids and metrics.
    """
    doc_texts = [d["full_text"] for d in docs]
    doc_ids = [d["id"] for d in docs]

    if is_colbert:
        query_embs = encode_colbert(model, [query], batch_size=1, is_query=True)
        doc_embs = encode_colbert(model, doc_texts, batch_size=args.batch_size, is_query=False)
        scores = score_colbert(query_embs[0], doc_embs)
    else:
        query_emb = encode_dpr(model, [query], batch_size=1, is_query=True)[0]
        doc_embs = encode_dpr(model, doc_texts, batch_size=args.batch_size, is_query=False)
        scores = score_dpr(query_emb, doc_embs)

    # Select top-k documents (k = number of gold docs)
    k = max(len(gold_ids), 1)
    top_indices = np.argsort(scores)[::-1][:k]
    predicted_ids = set(doc_ids[i] for i in top_indices)

    metrics = {
        "exact_match": float(retrieval_exact_match(predicted_ids, gold_ids)),
        "recall": retrieval_recall(predicted_ids, gold_ids),
        "precision": retrieval_precision(predicted_ids, gold_ids),
        "f1": retrieval_f1(predicted_ids, gold_ids),
    }
    return predicted_ids, metrics, scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate dense retrieval baselines")
    parser.add_argument("--mode", type=str, required=True, choices=["dpr", "colbert"],
                        help="Model type")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained model or HuggingFace model name")
    parser.add_argument("--eval-data", type=str, required=True,
                        help="Pre-generated retrieval JSONL (same as evaluate_retrieval.py)")
    parser.add_argument("--task", type=str, default="auto",
                        choices=["auto", "nq", "hotpotqa", "multi-hotpotqa"])
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Encoding batch size")
    parser.add_argument("--output-file", type=str, default="outputs/retrieval_baseline_results.json")
    args = parser.parse_args()

    # Load eval data
    print(f"Loading eval data from {args.eval_data}")
    raw = load_jsonl(args.eval_data)
    if args.max_test_samples and len(raw) > args.max_test_samples:
        random.seed(42)
        raw = random.sample(raw, args.max_test_samples)
    print(f"  {len(raw)} examples")

    # Detect task
    task = args.task if args.task != "auto" else detect_task(raw)
    print(f"  Task: {task}")

    # Load model
    is_colbert = args.mode == "colbert"
    print(f"Loading {args.mode} model from {args.model_path}...")
    if is_colbert:
        from pylate import models
        import torch
        model = models.ColBERT(model_name_or_path=args.model_path,
                               model_kwargs={"dtype": torch.bfloat16})
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(args.model_path)
    print("  Model loaded")

    # Evaluate
    if task == "multi-hotpotqa":
        all_per_query = []
        all_agg = []
        details = []

        for i, ex in enumerate(raw):
            docs, question_block = parse_documents_from_input(ex["input"])
            questions = parse_multi_questions(question_block)

            # Parse per-query gold IDs
            gold_output = ex["output"]
            per_query_gold = []
            for part in gold_output.split(";"):
                part = re.sub(r'^Q\d+:\s*', '', part.strip())
                per_query_gold.append(parse_doc_ids(part))
            num_q = len(questions)

            while len(per_query_gold) < num_q:
                per_query_gold.append(set())
            per_query_gold = per_query_gold[:num_q]

            per_query_metrics = []
            for q_idx, (question, gold_ids) in enumerate(zip(questions, per_query_gold)):
                _, metrics, _ = evaluate_single_query(
                    question, docs, gold_ids, model, args, is_colbert
                )
                per_query_metrics.append(metrics)

            all_per_query.append(per_query_metrics)
            n = len(per_query_metrics)
            agg = {
                "exact_match": sum(m["exact_match"] for m in per_query_metrics) / n,
                "recall": sum(m["recall"] for m in per_query_metrics) / n,
                "precision": sum(m["precision"] for m in per_query_metrics) / n,
                "f1": sum(m["f1"] for m in per_query_metrics) / n,
                "all_correct": float(all(m["exact_match"] == 1.0 for m in per_query_metrics)),
            }
            all_agg.append(agg)
            details.append({
                "gold_output": ex["output"],
                **agg,
            })

            if (i + 1) % 20 == 0:
                running_em = sum(a["exact_match"] for a in all_agg) / len(all_agg)
                print(f"  [{i+1}/{len(raw)}] running EM: {running_em:.1%}")

        n_total = len(all_agg)
        overall = {k: sum(a[k] for a in all_agg) / n_total
                   for k in ["exact_match", "recall", "precision", "f1", "all_correct"]}

        num_q = len(all_per_query[0]) if all_per_query else 0
        per_position = []
        for pos in range(num_q):
            pos_metrics = {}
            for k in ["exact_match", "recall", "precision", "f1"]:
                pos_metrics[k] = sum(pq[pos][k] for pq in all_per_query) / n_total
            per_position.append(pos_metrics)

        results = {
            "args": vars(args),
            "task": task,
            "overall": overall,
            "per_position": per_position,
            "details": details,
        }

        print(f"\n{'='*60}")
        print(f"RESULTS ({len(raw)} examples, {num_q} queries each)")
        print(f"{'='*60}")
        print(f"  Overall EM:        {overall['exact_match']:.1%}")
        print(f"  Overall Recall:    {overall['recall']:.1%}")
        print(f"  Overall Precision: {overall['precision']:.1%}")
        print(f"  Overall F1:        {overall['f1']:.1%}")
        print(f"  All-correct:       {overall['all_correct']:.1%}")

        print(f"\nPer-position:")
        for pos, m in enumerate(per_position):
            print(f"  Q{pos+1}: EM={m['exact_match']:.1%}  R={m['recall']:.1%}  "
                  f"P={m['precision']:.1%}  F1={m['f1']:.1%}")

    else:
        # NQ or HotpotQA (single query)
        metric_keys = ["exact_match", "recall", "precision", "f1"]
        results_list = []
        details = []

        for i, ex in enumerate(raw):
            docs, question = parse_documents_from_input(ex["input"])
            gold_ids = parse_doc_ids(ex["output"])

            predicted_ids, metrics, scores = evaluate_single_query(
                question, docs, gold_ids, model, args, is_colbert
            )

            results_list.append(metrics)
            details.append({
                "gold_ids": sorted(gold_ids),
                "predicted_ids": sorted(predicted_ids),
                **metrics,
            })

            if (i + 1) % 100 == 0:
                running = aggregate(results_list, metric_keys)
                print(f"  [{i+1}/{len(raw)}] running EM: {running['exact_match']:.1%}")

        overall = aggregate(results_list, metric_keys)
        results = {
            "args": vars(args),
            "task": task,
            "overall": overall,
            "details": details,
        }

        print(f"\n{'='*60}")
        print(f"RESULTS ({len(raw)} examples, task={task})")
        print(f"{'='*60}")
        print(f"  EM:        {overall['exact_match']:.1%}")
        print(f"  Recall:    {overall['recall']:.1%}")
        print(f"  Precision: {overall['precision']:.1%}")
        print(f"  F1:        {overall['f1']:.1%}")

    # Show samples
    print(f"\n--- Samples ---")
    for d in details[:5]:
        print(f"  {d}")
        print()

    save_results(args.output_file, results)
    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
