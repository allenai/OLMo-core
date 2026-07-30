"""Train dense retrieval baselines: single-vector (DPR) and multi-vector (ColBERT).

Trains on triplet data (query, positive, negative) produced by
convert_to_retrieval_triplets.py. Uses sentence-transformers for DPR and
pylate for ColBERT.

These baselines are compared against the long-context LLM retrieval results
from evaluate_retrieval.py.

Requirements (install in corpus-reasoning-retrieval env):
    pip install sentence-transformers pylate torch

Usage:
    # Single-vector DPR baseline (single GPU)
    python scripts/train_retrieval_baseline.py --mode dpr \
        --train-data data/hotpotqa_train_k20_retrieval_triplets_52k.jsonl

    # Multi-vector ColBERT baseline (single GPU)
    python scripts/train_retrieval_baseline.py --mode colbert \
        --train-data data/hotpotqa_train_k20_retrieval_triplets_52k.jsonl

    # Multi-GPU via torchrun (required for ModernBERT with torch.compile)
    torchrun --nproc_per_node=4 scripts/train_retrieval_baseline.py --mode dpr \
        --train-data data/hotpotqa_train_k20_retrieval_triplets_52k.jsonl

    torchrun --nproc_per_node=4 scripts/train_retrieval_baseline.py --mode colbert \
        --train-data data/hotpotqa_train_k20_retrieval_triplets_52k.jsonl
"""

import argparse

from datasets import Dataset

from corpus_reasoning.lib.io import load_jsonl


def load_triplet_data(path, max_samples=None):
    """Load triplet JSONL and return a HuggingFace Dataset."""
    raw = load_jsonl(path)
    if max_samples and len(raw) > max_samples:
        raw = raw[:max_samples]
    return Dataset.from_list(raw)


def train_dpr(args):
    """Train a single-vector (DPR-style) retrieval model."""
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
    from sentence_transformers.evaluation import TripletEvaluator

    model = SentenceTransformer(args.model_name)

    train_dataset = load_triplet_data(args.train_data, args.max_train_samples)
    if args.eval_data:
        eval_dataset = load_triplet_data(args.eval_data, args.max_eval_samples)
    else:
        split = train_dataset.train_test_split(test_size=min(1000, len(train_dataset) // 10), seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]

    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    loss = CachedMultipleNegativesRankingLoss(
        model, mini_batch_size=args.mini_batch_size, scale=1.0 / args.temperature
    )

    model_shortname = args.model_name.split("/")[-1]
    run_name = f"{model_shortname}-dpr-{args.dataset_tag}-lr{args.lr}"

    training_args = SentenceTransformerTrainingArguments(
        output_dir=f"outputs/{run_name}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_ratio=0.05,
        fp16=False,
        bf16=True,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        save_strategy="no",
        logging_steps=1,
        run_name=run_name,
        dataloader_pin_memory=True,
        dataloader_drop_last=True,
    )

    evaluator = TripletEvaluator(
        anchors=eval_dataset["query"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
        name="retrieval-dev",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()

    save_path = f"outputs/{run_name}/final"
    model.save_pretrained(save_path)
    print(f"Model saved to {save_path}")

    evaluator(model)
    return save_path


def train_colbert(args):
    """Train a multi-vector (ColBERT) retrieval model."""
    from pylate import models, losses, evaluation, utils
    import torch

    from sentence_transformers import (
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )

    model = models.ColBERT(
        model_name_or_path=args.model_name,
        model_kwargs={"dtype": torch.bfloat16},
        embedding_size=args.embedding_dim,
        document_length=args.doc_length,
        query_length=args.query_length,
        skiplist_words=[],
    )

    train_dataset = load_triplet_data(args.train_data, args.max_train_samples)
    if args.eval_data:
        eval_dataset = load_triplet_data(args.eval_data, args.max_eval_samples)
    else:
        split = train_dataset.train_test_split(test_size=min(1000, len(train_dataset) // 10), seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]

    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    loss = losses.CachedContrastive(
        model, mini_batch_size=args.mini_batch_size, temperature=args.temperature
    )

    model_shortname = args.model_name.split("/")[-1]
    run_name = f"{model_shortname}-colbert-{args.dataset_tag}-lr{args.lr}-emb{args.embedding_dim}"

    training_args = SentenceTransformerTrainingArguments(
        output_dir=f"outputs/{run_name}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_ratio=0.05,
        fp16=False,
        bf16=True,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        save_strategy="no",
        logging_steps=1,
        run_name=run_name,
        dataloader_pin_memory=True,
        dataloader_drop_last=True,
        ddp_find_unused_parameters=False,
    )

    evaluator = evaluation.ColBERTTripletEvaluator(
        anchors=eval_dataset["query"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
        name="retrieval-dev",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
        data_collator=utils.ColBERTCollator(tokenize_fn=model.tokenize),
    )
    trainer.train()

    save_path = f"outputs/{run_name}/final"
    model.save_pretrained(save_path)
    print(f"Model saved to {save_path}")

    evaluator(model)
    return save_path


def main():
    parser = argparse.ArgumentParser(description="Train dense retrieval baselines")
    parser.add_argument("--mode", type=str, required=True, choices=["dpr", "colbert"],
                        help="Model type: 'dpr' for single-vector, 'colbert' for multi-vector")
    parser.add_argument("--train-data", type=str, required=True,
                        help="Triplet JSONL training data (from convert_to_retrieval_triplets.py)")
    parser.add_argument("--eval-data", type=str, default=None,
                        help="Triplet JSONL eval data (auto-split from train if not provided)")
    parser.add_argument("--model-name", type=str, default="answerdotai/ModernBERT-base",
                        help="Base model name or path")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Per-device batch size")
    parser.add_argument("--mini-batch-size", type=int, default=16,
                        help="Mini-batch size for cached contrastive loss")
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--lr-scheduler-type", type=str, default="linear")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--dataset-tag", type=str, default="hotpotqa",
                        help="Tag for run name (e.g. hotpotqa, nq)")

    # ColBERT-specific args
    parser.add_argument("--embedding-dim", type=int, default=128,
                        help="ColBERT embedding dimension")
    parser.add_argument("--doc-length", type=int, default=512,
                        help="ColBERT max document token length")
    parser.add_argument("--query-length", type=int, default=64,
                        help="ColBERT max query token length")

    args = parser.parse_args()

    print(f"Training {args.mode} model: {args.model_name}")
    print(f"  Data: {args.train_data}")
    print(f"  LR: {args.lr}, Epochs: {args.epochs}, Batch: {args.batch_size}")

    if args.mode == "dpr":
        train_dpr(args)
    else:
        train_colbert(args)


if __name__ == "__main__":
    main()
