"""Train a small AutoModelForSequenceClassification on a pair JSONL.

Task-agnostic: consumes the unified pair JSONL format produced by any
scripts/data/prepare_*_pair_classifier.py:
    {"text_a": str, "text_b": str, "label": 0|1, "_meta": {...}}

Default backbone: microsoft/deberta-v3-small (~44M params, well under 100M).
Tokenization treats text_a as the anchor (query / first sentence) and text_b
as the candidate (doc / second sentence); when sequences are too long, only
text_b is truncated.

Usage:
    python scripts/train/train_pair_classifier.py \
        --train-file data/contradiction_pair_train.jsonl \
        --eval-file  data/contradiction_pair_eval.jsonl \
        --output-dir checkpoints/contradiction_pair_clf \
        --max-length 256 --num-epochs 3
"""

import argparse
import json
import os

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from corpus_reasoning.lib.io import load_jsonl


def load_pair_dataset(path: str) -> Dataset:
    rows = load_jsonl(path)
    cleaned = [
        {"text_a": r["text_a"], "text_b": r["text_b"], "label": int(r["label"])}
        for r in rows
    ]
    return Dataset.from_list(cleaned)


def make_tokenize_fn(tokenizer, max_length: int):
    def fn(batch):
        # longest_first: truncate whichever sequence is longer. only_second
        # fails when text_a alone already exceeds max_length (long PubMed
        # sentences); longest_first handles that case gracefully.
        return tokenizer(
            batch["text_a"],
            batch["text_b"],
            truncation="longest_first",
            max_length=max_length,
        )
    return fn


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()[:, 1]
    preds = (probs >= 0.5).astype(int)
    out = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }
    try:
        out["auc"] = roc_auc_score(labels, probs)
    except ValueError:
        out["auc"] = float("nan")
    return out


class WeightedTrainer(Trainer):
    """Trainer with optional class-balanced cross-entropy."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self._class_weights is not None:
            w = self._class_weights.to(device=logits.device, dtype=logits.dtype)
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=w)
        else:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss


def derive_class_weights(train_ds: Dataset) -> torch.Tensor:
    labels = np.array(train_ds["label"])
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    total = n_pos + n_neg
    # inverse-frequency, normalized to mean 1
    w_neg = total / (2.0 * max(n_neg, 1))
    w_pos = total / (2.0 * max(n_pos, 1))
    return torch.tensor([w_neg, w_pos], dtype=torch.float)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-file", required=True)
    p.add_argument("--eval-file", required=True)
    p.add_argument("--output-dir", required=True)
    # ELECTRA-small (~14M): stable training, well under 100M. DeBERTa-v3 was
    # tried first but NaNs out (gradient instability ~step 300, bf16 and fp32).
    p.add_argument("--model-name", default="google/electra-small-discriminator")
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--num-epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--class-weight", choices=["none", "balanced"], default="balanced")
    p.add_argument("--early-stopping-patience", type=int, default=2)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2,
    )

    train_ds = load_pair_dataset(args.train_file)
    eval_ds = load_pair_dataset(args.eval_file)

    tok_fn = make_tokenize_fn(tokenizer, args.max_length)
    train_ds = train_ds.map(tok_fn, batched=True, remove_columns=["text_a", "text_b"])
    eval_ds = eval_ds.map(tok_fn, batched=True, remove_columns=["text_a", "text_b"])

    class_weights = derive_class_weights(train_ds) if args.class_weight == "balanced" else None
    if class_weights is not None:
        print(f"Class weights (neg, pos): {class_weights.tolist()}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to=[],
        seed=args.seed,
        bf16=args.bf16,
        fp16=args.fp16,
    )

    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
        ))

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=callbacks,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(json.dumps({"final_eval": metrics}, indent=2))

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "final_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
