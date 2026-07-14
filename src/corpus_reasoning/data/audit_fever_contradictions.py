"""Post-hoc NLI audit for FEVER contradiction examples — find false negatives.

`generate_fever_contradiction_data.py` constructs (REFUTES claim, evidence
sentence) pairs as guaranteed contradictions, but the per-example distractors
can occasionally also contradict a gold REFUTES claim. NEI claims are safer
than SUPPORTS evidence sentences (which is why the generator uses NEI-only
gold clusters), but for very broad REFUTES claims like "X does not have an
acting career", any topical NEI claim mentioning X's acting still leaks.

This script runs an NLI model over (other_doc, gold_REFUTES_claim) pairs in
each example and flags any prediction with contradiction probability above
threshold. Two output modes:

  --mode audit    write a side file listing every suspect (default).
  --mode relabel  write a new JSONL with extended `gold_doc_indices` so each
                  high-confidence suspect becomes an additional gold pair
                  (paired with the original gold REFUTES claim).

Both modes also print summary stats: how many examples have suspects, mean
suspects per example, distribution of contradiction probabilities.

Usage:
    python scripts/data/audit_fever_contradictions.py \\
        --input data/contradiction_train_fever_n100_k3.jsonl \\
        --mode audit
    python scripts/data/audit_fever_contradictions.py \\
        --input data/contradiction_train_fever_n100_k3.jsonl \\
        --mode relabel --threshold 0.95
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from corpus_reasoning.lib.io import load_jsonl, save_jsonl


def _refutes_claim_index(pair, docs):
    """Return the index (within `pair`) of the REFUTES claim half.

    The pair is (REFUTES claim, evidence sentence). The REFUTES claim is the
    short FEVER claim; the evidence sentence is a longer Wikipedia sentence.
    Length is a reliable proxy — Wikipedia sentences are almost always
    longer than the FEVER claims about them.
    """
    a, b = pair
    return a if len(docs[a - 1]["text"]) <= len(docs[b - 1]["text"]) else b


def _build_tasks(examples):
    """Yield (example_idx, gold_pair_idx, suspect_doc_idx, premise, hypothesis).

    For each gold pair, identify the REFUTES claim and emit one task per
    other document in the example with premise=other_doc, hypothesis=REFUTES
    claim. The NLI model predicts whether premise contradicts hypothesis —
    a high probability means the other doc states something the REFUTES claim
    contradicts, i.e. an unlabelled contradiction pair.
    """
    for ex_idx, ex in enumerate(examples):
        docs = ex["documents"]
        gold_pairs = ex["gold_doc_indices"]
        gold_set = {i for pair in gold_pairs for i in pair}
        for pi, pair in enumerate(gold_pairs):
            claim_idx = _refutes_claim_index(pair, docs)
            claim_text = docs[claim_idx - 1]["text"]
            for d_idx in range(1, len(docs) + 1):
                if d_idx in gold_set:
                    continue
                yield (ex_idx, pi, d_idx, docs[d_idx - 1]["text"], claim_text)


def _run_nli_batched(tasks, model, tokenizer, contradiction_label_id, batch_size,
                     device):
    """Run NLI on a list of tasks; return list of contradiction probabilities."""
    probs = [None] * len(tasks)
    for start in tqdm(range(0, len(tasks), batch_size), desc="NLI"):
        batch = tasks[start:start + batch_size]
        premises = [t[3] for t in batch]
        hypotheses = [t[4] for t in batch]
        enc = tokenizer(
            premises, hypotheses,
            padding=True, truncation=True, max_length=256,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        batch_probs = torch.softmax(logits, dim=-1)[:, contradiction_label_id]
        for i, p in enumerate(batch_probs.tolist()):
            probs[start + i] = p
    return probs


def _summarize(suspects, examples):
    n_examples = len(examples)
    n_with_suspects = len({s["example_idx"] for s in suspects})
    per_ex_counts = [0] * n_examples
    for s in suspects:
        per_ex_counts[s["example_idx"]] += 1
    nonzero = [c for c in per_ex_counts if c > 0]
    print()
    print("=" * 60)
    print(f"Audit summary")
    print("=" * 60)
    print(f"  Examples:                       {n_examples}")
    print(f"  Examples with ≥1 suspect:       {n_with_suspects} "
          f"({n_with_suspects / n_examples * 100:.1f}%)")
    print(f"  Total suspect (gold, doc) pairs: {len(suspects)}")
    if nonzero:
        print(f"  Suspects per affected example:  "
              f"mean={statistics.mean(nonzero):.2f} "
              f"median={statistics.median(nonzero):.0f} "
              f"max={max(nonzero)}")
    if suspects:
        probs = sorted(s["contradiction_prob"] for s in suspects)
        print(f"  Contradiction prob distribution: "
              f"min={probs[0]:.2f} median={probs[len(probs) // 2]:.2f} "
              f"max={probs[-1]:.2f}")


def _emit_audit(suspects, examples, dst):
    rows = []
    for s in suspects:
        ex = examples[s["example_idx"]]
        docs = ex["documents"]
        pair = ex["gold_doc_indices"][s["gold_pair_idx"]]
        claim_idx = _refutes_claim_index(pair, docs)
        rows.append({
            "example_idx": s["example_idx"],
            "gold_pair": pair,
            "gold_refutes_claim": docs[claim_idx - 1]["text"],
            "suspect_doc_idx": s["suspect_doc_idx"],
            "suspect_text": docs[s["suspect_doc_idx"] - 1]["text"],
            "contradiction_prob": round(s["contradiction_prob"], 4),
        })
    save_jsonl(str(dst), rows)
    print(f"\nAudit written: {dst} ({len(rows)} suspect pairs)")


def _emit_relabeled(suspects, examples, dst):
    """Extend each example's `gold_doc_indices` with new pairs from suspects.

    Each suspect is paired with the REFUTES claim it contradicted. Multiple
    suspects flagged against the same gold pair each become their own new
    gold pair anchored on that same REFUTES claim.
    """
    by_example = {}
    for s in suspects:
        by_example.setdefault(s["example_idx"], []).append(s)

    out = []
    n_added = 0
    for ex_idx, ex in enumerate(examples):
        new_ex = dict(ex)
        gold_pairs = list(ex["gold_doc_indices"])
        if ex_idx in by_example:
            docs = ex["documents"]
            for s in by_example[ex_idx]:
                pair = ex["gold_doc_indices"][s["gold_pair_idx"]]
                claim_idx = _refutes_claim_index(pair, docs)
                new_pair = sorted([claim_idx, s["suspect_doc_idx"]])
                if new_pair not in gold_pairs:
                    gold_pairs.append(new_pair)
                    n_added += 1
        new_ex["gold_doc_indices"] = sorted(gold_pairs)
        out.append(new_ex)
    save_jsonl(str(dst), out)
    print(f"\nRelabeled JSONL written: {dst} (+{n_added} new gold pairs)")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input", required=True, help="Path to FEVER contradiction JSONL")
    p.add_argument("--mode", choices=["audit", "relabel"], default="audit")
    p.add_argument("--model", default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
                   help="HuggingFace NLI model. Must output {entailment, "
                        "neutral, contradiction} 3-way logits. Default is "
                        "trained on MNLI+FEVER+ANLI which fits this task. "
                        "Avoid `cross-encoder/nli-deberta-v3-base` — it gives "
                        "wildly over-confident contradiction scores when used "
                        "via plain AutoModelForSequenceClassification.")
    p.add_argument("--threshold", type=float, default=0.95,
                   help="Flag pairs with contradiction probability > threshold. "
                        "Lower (0.7-0.85) for high-recall manual review; "
                        "higher (0.95-0.99) for relabel mode where you want "
                        "high precision. NLI models tend to over-call "
                        "topically related but logically unrelated claims, "
                        "so use 0.95+ before injecting suspects into training.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-examples", type=int, default=0,
                   help="Audit only the first N examples (0 = all)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default=None,
                   help="Where to write outputs (default: same dir as input)")
    args = p.parse_args()

    src = Path(args.input)
    examples = load_jsonl(str(src))
    if args.max_examples and len(examples) > args.max_examples:
        examples = examples[:args.max_examples]
    print(f"Loaded {len(examples)} examples from {src}")

    print(f"Loading NLI model: {args.model} on {args.device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.eval().to(args.device)

    # Locate the contradiction label id from id2label
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    contradiction_id = next(
        (i for i, lab in id2label.items()
         if "contradict" in lab or lab == "contradiction"),
        None,
    )
    if contradiction_id is None:
        raise RuntimeError(
            f"Could not find contradiction label in model.config.id2label "
            f"({id2label}). Pass a 3-way NLI model."
        )
    print(f"  Contradiction label id: {contradiction_id} ({id2label[contradiction_id]})")

    tasks = list(_build_tasks(examples))
    print(f"NLI tasks: {len(tasks)}")
    probs = _run_nli_batched(
        tasks, model, tokenizer, contradiction_id,
        args.batch_size, args.device,
    )

    suspects = []
    for (ex_idx, pi, d_idx, _premise, _hyp), prob in zip(tasks, probs):
        if prob > args.threshold:
            suspects.append({
                "example_idx": ex_idx,
                "gold_pair_idx": pi,
                "suspect_doc_idx": d_idx,
                "contradiction_prob": prob,
            })

    _summarize(suspects, examples)

    out_dir = Path(args.output_dir) if args.output_dir else src.parent
    if args.mode == "audit":
        dst = out_dir / f"{src.stem}.audit.jsonl"
        _emit_audit(suspects, examples, dst)
    else:
        dst = out_dir / f"{src.stem}.relabeled.jsonl"
        _emit_relabeled(suspects, examples, dst)


if __name__ == "__main__":
    main()
