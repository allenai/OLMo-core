"""Generate human-readable example files showing exactly what the model sees.

For each task variant, shows 3 training examples and 3 eval examples with
the full alpaca-wrapped prompt. Eval examples use the same code path as the
actual eval scripts (HELMET data for QA, generated JSONL for retrieval).

Usage:
    python scripts/generate_examples.py
"""

import json
import re
from pathlib import Path

from corpus_reasoning.lib.io import format_alpaca_prompt, insert_dummy_tokens
from corpus_reasoning.lib.data_format import build_prompt
from corpus_reasoning.lib.prompts import PASSAGE_TEMPLATE, QA_INSTRUCTION

EXAMPLES_DIR = Path("examples")
EXAMPLES_DIR.mkdir(exist_ok=True)

NUM_EXAMPLES = 3


def count_docs(input_text):
    """Count documents in input text."""
    n = input_text.count("Document (Title:") + input_text.count("Document [")
    if n == 0:
        n = input_text.count("Document:")
    if n == 0:
        n = input_text.count("Passage [")
    return n


def fmt_unified(task="qa", query_position="after", before_dummy=0, after_dummy=0):
    """Return a formatter function for unified-format examples."""
    def _fmt(ex):
        return build_prompt(ex, task=task, query_position=query_position,
                            before_dummy=before_dummy, after_dummy=after_dummy,
                            use_alpaca=True)
    return _fmt


def format_helmet_eval_example(sample, instruction, query_position="after",
                               before_dummy=0, after_dummy=0):
    """Format a HELMET eval example the same way evaluate_helmet_rag.py does for trained models.

    This replicates the alpaca-format code path from evaluate_helmet_rag.py:load_dataset_for_eval
    with use_alpaca=True, shots=0 (trained model eval). HELMET data uses its own format
    (ctxs list), not our unified format.
    """
    context = "\n\n".join(
        PASSAGE_TEMPLATE.format(**ctx) for ctx in sample.get("ctxs", [])
    )
    question = sample["question"]

    if query_position == "both":
        input_text = f"Question: {question}\n\n{context}\n\nQuestion: {question}"
    elif query_position == "before":
        input_text = f"Question: {question}\n\n{context}"
    else:
        input_text = f"{context}\n\nQuestion: {question}"

    if before_dummy > 0 or after_dummy > 0:
        input_text = insert_dummy_tokens(input_text, before_dummy, after_dummy)

    prompt = format_alpaca_prompt(instruction, input_text)
    answers = sample["answers"]
    return prompt, answers


def write_section(f, label, prompt, output, idx):
    """Write one example section to the file."""
    doc_count = count_docs(prompt)
    f.write(f"{'=' * 80}\n")
    f.write(f"  {label} Example {idx}\n")
    f.write(f"  Documents: {doc_count} | Prompt chars: {len(prompt):,} | Output chars: {len(output):,}\n")
    f.write(f"{'=' * 80}\n\n")
    f.write("--- MODEL INPUT (prompt) ---\n\n")
    f.write(prompt)
    f.write("\n--- EXPECTED OUTPUT (completion) ---\n\n")
    if isinstance(output, list):
        f.write(f"{output[0]}  (gold answers: {output})\n")
    else:
        f.write(output + "\n")
    f.write("\n\n")


# ── Example definitions ──
# Each entry: (name, description, train_config, eval_config)
# train_config: (file, formatter_fn)
# eval_config: (file, formatter_fn) or None

TASK_EXAMPLES = []


def add_task(name, description, train_file, eval_file, train_fmt, eval_fmt):
    TASK_EXAMPLES.append((name, description, train_file, eval_file, train_fmt, eval_fmt))


# ── Formatter shortcuts ──
# Training data uses unified format; HELMET eval data uses its own KILT format.

fmt_qa = fmt_unified(task="qa")
fmt_reorder = fmt_unified(task="reorder")
fmt_reorder_qboth = fmt_unified(task="reorder", query_position="both")
fmt_contradiction_qboth = fmt_unified(task="contradiction", query_position="both")
fmt_qa_qboth = fmt_unified(task="qa", query_position="both")
fmt_retrieval = fmt_unified(task="retrieval")
fmt_qa_qboth_dummy = fmt_unified(task="qa", query_position="both", before_dummy=10)

def fmt_helmet_qa_eval(sample):
    return format_helmet_eval_example(sample, QA_INSTRUCTION)

def fmt_helmet_qa_eval_qboth(sample):
    return format_helmet_eval_example(sample, QA_INSTRUCTION, query_position="both")

def fmt_helmet_qa_eval_qboth_dummy(sample):
    return format_helmet_eval_example(sample, QA_INSTRUCTION, query_position="both",
                                       before_dummy=10)


# QA tasks
add_task("nq_qa",
         "NQ question-answering: 20 documents, output is the answer text",
         "data/nq_train_k20_random_2500.jsonl",
         "data/data/kilt/nq-dev-multikilt_1000_k20_dep6.jsonl",
         fmt_qa, fmt_helmet_qa_eval)

add_task("nq_qa_qboth",
         "NQ question-answering with query-both: question before and after documents",
         "data/nq_train_k20_random_2500.jsonl",
         "data/data/kilt/nq-dev-multikilt_1000_k20_dep6.jsonl",
         fmt_qa_qboth, fmt_helmet_qa_eval_qboth)

add_task("hotpotqa_qa",
         "HotpotQA multi-hop QA: 2 supporting docs among 20 total, output is the answer",
         "data/hotpotqa_train_k20_shuffled_bridge_2500.jsonl",
         "data/data/kilt/hotpotqa-dev-multikilt_1000_k20_dep3.jsonl",
         fmt_qa, fmt_helmet_qa_eval)

add_task("hotpotqa_qa_qboth",
         "HotpotQA multi-hop QA with query-both",
         "data/hotpotqa_train_k20_shuffled_bridge_2500.jsonl",
         "data/data/kilt/hotpotqa-dev-multikilt_1000_k20_dep3.jsonl",
         fmt_qa_qboth, fmt_helmet_qa_eval_qboth)

add_task("multi_hotpotqa_qa",
         "Multi-query HotpotQA QA: 10 queries, comma-separated answers",
         "data/multi_hotpotqa_train_n10_k50_bridge_5000.jsonl",
         "data/multi_hotpotqa_train_n10_k50_bridge_5000.jsonl",
         fmt_qa, fmt_qa)

# Retrieval tasks
add_task("nq_retrieval",
         "NQ retrieval: 20 docs with IDs, output is the relevant document ID",
         "data/nq_train_k20_random_1000.jsonl",
         "data/nq_train_k20_random_500.jsonl",
         fmt_retrieval, fmt_retrieval)

add_task("hotpotqa_retrieval",
         "HotpotQA retrieval: 20 docs with IDs, output is the 2 relevant document IDs",
         "data/hotpotqa_train_k20_shuffled_bridge_2500.jsonl",
         "data/hotpotqa_eval_k20_shuffled_bridge_500.jsonl",
         fmt_retrieval, fmt_retrieval)

add_task("hotpotqa_qa_qboth_dummy_before",
         "HotpotQA QA with query-both and 10 dummy tokens before documents",
         "data/hotpotqa_train_k10_shuffled_bridge_5000.jsonl",
         "data/data/kilt/hotpotqa-dev-multikilt_1000_k20_dep3.jsonl",
         fmt_qa_qboth_dummy, fmt_helmet_qa_eval_qboth_dummy)

add_task("reorder_gutenberg",
         "Narrative reordering: N shuffled passages from a Gutenberg novel, output the permutation (qboth)",
         "data/reorder_gutenberg_n20_train_1995.jsonl",
         "data/reorder_gutenberg_n20_eval_105.jsonl",
         fmt_reorder_qboth, fmt_reorder_qboth)

add_task("contradiction_pubmed_qboth",
         "PubMed contradiction pairs with query-both (task instruction placed before and after the claims)",
         "data/contradiction_train_pubmed_both_n100_k3.jsonl",
         "data/contradiction_eval_pubmed_both_n100_k3.jsonl",
         fmt_contradiction_qboth, fmt_contradiction_qboth)

add_task("multi_hotpotqa_retrieval",
         "Multi-query HotpotQA retrieval: per-query relevant document IDs",
         "data/multi_hotpotqa_train_n10_suponly_bridge_500.jsonl",
         "data/multi_hotpotqa_eval_n10_suponly_bridge_100.jsonl",
         fmt_retrieval, fmt_retrieval)


def load_examples(path, n, offset=0):
    """Load n examples from a JSONL file, starting at offset."""
    examples = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            if i >= offset + n:
                break
            if line.strip():
                examples.append(json.loads(line))
    return examples


def generate_example_file(name, description, train_file, eval_file, train_fmt, eval_fmt):
    """Generate one example file with train + eval examples."""
    out_path = EXAMPLES_DIR / f"{name}.txt"

    with open(out_path, "w") as f:
        f.write(f"# {description}\n")
        f.write(f"#\n")
        f.write(f"# Train data: {train_file}\n")
        f.write(f"# Eval data:  {eval_file}\n")
        f.write(f"# Examples per section: {NUM_EXAMPLES}\n")
        f.write(f"#\n")
        f.write(f"# This file shows the exact prompt the model sees during training and evaluation.\n")
        f.write(f"# Training uses Axolotl with alpaca format. Eval uses the corresponding eval script\n")
        f.write(f"# with alpaca format + 0 few-shot demos (for trained/finetuned models).\n")
        f.write(f"\n\n")

        # ── Training examples ──
        if Path(train_file).exists():
            train_data = load_examples(train_file, NUM_EXAMPLES)
            f.write(f"{'#' * 80}\n")
            f.write(f"#  TRAINING EXAMPLES (from {train_file})\n")
            f.write(f"#  Formatted by: Axolotl alpaca format (instruction + input -> output)\n")
            f.write(f"{'#' * 80}\n\n")
            for i, ex in enumerate(train_data, 1):
                prompt, output = train_fmt(ex)
                write_section(f, "TRAIN", prompt, output, i)
        else:
            f.write(f"# TRAIN: {train_file} not found, skipping.\n\n")

        # ── Eval examples ──
        if Path(eval_file).exists():
            # If eval and train are the same file, offset to show different examples
            eval_offset = NUM_EXAMPLES if eval_file == train_file else 0
            eval_data = load_examples(eval_file, NUM_EXAMPLES, offset=eval_offset)
            f.write(f"{'#' * 80}\n")
            f.write(f"#  EVAL EXAMPLES (from {eval_file})\n")
            f.write(f"#  Formatted by: eval script with use_alpaca=True, shots=0\n")
            f.write(f"{'#' * 80}\n\n")
            for i, ex in enumerate(eval_data, 1):
                prompt, output = eval_fmt(ex)
                write_section(f, "EVAL", prompt, output, i)
        else:
            f.write(f"# EVAL: {eval_file} not found, skipping.\n\n")

    print(f"  {name} -> {out_path}")


def main():
    print(f"Generating example files in examples/ ({NUM_EXAMPLES} examples each, train + eval)...")
    for name, description, train_file, eval_file, train_fmt, eval_fmt in TASK_EXAMPLES:
        generate_example_file(name, description, train_file, eval_file, train_fmt, eval_fmt)
    print(f"\nDone. See examples/ directory.")


if __name__ == "__main__":
    main()
