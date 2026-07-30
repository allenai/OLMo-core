"""Generate chain-of-thought reasoning for retrieval data using an LLM API.

Given unified JSONL with documents and gold_doc_indices, prompts an LLM with
only the gold documents + question to produce a short CoT explanation of why
those documents are relevant. The CoT is stored in the `chain_of_thought` field.

Routes via `scripts/lib/llm_request_client.ParallelResponsesClient`, which
handles both OpenAI (via OPENAI_API_KEY) and Gemini (via Vertex AI using
GOOGLE_CLOUD_PROJECT + GOOGLE_GENAI_USE_VERTEXAI=true +
GOOGLE_APPLICATION_CREDENTIALS).

Usage:
    python scripts/data/generate_cot.py \
        --input-data data/hotpotqa_train_k100_bridge_unified_hn98_2000.jsonl \
        --model gemini-2.5-flash-lite

    # Resume interrupted generation (skips examples that already have CoTs)
    python scripts/data/generate_cot.py \
        --input-data <input.jsonl> --output <output.jsonl> \
        --model gemini-2.5-flash-lite --resume
"""

import argparse
from pathlib import Path

from corpus_reasoning.lib.io import load_jsonl, save_jsonl
from corpus_reasoning.lib.llm_request_client import ParallelResponsesClient


PROMPT_TEMPLATE = """\
You are a helpful assistant that explains why specific documents are \
relevant to answering a question. Be concise (2-4 sentences). Reference \
documents by their ID (e.g., [1], [2]). Focus on what information each \
document provides and how they connect to answer the question.

Question: {question}

{documents}

Gold answer: {answer}

Explain briefly why the above document(s) are relevant to answering the \
question. Reference each document by its [ID]. Keep your reasoning to 2-4 \
sentences."""


def build_cot_prompt(example):
    """Build the prompt for CoT generation using only gold documents."""
    gold_indices = example["gold_doc_indices"]
    documents = example["documents"]
    query = example["queries"][0]
    answer = example["answers"][0]

    # Format only gold documents with their actual position IDs (1-indexed)
    doc_parts = []
    for gi in sorted(gold_indices):
        doc = documents[gi]
        doc_id = gi + 1  # 1-indexed
        title = doc.get("title")
        if title:
            doc_parts.append(f"Document [{doc_id}]: {title}\n{doc['text']}")
        else:
            doc_parts.append(f"Document [{doc_id}]:\n{doc['text']}")

    return PROMPT_TEMPLATE.format(
        question=query,
        documents="\n\n".join(doc_parts),
        answer=answer,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate chain-of-thought reasoning for retrieval data")
    parser.add_argument("--input-data", required=True,
                        help="Input unified JSONL file")
    parser.add_argument("--output", default=None,
                        help="Output JSONL file (default: input with _cot suffix)")
    parser.add_argument("--model", default="gemini-2.5-flash-lite",
                        help="LLM model name (default: gemini-2.5-flash-lite)")
    parser.add_argument("--max-concurrent", type=int, default=20,
                        help="Max concurrent API requests (default: 20)")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Sampling temperature (default: 0.3)")
    parser.add_argument("--num-examples", type=int, default=0,
                        help="Process only first N examples (0=all)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume: load output file and skip examples with existing CoTs")
    args = parser.parse_args()

    # Determine output path
    if args.output is None:
        inp = Path(args.input_data)
        args.output = str(inp.parent / f"{inp.stem}_cot{inp.suffix}")

    # Load data
    print(f"Loading {args.input_data}...")
    examples = load_jsonl(args.input_data)
    print(f"  Loaded {len(examples)} examples")

    if args.num_examples > 0:
        examples = examples[:args.num_examples]
        print(f"  Using first {len(examples)} examples")

    # Resume: load existing output and skip completed examples
    if args.resume and Path(args.output).exists():
        print(f"Resuming from {args.output}...")
        existing = load_jsonl(args.output)
        existing_cots = {}
        for ex in existing:
            key = (ex["queries"][0], tuple(ex["gold_doc_indices"]))
            if ex.get("chain_of_thought"):
                existing_cots[key] = ex["chain_of_thought"]

        to_generate = []
        to_generate_indices = []
        for i, ex in enumerate(examples):
            key = (ex["queries"][0], tuple(ex["gold_doc_indices"]))
            if key in existing_cots:
                ex["chain_of_thought"] = existing_cots[key]
            else:
                to_generate.append(ex)
                to_generate_indices.append(i)

        print(f"  {len(examples) - len(to_generate)} already have CoTs, "
              f"{len(to_generate)} remaining")
    else:
        to_generate = examples
        to_generate_indices = list(range(len(examples)))

    if not to_generate:
        print("All examples already have CoTs. Saving and exiting.")
        save_jsonl(args.output, examples)
        print(f"  Saved {len(examples)} examples to {args.output}")
        return

    client = ParallelResponsesClient(
        max_concurrent=args.max_concurrent, use_cache=True,
    )
    print(f"Generating CoTs with {args.model} "
          f"(concurrency={args.max_concurrent}, temp={args.temperature})...")

    prompts = [build_cot_prompt(ex) for ex in to_generate]
    responses = client.run(
        model=args.model, prompts=prompts,
        temperature=args.temperature, max_output_tokens=256,
    )
    cots = [
        (r.get("response") or "").strip() if r.get("success", True) else None
        for r in responses
    ]

    # Merge results back
    success = 0
    fail = 0
    for idx, cot in zip(to_generate_indices, cots):
        if cot:
            examples[idx]["chain_of_thought"] = cot
            success += 1
        else:
            fail += 1

    print(f"  Generated {success} CoTs, {fail} failures")

    # Save
    save_jsonl(args.output, examples)
    print(f"  Saved {len(examples)} examples to {args.output}")

    if fail > 0:
        print(f"  Tip: re-run with --resume to retry {fail} failed examples")


if __name__ == "__main__":
    main()
