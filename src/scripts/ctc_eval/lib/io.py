"""Shared I/O utilities and prompt templates.

This module provides the core data loading/saving functions and the alpaca prompt
template used throughout the pipeline. Every training example and eval prompt flows
through format_alpaca_prompt() to ensure training/eval prompt alignment.

Key exports:
  - ALPACA_TEMPLATE / format_alpaca_prompt: The prompt wrapper for all trained models.
  - load_jsonl / save_jsonl: Read/write alpaca-format training data.
  - insert_dummy_tokens: Inject padding tokens around documents for ablation studies.
  - print_dataset_stats: Quick summary of generated dataset files.
"""

import json
import re
from pathlib import Path


# Alpaca prompt template — the exact wrapper Axolotl applies during training.
# Evaluation scripts must use this same template for trained models (LoRA or full FT)
# to match the prompt distribution the model was trained on. Base model evaluation
# uses HELMET templates instead (see lib/prompts.py).
ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)


def format_alpaca_prompt(instruction: str, input_text: str) -> str:
    return ALPACA_TEMPLATE.format(instruction=instruction, input=input_text)


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path: str, examples: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def save_results(path: str, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def insert_dummy_tokens(input_text: str, before_dummy: int = 0, after_dummy: int = 0,
                        dummy_token: str = "* ") -> str:
    """Insert dummy tokens before and/or after the document block in the input text.

    Used for ablation studies to test whether the model relies on positional
    proximity between the query and documents. Dummy tokens push documents
    further from the query in the sequence, testing if the model can still
    retrieve/answer correctly with increased distance.

    The input text has the structure:
        [optional: Question: ...\\n\\n] Document 1\\n\\nDocument 2\\n\\n...\\n\\nQuestion: ...

    This function finds the document block boundaries and inserts repeated
    dummy tokens ("* * * ...") before and/or after it.

    Args:
        input_text: The 'input' field of an alpaca example (docs + question).
        before_dummy: Number of dummy token repetitions to insert before documents.
        after_dummy: Number of dummy token repetitions to insert after documents.
        dummy_token: The token string to repeat (default "* ").

    Returns:
        Modified input text with dummy tokens inserted.
    """
    if before_dummy == 0 and after_dummy == 0:
        return input_text

    # Locate the start of the document block by finding the first "Document" token.
    # Handles all naming conventions: "Document (Title:", "Document:", "Document ["
    doc_match = re.search(r'Document[\s\[\(:]', input_text)
    if not doc_match:
        return input_text
    doc_start_idx = doc_match.start()

    # Locate the end of the document block by finding the trailing question.
    # For single-query tasks: "\n\nQuestion: ..."
    # For multi-query tasks: "\nQuestion 1: ..."
    doc_end_idx = len(input_text)
    trailing = input_text.rfind("\n\nQuestion:")
    if trailing > doc_start_idx:
        doc_end_idx = trailing
    else:
        trailing = input_text.rfind("\nQuestion 1:")
        if trailing > doc_start_idx:
            doc_end_idx = trailing

    # Split into three segments: before docs, docs, after docs (question)
    before_text = input_text[:doc_start_idx]
    doc_text = input_text[doc_start_idx:doc_end_idx]
    after_text = input_text[doc_end_idx:]

    # Reassemble with dummy tokens injected at the boundaries
    result = before_text
    if before_dummy > 0:
        result += dummy_token * before_dummy + "\n\n"
    result += doc_text
    if after_dummy > 0:
        result += "\n\n" + dummy_token * after_dummy
    result += after_text

    return result


def print_dataset_stats(examples: list[dict], label: str, path: str) -> None:
    if not examples:
        print(f"\n{label}: 0 examples (skipped)")
        return
    input_lens = [len(ex.get("input", "")) for ex in examples]
    output_lens = [len(ex.get("output", "")) for ex in examples]
    size_mb = Path(path).stat().st_size / 1024 / 1024
    print(f"\n{label}: {len(examples)} examples -> {path}")
    print(f"  Avg input:  {sum(input_lens)/len(input_lens):,.0f} chars")
    print(f"  Avg output: {sum(output_lens)/len(output_lens):,.0f} chars")
    print(f"  File size:  {size_mb:.1f} MB")
