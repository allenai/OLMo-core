"""Pre-tokenize augmented contradiction JSONL into torchtitan-ready SFT tensors.

Each example becomes (tokens, labels) where:
  tokens : full input sequence (BOS + prompt + answer + EOS)
  labels : same length; -100 on prompt tokens, real ids on answer + EOS tokens

The dataloader at training time does the standard (x[:-1], x[1:]) shift.

Prompt format (query_position=both, matches existing contradiction configs):
  {INSTRUCTION}\\n\\n{claim1}\\n\\n...\\n\\n{claimN}\\n\\n{INSTRUCTION}\\n\\nAnswer: {json}<eos>

Output: a single .pt file = list[dict] with `tokens` and `labels` int32 tensors,
plus a small `meta` JSON sidecar describing the file (tokenizer, prompt format,
seq lengths per example).

Usage:
    python scripts/data/tokenize_contradiction_for_titan.py \\
        --input data/contradiction_train_pubmed_both_n23000_k3_first10.jsonl \\
        --tokenizer Qwen/Qwen3-4B-Base \\
        --output data/contradiction_n23000_first10_qwen3_tok.pt
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from corpus_reasoning.lib.prompts import CONTRADICTION_INSTRUCTION, CLAIM_TEMPLATE


IGNORE_INDEX = -100


def build_prompt_and_answer(example: dict) -> tuple[str, str]:
    """Return (prompt_text, answer_text) for one contradiction example.

    Mirrors the qboth format used in configs/contradiction_pubmed_*_qboth_*.yml:
    INSTRUCTION → claims → INSTRUCTION → "Answer: " (the preamble is part of
    the prompt so the completion begins right at the JSON list).
    """
    claims = "\n\n".join(
        CLAIM_TEMPLATE.format(id=i + 1, text=doc["text"])
        for i, doc in enumerate(example["documents"])
    )
    prompt = (
        f"{CONTRADICTION_INSTRUCTION}\n\n"
        f"{claims}\n\n"
        f"{CONTRADICTION_INSTRUCTION}\n\n"
        f"Answer: "
    )
    answer = json.dumps(example["gold_doc_indices"])
    return prompt, answer


def tokenize_example(tokenizer, prompt: str, answer: str) -> tuple[list[int], list[int]]:
    """Tokenize a single SFT pair. Returns (tokens, labels) full-length lists."""
    # Disable HF's per-call truncation warnings; we know these are long.
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    eos = tokenizer.eos_token_id
    assert eos is not None, "tokenizer must have eos_token_id"

    tokens = prompt_ids + answer_ids + [eos]
    labels = [IGNORE_INDEX] * len(prompt_ids) + answer_ids + [eos]
    assert len(tokens) == len(labels)
    return tokens, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="augmented contradiction JSONL")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--output", required=True, help="output .pt path")
    args = ap.parse_args()

    print(f"Loading tokenizer: {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    print(f"  eos_token_id={tok.eos_token_id} bos_token_id={tok.bos_token_id}")

    src = Path(args.input)
    print(f"Reading {src}")
    examples = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
    print(f"  {len(examples)} examples")

    out = []
    meta = {
        "tokenizer": args.tokenizer,
        "ignore_index": IGNORE_INDEX,
        "prompt_format": "qboth + Answer:",
        "per_example": [],
    }
    for i, ex in enumerate(examples):
        prompt, answer = build_prompt_and_answer(ex)
        tokens, labels = tokenize_example(tok, prompt, answer)
        n_prompt = sum(1 for l in labels if l == IGNORE_INDEX)
        n_answer = len(labels) - n_prompt
        out.append({
            "tokens": torch.tensor(tokens, dtype=torch.int32),
            "labels": torch.tensor(labels, dtype=torch.int64),  # int64 for CE loss
        })
        meta["per_example"].append({
            "idx": i,
            "n_docs": len(ex["documents"]),
            "n_tokens": len(tokens),
            "n_prompt_tokens": n_prompt,
            "n_answer_tokens": n_answer,
            "gold_pairs": ex["gold_doc_indices"],
        })
        print(f"  ex {i}: n_docs={len(ex['documents'])} "
              f"tokens={len(tokens)} prompt={n_prompt} answer={n_answer}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
