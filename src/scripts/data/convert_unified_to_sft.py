"""
Convert corpus-reasoning *unified JSONL* (any task in the suite) into OLMo-core SFT datasets
tokenized with the Qwen3 chat template.

Generalizes ``convert_longctx_tasks_to_sft.py`` (which hand-mirrors only oolong + contradiction)
to the **full task roster** by dispatching prompt construction through the vendored
``olmo_core.data.corpus_reasoning_prompts.build_prompt`` -- the SAME builder the oe-eval ``cr_*``
suite uses for scoring-time prompts, so training and eval inputs are byte-identical.

Per row the task/cot_mode come from ``_task`` / ``_cot_mode`` (combined multitask file) or, absent
those, from ``--task`` / ``--cot-mode`` (single-task file -- the suite's per-task train files). The
bare ``build_prompt(..., use_alpaca=False)`` user content is wrapped in a single-turn Qwen3 chat
instance; the loss mask covers the assistant turn (char-offset derived).

Output matches ``convert_rlhn_to_sft.py`` / ``convert_longctx_tasks_to_sft.py`` -- the format the
Qwen3-4B SFT scripts consume, and a valid ``NumpyDocumentSource`` (EOS-separated documents) for
``LandmarkPackingInstanceSource`` / ``ConcatAndChunk`` packing:

  * ``token_ids_part_NNNNNN.npy``  -- raw uint32, documents concatenated, each EOS(151643)-terminated.
  * ``labels_mask_NNNNNN.npy``     -- raw bool, True only on assistant-response tokens.

Run (CPU; via gantry with weka mounted and the unified JSONL on weka)::

    python src/scripts/data/convert_unified_to_sft.py \\
        --task contradiction --cot-mode template \\
        --input-jsonl '/weka/.../cr_suite_data/contradiction_train_pubmed_both_n*_k3.jsonl' \\
        --out-dir /weka/.../prasanns/suite_it_sft_qwen/contradiction
"""

import argparse
import glob
import json
import logging
import os
from typing import Iterator, List, Optional, Tuple

import numpy as np

from olmo_core.data.corpus_reasoning_prompts import build_prompt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger("convert_unified")

DEFAULT_TOKENIZER = "Qwen/Qwen3-0.6B"
EOS_TOKEN_ID = 151643  # <|endoftext|> -- OLMo-core document separator (Qwen3 turns end in <|im_end|>)
LANDMARK_TOKEN_ID = 151860  # reserved id inserted later by the landmark sources; must NOT appear
TOKEN_DTYPE = np.uint32
MASK_DTYPE = np.bool_


def render_chat(tok, messages: List[dict], add_generation_prompt: bool) -> str:
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )


def tokenize_instance(
    tok, user_content: str, answer: str
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Tokenize one (user, assistant) chat instance -> (token_ids, labels_mask), or None if the body
    contains a reserved id. Loss mask is char-offset derived (every token starting at/after the
    assistant header boundary), so it covers the whole assistant turn incl. the closing ``<|im_end|>``."""
    messages = [{"role": "user", "content": user_content}]
    prompt_str = render_chat(tok, messages, add_generation_prompt=True)
    full_str = render_chat(
        tok, messages + [{"role": "assistant", "content": answer}], add_generation_prompt=False
    )
    if not full_str.startswith(prompt_str):
        raise RuntimeError(
            "Rendered prompt is not a prefix of the full conversation; cannot derive the loss mask."
        )
    if not tok.is_fast:
        raise RuntimeError("A fast tokenizer is required for offset-based mask derivation.")

    enc = tok(full_str, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    if EOS_TOKEN_ID in ids:  # the appended EOS separator must not occur inside the body
        return None
    boundary = len(prompt_str)
    token_ids = np.asarray(list(ids) + [EOS_TOKEN_ID], dtype=TOKEN_DTYPE)
    if LANDMARK_TOKEN_ID in token_ids:
        return None
    mask = np.zeros(token_ids.shape, dtype=MASK_DTYPE)
    for i, (start, _end) in enumerate(offsets):
        if start >= boundary:
            mask[i] = True
    return token_ids, mask


def iter_examples(jsonl_patterns: List[str], limit: int, limit_per_file: int = 0) -> Iterator[dict]:
    paths: List[str] = []
    for pattern in jsonl_patterns:
        matched = sorted(glob.glob(pattern))
        if not matched and os.path.exists(pattern):
            matched = [pattern]
        paths.extend(matched)
    if not paths:
        raise FileNotFoundError(f"No JSONL files matched: {jsonl_patterns}")
    log.info(f"reading {len(paths)} JSONL file(s)")
    n_yielded = 0
    for path in paths:
        n_from_file = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                example = json.loads(line)
                if "ex" in example and "documents" not in example:
                    example = example["ex"]
                yield example
                n_yielded += 1
                n_from_file += 1
                if limit > 0 and n_yielded >= limit:
                    return
                if limit_per_file > 0 and n_from_file >= limit_per_file:
                    break


class ShardWriter:
    """Buffers (token_ids, labels_mask) and flushes raw paired shard files."""

    def __init__(self, out_dir: str, flush_tokens: int):
        self.out_dir = out_dir
        self.flush_tokens = flush_tokens
        self.tok_buf: List[np.ndarray] = []
        self.mask_buf: List[np.ndarray] = []
        self.buffered = 0
        self.part = 0
        self.total_tokens = 0
        self.total_loss_tokens = 0

    def add(self, token_ids: np.ndarray, mask: np.ndarray) -> None:
        self.tok_buf.append(token_ids)
        self.mask_buf.append(mask)
        self.buffered += token_ids.size
        self.total_loss_tokens += int(mask.sum())
        if self.buffered >= self.flush_tokens:
            self.flush()

    def flush(self) -> None:
        if self.buffered == 0:
            return
        tokens = np.concatenate(self.tok_buf)
        masks = np.concatenate(self.mask_buf)
        assert tokens.size == masks.size
        tok_path = os.path.join(self.out_dir, f"token_ids_part_{self.part:06d}.npy")
        mask_path = os.path.join(self.out_dir, f"labels_mask_{self.part:06d}.npy")
        tokens.tofile(tok_path + ".tmp")
        masks.tofile(mask_path + ".tmp")
        os.replace(tok_path + ".tmp", tok_path)
        os.replace(mask_path + ".tmp", mask_path)
        self.total_tokens += int(tokens.size)
        log.info(f"wrote part {self.part:06d}: {tokens.size:,} tokens; total {self.total_tokens:,}")
        self.part += 1
        self.tok_buf = []
        self.mask_buf = []
        self.buffered = 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-jsonl", nargs="+", required=True, help="unified JSONL path(s)/glob(s)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--task", default=None,
        help="task type for every row (single-task file). Per-row '_task' overrides this.",
    )
    parser.add_argument(
        "--cot-mode", default="label",
        help="cot_mode for every row. Per-row '_cot_mode' overrides this.",
    )
    parser.add_argument(
        "--query-position", default="both", choices=("before", "after", "both"),
        help="where the task ask sits relative to the context (must match the eval setting).",
    )
    parser.add_argument("--max-seq-len", type=int, default=64512)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--flush-tokens", type=int, default=100_000_000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--limit-per-file", type=int, default=0)
    parser.add_argument("--print-examples", type=int, default=2)
    args = parser.parse_args()

    if args.task is None:
        log.info("--task not set; every row must carry a '_task' field (combined multitask file).")

    os.makedirs(args.out_dir, exist_ok=True)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    assert tok.vocab_size <= np.iinfo(TOKEN_DTYPE).max

    writer = ShardWriter(args.out_dir, args.flush_tokens)
    n_written = n_skipped_too_long = n_skipped_bad = 0
    by_task: dict = {}

    for example in iter_examples(args.input_jsonl, args.limit, args.limit_per_file):
        task = example.get("_task", args.task)
        cot_mode = example.get("_cot_mode", args.cot_mode)
        if task is None:
            raise ValueError("Row has no '_task' and --task was not provided.")
        user_content, answer = build_prompt(
            example, task=task, query_position=args.query_position,
            use_alpaca=False, cot_mode=cot_mode,
        )
        # Cheap pre-filter before paying for a long tokenization.
        if len(user_content) + len(answer) > args.max_seq_len * 8:
            n_skipped_too_long += 1
            continue
        result = tokenize_instance(tok, user_content, answer)
        if result is None:
            n_skipped_bad += 1
            continue
        token_ids, mask = result
        if token_ids.size > args.max_seq_len:
            n_skipped_too_long += 1
            continue
        if not mask.any():
            n_skipped_bad += 1
            continue

        if n_written < args.print_examples:
            log.info(
                "EXAMPLE %d [%s/%s] (%d tokens, %d loss):\n%s",
                n_written, task, cot_mode, token_ids.size, int(mask.sum()),
                tok.decode(token_ids[:800].tolist()),
            )
        writer.add(token_ids, mask)
        by_task[task] = by_task.get(task, 0) + 1
        n_written += 1
        if n_written % 500 == 0:
            log.info(f"{n_written:,} instances written")

    writer.flush()
    meta = {
        "input_jsonl": args.input_jsonl,
        "task": args.task,
        "cot_mode": args.cot_mode,
        "query_position": args.query_position,
        "tokenizer": args.tokenizer,
        "eos_token_id": EOS_TOKEN_ID,
        "dtype": "uint32",
        "mask_dtype": "bool",
        "max_seq_len": args.max_seq_len,
        "num_instances": n_written,
        "num_instances_by_task": by_task,
        "num_tokens": writer.total_tokens,
        "num_loss_tokens": writer.total_loss_tokens,
        "num_parts": writer.part,
        "skipped_too_long": n_skipped_too_long,
        "skipped_bad": n_skipped_bad,
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log.info("DONE: %s", json.dumps(meta))


if __name__ == "__main__":
    main()
