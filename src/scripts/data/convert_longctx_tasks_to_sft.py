"""
Convert corpus-reasoning long-context task data (oolong, contradiction) into OLMo-core SFT
datasets tokenized with the Qwen3 chat template.

Input is the corpus-reasoning *unified JSONL* format (one example per line)::

    {"documents": [{"text": ...}, ...], "queries": [...], "answers": [...],
     "gold_doc_indices": [...], "_meta": {...}, "source": ...}

Prompt construction mirrors corpus-reasoning ``build_prompt()`` (scripts/lib/data_format.py on
the ``task-suite-expansion`` branch) with ``query_position="both"``, transplanted from the alpaca
template into a single-turn Qwen3 chat instance. The instruction strings below are copied
VERBATIM from corpus-reasoning ``scripts/lib/prompts.py`` so fine-tuned models stay comparable
with that repo's baselines. Do not edit casually.

``oolong`` (the Oolong benchmark, arXiv:2511.02817, ported via generate_oolong_data.py)::

    <user>
    {OOLONG_INSTRUCTION}

    Question: {query}

    {context_window_text}                # documents shown verbatim, no per-document wrapper

    Question: {query}
    <assistant>
    {answer}                             # or with --cot-mode plan:
                                         # Reasoning: to answer this, {plan}\\nAnswer: {answer}

``contradiction`` (numbered-claims contradiction detection)::

    <user>
    {GENERIC_INSTRUCTION}

    {CONTRADICTION_INSTRUCTION}

    Claim 1: {text}

    Claim 2: {text}
    ...

    {CONTRADICTION_INSTRUCTION}
    <assistant>
    [[1, 4], [3, 7]]                     # or with --cot-mode enumerate:
                                         # Reasoning: 1: No conflict, 2: conflict with 9, ...
                                         # Contradicting pairs: [[1, 4], [3, 7]]

Output format matches ``convert_rlhn_to_sft.py`` (what the Qwen3-4B SFT scripts consume):

  * ``token_ids_part_NNNNNN.npy``  -- raw (headerless) ``uint32`` token IDs, documents
    concatenated and each terminated by the EOS id ``151643``.
  * ``labels_mask_NNNNNN.npy``     -- raw (headerless) ``bool``, parallel to the token file,
    ``True`` only on the assistant-response tokens (loss is computed there).

Both files are written with ``ndarray.tofile`` (NOT ``np.save``) because the reader memmaps them
as raw arrays.

CRITICAL -- EOS alignment: ``TokenizerConfig.qwen3().eos_token_id`` is ``151643``
(``<|endoftext|>``) and OLMo-core splits documents on that id; the Qwen3 chat template ends turns
with ``<|im_end|>`` = ``151645``, which is NOT a boundary, so we explicitly append ``151643``
after each conversation.

Run (CPU, e.g. via gantry with the weka bucket mounted and the JSONL in a Beaker dataset)::

    python src/scripts/data/convert_longctx_tasks_to_sft.py \\
        --task oolong \\
        --input-jsonl '/input/oolong_test_synth_ctx*_splittrain.jsonl' \\
        --out-dir /weka/oe-training-default/ai2-llm/checkpoints/$USER/longctx_sft_qwen/oolong

    python src/scripts/data/convert_longctx_tasks_to_sft.py \\
        --task contradiction --cot-mode enumerate \\
        --input-jsonl '/input/contradiction_train_pubmed_both_n*_k3.jsonl' \\
        --out-dir /weka/oe-training-default/ai2-llm/checkpoints/$USER/longctx_sft_qwen/contradiction_cot

Then point the longctx SFT scripts' ``DATA_ROOT`` at the parent ``longctx_sft_qwen`` directory.
"""

import argparse
import glob
import json
import logging
import os
from typing import Iterator, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger("convert_longctx")

# Any Qwen3 tokenizer is identical for our purposes; matches olmo_core.data.TokenizerConfig.qwen3()
# (vocab 151936, eos/bos/pad 151643).
DEFAULT_TOKENIZER = "Qwen/Qwen3-0.6B"

EOS_TOKEN_ID = 151643  # <|endoftext|> -- OLMo-core document separator (see module docstring)
LANDMARK_TOKEN_ID = 151860  # reserved id inserted later by LandmarkInstanceSource; must NOT appear

TOKEN_DTYPE = np.uint32
MASK_DTYPE = np.bool_

# ---------------------------------------------------------------------------
# Prompt strings mirrored VERBATIM from corpus-reasoning scripts/lib/prompts.py
# (task-suite-expansion branch).
# ---------------------------------------------------------------------------

GENERIC_INSTRUCTION = "You will be asked to do a long-context processing task for a context."

CONTRADICTION_INSTRUCTION = (
    "Given the following corpus of numbered claims, identify all pairs of claims "
    "that contradict each other. A pair of claims is contradictory if they cannot "
    "both be true at the same time.\n\n"
    "Output your answer as a JSON list of pairs, where each pair is a list of two "
    "claim IDs. For example: [[1, 4], [3, 7]]\n"
    "If there are no contradicting pairs, output: []"
)

CLAIM_TEMPLATE = "Claim {id}: {text}"

OOLONG_INSTRUCTION = (
    "Read the data below and answer the question. Compute the exact answer by "
    "analyzing every item; do not guess or approximate."
)

# Special-token strings stripped from body text so they can't be re-parsed as control tokens.
_SPECIAL_STRINGS = ("<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>")


def sanitize(text: Optional[str]) -> str:
    text = text or ""
    for s in _SPECIAL_STRINGS:
        text = text.replace(s, " ")
    return text.strip()


# ---------------------------------------------------------------------------
# CoT builders mirrored from corpus-reasoning scripts/lib/data_format.py.
# ---------------------------------------------------------------------------


def build_oolong_plan_cot(example: dict) -> str:
    """Generic aggregation-plan CoT (mirrors ``_build_oolong_plan_cot``)."""
    tg = (example.get("_meta") or {}).get("task_group", "")
    plan = {
        "counting": "go through every item, determine its label, tally the counts, "
        "then compare the totals.",
        "user": "go through every item, note its user id, tally per user, then find "
        "the most frequent.",
        "timeline": "go through the items in date order, track the per-period label "
        "counts, then compare across periods.",
    }.get(tg, "analyze every item and aggregate the result before answering.")
    return f"Reasoning: to answer this, {plan}"


def build_contradiction_enumerate_cot(example: dict) -> str:
    """Per-claim verdict CoT that scales with N (mirrors ``_build_contradiction_enumerate_cot``)."""
    n = len(example["documents"])
    partners: dict = {}
    for pair in example["gold_doc_indices"]:
        a, b = pair
        partners.setdefault(a, []).append(b)
        partners.setdefault(b, []).append(a)
    parts = []
    for i in range(1, n + 1):
        if i in partners:
            parts.append(f"{i}: conflict with {', '.join(str(p) for p in sorted(partners[i]))}")
        else:
            parts.append(f"{i}: No conflict")
    return "Reasoning: " + ", ".join(parts)


# ---------------------------------------------------------------------------
# Per-task (user_content, answer) builders.
# ---------------------------------------------------------------------------


def build_oolong_instance(example: dict, query_position: str, cot_mode: str) -> Tuple[str, str]:
    # Documents shown verbatim, no per-document wrapper (benchmark pre-formats the block).
    context = "\n\n".join(sanitize(doc["text"]) for doc in example["documents"])
    queries = example["queries"]
    questions = (
        f"Question: {sanitize(queries[0])}"
        if len(queries) == 1
        else "\n".join(f"Question {i+1}: {sanitize(q)}" for i, q in enumerate(queries))
    )
    if query_position == "before":
        input_text = f"{questions}\n\n{context}"
    elif query_position == "both":
        input_text = f"{questions}\n\n{context}\n\n{questions}"
    else:  # "after"
        input_text = f"{context}\n\n{questions}"
    user_content = f"{OOLONG_INSTRUCTION}\n\n{input_text}"

    answer = sanitize(str(example["answers"][0]))
    if cot_mode == "plan":
        answer = f"{build_oolong_plan_cot(example)}\nAnswer: {answer}"
    return user_content, answer


def build_contradiction_instance(
    example: dict, query_position: str, cot_mode: str
) -> Tuple[str, str]:
    # 1-indexed claims separated by blank lines (mirrors _format_documents(task="contradiction")).
    claims = "\n\n".join(
        CLAIM_TEMPLATE.format(id=i + 1, text=sanitize(doc["text"]))
        for i, doc in enumerate(example["documents"])
    )
    instr = CONTRADICTION_INSTRUCTION
    if query_position == "before":
        input_text = f"{instr}\n\n{claims}"
    elif query_position == "both":
        input_text = f"{instr}\n\n{claims}\n\n{instr}"
    else:  # "after"
        input_text = f"{claims}\n\n{instr}"
    user_content = f"{GENERIC_INSTRUCTION}\n\n{input_text}"

    # gold_doc_indices holds the contradiction pairs as 1-indexed claim-ID pairs.
    answer = json.dumps(example["gold_doc_indices"])
    if cot_mode == "enumerate":
        answer = f"{build_contradiction_enumerate_cot(example)}\nContradicting pairs: {answer}"
    return user_content, answer


def iter_examples(jsonl_patterns: List[str], limit: int, limit_per_file: int = 0) -> Iterator[dict]:
    paths: List[str] = []
    for pattern in jsonl_patterns:
        matched = sorted(glob.glob(pattern))
        if not matched and os.path.exists(pattern):
            matched = [pattern]
        paths.extend(matched)
    if not paths:
        raise FileNotFoundError(f"No JSONL files matched: {jsonl_patterns}")
    log.info(f"reading {len(paths)} JSONL file(s): {paths}")

    n_yielded = 0
    for path in paths:
        n_from_file = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                example = json.loads(line)
                # Some suite files wrap the example under 'ex' (see corpus-reasoning task-suite).
                if "ex" in example and "documents" not in example:
                    example = example["ex"]
                yield example
                n_yielded += 1
                n_from_file += 1
                if limit > 0 and n_yielded >= limit:
                    return
                if limit_per_file > 0 and n_from_file >= limit_per_file:
                    break


# ---------------------------------------------------------------------------
# Chat-template tokenization with offset-derived loss mask (see convert_rlhn_to_sft.py).
# ---------------------------------------------------------------------------


def render_chat(tok, messages: List[dict], add_generation_prompt: bool) -> str:
    """Render a chat to a string with the Qwen3 template (``enable_thinking`` left at its default)."""
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )


def tokenize_instance(
    tok, user_content: str, answer: str
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Tokenize one (user, assistant) chat instance and build its (token_ids, labels_mask) arrays,
    or ``None`` if the body contains a reserved id.

    The loss mask is derived from **character offsets**: render the prompt (conversation up to
    the assistant header) and the full conversation as strings, confirm prefix consistency,
    tokenize the full string once with ``return_offsets_mapping``, and mark every token that
    starts at or after the prompt boundary. Loss therefore covers the whole assistant turn after
    the header, including the (empty) think block the Qwen3 template injects and the closing
    ``<|im_end|>``.
    """
    messages = [{"role": "user", "content": user_content}]
    prompt_str = render_chat(tok, messages, add_generation_prompt=True)
    full_str = render_chat(
        tok, messages + [{"role": "assistant", "content": answer}], add_generation_prompt=False
    )

    if not full_str.startswith(prompt_str):
        raise RuntimeError(
            "Rendered prompt is not a string prefix of the full conversation; cannot derive the "
            "loss mask. Check the tokenizer's chat template."
        )

    if not tok.is_fast:
        raise RuntimeError("A fast tokenizer is required for offset-based mask derivation.")

    enc = tok(full_str, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    # The appended EOS is the OLMo-core document separator; it must not occur inside the body.
    if EOS_TOKEN_ID in ids:
        return None

    boundary = len(prompt_str)
    token_ids = np.asarray(list(ids) + [EOS_TOKEN_ID], dtype=TOKEN_DTYPE)
    if LANDMARK_TOKEN_ID in token_ids:
        return None
    # Loss on response tokens (start char >= boundary); prompt and trailing separator are masked.
    mask = np.zeros(token_ids.shape, dtype=MASK_DTYPE)
    for i, (start, _end) in enumerate(offsets):
        if start >= boundary:
            mask[i] = True
    return token_ids, mask


class ShardWriter:
    """Buffers (token_ids, labels_mask) arrays and flushes raw paired shard files."""

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
        log.info(
            f"wrote part {self.part:06d}: {tokens.size:,} tokens "
            f"({int(masks.sum()):,} with loss); total {self.total_tokens:,}"
        )
        self.part += 1
        self.tok_buf = []
        self.mask_buf = []
        self.buffered = 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", required=True, choices=("oolong", "contradiction"))
    parser.add_argument(
        "--input-jsonl",
        nargs="+",
        required=True,
        help="One or more unified-format JSONL paths or globs (quote globs).",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--cot-mode",
        default="none",
        choices=("none", "plan", "enumerate"),
        help="CoT prefix for the target: 'plan' (oolong only), 'enumerate' (contradiction only), "
        "or 'none' for plain answers.",
    )
    parser.add_argument(
        "--cot-fraction",
        type=float,
        default=1.0,
        help="When --cot-mode is set, the fraction of examples (chosen by a seeded RNG) that get "
        "the CoT target; the rest get plain answers. 0.5 gives a 50/50 'cotmix' dataset so one "
        "model learns both modes and the eval-time response prefix selects between them.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Seed for the CoT-mix RNG.")
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=64512,
        help="Total token budget per instance (template + content + answer + EOS). Instances "
        "exceeding the budget are skipped (the answer sits at the end, so truncation would drop "
        "it). Default 64512 = the landmark CONTENT_SEQUENCE_LENGTH; safe for the dense run too.",
    )
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--query-position",
        default="both",
        choices=("before", "after", "both"),
        help="Where the question/instruction sits relative to the context; 'both' matches the "
        "corpus-reasoning baseline configs.",
    )
    parser.add_argument(
        "--flush-tokens", type=int, default=100_000_000, help="Tokens per shard file."
    )
    parser.add_argument("--limit", type=int, default=0, help="Process at most N rows (0 = all).")
    parser.add_argument(
        "--limit-per-file",
        type=int,
        default=0,
        help="Take at most N rows from each input file (0 = all). Used to subsample e.g. the "
        "contradiction ladder evenly across context sizes.",
    )
    parser.add_argument(
        "--print-examples", type=int, default=2, help="Print the first N assembled instances."
    )
    args = parser.parse_args()

    if args.cot_mode == "plan" and args.task != "oolong":
        parser.error("--cot-mode plan is only valid with --task oolong")
    if args.cot_mode == "enumerate" and args.task != "contradiction":
        parser.error("--cot-mode enumerate is only valid with --task contradiction")

    os.makedirs(args.out_dir, exist_ok=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    assert tok.vocab_size <= np.iinfo(TOKEN_DTYPE).max
    if getattr(tok, "eos_token_id", None) not in (None, EOS_TOKEN_ID):
        log.warning(
            f"Tokenizer eos_token_id={tok.eos_token_id} != expected OLMo-core separator "
            f"{EOS_TOKEN_ID}; the appended separator is hard-coded to {EOS_TOKEN_ID}."
        )

    build = build_oolong_instance if args.task == "oolong" else build_contradiction_instance

    writer = ShardWriter(args.out_dir, args.flush_tokens)
    n_written = 0
    n_skipped_too_long = 0
    n_skipped_bad = 0

    import random

    rng = random.Random(args.seed)
    n_cot = 0

    for example in iter_examples(args.input_jsonl, args.limit, args.limit_per_file):
        use_cot = args.cot_mode != "none" and rng.random() < args.cot_fraction
        if use_cot:
            n_cot += 1
        user_content, answer = build(
            example, args.query_position, args.cot_mode if use_cot else "none"
        )

        # Cheap pre-filter: below ~2 chars/token, anything this long can never fit the token
        # budget; skip before paying for a full 100k+-token tokenization.
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
                "EXAMPLE %d (%d tokens, %d with loss):\n%s",
                n_written,
                token_ids.size,
                int(mask.sum()),
                tok.decode(token_ids[:1200].tolist())
                + (
                    "\n...[truncated]...\n" + tok.decode(token_ids[-400:].tolist())
                    if token_ids.size > 1600
                    else ""
                ),
            )

        writer.add(token_ids, mask)
        n_written += 1
        if n_written % 200 == 0:
            log.info(f"{n_written:,} instances written so far")

    writer.flush()

    meta = {
        "task": args.task,
        "input_jsonl": args.input_jsonl,
        "limit_per_file": args.limit_per_file,
        "cot_mode": args.cot_mode,
        "cot_fraction": args.cot_fraction,
        "num_cot_instances": n_cot,
        "seed": args.seed,
        "query_position": args.query_position,
        "tokenizer": args.tokenizer,
        "eos_token_id": EOS_TOKEN_ID,
        "dtype": "uint32",
        "mask_dtype": "bool",
        "max_seq_len": args.max_seq_len,
        "num_instances": n_written,
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
