"""
Convert corpus-reasoning RAG-QA task data (Natural Questions, HotpotQA) into an OLMo-core SFT
dataset tokenized with the Qwen3 chat template. This is the "NEAR" rung of the HELMET-RAG
in-distribution experiment: same task/datasets/answer-output as the HELMET RAG eval, built from
corpus-reasoning's BM25 / wikipedia-dpr-100w generators (``generate_nq_training_data.py`` /
``generate_hotpotqa_data.py``), which mirror HELMET's QA prompt format.

Input is the corpus-reasoning *unified JSONL* format (one example per line)::

    {"documents": [{"title": ..., "text": ...}, ...], "queries": [...], "answers": [...],
     "gold_doc_indices": [...], "hard_neg_indices": [...], "source": "nq" | "hotpotqa"}

Prompt construction mirrors corpus-reasoning ``build_prompt(task="qa", query_position="after")``
(scripts/lib/data_format.py + prompts.py), transplanted from the alpaca template into a single-turn
Qwen3 chat instance -- exactly the transplant done by ``convert_longctx_tasks_to_sft.py``. The
instruction and passage templates below are copied VERBATIM from corpus-reasoning
``scripts/lib/prompts.py`` so fine-tuned models stay comparable with that repo's HELMET baselines.
Do not edit casually.

Each row becomes::

    <user>
    {QA_INSTRUCTION}

    Document (Title: {t1}): {text1}

    Document (Title: {t2}): {text2}
    ...
    Question: {query}
    <assistant>
    {answer}                             # the bare short answer (answers[0]); SubEM-friendly

Output format matches ``convert_rlhn_to_sft.py`` / ``convert_longctx_tasks_to_sft.py`` (what the
Qwen3-4B SFT scripts consume):

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

Multiple input files (e.g. NQ + HotpotQA) are read ROUND-ROBIN so the two datasets interleave, and
``--target-tokens`` stops emission once that many output tokens have been written -- used to match
the total training-token count of amandab's ``rlhn_sft_qwen_63k`` (read its ``metadata.json``
``num_tokens`` and pass it here).

Run (CPU, e.g. via gantry with the weka bucket mounted and the JSONL in a Beaker dataset)::

    python src/scripts/data/convert_rag_tasks_to_sft.py \\
        --input-jsonl '/input/nq_train_*.jsonl' '/input/hotpotqa_train_*.jsonl' \\
        --target-tokens 123456789 \\
        --out-dir /weka/oe-training-default/ai2-llm/checkpoints/$USER/rag_sft_qwen/nq_hotpotqa_near

Then point the SFT scripts' dataset path at ``--out-dir`` (a run name containing 'rag' selects it).
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
log = logging.getLogger("convert_rag")

# Any Qwen3 tokenizer is identical for our purposes; matches olmo_core.data.TokenizerConfig.qwen3()
# (vocab 151936, eos/bos/pad 151643).
DEFAULT_TOKENIZER = "Qwen/Qwen3-0.6B"

EOS_TOKEN_ID = 151643  # <|endoftext|> -- OLMo-core document separator (see module docstring)
LANDMARK_TOKEN_ID = 151860  # reserved id inserted later by LandmarkInstanceSource; must NOT appear

TOKEN_DTYPE = np.uint32
MASK_DTYPE = np.bool_

# ---------------------------------------------------------------------------
# Prompt strings mirrored VERBATIM from corpus-reasoning scripts/lib/prompts.py.
# QA_INSTRUCTION is HELMET's QA ask; PASSAGE_TEMPLATE "matches HELMET eval format exactly"
# (per that file's comment). Do not edit casually.
# ---------------------------------------------------------------------------

QA_INSTRUCTION = (
    "Use the given documents to write a concise and short answer to the question. "
    "Write your answer in the following format:\nAnswer: [answer]"
)

PASSAGE_TEMPLATE = "Document (Title: {title}): {text}"
PASSAGE_TEMPLATE_NO_TITLE = "Document: {text}"

# Special-token strings stripped from body text so they can't be re-parsed as control tokens.
_SPECIAL_STRINGS = ("<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>")


def sanitize(text: Optional[str]) -> str:
    text = text or ""
    for s in _SPECIAL_STRINGS:
        text = text.replace(s, " ")
    return text.strip()


# ---------------------------------------------------------------------------
# Per-example (user_content, answer) builder. Mirrors corpus-reasoning
# build_prompt(task="qa", ...): docs formatted with PASSAGE_TEMPLATE, joined by blank lines,
# the question placed before/after/both, instruction = QA_INSTRUCTION, target = bare answer.
# ---------------------------------------------------------------------------


def _format_doc(doc: dict) -> str:
    title = sanitize(doc.get("title"))
    text = sanitize(doc.get("text"))
    if title:
        return PASSAGE_TEMPLATE.format(title=title, text=text)
    return PASSAGE_TEMPLATE_NO_TITLE.format(text=text)


def _questions_block(queries: List[str]) -> str:
    # Mirrors corpus-reasoning _build_questions_block: single -> "Question: q";
    # multi -> "Question 1: q1\nQuestion 2: q2" (multi-query HotpotQA; rare here).
    if len(queries) == 1:
        return f"Question: {queries[0]}"
    return "\n".join(f"Question {i + 1}: {q}" for i, q in enumerate(queries))


def build_rag_instance(example: dict, query_position: str) -> Tuple[str, str]:
    context = "\n\n".join(_format_doc(d) for d in example["documents"])
    queries = [sanitize(q) for q in example["queries"]]
    questions = _questions_block(queries)
    if query_position == "before":
        input_text = f"{questions}\n\n{context}"
    elif query_position == "both":
        input_text = f"{questions}\n\n{context}\n\n{questions}"
    else:  # "after" (HELMET default)
        input_text = f"{context}\n\n{questions}"
    user_content = f"{QA_INSTRUCTION}\n\n{input_text}"

    answers = example["answers"]
    if len(queries) > 1:  # multi-query QA -> comma-joined (mirrors _build_output)
        answer = ", ".join(sanitize(str(a)) for a in answers)
    else:
        answer = sanitize(str(answers[0]))
    return user_content, answer


def _iter_one_file(path: str) -> Iterator[dict]:
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


def iter_examples_round_robin(jsonl_patterns: List[str], limit: int) -> Iterator[dict]:
    """Yield examples round-robin across the matched files so e.g. NQ and HotpotQA interleave."""
    paths: List[str] = []
    for pattern in jsonl_patterns:
        matched = sorted(glob.glob(pattern))
        if not matched and os.path.exists(pattern):
            matched = [pattern]
        paths.extend(matched)
    if not paths:
        raise FileNotFoundError(f"No JSONL files matched: {jsonl_patterns}")
    log.info(f"reading {len(paths)} JSONL file(s) round-robin: {paths}")

    iterators = [_iter_one_file(p) for p in paths]
    n_yielded = 0
    while iterators:
        next_round = []
        for it in iterators:
            try:
                example = next(it)
            except StopIteration:
                continue
            next_round.append(it)
            yield example
            n_yielded += 1
            if limit > 0 and n_yielded >= limit:
                return
        iterators = next_round


# ---------------------------------------------------------------------------
# Chat-template tokenization with offset-derived loss mask (see convert_longctx_tasks_to_sft.py).
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

    @property
    def tokens_written_or_buffered(self) -> int:
        return self.total_tokens + self.buffered

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
    parser.add_argument(
        "--input-jsonl",
        nargs="+",
        required=True,
        help="One or more unified-format JSONL paths or globs (quote globs). Multiple files are "
        "read round-robin so e.g. NQ and HotpotQA interleave.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=0,
        help="Stop once this many output tokens have been written (0 = no cap). Use to match the "
        "total training-token count of amandab/rlhn_sft_qwen_63k (its metadata.json num_tokens).",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=64512,
        help="Total token budget per instance (template + context + answer + EOS). Instances "
        "exceeding the budget are skipped (the answer sits at the end, so truncation would drop "
        "it). Default 64512 = the landmark CONTENT_SEQUENCE_LENGTH; safe for the dense run too.",
    )
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--query-position",
        default="after",
        choices=("before", "after", "both"),
        help="Where the question sits relative to the documents; 'after' matches the HELMET RAG "
        "eval and corpus-reasoning's build_prompt default.",
    )
    parser.add_argument(
        "--flush-tokens", type=int, default=100_000_000, help="Tokens per shard file."
    )
    parser.add_argument("--limit", type=int, default=0, help="Process at most N rows (0 = all).")
    parser.add_argument(
        "--print-examples", type=int, default=2, help="Print the first N assembled instances."
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    assert tok.vocab_size <= np.iinfo(TOKEN_DTYPE).max
    if getattr(tok, "eos_token_id", None) not in (None, EOS_TOKEN_ID):
        log.warning(
            f"Tokenizer eos_token_id={tok.eos_token_id} != expected OLMo-core separator "
            f"{EOS_TOKEN_ID}; the appended separator is hard-coded to {EOS_TOKEN_ID}."
        )

    writer = ShardWriter(args.out_dir, args.flush_tokens)
    n_written = 0
    n_skipped_too_long = 0
    n_skipped_bad = 0
    per_source: dict = {}

    for example in iter_examples_round_robin(args.input_jsonl, args.limit):
        if args.target_tokens > 0 and writer.tokens_written_or_buffered >= args.target_tokens:
            log.info(
                f"reached target of {args.target_tokens:,} tokens "
                f"({writer.tokens_written_or_buffered:,} written/buffered); stopping."
            )
            break

        user_content, answer = build_rag_instance(example, args.query_position)

        # Cheap pre-filter: below ~2 chars/token, anything this long can never fit the token
        # budget; skip before paying for a full 60k+-token tokenization.
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
                "EXAMPLE %d (source=%s, %d tokens, %d with loss):\n%s",
                n_written,
                example.get("source", "?"),
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
        src = example.get("source", "?")
        per_source[src] = per_source.get(src, 0) + 1
        if n_written % 200 == 0:
            log.info(
                f"{n_written:,} instances written so far "
                f"({writer.tokens_written_or_buffered:,} tokens); by source: {per_source}"
            )

    writer.flush()

    meta = {
        "task": "rag",
        "input_jsonl": args.input_jsonl,
        "query_position": args.query_position,
        "tokenizer": args.tokenizer,
        "eos_token_id": EOS_TOKEN_ID,
        "dtype": "uint32",
        "mask_dtype": "bool",
        "max_seq_len": args.max_seq_len,
        "target_tokens": args.target_tokens,
        "num_instances": n_written,
        "num_instances_by_source": per_source,
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
