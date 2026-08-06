"""
Re-tokenize existing **Qwen3** SFT shards into **Qwen3.5** shards, preserving document order.

Motivation: the canonical 32k 5-task ladders (``prasanns/single_task_ladders_v2/*`` and
``single_task_ladders_p10/nq``) exist only as Qwen3-tokenized shards; their source JSONL lives on the
Berkeley cluster and is not reachable from weka. This script rebuilds the Qwen3.5 twin from the
shards themselves.

**This is not a blind decode/re-encode.** Both chat templates render a single-turn instance to the
identical structure::

    <|im_start|>user\\n{USER}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n{ANSWER}<|im_end|>\\n

so the script decodes each document, *parses* ``USER``/``ANSWER`` back out with those fixed
delimiters, and then re-renders through **Qwen3.5's own** ``apply_chat_template`` with an
offset-derived loss mask -- exactly what ``convert_unified_to_sft.py`` does from JSONL. The output is
therefore equivalent to a rebuild from source, not a transliteration of Qwen3's rendering.

The only place the two templates differ is where the generation boundary sits: Qwen3 puts the empty
``<think>`` block *after* ``add_generation_prompt``, Qwen3.5 *before* it. That changes which tokens
the loss mask covers (Qwen3 masks in the think block, Qwen3.5 does not). Re-rendering per-template
gives each model the mask its own template implies, which is the correct behaviour.

Order is preserved exactly: parts are read in sorted glob order, documents in file order, and the
writer appends in arrival order. Only part *boundaries* move (parts flush on a token count, and
Qwen3.5 tokenizes the same text to a different length).

Any document that does not parse is **skipped and counted**, never silently mangled; the run fails
loudly if the skip rate exceeds ``--max-skip-frac``.

Run (CPU, via gantry with weka mounted)::

    python src/scripts/data/retokenize_sft_shards_qwen3_to_qwen35.py \\
        --in-dir  /weka/.../prasanns/single_task_ladders_v2/contradiction \\
        --out-dir /weka/.../prasanns/single_task_ladders_v2_qwen35/contradiction
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
log = logging.getLogger("retokenize_qwen35")

TOKEN_DTYPE = np.uint32
MASK_DTYPE = np.bool_

# Document separator appended by the converters (NOT tok.eos_token_id, which is <|im_end|>).
SRC_EOS_DEFAULT = 151643  # Qwen3 <|endoftext|>
DST_EOS_DEFAULT = 248044  # Qwen3.5 <|endoftext|>
DST_LANDMARK_DEFAULT = 248200  # reserved row the landmark sources insert; must not occur in a body

USER_PREFIX = "<|im_start|>user\n"
ASSISTANT_SEP = "<|im_end|>\n<|im_start|>assistant\n"
THINK_BLOCK = "<think>\n\n</think>\n\n"
TURN_END = "<|im_end|>"


def iter_documents(in_dir: str, src_eos: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Yield ``(token_ids, labels_mask)`` per document, in shard order, separator stripped.

    :param in_dir: Directory holding ``token_ids_part_*.npy`` + ``labels_mask_*.npy``.
    :param src_eos: The document-separator id used when the shards were written.

    :raises FileNotFoundError: If no token parts are found.
    :raises ValueError: If a token part has no matching mask part or differing lengths.
    """
    tok_paths = sorted(glob.glob(os.path.join(in_dir, "token_ids_part_*.npy")))
    if not tok_paths:
        raise FileNotFoundError(f"no token_ids_part_*.npy under {in_dir}")
    for tok_path in tok_paths:
        part = os.path.basename(tok_path).replace("token_ids_part_", "").replace(".npy", "")
        mask_path = os.path.join(in_dir, f"labels_mask_{part}.npy")
        if not os.path.exists(mask_path):
            raise ValueError(f"missing mask part for {tok_path}")
        tokens = np.fromfile(tok_path, dtype=TOKEN_DTYPE)
        masks = np.fromfile(mask_path, dtype=MASK_DTYPE)
        if tokens.size != masks.size:
            raise ValueError(f"{tok_path}: {tokens.size} tokens vs {masks.size} mask entries")
        # Documents are separator-terminated; split after each separator.
        (sep_idx,) = np.nonzero(tokens == src_eos)
        start = 0
        for end in sep_idx:
            yield tokens[start:end], masks[start:end]  # drop the separator itself
            start = int(end) + 1
        if start < tokens.size:  # trailing fragment without a separator (shouldn't happen)
            log.warning(f"{tok_path}: {tokens.size - start} trailing tokens with no separator")


def parse_chat(text: str) -> Optional[Tuple[str, str]]:
    """
    Recover ``(user_content, answer)`` from a rendered single-turn Qwen chat string.

    :param text: The decoded document (special tokens kept, separator already stripped).

    :returns: The user content and the assistant answer, or ``None`` if ``text`` does not have the
        expected single-turn shape.
    """
    if not text.startswith(USER_PREFIX):
        return None
    body = text[len(USER_PREFIX) :]
    sep_at = body.find(ASSISTANT_SEP)
    if sep_at < 0:
        return None
    user_content = body[:sep_at]
    answer = body[sep_at + len(ASSISTANT_SEP) :]
    if answer.startswith(THINK_BLOCK):  # Qwen3 renders the empty think block after the boundary
        answer = answer[len(THINK_BLOCK) :]
    # Strip the closing turn marker (with or without its trailing newline).
    if answer.endswith(TURN_END + "\n"):
        answer = answer[: -(len(TURN_END) + 1)]
    elif answer.endswith(TURN_END):
        answer = answer[: -len(TURN_END)]
    else:
        return None
    # A second user turn would mean this is not the single-turn shape the converters emit.
    if USER_PREFIX in user_content or ASSISTANT_SEP in answer:
        return None
    return user_content, answer


def render_and_mask(
    tok, user_content: str, answer: str, dst_eos: int, dst_landmark: int
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Render one instance with the destination chat template and derive its loss mask.

    :param tok: The destination fast tokenizer.
    :param user_content: The user turn's content.
    :param answer: The assistant turn's content.
    :param dst_eos: Separator id appended after the instance.
    :param dst_landmark: Reserved id that must not appear in the body.

    :returns: ``(token_ids, labels_mask)``, or ``None`` if a reserved id appears in the body.

    :raises RuntimeError: If the tokenizer is not fast, or the prompt is not a prefix of the full
        rendering (which would make the offset-derived mask meaningless).
    """
    messages = [{"role": "user", "content": user_content}]
    prompt_str = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_str = tok.apply_chat_template(
        messages + [{"role": "assistant", "content": answer}],
        tokenize=False,
        add_generation_prompt=False,
    )
    if not full_str.startswith(prompt_str):
        raise RuntimeError("rendered prompt is not a prefix of the full conversation")
    if not tok.is_fast:
        raise RuntimeError("a fast tokenizer is required for offset-based mask derivation")

    enc = tok(full_str, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    if dst_eos in ids:
        return None
    token_ids = np.asarray(list(ids) + [dst_eos], dtype=TOKEN_DTYPE)
    if dst_landmark in token_ids:
        return None
    boundary = len(prompt_str)
    mask = np.zeros(token_ids.shape, dtype=MASK_DTYPE)
    for i, (start, _end) in enumerate(enc["offset_mapping"]):
        if start >= boundary:
            mask[i] = True
    return token_ids, mask


class ShardWriter:
    """Buffers ``(token_ids, labels_mask)`` and flushes raw paired shard files."""

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
    parser.add_argument("--in-dir", required=True, help="directory of Qwen3 shards")
    parser.add_argument("--out-dir", required=True, help="directory to write Qwen3.5 shards")
    parser.add_argument("--src-tokenizer", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dst-tokenizer", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--src-eos", type=int, default=SRC_EOS_DEFAULT)
    parser.add_argument("--dst-eos", type=int, default=DST_EOS_DEFAULT)
    parser.add_argument("--dst-landmark", type=int, default=DST_LANDMARK_DEFAULT)
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=40960,
        help="drop re-tokenized instances longer than this (match the source build's value).",
    )
    parser.add_argument("--flush-tokens", type=int, default=100_000_000)
    parser.add_argument("--limit", type=int, default=0, help="stop after N documents (smoke test)")
    parser.add_argument("--print-examples", type=int, default=2)
    parser.add_argument(
        "--max-skip-frac",
        type=float,
        default=0.01,
        help="fail if more than this fraction of documents are skipped.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="parse/re-render but write nothing (use with --limit to validate a source dir).",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    src_tok = AutoTokenizer.from_pretrained(args.src_tokenizer)
    dst_tok = AutoTokenizer.from_pretrained(args.dst_tokenizer)
    assert dst_tok.vocab_size <= np.iinfo(TOKEN_DTYPE).max

    if not args.dry_run:
        os.makedirs(args.out_dir, exist_ok=True)
    writer = ShardWriter(args.out_dir, args.flush_tokens)

    n_read = n_written = 0
    n_skip_parse = n_skip_long = n_skip_bad = 0
    src_tokens = 0

    for token_ids, mask in iter_documents(args.in_dir, args.src_eos):
        n_read += 1
        src_tokens += int(token_ids.size)
        text = src_tok.decode(token_ids.tolist(), skip_special_tokens=False)
        parsed = parse_chat(text)
        if parsed is None:
            n_skip_parse += 1
            if n_skip_parse <= 3:
                log.warning("doc %d did not parse; head=%r", n_read - 1, text[:200])
            continue
        user_content, answer = parsed
        result = render_and_mask(dst_tok, user_content, answer, args.dst_eos, args.dst_landmark)
        if result is None:
            n_skip_bad += 1
            continue
        new_ids, new_mask = result
        if new_ids.size > args.max_seq_len:
            n_skip_long += 1
            continue
        if not new_mask.any():
            n_skip_bad += 1
            continue

        if n_written < args.print_examples:
            log.info(
                "EXAMPLE %d: src %d tok -> dst %d tok (%d loss)\n--- decoded dst head ---\n%s",
                n_written,
                token_ids.size,
                new_ids.size,
                int(new_mask.sum()),
                dst_tok.decode(new_ids[:400].tolist()),
            )
            # Cross-check: the source mask covered the assistant span; so should the new one.
            log.info(
                "  src loss tokens=%d dst loss tokens=%d", int(mask.sum()), int(new_mask.sum())
            )

        if not args.dry_run:
            writer.add(new_ids, new_mask)
        n_written += 1
        if n_written % 1000 == 0:
            log.info(f"{n_written:,} instances re-tokenized")
        if args.limit and n_read >= args.limit:
            break

    if not args.dry_run:
        writer.flush()

    n_skipped = n_skip_parse + n_skip_long + n_skip_bad
    skip_frac = n_skipped / max(n_read, 1)
    meta = {
        "source_shards": args.in_dir,
        "built_by": "retokenize_sft_shards_qwen3_to_qwen35.py",
        "src_tokenizer": args.src_tokenizer,
        "tokenizer": args.dst_tokenizer,
        "src_eos_token_id": args.src_eos,
        "eos_token_id": args.dst_eos,
        "landmark_token_id": args.dst_landmark,
        "dtype": "uint32",
        "mask_dtype": "bool",
        "max_seq_len": args.max_seq_len,
        "num_documents_read": n_read,
        "num_instances": n_written,
        "num_tokens": writer.total_tokens,
        "num_loss_tokens": writer.total_loss_tokens,
        "num_parts": writer.part,
        "src_num_tokens": src_tokens,
        "skipped_parse": n_skip_parse,
        "skipped_too_long": n_skip_long,
        "skipped_bad": n_skip_bad,
        "skip_fraction": round(skip_frac, 6),
        "order_preserved": True,
    }
    log.info("summary: %s", json.dumps(meta, indent=2))
    if not args.dry_run:
        with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

    if skip_frac > args.max_skip_frac:
        raise SystemExit(
            f"skip fraction {skip_frac:.4f} exceeds --max-skip-frac {args.max_skip_frac}; "
            f"parse={n_skip_parse} long={n_skip_long} bad={n_skip_bad}. Refusing to call this a "
            f"faithful re-tokenization."
        )


if __name__ == "__main__":
    main()
