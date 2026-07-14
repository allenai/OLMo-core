"""
Convert ``allenai/tulu-3-sft-mixture`` multi-turn chat data into an OLMo-core SFT dataset
tokenized with the Qwen3 chat template, in the EXACT shard format produced by
``OLMo-core/src/scripts/data/convert_longctx_tasks_to_sft.py`` so it can be mixed via
``NumpyDocumentSource``.

Output (one shard, no flushing for ~25M tokens):

  * ``token_ids_part_000000.npy`` -- raw (headerless) ``uint32`` token IDs written with
    ``ndarray.tofile`` (NOT ``np.save``). All instances concatenated; each instance is the full
    Qwen3-chat-rendered conversation with the OLMo-core document separator EOS id ``151643``
    appended at the end.
  * ``labels_mask_000000.npy`` -- raw (headerless) ``bool`` written with ``tofile``, parallel to
    the token file, ``True`` ONLY on assistant-response tokens (completion-only loss), ``False``
    on system/user/template/separator tokens.  NOTE the naming asymmetry: the token file has the
    ``part_`` infix, the mask file does not (matches the longctx converter / NumpyDocumentSource).
  * ``metadata.json`` -- num_instances, num_tokens, num_loss_tokens, etc.

Loss-mask derivation (multi-turn robust): the Qwen3 template renders a conversation as a flat
concatenation of ``<|im_start|>{role}\n...{content}...<|im_end|>\n`` segments (it also injects an
empty ``<think>\n\n</think>`` block into the *last* assistant turn -- and into any assistant turn
that ends a partial render, which is why incremental prefix rendering is unreliable). We therefore
render the whole conversation ONCE, scan for ``<|im_start|>assistant\n`` segment headers, and mark
every token whose start char falls inside an assistant segment's response body (everything after
the header through that segment's trailing ``<|im_end|>\n``). This puts loss on the entire
assistant turn -- including the empty think block and the closing ``<|im_end|>`` -- exactly like
``tokenize_instance`` in the longctx converter does for the single-turn case.

CRITICAL -- EOS alignment: the Qwen3 chat template ends turns with ``<|im_end|>`` = ``151645``,
which is NOT the OLMo-core document boundary. OLMo-core splits documents on
``TokenizerConfig.qwen3().eos_token_id`` = ``151643`` (``<|endoftext|>``), so we explicitly append
``151643`` (masked False) after each conversation.

Run::

    TOKENIZERS_PARALLELISM=false \
    /scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python \
        /scratch/users/prasann/corpus-reasoning/scripts/data/convert_tulu3_to_sft.py
"""

import argparse
import json
import logging
import os
from typing import List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger("convert_tulu3")

DEFAULT_TOKENIZER = "Qwen/Qwen3-0.6B"

EOS_TOKEN_ID = 151643  # <|endoftext|> -- OLMo-core document separator
LANDMARK_TOKEN_ID = 151860  # reserved id; must NOT appear in the body

TOKEN_DTYPE = np.uint32
MASK_DTYPE = np.bool_

ASSISTANT_HEADER = "<|im_start|>assistant\n"
IM_START = "<|im_start|>"


def assistant_spans(full_str: str) -> List[Tuple[int, int]]:
    """Return [start, end) character spans of every assistant *response body* in a rendered Qwen3
    conversation. The body starts right after the ``<|im_start|>assistant\\n`` header and runs to
    the start of the next ``<|im_start|>`` (or end of string), so it includes the closing
    ``<|im_end|>\\n`` and any empty ``<think></think>`` block -- matching the single-turn converter.
    """
    spans: List[Tuple[int, int]] = []
    pos = 0
    n = len(full_str)
    while True:
        h = full_str.find(ASSISTANT_HEADER, pos)
        if h == -1:
            break
        body_start = h + len(ASSISTANT_HEADER)
        nxt = full_str.find(IM_START, body_start)
        body_end = nxt if nxt != -1 else n
        spans.append((body_start, body_end))
        pos = body_end
    return spans


def tokenize_conversation(
    tok, messages: List[dict]
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Render a full multi-turn conversation and build (token_ids, labels_mask), or ``None`` if it
    contains a reserved id."""
    full_str = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    if not tok.is_fast:
        raise RuntimeError("A fast tokenizer is required for offset-based mask derivation.")

    spans = assistant_spans(full_str)
    if not spans:
        return None

    enc = tok(full_str, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    # The appended EOS is the OLMo-core document separator; it must not occur inside the body.
    if EOS_TOKEN_ID in ids:
        return None

    token_ids = np.asarray(list(ids) + [EOS_TOKEN_ID], dtype=TOKEN_DTYPE)
    if LANDMARK_TOKEN_ID in token_ids:
        return None

    mask = np.zeros(token_ids.shape, dtype=MASK_DTYPE)
    si = 0
    for i, (start, _end) in enumerate(offsets):
        # advance span pointer
        while si < len(spans) and start >= spans[si][1]:
            si += 1
        if si < len(spans) and spans[si][0] <= start < spans[si][1]:
            mask[i] = True
    # trailing appended EOS stays False
    return token_ids, mask


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", default="/scratch/users/prasann/cpt_data/tulu3_sft_qwen")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--target-tokens", type=int, default=25_000_000,
        help="Stop once this many tokens have been accumulated (cap).",
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=8192,
        help="Skip any single conversation whose rendered length exceeds this many tokens.",
    )
    parser.add_argument("--print-examples", type=int, default=2)
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.makedirs(args.out_dir, exist_ok=True)

    from transformers import AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    assert tok.vocab_size <= np.iinfo(TOKEN_DTYPE).max
    log.info("tokenizer loaded: vocab=%d is_fast=%s", tok.vocab_size, tok.is_fast)

    ds = load_dataset(
        "allenai/tulu-3-sft-mixture", split="train", streaming=True
    )

    tok_buf: List[np.ndarray] = []
    mask_buf: List[np.ndarray] = []
    total_tokens = 0
    total_loss_tokens = 0
    n_written = 0
    n_skipped_too_long = 0
    n_skipped_bad = 0
    n_seen = 0

    for row in ds:
        n_seen += 1
        messages = row.get("messages")
        if not messages:
            n_skipped_bad += 1
            continue
        # normalize to {role, content}
        msgs = [
            {"role": m["role"], "content": m.get("content", "") or ""}
            for m in messages
            if m.get("role") in ("system", "user", "assistant")
        ]
        if not any(m["role"] == "assistant" for m in msgs):
            n_skipped_bad += 1
            continue

        result = tokenize_conversation(tok, msgs)
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
                n_written, token_ids.size, int(mask.sum()),
                tok.decode(token_ids[:1200].tolist()),
            )

        tok_buf.append(token_ids)
        mask_buf.append(mask)
        total_tokens += int(token_ids.size)
        total_loss_tokens += int(mask.sum())
        n_written += 1
        if n_written % 500 == 0:
            log.info(
                "%d instances, %s tokens (%.1f%% loss), %d seen",
                n_written, f"{total_tokens:,}",
                100.0 * total_loss_tokens / max(total_tokens, 1), n_seen,
            )

        if total_tokens >= args.target_tokens:
            log.info("reached target tokens (%s); stopping", f"{total_tokens:,}")
            break

    tokens = np.concatenate(tok_buf)
    masks = np.concatenate(mask_buf)
    assert tokens.size == masks.size

    tok_path = os.path.join(args.out_dir, "token_ids_part_000000.npy")
    mask_path = os.path.join(args.out_dir, "labels_mask_000000.npy")
    tokens.tofile(tok_path + ".tmp")
    masks.tofile(mask_path + ".tmp")
    os.replace(tok_path + ".tmp", tok_path)
    os.replace(mask_path + ".tmp", mask_path)

    meta = {
        "dataset": "allenai/tulu-3-sft-mixture",
        "split": "train",
        "tokenizer": args.tokenizer,
        "eos_token_id": EOS_TOKEN_ID,
        "dtype": "uint32",
        "mask_dtype": "bool",
        "max_seq_len": args.max_seq_len,
        "target_tokens": args.target_tokens,
        "num_instances": n_written,
        "num_tokens": int(tokens.size),
        "num_loss_tokens": int(masks.sum()),
        "num_parts": 1,
        "num_seen": n_seen,
        "skipped_too_long": n_skipped_too_long,
        "skipped_bad": n_skipped_bad,
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log.info("DONE: %s", json.dumps(meta))


if __name__ == "__main__":
    main()
