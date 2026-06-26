"""
Convert **unified-format** task JSONL into **document-chunked** SFT shards for OLMo-core -- either the
**dense** layout (``--emit dense``, for :class:`~olmo_core.nn.attention.DocumentChunkedAttention`) or
the **landmark** layout (``--emit landmark``, for
:class:`~olmo_core.nn.attention.DocumentLandmarkAttention`).

Document boundaries use **registered special tokens** ``<|box_start|>`` / ``<|box_end|>`` (real Qwen3
reserved tokens; ids 151648 / 151649): each context document/item-line is wrapped at the *string*
level and the tokenizer emits the boundary ids natively. All the shared logic (prompt rendering,
wrapping, tokenization, segmentation) lives in
:func:`olmo_core.data.document_chunk_landmark.segment_prompt_to_chunks`, which is also used by the
native eval harness so train and eval token layouts match exactly.

  * ``--chunk-by line`` (OOLONG): each item line matching ``--item-regex`` (default ``||``) is a
    document; intro / instruction / question lines stay FREE. **Document count scales with context
    length.**
  * ``--chunk-by document`` (absence / retrieval / contradiction / ...): each ``documents[i]`` is a
    document.

The dense layout is just the wrapped tokens + EOS. The landmark layout additionally first-fit packs
documents into landmark windows (``block = mem_freq + 1``) with a landmark at every block-end (see
:func:`~olmo_core.data.document_chunk_landmark.emit_document_chunk_landmark`).

Output (raw, headerless; matches the other longctx SFT shards / NumpyPaddedFSLDataset reader):
  * ``token_ids_part_NNNNNN.npy``  -- ``uint32`` token ids, EOS-terminated (PadToLength pads at load).
  * ``labels_mask_part_NNNNNN.npy`` -- ``bool``, True only on answer tokens.

Example::

    python src/scripts/data/convert_unified_to_document_landmark.py \\
        --emit dense --task oolong --chunk-by line \\
        --input-jsonl '/scratch/.../oolong_test_synth_ctx2048_splittrain.jsonl' \\
        --cot-mode plan --seq-len 4096 --out-dir /scratch/.../oolong_ctx2048_docdense
"""

import argparse
import glob
import json
import logging
import os
from typing import List, Optional, Tuple

import numpy as np
from transformers import AutoTokenizer

from olmo_core.data.document_chunk_landmark import (
    DOC_END_ID,
    DOC_START_ID,
    emit_document_chunk_dense,
    emit_document_chunk_landmark,
    segment_prompt_to_chunks,
)

log = logging.getLogger("convert_unified_doc_chunked")

# Qwen3 ids (match olmo_core.data.TokenizerConfig.qwen3() + the document_chunk_landmark defaults).
EOS_TOKEN_ID = 151643  # <|endoftext|>
LANDMARK_TOKEN_ID = (
    151860  # reserved id reused as the landmark/memory token (matches the base ckpt)
)
PAD_TOKEN_ID = 151863  # dedicated interior window-fill padding (never supervised; runtime -> PAD)
DEFAULT_TOKENIZER = "Qwen/Qwen3-4B"

TOKEN_DTYPE = np.uint32
MASK_DTYPE = np.bool_
# Ids the converter inserts; if any already occur in a rendered prompt, the example is ambiguous.
RESERVED_INSERTED = (LANDMARK_TOKEN_ID, PAD_TOKEN_ID)


def tokenize_example(
    tok,
    example: dict,
    task: str,
    *,
    emit: str,
    query_position: str,
    cot_mode: str,
    mem_freq: int,
    seq_len: int,
    chunk_by: str,
    item_regex: str,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Tokenize one unified example into a document-chunked instance, or ``None`` to skip."""
    segments, ids, _mask = segment_prompt_to_chunks(
        tok,
        example,
        task,
        query_position=query_position,
        cot_mode=cot_mode,
        chunk_by=chunk_by,
        item_regex=item_regex,
        include_answer=True,
        doc_start_id=DOC_START_ID,
        doc_end_id=DOC_END_ID,
    )
    if any(t in RESERVED_INSERTED for t in ids):
        return None  # the rendered prompt already contains a reserved inserted id -> ambiguous

    if emit == "dense":
        out_ids, out_mask = emit_document_chunk_dense(segments)
    else:  # landmark
        out_ids, out_mask = emit_document_chunk_landmark(
            segments, mem_freq=mem_freq, mem_id=LANDMARK_TOKEN_ID, pad_id=PAD_TOKEN_ID
        )

    # Append the EOS document separator (PadToLengthInstanceSource pads each instance up to seq_len).
    out_ids.append(EOS_TOKEN_ID)
    out_mask.append(False)
    if len(out_ids) > seq_len:
        return None  # too long for this sequence length -> drop (raise --seq-len to keep)
    return np.asarray(out_ids, dtype=TOKEN_DTYPE), np.asarray(out_mask, dtype=MASK_DTYPE)


def iter_examples(patterns: List[str], limit: int) -> List[dict]:
    paths: List[str] = []
    for pat in patterns:
        paths.extend(sorted(glob.glob(pat)) or ([pat] if os.path.exists(pat) else []))
    if not paths:
        raise FileNotFoundError(f"No JSONL matched: {patterns}")
    out: List[dict] = []
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                if "ex" in ex and "documents" not in ex:
                    ex = ex["ex"]
                out.append(ex)
                if limit > 0 and len(out) >= limit:
                    return out
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input-jsonl", nargs="+", required=True)
    p.add_argument(
        "--task", required=True, help="Unified task name (oolong/retrieval/absence/...)."
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--emit",
        default="landmark",
        choices=["landmark", "dense"],
        help="'landmark': pack into landmark windows (DocumentLandmarkAttention). 'dense': wrapped "
        "tokens only, no landmarks (DocumentChunkedAttention).",
    )
    p.add_argument(
        "--seq-len", type=int, default=4096, help="Max instance length; longer is dropped."
    )
    p.add_argument(
        "--mem-freq",
        type=int,
        default=63,
        help="(landmark) tokens between landmarks; block = mem_freq+1.",
    )
    p.add_argument("--query-position", default="both", choices=["before", "after", "both"])
    p.add_argument("--cot-mode", default="plan", help="Prompt CoT mode (oolong: 'plan' or 'none').")
    p.add_argument(
        "--chunk-by",
        default="line",
        choices=["document", "line"],
        help="'line': each item line is a document (OOLONG). 'document': each documents[i].",
    )
    p.add_argument(
        "--item-regex",
        default=r"\|\|",
        help="In --chunk-by=line mode, a line is a document iff this regex matches it (default "
        "matches OOLONG's 'Date: ... || ...' items).",
    )
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shard-tokens", type=int, default=20_000_000)
    args = p.parse_args()

    if args.emit == "landmark" and args.seq_len % (args.mem_freq + 1) != 0:
        raise SystemExit(
            f"--seq-len must be a multiple of block_size (mem_freq+1={args.mem_freq + 1}) for landmark."
        )

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    os.makedirs(args.out_dir, exist_ok=True)
    examples = iter_examples(args.input_jsonl, args.limit)
    log.info(
        f"loaded {len(examples)} examples (task={args.task}, emit={args.emit}, "
        f"chunk_by={args.chunk_by}); tokenizing -> {args.out_dir}"
    )

    tok_buf: List[np.ndarray] = []
    mask_buf: List[np.ndarray] = []
    part = 0
    kept = dropped = 0
    buffered = 0

    def flush():
        nonlocal part
        if not tok_buf:
            return
        np.concatenate(tok_buf).tofile(os.path.join(args.out_dir, f"token_ids_part_{part:06d}.npy"))
        np.concatenate(mask_buf).tofile(
            os.path.join(args.out_dir, f"labels_mask_part_{part:06d}.npy")
        )
        part += 1
        tok_buf.clear()
        mask_buf.clear()

    lengths: List[int] = []
    for ex in examples:
        res = tokenize_example(
            tok,
            ex,
            args.task,
            emit=args.emit,
            query_position=args.query_position,
            cot_mode=args.cot_mode,
            mem_freq=args.mem_freq,
            seq_len=args.seq_len,
            chunk_by=args.chunk_by,
            item_regex=args.item_regex,
        )
        if res is None:
            dropped += 1
            continue
        ids, mask = res
        tok_buf.append(ids)
        mask_buf.append(mask)
        kept += 1
        lengths.append(len(ids))
        buffered += len(ids)
        if buffered >= args.shard_tokens:
            flush()
            buffered = 0
    flush()
    if lengths:
        arr = np.asarray(lengths)
        log.info(
            f"length tokens: min {arr.min()} / p50 {int(np.percentile(arr, 50))} / "
            f"p90 {int(np.percentile(arr, 90))} / max {arr.max()}"
        )
    log.info(
        f"done: kept {kept}, dropped {dropped} (too long / reserved collision). shards: {part}"
    )


if __name__ == "__main__":
    main()
