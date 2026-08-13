"""
Build **summary-token** SFT shards from existing **document-chunked** (box-marker) shards.

The summary-token layout is the document-chunked layout plus a run of ``<|summ|>`` tokens after each
context document. Because the documents are already tokenized and already carry their
``<|box_start|>``/``<|box_end|>`` markers, the runs can be inserted directly into the token stream --
no re-tokenization, no source JSONL, and the documents stay **byte-identical** to the doc-chunked
arms, which is what keeps the two families comparable.

The insertion goes through the production emitter
(:func:`~olmo_core.data.document_chunk_landmark.emit_document_chunk_summary`) over segments recovered
with :func:`~olmo_core.data.document_chunk_landmark.find_chunk_spans`, so what is written here is the
same layout the converter's ``--emit summary`` produces, and the same one
:func:`~olmo_core.nn.attention.summary_mask.build_summary_roles` reads back at training time.

Input and output are the raw headerless shard pairs the rest of the pipeline uses::

    token_ids_part_NNNNNN.npy   uint32, EOS-separated examples
    labels_mask_part_NNNNNN.npy bool, True only on answer tokens

Usage::

    python src/scripts/data/build_summary_token_shards.py \\
        --in-dir  /weka/.../xlong5_2k256k_qwen35/shards_chunked/contradiction_train \\
        --out-dir /weka/.../summtoken_5task_xlong/contra_summary \\
        --marker-set qwen3_5 --num-summary-tokens 5
"""

import argparse
import glob
import json
import os
from typing import List, Tuple

import numpy as np

from olmo_core.data.document_chunk_landmark import (
    ChunkSegment,
    ReservedIds,
    emit_document_chunk_summary,
    find_chunk_spans,
    reserved_ids,
)

TOKEN_DTYPE = np.uint32
MASK_DTYPE = np.bool_


def add_summary_runs(
    ids: List[int], mask: List[bool], ids_set: ReservedIds, n_summary: int
) -> Tuple[List[int], List[bool]]:
    """Insert a ``<|summ|>`` run after every context document of one already-tokenized example."""
    spans = find_chunk_spans(ids, doc_start_id=ids_set.doc_start, doc_end_id=ids_set.doc_end)
    segments: List[ChunkSegment] = []
    cursor = 0
    for start, end in spans:
        if start > cursor:
            segments.append(ChunkSegment(ids[cursor:start], mask[cursor:start], False))
        segments.append(ChunkSegment(ids[start:end], mask[start:end], True))
        cursor = end
    if cursor < len(ids):
        segments.append(ChunkSegment(ids[cursor:], mask[cursor:], False))
    return emit_document_chunk_summary(
        segments, summary_token_id=ids_set.summary, n_summary_tokens=n_summary
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", required=True, help="a document-chunked (box-marker) shard dir")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--marker-set", default="qwen3_5")
    ap.add_argument("--num-summary-tokens", type=int, default=5)
    ap.add_argument("--dtype", default="uint32", help="input token dtype (see its metadata.json)")
    ap.add_argument(
        "--max-len",
        type=int,
        default=0,
        help="drop examples longer than this AFTER insertion (0 = keep everything). Set it to the "
        "training sequence_length so over-long examples are dropped here, visibly and counted, "
        "rather than silently at load time.",
    )
    args = ap.parse_args()

    ids_set = reserved_ids(args.marker_set)
    os.makedirs(args.out_dir, exist_ok=True)

    tok_paths = sorted(glob.glob(os.path.join(args.in_dir, "token_ids_part_*.npy")))
    if not tok_paths:
        raise SystemExit(f"no token_ids_part_*.npy in {args.in_dir}")

    n_examples = n_dropped = n_docs_total = 0
    len_before = len_after = 0
    for part, tok_path in enumerate(tok_paths):
        mask_path = tok_path.replace("token_ids_part_", "labels_mask_part_")
        toks = np.memmap(tok_path, dtype=np.dtype(args.dtype), mode="r")
        masks = np.memmap(mask_path, dtype=MASK_DTYPE, mode="r")
        if len(toks) != len(masks):
            raise SystemExit(f"{tok_path}: tokens ({len(toks)}) and mask ({len(masks)}) disagree")

        out_ids: List[np.ndarray] = []
        out_mask: List[np.ndarray] = []
        start = 0
        for eos in np.flatnonzero(np.asarray(toks) == ids_set.eos):
            end = int(eos) + 1
            ex_ids = [int(t) for t in toks[start:end]]
            ex_mask = [bool(m) for m in masks[start:end]]
            start = end

            n_docs = len(
                find_chunk_spans(ex_ids, doc_start_id=ids_set.doc_start, doc_end_id=ids_set.doc_end)
            )
            # A stray inserted id would renumber every document downstream, so refuse the example.
            if any(t == ids_set.summary for t in ex_ids):
                n_dropped += 1
                continue

            new_ids, new_mask = add_summary_runs(ex_ids, ex_mask, ids_set, args.num_summary_tokens)
            if args.max_len and len(new_ids) > args.max_len:
                n_dropped += 1
                continue

            assert (
                len(new_ids) == len(ex_ids) + n_docs * args.num_summary_tokens
            ), "summary insertion changed the length by an unexpected amount"
            n_examples += 1
            n_docs_total += n_docs
            len_before += len(ex_ids)
            len_after += len(new_ids)
            out_ids.append(np.asarray(new_ids, dtype=TOKEN_DTYPE))
            out_mask.append(np.asarray(new_mask, dtype=MASK_DTYPE))

        if out_ids:
            np.concatenate(out_ids).tofile(
                os.path.join(args.out_dir, f"token_ids_part_{part:06d}.npy")
            )
            np.concatenate(out_mask).tofile(
                os.path.join(args.out_dir, f"labels_mask_part_{part:06d}.npy")
            )
        print(
            f"  part {part:06d}: {len(out_ids)} examples -> {os.path.basename(args.out_dir)}",
            flush=True,
        )

    meta = {
        "source": args.in_dir,
        "emit": "summary",
        "marker_set": args.marker_set,
        "num_summary_tokens": args.num_summary_tokens,
        "summary_token_id": ids_set.summary,
        "doc_start_id": ids_set.doc_start,
        "doc_end_id": ids_set.doc_end,
        "eos_token_id": ids_set.eos,
        "pad_token_id": ids_set.pad,
        "dtype": np.dtype(TOKEN_DTYPE).name,
        "n_examples": n_examples,
        "n_dropped": n_dropped,
        "max_len": args.max_len or None,
        "mean_docs_per_example": (n_docs_total / n_examples) if n_examples else 0,
        "mean_len_before": (len_before / n_examples) if n_examples else 0,
        "mean_len_after": (len_after / n_examples) if n_examples else 0,
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(
        f"{args.out_dir}: {n_examples} examples ({n_dropped} dropped), "
        f"{meta['mean_docs_per_example']:.1f} docs/example, "
        f"mean len {meta['mean_len_before']:.0f} -> {meta['mean_len_after']:.0f}"
    )


if __name__ == "__main__":
    main()
