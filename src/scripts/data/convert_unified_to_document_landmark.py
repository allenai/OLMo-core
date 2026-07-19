"""
Convert **unified-format** task JSONL into **document-chunked** SFT shards for OLMo-core -- either the
**dense** layout (``--emit dense``, for :class:`~olmo_core.nn.attention.DocumentChunkedAttention`) or
the **landmark** layout (``--emit landmark``, for
:class:`~olmo_core.nn.attention.DocumentLandmarkAttention`).

Document boundaries use **registered special tokens** ``<|box_start|>`` / ``<|box_end|>``: each
context document/item-line is wrapped at the *string* level and the tokenizer emits the boundary ids
natively. The boundary/EOS/landmark/pad **ids are tokenizer-specific** -- select them with
``--marker-set`` (``qwen3``: 151648/151649/151643, the default; ``qwen3_5``: 248049/248050/248044,
matching ``Qwen/Qwen3.5-*``); the resolved ids are verified against ``--tokenizer`` at startup and
recorded in the shard's ``metadata.json``. All the shared logic (prompt rendering,
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

from olmo_core.data.document_chunk_landmark import (  # canonical ids -- never retype
    DOC_END_STR,
    DOC_START_STR,
    RESERVED_IDS,
    ReservedIds,
    emit_document_chunk_dense,
    emit_document_chunk_landmark,
    reserved_ids,
    segment_prompt_to_chunks,
)

log = logging.getLogger("convert_unified_doc_chunked")

DEFAULT_TOKENIZER = "Qwen/Qwen3-4B"

TOKEN_DTYPE = np.uint32
MASK_DTYPE = np.bool_


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
    use_titles: bool,
    free_pad_repeat: int = 0,
    repeat_doc_text: int = 1,
    summary_every_k: int = 0,
    ids_set: ReservedIds = RESERVED_IDS["qwen3"],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Tokenize one unified example into a document-chunked instance, or ``None`` to skip.

    :param ids_set: The tokenizer-specific reserved ids (boundary markers / EOS / landmark / pad).
        Defaults to the Qwen3 set, which keeps every existing invocation byte-identical.
    """
    segments, ids, _mask = segment_prompt_to_chunks(
        tok,
        example,
        task,
        query_position=query_position,
        cot_mode=cot_mode,
        chunk_by=chunk_by,
        item_regex=item_regex,
        include_answer=True,
        use_titles=use_titles,
        doc_start_id=ids_set.doc_start,
        doc_end_id=ids_set.doc_end,
        free_pad_repeat=free_pad_repeat,
        repeat_doc_text=repeat_doc_text,
        summary_every_k=summary_every_k,
    )
    # Ids the converter inserts; if any already occur in a rendered prompt, the example is ambiguous.
    if any(t in (ids_set.landmark, ids_set.pad) for t in ids):
        return None  # the rendered prompt already contains a reserved inserted id -> ambiguous

    if emit == "dense":
        out_ids, out_mask = emit_document_chunk_dense(segments)
    else:  # landmark
        out_ids, out_mask = emit_document_chunk_landmark(
            segments, mem_freq=mem_freq, mem_id=ids_set.landmark, pad_id=ids_set.pad
        )

    # Append the EOS document separator (PadToLengthInstanceSource pads each instance up to seq_len).
    out_ids.append(ids_set.eos)
    out_mask.append(False)
    if len(out_ids) > seq_len:
        return None  # too long for this sequence length -> drop (raise --seq-len to keep)
    return np.asarray(out_ids, dtype=TOKEN_DTYPE), np.asarray(out_mask, dtype=MASK_DTYPE)


def _gold_fingerprint_entry(out_ids: np.ndarray, example: dict):
    """
    Build the ``(content_fingerprint, sorted gold chunk indices)`` sidecar entry for one emitted
    instance, or ``None`` if the example has no ``gold_doc_indices``.

    The fingerprint is over the emitted content ids (which end with the single real EOS; the loader
    right-pads afterwards), so it matches ``content_fingerprint_from_row(input_ids, eos)`` at train
    time. Only valid for the **dense** doc-chunked layout, where wrapped-doc / chunk index equals the
    document's stream order (== ``Claim id - 1``); the landmark layout repacks documents into windows,
    so its chunk indexing does not line up with ``gold_doc_indices`` and is rejected by the caller.
    """
    from olmo_core.nn.attention.gold_grad_mask import (
        content_fingerprint,
        gold_chunks_from_gold_doc_indices,
    )

    gdi = example.get("gold_doc_indices")
    if not gdi:
        return None
    fp = content_fingerprint(out_ids.tolist())
    return fp, sorted(gold_chunks_from_gold_doc_indices(gdi))


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
    p.add_argument(
        "--use-titles",
        action="store_true",
        help="Render document titles (Document [N] (Title: ...): ...). Off by default so a "
        "per-document title can't shortcut the task (e.g. review-outlier titles that name the "
        "product category). MUST match the eval (eval_lc_native_docchunk.py --use-titles).",
    )
    p.add_argument(
        "--free-pad-repeat",
        type=int,
        default=0,
        help="Append N repeats of FREE_PAD_SENTENCE after the documents and before the answer. These "
        "are FREE tokens -- they attend the WHOLE context under every cross_doc_mode -- so this widens "
        "the budget of positions that can do cross-document comparison. The knob for the "
        "'do the FREE tokens saturate at 100 documents?' experiment. "
        "MUST match the eval (--free-pad-repeat), or the model sees a prompt layout it never trained on.",
    )
    p.add_argument(
        "--repeat-doc-text",
        type=int,
        default=1,
        help="Repeat each document's TEXT N times inside its chunk. Grows the chunk without "
        "changing the document COUNT or the document-level attention graph -- the control for "
        "--free-pad-repeat (adds tokens that are NEVER free). MUST match the eval.",
    )
    p.add_argument(
        "--summary-every-k",
        type=int,
        default=0,
        help="Emit the summary_attention layout: one extra 'Summary of claims X to Y: [X]...[Y]' span "
        "-- its own chunk -- after every K documents, so chunk indices run on a stride of K+1 and the "
        "summary_attention mask identifies a span as (chunk_id %% (K+1)) == K. 0 (default) = off. "
        "MUST match the eval (--summary-every-k) and the model's summary_every_k, or every chunk role "
        "is silently rebound. The bandwidth/relay knobs are NOT here -- they live on the attention "
        "config, so ONE shard serves every rung of the bandwidth ladder.",
    )
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument(
        "--marker-set",
        default="qwen3",
        choices=sorted(RESERVED_IDS),
        help="Which tokenizer family's reserved ids to use (boundary markers / EOS / landmark / "
        "pad). MUST match --tokenizer -- the boundary ids are verified against it at startup. "
        "Default 'qwen3' keeps existing invocations byte-identical.",
    )
    p.add_argument(
        "--doc-start-id",
        type=int,
        default=None,
        help="Override the <|box_start|> id from --marker-set (rarely needed).",
    )
    p.add_argument(
        "--doc-end-id",
        type=int,
        default=None,
        help="Override the <|box_end|> id from --marker-set (rarely needed).",
    )
    p.add_argument(
        "--eos-token-id",
        type=int,
        default=None,
        help="Override the EOS terminator id from --marker-set (rarely needed).",
    )
    p.add_argument(
        "--landmark-id",
        type=int,
        default=None,
        help="(landmark) Override the landmark token id from --marker-set.",
    )
    p.add_argument(
        "--pad-id",
        type=int,
        default=None,
        help="(landmark) Override the window-fill pad id from --marker-set.",
    )
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shard-tokens", type=int, default=20_000_000)
    p.add_argument(
        "--emit-gold-sidecar",
        action="store_true",
        help="Also write gold_fingerprints.json {content_fingerprint -> [gold chunk idx]} aligned to "
        "the emitted shards, for gold-document gradient masking (olmo_core.nn.attention.gold_grad_mask). "
        "Dense layout only (the landmark repacking breaks the chunk-index == Claim-id-1 alignment).",
    )
    args = p.parse_args()

    if args.emit == "landmark" and args.seq_len % (args.mem_freq + 1) != 0:
        raise SystemExit(
            f"--seq-len must be a multiple of block_size (mem_freq+1={args.mem_freq + 1}) for landmark."
        )
    if args.emit_gold_sidecar and args.emit != "dense":
        raise SystemExit("--emit-gold-sidecar requires --emit dense (chunk index == Claim id - 1).")

    # Resolve the reserved-id set: --marker-set base, explicit --*-id flags override field-by-field.
    ids_set = reserved_ids(args.marker_set)
    overrides = {
        "doc_start": args.doc_start_id,
        "doc_end": args.doc_end_id,
        "eos": args.eos_token_id,
        "landmark": args.landmark_id,
        "pad": args.pad_id,
    }
    ids_set = ids_set._replace(**{k: v for k, v in overrides.items() if v is not None})

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    # A shard built with one id set and a model/eval expecting another produces plausible-looking
    # numbers, not a crash -- so verify the resolved ids against THIS tokenizer, loudly.
    for name, tok_str, want in (
        ("doc_start", DOC_START_STR, ids_set.doc_start),
        ("doc_end", DOC_END_STR, ids_set.doc_end),
    ):
        got = tok.convert_tokens_to_ids(tok_str)
        if got != want:
            raise SystemExit(
                f"marker-id mismatch: tokenizer {args.tokenizer!r} maps {tok_str!r} -> {got}, but "
                f"the resolved id set (--marker-set {args.marker_set}) expects {name}={want}. "
                "Pass the --marker-set matching the tokenizer (e.g. --marker-set qwen3_5 for a "
                "Qwen3.5 tokenizer) or explicit --doc-start-id/--doc-end-id overrides."
            )
    if ids_set.eos >= len(tok):
        raise SystemExit(
            f"eos id {ids_set.eos} is out of range for tokenizer {args.tokenizer!r} "
            f"(len {len(tok)}); --marker-set {args.marker_set} does not match this tokenizer."
        )
    log.info(
        f"marker set '{args.marker_set}': doc_start={ids_set.doc_start} doc_end={ids_set.doc_end} "
        f"eos={ids_set.eos} landmark={ids_set.landmark} pad={ids_set.pad}"
    )

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

    gold_table: dict = {}
    n_gold = 0
    lengths: List[int] = []
    n_loss_tokens = 0
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
            use_titles=args.use_titles,
            free_pad_repeat=args.free_pad_repeat,
            repeat_doc_text=args.repeat_doc_text,
            summary_every_k=args.summary_every_k,
            ids_set=ids_set,
        )
        if res is None:
            dropped += 1
            continue
        ids, mask = res
        if args.emit_gold_sidecar:
            entry = _gold_fingerprint_entry(ids, ex)
            if entry is not None:
                gold_table[entry[0]] = entry[1]
                n_gold += 1
        tok_buf.append(ids)
        mask_buf.append(mask)
        kept += 1
        lengths.append(len(ids))
        n_loss_tokens += int(np.asarray(mask).sum())
        buffered += len(ids)
        if buffered >= args.shard_tokens:
            flush()
            buffered = 0
    flush()

    # ---- metadata.json: a HARD requirement of the docchunk trainers ----
    # They read `num_instances` (which drives the mask-mixing anneal, `mix_total_forwards`) and
    # `max_example_len` (to refuse a --seq-len that would make PadToLength silently SKIP long
    # examples). This file used to be absent and got hand-backfilled per shard, which meant a shard
    # could carry a WRONG num_instances -- silently corrupting the curriculum -- or none at all, and
    # the trainer would simply die. Write it here so the shard is self-describing by construction.
    meta = {
        "task": args.task,
        "emit": args.emit,
        "cot_mode": args.cot_mode,
        "chunk_by": args.chunk_by,
        "wrap_docs": True,
        "free_pad_repeat": args.free_pad_repeat,
        "summary_every_k": args.summary_every_k,
        "use_titles": args.use_titles,
        "tokenizer": args.tokenizer,
        "marker_set": args.marker_set,
        "eos_token_id": ids_set.eos,
        "doc_start_id": ids_set.doc_start,
        "doc_end_id": ids_set.doc_end,
        "landmark_token_id": ids_set.landmark if args.emit == "landmark" else None,
        "pad_token_id": ids_set.pad if args.emit == "landmark" else None,
        "mem_freq": args.mem_freq if args.emit == "landmark" else None,
        "dtype": np.dtype(TOKEN_DTYPE).name,
        "mask_dtype": "bool",
        "num_instances": kept,
        "num_dropped": dropped,
        "num_tokens": int(sum(lengths)),
        "num_loss_tokens": n_loss_tokens,
        "max_example_len": int(max(lengths)) if lengths else 0,
        "min_example_len": int(min(lengths)) if lengths else 0,
    }
    meta_path = os.path.join(args.out_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(
        f"metadata.json -> {meta_path} (num_instances={kept}, max_example_len="
        f"{meta['max_example_len']}, num_loss_tokens={n_loss_tokens})"
    )
    if args.emit_gold_sidecar:
        gold_path = os.path.join(args.out_dir, "gold_fingerprints.json")
        with open(gold_path, "w") as f:
            json.dump(gold_table, f)
        log.info(f"gold sidecar: {n_gold} examples with gold_doc_indices -> {gold_path}")
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
