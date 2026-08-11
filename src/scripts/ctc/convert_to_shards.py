"""
Unified task JSONL -> olmo-core SFT shards, in the dense or landmark document-chunked layout.

This is the bridge between the two halves of the pipeline: ``ctc-data`` produces task JSONL, this
produces the ``.npy`` shards a trainer reads. It lives here rather than in the ``ctc`` package
because it writes olmo-core's format and imports olmo-core -- ``ctc`` itself stays dependency-free.

Document boundaries are **registered special tokens** ``<|box_start|>`` / ``<|box_end|>``: each
context document (or item line) is wrapped at the string level and the tokenizer emits the boundary
ids natively. Those ids are tokenizer-specific -- pick them with ``--marker-set`` and they are
verified against ``--tokenizer`` at startup. All the shared rendering, wrapping, tokenizing and
segmenting lives in :func:`olmo_core.data.document_chunk_landmark.segment_prompt_to_chunks`, which
the native eval harness also uses, so train and eval token layouts match by construction.

**Every shard directory gets a format fingerprint.** That is what closes the loop: this writes it
beside the shards, ``ctc.train.FormatFingerprintCallback`` collects it into every checkpoint, and
``ctc-eval`` refuses to grade a checkpoint against a format it was not trained on. Converting is
also the only place ``doc_id_range`` can honestly be measured, because it is the only place that
sees the training data -- measuring it from an eval file produces a field that contains itself.

Output (raw, headerless; matches ``NumpyPaddedFSLDataset``):

* ``token_ids_part_NNNNNN.npy`` -- ``uint32`` ids, EOS-terminated.
* ``labels_mask_part_NNNNNN.npy`` -- ``bool``, True only on answer tokens.
* ``metadata.json`` -- shard stats the docchunk trainers require.
* ``format_fingerprint.json`` -- the format guard's record.

Example::

    python src/scripts/ctc/convert_to_shards.py \\
        --task contradiction --emit dense --chunk-by document --marker-set qwen3_5 \\
        --tokenizer /path/to/Qwen3.5-4B-Base --query-position both --seq-len 40960 \\
        --input-jsonl /data/ctc/v3/contradiction/train.jsonl \\
        --out-dir /data/ctc/shards/contradiction
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import multiprocessing as mp
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("ctc.convert")

DEFAULT_TOKENIZER = "Qwen/Qwen3-4B"
TOKEN_DTYPE = np.uint32
MASK_DTYPE = np.bool_

_W: Dict[str, Any] = {}


# ── guards that need neither olmo-core nor a tokenizer ──────────────────────────────────────────


def check_item_regex(item_regex: str) -> None:
    """
    Reject an ``--item-regex`` that matches every line.

    A regex matching the empty string matches *every* line, so the instruction, question and header
    lines each become their own chunk and the blank lines between them stay FREE -- bridging chunks
    that should be isolated, and mismatching the eval layout, which keeps the preamble FREE. The
    bare ``'||'`` is exactly this: an alternation of two empty branches. It silently produced the
    oolong chunk leak (2019 inter-chunk FREE tokens, ~5/example), so it fails loudly now.

    :param item_regex: The pattern to check.

    :raises SystemExit: If the pattern is invalid or matches the empty string.
    """
    try:
        compiled = re.compile(item_regex)
    except re.error as e:
        raise SystemExit(f"--item-regex {item_regex!r} is not a valid regex: {e}")
    if compiled.search("") is not None:
        raise SystemExit(
            f"--item-regex {item_regex!r} matches the EMPTY STRING, so it matches every line: the "
            "instruction/question/header lines would each be wrapped as their own chunk and the "
            "blank lines between them would stay FREE, bridging chunks and mismatching the eval "
            r"layout. Did you mean '\|\|' (escaped) rather than '||'?"
        )


def check_cot_mode(cot_mode: str) -> None:
    """
    Refuse a CoT mode this repo's evaluator cannot render.

    ``ctc`` dropped the CoT prompt builders during the port: 150 of 150 CTC-suite result rows are
    no-cot, ``cot_mode`` was never recorded in them, and the eval path renders ``"none"``
    unconditionally. A shard built with a CoT preamble therefore trains fine, grades fine, and
    grades the wrong thing. The pre-migration tree had exactly this pair live --
    ``TASK_CFG["oolong"]["cot"] == "plan"`` against every converter building ``--cot-mode none``.

    :param cot_mode: The requested mode.

    :raises SystemExit: For anything but ``"none"``.
    """
    if cot_mode != "none":
        raise SystemExit(
            f"--cot-mode {cot_mode!r} is not supported: ctc's eval path renders cot_mode='none' "
            "unconditionally, so a shard built this way could not be evaluated by this repo. See "
            "ctc/src/ctc/tasks/README.md for why the CoT builders were dropped."
        )


def check_marker_ids(tok, tokenizer_name: str, marker_set: str, ids_set) -> None:
    """
    Verify the resolved reserved ids against the tokenizer actually being used.

    A shard built with one id set and graded by a model expecting another produces plausible
    numbers, not a crash, so this is checked rather than trusted.

    :param tok: The loaded tokenizer.
    :param tokenizer_name: Its name, for the error message.
    :param marker_set: The ``--marker-set`` name, for the error message.
    :param ids_set: The resolved :class:`~olmo_core.data.document_chunk_landmark.ReservedIds`.

    :raises SystemExit: On any mismatch.
    """
    from olmo_core.data.document_chunk_landmark import DOC_END_STR, DOC_START_STR

    for name, token, want in (
        ("doc_start", DOC_START_STR, ids_set.doc_start),
        ("doc_end", DOC_END_STR, ids_set.doc_end),
    ):
        got = tok.convert_tokens_to_ids(token)
        if got != want:
            raise SystemExit(
                f"marker-id mismatch: tokenizer {tokenizer_name!r} maps {token!r} -> {got}, but "
                f"--marker-set {marker_set} expects {name}={want}. Pass the marker set matching the "
                "tokenizer (--marker-set qwen3_5 for a Qwen3.5 tokenizer) or explicit id overrides."
            )
    if ids_set.eos >= len(tok):
        raise SystemExit(
            f"eos id {ids_set.eos} is out of range for tokenizer {tokenizer_name!r} "
            f"(len {len(tok)}); --marker-set {marker_set} does not match this tokenizer."
        )


def load_examples(patterns: Sequence[str], limit: int = 0) -> List[dict]:
    """
    :param patterns: Paths or globs of unified-format JSONL.
    :param limit: Stop after this many examples; 0 for all.

    :returns: The examples, unwrapped from any ``{"ex": ...}`` envelope.

    :raises FileNotFoundError: If nothing matched.
    """
    paths: List[str] = []
    for pattern in patterns:
        paths.extend(sorted(glob.glob(pattern)) or ([pattern] if os.path.exists(pattern) else []))
    if not paths:
        raise FileNotFoundError(f"no JSONL matched: {list(patterns)}")

    out: List[dict] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                example = json.loads(line)
                if "ex" in example and "documents" not in example:
                    example = example["ex"]
                out.append(example)
                if limit and len(out) >= limit:
                    return out
    return out


# ── the fingerprint, written beside the shards ──────────────────────────────────────────────────


def build_fingerprint(args: argparse.Namespace, ids_set, examples: Sequence[dict]):
    """
    Derive this build's format fingerprint from the options actually used.

    Derived from the resolved arguments rather than declared alongside them: a launcher's
    declaration is a claim about the data, and when the two disagree the launcher is the one that is
    wrong. ``doc_id_range`` is *measured* from the examples being converted, which is the training
    data -- the only honest place to take it from.

    :param args: The parsed converter arguments.
    :param ids_set: The resolved reserved ids.
    :param examples: The examples being converted.

    :returns: The :class:`~ctc.format.fingerprint.FormatFingerprint`.
    """
    from ctc import tasks
    from ctc.format import registry
    from ctc.format.documents import visible_doc_id_range
    from ctc.format.fingerprint import chunk_layout_for

    tasks.load_all()
    spec = registry.get(args.task)

    doc_markers = not args.no_doc_markers
    markers: Optional[Tuple[int, ...]] = None
    if doc_markers:
        markers = (ids_set.doc_start, ids_set.doc_end)
        if args.emit == "landmark":
            markers = markers + (ids_set.landmark, ids_set.pad)

    return spec.fingerprint(
        query_position=args.query_position,
        chunk_layout=chunk_layout_for(args.emit, args.chunk_by, doc_markers),
        marker_token_ids=markers,
        tokenizer=args.tokenizer,
        doc_id_range=visible_doc_id_range(examples, args.task),
        data_paths=tuple(str(Path(p).resolve()) for p in args.input_jsonl),
        notes={
            "provenance": "measured",
            "measured_over": len(examples),
            "measured_from": "the training data being converted",
            "cot_mode": args.cot_mode,
            "use_titles": args.use_titles,
            "emit": args.emit,
        },
    )


# ── tokenization ────────────────────────────────────────────────────────────────────────────────


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
    ids_set=None,
    doc_markers: bool = True,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Tokenize one unified example into a document-chunked instance.

    :param tok: The tokenizer.
    :param example: A unified-format example.
    :param task: Task name.
    :param emit: ``"dense"`` or ``"landmark"``.
    :param query_position: ``"before"``, ``"after"`` or ``"both"``.
    :param cot_mode: Prompt CoT mode.
    :param mem_freq: Landmark spacing; block size is ``mem_freq + 1``.
    :param seq_len: Max instance length; longer instances are dropped.
    :param chunk_by: ``"document"`` or ``"line"``.
    :param item_regex: In ``line`` mode, a line is a document iff this matches.
    :param use_titles: Render document titles. Off by default so a title cannot shortcut the task,
        and it MUST match the eval.
    :param free_pad_repeat: Repeats of the free-pad sentence after the documents.
    :param repeat_doc_text: Repeat each document's text N times inside its chunk.
    :param summary_every_k: Emit a summary span every K documents; 0 is off.
    :param ids_set: Resolved reserved ids.
    :param doc_markers: Emit the boundary tokens. ``False`` renders the *same* prompt with empty
        boundary strings, so a marker/no-marker shard pair differs only by 2 tokens per document.

    :returns: ``(ids, labels_mask)``, or ``None`` to drop the example -- too long for ``seq_len``,
        or the rendered prompt already contains a reserved inserted id, which would make its chunk
        structure ambiguous.
    """
    from olmo_core.data.document_chunk_landmark import (
        DOC_END_STR,
        DOC_START_STR,
        emit_document_chunk_dense,
        emit_document_chunk_landmark,
        segment_prompt_to_chunks,
    )

    segments, ids, _ = segment_prompt_to_chunks(
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
        doc_start_str=DOC_START_STR if doc_markers else "",
        doc_end_str=DOC_END_STR if doc_markers else "",
        free_pad_repeat=free_pad_repeat,
        repeat_doc_text=repeat_doc_text,
        summary_every_k=summary_every_k,
    )
    if any(t in (ids_set.landmark, ids_set.pad) for t in ids):
        return None

    if emit == "dense":
        out_ids, out_mask = emit_document_chunk_dense(segments)
    else:
        out_ids, out_mask = emit_document_chunk_landmark(
            segments, mem_freq=mem_freq, mem_id=ids_set.landmark, pad_id=ids_set.pad
        )

    out_ids.append(ids_set.eos)
    out_mask.append(False)
    if len(out_ids) > seq_len:
        return None
    return np.asarray(out_ids, dtype=TOKEN_DTYPE), np.asarray(out_mask, dtype=MASK_DTYPE)


def _worker_init(tokenizer_name: str, kwargs: Dict[str, Any], ids_set) -> None:
    from transformers import AutoTokenizer

    _W["tok"] = AutoTokenizer.from_pretrained(tokenizer_name)
    _W["kwargs"] = kwargs
    _W["ids_set"] = ids_set


def _worker_tokenize(example: dict):
    return tokenize_example(_W["tok"], example, ids_set=_W["ids_set"], **_W["kwargs"])


def _gold_sidecar_entry(out_ids: np.ndarray, example: dict):
    """
    ``(content_fingerprint, gold chunk indices)`` for one instance, or ``None``.

    Dense layout only: there, a wrapped document's chunk index equals its stream order, so it lines
    up with ``gold_doc_indices``. The landmark layout repacks documents into windows, which breaks
    that alignment -- hence the caller's hard refusal rather than a silently wrong sidecar.

    :param out_ids: The emitted ids.
    :param example: The source example.

    :returns: The entry, or ``None`` when the example has no gold.
    """
    from olmo_core.nn.attention.gold_grad_mask import (
        content_fingerprint,
        gold_chunks_from_gold_doc_indices,
    )

    gold = example.get("gold_doc_indices")
    if not gold:
        return None
    return content_fingerprint(out_ids.tolist()), sorted(gold_chunks_from_gold_doc_indices(gold))


# ── CLI ─────────────────────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """:returns: The converter's argument parser."""
    from olmo_core.data.document_chunk_landmark import RESERVED_IDS

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input-jsonl", nargs="+", required=True)
    p.add_argument("--task", required=True, help="unified task name")
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--emit",
        default="landmark",
        choices=["landmark", "dense"],
        help="'landmark': pack into landmark windows (DocumentLandmarkAttention). "
        "'dense': wrapped tokens only (DocumentChunkedAttention).",
    )
    p.add_argument(
        "--seq-len", type=int, default=4096, help="max instance length; longer is dropped"
    )
    p.add_argument("--mem-freq", type=int, default=63, help="(landmark) block = mem_freq + 1")
    p.add_argument("--query-position", default="both", choices=["before", "after", "both"])
    p.add_argument(
        "--cot-mode",
        default="none",
        help="prompt CoT mode. Only 'none' is supported -- see :func:`check_cot_mode`. The "
        "pre-migration default was 'plan', which this repo's evaluator cannot render.",
    )
    p.add_argument(
        "--chunk-by",
        default="line",
        choices=["document", "line"],
        help="'line': each matching item line is a document (OOLONG only). "
        "'document': each documents[i]. Everything except oolong wants 'document'.",
    )
    p.add_argument(
        "--item-regex",
        default=r"\|\|",
        help=r"(line mode) a line is a document iff this matches. Pass it ESCAPED ('\|\|'): the "
        "bare '||' is an alternation of empty branches and matches every line. Rejected at startup.",
    )
    p.add_argument(
        "--use-titles",
        action="store_true",
        help="render document titles. Off by default so a title cannot shortcut the task. MUST "
        "match the eval.",
    )
    p.add_argument(
        "--free-pad-repeat",
        type=int,
        default=0,
        help="append N free-pad sentences after the documents. These are FREE tokens under every "
        "cross-document mode, so this widens the budget of positions that can compare across "
        "documents. MUST match the eval.",
    )
    p.add_argument(
        "--repeat-doc-text",
        type=int,
        default=1,
        help="repeat each document's text N times inside its chunk -- grows the chunk without "
        "changing the document count, the control for --free-pad-repeat. MUST match the eval.",
    )
    p.add_argument(
        "--summary-every-k",
        type=int,
        default=0,
        help="emit a summary span every K documents, so chunk indices run on a stride of K+1. MUST "
        "match the eval and the model's summary_every_k, or every chunk role is silently rebound.",
    )
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument(
        "--marker-set",
        default="qwen3",
        choices=sorted(RESERVED_IDS),
        help="which tokenizer family's reserved ids to use. MUST match --tokenizer; verified at "
        "startup. Every Qwen3.5 build needs --marker-set qwen3_5.",
    )
    for flag, field in (
        ("--doc-start-id", "doc_start"),
        ("--doc-end-id", "doc_end"),
        ("--eos-token-id", "eos"),
        ("--landmark-id", "landmark"),
        ("--pad-id", "pad"),
    ):
        p.add_argument(flag, type=int, default=None, help=f"override {field} from --marker-set")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument(
        "--num-proc",
        type=int,
        default=0,
        help="tokenizer worker processes; 0 = auto. Output is byte-identical regardless: the "
        "tokenize path has no RNG and results are consumed in input order.",
    )
    p.add_argument("--shard-tokens", type=int, default=20_000_000)
    p.add_argument(
        "--no-doc-markers",
        action="store_true",
        help="render without the boundary tokens -- the plain full-attention baseline. Everything "
        "else is identical, so a marker/no-marker pair differs only by 2 tokens per document.",
    )
    p.add_argument(
        "--emit-gold-sidecar",
        action="store_true",
        help="also write gold_fingerprints.json for gold-document gradient masking. Dense only.",
    )
    p.add_argument(
        "--no-fingerprint",
        action="store_true",
        help="skip writing format_fingerprint.json. Only for a throwaway shard: without it the "
        "eval-side format guard cannot verify anything this shard trains, and an unfingerprinted "
        "checkpoint grades with a warning that is easy to miss.",
    )
    return p


def resolve_ids(args: argparse.Namespace):
    """
    :param args: Parsed arguments.

    :returns: The reserved id set, with any explicit ``--*-id`` overrides applied field by field.
    """
    from olmo_core.data.document_chunk_landmark import reserved_ids

    overrides = {
        "doc_start": args.doc_start_id,
        "doc_end": args.doc_end_id,
        "eos": args.eos_token_id,
        "landmark": args.landmark_id,
        "pad": args.pad_id,
    }
    return reserved_ids(args.marker_set)._replace(
        **{k: v for k, v in overrides.items() if v is not None}
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    :param argv: Argument list; defaults to ``sys.argv[1:]``.

    :returns: Process exit status.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)

    if args.emit == "landmark" and args.seq_len % (args.mem_freq + 1) != 0:
        raise SystemExit(
            f"--seq-len must be a multiple of the block size (mem_freq+1={args.mem_freq + 1}) "
            "for the landmark layout"
        )
    if args.emit_gold_sidecar and args.emit != "dense":
        raise SystemExit("--emit-gold-sidecar requires --emit dense (chunk index == gold id - 1)")
    if args.chunk_by == "line":
        check_item_regex(args.item_regex)
    check_cot_mode(args.cot_mode)

    from transformers import AutoTokenizer

    ids_set = resolve_ids(args)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    check_marker_ids(tok, args.tokenizer, args.marker_set, ids_set)
    log.info(
        f"marker set '{args.marker_set}': doc_start={ids_set.doc_start} doc_end={ids_set.doc_end} "
        f"eos={ids_set.eos} landmark={ids_set.landmark} pad={ids_set.pad}"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    examples = load_examples(args.input_jsonl, args.limit)
    log.info(
        f"loaded {len(examples)} examples (task={args.task}, emit={args.emit}, "
        f"chunk_by={args.chunk_by}); tokenizing -> {out_dir}"
    )

    tok_kwargs: Dict[str, Any] = dict(
        task=args.task,
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
        doc_markers=not args.no_doc_markers,
    )

    n_proc = args.num_proc if args.num_proc > 0 else min(8, os.cpu_count() or 1)
    pool = None
    if n_proc > 1:
        # imap, NOT imap_unordered: shard content, the gold sidecar and the length stats must not
        # depend on worker scheduling.
        pool = mp.Pool(
            n_proc, initializer=_worker_init, initargs=(args.tokenizer, tok_kwargs, ids_set)
        )
        results: Any = pool.imap(_worker_tokenize, examples, chunksize=8)
        log.info(f"tokenizing with {n_proc} worker processes")
    else:
        results = (tokenize_example(tok, ex, ids_set=ids_set, **tok_kwargs) for ex in examples)

    tok_buf: List[np.ndarray] = []
    mask_buf: List[np.ndarray] = []
    lengths: List[int] = []
    gold_table: Dict[str, Any] = {}
    part = kept = dropped = buffered = n_loss_tokens = 0

    def flush() -> None:
        nonlocal part
        if not tok_buf:
            return
        np.concatenate(tok_buf).tofile(str(out_dir / f"token_ids_part_{part:06d}.npy"))
        np.concatenate(mask_buf).tofile(str(out_dir / f"labels_mask_part_{part:06d}.npy"))
        part += 1
        tok_buf.clear()
        mask_buf.clear()

    for example, result in zip(examples, results):
        if result is None:
            dropped += 1
            continue
        ids, mask = result
        if args.emit_gold_sidecar:
            entry = _gold_sidecar_entry(ids, example)
            if entry is not None:
                gold_table[entry[0]] = entry[1]
        tok_buf.append(ids)
        mask_buf.append(mask)
        kept += 1
        lengths.append(len(ids))
        n_loss_tokens += int(np.asarray(mask).sum())
        buffered += len(ids)
        if buffered >= args.shard_tokens:
            flush()
            buffered = 0
    if pool is not None:
        pool.close()
        pool.join()
    flush()

    # metadata.json is a hard requirement of the docchunk trainers: they read num_instances (which
    # drives the mask-mixing anneal) and max_example_len (to refuse a --seq-len that would make
    # PadToLength silently skip long examples). It used to be hand-backfilled per shard, so a shard
    # could carry a wrong num_instances and quietly corrupt the curriculum. Written here so the
    # shard is self-describing by construction.
    meta = {
        "task": args.task,
        "emit": args.emit,
        "cot_mode": args.cot_mode,
        "chunk_by": args.chunk_by,
        # Recorded so a shard's line-wrapping is auditable from metadata alone. Without it, telling
        # a good oolong shard from one built with the bare '||' regex means scanning raw token ids
        # for inter-chunk FREE gaps -- which is how a bad shard sat undetected on disk for weeks.
        "item_regex": args.item_regex if args.chunk_by == "line" else None,
        "doc_markers": not args.no_doc_markers,
        "free_pad_repeat": args.free_pad_repeat,
        "repeat_doc_text": args.repeat_doc_text,
        "summary_every_k": args.summary_every_k,
        "use_titles": args.use_titles,
        "query_position": args.query_position,
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
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info(
        f"metadata.json (num_instances={kept}, max_example_len={meta['max_example_len']}, "
        f"num_loss_tokens={n_loss_tokens})"
    )

    if not args.no_fingerprint:
        from ctc.format.fingerprint import FingerprintSet

        fingerprint = build_fingerprint(args, ids_set, examples)
        path = FingerprintSet([fingerprint]).write(out_dir)
        log.info(
            f"format fingerprint -> {path} (task={fingerprint.task}, "
            f"query_position={fingerprint.query_position}, "
            f"chunk_layout={fingerprint.chunk_layout}, doc_id_range={fingerprint.doc_id_range})"
        )
    else:
        log.warning(
            "no format_fingerprint.json written: a checkpoint trained on this shard cannot have "
            "its eval format verified"
        )

    if args.emit_gold_sidecar:
        (out_dir / "gold_fingerprints.json").write_text(json.dumps(gold_table), encoding="utf-8")
        log.info(f"gold sidecar: {len(gold_table)} examples with gold_doc_indices")
    if lengths:
        arr = np.asarray(lengths)
        log.info(
            f"length tokens: min {arr.min()} / p50 {int(np.percentile(arr, 50))} / "
            f"p90 {int(np.percentile(arr, 90))} / max {arr.max()}"
        )
    log.info(
        f"done: kept {kept}, dropped {dropped} (too long / reserved collision); {part} shard(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
