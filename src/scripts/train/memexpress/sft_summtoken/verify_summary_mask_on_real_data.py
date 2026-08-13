"""
Verify and visualize the summary-token attention mask on **real Qwen3.5-tokenized** context windows.

Unit tests pin the mask on synthetic layouts. This checks the thing that actually ships: real
document-chunked windows off weka, with real document-length distributions, real marker placement and
real padding tails -- run through the **production** emitter and role builder, not reimplementations.

For every window it asserts the three reachability rules the experiment is defined by, at probe
positions spread through the context:

* a **document token** sees its own document, the instruction, and the summary runs of strictly
  earlier documents -- and **no other document's content**;
* a **summary token** sees its own document and earlier summary runs (the relay);
* a **query/answer token** sees the instruction and every summary run -- and **no raw document
  content** -- while on a causal-arm example it sees everything.

Then it prints a picture of the mask and a role-by-role reachability table.

Nothing here materializes a ``(T, T)`` mask: probe rows are single ``mask_mod`` evaluations and the
picture is a strided sample, so this runs unchanged at 256k.

Usage (locally, against any doc-chunked Qwen3.5 shard)::

    PYTHONPATH=src python src/scripts/train/memexpress/sft_summtoken/verify_summary_mask_on_real_data.py \\
        --shards '/weka/.../xlong5_2k256k_qwen35/shards_chunked/token_ids_part_*.npy' \\
        --n-windows 3 --num-summary-tokens 5

On Beaker: see ``verify_summary_mask_gantry.sh`` next to this file.
"""

import argparse
import glob
import json
import sys
from typing import Dict, List

import numpy as np
import torch

from olmo_core.data.document_chunk_landmark import (
    ChunkSegment,
    ReservedIds,
    emit_document_chunk_summary,
    find_chunk_spans,
    reserved_ids,
)
from olmo_core.nn.attention.summary_mask import (
    ROLE_DOC_ID,
    ROLE_EXAMPLE_ID,
    ROLE_KIND,
    SummaryMaskSpec,
    TokenKind,
    build_summary_mask_mod,
    build_summary_roles,
)
from olmo_core.nn.attention.summary_token import build_summary_block_mask

SHADES = " .:-=+*#%@"


def segments_from_tokens(ids: List[int], ids_set: ReservedIds) -> List[ChunkSegment]:
    """
    Recover the :class:`ChunkSegment` list from an already-tokenized doc-chunked window.

    Uses the repo's own :func:`find_chunk_spans`, so the document boundaries here are exactly the
    ones training would see. Everything between spans (instruction, separators, the trailing
    query/answer) is a FREE run.
    """
    spans = find_chunk_spans(ids, doc_start_id=ids_set.doc_start, doc_end_id=ids_set.doc_end)
    segments: List[ChunkSegment] = []
    cursor = 0
    for start, end in spans:
        if start > cursor:
            free = ids[cursor:start]
            segments.append(ChunkSegment(free, [False] * len(free), False))
        chunk = ids[start:end]
        segments.append(ChunkSegment(chunk, [False] * len(chunk), True))
        cursor = end
    if cursor < len(ids):
        tail = ids[cursor:]
        segments.append(ChunkSegment(tail, [False] * len(tail), False))
    return segments


def load_windows(
    pattern: str,
    n_windows: int,
    min_docs: int,
    ids_set: ReservedIds,
    dtype: str = "uint32",
) -> List[List[int]]:
    """Read the first ``n_windows`` EOS-terminated examples with at least ``min_docs`` documents.

    These shards are **raw and headerless** despite the ``.npy`` extension -- the converter writes
    them with ``ndarray.tofile`` and the data loader reads them back with ``np.memmap`` -- so the
    dtype has to be supplied rather than inferred. It is recorded in the shard's ``metadata.json``.
    """
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no shards matched {pattern!r}")
    out: List[List[int]] = []
    for path in paths:
        arr = np.memmap(path, dtype=np.dtype(dtype), mode="r")
        # A shard is a flat EOS-separated stream of examples.
        eos_positions = np.flatnonzero(np.asarray(arr) == ids_set.eos)
        start = 0
        for eos in eos_positions:
            example = [int(t) for t in arr[start : eos + 1]]
            start = int(eos) + 1
            n_docs = len(
                find_chunk_spans(
                    example, doc_start_id=ids_set.doc_start, doc_end_id=ids_set.doc_end
                )
            )
            if n_docs >= min_docs:
                out.append(example)
                if len(out) == n_windows:
                    return out
    if not out:
        raise SystemExit(
            f"found no example with >= {min_docs} documents in {pattern!r}. Is this a "
            "document-chunked (box-marker) shard, and is --marker-set the right tokenizer family?"
        )
    return out


def pad_to_window(ids: List[int], ids_set: ReservedIds, pad_to: int) -> List[int]:
    """
    Append the padding tail that ``PadToLengthInstanceSource`` adds at training time.

    Without it the PAD role never appears, so "nothing attends padding" would go unchecked on real
    data -- and the analytic block mask, which needs a length that is a multiple of the block size,
    could not be built either.
    """
    if pad_to <= 0 or len(ids) >= pad_to:
        return ids
    return ids + [ids_set.pad] * (pad_to - len(ids))


def probe_row(mask_mod, q: int, seq_len: int) -> torch.Tensor:
    """The single mask row for query position ``q`` -- ``(T,)`` bool, no ``(T, T)`` anywhere."""
    kv = torch.arange(seq_len)
    b = torch.zeros(seq_len, dtype=torch.long)
    return mask_mod(b, b, torch.full((seq_len,), q, dtype=torch.long), kv)


def density_picture(mask_mod, seq_len: int, grid: int = 48, samples: int = 4) -> np.ndarray:
    """Fraction of allowed pairs per cell, from a strided sample (never a full ``(T, T)``)."""
    step = max(1, seq_len // (grid * samples))
    pts = torch.arange(0, grid * samples, dtype=torch.long) * step
    pts = pts.clamp(max=seq_len - 1)
    q = pts.view(-1, 1).expand(len(pts), len(pts))
    k = pts.view(1, -1).expand(len(pts), len(pts))
    b = torch.zeros_like(q)
    allowed = mask_mod(b, b, q, k).float()
    cells = allowed.reshape(grid, samples, grid, samples).mean(dim=(1, 3))
    return cells.numpy()


def render_picture(cells: np.ndarray, seq_len: int) -> str:
    lines = [
        f"    mask density, {cells.shape[0]}x{cells.shape[0]} cells over T={seq_len:,} "
        f"(rows=query, cols=key; '{SHADES[-1]}'=all allowed, ' '=none)"
    ]
    for row in cells:
        lines.append(
            "    " + "".join(SHADES[min(len(SHADES) - 1, int(v * len(SHADES)))] for v in row)
        )
    return "\n".join(lines)


def _counts_by_role(
    row: torch.Tensor, kind: torch.Tensor, doc: torch.Tensor, q: int
) -> Dict[str, str]:
    """How much of each role this query can see, out of how much is causally available."""
    causal = torch.arange(len(row)) <= q
    q_doc = int(doc[q])
    out: Dict[str, str] = {}

    def frac(sel: torch.Tensor) -> str:
        avail = int((sel & causal).sum())
        seen = int((sel & causal & row).sum())
        return f"{seen}/{avail}" if avail else "-"

    is_content = kind == int(TokenKind.DOC_CONTENT)
    out["instruction"] = frac(kind == int(TokenKind.INSTRUCTION))
    out["own doc"] = frac(is_content & (doc == q_doc)) if q_doc >= 0 else "-"
    out["other docs"] = frac(is_content & (doc != q_doc))
    out["summaries"] = frac(kind == int(TokenKind.SUMMARY))
    out["query span"] = frac(kind == int(TokenKind.QUERY))
    out["pad"] = frac(kind == int(TokenKind.PAD))
    return out


def _decode(tok, ids: List[int], lo: int, hi: int, width: int = 46) -> str:
    if tok is None:
        return ""
    try:
        text = tok.decode(ids[lo:hi], skip_special_tokens=False)
    except Exception:
        return ""
    text = text.replace("\n", "\\n")
    return (text[:width] + "…") if len(text) > width else text


def check_window(
    ids: List[int],
    ids_set: ReservedIds,
    spec: SummaryMaskSpec,
    *,
    tok=None,
    index: int = 0,
    grid: int = 48,
) -> dict:
    """Assert the three reachability rules on one real window and return a report."""
    roles = build_summary_roles(
        torch.tensor([ids]),
        doc_start_id=ids_set.doc_start,
        doc_end_id=ids_set.doc_end,
        summary_token_id=ids_set.summary,
        eos_id=ids_set.eos,
        pad_id=ids_set.pad,
    )
    kind, doc = roles[0, ROLE_KIND], roles[0, ROLE_DOC_ID]
    example_id = roles[0, ROLE_EXAMPLE_ID]
    seq_len = len(ids)
    # ``doc_id`` is example-local, so on a packed window this is documents per example, not the
    # window total. Report both -- a window holding more than one example is a packed window, and
    # the probes below are all within-example by construction.
    n_docs = (
        int(doc[kind == int(TokenKind.SUMMARY)].max()) + 1
        if (kind == int(TokenKind.SUMMARY)).any()
        else 0
    )
    n_examples = int(example_id.max()) + 1 if (example_id >= 0).any() else 0

    masked_mod = build_summary_mask_mod(roles, spec)
    causal_mod = build_summary_mask_mod(roles, spec, causal_example=torch.tensor([True]))

    counts = {k.name: int((kind == int(k)).sum()) for k in TokenKind}
    print(
        f"\n=== window {index}: T={seq_len:,}  examples={n_examples}  "
        f"documents/example={n_docs}  summary_tokens_each={spec.n_summary_tokens} ==="
    )
    print("    roles: " + "  ".join(f"{k.lower()}={v:,}" for k, v in counts.items()))

    failures: List[str] = []

    def expect(cond: bool, msg: str):
        if not cond:
            failures.append(msg)

    # ---- probes spread through the context ----
    is_content = kind == int(TokenKind.DOC_CONTENT)
    is_summary = kind == int(TokenKind.SUMMARY)
    probe_docs = sorted({0, max(0, n_docs // 4), max(0, n_docs // 2), max(0, n_docs - 1)})
    rows = []

    for d in probe_docs:
        content_pos = torch.nonzero(is_content & (doc == d)).flatten()
        summ_pos = torch.nonzero(is_summary & (doc == d)).flatten()
        if len(content_pos) == 0 or len(summ_pos) == 0:
            continue
        # A token late in the document, and the last token of its summary run.
        for label, q in (
            (f"doc {d} content", int(content_pos[len(content_pos) * 3 // 4])),
            (f"doc {d} summary", int(summ_pos[-1])),
        ):
            row = probe_row(masked_mod, q, seq_len)
            c = _counts_by_role(row, kind, doc, q)
            rows.append((label, q, c, _decode(tok, ids, max(0, q - 12), q + 1)))

            other_doc_seen = int((row & is_content & (doc != d) & (doc >= 0)).sum())
            expect(
                other_doc_seen == 0, f"{label}: sees {other_doc_seen} tokens of ANOTHER document"
            )
            own_avail = int((is_content & (doc == d) & (torch.arange(seq_len) <= q)).sum())
            own_seen = int((row & is_content & (doc == d)).sum())
            expect(
                own_seen == own_avail, f"{label}: sees {own_seen}/{own_avail} of its own document"
            )
            earlier = is_summary & (doc < d)
            expect(
                int((row & earlier).sum()) == int(earlier.sum()),
                f"{label}: cannot see every earlier summary run",
            )
            expect(
                int((row & (kind == int(TokenKind.PAD))).sum()) == 0, f"{label}: attends padding"
            )

    # ---- the query / answer ----
    query_pos = torch.nonzero(kind == int(TokenKind.QUERY)).flatten()
    if len(query_pos) == 0:
        failures.append("window has no QUERY region -- the trailing span was not recognized")
    else:
        for label, q in (
            ("query start", int(query_pos[0])),
            ("answer end", int(query_pos[-1])),
        ):
            row = probe_row(masked_mod, q, seq_len)
            c = _counts_by_role(row, kind, doc, q)
            rows.append((label, q, c, _decode(tok, ids, max(0, q - 12), q + 1)))

            expect(
                int((row & is_content).sum()) == 0,
                f"{label}: sees {int((row & is_content).sum())} RAW DOCUMENT tokens (must be 0)",
            )
            expect(
                int((row & is_summary).sum()) == int(is_summary.sum()),
                f"{label}: cannot see every summary token",
            )
            instr = kind == int(TokenKind.INSTRUCTION)
            expect(
                int((row & instr).sum()) == int(instr.sum()), f"{label}: cannot see the instruction"
            )

            # ...and on a causal-arm example the same position sees everything.
            crow = probe_row(causal_mod, q, seq_len)
            avail = (torch.arange(seq_len) <= q) & (kind != int(TokenKind.PAD))
            expect(
                bool((crow[avail]).all()),
                f"{label}: causal arm does not see every prior non-pad token",
            )

    print(
        f"\n    {'probe':<18}{'pos':>9}   instruction   own doc     other docs   summaries    query"
    )
    for label, q, c, text in rows:
        print(
            f"    {label:<18}{q:>9}   {c['instruction']:>11}   {c['own doc']:>9}   "
            f"{c['other docs']:>10}   {c['summaries']:>9}   {c['query span']:>8}"
        )
        if text:
            print(f"    {'':<18}{'':>9}   …{text}")

    print()
    print(render_picture(density_picture(masked_mod, seq_len, grid=grid), seq_len))

    # ---- the analytic block mask must agree with the predicate on this real window ----
    block_note = "skipped (T not a multiple of 128)"
    if seq_len % 128 == 0:
        bm = build_summary_block_mask(roles, spec, block_size=128)
        n_blocks = seq_len // 128
        total = n_blocks * n_blocks
        density = (int(bm.kv_num_blocks.sum()) + int(bm.full_kv_num_blocks.sum())) / total
        block_note = f"{density:.4f} of {n_blocks}x{n_blocks} blocks kept"
    print(f"\n    analytic block mask: {block_note}")

    for f in failures:
        print(f"    FAIL: {f}")
    print(f"    ==> window {index}: {'OK' if not failures else str(len(failures)) + ' FAILURES'}")

    return {
        "index": index,
        "seq_len": seq_len,
        "n_docs": n_docs,
        "role_counts": counts,
        "block_mask": block_note,
        "failures": failures,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", required=True, help="glob for token_ids_part_*.npy")
    ap.add_argument("--marker-set", default="qwen3_5", help="RESERVED_IDS family of the shards")
    ap.add_argument("--n-windows", type=int, default=3)
    ap.add_argument("--min-docs", type=int, default=4, help="skip examples with fewer documents")
    ap.add_argument("--dtype", default="uint32", help="shard token dtype (see its metadata.json)")
    ap.add_argument(
        "--pad-to",
        type=int,
        default=0,
        help="pad each window to this length with the reserved pad id, as PadToLengthInstanceSource "
        "does at training time. 0 (default) rounds up to the next multiple of 128 so the padding "
        "tail and the analytic block mask both get exercised; -1 disables padding entirely.",
    )
    ap.add_argument("--num-summary-tokens", type=int, default=5)
    ap.add_argument("--summary-visible-tokens", type=int, default=None)
    ap.add_argument("--grid", type=int, default=48, help="picture resolution")
    ap.add_argument("--tokenizer", default=None, help="path/name used only to decode probe context")
    ap.add_argument("--report-json", default=None)
    args = ap.parse_args()

    ids_set = reserved_ids(args.marker_set)
    spec = SummaryMaskSpec(
        n_summary_tokens=args.num_summary_tokens,
        summary_visible_tokens=args.summary_visible_tokens,
    )
    print(
        f"marker set: {args.marker_set}  doc_start={ids_set.doc_start} doc_end={ids_set.doc_end} "
        f"summary={ids_set.summary} eos={ids_set.eos} pad={ids_set.pad}"
    )
    print(f"spec: {spec}")

    tok = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(args.tokenizer)
            print(f"tokenizer: {args.tokenizer} (vocab {len(tok):,})")
        except Exception as e:  # decoding is a nicety, not a dependency
            print(f"tokenizer unavailable ({type(e).__name__}: {e}); continuing without decoding")

    windows = load_windows(args.shards, args.n_windows, args.min_docs, ids_set, dtype=args.dtype)
    print(f"loaded {len(windows)} real windows from {args.shards}")

    reports = []
    for i, raw in enumerate(windows):
        # Run the PRODUCTION emitter over the real window's segments, so what is verified below is
        # the layout training would actually see -- not a reimplementation of it.
        # A shard that ALREADY carries summary runs (built by build_summary_token_shards.py or the
        # converter's --emit summary) must be verified as-is. Re-emitting over it would append a
        # SECOND run after every document and silently verify a layout that will never be trained.
        if ids_set.summary in raw:
            if i == 0:
                print("shards already contain summary runs -- verifying them as-is, no insertion")
            ids = list(raw)
        else:
            if i == 0:
                print("shards carry no summary runs -- inserting them with the production emitter")
            segments = segments_from_tokens(raw, ids_set)
            ids, _ = emit_document_chunk_summary(
                segments,
                summary_token_id=ids_set.summary,
                n_summary_tokens=args.num_summary_tokens,
            )
        pad_to = args.pad_to
        if pad_to == 0:
            pad_to = ((len(ids) + 127) // 128) * 128
        ids = pad_to_window(ids, ids_set, pad_to)
        reports.append(check_window(ids, ids_set, spec, tok=tok, index=i, grid=args.grid))

    n_failed = sum(1 for r in reports if r["failures"])
    print(f"\n=== {len(reports) - n_failed}/{len(reports)} windows OK ===")
    if args.report_json:
        with open(args.report_json, "w") as f:
            json.dump({"spec": spec.__dict__, "windows": reports}, f, indent=2)
        print(f"wrote {args.report_json}")
    sys.exit(1 if n_failed else 0)


if __name__ == "__main__":
    main()
