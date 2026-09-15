"""
Sibling of ``repro_xhdr_compaction.py`` for the keep-token knobs: reproduce EXACTLY what the
trainer builds for one outlier row and decode a pooled document's compacted span so a human can
see the framing. CPU only. Covers both selectors:

* ``--st-header-extra-tokens K`` / ``--st-keep-token-rule first --st-keep-token-k K``
  (``mark_doc_headers_free(..., extra_tokens=K)``) -- ``Document [N]: <first K body tokens><SLOT>``.
* ``--st-keep-token-rule rule --st-keep-token-k K`` (``keep_token_scores`` +
  ``mark_doc_topk_tokens_free``) -- the K highest-scoring body tokens by the cheap feature rule,
  kept at their ORIGINAL positions, so the span reads
  ``Document [N]: <scattered real tokens><SLOT>``.

The mechanical assertions live in the CPU pytests
(``src/test/nn/attention/document_chunked_test.py::test_keep_token_*`` and
``src/test/nn/pooled_soft_token_keep_token_test.py``); this script is for eyeballing real data.

Same pipeline as repro_xhdr_compaction.py: segment_prompt_to_chunks + emit_document_chunk_dense ->
build_chunk_ids_from_tokens -> mark_doc_headers_free -> [keep_token_scores +
mark_doc_topk_tokens_free] -> resolve_keep_docs -> compact_pooled_rows, decoded.

    # first-K (unchanged behaviour)
    python debug/ds64/repro_firstk_compaction.py --tokenizer <dir> --input <unified jsonl>
    # the feature rule at K=8, IDF taken over the row itself when no shard is given
    python debug/ds64/repro_firstk_compaction.py --tokenizer <dir> --input <unified jsonl> \\
        --rule rule --extra-tokens 8 [--idf-shard <ds64 shard dir>]
"""

from __future__ import annotations

import argparse
import json
import re
import sys

sys.path.insert(0, "src")

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from olmo_core.data.document_chunk_landmark import (  # noqa: E402
    RESERVED_IDS,
    emit_document_chunk_dense,
    segment_prompt_to_chunks,
)
from olmo_core.nn.attention.chunked_mask import (  # noqa: E402
    build_chunk_ids_from_tokens,
    mark_doc_headers_free,
    mark_doc_topk_tokens_free,
)
from olmo_core.nn.attention.pooled_doc_kv import resolve_keep_docs  # noqa: E402
from olmo_core.nn.pooled_soft_token import (  # noqa: E402
    DEFAULT_KEEP_TOKEN_WEIGHTS,
    KeepTokenTables,
    build_token_idf,
    build_token_piece_tables,
    compact_pooled_rows,
    first_last_scores,
    keep_token_scores,
    parse_keep_token_weights,
)

IGN = -100


def _shard_head_ids(shard_dir, n_rows):
    """``token_ids_part_*.npy`` is a RAW HEADERLESS array despite the extension -- np.load dies."""
    import glob as _glob
    import os

    import numpy as np

    parts = sorted(_glob.glob(f"{shard_dir}/token_ids_part_*.npy"))
    if not parts:
        raise SystemExit(f"{shard_dir} has no token_ids_part_*.npy")
    meta = json.load(open(f"{shard_dir}/metadata.json"))
    dtype = np.dtype(meta.get("dtype") or "uint32")
    n_total = os.path.getsize(parts[0]) // dtype.itemsize
    arr = np.memmap(parts[0], dtype=dtype, mode="r", shape=(n_total,))
    row_len = int(meta.get("max_example_len") or 65536)
    return np.asarray(arr[: min(n_total, max(1, n_rows) * max(1, row_len))], dtype=np.int64)


def render(tok, ex, ids_set, use_titles=False):
    segs, _ids, _m = segment_prompt_to_chunks(
        tok,
        ex,
        "outlier",
        query_position="after",
        cot_mode="none",
        chunk_by="document",
        item_regex=None,
        include_answer=True,
        use_titles=use_titles,
        doc_start_id=ids_set.doc_start,
        doc_end_id=ids_set.doc_end,
    )
    out_ids, out_mask = emit_document_chunk_dense(segs)
    out_ids.append(ids_set.eos)
    out_mask.append(False)
    return out_ids, out_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument(
        "--keep-prob",
        type=float,
        default=0.0,
        help="--st-keep-prob (0.0 = xhdr00/cc00's gold-blind, every doc pooled)",
    )
    ap.add_argument(
        "--stop-id",
        type=int,
        default=5491,
        help="outlier's 'Document [N]:' fused ']:' token; 25 for contradiction/oolong ':'",
    )
    ap.add_argument("--extra-tokens", default="0,8,16,32", help="comma list of K values to compare")
    ap.add_argument(
        "--rule",
        choices=["first", "first_last", "rule"],
        default="first",
        help="which keep-token selector to show: 'first' = --st-header-extra-tokens K (the "
        "original behaviour of this script); 'first_last' = --st-keep-token-rule first_last, the "
        "first K//2 + last K-K//2 body tokens; 'rule' = --st-keep-token-rule rule, the "
        "per-document top-K by the cheap feature score",
    )
    ap.add_argument(
        "--weights",
        default=None,
        help="--st-keep-token-weights (inline JSON or a path); default = the documented "
        "PLACEHOLDER olmo_core.nn.pooled_soft_token.DEFAULT_KEEP_TOKEN_WEIGHTS",
    )
    ap.add_argument(
        "--idf-shard",
        default=None,
        help="ds64 shard dir to take the IDF table's token frequencies from (the trainer uses the "
        "training shard). Without it the frequencies come from THIS ROW, which is enough to see "
        "the rule's shape but is not what a run would use",
    )
    ap.add_argument("--idf-rows", type=int, default=512)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if getattr(tok, "chat_template", None) is None:
        tok.chat_template = (
            "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
            "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        )
    ids_set = RESERVED_IDS["qwen3_5"]

    with open(args.input) as f:
        for i, line in enumerate(f):
            if i == args.row:
                ex = json.loads(line)
                break
    print(f"row {args.row}: {len(ex['documents'])} docs, gold={ex.get('gold_doc_indices')}")

    ids, mask = render(tok, ex, ids_set)
    t = torch.tensor(ids).unsqueeze(0)
    lm = torch.tensor(mask).unsqueeze(0)
    # trainer-style already-shifted labels: label[t] = target for position t, only on answer tokens
    labels = torch.full_like(t, IGN)
    labels[:, :-1] = torch.where(lm[:, 1:], t[:, 1:], torch.full_like(t[:, 1:], IGN))
    print(f"content len {t.shape[1]}, label tokens {(labels != IGN).sum().item()}")

    base_chunk = build_chunk_ids_from_tokens(
        t,
        doc_start_id=ids_set.doc_start,
        doc_end_id=ids_set.doc_end,
        eos_id=ids_set.eos,
        mode="chunked",
    )
    n_docs = int(base_chunk.max().item()) + 1
    print(f"n_docs from chunk ids = {n_docs}")

    tables = weights = sd = None
    if args.rule == "rule":
        vocab = int(max(len(tok), int(t.max()) + 1, ids_set.pad + 1))
        if args.idf_shard:
            freq_ids = _shard_head_ids(args.idf_shard, args.idf_rows)
        else:
            freq_ids = t[0].numpy()
        idf = build_token_idf(freq_ids, vocab)
        sent_end, is_cap, is_dig, tok_len = build_token_piece_tables(
            tok.convert_ids_to_tokens(list(range(vocab)))
        )
        tables = KeepTokenTables(
            idf=torch.from_numpy(idf),
            sent_end=torch.from_numpy(sent_end),
            is_cap=torch.from_numpy(is_cap),
            is_dig=torch.from_numpy(is_dig),
            tok_len=torch.from_numpy(tok_len),
        )
        weights, sd = parse_keep_token_weights(args.weights)
        print(
            f"rule weights={weights} sd={sd or '{}'}"
            + ("" if args.weights else "  (PLACEHOLDER: " + str(DEFAULT_KEEP_TOKEN_WEIGHTS) + ")")
        )
        print(
            f"idf over {len(freq_ids)} tokens of "
            f"{args.idf_shard or 'THIS ROW (not what a run uses -- pass --idf-shard)'}"
        )

    for K in [int(x) for x in args.extra_tokens.split(",")]:
        chunk = mark_doc_headers_free(
            base_chunk,
            t,
            doc_start_id=ids_set.doc_start,
            doc_end_id=ids_set.doc_end,
            stop_id=args.stop_id,
            stop_count=1,
            extra_tokens=K if args.rule == "first" else 0,
            cap=32,
        )
        if args.rule != "first" and K > 0:
            scores = (
                first_last_scores(
                    t,
                    chunk,
                    doc_start_id=ids_set.doc_start,
                    doc_end_id=ids_set.doc_end,
                    n_docs=n_docs,
                )
                if args.rule == "first_last"
                else keep_token_scores(
                    t,
                    chunk,
                    tables=tables,
                    weights=weights,
                    sd=sd,
                    doc_start_id=ids_set.doc_start,
                    doc_end_id=ids_set.doc_end,
                    n_docs=n_docs,
                )
            )
            chunk = mark_doc_topk_tokens_free(
                chunk,
                t,
                scores,
                doc_start_id=ids_set.doc_start,
                doc_end_id=ids_set.doc_end,
                k=K,
                n_docs=n_docs,
            )
        freed = int(((chunk < 0) & (base_chunk >= 0)).sum())
        keep = resolve_keep_docs(chunk, n_docs, holder=None, keep_prob=args.keep_prob, keep_seed=0)
        cb = compact_pooled_rows(
            t,
            labels,
            chunk,
            keep,
            placeholder_id=ids_set.landmark,
            pad_token_id=ids_set.eos,
            ignore_index=IGN,
        )
        T2 = int(cb.row_lens[0].item())
        new_ids = cb.input_ids[0, :T2]
        n_slots = int((new_ids == ids_set.landmark).sum())
        print(
            f"\n--- {args.rule} K={K}: kept_docs={int(keep.sum())}/{n_docs} freed_tokens={freed} "
            f"len {t.shape[1]} -> {T2} (x{t.shape[1]/max(T2,1):.1f})  slots={n_slots}"
        )
        # Decode the whole compacted row with <SLOT>/<BS>/<BE> markers.
        show = []
        for k in range(T2):
            tid = int(new_ids[k])
            show.append(
                "<SLOT>"
                if tid == ids_set.landmark
                else (
                    "<BS>"
                    if tid == ids_set.doc_start
                    else ("<BE>" if tid == ids_set.doc_end else tok.decode([tid]))
                )
            )
        decoded = "".join(show)
        # One real, framed "Document [N]: <K real body tokens><SLOT>" span, for eyeballing.
        m = re.search(r"Document \[\d+\]:[^<]{0,200}?<SLOT>", decoded)
        print(
            "    one framed pooled doc:",
            repr(m.group(0)) if m else "(none found -- try a later row/K)",
        )
        print("    head:", repr(decoded[:500]))


if __name__ == "__main__":
    main()
