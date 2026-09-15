"""
Sibling of ``repro_xhdr_compaction.py`` for the ``--st-header-extra-tokens K`` knob
(``mark_doc_headers_free(..., extra_tokens=K)``): reproduce EXACTLY what the trainer builds for
one outlier row under ``--st-header-stop-id 5491 --st-header-stop-count 1 --st-header-extra-tokens
K``, for a few K values, and decode ONE pooled doc's compacted span so a human can see the
``Document [N]: <K real body tokens> <SLOT>`` framing (see the CPU pytest in
``src/test/nn/attention/document_chunked_test.py::test_mark_doc_headers_free_extra_tokens_*`` for
the mechanical assertions; this script is for eyeballing real data). CPU only.

Same pipeline as repro_xhdr_compaction.py: segment_prompt_to_chunks + emit_document_chunk_dense ->
build_chunk_ids_from_tokens -> mark_doc_headers_free -> resolve_keep_docs -> compact_pooled_rows,
decoded.

    python debug/ds64/repro_firstk_compaction.py --tokenizer <dir> --input <unified jsonl>
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
)
from olmo_core.nn.attention.pooled_doc_kv import resolve_keep_docs  # noqa: E402
from olmo_core.nn.pooled_soft_token import compact_pooled_rows  # noqa: E402

IGN = -100


def render(tok, ex, ids_set, use_titles=False):
    segs, _ids, _m = segment_prompt_to_chunks(
        tok, ex, "outlier", query_position="after", cot_mode="none", chunk_by="document",
        item_regex=None, include_answer=True, use_titles=use_titles,
        doc_start_id=ids_set.doc_start, doc_end_id=ids_set.doc_end,
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
    ap.add_argument("--keep-prob", type=float, default=0.0, help="--st-keep-prob (0.0 = xhdr00/cc00's gold-blind, every doc pooled)")
    ap.add_argument("--stop-id", type=int, default=5491, help="outlier's 'Document [N]:' fused ']:' token; 25 for contradiction/oolong ':'")
    ap.add_argument("--extra-tokens", default="0,8,16,32", help="comma list of K values to compare")
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
        t, doc_start_id=ids_set.doc_start, doc_end_id=ids_set.doc_end, eos_id=ids_set.eos,
        mode="chunked",
    )
    n_docs = int(base_chunk.max().item()) + 1
    print(f"n_docs from chunk ids = {n_docs}")

    for K in [int(x) for x in args.extra_tokens.split(",")]:
        chunk = mark_doc_headers_free(
            base_chunk, t, doc_start_id=ids_set.doc_start, doc_end_id=ids_set.doc_end,
            stop_id=args.stop_id, stop_count=1, extra_tokens=K, cap=32,
        )
        freed = int(((chunk < 0) & (base_chunk >= 0)).sum())
        keep = resolve_keep_docs(chunk, n_docs, holder=None, keep_prob=args.keep_prob, keep_seed=0)
        cb = compact_pooled_rows(
            t, labels, chunk, keep, placeholder_id=ids_set.landmark,
            pad_token_id=ids_set.eos, ignore_index=IGN,
        )
        T2 = int(cb.row_lens[0].item())
        new_ids = cb.input_ids[0, :T2]
        n_slots = int((new_ids == ids_set.landmark).sum())
        print(
            f"\n--- extra_tokens={K}: kept_docs={int(keep.sum())}/{n_docs} freed_tokens={freed} "
            f"len {t.shape[1]} -> {T2} (x{t.shape[1]/max(T2,1):.1f})  slots={n_slots}"
        )
        # Decode the whole compacted row with <SLOT>/<BS>/<BE> markers.
        show = []
        for k in range(T2):
            tid = int(new_ids[k])
            show.append(
                "<SLOT>" if tid == ids_set.landmark
                else ("<BS>" if tid == ids_set.doc_start
                      else ("<BE>" if tid == ids_set.doc_end else tok.decode([tid])))
            )
        decoded = "".join(show)
        # One real, framed "Document [N]: <K real body tokens><SLOT>" span, for eyeballing.
        m = re.search(r"Document \[\d+\]:[^<]{0,200}?<SLOT>", decoded)
        print("    one framed pooled doc:", repr(m.group(0)) if m else "(none found -- try a later row/K)")
        print("    head:", repr(decoded[:500]))


if __name__ == "__main__":
    main()
