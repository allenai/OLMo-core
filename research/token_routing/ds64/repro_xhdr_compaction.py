"""
Reproduce EXACTLY what the trainer builds for one outlier row under the xhdr* flags
(``--st-gold-blind --st-keep-prob P --st-header-stop-id 5491 --st-header-stop-count 1``) and,
for contrast, under kvgb* (no header). CPU only.

Renders the row with the same call the ds64 shards were tokenized with
(convert_unified_to_document_landmark.py -> segment_prompt_to_chunks + emit_document_chunk_dense),
then runs build_chunk_ids_from_tokens -> mark_doc_headers_free -> resolve_keep_docs ->
compact_pooled_rows and decodes the compacted stream, with the label mask.

    python research/token_routing/ds64/repro_xhdr_compaction.py --tokenizer <dir> --input <unified jsonl>
"""
from __future__ import annotations

import argparse
import json
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
    ap.add_argument("--keep-probs", default="0.0,0.3333")
    ap.add_argument("--stop-id", type=int, default=5491)
    ap.add_argument("--pad-to", type=int, default=0, help="right-pad with EOS to this length (trainer pads to seq_len)")
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
    n = len(ids)
    if args.pad_to and args.pad_to > n:
        ids = ids + [ids_set.eos] * (args.pad_to - n)
        mask = mask + [False] * (args.pad_to - n)
    t = torch.tensor(ids).unsqueeze(0)
    lm = torch.tensor(mask).unsqueeze(0)
    # trainer-style already-shifted labels: label[t] = target for position t, only on answer tokens
    labels = torch.full_like(t, IGN)
    labels[:, :-1] = torch.where(lm[:, 1:], t[:, 1:], torch.full_like(t[:, 1:], IGN))
    print(f"content len {n}, padded len {t.shape[1]}, label tokens {(labels != IGN).sum().item()}")
    print("ANSWER REGION (label-bearing) decode:",
          repr(tok.decode(t[0][(labels[0] != IGN)].tolist()))[:300])

    base_chunk = build_chunk_ids_from_tokens(
        t, doc_start_id=ids_set.doc_start, doc_end_id=ids_set.doc_end, eos_id=ids_set.eos,
        mode="chunked",
    )
    n_docs = int(base_chunk.max().item()) + 1
    print(f"n_docs from chunk ids = {n_docs}")

    for hdr in (False, True):
        chunk = base_chunk
        if hdr:
            chunk = mark_doc_headers_free(
                base_chunk, t, doc_start_id=ids_set.doc_start, doc_end_id=ids_set.doc_end,
                stop_id=args.stop_id, stop_count=1, cap=32,
            )
        freed = int(((chunk < 0) & (base_chunk >= 0)).sum())
        # any header token carrying a LABEL?
        hdr_lab = int(((chunk < 0) & (base_chunk >= 0) & (labels != IGN)).sum())
        for kp in [float(x) for x in args.keep_probs.split(",")]:
            keep = resolve_keep_docs(chunk, n_docs, holder=None, keep_prob=kp, keep_seed=0)
            cb = compact_pooled_rows(
                t, labels, chunk, keep, placeholder_id=ids_set.landmark,
                pad_token_id=ids_set.eos, ignore_index=IGN,
            )
            T2 = int(cb.row_lens[0].item())
            new_ids = cb.input_ids[0, :T2]
            new_lab = cb.labels[0, :T2]
            pos = cb.position_ids[0, :T2]
            nlab = int((new_lab != IGN).sum())
            print(f"\n--- header_real={hdr} keep_prob={kp}: kept_docs={int(keep.sum())}/{n_docs} "
                  f"freed_hdr_tokens={freed} hdr_tokens_with_label={hdr_lab} "
                  f"len {t.shape[1]} -> {T2} (x{t.shape[1]/max(T2,1):.1f})  labels {int((labels!=IGN).sum())} -> {nlab}")
            print("    compacted answer-region decode:",
                  repr(tok.decode(new_ids[new_lab != IGN].tolist()))[:200])
            # first 160 compacted tokens, one string with <SLOT> markers
            show = []
            for k in range(min(160, T2)):
                tid = int(new_ids[k])
                show.append("<SLOT>" if tid == ids_set.landmark else
                            ("<BS>" if tid == ids_set.doc_start else
                             ("<BE>" if tid == ids_set.doc_end else tok.decode([tid]))))
            print("    head:", repr("".join(show))[:900])
            print("    positions head:", pos[:40].tolist())
            print("    positions monotonic:", bool((pos[1:] >= pos[:-1]).all()))
            # tail (answer region)
            tail = []
            for k in range(max(0, T2 - 60), T2):
                tid = int(new_ids[k])
                tail.append("<SLOT>" if tid == ids_set.landmark else tok.decode([tid]))
            print("    tail:", repr("".join(tail))[:500])


if __name__ == "__main__":
    main()
