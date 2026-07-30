"""Probe: can segment_prompt_to_chunks render the SAME prompt with no document markers?

For a clean full-vs-chunked comparison the two shard sets must differ ONLY by the
``<|box_start|>``/``<|box_end|>`` tokens -- same pools, same build_prompt rendering, same chat
template, same tokenizer, same EOS. ``segment_prompt_to_chunks`` takes ``doc_start_str`` /
``doc_end_str`` as parameters, so passing empty strings should give exactly that.

Verifies:
  1. the marker-free stream contains ZERO marker ids,
  2. its length is the chunked length minus ~2 tokens per document,
  3. the label masks agree (same answer span).
"""

import json
import sys

sys.path.insert(0, "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/src")

from transformers import AutoTokenizer

from olmo_core.data.document_chunk_landmark import (
    DOC_END_STR,
    DOC_START_STR,
    emit_document_chunk_dense,
    reserved_ids,
    segment_prompt_to_chunks,
)

TOKENIZER = "/scratch/users/prasann/hf_models/Qwen3.5-4B-Base"
POOL = "/data/prasann/xlong5/pools/contradiction/contra_banded_2k-256k_shard0.jsonl"

ids = reserved_ids("qwen3_5")
tok = AutoTokenizer.from_pretrained(TOKENIZER)

with open(POOL) as f:
    for _ in range(3):
        ex = json.loads(f.readline())

n_docs = len(ex["documents"])
common = dict(
    query_position="both", cot_mode="none", chunk_by="document",
    include_answer=True, use_titles=False,
    doc_start_id=ids.doc_start, doc_end_id=ids.doc_end,
)

seg_m, _, _ = segment_prompt_to_chunks(
    tok, ex, "contradiction", doc_start_str=DOC_START_STR, doc_end_str=DOC_END_STR, **common
)
seg_p, _, _ = segment_prompt_to_chunks(tok, ex, "contradiction", doc_start_str="", doc_end_str="", **common)

mk_ids, mk_mask = emit_document_chunk_dense(seg_m)
pl_ids, pl_mask = emit_document_chunk_dense(seg_p)

n_mk = sum(1 for t in mk_ids if t in (ids.doc_start, ids.doc_end))
n_pl = sum(1 for t in pl_ids if t in (ids.doc_start, ids.doc_end))

print(f"documents            : {n_docs}")
print(f"chunked   len        : {len(mk_ids):>8}  marker_ids={n_mk}")
print(f"markerless len       : {len(pl_ids):>8}  marker_ids={n_pl}")
print(f"delta                : {len(mk_ids) - len(pl_ids)}  (expect ~2*n_docs = {2 * n_docs})")
print(f"answer tokens equal  : {sum(mk_mask)} vs {sum(pl_mask)} -> {sum(mk_mask) == sum(pl_mask)}")
print()
print("VERDICT:", "OK - markerless path is clean" if (n_pl == 0 and n_mk == 2 * n_docs
      and sum(mk_mask) == sum(pl_mask)) else "UNEXPECTED - inspect before using")
