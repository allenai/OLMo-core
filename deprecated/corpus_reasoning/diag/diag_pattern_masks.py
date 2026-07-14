"""Verify the 6 non-standard attention patterns actually produce different
dense masks on a real training example.

If two different patterns produce the same mask (within context-context cells),
that's a code bug — their losses would trivially be similar.
"""

import os, sys
# sys.path hack removed: corpus_reasoning is a package on PYTHONPATH=src
import torch
from transformers import AutoTokenizer

from corpus_reasoning.lib.chunked_attention import (
    setup_tokenizer, AttentionPattern,
    build_dense_bool_mask, PAD_CHUNK_ID, FREE_CHUNK_ID,
)
from corpus_reasoning.train.train_chunked_fast import ChunkedDataset, build_chunk_ids


MODEL = "Qwen/Qwen3.5-0.8B-Base"
DATA = "data/hotpotqa_train_k100_bridge_unified_hn98_2000_cot.jsonl"

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
doc_start_id, doc_end_id = setup_tokenizer(tok)

ds = ChunkedDataset(
    DATA, tok, max_len=16384,
    doc_start_id=doc_start_id, doc_end_id=doc_end_id,
    query_position="both", train_on_inputs=False, task="cot_retrieval",
)
ex = ds[0]
ids = ex["input_ids"]
S = ids.numel()
chunk_id = build_chunk_ids(ids, doc_start_id, doc_end_id).unsqueeze(0)  # (1, S)
num_docs = int((chunk_id[chunk_id >= 0].unique()).numel())
is_free = (chunk_id == FREE_CHUNK_ID)
is_doc = (chunk_id >= 0)

# is_anchor: doc_end positions
is_anchor = torch.zeros_like(chunk_id, dtype=torch.bool)
is_anchor[0] = (ids == doc_end_id)

# doc_random for bigbird
from corpus_reasoning.lib.chunked_attention import build_random_doc_edges
doc_random = build_random_doc_edges(
    num_docs=num_docs, num_edges=2, seed=42, max_docs=num_docs,
).unsqueeze(0)

patterns = [
    AttentionPattern(name="standard"),
    AttentionPattern(name="chunked"),
    AttentionPattern(name="doc_window", doc_window_k=1),
    AttentionPattern(name="doc_window", doc_window_k=4),
    AttentionPattern(name="last_token_anchor"),
    AttentionPattern(name="token_window", token_window_w=512),
    AttentionPattern(name="bigbird", doc_window_k=1, num_random_doc_edges=2, random_seed=42),
]

# Causal only mask — reference for what "all allowed" looks like
causal_bool = torch.tril(torch.ones(S, S, dtype=torch.bool))

masks = {}
for p in patterns:
    kwargs = {}
    if p.needs_anchor_tensor():
        kwargs["is_anchor"] = is_anchor
    if p.needs_random_edges():
        kwargs["doc_random"] = doc_random
    m = build_dense_bool_mask(p, chunk_id, **kwargs)[0]  # (S, S) bool
    masks[p.tag()] = m

# Summary: per pattern, count how many context-context cells (both doc) are
# True (below causal), vs how many "FREE rows attend everywhere causal" (should
# be identical across patterns).
doc_rows = is_doc[0].unsqueeze(1)  # (S,1)
doc_cols = is_doc[0].unsqueeze(0)  # (1,S)
dd_mask = doc_rows & doc_cols & causal_bool  # doc-to-doc causal cells
dd_total = dd_mask.sum().item()
free_rows = is_free[0].unsqueeze(1) & causal_bool
free_rows_total = free_rows.sum().item()

print(f"seq_len={S}, num_docs={num_docs}")
print(f"causal doc-doc cells (possible context-context connections): {dd_total}")
print(f"causal free-row cells (FREE attending to anything non-pad): {free_rows_total}\n")
print(f"{'pattern':25} {'context-context True':>22} {'FREE-row True':>15}")
print("-" * 70)
for tag, m in masks.items():
    cc_true = (m & dd_mask).sum().item()
    fr_true = (m & free_rows).sum().item()
    pct_cc = 100 * cc_true / max(dd_total, 1)
    print(f"{tag:25} {cc_true:>15d} ({pct_cc:5.2f}%)  {fr_true:>15d}")

# Pairwise mask equality check.
print("\nPairwise mask identity (1 = bit-identical):")
tags = list(masks.keys())
print(f"{'':25}  " + "  ".join(f"{t[:10]:>10}" for t in tags))
for t1 in tags:
    row = f"{t1:25}  "
    for t2 in tags:
        eq = torch.equal(masks[t1], masks[t2])
        row += f"  {'1' if eq else '.':>10}"
    print(row)
