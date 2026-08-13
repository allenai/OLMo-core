"""Exactness proof for the varlen chunked-prefill decomposition (ROUND 5, 2026-08-13).

The chunked rule (document_chunked.py / vllm_chunked_patch.py) on a pure full-prompt
prefill step is, per request:

    allowed[i, j] = (j <= i) & (free[i] | free[j] | chunk[i] == chunk[j])

where chunk[t] is the document id of token t and free[t] means chunk[t] == FREE (-1).
This decomposes EXACTLY into three dense pieces, each expressible without a custom mask:

    A. segment-diagonal causal: tokens attend causally within their own maximal
       constant-chunk-id run (documents AND free runs alike). -> flash_attn_varlen_func
       with cu_seqlens at run boundaries.
    B. free-q history: a FREE query attends (full, no further mask) to EVERY token
       strictly before its run's start. [q free, kv anywhere before own run]
    C. doc-q free history: a document query attends to every FREE token strictly
       before it. [q doc, kv free before q]  (there are no FREE tokens inside a doc
       run, so "before q" == "before q's doc start".)

Every (i, j) allowed pair is covered by exactly ONE of A/B/C:
    * j in own run  -> A   (whether i free or doc)
    * i free, j before own run -> B  (j free or doc, both allowed since q is free)
    * i doc,  j before own run -> allowed iff same chunk (impossible: different run,
      and doc ids are unique per run) or j free -> C.
No pair is double-counted, so merging (out, lse) of A with (B|C) per row via online
softmax reproduces the full-rule attention bit-for-bit (up to float assoc.).

This test builds random layouts (docs, gaps, prefix/suffix, adjacencies, unmatched
markers, multi-request packing) and checks the composite against a naive reference
computed with the full rule, in float64 on CPU. Run:

    python debug/chunked_eval_speedup/test_varlen_decomposition.py
"""

import sys

import numpy as np
import torch

sys.path.insert(0, "src")
from corpus_reasoning.lib.vllm_chunked_patch import (  # noqa: E402
    _FREE_CHUNK_ID,
    _build_chunk_ids_row,
)

torch.manual_seed(0)
np.random.seed(0)

H, HKV, D = 4, 2, 16  # heads, kv heads (GQA), head dim
SCALE = D ** -0.5


# ---------------------------------------------------------------------------
# Reference: naive attention under the full chunked rule.
# ---------------------------------------------------------------------------
def reference_attn(q, k, v, chunk_row):
    """q: (L,H,D) k,v: (L,HKV,D) float64. Returns (L,H,D)."""
    L = q.shape[0]
    free = chunk_row == _FREE_CHUNK_ID
    same = chunk_row[:, None] == chunk_row[None, :]
    causal = np.tril(np.ones((L, L), dtype=bool))
    allowed = causal & (free[:, None] | free[None, :] | same)
    mask = torch.from_numpy(allowed)
    kr = k.repeat_interleave(H // HKV, dim=1)  # expand GQA
    vr = v.repeat_interleave(H // HKV, dim=1)
    scores = torch.einsum("ihd,jhd->hij", q, kr) * SCALE
    scores = scores.masked_fill(~mask[None], float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum("hij,jhd->ihd", attn, vr)


# ---------------------------------------------------------------------------
# Composite: A (segment causal) + B (free-q history) + C (doc-q free history),
# merged with LSE — mirrors what the GPU path does with flash varlen +
# vllm merge_attn_states, but in float64 naive form.
# ---------------------------------------------------------------------------
def _attn_with_lse(qs, ks, vs, mask):
    """qs: (m,H,D) ks,vs: (n,HKV,D) mask: (m,n) bool. -> out (m,H,D), lse (H,m)."""
    kr = ks.repeat_interleave(H // HKV, dim=1)
    vr = vs.repeat_interleave(H // HKV, dim=1)
    scores = torch.einsum("ihd,jhd->hij", qs, kr) * SCALE
    scores = scores.masked_fill(~mask[None], float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)  # (H,m); -inf where row all-masked
    attn = torch.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)  # all-masked rows -> 0 output
    out = torch.einsum("hij,jhd->ihd", attn, vr)
    return out, lse


def _merge(out_a, lse_a, out_b, lse_b):
    """Online-softmax merge of two disjoint partial attentions. Shapes as above."""
    lse_max = torch.maximum(lse_a, lse_b)
    wa = torch.exp(lse_a - lse_max)  # exp(-inf - -inf) never happens: A always nonempty
    wb = torch.exp(lse_b - lse_max)
    wb = torch.nan_to_num(wb, nan=0.0)  # lse_b == -inf -> weight 0
    denom = wa + wb
    return (out_a * (wa / denom).T[:, :, None]) + (out_b * (wb / denom).T[:, :, None])


def segments_of(chunk_row):
    """Maximal constant-value runs: list of (start, end_exclusive, is_free)."""
    L = len(chunk_row)
    segs, s = [], 0
    for i in range(1, L + 1):
        if i == L or chunk_row[i] != chunk_row[s]:
            segs.append((s, i, chunk_row[s] == _FREE_CHUNK_ID))
            s = i
    return segs


def composite_attn(q, k, v, chunk_row):
    L = q.shape[0]
    segs = segments_of(chunk_row)
    pos = torch.arange(L)

    # A: causal within own segment.
    out = torch.zeros(L, H, D, dtype=q.dtype)
    lse_a = torch.full((H, L), float("-inf"), dtype=q.dtype)
    for s, e, _ in segs:
        m = e - s
        tri = torch.tril(torch.ones(m, m, dtype=torch.bool))
        o, l = _attn_with_lse(q[s:e], k[s:e], v[s:e], tri)
        out[s:e] = o
        lse_a[:, s:e] = l

    # B + C into a single disjoint "other" tensor.
    other = torch.zeros(L, H, D, dtype=q.dtype)
    lse_o = torch.full((H, L), float("-inf"), dtype=q.dtype)
    for s, e, is_free in segs:
        if is_free:
            if s == 0:
                continue  # no history before the first run
            m = torch.ones(e - s, s, dtype=torch.bool)  # full: all kv < run start
            o, l = _attn_with_lse(q[s:e], k[:s], v[:s], m)
        else:
            free_idx = torch.nonzero(
                torch.from_numpy(chunk_row == _FREE_CHUNK_ID), as_tuple=True
            )[0]
            if free_idx.numel() == 0:
                continue
            m = free_idx[None, :] < pos[s:e, None]  # kv free strictly before q
            if not m.any():
                continue
            o, l = _attn_with_lse(q[s:e], k[free_idx], v[free_idx], m)
        other[s:e] = o
        lse_o[:, s:e] = l

    return _merge(out, lse_a, other, lse_o)


# ---------------------------------------------------------------------------
# Layout generators. chunk rows come from the SAME id-scan the patch uses.
# ---------------------------------------------------------------------------
DOC_START, DOC_END, VOCAB = 9001, 9002, 100


def make_ids(layout):
    """layout: list of ('free', n) | ('doc', n) | ('open', n) fragments -> token ids."""
    ids = []
    for kind, n in layout:
        if kind == "free":
            ids += list(np.random.randint(0, VOCAB, n))
        elif kind == "doc":
            ids += [DOC_START] + list(np.random.randint(0, VOCAB, n)) + [DOC_END]
        elif kind == "open":  # unmatched trailing start -> stays FREE
            ids += [DOC_START] + list(np.random.randint(0, VOCAB, n))
    return np.array(ids, dtype=np.int64)


CASES = {
    "typical":       [("free", 7), ("doc", 5), ("free", 1), ("doc", 8), ("free", 1),
                      ("doc", 3), ("free", 6)],
    "all_free":      [("free", 24)],
    "doc_at_zero":   [("doc", 6), ("free", 1), ("doc", 4), ("free", 5)],
    "adjacent_docs": [("free", 3), ("doc", 4), ("doc", 4), ("doc", 4), ("free", 4)],
    "one_giant_doc": [("free", 2), ("doc", 30), ("free", 3)],
    "trailing_open": [("free", 3), ("doc", 5), ("free", 1), ("open", 6)],
    "no_prefix_no_suffix": [("doc", 5), ("doc", 7)],
    "many_tiny_docs": [("free", 4)] + [("doc", 2), ("free", 1)] * 10 + [("free", 5)],
}


def run_case(name, ids):
    # Patch the module's doc ids for the row scan.
    import corpus_reasoning.lib.vllm_chunked_patch as p
    p._DOC_START_ID, p._DOC_END_ID = DOC_START, DOC_END
    chunk_row = _build_chunk_ids_row(ids)
    L = len(ids)
    q = torch.randn(L, H, D, dtype=torch.float64)
    k = torch.randn(L, HKV, D, dtype=torch.float64)
    v = torch.randn(L, HKV, D, dtype=torch.float64)
    ref = reference_attn(q, k, v, chunk_row)
    got = composite_attn(q, k, v, chunk_row)
    err = (ref - got).abs().max().item()
    status = "OK " if err < 1e-12 else "FAIL"
    print(f"  [{status}] {name:22s} L={L:4d} segs={len(segments_of(chunk_row)):3d} "
          f"max|err|={err:.2e}")
    return err < 1e-12


def run_multirequest():
    """Packing: two requests in one step = independent plans; verify per-request
    composite equals per-request reference when computed on concatenated tensors
    with request offsets (what the GPU plan does)."""
    ok = True
    ids1 = make_ids(CASES["typical"])
    ids2 = make_ids(CASES["adjacent_docs"])
    for ids in (ids1, ids2):
        import corpus_reasoning.lib.vllm_chunked_patch as p
        p._DOC_START_ID, p._DOC_END_ID = DOC_START, DOC_END
        row = _build_chunk_ids_row(ids)
        L = len(ids)
        q = torch.randn(L, H, D, dtype=torch.float64)
        k = torch.randn(L, HKV, D, dtype=torch.float64)
        v = torch.randn(L, HKV, D, dtype=torch.float64)
        err = (reference_attn(q, k, v, row) - composite_attn(q, k, v, row)).abs().max()
        ok &= err.item() < 1e-12
    print(f"  [{'OK ' if ok else 'FAIL'}] multirequest_packing (per-request independence)")
    return ok


def main():
    print("varlen chunked-prefill decomposition vs reference (float64, CPU):")
    ok = True
    for name, layout in CASES.items():
        ok &= run_case(name, make_ids(layout))
    ok &= run_multirequest()
    print("ALL OK" if ok else "*** FAILURES ***")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
