"""Profile + correctness-check the chunked-attention path, to decide whether a
custom kernel beats compiled FlexAttention and where the time goes.

For a representative chunked sequence (free prefix + N docs + free query):
  1. CORRECTNESS: FlexAttention(chunked mask_mod) vs a dense-mask SDPA reference
     built from `build_dense_bool_mask` — report max-abs diff (the parity bar the
     repo's tests use is bf16 noise ~1e-3).
  2. PROFILE: time create_block_mask, the flex kernel, and dense SDPA; report the
     BlockMask sparsity (fraction of 128-blocks actually computed) so we know how
     much room a structural (prefix+block-diagonal) kernel could win.

Usage (GPU): python -m scripts.debug.profile_chunked_kernel --n-docs 60 --doc-len 150
"""
import argparse
import time

import torch
import torch.nn.functional as F

from corpus_reasoning.lib.chunked_attention import (
    AttentionPattern, build_flex_mask_mod, build_dense_bool_mask,
    FREE_CHUNK_ID,
)
from torch.nn.attention.flex_attention import flex_attention, create_block_mask


def make_chunk_ids(prefix, n_docs, doc_len, query, device):
    """[free prefix][n_docs x doc_len][free query] -> (1,S) chunk_ids."""
    ids = [FREE_CHUNK_ID] * prefix
    for d in range(n_docs):
        ids += [d] * doc_len
    ids += [FREE_CHUNK_ID] * query
    return torch.tensor([ids], dtype=torch.int32, device=device)


def cuda_time(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", type=int, default=256)
    ap.add_argument("--n-docs", type=int, default=60)
    ap.add_argument("--doc-len", type=int, default=150)
    ap.add_argument("--query", type=int, default=128)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    dev = "cuda"
    dt = getattr(torch, args.dtype)
    chunk_ids = make_chunk_ids(args.prefix, args.n_docs, args.doc_len, args.query, dev)
    S = chunk_ids.shape[1]
    H, D = args.heads, args.head_dim
    print(f"S={S}  (prefix {args.prefix} + {args.n_docs}x{args.doc_len} docs + query {args.query})  "
          f"H={H} D={D} dtype={args.dtype}")

    torch.manual_seed(0)
    q = torch.randn(1, H, S, D, device=dev, dtype=dt)
    k = torch.randn(1, H, S, D, device=dev, dtype=dt)
    v = torch.randn(1, H, S, D, device=dev, dtype=dt)

    pat = AttentionPattern(name="chunked")
    mask_mod = build_flex_mask_mod(pat, chunk_ids)

    # block mask + sparsity
    t0 = time.perf_counter()
    bm = create_block_mask(mask_mod, B=1, H=None, Q_LEN=S, KV_LEN=S, device=dev, _compile=True)
    torch.cuda.synchronize()
    bm_ms = (time.perf_counter() - t0) * 1e3
    try:
        sparsity = bm.sparsity()  # % of blocks MASKED OUT (skipped)
    except Exception:
        sparsity = float("nan")

    flex_c = torch.compile(flex_attention)
    out_flex = flex_c(q, k, v, block_mask=bm)

    # dense reference (bf16 noise bar)
    dense = build_dense_bool_mask(pat, chunk_ids)  # (1,S,S) bool
    fmask = torch.where(dense, 0.0, float("-inf")).to(dt).unsqueeze(1)  # (1,1,S,S)
    out_ref = F.scaled_dot_product_attention(q, k, v, attn_mask=fmask)

    diff = (out_flex.float() - out_ref.float()).abs()
    print(f"\nCORRECTNESS flex vs dense-SDPA: max|Δ|={diff.max():.2e}  mean|Δ|={diff.mean():.2e}")

    # profile
    bm_rebuild = cuda_time(lambda: create_block_mask(mask_mod, B=1, H=None, Q_LEN=S, KV_LEN=S, device=dev, _compile=True), iters=10)
    flex_ms = cuda_time(lambda: flex_c(q, k, v, block_mask=bm))
    try:
        sdpa_ms = cuda_time(lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=fmask))
    except torch.cuda.OutOfMemoryError:
        sdpa_ms = float("nan")
    print(f"\nPROFILE (ms/forward):")
    print(f"  create_block_mask (rebuild): {bm_rebuild:7.3f}")
    print(f"  flex_attention kernel:       {flex_ms:7.3f}")
    print(f"  dense SDPA (ref):            {sdpa_ms:7.3f}")
    print(f"  BlockMask sparsity (skipped): {sparsity:.1%}")
    # structural ideal: a doc row attends to prefix + own doc; free rows dense-causal
    ideal_kv = args.prefix + args.doc_len  # per doc row (approx)
    print(f"  structural lower-bound kv/doc-row ~{ideal_kv} of {S} "
          f"({ideal_kv/S:.1%}) — the ceiling a prefix+block-diagonal kernel targets")


if __name__ == "__main__":
    main()
