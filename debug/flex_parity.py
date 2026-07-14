"""FlexAttention-vs-dense parity for DocumentChunkedAttention on a REAL n=100 chunk_ids tensor at
T>=8192 (the regime training actually uses). Any docchunk run at seq_len >= _FLEX_MIN_SEQ_LEN takes
the flex path, so a mismatch here invalidates those runs.

Checks, in order:
 1. mask_mod vs build_chunked_allowed_mask, elementwise (pure logic -- no CUDA needed for this part)
 2. flex_attention output vs dense-masked SDPA output on real q/k/v (CUDA)
"""
import numpy as np, torch, torch.nn.functional as F
from olmo_core.data import TokenizerConfig
from olmo_core.nn.attention.chunked_mask import (
    AttentionPattern, build_chunk_ids_from_tokens, build_chunked_allowed_mask,
    build_chunked_mask_mod,
)

ROOT = "/scratch/users/prasann/longctx_sft_qwen"
tc = TokenizerConfig.qwen3()
SEQ = 8192
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={dev}")

t = np.asarray(np.memmap(f"{ROOT}/contra_n100_v2_free60/token_ids_part_000000.npy", dtype=np.uint32, mode="r"), dtype=np.int64)
e = int(np.where(t == 151643)[0][0])
ex = t[: e + 1]
inst = np.full(SEQ, tc.pad_token_id, dtype=np.int64)
inst[: len(ex)] = ex
ids = torch.from_numpy(inst).unsqueeze(0).to(dev)
cids = build_chunk_ids_from_tokens(ids, 151648, 151649, tc.eos_token_id)
print(f"example len={len(ex)}  chunks={int(cids.max())+1}")

for pat_name in ["chunked", "standard", "random_doc"]:
    pat = AttentionPattern(name=pat_name, doc_keep_prob=0.5)
    dense = build_chunked_allowed_mask(pat, cids)[0]           # (T,T) bool
    mm = build_chunked_mask_mod(pat, cids)
    if mm is None:
        print(f"{pat_name}: no mask_mod (dense-only)"); continue
    # Evaluate mask_mod elementwise over the full (T,T) grid, exactly as flex would inside a block.
    q = torch.arange(SEQ, device=dev).view(-1, 1).expand(SEQ, SEQ)
    kv = torch.arange(SEQ, device=dev).view(1, -1).expand(SEQ, SEQ)
    b = torch.zeros_like(q)
    h = torch.zeros_like(q)
    modmask = mm(b, h, q, kv)
    same = torch.equal(modmask, dense)
    print(f"{pat_name:10s}: mask_mod == dense allowed-mask ? {same}")
    if not same:
        d = (modmask ^ dense)
        idx = d.nonzero()[:5]
        print(f"    MISMATCH at {int(d.sum())} positions, e.g. {idx.tolist()}")

if dev == "cuda":
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    torch.manual_seed(0)
    B, H, D = 1, 4, 64
    q_ = torch.randn(B, H, SEQ, D, device=dev, dtype=torch.bfloat16)
    k_ = torch.randn(B, H, SEQ, D, device=dev, dtype=torch.bfloat16)
    v_ = torch.randn(B, H, SEQ, D, device=dev, dtype=torch.bfloat16)
    scale = D ** -0.5
    for pat_name in ["chunked", "standard"]:
        pat = AttentionPattern(name=pat_name)
        mm = build_chunked_mask_mod(pat, cids)
        for bs in (128, 64):
            bm = create_block_mask(mm, B, None, SEQ, SEQ, device=dev, BLOCK_SIZE=(bs, bs))
            out_flex = flex_attention(q_, k_, v_, block_mask=bm, scale=scale)
            allowed = build_chunked_allowed_mask(pat, cids)          # (1,T,T)
            bias = torch.where(allowed.unsqueeze(1), 0.0, torch.finfo(torch.bfloat16).min).to(torch.bfloat16)
            out_dense = F.scaled_dot_product_attention(q_, k_, v_, attn_mask=bias, is_causal=False, scale=scale)
            # compare only rows that are not fully-masked pad rows
            real = torch.arange(SEQ, device=dev) < len(ex)
            df = (out_flex - out_dense)[:, :, real].abs().max().item()
            print(f"{pat_name:10s} block={bs:3d}: max|flex - dense| on real rows = {df:.5f}")
