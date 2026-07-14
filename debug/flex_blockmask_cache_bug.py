"""Demonstrate the stale-BlockMask-cache bug in DocumentChunkedAttention.

``_get_or_build_block_mask`` keys its single-slot cache on ``(id(cids), cids.data_ptr(),
cids._version, ...)``. But ``chunk_ids`` is rebuilt from the token stream on EVERY forward and the
previous tensor is freed -- so CPython recycles the ``id`` and the CUDA caching allocator hands back
the same ``data_ptr`` for the same-shape tensor. The key then collides and the block mask built for a
PREVIOUS batch is silently reused.

Only bites on the FlexAttention path (seq_len >= _FLEX_MIN_SEQ_LEN = 8192); the dense path rebuilds
the mask every call. Result: every docchunk run at seq_len >= 8192 attends with the wrong example's
document mask -> training flatlines.
"""
import torch
from olmo_core.nn.attention import document_chunked as dc
from olmo_core.nn.attention.chunked_mask import AttentionPattern, build_chunked_mask_mod

T, B = 8192, 1
dev = "cuda" if torch.cuda.is_available() else "cpu"
pat = AttentionPattern(name="chunked")


def make_cids(n_chunks: int) -> torch.Tensor:
    """A fresh (B,T) chunk_ids tensor -- exactly what _prepare_inputs produces each forward."""
    c = torch.full((B, T), -1, dtype=torch.int32, device=dev)
    span = 60
    for i in range(n_chunks):
        c[0, 100 + i * span : 100 + (i + 1) * span] = i
    return c


keys, ptrs = [], []
masks = []
for i, n_chunks in enumerate([100, 5, 100, 3]):
    cids = make_cids(n_chunks)          # fresh tensor, previous one is now unreferenced
    mm = build_chunked_mask_mod(pat, cids)
    key = (id(cids), cids.data_ptr(), cids._version, B, T, 128, "chunked", str(cids.device))
    keys.append(key)
    ptrs.append((id(cids), cids.data_ptr()))
    bm = dc._get_or_build_block_mask(mm, key, B=B, T=T, device=cids.device, block_size=128)
    # num_blocks in the block mask reflects how much is unmasked -> a fingerprint of the mask
    nb = int(bm.kv_num_blocks.sum())
    masks.append(nb)
    print(f"forward {i}: n_chunks={n_chunks:3d}  id={id(cids)}  ptr={cids.data_ptr()}  "
          f"cache_key_seen_before={key in keys[:-1]}  kv_blocks={nb}")
    del cids, mm

print()
print(f"distinct (id, data_ptr) pairs across 4 forwards: {len(set(ptrs))} (1 => keys ALWAYS collide)")
uniq = len(set(masks))
print(f"distinct block masks actually used: {uniq}")
if uniq == 1:
    print("BUG CONFIRMED: every forward reused the FIRST forward's block mask, "
          "even though the documents changed completely.")
else:
    print("masks differed -> cache is behaving")
