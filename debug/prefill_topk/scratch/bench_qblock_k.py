"""qblock prefill cost vs k at Qwen3-4B shapes, against both dense references."""
import time

import torch

from olmo_core.nn.attention.landmark_compressive import fused_compressive_landmark_attention
from olmo_core.nn.attention.landmark_prefill_sparse import landmark_topk_prefill_sparse

DEV, H, D, Lb, NL = "cuda", 32, 128, 64, 36


def t(f, it=5):
    for _ in range(2):
        f()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(it):
        f()
    torch.cuda.synchronize()
    return (time.time() - t0) / it


for T in (16384, 32768):
    q = torch.randn(1, H, T, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(1, H, T, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(1, H, T, D, device=DEV, dtype=torch.bfloat16)
    is_mem = (torch.arange(T, device=DEV) % Lb) == (Lb - 1)
    scale = D**-0.5
    nb = T // Lb
    flash = t(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True))
    dense = t(lambda: fused_compressive_landmark_attention(q, k, v, is_mem, sm_scale=scale, block_size=Lb))
    print(f"\nT={T} ({nb} blocks)   dense flash {flash*NL:.2f} s/prompt | dense landmark {dense*NL:.2f} s/prompt")
    for kk in (8, 16, 32, nb // 10, nb // 4):
        s = t(lambda x=kk: landmark_topk_prefill_sparse(q, k, v, block_size=Lb, softmax_scale=scale,
                                                        top_k=x, compressive=True, mode="qblock"))
        pct = 100 * kk / nb
        print(f"  qblock k={kk:>3d} ({pct:>4.1f}% of blocks): {s*NL:>6.3f} s/prompt   "
              f"{dense/s:>5.1f}x dense-landmark  {flash/s:>5.1f}x flash")
    del q, k, v
    torch.cuda.empty_cache()
