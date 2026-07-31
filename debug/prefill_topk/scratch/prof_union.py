"""Split union-mode cost into candidate-building (host torch ops) vs the fused kernel."""
import time

import torch

from olmo_core.nn.attention.landmark_compressive import fused_compressive_landmark_attention
from olmo_core.nn.attention.landmark_prefill_sparse import (
    build_candidates,
    landmark_topk_prefill_sparse,
)

DEV, H, D, Lb = "cuda", 16, 128, 64


def t(f, it=5):
    for _ in range(2):
        f()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(it):
        f()
    torch.cuda.synchronize()
    return (time.time() - t0) / it * 1e3


for T in (16384,):
    q = torch.randn(1, H, T, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(1, H, T, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(1, H, T, D, device=DEV, dtype=torch.bfloat16)
    is_mem = (torch.arange(T, device=DEV) % Lb) == (Lb - 1)
    scale = D**-0.5
    dense = t(lambda: fused_compressive_landmark_attention(q, k, v, is_mem, sm_scale=scale, block_size=Lb))
    print(f"T={T}  dense landmark kernel: {dense:.2f} ms")
    for mode in ("union", "qblock"):
        for kk in (8, 27):
            build = t(lambda: build_candidates(q, k, block_size=Lb, softmax_scale=scale, top_k=kk, mode=mode))
            total = t(lambda: landmark_topk_prefill_sparse(q, k, v, block_size=Lb, softmax_scale=scale, top_k=kk, compressive=True, mode=mode))
            print(f"  {mode:6s} k={kk:<3d} build={build:6.2f} ms  total={total:6.2f} ms  "
                  f"kernel~={total-build:6.2f} ms   build is {100*build/total:.0f}% of total")
