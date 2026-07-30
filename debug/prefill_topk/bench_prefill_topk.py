"""
Wall-clock + peak-memory of the eager prefill top-k path vs the fused Triton prefill kernel, at the
shapes a Qwen3-4B compressive-landmark contra ladder actually runs (H=32 q-heads, D=128, 36 layers).

    python debug/prefill_topk/bench_prefill_topk.py
"""

import time

import torch

from olmo_core.nn.attention.landmark_compressive import fused_compressive_landmark_attention
from olmo_core.nn.attention.landmark_prefill_topk import (
    landmark_topk_prefill_attention,
    landmark_topk_prefill_attention_fast,
)

DEV = "cuda"
H, D, Lb = 32, 128, 64
N_LAYERS = 36


def _time(fn, warmup=1, iters=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters, torch.cuda.max_memory_allocated() / 2**30


def main():
    print(f"{'T':>7} {'path':>22} {'s/layer':>9} {'s/prompt(36L)':>14} {'peakGB':>8}")
    for T in [2048, 8192, 16384, 32768]:
        q = torch.randn(1, H, T, D, device=DEV, dtype=torch.bfloat16)
        k = torch.randn(1, H, T, D, device=DEV, dtype=torch.bfloat16)
        v = torch.randn(1, H, T, D, device=DEV, dtype=torch.bfloat16)
        is_mem = (torch.arange(T, device=DEV) % Lb) == (Lb - 1)
        scale = D**-0.5

        t, m = _time(
            lambda: fused_compressive_landmark_attention(
                q, k, v, is_mem, sm_scale=scale, block_size=Lb
            )
        )
        print(f"{T:>7} {'fused kernel':>22} {t:>9.3f} {t*N_LAYERS:>14.1f} {m:>8.2f}")

        t, m = _time(
            lambda: landmark_topk_prefill_attention_fast(
                q,
                k,
                v,
                block_size=Lb,
                softmax_scale=scale,
                top_k=max(1, (T // Lb) // 10),
                compressive=True,
            )
        )
        print(f"{T:>7} {'fused topk (alpha=0)':>22} {t:>9.3f} {t*N_LAYERS:>14.1f} {m:>8.2f}")

        for tile in (128, 512):
            try:
                t, m = _time(
                    lambda: landmark_topk_prefill_attention(
                        q,
                        k,
                        v,
                        block_size=Lb,
                        softmax_scale=scale,
                        top_k=max(1, (T // Lb) // 10),
                        compressive=True,
                        nonselected_mass=0.1,
                        query_tile=tile,
                    )
                )
                print(
                    f"{T:>7} {f'eager topk tile={tile}':>22} {t:>9.3f} {t*N_LAYERS:>14.1f} {m:>8.2f}"
                )
            except torch.cuda.OutOfMemoryError:
                print(f"{T:>7} {f'eager topk tile={tile}':>22} {'OOM':>9}")
                torch.cuda.empty_cache()
        del q, k, v
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
