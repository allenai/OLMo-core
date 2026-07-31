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

        # "normal prefill" reference #1: ordinary dense causal attention (what a non-landmark model
        # runs). SDPA picks its flash kernel for these shapes.
        t, m = _time(
            lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        )
        print(f"{T:>7} {'dense causal (SDPA)':>22} {t:>9.3f} {t*N_LAYERS:>14.1f} {m:>8.2f}")

        # "normal prefill" reference #2: the landmark prefill this model actually uses today.
        t, m = _time(
            lambda: fused_compressive_landmark_attention(
                q, k, v, is_mem, sm_scale=scale, block_size=Lb
            )
        )
        print(f"{T:>7} {'landmark kernel':>22} {t:>9.3f} {t*N_LAYERS:>14.1f} {m:>8.2f}")

        for k_blocks, label in ((max(1, (T // Lb) // 10), "topk 10%"), (8, "topk k=8")):
            t, m = _time(
                lambda kb=k_blocks: landmark_topk_prefill_attention_fast(
                    q,
                    k,
                    v,
                    block_size=Lb,
                    softmax_scale=scale,
                    top_k=kb,
                    compressive=True,
                )
            )
            print(f"{T:>7} {f'fused {label}':>22} {t:>9.3f} {t*N_LAYERS:>14.1f} {m:>8.2f}")

        # What a GATHER-based implementation would have to touch: T/B landmark keys + (k+1)*B
        # content keys per query, vs T/2 for the dense landmark prefill. FLOP-ratio ceiling only --
        # a real kernel gives up some of it to gather overhead and worse tensor-core utilisation.
        n_blk = T // Lb
        for k_blocks in (max(1, n_blk // 10), 8):
            frac = (T / Lb + (k_blocks + 1) * Lb) / (T / 2)
            print(
                f"{T:>7} {f'[ceiling k={k_blocks}]':>22} {'':>9} {'':>14} "
                f"  {1/frac:>5.1f}x fewer key-touches than the landmark kernel"
            )

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
