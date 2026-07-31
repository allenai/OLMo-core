"""
Validation + speed for ``landmark_prefill_sparse`` (the skip-work prefill).

1. ``mode="union"`` must equal the masked fused path (``landmark_prefill_topk``) — it is the same
   per-token selection, just iterated over a candidate list instead of every past block.
2. ``top_k=None`` must equal the ordinary landmark prefill kernel.
3. ``mode="qblock"`` is an approximation; report how far it moves, do not require equality.
4. Speed vs the dense landmark kernel and the masked top-k path.
5. Union statistics on random data (the real-model number is what actually decides "union" mode —
   see ``bench_selection_stats.py``).

    python debug/prefill_topk/test_prefill_sparse.py
"""

import sys
import time

import torch

from olmo_core.nn.attention.landmark_compressive import fused_compressive_landmark_attention
from olmo_core.nn.attention.landmark_prefill_sparse import (
    landmark_topk_prefill_sparse,
    selection_stats,
)
from olmo_core.nn.attention.landmark_prefill_topk import landmark_topk_prefill_attention_fast

DEV = "cuda"
DTYPE = torch.bfloat16
TOL = 3e-2


def _mk(B, H, T, D, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (
        torch.randn(B, H, T, D, generator=g, device=DEV, dtype=DTYPE),
        torch.randn(B, H, T, D, generator=g, device=DEV, dtype=DTYPE),
        torch.randn(B, H, T, D, generator=g, device=DEV, dtype=DTYPE),
    )


def _time(fn, warmup=2, iters=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters


def test_union_equals_masked():
    print("[1] mode='union' == the masked fused top-k path (same per-token selection)")
    ok = True
    for B, H, T, D, Lb in [(1, 8, 1024, 64, 64), (2, 4, 2048, 128, 64)]:
        q, k, v = _mk(B, H, T, D, seed=1)
        scale = D**-0.5
        for compressive in (False, True):
            for top_k in [1, 3, 8, None]:
                ref = landmark_topk_prefill_attention_fast(
                    q, k, v, block_size=Lb, softmax_scale=scale, top_k=top_k,
                    compressive=compressive,
                )
                got = landmark_topk_prefill_sparse(
                    q, k, v, block_size=Lb, softmax_scale=scale, top_k=top_k,
                    compressive=compressive, mode="union",
                )
                err = (got.float() - ref.float()).abs().max().item()
                passed = err <= TOL
                ok &= passed
                print(
                    f"  {'PASS' if passed else 'FAIL'}  T{T} D{D} compressive={compressive} "
                    f"k={top_k}: max_abs_err={err:.3e}"
                )
    return ok


def test_dense_limit():
    print("[2] top_k=None == the ordinary landmark prefill kernel")
    B, H, T, D, Lb = 1, 8, 1024, 64, 64
    q, k, v = _mk(B, H, T, D, seed=2)
    scale = D**-0.5
    is_mem = (torch.arange(T, device=DEV) % Lb) == (Lb - 1)
    ref = fused_compressive_landmark_attention(q, k, v, is_mem, sm_scale=scale, block_size=Lb)
    got = landmark_topk_prefill_sparse(
        q, k, v, block_size=Lb, softmax_scale=scale, top_k=None, compressive=True, mode="union"
    )
    err = (got.float() - ref.float()).abs().max().item()
    print(f"  {'PASS' if err <= TOL else 'FAIL'}  max_abs_err={err:.3e}")
    return err <= TOL


def test_qblock_divergence():
    print("[3] mode='qblock' divergence from exact per-token selection (informational)")
    B, H, T, D, Lb = 1, 8, 2048, 128, 64
    q, k, v = _mk(B, H, T, D, seed=3)
    scale = D**-0.5
    for top_k in (3, 8):
        exact = landmark_topk_prefill_sparse(
            q, k, v, block_size=Lb, softmax_scale=scale, top_k=top_k, compressive=True,
            mode="union",
        )
        approx = landmark_topk_prefill_sparse(
            q, k, v, block_size=Lb, softmax_scale=scale, top_k=top_k, compressive=True,
            mode="qblock",
        )
        d = (approx.float() - exact.float()).abs()
        rel = d.mean().item() / exact.float().abs().mean().item()
        print(f"  k={top_k}: mean_abs_diff={d.mean().item():.4f} (rel {rel:.3f}), max={d.max().item():.3f}")
    return True


def bench():
    print("[4] speed, H=32 D=128 (per layer; x36 for a Qwen3-4B prompt)")
    H, D, Lb = 32, 128, 64
    print(f"{'T':>7} {'path':>28} {'ms/layer':>9} {'s/prompt(36L)':>14}")
    for T in (8192, 16384, 32768):
        q, k, v = _mk(1, H, T, D, seed=4)
        scale = D**-0.5
        is_mem = (torch.arange(T, device=DEV) % Lb) == (Lb - 1)
        n_blocks = T // Lb

        t = _time(
            lambda: fused_compressive_landmark_attention(
                q, k, v, is_mem, sm_scale=scale, block_size=Lb
            )
        )
        print(f"{T:>7} {'dense landmark kernel':>28} {t*1e3:>9.2f} {t*36:>14.2f}")

        for tk in (max(1, n_blocks // 10), 8):
            t = _time(
                lambda kk=tk: landmark_topk_prefill_attention_fast(
                    q, k, v, block_size=Lb, softmax_scale=scale, top_k=kk, compressive=True
                )
            )
            print(f"{T:>7} {f'masked top-k k={tk}':>28} {t*1e3:>9.2f} {t*36:>14.2f}")
            for mode in ("union", "qblock"):
                t = _time(
                    lambda kk=tk, mm=mode: landmark_topk_prefill_sparse(
                        q, k, v, block_size=Lb, softmax_scale=scale, top_k=kk,
                        compressive=True, mode=mm,
                    )
                )
                print(f"{T:>7} {f'SPARSE {mode} k={tk}':>28} {t*1e3:>9.2f} {t*36:>14.2f}")

        st = selection_stats(q, k, block_size=Lb, softmax_scale=scale, top_k=8)
        print(
            f"{T:>7} {'[union stats k=8, RANDOM q/k]':>28}  mean={st['union_mean']:.1f} "
            f"({st['union_over_k']:.1f}x k), covers {st['frac_of_past_blocks']*100:.0f}% of past blocks"
        )
        del q, k, v
        torch.cuda.empty_cache()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        sys.exit("needs a CUDA device")
    ok = True
    ok &= test_union_equals_masked()
    ok &= test_dense_limit()
    ok &= test_qblock_divergence()
    print("ALL PASS" if ok else "FAILURES ABOVE")
    bench()
    sys.exit(0 if ok else 1)
