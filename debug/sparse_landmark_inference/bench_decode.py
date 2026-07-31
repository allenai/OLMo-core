"""
Where does a landmark DECODE step actually spend its time?

Decode dominates long-context landmark eval (~81% of per-example wall clock at 32k), and it is the
part that is top-k only *numerically*: ``_decode_one_eval`` scores the query against the WHOLE KV
cache and then masks the non-selected blocks to zero. This benchmark separates the two costs that
hides:

  A. **constant factor** -- ``repeat_kv`` expands 8 KV heads to 32 with ``expand().reshape()``. A
     reshape of a stride-0 expand cannot be a view, so it materializes a 4x copy of the entire cache
     every step, every layer, before a single score is computed.
  B. **asymptotic** -- attention touches all ``T`` keys instead of the ``T/B + k*B`` a genuinely
     sparse retrieval would.

Reference points: a dense single-query SDPA (what a normal model's decode step costs) and the
memory-bandwidth floor for each variant.

    python debug/sparse_landmark_inference/bench_decode.py
"""

import time

import torch

from olmo_core.nn.attention.landmark import repeat_kv
from olmo_core.nn.attention.landmark_compressive import FastCompressiveLandmarkAttention
from olmo_core.nn.attention.landmark_sparse_decode import sparse_landmark_decode

DEV = "cuda"
DTYPE = torch.bfloat16
H, H_KV, D, Lb = 32, 8, 128, 64
N_LAYERS = 36
BYTES = 2


def _time(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters


def _mk_attn(top_k):
    attn = FastCompressiveLandmarkAttention.__new__(FastCompressiveLandmarkAttention)
    attn.block_size = Lb
    attn.mem_freq = Lb - 1
    attn.softmax_scale = D**-0.5
    attn._eval_decode_mode = "extend_last_block"
    attn._eval_top_k = top_k
    attn._ragged_qpos = None
    attn.nonselected_landmark_mass = 0.1
    return attn


def main():
    print(f"{'T':>7} {'stage':>34} {'ms/layer':>9} {'ms/token(36L)':>14} {'GB/token':>9}")
    for T in [8192, 16384, 32768]:
        n_blocks = T // Lb
        top_k = max(1, n_blocks // 10)
        q = torch.randn(1, H, 1, D, device=DEV, dtype=DTYPE)
        k_cache = torch.randn(1, H_KV, T, D, device=DEV, dtype=DTYPE)
        v_cache = torch.randn(1, H_KV, T, D, device=DEV, dtype=DTYPE)

        kv_gb = 2 * H_KV * T * D * BYTES * N_LAYERS / 2**30
        kv_gb_expanded = 2 * H * T * D * BYTES * N_LAYERS / 2**30

        # --- A: the repeat_kv expansion alone (no attention at all) ---
        t = _time(lambda: (repeat_kv(k_cache, H // H_KV), repeat_kv(v_cache, H // H_KV)))
        print(
            f"{T:>7} {'repeat_kv expansion ONLY':>34} {t*1e3:>9.3f} {t*1e3*N_LAYERS:>14.1f} "
            f"{kv_gb + kv_gb_expanded:>9.2f}"
        )

        kh = repeat_kv(k_cache, H // H_KV)
        vh = repeat_kv(v_cache, H // H_KV)

        # --- dense single-query SDPA on the ALREADY-expanded cache (normal decode, no landmarks) ---
        t = _time(lambda: torch.nn.functional.scaled_dot_product_attention(q, kh, vh))
        print(
            f"{T:>7} {'dense SDPA (pre-expanded)':>34} {t*1e3:>9.3f} {t*1e3*N_LAYERS:>14.1f} "
            f"{kv_gb_expanded:>9.2f}"
        )

        # --- dense SDPA with GQA (no expansion; PyTorch handles the head grouping) ---
        t = _time(
            lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k_cache, v_cache, enable_gqa=True
            )
        )
        print(
            f"{T:>7} {'dense SDPA (GQA, no expand)':>34} {t*1e3:>9.3f} {t*1e3*N_LAYERS:>14.1f} "
            f"{kv_gb:>9.2f}"
        )

        # --- the shipped landmark decode: expansion + masked grouped softmax over all T keys ---
        for tk, label in ((None, "landmark decode DENSE"), (top_k, f"landmark decode top-{top_k}")):
            attn = _mk_attn(tk)
            attn._eval_prompt_len = T

            def _step(a=attn):
                khh = repeat_kv(k_cache, H // H_KV)
                vhh = repeat_kv(v_cache, H // H_KV)
                return a._decode_one(q, khh, vhh, T)

            t = _time(_step, warmup=2, iters=5)
            print(
                f"{T:>7} {f'{label} (as shipped)':>34} {t*1e3:>9.3f} {t*1e3*N_LAYERS:>14.1f} "
                f"{kv_gb + kv_gb_expanded:>9.2f}"
            )

        # --- the sparse implementation: landmark scan + gather of k blocks + local ---
        for tk in (top_k, 16, 8):
            t = _time(
                lambda kk=tk: sparse_landmark_decode(
                    q,
                    k_cache,
                    v_cache,
                    block_size=Lb,
                    softmax_scale=D**-0.5,
                    section_start=(T // Lb) * Lb,
                    total=T,
                    top_k=kk,
                    compressive=True,
                    nonselected_mass=0.1,
                ),
                warmup=3,
                iters=10,
            )
            # per-q-head gather => n_rep duplication on the block reads
            gb = (H_KV * n_blocks + H * (tk * Lb) + H_KV * Lb) * D * BYTES * 2 * N_LAYERS / 2**30
            print(
                f"{T:>7} {f'SPARSE decode k={tk}':>34} {t*1e3:>9.3f} {t*1e3*N_LAYERS:>14.1f} "
                f"{gb:>9.2f}"
            )

        # --- what a genuinely sparse retrieval would have to touch ---
        touched = n_blocks + (top_k + 1) * Lb
        print(
            f"{T:>7} {f'[sparse floor k={top_k}]':>34} {'':>9} {'':>14} "
            f"{kv_gb * touched / T:>9.3f}  ({T/touched:.1f}x fewer key-bytes)"
        )
        del k_cache, v_cache, kh, vh
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
