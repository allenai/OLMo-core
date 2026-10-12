"""Resolve the GDN wall-clock/FLOP gap: bad accounting, bandwidth, or occupancy?

GDN measures ~7.7-8.7% of counted FLOPs but 15-21% of step wall-clock (~115 TF/s vs the FFN's
~595). Three candidate explanations, and they imply completely different fixes:

  A. ACCOUNTING. `num_flops_per_token` prices the recurrence as `2 * 4 * state_size` -- the
     SEQUENTIAL recurrent form. Training runs `chunk_gated_delta_rule`, whose intra-chunk matmuls
     scale with chunk size C. If the true count is much higher, there is no inefficiency to fix and
     every FLOP-share table we have built is wrong.
  B. BANDWIDTH. If the block moves bytes near HBM peak, low TF/s is expected and the only lever is
     less traffic (e.g. a smaller `expand_v`).
  C. OCCUPANCY/LATENCY. The inter-chunk scan is sequential: T/C steps with limited parallel work
     each. Then neither roofline binds and the lever is more parallelism, not fewer bytes/FLOPs.

Distinguished by measuring the delta-rule kernel ALONE across sequence lengths and pricing it
against both rooflines. If time grows linearly with T while achieved TF/s and GB/s both stay far
below peak, it is C.

    python research/kv_route/flexcompute_40x/resolve_gdn_gap.py
"""

import argparse
import time

import torch

from olmo_core.nn.attention import Attention
from olmo_core.nn.transformer import TransformerConfig

VOCAB = 248320
# H200 SXM: ~990 TFLOP/s bf16 dense, ~4.8 TB/s HBM3e.
PEAK_TF, PEAK_BW = 990.0, 4.8e12


def timed(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="2048,8192,32768")
    ap.add_argument("--chunk", type=int, default=64, help="fla chunk size, for the analytic count")
    args = ap.parse_args()
    dev = torch.device("cuda")

    from olmo_core.nn.attention.flash_linear_attn_api import dispatch_chunk_gated_delta_rule

    m = TransformerConfig.qwen3_5_4B(vocab_size=VOCAB).build(init_device="meta")
    g = [b.attention for b in m.blocks.values() if type(b.attention) is not Attention][0]
    H, DK, DV, C = g.n_v_heads, g.head_k_dim, g.head_v_dim, args.chunk
    state = H * DK * DV
    formula_per_tok = 2 * 4 * state
    # Chunked form, per token per head: intra QK^T (C*DK), (attn)V (C*DV), KK^T for the UT
    # transform (C*DK), w/u recompute (~C*DV), state read Q@S (DK*DV), state write K^T@U (DK*DV).
    chunk_per_tok = 2 * H * (C * DK + C * DV + C * DK + C * DV + DK * DV + DK * DV)
    print(f"# 4B GDN: H={H} DK={DK} DV={DV} chunk={C}")
    print(f"# formula recurrence : {formula_per_tok:>12,} FLOP/token")
    print(
        f"# chunked (analytic) : {chunk_per_tok:>12,} FLOP/token   "
        f"-> {chunk_per_tok/formula_per_tok:.2f}x the formula"
    )
    print(
        f"# total counted/token: {g.num_flops_per_token(8192):>12,}  "
        f"(recurrence is {100*formula_per_tok/g.num_flops_per_token(8192):.1f}% of it)"
    )

    print(
        f"\n{'T':>7} {'ms':>9} {'ms/token':>10} | {'TF/s(chunked)':>14} {'%peak':>6} | "
        f"{'GB/s':>8} {'%peak':>6}"
    )
    for T in [int(x) for x in args.lengths.split(",")]:
        q = torch.randn(1, T, H, DK, device=dev, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(1, T, H, DK, device=dev, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(1, T, H, DV, device=dev, dtype=torch.bfloat16, requires_grad=True)
        gg = torch.rand(1, T, H, device=dev, dtype=torch.float32, requires_grad=True) * -0.1
        beta = torch.rand(1, T, H, device=dev, dtype=torch.bfloat16, requires_grad=True)

        def step():
            # Fresh leaves each iteration: reusing them meant the second backward hit a graph
            # whose saved tensors had already been freed.
            qi, ki, vi = (t.detach().requires_grad_(True) for t in (q, k, v))
            gi, bi = (t.detach().requires_grad_(True) for t in (gg, beta))
            o, _ = dispatch_chunk_gated_delta_rule(
                q=qi,
                k=ki,
                v=vi,
                g=gi,
                beta=bi,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
            )
            o.sum().backward()

        try:
            t = timed(step)
        except Exception as e:  # noqa: BLE001
            print(f"{T:>7}  failed: {type(e).__name__}: {e}")
            continue
        # fwd+bwd ~ 3x the forward FLOPs; traffic = q,k,v,o read+written a few times over
        flops = chunk_per_tok * T * 3 / 2  # chunk_per_tok already has the 2x for MAC
        bytes_moved = 2 * T * H * (DK + DK + DV + DV) * 2 * 3  # in+out, fwd+bwd, bf16
        print(
            f"{T:>7} {t*1e3:>8.3f} {t/T*1e6:>9.4f}us | {flops/t/1e12:>13.1f} "
            f"{100*flops/t/1e12/PEAK_TF:>5.1f}% | {bytes_moved/t/1e9:>7.1f} "
            f"{100*bytes_moved/t/PEAK_BW:>5.1f}%"
        )


if __name__ == "__main__":
    main()
