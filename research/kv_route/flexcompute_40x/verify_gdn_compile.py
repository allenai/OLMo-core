"""Verify the two surprising results from the fusion sizing, with enough warmup to trust them.

1. torch.compile appeared to make the GDN block SLOWER (0.79x). Training enables compile by
   default, so if real this is a regression every run is paying. Needs generous warmup so no
   compilation or recompilation lands inside the timed window.
2. The standalone 3-conv measurement (3.376 ms) is ~5x larger than the share the in-block kernel
   profile attributed to the convs (10.9% of 6.07 ms = 0.66 ms). One of the two is misleading;
   this re-measures the convs in isolation against the FFN as a sanity anchor.

  python research/kv_route/flexcompute_40x/verify_gdn_compile.py --tokens 8192
"""

import argparse
import time

import torch

from olmo_core.nn.attention import Attention
from olmo_core.nn.transformer import TransformerConfig

VOCAB = 248320


def timed(fn, warmup, iters):
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
    return ts[len(ts) // 2] * 1e3, ts[0] * 1e3, ts[-1] * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=8192)
    args = ap.parse_args()
    dev = torch.device("cuda")
    T = args.tokens

    model = TransformerConfig.qwen3_5_4B(vocab_size=VOCAB).build(init_device="cpu").to(dev)
    model.init_weights(max_seq_len=T, device=dev)
    model = model.to(torch.bfloat16)
    gdn = [b.attention for b in model.blocks.values() if type(b.attention) is not Attention][0]
    x = torch.randn(1, T, model.d_model, device=dev, dtype=torch.bfloat16, requires_grad=True)

    def blk(mod):
        def run():
            mod.zero_grad(set_to_none=True)
            if x.grad is not None:
                x.grad = None
            mod(x).sum().backward()

        return run

    med, lo, hi = timed(blk(gdn), warmup=10, iters=30)
    print(f"GDN eager    median {med:7.3f} ms  (p0 {lo:7.3f} / pmax {hi:7.3f})")

    cgdn = torch.compile(gdn)
    med_c, lo_c, hi_c = timed(blk(cgdn), warmup=25, iters=30)
    print(
        f"GDN compiled median {med_c:7.3f} ms  (p0 {lo_c:7.3f} / pmax {hi_c:7.3f})"
        f"  -> {med/med_c:.2f}x vs eager"
    )

    # conv isolation, with the three convs sharing ONE backward (as they do inside the block)
    qd, kd, vd = gdn.w_q.out_features, gdn.w_k.out_features, gdn.w_v.out_features
    tq = torch.randn(1, T, qd, device=dev, dtype=torch.bfloat16, requires_grad=True)
    tk = torch.randn(1, T, kd, device=dev, dtype=torch.bfloat16, requires_grad=True)
    tv = torch.randn(1, T, vd, device=dev, dtype=torch.bfloat16, requires_grad=True)

    def conv3():
        for t in (tq, tk, tv):
            t.grad = None
        o = gdn.q_conv1d(x=tq).sum() + gdn.k_conv1d(x=tk).sum() + gdn.v_conv1d(x=tv).sum()
        o.backward()

    def conv3_fwd_only():
        with torch.no_grad():
            gdn.q_conv1d(x=tq)
            gdn.k_conv1d(x=tk)
            gdn.v_conv1d(x=tv)

    m3, _, _ = timed(conv3, 5, 20)
    mf, _, _ = timed(conv3_fwd_only, 5, 20)
    print(f"\n3 convs fwd+bwd {m3:7.3f} ms   fwd only {mf:7.3f} ms   (bwd ~{m3-mf:.3f} ms)")
    print(
        f"GDN block total {med:7.3f} ms -> convs are {100*m3/med:.1f}% of the block by this measure"
    )


if __name__ == "__main__":
    main()
