"""Size the two GDN fusion opportunities before doing surgery on the module.

The kernel profile showed a GDN block spending only 42% of its time in the GEMMs that carry 95% of
its FLOPs. Two structural causes are visible in `GatedDeltaNet.forward`:

  * SIX separate input projections off the same x (w_q 2048, w_k 2048, w_v 4096, w_g 4096, and
    w_a/w_b at 32 outputs each -- the last two are far too small to fill a GPU), where one fused
    2560 -> 12352 GEMM would do.
  * THREE separate causal convs (q, k, v), each its own launch streaming its own tensor.

Both fusions are numerically exact: concatenating independent linear maps, and concatenating
depthwise convs along the channel axis. This measures the ceiling on each WITHOUT touching the
real module, plus what torch.compile alone buys (training enables compile by default, so the eager
profile may overstate the elementwise share).

    python research/kv_route/flexcompute_40x/bench_gdn_fusion.py --tokens 8192
"""

import argparse
import time

import torch
import torch.nn.functional as F

from olmo_core.nn.attention import Attention
from olmo_core.nn.transformer import TransformerConfig

VOCAB = 248320


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
    return ts[len(ts) // 2] * 1e3


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
    d = model.d_model
    x = torch.randn(1, T, d, device=dev, dtype=torch.bfloat16, requires_grad=True)

    # --- 1. projection fusion ceiling -------------------------------------------------
    dims = [
        gdn.w_q.out_features,
        gdn.w_k.out_features,
        gdn.w_v.out_features,
        gdn.w_a.out_features,
        gdn.w_b.out_features,
        gdn.w_g.out_features,
    ]
    ws = [torch.randn(o, d, device=dev, dtype=torch.bfloat16, requires_grad=True) for o in dims]
    wf = torch.randn(sum(dims), d, device=dev, dtype=torch.bfloat16, requires_grad=True)
    xf = x.view(T, d)

    def sep():
        for w in ws:
            w.grad = None
        sum(F.linear(xf, w).sum() for w in ws).backward()

    def fused():
        wf.grad = None
        F.linear(xf, wf).sum().backward()

    t_sep, t_fus = timed(sep), timed(fused)
    print(f"projections  dims={dims} total={sum(dims)}")
    print(f"  6 separate GEMMs : {t_sep:7.3f} ms")
    print(
        f"  1 fused GEMM     : {t_fus:7.3f} ms   -> {t_sep/t_fus:.2f}x, saves {t_sep-t_fus:.3f} ms/block"
    )

    # --- 2. conv fusion ceiling -------------------------------------------------------
    qd, kd, vd = gdn.w_q.out_features, gdn.w_k.out_features, gdn.w_v.out_features
    convs = [gdn.q_conv1d, gdn.k_conv1d, gdn.v_conv1d]
    tq = torch.randn(1, T, qd, device=dev, dtype=torch.bfloat16, requires_grad=True)
    tk = torch.randn(1, T, kd, device=dev, dtype=torch.bfloat16, requires_grad=True)
    tv = torch.randn(1, T, vd, device=dev, dtype=torch.bfloat16, requires_grad=True)
    tcat = torch.randn(1, T, qd + kd + vd, device=dev, dtype=torch.bfloat16, requires_grad=True)

    def conv3():
        for t in (tq, tk, tv):
            t.grad = None
        (convs[0](x=tq).sum() + convs[1](x=tk).sum() + convs[2](x=tv).sum()).backward()

    try:
        t_c3 = timed(conv3)
        # one conv over the concatenated channels, using the widest conv module as a stand-in
        big = (
            type(convs[0])(hidden_size=qd + kd + vd, kernel_size=gdn.conv_size, activation="silu")
            .to(dev)
            .to(torch.bfloat16)
        )

        def conv1():
            tcat.grad = None
            big(x=tcat).sum().backward()

        t_c1 = timed(conv1)
        print(f"\nconvs        channels {qd}+{kd}+{vd} = {qd+kd+vd}")
        print(f"  3 separate convs : {t_c3:7.3f} ms")
        print(
            f"  1 fused conv     : {t_c1:7.3f} ms   -> {t_c3/t_c1:.2f}x, saves {t_c3-t_c1:.3f} ms/block"
        )
    except Exception as e:  # noqa: BLE001
        print(f"\nconv fusion sizing skipped: {type(e).__name__}: {e}")

    # --- 3. what compile alone buys on the whole block --------------------------------
    def blk(mod):
        def run():
            mod.zero_grad(set_to_none=True)
            if x.grad is not None:
                x.grad = None
            mod(x).sum().backward()

        return run

    t_eager = timed(blk(gdn), warmup=3, iters=10)
    try:
        cgdn = torch.compile(gdn)
        t_comp = timed(blk(cgdn), warmup=5, iters=10)
        print("\nwhole GDN block")
        print(f"  eager    : {t_eager:7.3f} ms")
        print(f"  compiled : {t_comp:7.3f} ms   -> {t_eager/t_comp:.2f}x")
    except Exception as e:  # noqa: BLE001
        print(f"\ncompile failed: {type(e).__name__}: {e}")
        print(f"  eager    : {t_eager:7.3f} ms")


if __name__ == "__main__":
    main()
