"""What IS the routed-FFN plateau, and where does mixed-rung routing really bottom out?

Two questions the first linearity sweep could not answer:

  1. WHY the plateau. It is not the router -- `_NestedLadderFn` is timed directly and the router
     lives outside it, and the routing-free `dense-narrow` curve plateaus too. The hypothesis is
     width-INDEPENDENT activation traffic: an FFN of any hidden width still reads x (T x d_model)
     and writes an output (T x d_model). This prices that traffic against HBM peak, and adds an
     `act_only` control (an elementwise pass over x, fwd+bwd) that moves the same activations with
     essentially no compute -- the irreducible floor.
  2. HOW LOW mixed routing goes. The old mix pinned 10% of tokens to the FULL rung, so its mean
     cost could not fall below 0.1. Here the full-rung fraction is p = c/2, so the full rung
     supplies half the budget and the mean reaches c for any c. That is what real routing does.

    python research/kv_route/flexcompute_40x/ffn_floor_probe.py --tokens 32768
"""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from olmo_core.nn.nested_ffn_moe import _NestedLadderFn
from olmo_core.nn.transformer import TransformerConfig

OUT = os.path.dirname(os.path.abspath(__file__))
VOCAB = 248320
PEAK_BW = 4.8e12
FRACS = [
    1.0,
    0.5,
    0.25,
    0.125,
    0.0625,
    0.03125,
    0.015625,
    0.0078125,
    0.00390625,
    0.001953125,
    0.0009765625,
]


def timed(fn, warmup=4, iters=14):
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
    ap.add_argument("--tokens", type=int, default=32768)
    ap.add_argument("--scale", default="4B")
    args = ap.parse_args()
    dev = torch.device("cuda")
    T = args.tokens

    cfg = {"4B": TransformerConfig.qwen3_5_4B, "0.8B": TransformerConfig.qwen3_5_0_8B}[args.scale]
    m = cfg(vocab_size=VOCAB).build(init_device="meta")
    d = m.d_model
    Fh = list(m.blocks.values())[0].feed_forward.w1.out_features
    del m
    print(
        f"# {torch.cuda.get_device_name(0)}  {args.scale}: d_model={d} ffn_hidden={Fh} tokens={T}"
    )

    torch.manual_seed(0)
    w1 = torch.randn(Fh, d, device=dev, dtype=torch.bfloat16, requires_grad=True)
    w3 = torch.randn(Fh, d, device=dev, dtype=torch.bfloat16, requires_grad=True)
    w2 = torch.randn(d, Fh, device=dev, dtype=torch.bfloat16, requires_grad=True)
    x = torch.randn(T, d, device=dev, dtype=torch.bfloat16, requires_grad=True)
    allidx = torch.arange(T, device=dev)

    # Control: move the same activations with no FFN compute at all.
    def act_only():
        x.grad = None
        (x * 1.0001).sum().backward()

    t_act = timed(act_only)
    # x read + out write, fwd; roughly doubled in bwd
    act_bytes = 2 * T * d * 2 * 2
    print(
        f"\nactivation-only control: {t_act:7.3f} ms   "
        f"({act_bytes/t_act*1e3/1e9:.0f} GB/s = {100*act_bytes/(t_act/1e3)/PEAK_BW:.1f}% of HBM peak)"
    )

    rows = []
    print(
        f"\n{'c':>10} {'p_full':>7} {'w_narrow':>9} | {'dense ms':>9} {'ladder ms':>10} "
        f"{'mixed ms':>9} | {'mixed vs full':>13}"
    )
    full_ref = {}
    for c in FRACS:
        w = max(8, int(round(c * Fh / 8)) * 8)

        def dense():
            for t in (w1, w3, w2, x):
                t.grad = None
            h1, h3 = F.linear(x, w1[:w]), F.linear(x, w3[:w])
            F.linear(F.silu(h1) * h3, w2[:, :w]).sum().backward()

        empty = torch.empty(0, dtype=torch.long, device=dev)
        one_w = [Fh] if w == Fh else [Fh, w]
        one_i = (allidx,) if w == Fh else (empty, allidx)

        def ladder():
            for t in (w1, w3, w2, x):
                t.grad = None
            _NestedLadderFn.apply(one_w, x, w1, w3, w2, *one_i).sum().backward()

        # realistic mix: full rung carries HALF the budget -> p = c/2, reaches ANY c
        if c >= 1.0:
            mw, mi, p_full, wn = [Fh], (allidx,), 1.0, Fh
        else:
            p_full = c / 2
            n_full = max(1, int(round(p_full * T)))
            p_full = n_full / T
            wn_frac = max((c - p_full) / (1 - p_full), 1e-6)
            wn = max(8, int(round(wn_frac * Fh / 8)) * 8)
            mw, mi = [Fh, wn], (allidx[:n_full], allidx[n_full:])

        def mixed():
            for t in (w1, w3, w2, x):
                t.grad = None
            _NestedLadderFn.apply(mw, x, w1, w3, w2, *mi).sum().backward()

        td, tl, tm = timed(dense), timed(ladder), timed(mixed)
        if c == 1.0:
            full_ref = dict(dense=td, ladder=tl, mixed=tm)
        rows.append(dict(c=c, p_full=p_full, w_narrow=wn, dense_ms=td, ladder_ms=tl, mixed_ms=tm))
        print(
            f"{c:>10.6f} {p_full:>7.4f} {wn:>9} | {td:>9.3f} {tl:>10.3f} {tm:>9.3f} | "
            f"{full_ref['mixed']/tm:>12.1f}x"
        )

    json.dump(
        {"tokens": T, "d_model": d, "ffn_hidden": Fh, "act_only_ms": t_act, "rows": rows},
        open(os.path.join(OUT, f"ffn_floor_probe_{T}.json"), "w"),
        indent=1,
    )
    print(f"\n# wrote ffn_floor_probe_{T}.json")


if __name__ == "__main__":
    main()
