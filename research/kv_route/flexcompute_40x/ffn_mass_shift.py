"""Shift routing mass from wide rungs to cheap ones, and watch the speedup curve.

The narrow-only sweep plateaus at c ~ 1/32 (width ~288): below that, narrowing a rung buys nothing
because a token still reads x, writes a full-width output, and runs a badly-shaped tiny GEMM. So a
ladder whose rungs all sit AT OR ABOVE that knee, plus a null rung to absorb the rest of the
budget, should dominate the real ladder -- which spends four of its six rungs below the knee.

Rung sets compared (all with a null rung last):
  above-knee   [9216, 2304, 576, 288, 0]   every working rung at or above the plateau knee
  real ladder  [9216, 576, 144, 36, 9, 0]  the trained ladder's geometry (1,16,64,256,1024 + null)
  full-or-null [9216, 0]                   the degenerate case, best so far at 19.2x

Mass is swept with a geometric tilt p_i ~ r^i over rung index i. r < 1 concentrates on WIDE rungs
(expensive); r > 1 pushes mass down the ladder toward narrow and null (cheap). Each point reports
the realised mean cost, so the curve is speedup against actual compute, not against the knob.

    python research/kv_route/flexcompute_40x/ffn_mass_shift.py --tokens 32768
"""

import argparse
import json
import os
import time

import torch

from olmo_core.nn.nested_ffn_moe import _NestedLadderFn
from olmo_core.nn.transformer import TransformerConfig

OUT = os.path.dirname(os.path.abspath(__file__))
VOCAB = 248320
RS = [0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 4.5, 7.0, 11.0, 18.0]


def timed(fn, warmup=3, iters=11):
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
    args = ap.parse_args()
    dev = torch.device("cuda")
    T = args.tokens
    m = TransformerConfig.qwen3_5_4B(vocab_size=VOCAB).build(init_device="meta")
    d = m.d_model
    Fh = list(m.blocks.values())[0].feed_forward.w1.out_features
    del m
    print(f"# {torch.cuda.get_device_name(0)} d_model={d} hidden={Fh} tokens={T}")

    torch.manual_seed(0)
    x = torch.randn(T, d, device=dev, dtype=torch.bfloat16, requires_grad=True)
    w1 = torch.randn(Fh, d, device=dev, dtype=torch.bfloat16, requires_grad=True)
    w3 = torch.randn(Fh, d, device=dev, dtype=torch.bfloat16, requires_grad=True)
    w2 = torch.randn(d, Fh, device=dev, dtype=torch.bfloat16, requires_grad=True)
    idx = torch.arange(T, device=dev)

    # The TRAINED ladder is divisors 1,16,64,256,1024,9728 -> the widths below, and it has NO null
    # rung: its cheapest option is a width-1 FFN, not "nothing". An earlier version of this script
    # substituted a null for that width-1 rung, which flattered it.
    SETS = {
        # as trained: six rungs, cheapest option is a width-1 FFN, no null
        "trained ladder": [Fh, 576, 144, 36, 9, 1],
        # ADD a seventh rung: every trained option kept, plus "nothing" as an escape hatch
        "trained + null (7)": [Fh, 576, 144, 36, 9, 1, 0],
        # REPLACE the width-1 rung with null: same rung count, worst option swapped out
        "trained, w1->null": [Fh, 576, 144, 36, 9, 0],
        "above-knee + null": [Fh, 2304, 576, 288, 0],
        "full-or-null": [Fh, 0],
    }

    def measure(widths, counts):
        offs, idxs = 0, []
        for n in counts:
            idxs.append(idx[offs : offs + n])
            offs += n

        def f():
            for t in (x, w1, w3, w2):
                t.grad = None
            _NestedLadderFn.apply(widths, x, w1, w3, w2, *idxs).sum().backward()

        mean = sum(n * w / Fh for w, n in zip(widths, counts)) / T
        return timed(f), mean

    dense_ms, _ = measure([Fh], [T])
    print(f"# dense reference (all tokens full width): {dense_ms:.3f} ms\n")

    out = {"tokens": T, "hidden": Fh, "dense_ms": dense_ms, "sets": {}}
    for name, widths in SETS.items():
        n_r = len(widths)
        print(f"=== {name}   widths {widths}")
        print(f"{'tilt r':>7} {'mean c':>9} {'ms':>8} {'speedup':>8}   mass by rung (%)")
        recs = []
        for r in RS:
            w_ = [r**i for i in range(n_r)]
            s = sum(w_)
            counts = [max(0, int(round(T * wi / s))) for wi in w_]
            counts[-1] = T - sum(counts[:-1])
            if counts[-1] < 0:
                continue
            t, mc = measure(widths, counts)
            mass = " ".join(f"{100*c/T:5.1f}" for c in counts)
            recs.append(
                dict(r=r, mean_c=mc, ms=t, speedup=dense_ms / t, mass=[c / T for c in counts])
            )
            print(f"{r:>7.2f} {mc:>9.5f} {t:>8.3f} {dense_ms/t:>7.2f}x   {mass}")
        out["sets"][name] = dict(widths=widths, rows=recs)
        print()

    json.dump(out, open(os.path.join(OUT, f"ffn_ladder_compare_{T}.json"), "w"), indent=1)
    print(f"# wrote ffn_mass_shift_{T}.json")


if __name__ == "__main__":
    main()
