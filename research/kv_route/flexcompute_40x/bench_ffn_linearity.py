"""Is routed-FFN wall-clock a LINEAR function of routed-FFN FLOPs?

Sweeps the routed cost fraction c (mean FFN width / full width) and times fwd+bwd of one FFN at
4B geometry. If time = a + b*c with a ~ 0, FLOPs convert one-for-one and an FFN router's speedup
equals its FLOP saving. Any a > 0 is a floor no routing can go below, and it is what turns a 17x
FLOP saving into a 1.05x step.

Three curves separate the causes:

  dense-narrow  a plain FFN whose hidden width IS c*F. The ideal: no routing machinery at all,
                so its intercept is pure GEMM/bandwidth inefficiency at small width.
  ladder-1rung  the real routed kernel with every token on ONE rung of width c*F. Adds the
                ladder's autograd node and per-rung bookkeeping, but no token scatter.
  ladder-mixed  the realistic case: 10% of tokens on the full rung, the rest on a narrow rung
                chosen so the MEAN cost is c. Adds index_select/index_copy and multi-rung
                backward, which is what training actually pays.

    python research/kv_route/flexcompute_40x/bench_ffn_linearity.py --tokens 8192
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
FRACS = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.00390625, 0.0009765625]


def timed(fn, warmup=3, iters=12):
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


def fit_line(xs, ys):
    """Least-squares y = a + b x; returns (intercept, slope, R^2)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx else 0.0
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return a, b, (1 - ss_res / ss_tot if ss_tot else 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--scale", default="4B")
    ap.add_argument("--iters", type=int, default=12)
    args = ap.parse_args()

    device = torch.device("cuda")
    cfg = {"4B": TransformerConfig.qwen3_5_4B, "0.8B": TransformerConfig.qwen3_5_0_8B}[args.scale]
    model = cfg(vocab_size=VOCAB).build(init_device="meta")
    block = list(model.blocks.values())[0]
    d = model.d_model
    Fh = block.feed_forward.w1.out_features
    del model
    print(
        f"# {torch.cuda.get_device_name(0)}  {args.scale}: d_model={d} ffn_hidden={Fh} "
        f"tokens={args.tokens}",
        flush=True,
    )

    torch.manual_seed(0)
    w1 = torch.randn(Fh, d, device=device, dtype=torch.bfloat16, requires_grad=True)
    w3 = torch.randn(Fh, d, device=device, dtype=torch.bfloat16, requires_grad=True)
    w2 = torch.randn(d, Fh, device=device, dtype=torch.bfloat16, requires_grad=True)
    x = torch.randn(args.tokens, d, device=device, dtype=torch.bfloat16, requires_grad=True)
    N = args.tokens
    all_idx = torch.arange(N, device=device)

    rows = []
    for c in FRACS:
        w = max(8, int(round(c * Fh / 8)) * 8)
        c_eff = w / Fh

        def dense_narrow():
            for t in (w1, w3, w2, x):
                t.grad = None
            h1 = F.linear(x, w1[:w])
            h3 = F.linear(x, w3[:w])
            F.linear(F.silu(h1) * h3, w2[:, :w]).sum().backward()

        # widths[0] MUST be the full width: the backward treats it as such and ASSIGNS the
        # weight gradient there. Passing a lone narrow width returns a [w, d] gradient where
        # autograd expects [F, d]. Real routing always carries the full rung, often with no
        # tokens on it -- which is the case measured here (empty full-rung index).
        empty = torch.empty(0, dtype=torch.long, device=device)
        one_widths = [Fh] if w == Fh else [Fh, w]
        one_idxs = (all_idx,) if w == Fh else (empty, all_idx)

        def ladder_1rung():
            for t in (w1, w3, w2, x):
                t.grad = None
            _NestedLadderFn.apply(one_widths, x, w1, w3, w2, *one_idxs).sum().backward()

        # mean cost c = 0.1*1 + 0.9*(w_narrow/Fh)  ->  w_narrow = Fh*(c-0.1)/0.9
        n_full = max(1, N // 10)
        wn = max(8, int(round(max(Fh * (c_eff - 0.1) / 0.9, 8) / 8)) * 8)
        idx_full = all_idx[:n_full]
        idx_rest = all_idx[n_full:]
        mixed_cost = 0.1 + 0.9 * wn / Fh

        mix_widths = [Fh] if wn == Fh else [Fh, wn]
        mix_idxs = (all_idx,) if wn == Fh else (idx_full, idx_rest)

        def ladder_mixed():
            for t in (w1, w3, w2, x):
                t.grad = None
            _NestedLadderFn.apply(mix_widths, x, w1, w3, w2, *mix_idxs).sum().backward()

        row = {
            "c_requested": c,
            "c_eff": c_eff,
            "width": w,
            "dense_narrow_ms": timed(dense_narrow, iters=args.iters) * 1e3,
            "ladder_1rung_ms": timed(ladder_1rung, iters=args.iters) * 1e3,
            "ladder_mixed_ms": timed(ladder_mixed, iters=args.iters) * 1e3,
            "mixed_cost": mixed_cost,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    print("\n# linear fit  time_ms = a + b*c   (a = the floor routing cannot go below)")
    base = {k: rows[0][k] for k in ("dense_narrow_ms", "ladder_1rung_ms", "ladder_mixed_ms")}
    for key in ("dense_narrow_ms", "ladder_1rung_ms"):
        a, b, r2 = fit_line([r["c_eff"] for r in rows], [r[key] for r in rows])
        print(
            f"#   {key:17} a={a:7.3f} ms  b={b:8.3f} ms  R2={r2:.4f}  "
            f"floor={100*a/base[key]:5.1f}% of full  max_speedup={base[key]/a if a>0 else float('inf'):6.1f}x"
        )
    a, b, r2 = fit_line([r["mixed_cost"] for r in rows], [r["ladder_mixed_ms"] for r in rows])
    print(
        f"#   {'ladder_mixed_ms':17} a={a:7.3f} ms  b={b:8.3f} ms  R2={r2:.4f}  "
        f"floor={100*a/base['ladder_mixed_ms']:5.1f}% of full  "
        f"max_speedup={base['ladder_mixed_ms']/a if a>0 else float('inf'):6.1f}x"
    )

    with open(os.path.join(OUT, f"bench_ffn_linearity_{args.scale}_{args.tokens}.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
