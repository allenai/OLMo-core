"""How much do NULL rungs help? Three ways to spend the same mean FFN budget.

A router can reach a low mean cost c two very different ways: make the rungs NARROWER, or send more
tokens to a zero-width (null / AdaMoE) rung that does no work at all. The one-off measurement said
adding a null rung to the 6-rung ladder recovered 15% at matched cost -- this sweeps it across
three decades so the shape is visible in log-log.

  narrow-only   [full, w]           p_full = c/2, w shrinks with c. No null rung. (the ladder's way)
  full-or-null  [full, 0]           fraction c on full, the rest do NOTHING. Pure AdaMoE.
  ladder+null   [full, F/16, F/64, 0]  fixed small mid rungs, the null fraction absorbs the budget.

Same mean cost on every curve, so any difference is HOW the budget is spent, not how much.

    python research/kv_route/flexcompute_40x/ffn_null_rung_sweep.py --tokens 32768
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
CS = [
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


def timed(fn, warmup=4, iters=12):
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
    ap.add_argument("--scale", default="4B", choices=["0.8B", "2B", "4B", "9B", "27B"])
    args = ap.parse_args()
    dev = torch.device("cuda")
    T = args.tokens
    fac = {
        "0.8B": TransformerConfig.qwen3_5_0_8B,
        "2B": TransformerConfig.qwen3_5_2B,
        "4B": TransformerConfig.qwen3_5_4B,
        "9B": TransformerConfig.qwen3_5_9B,
        "27B": TransformerConfig.qwen3_5_27B,
    }[args.scale]
    m = fac(vocab_size=VOCAB).build(init_device="meta")
    d = m.d_model
    Fh = list(m.blocks.values())[0].feed_forward.w1.out_features
    del m
    print(
        f"# {torch.cuda.get_device_name(0)} scale={args.scale} d_model={d} hidden={Fh} tokens={T}"
    )

    torch.manual_seed(0)
    x = torch.randn(T, d, device=dev, dtype=torch.bfloat16, requires_grad=True)
    w1 = torch.randn(Fh, d, device=dev, dtype=torch.bfloat16, requires_grad=True)
    w3 = torch.randn(Fh, d, device=dev, dtype=torch.bfloat16, requires_grad=True)
    w2 = torch.randn(d, Fh, device=dev, dtype=torch.bfloat16, requires_grad=True)
    idx = torch.arange(T, device=dev)

    def safe_run(widths, counts):
        """OOM at the widest rung is a real answer at 27B/128k, not a reason to lose the sweep."""
        try:
            return run(widths, counts)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return float("nan"), float("nan")

    def run(widths, counts):
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

    rows = []
    print(f"\n{'target c':>10} | {'narrow-only':>22} | {'full-or-null':>22} | {'ladder+null':>22}")
    print(f"{'':>10} | {'mean c':>8} {'ms':>7} | {'mean c':>8} {'ms':>7} | {'mean c':>8} {'ms':>7}")
    for c in CS:
        rec = {"c": c}
        # narrow-only
        p = min(c / 2, 1.0)
        nf = max(1, int(round(p * T))) if c < 1 else T
        wn = (
            max(8, int(round(max((c - nf / T) / (1 - nf / T), 1e-6) * Fh / 8)) * 8) if c < 1 else Fh
        )
        t, mc = safe_run([Fh] if c >= 1 else [Fh, wn], [T] if c >= 1 else [nf, T - nf])
        rec["narrow_ms"], rec["narrow_c"] = t, mc
        # full-or-null
        nf2 = max(1, int(round(c * T)))
        t2, mc2 = safe_run([Fh] if c >= 1 else [Fh, 0], [T] if c >= 1 else [nf2, T - nf2])
        rec["null_ms"], rec["null_c"] = t2, mc2
        # ladder + null: small fixed mid rungs, null absorbs the rest
        w_mid1, w_mid2 = Fh // 16, Fh // 64
        n1 = max(1, int(round(min(c, 1.0) * 0.10 * T)))
        n2 = max(1, int(round(min(c, 1.0) * 0.20 * T)))
        budget_mid = (n1 * w_mid1 + n2 * w_mid2) / Fh / T
        nf3 = max(1, int(round(max(c - budget_mid, 0.0) * T)))
        rest = T - n1 - n2 - nf3
        if c >= 1 or rest < 0:
            t3, mc3 = safe_run([Fh], [T])
        else:
            t3, mc3 = safe_run([Fh, w_mid1, w_mid2, 0], [nf3, n1, n2, rest])
        rec["ln_ms"], rec["ln_c"] = t3, mc3
        rows.append(rec)
        print(
            f"{c:>10.6f} | {mc:>8.5f} {t:>7.3f} | {mc2:>8.5f} {t2:>7.3f} | {mc3:>8.5f} {t3:>7.3f}"
        )

    json.dump(
        {"tokens": T, "hidden": Fh, "rows": rows},
        open(os.path.join(OUT, f"ffn_null_rung_{args.scale}_{T}.json"), "w"),
        indent=1,
    )
    print(f"\n# wrote ffn_null_rung_{args.scale}_{T}.json")


if __name__ == "__main__":
    main()
