"""
How close can the three routers (FFN width, KV keep, block skip) get to a 40x TRAINING speedup?

Decomposes Qwen3/Qwen3.5 per-token training FLOPs into the five buckets the routers act on, then
prices every combination of router operating points, including the un-routed LM-head term that
sets the Amdahl floor.

    python research/kv_route/flexcompute_40x/ceiling.py
"""

import json
import os

from olmo_core.nn.attention import Attention
from olmo_core.nn.transformer import TransformerConfig

OUT = os.path.dirname(os.path.abspath(__file__))
VOCAB = 248320
SCALES = [
    ("qwen3_5-0.8B", TransformerConfig.qwen3_5_0_8B),
    ("qwen3_5-4B", TransformerConfig.qwen3_5_4B),
    ("qwen3_5-27B", TransformerConfig.qwen3_5_27B),
    ("qwen3-4B", TransformerConfig.qwen3_4B),
]
LENGTHS = [2048, 8192, 16384, 32768, 65536, 131072, 262144]


def buckets(model, L):
    """Per-token training FLOPs split into the five buckets, at context length ``L``."""
    ffn = gdn = proj = score = 0.0
    n_attn = n_gdn = 0
    for b in model.blocks.values():
        ffn += b.feed_forward.num_flops_per_token(L)
        a = b.attention
        if type(a) is Attention:
            p = a.num_flops_per_token(0)
            proj += p
            score += a.num_flops_per_token(L) - p
            n_attn += 1
        else:
            gdn += a.num_flops_per_token(L)
            n_gdn += 1
    head = float(model.lm_head.num_flops_per_token(L))
    return dict(
        ffn=ffn,
        gdn=gdn,
        proj=proj,
        score=score,
        head=head,
        n_attn=n_attn,
        n_gdn=n_gdn,
        total=ffn + gdn + proj + score + head,
    )


def routed_total(b, *, skip_keep=1.0, ffn_cost=1.0, kv_keep=1.0, head_frac=1.0):
    """Per-token FLOPs under a router setting.

    :param skip_keep: fraction of (token, block) pairs actually executed (block-skip router).
    :param ffn_cost: mean FFN width cost as a fraction of the dense FFN (nested-width router).
    :param kv_keep: fraction of keys each query still attends to (KV-keep router).
    :param head_frac: fraction of positions the LM head is computed at (1.0 = today's training).

    A skipped token pays nothing in that block -- FFN, mixer and projections all go away -- so the
    skip factor multiplies everything except the LM head, which lives outside the blocks.
    """
    inside = (
        skip_keep * (ffn_cost * b["ffn"] + b["gdn"] + b["proj"]) + skip_keep * kv_keep * b["score"]
    )
    return inside + head_frac * b["head"]


# Router operating points. "free" and "aggressive" are MEASURED on 4B; "floor" is the mechanism's
# limit, not a point anyone has trained to quality.
POINTS = {
    # name:            (skip_keep, ffn_cost, kv_keep, head_frac)
    "dense": (1.00, 1.000, 1.00, 1.0),
    "ffn-free (L12+ 127x)": (1.00, 0.338, 1.00, 1.0),
    "ffn-all 20x": (1.00, 0.050, 1.00, 1.0),
    "ffn-all 40x": (1.00, 0.025, 1.00, 1.0),
    "kv-0.5": (1.00, 1.000, 0.50, 1.0),
    "3router measured": (0.31, 0.050, 0.50, 1.0),
    "3router floor": (0.11, 0.025, 0.10, 1.0),
    "3router floor +head": (0.11, 0.025, 0.10, 0.02),
    "ALL->0 (Amdahl)": (0.00, 0.000, 0.00, 1.0),
    "ALL->0 +head": (0.00, 0.000, 0.00, 0.0),
}


def main():
    out = {}
    for name, fac in SCALES:
        model = fac(vocab_size=VOCAB).build(init_device="meta")
        per_len = {}
        for L in LENGTHS:
            b = buckets(model, L)
            shares = {k: b[k] / b["total"] for k in ("ffn", "gdn", "proj", "score", "head")}
            speeds = {}
            for pname, (s, c, k, h) in POINTS.items():
                r = routed_total(b, skip_keep=s, ffn_cost=c, kv_keep=k, head_frac=h)
                speeds[pname] = float("inf") if r <= 0 else b["total"] / r
            per_len[L] = dict(shares=shares, speedups=speeds, n_attn=b["n_attn"], n_gdn=b["n_gdn"])
        out[name] = per_len

    with open(os.path.join(OUT, "ceiling.json"), "w") as f:
        json.dump(out, f, indent=1)

    for name in out:
        print(
            f"\n### {name}   (full-attn layers: {out[name][LENGTHS[0]]['n_attn']}, "
            f"GDN: {out[name][LENGTHS[0]]['n_gdn']})"
        )
        print(
            f"{'ctx':>8} | {'FFN':>5} {'GDN':>5} {'proj':>5} {'score':>6} {'head':>5} | "
            + " ".join(f"{p:>21}" for p in POINTS)
        )
        for L in LENGTHS:
            s = out[name][L]["shares"]
            sp = out[name][L]["speedups"]
            print(
                f"{L:>8} | "
                + " ".join(f"{100*s[k]:>4.1f}%" for k in ("ffn", "gdn", "proj"))
                + f" {100*s['score']:>5.1f}% {100*s['head']:>4.1f}% | "
                + " ".join(f"{sp[p]:>20.2f}x" for p in POINTS)
            )


if __name__ == "__main__":
    main()
