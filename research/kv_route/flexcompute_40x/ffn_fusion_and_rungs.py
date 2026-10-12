"""Two FFN questions: does fusing w1/w3 help, and what do more rungs (and null rungs) cost?

PART A -- w1/w3 fusion. `FeedForward.forward` is `w2(act(w1(x)) * w3(x))`, so w1 and w3 EACH read
x. Fusing them into one `d_model -> 2*hidden` GEMM reads x once and writes h1|h3 contiguously --
exact, same arithmetic, one less pass over the activations. Unlike the GDN qkv fusion (which left
HBM traffic unchanged and bought nothing), this removes a read of the tensor the floor is made of.

PART B -- rung count, null rungs, and scattered assignment. The earlier "mixed" arm was TWO rungs
with CONTIGUOUS indices, which flatters it twice over: the real ladder has six rungs, and a real
router assigns per token, so `index_select`/`index_copy` hit scattered rows. Each configuration is
run with contiguous and with permuted indices, at matched mean cost, so both effects are priced.

    python research/kv_route/flexcompute_40x/ffn_fusion_and_rungs.py --tokens 32768
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
    args = ap.parse_args()
    dev = torch.device("cuda")
    T = args.tokens

    m = TransformerConfig.qwen3_5_4B(vocab_size=VOCAB).build(init_device="meta")
    d = m.d_model
    Fh = list(m.blocks.values())[0].feed_forward.w1.out_features
    del m
    print(f"# {torch.cuda.get_device_name(0)}  d_model={d} hidden={Fh} tokens={T}")

    torch.manual_seed(0)
    x = torch.randn(T, d, device=dev, dtype=torch.bfloat16, requires_grad=True)
    w1 = torch.randn(Fh, d, device=dev, dtype=torch.bfloat16, requires_grad=True)
    w3 = torch.randn(Fh, d, device=dev, dtype=torch.bfloat16, requires_grad=True)
    w2 = torch.randn(d, Fh, device=dev, dtype=torch.bfloat16, requires_grad=True)
    w13 = torch.randn(2 * Fh, d, device=dev, dtype=torch.bfloat16, requires_grad=True)

    print("\n=== A. w1/w3 fusion (one GEMM d->2*hidden vs two d->hidden) ===")
    print(f"{'width':>7} {'separate ms':>12} {'fused ms':>10} {'speedup':>9}")
    partA = []
    for w in (Fh, Fh // 4, Fh // 16, 576, 144, 40, 8):

        def sep(w=w):
            for t in (x, w1, w3, w2):
                t.grad = None
            h1, h3 = F.linear(x, w1[:w]), F.linear(x, w3[:w])
            F.linear(F.silu(h1) * h3, w2[:, :w]).sum().backward()

        def fus(w=w):
            for t in (x, w13, w2):
                t.grad = None
            h = F.linear(x, torch.cat([w13[:w], w13[Fh : Fh + w]], 0))
            h1, h3 = h[:, :w], h[:, w:]
            F.linear(F.silu(h1) * h3, w2[:, :w]).sum().backward()

        # fused-contiguous variant: the realistic layout, w13 stored as [gate|up] interleaved by rung
        def fus2(w=w):
            for t in (x, w13, w2):
                t.grad = None
            h = F.linear(x, w13[: 2 * w])
            h1, h3 = h[:, :w], h[:, w:]
            F.linear(F.silu(h1) * h3, w2[:, :w]).sum().backward()

        ts, tf = timed(sep), timed(fus2)
        partA.append(dict(width=w, separate_ms=ts, fused_ms=tf))
        print(f"{w:>7} {ts:>12.3f} {tf:>10.3f} {ts/tf:>8.2f}x")

    print("\n=== B. rung count, null rungs, scattered vs contiguous assignment ===")
    perm = torch.randperm(T, device=dev)
    cfgs = {
        "2 rungs": [(Fh, 0.0039), (40, 0.9961)],
        "3 rungs": [(Fh, 0.0039), (144, 0.05), (8, 0.9461)],
        "6 rungs (ladder)": [
            (Fh, 0.0039),
            (1152, 0.01),
            (288, 0.04),
            (72, 0.10),
            (16, 0.30),
            (8, 0.5461),
        ],
        "6 rungs + null": [
            (Fh, 0.0039),
            (1152, 0.01),
            (288, 0.04),
            (72, 0.10),
            (16, 0.30),
            (0, 0.5461),
        ],
        "3 rungs + null 50%": [(Fh, 0.0039), (144, 0.05), (0, 0.9461)],
    }
    print(f"{'config':<20} {'mean c':>8} {'contig ms':>10} {'scatter ms':>11} {'scatter cost':>13}")
    partB = []
    for name, spec in cfgs.items():
        widths = [w for w, _ in spec]
        fracs = [f for _, f in spec]
        counts = [max(0, int(round(f * T))) for f in fracs]
        counts[-1] = T - sum(counts[:-1])
        mean_c = sum(c_ * w / Fh for (w, _), c_ in zip(spec, counts)) / T
        offs, idx_c = 0, []
        for n in counts:
            idx_c.append(torch.arange(offs, offs + n, device=dev))
            offs += n
        idx_s = []
        offs = 0
        for n in counts:
            idx_s.append(perm[offs : offs + n].contiguous())
            offs += n

        def run(idxs):
            def f():
                for t in (x, w1, w3, w2):
                    t.grad = None
                _NestedLadderFn.apply(widths, x, w1, w3, w2, *idxs).sum().backward()

            return f

        tc, tsc = timed(run(idx_c)), timed(run(idx_s))
        partB.append(dict(config=name, mean_c=mean_c, contig_ms=tc, scatter_ms=tsc))
        print(f"{name:<20} {mean_c:>8.5f} {tc:>10.3f} {tsc:>11.3f} {100*(tsc/tc-1):>12.1f}%")

    json.dump(
        {"tokens": T, "A": partA, "B": partB},
        open(os.path.join(OUT, f"ffn_fusion_rungs_{T}.json"), "w"),
        indent=1,
    )
    print(f"\n# wrote ffn_fusion_rungs_{T}.json")


if __name__ == "__main__":
    main()
