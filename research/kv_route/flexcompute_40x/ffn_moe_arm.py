"""A standard top-k MoE priced on the same axis as the nested-ladder arms.

`ffn_null_rung_sweep.py` asks how three *routing* families spend the same mean FFN budget c
(narrow-only / full-or-null / ladder+null). This adds the family everyone else uses: split the
dense FFN's hidden units into E equal experts and activate top-k of them.

The expert family is pinned to the "same FFN parameters" constraint every production MoE uses --
E * W_e = F_hidden -- so an arm here has the *same FFN weight count as the dense model it
replaces*, and its mean compute is exactly c = k / E. That makes it directly comparable to a
router that shrinks or nulls rungs on the dense FFN: same params, same x-axis, same timing
harness, same fwd+bwd.

Then, separately, the arms that matter most: the expert geometry of the MoE models Qwen actually
shipped (`ANCHORS`), dropped into this scale's d_model. Those do NOT hold E * W_e = F_hidden --
real MoEs keep their experts wide (512-1024 units) and let total params grow instead, which is
precisely the thing this page is testing, so they land wherever they land on the x-axis.

Three cost levels per point, because "what does MoE cost" depends on what you count:

  gemm    balanced capacity groups through `torch.bmm` -- literally olmo_core's `MoEMLP.forward`.
          No gate, no permute. This is the apples-to-apples number against the ladder arms, which
          are also handed pinned contiguous index lists. It is also a FLOOR: a perfect MoE kernel
          cannot beat a balanced bmm.
  full    gate + top-k + permute + bmm + unpermute + weighted combine, capacity-based. What a
          step actually pays.
  grouped dropless variable-size groups via `torch._grouped_mm` where available, for the reader
          who objects that capacity padding is the slow part. Skipped if the op is missing.

    python research/kv_route/flexcompute_40x/ffn_moe_arm.py --tokens 32768 --scale 4B
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from olmo_core.nn.transformer import TransformerConfig

OUT = os.path.dirname(os.path.abspath(__file__))
VOCAB = 248320

# Same target grid as ffn_null_rung_sweep.py so the curves share x ticks.
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


# Real shipped Qwen3.5 MoE geometry: (name, num_experts, moe_intermediate_size, num_experts_per_tok).
# Every one also carries ONE always-live shared expert of the same width, so k_live = k + 1.
# Read from each model's config.json on the Hub.
ANCHORS = [
    ("Qwen3.5-35B-A3B", 256, 512, 8),
    ("Qwen3.5-122B-A10B", 256, 1024, 8),
    ("Qwen3.5-397B-A17B", 512, 1024, 10),
]


def timed(fn, warmup=4, iters=12):
    """Identical harness to ffn_null_rung_sweep.timed -- median of `iters` after warmup."""
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


class TopKMoE(torch.nn.Module):
    """Top-k MoE over E experts of width W_e, held to E * W_e == F_hidden."""

    def __init__(self, d: int, hidden: int, n_experts: int, k: int, dev, dtype=torch.bfloat16):
        super().__init__()
        self.d, self.E, self.k = d, n_experts, k
        self.We = hidden // n_experts
        g = torch.Generator(device="cpu").manual_seed(1234)

        def p(*shape):
            t = torch.randn(*shape, generator=g, dtype=torch.float32) * (1.0 / math.sqrt(d))
            return torch.nn.Parameter(t.to(device=dev, dtype=dtype))

        self.wg = p(d, n_experts)
        self.w1 = p(n_experts, d, self.We)
        self.w3 = p(n_experts, d, self.We)
        self.w2 = p(n_experts, self.We, d)

    @classmethod
    def with_width(cls, d: int, We: int, n_experts: int, k: int, dev) -> "TopKMoE":
        """Experts of a GIVEN width -- total FFN params are whatever that implies, as in a real MoE."""
        return cls(d, We * n_experts, n_experts, k, dev)

    def gemm_only(self, xs: torch.Tensor) -> torch.Tensor:
        """xs: (E, cap, d) -- the balanced-capacity bmm path, == olmo_core MoEMLP.forward."""
        return torch.bmm(F.silu(torch.bmm(xs, self.w1)) * torch.bmm(xs, self.w3), self.w2)

    def full(self, x: torch.Tensor) -> torch.Tensor:
        """Gate, permute to capacity slots, bmm, unpermute, combine."""
        T = x.shape[0]
        cap = int(math.ceil(T * self.k / self.E))
        logits = x @ self.wg
        weights, sel = torch.topk(logits.float(), self.k, dim=-1)  # (T, k)
        weights = weights.softmax(dim=-1).to(x.dtype)

        flat_e = sel.reshape(-1)  # (T*k,)
        tok = torch.arange(T, device=x.device).repeat_interleave(self.k)
        order = torch.argsort(flat_e, stable=True)
        e_sorted, tok_sorted = flat_e[order], tok[order]
        counts = torch.bincount(flat_e, minlength=self.E)
        starts = torch.cumsum(counts, 0) - counts
        rank = torch.arange(flat_e.numel(), device=x.device) - starts[e_sorted]
        keep = rank < cap  # capacity drop, as in any capacity-based MoE

        slot = e_sorted * cap + rank.clamp(max=cap - 1)
        gather = torch.zeros(self.E * cap, dtype=torch.long, device=x.device)
        gather[slot[keep]] = tok_sorted[keep]
        xs = x.index_select(0, gather).view(self.E, cap, self.d)

        ys = self.gemm_only(xs).view(self.E * cap, self.d)

        out = torch.zeros_like(x)
        w_flat = weights.reshape(-1)[order]
        contrib = ys.index_select(0, slot[keep]) * w_flat[keep].unsqueeze(-1)
        out.index_add_(0, tok_sorted[keep], contrib)
        return out

    def grouped(self, xs: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
        """Dropless variable-size groups, if torch exposes _grouped_mm."""
        gm = torch._grouped_mm  # type: ignore[attr-defined]
        h = F.silu(gm(xs, self.w1, offs=offs)) * gm(xs, self.w3, offs=offs)
        return gm(h, self.w2, offs=offs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=32768)
    ap.add_argument("--scale", default="4B", choices=["0.8B", "2B", "4B", "9B", "27B"])
    ap.add_argument("--ks", default="8,2", help="top-k values to sweep (comma separated)")
    ap.add_argument("--tag", default="")
    ap.add_argument(
        "--grouped",
        action="store_true",
        help="also try torch._grouped_mm. OFF by default: its backward raises on every shape we "
        "tried, and at 27B it took the CUDA context down with it (no traceback, exit 1).",
    )
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
    has_grouped = args.grouped and hasattr(torch, "_grouped_mm")
    print(
        f"# {torch.cuda.get_device_name(0)} scale={args.scale} d_model={d} hidden={Fh} "
        f"tokens={T} torch={torch.__version__} grouped_mm={has_grouped}"
    )

    torch.manual_seed(0)
    x = torch.randn(T, d, device=dev, dtype=torch.bfloat16, requires_grad=True)

    def bench(fn):
        def f():
            if x.grad is not None:
                x.grad = None
            fn().sum().backward()

        try:
            return timed(f)
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            torch.cuda.empty_cache()
            return float("nan")

    rows = []
    for k in [int(s) for s in args.ks.split(",")]:
        print(f"\n### top-{k} MoE, E*W_e = {Fh} (same FFN params as dense)")
        print(
            f"{'target c':>10} {'E':>6} {'W_e':>6} {'c':>10} | {'gemm ms':>9} {'full ms':>9} "
            f"{'grouped ms':>10}"
        )
        for c in CS:
            E = int(round(k / c))
            if E < k:
                continue
            if Fh % E != 0:
                E = max(k, min([e for e in _divisors(Fh) if e >= k], key=lambda e: abs(e - E)))
            We = Fh // E
            if We < 1:
                continue
            c_real = k / E
            cap = int(math.ceil(T * k / E))
            moe = xs = None
            t_gemm = t_full = t_grp = float("nan")
            try:
                moe = TopKMoE(d, Fh, E, k, dev)
                xs = torch.randn(E, cap, d, device=dev, dtype=torch.bfloat16, requires_grad=True)
                t_gemm = bench_on(xs, moe.gemm_only)
                t_full = bench(lambda: moe.full(x))
                if has_grouped and We % 16 == 0 and cap % 16 == 0:
                    offs = torch.arange(1, E + 1, device=dev, dtype=torch.int32) * cap
                    xg = torch.randn(
                        E * cap, d, device=dev, dtype=torch.bfloat16, requires_grad=True
                    )
                    t_grp = bench_on(xg, lambda z: moe.grouped(z, offs))
            except Exception as exc:  # a point we cannot fit is a datum, not a lost sweep
                print(f"#   skipped E={E} We={We}: {type(exc).__name__}: {exc}")

            rec = {
                "c_target": c,
                "k": k,
                "E": E,
                "We": We,
                "c": c_real,
                "gemm_ms": t_gemm,
                "full_ms": t_full,
                "grouped_ms": t_grp,
            }
            rows.append(rec)
            print(
                f"{c:>10.6f} {E:>6d} {We:>6d} {c_real:>10.6f} | {t_gemm:>9.3f} {t_full:>9.3f} "
                f"{t_grp:>10.3f}"
            )
            del moe, xs
            torch.cuda.empty_cache()

    # Anchors: real Qwen3.5 MoE expert geometry dropped into THIS scale's d_model. The shared
    # expert is always live, so it counts as one more active expert. c is the active FFN width
    # as a fraction of this dense model's FFN -- i.e. what the swap does to per-token FFN compute.
    print(f"\n### real Qwen3.5 MoE geometries at d_model={d} (E x W_e, k routed + 1 shared)")
    print(f"{'model':>18} {'E':>5} {'W_e':>6} {'k+1':>5} {'c':>8} | {'gemm ms':>9} {'full ms':>9}")
    for name, E, We, k_routed in ANCHORS:
        k_live = k_routed + 1  # shared expert
        cap = int(math.ceil(T * k_live / E))
        moe = xs = None
        t_gemm = t_full = float("nan")
        try:
            moe = TopKMoE.with_width(d, We, E, k_live, dev)
            xs = torch.randn(E, cap, d, device=dev, dtype=torch.bfloat16, requires_grad=True)
            t_gemm = bench_on(xs, moe.gemm_only)
            t_full = bench(lambda: moe.full(x))
        except Exception as exc:
            print(f"#   skipped {name}: {type(exc).__name__}: {exc}")
        c_real = k_live * We / Fh
        rows.append(
            {
                "anchor": name,
                "k": k_live,
                "E": E,
                "We": We,
                "c": c_real,
                "gemm_ms": t_gemm,
                "full_ms": t_full,
                "grouped_ms": float("nan"),
            }
        )
        print(
            f"{name:>18} {E:>5d} {We:>6d} {k_live:>5d} {c_real:>8.4f} | {t_gemm:>9.3f} "
            f"{t_full:>9.3f}"
        )
        del moe, xs
        torch.cuda.empty_cache()

    tag = f"_{args.tag}" if args.tag else ""
    path = os.path.join(OUT, f"ffn_moe_arm_{args.scale}_{T}{tag}.json")
    json.dump({"tokens": T, "hidden": Fh, "d_model": d, "rows": rows}, open(path, "w"), indent=1)
    print(f"\n# wrote {os.path.basename(path)}")


def _divisors(n: int):
    out = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            out.append(i)
            out.append(n // i)
        i += 1
    return sorted(set(out))


def bench_on(inp: torch.Tensor, fn):
    def f():
        if inp.grad is not None:
            inp.grad = None
        fn(inp).sum().backward()

    try:
        return timed(f)
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        torch.cuda.empty_cache()
        return float("nan")


if __name__ == "__main__":
    main()
