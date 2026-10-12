"""What does ONE per-block skip-routing decision cost, and what does a span of K amortise?

Context (README, "Why GDN is slow: RE-RESOLVED"): a 4B/8k step is 559 ms wall against 546 ms of
GPU-busy time -- 2.4% idle -- so the model sits at the CPU/GPU crossover. The FFN router's
per-layer prologue costs 7.8% on top of its kernel and contains a host sync
(``bincount(...).tolist()``). The block-skip router (:mod:`olmo_core.nn.block_skip`) has the same
shape: per routed block it runs a ``Linear(d_model, 1)`` on the RMS-normalised block input, takes a
hard decision, calls ``holder.accumulate`` (which does ``keep.sum().item()`` -- ANOTHER host sync)
and then mixes the residual with a straight-through coefficient.

The interesting property of a SPAN decision -- one router evaluation covering K consecutive blocks
-- is that all of that is paid once per K blocks instead of once per block. This script measures:

  1. component  -- each piece of the per-block routing prologue in isolation, fwd+bwd.
  2. block      -- a real 4B attention block and a real 4B GDN block, stock vs wrapped in
                   ``block_skip_forward``, so the overhead is quoted as a share of a block.
  3. spanK      -- a REAL stack of K blocks run two ways: K per-block decisions vs ONE span
                   decision applied to all K. K = 1, 2, 4, 8. This is the amortisation, measured
                   rather than modelled.
  4. launches   -- kernel-launch counts per arm from the profiler, since launches (not FLOPs) are
                   what binds here.

Discipline (this box drifts up to 2x CPU-side within one process): every arm is interleaved in one
process, a control arm is re-measured at the start and end of every round, and p10/median/p90 are
reported alongside the control's own spread. Nothing is compared across jobs.

    python research/kv_route/flexcompute_40x/mlskip_router_overhead.py --scale 4B --tokens 8192
"""

import argparse
import json
import os
import time
from typing import Dict, List

import torch

from olmo_core.nn.attention import Attention
from olmo_core.nn.block_skip import BlockSkipHolder, BlockSkipRouter, block_skip_forward
from olmo_core.nn.router_fn import router_logits
from olmo_core.nn.transformer import TransformerConfig

OUT = os.path.dirname(os.path.abspath(__file__))
VOCAB = 248320
FACTORY = {
    "4B": TransformerConfig.qwen3_5_4B,
    "0.8B": TransformerConfig.qwen3_5_0_8B,
    "q3-4B": TransformerConfig.qwen3_4B,
}


def percentiles(ts: List[float]):
    ts = sorted(ts)
    n = len(ts)
    return (
        ts[int(round(0.10 * (n - 1)))],
        ts[int(round(0.50 * (n - 1)))],
        ts[int(round(0.90 * (n - 1)))],
    )


def time_once(fn, setup=None) -> float:
    """Time one call. ``setup`` (arm reconfiguration) runs OUTSIDE the timed region."""
    if setup is not None:
        setup()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def interleave(arms: Dict[str, object], rounds: int, samples: int, ctl: str, setups=None):
    """Run every arm `rounds*samples` times, re-measuring `ctl` at both ends of every round."""
    setups = setups or {}
    labels = [k for k in arms if k != ctl]
    out = {k: [] for k in arms}
    ctl_rounds = []
    for r in range(rounds):
        order = labels[r % max(1, len(labels)) :] + labels[: r % max(1, len(labels))]
        c = [time_once(arms[ctl], setups.get(ctl)) for _ in range(samples)]
        for lbl in order:
            out[lbl] += [time_once(arms[lbl], setups.get(lbl)) for _ in range(samples)]
        c += [time_once(arms[ctl], setups.get(ctl)) for _ in range(samples)]
        out[ctl] += c
        ctl_rounds.append(sorted(c)[len(c) // 2])
    return out, ctl_rounds


def count_launches(fn, iters: int = 3) -> float:
    """Kernel launches per call, from the profiler's CUDA events."""
    from torch.profiler import ProfilerActivity, profile

    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    n = 0
    for e in prof.events():
        if (
            str(getattr(e, "device_type", "")).endswith("CUDA")
            and e.name
            and "Memcpy" not in e.name
        ):
            n += 1
    return n / iters


def report(name: str, samples: Dict[str, List[float]], ctl_rounds, out: dict, base: str = None):
    print(f"\n## {name}")
    print("   control rounds (ms): " + " ".join(f"{t*1e3:.3f}" for t in ctl_rounds))
    print(f"   control spread {max(ctl_rounds)/min(ctl_rounds):.3f}x")
    med = {k: percentiles(v)[1] for k, v in samples.items()}
    b = med.get(base) if base else None
    print(f"{'arm':>22} {'p10':>9} {'med':>9} {'p90':>9} {'p90/p10':>8} {'vs base':>9}")
    for k, v in samples.items():
        p10, m, p90 = percentiles(v)
        rel = f"{m/b:8.3f}x" if b else "        -"
        print(f"{k:>22} {p10*1e3:>8.4f} {m*1e3:>8.4f} {p90*1e3:>8.4f} {p90/p10:>7.3f}x {rel:>9}")
        out.setdefault(name, {})[k] = dict(p10=p10, median=m, p90=p90)
    out.setdefault(name + "_control", {})["rounds"] = ctl_rounds
    return med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="4B")
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--spans", default="1,2,4,8")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = torch.device("cuda")
    T = args.tokens
    torch.manual_seed(0)

    cfg = FACTORY[args.scale](vocab_size=VOCAB)
    model = cfg.build(init_device="cpu")
    d = model.d_model
    n_layers = len(model.blocks)
    print(
        f"# GPU {torch.cuda.get_device_name(0)}  scale={args.scale} d_model={d} "
        f"n_layers={n_layers} tokens={T}"
    )
    out: dict = dict(scale=args.scale, tokens=T, d_model=d, n_layers=n_layers)

    # ---------------------------------------------------------------- 1. components
    h = torch.randn(1, T, d, device=dev, dtype=torch.bfloat16, requires_grad=True)
    router = BlockSkipRouter(d, device=dev, dtype=torch.bfloat16)
    holder = BlockSkipHolder(target=0.5)
    holder.routed_layers = list(range(n_layers))
    holder.begin_forward()

    with torch.no_grad():
        logits0 = router_logits(h, router).squeeze(-1)
    keep0 = logits0 > 0
    p0 = torch.sigmoid(logits0)
    dout = torch.randn_like(h)

    def a_gemm():
        h.grad = None
        router.zero_grad(set_to_none=True)
        lg = router_logits(h, router).squeeze(-1)
        lg.sum().backward()

    def a_decide():
        lg = router_logits(h, router).squeeze(-1).detach()
        torch.sigmoid(lg)
        return lg > 0

    def a_accum():
        holder._seen = set()
        holder.accumulate(exp_keep=p0.mean(), keep=keep0, layer_idx=0)

    def a_sync():
        float(keep0.sum().item())  # the host sync inside accumulate, in isolation

    def a_mix():
        h.grad = None
        x = h * 1.0
        p_sel = torch.where(keep0, p0, 1.0 - p0)
        coef = (1.0 + p_sel - p_sel.detach()).to(h.dtype)[:, :, None]
        mixed = h + coef * (x - h)
        torch.where(keep0[:, :, None], mixed, h).backward(dout)

    comps = {
        "CTL": a_gemm,
        "router gemm f+b": a_gemm,
        "decide (>0)": a_decide,
        "accumulate": a_accum,
        "  of which .item()": a_sync,
        "straight-through mix": a_mix,
    }
    for f in comps.values():
        for _ in range(5):
            f()
    s, cr = interleave(comps, args.rounds, args.samples, "CTL")
    report("components", s, cr, out)
    comp_launch = {k: count_launches(v) for k, v in comps.items() if k != "CTL"}
    print("   launches/call: " + "  ".join(f"{k}={v:.1f}" for k, v in comp_launch.items()))
    out["component_launches"] = comp_launch

    # ---------------------------------------------------------------- 2. block level
    model = model.to(dev)
    model.init_weights(max_seq_len=T, device=dev)
    model = model.to(torch.bfloat16)
    blocks = list(model.blocks.values())
    attn_i = next((i for i, b in enumerate(blocks) if type(b.attention) is Attention), None)
    gdn_i = next((i for i, b in enumerate(blocks) if type(b.attention) is not Attention), None)
    print(f"# first attention block L{attn_i}, first GDN block L{gdn_i}")

    from olmo_core.nn.block_skip import block_skip_decide, set_block_skip_span

    model.enable_block_skip(start_layer=0, target=0.5)
    bholder = model._block_skip["holder"]
    bholder.begin_forward()

    def reset_holder():
        # ``_reset`` also drops the accumulated ``exp_keep`` tensors; without it every timed
        # iteration would retain its router's autograd graph for the life of the process.
        bholder._reset()
        bholder._choice_cache = {}
        bholder._span_decision = {}
        bholder._span_input = {}

    def block_arms(i: int, tag: str):
        """Four arms that separate the two things block skipping adds to a block."""
        blk = blocks[i]
        x = torch.randn(1, T, d, device=dev, dtype=torch.bfloat16, requires_grad=True)
        g = torch.randn_like(x)
        keep_all = torch.ones(1, T, dtype=torch.bool, device=dev)

        def clear():
            x.grad = None
            blk.zero_grad(set_to_none=True)

        def stock():  # the block as it runs today
            clear()
            blk._bskip_orig_forward(x).backward(g)

        def keymask():  # + the masked mixer path, no router at all
            clear()
            blk._bskip_orig_forward(x, block_keep=keep_all).backward(g)

        def prologue():  # + the ROUTER prologue and mix, mixer left on its fast path
            clear()
            reset_holder()
            p_, keep_ = block_skip_decide(blk, x)
            out_ = blk._bskip_orig_forward(x)
            p_sel = torch.where(keep_, p_, 1.0 - p_)
            coef = (1.0 + p_sel - p_sel.detach()).to(x.dtype)[:, :, None]
            torch.where(keep_[:, :, None], x + coef * (out_ - x), x).backward(g)

        def full():  # both, i.e. the deployed block_skip path
            clear()
            reset_holder()
            block_skip_forward(blk, x, {}).backward(g)

        return {
            f"{tag} stock": stock,
            f"{tag} +keymask only": keymask,
            f"{tag} +router+mix only": prologue,
            f"{tag} full block_skip": full,
        }

    blk_arms: Dict[str, object] = {}
    for i, tag in ((attn_i, "attn"), (gdn_i, "gdn")):
        if i is not None:
            blk_arms.update(block_arms(i, tag))
    ctl_key = list(blk_arms)[0]
    blk_arms = {"CTL": blk_arms[ctl_key], **blk_arms}
    for f in blk_arms.values():
        for _ in range(4):
            f()
    s_, cr = interleave(blk_arms, args.rounds, args.samples, "CTL")
    med = report("blocks", s_, cr, out)
    for tag in ("attn", "gdn"):
        if f"{tag} stock" in med:
            a = med[f"{tag} stock"]
            print(
                f"   {tag}: key-mask path {100*(med[f'{tag} +keymask only']-a)/a:+.2f}%, "
                f"router+mix {100*(med[f'{tag} +router+mix only']-a)/a:+.2f}%, "
                f"both {100*(med[f'{tag} full block_skip']-a)/a:+.2f}% of the stock block"
            )
            out.setdefault("block_overhead", {})[tag] = {
                k: med[f"{tag} {k}"]
                for k in ("stock", "+keymask only", "+router+mix only", "full block_skip")
            }
    blk_launch = {k: count_launches(v, 2) for k, v in blk_arms.items() if k != "CTL"}
    print("   launches/call: " + "  ".join(f"{k}={v:.0f}" for k, v in blk_launch.items()))
    out["block_launches"] = blk_launch

    # ---------------------------------------------------------------- 3. span-K, measured
    spans = [int(x) for x in args.spans.split(",")]
    K = max(spans)
    stack = blocks[:K]
    x = torch.randn(1, T, d, device=dev, dtype=torch.bfloat16, requires_grad=True)
    g = torch.randn_like(x)

    def run_stack(k: int, routed: bool):
        sub = stack[:k]

        def run():
            x.grad = None
            for b in sub:
                b.zero_grad(set_to_none=True)
            reset_holder()
            hh = x
            for b in sub:
                if routed:
                    pp, kk = block_skip_decide(b, hh)
                    hh = block_skip_forward(b, hh, {}, pp, kk)
                else:
                    hh = b._bskip_orig_forward(hh)
            hh.backward(g)

        return run

    def set_span(sp: int, lazy: bool):
        def setup():
            set_block_skip_span(model, sp)
            bholder.lazy_span_mix = lazy

        return setup

    span_arms, span_setups = {}, {}
    for k in spans:
        span_arms[f"K={k} plain"] = run_stack(k, False)
        span_setups[f"K={k} plain"] = set_span(1, False)
        span_arms[f"K={k} perblock"] = run_stack(k, True)
        span_setups[f"K={k} perblock"] = set_span(1, False)
        if k >= 2:  # a span of 1 is per-block routing; a span of k>K would defer past the stack
            span_arms[f"K={k} span"] = run_stack(k, True)
            span_setups[f"K={k} span"] = set_span(k, False)
            span_arms[f"K={k} span+lazymix"] = run_stack(k, True)
            span_setups[f"K={k} span+lazymix"] = set_span(k, True)
    span_arms["CTL"] = span_arms[f"K={spans[0]} plain"]
    span_setups["CTL"] = span_setups[f"K={spans[0]} plain"]
    for lbl, f in span_arms.items():
        span_setups[lbl]()
        for _ in range(3):
            f()

    # ---- exactness: at keep-all init the three routed variants must equal the plain stack
    print("\n## exactness at keep-all init (untrained router: bias +10 -> keep everything)")
    with torch.no_grad():
        refs = {}
        exact_labels = [f"K={K} plain", f"K={K} perblock"]
        if K >= 2:
            exact_labels += [f"K={K} span", f"K={K} span+lazymix"]
        for lbl in exact_labels:
            span_setups[lbl]()
            reset_holder()
            hh = x
            for b in stack:
                if "plain" in lbl:
                    hh = b._bskip_orig_forward(hh)
                else:
                    pp, kk = block_skip_decide(b, hh)
                    hh = block_skip_forward(b, hh, {}, pp, kk)
            refs[lbl] = hh.float().clone()
        base = refs[f"K={K} plain"]
        scale_ = base.abs().max().item()
        eps_bf16 = 2**-8  # 3.9e-3; one ULP is up to 2x this after a rounded add
        print(f"   reference |h|max = {scale_:.3f}; bf16 eps = {eps_bf16:.2e}")
        for lbl, v in refs.items():
            dmax = (v - base).abs().max().item()
            rel = dmax / scale_
            tag = "  EXACT" if dmax == 0.0 else f"  = {rel/eps_bf16:.2f} bf16 eps"
            print(f"   vs plain  {lbl:>22}: max|d| = {dmax:.3e}  rel = {rel:.3e}{tag}")
            out.setdefault("exactness", {})[lbl] = dict(max_abs=dmax, rel=rel)
        if f"K={K} span+lazymix" in refs:
            for a_, b_ in (
                (f"K={K} perblock", f"K={K} span"),
                (f"K={K} span", f"K={K} span+lazymix"),
            ):
                dm = (refs[a_] - refs[b_]).abs().max().item()
                print(
                    f"   {a_} vs {b_}: max|d| = {dm:.3e}  rel = {dm/scale_:.3e}"
                    f"{'  EXACT' if dm == 0.0 else ''}"
                )
                out.setdefault("exactness_pairs", {})[f"{a_}|{b_}"] = dm

    s_, cr = interleave(span_arms, args.rounds, args.samples, "CTL", span_setups)
    med = report("spanK", s_, cr, out)
    print(
        f"\n{'K':>4} {'plain ms':>10} {'perblock':>10} {'span':>10} {'span+lazy':>11} "
        f"{'perblk oh%':>11} {'span oh%':>10} {'lazy oh%':>10}"
    )
    for k in spans:
        pl = med[f"K={k} plain"]
        pb = med[f"K={k} perblock"]
        sp = med.get(f"K={k} span")
        sl = med.get(f"K={k} span+lazymix")
        f3 = lambda v: f"{v*1e3:>9.3f}" if v else f"{'-':>9}"  # noqa: E731
        fp = lambda v: f"{100*(v-pl)/pl:>9.2f}%" if v else f"{'-':>10}"  # noqa: E731
        print(
            f"{k:>4} {pl*1e3:>9.3f} {pb*1e3:>9.3f} {f3(sp)} {f3(sl):>10} "
            f"{100*(pb-pl)/pl:>10.2f}% {fp(sp)} {fp(sl)}"
        )
        out.setdefault("span_summary", {})[str(k)] = dict(
            plain=pl, perblock=pb, span=sp, span_lazy=sl
        )
    span_launch = {}
    for lbl, f in span_arms.items():
        if lbl == "CTL":
            continue
        span_setups[lbl]()
        span_launch[lbl] = count_launches(f, 2)
    print("\n   launches/call (and per routed block over the plain stack):")
    for k in spans:
        pl = span_launch[f"K={k} plain"]
        row = [f"plain {pl:.0f}"]
        for v in ("perblock", "span", "span+lazymix"):
            n = span_launch.get(f"K={k} {v}")
            if n is not None:
                row.append(f"{v} {n:.0f} (+{(n-pl)/k:.1f}/blk)")
        print(f"     K={k}: " + "  ".join(row))
    out["span_launches"] = span_launch
    set_block_skip_span(model, 1)

    path = args.out or os.path.join(OUT, f"mlskip_router_overhead_{args.scale}_{T}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
