"""Do the three routers' FLOP cuts convert into TRAINING wall-clock at 4B?

The FLOP ceiling work (this directory) prices the routers analytically. This measures them, all in
ONE process with the arms INTERLEAVED and a dense control re-measured every round, because eager
timings on this box drift ~2x within a process (see the "Why GDN is slow: RE-RESOLVED" section).

Method
------
One 4B model is built once and all three routers are installed once, so every arm runs the same
parameters and the same allocator state. An arm is a triple of router operating points; the model
switches between them by

* ``holder.enabled`` (FFN, KV), and stashing ``block._bskip`` (block skip), and
* FORCED routing decisions: each holder caches its per-forward decisions in ``_choice_cache`` (so
  an activation-checkpoint recompute replays them), and this script wraps ``begin_forward`` to
  refill that cache with a decision pattern of a chosen cost. The routers themselves still run --
  logits, straight-through coefficients, gains, budget accumulation -- so the measurement includes
  every bit of router overhead. Only WHICH rung/keep a token gets is pinned, which is what makes
  the arms comparable at a stated operating point instead of at whatever an untrained router does
  (untrained = keep everything, i.e. no FLOP cut at all).

FLOPs are the analytic per-token training count from ``ceiling.buckets`` / ``ceiling.routed_total``
priced at the arm's forced operating point, so "FLOP reduction" and "wall-clock reduction" in the
output are the two sides the conversion ratio compares.

    python research/kv_route/flexcompute_40x/bench_router_matrix.py --ctx 8192 --rounds 3
"""

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch

from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.transformer import TransformerConfig

from ceiling import buckets, routed_total  # noqa: E402  (same directory)

VOCAB = 248320
OUT = os.path.dirname(os.path.abspath(__file__))
SCALES = {
    "0.8B": TransformerConfig.qwen3_5_0_8B,
    "4B": TransformerConfig.qwen3_5_4B,
    "9B": TransformerConfig.qwen3_5_9B,
    "27B": TransformerConfig.qwen3_5_27B,
}

#: Shape of the working (non-null) part of the FFN rung distribution, over rungs 0..n-2. Chosen to
#: resemble the MEASURED trained histograms in ``rungs_v11-*.json`` (a few percent on the full rung,
#: most working mass on the middle rungs); an arm then moves ``f_null`` of the tokens to the null
#: rung, which is the axis that actually varies the cost.
WORK_SHAPE = [0.08, 0.08, 0.28, 0.12, 0.44]


class Arm:
    """One operating point: FFN mean cost, KV keep fraction, block-skip keep fraction."""

    def __init__(
        self,
        name: str,
        *,
        ffn: Optional[float] = None,  # f_null, or None to disable the FFN router
        kv: Optional[float] = None,  # keep fraction, or None to disable
        bskip: Optional[float] = None,  # keep fraction, or None to disable
        head_gate: bool = False,
        compact: bool = False,  # gathered block skip instead of masked
        static_ffn: bool = False,  # delete the routed FFNs outright (no router at all)
        capacity: bool = False,  # fixed-capacity (top-k) block skip: static shapes, no host sync
    ):
        self.name = name
        self.ffn, self.kv, self.bskip, self.head_gate = ffn, kv, bskip, head_gate
        self.compact = compact
        self.static_ffn = static_ffn
        self.capacity = capacity


def build_arms(spec: str) -> List[Arm]:
    all_arms = [
        Arm("dense"),
        # FFN router: f_null=-1 is the "router runs, every token takes the full rung" overhead probe
        Arm("ffn-overhead", ffn=-1.0),
        Arm("ffn-fnull0.00", ffn=0.00),
        Arm("ffn-fnull0.50", ffn=0.50),
        Arm("ffn-fnull0.75", ffn=0.75),
        Arm("ffn-fnull1.00", ffn=1.00),
        Arm("kv-1.00", kv=1.00),
        Arm("kv-0.50", kv=0.50),
        Arm("kv-0.25", kv=0.25),
        Arm("bskip-1.00", bskip=1.00),
        Arm("bskip-0.50", bskip=0.50),
        Arm("bskip-0.25", bskip=0.25),
        Arm("bskipC-1.00", bskip=1.00, compact=True),
        Arm("bskipC-0.50", bskip=0.50, compact=True),
        Arm("bskipC-0.25", bskip=0.25, compact=True),
        Arm("all3-mid", ffn=0.75, kv=0.50, bskip=0.50),
        Arm("all3-aggr", ffn=1.00, kv=0.25, bskip=0.25),
        Arm("all3C-mid", ffn=0.75, kv=0.50, bskip=0.50, compact=True),
        Arm("all3C-aggr", ffn=1.00, kv=0.25, bskip=0.25, compact=True),
        Arm("bskipC-0.50+headgate", bskip=0.50, compact=True, head_gate=True),
        Arm("all3C-aggr+headgate", ffn=1.00, kv=0.25, bskip=0.25, compact=True, head_gate=True),
        # Static deletion of the routed FFNs: the same function an all-null router expresses, with
        # no router GEMM, no gather and no per-layer host sync (another agent measured 1.177x vs
        # 1.082x for the router expressing it). Composed with the rest of the stack below.
        Arm("ffn-static", static_ffn=True),
        Arm("staticFFN+bskipC-0.25", static_ffn=True, bskip=0.25, compact=True),
        Arm(
            "staticFFN+bskipC-0.25+headgate",
            static_ffn=True,
            bskip=0.25,
            compact=True,
            head_gate=True,
        ),
        Arm("bskipK-0.25", bskip=0.25, compact=True, capacity=True),
        Arm(
            "staticFFN+bskipK-0.25+headgate",
            static_ffn=True,
            bskip=0.25,
            compact=True,
            capacity=True,
            head_gate=True,
        ),
        Arm(
            "staticFFN+kv0.25+bskipC-0.25+headgate",
            static_ffn=True,
            kv=0.25,
            bskip=0.25,
            compact=True,
            head_gate=True,
        ),
        Arm("headgate", head_gate=True),
        Arm("all3-mid+headgate", ffn=0.75, kv=0.50, bskip=0.50, head_gate=True),
    ]
    if spec == "all":
        return all_arms
    wanted = set(spec.split(","))
    return [a for a in all_arms if a.name in wanted]


def rung_fracs(f_null: float, n_rungs: int) -> List[float]:
    """Token fractions per rung for a given null share (``f_null < 0`` = all on the full rung)."""
    if f_null < 0:
        return [1.0] + [0.0] * (n_rungs - 1)
    work = WORK_SHAPE[: n_rungs - 1]
    work = [w / sum(work) * (1.0 - f_null) for w in work]
    return work + [f_null]


def forced_choice(n: int, fracs: List[float], device) -> torch.Tensor:
    """A deterministic per-token rung assignment with the given fractions (contiguous blocks).

    Contiguous rather than random on purpose: the routed FFN gathers tokens per rung anyway
    (``argsort`` + ``split``), so the layout does not change its kernels, and a fixed pattern keeps
    every arm's decision identical across rounds.
    """
    out = torch.empty(n, dtype=torch.long, device=device)
    start = 0
    for g, f in enumerate(fracs):
        end = n if g == len(fracs) - 1 else min(n, start + int(round(f * n)))
        out[start:end] = g
        start = end
    return out


def forced_keep(B: int, T: int, frac: float, device) -> torch.Tensor:
    """A (B, T) bool keep-mask with exactly ``round(frac*T)`` kept, spread evenly over positions."""
    k = int(round(frac * T))
    keep = torch.zeros(B, T, dtype=torch.bool, device=device)
    if k > 0:
        idx = torch.linspace(0, T - 1, k, device=device).round().long().unique()
        keep[:, idx] = True
    return keep


class Forcer:
    """Refills every holder's decision cache after ``begin_forward`` clears it."""

    def __init__(self, model, B: int, T: int, device):
        self.model, self.B, self.T, self.device = model, B, T, device
        self.ffn_fracs: Optional[List[float]] = None
        self.kv_keep: Optional[float] = None
        self.bskip_keep: Optional[float] = None
        for cfg, filler in (
            (model._nested_ffn_moe, self._fill_ffn),
            (model._kv_route, self._fill_kv),
            (model._block_skip, self._fill_bskip),
        ):
            holder = cfg["holder"]
            orig = holder.begin_forward

            def wrapped(*a, _orig=orig, _filler=filler, _h=holder, **kw):
                _orig(*a, **kw)
                _filler(_h)

            holder.begin_forward = wrapped

    def _fill_ffn(self, holder) -> None:
        if self.ffn_fracs is None or not holder.enabled:
            return
        n = self.B * self.T
        choice = forced_choice(n, self.ffn_fracs, self.device)
        for li in self.model._nested_ffn_moe.get("routed_layers", []):
            holder._choice_cache[(int(li), n)] = choice

    def _fill_kv(self, holder) -> None:
        if self.kv_keep is None or not holder.enabled:
            return
        keep = forced_keep(self.B, self.T, self.kv_keep, self.device)
        for li in self.model._kv_route["routed"]:
            holder._choice_cache[(int(li), self.B, self.T)] = keep

    def _fill_bskip(self, holder) -> None:
        if self.bskip_keep is None or not holder.enabled:
            return
        keep = forced_keep(self.B, self.T, self.bskip_keep, self.device)
        # Under fixed capacity the real router hands the compacted path its indices for free
        # (they ARE the top-k), so the forced arm must supply them too or it would measure a
        # ``nonzero`` sync the capacity router never pays.
        idx = keep[0].nonzero(as_tuple=True)[0][None, :] if holder.capacity_frac else None
        for li in self.model._block_skip["routed"]:
            holder._choice_cache[(int(li), self.B, self.T)] = keep
            if idx is not None:
                holder._idx_cache[(int(li), self.B, self.T)] = idx


def timed(step, warmup: int, iters: int):
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts


def profile_step(step, iters: int = 3):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            step()
        torch.cuda.synchronize()
    tot, cnt = 0.0, 0
    per_kernel: Dict[str, float] = defaultdict(float)
    for e in prof.key_averages():
        if e.self_device_time_total > 0:
            tot += e.self_device_time_total
            cnt += e.count
            per_kernel[e.key] += e.self_device_time_total
    return tot / iters / 1e6, cnt / iters, {k: v / iters / 1e6 for k, v in per_kernel.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="4B")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--arms", default="all")
    # 0 = route every layer, matching the trainer default and the trained checkpoints.
    ap.add_argument("--ffn-start-layer", type=int, default=0)
    ap.add_argument("--divisors", default="1,2.4,4.9,10,32")
    ap.add_argument("--attn-backend", default="flash_2")
    ap.add_argument("--kernels", action="store_true", help="dump each profiled arm's top kernels")
    ap.add_argument("--loss-density", type=float, default=0.0064, help="supervised label fraction")
    ap.add_argument(
        "--docs",
        type=int,
        default=0,
        help="pack the sequence as this many equal documents (the --pack training path). Matters "
        "for the compacted block skip: every block then re-derives its own cu_doc_lens, which "
        "misses fla's identity-keyed prepare_chunk_indices cache",
    )
    ap.add_argument(
        "--ac",
        action="store_true",
        help="full-block activation checkpointing (as the trainer uses)",
    )
    ap.add_argument("--compact-bskip", action="store_true", help="gathered block skip, not masked")
    ap.add_argument(
        "--profile", action="store_true", help="also profile each arm (GPU-busy, launches)"
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = torch.device("cuda")
    T = args.ctx
    B = 1
    print(f"# GPU {torch.cuda.get_device_name(0)} torch {torch.__version__}", flush=True)
    print(f"# scale={args.scale} ctx={T} rounds={args.rounds} iters={args.iters}", flush=True)

    from olmo_core.nn.attention import AttentionBackendName

    # flash_2 is what the trainer uses; the torch backend refuses intra-document masking, which
    # the packed (--docs) arms need.
    cfg = SCALES[args.scale](vocab_size=VOCAB, attn_backend=AttentionBackendName(args.attn_backend))
    cfg.lm_head.loss_implementation = LMLossImplementation.fused_linear
    torch.manual_seed(0)
    model = cfg.build(init_device="cpu").to(dev)
    model.init_weights(max_seq_len=T, device=dev)
    model = model.to(torch.bfloat16)
    model.train()

    model.enable_nested_ffn_moe(
        start_layer=args.ffn_start_layer,
        divisors=[float(x) for x in args.divisors.split(",")],
        include_null=True,
        target_cost=0.01,
        explore_prob=0.0,
        width_multiple=8,
    )
    model._nested_ffn_moe["routed_layers"] = [
        i for i in range(len(model.blocks)) if i >= args.ffn_start_layer
    ]
    model.enable_kv_route(start_layer=0, target=0.5, explore_prob=0.0)
    model.enable_block_skip(start_layer=0, target=0.5, compact=args.compact_bskip)
    # arms flip this per-arm; the constructor value is only the default
    model.budget_attach = True
    if args.ac:
        from olmo_core.nn.transformer import TransformerActivationCheckpointingMode

        model.apply_activation_checkpointing(TransformerActivationCheckpointingMode.full)

    widths = model._nested_ffn_moe["widths"]
    costs = model._nested_ffn_moe["costs"]
    n_rungs = len(widths)
    print(f"# rungs widths={widths} costs={[round(c,4) for c in costs]}", flush=True)
    print(
        f"# kv-routed attention layers: {model._kv_route['routed']}\n"
        f"# block-skip routed blocks: {len(model._block_skip['routed'])}",
        flush=True,
    )

    bkt = buckets(model, T)
    dense_flops = bkt["total"]
    n_layers = len(model.blocks)
    ffn_routed_frac = (
        sum(1 for i in range(n_layers) if i >= args.ffn_start_layer) / n_layers
    )  # rough: all blocks have the same FFN size on this family

    forcer = Forcer(model, B, T, dev)
    ffn_holder = model._nested_ffn_moe["holder"]
    kv_holder = model._kv_route["holder"]
    bs_holder = model._block_skip["holder"]
    bskip_cfgs = {k: b._bskip for k, b in model.blocks.items()}

    n_sup = max(1, int(T * args.loss_density))
    labels = torch.full((B, T), -100, dtype=torch.long, device=dev)
    labels[:, -n_sup:] = torch.randint(0, VOCAB, (B, n_sup), device=dev)
    ids = torch.randint(0, VOCAB, (B, T), device=dev)

    doc_kwargs: Dict[str, Any] = {}
    if args.docs > 1:
        per = T // args.docs
        lens = [per] * (args.docs - 1) + [T - per * (args.docs - 1)]
        doc_kwargs = {
            "doc_lens": torch.tensor([lens], dtype=torch.int32, device=dev),
            "max_doc_lens": [max(lens)],
        }

    def step():
        model.zero_grad(set_to_none=True)
        out = model(ids, labels=labels, loss_reduction="sum", return_logits=False, **doc_kwargs)
        loss = out.loss
        if getattr(out, "auxiliary_losses", None):
            for v in out.auxiliary_losses.values():
                loss = loss + v
        loss.backward()

    class _Identity(torch.nn.Module):
        """Stands in for a statically deleted FFN: same shape out, no parameters in the graph."""

        def forward(self, x, *args, **kwargs):
            return x

    def _inner(mod):
        """Unwrap an activation-checkpoint wrapper.

        Under ``--ac`` ``model.blocks`` holds ``CheckpointWrapper`` objects. Reading
        ``wrapper.feed_forward`` forwards to the inner block, but ASSIGNING to it sets an attribute
        on the wrapper that nothing ever reads -- so the static-FFN arm silently became a no-op
        (9B/8k measured ffn-static at exactly 1.000x while the equivalent all-null ROUTER measured
        1.253x, which is what caught it).
        """
        return getattr(mod, "_checkpoint_wrapped_module", mod)

    real_ffn = {k: _inner(b).feed_forward for k, b in model.blocks.items()}
    dead_ffn = _Identity().to(dev)

    def apply(arm: Arm):
        for k, b in model.blocks.items():
            _inner(b).feed_forward = (
                dead_ffn if (arm.static_ffn and int(k) >= args.ffn_start_layer) else real_ffn[k]
            )
        ffn_holder.enabled = arm.ffn is not None and not arm.static_ffn
        kv_holder.enabled = arm.kv is not None
        bs_holder.enabled = arm.bskip is not None
        bs_holder.compact = arm.compact
        bs_holder.capacity_frac = arm.bskip if arm.capacity else None
        for k, b in model.blocks.items():
            b._bskip = bskip_cfgs[k] if arm.bskip is not None else None
        forcer.ffn_fracs = rung_fracs(arm.ffn, n_rungs) if arm.ffn is not None else None
        forcer.kv_keep = arm.kv
        forcer.bskip_keep = arm.bskip
        model.lm_head._sparse_loss_positions = arm.head_gate

    def arm_flops(arm: Arm) -> float:
        """Analytic per-token training FLOPs at this arm's operating point."""
        ffn_cost = 1.0
        if arm.static_ffn:
            ffn_cost = 1 - ffn_routed_frac
        elif arm.ffn is not None:
            fr = rung_fracs(arm.ffn, n_rungs)
            routed_cost = sum(f * c for f, c in zip(fr, costs))
            # un-routed layers keep the dense FFN
            ffn_cost = (1 - ffn_routed_frac) + ffn_routed_frac * routed_cost
        skip = arm.bskip if arm.bskip is not None else 1.0
        kv = arm.kv if arm.kv is not None else 1.0
        # a skipped token is neither a query nor a key, so the score term goes as skip * (skip*kv)
        return routed_total(
            bkt,
            skip_keep=skip,
            ffn_cost=ffn_cost,
            kv_keep=kv * skip,
            head_frac=args.loss_density if arm.head_gate else 1.0,
        )

    arms = build_arms(args.arms)
    samples: Dict[str, List[float]] = {a.name: [] for a in arms}
    control: List[float] = []

    for rnd in range(args.rounds):
        apply(arms[0] if arms[0].name == "dense" else Arm("dense"))
        try:
            control.append(timed(step, 1, 3)[1])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            control.append(float("nan"))
        print(f"# round {rnd}: control median {control[-1]*1e3:.1f} ms", flush=True)
        for arm in arms:
            apply(arm)
            try:
                t_arm = time.perf_counter()
                ts = timed(step, 1 if rnd else 2, args.iters)
                samples[arm.name].extend(ts)
                print(
                    f"#   [{rnd}] {arm.name}: median {ts[len(ts) // 2] * 1e3:.1f} ms "
                    f"({time.perf_counter() - t_arm:.0f}s)",
                    flush=True,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"#   {arm.name}: OOM", flush=True)
            except Exception as exc:  # keep the matrix going; report which arm broke
                torch.cuda.empty_cache()
                print(f"#   {arm.name}: ERROR {type(exc).__name__}: {exc}", flush=True)

    prof: Dict[str, Dict[str, float]] = {}
    if args.profile:
        for arm in arms:
            apply(arm)
            try:
                gpu, launches, per_kernel = profile_step(step)
                prof[arm.name] = {"gpu_busy_s": gpu, "launches": launches}
                if args.kernels:
                    # Where the GPU time that survived the routers actually goes. Printed only for
                    # the arms asked for, because this is the question "what still limits
                    # conversion" reduces to once the FLOPs are gone.
                    top = sorted(per_kernel.items(), key=lambda kv: -kv[1])[:12]
                    print(f"#   kernels {arm.name} (total {gpu * 1e3:.1f} ms):", flush=True)
                    for name, t in top:
                        print(f"#     {t * 1e3:>8.1f} ms  {name[:72]}", flush=True)
            except Exception as exc:
                print(f"#   profile {arm.name}: {type(exc).__name__}: {exc}", flush=True)

    def stats(xs: List[float]):
        if not xs:
            return None
        xs = sorted(xs)
        p = lambda q: xs[min(len(xs) - 1, int(q * len(xs)))]  # noqa: E731
        return dict(p10=p(0.1), median=xs[len(xs) // 2], p90=p(0.9), n=len(xs))

    dense_stat = stats(samples.get("dense", []))
    rows = []
    for arm in arms:
        s = stats(samples[arm.name])
        f = arm_flops(arm)
        row = dict(
            arm=arm.name,
            ffn_fnull=arm.ffn,
            kv_keep=arm.kv,
            bskip_keep=arm.bskip,
            compact=arm.compact,
            capacity=arm.capacity,
            static_ffn=arm.static_ffn,
            head_gate=arm.head_gate,
            flops_per_token=f,
            flop_reduction=dense_flops / f if f > 0 else float("inf"),
            **(s or {}),
        )
        if s and dense_stat:
            row["wall_speedup"] = dense_stat["median"] / s["median"]
            row["conversion"] = row["wall_speedup"] / row["flop_reduction"]
        if arm.name in prof:
            row.update(prof[arm.name])
        rows.append(row)

    print(
        f"\n{'arm':>20} {'FLOPx':>7} {'p10 ms':>8} {'med ms':>8} {'p90 ms':>8} "
        f"{'wallx':>7} {'conv':>6}"
    )
    for r in rows:
        if "median" not in r or r["median"] is None:
            print(f"{r['arm']:>20} {r['flop_reduction']:>7.2f} {'-':>8} {'(no samples)':>26}")
            continue
        print(
            f"{r['arm']:>20} {r['flop_reduction']:>7.2f} {r['p10']*1e3:>8.1f} "
            f"{r['median']*1e3:>8.1f} {r['p90']*1e3:>8.1f} "
            f"{r.get('wall_speedup', float('nan')):>7.3f} {r.get('conversion', float('nan')):>6.2f}"
        )
    cd = [c for c in control if c == c]
    if cd:
        print(
            f"\n# control drift across rounds: {max(cd)/min(cd):.3f}x "
            f"({' '.join(f'{c*1e3:.0f}' for c in control)} ms)"
        )
    out = args.out or os.path.join(OUT, f"router_matrix_{args.scale}_{T}.json")
    with open(out, "w") as fh:
        json.dump(
            dict(
                scale=args.scale,
                ctx=T,
                dense_flops_per_token=dense_flops,
                buckets={k: v for k, v in bkt.items()},
                widths=widths,
                costs=costs,
                control=control,
                rows=rows,
            ),
            fh,
            indent=1,
        )
    print(f"# wrote {out}", flush=True)


if __name__ == "__main__":
    main()
