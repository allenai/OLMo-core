"""Static layer pruning: what does per-token routing cost to express a CONSTANT?

`dump_rung_histograms.py` dumped the real per-layer rung distributions from two trained FFN-routing
checkpoints (`rungs_v11-*.json`, `rungs_v12-*.json`; Qwen3-4B, 36 layers). The router did not learn
per-token routing -- it learned LAYER PRUNING: at target .02 **22 of 36 layers are >99% null** and
at target .05, 13 of 36. For such a layer the whole per-token apparatus (router GEMM, argmax,
``bincount(...).tolist()`` HOST SYNC, stable argsort, split, ladder call, ``zeros_like`` output,
holder accumulation) runs every forward to produce a decision that never varies.

The degenerate case of multi-layer skipping is therefore free: DELETE the FFN of those layers. No
router, no mask, no gather, no launches -- and it is a deployed-model change, not a training-recipe
change. This measures the three states at both the FFN level and the whole-model level:

  (a) dense       -- the full FFN
  (b) routed-null -- the deployed nested-ffn path with the router pinned to the null rung
  (c) pruned      -- the FFN structurally removed from the block

(b) - (c) is the cost of the machinery; (a) - (c) is the total win.

    python research/kv_route/flexcompute_40x/mlskip_static_prune.py --scale q3-4B --tokens 8192

The layer set for the model-level arms is read from the rung dumps rather than assumed, so the
"22 layers" claim and the measurement stay tied to the same file.
"""

import argparse
import json
import os
import time
from typing import Dict, List

import torch
import torch.nn as nn

from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.transformer import TransformerConfig

OUT = os.path.dirname(os.path.abspath(__file__))
VOCAB = 248320
SCALES = {
    "q3-4B": TransformerConfig.qwen3_4B,
    "4B": TransformerConfig.qwen3_5_4B,
    "0.8B": TransformerConfig.qwen3_5_0_8B,
}
# The trained geometry in the dumps: divisors 1, 16, 64, 256, 1024, 9728 (+ null).
DIVISORS = (1, 16, 64, 256, 1024, 9728)


class NoFFNBlock(nn.Module):
    """``TransformerBlock`` with the feed-forward half structurally absent.

    Not ``block.feed_forward = Identity()`` (the ablation style used in `bench_step_partition.py`):
    that still runs ``feed_forward_norm`` and the residual add, which is exactly the overhead this
    script is trying to price. Here the FFN norm, the FFN and its residual stream are all gone.
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x, *, loss_div_factor=None, **kwargs):
        del loss_div_factor
        b = self.inner
        return b.attention_residual_stream(x, b.attention(b.attention_norm(x), **kwargs))


def percentiles(ts: List[float]):
    ts = sorted(ts)
    n = len(ts)
    return (
        ts[int(round(0.10 * (n - 1)))],
        ts[int(round(0.50 * (n - 1)))],
        ts[int(round(0.90 * (n - 1)))],
    )


def time_once(fn) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def interleave(arms: Dict[str, object], rounds: int, samples: int, ctl: str):
    labels = [k for k in arms if k != ctl]
    out = {k: [] for k in arms}
    ctl_rounds = []
    for r in range(rounds):
        order = labels[r % max(1, len(labels)) :] + labels[: r % max(1, len(labels))]
        c = [time_once(arms[ctl]) for _ in range(samples)]
        for lbl in order:
            out[lbl] += [time_once(arms[lbl]) for _ in range(samples)]
        c += [time_once(arms[ctl]) for _ in range(samples)]
        out[ctl] += c
        ctl_rounds.append(sorted(c)[len(c) // 2])
    return out, ctl_rounds


def count_launches(fn, iters: int = 3) -> float:
    from torch.profiler import ProfilerActivity, profile

    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    n = sum(
        1
        for e in prof.events()
        if str(getattr(e, "device_type", "")).endswith("CUDA") and e.name and "Memcpy" not in e.name
    )
    return n / iters


def report(name: str, samples, ctl_rounds, out: dict, base=None):
    print(f"\n## {name}")
    print("   control rounds (ms): " + " ".join(f"{t*1e3:.3f}" for t in ctl_rounds))
    print(f"   control spread {max(ctl_rounds)/min(ctl_rounds):.3f}x")
    med = {k: percentiles(v)[1] for k, v in samples.items()}
    b = med.get(base) if base else None
    print(f"{'arm':>24} {'p10':>9} {'med':>9} {'p90':>9} {'p90/p10':>8} {'vs base':>9}")
    for k, v in samples.items():
        p10, m, p90 = percentiles(v)
        rel = f"{b/m:8.3f}x" if b else "        -"
        print(f"{k:>24} {p10*1e3:>8.3f} {m*1e3:>8.3f} {p90*1e3:>8.3f} {p90/p10:>7.3f}x {rel:>9}")
        out.setdefault(name, {})[k] = dict(p10=p10, median=m, p90=p90)
    out[name + "_control"] = ctl_rounds
    return med


def null_layers_from_dump(path: str, thresh: float = 0.99):
    d = json.load(open(path))
    u = d["usage_by_layer"]
    items = u.items() if isinstance(u, dict) else enumerate(u)
    out = []
    for li, row in items:
        if sum(row) > 0 and row[-1] / sum(row) > thresh:
            out.append(int(li))
    return sorted(out), len(u)


def pin_router(ff: nn.Module, rung: int) -> None:
    """Pin this layer's rung router to a constant choice (weights 0, one-hot bias)."""
    with torch.no_grad():
        ff._nffn_router.w.weight.zero_()
        ff._nffn_router.w.bias.zero_()
        ff._nffn_router.w.bias[rung] = 20.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="q3-4B", choices=list(SCALES))
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--dump", default=os.path.join(OUT, "rungs_v12-t02-L0-d6-warmv10.json"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = torch.device("cuda")
    T = args.tokens
    torch.manual_seed(0)

    null_layers, dump_layers = null_layers_from_dump(args.dump)
    cfg = SCALES[args.scale](vocab_size=VOCAB)
    cfg.lm_head.loss_implementation = LMLossImplementation.fused_linear
    model = cfg.build(init_device="meta")
    model.init_weights(max_seq_len=T, device=torch.device("cpu"))
    model = model.to(torch.bfloat16).to(dev)
    model.train()
    d = model.d_model
    n_layers = len(model.blocks)
    hidden = list(model.blocks.values())[0].feed_forward.w1.out_features
    print(
        f"# GPU {torch.cuda.get_device_name(0)} scale={args.scale} d={d} hidden={hidden} "
        f"n_layers={n_layers} tokens={T}"
    )
    print(
        f"# dump {os.path.basename(args.dump)}: {len(null_layers)}/{dump_layers} layers >99% null "
        f"-> {null_layers}"
    )
    if dump_layers != n_layers:
        print(
            f"# WARNING: dump has {dump_layers} layers, model has {n_layers}; "
            f"null set clipped to the model"
        )
    null_layers = [li for li in null_layers if li < n_layers]
    out = dict(
        scale=args.scale,
        tokens=T,
        n_layers=n_layers,
        hidden=hidden,
        null_layers=null_layers,
        dump=os.path.basename(args.dump),
    )

    # width_multiple=1: the trained checkpoint's rungs are 9728/608/152/38/9/1, which the default
    # rounding-to-8 would collapse into duplicates (38->32, 9->8, 1->8 -> ValueError).
    model.enable_nested_ffn_moe(
        start_layer=0,
        divisors=DIVISORS,
        include_null=True,
        target_cost=0.02,
        width_multiple=1,
    )
    nffn_widths = list(model.blocks["0"].feed_forward._nffn_widths)
    print(f"# rung widths {nffn_widths}  (null = rung {len(nffn_widths)-1})")
    out["widths"] = nffn_widths
    NULL = len(nffn_widths) - 1
    blocks = list(model.blocks.values())
    real_ff = [b.feed_forward for b in blocks]
    holder = model._nested_ffn_moe["holder"]

    # ------------------------------------------------------------- FFN level (one layer)
    print("\n# ---- FFN level: one layer's feed-forward, fwd+bwd ----")
    ff = blocks[n_layers // 2].feed_forward
    orig_fwd = ff._nffn_orig_forward
    x = torch.randn(1, T, d, device=dev, dtype=torch.bfloat16, requires_grad=True)
    g = torch.randn_like(x)

    def ffn_dense():
        x.grad = None
        ff.zero_grad(set_to_none=True)
        orig_fwd(x).backward(g)

    def ffn_routed(rung: int):
        def run():
            x.grad = None
            ff.zero_grad(set_to_none=True)
            holder.begin_forward(collect_loss=True)
            pin_router(ff, rung)
            ff(x).backward(g)

        return run

    def ffn_pruned():
        x.grad = None  # the FFN is gone: nothing runs, nothing is added to the residual

    arms = {
        "CTL": ffn_dense,
        "ffn dense": ffn_dense,
        "ffn routed-null": ffn_routed(NULL),
        "ffn routed-full": ffn_routed(0),
        "ffn pruned (absent)": ffn_pruned,
    }
    for f in arms.values():
        for _ in range(4):
            f()
    s, cr = interleave(arms, args.rounds, args.samples, "CTL")
    med = report("ffn_level", s, cr, out, base="ffn dense")
    mach = med["ffn routed-null"] - med["ffn pruned (absent)"]
    print(
        f"   routing machinery to express 'always null': {mach*1e3:.3f} ms/layer "
        f"= {100*mach/med['ffn dense']:.1f}% of a dense FFN"
    )
    print(
        f"   x {len(null_layers)} always-null layers = {mach*len(null_layers)*1e3:.1f} ms/step "
        f"(CALCULATED from the per-layer MEASURED value)"
    )
    out["ffn_machinery_ms"] = mach
    ffl = {k: count_launches(v) for k, v in arms.items() if k != "CTL"}
    print("   launches/call: " + "  ".join(f"{k}={v:.1f}" for k, v in ffl.items()))
    out["ffn_launches"] = ffl

    # ------------------------------------------------------------- model level
    print("\n# ---- model level: full fwd+bwd step ----")
    labels = torch.full((1, T), -100, dtype=torch.long, device=dev)
    k = max(1, T // 512)
    labels[:, -k:] = torch.randint(0, VOCAB, (1, k), device=dev)
    input_ids = torch.randint(0, VOCAB, (1, T), device=dev)

    def step():
        model.zero_grad(set_to_none=True)
        o = model(input_ids, labels=labels, loss_reduction="sum", return_logits=False)
        o.loss.backward()

    def restore_dense():
        for key, b in list(model.blocks.items()):
            model.blocks[key] = b.inner if isinstance(b, NoFFNBlock) else b
        for b, ff_ in zip(list(model.blocks.values()), real_ff):
            b.feed_forward = ff_
            if hasattr(ff_, "_nffn_orig_forward"):
                ff_.forward = ff_._nffn_orig_forward  # dense: bypass the routed path

    def set_routed(rung_of):
        restore_dense()
        import types

        from olmo_core.nn.nested_ffn_moe import _nested_forward

        for li, b in enumerate(model.blocks.values()):
            ff_ = b.feed_forward
            ff_.forward = types.MethodType(_nested_forward, ff_)
            pin_router(ff_, rung_of(li))

    def set_pruned(layers):
        restore_dense()
        for li in layers:
            model.blocks[str(li)] = NoFFNBlock(model.blocks[str(li)])

    nl = set(null_layers)

    # Each arm reconfigures the model, so they cannot be interleaved cheaply at the module level;
    # instead the SETUP is applied once per round and the timings interleaved within the round.
    configs = {
        "dense (all FFNs)": lambda: restore_dense(),
        "routed: null x%d, full elsewhere"
        % len(null_layers): lambda: set_routed(lambda li: NULL if li in nl else 0),
        "pruned: %d FFNs deleted" % len(null_layers): lambda: set_pruned(null_layers),
        "routed everywhere (all null)": lambda: set_routed(lambda li: NULL),
        "pruned: ALL FFNs deleted": lambda: set_pruned(range(n_layers)),
    }
    msamples = {k: [] for k in configs}
    ctl_rounds = []
    for name, setup in configs.items():
        setup()
        for _ in range(2):
            step()
    for r in range(args.rounds):
        order = list(configs)[r % len(configs) :] + list(configs)[: r % len(configs)]
        for name in order:
            configs[name]()
            msamples[name] += [time_once(step) for _ in range(args.samples)]
        configs["dense (all FFNs)"]()
        ctl_rounds.append(sorted([time_once(step) for _ in range(args.samples)])[args.samples // 2])
    med = report("model_level", msamples, ctl_rounds, out, base="dense (all FFNs)")
    routed_k = [k for k in configs if k.startswith("routed: null")][0]
    pruned_k = [k for k in configs if k.startswith("pruned: %d" % len(null_layers))][0]
    print(
        f"\n   dense -> pruned  : {med['dense (all FFNs)']/med[pruned_k]:.4f}x "
        f"({1e3*(med['dense (all FFNs)']-med[pruned_k]):.1f} ms saved)"
    )
    print(
        f"   dense -> routed  : {med['dense (all FFNs)']/med[routed_k]:.4f}x "
        f"({1e3*(med['dense (all FFNs)']-med[routed_k]):.1f} ms saved)"
    )
    print(
        f"   routed -> pruned : {med[routed_k]/med[pruned_k]:.4f}x "
        f"({1e3*(med[routed_k]-med[pruned_k]):.1f} ms = the per-token machinery)"
    )
    out["model_summary"] = dict(
        dense=med["dense (all FFNs)"], routed=med[routed_k], pruned=med[pruned_k]
    )

    print("\n   launches/step:")
    for name, setup in configs.items():
        setup()
        n = count_launches(step, 2)
        print(f"     {name:>34}: {n:.0f}")
        out.setdefault("model_launches", {})[name] = n
    restore_dense()

    path = args.out or os.path.join(OUT, f"mlskip_static_prune_{args.scale}_{T}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
