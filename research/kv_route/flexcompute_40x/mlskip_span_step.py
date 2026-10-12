"""Span skipping on a REAL 4B step: launches, wall-clock and GPU-busy, span-K vs per-block.

`mlskip_router_overhead.py` prices one decision and a short stack. This is the same question at
the level the answer has to hold at: a full Qwen3.5-4B forward+backward with block skipping
enabled on all 32 blocks, at a MATCHED mean keep fraction, with the span varied.

Matching the keep fraction matters: the deployed block-skip path is mask-based (the block runs on
every token and skipped tokens' outputs are discarded), so keep affects the mixer's key set and the
router's own work but NOT the block's FLOPs. Holding it fixed therefore isolates exactly the thing
spans change -- how many times per step the routing prologue runs.

The routers are pinned (a patched ``router_logits``) rather than trained, so every arm makes the
same decision on the same tokens and the comparison is not confounded by a different keep pattern.

    python research/kv_route/flexcompute_40x/mlskip_span_step.py --scale 4B --tokens 8192 --keep 1.0,0.5
"""

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Dict, List

import torch

import olmo_core.nn.router_fn as rf
from olmo_core.nn.block_skip import set_block_skip_span
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.transformer import TransformerConfig

OUT = os.path.dirname(os.path.abspath(__file__))
VOCAB = 248320
SCALES = {
    "4B": TransformerConfig.qwen3_5_4B,
    "0.8B": TransformerConfig.qwen3_5_0_8B,
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
    if setup is not None:
        setup()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def gpu_busy_and_launches(fn, iters: int = 3):
    """(GPU-busy seconds per call, kernel launches per call) from the CUDA profiler."""
    from torch.profiler import ProfilerActivity, profile

    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    tot, cnt = 0.0, 0
    per_kernel = defaultdict(float)
    for e in prof.key_averages():
        if e.self_device_time_total > 0:
            tot += e.self_device_time_total
            cnt += e.count
            per_kernel[e.key] += e.self_device_time_total
    return tot / iters / 1e6, cnt / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="4B", choices=list(SCALES))
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--keep", default="1.0,0.5")
    ap.add_argument("--spans", default="1,2,4,8")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = torch.device("cuda")
    T = args.tokens
    spans = [int(x) for x in args.spans.split(",")]
    keeps = [float(x) for x in args.keep.split(",")]

    torch.manual_seed(0)
    cfg = SCALES[args.scale](vocab_size=VOCAB)
    cfg.lm_head.loss_implementation = LMLossImplementation.fused_linear
    model = cfg.build(init_device="meta")
    model.init_weights(max_seq_len=T, device=torch.device("cpu"))
    model = model.to(torch.bfloat16).to(dev)
    model.train()
    n_layers = len(model.blocks)
    print(
        f"# GPU {torch.cuda.get_device_name(0)} scale={args.scale} n_layers={n_layers} "
        f"tokens={T} params={sum(p.numel() for p in model.parameters())/1e9:.3f}B"
    )

    model.enable_block_skip(start_layer=0, target=0.5)
    holder = model._block_skip["holder"]
    routed = model._block_skip["routed"]
    print(f"# routed blocks: {len(routed)}")

    labels = torch.full((1, T), -100, dtype=torch.long, device=dev)
    k = max(1, T // 512)
    labels[:, -k:] = torch.randint(0, VOCAB, (1, k), device=dev)
    input_ids = torch.randint(0, VOCAB, (1, T), device=dev)

    def step():
        model.zero_grad(set_to_none=True)
        o = model(input_ids, labels=labels, loss_reduction="sum", return_logits=False)
        o.loss.backward()

    # ---- pin the decision so every arm skips exactly the same tokens in exactly the same blocks
    orig_logits = rf.router_logits
    pinned = {"pattern": None}

    def patched(h, router, eps=1e-6):
        out = orig_logits(h, router, eps)
        pat = pinned["pattern"]
        return out if pat is None else out * 0.0 + pat.to(out.dtype)[..., None]

    out: dict = dict(scale=args.scale, tokens=T, n_layers=n_layers, spans=spans, keeps=keeps)
    rows = []
    rf.router_logits = patched
    try:
        for keep_frac in keeps:
            g = torch.Generator(device=dev).manual_seed(1234)
            r = torch.rand(1, T, device=dev, generator=g)
            pinned["pattern"] = torch.where(r < keep_frac, 5.0, -5.0)

            arms: Dict[str, object] = {}
            setups: Dict[str, object] = {}

            def mk_setup(span, lazy, enabled=True):
                def setup():
                    holder.enabled = enabled
                    set_block_skip_span(model, span)
                    holder.lazy_span_mix = lazy

                return setup

            arms["no routing"] = step
            setups["no routing"] = mk_setup(1, False, enabled=False)
            for sp in spans:
                if sp == 1:
                    arms["span=1 (per-block)"] = step
                    setups["span=1 (per-block)"] = mk_setup(1, False)
                else:
                    arms[f"span={sp}"] = step
                    setups[f"span={sp}"] = mk_setup(sp, False)
                    arms[f"span={sp}+lazymix"] = step
                    setups[f"span={sp}+lazymix"] = mk_setup(sp, True)
            arms["CTL"] = step
            setups["CTL"] = setups["no routing"]

            for lbl in arms:
                setups[lbl]()
                for _ in range(2):
                    step()

            labels_ = [x for x in arms if x != "CTL"]
            samples = {x: [] for x in arms}
            ctl_rounds = []
            for rd in range(args.rounds):
                order = labels_[rd % len(labels_) :] + labels_[: rd % len(labels_)]
                c = [time_once(step, setups["CTL"]) for _ in range(args.samples)]
                for lbl in order:
                    samples[lbl] += [time_once(step, setups[lbl]) for _ in range(args.samples)]
                c += [time_once(step, setups["CTL"]) for _ in range(args.samples)]
                samples["CTL"] += c
                ctl_rounds.append(sorted(c)[len(c) // 2])

            print(
                f"\n## keep = {keep_frac}   (mask-based skipping: keep changes the key set and "
                f"the router's work, not the block's FLOPs)"
            )
            print("   control rounds (ms): " + " ".join(f"{t*1e3:.1f}" for t in ctl_rounds))
            print(f"   control spread {max(ctl_rounds)/min(ctl_rounds):.3f}x")
            base = percentiles(samples["no routing"])[1]
            print(
                f"{'arm':>22} {'p10':>9} {'med':>9} {'p90':>9} {'vs no-routing':>14} "
                f"{'launches':>9} {'gpu-busy':>9} {'idle':>7}"
            )
            for lbl in arms:
                if lbl == "CTL":
                    continue
                p10, med, p90 = percentiles(samples[lbl])
                setups[lbl]()
                gpu, nlaunch = gpu_busy_and_launches(step)
                print(
                    f"{lbl:>22} {p10*1e3:>8.1f} {med*1e3:>8.1f} {p90*1e3:>8.1f} "
                    f"{med/base:>13.4f}x {nlaunch:>9.0f} {gpu*1e3:>8.1f} "
                    f"{100*(1-gpu/med):>6.1f}%"
                )
                rows.append(
                    dict(
                        keep=keep_frac,
                        arm=lbl,
                        p10=p10,
                        median=med,
                        p90=p90,
                        launches=nlaunch,
                        gpu_busy=gpu,
                        vs_base=med / base,
                    )
                )
            ctl_med = percentiles(samples["CTL"])[1]
            rows.append(dict(keep=keep_frac, arm="CTL", median=ctl_med, control_rounds=ctl_rounds))
            n_dec = {sp: (len(routed) + sp - 1) // sp for sp in spans}
            print("   decisions per step: " + "  ".join(f"span{sp}={n}" for sp, n in n_dec.items()))
    finally:
        rf.router_logits = orig_logits
        holder.enabled = True
        set_block_skip_span(model, 1)

    out["rows"] = rows
    path = args.out or os.path.join(OUT, f"mlskip_span_step_{args.scale}_{T}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
