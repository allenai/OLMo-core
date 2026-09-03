"""
Theoretical vs measured speedup of the routed nested-width FFN at FIXED cost levels.

Independent of accuracy: random weights, every routed layer's router is FORCED (one-hot bias) to
one rung, so the mean per-token FFN cost is exactly that rung's width fraction: 1 (dense path
through the routed code), 1/4, 1/16, 1/64, 1/256, 8/H, 1/H, 0 (null). Every layer is routed
(the theoretical maximum), so "FFN cost c" means the whole model's FFN runs at c.

Three shapes per (model, seq len, cost):
  train    forward + backward, batch 1 x L tokens, bf16, no activation checkpointing, fused loss
  prefill  forward only (no grad), batch 1 x L
  decode   forward only, batch 32 x 1 token, no KV cache -- the launch-bound regime where FFN is
           the whole layer and routing overhead can only hurt

Models: Qwen3.5 0.8B / 2B / 4B / 9B / 27B and a Qwen3.5-like 70B geometry (d 8192, 80 layers,
FFN 28672, 64/8 heads). Full-model timings where the model fits on one GPU; for every size a
"layer probe" builds the same geometry with 2 and 4 layers and takes per-layer time as the
difference, then predicts the full model as n_layers x per-layer + (embed+head) time. The probe is
validated against the full-model numbers where both exist.

Theoretical FLOP speedup = num_flops_per_token(L) / (num_flops_per_token(L) - ffn_flops * (1 - c)).

    python debug/flop_scaling/bench_ffn_speed.py --out results/flop_scaling/ffn_speed.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, List, Optional

import torch

from olmo_core.config import DType
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.transformer import TransformerConfig

VOCAB = 248320
COST_LEVELS = ["1", "1/4", "1/16", "1/64", "1/256", "8/H", "1/H", "0"]


def factory(name: str, n_layers: Optional[int] = None):
    kw = {"vocab_size": VOCAB, "dtype": DType.bfloat16}
    if n_layers is not None:
        kw["n_layers"] = n_layers
    if name == "q35-0.8B":
        return TransformerConfig.qwen3_5_0_8B(**kw)
    if name == "q35-2B":
        return TransformerConfig.qwen3_5_2B(**kw)
    if name == "q35-4B":
        return TransformerConfig.qwen3_5_4B(**kw)
    if name == "q35-9B":
        return TransformerConfig.qwen3_5_9B(**kw)
    if name == "q35-27B":
        return TransformerConfig.qwen3_5_27B(**kw)
    if name == "q35like-70B":
        return TransformerConfig.qwen3_5_like(
            d_model=8192, n_layers=n_layers or 80, n_heads=64, n_kv_heads=8, head_dim=128,
            intermediate_size=28672, vocab_size=VOCAB, dtype=DType.bfloat16,
        )
    raise SystemExit(f"unknown model {name}")


FULL_LAYERS = {"q35-0.8B": None, "q35-2B": None, "q35-4B": None, "q35-9B": None, "q35-27B": None, "q35like-70B": 80}


def build(name: str, n_layers: Optional[int], device: str, attn_backend: str):
    cfg = factory(name, n_layers)
    cfg.lm_head.loss_implementation = LMLossImplementation.fused_linear
    try:
        cfg.block.attention.backend = AttentionBackendName(attn_backend)  # type: ignore[attr-defined]
    except Exception:
        pass
    model = cfg.build(init_device=device)
    model.init_weights(max_seq_len=65536, device=torch.device(device))
    model.to(torch.bfloat16)
    return cfg, model


def hidden_size(model) -> int:
    for b in model.blocks.values():
        return int(b.feed_forward.w1.out_features)
    raise RuntimeError("no feed_forward")


def enable_routing(model, H: int):
    divisors = [1, 4, 16, 64, 256, H / 8, H]
    model.enable_nested_ffn_moe(start_layer=0, divisors=divisors, include_null=True, width_multiple=1,
                                target_cost=1.0, explore_prob=0.0)
    ffs = [b.feed_forward for b in model.blocks.values() if hasattr(b.feed_forward, "_nffn_router")]
    widths = list(ffs[0]._nffn_widths)
    return ffs, widths


def force_rung(ffs, rung: int):
    with torch.no_grad():
        for ff in ffs:
            ff._nffn_router.w.weight.zero_()
            ff._nffn_router.w.bias.zero_()
            ff._nffn_router.w.bias[rung] = 30.0


def cost_of(level: str, widths: List[int]) -> float:
    H = widths[0]
    if level == "0":
        return 0.0
    num, _, den = level.partition("/")
    if den == "":
        return 1.0
    if den == "H":
        return int(num) / H
    return int(num) / int(den)


def rung_for(level: str, widths: List[int]) -> int:
    if level == "0":
        return len(widths)  # null rung (appended last)
    target_w = max(1, round(cost_of(level, widths) * widths[0]))
    return min(range(len(widths)), key=lambda i: abs(widths[i] - target_w))


def timeit(fn, iters: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        ev0.record(); fn(); ev1.record(); torch.cuda.synchronize(); ts.append(ev0.elapsed_time(ev1))
    ts.sort()
    return ts[len(ts) // 2]


def shapes(model, L: int, device: str, decode_batch: int):
    x = torch.randint(0, VOCAB, (1, L), device=device)
    xd = torch.randint(0, VOCAB, (decode_batch, 1), device=device)

    def train():
        out = model(x, labels=x)
        loss = out.loss if hasattr(out, "loss") else out
        loss.backward()
        model.zero_grad(set_to_none=True)

    def prefill():
        with torch.no_grad():
            model(x)

    def decode():
        with torch.no_grad():
            model(xd)

    return {"train": train, "prefill": prefill, "decode": decode}


def theoretical(cfg_model, L: int, c: float, n_layers_full: int, n_layers_built: int):
    """Model-wide FLOP speedup for full model at seq L when every layer's FFN runs at cost c."""
    m = cfg_model
    fpt = float(m.num_flops_per_token(L))
    ffn = float(sum(b.feed_forward.num_flops_per_token(1) for b in m.blocks.values()))
    # scale per-layer quantities to the full depth when a probe (fewer layers) is used
    if n_layers_built != n_layers_full:
        embed_head = fpt - float(sum(b.num_flops_per_token(L) for b in m.blocks.values())) if hasattr(next(iter(m.blocks.values())), "num_flops_per_token") else 0.0
        per_layer = (fpt - embed_head) / n_layers_built
        fpt = embed_head + per_layer * n_layers_full
        ffn = ffn / n_layers_built * n_layers_full
    return fpt / (fpt - ffn * (1.0 - c)), ffn / fpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="q35-0.8B,q35-2B,q35-4B,q35-9B,q35-27B,q35like-70B")
    ap.add_argument("--seq-lens", default="2048,8192,32768")
    ap.add_argument("--levels", default=",".join(COST_LEVELS))
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--decode-batch", type=int, default=32)
    ap.add_argument("--attn-backend", default="flash_2")
    ap.add_argument("--full-max-params-b", type=float, default=10.0, help="full-model timings only below this size (B)")
    ap.add_argument("--out", default="results/flop_scaling/ffn_speed.json")
    args = ap.parse_args()
    device = "cuda"
    torch.manual_seed(0)
    results = {"gpu": torch.cuda.get_device_name(0), "runs": []}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    levels = args.levels.split(",")

    for name in args.models.split(","):
        cfg_full = factory(name)
        n_full = cfg_full.n_layers
        # ---- layer probe: 2 and 4 layers ----
        probe = {}
        for nl in (2, 4):
            _, model = build(name, nl, device, args.attn_backend)
            H = hidden_size(model)
            ffs, widths = enable_routing(model, H)
            nparams = sum(p.numel() for p in model.parameters())
            print(f"[{name} probe {nl}L] params {nparams/1e9:.2f}B H={H} widths={widths}", flush=True)
            for L in [int(s) for s in args.seq_lens.split(",")]:
                fns = shapes(model, L, device, args.decode_batch)
                for lvl in levels:
                    r = rung_for(lvl, widths); force_rung(ffs, r); c = widths[r] / H if r < len(widths) else 0.0
                    for shape, fn in fns.items():
                        if shape == "decode" and L != int(args.seq_lens.split(",")[0]):
                            continue
                        model.train(shape == "train")
                        try:
                            ms = timeit(fn, args.iters)
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache(); ms = float("nan")
                        probe[(nl, L, lvl, shape)] = ms
                        print(f"  {nl}L L={L} {shape:7s} cost={lvl:5s} (c={c:.5f}) {ms:9.2f} ms", flush=True)
            del model, ffs; torch.cuda.empty_cache()
        # per-layer + overhead from the two probes
        for L in [int(s) for s in args.seq_lens.split(",")]:
            for lvl in levels:
                for shape in ("train", "prefill", "decode"):
                    if (2, L, lvl, shape) not in probe:
                        continue
                    t2, t4 = probe[(2, L, lvl, shape)], probe[(4, L, lvl, shape)]
                    per_layer = (t4 - t2) / 2.0
                    overhead = t2 - 2 * per_layer
                    predicted = overhead + per_layer * n_full
                    c = cost_of(lvl, [H])
                    results["runs"].append({"model": name, "n_layers": n_full, "H": H, "L": L, "level": lvl, "cost": c,
                                            "shape": shape, "probe_per_layer_ms": per_layer, "probe_overhead_ms": overhead,
                                            "predicted_full_ms": predicted})
        # ---- full model where it fits ----
        if sum(p.numel() for p in cfg_full.build(init_device="meta").parameters()) / 1e9 <= args.full_max_params_b:
            _, model = build(name, None, device, args.attn_backend)
            H = hidden_size(model); ffs, widths = enable_routing(model, H)
            for L in [int(s) for s in args.seq_lens.split(",")]:
                fns = shapes(model, L, device, args.decode_batch)
                for lvl in levels:
                    r = rung_for(lvl, widths); force_rung(ffs, r); c = widths[r] / H if r < len(widths) else 0.0
                    th, ffn_frac = theoretical(model, L, c, n_full, n_full)
                    for shape, fn in fns.items():
                        if shape == "decode" and L != int(args.seq_lens.split(",")[0]):
                            continue
                        model.train(shape == "train")
                        try:
                            ms = timeit(fn, args.iters)
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache(); ms = float("nan")
                        for rr in results["runs"]:
                            if rr["model"] == name and rr["L"] == L and rr["level"] == lvl and rr["shape"] == shape:
                                rr["full_ms"] = ms; rr["theoretical_speedup"] = th; rr["ffn_flop_frac"] = ffn_frac
                        print(f"  FULL L={L} {shape:7s} cost={lvl:5s} {ms:9.2f} ms  (theory x{th:.2f}, ffn frac {ffn_frac:.2f})", flush=True)
            del model, ffs; torch.cuda.empty_cache()
        else:
            # theoretical numbers from the 4-layer probe's config, scaled to full depth
            _, model = build(name, 4, device, args.attn_backend)
            for rr in results["runs"]:
                if rr["model"] == name and "theoretical_speedup" not in rr:
                    th, ffn_frac = theoretical(model, rr["L"], rr["cost"], n_full, 4)
                    rr["theoretical_speedup"] = th; rr["ffn_flop_frac"] = ffn_frac
            del model; torch.cuda.empty_cache()
        json.dump(results, open(args.out, "w"), indent=1)
        print(f"[{name}] written {args.out}", flush=True)

    # ---- summary table: measured speedup vs cost=1 (routed code path) and vs theory ----
    lines = ["| model | L | shape | cost | theory x | full x | probe x | full ms | probe-pred ms |", "|---|---|---|---|---|---|---|---|---|"]
    by = {}
    for rr in results["runs"]:
        by[(rr["model"], rr["L"], rr["shape"], rr["level"])] = rr
    for rr in results["runs"]:
        base = by.get((rr["model"], rr["L"], rr["shape"], "1"))
        fx = (base["full_ms"] / rr["full_ms"]) if base and "full_ms" in base and "full_ms" in rr and rr["full_ms"] == rr["full_ms"] and rr["full_ms"] > 0 else None
        px = (base["predicted_full_ms"] / rr["predicted_full_ms"]) if base and rr["predicted_full_ms"] > 0 else None
        lines.append(f"| {rr['model']} | {rr['L']} | {rr['shape']} | {rr['level']} | {rr.get('theoretical_speedup', float('nan')):.2f} | "
                     f"{'-' if fx is None else f'{fx:.2f}'} | {'-' if px is None else f'{px:.2f}'} | "
                     f"{'-' if 'full_ms' not in rr else f'{rr['full_ms']:.1f}'} | {rr['predicted_full_ms']:.1f} |")
    open(args.out.replace(".json", ".md"), "w").write("# Routed-FFN speed: theoretical vs measured (%s)\n\n" % results["gpu"] + "\n".join(lines) + "\n")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
