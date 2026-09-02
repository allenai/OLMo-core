"""
Where does a Qwen3-4B training step's wall-clock go, and how much of it can FFN routing save?

The per-layer microbenchmark (bench_ffn_routing.py) shows the routed FFN is already ~3.7x faster
than dense in forward+backward. The end-to-end runs barely moved (dense ~6,920 TPS vs routed
5,500-6,750). This script measures a full single-GPU step at the training shape (1 x 6144 tokens,
bf16, full activation checkpointing, fused-linear loss -- what the launcher uses) and attributes
GPU time to attention / FFN / LM head / other, for the dense model and for a routed model driven
with the measured end-of-training rung mix.

Also times the eval shape: 8-way batched decode (8 tokens per forward), where every FFN call is
launch-bound and routing overhead can only hurt.

    srun -w sneetches -p jsteinhardt -q preemptive_high --gres=gpu:H200:1 \
      /data/prasann/conda/envs/corpus-reasoning-olmo/bin/python debug/ffnmoe/profile_train_step.py
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Dict, List

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from olmo_core.config import DType
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingMode,
    TransformerConfig,
)

# End-of-training hard routing from the job logs ([full, 1/4, 1/16, 1/64, null]).
RUNG_MIX = {
    "v3": [0.058, 0.017, 0.020, 0.072, 0.833],
    "v1": [0.055, 0.019, 0.043, 0.080, 0.803],
    "v2": [0.154, 0.001, 0.053, 0.094, 0.698],
}


def scoped(module: torch.nn.Module, name: str) -> None:
    """Wrap a module's forward in a profiler range so its kernels (incl. AC recompute and
    backward launched from inside it) can be attributed."""
    orig = module.forward

    def fwd(*a, **k):
        with record_function(name):
            return orig(*a, **k)

    module.forward = fwd


def force_mix(model, mix: List[float], n_tokens: int, device) -> None:
    """Make every routed layer's router emit a fixed per-token rung with the given marginals."""
    counts = [int(round(f * n_tokens)) for f in mix]
    counts[-1] += n_tokens - sum(counts)
    g = torch.Generator(device="cpu").manual_seed(0)
    for block in model.blocks.values():
        ff = block.feed_forward
        if not hasattr(ff, "_nffn_router"):
            continue
        choice = torch.cat([torch.full((c,), r, dtype=torch.long) for r, c in enumerate(counts)])
        choice = choice[torch.randperm(n_tokens, generator=g)].to(device)
        logits = torch.full((n_tokens, len(mix)), -10.0, device=device)
        logits[torch.arange(n_tokens, device=device), choice] = 10.0
        router = ff._nffn_router

        def fake(x, _logits=logits, _router=router):
            # keep a (tiny) real router matmul so its parameters get gradients like in training
            return _logits[: x.shape[0]] + 0.0 * _router.w(x).sum()

        router.forward = fake


def attribute(prof) -> Dict[str, float]:
    """GPU ms per scope from a profiler run (scopes nest: attn/ffn/lm_head are inside blocks)."""
    out: Dict[str, float] = {}
    for ev in prof.key_averages():
        if ev.key in ("attn", "ffn", "lm_head", "step", "forward", "backward"):
            out[ev.key] = ev.device_time_total / 1e3
    return out


def time_step(fn, iters: int) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / iters


def build(routed: bool, start_layer: int, seq_len: int, device, ac: bool):
    cfg = TransformerConfig.qwen3_4B(
        vocab_size=151936, dtype=DType.bfloat16, attn_backend=AttentionBackendName.flash_2
    )
    cfg.lm_head.loss_implementation = LMLossImplementation.fused_linear
    model = cfg.build(init_device=device)
    model.init_weights(max_seq_len=seq_len, device=torch.device(device))
    if routed:
        model.enable_nested_ffn_moe(start_layer=start_layer, divisors=(1, 4, 16, 64))
    if ac:
        model.apply_activation_checkpointing(TransformerActivationCheckpointingMode.full)
    for block in model.blocks.values():
        scoped(block.attention, "attn")
        scoped(block.feed_forward, "ffn")
    scoped(model.lm_head, "lm_head")
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=6144)
    ap.add_argument("--start-layer", type=int, default=12)
    ap.add_argument("--mix", default="v3", choices=sorted(RUNG_MIX))
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--no-ac", action="store_true")
    ap.add_argument("--out", default="debug/ffnmoe/profile_train_step.json")
    args = ap.parse_args()
    device = "cuda"
    results: Dict[str, Dict] = {}

    for arm in ("dense", "routed", "noffn"):
        routed = arm == "routed"
        model = build(routed, args.start_layer, args.seq_len, device, ac=not args.no_ac)
        holder = model._nested_ffn_moe["holder"] if routed else None
        if routed:
            force_mix(model, RUNG_MIX[args.mix], args.seq_len, device)
        if arm == "noffn":
            # The floor for ANY FFN-side saving: the routed layers' FFN costs nothing at all.
            for i, block in model.blocks.items():
                if int(i) >= args.start_layer:
                    block.feed_forward.forward = lambda x, _o=block.feed_forward.forward: torch.zeros_like(x)
        model.train()

        # In-model single-layer timing, with the model's own tensors: reconciles with
        # bench_ffn_routing.py (which timed a standalone layer) if the routed path behaves.
        ff_last = model.blocks[str(len(model.blocks) - 1)].feed_forward
        xin = torch.randn(1, args.seq_len, model.d_model, device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            if holder is not None:
                holder.begin_forward(collect_loss=False)
            ff_ms = time_step(lambda: ff_last(xin), 20)
        print(f"[{arm}] last layer FFN alone, in-model, fwd no_grad: {ff_ms:.2f} ms", flush=True)
        ids = torch.randint(5, 150000, (1, args.seq_len), device=device)
        labels = ids.clone()

        def train_step():
            if holder is not None:
                holder.begin_forward(collect_loss=True)
            with record_function("forward"):
                out = model(ids, labels=labels)
                loss = out if torch.is_tensor(out) else out.loss
            with record_function("backward"):
                loss.backward()
            model.zero_grad(set_to_none=True)

        ms = time_step(train_step, args.iters)
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            with record_function("step"):
                train_step()
            torch.cuda.synchronize()
        attr = attribute(prof)
        kernels = [e for e in prof.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA]
        gpu_busy = sum(e.device_time_total for e in kernels) / 1e3
        with open(args.out.replace(".json", f".{arm}.kernels.txt"), "w") as f:
            f.write(prof.key_averages().table(sort_by="device_time_total", row_limit=60))
        rec = {"train_step_ms": ms, "gpu_scope_ms": attr, "gpu_busy_ms": gpu_busy, "ffn_layer_fwd_ms": ff_ms}
        if holder is not None:
            rec["routing"] = holder.metrics()
        print(f"[{arm}] train step (1x{args.seq_len}, AC={'off' if args.no_ac else 'full'}): "
              f"{ms:.0f} ms wall, GPU busy {gpu_busy:.0f} ms | scopes: fwd {attr.get('forward', 0):.0f} "
              f"bwd {attr.get('backward', 0):.0f} | attn {attr.get('attn', 0):.0f} ffn {attr.get('ffn', 0):.0f} "
              f"lm_head {attr.get('lm_head', 0):.0f} (fwd+recompute, thread-local)", flush=True)

        # ---- eval shape: a decode step, 8 sequences x 1 token, KV cache not modelled (FFN is
        # per-token so the routing cost is the same; attention over a cache is not the point).
        model.eval()
        dec = torch.randint(5, 150000, (8, 1), device=device)
        with torch.no_grad():
            def decode_step():
                if holder is not None:
                    holder.begin_forward(collect_loss=False)
                model(dec)
            dms = time_step(decode_step, 20)
        rec["decode8_ms"] = dms
        print(f"[{arm}] decode step (8 seqs x 1 token, no cache): {dms:.1f} ms", flush=True)
        results[arm] = rec
        del model
        torch.cuda.empty_cache()

    d, r, z = results["dense"], results["routed"], results["noffn"]
    print(f"\nstep speedup routed/dense: {d['train_step_ms'] / r['train_step_ms']:.2f}x ; "
          f"floor (routed layers' FFN free): {d['train_step_ms'] / z['train_step_ms']:.2f}x ; "
          f"decode: routed {d['decode8_ms'] / r['decode8_ms']:.2f}x, floor {d['decode8_ms'] / z['decode8_ms']:.2f}x")
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "gpu": torch.cuda.get_device_name(0), **results}, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
