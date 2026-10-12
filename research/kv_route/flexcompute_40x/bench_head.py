"""
Measure what gating the LM head to loss positions actually buys, in wall-clock and memory.

Two benchmarks:
  * ``head``  -- the LM head alone (final norm + vocab projection + CE), fwd+bwd. Isolates the
    term, and checks the gated path returns the same loss as the dense one under BOTH loss
    implementations (the fused Liger kernel is GPU-only, so this is where it gets verified).
  * ``model`` -- a full Qwen3.5 fwd+bwd step, which is the number that matters and the one the
    FLOP share predicts.

    python research/kv_route/flexcompute_40x/bench_head.py --bench head --scale 4B
    python research/kv_route/flexcompute_40x/bench_head.py --bench model --scale 4B --lengths 2048,8192,32768
"""

import argparse
import json
import os
import time

import torch

from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.nn.lm_head import LMHeadConfig, LMLossImplementation
from olmo_core.nn.transformer import TransformerConfig

VOCAB = 248320
OUT = os.path.dirname(os.path.abspath(__file__))
SCALES = {
    "0.8B": TransformerConfig.qwen3_5_0_8B,
    "4B": TransformerConfig.qwen3_5_4B,
    "qwen3-4B": TransformerConfig.qwen3_4B,
}
# Loss densities measured on real task shards (research/kv_route/flexcompute_40x/README.md).
DENSITIES = {2048: 0.0064, 8192: 0.0064, 16384: 0.0018, 32768: 0.0018, 65536: 0.0018}


def make_labels(B, T, density, device):
    """Labels masked to a supervised suffix, matching how our SFT shards are built."""
    n_keep = max(1, int(round(T * density)))
    labels = torch.full((B, T), -100, dtype=torch.long, device=device)
    labels[:, -n_keep:] = torch.randint(0, VOCAB, (B, n_keep), device=device)
    return labels, n_keep


def timed(fn, warmup=3, iters=10):
    """Median wall-clock seconds and peak allocated GB for ``fn``."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2], torch.cuda.max_memory_allocated() / 2**30


def bench_head(args, device, results):
    d_model = SCALES[args.scale](vocab_size=VOCAB).d_model
    for loss_impl in (LMLossImplementation.default, LMLossImplementation.fused_linear):
        for T in args.lengths:
            density = DENSITIES.get(T, 0.0018)
            B = 1
            labels, n_keep = make_labels(B, T, density, device)
            x = torch.randn(B, T, d_model, device=device, dtype=torch.bfloat16)
            row = {
                "bench": "head",
                "loss_impl": str(loss_impl),
                "ctx": T,
                "density": density,
                "n_keep": n_keep,
            }
            losses = {}
            for sparse in (False, True):
                cfg = LMHeadConfig(
                    layer_norm=LayerNormConfig(name="rms"),
                    loss_implementation=loss_impl,
                    sparse_loss_positions=sparse,
                )
                torch.manual_seed(0)
                head = cfg.build(d_model=d_model, vocab_size=VOCAB, init_device="cpu").to(device)
                head = head.to(torch.bfloat16)

                # Bind by default argument: the loop deletes `head` at the end of each pass, so
                # a late-bound closure would reference an unbound name.
                def step(head=head, x=x, labels=labels):
                    head.zero_grad(set_to_none=True)
                    out = head(x, labels=labels, loss_reduction="sum", return_logits=False)
                    out.loss.backward()
                    return out.loss

                try:
                    losses[sparse] = float(step().detach())
                    t, mem = timed(step, iters=args.iters)
                except torch.cuda.OutOfMemoryError:
                    t, mem = float("nan"), float("nan")
                    torch.cuda.empty_cache()
                row["dense_s" if not sparse else "gated_s"] = t
                row["dense_gb" if not sparse else "gated_gb"] = mem
                del head
                torch.cuda.empty_cache()
            row["speedup"] = (
                row["dense_s"] / row["gated_s"]
                if row["gated_s"] == row["gated_s"]
                else float("nan")
            )
            row["loss_dense"], row["loss_gated"] = losses.get(False), losses.get(True)
            row["loss_rel_err"] = (
                abs(row["loss_gated"] - row["loss_dense"]) / max(abs(row["loss_dense"]), 1e-9)
                if None not in (row["loss_dense"], row["loss_gated"])
                else None
            )
            results.append(row)
            print(json.dumps(row), flush=True)


def bench_model(args, device, results):
    """Full fwd+bwd step, dense vs gated.

    Builds the model ONCE and flips the flag between measurements, interleaving the two arms
    A/B/A/B. Building a fresh model per arm -- the first version of this -- left the caching
    allocator in a different state for each and produced ratios that disagreed with themselves
    (2.09x at 2k next to 1.04x at 8k on the same model).
    """
    cfg = SCALES[args.scale](vocab_size=VOCAB)
    cfg.lm_head.loss_implementation = LMLossImplementation.fused_linear
    torch.manual_seed(0)
    model = cfg.build(init_device="cpu").to(device)
    model.init_weights(max_seq_len=max(args.lengths), device=device)
    model = model.to(torch.bfloat16)
    model.train()

    for T in args.lengths:
        density = DENSITIES.get(T, 0.0018)
        labels, n_keep = make_labels(1, T, density, device)
        input_ids = torch.randint(0, VOCAB, (1, T), device=device)

        def step():
            model.zero_grad(set_to_none=True)
            out = model(input_ids, labels=labels, loss_reduction="sum", return_logits=False)
            out.loss.backward()
            return out.loss

        def arm(sparse):
            model.lm_head._sparse_loss_positions = sparse
            return step()

        row = {
            "bench": "model",
            "scale": args.scale,
            "ctx": T,
            "density": density,
            "n_keep": n_keep,
            "iters": args.iters,
        }
        try:
            row["loss_dense"] = float(arm(False).detach())
            row["loss_gated"] = float(arm(True).detach())
            for _ in range(3):  # shared warmup, both arms
                arm(False)
                arm(True)
            ts = {False: [], True: []}
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            for _ in range(args.iters):  # interleaved A/B so drift hits both arms equally
                for sparse in (False, True):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    arm(sparse)
                    torch.cuda.synchronize()
                    ts[sparse].append(time.perf_counter() - t0)
            for sparse in (False, True):
                v = sorted(ts[sparse])
                k = "gated" if sparse else "dense"
                row[f"{k}_s"] = v[len(v) // 2]
                row[f"{k}_p10"], row[f"{k}_p90"] = v[len(v) // 10], v[(9 * len(v)) // 10]
            row["speedup"] = row["dense_s"] / row["gated_s"]
            row["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
            row["loss_rel_err"] = abs(row["loss_gated"] - row["loss_dense"]) / max(
                abs(row["loss_dense"]), 1e-9
            )
        except torch.cuda.OutOfMemoryError:
            row["error"] = "OOM"
            torch.cuda.empty_cache()
        results.append(row)
        print(json.dumps(row), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["head", "model"], default="head")
    ap.add_argument("--scale", default="4B", choices=list(SCALES))
    ap.add_argument("--lengths", default="2048,8192,32768")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.lengths = [int(x) for x in args.lengths.split(",")]

    assert torch.cuda.is_available(), "this benchmark needs a GPU"
    device = torch.device("cuda")
    print(f"# {torch.cuda.get_device_name(0)}  torch {torch.__version__}", flush=True)

    results = []
    if args.bench == "head":
        bench_head(args, device, results)
    else:
        bench_model(args, device, results)

    out = args.out or os.path.join(OUT, f"bench_{args.bench}_{args.scale}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"# wrote {out}", flush=True)


if __name__ == "__main__":
    main()
