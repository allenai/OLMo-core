"""Is the whole training step CPU-launch-bound, or only the isolated GDN kernel?

`gdn_bottleneck.py` found the delta-rule sequence spends 51.9% of its wall idle waiting on Python
at B=1. In a real step the CPU also has FFN/attention/head kernels to issue, and GPU work from one
layer can hide the CPU cost of the next -- or not. This measures, per model and context length:

  wall     median fwd+bwd step time (synced each iteration)
  cpu      time for the CPU to issue K iterations back to back with no sync in between
  gpu      sum of kernel self-time from the profiler, i.e. GPU-busy
  idle     1 - gpu/wall

CAVEAT on `cpu` at this scale: with ~6000 launches per step the CUDA queue (~1024 deep) fills, so
the CPU blocks inside `cudaLaunchKernel` and `cpu ~ wall` is expected whether or not the CPU is the
true constraint. `idle` is the trustworthy indicator at model level; `cpu` is trustworthy only for
the short isolated sequences in `gdn_bottleneck.py` / `gdn_levers.py`. Cross-check `cpu` against the
per-block CPU cost measured there instead of reading it as proof on its own. Interleaved arms are not needed here: each row is an independent
configuration, and both quantities in a row come from the same warm process.

    python research/kv_route/flexcompute_40x/step_launch_bound.py
"""

import argparse
import json
import time
from collections import defaultdict

import torch

from olmo_core.nn.attention import Attention
from olmo_core.nn.transformer import TransformerConfig

VOCAB = 248320


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--configs",
        default="4B:8192,4B:8192:compile,4B:16384,0.8B:8192,0.8B:8192:compile",
        help="scale:ctx[:compile]; the compile arms mirror what the trainer does (per-block)",
    )
    ap.add_argument("--out", default="research/kv_route/flexcompute_40x/step_launch_bound.json")
    args = ap.parse_args()
    dev = torch.device("cuda")
    print(f"# GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"\n{'model':>6} {'ctx':>7} {'cmp':>4} {'wall ms':>9} {'cpu ms':>9} {'cpu%':>6} "
        f"{'gpu ms':>9} {'idle%':>6} {'launches':>9}"
    )
    res = {}
    for spec in args.configs.split(","):
        parts = spec.split(":")
        scale, ctx = parts[0], parts[1]
        do_compile = len(parts) > 2 and parts[2] == "compile"
        T = int(ctx)
        builder = {"4B": TransformerConfig.qwen3_5_4B, "0.8B": TransformerConfig.qwen3_5_0_8B}[
            scale
        ]
        torch.manual_seed(0)
        model = builder(vocab_size=VOCAB).build(init_device="cpu").to(dev)
        model.init_weights(max_seq_len=T, device=dev)
        model = model.to(torch.bfloat16)
        if do_compile:
            model.apply_compile()
        ids = torch.randint(0, VOCAB, (1, T), device=dev)
        n_gdn = sum(1 for b in model.blocks.values() if type(b.attention) is not Attention)

        def step():
            model.zero_grad(set_to_none=True)
            model(ids).sum().backward()

        for _ in range(8 if do_compile else 4):
            step()
        torch.cuda.synchronize()
        ts = []
        for _ in range(8):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            step()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        ts.sort()
        wall = ts[len(ts) // 2]

        K = 5
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(K):
            step()
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        cpu, pipelined = (t1 - t0) / K, (t2 - t0) / K

        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                step()
            torch.cuda.synchronize()
        tot, cnt = 0.0, 0
        per_kernel = defaultdict(float)
        for e in prof.key_averages():
            if e.self_device_time_total > 0:
                tot += e.self_device_time_total
                cnt += e.count
                per_kernel[e.key] += e.self_device_time_total
        gpu = tot / 3 / 1e6
        print(
            f"{scale:>6} {T:>7} {'yes' if do_compile else '-':>4} {wall*1e3:>8.1f} "
            f"{cpu*1e3:>8.1f} {100*cpu/pipelined:>5.1f}% "
            f"{gpu*1e3:>8.1f} {100*(1-gpu/wall):>5.1f}% {cnt/3:>8.0f}"
        )
        gdn_keys = [
            k
            for k in per_kernel
            if "delta_rule" in k
            or "wy_repr" in k
            or "chunk_" in k
            or "l2norm" in k
            or "solve_tril" in k
            or "causal_conv1d" in k
            or "merge_16x16" in k
        ]
        gdn_gpu = sum(per_kernel[k] for k in gdn_keys) / 3 / 1e6
        print(
            f"       -> {n_gdn} GDN blocks; fla+conv kernels are {gdn_gpu*1e3:.1f} ms GPU-busy "
            f"({100*gdn_gpu/gpu:.1f}% of GPU-busy, {100*gdn_gpu/wall:.1f}% of wall)"
        )
        res[spec] = dict(
            compiled=do_compile,
            wall=wall,
            cpu=cpu,
            gpu_busy=gpu,
            launches=cnt / 3,
            gdn_gpu_busy=gdn_gpu,
        )
        model = None  # closures above have already run; drop the weights before the next config
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
