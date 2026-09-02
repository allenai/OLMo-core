"""
Why is the REAL routed FeedForward no faster than dense inside the model, when the standalone
re-implementation in bench_ffn_routing.py is 3.6x faster?

Profiles one olmo_core FeedForward (Qwen3-4B shape, bf16) with install_nested_ffn_moe applied,
driven with the v3 rung mix, at the training token count, and prints the kernel table -- so the
extra GPU work (or the CPU launch/sync floor) is visible by name.

    srun -w horton -p jsteinhardt -q preemptive_high --gres=gpu:H200:1 \
      /data/prasann/conda/envs/corpus-reasoning-olmo/bin/python debug/ffnmoe/profile_routed_layer.py
"""

import argparse
import time

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile

from olmo_core.config import DType
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.nested_ffn_moe import NestedFFNHolder, install_nested_ffn_moe, resolve_rung_widths

MIX = [0.058, 0.017, 0.020, 0.072, 0.833]


def timeit(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / iters


def gpu_span(fn, iters=20):
    """GPU-timeline span via CUDA events (includes idle gaps while the CPU launches)."""
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=6144)
    args = ap.parse_args()
    dev = "cuda"
    d_model, hidden = 2560, 9728
    ff = FeedForwardConfig(hidden_size=hidden, bias=False, dtype=DType.bfloat16).build(
        d_model=d_model, init_device=dev
    )
    dense_forward = ff.forward
    x = torch.randn(1, args.tokens, d_model, device=dev, dtype=torch.bfloat16)

    print(f"dense  fwd {timeit(lambda: dense_forward(x)):.3f} ms wall")

    widths, costs = resolve_rung_widths(hidden, (1, 4, 16, 64))
    holder = NestedFFNHolder(costs)
    blocks = nn.ModuleDict({"0": nn.Module()})
    blocks["0"].feed_forward = ff
    install_nested_ffn_moe(blocks, holder, start_layer=0, widths=widths, costs=costs, init_device=dev)
    ff.to(dev)
    print("router dtype", ff._nffn_router.w.weight.dtype, "gain dtype", ff._nffn_gain.dtype)

    n = args.tokens
    counts = [int(round(f * n)) for f in MIX]
    counts[-1] += n - sum(counts)
    choice = torch.cat([torch.full((c,), r, dtype=torch.long) for r, c in enumerate(counts)])
    choice = choice[torch.randperm(n, generator=torch.Generator().manual_seed(0))].to(dev)
    logits = torch.full((n, len(MIX)), -10.0, device=dev)
    logits[torch.arange(n, device=dev), choice] = 10.0
    ff._nffn_router.forward = lambda t, _l=logits: _l[: t.shape[0]]

    def routed_fwd():
        holder.begin_forward(collect_loss=False)
        with torch.no_grad():
            return ff(x)

    ff.eval()
    print(f"routed fwd {timeit(routed_fwd):.3f} ms wall, {gpu_span(routed_fwd):.3f} ms GPU span")
    with torch.no_grad():
        print(f"dense  fwd {gpu_span(lambda: dense_forward(x)):.3f} ms GPU span")

    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        routed_fwd()
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    # fwd+bwd, as in training
    ff.train()
    xg = x.detach().clone().requires_grad_(True)

    def routed_fb():
        holder.begin_forward(collect_loss=True)
        ff(xg).sum().backward()
        ff.zero_grad(set_to_none=True)

    def dense_fb():
        dense_forward(xg).sum().backward()
        ff.zero_grad(set_to_none=True)

    print(f"dense  fwd+bwd {timeit(dense_fb, 10):.3f} ms wall")
    print(f"routed fwd+bwd {timeit(routed_fb, 10):.3f} ms wall")
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        routed_fb()
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))


if __name__ == "__main__":
    main()
