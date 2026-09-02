"""
Does the nested-FFN FLOP saving convert into wall-clock?

The 4B contradiction arms give a real FLOP number (mean per-token FFN cost 0.046-0.065 of dense
on the routed layers) but almost no measured speedup. This isolates ONE FeedForward layer at the
training shape and asks where the FLOPs go:

* is the routed path itself slow (gather/scatter, many narrow GEMMs), or
* is the FFN just not a big enough share of the layer for a 20x FFN saving to show up?

Variants measured (forward, and forward+backward):

  dense           the unmodified FeedForward -- the thing we are trying to beat
  current         exactly what ``_nested_forward`` does today: per-rung ``index_select`` ->
                  narrow GEMM -> OUT-OF-PLACE ``index_copy``, then ``out * coef``
  inplace         same structure, but one preallocated output buffer + ``index_copy_`` and an
                  in-place coefficient multiply (removes ~5 full (N,D) allocations per layer)
  sorted          sort tokens by rung ONCE: one gather in, contiguous per-rung views (each GEMM
                  reads contiguous memory instead of an indexed gather), one scatter out
  twoway          collapse the ladder to null vs non-null: a single full-width GEMM over the
                  ~15% of tokens that are not null. This is the cheapest possible *shape* of the
                  idea and upper-bounds what kernel work can buy at this rung distribution.
  ideal_frac      dense FFN on a contiguous prefix of ``1 - p_null`` tokens, no routing overhead
                  at all. The floor: no implementation can beat this at this sparsity.

Run on a GPU node:

    srun -w sneetches -p jsteinhardt -q preemptive_high --gres=gpu:H200:1 --pty \
      /data/prasann/conda/envs/corpus-reasoning-olmo/bin/python \
      debug/ffnmoe/bench_ffn_routing.py
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

# Measured hard-routing fractions at the end of training, from the job logs
# (`[ffn-moe] step 3000: ... rungs=[...]`). Order is [full, 1/4, 1/16, 1/64, null].
RUNG_MIX = {
    "v1": [0.039, 0.000, 0.023, 0.082, 0.856],  # target 0.05, routed from L4
    "v3": [0.058, 0.017, 0.020, 0.072, 0.833],  # target 0.05, routed from L12
    "v2": [0.154, 0.001, 0.053, 0.094, 0.698],  # target 0.15, routed from L4
}


class FF(nn.Module):
    """A SwiGLU FFN with the same three-projection shape as Qwen3's."""

    def __init__(self, d_model: int, hidden: int, dtype: torch.dtype, device: str):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden, bias=False, dtype=dtype, device=device)
        self.w3 = nn.Linear(d_model, hidden, bias=False, dtype=dtype, device=device)
        self.w2 = nn.Linear(hidden, d_model, bias=False, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def slice_forward(self, x: torch.Tensor, width: int) -> torch.Tensor:
        h = F.silu(F.linear(x, self.w1.weight[:width])) * F.linear(x, self.w3.weight[:width])
        return F.linear(h, self.w2.weight[:, :width])


def make_choice(mix: List[float], n: int, device: str) -> torch.Tensor:
    """A fixed per-token rung assignment with the given marginals (shuffled, as routing is)."""
    counts = [int(round(f * n)) for f in mix]
    counts[-1] += n - sum(counts)
    choice = torch.cat(
        [torch.full((c,), g, dtype=torch.long, device=device) for g, c in enumerate(counts)]
    )
    return choice[torch.randperm(n, device=device)]


def v_dense(ff: FF, x, choice, widths, coef):
    return ff(x)


def v_current(ff: FF, x, choice, widths, coef):
    """Today's implementation: out-of-place index_copy per rung, out-of-place coef multiply."""
    out = torch.zeros_like(x)
    for g, width in enumerate(widths):
        idx = (choice == g).nonzero(as_tuple=True)[0]
        if width == 0 or idx.numel() == 0:
            continue
        xs = x.index_select(0, idx)
        ys = ff(xs) if width == widths[0] else ff.slice_forward(xs, width)
        out = out.index_copy(0, idx, ys)
    return out * coef[:, None]


def v_inplace(ff: FF, x, choice, widths, coef):
    """One preallocated buffer, in-place scatter and in-place scaling."""
    out = torch.zeros_like(x)
    for g, width in enumerate(widths):
        if width == 0:
            continue
        idx = (choice == g).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        xs = x.index_select(0, idx)
        ys = ff(xs) if width == widths[0] else ff.slice_forward(xs, width)
        out.index_copy_(0, idx, ys)
    return out.mul_(coef[:, None])


def v_sorted(ff: FF, x, choice, widths, coef):
    """Sort once; each rung's GEMM then reads a contiguous slice. One gather + one scatter."""
    order = torch.argsort(choice)
    xs_all = x.index_select(0, order)
    counts = torch.bincount(choice, minlength=len(widths)).tolist()
    out_all = torch.zeros_like(xs_all)
    start = 0
    for g, width in enumerate(widths):
        end = start + counts[g]
        if width != 0 and counts[g] > 0:
            xs = xs_all[start:end]
            ys = ff(xs) if width == widths[0] else ff.slice_forward(xs, width)
            out_all[start:end] = ys
        start = end
    out = torch.empty_like(out_all)
    out.index_copy_(0, order, out_all)
    return out.mul_(coef[:, None])


def v_twoway(ff: FF, x, choice, widths, coef):
    """Null vs non-null only: a single full-width GEMM over the tokens that get any FFN."""
    keep = (choice != len(widths) - 1).nonzero(as_tuple=True)[0]
    out = torch.zeros_like(x)
    if keep.numel() > 0:
        out.index_copy_(0, keep, ff(x.index_select(0, keep)))
    return out.mul_(coef[:, None])


def v_ideal(ff: FF, x, choice, widths, coef):
    """Dense FFN on a contiguous prefix of the non-null tokens. No routing overhead at all."""
    k = int((choice != len(widths) - 1).sum())
    out = torch.zeros_like(x)
    out[:k] = ff(x[:k])
    return out


def bench(
    fn: Callable, ff: FF, x: torch.Tensor, choice, widths, coef, *, backward: bool, iters: int
) -> float:
    """Median ms per call."""
    times: List[float] = []
    for i in range(iters + 3):
        if backward:
            xi = x.detach().clone().requires_grad_(True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn(ff, xi, choice, widths, coef)
            out.sum().backward()
        else:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                fn(ff, x, choice, widths, coef)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1e3
        if i >= 3:  # warmup
            times.append(dt)
        if backward:
            ff.zero_grad(set_to_none=True)
    times.sort()
    return times[len(times) // 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=2560)
    ap.add_argument("--hidden", type=int, default=9728)
    ap.add_argument("--tokens", type=int, default=12288, help="2 seqs x 6144, the training shape")
    ap.add_argument("--divisors", default="1,4,16,64")
    ap.add_argument("--mix", default="v1", choices=sorted(RUNG_MIX))
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--out", default="debug/ffnmoe/bench_ffn_routing.json")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "GPU required"
    device, dtype = "cuda", torch.bfloat16
    torch.manual_seed(0)

    divisors = [int(d) for d in args.divisors.split(",")]
    widths = [max(8, (args.hidden // d) // 8 * 8) for d in divisors] + [0]
    costs = [w / args.hidden for w in widths]
    mix = RUNG_MIX[args.mix]

    ff = FF(args.d_model, args.hidden, dtype, device)
    x = torch.randn(args.tokens, args.d_model, dtype=dtype, device=device)
    choice = make_choice(mix, args.tokens, device)
    coef = torch.ones(args.tokens, dtype=dtype, device=device)

    flop_cost = sum(f * c for f, c in zip(mix, costs))
    print(f"shape: {args.tokens} tokens x d_model {args.d_model}, hidden {args.hidden}, bf16")
    print(f"widths {widths}  costs {[round(c, 5) for c in costs]}")
    print(f"mix ({args.mix}) {mix}  ->  FLOP cost {flop_cost:.4f} = {1/flop_cost:.1f}x reduction\n")

    variants: List[Tuple[str, Callable]] = [
        ("dense", v_dense),
        ("current", v_current),
        ("inplace", v_inplace),
        ("sorted", v_sorted),
        ("twoway", v_twoway),
        ("ideal_frac", v_ideal),
    ]

    results: Dict[str, Dict[str, float]] = {}
    for mode, backward in (("fwd", False), ("fwd+bwd", True)):
        print(f"--- {mode} (median of {args.iters}) ---")
        base = None
        for name, fn in variants:
            ms = bench(fn, ff, x, choice, widths, coef, backward=backward, iters=args.iters)
            base = ms if name == "dense" else base
            results.setdefault(name, {})[mode] = ms
            print(f"  {name:12s} {ms:8.3f} ms   {base/ms:5.2f}x vs dense")
        print()

    payload = {
        "config": vars(args),
        "widths": widths,
        "costs": costs,
        "mix": mix,
        "flop_cost": flop_cost,
        "flop_speedup": 1.0 / flop_cost,
        "ms": results,
        "gpu": torch.cuda.get_device_name(0),
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
