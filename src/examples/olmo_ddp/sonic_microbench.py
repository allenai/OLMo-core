"""
Kernel-level microbenchmark: SonicMoE vs the grouped-MM routed-expert path.

Times the routed-expert section of the no-EP MoE block (permute -> experts -> weighted
unpermute for grouped-MM, or the fused SonicMoE call) in isolation, with random top-k
routing, for a list of ``T,H,I,E,K`` shapes. Reports forward and forward+backward latency,
TFLOPS using SonicMoE's own accounting (``6*T*I*H*K`` forward, ``18*T*I*H*K`` fwd+bwd for
GLU experts), and peak allocated memory, so numbers are directly comparable with
``sonic-moe/benchmarks/moe-cute.py``.

Example::

    PYTHONPATH=src python src/examples/olmo_ddp/sonic_microbench.py \\
        --shapes 65536,320,544,512,16 32768,2048,1024,64,8 --backends grouped_mm sonic
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable

import torch

from olmo_core.config import DType
from olmo_core.nn.moe.utils import moe_permute_no_compile, moe_unpermute_no_compile
from olmo_core.nn.moe.v2.routed_experts import (
    RoutedExperts,
    RoutedExpertsBackend,
    requires_host_side_split_sizes,
)


def _random_routing(T: int, E: int, K: int, device: torch.device):
    logits = torch.randn(T, E, device=device)
    weights, indices = logits.topk(K, dim=-1)
    weights = torch.softmax(weights.float(), dim=-1)
    return indices.to(torch.int32), weights


def _grouped_fn(
    experts: RoutedExperts, x: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor
):
    T, K = indices.shape
    counts = torch.bincount(indices.reshape(-1).long(), minlength=experts.num_experts)
    if requires_host_side_split_sizes():
        counts = counts.cpu()

    def fn() -> torch.Tensor:
        permuted, row_id_map = moe_permute_no_compile(
            inp=x, routing_map=indices, num_out_tokens=T * K, map_type="index"
        )
        h = experts(permuted, counts)
        return moe_unpermute_no_compile(
            inp=h,
            row_id_map=row_id_map,
            restore_shape=x.shape,
            map_type="index",
            merging_probs=weights,
        )

    return fn


def _sonic_fn(
    experts: RoutedExperts, x: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor
):
    from olmo_core.nn.moe.v2.sonic import sonic_moe_forward

    def fn() -> torch.Tensor:
        return sonic_moe_forward(x, indices, weights, experts)

    return fn


def _time(fn: Callable[[], None], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def bench(shape: tuple[int, ...], backend: str, warmup: int, iters: int, seed: int) -> dict:
    T, H, I, E, K = shape
    device = torch.device("cuda")
    torch.manual_seed(seed)
    experts = RoutedExperts(
        d_model=H,
        hidden_size=I,
        num_experts=E,
        bias=False,
        dtype=DType.bfloat16,
        backend=RoutedExpertsBackend(backend),
        init_device="cuda",
    )
    with torch.no_grad():
        for p in experts.parameters():
            p.normal_(std=0.02)
    x = torch.randn(T, H, device=device, dtype=torch.bfloat16, requires_grad=True)
    indices, weights = _random_routing(T, E, K, device)
    weights = weights.to(torch.bfloat16).requires_grad_(True)

    fn = (
        _grouped_fn(experts, x, indices, weights)
        if backend == "grouped_mm"
        else _sonic_fn(experts, x, indices, weights)
    )

    def fwd():
        with torch.no_grad():
            fn()

    grad_tensors = [x, weights, *experts.parameters()]

    def fwd_bwd():
        out = fn()
        out.float().sum().backward()
        for t in grad_tensors:
            t.grad = None

    fwd_ms = _time(fwd, warmup, iters)
    # Warm up fwd+bwd before measuring memory so JIT autotuning workspaces are excluded.
    for _ in range(warmup):
        fwd_bwd()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated()
    fwd_bwd()
    torch.cuda.synchronize()
    peak_mem_gib = (torch.cuda.max_memory_allocated() - base_mem) / 2**30
    fwd_bwd_ms = _time(fwd_bwd, 0, iters)

    flops_fwd = 6 * T * I * H * K
    flops_fwd_bwd = 18 * T * I * H * K
    result = {
        "backend": backend,
        "T": T,
        "H": H,
        "I": I,
        "E": E,
        "K": K,
        "rows_per_expert": T * K / E,
        "fwd_ms": round(fwd_ms, 3),
        "fwd_tflops": round(flops_fwd / (fwd_ms * 1e9), 1),
        "fwd_bwd_ms": round(fwd_bwd_ms, 3),
        "fwd_bwd_tflops": round(flops_fwd_bwd / (fwd_bwd_ms * 1e9), 1),
        "bwd_ms": round(fwd_bwd_ms - fwd_ms, 3),
        "peak_extra_mem_gib": round(peak_mem_gib, 2),
    }
    del experts, x, indices, weights
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes",
        nargs="+",
        required=True,
        help="One or more T,H,I,E,K shapes (tokens, hidden, intermediate, experts, top-k).",
    )
    parser.add_argument("--backends", nargs="+", default=["grouped_mm", "sonic"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"device: {torch.cuda.get_device_name()}  torch: {torch.__version__}", flush=True)
    results = []
    for shape_str in args.shapes:
        shape = tuple(int(v) for v in shape_str.split(","))
        assert len(shape) == 5, f"bad shape {shape_str}"
        for backend in args.backends:
            try:
                r = bench(shape, backend, args.warmup, args.iters, args.seed)
            except torch.cuda.OutOfMemoryError:
                r = {"backend": backend, "shape": shape_str, "error": "OOM"}
                torch.cuda.empty_cache()
            except Exception as exc:  # noqa: BLE001
                r = {"backend": backend, "shape": shape_str, "error": repr(exc)[:300]}
            print("RESULT " + json.dumps(r), flush=True)
            results.append(r)

    print("\n=== summary (fwd+bwd) ===")
    print(
        f"{'shape (T,H,I,E,K)':>28s} {'backend':>10s} {'fwd ms':>9s} {'f+b ms':>9s} {'f+b TFLOPS':>11s} {'peak GiB':>9s}"
    )
    for r in results:
        if "error" in r:
            print(f"{r['shape']:>28s} {r['backend']:>10s}  {r['error']}")
            continue
        shape = f"{r['T']},{r['H']},{r['I']},{r['E']},{r['K']}"
        print(
            f"{shape:>28s} {r['backend']:>10s} {r['fwd_ms']:9.3f} {r['fwd_bwd_ms']:9.3f} "
            f"{r['fwd_bwd_tflops']:11.1f} {r['peak_extra_mem_gib']:9.2f}"
        )


if __name__ == "__main__":
    main()
