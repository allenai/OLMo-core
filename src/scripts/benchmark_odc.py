"""
Standalone benchmark for :class:`olmo_core.nn.OutputDiscardCheckpoint`.

For each of several block types (fp32 cast, plain linear up-projection,
SiLU-then-down, SwiGLU FFN, RMSNorm + residual), runs the block at both
``N = 1`` (single-layer) and ``N = args.n_layers`` (stacked), and compares four
configurations end-to-end:

1. baseline               -- vanilla forward+backward, no recompute.
2. torch.utils.checkpoint -- standard activation checkpointing wrapping the
   inner region.
3. ODC (C++ if available) -- :class:`OutputDiscardCheckpoint` with the C++
   ``share_storage`` extension when it builds, otherwise the Python fallback.
4. ODC (python fallback)  -- :class:`OutputDiscardCheckpoint` with the Python
   fallback forced via monkey-patching ``_SHARED_STORAGE_LOADER._load``.

Reports peak GPU memory and forward / backward / total wall time.

ODC's memory win usually does *not* appear at ``N = 1`` because the
"savings window" between discard and recompute is zero -- and recompute has
to allocate its own saved-for-backward intermediates inside the wrapped
region. The multi-layer scenario surfaces the real workload pattern: each
layer's fat output is discarded while subsequent layers run forward.

This script is intentionally outside ``src/test/`` so CI does not pick it up.
Run manually:

.. code-block:: bash

    python src/scripts/benchmark_odc.py
    python src/scripts/benchmark_odc.py --n-layers 8 --d-model 8192 --d-ff 32768
    python src/scripts/benchmark_odc.py --only swiglu rms_norm
    python src/scripts/benchmark_odc.py --layers 1 4 8 --dtype bf16 fp32 --iters 5
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Callable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from olmo_core.nn import output_discard_checkpoint as odc_module
from olmo_core.nn.output_discard_checkpoint import OutputDiscardCheckpoint

WARMUP_ITERS = 3


@dataclass
class Result:
    name: str
    peak_mb: float
    fwd_ms: float
    bwd_ms: float

    @property
    def total_ms(self) -> float:
        return self.fwd_ms + self.bwd_ms


class BenchBlock(nn.Module):
    """
    Base class for a single benchmark block. Subclasses define ``_inner`` (the
    region we'd checkpoint) and ``_outer`` (the consumer of inner's output).
    Each block exposes three forward variants so the stack harness can run
    apples-to-apples comparisons.
    """

    def _inner(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _outer(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward_baseline(self, x: torch.Tensor) -> torch.Tensor:
        return self._outer(self._inner(x), x)

    def forward_torch_ckpt(self, x: torch.Tensor) -> torch.Tensor:
        h = torch_checkpoint(self._inner, x, use_reentrant=False)
        return self._outer(h, x)

    def forward_odc(self, x: torch.Tensor) -> torch.Tensor:
        ckpt = OutputDiscardCheckpoint()
        h = ckpt.checkpoint(self._inner, x)
        y = self._outer(h, x)
        ckpt.discard_output_and_register_recompute(y)
        return y


class Fp32CastBlock(BenchBlock):
    """Mimics the MoE router's pattern: x_fp32 = x.float(), then a fp32 linear."""

    def __init__(self, d_model: int, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=False, dtype=torch.float32, device=device)
        self.dtype = dtype

    def _inner(self, x: torch.Tensor) -> torch.Tensor:
        return x.float()

    def _outer(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.linear(h).to(self.dtype)

    def forward_odc(self, x: torch.Tensor) -> torch.Tensor:
        # When x is already fp32, ``x.float()`` is a no-op alias of x. Discarding
        # the "output" would resize the input tensor's storage to 0 and break
        # backward. The cast pattern is degenerate in this case; fall through
        # to baseline so the benchmark row truthfully reports "no ODC benefit".
        if x.dtype == torch.float32:
            return self.forward_baseline(x)
        return super().forward_odc(x)

    def forward_torch_ckpt(self, x: torch.Tensor) -> torch.Tensor:
        # Same degeneracy: nothing useful for torch.utils.checkpoint either.
        if x.dtype == torch.float32:
            return self.forward_baseline(x)
        return super().forward_torch_ckpt(x)


class UpProjBlock(BenchBlock):
    """Fat linear up-projection (no activation inside) followed by a down-projection."""

    def __init__(self, d_model: int, d_ff: int, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.up = nn.Linear(d_model, d_ff, bias=False, dtype=dtype, device=device)
        self.down = nn.Linear(d_ff, d_model, bias=False, dtype=dtype, device=device)

    def _inner(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)

    def _outer(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.down(h)


class SiluUpBlock(BenchBlock):
    """SiLU(up(x)) -> down(h). Activation inside ``_inner`` -> recompute saves extra."""

    def __init__(self, d_model: int, d_ff: int, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.up = nn.Linear(d_model, d_ff, bias=False, dtype=dtype, device=device)
        self.down = nn.Linear(d_ff, d_model, bias=False, dtype=dtype, device=device)

    def _inner(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.up(x))

    def _outer(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.down(h)


class SwiGLUBlock(BenchBlock):
    """OLMo-style SwiGLU FFN: w2(silu(w1(x)) * w3(x))."""

    def __init__(self, d_model: int, d_ff: int, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False, dtype=dtype, device=device)
        self.w3 = nn.Linear(d_model, d_ff, bias=False, dtype=dtype, device=device)
        self.w2 = nn.Linear(d_ff, d_model, bias=False, dtype=dtype, device=device)

    def _inner(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.w1(x)) * self.w3(x)

    def _outer(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.w2(h)


class RMSNormBlock(BenchBlock):
    """RMSNorm + Linear + residual. Norm output is what's discarded."""

    def __init__(self, d_model: int, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.norm = nn.RMSNorm(d_model, dtype=dtype, device=device)
        self.linear = nn.Linear(d_model, d_model, bias=False, dtype=dtype, device=device)

    def _inner(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)

    def _outer(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.linear(h) + x


class FP32SoftmaxBlock(BenchBlock):
    """
    fp32 softmax pattern (attention softmax / routing softmax in mixed precision).
    Inner: bf16 -> upcast -> softmax in fp32. The fp32 softmax output is what's
    discarded. Note that softmax saves its OUTPUT for backward, so discarding it
    forces a recompute of the full softmax in backward.
    """

    def __init__(self, d_model: int, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False, dtype=dtype, device=device)
        self.dtype = dtype

    def _inner(self, x: torch.Tensor) -> torch.Tensor:
        return x.float().softmax(dim=-1)

    def _outer(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.proj(h.to(self.dtype))


class Stack(nn.Module):
    """Stacks ``n_layers`` copies of a block factory and exposes the three variants."""

    def __init__(self, factory: Callable[[], BenchBlock], n_layers: int):
        super().__init__()
        self.blocks = nn.ModuleList([factory() for _ in range(n_layers)])

    def forward_baseline(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b.forward_baseline(x)
        return x

    def forward_torch_ckpt(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b.forward_torch_ckpt(x)
        return x

    def forward_odc(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b.forward_odc(x)
        return x


def _time_step(
    runner: Callable[[Stack, torch.Tensor], torch.Tensor],
    model: Stack,
    x: torch.Tensor,
) -> Tuple[float, float, float]:
    torch.cuda.synchronize()
    fwd_start = torch.cuda.Event(enable_timing=True)
    fwd_end = torch.cuda.Event(enable_timing=True)
    bwd_end = torch.cuda.Event(enable_timing=True)

    fwd_start.record()
    y = runner(model, x)
    loss = y.square().mean()
    fwd_end.record()
    loss.backward()
    bwd_end.record()
    torch.cuda.synchronize()

    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    fwd_ms = fwd_start.elapsed_time(fwd_end)
    bwd_ms = fwd_end.elapsed_time(bwd_end)
    return peak_mb, fwd_ms, bwd_ms


def _benchmark_one(
    name: str,
    runner: Callable[[Stack, torch.Tensor], torch.Tensor],
    model: Stack,
    x: torch.Tensor,
    timed_iters: int,
) -> Result:
    for _ in range(WARMUP_ITERS):
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
        if x.grad is not None:
            x.grad = None
        _time_step(runner, model, x)

    peak_mbs: List[float] = []
    fwd_mss: List[float] = []
    bwd_mss: List[float] = []
    for _ in range(timed_iters):
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
        if x.grad is not None:
            x.grad = None
        torch.cuda.reset_peak_memory_stats()
        peak_mb, fwd_ms, bwd_ms = _time_step(runner, model, x)
        peak_mbs.append(peak_mb)
        fwd_mss.append(fwd_ms)
        bwd_mss.append(bwd_ms)

    return Result(
        name=name,
        peak_mb=max(peak_mbs),
        fwd_ms=sum(fwd_mss) / len(fwd_mss),
        bwd_ms=sum(bwd_mss) / len(bwd_mss),
    )


def _force_python_fallback() -> Callable[[], None]:
    """Monkey-patch ODC to use the Python fallback. Returns a restorer."""
    loader = odc_module._SHARED_STORAGE_LOADER
    original = loader._load
    loader._load = lambda: None  # type: ignore[method-assign]
    return lambda: setattr(loader, "_load", original)


def _print_table(results: List[Result], baseline: Result) -> None:
    print(
        f"  {'config':<28} "
        f"{'peak (MB)':>12} {'fwd (ms)':>10} {'bwd (ms)':>10} {'total (ms)':>12}"
        f"  {'mem saved':>10}  {'time delta':>11}"
    )
    print(f"  {'-' * 28} {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 12}  {'-' * 10}  {'-' * 11}")
    for r in results:
        mem_pct = (1.0 - r.peak_mb / baseline.peak_mb) * 100.0
        time_pct = (r.total_ms / baseline.total_ms - 1.0) * 100.0
        mem_label = "(baseline)" if r is baseline else f"{mem_pct:+6.1f}%"
        time_label = "(baseline)" if r is baseline else f"{time_pct:+6.1f}%"
        print(
            f"  {r.name:<28} "
            f"{r.peak_mb:>12.1f} {r.fwd_ms:>10.2f} {r.bwd_ms:>10.2f} {r.total_ms:>12.2f}"
            f"  {mem_label:>10}  {time_label:>11}"
        )


# Each entry: (short name, description, factory). The factory takes
# (d_model, d_ff, dtype, device) and returns a BenchBlock. d_ff is ignored
# by blocks that don't need it.
def _block_registry(
    dtype: torch.dtype, device: torch.device
) -> List[Tuple[str, str, Callable[[int, int], BenchBlock]]]:
    return [
        (
            "fp32_cast",
            "x.float() under ODC; consumer is an fp32 Linear (router pattern)",
            lambda d_model, d_ff: Fp32CastBlock(d_model, dtype, device),
        ),
        (
            "up_proj",
            "fat Linear up-projection, no activation inside the discarded region",
            lambda d_model, d_ff: UpProjBlock(d_model, d_ff, dtype, device),
        ),
        (
            "silu_up",
            "silu(up(x)) inside; activation adds a saved intermediate to recompute",
            lambda d_model, d_ff: SiluUpBlock(d_model, d_ff, dtype, device),
        ),
        (
            "swiglu",
            "OLMo SwiGLU FFN; three fat intermediates saved during recompute",
            lambda d_model, d_ff: SwiGLUBlock(d_model, d_ff, dtype, device),
        ),
        (
            "rms_norm",
            "RMSNorm + Linear + residual; cheap recompute, small fat-output savings",
            lambda d_model, d_ff: RMSNormBlock(d_model, dtype, device),
        ),
        (
            "fp32_softmax",
            "softmax in fp32 (attention/routing pattern); 2x size if base dtype is bf16/fp16",
            lambda d_model, d_ff: FP32SoftmaxBlock(d_model, dtype, device),
        ),
    ]


def run_block(
    *,
    block_name: str,
    block_desc: str,
    factory: Callable[[int, int], BenchBlock],
    d_model: int,
    d_ff: int,
    batch: int,
    seq: int,
    n_layers: int,
    dtype: torch.dtype,
    device: torch.device,
    timed_iters: int,
) -> None:
    """Run all four configurations for one (block, n_layers) combo and print the table."""
    print(f"\n  [{block_name}] {block_desc}")

    torch.manual_seed(0)
    model = Stack(lambda: factory(d_model, d_ff), n_layers).to(device)
    x = torch.randn(batch, seq, d_model, dtype=dtype, device=device, requires_grad=True)

    runners: List[Tuple[str, Callable[[Stack, torch.Tensor], torch.Tensor]]] = [
        ("baseline", Stack.forward_baseline),
        ("torch.utils.checkpoint", Stack.forward_torch_ckpt),
        ("ODC (C++ if available)", Stack.forward_odc),
    ]

    results: List[Result] = []
    for name, method in runners:
        results.append(_benchmark_one(name, method, model, x, timed_iters))

    restore = _force_python_fallback()
    try:
        results.append(
            _benchmark_one("ODC (python fallback)", Stack.forward_odc, model, x, timed_iters)
        )
    finally:
        restore()

    _print_table(results, baseline=results[0])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--d-model", type=int, default=2048)
    parser.add_argument("--d-ff", type=int, default=8192)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--n-layers", type=int, default=4, help="Multi-layer stack depth.")
    parser.add_argument(
        "--dtype",
        nargs="+",
        choices=["fp32", "bf16", "fp16"],
        default=["bf16"],
        help="One or more dtypes to sweep. Each is run as a separate top-level "
        "group so you can compare precision-boundary effects (e.g. fp32_cast / "
        "fp32_softmax discard tensor doubles when base dtype is bf16/fp16).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=10,
        help="Number of timed iterations per configuration (after warmup).",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Restrict to a subset of block names (e.g. --only swiglu rms_norm).",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help="Stack depths to run per block. Default: [1, --n-layers]. "
        "Override with explicit depths, e.g. --layers 1 4 8.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required for memory and timing measurements. Skipping.", file=sys.stderr)
        return 0

    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    dtypes = [dtype_map[d] for d in args.dtype]
    device = torch.device("cuda")

    print(f"PyTorch {torch.__version__}  /  device {torch.cuda.get_device_name(device)}")

    # Touch ODC once so any C++ build cost is excluded from timings.
    _warm = OutputDiscardCheckpoint()
    del _warm
    odc_module._SHARED_STORAGE_LOADER._load()

    if args.layers is None:
        layer_counts = sorted({1, args.n_layers})
    else:
        layer_counts = sorted(set(args.layers))

    for dtype in dtypes:
        blocks = _block_registry(dtype, device)
        if args.only is not None:
            wanted = set(args.only)
            unknown = wanted - {name for name, _, _ in blocks}
            if unknown:
                print(f"unknown block name(s): {sorted(unknown)}", file=sys.stderr)
                return 2
            blocks = [b for b in blocks if b[0] in wanted]

        print(f"\n############### dtype = {dtype} ###############")
        for n_layers in layer_counts:
            print(
                f"\n========== n_layers={n_layers}  "
                f"(d_model={args.d_model}, d_ff={args.d_ff}, "
                f"batch={args.batch}, seq={args.seq}, dtype={dtype}) =========="
            )
            for block_name, block_desc, factory in blocks:
                run_block(
                    block_name=block_name,
                    block_desc=block_desc,
                    factory=factory,
                    d_model=args.d_model,
                    d_ff=args.d_ff,
                    batch=args.batch,
                    seq=args.seq,
                    n_layers=n_layers,
                    dtype=dtype,
                    device=device,
                    timed_iters=args.iters,
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())
