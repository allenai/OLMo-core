"""
Standalone benchmark for :class:`olmo_core.nn.OutputDiscardCheckpoint`.

Compares four configurations on a synthetic "fat-output" workload
(``Linear(d -> d_ff) -> activation -> Linear(d_ff -> d)``):

1. baseline            -- vanilla forward+backward, no recompute.
2. torch.utils.checkpoint -- standard activation checkpointing wrapping the
   up-projection + activation.
3. ODC (C++ ext.)      -- :class:`OutputDiscardCheckpoint` with the C++
   ``share_storage`` extension (if it can build on this machine).
4. ODC (Python fb)     -- :class:`OutputDiscardCheckpoint` with the Python
   fallback forced (via monkey-patching ``_get_share_storage``).

Reports peak GPU memory and forward / backward / total wall time for each.

This script is intentionally outside ``src/test/`` so CI does not pick it up.
Run manually:

.. code-block:: bash

    python src/scripts/benchmark_odc.py
    python src/scripts/benchmark_odc.py --d-model 8192 --d-ff 32768 --batch 4 --seq 4096
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

from olmo_core.nn import OutputDiscardCheckpoint
from olmo_core.nn import output_discard_checkpoint as odc_module

WARMUP_ITERS = 3
TIMED_ITERS = 10


@dataclass
class Result:
    name: str
    peak_mb: float
    fwd_ms: float
    bwd_ms: float

    @property
    def total_ms(self) -> float:
        return self.fwd_ms + self.bwd_ms


class FatOutputBlock(nn.Module):
    """Two linears with a SiLU in between -- the intermediate is ``d_ff``-wide."""

    def __init__(self, d_model: int, d_ff: int, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.up = nn.Linear(d_model, d_ff, bias=False, dtype=dtype, device=device)
        self.down = nn.Linear(d_ff, d_model, bias=False, dtype=dtype, device=device)

    def _inner(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.up(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self._inner(x))


def _run_baseline(model: FatOutputBlock, x: torch.Tensor) -> torch.Tensor:
    return model(x)


def _run_torch_checkpoint(model: FatOutputBlock, x: torch.Tensor) -> torch.Tensor:
    h = torch_checkpoint(model._inner, x, use_reentrant=False)
    return model.down(h)


def _run_odc(model: FatOutputBlock, x: torch.Tensor) -> torch.Tensor:
    ckpt = OutputDiscardCheckpoint()
    h = ckpt.checkpoint(model._inner, x)
    y = model.down(h)
    ckpt.discard_output_and_register_recompute(y)
    return y


def _time_step(
    runner: Callable[[FatOutputBlock, torch.Tensor], torch.Tensor],
    model: FatOutputBlock,
    x: torch.Tensor,
) -> Tuple[float, float, float]:
    """
    Time one forward / backward of ``runner`` and return
    ``(peak_mb, fwd_ms, bwd_ms)``. Caller should reset peak stats first.
    """
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
    runner: Callable[[FatOutputBlock, torch.Tensor], torch.Tensor],
    model: FatOutputBlock,
    x: torch.Tensor,
) -> Result:
    # Warmup.
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
    for _ in range(TIMED_ITERS):
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
    original = odc_module._get_share_storage
    odc_module._get_share_storage = lambda: None
    return lambda: setattr(odc_module, "_get_share_storage", original)


def _print_table(results: List[Result], baseline: Result) -> None:
    print()
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


def run_scenario(
    *,
    d_model: int,
    d_ff: int,
    batch: int,
    seq: int,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """
    Run all four configurations on a single shape and print the comparison.
    """
    print(
        f"\n=== d_model={d_model}, d_ff={d_ff}, batch={batch}, seq={seq}, "
        f"dtype={dtype}, device={device} ==="
    )
    print(
        f"    intermediate tensor size: "
        f"{batch * seq * d_ff * dtype.itemsize / 1024 / 1024:.1f} MB"
    )

    torch.manual_seed(0)
    model = FatOutputBlock(d_model=d_model, d_ff=d_ff, dtype=dtype, device=device)
    x = torch.randn(batch, seq, d_model, dtype=dtype, device=device, requires_grad=True)

    runners = [
        ("baseline", _run_baseline),
        ("torch.utils.checkpoint", _run_torch_checkpoint),
        ("ODC (C++ if available)", _run_odc),
    ]

    results: List[Result] = []
    for name, runner in runners:
        results.append(_benchmark_one(name, runner, model, x))

    # ODC with Python fallback forced.
    restore = _force_python_fallback()
    try:
        results.append(_benchmark_one("ODC (python fallback)", _run_odc, model, x))
    finally:
        restore()

    _print_table(results, baseline=results[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=4096)
    parser.add_argument("--d-ff", type=int, default=16384)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument(
        "--scenarios",
        choices=["one", "grid"],
        default="one",
        help="'one' runs the single shape from --d-model/--d-ff/--batch/--seq; "
        "'grid' sweeps a few common (model, ff, batch, seq) combos.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required for memory and timing measurements. Skipping.", file=sys.stderr)
        return 0

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    device = torch.device("cuda")

    print(f"PyTorch {torch.__version__}  /  device {torch.cuda.get_device_name(device)}")

    # Touch the ODC module once so any C++ build cost is excluded from timings.
    _warm = OutputDiscardCheckpoint()
    del _warm
    odc_module._get_share_storage()

    if args.scenarios == "one":
        run_scenario(
            d_model=args.d_model,
            d_ff=args.d_ff,
            batch=args.batch,
            seq=args.seq,
            dtype=dtype,
            device=device,
        )
    else:
        for d_model, d_ff in [(2048, 8192), (4096, 16384), (4096, 21845)]:
            for batch, seq in [(4, 2048), (2, 4096), (1, 8192)]:
                run_scenario(
                    d_model=d_model,
                    d_ff=d_ff,
                    batch=batch,
                    seq=seq,
                    dtype=dtype,
                    device=device,
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())
