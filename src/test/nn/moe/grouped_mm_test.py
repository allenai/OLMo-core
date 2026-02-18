import os
from collections.abc import Sequence

import pytest
import torch
import torch.nn.functional as F

from olmo_core.testing import requires_gpu


def _build_offs(tokens_per_group: Sequence[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(tokens_per_group, device=device, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )


@requires_gpu
@pytest.mark.skipif(not hasattr(F, "grouped_mm"), reason="Requires torch.nn.functional.grouped_mm")
def test_grouped_mm_matches_reference_with_tail_padding():
    device = torch.device("cuda")
    dtype = torch.bfloat16

    tokens_per_group = [4, 3, 5]
    offs = _build_offs(tokens_per_group, device=device)
    active_tokens = int(offs[-1].item())
    extra_pad_tokens = 9

    # Keep K/N aligned to grouped_mm kernel layout constraints.
    k, n = 16, 32
    a = torch.randn(active_tokens + extra_pad_tokens, k, device=device, dtype=dtype)
    b = torch.randn(len(tokens_per_group), k, n, device=device, dtype=dtype)

    out = F.grouped_mm(a, b, offs=offs)
    assert out.shape == (active_tokens + extra_pad_tokens, n)

    expected = torch.empty(active_tokens, n, device=device, dtype=dtype)
    start = 0
    for group_idx, end in enumerate(offs.tolist()):
        expected[start:end] = a[start:end] @ b[group_idx]
        start = end

    torch.testing.assert_close(out[:active_tokens], expected)


@requires_gpu
@pytest.mark.skipif(not hasattr(F, "grouped_mm"), reason="Requires torch.nn.functional.grouped_mm")
def test_grouped_mm_speed_vs_tail_padding():
    if os.getenv("OLMO_RUN_GROUPED_MM_PERF_TEST", "0") != "1":
        pytest.skip("Set OLMO_RUN_GROUPED_MM_PERF_TEST=1 to run grouped_mm padding perf test")

    device = torch.device("cuda")
    dtype = torch.bfloat16

    num_groups = 16
    tokens_per_group = [512] * num_groups
    offs = _build_offs(tokens_per_group, device=device)
    active_tokens = int(offs[-1].item())

    k, n = 1024, 2048
    pad_cases = [1, 64, 256, 1024, 4096, 8192]
    max_pad = max(pad_cases)

    print(f'active_tokens={active_tokens}, k={k}, n={n}, pad_cases={pad_cases}')

    a_max = torch.randn(active_tokens + max_pad, k, device=device, dtype=dtype)
    b = torch.randn(num_groups, k, n, device=device, dtype=dtype)

    warmup_iters = 10
    iters = 40

    timings_ms: dict[int, float] = {}
    with torch.no_grad():
        for pad_tokens in pad_cases:
            a = a_max[: active_tokens + pad_tokens]

            for _ in range(warmup_iters):
                F.grouped_mm(a, b, offs=offs)
            torch.cuda.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                F.grouped_mm(a, b, offs=offs)
            end.record()
            torch.cuda.synchronize()

            timings_ms[pad_tokens] = start.elapsed_time(end) / iters

    baseline = timings_ms[pad_cases[0]]
    allowed_ratio = 1.50
    allowed_abs_overhead_ms = 0.15

    print("\ngrouped_mm latency vs tail pad rows (rows beyond offs[-1]):")
    for pad_tokens in pad_cases:
        t_ms = timings_ms[pad_tokens]
        print(f"active={active_tokens}  pad={pad_tokens:5d}  time={t_ms:7.3f} ms  ratio={t_ms / baseline:5.2f}x")

    for pad_tokens in pad_cases[1:]:
        t_ms = timings_ms[pad_tokens]
        assert t_ms <= baseline * allowed_ratio + allowed_abs_overhead_ms, (
            f"Unexpected grouped_mm slowdown with tail padding: "
            f"pad={pad_tokens}, baseline={baseline:.3f} ms, observed={t_ms:.3f} ms"
        )


if __name__ == "__main__":
    test_grouped_mm_speed_vs_tail_padding()