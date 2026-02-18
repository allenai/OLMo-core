import os

import pytest
import torch

from olmo_core.config import DType
from olmo_core.nn.moe.v2.routed_experts import RoutedExperts, requires_host_side_split_sizes
from olmo_core.testing import requires_gpu, requires_grouped_gemm


def _batch_sizes_for_runtime(
    batch_sizes: list[int], device: torch.device, *, dtype: torch.dtype = torch.long
) -> torch.Tensor:
    sizes = torch.tensor(batch_sizes, dtype=dtype)
    return sizes if requires_host_side_split_sizes() else sizes.to(device=device)


def _layout_positions(num_rows: int, num_active: int, mode: str, device: torch.device) -> torch.Tensor:
    if mode == "tail":
        return torch.arange(0, num_active, device=device, dtype=torch.long)
    if mode == "head":
        return torch.arange(num_rows - num_active, num_rows, device=device, dtype=torch.long)
    if mode == "interleave":
        pos = torch.arange(0, num_rows, 2, device=device, dtype=torch.long)
        if pos.numel() < num_active:
            pos = torch.cat([pos, torch.arange(1, num_rows, 2, device=device, dtype=torch.long)], dim=0)
        return pos[:num_active].sort().values
    raise ValueError(f"Unknown mode '{mode}'")


def _build_x_with_pad_layout(
    active_by_expert: list[torch.Tensor], pad_by_expert: list[int], mode: str
) -> tuple[torch.Tensor, torch.Tensor]:
    x_parts: list[torch.Tensor] = []
    active_positions: list[torch.Tensor] = []

    offset = 0
    for expert_active, pad_count in zip(active_by_expert, pad_by_expert):
        active_count = expert_active.shape[0]
        seg_rows = active_count + pad_count

        seg = torch.zeros(seg_rows, expert_active.shape[1], device=expert_active.device, dtype=expert_active.dtype)
        seg_active_pos = _layout_positions(seg_rows, active_count, mode, expert_active.device)
        seg.index_copy_(0, seg_active_pos, expert_active)

        x_parts.append(seg)
        active_positions.append(seg_active_pos + offset)
        offset += seg_rows

    x = torch.cat(x_parts, dim=0)
    active_idx = torch.cat(active_positions, dim=0)
    return x, active_idx


@requires_gpu
@requires_grouped_gemm
def test_routed_experts_forward_matches_baseline_with_different_pad_positions():
    torch.manual_seed(42)
    device = torch.device("cuda")

    module = RoutedExperts(
        d_model=64,
        hidden_size=128,
        num_experts=4,
        bias=False,
        dtype=DType.float32,
        init_device="cuda",
    )
    module.eval()
    with torch.no_grad():
        module.w_up_gate.normal_(mean=0.0, std=0.02)
        module.w_down.normal_(mean=0.0, std=0.02)

    active_per_expert = [8, 6, 7, 5]
    pad_per_expert = [3, 4, 2, 5]

    active_by_expert = [
        torch.randn(n, module.d_model, device=device, dtype=module.w_up_gate.dtype)
        for n in active_per_expert
    ]

    x_baseline = torch.cat(active_by_expert, dim=0)
    baseline_sizes = _batch_sizes_for_runtime(active_per_expert, device)
    with torch.no_grad():
        y_baseline = module(x_baseline, baseline_sizes)

    padded_sizes = [a + p for a, p in zip(active_per_expert, pad_per_expert)]
    padded_sizes_tensor = _batch_sizes_for_runtime(padded_sizes, device)

    for mode in ("tail", "head", "interleave"):
        x_padded, active_idx = _build_x_with_pad_layout(active_by_expert, pad_per_expert, mode)
        with torch.no_grad():
            y_padded = module(x_padded, padded_sizes_tensor)
        torch.testing.assert_close(y_padded.index_select(0, active_idx), y_baseline)


@requires_gpu
@requires_grouped_gemm
def test_routed_experts_speed_vs_pad_positions():
    if os.getenv("OLMO_RUN_ROUTED_EXPERTS_PERF_TEST", "0") != "1":
        pytest.skip("Set OLMO_RUN_ROUTED_EXPERTS_PERF_TEST=1 to run RoutedExperts pad-position perf test")

    torch.manual_seed(42)
    device = torch.device("cuda")

    module = RoutedExperts(
        d_model=1024,
        hidden_size=2048,
        num_experts=8,
        bias=False,
        dtype=DType.bfloat16,
        init_device="cuda",
    )
    module.eval()
    with torch.no_grad():
        module.w_up_gate.normal_(mean=0.0, std=0.02)
        module.w_down.normal_(mean=0.0, std=0.02)

    active_per_expert = [192] * module.num_experts
    pad_per_expert = [64] * module.num_experts
    padded_sizes = [a + p for a, p in zip(active_per_expert, pad_per_expert)]
    padded_sizes_tensor = _batch_sizes_for_runtime(padded_sizes, device)

    active_by_expert = [
        torch.randn(n, module.d_model, device=device, dtype=module.w_up_gate.dtype)
        for n in active_per_expert
    ]

    layouts = ("tail", "head", "interleave")
    x_by_layout = {
        layout: _build_x_with_pad_layout(active_by_expert, pad_per_expert, layout)[0] for layout in layouts
    }

    warmup_iters = 8
    iters = 30
    timings_ms: dict[str, float] = {}

    with torch.no_grad():
        for layout in layouts:
            x = x_by_layout[layout]
            for _ in range(warmup_iters):
                module(x, padded_sizes_tensor)
            torch.cuda.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                module(x, padded_sizes_tensor)
            end.record()
            torch.cuda.synchronize()
            timings_ms[layout] = start.elapsed_time(end) / iters

    baseline = timings_ms["tail"]
    allowed_ratio = 1.35
    allowed_abs_overhead_ms = 0.25

    print("\nRoutedExperts latency vs pad-token positions in x (active tokens fixed):")
    for layout in layouts:
        t_ms = timings_ms[layout]
        print(f"  layout={layout:10s} time={t_ms:7.3f} ms ratio={t_ms / baseline:5.2f}x")

    for layout in layouts[1:]:
        t_ms = timings_ms[layout]
        assert t_ms <= baseline * allowed_ratio + allowed_abs_overhead_ms, (
            f"Unexpected RoutedExperts slowdown for pad layout '{layout}': "
            f"baseline={baseline:.3f} ms, observed={t_ms:.3f} ms"
        )

if __name__ == "__main__":
    test_routed_experts_speed_vs_pad_positions()
    