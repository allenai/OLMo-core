import torch
import torch.nn.functional as F

from olmo_core.config import DType
from olmo_core.nn.moe.utils import moe_permute_no_compile, moe_unpermute_no_compile
from olmo_core.nn.moe.v2.routed_experts import (
    RoutedExperts,
    requires_host_side_split_sizes,
)
from olmo_core.testing import requires_gpu, requires_grouped_gemm, requires_te


def _batch_sizes_for_runtime(
    batch_sizes: list[int], device: torch.device, *, dtype: torch.dtype = torch.long
) -> torch.Tensor:
    sizes = torch.tensor(batch_sizes, dtype=dtype)
    return sizes if requires_host_side_split_sizes() else sizes.to(device=device)


def _layout_positions(
    num_rows: int, num_active: int, mode: str, device: torch.device
) -> torch.Tensor:
    if mode == "tail":
        return torch.arange(0, num_active, device=device, dtype=torch.long)
    if mode == "head":
        return torch.arange(num_rows - num_active, num_rows, device=device, dtype=torch.long)
    if mode == "interleave":
        pos = torch.arange(0, num_rows, 2, device=device, dtype=torch.long)
        if pos.numel() < num_active:
            pos = torch.cat(
                [pos, torch.arange(1, num_rows, 2, device=device, dtype=torch.long)], dim=0
            )
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

        seg = torch.zeros(
            seg_rows, expert_active.shape[1], device=expert_active.device, dtype=expert_active.dtype
        )
        seg_active_pos = _layout_positions(seg_rows, active_count, mode, expert_active.device)
        seg.index_copy_(0, seg_active_pos, expert_active)

        x_parts.append(seg)
        active_positions.append(seg_active_pos + offset)
        offset += seg_rows

    x = torch.cat(x_parts, dim=0)
    active_idx = torch.cat(active_positions, dim=0)
    return x, active_idx


def _naive_routed_mlp_reference(
    x: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_weights: torch.Tensor,
    w_up_gate: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    x_cpu = x.detach().cpu().to(torch.float64)
    expert_indices_cpu = expert_indices.detach().cpu()
    expert_weights_cpu = expert_weights.detach().cpu().to(torch.float64)
    w_up_gate_cpu = w_up_gate.detach().cpu().to(torch.float64)
    w_down_cpu = w_down.detach().cpu().to(torch.float64)

    out = torch.zeros_like(x_cpu)
    num_tokens, top_k = expert_indices_cpu.shape
    for token_idx in range(num_tokens):
        token = x_cpu[token_idx]
        for route_idx in range(top_k):
            expert_idx = int(expert_indices_cpu[token_idx, route_idx])
            up_gate = token @ w_up_gate_cpu[expert_idx].T
            up, gate = up_gate.chunk(2, dim=-1)
            hidden = up * F.silu(gate)
            out[token_idx] += expert_weights_cpu[token_idx, route_idx] * (
                hidden @ w_down_cpu[expert_idx]
            )
    return out.to(device=x.device, dtype=x.dtype)


@requires_gpu
@requires_grouped_gemm
@requires_te
def test_no_ep_routed_core_matches_naive_reference():
    torch.manual_seed(17)
    device = torch.device("cuda")

    module = RoutedExperts(
        d_model=512,
        hidden_size=1024,
        num_experts=4,
        bias=False,
        dtype=DType.float32,
        init_device="cuda",
    )
    module.eval()
    with torch.no_grad():
        module.w_up_gate.normal_(mean=0.0, std=0.02)
        module.w_down.normal_(mean=0.0, std=0.02)

    x = torch.randn(9, module.d_model, device=device, dtype=torch.float32)
    expert_indices = torch.tensor(
        [
            [0, 1],
            [1, 3],
            [2, 0],
            [3, 2],
            [0, 3],
            [2, 1],
            [1, 0],
            [3, 0],
            [2, 3],
        ],
        device=device,
        dtype=torch.long,
    )
    expert_weights = torch.tensor(
        [
            [0.7, 0.3],
            [0.6, 0.4],
            [0.5, 0.5],
            [0.8, 0.2],
            [0.4, 0.6],
            [0.9, 0.1],
            [0.25, 0.75],
            [0.55, 0.45],
            [0.35, 0.65],
        ],
        device=device,
        dtype=torch.float32,
    )

    num_out_tokens = expert_indices.numel()
    batch_size_per_expert = torch.bincount(expert_indices.reshape(-1), minlength=module.num_experts)
    permuted, reverse = moe_permute_no_compile(
        inp=x,
        routing_map=expert_indices.int(),
        num_out_tokens=num_out_tokens,
        map_type="index",
    )
    actual_permuted = module(
        permuted,
        _batch_sizes_for_runtime(batch_size_per_expert.tolist(), device),
    )
    actual = moe_unpermute_no_compile(
        inp=actual_permuted,
        row_id_map=reverse,
        restore_shape=x.shape,
        map_type="index",
        merging_probs=expert_weights,
    )
    expected = _naive_routed_mlp_reference(
        x,
        expert_indices,
        expert_weights,
        module.w_up_gate,
        module.w_down,
    )

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@requires_gpu
@requires_grouped_gemm
@requires_te
def test_routed_experts_forward_matches_baseline_with_different_pad_positions():
    torch.manual_seed(42)
    device = torch.device("cuda")

    module = RoutedExperts(
        d_model=512,
        hidden_size=1024,
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
