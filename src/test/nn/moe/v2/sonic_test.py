import pytest
import torch
import torch.nn.functional as F

from olmo_core.config import DType
from olmo_core.nn.moe.v2.routed_experts import RoutedExperts, RoutedExpertsBackend
from olmo_core.nn.moe.v2.sonic import _prepare_sonic_inputs, sonic_moe_forward
from olmo_core.testing import requires_gpu
from olmo_core.testing.utils import requires_compute_capability


def _build_experts(*, device: str, dtype: DType) -> RoutedExperts:
    return RoutedExperts(
        d_model=512,
        hidden_size=512,
        num_experts=4,
        bias=False,
        dtype=dtype,
        backend=RoutedExpertsBackend.sonic,
        init_device=device,
    )


def test_prepare_sonic_inputs_uses_sorted_pairs_and_weight_views() -> None:
    experts = _build_experts(device="cpu", dtype=DType.float32)
    x = torch.randn(3, experts.d_model)
    expert_indices = torch.tensor([[2, 0], [3, 1], [0, 2]], dtype=torch.long)
    expert_weights = torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.8, 0.2]])

    token_indices, flat_expert_indices, router_scores, w1, w2 = _prepare_sonic_inputs(
        x,
        expert_indices,
        expert_weights,
        experts,
    )

    torch.testing.assert_close(token_indices, torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int32))
    torch.testing.assert_close(
        flat_expert_indices,
        torch.tensor([2, 0, 3, 1, 0, 2], dtype=torch.int32),
    )
    torch.testing.assert_close(router_scores, expert_weights.flatten().float())
    assert w1.shape == (2 * experts.hidden_size, experts.d_model, experts.num_experts)
    assert w2.shape == (experts.d_model, experts.hidden_size, experts.num_experts)
    assert w1.untyped_storage().data_ptr() == experts.w_up_gate.untyped_storage().data_ptr()
    assert w2.untyped_storage().data_ptr() == experts.w_down.untyped_storage().data_ptr()

    # With the fresh-run reinterpretation, Sonic considers OLMo's first half to be the gate.
    torch.testing.assert_close(
        w1[: experts.hidden_size, :, 1],
        experts.w_up_gate[1, : experts.hidden_size, :],
    )
    torch.testing.assert_close(
        w1[experts.hidden_size :, :, 1],
        experts.w_up_gate[1, experts.hidden_size :, :],
    )


def _naive_sonic_layout_moe(
    x: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_weights: torch.Tensor,
    experts: RoutedExperts,
) -> torch.Tensor:
    outputs = []
    for token_idx in range(x.shape[0]):
        token_out = torch.zeros_like(x[token_idx])
        for route_idx in range(expert_indices.shape[1]):
            expert_idx = int(expert_indices[token_idx, route_idx])
            gate_up = F.linear(x[token_idx], experts.w_up_gate[expert_idx])
            gate, up = gate_up.chunk(2, dim=-1)
            expert_out = (up * F.silu(gate)) @ experts.w_down[expert_idx]
            token_out = (
                token_out + expert_weights[token_idx, route_idx].to(expert_out.dtype) * expert_out
            )
        outputs.append(token_out)
    return torch.stack(outputs)


@requires_gpu
@requires_compute_capability(min_cc=9)
def test_sonic_forward_backward_matches_fresh_weight_semantics() -> None:
    pytest.importorskip("sonicmoe")
    torch.manual_seed(1234)
    experts = _build_experts(device="cuda", dtype=DType.bfloat16)
    experts.train()
    x = torch.randn(16, experts.d_model, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    expert_indices = torch.tensor(
        [[i % experts.num_experts, (i + 1) % experts.num_experts] for i in range(x.shape[0])],
        device="cuda",
        dtype=torch.long,
    )
    expert_weights = torch.rand(
        x.shape[0],
        expert_indices.shape[1],
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    normalized_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)

    actual = sonic_moe_forward(x, expert_indices, normalized_weights, experts)
    expected = _naive_sonic_layout_moe(x, expert_indices, normalized_weights, experts)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=5e-2)

    actual.float().square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert experts.w_up_gate.grad is not None and torch.isfinite(experts.w_up_gate.grad).all()
    assert experts.w_down.grad is not None and torch.isfinite(experts.w_down.grad).all()
    assert expert_weights.grad is not None and torch.isfinite(expert_weights.grad).all()
