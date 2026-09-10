import pytest
import torch
import torch.nn.functional as F

from olmo_core.nn.moe import MoERouterGatingFunction
from olmo_core.nn.moe.v2.router import MoERouterV2


def test_router_can_match_low_precision_linear() -> None:
    router = MoERouterV2(
        d_model=16,
        num_experts=8,
        top_k=2,
        router_logits_in_fp32=False,
        dtype=torch.bfloat16,
    )
    generator = torch.Generator().manual_seed(123)
    x = torch.randn(2, 4, 16, generator=generator, dtype=torch.bfloat16)
    with torch.no_grad():
        router.weight.copy_(torch.randn(router.weight.shape, generator=generator))

    actual = router.get_expert_logits(x)
    expected = F.linear(x, router.weight.view(8, 16))

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_router_uses_float32_linear_by_default() -> None:
    router = MoERouterV2(
        d_model=16,
        num_experts=8,
        top_k=2,
        dtype=torch.bfloat16,
    )
    x = torch.randn(2, 4, 16, dtype=torch.bfloat16)

    actual = router.get_expert_logits(x)

    assert actual.dtype == torch.float32


@pytest.mark.parametrize("routed_top_k", [4, 8, 12])
def test_router_can_normalize_expert_weights_against_reference_top_k(
    routed_top_k: int,
) -> None:
    num_experts = 16
    reference_top_k = 8
    router = MoERouterV2(
        d_model=num_experts,
        num_experts=num_experts,
        top_k=routed_top_k,
        normalize_expert_weights=1.0,
        expert_weight_normalization_top_k=reference_top_k,
    )
    logits = torch.linspace(-2, 2, num_experts).view(1, 1, -1)
    with torch.no_grad():
        router.weight.copy_(torch.eye(num_experts).reshape(-1))

    actual_weights, actual_indices, _, _ = router(logits, scores_only=False)
    scores = logits.softmax(dim=-1)
    expected_weights, expected_indices = scores.topk(routed_top_k, dim=-1)
    reference_weights = scores.topk(reference_top_k, dim=-1).values
    expected_weights = expected_weights / reference_weights.sum(dim=-1, keepdim=True)

    torch.testing.assert_close(actual_indices, expected_indices)
    torch.testing.assert_close(actual_weights, expected_weights)


def test_reference_top_k_requires_expert_weight_normalization() -> None:
    with pytest.raises(ValueError, match="requires normalize_expert_weights"):
        MoERouterV2(
            d_model=8,
            num_experts=8,
            top_k=4,
            expert_weight_normalization_top_k=8,
        )


def test_reference_top_k_rejects_topk_softmax_gating() -> None:
    with pytest.raises(ValueError, match="not supported with topk_softmax"):
        MoERouterV2(
            d_model=8,
            num_experts=8,
            top_k=4,
            normalize_expert_weights=1.0,
            expert_weight_normalization_top_k=8,
            gating_function=MoERouterGatingFunction.topk_softmax,
        )


def test_router_top_k_can_change_between_forwards() -> None:
    router = MoERouterV2(
        d_model=8,
        num_experts=8,
        top_k=4,
        normalize_expert_weights=1.0,
    )

    router.set_top_k(6)

    assert router.top_k == 6


def test_router_top_k_rejects_change_during_recompute() -> None:
    router = MoERouterV2(d_model=8, num_experts=8, top_k=4)
    router._recompute_cache = torch.zeros(1, 1, 4, dtype=torch.long)

    with pytest.raises(RuntimeError, match="recomputation is pending"):
        router.set_top_k(6)
