import torch
import torch.nn.functional as F

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
