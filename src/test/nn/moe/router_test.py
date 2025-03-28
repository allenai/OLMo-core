import pytest
import torch

from olmo_core.nn.moe.router import MoELinearRouter, MoERouterGatingFunction

from ...utils import DEVICES


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "uniform_expert_assignment",
    [
        pytest.param(True, id="uniform"),
        pytest.param(False, id="computed"),
    ],
)
@pytest.mark.parametrize(
    "gating_function",
    [
        pytest.param(MoERouterGatingFunction.softmax, id="softmax"),
        pytest.param(MoERouterGatingFunction.sigmoid, id="sigmoid"),
    ],
)
def test_router(
    device: torch.device, uniform_expert_assignment: bool, gating_function: MoERouterGatingFunction
):
    router = MoELinearRouter(
        d_model=128,
        num_experts=4,
        jitter_eps=0.1,
        top_k=2,
        normalize_expert_weights=True,
        uniform_expert_assignment=uniform_expert_assignment,
        gating_function=gating_function,
    ).to(device)

    x = torch.randn((2, 4, 128), device=device)
    logits, scores, weights, indices, bz_per_expert, batched_bz_per_expert = router(x)

    assert logits.shape == (2, 4, 4)
    assert scores.shape == (2, 4, 4)
    assert weights.shape == (2, 4, 2)
    assert indices.shape == (2, 4, 2)
    assert bz_per_expert.shape == (4,)
    assert batched_bz_per_expert.shape == (2, 4)

    # Scores should be normalized, otherwise LB loss doesn't work.
    scores_total = scores.sum(dim=-1)
    torch.testing.assert_close(scores_total, torch.ones_like(scores_total))


@pytest.mark.parametrize("device", DEVICES)
def test_router_with_bias_gamma(device: torch.device):
    router1 = MoELinearRouter(
        d_model=128,
        num_experts=4,
        top_k=2,
        bias_gamma=0.001,
    ).to(device)
    router1.reset_parameters()

    assert router1.score_bias is not None
    assert router1.score_bias.nonzero().sum().item() == 0  # type: ignore
    assert router1._cache is not None
    assert router1._cache["batch_size_per_expert"].nonzero().sum().item() == 0

    router2 = MoELinearRouter(
        d_model=128,
        num_experts=4,
        top_k=2,
    ).to(device)
    router2.reset_parameters()
    state_dict = router1.state_dict()
    del state_dict["score_bias"]
    router2.load_state_dict(state_dict)

    x = torch.randn((2, 4, 128), device=device)

    # At this point, the output should be exactly the same as it would be without a bias gamma.
    logits1, scores1, weights1, indices1, bz_per_expert1, _ = router1(x)
    logits2, scores2, weights2, indices2, bz_per_expert2, _ = router2(x)
    torch.testing.assert_close(logits1, logits2)
    torch.testing.assert_close(scores1, scores2)
    torch.testing.assert_close(weights1, weights2)
    torch.testing.assert_close(indices1, indices2)
    torch.testing.assert_close(bz_per_expert1, bz_per_expert2)

    assert router1._cache["batch_size_per_expert"].sum().item() == 8 * 2

    # Update the biases and check.
    router1.post_batch()
    assert router1.score_bias.nonzero().sum().item() > 0  # type: ignore
    assert router1._cache["batch_size_per_expert"].nonzero().sum().item() == 0
