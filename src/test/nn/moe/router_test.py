import pytest
import torch

from olmo_core.nn.moe.router import MoELinearRouter

from ...utils import DEVICES


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "uniform_expert_assignment",
    [pytest.param(True, id="uniform"), pytest.param(False, id="computed")],
)
def test_router(device: torch.device, uniform_expert_assignment: bool):
    router = MoELinearRouter(
        d_model=128,
        num_experts=4,
        jitter_eps=0.1,
        top_k=2,
        normalize_expert_weights=True,
        uniform_expert_assignment=uniform_expert_assignment,
    ).to(device)
    x = torch.randn((2, 4, 128), device=device)
    logits, scores, weights, indices = router(x)
    assert logits.shape == (8, 4)
    assert scores.shape == (8, 4)
    assert weights.shape == (8, 2)
    assert indices.shape == (8, 2)
