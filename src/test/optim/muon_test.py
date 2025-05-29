import pytest
import torch
import torch.nn as nn

from olmo_core.optim import MuonWithAuxAdamConfig, OptimGroupOverride
from olmo_core.testing import DEVICES


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(8, 16, bias=False)
        self.w2 = nn.Linear(16, 8, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.w1(x))


@pytest.mark.parametrize("device", DEVICES)
def test_muon_with_aux_adam(device: torch.device):
    config = MuonWithAuxAdamConfig(
        group_overrides=[
            OptimGroupOverride(
                params=["w1.weight"], opts=dict(use_muon=True, weight_decay=0.0, lr=0.02, momentum=0.95)
            ),
            OptimGroupOverride(params=["w2.weight"], opts=dict(use_muon=False, lr=3e-4, betas=(0.9, 0.95))),
        ]
    )
    model = Model().train().to(device)
    optim = config.build(model)

    for group in optim.param_groups:
        assert "initial_lr" in group

    # Take a step.
    optim.zero_grad(set_to_none=True)
    model(torch.randn(2, 8, device=device)).sum().backward()
    optim.step()
