import pytest
import torch
import torch.nn as nn

from olmo_core.optim import LionConfig
from olmo_core.testing import DEVICES


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(8, 16)
        self.w2 = nn.Linear(16, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.w1(x))


@pytest.mark.parametrize("device", DEVICES)
def test_lion(device: torch.device):
    config = LionConfig()
    model = Model().train().to(device)
    optim = config.build(model)

    for group in optim.param_groups:
        assert "initial_lr" in group

    # Take a step.
    optim.zero_grad(set_to_none=True)
    model(torch.randn(2, 8, device=device)).sum().backward()
    optim.step()
