import pytest
import torch
from torch import nn

from olmo_core.optim import SkipStepAdamWConfig
from olmo_core.optim.adamw import SkipStepAdamW
from olmo_core.testing import DEVICES


@pytest.mark.parametrize("metric", ["latest_loss", "latest_grad_norm"])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_skip_step_nonfinite_warmup_and_empty_finite_history(metric, bad_value):
    optim = SkipStepAdamW([nn.Parameter(torch.ones(1))], rolling_interval_length=2)
    setattr(optim, metric, torch.tensor(bad_value))
    assert optim.get_step_factor().item() == 0.0
    # No finite history yet: do not use undefined std/mean to reject the next batch.
    setattr(optim, metric, torch.tensor(1.0))
    assert optim.get_step_factor().item() == 1.0


def test_skip_step_large_finite_history():
    optim = SkipStepAdamW([nn.Parameter(torch.ones(1))], rolling_interval_length=4)
    for value in (1e20, 2e20, 3e20):
        optim.latest_loss = torch.tensor(value)
    assert optim.get_step_factor().item() == 1.0
    optim.latest_loss = torch.tensor(1e30)
    assert optim.get_step_factor().item() == 0.0


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(1024, 16)
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.wte(x)
        x = self.fc1(x)
        x = torch.relu(x)
        return self.fc2(x)


@pytest.mark.parametrize("device", DEVICES)
def test_skip_step_optimizer(device: torch.device):
    """Test that skip step optimizer skips steps with outlier losses."""
    model = MyModel().to(device)
    optim = SkipStepAdamWConfig(rolling_interval_length=2, sigma_factor=1).build(model)

    # Normal step - should not skip
    optim.zero_grad(set_to_none=True)
    loss = model(torch.randint(0, 128, (4, 8), device=device)).sum()
    optim.latest_loss = loss.detach()
    loss.backward()
    optim.step()
    assert torch.equal(optim.step_skipped.cpu().detach(), torch.tensor(False))

    # Outlier step - should skip
    optim.zero_grad(set_to_none=True)
    loss = model(torch.randint(0, 128, (4, 8), device=device)).sum()
    optim.latest_loss = torch.tensor(1e9, device=device)  # Outlier loss
    loss.backward()
    optim.step()
    assert torch.equal(optim.step_skipped.cpu().detach(), torch.tensor(True))

    # Another normal step
    optim.zero_grad(set_to_none=True)
    loss = model(torch.randint(0, 128, (4, 8), device=device)).sum()
    optim.latest_loss = loss.detach()
    loss.backward()
    optim.step()
    assert torch.equal(optim.step_skipped.cpu().detach(), torch.tensor(False))
