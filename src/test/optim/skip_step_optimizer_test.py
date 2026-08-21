import pytest
import torch
from torch import nn

from olmo_core.optim import SkipStepAdamWConfig
from olmo_core.testing import DEVICES


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


@pytest.mark.parametrize(
    ("current_loss", "current_grad_norm", "expected_loss_within", "expected_grad_within"),
    (
        (2.0, 1.0, False, True),
        (1.0, 2.0, True, False),
        (2.0, 2.0, False, False),
    ),
)
@pytest.mark.parametrize("device", DEVICES)
def test_skip_step_optimizer_caches_the_live_guard_reason(
    device: torch.device,
    current_loss: float,
    current_grad_norm: float,
    expected_loss_within: bool,
    expected_grad_within: bool,
) -> None:
    model = nn.Linear(1, 1).to(device)
    optim = SkipStepAdamWConfig(rolling_interval_length=6, sigma_factor=1).build(model)

    optim.latest_loss = torch.tensor(1.0, device=device)
    optim.latest_grad_norm = torch.tensor(1.0, device=device)
    assert optim.get_step_factor().item() == 1.0
    assert optim.guard_active.item() is False
    assert optim.guard_loss_within.item() is True
    assert optim.guard_gradient_within.item() is True

    optim.latest_loss = torch.tensor(1.0, device=device)
    optim.latest_grad_norm = torch.tensor(1.0, device=device)
    assert optim.get_step_factor().item() == 1.0
    assert optim.guard_active.item() is False

    optim.latest_loss = torch.tensor(current_loss, device=device)
    optim.latest_grad_norm = torch.tensor(current_grad_norm, device=device)
    expected_step_factor = float(expected_loss_within and expected_grad_within)
    assert optim.get_step_factor().item() == expected_step_factor
    assert optim.guard_active.item() is True
    assert optim.guard_loss_within.item() is expected_loss_within
    assert optim.guard_gradient_within.item() is expected_grad_within
