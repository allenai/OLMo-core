import pytest
import torch
from torch import nn

from olmo_core.optim import NoOpConfig, SkipStepAdamWConfig
from olmo_core.testing import DEVICES


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(128, 8)
        self.fc = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.wte(x)
        return self.fc(x)


@pytest.mark.parametrize("device", DEVICES)
def test_noop_vs_zero_lr_adamw(device: torch.device):
    """Test that NoOpOptimizer produces the same output as SkipStepAdamW with lr=0."""
    torch.manual_seed(42)

    # Create two identical models
    model1 = TinyModel().to(device)
    model2 = TinyModel().to(device)
    model2.load_state_dict(model1.state_dict())

    # Create optimizers: SkipStepAdamW with lr=0 and NoOpOptimizer
    optim1 = SkipStepAdamWConfig(
        lr=0.0,
        rolling_interval_length=128,
        sigma_factor=6,
    ).build(model1)

    optim2 = NoOpConfig(
        lr=1e-3,  # lr doesn't matter for NoOp
        rolling_interval_length=128,
        sigma_factor=6,
    ).build(model2)

    # Run both models for 10 steps
    for step in range(10):
        # Set the same seed for both models to ensure same input
        torch.manual_seed(100 + step)
        x = torch.randint(0, 128, (4, 8), device=device)

        # Model 1 with SkipStepAdamW (lr=0)
        optim1.zero_grad(set_to_none=True)
        out1 = model1(x)
        loss1 = out1.sum()
        optim1.latest_loss = loss1.detach()
        loss1.backward()
        optim1.step()

        # Model 2 with NoOpOptimizer
        optim2.zero_grad(set_to_none=True)
        out2 = model2(x)
        loss2 = out2.sum()
        optim2.latest_loss = loss2.detach()
        loss2.backward()
        optim2.step()

        # Verify that the models produce the same output
        torch.manual_seed(200 + step)
        test_input = torch.randint(0, 128, (2, 4), device=device)

        with torch.no_grad():
            test_out1 = model1(test_input)
            test_out2 = model2(test_input)

        assert torch.allclose(
            test_out1, test_out2, atol=1e-6
        ), f"Step {step}: Outputs differ between SkipStepAdamW(lr=0) and NoOpOptimizer"

    # Verify final model parameters are identical
    for (name1, param1), (name2, param2) in zip(
        model1.named_parameters(), model2.named_parameters()
    ):
        assert name1 == name2
        assert torch.equal(param1, param2), f"Parameter {name1} differs between models"


@pytest.mark.parametrize("device", DEVICES)
def test_noop_no_parameter_updates(device: torch.device):
    """Test that NoOpOptimizer doesn't update any parameters."""
    torch.manual_seed(42)

    model = TinyModel().to(device)
    optim = NoOpConfig(rolling_interval_length=2, sigma_factor=6).build(model)

    # Store initial parameters
    initial_params = {name: param.clone() for name, param in model.named_parameters()}

    # Run for 10 steps
    for step in range(10):
        optim.zero_grad(set_to_none=True)
        x = torch.randint(0, 128, (4, 8), device=device)
        out = model(x)
        loss = out.sum()
        optim.latest_loss = loss.detach()
        loss.backward()
        optim.step()

    # Verify parameters haven't changed
    for name, param in model.named_parameters():
        assert torch.equal(
            param, initial_params[name]
        ), f"Parameter {name} was modified by NoOpOptimizer"
