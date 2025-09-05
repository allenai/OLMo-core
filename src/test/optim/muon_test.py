import pytest
import torch
from torch import nn

from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.optim import MuonAdamW, MuonAdamWConfig
from olmo_core.testing import DEVICES


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(1024, 16)
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 16)
        self.bias = nn.Parameter(torch.zeros(16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.wte(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = x + self.bias
        return x

@pytest.mark.parametrize("device", DEVICES)
def test_muon_optimizer_basic(device: torch.device):
    model = MyModel().to(device)
    optimizer = MuonAdamW(model.parameters(), lr=0.01)

    x = torch.randint(0, 100, (4, 8), device=device)
    output = model(x)
    loss = output.sum()

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    assert len(optimizer.state) > 0

    # Check if parameters have momentum_buffer (Muon) and others have exp_avg (AdamW)
    has_momentum = False
    has_exp_avg = False

    for param in model.parameters():
        if param in optimizer.state:
            state = optimizer.state[param]
            if 'momentum_buffer' in state:
                has_momentum = True
            if 'exp_avg' in state:
                has_exp_avg = True

    assert has_momentum, "Should have momentum buffers for matrix parameters"
    assert has_exp_avg, "Should have AdamW state for non-matrix/excluded parameters"


@pytest.mark.parametrize("device", DEVICES)
def test_muon_config_build(device: torch.device):
    model = MyModel().to(device)
    config = MuonAdamWConfig(lr=0.02, weight_decay=0.1, ns_steps=3)
    optimizer = config.build(model)

    assert isinstance(optimizer, MuonAdamW)
    assert optimizer.defaults['lr'] == 0.02
    assert optimizer.defaults['weight_decay'] == 0.1
    assert optimizer.defaults['ns_steps'] == 3


@pytest.mark.parametrize("device", DEVICES)
def test_muon_parameter_selection(device: torch.device):
    model = MyModel().to(device)
    optimizer = MuonAdamW(model.parameters(), lr=0.01)

    x = torch.randint(0, 100, (4, 8), device=device)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()

    param_to_name = {param: name for name, param in model.named_parameters()}

    for param in model.parameters():
        if param in optimizer.state:
            state = optimizer.state[param]
            name = param_to_name.get(param, "unknown")

            if (param.ndim >= 2 and
                'embed' not in name.lower() and
                'head' not in name.lower()):
                assert 'momentum_buffer' in state, f"Parameter {name} should use Muon"
                assert 'exp_avg' not in state, f"{name} should not have AdamW state"
            else:
                assert 'exp_avg' in state, f"Parameter {name} should use AdamW"
                assert 'exp_avg_sq' in state, f"{name} should have AdamW second moment"
                assert 'momentum_buffer' not in state, f"{name} should not have Muon state"


@pytest.mark.parametrize("device", DEVICES)
def test_muon_newton_schulz_convergence(device: torch.device):
    optimizer = MuonAdamW([], lr=0.01)

    torch.manual_seed(42)
    grad_matrix = torch.randn(32, 16, device=device) 
    result = optimizer.zeropower_via_newtonschulz5(grad_matrix, steps=5)

    assert not torch.isnan(result).any(), "Result contains NaN values"
    assert not torch.allclose(result, torch.zeros_like(result)), "Result is all zeros"
    
    col_norms = torch.norm(result, dim=0)

    assert torch.all(col_norms > 0.1), f"Some columns have very small norms: {col_norms}"


@pytest.mark.parametrize("device", DEVICES)
def test_muon_checkpoint_fixed_fields(device: torch.device, tmp_path):
    config = MuonAdamWConfig()
    model = MyModel().train().to(device)
    optimizer = config.build(model)

    for group in optimizer.param_groups:
        assert "initial_lr" in group

    optimizer.zero_grad(set_to_none=True)
    model(torch.randint(0, 1024, (2, 8), device=device)).sum().backward()
    optimizer.step()

    for group in optimizer.param_groups:
        group["initial_lr"] = 1e-8

    save_model_and_optim_state(tmp_path, model, optimizer)
    load_model_and_optim_state(tmp_path, model, optimizer)

    for group in optimizer.param_groups:
        assert group["initial_lr"] == config.lr


def test_muon_config_defaults():
    config = MuonAdamWConfig()

    assert config.lr == 0.01
    assert config.betas == (0.95, 0.95)
    assert config.weight_decay == 0.0
    assert config.ns_steps == 5
    assert config.nesterov is True
    assert config.eps == 1e-8
    assert config.record_update_metrics is False
    assert config.selective_updates is False
    assert config.optimizer() == MuonAdamW