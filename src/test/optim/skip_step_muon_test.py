import pytest
import torch

from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.nn.transformer.model import Transformer
from olmo_core.optim import SkipStepMuonConfig
from olmo_core.optim.skip_step_optimizer import SkipStepOptimizer
from olmo_core.testing import DEVICES
from olmo_core.testing.utils import requires_dion


def build_transformer_model() -> Transformer:
    config = TransformerConfig.olmo2_30M(vocab_size=1024, n_layers=2)
    model = config.build()
    return model


@requires_dion
def test_skip_step_muon_config_to_optim():
    from olmo_core.optim.skip_step_muon import SkipStepMuon

    config = SkipStepMuonConfig()
    model = build_transformer_model()
    optim = config.build(model)

    assert isinstance(optim, SkipStepMuon)
    assert isinstance(optim, SkipStepOptimizer)
    assert len(optim.param_groups) == 4  # emb, matrix, vector, lm_head

    assert config.merge(["lr=1e-1"]).lr == 0.1


@requires_dion
@pytest.mark.parametrize("device", DEVICES)
def test_skip_step_muon(device: torch.device):
    config = SkipStepMuonConfig()
    model = build_transformer_model().train().to(device)
    optim = config.build(model)

    for group in optim.param_groups:
        assert "initial_lr" in group

    # Take a step.
    optim.zero_grad(set_to_none=True)
    optim.latest_loss = torch.tensor(1.0, device=device)
    model(torch.randint(0, 1024, (2, 8), device=device).int()).sum().backward()
    optim.step()


@requires_dion
@pytest.mark.parametrize("device", DEVICES)
def test_skip_step_muon_skips_on_outlier(device: torch.device):
    """Test that SkipStepMuon skips steps with outlier losses."""
    config = SkipStepMuonConfig(rolling_interval_length=2, sigma_factor=1)
    model = build_transformer_model().train().to(device)
    optim = config.build(model)

    # Normal step - should not skip (not enough history yet)
    optim.zero_grad(set_to_none=True)
    loss = model(torch.randint(0, 1024, (2, 8), device=device).int()).sum()
    optim.latest_loss = loss.detach()
    loss.backward()
    optim.step()
    assert torch.equal(optim.step_skipped.cpu().detach(), torch.tensor(False))

    # Outlier step - should skip
    optim.zero_grad(set_to_none=True)
    loss = model(torch.randint(0, 1024, (2, 8), device=device).int()).sum()
    optim.latest_loss = torch.tensor(1e9, device=device)
    loss.backward()
    optim.step()
    assert torch.equal(optim.step_skipped.cpu().detach(), torch.tensor(True))

    # Another normal step - should not skip
    optim.zero_grad(set_to_none=True)
    loss = model(torch.randint(0, 1024, (2, 8), device=device).int()).sum()
    optim.latest_loss = loss.detach()
    loss.backward()
    optim.step()
    assert torch.equal(optim.step_skipped.cpu().detach(), torch.tensor(False))


@requires_dion
@pytest.mark.parametrize("device", DEVICES)
def test_skip_step_muon_preserves_state_on_skip(device: torch.device):
    """Test that weights and optimizer state are unchanged when a step is skipped."""
    config = SkipStepMuonConfig(rolling_interval_length=2, sigma_factor=1)
    model = build_transformer_model().train().to(device)
    optim = config.build(model)

    # Take a normal step to build up loss history
    optim.zero_grad(set_to_none=True)
    loss = model(torch.randint(0, 1024, (2, 8), device=device).int()).sum()
    optim.latest_loss = loss.detach()
    loss.backward()
    optim.step()

    # Snapshot weights before the skip
    params_before = {n: p.clone() for n, p in model.named_parameters()}

    # Trigger a skip with an outlier loss
    optim.zero_grad(set_to_none=True)
    loss = model(torch.randint(0, 1024, (2, 8), device=device).int()).sum()
    optim.latest_loss = torch.tensor(1e9, device=device)
    loss.backward()
    optim.step()
    assert torch.equal(optim.step_skipped.cpu().detach(), torch.tensor(True))

    # Weights should be unchanged
    for n, p in model.named_parameters():
        assert torch.equal(p, params_before[n]), f"Parameter {n} changed during skipped step"
