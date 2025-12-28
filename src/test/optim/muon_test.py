import pytest
import torch

from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.nn.transformer.model import Transformer
from olmo_core.optim.muon import SkipStepMuon, SkipStepMuonConfig
from olmo_core.testing import DEVICES


def build_transformer_model() -> Transformer:
    config = TransformerConfig.olmo2_30M(vocab_size=1024)
    model = config.build()
    return model


def test_muon_config_to_optim():
    config = SkipStepMuonConfig()

    model = build_transformer_model()
    optim = config.build(model)

    assert isinstance(optim, SkipStepMuon)
    assert len(optim.param_groups) == 3

    assert config.merge(["lr=1e-1"]).lr == 0.1


@pytest.mark.parametrize("device", DEVICES)
def test_muon(device: torch.device, tmp_path):
    config = SkipStepMuonConfig()

    model = build_transformer_model().train().to(device)
    optim = config.build(model)

    for group in optim.param_groups:
        assert "initial_lr" in group

    optim.zero_grad(set_to_none=True)
    model(torch.randint(0, 1024, (2, 8), device=device).int()).sum().backward()
    optim.step()

    # Save and then restore a checkpoint, and make sure fixed fields reset.
    # Store original initial_lr values before modifying them.
    # TODO: double check this test.
    original_initial_lrs = [group["initial_lr"] for group in optim.param_groups]
    for group in optim.param_groups:
        group["initial_lr"] = 1e-8
    save_model_and_optim_state(tmp_path, model, optim)
    load_model_and_optim_state(tmp_path, model, optim)
    for group, original_lr in zip(optim.param_groups, original_initial_lrs):
        assert group["initial_lr"] == original_lr
