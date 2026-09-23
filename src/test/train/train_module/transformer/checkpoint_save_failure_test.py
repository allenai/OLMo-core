"""Native save must restore released optimizer storage even if writing fails."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from olmo_core.train.train_module.transformer import ddp_train_module


@pytest.mark.parametrize("failure_phase", ["buffers", "write"])
def test_failed_save_restores_live_optimizer(tmp_path, monkeypatch, failure_phase):
    states = {"parameter.main": torch.tensor([1.0])}
    optimizer = mock.Mock()
    optimizer.state_dict.return_value = states
    module = SimpleNamespace(
        _require_optimizer=lambda: optimizer,
        _persistent_model_buffer_state_dict=mock.Mock(return_value={}),
    )
    error = RuntimeError("injected save failure")
    monkeypatch.setattr(ddp_train_module, "_prepare_env_for_save", lambda path, **kwargs: path)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    write = mock.Mock(side_effect=error if failure_phase == "write" else None)
    monkeypatch.setattr(ddp_train_module.dist_cp.state_dict_saver, "save", write)
    if failure_phase == "buffers":
        module._persistent_model_buffer_state_dict.side_effect = error
    with pytest.raises(RuntimeError, match="injected save failure"):
        ddp_train_module.OLMoDDPTrainModule.save_state_dict_direct(module, tmp_path)
    optimizer.load_state_dict.assert_called_once_with(states, reset_optimizer_moments_on_load=False)
