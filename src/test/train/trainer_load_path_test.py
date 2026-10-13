from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.train.common import LoadStrategy
from olmo_core.train.train_module.train_module import assert_optimizer_lrs_nonzero
from olmo_core.train.trainer import Trainer


def _bare_trainer() -> Trainer:
    trainer = object.__new__(Trainer)
    trainer.no_checkpoints = False
    trainer._checkpoint_loaded = False
    trainer.load_strategy = LoadStrategy.if_available
    trainer.reset_data_loader_state_on_load_path = False
    trainer.train_module = SimpleNamespace(validate_load_path_checkpoint=Mock())
    return trainer


def test_prefer_explicit_checkpoint_does_not_fall_back_to_latest_endpoint():
    trainer = _bare_trainer()
    trainer.prefer_explicit_load_path = True
    trainer.load_path = "/checkpoints/step858"
    trainer.save_folder = "/checkpoints"  # Also contains the newer step954 endpoint.
    trainer.maybe_load_checkpoint = Mock()

    def load_checkpoint(path, **kwargs):
        assert path == "/checkpoints/step858"
        assert kwargs == {"reset_data_loader_state": False}
        trainer._checkpoint_loaded = True

    trainer.load_checkpoint = Mock(side_effect=load_checkpoint)

    trainer._load_initial_checkpoint()

    trainer.load_checkpoint.assert_called_once_with(
        "/checkpoints/step858", reset_data_loader_state=False
    )
    trainer.maybe_load_checkpoint.assert_not_called()
    trainer.train_module.validate_load_path_checkpoint.assert_called_once_with()


def test_default_resume_prefers_save_folder_over_load_path():
    trainer = _bare_trainer()
    trainer.prefer_explicit_load_path = False
    trainer.load_path = "/elsewhere/step100"
    trainer.save_folder = "/checkpoints"

    def maybe_load_checkpoint(path, **kwargs):
        assert path == "/checkpoints"
        trainer._checkpoint_loaded = True
        return True

    trainer.maybe_load_checkpoint = Mock(side_effect=maybe_load_checkpoint)
    trainer.load_checkpoint = Mock()

    trainer._load_initial_checkpoint()

    trainer.maybe_load_checkpoint.assert_called_once()
    trainer.load_checkpoint.assert_not_called()
    trainer.train_module.validate_load_path_checkpoint.assert_not_called()


def test_assert_optimizer_lrs_nonzero():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    assert_optimizer_lrs_nonzero(optimizer)

    optimizer.param_groups[0]["lr"] = 0.0
    with pytest.raises(OLMoConfigurationError, match="pre-decay checkpoint"):
        assert_optimizer_lrs_nonzero(optimizer)
