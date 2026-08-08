from types import SimpleNamespace

import torch
import torch.nn as nn

from olmo_core.train.train_module.train_module import TrainModule
from olmo_core.train.train_module.transformer.train_module import TransformerTrainModule


def _build_train_module_with_grads(
    param_groups: list[dict],
) -> tuple[TransformerTrainModule, nn.Parameter, nn.Parameter]:
    model = nn.Module()
    model.adamw_param = nn.Parameter(torch.zeros(1))
    model.muon_param = nn.Parameter(torch.zeros(1))
    model.adamw_param.grad = torch.tensor([3.0])
    model.muon_param.grad = torch.tensor([4.0])

    train_module = TransformerTrainModule.__new__(TransformerTrainModule)
    TrainModule.__init__(train_module)
    train_module.model = model  # type: ignore[assignment]
    train_module.optim = SimpleNamespace(param_groups=param_groups)  # type: ignore[assignment]
    return train_module, model.adamw_param, model.muon_param


def test_muon_gradient_clipping_only_clips_adamw_parameters():
    train_module, adamw_param, muon_param = _build_train_module_with_grads([])
    train_module.optim.param_groups = [
        {"params": [adamw_param], "algorithm": "adamw"},
        {"params": [muon_param], "algorithm": "muon"},
    ]

    total_norm = train_module._clip_grad_norm(1.0)

    torch.testing.assert_close(total_norm, torch.tensor(3.0))
    torch.testing.assert_close(adamw_param.grad, torch.tensor([1.0]))
    torch.testing.assert_close(muon_param.grad, torch.tensor([4.0]))


def test_gradient_clipping_still_clips_all_parameters_for_standard_optimizers():
    train_module, adamw_param, other_param = _build_train_module_with_grads([])
    train_module.optim.param_groups = [{"params": [adamw_param, other_param]}]

    total_norm = train_module._clip_grad_norm(1.0)

    torch.testing.assert_close(total_norm, torch.tensor(5.0))
    torch.testing.assert_close(adamw_param.grad, torch.tensor([0.6]))
    torch.testing.assert_close(other_param.grad, torch.tensor([0.8]))
