import pytest
import torch

from olmo_core.optim.config import INITIAL_LR_FIELD
from olmo_core.train.callbacks.optimizer_transition import (
    transition_optimizer_hyperparameters,
)


def test_transition_optimizer_hyperparameters_preserves_state_and_zero_decay_groups():
    regularized = torch.nn.Parameter(torch.tensor([1.0]))
    unregularized = torch.nn.Parameter(torch.tensor([2.0]))
    optimizer = torch.optim.AdamW(
        [
            {"params": [regularized], "weight_decay": 0.333},
            {"params": [unregularized], "weight_decay": 0.0},
        ],
        lr=5e-4,
    )
    regularized.grad = torch.tensor([0.5])
    unregularized.grad = torch.tensor([0.25])
    optimizer.step()
    state_before = {
        parameter: {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in state.items()
        }
        for parameter, state in optimizer.state.items()
    }

    adjusted = transition_optimizer_hyperparameters(
        optimizer,
        source_lr=5e-4,
        target_lr=1e-3,
        source_weight_decay=0.333,
        target_weight_decay=0.3,
    )

    assert adjusted == 1
    assert [group["lr"] for group in optimizer.param_groups] == [1e-3, 1e-3]
    assert [group[INITIAL_LR_FIELD] for group in optimizer.param_groups] == [1e-3, 1e-3]
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.3, 0.0]
    for parameter, state in optimizer.state.items():
        for key, value in state.items():
            expected = state_before[parameter][key]
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, expected)
            else:
                assert value == expected


def test_transition_optimizer_hyperparameters_rejects_unexpected_loaded_lr():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3, weight_decay=0.333)

    with pytest.raises(RuntimeError, match="loaded LR"):
        transition_optimizer_hyperparameters(
            optimizer,
            source_lr=5e-4,
            target_lr=1e-3,
            source_weight_decay=0.333,
            target_weight_decay=0.333,
        )


def test_transition_optimizer_hyperparameters_rejects_unexpected_weight_decay():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=5e-4, weight_decay=0.1)

    with pytest.raises(RuntimeError, match="neither zero nor expected"):
        transition_optimizer_hyperparameters(
            optimizer,
            source_lr=5e-4,
            target_lr=1e-3,
            source_weight_decay=0.333,
            target_weight_decay=0.333,
        )
