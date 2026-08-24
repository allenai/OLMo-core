import pytest
import torch

from olmo_core.distributed.checkpoint import (
    load_olmo_ddp_checkpoint_state,
    normalize_olmo_ddp_checkpoint_state,
    save_state_dict,
)


def test_load_olmo_ddp_checkpoint_state(tmp_path):
    expected = {
        "module.embeddings.weight.main": torch.arange(6).reshape(2, 3),
        "module.blocks.0.router.weight.main": torch.arange(8).reshape(2, 4),
    }
    save_state_dict(
        tmp_path,
        {
            **expected,
            "optim.module.embeddings.weight.exp_avg": torch.ones(2, 3),
        },
    )

    actual = load_olmo_ddp_checkpoint_state(tmp_path)

    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])


def test_load_olmo_ddp_checkpoint_state_rejects_non_ddp_checkpoint(tmp_path):
    save_state_dict(tmp_path, {"model.weight": torch.ones(2, 3)})

    with pytest.raises(RuntimeError, match="No OLMoDDP model tensors"):
        load_olmo_ddp_checkpoint_state(tmp_path)


def test_normalize_olmo_ddp_checkpoint_state():
    expected = {
        "embeddings.weight": torch.arange(6).reshape(2, 3),
        "blocks.0.router.weight": torch.arange(8).reshape(2, 4),
    }

    actual = normalize_olmo_ddp_checkpoint_state(
        {f"module.{name}.main": value for name, value in expected.items()}
    )

    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])


def test_normalize_olmo_ddp_checkpoint_state_restores_flat_parameter_shapes():
    model_state = {
        "embeddings.weight": torch.empty(2, 3),
        "blocks.0.router.weight": torch.empty(2, 4),
    }
    checkpoint_state = {
        f"module.{name}.main": torch.arange(parameter.numel())
        for name, parameter in model_state.items()
    }

    actual = normalize_olmo_ddp_checkpoint_state(checkpoint_state, model_state)

    assert {name: value.shape for name, value in actual.items()} == {
        name: value.shape for name, value in model_state.items()
    }


def test_normalize_olmo_ddp_checkpoint_state_rejects_wrong_numel():
    with pytest.raises(RuntimeError, match="has 5 elements, expected 6"):
        normalize_olmo_ddp_checkpoint_state(
            {"module.embeddings.weight.main": torch.ones(5)},
            {"embeddings.weight": torch.empty(2, 3)},
        )


def test_normalize_olmo_ddp_checkpoint_state_rejects_missing_model_parameter():
    with pytest.raises(RuntimeError, match="missing model parameters"):
        normalize_olmo_ddp_checkpoint_state(
            {"module.embeddings.weight.main": torch.ones(6)},
            {
                "embeddings.weight": torch.empty(2, 3),
                "norm.weight": torch.empty(3),
            },
        )


def test_normalize_olmo_ddp_checkpoint_state_rejects_unexpected_names():
    with pytest.raises(RuntimeError, match="Unexpected OLMoDDP checkpoint tensor names"):
        normalize_olmo_ddp_checkpoint_state(
            {
                "module.embeddings.weight.main": torch.ones(2, 3),
                "model.embeddings.weight": torch.ones(2, 3),
            }
        )
