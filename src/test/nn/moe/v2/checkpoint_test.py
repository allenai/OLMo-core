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


def test_normalize_olmo_ddp_checkpoint_state_rejects_unexpected_names():
    with pytest.raises(RuntimeError, match="Unexpected OLMoDDP checkpoint tensor names"):
        normalize_olmo_ddp_checkpoint_state(
            {
                "module.embeddings.weight.main": torch.ones(2, 3),
                "model.embeddings.weight": torch.ones(2, 3),
            }
        )
