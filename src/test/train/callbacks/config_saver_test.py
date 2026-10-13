import json

import pytest

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.train.callbacks.config_saver import (
    ConfigSaverCallback,
    checkpoint_compatibility_config,
    config_differences,
)


def compatibility_config(*, weight_decay: float = 0.1):
    return {
        "model": {
            "_CLASS_": "TransformerConfig",
            "d_model": 1024,
            "n_layers": 16,
            "rope": {"theta": 500000.0},
        },
        "train_module": {
            "_CLASS_": "TransformerTrainModuleConfig",
            "rank_microbatch_size": 8192,
            "max_sequence_length": 2048,
            "optim": {
                "_CLASS_": "AdamWConfig",
                "lr": 0.001,
                "weight_decay": weight_decay,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
            },
            "scheduler": {"_CLASS_": "WSD", "warmup": 1000},
            "autocast_precision": "bfloat16",
        },
        "init_seed": 12536,
        "dataset": {"paths": ["intentionally", "not", "compared"]},
    }


def test_checkpoint_compatibility_config_selects_complete_model_and_train_module():
    selected = checkpoint_compatibility_config(compatibility_config())

    assert set(selected) == {"model", "train_module", "init_seed"}
    assert selected["model"]["rope"]["theta"] == 500000.0
    assert selected["train_module"]["optim"]["betas"] == [0.9, 0.95]
    assert "rank_microbatch_size" not in selected["train_module"]


def test_config_differences_identifies_exact_nested_optimizer_mismatch():
    expected = checkpoint_compatibility_config(compatibility_config(weight_decay=0.1))
    actual = checkpoint_compatibility_config(compatibility_config(weight_decay=0.033))

    assert config_differences(expected, actual) == [
        "train_module.optim.weight_decay: checkpoint 0.033 != command 0.1"
    ]


def test_checkpoint_compatibility_config_requires_model_train_module_and_seed():
    config = compatibility_config()
    del config["model"]

    with pytest.raises(
        OLMoConfigurationError, match="missing required compatibility field 'model'"
    ):
        checkpoint_compatibility_config(config)


def test_runtime_microbatch_change_is_checkpoint_compatible():
    expected_config = compatibility_config()
    checkpoint_config = compatibility_config()
    checkpoint_config["train_module"]["rank_microbatch_size"] = 16384

    expected = checkpoint_compatibility_config(expected_config)
    actual = checkpoint_compatibility_config(checkpoint_config)

    assert config_differences(expected, actual) == []


def test_pre_checkpoint_load_validation_reads_saved_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(compatibility_config()))
    callback = ConfigSaverCallback(validate_checkpoint_config=True)
    callback._config = compatibility_config()

    callback.pre_checkpoint_loaded(tmp_path)


def test_pre_checkpoint_load_validation_is_a_noop_when_disabled(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(compatibility_config(weight_decay=0.033)))
    callback = ConfigSaverCallback()
    callback._config = compatibility_config(weight_decay=0.1)

    callback.pre_checkpoint_loaded(tmp_path)


def test_pre_checkpoint_load_validation_aborts_before_mismatched_restore(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(compatibility_config(weight_decay=0.033)))
    callback = ConfigSaverCallback(validate_checkpoint_config=True)
    callback._config = compatibility_config(weight_decay=0.1)

    with pytest.raises(
        OLMoConfigurationError,
        match=r"train_module\.optim\.weight_decay: checkpoint 0\.033 != command 0\.1",
    ):
        callback.pre_checkpoint_loaded(tmp_path)
