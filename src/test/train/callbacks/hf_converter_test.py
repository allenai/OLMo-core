"""Tests for HFConverterCallback."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from transformers import AutoConfig

from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.distributed.checkpoint import save_model_and_optim_state
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.train.callbacks.checkpointer import CheckpointerCallback
from olmo_core.train.callbacks.hf_converter import HFConverterCallback


@pytest.fixture
def tokenizer_config() -> TokenizerConfig:
    return TokenizerConfig.dolma2()


@pytest.fixture
def transformer_config(tokenizer_config: TokenizerConfig) -> TransformerConfig:
    return TransformerConfig.olmo2_190M(tokenizer_config.padded_vocab_size(), n_layers=2)


def test_post_train_converts_checkpoint(
    tmp_path: Path,
    transformer_config: TransformerConfig,
    tokenizer_config: TokenizerConfig,
):
    """End-to-end: the callback converts a saved checkpoint to HF format."""
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    model = transformer_config.build()
    save_model_and_optim_state(checkpoint_path / "model_and_optim", model)

    # Simulate what ConfigSaverCallback writes to the checkpoint.
    with open(checkpoint_path / "config.json", "w") as f:
        json.dump(
            {
                "model": transformer_config.as_config_dict(),
                "dataset": {"tokenizer": tokenizer_config.as_config_dict()},
            },
            f,
        )

    trainer = Mock()
    trainer.train_module.model = model
    checkpointer = CheckpointerCallback()
    checkpointer._latest_checkpoint_path = str(checkpoint_path)
    checkpointer._trainer = trainer
    trainer.callbacks = {"checkpointer": checkpointer}

    output_folder = tmp_path / "hf_output"
    callback = HFConverterCallback(
        enabled=True,
        output_folder=str(output_folder),
        validate=False,
    )
    callback._trainer = trainer
    callback.post_train()

    assert output_folder.exists()
    hf_config = AutoConfig.from_pretrained(output_folder)
    assert hf_config.hidden_size == transformer_config.d_model
    assert hf_config.num_hidden_layers == transformer_config.n_layers
