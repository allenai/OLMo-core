import json
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, Olmo2Config, PreTrainedModel

from examples.huggingface.convert_checkpoint_from_hf import convert_checkpoint_from_hf
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.nn.transformer.model import Transformer


@pytest.fixture
def tokenizer_config() -> TokenizerConfig:
    return TokenizerConfig.dolma2()


@pytest.fixture
def transformer_config(tokenizer_config: TokenizerConfig) -> TransformerConfig:
    return TransformerConfig.olmo2_190M(tokenizer_config.padded_vocab_size())


@pytest.fixture
def hf_model_path(
    tmp_path: Path, tokenizer_config: TokenizerConfig, transformer_config: TransformerConfig
) -> Path:
    hf_config = Olmo2Config(
        vocab_size=tokenizer_config.vocab_size,
        hidden_size=transformer_config.d_model,
        intermediate_size=3072,
        num_hidden_layers=transformer_config.n_layers,
        num_attention_heads=12,
        rope_theta=500_000,
        rms_norm_eps=1e-6,
    )
    hf_model = AutoModelForCausalLM.from_config(hf_config)

    model_path = tmp_path / "hf"
    hf_model.save_pretrained(tmp_path / "hf")
    del hf_model
    return model_path


@pytest.fixture
def olmo_core_model_path(tmp_path: Path, transformer_config: TransformerConfig) -> Path:
    model = transformer_config.build()

    model_path = tmp_path / "olmo_core"
    save_model_and_optim_state(model_path, model)
    del model
    return model_path


def _validate_models_match(hf_model: PreTrainedModel, olmo_core_model: Transformer):
    min_vocab_size = min(int(hf_model.vocab_size), olmo_core_model.vocab_size)
    assert min_vocab_size > 0

    rand_input = torch.randint(0, min_vocab_size, (4, 8))
    with torch.no_grad():
        hf_logits, *_ = hf_model(input_ids=rand_input, return_dict=False)

    olmo_core_model.eval()
    with torch.no_grad():
        logits = olmo_core_model(input_ids=rand_input)

    assert hf_logits.shape[-1] == hf_model.vocab_size
    assert logits.shape[-1] == olmo_core_model.vocab_size
    torch.testing.assert_close(hf_logits[..., :min_vocab_size], logits[..., :min_vocab_size])


def test_convert_checkpoint_from_hf_correct_config(
    tmp_path: Path,
    hf_model_path: Path,
    tokenizer_config: TokenizerConfig,
    transformer_config: TransformerConfig,
):
    output_dir = tmp_path / "olmo_core"
    convert_checkpoint_from_hf(
        hf_checkpoint_path=hf_model_path,
        output_path=output_dir,
        transformer_config_dict=transformer_config.as_config_dict(),
        tokenizer_config_dict=tokenizer_config.as_config_dict(),
        validate=False,
    )

    assert (output_dir / "config.json").is_file()
    experiment_config_dict = json.loads((output_dir / "config.json").read_text())

    assert "model" in experiment_config_dict
    assert experiment_config_dict["model"] == transformer_config.as_config_dict()

    assert "dataset" in experiment_config_dict
    assert "tokenizer" in experiment_config_dict["dataset"]
    assert experiment_config_dict["dataset"]["tokenizer"] == tokenizer_config.as_config_dict()


def test_convert_checkpoint_from_hf_correct_model(
    tmp_path: Path,
    hf_model_path: Path,
    tokenizer_config: TokenizerConfig,
    transformer_config: TransformerConfig,
):
    output_dir = tmp_path / "olmo_core-output"
    convert_checkpoint_from_hf(
        hf_checkpoint_path=hf_model_path,
        output_path=output_dir,
        transformer_config_dict=transformer_config.as_config_dict(),
        tokenizer_config_dict=tokenizer_config.as_config_dict(),
        validate=False,
    )

    olmo_core_model = transformer_config.build()
    load_model_and_optim_state(output_dir / "model_and_optim", olmo_core_model)

    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_path)

    _validate_models_match(hf_model, olmo_core_model)
