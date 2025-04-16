from pathlib import Path

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, Olmo2Config, PreTrainedModel

from examples.huggingface.convert_checkpoint_to_hf import convert_checkpoint_to_hf
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
def olmo_core_model_path(tmp_path: Path, transformer_config: TransformerConfig) -> Path:
    model = transformer_config.build()

    model_path = tmp_path / "olmo_core"
    save_model_and_optim_state(model_path / "model_and_optim", model)
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


def test_convert_checkpoint_to_hf_correct_config(
    tmp_path: Path,
    olmo_core_model_path: Path,
    tokenizer_config: TokenizerConfig,
    transformer_config: TransformerConfig,
):
    output_dir = tmp_path / "hf-output"
    convert_checkpoint_to_hf(
        original_checkpoint_path=olmo_core_model_path,
        output_path=output_dir,
        transformer_config_dict=transformer_config.as_config_dict(),
        tokenizer_config_dict=tokenizer_config.as_config_dict(),
        max_sequence_length=256,
        validate=False,
    )

    expected_hf_config = Olmo2Config(
        architectures=["Olmo2ForCausalLM"],
        vocab_size=tokenizer_config.vocab_size,
        hidden_size=transformer_config.d_model,
        intermediate_size=3072,
        num_hidden_layers=transformer_config.n_layers,
        num_attention_heads=12,
        rope_theta=500_000,
        rms_norm_eps=1e-6,
        max_position_embeddings=256,
        pad_token_id=tokenizer_config.pad_token_id,
        bos_token_id=tokenizer_config.bos_token_id,
        eos_token_id=tokenizer_config.eos_token_id,
        torch_dtype=torch.float32,
    )

    hf_config = AutoConfig.from_pretrained(output_dir)

    assert hf_config.to_diff_dict() == expected_hf_config.to_diff_dict()


def test_convert_checkpoint_to_hf_correct_model(
    tmp_path: Path,
    olmo_core_model_path: Path,
    tokenizer_config: TokenizerConfig,
    transformer_config: TransformerConfig,
):
    output_dir = tmp_path / "hf-output"
    convert_checkpoint_to_hf(
        original_checkpoint_path=olmo_core_model_path,
        output_path=output_dir,
        transformer_config_dict=transformer_config.as_config_dict(),
        tokenizer_config_dict=tokenizer_config.as_config_dict(),
        max_sequence_length=256,
        debug=True,
    )

    olmo_core_model = transformer_config.build()
    load_model_and_optim_state(olmo_core_model_path / "model_and_optim", model=olmo_core_model)

    hf_model = AutoModelForCausalLM.from_pretrained(output_dir)

    _validate_models_match(hf_model, olmo_core_model)
