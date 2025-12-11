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
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.nn.transformer.model import Transformer

try:
    from transformers import Olmo3Config
except ImportError:
    Olmo3Config = None  # type: ignore


@pytest.fixture
def tokenizer_config() -> TokenizerConfig:
    return TokenizerConfig.dolma2()


@pytest.fixture(
    params=["olmo2", "olmo3"],
    ids=["OLMo2-190M", "OLMo3-190M"],
)
def model_config(request, tokenizer_config: TokenizerConfig) -> tuple[str, TransformerConfig]:
    """Returns (model_family, transformer_config) tuple."""
    model_family = request.param

    if model_family == "olmo2":
        config = TransformerConfig.olmo2_190M(tokenizer_config.padded_vocab_size(), n_layers=4)
    elif model_family == "olmo3":
        config = TransformerConfig.olmo3_190M(
            tokenizer_config.padded_vocab_size(),
            attn_backend=AttentionBackendName.torch,  # Use torch backend for testing
            n_layers=4,
        )
    else:
        raise ValueError(f"Unknown model family: {model_family}")

    return model_family, config


@pytest.fixture
def olmo_core_model_path(
    tmp_path: Path, model_config: tuple[str, TransformerConfig]
) -> tuple[str, Path]:
    """Returns (model_family, model_path) tuple."""
    model_family, transformer_config = model_config
    model = transformer_config.build()

    model_path = tmp_path / f"olmo_core_{model_family}"
    save_model_and_optim_state(model_path / "model_and_optim", model)
    del model
    return model_family, model_path


def _get_expected_hf_config(
    model_family: str,
    transformer_config: TransformerConfig,
    tokenizer_config: TokenizerConfig,
) -> Olmo2Config | Olmo3Config:
    """Build expected HF config based on model family."""
    common_config = {
        "vocab_size": tokenizer_config.vocab_size,
        "hidden_size": transformer_config.d_model,
        "intermediate_size": 3072,
        "num_hidden_layers": transformer_config.n_layers,
        "num_attention_heads": 12,
        "rope_theta": 500_000,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 256,
        "pad_token_id": tokenizer_config.pad_token_id,
        "bos_token_id": tokenizer_config.bos_token_id,
        "eos_token_id": tokenizer_config.eos_token_id,
        "torch_dtype": torch.float32,
    }

    if model_family == "olmo2":
        return Olmo2Config(
            architectures=["Olmo2ForCausalLM"],
            **common_config,
        )
    elif model_family == "olmo3":
        if Olmo3Config is None:
            pytest.skip("The installed transformers version does not support Olmo3")

        # Compute expected layer_types and sliding_window from the transformer config
        sliding_window_config = transformer_config.block.attention.sliding_window
        assert sliding_window_config is not None

        pattern = sliding_window_config.pattern
        n_layers = transformer_config.n_layers
        force_first = sliding_window_config.force_full_attention_on_first_layer
        force_last = sliding_window_config.force_full_attention_on_last_layer

        # Generate layer types based on swa pattern
        layer_types = []
        for i in range(n_layers):
            if i == 0 and force_first:
                layer_types.append("full_attention")
            elif i == n_layers - 1 and force_last:
                layer_types.append("full_attention")
            else:
                window = pattern[i % len(pattern)]
                layer_types.append("full_attention" if window == -1 else "sliding_attention")

        # Find the sliding window size (first non -1 value in pattern)
        window_size = next((w for w in pattern if w != -1), None)
        assert window_size is not None

        return Olmo3Config(
            architectures=["Olmo3ForCausalLM"],
            sliding_window=window_size,
            layer_types=layer_types,
            **common_config,
        )
    else:
        raise ValueError(f"Unknown model family: {model_family}")


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

    # Using torch backend w/ validation on cpu we get bitwise identical results when comparing
    # logit outputs from an OlmoCore model and its HF-converted counterpart
    assert torch.equal(hf_logits[..., :min_vocab_size], logits[..., :min_vocab_size])


def test_convert_checkpoint_to_hf_correct_config(
    tmp_path: Path,
    olmo_core_model_path: tuple[str, Path],
    model_config: tuple[str, TransformerConfig],
    tokenizer_config: TokenizerConfig,
):
    model_family, model_path = olmo_core_model_path
    _, transformer_config = model_config

    expected_hf_config = _get_expected_hf_config(model_family, transformer_config, tokenizer_config)

    output_dir = tmp_path / f"hf-output-{model_family}"
    convert_checkpoint_to_hf(
        original_checkpoint_path=model_path,
        output_path=output_dir,
        transformer_config_dict=transformer_config.as_config_dict(),
        tokenizer_config_dict=tokenizer_config.as_config_dict(),
        max_sequence_length=256,
        validate=False,
    )
    hf_config = AutoConfig.from_pretrained(output_dir)

    assert hf_config.to_diff_dict() == expected_hf_config.to_diff_dict()


def test_convert_checkpoint_to_hf_correct_model(
    tmp_path: Path,
    olmo_core_model_path: tuple[str, Path],
    model_config: tuple[str, TransformerConfig],
    tokenizer_config: TokenizerConfig,
):
    model_family, model_path = olmo_core_model_path
    _, transformer_config = model_config

    if model_family == "olmo3" and Olmo3Config is None:
        pytest.skip("The installed transformers version does not support Olmo3")

    output_dir = tmp_path / f"hf-output-{model_family}"
    convert_checkpoint_to_hf(
        original_checkpoint_path=model_path,
        output_path=output_dir,
        transformer_config_dict=transformer_config.as_config_dict(),
        tokenizer_config_dict=tokenizer_config.as_config_dict(),
        max_sequence_length=256,
        debug=True,
    )

    olmo_core_model = transformer_config.build()
    load_model_and_optim_state(model_path / "model_and_optim", model=olmo_core_model)

    hf_model = AutoModelForCausalLM.from_pretrained(output_dir)

    _validate_models_match(hf_model, olmo_core_model)
