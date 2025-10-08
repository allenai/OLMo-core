import json
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, Olmo2Config, PreTrainedModel

from examples.huggingface.convert_checkpoint_from_hf import convert_checkpoint_from_hf
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.distributed.checkpoint import load_model_and_optim_state
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
        config = TransformerConfig.olmo2_190M(tokenizer_config.padded_vocab_size())
    elif model_family == "olmo3":
        config = TransformerConfig.olmo3_190M(
            tokenizer_config.padded_vocab_size(),
            attn_backend=AttentionBackendName.torch,  # Use torch backend for testing
        )
    else:
        raise ValueError(f"Unknown model family: {model_family}")

    return model_family, config


@pytest.fixture
def hf_model_path(
    tmp_path: Path, tokenizer_config: TokenizerConfig, model_config: tuple[str, TransformerConfig]
) -> Path:
    model_family, transformer_config = model_config
    common_config = {
        "vocab_size": tokenizer_config.vocab_size,
        "hidden_size": transformer_config.d_model,
        "intermediate_size": 3072,
        "num_hidden_layers": transformer_config.n_layers,
        "num_attention_heads": 12,
        "rope_theta": 500_000,
        "rms_norm_eps": 1e-6,
    }

    if model_family == "olmo2":
        hf_config = Olmo2Config(**common_config)
    elif model_family == "olmo3":
        pytest.skip("The installed transformers version does not support Olmo3")
        hf_config = Olmo3Config(**common_config)
    else:
        raise ValueError(f"Unknown model family: {model_family}")

    hf_model = AutoModelForCausalLM.from_config(hf_config)

    model_path = tmp_path / f"hf-{model_family}"
    hf_model.save_pretrained(model_path)
    del hf_model
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

    # Using torch backend w/ validation on cpu we get bitwise identical results when comparing
    # logit outputs from an OlmoCore model and its HF-converted counterpart
    assert torch.equal(hf_logits[..., :min_vocab_size], logits[..., :min_vocab_size])


def test_convert_checkpoint_from_hf_correct_config(
    tmp_path: Path,
    hf_model_path: Path,
    tokenizer_config: TokenizerConfig,
    model_config: tuple[str, TransformerConfig],
):
    model_family, transformer_config = model_config
    output_dir = tmp_path / f"olmo_core-{model_family}"
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
    model_config: tuple[str, TransformerConfig],
):
    model_family, transformer_config = model_config
    output_dir = tmp_path / f"olmo_core-output-{model_family}"
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
