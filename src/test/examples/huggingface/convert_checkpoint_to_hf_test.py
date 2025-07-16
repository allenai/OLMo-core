import gc
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig, Olmo2Config

from examples.huggingface.convert_checkpoint_to_hf import (
    convert_checkpoint_to_hf,
    load_olmo_model,
    validate_conversion,
)
from olmo_core.config import DType
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.distributed.checkpoint import (
    save_model_and_optim_state,
)
from olmo_core.nn.attention import SlidingWindowAttentionConfig
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.testing.utils import DEVICES, get_default_device


@pytest.fixture(autouse=True)
def cleanup_memory():
    """Fixture to clean up memory between tests."""
    yield
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture
def tokenizer_config() -> TokenizerConfig:
    return TokenizerConfig.dolma2()


@pytest.fixture
def transformer_config(tokenizer_config: TokenizerConfig) -> TransformerConfig:
    return TransformerConfig.olmo2_190M(tokenizer_config.padded_vocab_size())


@pytest.fixture
def transformer_config_with_swa(tokenizer_config: TokenizerConfig) -> TransformerConfig:
    """Create a transformer config with sliding window attention."""
    config = TransformerConfig.olmo2_190M(
        tokenizer_config.padded_vocab_size(),
        use_flash=True,
        n_kv_heads=4,
    )

    # Set up sliding window attention with a pattern
    # Pattern: [4096, 4096, 4096, -1] means 3 layers with window size 4096, then 1 layer with full attention
    config.block.attention.sliding_window = SlidingWindowAttentionConfig(
        pattern=[4096, 4096, 4096, -1],
        force_full_attention_on_first_layer=True,
        force_full_attention_on_last_layer=True,
    )
    config.block.attention.use_head_qk_norm = True

    return config


@pytest.fixture
def olmo_core_model_path(tmp_path: Path, transformer_config: TransformerConfig) -> Path:
    model = transformer_config.build()

    model_path = tmp_path / "olmo_core"
    save_model_and_optim_state(model_path / "model_and_optim", model)
    del model
    return model_path


@pytest.fixture
def olmo_core_model_path_with_swa(
    tmp_path: Path, transformer_config_with_swa: TransformerConfig
) -> Path:
    model = transformer_config_with_swa.build()

    model_path = tmp_path / "olmo_core_swa"
    save_model_and_optim_state(model_path / "model_and_optim", model)
    del model
    return model_path


def test_convert_checkpoint_to_hf_correct_config(
    tmp_path: Path,
    olmo_core_model_path: Path,
    tokenizer_config: TokenizerConfig,
    transformer_config: TransformerConfig,
):
    output_dir = tmp_path / "hf-output"

    # Load the original OLMo-core model from checkpoint
    olmo_core_model = load_olmo_model(
        olmo_core_model_path, transformer_config, get_default_device()
    )

    # Convert and save to HuggingFace format
    convert_checkpoint_to_hf(
        olmo_core_model,
        output_dir,
        tokenizer_config,
        max_sequence_length=256,
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


@pytest.mark.parametrize(
    "max_sequence_length, expected_sliding_window, expected_max_position_embeddings",
    [
        (8192 * 2, 4096, 8192 * 2),
        (4096 * 2, 2048, 4096 * 2),
        (1024 * 2, 512, 1024 * 2),
    ],
)
def test_convert_checkpoint_to_hf_with_swa_config(
    tmp_path: Path,
    olmo_core_model_path_with_swa: Path,
    tokenizer_config: TokenizerConfig,
    transformer_config_with_swa: TransformerConfig,
    max_sequence_length: int,
    expected_sliding_window: int,
    expected_max_position_embeddings: int,
):
    output_dir = tmp_path / "hf-output-swa"

    # Load the original OLMo-core model from checkpoint
    olmo_core_model = load_olmo_model(
        olmo_core_model_path_with_swa, transformer_config_with_swa, get_default_device()
    )

    # Convert and save to HuggingFace format
    convert_checkpoint_to_hf(
        olmo_core_model,
        output_dir,
        tokenizer_config,
        max_sequence_length=max_sequence_length,
    )

    hf_config = AutoConfig.from_pretrained(output_dir)

    # Verify that sliding window was detected and passed to HF config
    assert hasattr(hf_config, "sliding_window"), "HF config should have sliding_window attribute"
    assert hf_config.sliding_window == expected_sliding_window, (
        f"Expected sliding_window={expected_sliding_window}, got {hf_config.sliding_window}"
    )
    assert hf_config.max_position_embeddings == expected_max_position_embeddings, (
        f"Expected max_position_embeddings={expected_max_position_embeddings}, got {hf_config.max_position_embeddings}"
    )
    assert hf_config.model_type == "olmo3", (
        f"Expected model_type='olmo3', got {hf_config.model_type}"
    )
    assert hf_config.vocab_size == tokenizer_config.vocab_size, (
        f"Expected vocab_size={tokenizer_config.vocab_size}, got {hf_config.vocab_size}"
    )
    assert hf_config.pad_token_id == tokenizer_config.pad_token_id, (
        f"Expected pad_token_id={tokenizer_config.pad_token_id}, got {hf_config.pad_token_id}"
    )

    # Verify the GenerationConfig is as expected
    generation_config = hf_config.generation_config
    assert generation_config is not None, "GenerationConfig should not be None"
    assert generation_config.pad_token_id == tokenizer_config.pad_token_id, (
        f"Expected pad_token_id={tokenizer_config.pad_token_id}, got {generation_config.pad_token_id}"
    )
    assert generation_config.eos_token_id == tokenizer_config.eos_token_id, (
        f"Expected eos_token_id={tokenizer_config.eos_token_id}, got {generation_config.eos_token_id}"
    )


@pytest.mark.parametrize("window_size", [1, 8, 128, None])
@pytest.mark.parametrize("sequence_length", [64, 256])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [DType.float32, DType.bfloat16])  #  DType.float64
@pytest.mark.parametrize("use_natural_language", [True, False])
def test_convert_checkpoint_to_hf_correct_model(
    tmp_path: Path,
    tokenizer_config: TokenizerConfig,
    sequence_length: int,
    window_size: int | None,
    dtype: DType,
    device: torch.device,
    use_natural_language: bool,
):
    if dtype == DType.bfloat16 and device.type == "cpu":
        pytest.skip("bfloat16 dtype requires cuda")
    if window_size is not None and window_size > sequence_length:
        pytest.skip("window size is bigger than sequence length")

    # Create a transformer config with the specific window size and dtype for this test
    transformer_config = TransformerConfig.olmo2_190M(
        tokenizer_config.padded_vocab_size(),
        use_flash=True,
        n_kv_heads=4,
        dtype=dtype,  # Pass dtype during config creation so nested configs get it too
    )
    transformer_config.block.attention.use_head_qk_norm = True
    if window_size is not None:
        transformer_config.block.attention.sliding_window = SlidingWindowAttentionConfig(
            pattern=[window_size, window_size, window_size, -1],
            force_full_attention_on_first_layer=True,
            force_full_attention_on_last_layer=True,
        )

    olmo_core_model = transformer_config.build().to(device)

    target_dtype = dtype.as_pt()
    # Check that all parameters have the expected dtype
    for name, param in olmo_core_model.named_parameters():
        assert param.dtype == target_dtype, (
            f"Parameter {name} dtype mismatch: expected {target_dtype}, got {param.dtype}"
        )

    # Save the model to disk for later validation
    olmo_core_model_path = tmp_path / "olmo_core_model"
    save_model_and_optim_state(
        olmo_core_model_path / "model_and_optim", olmo_core_model, optim=None
    )

    # Convert and save to HuggingFace format
    output_dir = tmp_path / f"hf-output-swa-{dtype.value}"
    convert_checkpoint_to_hf(
        olmo_core_model,
        output_dir,
        tokenizer_config,
        max_sequence_length=sequence_length,
        dtype=dtype,
    )
    del olmo_core_model
    gc.collect()
    torch.cuda.empty_cache()

    validate_conversion(
        hf_path=output_dir,
        olmo_core_path=olmo_core_model_path,
        olmo_core_model_config=transformer_config,
        vocab_size=tokenizer_config.vocab_size,
        batch_size=2,
        sequence_length=sequence_length,
        dtype=dtype,
        device=device,
        debug=True,
        use_natural_language=use_natural_language,
    )
