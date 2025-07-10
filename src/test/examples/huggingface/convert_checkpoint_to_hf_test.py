from itertools import product
from pathlib import Path
from typing import List, Optional

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, Olmo2Config, PreTrainedModel

from examples.huggingface.convert_checkpoint_to_hf import convert_checkpoint_to_hf
from olmo_core.config import DType
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.nn.attention import SlidingWindowAttentionConfig
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.nn.transformer.model import Transformer
from olmo_core.testing.utils import DEVICES, get_default_device


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


def _validate_models_match(
    hf_model: PreTrainedModel,
    olmo_core_model: Transformer,
    batch_size: int = 4,
    seq_len: int = 8,
    device: torch.device | None = None,
):
    min_vocab_size = min(int(hf_model.vocab_size), olmo_core_model.vocab_size)
    assert min_vocab_size > 0

    # Move models to device
    device = device or get_default_device()
    hf_model = hf_model.to(device)
    olmo_core_model = olmo_core_model.to(device)

    # Put both models in eval mode
    hf_model.eval()
    olmo_core_model.eval()

    # Generate random input on the same device
    rand_input = torch.randint(0, min_vocab_size, (batch_size, seq_len), device=device)

    # Forward pass through both models
    with torch.no_grad():
        hf_logits, *_ = hf_model(input_ids=rand_input, return_dict=False)
        logits = olmo_core_model(input_ids=rand_input)

    # Validate output shapes
    expected_shape = (batch_size, seq_len)
    assert hf_logits.shape[:2] == expected_shape, (
        f"HF model output shape mismatch: expected {expected_shape}, got {hf_logits.shape[:2]}"
    )
    assert logits.shape[:2] == expected_shape, (
        f"OLMo model output shape mismatch: expected {expected_shape}, got {logits.shape[:2]}"
    )

    assert hf_logits.shape[-1] == hf_model.vocab_size, (
        f"HF vocab size mismatch: expected {hf_model.vocab_size}, got {hf_logits.shape[-1]}"
    )
    assert logits.shape[-1] == olmo_core_model.vocab_size, (
        f"OLMo vocab size mismatch: expected {olmo_core_model.vocab_size}, got {logits.shape[-1]}"
    )

    # Compare logits (accounting for potential vocab size differences)
    torch.testing.assert_close(
        hf_logits[..., :min_vocab_size],
        logits[..., :min_vocab_size],
        rtol=1e-5,
        atol=1e-5,
        msg=f"Logits mismatch for batch_size={batch_size}, seq_len={seq_len}",
    )

    # Additional validation: parameter count
    hf_param_count = sum(p.numel() for p in hf_model.parameters())
    olmo_param_count = sum(p.numel() for p in olmo_core_model.parameters())

    # Log parameter counts for debugging
    print(f"HF model parameters: {hf_param_count:,}")
    print(f"OLMo model parameters: {olmo_param_count:,}")

    # Test deterministic outputs
    test_input = torch.randint(0, min_vocab_size, (batch_size, seq_len), device=device)
    with torch.no_grad():
        hf_out1, *_ = hf_model(input_ids=test_input, return_dict=False)
        hf_out2, *_ = hf_model(input_ids=test_input, return_dict=False)
        olmo_out1 = olmo_core_model(input_ids=test_input)
        olmo_out2 = olmo_core_model(input_ids=test_input)

    torch.testing.assert_close(hf_out1, hf_out2, msg="HF model not deterministic")
    torch.testing.assert_close(olmo_out1, olmo_out2, msg="OLMo model not deterministic")


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

    # Test with various batch sizes and sequence lengths
    test_configs = [
        (1, 1),  # Single token
        (1, 8),  # Single sequence
        (4, 8),  # Default batch
        (2, 16),  # Different config
        (1, 32),  # Longer sequence
    ]

    for batch_size, seq_len in test_configs:
        _validate_models_match(hf_model, olmo_core_model, batch_size=batch_size, seq_len=seq_len)


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
    convert_checkpoint_to_hf(
        original_checkpoint_path=olmo_core_model_path_with_swa,
        output_path=output_dir,
        transformer_config_dict=transformer_config_with_swa.as_config_dict(),
        tokenizer_config_dict=tokenizer_config.as_config_dict(),
        max_sequence_length=max_sequence_length,
        validate=False,
    )

    # Verify that sliding window was detected and passed to HF config
    hf_config = AutoConfig.from_pretrained(output_dir)

    # The HuggingFace Olmo2Config should have sliding_window attribute
    # when SWA is detected in the model
    assert hasattr(hf_config, "sliding_window"), "HF config should have sliding_window attribute"

    # The sliding window should be set to the expected value
    assert hf_config.sliding_window == expected_sliding_window, (
        f"Expected sliding_window={expected_sliding_window}, got {hf_config.sliding_window}"
    )

    # Additional verifications
    # Verify that the max_position_embeddings is set correctly
    assert hf_config.max_position_embeddings == expected_max_position_embeddings, (
        f"Expected max_position_embeddings={expected_max_position_embeddings}, got {hf_config.max_position_embeddings}"
    )

    # Verify that the model type is correctly set
    assert hf_config.model_type == "olmo2", (
        f"Expected model_type='olmo2', got {hf_config.model_type}"
    )

    # Verify that the vocab size matches the tokenizer config
    assert hf_config.vocab_size == tokenizer_config.vocab_size, (
        f"Expected vocab_size={tokenizer_config.vocab_size}, got {hf_config.vocab_size}"
    )

    # Verify that the pad token id matches the tokenizer config
    assert hf_config.pad_token_id == tokenizer_config.pad_token_id, (
        f"Expected pad_token_id={tokenizer_config.pad_token_id}, got {hf_config.pad_token_id}"
    )


@pytest.mark.parametrize("device", DEVICES)  # Different devices
@pytest.mark.parametrize("dtype", [DType.float32, DType.bfloat16])  # Different dtypes
@pytest.mark.parametrize("window_size", [64, 512])  # Different window sizes
@pytest.mark.parametrize("sequence_length", [256, 1024])  # Different sequence lengths
def test_convert_checkpoint_to_hf_with_swa_model(
    tmp_path: Path,
    olmo_core_model_path_with_swa: Path,
    tokenizer_config: TokenizerConfig,
    transformer_config_with_swa: TransformerConfig,
    sequence_length: int,
    window_size: int,
    dtype: DType,
    device: torch.device,
):
    if dtype == DType.bfloat16 and device.type == "cpu":
        pytest.skip("bfloat16 dtype requires cuda")
    if window_size > sequence_length:
        pytest.skip("Skipping test because window size is bigger than sequence length")

    output_dir = tmp_path / f"hf-output-swa-{dtype.value}"

    # Override the window size and model dtype for testing
    test_config = transformer_config_with_swa.as_config_dict()
    test_config["block"]["attention"]["sliding_window"]["pattern"] = [
        window_size,
        window_size,
        window_size,
        -1,
    ]
    test_config["dtype"] = dtype.value

    convert_checkpoint_to_hf(
        original_checkpoint_path=olmo_core_model_path_with_swa,
        output_path=output_dir,
        transformer_config_dict=test_config,
        tokenizer_config_dict=tokenizer_config.as_config_dict(),
        max_sequence_length=sequence_length,
        dtype=dtype,
        device=device,
        debug=True,
        validation_sliding_window=window_size,  # Use the first window size for validation
    )

    # Load both models and verify outputs match
    olmo_core_model = TransformerConfig.from_dict(test_config).build()
    load_model_and_optim_state(
        olmo_core_model_path_with_swa / "model_and_optim", model=olmo_core_model
    )
    hf_model = AutoModelForCausalLM.from_pretrained(output_dir)

    # Verify model dtypes match the requested dtype
    expected_dtype = dtype.as_pt()
    hf_param_dtype = next(hf_model.parameters()).dtype
    assert hf_param_dtype == expected_dtype, (
        f"HF model dtype mismatch: expected {expected_dtype}, got {hf_param_dtype}"
    )
    olmo_param_dtype = next(olmo_core_model.parameters()).dtype
    assert olmo_param_dtype == expected_dtype, (
        f"OLMo model dtype mismatch: expected {expected_dtype}, got {olmo_param_dtype}"
    )

    _validate_models_match(hf_model, olmo_core_model, batch_size=2, seq_len=sequence_length)
