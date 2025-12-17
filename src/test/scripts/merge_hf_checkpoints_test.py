"""
Tests for the merge_hf_checkpoints.py script.
"""
import gc
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import torch
from click.testing import CliRunner
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Import the merge_hf_checkpoints module
spec = importlib.util.spec_from_file_location(
    "merge_hf_checkpoints", "src/scripts/merge_hf_checkpoints.py"
)
if spec is None or spec.loader is None:
    raise ImportError("Could not load merge_hf_checkpoints.py")
merge_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merge_module)


# ==================== Helper Functions ====================


def create_hf_checkpoint_with_seed(
    checkpoint_dir: Path,
    vocab_size: int = 100,
    hidden_size: int = 32,
    num_hidden_layers: int = 2,
    num_attention_heads: int = 2,
    seed: int = 42,
    dtype: torch.dtype = torch.float32,
    model_type: str = "olmo3",
) -> None:
    """
    Create a minimal HuggingFace checkpoint with a specific random seed for distinct weights.

    Args:
        checkpoint_dir: Directory to save the checkpoint
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension size
        num_hidden_layers: Number of layers
        num_attention_heads: Number of attention heads
        seed: Random seed for weight initialization
        dtype: Data type for model weights
        model_type: Model architecture type
    """

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed for reproducible weights
    torch.manual_seed(seed)

    # Create minimal config
    config_dict = {
        "model_type": model_type,
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "activation_function": "gelu",
        "resid_pdrop": 0.0,
        "embd_pdrop": 0.0,
        "attn_pdrop": 0.0,
        "layer_norm_epsilon": 1e-5,
        "initializer_range": 0.02,
        "bos_token_id": 0,
        "eos_token_id": 1,
    }

    # Save config
    config_path = checkpoint_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    # Create and save model
    config = AutoConfig.from_pretrained(checkpoint_dir)
    model = AutoModelForCausalLM.from_config(config)

    # Convert to specified dtype
    if dtype != torch.float32:
        model = model.to(dtype)

    # Save model
    model.save_pretrained(checkpoint_dir)

    # Create a very minimal tokenizer that should work
    # Use PreTrainedTokenizerFast format which is simpler
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": 1024,
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
    }

    tokenizer_config_path = checkpoint_dir / "tokenizer_config.json"
    with open(tokenizer_config_path, "w") as f:
        json.dump(tokenizer_config, f, indent=2)

    # Create a minimal tokenizer.json for PreTrainedTokenizerFast
    tokenizer_json: Dict[str, Any] = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": 0,
                "content": "<s>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
            {
                "id": 1,
                "content": "</s>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
            {
                "id": 2,
                "content": "<unk>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
            {
                "id": 3,
                "content": "<pad>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "<unk>",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "vocab": {
                "<s>": 0,
                "</s>": 1,
                "<unk>": 2,
                "<pad>": 3,
                **{f"token_{i}": i + 4 for i in range(min(vocab_size - 4, 96))},
            },
            "merges": [],
        },
    }

    tokenizer_path = checkpoint_dir / "tokenizer.json"
    with open(tokenizer_path, "w") as f:
        json.dump(tokenizer_json, f, indent=2)


def run_merge_hf_cli(
    model_paths: List[str],
    output_path: str,
    revisions: Optional[List[str]] = None,
    device: str = "cpu",
):
    """
    Run the merge_hf_checkpoints CLI using Click's test runner.

    Args:
        model_paths: List of checkpoint paths to merge
        output_path: Output directory for merged checkpoint
        revisions: Optional list of revisions
        device: Device to use

    Returns:
        Click test runner result
    """
    runner = CliRunner()
    args = []

    # Add model paths
    for model_path in model_paths:
        args.extend(["--model", model_path])

    # Add revisions if provided
    if revisions:
        for revision in revisions:
            args.extend(["--revisions", revision])

    # Add output path
    args.extend(["--output", output_path])

    # Add device
    args.extend(["--device", device])

    result = runner.invoke(merge_module.main, args)
    return result


def load_hf_checkpoint(checkpoint_dir: Path) -> tuple:
    """
    Load a HuggingFace checkpoint for verification.

    Args:
        checkpoint_dir: Directory containing the checkpoint

    Returns:
        Tuple of (model, config, tokenizer)
    """

    config = AutoConfig.from_pretrained(checkpoint_dir)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

    return model, config, tokenizer


def verify_averaged_weights_hf(
    merged_checkpoint: Path,
    source_checkpoints: List[Path],
    tolerance: float = 1e-5,
) -> None:
    """
    Verify that merged checkpoint weights are correctly averaged.

    Args:
        merged_checkpoint: Path to merged checkpoint
        source_checkpoints: List of source checkpoint paths
        tolerance: Numerical tolerance for comparison
    """
    # Load merged model
    merged_model = AutoModelForCausalLM.from_pretrained(merged_checkpoint)
    merged_state_dict = merged_model.state_dict()

    # Load source models and compute expected average
    source_state_dicts = []
    for ckpt in source_checkpoints:
        model = AutoModelForCausalLM.from_pretrained(ckpt)
        source_state_dicts.append(model.state_dict())
        del model
        gc.collect()

    # Verify each parameter is correctly averaged
    for param_name in merged_state_dict.keys():
        merged_value = merged_state_dict[param_name]

        # Compute expected average
        source_values = [state[param_name] for state in source_state_dicts]

        # Only average floating-point tensors
        if source_values[0].dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16]:
            # Convert all to float32 for averaging
            expected_avg = torch.zeros_like(source_values[0], dtype=torch.float32)
            for v in source_values:
                expected_avg += v.float()
            expected_avg /= len(source_values)
            expected_avg = expected_avg.to(source_values[0].dtype)
        else:
            # Non-floating tensors should be taken from first checkpoint
            expected_avg = source_values[0]

        # Compare
        torch.testing.assert_close(
            merged_value,
            expected_avg,
            atol=tolerance,
            rtol=tolerance,
            msg=f"Parameter '{param_name}' not correctly averaged",
        )


# ==================== Core Functionality Tests ====================


def test_merge_two_checkpoints(tmp_path):
    """Test basic 2-checkpoint merge."""
    # Create two checkpoints with different seeds
    ckpt1 = tmp_path / "checkpoint1"
    ckpt2 = tmp_path / "checkpoint2"
    output = tmp_path / "merged"

    create_hf_checkpoint_with_seed(ckpt1, seed=42)
    create_hf_checkpoint_with_seed(ckpt2, seed=123)

    # Run merge
    result = run_merge_hf_cli([str(ckpt1), str(ckpt2)], str(output))

    # Verify success
    assert result.exit_code == 0, f"Merge failed: {result.output}"
    assert output.exists()
    assert (output / "config.json").exists()
    assert (output / "pytorch_model.bin").exists() or (output / "model.safetensors").exists()
    assert (output / "tokenizer_config.json").exists()


def test_merge_three_checkpoints(tmp_path):
    """Test 3-checkpoint merge to verify averaging works with >2 models."""
    ckpts = [tmp_path / f"checkpoint{i}" for i in range(3)]
    output = tmp_path / "merged"

    for i, ckpt in enumerate(ckpts):
        create_hf_checkpoint_with_seed(ckpt, seed=42 + i)

    # Run merge
    result = run_merge_hf_cli([str(ckpt) for ckpt in ckpts], str(output))

    # Verify success
    assert result.exit_code == 0, f"Merge failed: {result.output}"
    assert output.exists()

    # Verify weights are averaged
    verify_averaged_weights_hf(output, ckpts)


def test_single_checkpoint(tmp_path):
    """Test edge case with single checkpoint (should work as copy)."""
    ckpt = tmp_path / "checkpoint"
    output = tmp_path / "merged"

    create_hf_checkpoint_with_seed(ckpt, seed=42)

    # Run merge with single checkpoint
    result = run_merge_hf_cli([str(ckpt)], str(output))

    # Verify success
    assert result.exit_code == 0, f"Merge failed: {result.output}"
    assert output.exists()

    # Load both models and verify they're identical
    original_model, _, _ = load_hf_checkpoint(ckpt)
    merged_model, _, _ = load_hf_checkpoint(output)

    for (n1, p1), (n2, p2) in zip(
        original_model.named_parameters(), merged_model.named_parameters()
    ):
        assert n1 == n2
        torch.testing.assert_close(p1, p2)


def test_weights_are_averaged_correctly(tmp_path):
    """Test that weights are mathematically correctly averaged."""
    # Create checkpoints with known weights
    ckpt1 = tmp_path / "checkpoint1"
    ckpt2 = tmp_path / "checkpoint2"
    output = tmp_path / "merged"

    # Create with different seeds for different weights
    create_hf_checkpoint_with_seed(ckpt1, seed=1, hidden_size=16, num_hidden_layers=1)
    create_hf_checkpoint_with_seed(ckpt2, seed=2, hidden_size=16, num_hidden_layers=1)

    # Run merge
    result = run_merge_hf_cli([str(ckpt1), str(ckpt2)], str(output))
    assert result.exit_code == 0

    # Verify weights are averaged
    verify_averaged_weights_hf(output, [ckpt1, ckpt2], tolerance=1e-5)


def test_dtype_preservation(tmp_path):
    """Test that dtypes are preserved correctly after merging."""
    ckpt1 = tmp_path / "checkpoint1_fp32"
    ckpt2 = tmp_path / "checkpoint2_fp32"
    output = tmp_path / "merged"

    # Create checkpoints with float32 (test conversion during averaging)
    create_hf_checkpoint_with_seed(ckpt1, seed=42, dtype=torch.float32, hidden_size=16)
    create_hf_checkpoint_with_seed(ckpt2, seed=123, dtype=torch.float32, hidden_size=16)

    # Run merge
    result = run_merge_hf_cli([str(ckpt1), str(ckpt2)], str(output))
    assert result.exit_code == 0

    # Load models and check dtypes
    original_model = AutoModelForCausalLM.from_pretrained(ckpt1)
    merged_model = AutoModelForCausalLM.from_pretrained(output)

    for (n1, p1), (n2, p2) in zip(
        original_model.named_parameters(), merged_model.named_parameters()
    ):
        assert p1.dtype == p2.dtype, f"Dtype mismatch for {n1}: {p1.dtype} vs {p2.dtype}"


def test_tokenizer_copied_from_first(tmp_path):
    """Test that tokenizer is copied from the first checkpoint."""
    ckpt1 = tmp_path / "checkpoint1"
    ckpt2 = tmp_path / "checkpoint2"
    output = tmp_path / "merged"

    # Create checkpoints with same vocab size but different seeds
    # (same model architecture but different weights)
    create_hf_checkpoint_with_seed(ckpt1, seed=42, vocab_size=100)
    create_hf_checkpoint_with_seed(ckpt2, seed=123, vocab_size=100)

    # Run merge
    result = run_merge_hf_cli([str(ckpt1), str(ckpt2)], str(output))

    # Should succeed even with different tokenizers
    assert result.exit_code == 0

    # Verify tokenizer files exist
    assert (output / "tokenizer_config.json").exists()
    # tokenizer.json should exist for PreTrainedTokenizerFast
    assert (output / "tokenizer.json").exists()

    # Verify it's from the first checkpoint by checking some config
    with open(ckpt1 / "tokenizer_config.json") as f:
        original_config = json.load(f)
    with open(output / "tokenizer_config.json") as f:
        merged_config = json.load(f)

    # Basic check that tokenizer was copied
    assert merged_config["tokenizer_class"] == original_config["tokenizer_class"]


def test_config_copied_from_first(tmp_path):
    """Test that config.json is based on the first checkpoint."""
    ckpt1 = tmp_path / "checkpoint1"
    ckpt2 = tmp_path / "checkpoint2"
    output = tmp_path / "merged"

    # Create checkpoints with same architecture
    create_hf_checkpoint_with_seed(ckpt1, seed=42, hidden_size=32)
    create_hf_checkpoint_with_seed(ckpt2, seed=123, hidden_size=32)

    # Modify second checkpoint's config slightly (non-architecture field)
    with open(ckpt2 / "config.json", "r") as f:
        config2 = json.load(f)
    config2["initializer_range"] = 0.05  # Different from default 0.02
    with open(ckpt2 / "config.json", "w") as f:
        json.dump(config2, f)

    # Run merge
    result = run_merge_hf_cli([str(ckpt1), str(ckpt2)], str(output))
    assert result.exit_code == 0

    # Verify config matches first checkpoint
    with open(ckpt1 / "config.json") as f:
        config1 = json.load(f)
    with open(output / "config.json") as f:
        merged_config = json.load(f)

    # Architecture fields should match
    assert config1["hidden_size"] == merged_config["hidden_size"]
    assert config1["num_hidden_layers"] == merged_config["num_hidden_layers"]
    # Non-architecture field should match first checkpoint
    assert merged_config["initializer_range"] == 0.02  # From first checkpoint


def test_device_parameter(tmp_path):
    """Test device parameter handling."""
    ckpt = tmp_path / "checkpoint"
    output = tmp_path / "merged"

    create_hf_checkpoint_with_seed(ckpt, seed=42, hidden_size=16)

    # Test with cpu (should always work)
    result = run_merge_hf_cli([str(ckpt)], str(output), device="cpu")
    assert result.exit_code == 0

    # Test with cuda if available
    if torch.cuda.is_available():
        output_cuda = tmp_path / "merged_cuda"
        result = run_merge_hf_cli([str(ckpt)], str(output_cuda), device="cuda")
        assert result.exit_code == 0


# ==================== CLI Interface Tests ====================


def test_revision_handling_single(tmp_path):
    """Test single revision applied to all models."""
    # This would need actual HF Hub models to test properly
    # For now, test with local paths (revisions ignored for local)
    ckpt1 = tmp_path / "checkpoint1"
    ckpt2 = tmp_path / "checkpoint2"
    output = tmp_path / "merged"

    create_hf_checkpoint_with_seed(ckpt1, seed=42)
    create_hf_checkpoint_with_seed(ckpt2, seed=123)

    # Run with single revision (ignored for local paths)
    result = run_merge_hf_cli([str(ckpt1), str(ckpt2)], str(output), revisions=["main"])
    assert result.exit_code == 0


def test_revision_count_mismatch(tmp_path):
    """Test that revision count mismatch raises error."""
    ckpt1 = tmp_path / "checkpoint1"
    ckpt2 = tmp_path / "checkpoint2"
    output = tmp_path / "merged"

    create_hf_checkpoint_with_seed(ckpt1, seed=42)
    create_hf_checkpoint_with_seed(ckpt2, seed=123)

    # Run with mismatched revision count (2 models, 3 revisions)
    with pytest.raises((Exception, SystemExit)) as exc_info:
        result = run_merge_hf_cli(
            [str(ckpt1), str(ckpt2)], str(output), revisions=["rev1", "rev2", "rev3"]
        )
        if result.exception:
            raise result.exception

    # Check the error message contains info about revision mismatch
    error_msg = str(exc_info.value).lower()
    assert ("must match" in error_msg or "must be" in error_msg) and "revision" in error_msg


def test_multiple_model_flags(tmp_path):
    """Test that CLI accepts multiple --model/-m flags."""
    ckpts = [tmp_path / f"checkpoint{i}" for i in range(4)]
    output = tmp_path / "merged"

    for i, ckpt in enumerate(ckpts):
        create_hf_checkpoint_with_seed(ckpt, seed=42 + i, hidden_size=16)

    # Test using multiple model flags
    runner = CliRunner()
    result = runner.invoke(
        merge_module.main,
        [
            "-m",
            str(ckpts[0]),
            "-m",
            str(ckpts[1]),
            "-m",
            str(ckpts[2]),
            "-m",
            str(ckpts[3]),
            "-o",
            str(output),
        ],
    )
    assert result.exit_code == 0
    assert output.exists()


def test_no_model_paths_error():
    """Test CLI error when no --model flags are provided."""
    runner = CliRunner()
    result = runner.invoke(merge_module.main, ["--output", "/tmp/output"])
    assert result.exit_code != 0


def test_cli_help():
    """Verify --help output works."""
    runner = CliRunner()
    result = runner.invoke(merge_module.main, ["--help"])
    assert result.exit_code == 0
    assert "merge" in result.output.lower() or "average" in result.output.lower()


def test_output_directory_creation(tmp_path):
    """Test that parent directories are created."""
    ckpt = tmp_path / "checkpoint"
    output = tmp_path / "deeply" / "nested" / "output" / "dir"

    create_hf_checkpoint_with_seed(ckpt, seed=42)

    # Output parent directories don't exist
    assert not output.parent.exists()

    # Run merge
    result = run_merge_hf_cli([str(ckpt)], str(output))

    # Should succeed and create directories
    assert result.exit_code == 0
    assert output.exists()
    assert (output / "config.json").exists()


# ==================== Compatibility & Error Tests ====================


def test_incompatible_vocab_sizes(tmp_path):
    """Test that merging checkpoints with different vocab sizes fails."""
    ckpt1 = tmp_path / "checkpoint1"
    ckpt2 = tmp_path / "checkpoint2"
    output = tmp_path / "merged"

    create_hf_checkpoint_with_seed(ckpt1, seed=42, vocab_size=100)
    create_hf_checkpoint_with_seed(ckpt2, seed=123, vocab_size=200)

    # Run merge - should fail due to incompatible shapes
    with pytest.raises((Exception, RuntimeError)) as exc_info:
        result = run_merge_hf_cli([str(ckpt1), str(ckpt2)], str(output))
        if result.exception:
            raise result.exception

    # Error should be about size mismatch
    assert "size" in str(exc_info.value).lower() or "shape" in str(exc_info.value).lower()


def test_incompatible_hidden_sizes(tmp_path):
    """Test that merging checkpoints with different hidden sizes fails."""
    ckpt1 = tmp_path / "checkpoint1"
    ckpt2 = tmp_path / "checkpoint2"
    output = tmp_path / "merged"

    create_hf_checkpoint_with_seed(ckpt1, seed=42, hidden_size=32)
    create_hf_checkpoint_with_seed(ckpt2, seed=123, hidden_size=64)

    # Run merge - should fail
    with pytest.raises((Exception, RuntimeError)) as exc_info:
        result = run_merge_hf_cli([str(ckpt1), str(ckpt2)], str(output))
        if result.exception:
            raise result.exception

    assert "size" in str(exc_info.value).lower() or "shape" in str(exc_info.value).lower()


def test_incompatible_layer_counts(tmp_path):
    """Test that merging checkpoints with different layer counts fails."""
    ckpt1 = tmp_path / "checkpoint1"
    ckpt2 = tmp_path / "checkpoint2"
    output = tmp_path / "merged"

    create_hf_checkpoint_with_seed(ckpt1, seed=42, num_hidden_layers=2)
    create_hf_checkpoint_with_seed(ckpt2, seed=123, num_hidden_layers=3)

    # Run merge - should fail
    with pytest.raises((Exception, RuntimeError)) as exc_info:
        result = run_merge_hf_cli([str(ckpt1), str(ckpt2)], str(output))
        if result.exception:
            raise result.exception

    # Should have a KeyError for missing layer (e.g., 'transformer.h.2.ln_1.weight')
    # or other mismatch error
    error_str = str(exc_info.value)
    # The error will mention a specific layer parameter that doesn't exist
    assert "transformer.h" in error_str or "KeyError" in str(type(exc_info.value))


def test_invalid_model_path():
    """Test handling of non-existent model path."""
    runner = CliRunner()
    result = runner.invoke(
        merge_module.main,
        [
            "--model",
            "/nonexistent/path",
            "--output",
            "/tmp/output",
        ],
    )
    assert result.exit_code != 0
    assert result.exception is not None


def test_output_overwrites_existing(tmp_path):
    """Test that existing output directory is used (exist_ok=True behavior)."""
    ckpt = tmp_path / "checkpoint"
    output = tmp_path / "output"

    create_hf_checkpoint_with_seed(ckpt, seed=42)

    # Create pre-existing output directory with a file
    output.mkdir()
    marker_file = output / "marker.txt"
    marker_file.write_text("existing content")

    # Run merge
    result = run_merge_hf_cli([str(ckpt)], str(output))

    # Should succeed
    assert result.exit_code == 0
    assert output.exists()
    assert (output / "config.json").exists()
    # Marker file is preserved (exist_ok=True doesn't delete existing files)
    assert marker_file.exists()


# ==================== Integration Tests ====================


def test_merged_model_inference(tmp_path):
    """Test that merged model can perform basic inference."""
    ckpt1 = tmp_path / "checkpoint1"
    ckpt2 = tmp_path / "checkpoint2"
    output = tmp_path / "merged"

    create_hf_checkpoint_with_seed(ckpt1, seed=42, vocab_size=50, hidden_size=32)
    create_hf_checkpoint_with_seed(ckpt2, seed=123, vocab_size=50, hidden_size=32)

    # Run merge
    result = run_merge_hf_cli([str(ckpt1), str(ckpt2)], str(output))
    assert result.exit_code == 0

    # Load model and try inference
    model = AutoModelForCausalLM.from_pretrained(output)

    # Create dummy input
    input_ids = torch.tensor([[0, 1, 2, 3]])

    # Forward pass should work
    with torch.no_grad():
        outputs = model(input_ids)
        assert outputs.logits is not None
        assert outputs.logits.shape == (1, 4, 50)  # batch_size, seq_len, vocab_size

    # Generate should work (even if output is nonsense with random weights)
    with torch.no_grad():
        generated = model.generate(input_ids, max_length=10, do_sample=False)
        assert generated is not None
        assert generated.shape[0] == 1  # batch size
        assert generated.shape[1] <= 10  # max length
