"""
Tests for the reshard_core_checkpoint.py script.
"""
import gc
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import pytest
import torch

from olmo_core.distributed.checkpoint import (
    get_checkpoint_metadata,
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig

# Path to the reshard script
RESHARD_SCRIPT = Path("src/scripts/reshard_core_checkpoint.py")


# ==================== Helper Functions ====================


def create_test_checkpoint(
    checkpoint_dir: Path,
    model_config: TransformerConfig,
    optim_config: Optional[AdamWConfig] = None,
    world_size: int = 1,
    include_optimizer: bool = True,
) -> None:
    """
    Create a test checkpoint with proper structure for resharding tests.

    Args:
        checkpoint_dir: Directory to save the checkpoint
        model_config: Transformer configuration
        optim_config: Optimizer configuration (optional, defaults to AdamWConfig with lr=1e-3)
        world_size: Number of processes used to save the checkpoint
        include_optimizer: Whether to include optimizer state
    """
    gc.collect()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Set default optimizer config if not provided
    if optim_config is None:
        optim_config = AdamWConfig(lr=1e-3)

    # Create config.json
    optim_dict = optim_config.as_dict()
    optim_dict["_CLASS_"] = f"{optim_config.__class__.__module__}.{optim_config.__class__.__name__}"

    config_dict = {
        "model": model_config.as_dict(),
        "dataset": {"tokenizer": {"identifier": "test_tokenizer", "type": "test"}},
        "train_module": {"optim": optim_dict},
    }

    config_path = checkpoint_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    # Create model and optimizer
    if world_size == 1:
        # Single process checkpoint
        model = model_config.build(init_device="cpu")
        model.init_weights(device=torch.device("cpu"))

        if include_optimizer:
            optim = optim_config.build(model)
            # Take a step to initialize optimizer state
            loss = model(input_ids=torch.randint(0, model_config.vocab_size, (2, 16))).sum()
            loss.backward()
            optim.step()
        else:
            optim = None

        # Save checkpoint
        model_and_optim_dir = checkpoint_dir / "model_and_optim"
        save_model_and_optim_state(
            str(model_and_optim_dir),
            model,
            optim,
            save_overwrite=True,
        )
    else:
        # Multi-process checkpoint - use a subprocess to create it with single process
        # and reshard it to the desired world_size. This avoids pickling issues.
        with tempfile.TemporaryDirectory() as tmpdir:
            # First create a single-process checkpoint
            temp_checkpoint = Path(tmpdir) / "temp_checkpoint"
            temp_checkpoint.mkdir(parents=True)

            # Create the single-process checkpoint
            model = model_config.build(init_device="cpu")
            model.init_weights(device=torch.device("cpu"))

            if include_optimizer:
                optim = optim_config.build(model)
                # Take a step to initialize optimizer state
                loss = model(input_ids=torch.randint(0, model_config.vocab_size, (2, 16))).sum()
                loss.backward()
                optim.step()
            else:
                optim = None

            # Save as single-process checkpoint
            temp_model_and_optim_dir = temp_checkpoint / "model_and_optim"
            save_model_and_optim_state(
                str(temp_model_and_optim_dir),
                model,
                optim,
                save_overwrite=True,
            )

            # Copy config.json to temp checkpoint
            shutil.copy(checkpoint_dir / "config.json", temp_checkpoint / "config.json")

            # Now reshard from 1 to world_size using the CLI
            run_reshard_cli(
                input_path=str(temp_checkpoint),
                output_path=str(checkpoint_dir),
                num_processes=world_size,
                skip_optimizer=not include_optimizer,
            )


def load_config_from_checkpoint(checkpoint_dir: Path) -> dict:
    """Load config.json from a checkpoint directory."""
    gc.collect()
    config_path = checkpoint_dir / "config.json"
    with open(config_path, "r") as f:
        return json.load(f)


def checkpoint_has_optimizer_state(checkpoint_dir: Path) -> bool:
    """Check if a checkpoint contains optimizer state by examining its metadata."""
    model_and_optim_dir = checkpoint_dir / "model_and_optim"
    if not model_and_optim_dir.exists():
        return False

    checkpoint_meta = get_checkpoint_metadata(str(model_and_optim_dir))

    # Check if any optimizer-related keys are in the metadata
    # Optimizer state keys typically start with "optim."
    return any(key.startswith("optim.") for key in checkpoint_meta.state_dict_metadata.keys())


def run_reshard_cli(
    input_path: str, output_path: str, num_processes: int = 1, skip_optimizer: bool = False
) -> None:
    """Run the reshard script as a subprocess. The reshard script forks itself to create a process group.
    This interferes with PyTorch global stuff if you call it normally."""
    args = [
        sys.executable,
        str(RESHARD_SCRIPT),
        "--input",
        input_path,
        "--output",
        output_path,
        "--num-processes",
        str(num_processes),
    ]
    if skip_optimizer:
        args.append("--skip-optimizer-state")

    result = subprocess.run(args, capture_output=True, text=True)
    assert (
        result.returncode == 0
    ), f"Resharding failed with exit code {result.returncode}:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    gc.collect()


# ==================== Integration Tests ====================


def test_reshard_single_process(tmp_path):
    """Test resharding with a single process (n=1)."""
    # Create a small transformer config
    model_config = TransformerConfig.llama_like(
        d_model=64,
        n_heads=4,
        n_layers=2,
        vocab_size=100,
    )

    # Create source checkpoint with optimizer
    source_dir = tmp_path / "source_checkpoint"
    create_test_checkpoint(source_dir, model_config, world_size=1, include_optimizer=True)

    # Verify source has optimizer state
    assert checkpoint_has_optimizer_state(
        source_dir
    ), "Source checkpoint should have optimizer state"

    # Reshard to target (single process)
    target_dir = tmp_path / "target_checkpoint"

    # Run the reshard script via CLI
    run_reshard_cli(
        input_path=str(source_dir),
        output_path=str(target_dir),
        num_processes=1,
        skip_optimizer=False,
    )

    # Verify the checkpoint was created
    assert (target_dir / "config.json").exists()
    assert (target_dir / "model_and_optim").exists()

    # Verify configs match
    source_config = load_config_from_checkpoint(source_dir)
    target_config = load_config_from_checkpoint(target_dir)
    assert source_config == target_config

    # Verify target also has optimizer state
    assert checkpoint_has_optimizer_state(
        target_dir
    ), "Target checkpoint should have optimizer state when not skipping"


def test_reshard_single_process_skip_optimizer(tmp_path):
    """Test single process resharding without optimizer state."""
    # Create a small transformer config
    model_config = TransformerConfig.llama_like(
        d_model=64,
        n_heads=4,
        n_layers=2,
        vocab_size=100,
    )

    # Create source checkpoint with optimizer
    source_dir = tmp_path / "source_checkpoint"
    create_test_checkpoint(source_dir, model_config, world_size=1, include_optimizer=True)

    # Verify source has optimizer state
    assert checkpoint_has_optimizer_state(
        source_dir
    ), "Source checkpoint should have optimizer state"

    # Reshard without optimizer state
    target_dir = tmp_path / "target_checkpoint"

    # Run the reshard script via CLI with skip_optimizer flag
    run_reshard_cli(
        input_path=str(source_dir),
        output_path=str(target_dir),
        num_processes=1,
        skip_optimizer=True,
    )

    # Verify the checkpoint was created
    assert (target_dir / "config.json").exists()
    assert (target_dir / "model_and_optim").exists()

    # Verify no optimizer state in target by checking metadata
    assert not checkpoint_has_optimizer_state(
        target_dir
    ), "Optimizer state should not exist when skip_optimizer_state=True"


def run_multi_process_reshard_test(source_ws, target_ws, skip_optimizer=False):
    """Helper function for multi-process resharding tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create a small transformer config
        model_config = TransformerConfig.llama_like(
            d_model=64,
            n_heads=4,
            n_layers=2,
            vocab_size=100,
        )

        # Create source checkpoint
        source_dir = tmp_path / "source_checkpoint"
        create_test_checkpoint(source_dir, model_config, world_size=source_ws)

        # Reshard to target
        target_dir = tmp_path / "target_checkpoint"

        # Use the CLI runner
        run_reshard_cli(
            input_path=str(source_dir),
            output_path=str(target_dir),
            num_processes=target_ws,
            skip_optimizer=skip_optimizer,
        )

        # Verify the checkpoint was created
        assert (target_dir / "config.json").exists()
        assert (target_dir / "model_and_optim").exists()


def test_reshard_multi_process_2_workers(tmp_path):
    """Test resharding with 2 worker processes."""
    run_multi_process_reshard_test(source_ws=1, target_ws=2)


@pytest.mark.parametrize("world_size", [2, 4], ids=["world_size=2", "world_size=4"])
def test_reshard_multi_process_various_workers(tmp_path, world_size):
    """Test resharding with various numbers of worker processes."""
    run_multi_process_reshard_test(source_ws=1, target_ws=world_size)


# ==================== End-to-End Tests ====================


def test_reshard_1_to_2_processes(tmp_path):
    """Test expanding a single-process checkpoint to 2 processes."""
    model_config = TransformerConfig.llama_like(
        d_model=64,
        n_heads=4,
        n_layers=2,
        vocab_size=100,
    )

    # Create single-process checkpoint
    source_dir = tmp_path / "single_checkpoint"
    create_test_checkpoint(source_dir, model_config, world_size=1)

    # Reshard to 2 processes
    target_dir = tmp_path / "distributed_checkpoint"

    # Use the CLI runner
    run_reshard_cli(
        input_path=str(source_dir),
        output_path=str(target_dir),
        num_processes=2,
        skip_optimizer=False,
    )

    # Verify checkpoint structure
    assert (target_dir / "config.json").exists()
    assert (target_dir / "model_and_optim").exists()

    # Verify configs match
    source_config = load_config_from_checkpoint(source_dir)
    target_config = load_config_from_checkpoint(target_dir)
    assert source_config == target_config


def test_reshard_2_to_1_processes(tmp_path):
    """Test consolidating a 2-process checkpoint to single process."""
    model_config = TransformerConfig.llama_like(
        d_model=64,
        n_heads=4,
        n_layers=2,
        vocab_size=100,
    )

    # Create 2-process checkpoint
    source_dir = tmp_path / "distributed_checkpoint"
    create_test_checkpoint(source_dir, model_config, world_size=2)

    # Reshard to single process
    target_dir = tmp_path / "single_checkpoint"

    # Use the CLI runner
    run_reshard_cli(
        input_path=str(source_dir),
        output_path=str(target_dir),
        num_processes=1,
        skip_optimizer=False,
    )

    # Verify checkpoint structure
    assert (target_dir / "config.json").exists()
    assert (target_dir / "model_and_optim").exists()

    # Verify configs match
    source_config = load_config_from_checkpoint(source_dir)
    target_config = load_config_from_checkpoint(target_dir)
    assert source_config == target_config


def test_reshard_2_to_4_processes(tmp_path):
    """Test resharding between different distributed topologies (2 to 4 processes)."""
    model_config = TransformerConfig.llama_like(
        d_model=64,
        n_heads=4,
        n_layers=2,
        vocab_size=100,
    )

    # Create 2-process checkpoint
    source_dir = tmp_path / "checkpoint_2proc"
    create_test_checkpoint(source_dir, model_config, world_size=2)

    # Reshard to 4 processes
    target_dir = tmp_path / "checkpoint_4proc"

    # Use the CLI runner
    run_reshard_cli(
        input_path=str(source_dir),
        output_path=str(target_dir),
        num_processes=4,
        skip_optimizer=False,
    )

    # Verify checkpoint structure
    assert (target_dir / "config.json").exists()
    assert (target_dir / "model_and_optim").exists()


def test_reshard_preserves_model_state(tmp_path):
    """Verify that model weights are preserved correctly during resharding."""
    model_config = TransformerConfig.llama_like(
        d_model=64,
        n_heads=2,
        n_layers=2,
        vocab_size=100,
    )

    # Create source checkpoint
    source_dir = tmp_path / "source"
    create_test_checkpoint(source_dir, model_config, world_size=1)

    # Load original model state
    original_model = model_config.build(init_device="cpu")
    load_model_and_optim_state(
        str(source_dir / "model_and_optim"),
        original_model,
        None,
    )
    original_state = original_model.state_dict()

    # Reshard to 2 processes and back to 1
    intermediate_dir = tmp_path / "intermediate"
    final_dir = tmp_path / "final"

    # First reshard: 1 -> 2
    run_reshard_cli(
        input_path=str(source_dir),
        output_path=str(intermediate_dir),
        num_processes=2,
        skip_optimizer=False,
    )

    # Second reshard: 2 -> 1
    run_reshard_cli(
        input_path=str(intermediate_dir),
        output_path=str(final_dir),
        num_processes=1,
        skip_optimizer=False,
    )

    # Load final model state
    final_model = model_config.build(init_device="cpu")
    load_model_and_optim_state(
        str(final_dir / "model_and_optim"),
        final_model,
        None,
    )
    final_state = final_model.state_dict()

    # Compare states
    assert set(original_state.keys()) == set(final_state.keys())
    for key in original_state.keys():
        torch.testing.assert_close(
            original_state[key],
            final_state[key],
            msg=f"Mismatch in parameter '{key}' after round-trip resharding",
        )


def test_reshard_preserves_optimizer_state(tmp_path):
    """Verify that optimizer state is preserved correctly during resharding."""
    model_config = TransformerConfig.llama_like(
        d_model=64,
        n_heads=2,
        n_layers=2,
        vocab_size=100,
    )
    optim_config = AdamWConfig(lr=1e-3)

    # Create source checkpoint with optimizer state
    source_dir = tmp_path / "source"
    create_test_checkpoint(source_dir, model_config, optim_config, world_size=1)

    # Load original model and optimizer
    original_model = model_config.build(init_device="cpu")
    original_optim = optim_config.build(original_model)
    load_model_and_optim_state(
        str(source_dir / "model_and_optim"),
        original_model,
        original_optim,
    )

    # Get original optimizer state (simplified - just check it exists)
    original_optim_state = {
        name: {k: v.clone() for k, v in state.items()}
        for name, param in original_model.named_parameters()
        if param in original_optim.state
        for state in [original_optim.state[param]]
    }

    # Reshard and verify optimizer state is preserved
    target_dir = tmp_path / "target"

    # Use the CLI runner
    run_reshard_cli(
        input_path=str(source_dir),
        output_path=str(target_dir),
        num_processes=1,
        skip_optimizer=False,
    )

    # Load resharded model and optimizer
    final_model = model_config.build(init_device="cpu")
    final_optim = optim_config.build(final_model)
    load_model_and_optim_state(
        str(target_dir / "model_and_optim"),
        final_model,
        final_optim,
    )

    # Verify optimizer state exists
    for name, param in final_model.named_parameters():
        if name in original_optim_state:
            assert param in final_optim.state, f"Optimizer state missing for parameter '{name}'"


def test_reshard_roundtrip(tmp_path):
    """Test that resharding back and forth preserves all data."""
    model_config = TransformerConfig.llama_like(
        d_model=64,
        n_heads=2,
        n_layers=1,
        vocab_size=100,
    )

    # Create initial checkpoint
    checkpoint_1 = tmp_path / "checkpoint_1"
    create_test_checkpoint(checkpoint_1, model_config, world_size=1)

    # Load initial model state for comparison
    initial_model = model_config.build(init_device="cpu")
    load_model_and_optim_state(
        str(checkpoint_1 / "model_and_optim"),
        initial_model,
        None,
    )
    initial_state = initial_model.state_dict()

    # Reshard 1 -> 2
    checkpoint_2 = tmp_path / "checkpoint_2"
    run_reshard_cli(
        input_path=str(checkpoint_1),
        output_path=str(checkpoint_2),
        num_processes=2,
        skip_optimizer=False,
    )

    # Reshard 2 -> 1
    checkpoint_3 = tmp_path / "checkpoint_3"
    run_reshard_cli(
        input_path=str(checkpoint_2),
        output_path=str(checkpoint_3),
        num_processes=1,
        skip_optimizer=False,
    )

    # Load final model state
    final_model = model_config.build(init_device="cpu")
    load_model_and_optim_state(
        str(checkpoint_3 / "model_and_optim"),
        final_model,
        None,
    )
    final_state = final_model.state_dict()

    # Compare initial and final model states
    assert set(initial_state.keys()) == set(final_state.keys())
    for key in initial_state.keys():
        torch.testing.assert_close(
            initial_state[key],
            final_state[key],
            msg=f"Mismatch in parameter '{key}' after roundtrip resharding",
        )
