"""
Trains a small model for a few steps for testing.
Adapted from: src/examples/llm/train.py

Run CPU test:
    pytest -v src/integration_tests/test_train_small_model.py::test_train_small_model_cpu

Run GPU test:
    pytest -v -m gpu src/integration_tests/test_train_small_model.py::test_train_small_model_gpu
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Set

import pytest
import torch

from olmo_core.config import Config, DType
from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, TokenizerConfig
from olmo_core.data.numpy_dataset import NumpyDatasetConfig
from olmo_core.distributed.checkpoint import load_state_dict
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig
from olmo_core.testing import run_distributed_test
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import CheckpointerCallback, ModelMergeCallback
from olmo_core.train.callbacks.callback import Callback
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DATA_PATH = "http://olmo-data.org/examples/c4-en/gpt2/c4-train.00000-00099.npy"


@dataclass
class WeightCaptureCallback(Callback):
    """Test callback that captures model weights at specified steps."""

    capture_steps: Set[int] = field(default_factory=set)
    captured_weights: Dict[int, Dict[str, torch.Tensor]] = field(default_factory=dict)

    def post_train_batch(self):
        if self.step in self.capture_steps:
            self.captured_weights[self.step] = {
                k: v.clone().cpu().float()
                for k, v in self.trainer.train_module.model.state_dict().items()
            }


@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    trainer: TrainerConfig
    train_module: TransformerTrainModuleConfig
    init_seed: int = 42


def train(config: ExperimentConfig):
    seed_all(config.init_seed)

    # Build components
    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)

    trainer.fit()

    # Get references to callbacks and config values
    weight_capture = trainer.callbacks["weight_capture"]
    model_merger = trainer.callbacks["model_merger"]
    max_steps = trainer.max_steps
    assert max_steps is not None
    merge_last_n_steps = model_merger.merge_last_n_steps
    capture_steps = set(range(max_steps - merge_last_n_steps + 1, max_steps + 1))  # steps 3,4,5

    # Check merged checkpoint was created
    merged_path = Path(config.trainer.save_folder) / f"step{max_steps}-merged"
    assert merged_path.exists(), f"Merged checkpoint not found at {merged_path}"

    model_and_optim_path = merged_path / "model_and_optim"
    assert model_and_optim_path.exists(), "model_and_optim directory not found"

    # Verify the expected steps were captured
    assert (
        weight_capture.captured_weights.keys() == capture_steps
    ), f"Expected to capture steps {capture_steps}, got {weight_capture.captured_weights.keys()}"

    # Load the merged checkpoint
    merged_state: Dict[str, Dict[str, torch.Tensor]] = {"model": {}, "optim": {}}
    for key in weight_capture.captured_weights[max_steps].keys():
        merged_state["model"][key] = torch.empty_like(
            weight_capture.captured_weights[max_steps][key]
        )
    load_state_dict(str(model_and_optim_path), merged_state)

    # Verify merged weights match expected average
    captured_list = list(weight_capture.captured_weights.values())
    for key in captured_list[0].keys():
        stacked = torch.stack([w[key] for w in captured_list])
        expected = stacked.mean(dim=0)
        actual = merged_state["model"][key].float()

        if not torch.allclose(actual, expected, rtol=1e-4, atol=1e-6):
            max_diff = (actual - expected).abs().max().item()
            raise AssertionError(
                f"Merged weights for '{key}' do not match expected average. "
                f"Max difference: {max_diff}"
            )

    log.info("Averaging verification passed")


def build_config(
    save_folder: Path, work_dir: Path, dp_config: Optional[TransformerDataParallelConfig] = None
) -> ExperimentConfig:
    max_steps = 5
    merge_last_n_steps = 3

    tokenizer_config = TokenizerConfig.gpt2()

    model_config = TransformerConfig.olmo3_30M(
        vocab_size=tokenizer_config.padded_vocab_size(),
    )

    dataset_config = NumpyFSLDatasetConfig(
        paths=[DATA_PATH],
        sequence_length=64,
        tokenizer=tokenizer_config,
        work_dir=str(work_dir),
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=64 * 4,
        seed=0,
        num_workers=0,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=64 * 2,
        max_sequence_length=64,
        optim=AdamWConfig(lr=1e-3),
        compile_model=False,
        dp_config=dp_config,
    )

    # Capture weights at steps that will be averaged (steps 3, 4, 5)
    capture_steps = set(range(max_steps - merge_last_n_steps + 1, max_steps + 1))
    weight_capture = WeightCaptureCallback(capture_steps=capture_steps)

    trainer_config = (
        TrainerConfig(
            save_folder=str(save_folder),
            save_overwrite=True,
            metrics_collect_interval=1,
            cancel_check_interval=1,
            max_duration=Duration.steps(max_steps),
        )
        .with_callback("checkpointer", CheckpointerCallback(save_interval=1000))
        .with_callback(
            "model_merger",
            ModelMergeCallback(
                merge_step=max_steps,
                merge_last_n_steps=merge_last_n_steps,
                enabled=True,
            ),
        )
        .with_callback("weight_capture", weight_capture)
    )

    return ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
    )


def test_train_small_model_cpu(tmp_path):
    save_folder = tmp_path / "checkpoints"
    work_dir = tmp_path / "work_dir"
    save_folder.mkdir()
    work_dir.mkdir()

    config = build_config(save_folder, work_dir)
    train(config)


def test_ephemeral_blocked_during_merge_window(tmp_path):
    """Verify that ephemeral checkpoints are blocked during the merge window
    but NOT at the merge step itself (off-by-one check)."""
    save_folder = tmp_path / "checkpoints"
    work_dir = tmp_path / "work_dir"
    save_folder.mkdir()
    work_dir.mkdir()

    max_steps = 10
    merge_step = 8
    merge_last_n_steps = 3  # merge window: steps 6, 7, 8
    ephemeral_save_interval = 2  # would normally save at steps 2, 4, 6, 8, 10

    tokenizer_config = TokenizerConfig.gpt2()

    config = ExperimentConfig(
        model=TransformerConfig.olmo3_30M(vocab_size=tokenizer_config.padded_vocab_size()),
        dataset=NumpyFSLDatasetConfig(
            paths=[DATA_PATH],
            sequence_length=64,
            tokenizer=tokenizer_config,
            work_dir=str(work_dir),
        ),
        data_loader=NumpyDataLoaderConfig(global_batch_size=64 * 4, seed=0, num_workers=0),
        train_module=TransformerTrainModuleConfig(
            rank_microbatch_size=64 * 2,
            max_sequence_length=64,
            optim=AdamWConfig(lr=1e-3),
            compile_model=False,
        ),
        trainer=TrainerConfig(
            save_folder=str(save_folder),
            save_overwrite=True,
            metrics_collect_interval=1,
            cancel_check_interval=1,
            max_duration=Duration.steps(max_steps),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=ephemeral_save_interval,
            ),
        )
        .with_callback(
            "model_merger",
            ModelMergeCallback(
                merge_step=merge_step,
                merge_last_n_steps=merge_last_n_steps,
                enabled=True,
            ),
        ),
    )

    seed_all(42)
    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)
    trainer.fit()

    # Check which checkpoints were saved
    checkpoint_steps = set()
    for path in save_folder.iterdir():
        name = path.name
        if name.startswith("step") and not name.endswith("-merged"):
            step = int(name.replace("step", ""))
            checkpoint_steps.add(step)

    # Step 6 is inside the merge window but NOT the merge step — should be blocked.
    # (Earlier ephemeral checkpoints at steps 2, 4 may have been cleaned up by
    # the checkpointer, so we don't assert on those.)
    assert 6 not in checkpoint_steps, (
        f"Ephemeral checkpoint at step 6 should have been blocked during merge window, "
        f"got checkpoints at: {checkpoint_steps}"
    )

    # Step 8 IS the merge step — ephemeral should NOT be blocked after merge completes
    # (off-by-one check: blocking ends at the merge step, not after it)
    assert 8 in checkpoint_steps, (
        f"Ephemeral checkpoint at step 8 (merge step) should NOT have been blocked, "
        f"got checkpoints at: {checkpoint_steps}"
    )

    # Step 10 is after the window — should exist
    assert 10 in checkpoint_steps, (
        f"Ephemeral checkpoint at step 10 (after window) should exist, "
        f"got checkpoints at: {checkpoint_steps}"
    )

    # Merged checkpoint should exist
    merged_path = save_folder / f"step{merge_step}-merged"
    assert merged_path.exists(), f"Merged checkpoint not found at {merged_path}"

    log.info(f"Ephemeral blocking test passed. Checkpoints at steps: {checkpoint_steps}")


def _run_train_gpu(tmp_path: Path):
    save_folder = tmp_path / "checkpoints"
    work_dir = tmp_path / "work_dir"
    save_folder.mkdir(exist_ok=True)
    work_dir.mkdir(exist_ok=True)

    config = build_config(
        save_folder,
        work_dir,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
    )
    train(config)


@pytest.mark.gpu
def test_train_small_model_gpu(tmp_path):
    run_distributed_test(
        _run_train_gpu,
        backend="nccl",
        start_method="spawn",
        func_args=(tmp_path,),
    )
