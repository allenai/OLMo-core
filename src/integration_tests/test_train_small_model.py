"""
Trains a small model for a few steps on CPU to test ModelMergeCallback.

Run with pytest:
    pytest -v src/integration_tests/test_train_small_model.py
"""

import logging
from pathlib import Path

import pytest

from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    LMEvaluatorCallbackConfig,
    ModelMergeCallback,
)
from olmo_core.train.train_module import TransformerTrainModuleConfig
from olmo_core.utils import seed_all

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DATA_PATH = "http://olmo-data.org/examples/c4-en/gpt2/c4-train.00000-00099.npy"


def train(save_folder: Path, work_dir: Path):
    """Train a small model for a few steps."""
    seed_all(42)

    max_steps = 5

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
    )

    eval_dataset_config = NumpyPaddedFSLDatasetConfig(
        paths=[DATA_PATH],
        sequence_length=64,
        tokenizer=tokenizer_config,
        work_dir=str(work_dir),
    )

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
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=eval_dataset_config,
                eval_interval=1000,  # don't eval during training, only at merge
                eval_duration=Duration.steps(2),  # keep it fast
            ),
        )
        .with_callback(
            "model_merger",
            ModelMergeCallback(
                merge_step=max_steps,
                merge_last_n_steps=3,
                validate=True,
            ),
        )
    )

    model = model_config.build(init_device="meta")
    train_module = train_module_config.build(model)
    dataset = dataset_config.build()
    data_loader = data_loader_config.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = trainer_config.build(train_module, data_loader)

    trainer.fit()

    # Check merged checkpoint was created
    merged_path = save_folder / f"step{max_steps}-merged"
    assert merged_path.exists(), f"Merged checkpoint not found at {merged_path}"


def test_train_small_model(tmp_path):
    save_folder = tmp_path / "checkpoints"
    work_dir = tmp_path / "work_dir"
    save_folder.mkdir()
    work_dir.mkdir()
    train(save_folder, work_dir)
