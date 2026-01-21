"""
Trains a small model for a few steps.
Adapted from: https://github.com/allenai/OLMo-core/blob/main/src/examples/llm/train.py

Run with torchrun:
    torchrun --nproc-per-node=1 src/integration_tests/train_small_model.py

Run with pytest:
    pytest -v -m gpu src/integration_tests/train_small_model.py
"""

import logging
import tempfile
from pathlib import Path

import pytest

from olmo_core.config import DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    LMEvaluatorCallbackConfig,
    ModelMergeCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
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
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
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


def main():
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_folder = Path(tmp_dir) / "checkpoints"
        work_dir = Path(tmp_dir) / "work_dir"
        save_folder.mkdir(parents=True)
        work_dir.mkdir(parents=True)
        train(save_folder, work_dir)


@pytest.mark.gpu
def test_train_small_model(tmp_path):
    save_folder = tmp_path / "checkpoints"
    work_dir = tmp_path / "work_dir"
    save_folder.mkdir()
    work_dir.mkdir()
    train(save_folder, work_dir)


if __name__ == "__main__":
    prepare_training_environment()
    try:
        main()
    finally:
        teardown_training_environment()
