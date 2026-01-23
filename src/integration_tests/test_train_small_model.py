"""
Trains a small model for a few steps to test ModelMergeCallback.

Run CPU test:
    pytest -v src/integration_tests/test_train_small_model.py::test_train_small_model

Run GPU test:
    pytest -v -m gpu src/integration_tests/test_train_small_model.py::test_train_small_model_fsdp
"""

import logging
import os
from pathlib import Path
from typing import Optional

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
from olmo_core.testing import run_distributed_test
from olmo_core.train import Duration, TrainerConfig
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

# Try local paths first (Weka/NFS), fall back to HTTP
DEFAULT_DATA_ROOT = "http://olmo-data.org/examples/c4-en/gpt2"
for dir in (
    "/weka/oe-training-default/ai2-llm/examples/c4-en/gpt2/",
    "/net/nfs/allennlp/llm-data/c4/en/",
):
    if os.path.exists(dir):
        DEFAULT_DATA_ROOT = dir
        break

DATA_PATH = f"{DEFAULT_DATA_ROOT}/c4-train.00000-00099.npy"
EVAL_DATA_PATH = f"{DEFAULT_DATA_ROOT}/c4-validation.00000-00008.npy"


def train(
    save_folder: Path,
    work_dir: Path,
    dp_config: Optional[TransformerDataParallelConfig] = None,
):
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
        dp_config=dp_config,
    )

    eval_dataset_config = NumpyPaddedFSLDatasetConfig(
        paths=[EVAL_DATA_PATH],
        metadata=[{"label": "c4-validation"}],
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
        # Checkpointer needed for ModelMergeCallback's process_group; high interval to skip regular saves
        .with_callback("checkpointer", CheckpointerCallback(save_interval=1000))
        .with_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=eval_dataset_config,
                eval_interval=1000,  # Don't eval during training, only at merge
                eval_duration=Duration.steps(2),  # Keep it fast
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
    """CPU test for basic ModelMergeCallback logic."""
    save_folder = tmp_path / "checkpoints"
    work_dir = tmp_path / "work_dir"
    save_folder.mkdir()
    work_dir.mkdir()
    train(save_folder, work_dir)


def _run_train_fsdp(tmp_path: Path):
    """Inner function that runs inside the distributed environment."""
    save_folder = tmp_path / "checkpoints"
    work_dir = tmp_path / "work_dir"
    save_folder.mkdir(exist_ok=True)
    work_dir.mkdir(exist_ok=True)
    train(
        save_folder,
        work_dir,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
    )


@pytest.mark.gpu
def test_train_small_model_fsdp(tmp_path):
    """GPU test for ModelMergeCallback with FSDP sharded checkpoints."""
    run_distributed_test(
        _run_train_fsdp,
        backend="nccl",
        start_method="spawn",
        func_args=(tmp_path,),
    )
