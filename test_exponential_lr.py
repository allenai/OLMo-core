"""
Test exponential learning rate scheduler for LR range testing.

This script does a short 10k step run where the learning rate exponentially increases
from 1e-9 to 10, allowing you to find the optimal learning rate by watching loss/metrics.

This follows the same pattern as src/examples/llm/train.py.

Launch with:
    python test_exponential_lr.py exponential_lr_test_370M

Or with config overrides:
    python test_exponential_lr.py exponential_lr_test_370M trainer.max_duration=5000
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, cast

import rich

from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.data.numpy_dataset import NumpyDatasetConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, ExponentialScheduler, OptimGroupOverride
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    GPUMemoryMonitorCallback,
    LMEvaluatorCallbackConfig,
    ProfilerCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

log = logging.getLogger(__name__)

# Use the same data paths as examples/llm/train.py
DEFAULT_DATA_ROOT = "http://olmo-data.org/examples/c4-en/gpt2"
for dir in (
    "/net/nfs/allennlp/llm-data/c4/en/",
    "/weka/oe-training-default/ai2-llm/examples/c4-en/gpt2/",
):
    if os.path.exists(dir):
        DEFAULT_DATA_ROOT = dir
        break
DATA_ROOT = os.environ.get("OLMO_DATA_ROOT", DEFAULT_DATA_ROOT).rstrip("/")
DATA_PATHS = [
    f"{DATA_ROOT}/c4-train.00000-00099.npy",
]
EVAL_DATA_PATHS = [f"{DATA_ROOT}/c4-validation.00000-00008.npy"]


@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    """Model config."""
    dataset: NumpyDatasetConfig
    """Dataset config."""
    data_loader: NumpyDataLoaderConfig
    """Data loader config."""
    trainer: TrainerConfig
    """Trainer config."""
    train_module: TransformerTrainModuleConfig
    """Train module config. Contains settings for optimizer."""
    init_seed: int = 12536
    """Random seed to initialize model weights."""
    load_path: Optional[str] = None
    """Path to load checkpoint from if no checkpoint is found in the save folder.
    Mainly used when you want to fine-tune from a pretrained model."""
    load_trainer_state: bool = False
    """Whether to load the trainer state (including data loader state) when loading from `load_path`.
    This only makes sense when trainer state is available in the checkpoint and you're resuming
    on the same dataset."""


def train(config: ExperimentConfig):
    if get_rank() == 0:
        rich.print(config)

    # Set RNG states on all devices.
    seed_all(config.init_seed)

    # Build components.
    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)

    # Save config to W&B and each checkpoint dir.
    config_dict = config.as_config_dict()
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    # If we have a load path set and there is no checkpoint in the save folder, load the
    # checkpoint from the load path.
    if not trainer.no_checkpoints and not trainer.maybe_load_checkpoint() and config.load_path:
        log.info(
            f"Loading checkpoint from {config.load_path} since no checkpoints were found in the save folder..."
        )
        trainer.load_checkpoint(config.load_path, load_trainer_state=config.load_trainer_state)

    # Train.
    trainer.fit()


def build_config(opts, overrides: List[str]) -> ExperimentConfig:
    save_folder = opts.save_folder
    if not save_folder:
        save_folder = f"/tmp/{opts.run_name}"

    work_dir = opts.work_dir
    if not work_dir:
        work_dir = "/tmp/dataset-cache"

    tokenizer_config = TokenizerConfig.gpt2()

    # Model config - use olmo2_370M or make it smaller
    try:
        factory = getattr(TransformerConfig, opts.model_factory)
    except AttributeError:
        raise ValueError(f"Unknown model factory: {opts.model_factory}")
    model_config = factory(
        vocab_size=tokenizer_config.padded_vocab_size(),
    )

    log.info(f"Using data root: {DATA_ROOT}")
    dataset_config = NumpyFSLDatasetConfig(
        paths=DATA_PATHS,
        sequence_length=opts.sequence_length,
        tokenizer=tokenizer_config,
        work_dir=work_dir,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=256 * 1024,  # NOTE: this is specified in tokens, not instances
        seed=0,
        num_workers=4,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=16 * 1024,  # NOTE: this is specified in tokens, not instances
        max_sequence_length=opts.sequence_length,
        optim=AdamWConfig(
            lr=10.0,  # Max LR for LR range test - exponentially increase from 1e-9 to 10
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        compile_model=False,  # Disable compilation for easier debugging
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
        ),
        max_grad_norm=1.0,
        # Use ExponentialScheduler for LR range test: 1e-9 -> 10 over 10k steps
        scheduler=ExponentialScheduler(lr_min=1e-9),
    )

    trainer_config = (
        TrainerConfig(
            save_folder=save_folder,
            save_overwrite=True,
            metrics_collect_interval=5,
            cancel_check_interval=5,
            max_duration=Duration.steps(opts.max_steps),  # Max steps for testing
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                enabled=False,  # Disable checkpointing to avoid overwriting anything
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=opts.run_name,
                cancel_check_interval=10,
                enabled=False,  # Disabled
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name="exponential_lr_scheduler_370M",
                entity="ai2-llm",  # AI2 LLM team entity
                project="bailey-testing",  # Test project
                group="exponential_lr_scheduler_370M",  # Group name as requested
                cancel_check_interval=10,
                enabled=True,  # Enable W&B logging
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("profiler", ProfilerCallback(enabled=False))
        .with_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=NumpyPaddedFSLDatasetConfig(
                    paths=EVAL_DATA_PATHS,
                    metadata=[{"label": "c4-validation"}],
                    sequence_length=opts.sequence_length,
                    tokenizer=tokenizer_config,
                    work_dir=work_dir,
                ),
                eval_interval=250,
                eval_duration=Duration.steps(50),
            ),
        )
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=["hellaswag"],
                tokenizer=tokenizer_config,
                eval_interval=250,
            ),
        )
    )

    config = ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
    )

    # Apply overrides.
    config = config.merge(overrides)

    return config


def parser_args():
    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        usage=f"python {sys.argv[0]} RUN_NAME [OPTIONS...] [CONFIG_OVERRIDES...]",
        description="Test exponential LR scheduler on c4.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_name", type=str, help="""The name of the run.""")
    parser.add_argument(
        "--model-factory",
        type=str,
        default="olmo2_370M",
        help="""The name of the model factory to use.
        Options: olmo2_30M (smallest), olmo2_190M, llama2_271M, olmo2_370M (default)""",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=2048,
        help="""The sequence length to train and eval on.""",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10_000,
        help="""Maximum number of training steps.""",
    )
    parser.add_argument(
        "--save-folder",
        type=str,
        help="""A local or remote directory to save checkpoints to.
        Defaults to a temporary directory if not provided.""",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        help="""A local working directory for dataset preprocessing.
        Defaults to a temporary directory if not provided.""",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="""Print the config and exit.""",
    )
    opts, overrides = parser.parse_known_args()
    return opts, overrides


def main():
    opts, overrides = parser_args()
    config = build_config(opts, overrides)

    if opts.dry_run:
        rich.print(config)
        return

    prepare_training_environment()
    train(config)
    teardown_training_environment()


if __name__ == "__main__":
    main()
