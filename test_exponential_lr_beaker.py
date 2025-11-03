"""
Beaker script to test exponential learning rate scheduler for LR range testing.

This script does a 10k step run where the learning rate exponentially increases
from 1e-9 to 10, allowing you to find the optimal learning rate by watching loss/metrics.

Usage:
    python test_exponential_lr_beaker.py dry_run exponential_lr_test_370M
    python test_exponential_lr_beaker.py launch exponential_lr_test_370M
    python test_exponential_lr_beaker.py train exponential_lr_test_370M  # (called by Beaker)
"""

import logging
import sys
from dataclasses import dataclass
from typing import List, Optional, cast

from olmo_core.config import Config, DType
from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, TokenizerConfig
from olmo_core.data.numpy_dataset import NumpyDatasetConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.internal.common import (
    build_launch_config,
    get_beaker_username,
    get_root_dir,
    get_work_dir,
)
from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, ExponentialScheduler, OptimGroupOverride
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_cli_environment,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    BeakerCallback,
    CheckpointerCallback,
    ConfigSaverCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

log = logging.getLogger(__name__)

#######################
#### CONFIGURATION ####
#######################

# Data configuration
SEQUENCE_LENGTH = 2048
TOKENIZER_CONFIG = TokenizerConfig.gpt2()

# C4 data path - using Weka on ai2/augusta (fast and reliable)
DATA_ROOT = "/weka/oe-training-default/ai2-llm/examples/c4-en/gpt2"
DATA_PATHS = [f"{DATA_ROOT}/c4-train.00000-00099.npy"]

GLOBAL_BATCH_SIZE = 32 * 1024  # tokens (reduced for single GPU testing)
RANK_MICROBATCH_SIZE = 16 * 1024  # tokens

# Model config - use olmo2_370M or smaller
MODEL_FACTORY = "olmo2_370M" 

# LR Range Test Configuration
LR_MIN = 1e-9
LR_MAX = 10.0
MAX_STEPS = 10_000

# Beaker configuration
BEAKER_CLUSTER = "ai2/jupiter"  # Has Weka access
NUM_NODES = 1
NUM_GPUS = 1  
BEAKER_WORKSPACE = "ai2/OLMo-core"
BEAKER_BUDGET = "ai2/oe-base"  # Using oe-base budget

# Logging
WANDB_PROJECT = "bailey-testing"
WANDB_ENTITY = "ai2-llm"
WANDB_GROUP = "exponential_lr_scheduler_370M"

###########################
#### END CONFIGURATION ####
###########################


@dataclass
class ExperimentConfig(Config):
    launch: BeakerLaunchConfig
    model: TransformerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = 12536


def build_config(script: str, run_name: str, overrides: List[str]) -> ExperimentConfig:
    beaker_user = get_beaker_username()
    assert beaker_user is not None

    # Use standard paths (augusta has Weka access)
    root_dir = get_root_dir(BEAKER_CLUSTER)
    work_dir = get_work_dir(root_dir)

    # Model config
    factory = getattr(TransformerConfig, MODEL_FACTORY)
    model_config = factory(vocab_size=TOKENIZER_CONFIG.padded_vocab_size())

    # Dataset config
    dataset_config = NumpyFSLDatasetConfig(
        paths=DATA_PATHS,
        sequence_length=SEQUENCE_LENGTH,
        tokenizer=TOKENIZER_CONFIG,
        work_dir=work_dir,
    )

    # Data loader config
    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=0,
        num_workers=4,
    )

    # Train module config with Exponential Scheduler
    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=RANK_MICROBATCH_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=AdamWConfig(
            lr=LR_MAX,  # Max LR for LR range test
            weight_decay=0.01,
            betas=(0.9, 0.999),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
            fused=True,
        ),
        compile_model=False,  # Disable compilation for easier debugging
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
        max_grad_norm=1.0,
        # Use ExponentialScheduler for LR range test: 1e-9 -> 10 over 10k steps
        scheduler=ExponentialScheduler(lr_min=LR_MIN),
    )

    # Trainer config
    # Save to a test-specific folder to avoid overwriting any real training runs
    trainer_config = (
        TrainerConfig(
            save_folder=f"{root_dir}/checkpoints/{beaker_user.lower()}/lr-range-tests/{run_name}",
            save_overwrite=True,  # Safe since checkpointing is disabled and in test folder
            metrics_collect_interval=50,  # Log every 50 steps (200 points over 10k)
            cancel_check_interval=5,
            max_duration=Duration.steps(MAX_STEPS),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                ephemeral_save_interval=500,  # Save every 500 steps (can resume if job crashes)
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                entity=WANDB_ENTITY,
                project=WANDB_PROJECT,
                group=WANDB_GROUP,
                cancel_check_interval=10,
                enabled=True,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("garbage_collector", GarbageCollectorCallback())
        .with_callback("beaker", BeakerCallback())
    )

    return ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
        launch=build_launch_config(
            name=run_name,
            root_dir=root_dir,
            cmd=["torchrun", "--nproc-per-node=1", script, "train", run_name, *overrides],
            cluster=BEAKER_CLUSTER,
            workspace=BEAKER_WORKSPACE,
            budget=BEAKER_BUDGET,
            num_nodes=NUM_NODES,
            num_gpus=NUM_GPUS,
        ),
    ).merge(overrides)


def train(config: ExperimentConfig):
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

    # Train.
    trainer.fit()


def launch(config: ExperimentConfig):
    config.launch.launch(follow=True)


if __name__ == "__main__":
    usage = f"""
Usage
=====

› python {sys.argv[0]} [dry_run|launch|train] RUN_NAME [OVERRIDES...]

  * dry_run: Print out the final config after applying overrides and exit. Useful for debugging.
  * launch:  Launch the script on Beaker as a batch job for training.
  * train:   Run the script for training locally. This should usually not be called directly.

Examples
========

Print the config:
› python {sys.argv[0]} dry_run exponential_lr_test_370M

Launch the training run as a Beaker batch job:
› python {sys.argv[0]} launch exponential_lr_test_370M

Launch with a smaller model:
› python {sys.argv[0]} launch exponential_lr_test_30M --model_factory=olmo2_30M

Launch with different LR range:
› python {sys.argv[0]} launch exponential_lr_test --train_module.optim.lr=5.0 --train_module.scheduler.lr_min=1e-8
    """.strip()

    if len(sys.argv) < 3:
        print(usage)
        sys.exit(1)

    script, cmd, run_name, *overrides = sys.argv

    if cmd == "train":
        prepare_training_environment()
    else:
        prepare_cli_environment()

    config = build_config(script, run_name, overrides)
    log.info(config)

    if cmd == "train":
        train(config)
        teardown_training_environment()
    elif cmd == "launch":
        launch(config)
    elif cmd == "dry_run":
        pass
    else:
        print(usage)
        sys.exit(1)
