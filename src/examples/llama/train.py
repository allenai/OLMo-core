"""
Example of how to train a Llama transformer language model.

Launch this with torchrun:

    torchrun --nproc-per-node=4 src/examples/llama/train.py run_name [OVERRIDES...]
"""

import logging
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, cast

import rich

from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyDatasetType,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
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

# The sequence length to train and eval on.
SEQUENCE_LENGTH = 1024

# Check for the data on common Ai2 drives. If those don't exist we'll stream the data over the internet,
# which can be a lot slower. Alternatively you can download the files with wget, for example:
#  > wget http://olmo-data.org/examples/c4-en/gpt2/c4-train.00000-00099.npy
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
    # Uncomment for full dataset which might not be available on NFS or Weka.
    #  f"{DATA_ROOT}/c4-train.00100-00199.npy",
    #  f"{DATA_ROOT}/c4-train.00200-00299.npy",
    #  f"{DATA_ROOT}/c4-train.00300-00399.npy",
    #  f"{DATA_ROOT}/c4-train.00400-00499.npy",
    #  f"{DATA_ROOT}/c4-train.00500-00599.npy",
    #  f"{DATA_ROOT}/c4-train.00600-00699.npy",
    #  f"{DATA_ROOT}/c4-train.00700-00799.npy",
    #  f"{DATA_ROOT}/c4-train.00800-00899.npy",
    #  f"{DATA_ROOT}/c4-train.00900-00999.npy",
    #  f"{DATA_ROOT}/c4-train.01000-01023.npy",
]
EVAL_DATA_PATHS = [f"{DATA_ROOT}/c4-validation.00000-00008.npy"]
DATA_WORK_DIR = "/tmp/dataset-cache"


# docs: start-define-config
@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = 12536
    load_path: Optional[str] = None
    dry_run: bool = False
    # docs: end-define-config


def build_config(run_name: str, overrides: List[str]) -> ExperimentConfig:
    tokenizer_config = TokenizerConfig.gpt2()

    model_config = TransformerConfig.llama2_271M(
        vocab_size=tokenizer_config.padded_vocab_size(),  # a little bigger than actual vocab size to make it a multiple of 128
    )

    log.info(f"Using data root: {DATA_ROOT}")
    dataset_config = NumpyDatasetConfig(
        paths=DATA_PATHS,
        name=NumpyDatasetType.fsl,
        sequence_length=SEQUENCE_LENGTH,
        max_target_sequence_length=8192,
        tokenizer=tokenizer_config,
        work_dir=DATA_WORK_DIR,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=256 * SEQUENCE_LENGTH,
        seed=0,
        num_workers=4,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=16 * SEQUENCE_LENGTH,
        max_sequence_length=dataset_config.effective_sequence_length,
        optim=AdamWConfig(
            lr=1e-3,
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
        ),
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=100),
    )

    trainer_config = (
        TrainerConfig(
            save_folder=f"/tmp/{run_name}",
            save_overwrite=True,
            metrics_collect_interval=5,
            cancel_check_interval=5,
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=100,
                save_async=True,
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=run_name,
                cancel_check_interval=10,
                enabled=False,  # change to true to enable
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                cancel_check_interval=10,
                enabled=False,  # change to true to enable
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("profiler", ProfilerCallback(enabled=False))
        .with_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=NumpyDatasetConfig(
                    paths=EVAL_DATA_PATHS,
                    metadata=[{"label": "c4-validation"}],
                    name=NumpyDatasetType.padded_fsl,
                    sequence_length=SEQUENCE_LENGTH,
                    tokenizer=tokenizer_config,
                    work_dir=DATA_WORK_DIR,
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
    # docs: start-config-merge
    config = config.merge(overrides)
    # docs: end-config-merge

    return config


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

    # docs: start-load-path
    # If we have a load path set and there is no checkpoint in the save folder, load the
    # checkpoint from the load path.
    if not trainer.no_checkpoints and not trainer.maybe_load_checkpoint() and config.load_path:
        log.info(
            f"Loading checkpoint from {config.load_path} since no checkpoints were found in the save folder..."
        )
        trainer.load_checkpoint(config.load_path)
    # docs: end-load-path

    # Train.
    trainer.fit()


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} run_name [OVERRIDES...]")
        sys.exit(1)

    run_name, *overrides = sys.argv[1:]
    config = build_config(run_name, overrides)

    if config.dry_run:
        rich.print(config)
        return

    prepare_training_environment()
    try:
        train(config)
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
