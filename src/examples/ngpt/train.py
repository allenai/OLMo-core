"""
Example of how to train an nGPT language model.

Launch this with torchrun:

    torchrun --nproc-per-node=4 src/examples/ngpt/train.py run_name [OVERRIDES...]
"""

import sys
from dataclasses import dataclass
from typing import List, cast

from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyDatasetType,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamConfig, CosWithWarmup
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

SEQUENCE_LENGTH = 1024

DATA_PATHS = [
    "http://olmo-data.org/examples/c4-en/gpt2/c4-train.00000-00099.npy",
    "http://olmo-data.org/examples/c4-en/gpt2/c4-train.00100-00199.npy",
    "http://olmo-data.org/examples/c4-en/gpt2/c4-train.00200-00299.npy",
    "http://olmo-data.org/examples/c4-en/gpt2/c4-train.00300-00399.npy",
    "http://olmo-data.org/examples/c4-en/gpt2/c4-train.00400-00499.npy",
    "http://olmo-data.org/examples/c4-en/gpt2/c4-train.00500-00599.npy",
    "http://olmo-data.org/examples/c4-en/gpt2/c4-train.00600-00699.npy",
    "http://olmo-data.org/examples/c4-en/gpt2/c4-train.00700-00799.npy",
    "http://olmo-data.org/examples/c4-en/gpt2/c4-train.00800-00899.npy",
    "http://olmo-data.org/examples/c4-en/gpt2/c4-train.00900-00999.npy",
    "http://olmo-data.org/examples/c4-en/gpt2/c4-train.01000-01023.npy",
]


@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = 12536


def build_config(run_name: str, overrides: List[str]) -> ExperimentConfig:
    tokenizer_config = TokenizerConfig.gpt2()

    model_config = TransformerConfig.ngpt_271M(
        vocab_size=tokenizer_config.padded_vocab_size(),  # a little bigger than actual vocab size to make it a multiple of 128
    )

    dataset_config = NumpyDatasetConfig(
        paths=DATA_PATHS,
        name=NumpyDatasetType.fsl,
        sequence_length=SEQUENCE_LENGTH,
        max_target_sequence_length=8192,
        tokenizer=tokenizer_config,
        work_dir="/tmp/dataset-cache",
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=256 * SEQUENCE_LENGTH,
        seed=0,
        num_workers=4,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=16 * SEQUENCE_LENGTH,
        max_sequence_length=dataset_config.effective_sequence_length,
        optim=AdamConfig(lr=1e-3),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
        ),
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=0),
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
                    paths=[
                        "http://olmo-data.org/examples/c4-en/gpt2/c4-validation.00000-00008.npy"
                    ],
                    metadata=[{"label": "c4-validation"}],
                    name=NumpyDatasetType.padded_fsl,
                    sequence_length=SEQUENCE_LENGTH,
                    tokenizer=tokenizer_config,
                    work_dir="/tmp/dataset-cache",
                ),
                eval_interval=250,
                eval_duration=Duration.steps(10),
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

    return ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
    ).merge(overrides)


def main(run_name: str, overrides: List[str]):
    config = build_config(run_name, overrides)

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
    cast(CometCallback, trainer.callbacks["comet"]).config = config_dict
    cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    # Train.
    trainer.fit()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} run_name [OVERRIDES...]")
        sys.exit(1)

    run_name, *overrides = sys.argv[1:]

    prepare_training_environment()
    try:
        main(run_name, overrides=overrides)
    finally:
        teardown_training_environment()
