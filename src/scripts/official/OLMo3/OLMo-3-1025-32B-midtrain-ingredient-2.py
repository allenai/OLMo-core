"""
Official mid-training script for OLMo-3-1025-32B. (ingredient 2).

We performed two mid-training runs (ingredient 1 and 2) and combined the final checkpoints together into a souped model.
"""

import argparse
from typing import List

from olmo_core.config import DType
from olmo_core.data import (
    DataMix,
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingMode,
    TransformerConfig,
)
from olmo_core.optim import LinearWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.script_utils import ExperimentConfig, main
from olmo_core.train import Duration, LoadStrategy, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    ConfigSaverCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

DEFAULT_SEQUENCE_LENGTH = 8192
GLOBAL_BATCH_SIZE = 4 * 1024 * 1024  # ~4M tokens
MAX_TOKENS = 100_000_000_000  # 100B
LR = 0.0002071235285
SEED = 42069


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    sequence_length = opts.sequence_length or DEFAULT_SEQUENCE_LENGTH
    tokenizer_config = TokenizerConfig.dolma2()

    model_config = TransformerConfig.olmo3_32B(
        vocab_size=tokenizer_config.padded_vocab_size(),  # pad to a multiple of 128
        attn_backend=AttentionBackendName.flash_2,
    )

    dataset_config = NumpyFSLDatasetConfig.from_data_mix(
        DataMix.OLMo_midtraining_mix_0925_ingredient2_100B,
        tokenizer=tokenizer_config,
        mix_base_dir=opts.data_root,
        work_dir=opts.work_dir,
        sequence_length=sequence_length,
        max_target_sequence_length=max(8192, sequence_length),
        instance_filter_config=InstanceFilterConfig(
            repetition_max_period=13, repetition_min_period=1, repetition_max_count=32
        ),
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE, seed=SEED, num_workers=4
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=sequence_length,
        max_sequence_length=sequence_length,
        optim=SkipStepAdamWConfig(
            lr=LR,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        scheduler=LinearWithWarmup(warmup=0, alpha_f=0.0),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            shard_degree=64,
        ),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.budget,
            activation_memory_budget=0.5,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=opts.save_folder,
            save_overwrite=True,
            load_path="https://olmo-checkpoints.org/ai2-llm/stego32-highlr-filter3/step656000/",
            load_strategy=LoadStrategy.always,
            load_trainer_state=False,
            load_optim_state=True,
            max_duration=Duration.tokens(MAX_TOKENS),
            work_dir=opts.work_dir,
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=None,
                save_async=False,
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=opts.name,
                cancel_check_interval=10,
                enabled=False,  # NOTE: change to true to enable
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=opts.name,
                cancel_check_interval=10,
                enabled=False,  # NOTE: change to true to enable
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


if __name__ == "__main__":
    main(build_config)
