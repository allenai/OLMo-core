"""
Train a 7B OLMo model. Run this script without any arguments to see usage info.
"""

import logging

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelConfig, DataParallelType
from olmo_core.internal.experiment import CommonComponents, main
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, OptimGroupOverride
from olmo_core.train import TrainerConfig
from olmo_core.train.callbacks import CheckpointerCallback, WandBCallback

log = logging.getLogger(__name__)


def build_model_config(common: CommonComponents) -> TransformerConfig:
    return TransformerConfig.olmo_7B(
        vocab_size=common.tokenizer.padded_vocab_size(),
        compile=True,
        dp_config=DataParallelConfig(
            name=DataParallelType.fsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
        ),
    )


def build_optim_config(common: CommonComponents) -> AdamWConfig:
    del common
    return AdamWConfig(
        lr=3e-4,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        group_overrides=[
            OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
        ],
        fused=True,
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    return (
        TrainerConfig(
            save_folder=common.save_folder,
            global_batch_size=1024 * 4096,
            rank_microbatch_size=2 * 4096,
            autocast_precision=DType.bfloat16,
            save_overwrite=True,
            data_seed=34521,
            data_loader_workers=4,
            metrics_collect_interval=10,
            cancel_check_interval=1,
            z_loss_multiplier=1e-5,
            fused_loss=True,
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=10_000,
                ephemeral_save_interval=250,
                save_async=True,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=common.run_name,
                entity="ai2-llm",
                project="OLMo-core-7B",
                enabled=True,
                cancel_check_interval=10,
            ),
        )
    )


if __name__ == "__main__":
    main(
        model_config_builder=build_model_config,
        optim_config_builder=build_optim_config,
        trainer_config_builder=build_trainer_config,
    )
