"""
Train a 1B nGPT model. Run this script without any arguments to see usage info.
"""

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.internal.experiment import CommonComponents, main
from olmo_core.nn.transformer import TransformerConfig, TransformerDataParallelConfig
from olmo_core.optim import AdamConfig, CosWithWarmup
from olmo_core.train import TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    MatrixNormalizerCallback,
    SchedulerCallback,
    WandBCallback,
)


def build_model_config(common: CommonComponents) -> TransformerConfig:
    return TransformerConfig.ngpt_1B(
        vocab_size=common.tokenizer.padded_vocab_size(),
        compile=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
        ),
    )


def build_optim_config(common: CommonComponents) -> AdamConfig:
    del common
    return AdamConfig(
        lr=4e-4,
        betas=(0.9, 0.95),
        fused=True,
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    return (
        TrainerConfig(
            save_folder=common.save_folder,
            rank_microbatch_size=4 * 4096,  # TODO: can we increase this?
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=1,
            z_loss_multiplier=1e-5,
            compile_loss=True,
        )
        .with_callback("lr_scheduler", SchedulerCallback(scheduler=CosWithWarmup(warmup_steps=0)))
        .with_callback("matrix_normalizer", MatrixNormalizerCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=10_000,
                ephemeral_save_interval=1000,
                save_async=True,
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=common.run_name,
                workspace="ai2",
                project="OLMo-core-1B",
                enabled=True,
                cancel_check_interval=10,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=common.run_name,
                entity="ai2-llm",
                project="OLMo-core-1B",
                enabled=False,
                cancel_check_interval=10,
            ),
        )
    )


if __name__ == "__main__":
    main(
        global_batch_size=1024 * 4096,
        model_config_builder=build_model_config,
        optim_config_builder=build_optim_config,
        trainer_config_builder=build_trainer_config,
    )
