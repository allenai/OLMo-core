"""
Train a Llama 8B long context model. Run this script without any arguments to see usage info.
"""

import logging

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.experiment import CommonComponents, main
from olmo_core.nn.attention import RingAttentionLoadBalancerType
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup
from olmo_core.train import TrainerConfig
from olmo_core.train.callbacks import CheckpointerCallback, CometCallback, WandBCallback
from olmo_core.train.callbacks.monkey_patcher import MonkeyPatcherCallback
from olmo_core.train.train_module import (
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)

log = logging.getLogger(__name__)

SEQ_LENGTH = 131072


def build_model_config(common: CommonComponents) -> TransformerConfig:
    model = TransformerConfig.llama3_8B(vocab_size=common.tokenizer.padded_vocab_size())
    model.block.attention.use_flash = True
    return model


def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
    return TransformerTrainModuleConfig(
        rank_microbatch_size=SEQ_LENGTH,
        max_sequence_length=common.dataset.effective_sequence_length,
        optim=AdamWConfig(
            lr=3e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            fused=True,
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
        ),
        cp_config=TransformerContextParallelConfig(
            degree=8, head_stride=4, load_balancer=RingAttentionLoadBalancerType.llama3
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=2000),
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    return (
        TrainerConfig(
            save_folder=common.save_folder,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=1,
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
            "comet",
            CometCallback(
                name=common.run_name,
                workspace="ai2",
                project="Llama-8B",
                enabled=True,
                cancel_check_interval=10,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=common.run_name,
                entity="ai2-llm",
                project="Llama-8B",
                enabled=False,
                cancel_check_interval=10,
            ),
        )
        .with_callback("monkey_patcher", MonkeyPatcherCallback(enabled=False))
    )


if __name__ == "__main__":
    main(
        global_batch_size=128 * SEQ_LENGTH,
        sequence_length=SEQ_LENGTH,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        intra_document_masking=True,
        include_default_evals=False,
    )
