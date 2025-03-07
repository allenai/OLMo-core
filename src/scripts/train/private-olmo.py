"""
Train an Nx7B OLMo2 model. Run this script without any arguments to see usage info.
"""

import logging

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.experiment import CommonComponents, main
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup
from olmo_core.train import TrainerConfig
from olmo_core.train.callbacks import CheckpointerCallback, CometCallback, WandBCallback
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerExpertParallelConfig,
    TransformerTensorParallelConfig,
    TransformerTrainModuleConfig,
)

log = logging.getLogger(__name__)

CONTEXT_LENGTH = 4096


def build_model_config(common: CommonComponents) -> TransformerConfig:
    d_model = 4096
    dropless = True  # TODO: ablate this?
    return TransformerConfig.llama_like_moe(
        vocab_size=common.tokenizer.padded_vocab_size(),
        d_model=d_model,
        n_layers=32,
        n_heads=32,
        num_experts=8,
        top_k=1,
        expert_hidden_size=11008,
        dropless=dropless,
        capacity_factor=None if dropless else 1.2,  # adjust as needed
        lb_loss_weight=0.01,
        z_loss_weight=0.001,
        reordered_norm=True,
        qk_norm=True,
        rope_theta=500_000,
        layer_norm_eps=1e-6,
        freeze_params=[
            "embeddings.*",
            "blocks.*.attention*",
            "blocks.*.feed_forward_norm.*",  # TODO: not sure if you want this frozen
            "lm_head.*",
        ],
    )


def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
    return TransformerTrainModuleConfig(
        rank_microbatch_size=1 * CONTEXT_LENGTH,
        max_sequence_length=common.dataset.effective_sequence_length,
        optim=AdamWConfig(
            lr=3e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            fused=True,
            #  group_overrides=[
            #      OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            #  ],
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            #  num_replicas=1,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            prefetch_factor=1,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        #  ep_config=TransformerExpertParallelConfig(degree=-1),
        tp_config=TransformerTensorParallelConfig(degree=8),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=None,  # TODO: Z-loss on router logits, not sure if you want this
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
                project="private-olmo",
                enabled=True,
                cancel_check_interval=10,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=common.run_name,
                entity="ai2-llm",
                project="private-olmo",
                enabled=False,
                cancel_check_interval=10,
            ),
        )
    )


if __name__ == "__main__":
    main(
        global_batch_size=16 * CONTEXT_LENGTH,  # TODO: adjust as needed
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        sequence_length=CONTEXT_LENGTH,
        include_default_evals=False,
    )
