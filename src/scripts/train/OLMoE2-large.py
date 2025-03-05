"""
Train a large OLMoE model. Run this script without any arguments to see usage info.
"""

import logging

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.experiment import CommonComponents, main
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import (
    AdamWConfig,
    CosWithWarmup,
    OptimGroupOverride,
    SkipStepAdamWConfig,
)
from olmo_core.train import TrainerConfig
from olmo_core.train.callbacks import CheckpointerCallback, CometCallback, WandBCallback
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerExpertParallelConfig,
    TransformerTrainModuleConfig,
)

log = logging.getLogger(__name__)

SKIP_STEP_BF16_OPTIM = False


def build_model_config(common: CommonComponents) -> TransformerConfig:
    d_model = 4096
    return TransformerConfig.llama_like_moe(
        vocab_size=common.tokenizer.padded_vocab_size(),
        d_model=d_model,
        n_layers=32,
        n_heads=32,
        num_experts=64,
        top_k=2,
        expert_hidden_size=int(0.5 * d_model),
        shared_expert_hidden_size=d_model * 2,
        capacity_factor=1.05,
        lb_loss_weight=0.01,
        z_loss_weight=0.001,
        reordered_norm=True,
        qk_norm=True,
        rope_theta=500_000,
        layer_norm_eps=1e-6,
    )


def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
    return TransformerTrainModuleConfig(
        rank_microbatch_size=1 * 4096,
        max_sequence_length=common.dataset.effective_sequence_length,
        optim=SkipStepAdamWConfig(
            lr=3e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
            compile=True,
            dtype=DType.bfloat16,
        )
        if SKIP_STEP_BF16_OPTIM
        else AdamWConfig(
            lr=3e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
            fused=True,
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.bfloat16,
            num_replicas=1,  # to enable full-way expert parallel
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        ep_config=TransformerExpertParallelConfig(degree=-1),
        float8_config=Float8Config(enabled=True),
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
                project="OLMoE",
                enabled=True,
                cancel_check_interval=10,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=common.run_name,
                entity="ai2-llm",
                project="OLMoE",
                enabled=False,
                cancel_check_interval=10,
            ),
        )
    )


if __name__ == "__main__":
    main(
        global_batch_size=2048 * 4096,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        include_default_evals=False,
    )
