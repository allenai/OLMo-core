"""
Train a large OLMoE model. Run this script without any arguments to see usage info.
"""

import dataclasses
import logging

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import AOFloat8LinearConfig, Float8Config
from olmo_core.internal.experiment import CommonComponents, main
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.moe import (
    MoEConfig,
    MoELoadBalancingLossGranularity,
    MoERouterConfig,
    MoERouterGatingFunction,
    MoEType,
)
from olmo_core.nn.transformer import (
    TransformerBlockType,
    TransformerConfig,
    TransformerType,
)
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import CheckpointerCallback, CometCallback, WandBCallback
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerExpertParallelConfig,
    TransformerTrainModuleConfig,
)

log = logging.getLogger(__name__)


SEQUENCE_LENGTH = 4096
GLOBAL_BATCH_SIZE = 2048 * SEQUENCE_LENGTH
MAX_DURATION = int(6e12)


def build_model_config(common: CommonComponents) -> TransformerConfig:
    d_model = 4096
    hidden_size = 11008

    config = TransformerConfig.llama_like(
        d_model=d_model,
        vocab_size=common.tokenizer.padded_vocab_size(),
        n_layers=32,
        n_heads=32,
        name=TransformerType.moe,
        block_name=TransformerBlockType.moe_hybrid_reordered_norm,
        qk_norm=True,
        rope_theta=500_000,
        layer_norm_eps=1e-6,
        feed_forward_moe=MoEConfig(
            name=MoEType.default,
            num_experts=128,
            hidden_size=1024,
            capacity_factor=1.05,
            router=MoERouterConfig(top_k=4, gating_function=MoERouterGatingFunction.sigmoid),
            lb_loss_weight=0.01,
            z_loss_weight=None,
            lb_loss_granularity=MoELoadBalancingLossGranularity.instance,
            scale_loss_by_num_layers=False,
        ),
        feed_forward=FeedForwardConfig(hidden_size=hidden_size, bias=False),
        init_std=0.01,
    )

    # First block will be a regular transformer block (no MoE component).
    config.block_overrides = {
        0: dataclasses.replace(
            config.block, name=TransformerBlockType.reordered_norm, feed_forward_moe=None
        ),
    }

    return config


def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
    return TransformerTrainModuleConfig(
        rank_microbatch_size=2 * SEQUENCE_LENGTH,
        max_sequence_length=common.dataset.effective_sequence_length,
        optim=SkipStepAdamWConfig(
            lr=3e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
            compile=True,
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.bfloat16,
            num_replicas=1,  # to enable full-way expert parallel
            prefetch_factor=1,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        ep_config=TransformerExpertParallelConfig(degree=-1),
        float8_config=Float8Config(
            ao=AOFloat8LinearConfig(
                enable_fsdp_float8_all_gather=True,
                force_recompute_fp8_weight_in_bwd=True,
                round_scales_to_power_of_2=True,
            ),
            enabled=False,
        ),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=2000),
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    cancel_check_interval = 10

    #  assert common.launch is not None
    #  assert len(common.launch.clusters) == 1
    #  cluster = common.launch.clusters[0]

    return (
        TrainerConfig(
            save_folder=common.save_folder,
            save_overwrite=True,
            metrics_collect_interval=5,
            cancel_check_interval=cancel_check_interval,
            max_duration=Duration.tokens(MAX_DURATION),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
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
                cancel_check_interval=cancel_check_interval,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=common.run_name,
                entity="ai2-llm",
                project="OLMoE",
                enabled=True,
                cancel_check_interval=cancel_check_interval,
            ),
        )
        # TODO: might not be able to run in-loop evals depending on parallel strategies
        #  .with_recommended_evals(
        #      common.tokenizer, SEQUENCE_LENGTH, cluster
        #  )
    )


if __name__ == "__main__":
    main(
        global_batch_size=GLOBAL_BATCH_SIZE,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        include_instance_filter=False,  # We use SkipStepOptimizer for this problem.
        include_default_evals=False,
        num_nodes=8,
    )
