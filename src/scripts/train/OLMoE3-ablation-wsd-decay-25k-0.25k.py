"""
Train an OLMoE model. Run this script without any arguments to see usage info.
"""

import logging
import math

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import AOFloat8LinearConfig, Float8Config
from olmo_core.internal.experiment import CommonComponents, main, ExperimentConfig
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
from olmo_core.optim import WSD, OptimGroupOverride, SchedulerUnits, SkipStepAdamWConfig
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import (
    BatchSizeSchedulerCallback,
    CheckpointerCallback,
    CometCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

log = logging.getLogger(__name__)



SEQUENCE_LENGTH = 4096
GLOBAL_BATCH_SIZE = (
    1024 * 4096
)  # batch size at step 0, let's keep this independent of the sequence length in case we change it.
# MAX_DURATION = int(500e9)  # int(6e12), don't forget to adjust the LR when you increase this
EVAL_INTERVAL = 1000
NUM_EXPERTS = 64
TOP_K = 8
NUM_LAYERS=16
MOE_HIDDEN_SIZE = 1024
USE_SHARED_MLP = False  # Use shared MLP in MoE blocks
SHARED_MLP_HIDDEN_SIZE = 2560  # Hidden size for shared MLP in MoE blocks

###### decay 
START_STEP = 25000
DECAY_STEPS= 250
LOAD_CKPT = f'/weka/oe-training-default/tianhua/ws-megatron/tmp/OLMoE3-ablation-wsd/step{START_STEP}'
TOTAL_TOKENS = (START_STEP+ DECAY_STEPS) * GLOBAL_BATCH_SIZE  # 4096 is the sequence length, 1024 is the batch size at step 0
MAX_DURATION = int(TOTAL_TOKENS)
############


def build_model_config(common: CommonComponents) -> TransformerConfig:
    d_model = 2048

    config = TransformerConfig.llama_like(
        d_model=d_model,
        vocab_size=common.tokenizer.padded_vocab_size(),
        n_layers=NUM_LAYERS,
        n_heads=16,
        name=TransformerType.moe,
        block_name=TransformerBlockType.moe_reordered_norm,
        qk_norm=True,
        rope_theta=500_000,
        layer_norm_eps=1e-6,
        feed_forward_moe=MoEConfig(
            name=MoEType.default,
            num_experts=NUM_EXPERTS,
            hidden_size=MOE_HIDDEN_SIZE,
            capacity_factor=1.25,
            router=MoERouterConfig(top_k=TOP_K, gating_function=MoERouterGatingFunction.sigmoid),
            shared_mlp=FeedForwardConfig(hidden_size=SHARED_MLP_HIDDEN_SIZE, bias=False) if USE_SHARED_MLP else None,
            lb_loss_weight=0.05,
            z_loss_weight=None,
            lb_loss_granularity=MoELoadBalancingLossGranularity.instance,
            scale_loss_by_num_layers=False,
        ),
        #  feed_forward=FeedForwardConfig(hidden_size=hidden_size, bias=False),
        init_std=0.01,
    )

    # First block will be a regular transformer block (no MoE component).
    #  config.block_overrides = {
    #      0: dataclasses.replace(
    #          config.block, name=TransformerBlockType.reordered_norm, feed_forward_moe=None
    #      ),
    #  }

    return config


def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
    return TransformerTrainModuleConfig(
        rank_microbatch_size=2 * SEQUENCE_LENGTH,
        max_sequence_length=common.dataset.effective_sequence_length,
        optim=SkipStepAdamWConfig(
            lr=1.6e-4
            * math.sqrt(
                GLOBAL_BATCH_SIZE / (4096 * 512)
            ),  # 1.6e-4 was used for 2M batch size, adjusting it accordingly
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
            #  num_replicas=1,  # to enable full-way expert parallel
            shard_degree=8,
            prefetch_factor=1,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
        ),
        #  ep_config=TransformerExpertParallelConfig(degree=-1),
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
        scheduler=WSD(
            units=SchedulerUnits.steps,
            warmup=2000,
            decay=(int(DECAY_STEPS)),
            decay_fraction=None,
        ),
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    cancel_check_interval = 10

    # assert common.launch is not None
    # assert len(common.launch.clusters) == 1
    # cluster = common.launch.clusters[0]
    cluster = 'ai2/jupiter-cirrascale-2'
    return (
        TrainerConfig(
            save_folder=f'/workspace/tmp/{common.run_name}',
            save_overwrite=True,
            metrics_collect_interval=5,
            cancel_check_interval=cancel_check_interval,
            max_duration=Duration.tokens(MAX_DURATION),
            load_path=LOAD_CKPT,  # Load from the before decay
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
                project="olmo3",
                enabled=False,
                cancel_check_interval=cancel_check_interval,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=common.run_name,
                entity="ai2-llm",
                # project="tianhua-moe",
                project="olmo3",
                enabled=True,
                cancel_check_interval=cancel_check_interval,
            ),
        )
        .with_callback(
            "batchwup",
            BatchSizeSchedulerCallback(
                batch_sizes=[GLOBAL_BATCH_SIZE, GLOBAL_BATCH_SIZE * 2, GLOBAL_BATCH_SIZE * 4],
                schedule=[
                    Duration.tokens(0),
                    Duration.tokens(167_772_160_000),
                    Duration.tokens(503_316_480_000),
                ],
            ),
        )
        # TODO: might not be able to run in-loop evals depending on parallel strategies
        .with_recommended_evals(
            common.tokenizer, SEQUENCE_LENGTH, cluster, task_set="fast", eval_interval=EVAL_INTERVAL
        )
    )


def finalize_config(config: ExperimentConfig):
    # add active & total params to the wandb name
    total_params_in_B = config.model.num_params/1000/1000/1000
    active_params_in_B = config.model.num_active_params/1000/1000/1000
    config.trainer.callbacks['wandb'].name += f"_{active_params_in_B:.2f}@{total_params_in_B:.2f}B"  # print to 2 decimal places
    config.trainer.callbacks['wandb'].name += f"_{TOP_K}K{NUM_EXPERTS}N"  # print to 2 decimal places
    
if __name__ == "__main__":
    main(
        global_batch_size=GLOBAL_BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        include_instance_filter=False,  # We use SkipStepOptimizer for this problem.
        include_default_evals=False,
        finalize_config=finalize_config,
        
    )
