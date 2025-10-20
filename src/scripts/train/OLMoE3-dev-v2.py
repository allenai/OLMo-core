"""
Train an OLMoE model. Run this script without any arguments to see usage info.
"""

import logging
import math
from dataclasses import replace
from typing import cast

import torch
import transformer_engine

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.parallel.pipeline_parallel import PipelineScheduleType
from olmo_core.float8 import AOFloat8LinearConfig, Float8Config
from olmo_core.internal.experiment import CommonComponents, ExperimentConfig, main
from olmo_core.nn.attention import SlidingWindowAttentionConfig
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.moe import (
    MoEConfig,
    MoELoadBalancingLossGranularity,
    MoERouterConfig,
    MoERouterGatingFunction,
    MoEType,
)
from olmo_core.nn.moe.v2.block import (
    MoERouterConfigV2,
    RoutedExpertsConfig,
    SharedExpertsConfig,
)
from olmo_core.nn.transformer import (
    MoEFusedV2TransformerConfig,
    TransformerBlockType,
    TransformerConfig,
    TransformerType,
)
from olmo_core.optim import (
    WSD,
    AdamWConfig,
    OptimGroupOverride,
    SchedulerUnits,
    SkipStepAdamWConfig,
)
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import (
    BatchSizeSchedulerCallback,
    CheckpointerCallback,
    CometCallback,
    NvidiaProfilerCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    MoEV2TransformerTrainModuleConfig,
    TransformerActivationCheckpointingConfig,
    TransformerActivationCheckpointingMode,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerExpertParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.train.train_module.transformer import TransformerPipelineParallelConfig
from olmo_core.train.train_module.transformer.moe_train_module import (
    MoEV2TransformerTrainModule,
)

log = logging.getLogger(__name__)


SEQUENCE_LENGTH = 8192
GLOBAL_BATCH_SIZE_SEQ = 512
GLOBAL_BATCH_SIZE = (GLOBAL_BATCH_SIZE_SEQ) * SEQUENCE_LENGTH
MAX_DURATION = int(1000e9)  # int(6e12), don't forget to adjust the LR when you increase this
EVAL_INTERVAL = 1000
LR = 1e-4

NUM_EXPERTS = 16
TOP_K = 4
D_MODEL = 1024
HEAD_DIM = 64
NUM_HEAD = D_MODEL // HEAD_DIM
NUM_KV_HEAD = 4
MOE_HIDDEN_SIZE = 1024
NUM_SHARED_EXPERTS = 1  # Number of shared experts in the shared MLP
SHARED_MLP_HIDDEN_SIZE = (
    1024  # Hidden size for shared MLP (or dense branch MLP in arctic) in MoE blocks
)

MICRO_BSZ = 2
# DP_DIM=2
EP_DIM = 2
PP_DIM = 2
NUM_LAYERS = 8
SPLIT_POINTS = None

# NUM_LAYERS= 30
# SPLIT_POINTS = [
#     4, 8 ,
#     12,16,
#     20,24 ,
#     28,
#     ]

# NUM_LAYERS= 15
# SPLIT_POINTS =[
#     4,8,
#     12,
# ]

# NUM_LAYERS= 24
# SPLIT_POINTS =[
#     6,12,
#     18,
# ]

# NUM_LAYERS= 46
# SPLIT_POINTS = [
#     6, 12 ,
#     18,24,
#     30,36 ,
#     42,
# ]


# SPLIT_POINTS = None
USE_COMPILE = False
USE_AC = False
USE_TBO = False

TAG = f"dev-ep{EP_DIM}-uni-rs-step1b-pp"
from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig, LMHeadType
from olmo_core.nn.rope import RoPEConfig, RoPEScalingConfig, RoPEType
from olmo_core.nn.transformer import TransformerBlockConfig


# from olmo_core.nn.moe.v2.block import LayerNormConfigV2
def build_model_config(common: CommonComponents) -> TransformerConfig:
    from olmo_core.nn.moe.v2.block import MoEFusedV2TransformerBlockConfig
    from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
    from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig

    d_model = D_MODEL
    dtype = DType.float32

    layer_norm = LayerNormConfig(
        name=LayerNormType.rms,
        eps=1e-6,
        bias=False,
        dtype=dtype,
    )
    config = MoEFusedV2TransformerConfig(
        d_model=d_model,
        two_batch_overlap=USE_TBO,
        vocab_size=common.tokenizer.padded_vocab_size(),
        n_layers=NUM_LAYERS,
        block=MoEFusedV2TransformerBlockConfig(
            name=TransformerBlockType.moe_fused_v2,
            attention=AttentionConfig(
                name=AttentionType.default,
                n_heads=NUM_HEAD,
                n_kv_heads=NUM_KV_HEAD,
                bias=False,
                rope=RoPEConfig(
                    name=RoPEType.default, theta=500_000, scaling=None, full_precision=True
                ),
                qk_norm=layer_norm,
                use_flash=True,
                use_head_qk_norm=True,
                dtype=dtype,
            ),
            attention_norm=layer_norm,
            routed_experts=RoutedExpertsConfig(
                d_model=d_model,
                hidden_size=MOE_HIDDEN_SIZE,
                num_experts=NUM_EXPERTS,
                bias=False,
                dtype=dtype,
            ),
            routed_experts_router=MoERouterConfigV2(
                d_model=d_model,
                num_experts=NUM_EXPERTS,
                top_k=TOP_K,
                gating_function=MoERouterGatingFunction.sigmoid,
                uniform_expert_assignment=True,
                lb_loss_weight=0.005,
                z_loss_weight=None,
                lb_loss_granularity=MoELoadBalancingLossGranularity.instance,
                dtype=dtype,
            ),
            shared_experts=SharedExpertsConfig(
                d_model=d_model,
                hidden_size=SHARED_MLP_HIDDEN_SIZE,
                num_experts=NUM_SHARED_EXPERTS,
                bias=False,
                dtype=dtype,
            )
            if NUM_SHARED_EXPERTS > 0
            else None,
            shared_experts_router=MoERouterConfigV2(
                d_model=d_model,
                num_experts=NUM_SHARED_EXPERTS,
                top_k=NUM_SHARED_EXPERTS,  # all experts are used
                gating_function=MoERouterGatingFunction.sigmoid,
                uniform_expert_assignment=False,
                lb_loss_weight=None,
                z_loss_weight=None,
                lb_loss_granularity=MoELoadBalancingLossGranularity.instance,
                dtype=dtype,
            )
            if NUM_SHARED_EXPERTS > 1
            else None,  # only need router if > 1 expert
            feed_forward_norm=layer_norm,
        ),
        lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=dtype),
        name=TransformerType.moe_fused_v2,
        init_std=0.01,
        dtype=dtype,
    )

    config.lm_head.loss_implementation = LMLossImplementation.fused_linear
    WINDOW_SIZE = 4095
    config.block.attention.sliding_window = SlidingWindowAttentionConfig(
        force_full_attention_on_first_layer=False,
        force_full_attention_on_last_layer=True,
        pattern=[WINDOW_SIZE, WINDOW_SIZE, WINDOW_SIZE, -1],
    )
    # config.block.attention.use_flash = True
    # config.block.attention.use_head_qk_norm = True

    # First block will be a regular transformer block (no MoE component).
    config.block_overrides = {
        0: TransformerBlockConfig(
            name=TransformerBlockType.reordered_norm,
            attention=AttentionConfig(
                name=AttentionType.default,
                n_heads=16,
                n_kv_heads=4,
                bias=False,
                rope=RoPEConfig(
                    name=RoPEType.default, theta=500_000, scaling=None, full_precision=True
                ),
                qk_norm=layer_norm,
                use_flash=True,
                use_head_qk_norm=True,
                dtype=dtype,
            ),
            feed_forward_moe=None,
            feed_forward=FeedForwardConfig(
                hidden_size=(TOP_K * MOE_HIDDEN_SIZE + SHARED_MLP_HIDDEN_SIZE * NUM_SHARED_EXPERTS),
                bias=False,
            ),  # dense mlp is twice as fast as moe mlp
            attention_norm=layer_norm,
            feed_forward_norm=layer_norm,
        )
    }

    return config


def build_train_module_config(common: CommonComponents) -> MoEV2TransformerTrainModuleConfig:
    from olmo_core.optim.moe_optimizer import MoEFusedV2OptimizerConfig

    return MoEV2TransformerTrainModuleConfig(
        rank_microbatch_size=MICRO_BSZ * SEQUENCE_LENGTH,
        max_sequence_length=common.dataset.effective_sequence_length,
        optim=MoEFusedV2OptimizerConfig(
            lr=LR,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(
                    params=["embeddings.weight", "*_norm.weight"], opts=dict(weight_decay=0.0)
                ),
                OptimGroupOverride(
                    params=["*w_up_gate"], opts=dict(weight_decay=0.1)
                )  # HACK: just to make a separate group to avoid OOM in RS
                # OptimGroupOverride(params=["embeddings.weight", ], opts=dict(weight_decay=0.0)) #TODO: fix
            ],
            # TODO: weight decay for norm?
            # fused=True,
            compile=False,
            dtype=DType.float32,
            # foreach=True
        ),
        compile_model=USE_COMPILE,
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.full,
        )
        if USE_AC
        else None,
        # FSDP
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.ddp,
            param_dtype=DType.bfloat16,  # TODO: not used?
            reduce_dtype=DType.float32,  # TODO: not used?
            shard_degree=None,
        ),
        ep_config=TransformerExpertParallelConfig(degree=EP_DIM)
        if EP_DIM != 1
        else None,  # EP=1 means no expert parallel
        pp_config=TransformerPipelineParallelConfig(
            degree=PP_DIM,
            # schedule=PipelineScheduleType.custom_1F1B,
            schedule=PipelineScheduleType.custom_interleaved_1F1B,
            use_custom_stage_implementation=True,  # use custom stage implementation that re-uses receive buffers across micro-batches
            split_points=SPLIT_POINTS,
        )
        if PP_DIM > 1
        else None,
        # float8_config=Float8Config(
        #     ao=AOFloat8LinearConfig(
        #         enable_fsdp_float8_all_gather=True,
        #         force_recompute_fp8_weight_in_bwd=True,
        #         round_scales_to_power_of_2=True,
        #     ),
        #     enabled=False,
        # ),
        float8_config=None,
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=WSD(
            units=SchedulerUnits.steps,
            warmup=500,
            # NOTE: be aware of when decay will happen relative to batch_wup schedule
            decay=(int(50e9 / GLOBAL_BATCH_SIZE)),
            decay_fraction=None,
        ),
    )


# WORK_DIR = "/jfs/tianhua-tao/ws-olmoe"
WORK_DIR = "/weka/oe-training-default/tianhua/ws-megatron"


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    cancel_check_interval = 10

    # cluster = 'ai2/augusta-google-1'

    return (
        TrainerConfig(
            # save_folder=f'{common.save_folder}/{common.run_name}_{D_MODEL}d_{NUM_LAYERS}L{MOE_HIDDEN_SIZE}M{SHARED_MLP_HIDDEN_SIZE}S_{NUM_EXPERTS}E{TOP_K}K_{TAG}',
            save_folder=f"{WORK_DIR}/{common.run_name}_{D_MODEL}d_{NUM_LAYERS}L{MOE_HIDDEN_SIZE}M{SHARED_MLP_HIDDEN_SIZE}S_{NUM_EXPERTS}E{TOP_K}K{NUM_SHARED_EXPERTS}S_{TAG}",
            save_overwrite=True,
            metrics_collect_interval=5,
            cancel_check_interval=cancel_check_interval,
            max_duration=Duration.tokens(MAX_DURATION),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=2000,
                ephemeral_save_interval=1000,
                save_async=False,
                pre_train_checkpoint=False,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=common.run_name,
                # entity="ai2-llm",
                # project="tianhua-moe",
                entity="tianhuat-ai2",
                project="olmo-core-moe",
                # project="olmo3",
                enabled=True,
                cancel_check_interval=cancel_check_interval,
            ),
        )
        .with_callback(
            "batchwup",
            BatchSizeSchedulerCallback(
                batch_sizes=[
                    GLOBAL_BATCH_SIZE,
                    GLOBAL_BATCH_SIZE * 2,
                    GLOBAL_BATCH_SIZE * 4,
                    GLOBAL_BATCH_SIZE * 8,
                ],
                schedule=[
                    Duration.tokens(0),
                    Duration.tokens(167_772_160_000),
                    Duration.tokens(503_316_480_000),
                    Duration.tokens(838_860_800_000),
                ],
            ),
        )
        .with_callback(
            "profiler",
            NvidiaProfilerCallback(
                enabled=False, profile_ranks=[0, 8, 16, 24], start=10, end=13  # NOTE: change this
            ),
        )
        # TODO: might not be able to run in-loop evals depending on parallel strategies
        # .with_recommended_evals(
        #     common.tokenizer, SEQUENCE_LENGTH, cluster, task_set="fast", eval_interval=EVAL_INTERVAL
        # )
    )


def finalize_config(config: ExperimentConfig):
    # config.dataset.mix = 'OLMo-mix-0625' # new dataset mix
    # config.dataset.mix_base_dir = "r2://ai2-llm" # only avail on Google Cloud
    # config.dataset.mix = "OLMoE-mix-0824-dev"

    # This will read stream data from the public endpoints by default, but that might be a lot slower
    # than reading data locally.
    # DATA_ROOT = "http://olmo-data.org"
    # DATA_ROOT = "/jfs/tianhua-tao/ws-olmoe/data"
    # DATA_ROOT = "/jfs/tianhua-tao/ws-olmoe/data"
    DATA_ROOT = "/weka/oe-training-default/ai2-llm"

    DATA_WORK_DIR = "/tmp/dataset-cache"
    # SAVE_ROOT = "/tmp/olmo-core/runs"  # NOTE: change this to what you want

    from olmo_core.data import (
        DataMix,
        NumpyDataLoaderConfig,
        NumpyDatasetConfig,
        NumpyDatasetType,
        TokenizerConfig,
    )

    dataset_config = NumpyDatasetConfig.from_data_mix(
        DataMix.OLMoE_mix_0824,
        tokenizer=TokenizerConfig.dolma2(),
        mix_base_dir=DATA_ROOT,
        sequence_length=SEQUENCE_LENGTH,
        max_target_sequence_length=max(8192, SEQUENCE_LENGTH),
        work_dir=f"{WORK_DIR}/{DATA_WORK_DIR}",
    )

    config.dataset = dataset_config
    config.dataset.mix = "OLMoE-mix-0824-dev"

    config.data_loader.num_workers = 1

    # add active & total params to the wandb name
    total_params_in_B = config.model.num_params / 1000 / 1000 / 1000
    active_params_in_B = config.model.num_active_params / 1000 / 1000 / 1000
    log.info(f"Total params: {total_params_in_B:.2f}B, Active params: {active_params_in_B:.2f}B")

    wandb_cb = cast(WandBCallback, config.trainer.callbacks["wandb"])
    assert isinstance(wandb_cb.name, str), "WandB callback name must be initialized"
    wandb_cb.name += f"_{active_params_in_B:.2f}@{total_params_in_B:.2f}B"
    wandb_cb.name += f"_{TOP_K}K{NUM_EXPERTS}N{NUM_SHARED_EXPERTS}S_{TAG}"


if __name__ == "__main__":
    main(
        global_batch_size=GLOBAL_BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        include_instance_filter=False,  # We use SkipStepOptimizer for this problem.
        include_default_evals=False,
        # intra_document_masking=True,
        finalize_config=finalize_config,
    )
