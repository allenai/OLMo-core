"""
Train an OLMoE model. Run this script without any arguments to see usage info.
"""

import logging
import math
import torch
import transformer_engine
from functools import partial

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.parallel.pipeline_parallel import PipelineScheduleType
from olmo_core.float8 import AOFloat8LinearConfig, Float8Config
from olmo_core.internal.experiment import CommonComponents, main, ExperimentConfig
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
from olmo_core.nn.moe.v2.block import SharedExpertsConfig, RoutedExpertsConfig, MoERouterConfigV2
from typing import cast
from olmo_core.train.callbacks import WandBCallback


from olmo_core.internal.experiment import (
    CommonComponents,
    DataComponents,
    build_config,
    main,
)
from olmo_core.data import (
    DataMix,
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
)
from olmo_core.nn.transformer import (
    TransformerBlockType,
    TransformerConfig,
    TransformerType,
    MoEFusedV2TransformerConfig,
)
from olmo_core.optim import WSD, OptimGroupOverride, SchedulerUnits, SkipStepAdamWConfig, AdamWConfig, CosWithWarmup
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import (
    BatchSizeSchedulerCallback,
    CheckpointerCallback,
    CometCallback,
    WandBCallback,
    NvidiaProfilerCallback
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
    TransformerActivationCheckpointingConfig,
    TransformerActivationCheckpointingMode,
    TransformerExpertParallelConfig,
    MoEV2TransformerTrainModuleConfig
)
from olmo_core.train.train_module.transformer import TransformerPipelineParallelConfig
from dataclasses import replace
from olmo_core.train.train_module.transformer.moe_train_module import MoEV2TransformerTrainModule

log = logging.getLogger(__name__)

def _get_split_points(original_num_layers: int, num_stages: int, minus_last_stage: int):
    assert original_num_layers % num_stages == 0, "Original number of layers must be divisible by number of stages"
    layers_per_stage = original_num_layers // num_stages

    new_num_layers = original_num_layers - minus_last_stage

    split_points = []
    for i in range(1, num_stages):
        split_points.append(i * layers_per_stage)
    # if minus_last_stage > 0:
    #     split_points.append(original_num_layers - minus_last_stage)
    return new_num_layers, split_points


SEQUENCE_LENGTH = 8192

# GLOBAL_BATCH_SIZE_SEQ=1024 + 512
GLOBAL_BATCH_SIZE_SEQ=512
GLOBAL_BATCH_SIZE = (
    (GLOBAL_BATCH_SIZE_SEQ) * SEQUENCE_LENGTH
)  

GLOBAL_BATCH_TOKENS_IN_M = SEQUENCE_LENGTH * GLOBAL_BATCH_SIZE_SEQ // 1024 // 1024

MAX_DURATION = int(1000e9)  # int(6e12), don't forget to adjust the LR when you increase this
EVAL_INTERVAL = 50
LR= 3e-4

NUM_EXPERTS = 64
TOP_K = 4
# D_MODEL=3072
# D_ATTN=3072
D_MODEL=3072
D_ATTN=D_MODEL
HEAD_DIM=128
NUM_HEAD = D_ATTN // HEAD_DIM
NUM_KV_HEAD=4
MOE_HIDDEN_SIZE = 2560
NUM_SHARED_EXPERTS = 1  # Number of shared experts in the shared MLP
SHARED_MLP_HIDDEN_SIZE = 2560  # Hidden size for shared MLP (or dense branch MLP in arctic) in MoE blocks

EFFECTIVE_MLP = (MOE_HIDDEN_SIZE * TOP_K + SHARED_MLP_HIDDEN_SIZE * NUM_SHARED_EXPERTS)
MLP_RATIO = EFFECTIVE_MLP / D_MODEL

# the first dense layer MLP
DENSE_LAYER_MLP = (TOP_K * MOE_HIDDEN_SIZE + SHARED_MLP_HIDDEN_SIZE * NUM_SHARED_EXPERTS) * 1

MICRO_BSZ = 1
# DP_DIM=2
EP_DIM=8
PP_DIM=4


NUM_LAYERS= 32

if PP_DIM > 1:
    MINUS_LAST_STAGE=1
    NUM_LAYERS, SPLIT_POINTS = _get_split_points(NUM_LAYERS, PP_DIM * 2, minus_last_stage=MINUS_LAST_STAGE)
else:
    SPLIT_POINTS = None

############


# SPLIT_POINTS = None
USE_COMPILE=True
USE_AC=False
USE_TBO=False
GRAD_ACC_IN_FP32=False
UNIFORM_ASSIGN=False
RANDOM_ASSIGN=True

SEED = 2026

TAG=f'dev-S{SEED}'

if UNIFORM_ASSIGN:
    TAG = 'U-' + TAG
elif RANDOM_ASSIGN:
    TAG = 'RA-' + TAG
else:
    TAG = 'R-' + TAG
if GRAD_ACC_IN_FP32:
    TAG += '-fp32acc'



from olmo_core.nn.lm_head import LMHeadConfig, LMHeadType
from olmo_core.nn.rope import RoPEConfig, RoPEScalingConfig, RoPEType
from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.layer_norm import LayerNormType, LayerNormConfig
from olmo_core.nn.transformer import TransformerBlockConfig

# from olmo_core.nn.moe.v2.block import LayerNormConfigV2
def build_model_config(common: CommonComponents) -> TransformerConfig:
    from olmo_core.nn.moe.v2.block import (
        MoEFusedV2TransformerBlockConfig
    )
    from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig
    from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
    
    d_model = D_MODEL
    dtype = DType.float32
    
    layer_norm = LayerNormConfig(
        name=LayerNormType.rms,
        eps=1e-6,
        bias=False,
        dtype=dtype,
    )
    config = MoEFusedV2TransformerConfig(
        init_seed=SEED,
        d_model=d_model,
        two_batch_overlap=USE_TBO,
        recompute_each_block=False,
        recompute_all_blocks_by_chunk=False,
        vocab_size=common.tokenizer.padded_vocab_size(),
        n_layers=NUM_LAYERS,
        block=MoEFusedV2TransformerBlockConfig(
            name=TransformerBlockType.moe_fused_v2,
            checkpoint_permute_moe_unpermute=True,
            checkpoint_attn=True,
            attention=AttentionConfig(
                name=AttentionType.default,
                n_heads=NUM_HEAD,
                n_kv_heads=NUM_KV_HEAD,
                bias=False,
                rope=RoPEConfig(name=RoPEType.default, theta=500_000, scaling=None, full_precision=True),
                qk_norm=layer_norm ,
                use_flash=True,
                use_head_qk_norm=True,
                dtype=dtype,
                d_attn=D_ATTN,
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
                uniform_expert_assignment=UNIFORM_ASSIGN,
                random_expert_assignment=RANDOM_ASSIGN,
                lb_loss_weight=0.005,
                z_loss_weight=None,
                lb_loss_granularity=MoELoadBalancingLossGranularity.instance,
                dtype=dtype,
                normalize_expert_weights=1.0,
                restore_weight_scale=True,
            ),
            shared_experts=SharedExpertsConfig(
                d_model=d_model,
                hidden_size=SHARED_MLP_HIDDEN_SIZE,
                num_experts=NUM_SHARED_EXPERTS,
                bias=False,
                dtype=dtype
            ) if NUM_SHARED_EXPERTS > 0 else None,
            shared_experts_router=MoERouterConfigV2(
                d_model=d_model,
                num_experts=NUM_SHARED_EXPERTS,
                top_k=NUM_SHARED_EXPERTS, # all experts are used
                gating_function=MoERouterGatingFunction.sigmoid,
                uniform_expert_assignment=False,
                lb_loss_weight=None,
                z_loss_weight=None,
                lb_loss_granularity=MoELoadBalancingLossGranularity.instance,
                dtype=dtype,
                normalize_expert_weights=1.0,
                restore_weight_scale=True,
            ) if NUM_SHARED_EXPERTS > 1 else None, # only need router if > 1 expert
            feed_forward_norm=layer_norm,
        ),
        lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=dtype),
        name=TransformerType.moe_fused_v2,

        init_std=0.01,
        dtype=dtype
    )
    
    # config.lm_head.loss_implementation = LMLossImplementation.fused_linear
    config.lm_head.loss_implementation = LMLossImplementation.cut_cross_entropy
    WINDOW_SIZE=4096
    config.block.attention.sliding_window = SlidingWindowAttentionConfig(
        force_full_attention_on_first_layer=False,
        force_full_attention_on_last_layer=True,
        pattern=[WINDOW_SIZE, WINDOW_SIZE, WINDOW_SIZE, -1]
    )
    # config.block.attention.use_flash = True
    # config.block.attention.use_head_qk_norm = True
    from olmo_core.nn.attention.backend import AttentionBackendName
    # First block will be a regular transformer block (no MoE component).
    config.block_overrides = {
        0: TransformerBlockConfig(
                name=TransformerBlockType.reordered_norm,
                attention=AttentionConfig(
                    name=AttentionType.default,
                    n_heads=NUM_HEAD,
                    n_kv_heads=NUM_KV_HEAD,
                    bias=False,
                    rope=RoPEConfig(name=RoPEType.default, theta=500_000, scaling=None, full_precision=True),
                    qk_norm=layer_norm ,
                    # use_flash=True,
                    backend=AttentionBackendName.flash_3,
                    use_head_qk_norm=True,
                    dtype=dtype,
                    d_attn=D_ATTN,
                ),
                feed_forward_moe=None,
                feed_forward=FeedForwardConfig(hidden_size=( DENSE_LAYER_MLP ), bias=False), # dense mlp is twice as fast as moe mlp
                attention_norm=layer_norm,
                feed_forward_norm=layer_norm,
            ) 
    }
    # flops = config.num_flops_per_token(4096)
    return config



def build_train_module_config(common: CommonComponents) -> MoEV2TransformerTrainModuleConfig:
    from olmo_core.optim.moe_optimizer import MoEFusedV2OptimizerConfig
    return MoEV2TransformerTrainModuleConfig(
        rank_microbatch_size=MICRO_BSZ * SEQUENCE_LENGTH,
        max_sequence_length=common.max_sequence_length,
        optim=MoEFusedV2OptimizerConfig(
            lr=LR,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight", "*_norm.weight"], opts=dict(weight_decay=0.0)),
                # OptimGroupOverride(params=["*w_up_gate"], opts=dict(weight_decay=0.1)) # HACK: just to make a separate group to avoid OOM in RS
                # OptimGroupOverride(params=["embeddings.weight", ], opts=dict(weight_decay=0.0)) #TODO: fix
            ],
            #TODO: weight decay for norm?
            # fused=True,
            compile=False,
            dtype=DType.float32,
            # foreach=True
        ),
        grad_accum_in_fp32=GRAD_ACC_IN_FP32,
        compile_model=USE_COMPILE,
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.full,
        ) if USE_AC else None,
        # FSDP
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.ddp,
            param_dtype=DType.bfloat16,  # TODO: not used?
            reduce_dtype=DType.float32, # TODO: not used?
            shard_degree=None,
        ),
        ep_config=TransformerExpertParallelConfig(degree=EP_DIM) if EP_DIM != 1 else None, # EP=1 means no expert parallel
        pp_config=TransformerPipelineParallelConfig(
            degree=PP_DIM,
            # schedule=PipelineScheduleType.custom_1F1B,
            schedule=PipelineScheduleType.custom_interleaved_1F1B,
            use_custom_stage_implementation=True,  # use custom stage implementation that re-uses receive buffers across micro-batches
            split_points=SPLIT_POINTS
        ) if PP_DIM > 1 else None,
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
        # scheduler=WSD(
        #     units=SchedulerUnits.steps,
        #     warmup=500,
        #     # NOTE: be aware of when decay will happen relative to batch_wup schedule
        #     decay=(int(50e9 / GLOBAL_BATCH_SIZE)),
        #     decay_fraction=None,
        # ),
        scheduler=CosWithWarmup(warmup_steps=2000),
    )

# WORK_DIR = "/jfs/tianhua-tao/ws-olmoe"
WORK_DIR = "/weka/oe-training-default/tianhua/ws-megatron"

def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    cancel_check_interval = 10
    
    cluster = 'ai2/augusta'
    # cluster = 'cirrascale'

    return (
        TrainerConfig(
            # load_path='/weka/oe-training-default/tianhua/ws-megatron/tmp/OLMoE3-dev-baseline_2048d_8L1024M2048S_16E6K_U-fsdp-old-dbg/step0',
            # save_folder=f'{WORK_DIR}/tmp/{common.run_name}_{D_MODEL}d{D_ATTN}a_{NUM_LAYERS}L{MOE_HIDDEN_SIZE}M{SHARED_MLP_HIDDEN_SIZE}S_{NUM_EXPERTS}E{TOP_K}K{NUM_SHARED_EXPERTS}S_{TAG}',
            save_folder=f'{common.save_folder}/{common.run_name}_{D_MODEL}d{D_ATTN}a_{NUM_LAYERS}L{MOE_HIDDEN_SIZE}M{SHARED_MLP_HIDDEN_SIZE}S_{NUM_EXPERTS}E{TOP_K}K{NUM_SHARED_EXPERTS}S_{TAG}',
            save_overwrite=True,
            metrics_collect_interval=5,
            cancel_check_interval=cancel_check_interval,
            max_duration=Duration.tokens(MAX_DURATION),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=200,
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
                entity="ai2-llm",
                project="olmoe-dev-v2",
                # project="olmo3",
                enabled=True,
                cancel_check_interval=cancel_check_interval,
            ),
        )
        # .with_callback(
        #     "batchwup",
        #     BatchSizeSchedulerCallback(
        #         batch_sizes=[GLOBAL_BATCH_SIZE, GLOBAL_BATCH_SIZE * 2, GLOBAL_BATCH_SIZE * 4, GLOBAL_BATCH_SIZE * 8, ],
        #         schedule=[
        #             Duration.tokens(0),
        #             Duration.tokens(167_772_160_000),
        #             Duration.tokens(503_316_480_000),
        #             Duration.tokens(838_860_800_000),
        #         ],
        #     ),
        # )
        .with_callback(
            "profiler", 
            NvidiaProfilerCallback(enabled=True, # NOTE: change this
                                   profile_ranks=[0, 8, 16, 24, 32, 40, 48, 56],
                                   start=11,
                                   end=14
            )
        )
        # TODO: might not be able to run in-loop evals depending on parallel strategies
        # .with_recommended_evals(
        #     # common.tokenizer, SEQUENCE_LENGTH, cluster, task_set="fast", eval_interval=EVAL_INTERVAL
        #     common.tokenizer, SEQUENCE_LENGTH, cluster, task_set="fast", eval_interval=EVAL_INTERVAL
        # )
    )


def finalize_config(config: ExperimentConfig):
    # add active & total params to the wandb name
    total_params_in_B = config.model.num_params/1000/1000/1000
    active_params_in_B = config.model.num_active_params/1000/1000/1000
    log.info(f"Total params: {total_params_in_B:.2f}B, Active params: {active_params_in_B:.2f}B")

    wandb_cb = cast(WandBCallback, config.trainer.callbacks['wandb'])
    assert isinstance(wandb_cb.name, str), "WandB callback name must be initialized"
    wandb_cb.name += f"_{active_params_in_B:.2f}@{total_params_in_B:.2f}B"
    wandb_cb.name += f"_{NUM_LAYERS}L{TOP_K}K{NUM_EXPERTS}N{NUM_SHARED_EXPERTS}S_{EP_DIM}EP{PP_DIM}PP_{TAG}"
    wandb_cb.group = f"_{active_params_in_B:.2f}@{total_params_in_B:.2f}B_{NUM_LAYERS}L{TOP_K}K{NUM_EXPERTS}N{NUM_SHARED_EXPERTS}S_{TAG}"



def build_data_components(
    common: CommonComponents,
    intra_document_masking: bool = False,
    include_instance_filter: bool = True,
) -> DataComponents:
    # DATA_ROOT = "/weka/oe-training-default/ai2-llm"
    # DATA_WORK_DIR = "/tmp/dataset-cache"

    dataset_config = NumpyFSLDatasetConfig.from_data_mix(
        DataMix.OLMo_mix_0925,
        # DataMix.OLMoE_mix_0824_dev,
        tokenizer=common.tokenizer,
        mix_base_dir=common.root_dir,
        work_dir=common.work_dir,
        sequence_length=common.max_sequence_length,
        max_target_sequence_length=max(common.max_sequence_length, 8192),
        generate_doc_lengths=intra_document_masking,
        instance_filter_config=None
        if not include_instance_filter
        else InstanceFilterConfig(
            repetition_max_period=13, repetition_min_period=1, repetition_max_count=32
        ),
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=common.global_batch_size, seed=34521, num_workers=2
    )

    return DataComponents(dataset=dataset_config, data_loader=data_loader_config)


if __name__ == "__main__":
    config_builder = partial(
        build_config,
        global_batch_size=GLOBAL_BATCH_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        data_config_builder=build_data_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        flight_recorder=True,
        include_instance_filter=True,  # We use SkipStepOptimizer for this problem.
        include_default_evals=False,
        finalize_config=finalize_config,
    )
        
    main(
        config_builder=config_builder,
    )
