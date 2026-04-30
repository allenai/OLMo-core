"""
Match 7B TPS
"""

import logging
import math
from functools import partial
from typing import cast

import torch

from olmo_core.config import DType
from olmo_core.data import (
    DataMix,
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.parallel.pipeline_parallel import PipelineScheduleType
from olmo_core.internal.experiment import (
    CommonComponents,
    DataComponents,
    ExperimentConfig,
    build_config,
    main,
)
from olmo_core.nn.attention import SlidingWindowAttentionConfig
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.moe import (
    MoELoadBalancingLossGranularity,
    MoERouterGatingFunction,
)
from olmo_core.nn.moe.v2.block import (
    MoERouterConfigV2,
)
from olmo_core.nn.transformer import (
    MoEFusedV2TransformerConfig,
    TransformerBlockType,
    TransformerConfig,
    TransformerType,
)
from olmo_core.optim import (
    OptimGroupOverride,
    SchedulerUnits,
)
from olmo_core.optim.scheduler import (
    ComposableScheduler,
    ComposableSchedulerMonkeyPatchDecay,
    ComposableSchedulerStage,
    ComposableSchedulerStageType,
)
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    NvidiaProfilerCallback,
    TorchMemoryHistoryCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    MoEV2TransformerTrainModuleConfig,
    TransformerActivationCheckpointingConfig,
    TransformerActivationCheckpointingMode,
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
)
from olmo_core.train.train_module.transformer import TransformerPipelineParallelConfig

log = logging.getLogger(__name__)


def _get_split_points(original_num_layers: int, num_stages: int, minus_last_stage: int):
    assert (
        original_num_layers % num_stages == 0
    ), "Original number of layers must be divisible by number of stages"
    layers_per_stage = original_num_layers // num_stages

    new_num_layers = original_num_layers - minus_last_stage

    split_points = []
    for i in range(1, num_stages):
        split_points.append(i * layers_per_stage)
    # if minus_last_stage > 0:
    #     split_points.append(original_num_layers - minus_last_stage)
    return new_num_layers, split_points


SEQUENCE_LENGTH = 8192

torch.set_float32_matmul_precision("high")


MAX_DURATION = int(6000e9)
EVAL_INTERVAL = 2000
SAVE_INTERVAL = 250

NUM_EXPERTS = 48
TOP_K = 2
D_MODEL = 2560
D_ATTN = 3072

HEAD_DIM = 64
NUM_HEAD = D_ATTN // HEAD_DIM
NUM_KV_HEAD = 4
MOE_HIDDEN_SIZE = 2560
NUM_SHARED_EXPERTS = 1  # Number of shared experts in the shared MLP
SHARED_MLP_HIDDEN_SIZE = (
    2560  # Hidden size for shared MLP (or dense branch MLP in arctic) in MoE blocks
)

EFFECTIVE_MLP = MOE_HIDDEN_SIZE * TOP_K + SHARED_MLP_HIDDEN_SIZE * NUM_SHARED_EXPERTS
MLP_RATIO = EFFECTIVE_MLP / D_MODEL

# the first dense layer MLP
DENSE_LAYER_MLP = TOP_K * MOE_HIDDEN_SIZE + SHARED_MLP_HIDDEN_SIZE * NUM_SHARED_EXPERTS

# DP_DIM=2
EP_DIM = 4
PP_DIM = 1

# ref
REF_NUM_NODES = 8

# stage 1
# MICRO_BSZ = 4
# GLOBAL_BATCH_SIZE_SEQ=(8 * 8) * (32) # start at 16M
# LR_REF_BSZ = 4 * 1024 * 1024

# stage 2
MICRO_BSZ = 4
GLOBAL_BATCH_SIZE_SEQ = (8 * 8) * (64)  # 32M
LR_REF_BSZ = 8 * 1024 * 1024
# EXPERT_LR: 0.5 -> 0.5 * math.sqrt(2)
# fix decay


GLOBAL_BATCH_SIZE = (GLOBAL_BATCH_SIZE_SEQ) * SEQUENCE_LENGTH
NUM_MICRO_BATCHES = GLOBAL_BATCH_SIZE_SEQ // (REF_NUM_NODES * 8) // MICRO_BSZ

GLOBAL_BATCH_TOKENS_IN_M = GLOBAL_BATCH_SIZE // 1024 // 1024

LR = 3e-4
LR = LR * math.sqrt(GLOBAL_BATCH_SIZE / LR_REF_BSZ)  # lr is for X Million token
NUM_LAYERS = 24

if PP_DIM > 1:
    MINUS_LAST_STAGE = 1
    NUM_LAYERS, SPLIT_POINTS = _get_split_points(
        NUM_LAYERS, PP_DIM * 2, minus_last_stage=MINUS_LAST_STAGE
    )
else:
    SPLIT_POINTS = None

############


# SPLIT_POINTS = None
USE_COMPILE = True
USE_NO_SYNC_EP = True
USE_AC = False
PER_LAYER_RECOMPUTE = False
USE_TBO = False
GRAD_ACC_IN_FP32 = True
GRAD_REDUCE_IN_FP32 = True
UNIFORM_ASSIGN = False
RANDOM_ASSIGN = False
USE_ROWWISE_A2A = True
USE_FP8 = False
ROWWISE_A2A_NBLOCKS = 256
SEED = 2026

TAG = "c3"


from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.rope import RoPEConfig, RoPEType
from olmo_core.nn.transformer import TransformerBlockConfig


# from olmo_core.nn.moe.v2.block import LayerNormConfigV2
def build_model_config(common: CommonComponents) -> TransformerConfig:
    from olmo_core.nn.attention.backend import AttentionBackendName
    from olmo_core.nn.moe.v2.block import MoEFusedV2TransformerBlockConfig
    from olmo_core.nn.moe.v2.fp8 import MoERowwiseFP8Config
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
        init_seed=SEED,
        d_model=d_model,
        two_batch_overlap=USE_TBO,
        recompute_each_block=PER_LAYER_RECOMPUTE,
        recompute_all_blocks_by_chunk=False,
        vocab_size=common.tokenizer.padded_vocab_size(),
        n_layers=NUM_LAYERS,
        block=MoEFusedV2TransformerBlockConfig(
            name=TransformerBlockType.moe_fused_v2,
            ep_no_sync=USE_NO_SYNC_EP,
            checkpoint_permute_moe_unpermute=False,
            checkpoint_attn=False,
            checkpoint_second_unpermute=False,
            ep_no_sync_share_combine_out=PER_LAYER_RECOMPUTE,  # if layer-recompute, want to make combine_out shared (not per-layer persistent) to save memory; extra copy overhead applies.
            ep_no_sync_share_dispatch_out=PER_LAYER_RECOMPUTE,  # if layer-recompute, want to make dispatch_out shared (not per-layer persistent) to save memory; extra copy overhead applies.
            ep_no_sync_shared_slots=2 if USE_TBO else 1,
            ep_no_sync_use_rowwise_all_to_all=USE_ROWWISE_A2A,
            ep_no_sync_rowwise_nblocks=ROWWISE_A2A_NBLOCKS,
            ep_no_sync_capacity_factor=1.25,
            rowwise_fp8=MoERowwiseFP8Config(enabled=USE_FP8) if USE_ROWWISE_A2A else None,
            attention=AttentionConfig(
                name=AttentionType.default,
                n_heads=NUM_HEAD,
                n_kv_heads=NUM_KV_HEAD,
                bias=False,
                rope=RoPEConfig(
                    name=RoPEType.default, theta=500_000, scaling=None, full_precision=True
                ),
                qk_norm=layer_norm,
                backend=AttentionBackendName.te,
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
                gating_function=MoERouterGatingFunction.softmax,
                uniform_expert_assignment=UNIFORM_ASSIGN,
                random_expert_assignment=RANDOM_ASSIGN,
                # lb_loss_weight=0.01,
                lb_loss_weight=0.02,
                # lb_loss_weight=0.018,
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
                normalize_expert_weights=1.0,
                restore_weight_scale=True,
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

    # config.lm_head.loss_implementation = LMLossImplementation.fused_linear
    config.lm_head.loss_implementation = LMLossImplementation.default
    WINDOW_SIZE = 512
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
                n_heads=NUM_HEAD,
                n_kv_heads=NUM_KV_HEAD,
                bias=False,
                rope=RoPEConfig(
                    name=RoPEType.default, theta=500_000, scaling=None, full_precision=True
                ),
                qk_norm=layer_norm,
                backend=AttentionBackendName.te,
                use_head_qk_norm=True,
                dtype=dtype,
                d_attn=D_ATTN,
            ),
            feed_forward_moe=None,
            feed_forward=FeedForwardConfig(
                hidden_size=(DENSE_LAYER_MLP), bias=False
            ),  # dense mlp is twice as fast as moe mlp
            attention_norm=layer_norm,
            feed_forward_norm=layer_norm,
        )
    }
    # flops = config.num_flops_per_token(4096)
    return config


# EXPERT_LR = LR * math.sqrt(TOP_K / NUM_EXPERTS)  # scale lr for expert params, # 1/4.8989 = 0.204
# EXPERT_LR = LR * 0.5  # scale lr for expert params, empirical choice
EXPERT_LR = LR * 0.5 * math.sqrt(2)
SCHED_WARMUP_TOKENS = int((40e9 // GLOBAL_BATCH_SIZE) * GLOBAL_BATCH_SIZE)
# SCHED_FAST_DECAY_TOKENS = int((35e9 // GLOBAL_BATCH_SIZE) * GLOBAL_BATCH_SIZE)
SCHED_LONG_DECAY_TOKENS = int((5960e9 // GLOBAL_BATCH_SIZE) * GLOBAL_BATCH_SIZE)
SCHED_MID_FRACTION = 0.8
SCHED_FINAL_FRACTION = 0.1

# patch
MONKEY_PATCH_DECAY_START_TOKENS = None
MONKEY_PATCH_DECAY_DURATION_TOKENS = int((200e9 // GLOBAL_BATCH_SIZE) * GLOBAL_BATCH_SIZE)
MONKEY_PATCH_DECAY_END_FRACTION = SCHED_FINAL_FRACTION
MONKEY_PATCH_DECAY_SHAPE = ComposableSchedulerStageType.cosine


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
                # OptimGroupOverride(params=["embeddings.weight", "*_norm.weight"], opts=dict(weight_decay=0.0)), # WRONG
                OptimGroupOverride(
                    params=["*embeddings.weight", "*norm.weight"], opts=dict(weight_decay=0.0)
                ),
                # OptimGroupOverride(params=["*w_up_gate", "*w_down","*routed_experts_router*"], opts=dict(lr=EXPERT_LR)),
                OptimGroupOverride(
                    params=[
                        "*w_up_gate",
                        "*w_down",
                    ],
                    opts=dict(lr=EXPERT_LR),
                ),
                # OptimGroupOverride(params=["*routed_experts_router*"], opts=dict(lr=ROUTER_LR)),
            ],
            compile=USE_COMPILE,
            dtype=DType.float32,
            sigma_factor=12,
            use_distributed=True,
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
            reduce_grads_in_fp32=GRAD_REDUCE_IN_FP32,
            accumulate_grads_in_fp32=GRAD_ACC_IN_FP32,
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
        float8_config=None,
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=ComposableScheduler(
            units=SchedulerUnits.tokens,
            stages=[
                ComposableSchedulerStage(
                    duration=SCHED_WARMUP_TOKENS,
                    shape=ComposableSchedulerStageType.linear,
                    start_lr_fraction=0.0,
                    end_lr_fraction=1.0,
                ),
                # ComposableSchedulerStage(
                #     duration=SCHED_FAST_DECAY_TOKENS,
                #     shape=ComposableSchedulerStageType.cosine,
                #     end_lr_fraction=SCHED_MID_FRACTION,
                # ),
                ComposableSchedulerStage(
                    duration=SCHED_LONG_DECAY_TOKENS,
                    shape=ComposableSchedulerStageType.cosine,
                    end_lr_fraction=SCHED_FINAL_FRACTION,
                ),
            ],
            monkey_patch_decay=(
                ComposableSchedulerMonkeyPatchDecay(
                    start=MONKEY_PATCH_DECAY_START_TOKENS,
                    duration=MONKEY_PATCH_DECAY_DURATION_TOKENS,
                    shape=MONKEY_PATCH_DECAY_SHAPE,
                    end_lr_fraction=MONKEY_PATCH_DECAY_END_FRACTION,
                )
                if MONKEY_PATCH_DECAY_START_TOKENS is not None
                else None
            ),
        ),
    )


WORK_DIR = "/workspace"


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    cancel_check_interval = 10

    cluster = "ai2/jupiter"
    # cluster = 'cirrascale'
    from olmo_core.train.checkpoint import CheckpointerConfig

    return (
        TrainerConfig(
            save_folder=f"{WORK_DIR}/checkpoint/{common.run_name}_{D_MODEL}d{D_ATTN}a_{NUM_LAYERS}L{MOE_HIDDEN_SIZE}M{SHARED_MLP_HIDDEN_SIZE}S_{NUM_EXPERTS}E{TOP_K}K{NUM_SHARED_EXPERTS}S_{TAG}",
            # save_folder=f'{common.save_folder}/{common.run_name}_{D_MODEL}d{D_ATTN}a_{NUM_LAYERS}L{MOE_HIDDEN_SIZE}M{SHARED_MLP_HIDDEN_SIZE}S_{NUM_EXPERTS}E{TOP_K}K{NUM_SHARED_EXPERTS}S_{TAG}',
            save_overwrite=True,
            checkpointer=CheckpointerConfig(
                save_thread_count=3, load_thread_count=2, throttle_uploads=True
            ),
            metrics_collect_interval=5,
            cancel_check_interval=cancel_check_interval,
            max_duration=Duration.tokens(MAX_DURATION),
            # steps_to_skip=[StepSkipRange(start=41312, stop=41329)]
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=SAVE_INTERVAL,
                ephemeral_save_interval=None,
                save_async=False,
                pre_train_checkpoint=True,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=common.run_name,
                entity="ai2-llm",
                project="olmoe-dev-v2",
                enabled=True,
                cancel_check_interval=cancel_check_interval,
            ),
        )
        .with_callback(
            "profiler",
            NvidiaProfilerCallback(
                enabled=False,  # NOTE: change this
                profile_ranks=list(range(0, 8 * 8, 8)),
                start=41,
                end=45,
            ),
        )
        .with_callback(
            "torch_mem_history",
            TorchMemoryHistoryCallback(
                enabled=False,  # NOTE: change this
                profile_ranks=list(range(0, 8 * 128, 8)),
                start=59161,
                end=59164,
                output_dir="/workspace/tmp",
            ),
        )
        # TODO: might not be able to run in-loop evals depending on parallel strategies
        # .with_recommended_evals(
        #     common.tokenizer, SEQUENCE_LENGTH, cluster, task_set="fast", eval_interval=EVAL_INTERVAL
        # )
    )


def finalize_config(config: ExperimentConfig):
    # add active & total params to the wandb name
    total_params_in_B = config.model.num_params / 1000 / 1000 / 1000
    active_params_in_B = config.model.num_active_params / 1000 / 1000 / 1000
    log.info(f"Total params: {total_params_in_B:.2f}B, Active params: {active_params_in_B:.2f}B")

    wandb_cb = cast(WandBCallback, config.trainer.callbacks["wandb"])
    wandb_original_name = wandb_cb.name
    assert isinstance(wandb_cb.name, str), "WandB callback name must be initialized"
    wandb_cb.name += f"_{active_params_in_B:.2f}@{total_params_in_B:.2f}B"
    wandb_cb.name += (
        f"_{NUM_LAYERS}L{TOP_K}K{NUM_EXPERTS}N{NUM_SHARED_EXPERTS}S_{EP_DIM}EP{PP_DIM}PP_{TAG}"
    )
    wandb_cb.group = f"{wandb_original_name}_{active_params_in_B:.2f}@{total_params_in_B:.2f}B_{NUM_LAYERS}L{TOP_K}K{NUM_EXPERTS}N{NUM_SHARED_EXPERTS}S_{TAG}"


def build_data_components(
    common: CommonComponents,
    intra_document_masking: bool = False,
    include_instance_filter: bool = True,
) -> DataComponents:
    # DATA_ROOT = "/weka/oe-training-default/ai2-llm"
    # DATA_WORK_DIR = "/tmp/dataset-cache"

    dataset_config = NumpyFSLDatasetConfig.from_data_mix(
        DataMix.OLMo_mix_0925,
        # DataMix.OLMo_mix_0625,
        # DataMix.OLMoE_mix_0824_dev,
        tokenizer=common.tokenizer,
        # mix_base_dir=common.root_dir,
        mix_base_dir="s3://ai2-llm",
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
        global_batch_size=common.global_batch_size, seed=34521, num_workers=8
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
