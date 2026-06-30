
"""
Long-context continuation for the OLMoE3-dev-260614-s002 MoE DDP checkpoint.

The model and OLMoDDP/EP shape mirrors
``src/scripts/train/OLMoE3-dev-260614-s002-midtrain.py`` so it can continue
from the selected mid-training checkpoint. The long-context phase follows the
official OLMo3 32B long-context recipe for token count, data, YaRN RoPE scaling,
batch size, and LR schedule.

Example:
  PYTHONPATH=src torchrun --standalone --nproc-per-node=8 \
      src/scripts/train/OLMoE3-dev-260614-s002-long-context.py train \
      OLMoE3-dev-260614-s002-long-context local
"""

from copy import deepcopy
from functools import partial
import logging
import math
import os
import socket
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional, cast


def _default_triton_cache_dir() -> None:
    if os.environ.get("TRITON_CACHE_DIR") or os.environ.get("OLMO_DISABLE_PER_RANK_TRITON_CACHE"):
        return
    local_rank = (
        os.environ.get("LOCAL_RANK")
        or os.environ.get("SLURM_LOCALID")
        or os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
        or os.environ.get("MV2_COMM_WORLD_LOCAL_RANK")
        or "0"
    )
    job_id = (
        os.environ.get("BEAKER_EXPERIMENT_ID")
        or os.environ.get("SLURM_JOB_ID")
        or os.environ.get("JOB_ID")
        or "default"
    )
    host = socket.gethostname().split(".")[0] or "host"
    base = Path(os.environ.get("OLMO_TRITON_CACHE_BASE", "/tmp/olmo-triton-cache"))
    cache_dir = base / str(job_id) / host / f"local_rank_{local_rank}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)


_default_triton_cache_dir()

# Keep this before any olmo_core imports: several modules import nvtx at import
# time, and NVTX_DISABLE only works if it is set before nvtx is imported.
USE_NV_PROFILE = True
if not USE_NV_PROFILE:
    os.environ["NVTX_DISABLE"] = "1"
os.environ.setdefault("NO_GCE_CHECK", "true")
os.environ.setdefault("OLMO_DATA_PREP_WORKERS", "8")
# The LC production run immediately loads the mid-training checkpoint. Avoid the
# expensive random-init DDP parameter broadcast before checkpoint load.
os.environ.setdefault("OLMO_DDP_INIT_SYNC", "0")
# All LC MoE blocks use no-sync rowwise EP, which uses the regular DeviceMesh EP
# group for symmetric memory. Avoid creating an unused high-priority EP-MP NCCL
# subgroup during startup.
os.environ.setdefault("OLMO_EP_MP_HIGH_PRIORITY_GROUP", "0")

import torch

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.parallel.pipeline_parallel import PipelineScheduleType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.internal.experiment import (
    CliContext,
    CommonComponents,
    DataComponents,
    ExperimentConfig,
    SubCmd,
    build_config,
    build_common_components as build_default_common_components,
    main,
)
from olmo_core.data import (
    DataMix,
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyPackedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.nn.attention import AttentionConfig, AttentionType, SlidingWindowAttentionConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig, LMLossImplementation
from olmo_core.nn.moe import (
    MoELoadBalancingLossGranularity,
    MoERouterGatingFunction,
)
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig
from olmo_core.nn.rope import RoPEConfig, RoPEType, YaRNRoPEScalingConfig
from olmo_core.nn.transformer import (
    OLMoDDPModelConfig,
    TransformerBlockType,
    TransformerType,
)
from olmo_core.optim import (
    LinearWithWarmup,
    OLMoDDPOptimizerConfig,
    OptimGroupOverride,
    SchedulerUnits,
)
from olmo_core.train import Duration, LoadStrategy, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    NvidiaProfilerCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    OLMoDDPTrainModuleConfig,
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
)
from olmo_core.train.train_module.transformer import TransformerPipelineParallelConfig

log = logging.getLogger(__name__)

def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


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


SEQUENCE_LENGTH = 65536
LOCAL_WORK_DIR = (
    "/weka/oe-training-default/ai2-llm/checkpoints/"
    "OLMoE3-dev-260614-s002-long-context/dataset-cache"
)
LOCAL_SAVE_ROOT = "/workspace/checkpoint"
WORK_DIR_OVERRIDE_ENV = "OLMO_LC_WORK_DIR"
SAVE_ROOT_ENV = "OLMO_LC_SAVE_ROOT"
MIX_BASE_DIR = os.environ.get("OLMOE3_LC_MIX_BASE_DIR", "/workspace/data")
NUM_NODES = 32
LOAD_PATH_ENV = "OLMOE3_LC_LOAD_PATH"
DEFAULT_LOAD_PATH = "/workspace/checkpoint/OLMoE3-dev-260614-s002-midtrain/step23842"
LOAD_PATH: Optional[str] = os.environ.get(LOAD_PATH_ENV, DEFAULT_LOAD_PATH) or None
SAVE_ROOT = os.environ.get(SAVE_ROOT_ENV, LOCAL_SAVE_ROOT)
LAUNCH_ENV_VARS_TO_FORWARD = (
    LOAD_PATH_ENV,
    SAVE_ROOT_ENV,
    "OLMOE3_LC_MIX_BASE_DIR",
    WORK_DIR_OVERRIDE_ENV,
    "OLMOE3_LC_USE_COMPILE",
    "OLMOE3_LC_PER_LAYER_RECOMPUTE",
)
LC_MAX_TOKENS = 100_000_000_000
LC_GLOBAL_BATCH_SIZE = 8 * 1024 * 1024
LC_RANK_MICROBATCH_SEQUENCES = 1
LC_LR = 0.0002071235285
LOAD_OPTIM_STATE = True
RESET_OPTIMIZER_STATES_ON_LOAD = True
RESET_OPTIMIZER_STATES_ON_RESUME = False
INTRA_DOCUMENT_MASKING = True
INCLUDE_INSTANCE_FILTER = True

torch.set_float32_matmul_precision('high')

IN_EVAL_MODE = len(sys.argv) > 1 and sys.argv[1] == "eval_checkpoints"


EVAL_INTERVAL = 2000
SAVE_INTERVAL = 1000

NUM_EXPERTS = 64
TOP_K = 4
ORIGINAL_TOP_K=None
D_MODEL=2048
D_ATTN=4 * 1024

HEAD_DIM=128
NUM_HEAD = D_ATTN // HEAD_DIM
NUM_KV_HEAD= NUM_HEAD // 4
MOE_HIDDEN_SIZE = 2048 + 512
NUM_SHARED_EXPERTS = 1  # Number of shared experts in the shared MLP
SHARED_MLP_HIDDEN_SIZE = 2048 + 512   # Hidden size for shared MLP (or dense branch MLP in arctic) in MoE blocks

EFFECTIVE_MLP = (MOE_HIDDEN_SIZE * TOP_K + SHARED_MLP_HIDDEN_SIZE * NUM_SHARED_EXPERTS)
MLP_RATIO = EFFECTIVE_MLP / D_MODEL

# the first dense layer MLP
DENSE_LAYER_MLP = (TOP_K * MOE_HIDDEN_SIZE + SHARED_MLP_HIDDEN_SIZE * NUM_SHARED_EXPERTS)

# DP_DIM=2
EP_DIM=8
PP_DIM=1

TAG=f'p1_lc64k'

USE_FP8=False

NUM_LAYERS=31

if PP_DIM > 1:
    MINUS_LAST_STAGE=1
    NUM_LAYERS, SPLIT_POINTS = _get_split_points(NUM_LAYERS, PP_DIM * 2, minus_last_stage=MINUS_LAST_STAGE)
else:
    SPLIT_POINTS = None


if IN_EVAL_MODE:
    LC_RANK_MICROBATCH_SEQUENCES=4
    EP_DIM=1
    PP_DIM=1
    NUM_LAYERS=31
USE_COMPILE=_env_bool("OLMOE3_LC_USE_COMPILE", True)
USE_NO_SYNC_EP=True
PER_LAYER_RECOMPUTE=_env_bool("OLMOE3_LC_PER_LAYER_RECOMPUTE", True)
USE_TBO=False
GRAD_ACC_IN_FP32=True
GRAD_REDUCE_IN_FP32=True
UNIFORM_ASSIGN=False
RANDOM_ASSIGN=False
USE_ROWWISE_A2A=True
USE_FP8_ATTN_QKV=USE_FP8
USE_FP8_ATTN_OUT=USE_FP8
USE_FP8_ATTN_SAVE_QKV=False
ROWWISE_A2A_NBLOCKS=128 if EP_DIM <=8 else 64 # for intra-node, can use more blocks to increase overlap; for inter-node, the bottleneck is the network, so fewer blocks can reduce overhead.
SEED = 2026
USE_MUON = False
USE_PERI_NORM = True
WANDB_ENABLED = True
EP_NO_SYNC_CAPACITY_FACTOR = 1.3125
EARLY_MOE_EP_NO_SYNC_CAPACITY_FACTOR = 1.3125
EARLY_MOE_CAPACITY_BLOCK_INDICES = (1, 2, 3)  # block 0 is dense; these are the first two MoE blocks.
FIRST_MOE_ROUTER_Z_LOSS_WEIGHT = 2e-6
ROPE_SCALING = YaRNRoPEScalingConfig(
    factor=8,
    beta_fast=32,
    beta_slow=1,
    old_context_len=8192,
)

def build_model_config(common: CommonComponents) -> OLMoDDPModelConfig:
    from olmo_core.nn.ddp.block import OLMoDDPTransformerBlockConfig
    from olmo_core.nn.moe.v2.ep_config import (
        ExpertParallelConfig,
        ExpertParallelPath,
        ExpertParallelSchedule,
    )
    from olmo_core.nn.moe.v2.fp8 import MoERowwiseFP8Config
    from olmo_core.nn.attention.backend import AttentionBackendName

    d_model = D_MODEL
    dtype = DType.float32

    layer_norm = LayerNormConfig(
        name=LayerNormType.rms,
        eps=1e-6,
        bias=False,
        dtype=dtype,
    )
    use_block_no_sync_ep = USE_NO_SYNC_EP and EP_DIM > 1
    block_ep_path = (
        ExpertParallelPath.rowwise_nvshmem
        if use_block_no_sync_ep and USE_ROWWISE_A2A
        else ExpertParallelPath.no_sync_1d
        if use_block_no_sync_ep
        else ExpertParallelPath.sync_1d
    )
    block_ep_schedule = (
        ExpertParallelSchedule.tbo
        if USE_TBO and block_ep_path == ExpertParallelPath.rowwise_nvshmem
        else ExpertParallelSchedule.normal
    )
    config = OLMoDDPModelConfig(
        init_seed=SEED,
        d_model=d_model,
        two_batch_overlap=USE_TBO,
        recompute_each_block=PER_LAYER_RECOMPUTE,
        recompute_all_blocks_by_chunk=False,
        # recompute_block_keys=["0", "1", "2"], # recompute dense layer
        vocab_size=common.tokenizer.padded_vocab_size(),
        n_layers=NUM_LAYERS,
        embed_scale=math.sqrt(d_model),
        embedding_norm=layer_norm,
        block=OLMoDDPTransformerBlockConfig(
            name=TransformerBlockType.moe_fused_v2,
            use_peri_norm=USE_PERI_NORM,
            ep=ExpertParallelConfig(
                path=block_ep_path,
                schedule=block_ep_schedule,
                share_combine_out=PER_LAYER_RECOMPUTE, # if layer-recompute, want to make combine_out shared (not per-layer persistent) to save memory; extra copy overhead applies.
                share_dispatch_out=PER_LAYER_RECOMPUTE, # if layer-recompute, want to make dispatch_out shared (not per-layer persistent) to save memory; extra copy overhead applies.
                shared_slots=2 if block_ep_schedule == ExpertParallelSchedule.tbo else 1,
                rowwise_nblocks=ROWWISE_A2A_NBLOCKS,
                capacity_factor=EP_NO_SYNC_CAPACITY_FACTOR,
            ),
            checkpoint_permute_moe_unpermute=False,
            checkpoint_attn=False,
            checkpoint_second_unpermute=False,
            rowwise_fp8=MoERowwiseFP8Config(enabled=USE_FP8, fused_autograd_recompute_swiglu=False) if USE_ROWWISE_A2A else None,
            attention=AttentionConfig(
                name=AttentionType.fused_v2,
                # name=AttentionType.default,
                n_heads=NUM_HEAD,
                n_kv_heads=NUM_KV_HEAD,
                bias=False,
                rope=RoPEConfig(name=RoPEType.default, theta=500_000, scaling=None, full_precision=True),
                qk_norm=layer_norm ,
                backend=AttentionBackendName.flash_4,
                use_head_qk_norm=True,
                dtype=dtype,
                d_attn=D_ATTN,
                mxfp8_qkv_projection=USE_FP8_ATTN_QKV,
                mxfp8_out_projection=USE_FP8_ATTN_OUT,
                mxfp8_save_qkv_for_backward=USE_FP8_ATTN_SAVE_QKV,
                use_recompute_qkv_prep=False,
                # use_recompute_qkv_prep=not PER_LAYER_RECOMPUTE, # only enable when not doing per-layer recompute
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
                lb_loss_weight=0.015,
                # z_loss_weight=1e-5,
                z_loss_weight=1e-4,
                lb_loss_granularity=MoELoadBalancingLossGranularity.local_batch,
                dtype=dtype,
                normalize_expert_weights=1.0,
                restore_weight_scale=True,
                # use_recompute_fp32_cast=not PER_LAYER_RECOMPUTE, # only enable when not doing per-layer recompute
                use_recompute_fp32_cast=False,
                original_top_k=ORIGINAL_TOP_K
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
                lb_loss_granularity=MoELoadBalancingLossGranularity.local_batch,
                dtype=dtype,
                normalize_expert_weights=1.0,
                restore_weight_scale=True,
                # use_recompute_fp32_cast=not PER_LAYER_RECOMPUTE, # only enable when not doing per-layer recompute
                use_recompute_fp32_cast=False,
            ) if NUM_SHARED_EXPERTS > 1 else None, # only need router if > 1 expert
            feed_forward_norm=layer_norm,
        ),
        lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=dtype),
        name=TransformerType.moe_fused_v2,

        init_std=0.01,
        dtype=dtype
    )

    # config.lm_head.loss_implementation = LMLossImplementation.fused_linear
    config.lm_head.loss_implementation = LMLossImplementation.default
    WINDOW_SIZE=2048
    config.block.attention.sliding_window = SlidingWindowAttentionConfig(
        force_full_attention_on_first_layer=False,
        force_full_attention_on_last_layer=True,
        pattern=[WINDOW_SIZE, -1]
    )

    dense_block_config = OLMoDDPTransformerBlockConfig(
        name=TransformerBlockType.moe_fused_v2,
        use_peri_norm=USE_PERI_NORM,
        rowwise_fp8=MoERowwiseFP8Config(enabled=USE_FP8, fused_autograd_recompute_swiglu=False) if USE_ROWWISE_A2A else None,
        attention=AttentionConfig(
            name=AttentionType.fused_v2,
            # name=AttentionType.default,
            n_heads=NUM_HEAD,
            n_kv_heads=NUM_KV_HEAD,
            bias=False,
            rope=RoPEConfig(name=RoPEType.default, theta=500_000, scaling=None, full_precision=True),
            qk_norm=layer_norm ,
            backend=AttentionBackendName.flash_4,
            use_head_qk_norm=True,
            dtype=dtype,
            d_attn=D_ATTN,
            mxfp8_qkv_projection=USE_FP8_ATTN_QKV,
            mxfp8_out_projection=USE_FP8_ATTN_OUT,
            mxfp8_save_qkv_for_backward=USE_FP8_ATTN_SAVE_QKV,
            use_recompute_qkv_prep=False,
            # use_recompute_qkv_prep=not PER_LAYER_RECOMPUTE, # only enable when not doing per-layer recompute
        ),
        routed_experts=None,
        routed_experts_router=None,
        shared_experts=SharedExpertsConfig(
            d_model=d_model,
            hidden_size=DENSE_LAYER_MLP,
            num_experts=1,
            bias=False,
            dtype=dtype,
        ),
        shared_experts_router=None,
        attention_norm=layer_norm,
        feed_forward_norm=layer_norm,
    )
    early_moe_block_config = deepcopy(config.block)
    if not isinstance(early_moe_block_config, OLMoDDPTransformerBlockConfig):
        raise TypeError(
            "early MoE capacity override requires OLMoDDPTransformerBlockConfig, "
            f"got {type(early_moe_block_config).__name__}"
        )
    if early_moe_block_config.ep is None:
        raise RuntimeError("early MoE capacity override requires block.ep to be configured")
    early_moe_block_config.ep.capacity_factor = EARLY_MOE_EP_NO_SYNC_CAPACITY_FACTOR
    early_moe_block_config.ep.validate()

    first_moe_block_config = deepcopy(early_moe_block_config)
    if first_moe_block_config.routed_experts_router is None:
        raise RuntimeError("first MoE router z-loss override requires routed_experts_router")
    first_moe_block_config.routed_experts_router.z_loss_weight = FIRST_MOE_ROUTER_Z_LOSS_WEIGHT

    # First block will be a regular transformer block (no MoE component).
    config.block_overrides = {
        0: deepcopy(dense_block_config),
        **{
            block_idx: deepcopy(early_moe_block_config)
            for block_idx in EARLY_MOE_CAPACITY_BLOCK_INDICES
        },
        1: first_moe_block_config,
        # 1: deepcopy(dense_block_config),
        # 2: deepcopy(dense_block_config),

        # also make last layer dense
        # NUM_LAYERS-1: deepcopy(dense_block_config),
    }

    for block_idx in range(NUM_LAYERS):
        block_config = deepcopy(config.block_overrides.get(block_idx, config.block))
        sliding_window_config = block_config.attention.sliding_window
        if (
            sliding_window_config is not None
            and sliding_window_config.should_use_swa(block_idx, NUM_LAYERS)
        ):
            continue

        if block_config.attention is None or block_config.attention.rope is None:
            raise RuntimeError(f"cannot apply YaRN RoPE scaling to block {block_idx}")
        block_config.attention.rope = replace(block_config.attention.rope, scaling=ROPE_SCALING)
        config.block_overrides[block_idx] = block_config

    return config

def build_train_module_config(
    common: CommonComponents,
    *,
    rank_microbatch_sequences: int,
    lr: float,
    reset_optimizer_states_on_load: bool,
    reset_optimizer_states_on_resume: bool,
) -> OLMoDDPTrainModuleConfig:
    return OLMoDDPTrainModuleConfig(
        rank_microbatch_size=rank_microbatch_sequences * common.max_sequence_length,
        max_sequence_length=common.max_sequence_length,
        optim=OLMoDDPOptimizerConfig(
            lr=lr,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            muon_adjust_lr_fn="match_rms_adamw" if USE_MUON else None,
            group_overrides=[
                OptimGroupOverride(
                    params=[
                        "*embeddings.weight",
                        "*embedding_norm.weight",
                        "*q_norm.weight",
                        "*k_norm.weight",
                        "*input_norm.weight",
                        # "*norm.weight",
                        # "*lm_head.w_out.weight",
                        "*lm_head.norm.weight",
                        "*attention_norm.weight",
                        "*feed_forward_norm.weight",
                    ],
                    opts=dict(weight_decay=0.0, use_muon=False),
                ),
                # OptimGroupOverride(
                #     params=[
                #         "*.w_q.weight",
                #         "*.w_k.weight",
                #         "*.w_v.weight",
                #         "*attention.w_out.weight", # to exclude "lm_head.w_out.weight"
                #         "*.w1.weight", "*.w2.weight", "*.w3.weight"
                #         ], # attention + dense mlp
                #     opts=dict(use_muon=USE_MUON),
                # ),
            ],
            compile=USE_COMPILE,
            dtype=DType.float32,
            sigma_factor=12,
            use_distributed=True
        ),
        compile_model=USE_COMPILE,
        ac_config=None, # AC handled elsewhere for MoE
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.ddp,
            reduce_grads_in_fp32=GRAD_REDUCE_IN_FP32,
            accumulate_grads_in_fp32=GRAD_ACC_IN_FP32,
        ),
        ep_config=TransformerExpertParallelConfig(degree=EP_DIM) if EP_DIM != 1 else None, # EP=1 means no expert parallel
        pp_config=TransformerPipelineParallelConfig(
            degree=PP_DIM,
            # schedule=PipelineScheduleType.custom_1F1B,
            # schedule=PipelineScheduleType.custom_1F1B_V,  # V placement for comparison against interleaved 1F1B
            schedule=PipelineScheduleType.custom_interleaved_1F1B,
            use_custom_stage_implementation=True,  # use custom stage implementation that re-uses receive buffers across micro-batches
            p2p_use_separate_group=True,
            p2p_backend="nccl",
            # p2p_backend="nccl_rma_ack",
            # p2p_nccl_min_ctas=1,
            # p2p_nccl_max_ctas=2,
            # forward_pull_ahead_extra_activations=[1, 0, 1, 1],
            split_points=SPLIT_POINTS,
        ) if PP_DIM > 1 else None,

        float8_config=None,
        z_loss_multiplier=1e-4,
        max_grad_norm=1.0,

        scheduler=LinearWithWarmup(units=SchedulerUnits.steps, warmup=200, alpha_f=0.0),
        reset_optimizer_states_on_load=reset_optimizer_states_on_load,
        reset_optimizer_states_on_resume=reset_optimizer_states_on_resume,
    )

def build_trainer_config(
    common: CommonComponents,
    *,
    load_path: Optional[str],
    max_duration: Duration,
    load_optim_state: bool,
) -> TrainerConfig:
    cancel_check_interval = 10

    cluster = 'ai2/jupiter'
    # cluster = 'cirrascale'
    from olmo_core.train.checkpoint import CheckpointerConfig

    config = (
        TrainerConfig(
            save_folder=common.save_folder,
            load_path=load_path,
            load_strategy=LoadStrategy.always if load_path else LoadStrategy.if_available,
            load_trainer_state=False,
            load_optim_state=load_optim_state,
            save_overwrite=True,
            checkpointer=CheckpointerConfig(
                save_thread_count=1, load_thread_count=8, throttle_uploads=True
            ),
            metrics_collect_interval=50,
            cancel_check_interval=cancel_check_interval,
            max_duration=max_duration,
            checkpoints_to_eval=[
                f"{common.save_folder}/step*000",
            ],
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=SAVE_INTERVAL,
                ephemeral_save_interval=None,
                save_async=False,
                pre_train_checkpoint=False,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=common.run_name,
                entity="ai2-llm",
                project="olmoe-dev-v2",
                enabled=WANDB_ENABLED,
                cancel_check_interval=cancel_check_interval,
            ),
        )
        .with_callback(
            "profiler",
            NvidiaProfilerCallback(
                enabled=USE_NV_PROFILE,
                profile_ranks=list(range(0, 8 * 8, 8)),
                start=3031,
                end=3034,
            ),
        )
    )
    if IN_EVAL_MODE:
        config = config.with_recommended_evals(
            common.tokenizer, SEQUENCE_LENGTH, cluster, task_set="fast", eval_interval=EVAL_INTERVAL
        )

    return config


def finalize_config(config: ExperimentConfig):
    # add active & total params to the wandb name
    total_params_in_B = config.model.num_params/1000/1000/1000
    active_params_in_B = config.model.num_active_params/1000/1000/1000
    log.info(f"Total params: {total_params_in_B:.2f}B, Active params: {active_params_in_B:.2f}B")

    wandb_cb = cast(WandBCallback, config.trainer.callbacks['wandb'])
    wandb_original_name = wandb_cb.name
    assert isinstance(wandb_cb.name, str), "WandB callback name must be initialized"
    wandb_cb.name += f"_{active_params_in_B:.2f}@{total_params_in_B:.2f}B"
    wandb_cb.name += f"_{NUM_LAYERS}L{TOP_K}K{NUM_EXPERTS}N{NUM_SHARED_EXPERTS}S_{TAG}"
    # _{EP_DIM}EP{PP_DIM}PP
    wandb_cb.group = f"{wandb_original_name}_{active_params_in_B:.2f}@{total_params_in_B:.2f}B_{NUM_LAYERS}L{TOP_K}K{NUM_EXPERTS}N{NUM_SHARED_EXPERTS}S_{TAG}"


def build_data_components(
    common: CommonComponents,
    *,
    mix_base_dir: str,
    intra_document_masking: bool,
    include_instance_filter: bool,
) -> DataComponents:
    dataset_config = NumpyPackedFSLDatasetConfig.from_data_mix(
        DataMix.OLMo_longmino_mix_0925,
        mix_base_dir=mix_base_dir,
        tokenizer=common.tokenizer,
        work_dir=common.work_dir,
        sequence_length=common.max_sequence_length,
        generate_doc_lengths=intra_document_masking,
        source_group_size=8,
        source_permutation_seed=123,
        instance_filter_config=None
        if not include_instance_filter
        else InstanceFilterConfig(
            repetition_max_period=13, repetition_min_period=1, repetition_max_count=32
        ),
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=common.global_batch_size,
        seed=34521,
        num_workers=8,
        prefetch_factor=8,
    )

    return DataComponents(dataset=dataset_config, data_loader=data_loader_config)


def build_local_aware_common_components(
    cli_context: CliContext,
    *,
    tokenizer: TokenizerConfig,
    global_batch_size: int,
    max_sequence_length: int,
    beaker_image: str,
    num_nodes: int,
    beaker_workspace: str,
    work_dir: str,
    save_root: str,
    num_execution_units: Optional[int] = None,
    flight_recorder: bool = False,
) -> CommonComponents:
    def with_save_folder(common: CommonComponents) -> CommonComponents:
        save_folder = os.path.join(save_root, cli_context.run_name)
        if common.save_folder != save_folder:
            save_root_source = SAVE_ROOT_ENV if SAVE_ROOT_ENV in os.environ else "LC script default"
            log.info(
                "Overriding save_folder from '%s' to '%s' using %s",
                common.save_folder,
                save_folder,
                save_root_source,
            )
            common = replace(common, save_folder=save_folder)
        return common

    def forward_env_to_launch(common: CommonComponents) -> CommonComponents:
        if common.launch is None:
            return common

        from olmo_core.launch.beaker import BeakerEnvVar

        existing_env_vars = {env_var.name for env_var in common.launch.env_vars}
        for env_var_name in LAUNCH_ENV_VARS_TO_FORWARD:
            if env_var_name in os.environ and env_var_name not in existing_env_vars:
                common.launch.env_vars.append(
                    BeakerEnvVar(name=env_var_name, value=os.environ[env_var_name])
                )
        return common

    override_work_dir = os.environ.get(WORK_DIR_OVERRIDE_ENV)
    if cli_context.cluster not in ("local", "localhost"):
        common = build_default_common_components(
            cli_context,
            tokenizer=tokenizer,
            global_batch_size=global_batch_size,
            max_sequence_length=max_sequence_length,
            beaker_image=beaker_image,
            num_nodes=num_nodes,
            beaker_workspace=beaker_workspace,
            num_execution_units=num_execution_units,
            flight_recorder=flight_recorder,
        )
        if override_work_dir:
            log.info(
                "Overriding dataset work_dir from '%s' to '%s' via %s",
                common.work_dir,
                override_work_dir,
                WORK_DIR_OVERRIDE_ENV,
            )
            common = replace(common, work_dir=override_work_dir)
        common = with_save_folder(common)
        return forward_env_to_launch(common)

    common = with_save_folder(CommonComponents(
        run_name=cli_context.run_name,
        root_dir="/workspace",
        work_dir=override_work_dir or work_dir,
        save_folder=os.path.join(save_root, cli_context.run_name),
        launch=None,
        tokenizer=tokenizer,
        max_sequence_length=max_sequence_length,
        global_batch_size=global_batch_size,
    ))
    if override_work_dir:
        log.info("Using dataset work_dir '%s' from %s", override_work_dir, WORK_DIR_OVERRIDE_ENV)
    return common


def make_config_builder(cli_context: CliContext) -> ExperimentConfig:
    if cli_context.cmd in (SubCmd.launch, SubCmd.train, SubCmd.train_single) and LOAD_PATH is None:
        raise OLMoConfigurationError(
            f"Set {LOAD_PATH_ENV} to the final mid-training checkpoint before launching "
            "production long-context training."
        )

    max_duration = Duration.tokens(LC_MAX_TOKENS)

    config_builder = partial(
        build_config,
        common_config_builder=partial(
            build_local_aware_common_components,
            work_dir=LOCAL_WORK_DIR,
            save_root=SAVE_ROOT,
        ),
        data_config_builder=partial(
            build_data_components,
            mix_base_dir=MIX_BASE_DIR,
            intra_document_masking=INTRA_DOCUMENT_MASKING,
            include_instance_filter=INCLUDE_INSTANCE_FILTER,
        ),
        model_config_builder=build_model_config,
        train_module_config_builder=partial(
            build_train_module_config,
            rank_microbatch_sequences=LC_RANK_MICROBATCH_SEQUENCES,
            lr=LC_LR,
            reset_optimizer_states_on_load=RESET_OPTIMIZER_STATES_ON_LOAD,
            reset_optimizer_states_on_resume=RESET_OPTIMIZER_STATES_ON_RESUME,
        ),
        trainer_config_builder=partial(
            build_trainer_config,
            load_path=LOAD_PATH,
            max_duration=max_duration,
            load_optim_state=LOAD_OPTIM_STATE,
        ),
        tokenizer=TokenizerConfig.dolma2(),
        global_batch_size=LC_GLOBAL_BATCH_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        num_nodes=NUM_NODES,
        include_default_evals=False,
        flight_recorder=True,
        finalize_config=finalize_config,
    )
    return cast(ExperimentConfig, config_builder(cli_context))


if __name__ == "__main__":
    main(
        config_builder=make_config_builder,
    )
