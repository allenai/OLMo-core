"""
``script_utils``-style variant of ``OLMoE3-dev-260401-test-refactor.py``.

Same model/optim/data/scheduler config, but driven by the OLMo3-official
launch pattern (``olmo_core.script_utils.main``). Run via ``torchrun``
directly or wrap with ``python -m olmo_core.launch.beaker``::

    python -m olmo_core.launch.beaker \\
      --name=olmoe3-dev-test --cluster ai2/jupiter \\
      --nodes 8 --gpus 8 --weka oe-training-default \\
      --beaker-image=petew/olmo-core-tch270cu128-2025-05-16 \\
      --workspace=ai2/OLMo-core --budget=ai2/oe-training \\
      -- \\
      python src/scripts/train/OLMoE3-dev-260401-launchable.py \\
        --save-folder=/weka/oe-training-default/ai2-llm/checkpoints/$USER/olmoe3-dev-test \\
        --name=olmoe3-dev-test \\
        --data-root=/weka/oe-training-default/ai2-llm
"""

import argparse
import logging
import math
from dataclasses import dataclass
from typing import List, Optional, cast

import torch

from olmo_core.config import Config, DType
from olmo_core.data import (
    DataMix,
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.parallel.pipeline_parallel import PipelineScheduleType
from olmo_core.nn.attention import (
    AttentionConfig,
    AttentionType,
    SlidingWindowAttentionConfig,
)
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig, LMLossImplementation
from olmo_core.nn.moe import MoELoadBalancingLossGranularity, MoERouterGatingFunction
from olmo_core.nn.moe.v2.block import MoERouterConfigV2
from olmo_core.nn.rope import RoPEConfig, RoPEType
from olmo_core.nn.transformer import (
    MoEFusedV2TransformerConfig,
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
    TransformerType,
)
from olmo_core.optim import OptimGroupOverride, SchedulerUnits
from olmo_core.optim.scheduler import (
    ComposableScheduler,
    ComposableSchedulerMonkeyPatchDecay,
    ComposableSchedulerStage,
    ComposableSchedulerStageType,
)
from olmo_core.script_utils import main
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    NvidiaProfilerCallback,
    TorchMemoryHistoryCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    MoEV2TransformerTrainModuleConfig,
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
)
from olmo_core.train.train_module.transformer import TransformerPipelineParallelConfig

log = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


# Local ExperimentConfig with the right ``train_module`` annotation. The version in
# ``olmo_core.script_utils`` annotates ``train_module`` as ``TransformerTrainModuleConfig``,
# but this script uses ``MoEV2TransformerTrainModuleConfig`` (a sibling, not subclass).
# ``Config.merge()`` reads field annotations to coerce types, so passing a MoEV2 instance
# into the script_utils class fails coercion. ``script_utils.main`` accesses the config
# duck-typed, so a local class is sufficient.
@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: MoEV2TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = 12536
    load_path: Optional[str] = None

# -----------------------------------------------------------------------------
# Hyperparameters (kept identical to OLMoE3-dev-260401-test-refactor.py).
# -----------------------------------------------------------------------------

DEFAULT_SEQUENCE_LENGTH = 8192

EVAL_INTERVAL = 2000
SAVE_INTERVAL = 1000

NUM_EXPERTS = 24
TOP_K = 4
D_MODEL = 2560
D_ATTN = 3072

HEAD_DIM = 128
NUM_HEAD = D_ATTN // HEAD_DIM
NUM_KV_HEAD = NUM_HEAD // 2
MOE_HIDDEN_SIZE = 2560
NUM_SHARED_EXPERTS = 1
SHARED_MLP_HIDDEN_SIZE = 1280

EFFECTIVE_MLP = MOE_HIDDEN_SIZE * TOP_K + SHARED_MLP_HIDDEN_SIZE * NUM_SHARED_EXPERTS
MLP_RATIO = EFFECTIVE_MLP / D_MODEL

DENSE_LAYER_MLP = TOP_K * MOE_HIDDEN_SIZE + SHARED_MLP_HIDDEN_SIZE * NUM_SHARED_EXPERTS

EP_DIM = 1
PP_DIM = 1

REF_NUM_NODES = 8

# stage 1 - 1M
MAX_DURATION = int(25.5e9)
MICRO_BSZ = 4
GLOBAL_BATCH_SIZE_SEQ = (8 * 8) * 2 * 1

GLOBAL_BATCH_SIZE = GLOBAL_BATCH_SIZE_SEQ * DEFAULT_SEQUENCE_LENGTH
NUM_MICRO_BATCHES = GLOBAL_BATCH_SIZE_SEQ // (REF_NUM_NODES * 8) // MICRO_BSZ * PP_DIM
GLOBAL_BATCH_TOKENS_IN_M = GLOBAL_BATCH_SIZE // 1024 // 1024

SCHED_WARMUP_TOKENS = int((10e9 // GLOBAL_BATCH_SIZE) * GLOBAL_BATCH_SIZE)
SCHED_FAST_DECAY_TOKENS = int((0e9 // GLOBAL_BATCH_SIZE) * GLOBAL_BATCH_SIZE)
SCHED_LONG_DECAY_TOKENS = int((5990e9 // GLOBAL_BATCH_SIZE) * GLOBAL_BATCH_SIZE)
SCHED_MID_FRACTION = 1.0
SCHED_FINAL_FRACTION = 0.1

LR = 4e-4
LR = LR / SCHED_MID_FRACTION
LR = LR * math.sqrt(GLOBAL_BATCH_SIZE / (4 * 1024 * 1024))
EXPERT_LR = LR

NUM_LAYERS = 8


def _get_split_points(original_num_layers: int, num_stages: int, minus_last_stage: int):
    assert (
        original_num_layers % num_stages == 0
    ), "Original number of layers must be divisible by number of stages"
    layers_per_stage = original_num_layers // num_stages
    new_num_layers = original_num_layers - minus_last_stage
    split_points = [i * layers_per_stage for i in range(1, num_stages)]
    return new_num_layers, split_points


if PP_DIM > 1:
    MINUS_LAST_STAGE = 1
    NUM_LAYERS, SPLIT_POINTS = _get_split_points(
        NUM_LAYERS, PP_DIM * 2, minus_last_stage=MINUS_LAST_STAGE
    )
else:
    SPLIT_POINTS = None

USE_COMPILE = True
USE_NO_SYNC_EP = True
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
USE_MUON = False
USE_PERI_NORM = True
PRODUCTION_RUN = False

TAG = "p1"

MONKEY_PATCH_DECAY_START_TOKENS = None
MONKEY_PATCH_DECAY_DURATION_TOKENS = int((200e9 // GLOBAL_BATCH_SIZE) * GLOBAL_BATCH_SIZE)
MONKEY_PATCH_DECAY_END_FRACTION = SCHED_FINAL_FRACTION
MONKEY_PATCH_DECAY_SHAPE = ComposableSchedulerStageType.cosine


# -----------------------------------------------------------------------------
# Builders.
# -----------------------------------------------------------------------------


def build_model_config(tokenizer_config: TokenizerConfig) -> MoEFusedV2TransformerConfig:
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
        vocab_size=tokenizer_config.padded_vocab_size(),
        n_layers=NUM_LAYERS,
        embed_scale=math.sqrt(d_model),
        embedding_norm=layer_norm,
        block=MoEFusedV2TransformerBlockConfig(
            name=TransformerBlockType.moe_fused_v2,
            use_peri_norm=USE_PERI_NORM,
            ep_no_sync=USE_NO_SYNC_EP,
            checkpoint_permute_moe_unpermute=False,
            checkpoint_attn=False,
            checkpoint_second_unpermute=False,
            ep_no_sync_share_combine_out=PER_LAYER_RECOMPUTE,
            ep_no_sync_share_dispatch_out=PER_LAYER_RECOMPUTE,
            ep_no_sync_shared_slots=2 if USE_TBO else 1,
            ep_no_sync_use_rowwise_all_to_all=USE_ROWWISE_A2A,
            ep_no_sync_rowwise_nblocks=ROWWISE_A2A_NBLOCKS,
            ep_no_sync_capacity_factor=1.25,
            rowwise_fp8=MoERowwiseFP8Config(enabled=USE_FP8) if USE_ROWWISE_A2A else None,
            sequence_mixer=AttentionConfig(
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
                use_recompute_qkv_prep=False,
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
                lb_loss_weight=0.01,
                z_loss_weight=1e-5,
                lb_loss_granularity=MoELoadBalancingLossGranularity.instance,
                dtype=dtype,
                normalize_expert_weights=1.0,
                restore_weight_scale=True,
                use_recompute_fp32_cast=False,
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
                top_k=NUM_SHARED_EXPERTS,
                gating_function=MoERouterGatingFunction.sigmoid,
                uniform_expert_assignment=False,
                lb_loss_weight=None,
                z_loss_weight=None,
                lb_loss_granularity=MoELoadBalancingLossGranularity.instance,
                dtype=dtype,
                normalize_expert_weights=1.0,
                restore_weight_scale=True,
                use_recompute_fp32_cast=False,
            )
            if NUM_SHARED_EXPERTS > 1
            else None,
            feed_forward_norm=layer_norm,
        ),
        lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=dtype),
        name=TransformerType.moe_fused_v2,
        init_std=0.01,
        dtype=dtype,
    )

    config.lm_head.loss_implementation = LMLossImplementation.default
    WINDOW_SIZE = 2048
    config.block.sequence_mixer.sliding_window = SlidingWindowAttentionConfig(
        force_full_attention_on_first_layer=False,
        force_full_attention_on_last_layer=True,
        pattern=[WINDOW_SIZE, -1],
    )

    dense_block_config = TransformerBlockConfig(
        name=TransformerBlockType.peri_norm if USE_PERI_NORM else TransformerBlockType.default,
        sequence_mixer=AttentionConfig(
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
            use_recompute_qkv_prep=False,
        ),
        feed_forward_moe=None,
        feed_forward=FeedForwardConfig(hidden_size=DENSE_LAYER_MLP, bias=False),
        attention_norm=layer_norm,
        feed_forward_norm=layer_norm,
    )
    from copy import deepcopy

    # First block is a regular (non-MoE) transformer block.
    config.block_overrides = {0: deepcopy(dense_block_config)}

    return config


def build_train_module_config(sequence_length: int) -> MoEV2TransformerTrainModuleConfig:
    from olmo_core.optim.moe_optimizer import MoEFusedV2OptimizerConfig

    return MoEV2TransformerTrainModuleConfig(
        rank_microbatch_size=MICRO_BSZ * sequence_length,
        max_sequence_length=sequence_length,
        optim=MoEFusedV2OptimizerConfig(
            lr=LR,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            muon_adjust_lr_fn="match_rms_adamw" if USE_MUON else None,
            group_overrides=[
                OptimGroupOverride(
                    params=[
                        "*embedding_norm.weight",
                        "*q_norm.weight",
                        "*k_norm.weight",
                        "*input_norm.weight",
                        "*lm_head.norm.weight",
                        "*attention_norm.weight",
                        "*feed_forward_norm.weight",
                    ],
                    opts=dict(weight_decay=0.0, use_muon=False),
                ),
                OptimGroupOverride(
                    params=["*routed_experts.w_up_gate", "*routed_experts.w_down"],
                    opts=dict(lr=EXPERT_LR, use_muon=USE_MUON),
                ),
            ],
            compile=USE_COMPILE,
            dtype=DType.float32,
            sigma_factor=12,
            use_distributed=True,
        ),
        compile_model=USE_COMPILE,
        ac_config=None,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.ddp,
            reduce_grads_in_fp32=GRAD_REDUCE_IN_FP32,
            accumulate_grads_in_fp32=GRAD_ACC_IN_FP32,
        ),
        ep_config=TransformerExpertParallelConfig(degree=EP_DIM) if EP_DIM != 1 else None,
        pp_config=TransformerPipelineParallelConfig(
            degree=PP_DIM,
            schedule=PipelineScheduleType.custom_interleaved_1F1B,
            use_custom_stage_implementation=True,
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


def build_trainer_config(opts: argparse.Namespace) -> TrainerConfig:
    from olmo_core.train.checkpoint import CheckpointerConfig

    cancel_check_interval = 10
    return (
        TrainerConfig(
            save_folder=opts.save_folder,
            save_overwrite=True,
            checkpointer=CheckpointerConfig(
                save_thread_count=3, load_thread_count=2, throttle_uploads=True
            ),
            metrics_collect_interval=2,
            cancel_check_interval=cancel_check_interval,
            max_duration=Duration.tokens(MAX_DURATION),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=SAVE_INTERVAL,
                ephemeral_save_interval=None,
                save_async=False,
                pre_train_checkpoint=PRODUCTION_RUN,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=opts.name,
                entity="ai2-llm",
                project="olmoe-dev-v2",
                enabled=PRODUCTION_RUN,
                cancel_check_interval=cancel_check_interval,
            ),
        )
        .with_callback(
            "profiler",
            NvidiaProfilerCallback(
                enabled=False,
                profile_ranks=list(range(0, 8 * 8, 8)),
                start=1021,
                end=1026,
            ),
        )
        .with_callback(
            "torch_mem_history",
            TorchMemoryHistoryCallback(
                enabled=False,
                profile_ranks=list(range(0, 8 * 128, 8)),
                start=59161,
                end=59164,
                output_dir="/workspace/tmp",
            ),
        )
    )


def build_dataset_config(
    opts: argparse.Namespace,
    tokenizer_config: TokenizerConfig,
    sequence_length: int,
    intra_document_masking: bool = False,
    include_instance_filter: bool = True,
) -> NumpyFSLDatasetConfig:
    return NumpyFSLDatasetConfig.from_data_mix(
        DataMix.OLMoE_mix_0824_dev,
        tokenizer=tokenizer_config,
        mix_base_dir=opts.data_root,
        work_dir=opts.work_dir,
        sequence_length=sequence_length,
        max_target_sequence_length=max(sequence_length, 8192),
        generate_doc_lengths=intra_document_masking,
        instance_filter_config=None
        if not include_instance_filter
        else InstanceFilterConfig(
            repetition_max_period=13, repetition_min_period=1, repetition_max_count=32
        ),
    )


def finalize_config(config: ExperimentConfig, opts: argparse.Namespace) -> None:
    total_params_in_B = config.model.num_params / 1000 / 1000 / 1000
    active_params_in_B = config.model.num_active_params / 1000 / 1000 / 1000
    log.info(f"Total params: {total_params_in_B:.2f}B, Active params: {active_params_in_B:.2f}B")

    wandb_cb = cast(WandBCallback, config.trainer.callbacks["wandb"])
    assert isinstance(wandb_cb.name, str), "WandB callback name must be initialized"
    wandb_original_name = wandb_cb.name
    wandb_cb.name += f"_{active_params_in_B:.2f}@{total_params_in_B:.2f}B"
    wandb_cb.name += f"_{NUM_LAYERS}L{TOP_K}K{NUM_EXPERTS}N{NUM_SHARED_EXPERTS}S_{TAG}"
    wandb_cb.group = (
        f"{wandb_original_name}_{active_params_in_B:.2f}@{total_params_in_B:.2f}B"
        f"_{NUM_LAYERS}L{TOP_K}K{NUM_EXPERTS}N{NUM_SHARED_EXPERTS}S_{TAG}"
    )


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    sequence_length = opts.sequence_length or DEFAULT_SEQUENCE_LENGTH
    tokenizer_config = TokenizerConfig.dolma2()

    model_config = build_model_config(tokenizer_config)
    dataset_config = build_dataset_config(opts, tokenizer_config, sequence_length)
    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=34521,
        num_workers=8,
    )
    train_module_config = build_train_module_config(sequence_length)
    trainer_config = build_trainer_config(opts)

    config = ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
        init_seed=SEED,
    ).merge(overrides)

    finalize_config(config, opts)
    return config


if __name__ == "__main__":
    main(build_config)
