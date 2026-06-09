"""
SFT entry point for OLMo-core MoE V2 models.

This is intentionally scoped to architectures OLMo-core already supports. It
does not try to import Qwen3.5 weights or reproduce Qwen3.5 hybrid attention.
Use it first for random-init plumbing, then point ``--load-path`` at a
supported MoE V2 checkpoint when one is available.

Examples:
  python src/scripts/train/sft/OlmoE3-SFT.py dry_run moe-sft-smoke local \
      --dataset-path=/workspace/tasks/june8/scratch/sft_debug_dataset \
      --trainer.max_duration.value=2

  torchrun --nproc-per-node=1 src/scripts/train/sft/OlmoE3-SFT.py train moe-sft-smoke local \
      --dataset-path=/workspace/tasks/june8/scratch/sft_debug_dataset \
      --trainer.max_duration.value=2
"""

import logging
import math
import os
from dataclasses import replace
from functools import partial
from typing import Any, List, Optional, Tuple, cast

import yaml

# Keep this before olmo_core imports. Several modules import nvtx at import time.
os.environ.setdefault("NVTX_DISABLE", "1")

from olmo_core.config import DType
from olmo_core.data import NumpyDataLoaderConfig, NumpyPackedFSLDatasetConfig, TokenizerConfig
from olmo_core.data.types import LongDocStrategy
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.parallel.pipeline_parallel import PipelineScheduleType
from olmo_core.internal.experiment import (
    CliContext,
    CommonComponents,
    DataComponents,
    ExperimentConfig,
    build_common_components as build_default_common_components,
    build_config,
    main,
)
from olmo_core.nn.attention import AttentionConfig, AttentionType, SlidingWindowAttentionConfig
from olmo_core.nn.attention.backend import AttentionBackendName
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig, LMLossImplementation
from olmo_core.nn.moe import (
    MoELoadBalancingLossGranularity,
    MoERouterGatingFunction,
)
from olmo_core.nn.moe.v2.block import MoEFusedV2TransformerBlockConfig
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig
from olmo_core.nn.rope import RoPEConfig, RoPEType
from olmo_core.nn.transformer import (
    MoEFusedV2TransformerConfig,
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
    TransformerType,
)
from olmo_core.optim import LinearWithWarmup, OptimGroupOverride
from olmo_core.optim.moe_optimizer import MoEFusedV2OptimizerConfig
from olmo_core.train import Duration, LoadStrategy, TrainerConfig
from olmo_core.train.callbacks import CheckpointerCallback
from olmo_core.train.callbacks.wandb import WandBCallback
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.train_module import (
    MoEV2TransformerTrainModuleConfig,
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
)
from olmo_core.train.train_module.transformer import TransformerPipelineParallelConfig

log = logging.getLogger(__name__)

DEFAULT_DATASET_PATH = "/workspace/tasks/june8/scratch/sft_debug_dataset"
LOCAL_WORK_DIR = "/workspace/tasks/june8/scratch/dataset-cache"
LOCAL_SAVE_ROOT = "/workspace/checkpoint"

SEQUENCE_LENGTH = 2048
GLOBAL_BATCH_SIZE = 8 * SEQUENCE_LENGTH
RANK_MICROBATCH_SEQUENCES = 1

D_MODEL = 512
D_ATTN = 512
HEAD_DIM = 128
NUM_HEADS = D_ATTN // HEAD_DIM
NUM_KV_HEADS = max(1, NUM_HEADS // 4)
NUM_LAYERS = 4

NUM_EXPERTS = 8
TOP_K = 2
MOE_HIDDEN_SIZE = 1024
NUM_SHARED_EXPERTS = 0
SHARED_EXPERT_HIDDEN_SIZE = 512

DENSE_FIRST_LAYER = True
DENSE_LAYER_HIDDEN_SIZE = TOP_K * MOE_HIDDEN_SIZE

EP_DEGREE = 1
PP_DEGREE = 1

LR = 5e-5
WEIGHT_DECAY = 0.0
MAX_STEPS = 10
SAVE_INTERVAL = 50
EPHEMERAL_SAVE_INTERVAL = 25

COMPILE_MODEL = False
COMPILE_OPTIMIZER = False
USE_PERI_NORM = True
USE_ROWWISE_A2A = False
ATTENTION_BACKEND = AttentionBackendName.flash_4
INTRA_DOCUMENT_MASKING = True


def _parse_override(arg: str) -> Tuple[str, Any]:
    if "=" not in arg:
        name, value = arg, "true"
    else:
        name, value = arg.split("=", 1)
    name = name.strip(" -").replace("-", "_")
    if not value or value.isspace():
        parsed_value = ""
    else:
        parsed_value = yaml.safe_load(value)
    return name, parsed_value


def _pop_override(overrides: List[str], key: str, default: Any) -> Any:
    for idx, override in enumerate(list(overrides)):
        name, value = _parse_override(override)
        if name == key:
            overrides.pop(idx)
            return value
    return default


def _pop_int_override(overrides: List[str], key: str, default: int) -> int:
    return int(_pop_override(overrides, key, default))


def _pop_optional_str_override(overrides: List[str], key: str) -> Optional[str]:
    value = _pop_override(overrides, key, None)
    if value in (None, ""):
        return None
    return str(value)


def _split_points(num_layers: int, pp_degree: int) -> Optional[List[int]]:
    if pp_degree <= 1:
        return None
    if num_layers % pp_degree != 0:
        raise ValueError(
            f"NUM_LAYERS ({num_layers}) must be divisible by PP_DEGREE ({pp_degree})"
        )
    layers_per_stage = num_layers // pp_degree
    return [stage * layers_per_stage for stage in range(1, pp_degree)]


def build_local_aware_common_components(
    cli_context: CliContext,
    *,
    tokenizer: TokenizerConfig,
    global_batch_size: int,
    max_sequence_length: int,
    beaker_image: str,
    num_nodes: int,
    beaker_workspace: str,
    num_execution_units: Optional[int] = None,
    flight_recorder: bool = False,
) -> CommonComponents:
    if cli_context.cluster != "local":
        return build_default_common_components(
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

    return CommonComponents(
        run_name=cli_context.run_name,
        root_dir="/workspace",
        work_dir=LOCAL_WORK_DIR,
        save_folder=os.path.join(LOCAL_SAVE_ROOT, cli_context.run_name),
        launch=None,
        tokenizer=tokenizer,
        max_sequence_length=max_sequence_length,
        global_batch_size=global_batch_size,
    )


def build_sft_data_components(
    common: CommonComponents,
    *,
    dataset_path: str,
    intra_document_masking: bool,
) -> DataComponents:
    clean_path = dataset_path.rstrip("/")
    dataset = NumpyPackedFSLDatasetConfig(
        tokenizer=common.tokenizer,
        work_dir=common.work_dir,
        paths=[f"{clean_path}/token_ids_part_*.npy"],
        expand_glob=True,
        label_mask_paths=[f"{clean_path}/labels_mask_*.npy"],
        generate_doc_lengths=intra_document_masking,
        long_doc_strategy=LongDocStrategy.truncate,
        sequence_length=common.max_sequence_length,
    )
    data_loader = NumpyDataLoaderConfig(
        global_batch_size=common.global_batch_size,
        seed=34521,
        num_workers=4,
    )
    return DataComponents(dataset=dataset, data_loader=data_loader)


def _attention_config(
    dtype: DType,
    layer_norm: LayerNormConfig,
    attention_backend: AttentionBackendName,
) -> AttentionConfig:
    return AttentionConfig(
        name=AttentionType.default,
        n_heads=NUM_HEADS,
        n_kv_heads=NUM_KV_HEADS,
        bias=False,
        rope=RoPEConfig(
            name=RoPEType.default,
            theta=500_000,
            scaling=None,
            full_precision=True,
        ),
        qk_norm=layer_norm,
        backend=attention_backend,
        use_head_qk_norm=True,
        dtype=dtype,
        d_attn=D_ATTN,
    )


def build_model_config(
    common: CommonComponents,
    *,
    attention_backend: AttentionBackendName,
) -> TransformerConfig:
    del common
    dtype = DType.float32
    layer_norm = LayerNormConfig(
        name=LayerNormType.rms,
        eps=1e-6,
        bias=False,
        dtype=dtype,
    )

    block = MoEFusedV2TransformerBlockConfig(
        name=TransformerBlockType.moe_fused_v2,
        use_peri_norm=USE_PERI_NORM,
        ep_no_sync=EP_DEGREE > 1,
        ep_no_sync_use_rowwise_all_to_all=USE_ROWWISE_A2A,
        ep_no_sync_capacity_factor=1.25,
        checkpoint_attn=False,
        checkpoint_permute_moe_unpermute=False,
        checkpoint_second_unpermute=False,
        attention=_attention_config(dtype, layer_norm, attention_backend),
        attention_norm=layer_norm,
        routed_experts=RoutedExpertsConfig(
            d_model=D_MODEL,
            hidden_size=MOE_HIDDEN_SIZE,
            num_experts=NUM_EXPERTS,
            bias=False,
            dtype=dtype,
        ),
        routed_experts_router=MoERouterConfigV2(
            d_model=D_MODEL,
            num_experts=NUM_EXPERTS,
            top_k=TOP_K,
            gating_function=MoERouterGatingFunction.softmax,
            lb_loss_weight=0.01,
            z_loss_weight=1e-3,
            lb_loss_granularity=MoELoadBalancingLossGranularity.instance,
            dtype=dtype,
            normalize_expert_weights=1.0,
            restore_weight_scale=True,
            original_top_k=TOP_K,
        ),
        shared_experts=(
            SharedExpertsConfig(
                d_model=D_MODEL,
                hidden_size=SHARED_EXPERT_HIDDEN_SIZE,
                num_experts=NUM_SHARED_EXPERTS,
                bias=False,
                dtype=dtype,
            )
            if NUM_SHARED_EXPERTS > 0
            else None
        ),
        shared_experts_router=(
            MoERouterConfigV2(
                d_model=D_MODEL,
                num_experts=NUM_SHARED_EXPERTS,
                top_k=NUM_SHARED_EXPERTS,
                gating_function=MoERouterGatingFunction.sigmoid,
                lb_loss_weight=None,
                z_loss_weight=None,
                lb_loss_granularity=MoELoadBalancingLossGranularity.instance,
                dtype=dtype,
                normalize_expert_weights=1.0,
                restore_weight_scale=True,
            )
            if NUM_SHARED_EXPERTS > 1
            else None
        ),
        feed_forward_norm=layer_norm,
    )

    config = MoEFusedV2TransformerConfig(
        name=TransformerType.moe_fused_v2,
        init_seed=2026,
        d_model=D_MODEL,
        vocab_size=TokenizerConfig.dolma2().padded_vocab_size(),
        n_layers=NUM_LAYERS,
        embed_scale=math.sqrt(D_MODEL),
        embedding_norm=layer_norm,
        block=block,
        lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=dtype),
        init_std=0.01,
        dtype=dtype,
        recompute_each_block=False,
        recompute_all_blocks_by_chunk=False,
    )
    config.lm_head.loss_implementation = LMLossImplementation.default
    config.block.attention.sliding_window = SlidingWindowAttentionConfig(
        force_full_attention_on_first_layer=False,
        force_full_attention_on_last_layer=True,
        pattern=[-1],
    )

    if DENSE_FIRST_LAYER:
        config.block_overrides = {
            0: TransformerBlockConfig(
                name=TransformerBlockType.peri_norm
                if USE_PERI_NORM
                else TransformerBlockType.default,
                attention=_attention_config(dtype, layer_norm, attention_backend),
                attention_norm=layer_norm,
                feed_forward_norm=layer_norm,
                feed_forward=FeedForwardConfig(
                    hidden_size=DENSE_LAYER_HIDDEN_SIZE,
                    bias=False,
                    dtype=dtype,
                ),
                feed_forward_moe=None,
            )
        }

    return config


def build_train_module_config(common: CommonComponents) -> MoEV2TransformerTrainModuleConfig:
    rank_microbatch_size = RANK_MICROBATCH_SEQUENCES * common.max_sequence_length
    split_points = _split_points(NUM_LAYERS, PP_DEGREE)

    return MoEV2TransformerTrainModuleConfig(
        rank_microbatch_size=rank_microbatch_size,
        max_sequence_length=common.max_sequence_length,
        optim=MoEFusedV2OptimizerConfig(
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(
                    params=[
                        "*embeddings.weight",
                        "*embedding_norm.weight",
                        "*q_norm.weight",
                        "*k_norm.weight",
                        "*lm_head.norm.weight",
                        "*attention_norm.weight",
                        "*feed_forward_norm.weight",
                    ],
                    opts={"weight_decay": 0.0, "use_muon": False},
                )
            ],
            compile=COMPILE_OPTIMIZER,
            dtype=DType.float32,
            sigma_factor=12,
            use_distributed=True,
        ),
        compile_model=COMPILE_MODEL,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.ddp,
            reduce_grads_in_fp32=True,
            accumulate_grads_in_fp32=True,
        ),
        ep_config=TransformerExpertParallelConfig(degree=EP_DEGREE)
        if EP_DEGREE > 1
        else None,
        pp_config=(
            TransformerPipelineParallelConfig(
                degree=PP_DEGREE,
                schedule=PipelineScheduleType.custom_interleaved_1F1B,
                use_custom_stage_implementation=True,
                p2p_use_separate_group=True,
                p2p_backend="nccl",
                split_points=split_points,
            )
            if PP_DEGREE > 1
            else None
        ),
        ac_config=None,
        float8_config=None,
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=LinearWithWarmup(warmup_fraction=0.03, alpha_f=0.0),
    )


def build_trainer_config(
    common: CommonComponents,
    *,
    load_path: Optional[str],
) -> TrainerConfig:
    return (
        TrainerConfig(
            save_folder=common.save_folder,
            load_path=load_path,
            load_strategy=LoadStrategy.if_available,
            load_trainer_state=False,
            load_optim_state=False,
            save_overwrite=True,
            checkpointer=CheckpointerConfig(
                save_thread_count=1,
                load_thread_count=8,
                throttle_uploads=True,
            ),
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.steps(MAX_STEPS),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=SAVE_INTERVAL,
                ephemeral_save_interval=EPHEMERAL_SAVE_INTERVAL,
                save_async=False,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=common.run_name,
                entity="ai2-llm",
                project="olmoe-sft",
                enabled=False,
                cancel_check_interval=10,
            ),
        )
    )


def finalize_config(config: ExperimentConfig) -> None:
    wandb_cb = config.trainer.callbacks.get("wandb")
    if isinstance(wandb_cb, WandBCallback) and wandb_cb.name is not None:
        total_params_in_b = config.model.num_params / 1_000_000_000
        active_params_in_b = config.model.num_active_params / 1_000_000_000
        wandb_cb.name += f"_{active_params_in_b:.2f}@{total_params_in_b:.2f}B"


def make_config_builder(cli_context: CliContext) -> ExperimentConfig:
    overrides = list(cli_context.overrides)
    dataset_path = str(_pop_override(overrides, "dataset_path", DEFAULT_DATASET_PATH))
    seq_len = _pop_int_override(overrides, "seq_len", SEQUENCE_LENGTH)
    num_nodes = _pop_int_override(overrides, "num_nodes", 1)
    global_batch_size = _pop_int_override(overrides, "global_batch_size", 8 * seq_len)
    load_path = _pop_optional_str_override(overrides, "load_path")
    attention_backend = AttentionBackendName(
        str(_pop_override(overrides, "attention_backend", ATTENTION_BACKEND.value))
    )
    intra_document_masking = bool(
        _pop_override(overrides, "intra_document_masking", INTRA_DOCUMENT_MASKING)
    )

    clean_context = replace(cli_context, overrides=overrides)
    config_builder = partial(
        build_config,
        common_config_builder=build_local_aware_common_components,
        data_config_builder=partial(
            build_sft_data_components,
            dataset_path=dataset_path,
            intra_document_masking=intra_document_masking,
        ),
        model_config_builder=partial(
            build_model_config,
            attention_backend=attention_backend,
        ),
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=partial(build_trainer_config, load_path=load_path),
        finalize_config=finalize_config,
        tokenizer=TokenizerConfig.dolma2(),
        global_batch_size=global_batch_size,
        max_sequence_length=seq_len,
        num_nodes=num_nodes,
        include_default_evals=False,
    )
    return cast(ExperimentConfig, config_builder(clean_context))


if __name__ == "__main__":
    main(config_builder=make_config_builder)
