"""
SFT pilot for the OLMoE3-dev-260429-t001 MoE V2 checkpoint.

This follows the dense OLMo-3 SFT data convention: pass a pre-tokenized SFT
dataset directory containing ``token_ids_part_*.npy`` and
``labels_mask_*.npy``. The model shape mirrors
``src/scripts/train/OLMoE3-dev-260429-t001.py`` so it can load the older
checkpoint even if newer pretraining examples drift.

Example:
  PYTHONPATH=src torchrun --standalone --nproc-per-node=8 \
      src/scripts/train/sft/OLMoE3-dev-260429-t001-SFT.py train \
      OLMoE3-dev-260429-t001-sft-pilot local \
      --dataset-path=/path/to/packed/sft
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
from olmo_core.nn.moe import MoELoadBalancingLossGranularity, MoERouterGatingFunction
from olmo_core.nn.moe.v2.block import MoEFusedV2TransformerBlockConfig
from olmo_core.nn.moe.v2.fp8 import MoERowwiseFP8Config
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

DEFAULT_LOAD_PATH = (
    "/workspace/checkpoint/"
    "OLMoE3-dev-260429-t001_2048d2560a_16L2048M1536S_40E4K1S_p1/step112000"
)
DEFAULT_DATASET_PATH = ""

LOCAL_WORK_DIR = "/weka/oe-training-default/ai2-llm/checkpoints/tianhuat/dataset-cache"
LOCAL_SAVE_ROOT = "/workspace/checkpoint"

SEQUENCE_LENGTH = 8192
RANK_MICROBATCH_SEQUENCES = 2
GLOBAL_BATCH_SIZE = 8 * RANK_MICROBATCH_SEQUENCES * SEQUENCE_LENGTH

NUM_EXPERTS = 40
TOP_K = 4
D_MODEL = 2048
D_ATTN = 2560
HEAD_DIM = 128
NUM_HEADS = D_ATTN // HEAD_DIM
NUM_KV_HEADS = NUM_HEADS // 2
MOE_HIDDEN_SIZE = 2048
NUM_SHARED_EXPERTS = 1
SHARED_MLP_HIDDEN_SIZE = 1536
DENSE_LAYER_HIDDEN_SIZE = TOP_K * MOE_HIDDEN_SIZE + SHARED_MLP_HIDDEN_SIZE
NUM_LAYERS = 16

EP_DEGREE = 8
PP_DEGREE = 1

LR = 5e-5
WEIGHT_DECAY = 0.0
MAX_STEPS = 100
SAVE_INTERVAL = 1000
EPHEMERAL_SAVE_INTERVAL = 500

COMPILE_MODEL = False
COMPILE_OPTIMIZER = False
USE_PERI_NORM = True
USE_NO_SYNC_EP = True
USE_ROWWISE_A2A = True
USE_FP8 = False
ROWWISE_A2A_NBLOCKS = 256
ATTENTION_BACKEND = AttentionBackendName.flash_4
INTRA_DOCUMENT_MASKING = True
SEED = 2026


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


def _pop_float_override(overrides: List[str], key: str, default: float) -> float:
    return float(_pop_override(overrides, key, default))


def _pop_str_override(overrides: List[str], key: str, default: str) -> str:
    value = _pop_override(overrides, key, default)
    if value in (None, ""):
        return default
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
    work_dir: str,
    save_root: str,
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
        work_dir=work_dir,
        save_folder=os.path.join(save_root, cli_context.run_name),
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
        use_recompute_qkv_prep=False,
    )


def build_model_config(
    common: CommonComponents,
    *,
    attention_backend: AttentionBackendName,
) -> TransformerConfig:
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
        ep_no_sync=USE_NO_SYNC_EP,
        checkpoint_permute_moe_unpermute=False,
        checkpoint_attn=False,
        checkpoint_second_unpermute=False,
        ep_no_sync_share_combine_out=False,
        ep_no_sync_share_dispatch_out=False,
        ep_no_sync_shared_slots=1,
        ep_no_sync_use_rowwise_all_to_all=USE_ROWWISE_A2A,
        ep_no_sync_rowwise_nblocks=ROWWISE_A2A_NBLOCKS,
        ep_no_sync_capacity_factor=1.25,
        rowwise_fp8=MoERowwiseFP8Config(enabled=USE_FP8) if USE_ROWWISE_A2A else None,
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
            uniform_expert_assignment=False,
            random_expert_assignment=False,
            lb_loss_weight=0.01,
            z_loss_weight=1e-3,
            lb_loss_granularity=MoELoadBalancingLossGranularity.instance,
            dtype=dtype,
            normalize_expert_weights=1.0,
            restore_weight_scale=True,
            use_recompute_fp32_cast=False,
        ),
        shared_experts=SharedExpertsConfig(
            d_model=D_MODEL,
            hidden_size=SHARED_MLP_HIDDEN_SIZE,
            num_experts=NUM_SHARED_EXPERTS,
            bias=False,
            dtype=dtype,
        ),
        shared_experts_router=None,
        feed_forward_norm=layer_norm,
    )

    config = MoEFusedV2TransformerConfig(
        name=TransformerType.moe_fused_v2,
        init_seed=SEED,
        d_model=D_MODEL,
        two_batch_overlap=False,
        recompute_each_block=False,
        recompute_all_blocks_by_chunk=False,
        vocab_size=common.tokenizer.padded_vocab_size(),
        n_layers=NUM_LAYERS,
        embed_scale=math.sqrt(D_MODEL),
        embedding_norm=layer_norm,
        block=block,
        lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=dtype),
        init_std=0.01,
        dtype=dtype,
    )
    config.lm_head.loss_implementation = LMLossImplementation.default
    config.block.attention.sliding_window = SlidingWindowAttentionConfig(
        force_full_attention_on_first_layer=False,
        force_full_attention_on_last_layer=True,
        pattern=[2048, -1],
    )
    config.block_overrides = {
        0: TransformerBlockConfig(
            name=TransformerBlockType.peri_norm
            if USE_PERI_NORM
            else TransformerBlockType.default,
            attention=_attention_config(dtype, layer_norm, attention_backend),
            attention_norm=layer_norm,
            feed_forward=FeedForwardConfig(
                hidden_size=DENSE_LAYER_HIDDEN_SIZE,
                bias=False,
                dtype=dtype,
            ),
            feed_forward_moe=None,
            feed_forward_norm=layer_norm,
        )
    }
    return config


def build_train_module_config(
    common: CommonComponents,
    *,
    rank_microbatch_sequences: int,
    lr: float,
    ep_degree: int,
) -> MoEV2TransformerTrainModuleConfig:
    rank_microbatch_size = rank_microbatch_sequences * common.max_sequence_length
    split_points = _split_points(NUM_LAYERS, PP_DEGREE)

    return MoEV2TransformerTrainModuleConfig(
        rank_microbatch_size=rank_microbatch_size,
        max_sequence_length=common.max_sequence_length,
        optim=MoEFusedV2OptimizerConfig(
            lr=lr,
            weight_decay=WEIGHT_DECAY,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(
                    params=[
                        "*embeddings.weight",
                        "*embedding_norm.weight",
                        "*q_norm.weight",
                        "*k_norm.weight",
                        "*input_norm.weight",
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
        ac_config=None,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.ddp,
            reduce_grads_in_fp32=True,
            accumulate_grads_in_fp32=True,
        ),
        ep_config=TransformerExpertParallelConfig(degree=ep_degree) if ep_degree > 1 else None,
        pp_config=(
            TransformerPipelineParallelConfig(
                degree=PP_DEGREE,
                schedule=PipelineScheduleType.custom_interleaved_1F1B,
                use_custom_stage_implementation=True,
                split_points=split_points,
            )
            if PP_DEGREE > 1
            else None
        ),
        float8_config=None,
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=LinearWithWarmup(warmup_fraction=0.03, alpha_f=0.0),
        reset_optimizer_states_on_load=True,
    )


def build_trainer_config(
    common: CommonComponents,
    *,
    load_path: str,
    max_steps: int,
) -> TrainerConfig:
    cancel_check_interval = 10
    return (
        TrainerConfig(
            save_folder=common.save_folder,
            load_path=load_path,
            load_strategy=LoadStrategy.always,
            load_trainer_state=False,
            load_optim_state=False,
            save_overwrite=True,
            checkpointer=CheckpointerConfig(
                save_thread_count=1,
                load_thread_count=8,
                throttle_uploads=True,
            ),
            metrics_collect_interval=1,
            cancel_check_interval=cancel_check_interval,
            max_duration=Duration.steps(max_steps),
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
                cancel_check_interval=cancel_check_interval,
            ),
        )
    )


def finalize_config(config: ExperimentConfig) -> None:
    total_params_in_b = config.model.num_params / 1_000_000_000
    active_params_in_b = config.model.num_active_params / 1_000_000_000
    log.info("Total params: %.2fB, active params: %.2fB", total_params_in_b, active_params_in_b)
    wandb_cb = config.trainer.callbacks.get("wandb")
    if isinstance(wandb_cb, WandBCallback) and wandb_cb.name is not None:
        wandb_cb.name += f"_{active_params_in_b:.2f}@{total_params_in_b:.2f}B"
        wandb_cb.group = config.run_name


def make_config_builder(cli_context: CliContext) -> ExperimentConfig:
    overrides = list(cli_context.overrides)
    dataset_path = _pop_str_override(overrides, "dataset_path", DEFAULT_DATASET_PATH)
    seq_len = _pop_int_override(overrides, "seq_len", SEQUENCE_LENGTH)
    num_nodes = _pop_int_override(overrides, "num_nodes", 1)
    global_batch_size = _pop_int_override(
        overrides, "global_batch_size", 8 * RANK_MICROBATCH_SEQUENCES * seq_len
    )
    rank_microbatch_sequences = _pop_int_override(
        overrides, "rank_microbatch_sequences", RANK_MICROBATCH_SEQUENCES
    )
    max_steps = _pop_int_override(overrides, "max_steps", MAX_STEPS)
    load_path = _pop_str_override(overrides, "load_path", DEFAULT_LOAD_PATH)
    work_dir = _pop_str_override(overrides, "work_dir", LOCAL_WORK_DIR)
    save_root = _pop_str_override(overrides, "save_root", LOCAL_SAVE_ROOT)
    lr = _pop_float_override(overrides, "lr", LR)
    ep_degree = _pop_int_override(overrides, "ep_degree", EP_DEGREE)
    attention_backend = AttentionBackendName(
        str(_pop_override(overrides, "attention_backend", ATTENTION_BACKEND.value))
    )
    intra_document_masking = bool(
        _pop_override(overrides, "intra_document_masking", INTRA_DOCUMENT_MASKING)
    )

    clean_context = replace(cli_context, overrides=overrides)
    if not dataset_path:
        raise ValueError(
            "--dataset-path is required and should point at a packed SFT dataset "
            "with token_ids_part_*.npy and labels_mask_*.npy files."
        )

    config_builder = partial(
        build_config,
        common_config_builder=partial(
            build_local_aware_common_components,
            work_dir=work_dir,
            save_root=save_root,
        ),
        data_config_builder=partial(
            build_sft_data_components,
            dataset_path=dataset_path,
            intra_document_masking=intra_document_masking,
        ),
        model_config_builder=partial(build_model_config, attention_backend=attention_backend),
        train_module_config_builder=partial(
            build_train_module_config,
            rank_microbatch_sequences=rank_microbatch_sequences,
            lr=lr,
            ep_degree=ep_degree,
        ),
        trainer_config_builder=partial(
            build_trainer_config,
            load_path=load_path,
            max_steps=max_steps,
        ),
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
