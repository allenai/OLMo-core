"""
Mid-training pilot for the OLMoE3-dev-260429-t001 MoE V2 checkpoint.

The model shape mirrors ``src/scripts/train/OLMoE3-dev-260429-t001.py`` so it
can load the existing pretraining checkpoint. The data path follows the dense
OLMo3 mid-training recipe by using the OLMo3 32B source-mixture YAML.

Example:
  PYTHONPATH=src torchrun --standalone --nproc-per-node=8 \
      src/scripts/train/midtraining/OLMoE3-dev-260429-t001-midtraining.py train \
      OLMoE3-dev-260429-t001-midtrain-pilot local
"""

import logging
import math
import os
import random
from dataclasses import replace
from functools import partial
from glob import glob
from pathlib import Path
from typing import Any, List, Optional, Tuple, cast

import torch
import yaml

# Keep this before olmo_core imports. Several modules import nvtx at import time.
os.environ.setdefault("NVTX_DISABLE", "1")
# Only the flan source currently falls back to GCS on local runs. This avoids
# repeated GCE metadata probes on non-GCE hosts while leaving ADC/public access intact.
os.environ.setdefault("NO_GCE_CHECK", "true")
# Keep source-mixture index preparation from spawning one process per visible CPU.
os.environ.setdefault("OLMO_DATA_PREP_WORKERS", "8")

from olmo_core.config import DType
from olmo_core.data import (
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.data.source_mixture import SourceMixtureDatasetConfig, SourceMixtureList
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

torch.set_float32_matmul_precision("high")

SRC_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_MIX_YAML = str(
    SRC_ROOT / "olmo_core/data/source_mixtures/OLMo3-32B-midtraining-modelnamefilter.yaml"
)
DEFAULT_LOAD_PATH = (
    "/workspace/checkpoint/"
    "OLMoE3-dev-260429-t001_2048d2560a_16L2048M1536S_40E4K1S_p1/step112000"
)

LOCAL_WORK_DIR = "/weka/oe-training-default/ai2-llm/checkpoints/OLMoE3-dev-260429-t001-midtraining/dataset-cache"
LOCAL_SAVE_ROOT = "/workspace/checkpoint"
SOURCE_DATA_BASE_DIR = "/weka/oe-training-default/ai2-llm"

SEQUENCE_LENGTH = 8192
RANK_MICROBATCH_SEQUENCES = 2
GLOBAL_BATCH_SIZE = 8 * RANK_MICROBATCH_SEQUENCES * SEQUENCE_LENGTH
MAX_STEPS = 100
REQUESTED_TOKENS = GLOBAL_BATCH_SIZE * MAX_STEPS

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
WEIGHT_DECAY = 0.1
SAVE_INTERVAL = 1000
EPHEMERAL_SAVE_INTERVAL = 500

COMPILE_MODEL = False
COMPILE_OPTIMIZER = False
USE_PERI_NORM = True
USE_NO_SYNC_EP = True
USE_ROWWISE_A2A = True
USE_FP8 = False
ROWWISE_A2A_NBLOCKS = 256
ATTENTION_BACKEND = AttentionBackendName.te
INTRA_DOCUMENT_MASKING = False
INCLUDE_INSTANCE_FILTER = True
LOAD_OPTIM_STATE = False
SEED = 2026
SOURCE_MIX_PROCESSES = 16
SOURCE_DATA_LOCAL_MISSING_FALLBACK = True
SOURCE_MAX_PATHS_PER_SOURCE = 8


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


def _pop_bool_override(overrides: List[str], key: str, default: bool) -> bool:
    value = _pop_override(overrides, key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _pop_int_override(overrides: List[str], key: str, default: int) -> int:
    return int(_pop_override(overrides, key, default))


def _pop_float_override(overrides: List[str], key: str, default: float) -> float:
    return float(_pop_override(overrides, key, default))


def _pop_str_override(overrides: List[str], key: str, default: str) -> str:
    value = _pop_override(overrides, key, default)
    if value in (None, ""):
        return default
    return str(value)


def _pop_optional_str_override(
    overrides: List[str], key: str, default: Optional[str]
) -> Optional[str]:
    value = _pop_override(overrides, key, default)
    if value is None:
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


def _local_path_matches(path: str) -> bool:
    if "*" in path:
        return bool(glob(path, recursive=True))
    return Path(path).exists()


def _rewrite_source_paths(
    source_list: SourceMixtureList,
    source_data_base_dir: Optional[str],
    *,
    local_missing_fallback: bool,
) -> None:
    if not source_data_base_dir:
        return

    clean_base = source_data_base_dir.rstrip("/")
    if not Path(clean_base).exists():
        log.warning("Source data base dir '%s' does not exist; keeping source YAML paths", clean_base)
        return

    for source in source_list.sources:
        rewritten_paths = []
        for path in source.paths:
            if path.startswith("gs://ai2-llm/"):
                rewritten_path = path.replace("gs://ai2-llm", clean_base, 1)
            elif path.startswith("s3://ai2-llm/"):
                rewritten_path = path.replace("s3://ai2-llm", clean_base, 1)
            else:
                rewritten_path = path

            if (
                rewritten_path != path
                and local_missing_fallback
                and not _local_path_matches(rewritten_path)
            ):
                log.warning(
                    "No local matches for '%s'; falling back to source YAML path '%s'",
                    rewritten_path,
                    path,
                )
                rewritten_paths.append(path)
            else:
                rewritten_paths.append(rewritten_path)
        source.paths = rewritten_paths
        source._resolved_paths = None


def _limit_source_paths(
    source_list: SourceMixtureList,
    max_paths_per_source: int,
    *,
    seed: int,
) -> None:
    if max_paths_per_source <= 0:
        return

    for source in source_list.sources:
        resolved_paths = list(source.resolved_paths)
        if len(resolved_paths) <= max_paths_per_source:
            continue

        rng = random.Random(f"{seed}:{source.source_name}")
        selected_indices = sorted(rng.sample(range(len(resolved_paths)), max_paths_per_source))
        selected_paths = [resolved_paths[idx] for idx in selected_indices]
        log.info(
            "Limiting source '%s' from %d to %d resolved paths for this pilot run",
            source.source_name,
            len(resolved_paths),
            len(selected_paths),
        )
        source.paths = selected_paths
        source._resolved_paths = selected_paths


def build_midtraining_data_components(
    common: CommonComponents,
    *,
    source_mix_yaml: str,
    source_data_base_dir: Optional[str],
    source_data_local_missing_fallback: bool,
    source_max_paths_per_source: int,
    requested_tokens: int,
    source_mix_processes: int,
    intra_document_masking: bool,
    include_instance_filter: bool,
) -> DataComponents:
    source_list = SourceMixtureList.from_yaml(source_mix_yaml)
    _rewrite_source_paths(
        source_list,
        source_data_base_dir,
        local_missing_fallback=source_data_local_missing_fallback,
    )
    _limit_source_paths(source_list, source_max_paths_per_source, seed=SEED)
    source_list.validate()

    dataset = NumpyFSLDatasetConfig.from_src_mix(
        src_mix=SourceMixtureDatasetConfig(
            source_list=source_list,
            requested_tokens=requested_tokens,
            global_batch_size=common.global_batch_size,
            processes=source_mix_processes,
            seed=SEED,
            render_tables=False,
            quiet=True,
        ),
        tokenizer=common.tokenizer,
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
    data_loader = NumpyDataLoaderConfig(
        global_batch_size=common.global_batch_size,
        seed=34521,
        num_workers=4,
        ignore_fingerprint_mismatch=True,
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
    load_optim_state: bool,
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
        scheduler=LinearWithWarmup(warmup=0, alpha_f=0.0),
        reset_optimizer_states_on_load=not load_optim_state,
    )


def build_trainer_config(
    common: CommonComponents,
    *,
    load_path: str,
    max_steps: int,
    load_optim_state: bool,
) -> TrainerConfig:
    cancel_check_interval = 10
    return (
        TrainerConfig(
            save_folder=common.save_folder,
            load_path=load_path,
            load_strategy=LoadStrategy.always,
            load_trainer_state=False,
            load_optim_state=load_optim_state,
            save_overwrite=True,
            checkpointer=CheckpointerConfig(
                save_thread_count=1,
                load_thread_count=8,
                throttle_uploads=True,
            ),
            metrics_collect_interval=10,
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
                project="olmoe-midtraining",
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
        wandb_cb.name += f"_{active_params_in_b:.2f}@{total_params_in_b:.2f}B-midtrain"
        wandb_cb.group = config.run_name


def make_config_builder(cli_context: CliContext) -> ExperimentConfig:
    overrides = list(cli_context.overrides)
    seq_len = _pop_int_override(overrides, "seq_len", SEQUENCE_LENGTH)
    num_nodes = _pop_int_override(overrides, "num_nodes", 1)
    global_batch_size = _pop_int_override(
        overrides, "global_batch_size", 8 * RANK_MICROBATCH_SEQUENCES * seq_len
    )
    rank_microbatch_sequences = _pop_int_override(
        overrides, "rank_microbatch_sequences", RANK_MICROBATCH_SEQUENCES
    )
    max_steps = _pop_int_override(overrides, "max_steps", MAX_STEPS)
    requested_tokens = _pop_int_override(
        overrides, "requested_tokens", global_batch_size * max_steps
    )
    load_path = _pop_str_override(overrides, "load_path", DEFAULT_LOAD_PATH)
    source_mix_yaml = _pop_str_override(overrides, "source_mix_yaml", DEFAULT_SOURCE_MIX_YAML)
    source_data_base_dir = _pop_optional_str_override(
        overrides, "source_data_base_dir", SOURCE_DATA_BASE_DIR
    )
    source_data_local_missing_fallback = _pop_bool_override(
        overrides,
        "source_data_local_missing_fallback",
        SOURCE_DATA_LOCAL_MISSING_FALLBACK,
    )
    source_max_paths_per_source = _pop_int_override(
        overrides, "source_max_paths_per_source", SOURCE_MAX_PATHS_PER_SOURCE
    )
    work_dir = _pop_str_override(overrides, "work_dir", LOCAL_WORK_DIR)
    save_root = _pop_str_override(overrides, "save_root", LOCAL_SAVE_ROOT)
    lr = _pop_float_override(overrides, "lr", LR)
    ep_degree = _pop_int_override(overrides, "ep_degree", EP_DEGREE)
    source_mix_processes = _pop_int_override(
        overrides, "source_mix_processes", SOURCE_MIX_PROCESSES
    )
    attention_backend = AttentionBackendName(
        str(_pop_override(overrides, "attention_backend", ATTENTION_BACKEND.value))
    )
    intra_document_masking = _pop_bool_override(
        overrides, "intra_document_masking", INTRA_DOCUMENT_MASKING
    )
    include_instance_filter = _pop_bool_override(
        overrides, "include_instance_filter", INCLUDE_INSTANCE_FILTER
    )
    load_optim_state = _pop_bool_override(overrides, "load_optim_state", LOAD_OPTIM_STATE)

    if not load_path:
        raise ValueError("--load-path is required for mid-training continuation.")
    if requested_tokens < global_batch_size * max_steps:
        raise ValueError(
            "requested_tokens must cover the requested run: "
            f"{requested_tokens} < {global_batch_size * max_steps}"
        )

    clean_context = replace(cli_context, overrides=overrides)
    config_builder = partial(
        build_config,
        common_config_builder=partial(
            build_local_aware_common_components,
            work_dir=work_dir,
            save_root=save_root,
        ),
        data_config_builder=partial(
            build_midtraining_data_components,
            source_mix_yaml=source_mix_yaml,
            source_data_base_dir=source_data_base_dir,
            source_data_local_missing_fallback=source_data_local_missing_fallback,
            source_max_paths_per_source=source_max_paths_per_source,
            requested_tokens=requested_tokens,
            source_mix_processes=source_mix_processes,
            intra_document_masking=intra_document_masking,
            include_instance_filter=include_instance_filter,
        ),
        model_config_builder=partial(build_model_config, attention_backend=attention_backend),
        train_module_config_builder=partial(
            build_train_module_config,
            rank_microbatch_sequences=rank_microbatch_sequences,
            lr=lr,
            ep_degree=ep_degree,
            load_optim_state=load_optim_state,
        ),
        trainer_config_builder=partial(
            build_trainer_config,
            load_path=load_path,
            max_steps=max_steps,
            load_optim_state=load_optim_state,
        ),
        finalize_config=finalize_config,
        tokenizer=TokenizerConfig.dolma2(),
        global_batch_size=global_batch_size,
        max_sequence_length=seq_len,
        num_nodes=num_nodes,
        include_default_evals=False,
        flight_recorder=True,
    )
    return cast(ExperimentConfig, config_builder(clean_context))


if __name__ == "__main__":
    main(config_builder=make_config_builder)
