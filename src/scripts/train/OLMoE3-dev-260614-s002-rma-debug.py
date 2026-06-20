import logging
import math
import os
import socket
from copy import deepcopy
from functools import partial
from pathlib import Path


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

# Keep this before any olmo_core imports because several modules import nvtx at import time.
os.environ.setdefault("NVTX_DISABLE", "1")

import torch

from olmo_core.config import DType
from olmo_core.data.composable import ComposableDataLoaderConfig, RandomInstanceSourceConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.parallel.pipeline_parallel import PipelineScheduleType
from olmo_core.internal.experiment import (
    CliContext,
    CommonComponents,
    DataComponents,
    ExperimentConfig,
    build_config,
    main,
)
from olmo_core.nn.attention import AttentionBackendName, AttentionConfig, AttentionType
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlockConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig, LMLossImplementation
from olmo_core.nn.moe import (
    MoELoadBalancingLossGranularity,
    MoERouterGatingFunction,
)
from olmo_core.nn.moe.v2.fp8 import MoERowwiseFP8Config
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig
from olmo_core.nn.rope import RoPEConfig, RoPEType
from olmo_core.nn.transformer import OLMoDDPModelConfig, TransformerBlockType, TransformerType
from olmo_core.optim import OLMoDDPOptimizerConfig
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import CheckpointerCallback, WandBCallback
from olmo_core.train.common import LoadStrategy
from olmo_core.train.train_module import (
    OLMoDDPTrainModuleConfig,
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
)
from olmo_core.train.train_module.transformer import TransformerPipelineParallelConfig

log = logging.getLogger(__name__)


def _get_split_points(original_num_layers: int, num_stages: int, minus_last_stage: int = 0):
    if original_num_layers % num_stages != 0:
        raise ValueError(
            f"original_num_layers ({original_num_layers}) must be divisible by num_stages ({num_stages})"
        )
    layers_per_stage = original_num_layers // num_stages
    new_num_layers = original_num_layers - minus_last_stage
    split_points = [i * layers_per_stage for i in range(1, num_stages)]
    return new_num_layers, split_points


WORK_DIR = "/workspace"
SEED = 2026

SEQUENCE_LENGTH = 512
D_MODEL = 4096
D_ATTN = D_MODEL
HEAD_DIM = 64
NUM_HEAD = D_ATTN // HEAD_DIM
NUM_KV_HEAD = max(1, NUM_HEAD // 4)

NUM_EXPERTS = 8
TOP_K = 2
ORIGINAL_TOP_K = None
MOE_HIDDEN_SIZE = 512
NUM_SHARED_EXPERTS = 1
SHARED_MLP_HIDDEN_SIZE = 512
DENSE_LAYER_MLP = TOP_K * MOE_HIDDEN_SIZE + SHARED_MLP_HIDDEN_SIZE * NUM_SHARED_EXPERTS

EP_DIM = 2
PP_DIM = 2
ORIGINAL_NUM_LAYERS = 8
if PP_DIM > 1:
    NUM_LAYERS, SPLIT_POINTS = _get_split_points(
        ORIGINAL_NUM_LAYERS,
        PP_DIM * 2,
        minus_last_stage=0,
    )
else:
    NUM_LAYERS = ORIGINAL_NUM_LAYERS
    SPLIT_POINTS = None

MICRO_BSZ = 1
GLOBAL_BATCH_SIZE_SEQ = 16
GLOBAL_BATCH_SIZE = GLOBAL_BATCH_SIZE_SEQ * SEQUENCE_LENGTH
MAX_STEPS = 5
RANDOM_NUM_INSTANCES = 4096

P2P_BACKEND = "nccl_rma_ack"
USE_COMPILE = False
USE_NO_SYNC_EP = True
USE_ROWWISE_A2A = True
USE_RANDOM_ROUTING = True
USE_FP8 = False
USE_PERI_NORM = True
GRAD_ACC_IN_FP32 = True
GRAD_REDUCE_IN_FP32 = True
ROWWISE_A2A_NBLOCKS = 256

assert D_ATTN % HEAD_DIM == 0, "D_ATTN must be divisible by HEAD_DIM"
assert NUM_HEAD > 0, "NUM_HEAD must be positive"
assert NUM_EXPERTS >= TOP_K, "NUM_EXPERTS must be >= TOP_K"
assert NUM_EXPERTS % EP_DIM == 0, "NUM_EXPERTS must be divisible by EP_DIM"

torch.set_float32_matmul_precision("high")


def build_debug_common(
    cli_context: CliContext,
    *,
    tokenizer,
    global_batch_size: int,
    max_sequence_length: int,
    **_: object,
) -> CommonComponents:
    work_dir = str(Path(WORK_DIR) / "tmp" / "olmo-rma-debug-work" / cli_context.run_name)
    save_folder = str(Path(WORK_DIR) / "checkpoint" / cli_context.run_name)
    return CommonComponents(
        run_name=cli_context.run_name,
        root_dir=WORK_DIR,
        work_dir=work_dir,
        save_folder=save_folder,
        launch=None,
        tokenizer=tokenizer,
        max_sequence_length=max_sequence_length,
        global_batch_size=global_batch_size,
    )


def _layer_norm_config(dtype: DType) -> LayerNormConfig:
    return LayerNormConfig(
        name=LayerNormType.rms,
        eps=1e-6,
        bias=False,
        dtype=dtype,
    )


def _attention_config(layer_norm: LayerNormConfig, dtype: DType) -> AttentionConfig:
    return AttentionConfig(
        name=AttentionType.default,
        n_heads=NUM_HEAD,
        n_kv_heads=NUM_KV_HEAD,
        bias=False,
        rope=RoPEConfig(name=RoPEType.default, theta=500_000, scaling=None, full_precision=True),
        qk_norm=layer_norm,
        backend=AttentionBackendName.torch,
        use_head_qk_norm=True,
        dtype=dtype,
        d_attn=D_ATTN,
    )


def _rowwise_fp8_config() -> MoERowwiseFP8Config | None:
    if USE_ROWWISE_A2A and USE_FP8:
        return MoERowwiseFP8Config(enabled=True, fused_autograd_recompute_swiglu=False)
    return None


def build_model_config(common: CommonComponents) -> OLMoDDPModelConfig:
    dtype = DType.float32
    layer_norm = _layer_norm_config(dtype)
    attention = _attention_config(layer_norm, dtype)

    moe_block = OLMoDDPTransformerBlockConfig(
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
        rowwise_fp8=_rowwise_fp8_config(),
        attention=attention,
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
            random_expert_assignment=USE_RANDOM_ROUTING,
            lb_loss_weight=0.01,
            z_loss_weight=1e-5,
            lb_loss_granularity=MoELoadBalancingLossGranularity.instance,
            dtype=dtype,
            normalize_expert_weights=1.0,
            restore_weight_scale=True,
            use_recompute_fp32_cast=False,
            original_top_k=ORIGINAL_TOP_K,
        ),
        shared_experts=(
            SharedExpertsConfig(
                d_model=D_MODEL,
                hidden_size=SHARED_MLP_HIDDEN_SIZE,
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
            else None
        ),
        feed_forward_norm=layer_norm,
    )

    dense_block = OLMoDDPTransformerBlockConfig(
        name=TransformerBlockType.moe_fused_v2,
        use_peri_norm=USE_PERI_NORM,
        attention=attention,
        attention_norm=layer_norm,
        routed_experts=None,
        routed_experts_router=None,
        shared_experts=SharedExpertsConfig(
            d_model=D_MODEL,
            hidden_size=DENSE_LAYER_MLP,
            num_experts=1,
            bias=False,
            dtype=dtype,
        ),
        shared_experts_router=None,
        feed_forward_norm=layer_norm,
    )

    config = OLMoDDPModelConfig(
        init_seed=SEED,
        d_model=D_MODEL,
        two_batch_overlap=False,
        recompute_each_block=False,
        recompute_all_blocks_by_chunk=False,
        vocab_size=common.tokenizer.padded_vocab_size(),
        n_layers=NUM_LAYERS,
        embed_scale=math.sqrt(D_MODEL),
        embedding_norm=layer_norm,
        block=moe_block,
        block_overrides={0: deepcopy(dense_block)},
        lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=dtype),
        name=TransformerType.moe_fused_v2,
        init_std=0.01,
        dtype=dtype,
    )
    config.lm_head.loss_implementation = LMLossImplementation.default
    return config


def build_train_module_config(common: CommonComponents) -> OLMoDDPTrainModuleConfig:
    return OLMoDDPTrainModuleConfig(
        rank_microbatch_size=MICRO_BSZ * SEQUENCE_LENGTH,
        max_sequence_length=common.max_sequence_length,
        optim=OLMoDDPOptimizerConfig(
            lr=1e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
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
        pp_config=(
            TransformerPipelineParallelConfig(
                degree=PP_DIM,
                schedule=PipelineScheduleType.custom_interleaved_1F1B,
                use_custom_stage_implementation=True,
                p2p_use_separate_group=True,
                p2p_backend=P2P_BACKEND,
                split_points=SPLIT_POINTS,
            )
            if PP_DIM > 1
            else None
        ),
        float8_config=None,
        z_loss_multiplier=1e-4,
        max_grad_norm=1.0,
        scheduler=None,
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    return (
        TrainerConfig(
            save_folder=common.save_folder,
            work_dir=str(Path(common.work_dir) / "trainer"),
            save_overwrite=True,
            load_strategy=LoadStrategy.never,
            metrics_collect_interval=1,
            cancel_check_interval=1,
            max_duration=Duration.steps(MAX_STEPS),
            no_checkpoints=True,
            no_evals=True,
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=None,
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
                enabled=False,
                cancel_check_interval=1,
            ),
        )
    )


def build_data_components(common: CommonComponents) -> DataComponents:
    return DataComponents(
        dataset=[
            RandomInstanceSourceConfig(
                tokenizer=common.tokenizer,
                sequence_length=SEQUENCE_LENGTH,
                max_sequence_length=SEQUENCE_LENGTH,
                avg_document_length=SEQUENCE_LENGTH,
                num_instances=RANDOM_NUM_INSTANCES,
                seed=SEED,
                label="random-debug",
            )
        ],
        data_loader=ComposableDataLoaderConfig(
            tokenizer=common.tokenizer,
            global_batch_size=common.global_batch_size,
            work_dir=str(Path(common.work_dir) / "data"),
            shuffle=False,
            num_workers=0,
            prefetch_factor=None,
            display_source_visualization=False,
        ),
    )


def finalize_config(config: ExperimentConfig) -> None:
    log.info(
        "s002 RMA debug config: backend=%s, PP=%s, EP=%s, layers=%s, split_points=%s, "
        "d_model=%s, d_attn=%s, seq_len=%s, global_batch_seq=%s, micro_bsz=%s, rowwise_a2a=%s, random_routing=%s",
        P2P_BACKEND,
        PP_DIM,
        EP_DIM,
        NUM_LAYERS,
        SPLIT_POINTS,
        D_MODEL,
        D_ATTN,
        SEQUENCE_LENGTH,
        GLOBAL_BATCH_SIZE_SEQ,
        MICRO_BSZ,
        USE_ROWWISE_A2A,
        USE_RANDOM_ROUTING,
    )
    log.info(
        "Total params: %.3fB, active params: %.3fB",
        config.model.num_params / 1_000_000_000,
        config.model.num_active_params / 1_000_000_000,
    )


if __name__ == "__main__":
    config_builder = partial(
        build_config,
        common_config_builder=build_debug_common,
        data_config_builder=build_data_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        finalize_config=finalize_config,
        global_batch_size=GLOBAL_BATCH_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        include_default_evals=False,
        flight_recorder=False,
    )
    main(config_builder=config_builder)
