"""
Tiny active-275M MoE ladder script derived from ``OLMoE3-dev-260401-launchable.py``.

The model is configured for ~278M active params / ~1.13B total params. By default
it trains for 1x Chinchilla according to the repo default: active non-embedding
params. Use ``--chinchilla-multiple`` for longer runs.
The data mix is loaded from S3, so the Beaker workspace must have AWS env-key
secrets available. This script also unsets ``S3_PROFILE`` when ``--data-root`` is
an S3 URI and env AWS keys are present, so boto3 uses the Beaker-provided keys
instead of looking for a named profile file.

Before launching, make sure these Beaker secrets exist in
``ai2/OLMo-3-moe-experiments``:

- ``AWS_ACCESS_KEY_ID``
- ``AWS_SECRET_ACCESS_KEY``
- ``jacobm_WANDB_API_KEY`` (mapped into the job as ``WANDB_API_KEY``)

The MoE EP path needs the symmetric-memory CUDA extension. The launch below sets
``OLMO_SYMM_VDEV2D_AUTO_BUILD=1`` so the job builds it on the node.

Run from a pushed branch with ``uv``::

    uv run --extra dev --extra beaker python -m olmo_core.launch.beaker \\
      --name=olmoe3-tiny-275m-cx1-lr3e-4 \\
      --cluster ai2/titan \\
      --nodes 1 \\
      --gpus 8 \\
      --weka oe-training-default \\
      --beaker-image=tianhuat/olmo-core-torch211-2404-cu128 \\
      --workspace=ai2/OLMo-3-moe-experiments \\
      --budget=ai2/oe-other \\
      --priority urgent \\
      --env OLMO_SYMM_VDEV2D_AUTO_BUILD=1 \\
      --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY WANDB_API_KEY=jacobm_WANDB_API_KEY \\
      -- \\
      python src/scripts/train/jacobm_olmoe_ladder/tiny_275m.py \\
        --save-folder=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx1-lr3e-4 \\
        --name=olmoe3-tiny-275m-cx1-lr3e-4 \\
        --data-root=s3://ai2-llm \\
        --lr=3e-4 \\
        --chinchilla-multiple=1 \\
        --tag=lr3e-4-cx1

Eval over saved checkpoints (no training): pass ``--eval-checkpoints`` (one or more
paths; globs supported). When set, the train module / trainer are built with
``eval_only=True`` and ``Trainer.eval_checkpoints`` runs instead of ``Trainer.fit``.
The script also forces ``MICRO_BSZ=4`` and ``EP_DIM=1`` in eval mode (mirrors the
``IN_EVAL_MODE`` branch in the original test-refactor script)::

    python -m olmo_core.launch.beaker --name=olmoe3-dev-eval --cluster ai2/jupiter \\
      --nodes 1 --gpus 8 --weka oe-training-default \\
      -- \\
      python src/scripts/train/OLMoE3-tiny-275m-active-smoketest.py \\
        --save-folder=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-dev-eval \\
        --name=olmoe3-dev-eval \\
        --data-root=/weka/oe-training-default/ai2-llm \\
        --eval-checkpoints "/weka/.../checkpoints/jacobm/olmoe3/some-run/step*"
"""

import argparse
import logging
import math
import os
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
    ComposableSchedulerStage,
    ComposableSchedulerStageType,
    OverrideDecay,
)
from olmo_core.script_utils import get_cli_parser, main
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


@dataclass(frozen=True)
class ModelSizeSpec:
    d_model: int
    d_attn: int
    n_layers: int
    head_dim: int
    num_experts: int
    top_k: int
    moe_hidden_size: int
    num_shared_experts: int
    shared_mlp_hidden_size: int
    dense_prefix_layers: int
    dense_layer_mlp: int


MODEL_SIZE_SPECS = {
    "275m": ModelSizeSpec(
        d_model=768,
        d_attn=1024,
        n_layers=12,
        head_dim=128,
        num_experts=48,
        top_k=4,
        moe_hidden_size=768,
        num_shared_experts=1,
        shared_mlp_hidden_size=384,
        dense_prefix_layers=1,
        dense_layer_mlp=3456,
    ),
    "810m": ModelSizeSpec(
        d_model=1280,
        d_attn=1536,
        n_layers=20,
        head_dim=128,
        num_experts=48,
        top_k=4,
        moe_hidden_size=1280,
        num_shared_experts=1,
        shared_mlp_hidden_size=640,
        dense_prefix_layers=1,
        dense_layer_mlp=5760,
    ),
    "1p2b": ModelSizeSpec(
        d_model=1536,
        d_attn=2048,
        n_layers=22,
        head_dim=128,
        num_experts=48,
        top_k=4,
        moe_hidden_size=1536,
        num_shared_experts=1,
        shared_mlp_hidden_size=768,
        dense_prefix_layers=1,
        dense_layer_mlp=6912,
    ),
}


def prepare_s3_environment(opts: argparse.Namespace) -> None:
    if (
        str(opts.data_root).startswith("s3://")
        and os.getenv("AWS_ACCESS_KEY_ID")
        and os.getenv("AWS_SECRET_ACCESS_KEY")
    ):
        os.environ.pop("S3_PROFILE", None)


def get_parser() -> argparse.ArgumentParser:
    parser = get_cli_parser()
    parser.add_argument(
        "--model-size",
        choices=sorted(MODEL_SIZE_SPECS),
        default="275m",
        help="MoE A0 active-parameter scale to train.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LR,
        help="Peak learning rate for all parameter groups before group-specific overrides.",
    )
    parser.add_argument(
        "--chinchilla-multiple",
        type=float,
        default=CHINCHILLA_MULTIPLE,
        help="Training duration multiple, based on active non-embedding parameters.",
    )
    parser.add_argument(
        "--global-batch-size-seq",
        type=int,
        default=GLOBAL_BATCH_SIZE_SEQ,
        help="Global batch size in sequences. Tokens per step are this times sequence length.",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=REF_NUM_NODES,
        help="Number of training nodes. Used to derive gradient accumulation from global batch size.",
    )
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        default=REF_GPUS_PER_NODE,
        help="Number of training GPUs per node. Used to derive gradient accumulation from global batch size.",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=MICRO_BSZ,
        help="Per-rank microbatch size in sequences.",
    )
    parser.add_argument(
        "--ep-dim",
        type=int,
        default=EP_DIM,
        help="Expert-parallel degree. Use 1 to disable expert parallelism.",
    )
    parser.add_argument(
        "--use-rowwise-a2a",
        action=argparse.BooleanOptionalAction,
        default=USE_ROWWISE_A2A,
        help="Use the rowwise all-to-all EP path. Disable for the slower dropless EP sanity check.",
    )
    parser.add_argument(
        "--warmup-fraction",
        type=float,
        default=SCHED_WARMUP_FRACTION,
        help="Fraction of total training tokens used for linear LR warmup.",
    )
    parser.add_argument(
        "--ladder-evals",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable scaling-ladders-style in-loop evals during training. "
            "Eval-checkpoint mode always enables these callbacks."
        ),
    )
    parser.add_argument(
        "--eval-task-set",
        type=str,
        default="fast",
        help="Task group to use for downstream ladder evals.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=EVAL_INTERVAL,
        help="Step interval for ladder evals during training/eval-checkpoint runs.",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help=(
            "Permanent checkpoint interval in steps. Defaults to only the final checkpoint; "
            "periodic resume checkpoints are controlled by --ephemeral-save-interval."
        ),
    )
    parser.add_argument(
        "--ephemeral-save-interval",
        type=int,
        default=SAVE_INTERVAL,
        help=(
            "Temporary checkpoint interval in steps. OLMo Core keeps only the latest "
            "ephemeral checkpoint, so this gives resume safety without retaining every save."
        ),
    )
    parser.add_argument(
        "--pre-train-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to save a persistent step-0 checkpoint before training.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=TAG,
        help="Short tag appended to WandB run/group names.",
    )
    return parser


def configure_sweep_hparams(opts: argparse.Namespace, sequence_length: int, max_duration_tokens: int) -> None:
    global CHINCHILLA_MULTIPLE
    global REF_NUM_NODES, REF_GPUS_PER_NODE, MICRO_BSZ, EP_DIM
    global GLOBAL_BATCH_SIZE_SEQ, GLOBAL_BATCH_SIZE, NUM_MICRO_BATCHES, GLOBAL_BATCH_TOKENS_IN_M
    global SCHED_WARMUP_FRACTION, SCHED_WARMUP_TOKENS
    global LR, EXPERT_LR, TAG, MONKEY_PATCH_DECAY_DURATION_TOKENS
    global USE_ROWWISE_A2A

    if opts.chinchilla_multiple <= 0:
        raise ValueError("--chinchilla-multiple must be > 0")
    if opts.lr <= 0:
        raise ValueError("--lr must be > 0")
    if opts.global_batch_size_seq <= 0:
        raise ValueError("--global-batch-size-seq must be > 0")
    if opts.num_nodes <= 0:
        raise ValueError("--num-nodes must be > 0")
    if opts.gpus_per_node <= 0:
        raise ValueError("--gpus-per-node must be > 0")
    if opts.micro_batch_size <= 0:
        raise ValueError("--micro-batch-size must be > 0")
    if opts.ep_dim <= 0:
        raise ValueError("--ep-dim must be > 0")
    if not 0 < opts.warmup_fraction < 1:
        raise ValueError("--warmup-fraction must be between 0 and 1")
    if opts.save_interval is not None and opts.save_interval < 1:
        raise ValueError("--save-interval must be >= 1 when set")
    if opts.ephemeral_save_interval is not None and opts.ephemeral_save_interval < 1:
        raise ValueError("--ephemeral-save-interval must be >= 1 when set")
    if (
        opts.save_interval is not None
        and opts.ephemeral_save_interval is not None
        and opts.ephemeral_save_interval >= opts.save_interval
    ):
        raise ValueError("--ephemeral-save-interval must be less than --save-interval when both are set")

    CHINCHILLA_MULTIPLE = opts.chinchilla_multiple
    REF_NUM_NODES = opts.num_nodes
    REF_GPUS_PER_NODE = opts.gpus_per_node
    MICRO_BSZ = opts.micro_batch_size
    EP_DIM = opts.ep_dim
    USE_ROWWISE_A2A = opts.use_rowwise_a2a
    GLOBAL_BATCH_SIZE_SEQ = opts.global_batch_size_seq
    GLOBAL_BATCH_SIZE = GLOBAL_BATCH_SIZE_SEQ * sequence_length
    NUM_MICRO_BATCHES = GLOBAL_BATCH_SIZE_SEQ // (REF_NUM_NODES * REF_GPUS_PER_NODE) // MICRO_BSZ * PP_DIM
    if NUM_MICRO_BATCHES <= 0:
        raise ValueError("global batch size is too small for the configured node/GPU/microbatch setup")
    if GLOBAL_BATCH_SIZE_SEQ % (REF_NUM_NODES * REF_GPUS_PER_NODE * MICRO_BSZ) != 0:
        raise ValueError("--global-batch-size-seq must divide evenly across nodes, GPUs per node, and MICRO_BSZ")
    GLOBAL_BATCH_TOKENS_IN_M = GLOBAL_BATCH_SIZE // 1024 // 1024

    SCHED_WARMUP_FRACTION = opts.warmup_fraction
    SCHED_WARMUP_TOKENS = int(((max_duration_tokens * SCHED_WARMUP_FRACTION) // GLOBAL_BATCH_SIZE) * GLOBAL_BATCH_SIZE)
    SCHED_WARMUP_TOKENS = max(GLOBAL_BATCH_SIZE, SCHED_WARMUP_TOKENS)

    LR = opts.lr
    EXPERT_LR = LR
    TAG = opts.tag
    MONKEY_PATCH_DECAY_DURATION_TOKENS = int((200e9 // GLOBAL_BATCH_SIZE) * GLOBAL_BATCH_SIZE)


def configure_model_size(opts: argparse.Namespace) -> None:
    global NUM_EXPERTS, TOP_K, D_MODEL, D_ATTN, HEAD_DIM, NUM_HEAD, NUM_KV_HEAD
    global MOE_HIDDEN_SIZE, NUM_SHARED_EXPERTS, SHARED_MLP_HIDDEN_SIZE
    global EFFECTIVE_MLP, MLP_RATIO, DENSE_LAYER_MLP, NUM_LAYERS, SPLIT_POINTS

    spec = MODEL_SIZE_SPECS[opts.model_size]
    if spec.dense_prefix_layers != 1:
        raise ValueError("Only one dense prefix layer is currently implemented")

    NUM_EXPERTS = spec.num_experts
    TOP_K = spec.top_k
    D_MODEL = spec.d_model
    D_ATTN = spec.d_attn
    HEAD_DIM = spec.head_dim
    NUM_HEAD = D_ATTN // HEAD_DIM
    NUM_KV_HEAD = NUM_HEAD // 2
    MOE_HIDDEN_SIZE = spec.moe_hidden_size
    NUM_SHARED_EXPERTS = spec.num_shared_experts
    SHARED_MLP_HIDDEN_SIZE = spec.shared_mlp_hidden_size
    EFFECTIVE_MLP = MOE_HIDDEN_SIZE * TOP_K + SHARED_MLP_HIDDEN_SIZE * NUM_SHARED_EXPERTS
    MLP_RATIO = EFFECTIVE_MLP / D_MODEL
    DENSE_LAYER_MLP = spec.dense_layer_mlp
    NUM_LAYERS = spec.n_layers

    if PP_DIM > 1:
        _, SPLIT_POINTS = _get_split_points(NUM_LAYERS, PP_DIM * 2, minus_last_stage=1)
    else:
        SPLIT_POINTS = None


def consume_script_overrides(opts: argparse.Namespace, overrides: List[str]) -> List[str]:
    """Consume script-level key=value overrides before Config.merge sees them."""
    filtered_overrides: List[str] = []
    for override in overrides:
        if override.startswith("model_size="):
            opts.model_size = override.split("=", 1)[1]
            continue
        filtered_overrides.append(override)
    if opts.model_size not in MODEL_SIZE_SPECS:
        raise ValueError(f"--model-size must be one of {sorted(MODEL_SIZE_SPECS)}")
    return filtered_overrides


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
# Hyperparameters.
# -----------------------------------------------------------------------------

DEFAULT_SEQUENCE_LENGTH = 8192

EVAL_INTERVAL = 2000
SAVE_INTERVAL = 1000

NUM_EXPERTS = 48
TOP_K = 4
D_MODEL = 768
D_ATTN = 1024

HEAD_DIM = 128
NUM_HEAD = D_ATTN // HEAD_DIM
NUM_KV_HEAD = NUM_HEAD // 2
MOE_HIDDEN_SIZE = 768
NUM_SHARED_EXPERTS = 1
SHARED_MLP_HIDDEN_SIZE = 384

EFFECTIVE_MLP = MOE_HIDDEN_SIZE * TOP_K + SHARED_MLP_HIDDEN_SIZE * NUM_SHARED_EXPERTS
MLP_RATIO = EFFECTIVE_MLP / D_MODEL

DENSE_LAYER_MLP = TOP_K * MOE_HIDDEN_SIZE + SHARED_MLP_HIDDEN_SIZE * NUM_SHARED_EXPERTS

EP_DIM = 1
PP_DIM = 1

REF_NUM_NODES = 1
REF_GPUS_PER_NODE = 8

# Chinchilla multiple is based on active non-embedding parameters.
CHINCHILLA_MULTIPLE = 1.0
MICRO_BSZ = 1
GLOBAL_BATCH_SIZE_SEQ = (8 * 8) * 4 * 1

GLOBAL_BATCH_SIZE = GLOBAL_BATCH_SIZE_SEQ * DEFAULT_SEQUENCE_LENGTH
NUM_MICRO_BATCHES = GLOBAL_BATCH_SIZE_SEQ // (REF_NUM_NODES * REF_GPUS_PER_NODE) // MICRO_BSZ * PP_DIM
GLOBAL_BATCH_TOKENS_IN_M = GLOBAL_BATCH_SIZE // 1024 // 1024

SCHED_WARMUP_FRACTION = 0.1
SCHED_WARMUP_TOKENS = GLOBAL_BATCH_SIZE
SCHED_FAST_DECAY_TOKENS = int((0e9 // GLOBAL_BATCH_SIZE) * GLOBAL_BATCH_SIZE)
SCHED_MID_FRACTION = 1.0
SCHED_FINAL_FRACTION = 0.1

LR = 3e-4
EXPERT_LR = LR

NUM_LAYERS = 12


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
PRODUCTION_RUN = True

TAG = "cx1"

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


def build_train_module_config(
    sequence_length: int, max_duration_tokens: int, in_eval_mode: bool = False
) -> MoEV2TransformerTrainModuleConfig:
    from olmo_core.optim.moe_optimizer import MoEFusedV2OptimizerConfig

    # Eval-mode overrides: smaller micro-batch and no expert parallelism. Mirrors the
    # IN_EVAL_MODE branch in OLMoE3-dev-260401-test-refactor.py — eval needs less memory
    # per rank and EP/grad-sync setup is wasted when there's no backward pass.
    micro_bsz = 4 if in_eval_mode else MICRO_BSZ
    ep_dim = 1 if in_eval_mode else EP_DIM

    return MoEV2TransformerTrainModuleConfig(
        rank_microbatch_size=micro_bsz * sequence_length,
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
        ep_config=TransformerExpertParallelConfig(degree=ep_dim) if ep_dim != 1 else None,
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
                    duration=max(max_duration_tokens - SCHED_WARMUP_TOKENS, GLOBAL_BATCH_SIZE),
                    shape=ComposableSchedulerStageType.cosine,
                    end_lr_fraction=SCHED_FINAL_FRACTION,
                ),
            ],
            override_decay=(
                OverrideDecay(
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


def build_trainer_config(
    opts: argparse.Namespace,
    tokenizer_config: TokenizerConfig,
    sequence_length: int,
    max_duration_tokens: int,
    in_eval_mode: bool = False,
) -> TrainerConfig:
    from olmo_core.train.checkpoint import CheckpointerConfig

    cancel_check_interval = 10
    config = (
        TrainerConfig(
            save_folder=opts.save_folder,
            save_overwrite=True,
            checkpointer=CheckpointerConfig(
                save_thread_count=3, load_thread_count=2, throttle_uploads=True
            ),
            metrics_collect_interval=2,
            cancel_check_interval=cancel_check_interval,
            max_duration=Duration.tokens(max_duration_tokens),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=opts.save_interval,
                ephemeral_save_interval=opts.ephemeral_save_interval,
                save_async=False,
                pre_train_checkpoint=opts.pre_train_checkpoint,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=opts.name,
                entity="ai2-llm",
                project="jacobm-olmoe-ladder",
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

    if in_eval_mode or opts.ladder_evals:
        config = add_ladder_evals(config, opts, tokenizer_config, sequence_length)

    return config


def add_ladder_evals(
    config: TrainerConfig,
    opts: argparse.Namespace,
    tokenizer_config: TokenizerConfig,
    sequence_length: int,
) -> TrainerConfig:
    # Mirrors scaling-ladders' TrainerConfig.with_recommended_evals(..., task_set="fast"),
    # while keeping data paths explicit for this standalone script.
    from olmo_core.data import DataMix, NumpyPaddedFSLDatasetConfig
    from olmo_core.eval.task_groups import TASK_GROUPS
    from olmo_core.train.callbacks import (
        DownstreamEvaluatorCallbackConfig,
        LMEvaluatorCallbackConfig,
    )

    try:
        tasks = sorted(TASK_GROUPS[opts.eval_task_set])
    except KeyError as e:
        raise ValueError(f"Task set not recognized: {opts.eval_task_set}") from e

    return config.with_callback(
        "downstream_evaluator",
        DownstreamEvaluatorCallbackConfig(
            tasks=tasks,
            tokenizer=tokenizer_config,
            eval_interval=opts.eval_interval,
            eval_on_finish=True,
        ),
    ).with_callback(
        "lm_evaluator",
        LMEvaluatorCallbackConfig(
            eval_dataset=NumpyPaddedFSLDatasetConfig.from_data_mix(
                DataMix.v3_small_ppl_validation,
                mix_base_dir=opts.data_root,
                sequence_length=sequence_length,
                tokenizer=tokenizer_config,
                work_dir=opts.work_dir,
            ),
            eval_interval=opts.eval_interval,
            eval_on_finish=True,
        ),
    )


def build_dataset_config(
    opts: argparse.Namespace,
    tokenizer_config: TokenizerConfig,
    sequence_length: int,
    intra_document_masking: bool = False,
    include_instance_filter: bool = True,
) -> NumpyFSLDatasetConfig:
    return NumpyFSLDatasetConfig.from_data_mix(
        DataMix.OLMo_mix_0925,
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
    active_non_embedding_params_in_B = (
        config.model.num_active_non_embedding_params / 1000 / 1000 / 1000
    )
    log.info(
        f"Total params: {total_params_in_B:.2f}B, Active params: {active_params_in_B:.2f}B, "
        f"Active non-embedding params: {active_non_embedding_params_in_B:.2f}B"
    )
    log.info(
        f"Training for {CHINCHILLA_MULTIPLE:g}x Chinchilla: "
        f"{config.trainer.max_duration.value / 1000 / 1000 / 1000:.2f}B tokens"
    )

    wandb_cb = cast(WandBCallback, config.trainer.callbacks["wandb"])
    assert isinstance(wandb_cb.name, str), "WandB callback name must be initialized"
    wandb_original_name = wandb_cb.name
    wandb_cb.name += f"_{active_params_in_B:.2f}@{total_params_in_B:.2f}B"
    wandb_cb.name += f"_{opts.model_size}_{NUM_LAYERS}L{TOP_K}K{NUM_EXPERTS}N{NUM_SHARED_EXPERTS}S_{TAG}"
    wandb_cb.group = (
        f"{wandb_original_name}_{active_params_in_B:.2f}@{total_params_in_B:.2f}B"
        f"_{opts.model_size}_{NUM_LAYERS}L{TOP_K}K{NUM_EXPERTS}N{NUM_SHARED_EXPERTS}S_{TAG}"
    )


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    prepare_s3_environment(opts)
    overrides = consume_script_overrides(opts, overrides)
    sequence_length = opts.sequence_length or DEFAULT_SEQUENCE_LENGTH
    tokenizer_config = TokenizerConfig.dolma2()
    in_eval_mode = bool(getattr(opts, "eval_checkpoints", None))

    configure_model_size(opts)
    model_config = build_model_config(tokenizer_config)
    max_duration_tokens = Duration.chinchilla_tokens(
        opts.chinchilla_multiple,
        model_params=model_config.num_active_non_embedding_params,
    ).value
    configure_sweep_hparams(opts, sequence_length, max_duration_tokens)
    dataset_config = build_dataset_config(opts, tokenizer_config, sequence_length)
    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=34521,
        num_workers=8,
    )
    train_module_config = build_train_module_config(
        sequence_length,
        max_duration_tokens=max_duration_tokens,
        in_eval_mode=in_eval_mode,
    )
    trainer_config = build_trainer_config(
        opts,
        tokenizer_config,
        sequence_length,
        max_duration_tokens=max_duration_tokens,
        in_eval_mode=in_eval_mode,
    )

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
    main(build_config, parser=get_parser())
