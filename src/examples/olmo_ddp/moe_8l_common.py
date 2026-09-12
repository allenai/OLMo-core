"""Shared constants and experiment plumbing for the tech-report MoE benchmark.

The two entry points in this directory intentionally share the macro model
shape, data configuration, rank microbatch, optimizer hyperparameters, and
measurement window. They do *not* share a model implementation: the OLMo DDP
DDP path uses ``OLMoDDPModel`` while the FSDP-based path uses the generic MoE
Transformer, because both stacks explicitly reject the opposite parallelizer.
See ``README.md`` in this directory before interpreting the comparison.
"""

from __future__ import annotations

import logging
import os
import socket
from pathlib import Path
from typing import Callable, cast


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _set_default_triton_cache_dir() -> None:
    if os.environ.get("TRITON_CACHE_DIR") or os.environ.get("OLMO_DISABLE_PER_RANK_TRITON_CACHE"):
        return
    local_rank = (
        os.environ.get("LOCAL_RANK")
        or os.environ.get("SLURM_LOCALID")
        or os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
        or "0"
    )
    job_id = (
        os.environ.get("BEAKER_EXPERIMENT_ID")
        or os.environ.get("SLURM_JOB_ID")
        or os.environ.get("JOB_ID")
        or "tech-report"
    )
    host = socket.gethostname().split(".")[0] or "host"
    cache_root = Path(os.environ.get("OLMO_TRITON_CACHE_BASE", "/tmp/olmo-triton-cache"))
    cache_dir = cache_root / str(job_id) / host / f"local_rank_{local_rank}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)


PROFILE = _env_flag("TECH_REPORT_PROFILE", False)
PROFILE_START = int(os.environ.get("TECH_REPORT_PROFILE_START", 20))
PROFILE_END = int(os.environ.get("TECH_REPORT_PROFILE_END", 25))
if PROFILE_END <= PROFILE_START:
    raise ValueError(
        "TECH_REPORT_PROFILE_END must be greater than TECH_REPORT_PROFILE_START; "
        f"got {PROFILE_START=} and {PROFILE_END=}"
    )
if not PROFILE:
    # NVTX reads this at import time.
    os.environ.setdefault("NVTX_DISABLE", "1")
_set_default_triton_cache_dir()

import torch  # noqa: E402

from olmo_core.data import (  # noqa: E402
    DataMix,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
)
from olmo_core.internal.experiment import (  # noqa: E402
    CommonComponents,
    DataComponents,
    ExperimentConfig,
)
from olmo_core.train import Duration, TrainerConfig  # noqa: E402
from olmo_core.train.callbacks import (  # noqa: E402
    NvidiaProfilerCallback,
    SpeedMonitorCallback,
    WandBCallback,
)

log = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")

# Matched architecture.
SEED = 2026
SEQUENCE_LENGTH = int(os.environ.get("TECH_REPORT_SEQUENCE_LENGTH", 8192))
GLOBAL_BATCH_SIZE = int(os.environ.get("TECH_REPORT_GLOBAL_BATCH_SIZE", 4 * 1024 * 1024))
RANK_MICROBATCH_SEQUENCES = int(os.environ.get("TECH_REPORT_RANK_MICROBATCH_SEQUENCES", 2))
MAX_STEPS = int(os.environ.get("TECH_REPORT_MAX_STEPS", 100))
EP_PATH_NAME = os.environ.get("TECH_REPORT_EP_PATH", "auto").strip().lower()
ROWWISE_GET_NBLOCKS = int(os.environ.get("TECH_REPORT_ROWWISE_GET_NBLOCKS", 256))
ROWWISE_PUT_NBLOCKS = int(os.environ.get("TECH_REPORT_ROWWISE_PUT_NBLOCKS", 256))
ROWWISE_WEIGHTED_PUT_NBLOCKS = int(os.environ.get("TECH_REPORT_ROWWISE_WEIGHTED_PUT_NBLOCKS", 128))
MXFP8_MLP = _env_flag("TECH_REPORT_MXFP8_MLP", False)
MXFP8_ATTN_QKV = _env_flag("TECH_REPORT_MXFP8_ATTN_QKV", False)
MXFP8_ATTN_OUT = _env_flag("TECH_REPORT_MXFP8_ATTN_OUT", False)
MXFP8_ATTN_SAVE_QKV = _env_flag("TECH_REPORT_MXFP8_ATTN_SAVE_QKV", False)
FORCE_FUSED_ATTENTION = _env_flag("TECH_REPORT_FORCE_FUSED_ATTENTION", False)
RECOMPUTE_EACH_BLOCK = _env_flag("TECH_REPORT_RECOMPUTE_EACH_BLOCK", False)
TWO_BATCH_OVERLAP = _env_flag("TECH_REPORT_TWO_BATCH_OVERLAP", False)
UNIFORM_ROUTING = _env_flag("TECH_REPORT_UNIFORM_ROUTING", False)

SUPPORTED_NUM_EXPERTS = (8, 32, 48, 64, 128)
SUPPORTED_GLOBAL_BATCH_SIZES = tuple(
    kib_tokens * 1024 for kib_tokens in (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
)
if GLOBAL_BATCH_SIZE not in SUPPORTED_GLOBAL_BATCH_SIZES:
    raise ValueError(
        "TECH_REPORT_GLOBAL_BATCH_SIZE must be one of "
        f"{SUPPORTED_GLOBAL_BATCH_SIZES}; got {GLOBAL_BATCH_SIZE}"
    )

D_MODEL = 4096
D_ATTN = 4096
HEAD_DIM = 128
NUM_HEADS = D_ATTN // HEAD_DIM
NUM_KV_HEADS = NUM_HEADS // 4
NUM_LAYERS = 8

NUM_EXPERTS = int(os.environ.get("TECH_REPORT_NUM_EXPERTS", 64))
if NUM_EXPERTS not in SUPPORTED_NUM_EXPERTS:
    raise ValueError(
        f"TECH_REPORT_NUM_EXPERTS must be one of {SUPPORTED_NUM_EXPERTS}; got {NUM_EXPERTS}"
    )
TOP_K = 4
MOE_HIDDEN_SIZE = 4096
NUM_SHARED_EXPERTS = 1
SHARED_MLP_HIDDEN_SIZE = 4096
DENSE_LAYER_MLP = TOP_K * MOE_HIDDEN_SIZE + SHARED_MLP_HIDDEN_SIZE
CAPACITY_FACTOR = 1.25

# On 8-GPU nodes this gives one EP group for DDP and one HSDP shard
# group for the FSDP-based baseline. On 32 GPUs both have four replicas.
PARALLEL_DEGREE = int(os.environ.get("TECH_REPORT_PARALLEL_DEGREE", 8))
if NUM_EXPERTS % PARALLEL_DEGREE != 0:
    raise ValueError(
        f"NUM_EXPERTS ({NUM_EXPERTS}) must be divisible by TECH_REPORT_PARALLEL_DEGREE "
        f"({PARALLEL_DEGREE})"
    )

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
WARMUP_STEPS = 10


def benchmark_num_flops_per_token(vocab_size: int, seq_len: int = SEQUENCE_LENGTH) -> int:
    """Canonical idealized FLOPs/token used by both benchmark stacks."""
    projection_flops = 6 * (
        D_MODEL * D_ATTN + 2 * D_MODEL * NUM_KV_HEADS * HEAD_DIM + D_ATTN * D_MODEL
    )

    attention_positions = seq_len * (seq_len + 1) // 2
    attention_flops = projection_flops + (
        12 * NUM_HEADS * HEAD_DIM * attention_positions // seq_len
    )

    flops = attention_flops + 18 * D_MODEL * DENSE_LAYER_MLP
    for _ in range(1, NUM_LAYERS):
        flops += attention_flops
        flops += 6 * D_MODEL * NUM_EXPERTS
        flops += 18 * D_MODEL * MOE_HIDDEN_SIZE * TOP_K
        flops += 18 * D_MODEL * SHARED_MLP_HIDDEN_SIZE * NUM_SHARED_EXPERTS
    flops += 6 * D_MODEL * vocab_size
    return flops


def build_data_components(
    common: CommonComponents,
    intra_document_masking: bool = False,
    include_instance_filter: bool = False,
) -> DataComponents:
    del include_instance_filter
    dataset = NumpyFSLDatasetConfig.from_data_mix(
        DataMix.OLMo_mix_0925,
        tokenizer=common.tokenizer,
        mix_base_dir=os.environ.get("TECH_REPORT_DATA_ROOT", "s3://ai2-llm"),
        work_dir=common.work_dir,
        sequence_length=common.max_sequence_length,
        max_target_sequence_length=max(common.max_sequence_length, 8192),
        generate_doc_lengths=intra_document_masking,
        instance_filter_config=None,
    )
    data_loader = NumpyDataLoaderConfig(
        global_batch_size=common.global_batch_size,
        seed=34521,
        num_workers=8,
        prefetch_factor=8,
    )
    return DataComponents(dataset=dataset, data_loader=data_loader)


def build_trainer_config(
    common: CommonComponents,
    *,
    variant: str,
    flops_per_token_builder: Callable[[int, int], int] = benchmark_num_flops_per_token,
) -> TrainerConfig:
    save_root = Path(os.environ.get("TECH_REPORT_SAVE_ROOT", "/workspace/checkpoint/tech_report"))
    wandb_enabled = _env_flag("TECH_REPORT_WANDB", True)
    group = os.environ.get("TECH_REPORT_WANDB_GROUP", "moe-8l-fsdp-vs-ddp")

    return (
        TrainerConfig(
            save_folder=str(save_root / f"{common.run_name}-{variant}"),
            save_overwrite=True,
            no_checkpoints=True,
            no_evals=True,
            metrics_collect_interval=5,
            cancel_check_interval=10,
            max_duration=Duration.steps(MAX_STEPS),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=f"{common.run_name}-{variant}",
                group=group,
                entity="ai2-llm",
                project=os.environ.get("TECH_REPORT_WANDB_PROJECT", "olmoe-tech-report"),
                enabled=wandb_enabled,
                cancel_check_interval=10,
            ),
        )
        .with_callback(
            "profiler",
            NvidiaProfilerCallback(
                enabled=PROFILE,
                profile_ranks=[0],
                start=PROFILE_START,
                end=PROFILE_END,
            ),
        )
        .with_callback(
            "speed_monitor",
            SpeedMonitorCallback(
                num_flops_per_token=flops_per_token_builder(
                    common.tokenizer.padded_vocab_size(), common.max_sequence_length
                )
            ),
        )
    )


def finalize_config(config: ExperimentConfig, *, variant: str) -> None:
    active_b = config.model.num_active_params / 1e9
    total_b = config.model.num_params / 1e9
    log.info(
        "%s benchmark model: %.6fB active / %.6fB total parameters",
        variant,
        active_b,
        total_b,
    )
    wandb = cast(WandBCallback, config.trainer.callbacks["wandb"])
    wandb.name = f"{wandb.name}_{active_b:.3f}A-{total_b:.3f}T"
