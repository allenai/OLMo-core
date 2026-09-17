"""Portable OLMo 3.5 recipes, independent of Beaker campaigns or private paths.

These builders do not launch jobs, access data, or change process-wide kernel settings.
See OLMO3P5.md for qualification limits and explicit runtime setup.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path

from olmo3p5_models import build_model_config

from olmo_core.config import DType
from olmo_core.data import (
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.optim import OLMoDDPOptimizerConfig, OptimGroupOverride
from olmo_core.optim.scheduler import ConstantWithWarmup
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CheckpointReadyNotifierCallback,
)
from olmo_core.train.callbacks.checkpointer import CheckpointRemovalStrategy
from olmo_core.train.train_module import (
    OLMoDDPTrainModuleConfig,
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
)

BATCH_TOKENS = 16_777_216
SEQUENCE_LENGTH = 8192
WARMUP_STEPS = 2000
DATA_SEED = 928_543_231
INIT_SEED = 12536
DOLMA_MANIFEST_SHA256 = "992ea0c56506fe0e03140f7094b5f52022885f5accfafd390c50a8bf193b0c1b"


@dataclass(frozen=True)
class TrainingSettings:
    """Qualified topology, with an explicit LR and batch in tokens (not sequences)."""

    model_size: str
    num_gpus: int
    ep_degree: int
    microbatch_sequences: int
    peak_lr: float
    batch_tokens: int = BATCH_TOKENS
    metrics_collect_interval: int = 1

    @property
    def gradient_accumulation_steps(self) -> int:
        wave = self.num_gpus * self.microbatch_sequences * SEQUENCE_LENGTH
        if self.batch_tokens <= 0 or self.batch_tokens % wave:
            raise ValueError("Global batch must divide into whole microbatch waves")
        return self.batch_tokens // wave

    def validate(self) -> None:
        if self.model_size not in ("small", "medium"):
            raise ValueError("Large geometry is available, but no production topology is qualified")
        if min(self.num_gpus, self.ep_degree, self.microbatch_sequences) < 1:
            raise ValueError("GPU, EP and microbatch counts must be positive")
        if self.num_gpus % self.ep_degree or 512 % self.ep_degree:
            raise ValueError("EP degree must divide both GPU and expert counts")
        if not math.isfinite(self.peak_lr) or self.peak_lr <= 0:
            raise ValueError("A finite positive LR is required")
        if self.metrics_collect_interval < 1:
            raise ValueError("Metrics interval must be positive")
        _ = self.gradient_accumulation_steps


SMALL = TrainingSettings("small", 64, 1, 4, 1.1e-3)
# This is the medium CBS seed LR, NOT a tuned 14T hero LR.
MEDIUM_128 = TrainingSettings("medium", 128, 8, 2, 9.2e-4, metrics_collect_interval=5)
MEDIUM_64 = TrainingSettings("medium", 64, 8, 2, 9.2e-4, metrics_collect_interval=5)


def optimization_environment(*, ep_degree: int, torch_version: str) -> dict[str, str]:
    """Return opt-in switches; never apply a Torch-2.11 tie-order kernel to newer Torch."""
    return {
        "OLMO_PROFILE_FP32_GRAD_ADD_VECTORIZE": "1",
        "OLMO_PROFILE_SWIGLU_PAIRWISE": "1",
        "OLMO_PROFILE_EMO_DOCUMENT_POOL": "1",
        "OLMO_PROFILE_EMO_TOP16": "1" if torch_version.startswith("2.11.") else "0",
        "OLMO_PROFILE_ROUNDED_WGRAD": "1",
        "OLMO_PROFILE_ROUNDED_WGRAD_EP": "1" if ep_degree > 1 else "0",
        "OLMO_PROFILE_RS_SINGLE_PARAM_FAST_PATH": "1",
    }


def enable_optimizations(settings: TrainingSettings) -> dict[str, str]:
    """Explicit process-wide setup, called before model construction/compilation.

    MIN_CTAS is a private kernel-fun interface inherited from the qualified run.
    Fail closed if it changes; do not silently substitute another tuning policy.
    """
    import torch
    from kernel_fun._common import support

    import olmo_core.ops.moe as moe_ops

    settings.validate()
    if not hasattr(support, "MIN_CTAS"):
        raise RuntimeError("Requalify kernel-fun's CTA policy before using this recipe")
    flags = optimization_environment(ep_degree=settings.ep_degree, torch_version=torch.__version__)
    if flags["OLMO_PROFILE_EMO_TOP16"] == "0":
        logging.getLogger(__name__).warning(
            "Using native Torch top-k: the optimized tie-order kernel is qualified only on 2.11"
        )
    os.environ.update(flags)
    support.MIN_CTAS = 128
    moe_ops.pool_keep_mask = moe_ops.pool_keep_mask_inverse_scatter
    return flags


def build_model(settings: TrainingSettings, *, use_emo: bool):
    """Return the immutable model shape, with EMO selected explicitly."""
    settings.validate()
    return build_model_config(
        settings.model_size,
        eos_token_id=TokenizerConfig.dolma2().eos_token_id,
        use_emo=use_emo,
        init_seed=INIT_SEED,
    )


def build_train_module(settings: TrainingSettings, *, optimized: bool = True):
    """Build BF16-compute/FP32-state AdamW and the stable WSD trunk."""
    settings.validate()
    return OLMoDDPTrainModuleConfig(
        rank_microbatch_size=settings.microbatch_sequences * SEQUENCE_LENGTH,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=OLMoDDPOptimizerConfig(
            lr=settings.peak_lr,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            eps=1e-8,
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts={"weight_decay": 0.0}),
                OptimGroupOverride(
                    params=["*routed_experts.w_up_gate", "*routed_experts.w_down"], opts={}
                ),
            ],
            compile=True,
            dtype=DType.float32,
            sigma_factor=6,
            max_grad_norm=1.0,
            use_distributed=True,
        ),
        scheduler=ConstantWithWarmup(warmup=WARMUP_STEPS),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.ddp,
            accumulate_grads_in_fp32=True,
            reduce_grads_in_fp32=True,
            use_reduce_scatter=optimized,
        ),
        ep_config=(
            TransformerExpertParallelConfig(degree=settings.ep_degree)
            if settings.ep_degree > 1
            else None
        ),
        pp_config=None,
        tp_config=None,
        cp_config=None,
        ac_config=None,
        float8_config=None,
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )


def build_data(
    settings: TrainingSettings, *, manifest: str, data_root: str, work_dir: str
) -> tuple[NumpyFSLDatasetConfig, NumpyDataLoaderConfig]:
    """Use the exact Dolma 3.5 manifest order with a caller-selected local/S3 root."""
    settings.validate()
    # Mainline's manifest uses a tokenizer placeholder; production recorded the
    # same ordered paths after interpolation.
    payload = (
        Path(manifest)
        .read_bytes()
        .replace(b"{TOKENIZER}", b"allenai/dolma2-tokenizer")
        .rstrip(b"\n")
    )
    if hashlib.sha256(payload).hexdigest() != DOLMA_MANIFEST_SHA256:
        raise ValueError("Expected the production Dolma3p5-14t manifest, unchanged and in order")
    rows = list(csv.reader(payload.decode().splitlines()))
    paths = [f"{data_root.rstrip('/')}/{path}" for _, path in rows]
    dataset = NumpyFSLDatasetConfig(
        paths=paths,
        metadata=[{"label": label} for label, _ in rows],
        tokenizer=TokenizerConfig.dolma2(),
        work_dir=work_dir,
        sequence_length=SEQUENCE_LENGTH,
        max_target_sequence_length=SEQUENCE_LENGTH,
        generate_doc_lengths=False,
        instance_filter_config=InstanceFilterConfig(
            repetition_max_period=13, repetition_min_period=1, repetition_max_count=32
        ),
    )
    loader = NumpyDataLoaderConfig(
        global_batch_size=settings.batch_tokens,
        seed=DATA_SEED,
        num_workers=8,
        prefetch_factor=8,
        num_threads=4,
    )
    return dataset, loader


def build_trainer(
    settings: TrainingSettings,
    *,
    save_folder: str,
    work_dir: str,
    total_tokens: int = 14_000_000_000_000,
    stop_step: int | None = None,
    inbox_dir: str | None = None,
    run_id: str | None = None,
    lineage_id: str | None = None,
) -> TrainerConfig:
    """Synchronous immutable checkpoints; retention belongs to the separate uploader."""
    settings.validate()
    final_steps = math.ceil(total_tokens / settings.batch_tokens)
    if total_tokens <= 0 or (stop_step is not None and not 0 < stop_step <= final_steps):
        raise ValueError("Invalid token budget or stop step")
    config = TrainerConfig(
        save_folder=save_folder,
        work_dir=work_dir,
        max_duration=Duration.steps(final_steps),
        hard_stop=Duration.steps(stop_step) if stop_step is not None else None,
        metrics_collect_interval=settings.metrics_collect_interval,
        save_overwrite=False,
        load_optim_state=True,
        load_trainer_state=True,
    )
    config.add_callback(
        "checkpointer",
        CheckpointerCallback(
            save_interval=500,
            fixed_steps=list(range(100, 18_001, 100)) + list(range(18_250, 60_001, 250)),
            pre_train_checkpoint=None,
            save_async=False,
            remove=CheckpointRemovalStrategy.never,
            max_checkpoints=None,
        ),
    )
    if inbox_dir is not None:
        if not run_id or not lineage_id:
            raise ValueError("Uploader notifications require explicit run_id and lineage_id")
        config.add_callback(
            "checkpoint_ready",
            CheckpointReadyNotifierCallback(
                inbox_dir=inbox_dir, run_id=run_id, lineage_id=lineage_id
            ),
        )
    return config
