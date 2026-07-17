"""Upstream-main FSDP dense baseline for the tech-report MoE benchmark.

The dense MLP width matches the per-token selected MLP width of the paired MoE:
``top_k * expert_hidden_size + shared_hidden_size``. All other macro shape,
data, optimizer, precision, profiling, and timing settings are reused from the
upstream-main MoE FSDP entry point.
"""

from __future__ import annotations

import math
import os
from functools import partial

from moe_8l_fsdp import (
    D_MODEL,
    EXPERT_HIDDEN_SIZE,
    GLOBAL_BATCH_SIZE,
    HEAD_DIM,
    MAX_STEPS,
    NUM_HEADS,
    NUM_KV_HEADS,
    NUM_LAYERS,
    PROFILE,
    SEED,
    SEQUENCE_LENGTH,
    SHARED_HIDDEN_SIZE,
    TOP_K,
    NvidiaProfilerCallback,
    _env_flag,
    build_data_components,
    build_train_module_config,
    finalize_config,
)
from olmo_core.config import DType
from olmo_core.internal.experiment import (
    CommonComponents,
    ExperimentConfig,
    build_config,
    main,
)
from olmo_core.nn.attention.backend import AttentionBackendName
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.transformer import (
    TransformerBlockType,
    TransformerConfig,
    TransformerType,
)
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import SpeedMonitorCallback, WandBCallback

DENSE_HIDDEN_SIZE = TOP_K * EXPERT_HIDDEN_SIZE + SHARED_HIDDEN_SIZE


def benchmark_num_flops_per_token(vocab_size: int, seq_len: int = SEQUENCE_LENGTH) -> int:
    """Idealized dense FLOPs/token under the benchmark's shared convention."""
    projection_flops = 6 * (
        D_MODEL * D_MODEL
        + 2 * D_MODEL * NUM_KV_HEADS * HEAD_DIM
        + D_MODEL * D_MODEL
    )
    attention_positions = seq_len * (seq_len + 1) // 2
    attention_flops = projection_flops + (
        12 * NUM_HEADS * HEAD_DIM * attention_positions // seq_len
    )
    mlp_flops = 18 * D_MODEL * DENSE_HIDDEN_SIZE
    return NUM_LAYERS * (attention_flops + mlp_flops) + 6 * D_MODEL * vocab_size


def build_model_config(common: CommonComponents) -> TransformerConfig:
    return TransformerConfig.llama_like(
        init_seed=SEED,
        d_model=D_MODEL,
        vocab_size=common.tokenizer.padded_vocab_size(),
        n_layers=NUM_LAYERS,
        n_heads=NUM_HEADS,
        n_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        name=TransformerType.default,
        block_name=TransformerBlockType.reordered_norm,
        qk_norm=True,
        use_head_qk_norm=True,
        rope_theta=500_000,
        rope_full_precision=True,
        layer_norm_eps=1e-6,
        attn_backend=AttentionBackendName.flash_4,
        feed_forward=FeedForwardConfig(
            hidden_size=DENSE_HIDDEN_SIZE,
            bias=False,
            dtype=DType.float32,
        ),
        embed_scale=math.sqrt(D_MODEL),
        init_std=0.01,
        dtype=DType.float32,
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    wandb_enabled = _env_flag("TECH_REPORT_WANDB", bool(os.environ.get("WANDB_API_KEY")))
    return (
        TrainerConfig(
            save_folder=f"/workspace/checkpoint/tech_report-main/{common.run_name}",
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
                name=common.run_name,
                group=os.environ.get(
                    "TECH_REPORT_WANDB_GROUP", "dense-vs-moe-fsdp-main"
                ),
                entity="ai2-llm",
                project=os.environ.get("TECH_REPORT_WANDB_PROJECT", "olmoe-tech-report"),
                enabled=wandb_enabled,
                cancel_check_interval=10,
            ),
        )
        .with_callback(
            "nvidia_profiler",
            NvidiaProfilerCallback(
                enabled=PROFILE,
                start=21,
                end=24,
            ),
        )
        .with_callback(
            "speed_monitor",
            SpeedMonitorCallback(
                num_flops_per_token=benchmark_num_flops_per_token(
                    common.tokenizer.padded_vocab_size(), common.max_sequence_length
                )
            ),
        )
    )


def finalize_dense_config(config: ExperimentConfig) -> None:
    finalize_config(config)
    dense_params = config.model.num_params
    if dense_params != config.model.num_active_params:
        raise RuntimeError(
            "Dense benchmark must have identical active and total parameter counts; "
            f"got {config.model.num_active_params:,d} active and {dense_params:,d} total"
        )


if __name__ == "__main__":
    config_builder = partial(
        build_config,
        global_batch_size=GLOBAL_BATCH_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        data_config_builder=build_data_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        finalize_config=finalize_dense_config,
        include_instance_filter=False,
        include_default_evals=False,
        flight_recorder=True,
    )
    main(config_builder=config_builder)
