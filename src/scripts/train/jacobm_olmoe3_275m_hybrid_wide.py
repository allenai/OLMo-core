#!/usr/bin/env python3
"""Train the expand_v=1 275M integration-wide hybrid control from scratch."""

# ruff: noqa: E402

from __future__ import annotations

import logging
import os
import socket
from functools import partial
from pathlib import Path
from typing import Any, cast


def configure_rank_local_compile_cache() -> None:
    local_rank = os.environ.get("LOCAL_RANK", "0")
    job_id = os.environ.get("BEAKER_EXPERIMENT_ID", "local")
    host = socket.gethostname().split(".")[0]
    cache_dir = Path("/tmp/olmo-compile-cache") / job_id / host / f"rank{local_rank}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRITON_CACHE_DIR", str(cache_dir / "triton"))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_dir / "inductor"))


configure_rank_local_compile_cache()
os.environ.setdefault("NVTX_DISABLE", "1")
os.environ.setdefault("NO_GCE_CHECK", "true")
os.environ.setdefault("OLMO_DDP_INIT_SYNC", "0")
os.environ.setdefault("OLMO_DATA_PREP_WORKERS", "8")

import torch

from olmo_core.config import DType
from olmo_core.data import (
    DataMix,
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.internal.experiment import (
    CliContext,
    CommonComponents,
    DataComponents,
    ExperimentConfig,
    build_config,
    main,
)
from olmo_core.optim import OLMoDDPOptimizerConfig, OptimGroupOverride, SchedulerUnits
from olmo_core.optim.scheduler import (
    ComposableScheduler,
    ComposableSchedulerStage,
    ComposableSchedulerStageType,
)
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import CheckpointerCallback, SpeedMonitorCallback, WandBCallback
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.train_module import OLMoDDPTrainModuleConfig, TransformerDataParallelConfig
from scripts.train.jacobm_olmoe_ladder.v1.hybrid_wide_275m_model import (
    MAX_ACTIVE_PARAMETER_DELTA_FRACTION,
    build_hybrid_model_config,
    load_wide_model_config,
)


log = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")

SEQUENCE_LENGTH = int(os.environ.get("OLMOE3_HYBRID_SEQUENCE_LENGTH", "8192"))
GLOBAL_BATCH_SIZE = int(os.environ.get("OLMOE3_HYBRID_GLOBAL_BATCH_SIZE", "262144"))
WORLD_SIZE = int(os.environ.get("OLMOE3_HYBRID_WORLD_SIZE", "2"))
RANK_MICROBATCH_SEQUENCES = int(os.environ.get("OLMOE3_HYBRID_RANK_MICROBATCH_SEQUENCES", "16"))
LEARNING_RATE = float(os.environ.get("OLMOE3_HYBRID_LR", "1.6e-3"))
CHINCHILLA_MULTIPLE = float(os.environ.get("OLMOE3_HYBRID_CHINCHILLA_MULTIPLE", "1"))
MAX_TOKENS_OVERRIDE = os.environ.get("OLMOE3_HYBRID_MAX_TOKENS")
USE_COMPILE = os.environ.get("OLMOE3_HYBRID_USE_COMPILE", "1") != "0"
WANDB_ENABLED = os.environ.get("OLMOE3_HYBRID_WANDB", "1") != "0"
SAVE_INTERVAL = int(os.environ.get("OLMOE3_HYBRID_SAVE_INTERVAL", "1000"))
EPHEMERAL_SAVE_INTERVAL = int(os.environ.get("OLMOE3_HYBRID_EPHEMERAL_SAVE_INTERVAL", "500"))
SAVE_ROOT = os.environ.get(
    "OLMOE3_HYBRID_SAVE_ROOT",
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmo-ddp/pretraining",
)
WORK_DIR = os.environ.get(
    "OLMOE3_HYBRID_WORK_DIR",
    "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/dataset-cache",
)
DATA_ROOT = os.environ.get("OLMOE3_HYBRID_DATA_ROOT", "s3://ai2-llm")


def build_model_config(_common: CommonComponents):
    return build_hybrid_model_config()


def max_tokens() -> int:
    if MAX_TOKENS_OVERRIDE is not None:
        return int(MAX_TOKENS_OVERRIDE)
    model = build_hybrid_model_config()
    return int(
        Duration.chinchilla_tokens(
            CHINCHILLA_MULTIPLE,
            model_params=model.num_active_non_embedding_params,
        ).value
    )


def build_train_module_config(common: CommonComponents) -> OLMoDDPTrainModuleConfig:
    duration = max_tokens()
    warmup_tokens = max(
        GLOBAL_BATCH_SIZE,
        int((duration * 0.1 // GLOBAL_BATCH_SIZE) * GLOBAL_BATCH_SIZE),
    )
    return OLMoDDPTrainModuleConfig(
        rank_microbatch_size=RANK_MICROBATCH_SEQUENCES * common.max_sequence_length,
        max_sequence_length=common.max_sequence_length,
        optim=OLMoDDPOptimizerConfig(
            lr=LEARNING_RATE,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(
                    params=[
                        "*embedding_norm.weight",
                        "*q_norm.weight",
                        "*k_norm.weight",
                        "*o_norm.weight",
                        "*input_norm.weight",
                        "*lm_head.norm.weight",
                        "*attention_norm.weight",
                        "*feed_forward_norm.weight",
                    ],
                    opts={"weight_decay": 0.0},
                ),
                OptimGroupOverride(
                    params=["*routed_experts.w_up_gate", "*routed_experts.w_down"],
                    opts={"lr": LEARNING_RATE},
                ),
            ],
            compile=USE_COMPILE,
            dtype=DType.float32,
            sigma_factor=12,
            max_grad_norm=1.0,
            use_distributed=True,
        ),
        scheduler=ComposableScheduler(
            units=SchedulerUnits.tokens,
            stages=[
                ComposableSchedulerStage(
                    duration=warmup_tokens,
                    shape=ComposableSchedulerStageType.linear,
                    start_lr_fraction=0.0,
                    end_lr_fraction=1.0,
                ),
                ComposableSchedulerStage(
                    duration=max(duration - warmup_tokens, GLOBAL_BATCH_SIZE),
                    shape=ComposableSchedulerStageType.cosine,
                    end_lr_fraction=0.1,
                ),
            ],
        ),
        compile_model=USE_COMPILE,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.ddp,
            reduce_grads_in_fp32=True,
            accumulate_grads_in_fp32=True,
        ),
        ep_config=None,
        pp_config=None,
        tp_config=None,
        cp_config=None,
        ac_config=None,
        float8_config=None,
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )


def build_data_components(common: CommonComponents) -> DataComponents:
    dataset = NumpyFSLDatasetConfig.from_data_mix(
        DataMix.OLMo_mix_0925,
        tokenizer=common.tokenizer,
        mix_base_dir=DATA_ROOT,
        work_dir=common.work_dir,
        sequence_length=common.max_sequence_length,
        max_target_sequence_length=max(common.max_sequence_length, 8192),
        generate_doc_lengths=False,
        instance_filter_config=InstanceFilterConfig(
            repetition_max_period=13,
            repetition_min_period=1,
            repetition_max_count=32,
        ),
    )
    return DataComponents(
        dataset=dataset,
        data_loader=NumpyDataLoaderConfig(
            global_batch_size=common.global_batch_size,
            seed=34521,
            num_workers=8,
        ),
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    return (
        TrainerConfig(
            save_folder=common.save_folder,
            save_overwrite=False,
            checkpointer=CheckpointerConfig(
                save_thread_count=3,
                load_thread_count=8,
                throttle_uploads=True,
            ),
            metrics_collect_interval=10,
            cancel_check_interval=10,
            async_bookkeeping=False,
            max_duration=Duration.tokens(max_tokens()),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=SAVE_INTERVAL,
                ephemeral_save_interval=EPHEMERAL_SAVE_INTERVAL,
                save_async=False,
                pre_train_checkpoint=False,
            ),
        )
        .with_callback("speed_monitor", SpeedMonitorCallback())
        .with_callback(
            "wandb",
            WandBCallback(
                name=common.run_name,
                group="olmoe3-275m-integration-wide-hybrid",
                project="jacobm-olmoe-ladder",
                entity="ai2-llm",
                enabled=WANDB_ENABLED,
                cancel_check_interval=10,
                tags=[
                    "pretraining",
                    "275m",
                    "integration-wide",
                    "hybrid",
                    "gdn",
                    "expand-v-1",
                    "olmo-ddp",
                    "ep1",
                ],
            ),
        )
    )


def build_local_common_components(
    cli_context: CliContext,
    *,
    tokenizer: TokenizerConfig,
    global_batch_size: int,
    max_sequence_length: int,
    **_kwargs: Any,
) -> CommonComponents:
    if cli_context.cluster not in {"local", "localhost"}:
        raise ValueError("Launch this script inside Beaker and pass cluster='local'")
    return CommonComponents(
        run_name=cli_context.run_name,
        root_dir=DATA_ROOT,
        work_dir=WORK_DIR,
        save_folder=os.path.join(SAVE_ROOT, cli_context.run_name),
        launch=None,
        tokenizer=tokenizer,
        max_sequence_length=max_sequence_length,
        global_batch_size=global_batch_size,
    )


def finalize_config(config: ExperimentConfig) -> None:
    if config.train_module.ep_config is not None:
        raise ValueError("The 275M hybrid sweep must run without expert parallelism")
    global_sequences = GLOBAL_BATCH_SIZE // SEQUENCE_LENGTH
    denominator = WORLD_SIZE * RANK_MICROBATCH_SEQUENCES
    if global_sequences % denominator:
        raise ValueError(
            f"Global sequence batch {global_sequences} is not divisible by "
            f"world_size*rank_microbatch_sequences={denominator}"
        )
    base = load_wide_model_config()
    delta_fraction = (config.model.num_active_params - base.num_active_params) / base.num_active_params
    if abs(delta_fraction) > MAX_ACTIVE_PARAMETER_DELTA_FRACTION:
        raise ValueError(f"Hybrid active-parameter delta is too large: {delta_fraction:.4%}")
    log.info(
        "Hybrid wide config: active=%s total=%s active_delta=%+.4f%% tokens=%s "
        "global_sequences=%s rank_microbatch_sequences=%s grad_accum=%s lr=%s EP=off",
        f"{config.model.num_active_params:,}",
        f"{config.model.num_params:,}",
        100 * delta_fraction,
        f"{max_tokens():,}",
        global_sequences,
        RANK_MICROBATCH_SEQUENCES,
        global_sequences // denominator,
        LEARNING_RATE,
    )


def make_config(cli_context: CliContext) -> ExperimentConfig:
    builder = partial(
        build_config,
        common_config_builder=build_local_common_components,
        data_config_builder=build_data_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        tokenizer=TokenizerConfig.dolma2(),
        global_batch_size=GLOBAL_BATCH_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        num_nodes=1,
        include_default_evals=False,
        finalize_config=finalize_config,
    )
    return cast(ExperimentConfig, builder(cli_context))


if __name__ == "__main__":
    main(config_builder=make_config)
