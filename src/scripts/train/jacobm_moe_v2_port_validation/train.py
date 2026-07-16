#!/usr/bin/env python3
"""Train exact first-hybrid controls on the mergeable moe-v2-core port."""

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

import torch

from olmo_core.data import (
    DataMix,
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.internal.experiment import (
    CliContext,
    CommonComponents,
    DataComponents,
    ExperimentConfig,
    build_config,
    main,
)
from olmo_core.nn.moe.v2.ep_config import ExpertParallelPath
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import (
    BeakerCallback,
    CheckpointerCallback,
    CheckpointRemovalStrategy,
    ConfigSaverCallback,
    SpeedMonitorCallback,
    WandBCallback,
)
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.train_module import (
    OLMoDDPTrainModuleConfig,
    TransformerExpertParallelConfig,
)
from scripts.train.jacobm_moe_v2_port_validation.config_adapter import (
    adapt_train_module_payload,
    build_model_config as build_adapted_model_config,
    load_recorded_config,
)

log = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


SOURCE_CONFIGS = {
    "275m": Path(
        "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmo-ddp/"
        "pretraining/pt-275m-intwide-hybrid-gdn-ev1-cx1-lr1p6e-3-r1/step16108/config.json"
    ),
    "1p2b": Path(
        "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmo-ddp/"
        "pretraining/1p2b-intwide-hybrid-gdn-ev1-cx1-lr4e-4-ep8-sync/step86558/config.json"
    ),
}

MODEL_SIZE = os.environ.get("OLMOE3_PORT_MODEL_SIZE", "275m")
SEQUENCE_LENGTH = int(os.environ.get("OLMOE3_PORT_SEQUENCE_LENGTH", "8192"))
GLOBAL_BATCH_SIZE = int(os.environ.get("OLMOE3_PORT_GLOBAL_BATCH_SIZE", "262144"))
WORLD_SIZE = int(os.environ.get("OLMOE3_PORT_WORLD_SIZE", "2"))
EP_SIZE = int(os.environ.get("OLMOE3_PORT_EP_SIZE", "1"))
EP_PATH = ExpertParallelPath(
    os.environ.get("OLMOE3_PORT_EP_PATH", ExpertParallelPath.rowwise_nvshmem.value)
)
RANK_MICROBATCH_SEQUENCES = int(os.environ.get("OLMOE3_PORT_RANK_MICROBATCH_SEQUENCES", "16"))
LEARNING_RATE = float(os.environ.get("OLMOE3_PORT_LR", "1.6e-3"))
CHINCHILLA_MULTIPLE = float(os.environ.get("OLMOE3_PORT_CHINCHILLA_MULTIPLE", "1"))
HARD_STOP_STEPS = int(os.environ.get("OLMOE3_PORT_HARD_STOP_STEPS", "0"))
USE_COMPILE = env_bool("OLMOE3_PORT_USE_COMPILE", True)
WANDB_ENABLED = env_bool("OLMOE3_PORT_WANDB", True)
CHECKPOINTS_ENABLED = env_bool("OLMOE3_PORT_CHECKPOINTS", True)
SAVE_INTERVAL = int(os.environ.get("OLMOE3_PORT_SAVE_INTERVAL", "999999999"))
EPHEMERAL_SAVE_INTERVAL = int(os.environ.get("OLMOE3_PORT_EPHEMERAL_SAVE_INTERVAL", "500"))
CHECKPOINT_REMOVAL = CheckpointRemovalStrategy(
    os.environ.get("OLMOE3_PORT_CHECKPOINT_REMOVAL", "ephemeral_only")
)
SAVE_ROOT = os.environ.get(
    "OLMOE3_PORT_SAVE_ROOT",
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmo-ddp/"
    "port-validation/0cdcc8b81/pretraining",
)
WORK_DIR = os.environ.get(
    "OLMOE3_PORT_WORK_DIR",
    "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/dataset-cache",
)
DATA_ROOT = os.environ.get("OLMOE3_PORT_DATA_ROOT", "s3://ai2-llm")
SOURCE_CONFIG = Path(
    os.environ.get("OLMOE3_PORT_SOURCE_CONFIG", str(SOURCE_CONFIGS.get(MODEL_SIZE, "")))
)


def model_config():
    model = build_adapted_model_config(SOURCE_CONFIG)
    if EP_SIZE > 1:
        if model.block.ep is not None:
            model.block.ep.path = EP_PATH
        for block in (model.block_overrides or {}).values():
            if block.ep is not None:
                block.ep.path = EP_PATH
    model.validate()
    return model


def build_model_config(_common: CommonComponents):
    return model_config()


def max_tokens() -> int:
    model = model_config()
    return int(
        Duration.chinchilla_tokens(
            CHINCHILLA_MULTIPLE,
            model_params=model.num_active_non_embedding_params,
        ).value
    )


def build_train_module_config(common: CommonComponents) -> OLMoDDPTrainModuleConfig:
    recorded = load_recorded_config(SOURCE_CONFIG)
    config = OLMoDDPTrainModuleConfig.from_dict(
        adapt_train_module_payload(recorded["train_module"])
    )
    config.rank_microbatch_size = RANK_MICROBATCH_SEQUENCES * common.max_sequence_length
    config.max_sequence_length = common.max_sequence_length
    config.optim.lr = LEARNING_RATE
    for override in config.optim.group_overrides or []:
        if "lr" in override.opts:
            override.opts["lr"] = LEARNING_RATE
    config.optim.compile = USE_COMPILE
    config.compile_model = USE_COMPILE
    config.ep_config = TransformerExpertParallelConfig(degree=EP_SIZE) if EP_SIZE > 1 else None
    config.pp_config = None
    config.tp_config = None
    config.cp_config = None
    config.ac_config = None
    config.float8_config = None
    return config


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
    trainer = TrainerConfig(
        save_folder=common.save_folder,
        save_overwrite=False,
        no_checkpoints=not CHECKPOINTS_ENABLED,
        checkpointer=CheckpointerConfig(
            save_thread_count=3,
            load_thread_count=8,
            throttle_uploads=True,
        ),
        metrics_collect_interval=1 if HARD_STOP_STEPS else 10,
        cancel_check_interval=10,
        async_bookkeeping=False,
        max_duration=Duration.tokens(max_tokens()),
        hard_stop=Duration.steps(HARD_STOP_STEPS) if HARD_STOP_STEPS else None,
    )
    if CHECKPOINTS_ENABLED:
        trainer = trainer.with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=SAVE_INTERVAL,
                ephemeral_save_interval=EPHEMERAL_SAVE_INTERVAL,
                save_async=False,
                pre_train_checkpoint=False,
                remove=CHECKPOINT_REMOVAL,
            ),
        )
    return (
        trainer.with_callback("speed_monitor", SpeedMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("beaker", BeakerCallback())
        .with_callback(
            "wandb",
            WandBCallback(
                name=common.run_name,
                group="moe-v2-core-port-validation-0cdcc8b81",
                project="jacobm-olmoe-ladder",
                entity="ai2-llm",
                enabled=WANDB_ENABLED,
                cancel_check_interval=10,
                tags=[
                    "pretraining",
                    MODEL_SIZE,
                    "integration-wide",
                    "expand-v-1",
                    "hybrid",
                    "gdn",
                    "moe-v2-core-port",
                    "0cdcc8b81",
                    f"ep{EP_SIZE}",
                    "smoke" if HARD_STOP_STEPS else "full-run",
                    "no-in-loop-evals",
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
        raise ValueError("Launch inside Beaker and pass cluster='local'")
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
    if MODEL_SIZE not in SOURCE_CONFIGS:
        raise ValueError(f"Unsupported validation size {MODEL_SIZE!r}")
    if not SOURCE_CONFIG.is_file():
        raise FileNotFoundError(SOURCE_CONFIG)
    if WORLD_SIZE < 1 or EP_SIZE < 1 or WORLD_SIZE % EP_SIZE:
        raise ValueError(f"EP size {EP_SIZE} must divide world size {WORLD_SIZE}")
    if GLOBAL_BATCH_SIZE % SEQUENCE_LENGTH:
        raise ValueError("Global batch size must contain a whole number of sequences")
    global_sequences = GLOBAL_BATCH_SIZE // SEQUENCE_LENGTH
    if global_sequences % WORLD_SIZE:
        raise ValueError(
            f"Global sequence batch {global_sequences} must divide world size {WORLD_SIZE}"
        )
    rank_sequences = global_sequences // WORLD_SIZE
    effective_microbatch = min(rank_sequences, RANK_MICROBATCH_SEQUENCES)
    if rank_sequences % effective_microbatch:
        raise ValueError(
            f"Rank batch {rank_sequences} is not divisible by microbatch {effective_microbatch}"
        )
    source_model = build_adapted_model_config(SOURCE_CONFIG)
    if config.model.as_dict() != model_config().as_dict():
        raise ValueError("Built model config drifted from the narrowly adapted source config")
    log.info(
        "Port validation: commit=0cdcc8b81 size=%s active=%s active_non_embedding=%s total=%s "
        "tokens=%s global_sequences=%s world=%s EP=%s EP_path=%s rank_sequences=%s "
        "MB_cap=%s effective_MB=%s accum=%s lr=%s hard_stop=%s source=%s",
        MODEL_SIZE,
        f"{source_model.num_active_params:,}",
        f"{source_model.num_active_non_embedding_params:,}",
        f"{source_model.num_params:,}",
        f"{max_tokens():,}",
        global_sequences,
        WORLD_SIZE,
        EP_SIZE,
        EP_PATH.value,
        rank_sequences,
        RANK_MICROBATCH_SEQUENCES,
        effective_microbatch,
        rank_sequences // effective_microbatch,
        LEARNING_RATE,
        HARD_STOP_STEPS or "off",
        SOURCE_CONFIG,
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
