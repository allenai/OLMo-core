#!/usr/bin/env python3
"""Continue a converted OLMoE3 checkpoint at 65k context.

The generic ``OLMOE3_LC_*`` environment variables take precedence. The legacy
``OLMOE3_275M_LC_*`` names remain as fallbacks so the original 275M job stays
reproducible.
"""

# ruff: noqa: E402

from __future__ import annotations

import json
import logging
import os
import socket
from copy import deepcopy
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any, Dict, cast


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
from cached_path import cached_path

from olmo_core.config import DType
from olmo_core.data import (
    DataMix,
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyPackedFSLDatasetConfig,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.internal.experiment import (
    CliContext,
    CommonComponents,
    DataComponents,
    ExperimentConfig,
    SubCmd,
    build_config,
    main,
)
from olmo_core.io import join_path
from olmo_core.nn.moe.v2.ep_config import ExpertParallelPath
from olmo_core.nn.rope import YaRNRoPEScalingConfig
from olmo_core.nn.transformer import OLMoDDPModelConfig
from olmo_core.optim import ConstantWithWarmup, OLMoDDPOptimizerConfig, OptimGroupOverride
from olmo_core.train import Duration, LoadStrategy, TrainerConfig
from olmo_core.train.callbacks import (
    BeakerCallback,
    CheckpointerCallback,
    CheckpointRemovalStrategy,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    LMEvaluatorCallbackConfig,
    SpeedMonitorCallback,
    WandBCallback,
)
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.train_module import (
    OLMoDDPTrainModuleConfig,
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
)


log = logging.getLogger(__name__)


def lc_env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(f"OLMOE3_LC_{name}")
    if value is None:
        value = os.environ.get(f"OLMOE3_275M_LC_{name}")
    return default if value is None else value


def lc_bool(name: str, default: bool) -> bool:
    value = lc_env(name)
    if value is None:
        return default
    if value.lower() in {"1", "true", "yes", "on"}:
        return True
    if value.lower() in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"OLMOE3_LC_{name} must be a boolean, got {value!r}")


MODEL_SIZE = cast(str, lc_env("MODEL_SIZE", "275m"))
FAMILY = cast(str, lc_env("FAMILY", "baseline"))
SEQUENCE_LENGTH = int(cast(str, lc_env("SEQUENCE_LENGTH", "65536")))
GLOBAL_BATCH_SIZE = int(cast(str, lc_env("GLOBAL_BATCH_SIZE", str(2 * 1024 * 1024))))
MAX_TOKENS = int(cast(str, lc_env("MAX_TOKENS", "100000000000")))
RANK_MICROBATCH_SEQUENCES = int(cast(str, lc_env("RANK_MICROBATCH_SEQUENCES", "4")))
EXPECTED_WORLD_SIZE = int(cast(str, lc_env("WORLD_SIZE", "8")))
EP_SIZE = int(cast(str, lc_env("EP_SIZE", "1")))
EP_PATH = ExpertParallelPath(cast(str, lc_env("EP_PATH", ExpertParallelPath.sync_1d.value)))
LEARNING_RATE = float(cast(str, lc_env("LR", "1e-4")))
WARMUP_STEPS = int(cast(str, lc_env("WARMUP_STEPS", "2000")))
HARD_STOP_STEPS = int(cast(str, lc_env("HARD_STOP_STEPS", "0")))
USE_COMPILE = lc_bool("USE_COMPILE", True)
WANDB_ENABLED = lc_bool("WANDB", True)
EVALS_ENABLED = lc_bool("EVALS", False)
EVAL_INTERVAL = int(cast(str, lc_env("EVAL_INTERVAL", "1000")))
EVAL_STEPS = int(cast(str, lc_env("EVAL_STEPS", "0")))
EVAL_SEQUENCE_LENGTH = int(cast(str, lc_env("EVAL_SEQUENCE_LENGTH", "8192")))
EVAL_TASK_SET = cast(str, lc_env("EVAL_TASK_SET", "fast"))
EVAL_ON_FINISH = lc_bool("EVAL_ON_FINISH", False)
ASYNC_BOOKKEEPING = lc_bool("ASYNC_BOOKKEEPING", False)
SAVE_INTERVAL = int(cast(str, lc_env("SAVE_INTERVAL", "5000")))
EPHEMERAL_SAVE_INTERVAL = int(cast(str, lc_env("EPHEMERAL_SAVE_INTERVAL", "1000")))
CHECKPOINT_REMOVAL = CheckpointRemovalStrategy(
    cast(str, lc_env("CHECKPOINT_REMOVAL", CheckpointRemovalStrategy.ephemeral_only.value))
)

LOAD_PATH = lc_env("LOAD_PATH")
SAVE_ROOT = cast(
    str,
    lc_env(
        "SAVE_ROOT",
        "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmo-ddp/long-context",
    ),
)
WORK_DIR = cast(
    str,
    lc_env(
        "WORK_DIR",
        "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/dataset-cache/long-context-65k",
    ),
)
LC_DATA_GLOB = cast(
    str,
    lc_env(
        "DATA_GLOB",
        "/weka/oe-training-default/ai2-llm/preprocessed/tylerr/"
        "lc-reshard-final-cleaned/v0.1/allenai/dolma2-tokenizer/*.npy",
    ),
)
EVAL_DATA_ROOT = cast(
    str,
    lc_env("EVAL_DATA_ROOT", "/weka/oe-training-default/ai2-llm"),
)

ROPE_SCALING = YaRNRoPEScalingConfig(
    factor=8,
    beta_fast=32,
    beta_slow=1,
    old_context_len=8192,
)

torch.set_float32_matmul_precision("high")


def load_checkpoint_config() -> Dict[str, Any]:
    if LOAD_PATH is None:
        raise OLMoConfigurationError("Set OLMOE3_LC_LOAD_PATH to a converted OLMoDDP checkpoint")
    path = cached_path(join_path(LOAD_PATH, "config.json"), quiet=True)
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def build_model_config(common: CommonComponents) -> OLMoDDPModelConfig:
    config = OLMoDDPModelConfig.from_dict(load_checkpoint_config()["model"])
    if config.vocab_size != common.tokenizer.padded_vocab_size():
        raise ValueError(
            f"Checkpoint vocab size {config.vocab_size} does not match tokenizer "
            f"size {common.tokenizer.padded_vocab_size()}"
        )

    config.recompute_each_block = True
    config.recompute_all_blocks_by_chunk = False
    config.recompute_block_keys = None

    if isinstance(config.block, dict):
        raise TypeError("This continuation script expects a single base block config")
    overrides = dict(config.block_overrides or {})
    for block_idx in range(config.n_layers):
        block_config = deepcopy(overrides.get(block_idx, config.block))
        attention = block_config.attention
        if attention is None or attention.rope is None:
            raise RuntimeError(f"Block {block_idx} does not contain RoPE attention")
        sliding_window = attention.sliding_window
        uses_sliding_window = sliding_window is not None and sliding_window.should_use_swa(
            block_idx, config.n_layers
        )
        if not uses_sliding_window:
            block_config.attention.rope = replace(attention.rope, scaling=ROPE_SCALING)
        overrides[block_idx] = block_config
    config.block_overrides = overrides
    if EP_SIZE > 1:
        if config.block.ep is not None:
            config.block.ep.path = EP_PATH
        for block_config in config.block_overrides.values():
            if block_config.ep is not None:
                block_config.ep.path = EP_PATH
    config.validate()
    return config


def build_train_module_config(common: CommonComponents) -> OLMoDDPTrainModuleConfig:
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
                        "*input_norm.weight",
                        "*lm_head.norm.weight",
                        "*attention_norm.weight",
                        "*feed_forward_norm.weight",
                    ],
                    opts={"weight_decay": 0.0, "use_muon": False},
                ),
                OptimGroupOverride(
                    params=["*routed_experts.w_up_gate", "*routed_experts.w_down"],
                    opts={"lr": LEARNING_RATE, "use_muon": False},
                ),
            ],
            compile=USE_COMPILE,
            dtype=DType.float32,
            sigma_factor=12,
            max_grad_norm=1.0,
            use_distributed=True,
        ),
        scheduler=ConstantWithWarmup(warmup=WARMUP_STEPS),
        compile_model=USE_COMPILE,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.ddp,
            reduce_grads_in_fp32=True,
            accumulate_grads_in_fp32=True,
        ),
        ep_config=TransformerExpertParallelConfig(degree=EP_SIZE) if EP_SIZE > 1 else None,
        pp_config=None,
        tp_config=None,
        cp_config=None,
        ac_config=None,
        float8_config=None,
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        reset_optimizer_states_on_load=True,
        reset_optimizer_states_on_resume=False,
    )


def build_data_components(common: CommonComponents) -> DataComponents:
    dataset_config = NumpyPackedFSLDatasetConfig.glob(
        LC_DATA_GLOB,
        tokenizer=common.tokenizer,
        work_dir=common.work_dir,
        sequence_length=common.max_sequence_length,
        source_group_size=8,
        source_permutation_seed=123,
        instance_filter_config=InstanceFilterConfig(),
    )
    return DataComponents(
        dataset=dataset_config,
        data_loader=NumpyDataLoaderConfig(
            global_batch_size=common.global_batch_size,
            seed=119_105_108_108 % (2**31 - 1),
            num_workers=16,
            prefetch_factor=8,
        ),
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    assert LOAD_PATH is not None
    if EVAL_TASK_SET == "hellaswag":
        downstream_tasks = ["hellaswag"]
    else:
        from olmo_core.eval.task_groups import TASK_GROUPS

        try:
            downstream_tasks = sorted(TASK_GROUPS[EVAL_TASK_SET])
        except KeyError as error:
            raise ValueError(f"Task set not recognized: {EVAL_TASK_SET}") from error
    eval_duration = Duration.steps(EVAL_STEPS) if EVAL_STEPS > 0 else Duration.epochs(1)
    return (
        TrainerConfig(
            save_folder=common.save_folder,
            load_path=LOAD_PATH,
            load_strategy=LoadStrategy.always,
            load_trainer_state=False,
            load_optim_state=False,
            save_overwrite=False,
            checkpointer=CheckpointerConfig(
                save_thread_count=3,
                load_thread_count=8,
                throttle_uploads=True,
            ),
            metrics_collect_interval=1 if HARD_STOP_STEPS else 10,
            cancel_check_interval=10,
            # Async metric callbacks can let ranks enqueue distributed bookkeeping
            # collectives in different orders when rank 0 stalls while logging to W&B.
            # Keep these collectives synchronous for long-running LC jobs.
            async_bookkeeping=ASYNC_BOOKKEEPING,
            max_duration=Duration.tokens(MAX_TOKENS),
            hard_stop=Duration.steps(HARD_STOP_STEPS) if HARD_STOP_STEPS else None,
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=SAVE_INTERVAL,
                ephemeral_save_interval=EPHEMERAL_SAVE_INTERVAL,
                # OLMoDDPTrainModule does not implement async checkpoint staging.
                save_async=False,
                pre_train_checkpoint=False,
                remove=CHECKPOINT_REMOVAL,
            ),
        )
        .with_callback("speed_monitor", SpeedMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("beaker", BeakerCallback())
        .with_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=NumpyPaddedFSLDatasetConfig.from_data_mix(
                    DataMix.v3_small_ppl_validation,
                    mix_base_dir=EVAL_DATA_ROOT,
                    sequence_length=EVAL_SEQUENCE_LENGTH,
                    tokenizer=common.tokenizer,
                    work_dir=common.work_dir,
                ),
                eval_interval=EVAL_INTERVAL,
                eval_duration=eval_duration,
                eval_on_finish=EVAL_ON_FINISH,
                enabled=EVALS_ENABLED,
            ),
        )
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=downstream_tasks,
                tokenizer=common.tokenizer,
                eval_interval=EVAL_INTERVAL,
                eval_duration=eval_duration,
                eval_on_finish=EVAL_ON_FINISH,
                enabled=EVALS_ENABLED,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=common.run_name,
                group=f"olmoe3-{MODEL_SIZE}-{FAMILY}-long-context",
                project="jacobm-olmoe-ladder",
                entity="ai2-llm",
                enabled=WANDB_ENABLED,
                cancel_check_interval=10,
                tags=[
                    "long-context",
                    MODEL_SIZE,
                    FAMILY,
                    "cx8",
                    "midtrained",
                    "olmo-ddp",
                    f"ep{EP_SIZE}",
                    EP_PATH.value if EP_SIZE > 1 else "no-ep",
                    "64k",
                    "smoke" if HARD_STOP_STEPS else "full-run",
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
        root_dir="/weka/oe-training-default/ai2-llm",
        work_dir=WORK_DIR,
        save_folder=os.path.join(SAVE_ROOT, cli_context.run_name),
        launch=None,
        tokenizer=tokenizer,
        max_sequence_length=max_sequence_length,
        global_batch_size=global_batch_size,
    )


def finalize_config(config: ExperimentConfig) -> None:
    if EXPECTED_WORLD_SIZE < 1 or EP_SIZE < 1 or EXPECTED_WORLD_SIZE % EP_SIZE:
        raise ValueError(f"EP size {EP_SIZE} must divide world size {EXPECTED_WORLD_SIZE}")
    if GLOBAL_BATCH_SIZE % SEQUENCE_LENGTH:
        raise ValueError("Global batch size must contain a whole number of sequences")
    expert_dp_degree = EXPECTED_WORLD_SIZE // EP_SIZE
    global_sequences = GLOBAL_BATCH_SIZE // SEQUENCE_LENGTH
    # The data loader shards the global sequence batch across every rank, including
    # ranks that participate in expert parallelism. Therefore gradient accumulation
    # is based on world size, not the number of DP groups.
    denominator = EXPECTED_WORLD_SIZE * RANK_MICROBATCH_SEQUENCES
    if global_sequences % denominator != 0:
        raise ValueError(
            f"Global sequence batch {global_sequences} is not divisible by "
            f"world_size*rank_microbatch_sequences={denominator}"
        )
    log.info(
        "Long-context config: tokens=%s seq_len=%s global_sequences=%s "
        "world=%s EP=%s EP_path=%s EP_DP=%s rank_microbatch_sequences=%s "
        "grad_accum_steps=%s lr=%s hard_stop_steps=%s compile=%s async_bookkeeping=%s",
        f"{MAX_TOKENS:,}",
        f"{SEQUENCE_LENGTH:,}",
        global_sequences,
        EXPECTED_WORLD_SIZE,
        EP_SIZE,
        EP_PATH.value if EP_SIZE > 1 else "off",
        expert_dp_degree,
        RANK_MICROBATCH_SEQUENCES,
        global_sequences // denominator,
        LEARNING_RATE,
        HARD_STOP_STEPS or "off",
        USE_COMPILE,
        ASYNC_BOOKKEEPING,
    )


def make_config(cli_context: CliContext) -> ExperimentConfig:
    if cli_context.cmd in {SubCmd.train, SubCmd.train_single, SubCmd.launch} and LOAD_PATH is None:
        raise OLMoConfigurationError("OLMOE3_LC_LOAD_PATH is required")
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
