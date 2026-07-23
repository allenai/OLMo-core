#!/usr/bin/env python3
"""Midtrain an active-matched integration-wide GDN hybrid checkpoint."""

from __future__ import annotations

import logging
import os
from functools import partial
from pathlib import Path
from typing import cast

from olmo_core.data import (
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.data.source_mixture import SourceMixtureDatasetConfig, SourceMixtureList
from olmo_core.internal.experiment import (
    CliContext,
    CommonComponents,
    DataComponents,
    ExperimentConfig,
    SubCmd,
    build_config,
    main,
)
from olmo_core.optim import ConstantWithWarmup
from olmo_core.train import Duration, LoadStrategy, TrainerConfig
from olmo_core.train.callbacks import WandBCallback
from olmo_core.train.train_module import OLMoDDPTrainModuleConfig
from scripts.train import jacobm_olmoe3_hybrid_scale as base


log = logging.getLogger(__name__)

SEED = 2026
LOAD_PATH = os.environ.get("OLMOE3_MT_LOAD_PATH")
MAX_TOKENS = int(os.environ.get("OLMOE3_MT_MAX_TOKENS", "100000000000"))
WARMUP_STEPS = int(os.environ.get("OLMOE3_MT_WARMUP_STEPS", "2000"))
SOURCE_MIX_PROCESSES = int(os.environ.get("OLMOE3_MT_SOURCE_PROCESSES", "16"))
SOURCE_MIX_YAML = os.environ.get(
    "OLMOE3_MT_SOURCE_MIX_YAML",
    str(
        Path(__file__).resolve().parents[2]
        / "olmo_core/data/source_mixtures/OLMo3-32B-midtraining-modelnamefilter.yaml"
    ),
)


def build_train_module_config(common: CommonComponents) -> OLMoDDPTrainModuleConfig:
    config = base.build_train_module_config(common)
    config.scheduler = ConstantWithWarmup(warmup=WARMUP_STEPS)
    config.reset_optimizer_states_on_load = True
    config.reset_optimizer_states_on_resume = False
    return config


def build_data_components(common: CommonComponents) -> DataComponents:
    source_list = SourceMixtureList.from_yaml(SOURCE_MIX_YAML)
    source_list.validate()
    dataset = NumpyFSLDatasetConfig.from_src_mix(
        src_mix=SourceMixtureDatasetConfig(
            source_list=source_list,
            requested_tokens=MAX_TOKENS,
            global_batch_size=common.global_batch_size,
            processes=SOURCE_MIX_PROCESSES,
            seed=SEED,
            render_tables=False,
            quiet=True,
        ),
        tokenizer=common.tokenizer,
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
    assert LOAD_PATH is not None
    trainer = base.build_trainer_config(common)
    trainer.load_path = LOAD_PATH
    trainer.load_strategy = LoadStrategy.always
    trainer.load_trainer_state = False
    trainer.load_optim_state = False
    trainer.max_duration = Duration.tokens(MAX_TOKENS)

    wandb = cast(WandBCallback, trainer.callbacks["wandb"])
    wandb.group = "olmoe3-integration-wide-hybrid-midtraining"
    wandb.tags = [
        "midtraining",
        base.MODEL_SIZE,
        "integration-wide",
        "expand-v-1",
        "hybrid",
        "gdn",
        "cx8",
        "olmo-ddp",
        f"ep{base.EP_SIZE}",
        "fresh-optimizer",
        "weight-only-load",
        "posthoc-validation",
    ]
    return trainer


def finalize_config(config: ExperimentConfig) -> None:
    assert LOAD_PATH is not None
    load_path = Path(LOAD_PATH)
    if not load_path.is_dir():
        raise ValueError(f"Midtraining source checkpoint does not exist: {load_path}")
    if MAX_TOKENS <= 0 or WARMUP_STEPS < 0:
        raise ValueError("MAX_TOKENS must be positive and WARMUP_STEPS must be nonnegative")
    if base.MODEL_VARIANT != "integration_wide_gdn_ev1":
        raise ValueError(
            "This first hybrid midtraining entrypoint requires integration_wide_gdn_ev1"
        )
    if base.WORLD_SIZE < 1 or base.EP_SIZE < 1 or base.WORLD_SIZE % base.EP_SIZE:
        raise ValueError("EP size must divide world size")
    global_sequences = base.GLOBAL_BATCH_SIZE // base.SEQUENCE_LENGTH
    if base.GLOBAL_BATCH_SIZE % base.SEQUENCE_LENGTH:
        raise ValueError("Global batch size must contain whole sequences")
    if global_sequences % base.WORLD_SIZE:
        raise ValueError("Global sequence batch must divide evenly across ranks")
    rank_sequences = global_sequences // base.WORLD_SIZE
    effective_microbatch = min(rank_sequences, base.RANK_MICROBATCH_SEQUENCES)
    if rank_sequences % effective_microbatch:
        raise ValueError("Rank sequence batch must divide evenly into microbatches")

    log.info(
        "Hybrid midtraining config: source=%s tokens=%s lr=%s warmup_steps=%s "
        "global_tokens=%s global_sequences=%s world=%s EP=%s rank_sequences=%s "
        "rank_microbatch=%s accumulation=%s active_params=%s total_params=%s",
        LOAD_PATH,
        f"{MAX_TOKENS:,}",
        base.LEARNING_RATE,
        WARMUP_STEPS,
        f"{base.GLOBAL_BATCH_SIZE:,}",
        global_sequences,
        base.WORLD_SIZE,
        base.EP_SIZE,
        rank_sequences,
        effective_microbatch,
        rank_sequences // effective_microbatch,
        f"{config.model.num_active_params:,}",
        f"{config.model.num_params:,}",
    )


def make_config(cli_context: CliContext) -> ExperimentConfig:
    if cli_context.cmd in {SubCmd.train, SubCmd.train_single, SubCmd.launch} and LOAD_PATH is None:
        raise ValueError("OLMOE3_MT_LOAD_PATH is required")
    builder = partial(
        build_config,
        common_config_builder=base.build_local_common_components,
        data_config_builder=build_data_components,
        model_config_builder=base.build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        tokenizer=TokenizerConfig.dolma2(),
        global_batch_size=base.GLOBAL_BATCH_SIZE,
        max_sequence_length=base.SEQUENCE_LENGTH,
        num_nodes=1,
        include_default_evals=False,
        finalize_config=finalize_config,
    )
    return cast(ExperimentConfig, builder(cli_context))


if __name__ == "__main__":
    main(config_builder=make_config)
