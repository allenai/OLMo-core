"""
Shared SFT training logic for the OLMo hybrid small suite.

This module is imported by sft_think.py and sft_instruct.py.
It provides build functions and the launch entrypoint; each launcher script
only needs to supply a per-size config dict (with lr, global_batch_size,
load_path) and a dataset path.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
from functools import partial
from typing import Dict

from arch import MODEL_CONFIGS
from arch import build_model_config as arch_build_model_config
from arch import parse_model_size

from olmo_core.config import DType
from olmo_core.data import NumpyDataLoaderConfig, NumpyPackedFSLDatasetConfig
from olmo_core.data.types import LongDocStrategy
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.experiment import (
    CommonComponents,
    DataComponents,
    build_config,
    main,
)
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingMode,
    TransformerConfig,
)
from olmo_core.optim import LinearWithWarmup, SkipStepAdamWConfig
from olmo_core.train import Duration, LoadStrategy, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    SpeedMonitorCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

SEQUENCE_LENGTH = 32_768
SEED = 34521


def build_model_config(
    common: CommonComponents,
    model_size: str,
    sft_configs: Dict[str, dict],
    attn_backend: AttentionBackendName = AttentionBackendName.flash_3,
) -> TransformerConfig:
    model_config = arch_build_model_config(common, model_size, attn_backend=attn_backend)
    if sft_configs[model_size].get("fused_linear_loss", False):
        model_config.lm_head.loss_implementation = LMLossImplementation.fused_linear
    return model_config


def build_train_module_config(
    common: CommonComponents, model_size: str, sft_configs: Dict[str, dict]
) -> TransformerTrainModuleConfig:
    sft_cfg = sft_configs[model_size]

    return TransformerTrainModuleConfig(
        rank_microbatch_size=common.max_sequence_length,
        max_sequence_length=common.max_sequence_length,
        optim=SkipStepAdamWConfig(
            lr=sft_cfg["lr"],
            weight_decay=0.0,
            betas=(0.9, 0.95),
            compile=False,
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.budget,
            activation_memory_budget=1,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=None,
        max_grad_norm=1.0,
        scheduler=LinearWithWarmup(
            warmup_fraction=0.03,
            alpha_f=0.0,
        ),
    )


def build_data_components(
    common: CommonComponents,
    dataset_path: str,
) -> DataComponents:
    clean_path = dataset_path.rstrip("/")
    dataset_config = NumpyPackedFSLDatasetConfig(
        tokenizer=common.tokenizer,
        work_dir=common.work_dir,
        paths=[f"{clean_path}/token_ids_part_*.npy"],
        expand_glob=True,
        label_mask_paths=[f"{clean_path}/labels_mask_*.npy"],
        generate_doc_lengths=True,
        long_doc_strategy=LongDocStrategy.truncate,
        sequence_length=common.max_sequence_length,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=common.global_batch_size, seed=SEED, num_workers=4
    )

    return DataComponents(dataset=dataset_config, data_loader=data_loader_config)


def build_trainer_config(
    common: CommonComponents, model_size: str, sft_configs: Dict[str, dict], tags: list[str]
) -> TrainerConfig:
    cancel_check_interval = 10
    sft_cfg = sft_configs[model_size]

    run_name = f"{common.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"

    return (
        TrainerConfig(
            load_strategy=LoadStrategy.always,
            load_path=sft_cfg["load_path"],
            load_trainer_state=False,
            load_optim_state=False,
            save_folder=common.save_folder,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=cancel_check_interval,
            max_duration=Duration.epochs(2),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=500,
                save_async=True,
            ),
        )
        .with_callback("speed_monitor", SpeedMonitorCallback())
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                group=common.run_name,
                project="hybrid-small-suite",
                entity="ai2-llm",
                cancel_check_interval=cancel_check_interval,
                enabled=True,
                tags=tags + [model_size],
            ),
        )
    )


def run_sft(sft_configs: Dict[str, dict], dataset_path: str, tags: list[str]):
    """Common entrypoint for SFT launcher scripts."""
    if len(sys.argv) < 3:
        raise SystemExit(
            f"Usage: {sys.argv[0]} <subcmd> <run_name> <cluster> [overrides...]\n"
            f"Run name must contain a model size: {list(MODEL_CONFIGS.keys())}"
        )

    model_size = parse_model_size(sys.argv[2])
    cfg = MODEL_CONFIGS[model_size]
    sft_cfg = sft_configs[model_size]

    CLUSTER_ATTN_BACKENDS = {
        "saturn": AttentionBackendName.flash_2,
        "jupiter": AttentionBackendName.flash_3,
        "titan": AttentionBackendName.flash_4,
    }
    cluster_arg = " ".join(sys.argv[2:4]).lower()
    attn_backend = AttentionBackendName.flash_3
    for cluster, backend in CLUSTER_ATTN_BACKENDS.items():
        if cluster in cluster_arg:
            attn_backend = backend
            break

    config_builder = partial(
        build_config,
        global_batch_size=sft_cfg["global_batch_size"],
        max_sequence_length=SEQUENCE_LENGTH,
        num_nodes=cfg["num_nodes"],
        data_config_builder=partial(build_data_components, dataset_path=dataset_path),
        model_config_builder=partial(build_model_config, model_size=model_size, sft_configs=sft_configs, attn_backend=attn_backend),
        train_module_config_builder=partial(build_train_module_config, model_size=model_size, sft_configs=sft_configs),
        trainer_config_builder=partial(build_trainer_config, model_size=model_size, sft_configs=sft_configs, tags=tags),
        include_default_evals=False,
        beaker_workspace="ai2/linear-rnns",
        num_execution_units=1,
    )
    main(config_builder=config_builder)
