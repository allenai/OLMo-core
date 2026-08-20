"""Checkpoint-free 8 Mi-token throughput tests for the provisional final family.

Example::

    python src/examples/olmo_ddp/olmoe3_final_family_throughput.py launch \
        final-family-0p5b-mb16 ai2/holmes
"""

from __future__ import annotations

import os
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from olmoe3_final_family import MODEL_SIZES, build_model_config, geometry

from olmo_core.config import DType
from olmo_core.data import (
    DataMix,
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.internal.common import get_beaker_username
from olmo_core.internal.experiment import (
    CliContext,
    CommonComponents,
    DataComponents,
    build_common_components as build_default_common_components,
    build_config,
    main,
)
from olmo_core.launch.beaker import BeakerEnvSecret, BeakerEnvVar
from olmo_core.launch.beaker_presets import get_preset
from olmo_core.optim import OLMoDDPOptimizerConfig, OptimGroupOverride
from olmo_core.optim.scheduler import CosWithWarmup
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import SpeedMonitorCallback, WandBCallback
from olmo_core.train.train_module import (
    OLMoDDPTrainModuleConfig,
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
)

SEQUENCE_LENGTH = 8192
GLOBAL_BATCH_SIZE = 8 * 1024 * 1024
MAX_STEPS = 50
WORKSPACE = "ai2/OLMo-3-moe-experiments"
WANDB_PROJECT = "olmoe3-final-family-throughput"
BEAKER_IMAGE = "akshitab/olmo-core-tch2110cu130-fa4-rma-2026-07-24"
PRESET = get_preset("olmo-ddp")

# Initial 64-GPU matrix. These all exactly divide the 1,024-sequence global
# batch; no production training setting is changed by this qualification.
SYSTEMS = {
    "0p5b": {"ep": 1, "rank_microbatch_sequences": 8},
    "0p9b": {"ep": 1, "rank_microbatch_sequences": 4},
    "2p0b": {"ep": 4, "rank_microbatch_sequences": 2},
    "3p8b": {"ep": 8, "rank_microbatch_sequences": 1},
}


def build_common_components(cli_context: CliContext, **kwargs) -> CommonComponents:
    """Apply the ordinary OLMoDDP launch environment to this isolated test."""

    common = build_default_common_components(cli_context, **kwargs)
    if (launch := common.launch) is not None:
        beaker_user = get_beaker_username()
        if beaker_user is None:
            raise RuntimeError("Could not determine Beaker username")
        secret_prefix = beaker_user.lower()
        launch.workspace = WORKSPACE
        launch.priority = "urgent"
        launch.min_runtime = "30m"
        launch.preemptible = None
        launch.budget = "ai2/oe-other"
        launch.gh_token_secret = f"{secret_prefix}_GITHUB_TOKEN"
        launch.beaker_image = BEAKER_IMAGE
        launch.no_python = True
        env = dict(PRESET.env_vars)
        env.update({"S3_PROFILE": "default", "PYTHONPATH": "src"})
        launch.env_vars = [
            BeakerEnvVar(name=name, value=value) for name, value in env.items()
        ]
        launch.post_setup = PRESET.post_setup
        launch.env_secrets = [
            BeakerEnvSecret(
                name=name,
                secret=f"{secret_prefix}_{suffix}",
                required=True,
            )
            for name, suffix in (
                ("BEAKER_TOKEN", "BEAKER_TOKEN"),
                ("WANDB_API_KEY", "WANDB_API_KEY"),
                ("AWS_ACCESS_KEY_ID", "AWS_ACCESS_KEY_ID"),
                ("AWS_SECRET_ACCESS_KEY", "AWS_SECRET_ACCESS_KEY"),
            )
        ]
        launch.aws_config_secret = f"{secret_prefix}_AWS_CONFIG"
        launch.aws_credentials_secret = f"{secret_prefix}_AWS_CREDENTIALS"
        launch.google_credentials_secret = f"{secret_prefix}_GOOGLE_CREDENTIALS"
        launch.shared_memory = "128GiB"
        launch.follow = False
        launch.step_soft_timeout = None
    return common


def build_data_components(common: CommonComponents) -> DataComponents:
    """Use the production 8K Dolma 3.5 input pipeline."""

    dataset = NumpyFSLDatasetConfig.from_data_mix(
        DataMix.Dolma3p5_14t,
        tokenizer=common.tokenizer,
        mix_base_dir="s3://ai2-llm",
        work_dir=common.work_dir,
        sequence_length=common.max_sequence_length,
        max_target_sequence_length=common.max_sequence_length,
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
            seed=928_543_231,
            num_workers=8,
            prefetch_factor=8,
            num_threads=4,
        ),
    )


def build_model_config_from_common(
    common: CommonComponents, model_size: str
):
    return build_model_config(
        model_size,
        vocab_size=common.tokenizer.padded_vocab_size(),
    )


def build_train_module_config(
    common: CommonComponents, model_size: str
) -> OLMoDDPTrainModuleConfig:
    system = SYSTEMS[model_size]
    ep = system["ep"]
    return OLMoDDPTrainModuleConfig(
        rank_microbatch_size=(
            system["rank_microbatch_sequences"] * common.max_sequence_length
        ),
        max_sequence_length=common.max_sequence_length,
        optim=OLMoDDPOptimizerConfig(
            lr=8e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(
                    params=["embeddings.weight"],
                    opts={"weight_decay": 0.0},
                ),
                OptimGroupOverride(
                    params=["*routed_experts.w_up_gate", "*routed_experts.w_down"],
                    opts={},
                ),
            ],
            compile=True,
            dtype=DType.float32,
            sigma_factor=6,
            max_grad_norm=1.0,
            use_distributed=True,
        ),
        scheduler=CosWithWarmup(warmup_steps=10),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.ddp,
            reduce_grads_in_fp32=True,
            accumulate_grads_in_fp32=True,
        ),
        ep_config=(TransformerExpertParallelConfig(degree=ep) if ep > 1 else None),
        pp_config=None,
        tp_config=None,
        cp_config=None,
        ac_config=None,
        float8_config=None,
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )


def build_trainer_config(common: CommonComponents, model_size: str) -> TrainerConfig:
    g = geometry(model_size)
    system = SYSTEMS[model_size]
    tags = [
        "final-family-throughput",
        f"size:{model_size}",
        f"ep:{system['ep']}",
        f"mb:{system['rank_microbatch_sequences']}",
        "gbs:8Mi",
    ]
    return (
        TrainerConfig(
            save_folder=f"/tmp/olmoe3-final-family-throughput/{common.run_name}",
            save_overwrite=True,
            no_checkpoints=True,
            no_evals=True,
            metrics_collect_interval=1,
            cancel_check_interval=10,
            max_duration=Duration.steps(MAX_STEPS),
        )
        .with_callback("speed_monitor", SpeedMonitorCallback())
        .with_callback(
            "wandb",
            WandBCallback(
                name=common.run_name,
                group="olmoe3-final-family-8Mi-64gpu",
                project=WANDB_PROJECT,
                entity="ai2-llm",
                enabled=True,
                tags=tags,
                notes=(
                    f"{g.expected_active_params:,} active / "
                    f"{g.expected_total_params:,} total parameters"
                ),
                cancel_check_interval=10,
            ),
        )
    )


def parse_model_size(value: str) -> str:
    normalized = value.lower()
    padded = f"-{normalized}-"
    for model_size in MODEL_SIZES:
        if f"-{model_size}-" in padded:
            return model_size
    raise ValueError(
        f"Run name {value!r} must contain one of these model sizes: {MODEL_SIZES}"
    )


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(
            f"Usage: {sys.argv[0]} <subcmd> <run_name> <cluster> [overrides...]"
        )

    model_size = parse_model_size(sys.argv[2])
    config_builder = partial(
        build_config,
        global_batch_size=GLOBAL_BATCH_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        num_nodes=8,
        common_config_builder=build_common_components,
        data_config_builder=build_data_components,
        model_config_builder=partial(
            build_model_config_from_common,
            model_size=model_size,
        ),
        train_module_config_builder=partial(
            build_train_module_config,
            model_size=model_size,
        ),
        trainer_config_builder=partial(
            build_trainer_config,
            model_size=model_size,
        ),
        beaker_image=BEAKER_IMAGE,
        beaker_workspace=WORKSPACE,
        include_default_evals=False,
        num_execution_units=1,
    )
    main(config_builder=config_builder)
