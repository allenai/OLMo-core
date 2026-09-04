"""Reproducible profiling harness for the small and medium OLMoE3 candidates.

The two presets intentionally expose only the best current production-scale
topologies. This is a throughput/profiling harness: it runs 100 steps by
default, writes no checkpoints, and performs no evaluations.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from olmoe3_small_medium_models import MODEL_SIZES, build_model_config

from olmo_core.config import DType
from olmo_core.data import (
    DataMix,
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.internal.common import get_beaker_username
from olmo_core.internal.experiment import CliContext, CommonComponents, DataComponents
from olmo_core.internal.experiment import (
    build_common_components as build_default_common_components,
)
from olmo_core.internal.experiment import build_config, main
from olmo_core.launch.beaker import BeakerEnvSecret, BeakerEnvVar
from olmo_core.launch.beaker_presets import get_preset
from olmo_core.optim import OLMoDDPOptimizerConfig, OptimGroupOverride
from olmo_core.optim.scheduler import WSD
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import SpeedMonitorCallback, WandBCallback
from olmo_core.train.train_module import (
    OLMoDDPTrainModuleConfig,
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
)

SEQUENCE_LENGTH = 8192
GLOBAL_BATCH_SIZE = 8 * 1024 * 1024
LEARNING_RATE = 8e-4
GPUS_PER_NODE = 8
MAX_STEPS = int(os.environ.get("OLMOE3_PROFILE_MAX_STEPS", "100"))
WORKSPACE = os.environ.get("OLMOE3_BEAKER_WORKSPACE", "ai2/olmo3p5-training")
BEAKER_PRIORITY = os.environ.get("OLMOE3_BEAKER_PRIORITY", "urgent")
ALLOWED_HOSTNAMES = tuple(
    hostname.strip()
    for hostname in os.environ.get("OLMOE3_ALLOWED_HOSTNAMES", "").split(",")
    if hostname.strip()
)
WANDB_PROJECT = os.environ.get("OLMOE3_WANDB_PROJECT", "olmoe3-production-profiling")
BEAKER_IMAGE = "akshitab/olmo-core-tch2110cu130-fa4-rma-2026-07-24"
PRESET = get_preset("olmo-ddp")


@dataclass(frozen=True)
class SystemConfig:
    """A production-scale profiling topology."""

    model_size: str
    num_nodes: int
    pp: int
    ep: int
    rank_microbatch_sequences: int

    @property
    def num_gpus(self) -> int:
        return self.num_nodes * GPUS_PER_NODE

    @property
    def dense_dp(self) -> int:
        return self.num_gpus // self.pp

    @property
    def gradient_accumulation_steps(self) -> int:
        global_batch_sequences = GLOBAL_BATCH_SIZE // SEQUENCE_LENGTH
        return global_batch_sequences // (self.dense_dp * self.rank_microbatch_sequences)

    def validate(self) -> None:
        global_batch_sequences = GLOBAL_BATCH_SIZE // SEQUENCE_LENGTH
        if self.model_size not in MODEL_SIZES:
            raise ValueError(f"Unknown model size {self.model_size!r}")
        if self.num_gpus % self.pp:
            raise ValueError(f"PP={self.pp} must divide {self.num_gpus} GPUs")
        if self.dense_dp % self.ep:
            raise ValueError(f"EP={self.ep} must divide dense DP={self.dense_dp}")
        microbatch_wave = self.dense_dp * self.rank_microbatch_sequences
        if global_batch_sequences % microbatch_wave:
            raise ValueError(
                f"{global_batch_sequences} global sequences must divide by "
                f"dense_dp={self.dense_dp} * rank MB={self.rank_microbatch_sequences}"
            )
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient accumulation must be positive")


SYSTEMS = {
    "small-64g": SystemConfig(
        model_size="small",
        num_nodes=8,
        pp=1,
        ep=1,
        rank_microbatch_sequences=4,
    ),
    "medium-128g": SystemConfig(
        model_size="medium",
        num_nodes=16,
        pp=1,
        ep=8,
        rank_microbatch_sequences=2,
    ),
}


def selected_system() -> tuple[str, SystemConfig]:
    preset_name = os.environ.get("OLMOE3_PROFILE_PRESET")
    if preset_name is None:
        raise ValueError(f"Set OLMOE3_PROFILE_PRESET to one of {tuple(SYSTEMS)}")
    try:
        system = SYSTEMS[preset_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown OLMOE3_PROFILE_PRESET={preset_name!r}; choose from {tuple(SYSTEMS)}"
        ) from exc
    system.validate()
    return preset_name, system


def build_common_components(
    cli_context: CliContext,
    preset_name: str,
    system: SystemConfig,
    **kwargs,
) -> CommonComponents:
    common = build_default_common_components(cli_context, **kwargs)
    if (launch := common.launch) is not None:
        beaker_user = get_beaker_username()
        if beaker_user is None:
            raise RuntimeError("Could not determine Beaker username")
        secret_prefix = beaker_user.lower()
        launch.workspace = WORKSPACE
        launch.priority = BEAKER_PRIORITY
        launch.min_runtime = "30m"
        launch.preemptible = False
        launch.budget = "ai2/oe-other"
        launch.gh_token_secret = f"{secret_prefix}_GITHUB_TOKEN"
        launch.beaker_image = BEAKER_IMAGE
        env = dict(PRESET.env_vars)
        env.update(
            {
                "S3_PROFILE": "default",
                "PYTHONPATH": "src",
                "OLMOE3_PROFILE_PRESET": preset_name,
                "OLMOE3_PROFILE_MAX_STEPS": str(MAX_STEPS),
                # Fail instead of silently allocating a new symmetric buffer after prewarm.
                "OLMO_EP_NO_SYNC_FORBID_RUNTIME_SYMM_ALLOC": "1",
            }
        )
        for name in (
            "OLMO_DISTRIBUTED_TIMEOUT_SECONDS",
            "OLMO_ROWWISE_VERBOSE_DEBUG_PRINT",
            "OLMO_ROWWISE_DEBUG_RANKS",
        ):
            if value := os.environ.get(name):
                env[name] = value
        launch.env_vars = [BeakerEnvVar(name=name, value=value) for name, value in env.items()]
        launch.post_setup = PRESET.post_setup
        launch.env_secrets = [
            BeakerEnvSecret(name=name, secret=f"{secret_prefix}_{suffix}", required=True)
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
        if ALLOWED_HOSTNAMES:
            launch.clusters = []
            launch.gpu_types = None
            launch.tags = None
            launch.hostnames = list(ALLOWED_HOSTNAMES)
        launch.follow = False
        launch.step_soft_timeout = None
    return common


def build_data_components(common: CommonComponents) -> DataComponents:
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


def build_model_config_from_common(common: CommonComponents, system: SystemConfig):
    return build_model_config(
        system.model_size,
        eos_token_id=common.tokenizer.eos_token_id,
        vocab_size=common.tokenizer.padded_vocab_size(),
    )


def build_train_module_config(
    common: CommonComponents,
    system: SystemConfig,
) -> OLMoDDPTrainModuleConfig:
    return OLMoDDPTrainModuleConfig(
        rank_microbatch_size=system.rank_microbatch_sequences * common.max_sequence_length,
        max_sequence_length=common.max_sequence_length,
        optim=OLMoDDPOptimizerConfig(
            lr=LEARNING_RATE,
            weight_decay=0.1,
            betas=(0.9, 0.95),
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
        scheduler=WSD(warmup=2, decay=2, decay_fraction=None),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.ddp,
            reduce_grads_in_fp32=True,
            accumulate_grads_in_fp32=True,
            use_reduce_scatter=False,
        ),
        ep_config=(TransformerExpertParallelConfig(degree=system.ep) if system.ep > 1 else None),
        pp_config=None,
        tp_config=None,
        cp_config=None,
        ac_config=None,
        float8_config=None,
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )


def build_trainer_config(
    common: CommonComponents,
    preset_name: str,
    system: SystemConfig,
) -> TrainerConfig:
    model = build_model_config_from_common(common, system)
    tags = [
        "olmoe3-production-candidate",
        "profile-handoff",
        f"system:{preset_name}",
        f"size:{system.model_size}",
        f"gpus:{system.num_gpus}",
        f"pp:{system.pp}",
        f"ep:{system.ep}",
        f"mb:{system.rank_microbatch_sequences}",
        f"grad-accum:{system.gradient_accumulation_steps}",
        "emo:true",
        "emo-min-pool:16",
        "emo-max-pool:512",
        "kda:cute-pr837",
        "moe:fused-v2",
        "mxfp8:false",
        "recompute:false",
        "shared-ep-outputs:false",
        "reduce-scatter:false",
    ]
    return (
        TrainerConfig(
            save_folder=f"/tmp/olmoe3-small-medium-profile/{common.run_name}",
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
                group="olmoe3-small-medium-profile-bf16",
                project=WANDB_PROJECT,
                entity="ai2-llm",
                enabled=True,
                tags=tags,
                notes=(
                    f"{model.num_active_params:,} active / {model.num_params:,} total; "
                    "512 experts, top-16; EMO 16->512; BF16; "
                    f"PP={system.pp}, EP={system.ep}, rank MB={system.rank_microbatch_sequences}, "
                    f"gradient accumulation={system.gradient_accumulation_steps}; "
                    "CuTe KDA PR837; no recomputation, MXFP8, shared EP outputs, or "
                    "reduce-scatter; WSD 2-step warmup / stable / 2-step decay"
                ),
                cancel_check_interval=10,
            ),
        )
    )


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(f"Usage: {sys.argv[0]} <subcmd> <run_name> <cluster> [overrides...]")
    preset_name, system = selected_system()
    config_builder = partial(
        build_config,
        global_batch_size=GLOBAL_BATCH_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        num_nodes=system.num_nodes,
        common_config_builder=partial(
            build_common_components,
            preset_name=preset_name,
            system=system,
        ),
        data_config_builder=build_data_components,
        model_config_builder=partial(build_model_config_from_common, system=system),
        train_module_config_builder=partial(build_train_module_config, system=system),
        trainer_config_builder=partial(
            build_trainer_config,
            preset_name=preset_name,
            system=system,
        ),
        beaker_image=BEAKER_IMAGE,
        beaker_workspace=WORKSPACE,
        include_default_evals=False,
        num_execution_units=1,
    )
    main(config_builder=config_builder)
