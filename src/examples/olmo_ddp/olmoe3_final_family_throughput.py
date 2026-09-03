"""Checkpoint-free 8 Mi-token throughput tests for the provisional final family.

Example::

    python src/examples/olmo_ddp/olmoe3_final_family_throughput.py launch \
        final-family-0p5b-mb16 ai2/holmes
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from olmoe3_emo import EMO_ENV_VARS, emo_note, emo_router_config, emo_tags
from olmoe3_final_family import MODEL_SIZES, build_model_config

from olmo_core.config import DType
from olmo_core.data import (
    DataMix,
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
)
from olmo_core.distributed.parallel import (
    DataParallelType,
    PipelineP2PBackend,
    PipelineScheduleType,
)
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
from olmo_core.train.train_module.transformer import TransformerPipelineParallelConfig

SEQUENCE_LENGTH = 8192
GLOBAL_BATCH_SIZE = 8 * 1024 * 1024
MAX_STEPS = int(os.environ.get("OLMOE3_THROUGHPUT_MAX_STEPS", "50"))
EP_CAPACITY_FACTOR = float(os.environ.get("OLMOE3_TEST_EP_CAPACITY_FACTOR", "1.25"))
EXPERT_VARIANT = os.environ.get("OLMOE3_EXPERT_VARIANT", "n512-k16-h1")
WORKSPACE = "ai2/OLMo-3-moe-experiments"
WANDB_PROJECT = "olmoe3-final-family-throughput"
BEAKER_IMAGE = "akshitab/olmo-core-tch2110cu130-fa4-rma-2026-07-24"
PRESET = get_preset("olmo-ddp")
GPUS_PER_NODE = 8
WSD_WARMUP_STEPS = 10
WSD_DECAY_STEPS = 10


@dataclass(frozen=True)
class SystemConfig:
    """One explicitly qualified model/parallelism configuration."""

    model_size: str
    num_nodes: int
    pp: int
    ep: int
    rank_microbatch_sequences: int
    recompute_each_block: bool = False

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
        """Fail before launch if a preset has invalid mesh or batch geometry."""

        global_batch_sequences = GLOBAL_BATCH_SIZE // SEQUENCE_LENGTH
        if self.num_nodes < 1:
            raise ValueError("num_nodes must be positive")
        if self.pp < 1 or self.num_gpus % self.pp:
            raise ValueError(f"PP={self.pp} must divide {self.num_gpus} GPUs")
        if self.ep < 1 or self.dense_dp % self.ep:
            raise ValueError(f"EP={self.ep} must divide dense DP={self.dense_dp}")
        if self.ep > GPUS_PER_NODE or GPUS_PER_NODE % self.ep:
            raise ValueError(f"EP={self.ep} must fit evenly within a {GPUS_PER_NODE}-GPU node")
        sequences_per_microbatch_wave = self.dense_dp * self.rank_microbatch_sequences
        if global_batch_sequences % sequences_per_microbatch_wave:
            raise ValueError(
                f"{global_batch_sequences} global sequences must divide by "
                f"dense_dp={self.dense_dp} * "
                f"rank_microbatch_sequences={self.rank_microbatch_sequences}"
            )


# Existing 8-GPU baselines remain the default when no preset is selected.
DEFAULT_SYSTEMS = {
    "0p5b": SystemConfig("0p5b", 1, 1, 1, 8),
    "0p9b": SystemConfig("0p9b", 1, 1, 1, 4),
    "2p0b": SystemConfig("2p0b", 1, 1, 4, 2),
    "3p8b": SystemConfig("3p8b", 1, 1, 8, 1, recompute_each_block=True),
}

# Ordered 64-GPU qualification matrix plus explicit OOM/topology fallbacks.
# EP is kept within each 8-GPU node; with PP as the leading mesh dimension,
# PP peers span nodes while EP rank groups remain node-local.
QUALIFICATION_SYSTEMS = {
    "8gpu-2p0b-pp1-ep4-mb1": SystemConfig("2p0b", 1, 1, 4, 1),
    "g64-0p5b-pp1-ep1-mb8": SystemConfig("0p5b", 8, 1, 1, 8),
    "g64-0p9b-pp1-ep1-mb4": SystemConfig("0p9b", 8, 1, 1, 4),
    "g64-2p0b-pp1-ep4-mb2": SystemConfig("2p0b", 8, 1, 4, 2),
    "g64-2p0b-pp1-ep8-mb4": SystemConfig("2p0b", 8, 1, 8, 4),
    "g64-2p0b-pp1-ep8-mb2": SystemConfig("2p0b", 8, 1, 8, 2),
    "g64-3p8b-pp2-ep8-mb2": SystemConfig("3p8b", 8, 2, 8, 2),
    "g64-3p8b-pp2-ep4-mb1": SystemConfig("3p8b", 8, 2, 4, 1),
    "g64-3p8b-pp2-ep8-mb1": SystemConfig("3p8b", 8, 2, 8, 1),
    "g64-3p8b-pp4-ep4-mb2": SystemConfig("3p8b", 8, 4, 4, 2),
}

EXPERT_VARIANTS = {
    "n512-k16-h1": {
        "num_experts": 512,
        "top_k": 16,
        "expert_hidden_multiplier": 1,
    },
    "n256-k8-h2": {
        "num_experts": 256,
        "top_k": 8,
        "expert_hidden_multiplier": 2,
    },
}


def expert_variant() -> dict[str, int]:
    """Return the selected routed-expert geometry."""

    try:
        return EXPERT_VARIANTS[EXPERT_VARIANT]
    except KeyError as exc:
        raise ValueError(
            f"Unknown OLMOE3_EXPERT_VARIANT={EXPERT_VARIANT!r}; "
            f"choose from {tuple(EXPERT_VARIANTS)}"
        ) from exc


def select_system(model_size: str) -> tuple[str, SystemConfig]:
    """Select and validate the requested system preset."""

    preset_name = os.environ.get("OLMOE3_THROUGHPUT_SYSTEM_PRESET")
    if preset_name is None:
        preset_name = f"8gpu-{model_size}-baseline"
        system = DEFAULT_SYSTEMS[model_size]
    else:
        try:
            system = QUALIFICATION_SYSTEMS[preset_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown OLMOE3_THROUGHPUT_SYSTEM_PRESET={preset_name!r}; "
                f"choose from {tuple(QUALIFICATION_SYSTEMS)}"
            ) from exc
        if system.model_size != model_size:
            raise ValueError(
                f"Preset {preset_name!r} is for {system.model_size}, but the run name "
                f"selects {model_size}"
            )
    system.validate()
    return preset_name, system


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
        env = dict(PRESET.env_vars)
        env.update({"S3_PROFILE": "default", "PYTHONPATH": "src"})
        # The train subcommand rebuilds its config inside the Beaker container,
        # so explicitly propagate any launch-time diagnostic overrides.
        for name in (
            "OLMOE3_THROUGHPUT_MAX_STEPS",
            "OLMOE3_THROUGHPUT_SYSTEM_PRESET",
            "OLMOE3_TEST_EP_CAPACITY_FACTOR",
            "OLMOE3_EXPERT_VARIANT",
            *EMO_ENV_VARS,
        ):
            if value := os.environ.get(name):
                env[name] = value
        launch.env_vars = [BeakerEnvVar(name=name, value=value) for name, value in env.items()]
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


def build_model_config_from_common(common: CommonComponents, system: SystemConfig):
    variant = expert_variant()
    emo = emo_router_config(
        eos_token_id=common.tokenizer.eos_token_id,
        num_experts=variant["num_experts"],
        top_k=variant["top_k"],
    )
    model = build_model_config(
        system.model_size,
        vocab_size=common.tokenizer.padded_vocab_size(),
        **variant,
        emo=emo,
    )
    # The original single-node 3.8B test needs block recomputation. The PP
    # qualification presets deliberately disable it to test the likely
    # production topology. Keep this as a system-specific setting rather than
    # baking it into the architecture definition.
    if system.model_size == "3p8b":
        model.recompute_each_block = system.recompute_each_block
        for block in [model.block, *model.block_overrides.values()]:
            if block.routed_experts is not None:
                block.ep.share_dispatch_out = True
                block.ep.share_combine_out = True
                block.ep.capacity_factor = EP_CAPACITY_FACTOR
    return model


def build_train_module_config(
    common: CommonComponents, system: SystemConfig
) -> OLMoDDPTrainModuleConfig:
    pp_config = None
    if system.pp > 1:
        pp_config = TransformerPipelineParallelConfig(
            degree=system.pp,
            schedule=PipelineScheduleType.custom_interleaved_1F1B,
            use_custom_stage_implementation=True,
            p2p_use_separate_group=True,
            p2p_backend=PipelineP2PBackend.nccl,
        )
    return OLMoDDPTrainModuleConfig(
        rank_microbatch_size=(system.rank_microbatch_sequences * common.max_sequence_length),
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
        scheduler=WSD(
            warmup=WSD_WARMUP_STEPS,
            decay=WSD_DECAY_STEPS,
            decay_fraction=None,
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.ddp,
            reduce_grads_in_fp32=True,
            accumulate_grads_in_fp32=True,
        ),
        ep_config=(TransformerExpertParallelConfig(degree=system.ep) if system.ep > 1 else None),
        pp_config=pp_config,
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
    variant = expert_variant()
    emo = emo_router_config(
        eos_token_id=common.tokenizer.eos_token_id,
        num_experts=variant["num_experts"],
        top_k=variant["top_k"],
    )
    model = build_model_config(
        system.model_size,
        vocab_size=common.tokenizer.padded_vocab_size(),
        **variant,
        emo=emo,
    )
    tags = [
        "final-family-throughput",
        "scheduler:wsd",
        f"system:{preset_name}",
        f"size:{system.model_size}",
        f"gpus:{system.num_gpus}",
        f"pp:{system.pp}",
        f"ep:{system.ep}",
        f"mb:{system.rank_microbatch_sequences}",
        f"grad-accum:{system.gradient_accumulation_steps}",
        f"recompute:{str(system.recompute_each_block).lower()}",
        f"ep-capacity:{EP_CAPACITY_FACTOR:g}",
        f"expert-variant:{EXPERT_VARIANT}",
        "gbs:8Mi",
        *emo_tags(emo),
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
                group=(f"olmoe3-final-family-8Mi-{system.num_gpus}gpu-wsd-{EXPERT_VARIANT}"),
                project=WANDB_PROJECT,
                entity="ai2-llm",
                enabled=True,
                tags=tags,
                notes=(
                    f"{model.num_active_params:,} active / "
                    f"{model.num_params:,} total parameters; "
                    f"{variant['num_experts']} experts, top-{variant['top_k']}, "
                    f"expert hidden multiplier {variant['expert_hidden_multiplier']}; "
                    f"PP={system.pp}, EP={system.ep}, "
                    f"rank MB={system.rank_microbatch_sequences}, "
                    f"grad accum={system.gradient_accumulation_steps}, "
                    f"recompute={system.recompute_each_block}; "
                    f"{emo_note(emo)}; "
                    f"WSD={WSD_WARMUP_STEPS} warmup / stable / "
                    f"{WSD_DECAY_STEPS} decay steps"
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
    raise ValueError(f"Run name {value!r} must contain one of these model sizes: {MODEL_SIZES}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(f"Usage: {sys.argv[0]} <subcmd> <run_name> <cluster> [overrides...]")

    model_size = parse_model_size(sys.argv[2])
    preset_name, system = select_system(model_size)
    config_builder = partial(
        build_config,
        global_batch_size=GLOBAL_BATCH_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        num_nodes=system.num_nodes,
        common_config_builder=build_common_components,
        data_config_builder=build_data_components,
        model_config_builder=partial(
            build_model_config_from_common,
            system=system,
        ),
        train_module_config_builder=partial(
            build_train_module_config,
            system=system,
        ),
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
