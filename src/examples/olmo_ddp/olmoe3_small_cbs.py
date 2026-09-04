"""Small OLMoE3 critical-batch pilot with durable Weka checkpoints.

The reference trajectory trains with an 8 Mi-token batch through 100.66B
tokens.  The branch resumes from reference step 4,000 with a 16 Mi-token
batch and a square-root-scaled learning rate, reaching the same token horizon.

Examples::

    OLMOE3_SMALL_CBS_PHASE=8mi python src/examples/olmo_ddp/olmoe3_small_cbs.py \
        launch olmoe3-small-cbs-8mi-100b-lr1p3em3-uploader-r1 ai2/holmes

    OLMOE3_SMALL_CBS_PHASE=16mi python src/examples/olmo_ddp/olmoe3_small_cbs.py \
        launch olmoe3-small-cbs-16mi-from-step4000-lr1p85em3-uploader-r1 ai2/holmes
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from functools import partial
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from olmoe3_deep_family import build_model_config
from olmoe3_final_family import NUM_EXPERTS, TOP_K

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
from olmo_core.launch.beaker import BeakerEnvSecret, BeakerEnvVar, BeakerWekaBucket
from olmo_core.launch.beaker_presets import get_preset
from olmo_core.nn.moe import EmoRouterConfig
from olmo_core.optim import OLMoDDPOptimizerConfig, OptimGroupOverride
from olmo_core.optim.config import INITIAL_LR_FIELD, LR_FIELD
from olmo_core.optim.scheduler import WSD
from olmo_core.train import Duration, LoadStrategy, TrainerConfig
from olmo_core.train.callbacks import (
    Callback,
    CheckpointReadyNotifierCallback,
    CheckpointerCallback,
    CheckpointRemovalStrategy,
    SpeedMonitorCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    OLMoDDPTrainModuleConfig,
    TransformerDataParallelConfig,
)

SEQUENCE_LENGTH = 8192
NUM_NODES = 8
RANK_MICROBATCH_SEQUENCES = 4
WARMUP_STEPS = 2_000
TARGET_TOKENS = 100_663_296_000
REFERENCE_LR = 1.3e-3
BRANCH_LR = 1.85e-3
REFERENCE_STEP = 4_000

CHECKPOINT_MOUNT = "/weka/olmo-3p5-checkpoints"
CHECKPOINT_BASE = f"{CHECKPOINT_MOUNT}/production-cbs"
UPLOADER_INBOX = f"{CHECKPOINT_MOUNT}/uploader/control/inbox"
REFERENCE_RUN_ID = "olmoe3-small-cbs-8mi-100b-lr1p3em3-uploader-r1"
BRANCH_RUN_ID = "olmoe3-small-cbs-16mi-from-step4000-lr1p85em3-uploader-r1"
REFERENCE_CHECKPOINT = f"{CHECKPOINT_BASE}/{REFERENCE_RUN_ID}/step{REFERENCE_STEP}"

# Keep this pilot on the already-qualified S3 source while the one-time local
# Dolma 3.5 mirror is still being completed. Both lineages must use the same
# source and fingerprint when restoring trainer/data-loader state.
DATA_ROOT = "s3://ai2-llm"
WORK_DIR = "/tmp/olmoe3-small-cbs-dataset-cache"

WORKSPACE = os.environ.get("OLMOE3_BEAKER_WORKSPACE", "ai2/olmo3p5-training")
BEAKER_PRIORITY = os.environ.get("OLMOE3_BEAKER_PRIORITY", "urgent")
BEAKER_IMAGE = "akshitab/olmo-core-tch2110cu130-fa4-rma-2026-07-24"
PRESET = get_preset("olmo-ddp")


@dataclass(frozen=True)
class Phase:
    name: str
    run_id: str
    global_batch_size: int
    learning_rate: float
    checkpoint_interval: int
    load_path: str | None

    @property
    def gradient_accumulation_steps(self) -> int:
        sequences = self.global_batch_size // SEQUENCE_LENGTH
        return sequences // (NUM_NODES * 8 * RANK_MICROBATCH_SEQUENCES)


PHASES = {
    "8mi": Phase(
        name="8mi",
        run_id=REFERENCE_RUN_ID,
        global_batch_size=8 * 1024 * 1024,
        learning_rate=REFERENCE_LR,
        checkpoint_interval=500,
        load_path=None,
    ),
    "16mi": Phase(
        name="16mi",
        run_id=BRANCH_RUN_ID,
        global_batch_size=16 * 1024 * 1024,
        learning_rate=BRANCH_LR,
        checkpoint_interval=250,
        load_path=REFERENCE_CHECKPOINT,
    ),
}


def selected_phase() -> Phase:
    name = os.environ.get("OLMOE3_SMALL_CBS_PHASE", "8mi").lower()
    try:
        phase = PHASES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown OLMOE3_SMALL_CBS_PHASE={name!r}; choose from {tuple(PHASES)}"
        ) from exc
    if phase.gradient_accumulation_steps < 1:
        raise ValueError("gradient accumulation must be positive")
    return phase


def emo_config(common: CommonComponents) -> EmoRouterConfig:
    config = EmoRouterConfig(
        eos_token_id=common.tokenizer.eos_token_id,
        min_document_expert_pool=16,
        max_document_expert_pool=NUM_EXPERTS,
        eval_document_expert_pool=NUM_EXPERTS,
    )
    config.validate_for_router(num_experts=NUM_EXPERTS, top_k=TOP_K)
    return config


@dataclass
class LearningRateGuardCallback(Callback):
    """Fail before training if resume changed the configured peak LR."""

    expected_lr: float = 0.0

    def _check(self, source: str) -> None:
        optim = getattr(self.trainer.train_module, "optim", None)
        if optim is None:
            raise RuntimeError("OLMoDDP optimizer is not available for LR validation")
        for index, group in enumerate(optim.param_groups):
            for field in (LR_FIELD, INITIAL_LR_FIELD):
                value: Any = group.get(field)
                if value is None:
                    raise RuntimeError(f"optimizer group {index} is missing {field!r}")
                if hasattr(value, "detach"):
                    value = value.detach().float().cpu().item()
                actual = float(value)
                if not math.isclose(actual, self.expected_lr, rel_tol=1e-6, abs_tol=1e-9):
                    raise RuntimeError(
                        f"{source}: optimizer group {index} {field}={actual:.12g}; "
                        f"expected {self.expected_lr:.12g}"
                    )

    def post_checkpoint_loaded(self, path) -> None:
        self._check(f"after loading {path}")

    def pre_train(self) -> None:
        self._check("before training")


def build_common_components(cli_context: CliContext, phase: Phase, **kwargs) -> CommonComponents:
    if cli_context.run_name != phase.run_id:
        raise ValueError(
            f"Phase {phase.name!r} requires run name {phase.run_id!r}, got {cli_context.run_name!r}"
        )
    common = build_default_common_components(cli_context, **kwargs)
    if (launch := common.launch) is not None:
        beaker_user = get_beaker_username()
        if beaker_user is None:
            raise RuntimeError("Could not determine Beaker username")
        secret_prefix = beaker_user.lower()
        launch.workspace = WORKSPACE
        launch.priority = BEAKER_PRIORITY
        launch.min_runtime = "8h"
        launch.preemptible = None
        launch.retries = 3
        launch.budget = "ai2/oe-other"
        launch.gh_token_secret = f"{secret_prefix}_GITHUB_TOKEN"
        launch.beaker_image = BEAKER_IMAGE
        launch.weka_buckets = [
            BeakerWekaBucket("olmo-3p5-checkpoints", CHECKPOINT_MOUNT),
        ]
        launch.shared_filesystem = True
        env = dict(PRESET.env_vars)
        env.update(
            {
                "S3_PROFILE": "default",
                "PYTHONPATH": "src",
                "OLMOE3_SMALL_CBS_PHASE": phase.name,
            }
        )
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
        launch.follow = False
        launch.step_soft_timeout = None
    common.save_folder = f"{CHECKPOINT_BASE}/{phase.run_id}"
    common.work_dir = WORK_DIR
    return common


def build_data_components(common: CommonComponents) -> DataComponents:
    dataset = NumpyFSLDatasetConfig.from_data_mix(
        DataMix.Dolma3p5_14t,
        tokenizer=common.tokenizer,
        mix_base_dir=DATA_ROOT,
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


def build_model_config_from_common(common: CommonComponents):
    model = build_model_config(
        "small",
        vocab_size=common.tokenizer.padded_vocab_size(),
        emo=emo_config(common),
        mxfp8_mlp=False,
    )
    if model.recompute_each_block or model.recompute_all_blocks_by_chunk:
        raise ValueError("Recomputation is not allowed for the production candidate")
    return model


def build_train_module_config(common: CommonComponents, phase: Phase) -> OLMoDDPTrainModuleConfig:
    return OLMoDDPTrainModuleConfig(
        rank_microbatch_size=RANK_MICROBATCH_SEQUENCES * common.max_sequence_length,
        max_sequence_length=common.max_sequence_length,
        optim=OLMoDDPOptimizerConfig(
            lr=phase.learning_rate,
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
        # A one-step tail keeps this a WSD schedule without decaying any
        # optimizer update before the fixed 100.66B-token endpoint.
        scheduler=WSD(warmup=WARMUP_STEPS, decay=1, decay_fraction=None),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.ddp,
            reduce_grads_in_fp32=True,
            accumulate_grads_in_fp32=True,
            use_reduce_scatter=False,
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


def build_trainer_config(common: CommonComponents, phase: Phase) -> TrainerConfig:
    tags = [
        "small-cbs",
        f"phase:{phase.name}",
        "scheduler:wsd",
        "attention:default-scalable-softmax",
        "kda:cute-pr837",
        "moe:fused-v2",
        "recompute:false",
        "mxfp8-mlp:false",
        "share-ep-outputs:false",
        "reduce-scatter:false",
        "emo:true",
        "emo-min-pool:16",
        "emo-max-pool:512",
        "gpus:64",
        "pp:1",
        "ep:1",
        "mb:4",
        f"grad-accum:{phase.gradient_accumulation_steps}",
        f"gbs-tokens:{phase.global_batch_size}",
        f"peak-lr:{phase.learning_rate}",
    ]
    trainer = TrainerConfig(
        save_folder=common.save_folder,
        work_dir=common.work_dir,
        save_overwrite=False,
        load_path=phase.load_path,
        load_strategy=(LoadStrategy.always if phase.load_path else LoadStrategy.if_available),
        load_optim_state=True,
        load_trainer_state=True,
        metrics_collect_interval=1,
        cancel_check_interval=10,
        max_duration=Duration.tokens(TARGET_TOKENS),
        no_evals=True,
    )
    return (
        trainer.with_callback("speed_monitor", SpeedMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=phase.checkpoint_interval,
                ephemeral_save_interval=None,
                pre_train_checkpoint=None,
                # OLMoDDP checkpoints currently require the synchronous path.
                save_async=False,
                remove=CheckpointRemovalStrategy.never,
                max_checkpoints=None,
            ),
        )
        .with_callback(
            "checkpoint_ready_notifier",
            CheckpointReadyNotifierCallback(
                inbox_dir=UPLOADER_INBOX,
                run_id=phase.run_id,
                lineage_id=phase.run_id,
            ),
        )
        .with_callback(
            "learning_rate_guard",
            LearningRateGuardCallback(expected_lr=phase.learning_rate),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=phase.run_id,
                group="olmoe3-small-critical-batch",
                project="olmoe3-production-cbs",
                entity="ai2-llm",
                enabled=True,
                tags=tags,
                notes=(
                    "0.794B-active / 12.496B-total, 16-layer production small candidate; "
                    "512 experts, top-16; EMO 16->512; BF16; 64 B300 GPUs; PP1/EP1/MB4; "
                    "new CuTe KDA PR837; no recomputation, MXFP8, shared EP outputs, or "
                    f"reduce-scatter; phase={phase.name}, GBS={phase.global_batch_size:,}, "
                    f"peak LR={phase.learning_rate:.8g}, checkpoint interval={phase.checkpoint_interval}"
                ),
                cancel_check_interval=10,
            ),
        )
    )


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(f"Usage: {sys.argv[0]} <subcmd> <run_name> <cluster> [overrides...]")
    phase = selected_phase()
    config_builder = partial(
        build_config,
        global_batch_size=phase.global_batch_size,
        max_sequence_length=SEQUENCE_LENGTH,
        num_nodes=NUM_NODES,
        common_config_builder=partial(build_common_components, phase=phase),
        data_config_builder=build_data_components,
        model_config_builder=build_model_config_from_common,
        train_module_config_builder=partial(build_train_module_config, phase=phase),
        trainer_config_builder=partial(build_trainer_config, phase=phase),
        beaker_image=BEAKER_IMAGE,
        beaker_workspace=WORKSPACE,
        include_default_evals=False,
        num_execution_units=1,
    )
    main(config_builder=config_builder)
