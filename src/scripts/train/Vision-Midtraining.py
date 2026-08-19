"""Joint text-and-vision midtraining from the locked vision-alignment checkpoint.

This recipe is deliberately separate from ``Molmo2-Stage1.py`` and
``Vision-Alignment.py``. It model-only loads the permanent vision-alignment treatment
checkpoint, keeps the vision encoder frozen, and trains the complete language model plus
connector on the original Stage-1 visual sources, optionally mixed with the historical OLMo 3
7B midtraining source mixture.

The deterministic ``audit`` command measures mean supervised loss weight for every source.
Real training requires that SHA-pinned receipt; synthetic smoke uses unit means. Existing
checkpoints in the output folder use Trainer's normal full-state resume path, while a fresh
output folder falls back to the pinned parent with fresh optimizer, trainer, and data state.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np

from olmo_core.config import Config, DType, StrEnum
from olmo_core.data import InstanceFilterConfig, NumpyFSLDatasetConfig, TokenizerConfig
from olmo_core.data.data_loader import DataLoaderBase
from olmo_core.data.multimodal import (
    CoSynPointDatasetConfig,
    MixtureDataLoader,
    MultimodalCollatorConfig,
    NumpyFSLTextDatasetConfig,
    PixMoCapDatasetConfig,
    PixMoCountDatasetConfig,
    PixMoPointsDatasetConfig,
)
from olmo_core.data.multimodal.mixture_weights import (
    expected_loss_mass,
    sampling_weights_from_loss_mass,
)
from olmo_core.data.multimodal.paths import PIXMO_DATASETS
from olmo_core.data.multimodal.vision_alignment_sources import (
    VISION_ALIGNMENT_TOKENIZER_FINGERPRINT,
    VISION_ALIGNMENT_TOKENIZER_ID,
    VISION_ALIGNMENT_TOKENIZER_REVISION,
    load_pinned_vision_alignment_tokenizer,
)
from olmo_core.data.source_mixture import SourceMixtureDatasetConfig, SourceMixtureList
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.internal.common import build_launch_config, get_root_dir
from olmo_core.launch.beaker import BeakerEnvVar, BeakerLaunchConfig
from olmo_core.nn.vision import Molmo2TokenIds, MultimodalLMConfig
from olmo_core.optim import (
    CosWithWarmup,
    OLMoDDPOptimizerConfig,
    OptimGroupOverride,
    PerGroupScheduler,
    SchedulerUnits,
)
from olmo_core.train import (
    Checkpointer,
    CheckpointerConfig,
    Duration,
    LoadStrategy,
    TrainerConfig,
    prepare_cli_environment,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    BeakerCallback,
    CheckpointerCallback,
    ConfigSaverCallback,
    ConsoleLoggerCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    MultimodalOLMoDDPTrainModuleConfig,
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
)
from olmo_core.utils import seed_all

log = logging.getLogger(__name__)

RECIPE_VERSION = 2
PARENT_CHECKPOINT = (
    "/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/"
    "checkpoints/vision-alignment-joint-v1/step12000"
)
PARENT_CONFIG_SHA256 = "64b302865831b5aaf11e86e142a85b3467a06b93d6c214fb67f7f94a45c4ddc8"
PARENT_MARKER_SHA256 = "77dfdeec42fe7990f4b3b9c4eeecd480edcf5066c110603b115920af38423d03"
PARENT_DCP_METADATA_SHA256 = "44cc94aa5b69bb774e45561062476d4e97a3d6ef3ff6e5ab40f53591a42a651f"

TOKENIZER_ID = VISION_ALIGNMENT_TOKENIZER_ID
TOKENIZER_REVISION = VISION_ALIGNMENT_TOKENIZER_REVISION
TOKENIZER_FINGERPRINT = VISION_ALIGNMENT_TOKENIZER_FINGERPRINT
HF_CACHE_DIR = "/weka/oe-training-default/rustin/hf-cache/hub"

EXPERIMENT_ROOT = "/weka/oe-training-default/rustin/experiments/vision-moe"
VISION_MIDTRAINING_ROOT = f"{EXPERIMENT_ROOT}/vision-midtraining"
BEAKER_CLUSTER = "ai2/holmes"
BEAKER_WORKSPACE = "ai2/molmofication"
BEAKER_BUDGET = "ai2/oe-other"
WANDB_PROJECT: str | None = "vision-midtraining"
WANDB_ENTITY: str | None = None

SEQUENCE_LENGTH = 8192
GLOBAL_BATCH_SIZE = 1_048_576
MAX_TOKENS = 10_485_760_000
EP_DEGREE = 8
DATA_SEED = 95818
INIT_SEED = 6198
MAX_CROPS = 8
PACK_BUFFER_SIZE = 48
PACK_MAX_CROPS = 16
PREFETCH_WORKERS = 8

LM_LR = 1e-5
CONNECTOR_LR = 2e-5
SCHEDULER_WARMUP_TOKENS = 209_715_200
SCHEDULER_ALPHA_F = 0.1
ROUTER_LB_LOSS_WEIGHT = 0.015
ROUTER_Z_LOSS_WEIGHT = 1e-4
MODEL_Z_LOSS_MULTIPLIER = 1e-4

TEXT_SOURCE_MIX_PATH = "src/olmo_core/data/source_mixtures/OLMo3-7B-midtraining.yaml"
TEXT_SOURCE_MIX_SHA256 = "15ee181c199bb89b118672737340153093ace8b5765fd87f760aa944669e2cff"
TEXT_SOURCE_MIX_SEED = 1337
TEXT_SOURCE_MIX_PROCESSES = 16
TEXT_WORK_DIR = f"{VISION_MIDTRAINING_ROOT}/data"
VISUAL_SOURCE_NAMES = (
    "cosyn_point",
    "pixmo_caption",
    "pixmo_count",
    "pixmo_points_basic",
    "pixmo_points_high_frequency",
    "pixmo_transcript",
)
TEXT_SOURCE_NAME = "text_midtraining"
_RUN_NAME_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,127}")
_AUDIT_ALGORITHM = "affine-coprime-v1"


class VisionMidtrainingArm(StrEnum):
    """Top-level text-to-vision supervised-loss-mass treatments."""

    v100 = "v100"
    vt50 = "vt50"
    vt80 = "vt80"
    vt90 = "vt90"
    t100 = "t100"


_TEXT_LOSS_MASS = {
    VisionMidtrainingArm.v100: 0.0,
    VisionMidtrainingArm.vt50: 0.5,
    VisionMidtrainingArm.vt80: 0.8,
    VisionMidtrainingArm.vt90: 0.9,
    VisionMidtrainingArm.t100: 1.0,
}
_VISUAL_LOSS_MASS = {
    "pixmo_caption": 0.50,
    "pixmo_transcript": 0.20,
    "pixmo_points_basic": 0.10,
    "pixmo_points_high_frequency": 0.04,
    "pixmo_count": 0.10,
    "cosyn_point": 0.06,
}


def _target_loss_mass(arm: VisionMidtrainingArm) -> dict[str, float]:
    """Derive the exact source loss-mass targets for an experiment arm."""
    text_mass = _TEXT_LOSS_MASS[arm]
    targets = (
        {
            name: round(visual_mass * (1.0 - text_mass), 12)
            for name, visual_mass in _VISUAL_LOSS_MASS.items()
        }
        if text_mass < 1.0
        else {}
    )
    if text_mass > 0:
        targets[TEXT_SOURCE_NAME] = text_mass
    return targets


def _active_source_names(arm: VisionMidtrainingArm) -> tuple[str, ...]:
    """Return the exact sources materialized by an experiment arm."""
    names: tuple[str, ...] = () if _TEXT_LOSS_MASS[arm] == 1.0 else VISUAL_SOURCE_NAMES
    if _TEXT_LOSS_MASS[arm] > 0:
        names = (*names, TEXT_SOURCE_NAME)
    return names


@dataclass
class ParentArtifactConfig(Config):
    """Pinned permanent checkpoint used for the fresh model-only transition."""

    checkpoint: str = PARENT_CHECKPOINT
    config_sha256: str = PARENT_CONFIG_SHA256
    marker_sha256: str = PARENT_MARKER_SHA256
    dcp_metadata_sha256: str = PARENT_DCP_METADATA_SHA256


@dataclass
class OptimizationConfig(Config):
    """Optimizer and token-scheduler choices for vision midtraining."""

    lm_lr: float = LM_LR
    connector_lr: float = CONNECTOR_LR
    scheduler_warmup_tokens: int = SCHEDULER_WARMUP_TOKENS
    scheduler_alpha_f: float = SCHEDULER_ALPHA_F
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8


@dataclass
class VisionMidtrainingDataConfig(Config):
    """Stage-1 visual sources plus the pinned OLMo 3 7B midtraining source list."""

    sequence_length: int = SEQUENCE_LENGTH
    pixmo_cap_path: str = f"{PIXMO_DATASETS}/cap"
    text_source_mix_path: str = TEXT_SOURCE_MIX_PATH
    text_source_mix_sha256: str = TEXT_SOURCE_MIX_SHA256
    text_work_dir: str = TEXT_WORK_DIR
    message_format: str = "document"
    loss_token_weighting: str = "none"
    max_crops: int = MAX_CROPS
    mean_loss_weight_receipt: str | None = None
    mean_loss_weight_receipt_sha256: str | None = None
    audit_output_path: str | None = None
    audit_samples_per_source: int = 4096
    audit_seed: int = 6198
    synthetic_smoke: bool = False
    synthetic_size: int = 256
    pack_sequences: bool = True
    pack_buffer_size: int = PACK_BUFFER_SIZE
    pack_max_crops: int = PACK_MAX_CROPS
    prefetch_workers: int = PREFETCH_WORKERS
    mixture_arm: VisionMidtrainingArm = VisionMidtrainingArm("vt50")


@dataclass
class VisionMidtrainingMetadataConfig(Config):
    """Small lineage record saved with every midtraining checkpoint."""

    recipe_version: int = RECIPE_VERSION
    parent_checkpoint: str = PARENT_CHECKPOINT
    parent_config_sha256: str = PARENT_CONFIG_SHA256
    source_contract_sha256: str = ""
    run_contract_sha256: str = ""


@dataclass
class ExperimentConfig(Config):
    """Complete configuration for one vision-midtraining run."""

    launch: BeakerLaunchConfig
    model: MultimodalLMConfig
    text_dataset: NumpyFSLTextDatasetConfig
    collator: MultimodalCollatorConfig
    train_module: MultimodalOLMoDDPTrainModuleConfig
    trainer: TrainerConfig
    parent: ParentArtifactConfig = field(default_factory=ParentArtifactConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    data: VisionMidtrainingDataConfig = field(default_factory=VisionMidtrainingDataConfig)
    vision_midtraining: VisionMidtrainingMetadataConfig = field(
        default_factory=VisionMidtrainingMetadataConfig
    )
    global_batch_size: int = GLOBAL_BATCH_SIZE
    max_tokens: int = MAX_TOKENS
    hard_stop_tokens: int | None = None
    data_seed: int = DATA_SEED
    init_seed: int = INIT_SEED
    checkpoint_interval: int = 2500
    ephemeral_checkpoint_interval: int | None = 500
    max_checkpoints: int = 6
    wandb_enabled: bool = True
    checkpoint_load_threads: int = 8
    required_run_name: str = ""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _strict_json_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"JSON object repeats key {key!r}")
        value[key] = item
    return value


def _load_parent_model_config(
    parent: ParentArtifactConfig,
) -> tuple[MultimodalLMConfig, Mapping[str, Any]]:
    """Verify and deserialize the exact multimodal model config from the parent."""
    checkpoint = Path(parent.checkpoint)
    config_path = checkpoint / "config.json"
    marker_path = checkpoint / ".metadata.json"
    dcp_metadata_path = checkpoint / "model_and_optim" / ".metadata"
    expected = (
        (config_path, parent.config_sha256, "config"),
        (marker_path, parent.marker_sha256, "permanent marker"),
        (dcp_metadata_path, parent.dcp_metadata_sha256, "DCP metadata"),
    )
    for path, digest, label in expected:
        if not path.is_file() or _sha256_file(path) != digest:
            raise ValueError(f"Parent {label} fingerprint mismatch for {path}")

    try:
        config = json.loads(config_path.read_bytes(), object_pairs_hook=_strict_json_object)
        marker = json.loads(marker_path.read_bytes(), object_pairs_hook=_strict_json_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Could not parse pinned parent checkpoint: {error}") from error
    if not isinstance(config, Mapping) or not isinstance(marker, Mapping):
        raise TypeError("Pinned parent config and marker must be JSON objects")
    if marker.get("ephemeral") is not False:
        raise ValueError("Vision midtraining requires a permanent parent checkpoint")
    alignment = config.get("vision_alignment")
    if not isinstance(alignment, Mapping) or alignment.get("phase") != "joint":
        raise ValueError("Vision midtraining parent must be the completed joint alignment phase")
    train_module = config.get("train_module")
    if not isinstance(train_module, Mapping):
        raise TypeError("Vision-alignment parent lacks its train-module config")
    freeze_patterns = train_module.get("freeze_params") or []
    if any(fnmatch.fnmatch("vision.patch_embedding.weight", str(p)) for p in freeze_patterns):
        raise ValueError("The locked treatment parent unexpectedly froze its vision encoder")
    model = config.get("model")
    if not isinstance(model, Mapping):
        raise TypeError("Vision-alignment parent lacks a multimodal model config")
    return MultimodalLMConfig.from_dict(dict(model)), config


def _load_tokenizer():
    return load_pinned_vision_alignment_tokenizer(
        identifier=TOKENIZER_ID,
        revision=TOKENIZER_REVISION,
        expected_fingerprint=TOKENIZER_FINGERPRINT,
        cache_dir=HF_CACHE_DIR,
    )


def _load_text_source_list(data: VisionMidtrainingDataConfig) -> SourceMixtureList:
    """Load the exact historical OLMo 3 7B midtraining source list."""
    if data.text_source_mix_path != TEXT_SOURCE_MIX_PATH:
        raise ValueError("The pinned OLMo 3 7B text source-list path may not be overridden")
    if data.text_source_mix_sha256 != TEXT_SOURCE_MIX_SHA256:
        raise ValueError("The pinned OLMo 3 7B text source-list SHA-256 may not be overridden")
    path = Path(data.text_source_mix_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[3] / path
    if not path.is_file() or _sha256_file(path) != data.text_source_mix_sha256:
        raise ValueError(f"Text source-list fingerprint mismatch for {path}")
    source_list = SourceMixtureList.from_yaml(path)
    source_list.validate()
    return source_list


def _build_text_dataset_config(
    data: VisionMidtrainingDataConfig,
    *,
    requested_tokens: int,
    global_batch_size: int,
) -> NumpyFSLTextDatasetConfig:
    """Wrap the pinned OLMo 3 7B NumpyFSL source mixture for multimodal collation."""
    tokenizer = TokenizerConfig.dolma2()
    dataset = NumpyFSLDatasetConfig.from_src_mix(
        SourceMixtureDatasetConfig(
            source_list=_load_text_source_list(data),
            requested_tokens=requested_tokens,
            global_batch_size=global_batch_size,
            processes=TEXT_SOURCE_MIX_PROCESSES,
            seed=TEXT_SOURCE_MIX_SEED,
        ),
        tokenizer=tokenizer,
        work_dir=data.text_work_dir,
        sequence_length=data.sequence_length,
        max_target_sequence_length=data.sequence_length,
        instance_filter_config=InstanceFilterConfig(
            repetition_max_period=13,
            repetition_min_period=1,
            repetition_max_count=32,
        ),
    )
    return NumpyFSLTextDatasetConfig(dataset=dataset)


def _build_train_module_config(
    optimization: OptimizationConfig,
    *,
    sequence_length: int,
) -> MultimodalOLMoDDPTrainModuleConfig:
    """Build frozen-vision OLMoDDP training with full-LM and connector updates."""
    no_decay_patterns = [
        "*lm.embeddings.weight",
        "*lm.embedding_norm.*",
        "*lm.blocks.*norm*.weight",
        "*lm.lm_head.norm.*",
    ]
    group_overrides = [
        OptimGroupOverride(
            params=["*connector.*"],
            opts={
                "lr": optimization.connector_lr,
                "weight_decay": 0.0,
                "scheduler_name": "connector",
            },
        ),
        OptimGroupOverride(
            params=["*vision.*"],
            opts={"lr": 0.0, "weight_decay": 0.0, "scheduler_name": "vision"},
        ),
        OptimGroupOverride(params=no_decay_patterns, opts={"weight_decay": 0.0}),
    ]
    scheduler = CosWithWarmup(
        warmup=optimization.scheduler_warmup_tokens,
        alpha_f=optimization.scheduler_alpha_f,
        units=SchedulerUnits.tokens,
    )
    return MultimodalOLMoDDPTrainModuleConfig(
        rank_microbatch_size=sequence_length,
        max_sequence_length=sequence_length,
        optim=OLMoDDPOptimizerConfig(
            lr=optimization.lm_lr,
            betas=optimization.betas,
            eps=optimization.eps,
            weight_decay=optimization.weight_decay,
            group_overrides=group_overrides,
            compile=False,
            foreach_chunk_size=50_000_000,
            sigma_factor=12,
            max_grad_norm=1.0,
            clip_grad_norm_by_scheduler_group=True,
            check_nan_inf_grad=True,
            use_distributed=True,
        ),
        freeze_params=["vision.*"],
        train_embedding_rows=None,
        vision_activation_checkpointing=False,
        connector_activation_checkpointing=True,
        response_logits_only=True,
        diagnostics_interval=100,
        z_loss_multiplier=MODEL_Z_LOSS_MULTIPLIER,
        max_grad_norm=1.0,
        compile_model=True,
        scheduler=PerGroupScheduler(
            schedulers={
                "connector": scheduler,
                "vision": scheduler,
            },
            default=scheduler,
        ),
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.ddp,
            reduce_dtype=DType.float32,
            only_allreduce_last_microbatch=True,
            reduce_grads_in_fp32=True,
            accumulate_grads_in_fp32=True,
        ),
        ep_config=TransformerExpertParallelConfig(degree=EP_DEGREE),
    )


def _build_source_configs(config: ExperimentConfig, token_ids: Molmo2TokenIds) -> dict[str, Any]:
    """Build the active named source configs without materializing their datasets."""
    data = config.data
    active_names = _active_source_names(data.mixture_arm)
    if data.synthetic_smoke and active_names == (TEXT_SOURCE_NAME,):
        raise ValueError("The text-only arm requires the real pinned text source")
    if active_names == (TEXT_SOURCE_NAME,):
        return {TEXT_SOURCE_NAME: config.text_dataset}
    if data.synthetic_smoke:
        return {
            name: PixMoCapDatasetConfig(
                dataset_path="synthetic",
                mode="caption",
                max_crops=data.max_crops,
                max_sequence_length=data.sequence_length,
                loss_token_weighting=data.loss_token_weighting,
                token_ids=token_ids,
                message_format=data.message_format,
                seed=index,
                synthetic_size=data.synthetic_size,
            )
            for index, name in enumerate(active_names)
        }

    shared = {
        "split": "train",
        "require_split": True,
        "max_crops": data.max_crops,
        "max_sequence_length": data.sequence_length,
        "loss_token_weighting": data.loss_token_weighting,
        "token_ids": token_ids,
        "message_format": data.message_format,
        "seed": 0,
    }
    sources = {
        "cosyn_point": CoSynPointDatasetConfig(**shared),
        "pixmo_caption": PixMoCapDatasetConfig(
            dataset_path=data.pixmo_cap_path,
            mode="caption",
            **shared,
        ),
        "pixmo_count": PixMoCountDatasetConfig(
            mode="grounded",
            counting="both",
            **shared,
        ),
        "pixmo_points_basic": PixMoPointsDatasetConfig(
            kind="basic",
            counting="both",
            both_mode="duplicate",
            **shared,
        ),
        "pixmo_points_high_frequency": PixMoPointsDatasetConfig(
            kind="high_frequency",
            counting="both",
            both_mode="duplicate",
            **shared,
        ),
        "pixmo_transcript": PixMoCapDatasetConfig(
            dataset_path=data.pixmo_cap_path,
            mode="transcript",
            require_transcript=True,
            **shared,
        ),
    }
    if TEXT_SOURCE_NAME in active_names:
        sources[TEXT_SOURCE_NAME] = config.text_dataset
    return sources


class _TextOnlyCollator:
    """Remove the mixed-arm dummy image after validating a globally text-only batch."""

    def __init__(self, collator: Any):
        self.collator = collator
        self.pad_sequence_length = collator.pad_sequence_length

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        batch = self.collator(examples)
        if bool(batch["image_crop_counts"].any()) or bool(batch["pooled_token_counts"].any()):
            raise ValueError("The text-only arm received an example with visual inputs")
        batch.pop("images")
        batch.pop("pooled_patches_idx")
        return batch


def _build_collator(config: ExperimentConfig):
    """Build the shared collator, skipping its dummy vision forward for the text-only arm."""
    collator = config.collator.build()
    if config.data.mixture_arm is VisionMidtrainingArm.t100:
        return _TextOnlyCollator(collator)
    return collator


def _source_contract_sha256(config: ExperimentConfig, token_ids: Molmo2TokenIds) -> str:
    source_configs = _build_source_configs(config, token_ids)
    return _canonical_sha256(
        {
            "sequence_length": config.data.sequence_length,
            "tokenizer": {
                "identifier": TOKENIZER_ID,
                "revision": TOKENIZER_REVISION,
                "fingerprint": TOKENIZER_FINGERPRINT,
            },
            "sources": {
                name: source_configs[name].as_config_dict() for name in sorted(source_configs)
            },
        }
    )


def _run_contract_sha256(config: ExperimentConfig) -> str:
    """Hash the complete training-state contract, excluding launch-only settings."""
    payload = config.as_config_dict()
    payload.pop("launch")
    payload["vision_midtraining"].pop("run_contract_sha256")
    return _canonical_sha256(payload)


def _validate_output_resume_contract(config: ExperimentConfig) -> None:
    """Reject a same-folder resume whose saved recipe identity differs."""
    try:
        checkpoint = Checkpointer.latest_checkpoint(config.trainer.save_folder)
    except FileNotFoundError:
        return
    path = Path(checkpoint) / "config.json"
    try:
        saved = json.loads(path.read_bytes(), object_pairs_hook=_strict_json_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Invalid resume checkpoint config {path}: {error}") from error
    if not isinstance(saved, Mapping):
        raise TypeError(f"Resume checkpoint config must be a JSON object: {path}")
    saved_metadata = saved.get("vision_midtraining")
    if not isinstance(saved_metadata, Mapping):
        raise TypeError(f"Resume checkpoint lacks vision-midtraining metadata: {path}")
    expected_metadata = config.vision_midtraining.as_config_dict()
    identity_fields = (
        "recipe_version",
        "parent_checkpoint",
        "parent_config_sha256",
        "source_contract_sha256",
        "run_contract_sha256",
    )
    if any(saved_metadata.get(key) != expected_metadata[key] for key in identity_fields):
        raise ValueError(f"Resume checkpoint belongs to a different run contract: {path}")
    if saved.get("parent") != config.parent.as_config_dict():
        raise ValueError(f"Resume checkpoint has a different pinned parent identity: {path}")


def _validate_router_objectives(model: MultimodalLMConfig) -> None:
    """Require the locked s002 router load-balancing and router z objectives unchanged."""
    routers = []
    for block in [model.lm.block, *(model.lm.block_overrides or {}).values()]:
        router = getattr(block, "routed_experts_router", None)
        if router is not None:
            routers.append(router)
    if not routers:
        raise ValueError("The locked s002 language model has no routed-expert router configs")
    if any(router.lb_loss_weight != ROUTER_LB_LOSS_WEIGHT for router in routers):
        raise ValueError("Vision midtraining must preserve s002 router load balancing")
    if any(router.z_loss_weight != ROUTER_Z_LOSS_WEIGHT for router in routers):
        raise ValueError("Vision midtraining must preserve s002 router z loss")


def _build_console_logger() -> ConsoleLoggerCallback:
    callback = ConsoleLoggerCallback()
    callback.metrics.extend(
        ["data/*", "multimodal/*", "optim/* grad norm", "optim/* clip coefficient"]
    )
    return callback


def _build_trainer_config(config: ExperimentConfig, run_name: str) -> TrainerConfig:
    """Build fresh-parent transition fields while retaining Trainer's exact resume behavior."""
    return (
        TrainerConfig(
            save_folder=f"{VISION_MIDTRAINING_ROOT}/checkpoints/{run_name}",
            checkpointer=CheckpointerConfig(load_thread_count=config.checkpoint_load_threads),
            save_overwrite=True,
            load_path=config.parent.checkpoint,
            load_strategy=LoadStrategy.always,
            load_optim_state=False,
            load_trainer_state=False,
            metrics_collect_interval=5,
            cancel_check_interval=5,
            max_duration=Duration.tokens(config.max_tokens),
            hard_stop=(
                Duration.tokens(config.hard_stop_tokens)
                if config.hard_stop_tokens is not None
                else None
            ),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=config.checkpoint_interval,
                ephemeral_save_interval=config.ephemeral_checkpoint_interval,
                save_async=False,
                pre_train_checkpoint=True,
                max_checkpoints=config.max_checkpoints,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                entity=WANDB_ENTITY,
                project=WANDB_PROJECT,
                enabled=config.wandb_enabled,
                cancel_check_interval=10,
                auto_resume=True,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("garbage_collector", GarbageCollectorCallback())
        .with_callback("beaker", BeakerCallback())
        .with_callback("console_logger", _build_console_logger())
    )


def _configure_launch_runtime(launch_config: BeakerLaunchConfig) -> None:
    from olmo_core.launch.beaker_presets import get_preset

    preset = get_preset("olmo-ddp")
    if preset.beaker_image is not None:
        launch_config.beaker_image = preset.beaker_image
    env = {item.name: item.value for item in launch_config.env_vars}
    env.update(dict(preset.env_vars))
    env.update(
        {
            "OLMO_USE_OWN_SYMM_MEM": "1",
            "OLMO_EP_MP_HIGH_PRIORITY_GROUP": "1",
            "OLMO_OWN_SYMM_PREWARM": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "8",
            "TORCH_LOGS": "-dynamo",
        }
    )
    launch_config.env_vars = [BeakerEnvVar(name=name, value=value) for name, value in env.items()]
    launch_config.post_setup = preset.post_setup


def _configure_launch_data_credentials(
    launch_config: BeakerLaunchConfig,
    data: VisionMidtrainingDataConfig,
) -> None:
    """Derive the narrow cloud-credential surface from the selected data arm."""
    launch_config.google_credentials_secret = (
        "GOOGLE_CREDENTIALS"
        if not data.synthetic_smoke and TEXT_SOURCE_NAME in _active_source_names(data.mixture_arm)
        else None
    )


def _normalized(values: Mapping[str, float]) -> dict[str, float]:
    total = float(sum(values.values()))
    return {name: float(value) / total for name, value in values.items()}


def _validated_mean_receipt(config: ExperimentConfig) -> dict[str, float]:
    """Load mean loss weights from the SHA-pinned deterministic audit summary."""
    path_value = config.data.mean_loss_weight_receipt
    digest = config.data.mean_loss_weight_receipt_sha256
    if not path_value or not digest:
        raise ValueError("Real midtraining requires a pinned source-mean audit summary")
    path = Path(path_value)
    if not path.is_file() or _sha256_file(path) != digest:
        raise ValueError(f"Source-mean audit summary fingerprint mismatch for {path}")
    try:
        receipt = json.loads(path.read_bytes(), object_pairs_hook=_strict_json_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Invalid source-mean audit summary {path}: {error}") from error
    if not isinstance(receipt, Mapping) or receipt.get("version") != 1:
        raise ValueError("Source-mean audit summary must use version 1")
    expected_metadata = {
        "algorithm": _AUDIT_ALGORITHM,
        "sequence_length": config.data.sequence_length,
        "tokenizer_fingerprint": TOKENIZER_FINGERPRINT,
        "audit_seed": config.data.audit_seed,
        "samples_per_source": config.data.audit_samples_per_source,
    }
    if any(receipt.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError("Source-mean audit summary metadata differs from the data contract")
    if receipt.get("source_contract_sha256") != config.vision_midtraining.source_contract_sha256:
        raise ValueError("Source-mean audit summary belongs to a different data contract")
    sources = receipt.get("sources")
    if not isinstance(sources, Mapping):
        raise TypeError("Source-mean audit summary lacks source measurements")
    expected_sources = set(_active_source_names(config.data.mixture_arm))
    if set(sources) != expected_sources:
        raise ValueError("Source-mean audit summary measures the wrong experiment sources")
    measured = {
        name: float(value["mean_loss_weight"])
        for name, value in sources.items()
        if isinstance(name, str) and isinstance(value, Mapping) and "mean_loss_weight" in value
    }
    if set(measured) != expected_sources:
        raise ValueError("Source-mean audit summary contains malformed source measurements")
    if any(value <= 0 or not math.isfinite(value) for value in measured.values()):
        raise ValueError("Source-mean audit summary contains an invalid mean loss weight")
    return measured


def _sampling_weights(config: ExperimentConfig) -> dict[str, float]:
    """Return calibrated example sampling probabilities and verify reconstructed mass."""
    target = _target_loss_mass(config.data.mixture_arm)
    source_names = _active_source_names(config.data.mixture_arm)
    means = (
        {name: 1.0 for name in source_names}
        if config.data.synthetic_smoke
        else _validated_mean_receipt(config)
    )
    weights = sampling_weights_from_loss_mass(target, means)
    delivered = expected_loss_mass(weights, means)
    expected = _normalized(target)
    if any(not math.isclose(delivered[name], expected[name], abs_tol=1e-12) for name in expected):
        raise ValueError("Calibrated sampling weights do not reconstruct target loss mass")
    return weights


def _validate_contract(
    config: ExperimentConfig,
    run_name: str,
    command: str,
) -> None:
    """Validate the narrow parent, trainability, data, and resume contract."""
    if _RUN_NAME_RE.fullmatch(run_name) is None or config.required_run_name != run_name:
        raise ValueError("Invalid or inconsistent vision-midtraining run name")
    if config.parent != ParentArtifactConfig():
        raise ValueError("Pinned parent artifact identities may not be overridden")
    metadata = config.vision_midtraining
    if metadata.parent_checkpoint != config.parent.checkpoint:
        raise ValueError("Saved midtraining lineage differs from the pinned parent")
    if metadata.parent_config_sha256 != config.parent.config_sha256:
        raise ValueError("Saved parent config fingerprint differs from the pin")

    if config.data.sequence_length != SEQUENCE_LENGTH:
        raise ValueError("Vision midtraining is fixed at sequence length 8192")
    if config.collator.pad_sequence_length != config.data.sequence_length:
        raise ValueError("Collator and data sequence lengths must match")
    if config.train_module.max_sequence_length != config.data.sequence_length:
        raise ValueError("Train module and data sequence lengths must match")
    if config.global_batch_size <= 0 or config.global_batch_size % SEQUENCE_LENGTH:
        raise ValueError("Global batch must be a positive multiple of sequence length")
    if config.max_tokens <= 0 or config.max_tokens % config.global_batch_size:
        raise ValueError("Token duration must be a positive whole number of global batches")
    if config.hard_stop_tokens is not None and (
        config.hard_stop_tokens <= 0
        or config.hard_stop_tokens >= config.max_tokens
        or config.hard_stop_tokens % config.global_batch_size
    ):
        raise ValueError(
            "Hard stop must be a positive whole number of global batches below max tokens"
        )
    if config.trainer.max_duration != Duration.tokens(config.max_tokens):
        raise ValueError("Vision midtraining duration must use token units")
    expected_hard_stop = (
        Duration.tokens(config.hard_stop_tokens) if config.hard_stop_tokens is not None else None
    )
    if config.trainer.hard_stop != expected_hard_stop:
        raise ValueError("Trainer hard stop differs from the configured pilot boundary")
    if config.trainer.load_path != config.parent.checkpoint:
        raise ValueError("Fresh midtraining must model-load the exact parent")
    if config.trainer.load_strategy is not LoadStrategy.always:
        raise ValueError("Fresh midtraining must require its parent checkpoint")
    if (
        config.trainer.load_optim_state is not False
        or config.trainer.load_trainer_state is not False
    ):
        raise ValueError("The parent transition requires fresh optimizer, trainer, and data state")
    if config.trainer.save_overwrite is not True:
        raise ValueError("save_overwrite=True is required for exact same-folder resumes")
    if (
        config.checkpoint_load_threads <= 0
        or config.trainer.checkpointer.load_thread_count != config.checkpoint_load_threads
    ):
        raise ValueError("Checkpoint load threading must be positive and wired to the trainer")

    train_module = config.train_module
    if train_module.freeze_params != ["vision.*"]:
        raise ValueError("Vision midtraining freezes only the vision encoder")
    if train_module.train_embedding_rows is not None:
        raise ValueError("Vision midtraining trains the complete LM embedding table")
    if train_module.vision_activation_checkpointing:
        raise ValueError("Frozen vision must not use activation checkpointing")
    if train_module.dp_config.name is not DataParallelType.ddp:
        raise ValueError("Vision midtraining requires OLMoDDP")
    if train_module.ep_config.degree != EP_DEGREE:
        raise ValueError("Vision midtraining requires EP=8")
    if train_module.optim.lr != config.optimization.lm_lr:
        raise ValueError("Default optimizer group must use the configured LM learning rate")
    if train_module.optim.weight_decay != config.optimization.weight_decay:
        raise ValueError("Default optimizer group must use midtraining weight decay")
    groups = {tuple(group.params): group.opts for group in train_module.optim.group_overrides or []}
    if groups.get(("*vision.*",), {}).get("lr") != 0.0:
        raise ValueError("Frozen vision must have an explicit zero-LR optimizer contract")
    if groups.get(("*connector.*",), {}).get("lr") != config.optimization.connector_lr:
        raise ValueError("Connector group must use its configured learning rate")

    target_loss_mass = _target_loss_mass(config.data.mixture_arm)
    if any(value <= 0 or not math.isfinite(value) for value in target_loss_mass.values()):
        raise ValueError("Every active target loss mass must be finite and positive")
    if not math.isclose(sum(target_loss_mass.values()), 1.0, abs_tol=1e-12):
        raise ValueError("Derived target loss mass must sum to one")
    if not config.data.synthetic_smoke:
        _load_text_source_list(config.data)
    if config.data.message_format != "document" or config.data.loss_token_weighting != "none":
        raise ValueError("Visual data must retain the Stage-1 formatter and loss weighting")
    if not config.data.pack_sequences:
        raise ValueError("Vision midtraining requires multimodal buffered packing")
    if train_module.source_loss_mass_targets != target_loss_mass:
        raise ValueError("Train-module source telemetry differs from the data target")

    receipt_fields = (
        config.data.mean_loss_weight_receipt,
        config.data.mean_loss_weight_receipt_sha256,
    )
    if bool(receipt_fields[0]) != bool(receipt_fields[1]):
        raise ValueError("Source-mean audit receipt path and SHA-256 must be set together")
    if command == "audit":
        if config.data.synthetic_smoke or not config.data.audit_output_path:
            raise ValueError("A source audit requires real data and an output path")
    elif command in {"train", "launch"}:
        if config.data.synthetic_smoke and any(receipt_fields):
            raise ValueError("Synthetic smoke must use internal unit source means")
        _sampling_weights(config)
    elif config.data.synthetic_smoke or any(receipt_fields):
        _sampling_weights(config)
    if config.launch.hostnames or config.launch.clusters != [BEAKER_CLUSTER]:
        raise ValueError("Vision midtraining may target the Holmes cluster, not exact hosts")
    if config.launch.workspace != BEAKER_WORKSPACE or config.launch.budget != BEAKER_BUDGET:
        raise ValueError("Vision midtraining workspace and budget are pinned")
    if config.launch.num_gpus != 8 or config.launch.num_nodes != 2:
        raise ValueError("Vision midtraining requires exactly two complete 8-GPU Holmes nodes")
    expected_google_secret = (
        "GOOGLE_CREDENTIALS"
        if not config.data.synthetic_smoke
        and TEXT_SOURCE_NAME in _active_source_names(config.data.mixture_arm)
        else None
    )
    if config.launch.google_credentials_secret != expected_google_secret:
        raise ValueError("Google credentials must be derived from the selected data arm")
    if {secret.name for secret in config.launch.env_secrets} != {
        "BEAKER_TOKEN",
        "WANDB_API_KEY",
    }:
        raise ValueError("Vision midtraining launch has an unexpected environment-secret surface")
    if metadata.run_contract_sha256 != _run_contract_sha256(config):
        raise ValueError("Derived run contract fingerprint is inconsistent")
    _validate_output_resume_contract(config)


def build_config(
    script: str,
    run_name: str,
    overrides: list[str],
) -> ExperimentConfig:
    """Build a vision-midtraining config from ordinary OLMo-core CLI overrides."""
    if _RUN_NAME_RE.fullmatch(run_name) is None:
        raise ValueError(f"Invalid vision-midtraining run name {run_name!r}")
    parent = ParentArtifactConfig()
    parent_model, _ = _load_parent_model_config(parent)
    tokenizer, token_ids = _load_tokenizer()
    if tokenizer.pad_token_id is None:
        raise ValueError(f"Tokenizer {TOKENIZER_ID!r} has no pad token")
    data = VisionMidtrainingDataConfig()
    optimization = OptimizationConfig()
    text_dataset = _build_text_dataset_config(
        data,
        requested_tokens=MAX_TOKENS,
        global_batch_size=GLOBAL_BATCH_SIZE,
    )
    collator = MultimodalCollatorConfig(
        pad_token_id=int(tokenizer.pad_token_id),
        label_ignore_index=-100,
        pad_sequence_length=data.sequence_length,
    )
    train_module = _build_train_module_config(optimization, sequence_length=data.sequence_length)
    placeholder = ExperimentConfig(
        launch=BeakerLaunchConfig(name=run_name, cmd=[]),
        model=parent_model,
        text_dataset=text_dataset,
        collator=collator,
        train_module=train_module,
        trainer=TrainerConfig(save_folder="unused"),
        parent=parent,
        optimization=optimization,
        data=data,
        required_run_name=run_name,
    )
    launch = build_launch_config(
        name=run_name,
        root_dir=get_root_dir(BEAKER_CLUSTER),
        cmd=[script, "train", run_name, *overrides],
        cluster=BEAKER_CLUSTER,
        workspace=BEAKER_WORKSPACE,
        budget=BEAKER_BUDGET,
        num_nodes=2,
    )
    launch.aws_config_secret = None
    launch.aws_credentials_secret = None
    launch.env_secrets = [
        secret for secret in launch.env_secrets if secret.name in ("BEAKER_TOKEN", "WANDB_API_KEY")
    ]
    _configure_launch_runtime(launch)
    placeholder.launch = launch

    config = placeholder.merge(overrides)
    if config.parent != parent:
        raise ValueError("Pinned parent artifact identities may not be overridden")
    if config.model != parent_model:
        raise ValueError("The exact deserialized parent model config may not be overridden")
    _configure_launch_data_credentials(config.launch, config.data)
    config.text_dataset = _build_text_dataset_config(
        config.data,
        requested_tokens=config.max_tokens,
        global_batch_size=config.global_batch_size,
    )
    config.collator = MultimodalCollatorConfig(
        pad_token_id=int(tokenizer.pad_token_id),
        label_ignore_index=-100,
        pad_sequence_length=config.data.sequence_length,
    )
    config.train_module = _build_train_module_config(
        config.optimization, sequence_length=config.data.sequence_length
    )
    config.train_module.source_loss_mass_targets = _target_loss_mass(config.data.mixture_arm)
    config.trainer = _build_trainer_config(config, run_name)
    config.vision_midtraining.parent_checkpoint = config.parent.checkpoint
    config.vision_midtraining.parent_config_sha256 = config.parent.config_sha256
    config.vision_midtraining.source_contract_sha256 = _source_contract_sha256(config, token_ids)
    config.vision_midtraining.run_contract_sha256 = _run_contract_sha256(config)
    _validate_router_objectives(config.model)
    return config


def _materialize_sources(
    tokenizer, token_ids: Molmo2TokenIds, config: ExperimentConfig
) -> tuple[list[Any], list[str]]:
    source_configs = _build_source_configs(config, token_ids)
    names = sorted(source_configs)
    datasets = []
    for name in names:
        source_config = source_configs[name]
        dataset = (
            source_config.build()
            if isinstance(source_config, NumpyFSLTextDatasetConfig)
            else source_config.build(tokenizer)
        )
        prepare = getattr(dataset, "prepare", None)
        if prepare is not None:
            prepare()
        datasets.append(dataset)
    return datasets, names


def _audit_indices(size: int, count: int, seed: int) -> list[int]:
    if size <= 0 or count <= 0:
        raise ValueError("Audit source size and sample count must be positive")
    count = min(size, count)
    start = seed % size
    stride = 2 * seed + 1
    while math.gcd(stride, size) != 1:
        stride += 2
    return [(start + index * stride) % size for index in range(count)]


def audit_source_means(config: ExperimentConfig) -> Path:
    """Measure deterministic mean loss weights and write a compact JSON summary."""
    output = Path(cast(str, config.data.audit_output_path))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit summary {output}")
    tokenizer, token_ids = _load_tokenizer()
    datasets, names = _materialize_sources(tokenizer, token_ids, config)
    sources: dict[str, Any] = {}
    for source_index, (name, dataset) in enumerate(zip(names, datasets)):
        indices = _audit_indices(
            len(dataset),
            config.data.audit_samples_per_source,
            config.data.audit_seed + source_index,
        )
        loss_weights = []
        getter = getattr(dataset, "get", None)
        for index in indices:
            example = getter(index, 0) if getter is not None else dataset[index]
            loss_weights.append(float(np.asarray(example["loss_masks"]).sum(dtype=np.float64)))
        fingerprint = getattr(dataset, "content_fingerprint", None)
        if fingerprint is None:
            fingerprint = getattr(dataset, "fingerprint", None)
        sources[name] = {
            "dataset_size": len(dataset),
            "sample_count": len(indices),
            "sample_indices_sha256": _canonical_sha256(indices),
            "mean_loss_weight": float(np.mean(loss_weights, dtype=np.float64)),
            "content_fingerprint": fingerprint,
        }
        log.info("Audited %s mean loss weight: %.12f", name, sources[name]["mean_loss_weight"])
    receipt = {
        "version": 1,
        "algorithm": _AUDIT_ALGORITHM,
        "source_contract_sha256": config.vision_midtraining.source_contract_sha256,
        "sequence_length": config.data.sequence_length,
        "tokenizer_fingerprint": TOKENIZER_FINGERPRINT,
        "audit_seed": config.data.audit_seed,
        "samples_per_source": config.data.audit_samples_per_source,
        "sources": sources,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    log.info("Wrote source-mean audit summary %s (sha256=%s)", output, _sha256_file(output))
    return output


def _validate_runtime_trainability(train_module) -> None:
    model = train_module.multimodal_model
    mismatches = []
    for name, parameter in model.named_parameters():
        expected = not name.startswith("vision.")
        if parameter.requires_grad != expected:
            mismatches.append((name, parameter.requires_grad, expected))
    if mismatches:
        raise RuntimeError(f"Frozen-vision runtime trainability mismatch: {mismatches[:8]}")
    for required in ("lm.embeddings.weight", "lm.lm_head.w_out.weight"):
        parameter = dict(model.named_parameters()).get(required)
        if parameter is None or not parameter.requires_grad:
            raise RuntimeError(f"Full-LM midtraining requires trainable parameter {required}")


def train(config: ExperimentConfig) -> None:
    """Run frozen-vision joint midtraining under torchrun."""
    os.environ.setdefault("OLMO_USE_OWN_SYMM_MEM", "1")
    os.environ.setdefault("OLMO_EP_MP_HIGH_PRIORITY_GROUP", "1")
    os.environ.setdefault("OLMO_OWN_SYMM_PREWARM", "1")
    seed_all(config.init_seed)
    tokenizer, token_ids = _load_tokenizer()
    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    _validate_runtime_trainability(train_module)
    datasets, names = _materialize_sources(tokenizer, token_ids, config)
    # Numpy preparation and cold visual-dataset opens can have large rank-local skew.
    import torch.distributed as dist

    dist.barrier()
    log.info("All ranks finished vision-midtraining dataset setup")
    weights_by_name = _sampling_weights(config)
    weights = [weights_by_name[name] for name in names]
    log.info("Vision-midtraining sampling probabilities: %s", dict(zip(names, weights)))

    dp_group = train_module.dp_process_group
    dp_world_size, dp_rank = get_world_size(dp_group), get_rank(dp_group)
    data_loader: DataLoaderBase = MixtureDataLoader(
        datasets,
        weights,
        _build_collator(config),
        work_dir=config.trainer.save_folder,
        global_batch_size=config.global_batch_size,
        seed=config.data_seed,
        pack=config.data.pack_sequences,
        pack_max_crops=config.data.pack_max_crops,
        pack_buffer_size=config.data.pack_buffer_size,
        prefetch_workers=config.data.prefetch_workers,
        dataset_names=names,
        allow_legacy_state_without_dataset_fingerprints=False,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
    )
    trainer = config.trainer.build(train_module, data_loader)
    config_dict = config.as_config_dict()
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict
    cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict
    trainer.fit()


def launch(config: ExperimentConfig) -> None:
    """Submit an audited real run or explicit synthetic smoke to Holmes."""
    config.launch.launch(follow=True)


def main() -> None:
    """Run the vision-midtraining CLI."""
    usage = f"""
Usage
=====

python {sys.argv[0]} [dry_run|audit|launch|train] RUN_NAME [--key=value ...]
""".strip()
    if len(sys.argv) < 3:
        print(usage)
        sys.exit(1)
    script, command, run_name, *raw_overrides = sys.argv
    if command not in {"dry_run", "audit", "launch", "train"}:
        print(usage)
        sys.exit(1)
    if command == "train":
        prepare_training_environment(timeout=timedelta(minutes=60))
    else:
        prepare_cli_environment()
    config = build_config(script, run_name, raw_overrides)
    _validate_contract(config, run_name, command)
    log.info("%s", config)
    if command == "audit":
        audit_source_means(config)
    elif command == "launch":
        launch(config)
    elif command == "train":
        train(config)
        teardown_training_environment()


if __name__ == "__main__":
    main()
