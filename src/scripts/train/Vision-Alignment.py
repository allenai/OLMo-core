"""Vision alignment continued pretraining for the bare s002 language model.

This is a new recipe, intentionally independent from :mod:`Molmo2-Stage1`.  It treats
multimodal adaptation as continued pretraining rather than instruction tuning and exposes
three explicit, model-only phase boundaries:

``bridge``
    Load the bare s002 checkpoint and pristine pinned SigLIP, then train only the connector
    and the six input-only image-token rows on document-formatted captions/transcripts.
``perception``
    Fork from a bridge checkpoint with a fresh optimizer and data cursor, unfreeze the vision
    encoder, and add audited perception sources while keeping the language model frozen.
``joint``
    Fork from a perception checkpoint, unfreeze the language model at a low learning rate,
    and begin exact native pretraining-data replay.

Each phase uses explicit continued-pretraining sources and fails closed when a pinned data
artifact is absent instead of silently substituting instruction-tuning data.

Run without arguments for usage.  No command in this file launches automatically.
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
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from olmo_core.config import Config, DType
from olmo_core.data.data_loader import DataLoaderBase
from olmo_core.data.multimodal import (
    MixtureDataLoader,
    MultimodalCollatorConfig,
    MultimodalDataLoader,
    NativeTextReplayDatasetConfig,
)
from olmo_core.data.multimodal.mixtures.vision_alignment import (
    VisionAlignmentMixtureConfig,
    expected_loss_mass,
)
from olmo_core.data.multimodal.native_text_replay import (
    NativeTextReplayVerificationReceipt,
)
from olmo_core.data.multimodal.paths import PIXMO_DATASETS
from olmo_core.data.multimodal.vision_alignment_joint_provenance import (
    JointVisualProjectionManifest,
    build_selected_joint_dataset,
    load_joint_visual_projection_manifest,
)
from olmo_core.data.multimodal.vision_alignment_joint_sources import (
    JOINT_VISUAL_SOURCE_NAMES,
)
from olmo_core.data.multimodal.vision_alignment_perception_provenance import (
    PERCEPTION_SOURCE_NAMES,
    PerceptionProvenanceManifest,
    build_selected_perception_dataset,
    load_perception_provenance_manifest,
)
from olmo_core.data.multimodal.vision_alignment_sources import (
    VISION_ALIGNMENT_FORMATTER_VERSION,
    VISION_ALIGNMENT_RECIPE_VERSION,
    VISION_ALIGNMENT_TOKENIZER_FINGERPRINT,
    VISION_ALIGNMENT_TOKENIZER_ID,
    VISION_ALIGNMENT_TOKENIZER_REVISION,
    VisionAlignmentSourceSpec,
    build_vision_alignment_dataset,
    build_vision_alignment_dataset_config,
    load_pinned_vision_alignment_tokenizer,
    runtime_dataset_fingerprint,
    serialized_example_sha256,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.eval import (
    Evaluator,
    MultimodalBlankImageEvaluator,
    MultimodalLMEvaluator,
)
from olmo_core.internal.common import build_launch_config, get_root_dir
from olmo_core.launch.beaker import BeakerEnvVar, BeakerLaunchConfig
from olmo_core.nn.vision import Molmo2TokenIds, MultimodalLMConfig
from olmo_core.optim import (
    CosWithWarmup,
    OLMoDDPOptimizerConfig,
    OptimGroupOverride,
    PerGroupScheduler,
)
from olmo_core.train import (
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
    EvaluatorCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
    WandBCallback,
)
from olmo_core.train.checkpoint import Checkpointer
from olmo_core.train.train_module import (
    MultimodalOLMoDDPTrainModuleConfig,
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
)
from olmo_core.utils import seed_all

log = logging.getLogger(__name__)

RECIPE_VERSION = VISION_ALIGNMENT_RECIPE_VERSION
FORMATTER_VERSION = VISION_ALIGNMENT_FORMATTER_VERSION

BASE_CHECKPOINT = "/weka/oe-training-default/robertb/s002-step125500"
BASE_CONFIG_SHA256 = "35ce23db053dd2204bc37783546f1b2f98eafb742488903773dd0ef3e5741146"
BASE_DATA_PATHS_SHA256 = "f1155957f4f249fc17e1c7067512e7d881ce6675c6b854d5ce089c649cec1c2d"
BASE_TRAINER_STATE_SHA256 = "451a536f6483b5347837251ab931c38c70434854c001d74456737592750170d3"
BASE_DATASET_FINGERPRINT = "37e1ae62dccee1f0cb5c3e416572e6e48218a6c644580fa5034f575880e08c11"
BASE_PARENT_MIX_SHA256 = "fcc6a82b9a5e868885decfbc30486967644c7ca482a7d687102f7ff597dbd7c9"
BASE_CHECKPOINT_MARKER_SHA256 = "77dfdeec42fe7990f4b3b9c4eeecd480edcf5066c110603b115920af38423d03"
BASE_CHECKPOINT_METADATA_SHA256 = "ce7a6c254c7b3aeca6d6a9b521328e09601211ddf98dc238e97b5b6c84c34633"
PARENT_TEXT_MIX = "OLMo-mix-0925"
MOLMO2_CONFIG_MODEL_ID = "allenai/Molmo2-4B"
MOLMO2_CONFIG_REVISION = "042abfa7a38879a376cec03d949eff0aefaa0600"
VISION_MODEL_ID = "google/siglip2-so400m-patch14-384"
VISION_REVISION = "e8e487298228002f3d8a82e0cd5c8ea9c567f57f"
VISION_FINGERPRINT = "9d9257ea672527b2e37cae7f61734afdf9280d3e77680f2c2d13d4da60aba6bf"
TOKENIZER_ID = VISION_ALIGNMENT_TOKENIZER_ID
TOKENIZER_REVISION = VISION_ALIGNMENT_TOKENIZER_REVISION
TOKENIZER_FINGERPRINT = VISION_ALIGNMENT_TOKENIZER_FINGERPRINT
HF_CACHE_DIR = "/weka/oe-training-default/rustin/hf-cache/hub"

EXPERIMENT_ROOT = "/weka/oe-training-default/rustin/experiments/vision-moe"
VISION_ALIGNMENT_ROOT = f"{EXPERIMENT_ROOT}/vision-alignment"
BEAKER_CLUSTER = "ai2/holmes"
BEAKER_WORKSPACE = "ai2/molmofication"
BEAKER_BUDGET = "ai2/oe-other"
WANDB_PROJECT: Optional[str] = "vision-alignment"
WANDB_ENTITY: Optional[str] = None

EP_DEGREE = 8
GLOBAL_BATCH_INSTANCES = 128
DATA_PREFETCH_WORKERS = 8
PACK_BUFFER_SIZE = 48
PACK_MAX_CROPS = 9
MAX_CROPS = 8
DATA_SEED = 95818
INIT_SEED = 6198
EVAL_SEED = 6198
_PERCEPTION_PROVENANCE_RUNTIME_CACHE: Dict[Tuple[str, str], PerceptionProvenanceManifest] = {}
_JOINT_PROJECTION_RUNTIME_CACHE: Dict[Tuple[str, str, str], JointVisualProjectionManifest] = {}
PERCEPTION_PROFILE_ROOT = "configs/vision_moe/vision_alignment/perception"
JOINT_PROFILE_ROOT = "configs/vision_moe/vision_alignment/joint"

_JOINT_AUDIT_FORMAT = "vision_alignment_joint_source_audit"
_JOINT_SEQUENCE_LENGTH = 8192
_JOINT_NATIVE_REPLAY_PARENT_OBJECTS = 950
_JOINT_NATIVE_REPLAY_SELECTION_ALGORITHM = "affine-grid-v1"
_JOINT_NATIVE_REPLAY_SELECTION_SEED = 6198

_RUN_NAME_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,127}")


class VisionAlignmentPhase(StrEnum):
    """The three optimizer/data phases of vision alignment."""

    bridge = "bridge"
    perception = "perception"
    joint = "joint"


class InitializationMode(StrEnum):
    """How a fresh vision-alignment phase obtains its model weights."""

    bare = "bare"
    checkpoint = "checkpoint"


class JointTrainabilityArm(StrEnum):
    """Vision-encoder trainability choices for the joint phase."""

    treatment = "treatment"
    frozen_vision_control = "frozen_vision_control"


@dataclass(frozen=True)
class _PhasePolicy:
    phase: VisionAlignmentPhase
    initialization_mode: InitializationMode
    expected_parent_phase: Optional[VisionAlignmentPhase]
    freeze_params: Tuple[str, ...]
    sequence_length: int
    rank_microbatch_instances: int
    max_steps: int
    connector_lr: float
    vision_lr: float
    lm_lr: float
    connector_warmup: int
    connector_t_max: Optional[int]
    vision_warmup: int
    lm_warmup: int


_PHASE_POLICIES: Dict[VisionAlignmentPhase, _PhasePolicy] = {
    VisionAlignmentPhase.bridge: _PhasePolicy(
        phase=VisionAlignmentPhase.bridge,
        initialization_mode=InitializationMode.bare,
        expected_parent_phase=None,
        freeze_params=(
            "vision.*",
            "lm.embedding_norm.*",
            "lm.blocks.*",
            "lm.lm_head.*",
        ),
        sequence_length=2560,
        rank_microbatch_instances=4,
        max_steps=1000,
        connector_lr=2e-4,
        vision_lr=0.0,
        lm_lr=0.0,
        connector_warmup=100,
        connector_t_max=250,
        vision_warmup=100,
        lm_warmup=100,
    ),
    VisionAlignmentPhase.perception: _PhasePolicy(
        phase=VisionAlignmentPhase.perception,
        initialization_mode=InitializationMode.checkpoint,
        expected_parent_phase=VisionAlignmentPhase.bridge,
        freeze_params=(
            "lm.embedding_norm.*",
            "lm.blocks.*",
            "lm.lm_head.*",
        ),
        sequence_length=2560,
        rank_microbatch_instances=4,
        max_steps=4000,
        connector_lr=5e-5,
        vision_lr=3e-6,
        lm_lr=0.0,
        connector_warmup=200,
        connector_t_max=None,
        vision_warmup=500,
        lm_warmup=500,
    ),
    VisionAlignmentPhase.joint: _PhasePolicy(
        phase=VisionAlignmentPhase.joint,
        initialization_mode=InitializationMode.checkpoint,
        expected_parent_phase=VisionAlignmentPhase.perception,
        freeze_params=("lm.lm_head.w_out.weight",),
        sequence_length=8192,
        rank_microbatch_instances=1,
        max_steps=16000,
        connector_lr=2e-5,
        vision_lr=2e-6,
        lm_lr=1e-6,
        connector_warmup=200,
        connector_t_max=None,
        vision_warmup=500,
        lm_warmup=500,
    ),
}


@dataclass
class ArtifactConfig(Config):
    """Pinned model, tokenizer, and parent-pretraining artifacts."""

    base_checkpoint: str = BASE_CHECKPOINT
    base_config_sha256: str = BASE_CONFIG_SHA256
    base_data_paths_sha256: str = BASE_DATA_PATHS_SHA256
    base_checkpoint_marker_sha256: str = BASE_CHECKPOINT_MARKER_SHA256
    base_checkpoint_metadata_sha256: str = BASE_CHECKPOINT_METADATA_SHA256
    parent_text_mix: str = PARENT_TEXT_MIX
    tokenizer_id: str = TOKENIZER_ID
    tokenizer_revision: str = TOKENIZER_REVISION
    tokenizer_fingerprint: str = TOKENIZER_FINGERPRINT
    molmo2_config_model_id: str = MOLMO2_CONFIG_MODEL_ID
    molmo2_config_revision: str = MOLMO2_CONFIG_REVISION
    vision_model_id: str = VISION_MODEL_ID
    vision_revision: str = VISION_REVISION
    vision_fingerprint: str = VISION_FINGERPRINT
    hf_cache_dir: str = HF_CACHE_DIR


@dataclass
class InitializationConfig(Config):
    """Initialization contract for a fresh phase or an exact same-phase resume."""

    mode: InitializationMode = InitializationMode.bare
    checkpoint: Optional[str] = None
    expected_parent_phase: Optional[VisionAlignmentPhase] = None
    parent_config_sha256: Optional[str] = None


@dataclass
class VisionAlignmentDataConfig(Config):
    """Document-formatted vision sources and calibrated sampling settings."""

    pixmo_cap_path: str = f"{PIXMO_DATASETS}/cap"
    sequence_length: int = 2560
    max_crops: int = MAX_CROPS
    message_format: str = "document"
    loss_token_weighting: str = "root_subsegments_root_tokens"
    caption_prompt: str = "Description:"
    transcript_prompt: str = "Transcript:"
    require_transcript: bool = True
    mixture: VisionAlignmentMixtureConfig = field(default_factory=VisionAlignmentMixtureConfig)
    source_audit_path: Optional[str] = None
    source_audit_fingerprint: Optional[str] = None
    perception_provenance_path: Optional[str] = None
    perception_provenance_sha256: Optional[str] = None
    joint_visual_projection_path: Optional[str] = None
    joint_visual_projection_sha256: Optional[str] = None
    allow_unpinned_synthetic_smoke: bool = False
    native_text_replay: Optional[NativeTextReplayDatasetConfig] = None
    native_text_replay_fingerprint: Optional[str] = None
    pack_sequences: bool = True
    pack_buffer_size: int = PACK_BUFFER_SIZE
    pack_max_crops: int = PACK_MAX_CROPS
    prefetch_workers: int = DATA_PREFETCH_WORKERS


@dataclass
class VisionAlignmentEvalConfig(Config):
    """Intrinsic held-out continued-pretraining evaluation settings."""

    interval: Optional[int] = 500
    examples_per_source: int = 512
    rank_batch_instances: int = 4
    seed: int = EVAL_SEED
    eval_on_startup: bool = True
    eval_on_finish: bool = True
    validation_manifest_path: Optional[str] = None
    validation_manifest_sha256: Optional[str] = None
    native_text_holdout: Optional[NativeTextReplayDatasetConfig] = None
    native_text_holdout_fingerprint: Optional[str] = None


def _build_evaluation_config(policy: _PhasePolicy) -> VisionAlignmentEvalConfig:
    """Build intrinsic evaluation at the phase's proven per-rank capacity."""
    return VisionAlignmentEvalConfig(rank_batch_instances=policy.rank_microbatch_instances)


@dataclass
class VisionAlignmentMetadataConfig(Config):
    """Identity and compatibility metadata persisted in every checkpoint config."""

    recipe_version: int = RECIPE_VERSION
    formatter_version: str = FORMATTER_VERSION
    phase: VisionAlignmentPhase = VisionAlignmentPhase.bridge
    lineage_id: str = ""
    parent_checkpoint: Optional[str] = None
    parent_config_sha256: Optional[str] = None
    data_contract_sha256: str = ""
    trainable_contract_sha256: str = ""


@dataclass
class ExperimentConfig(Config):
    """Complete configuration for one vision-alignment phase."""

    launch: BeakerLaunchConfig
    model: MultimodalLMConfig
    collator: MultimodalCollatorConfig
    train_module: MultimodalOLMoDDPTrainModuleConfig
    trainer: TrainerConfig
    phase: VisionAlignmentPhase
    artifacts: ArtifactConfig
    initialization: InitializationConfig
    data: VisionAlignmentDataConfig
    evaluation: VisionAlignmentEvalConfig
    vision_alignment: VisionAlignmentMetadataConfig
    global_batch_size: int
    joint_trainability_arm: JointTrainabilityArm = JointTrainabilityArm.treatment
    data_seed: int = DATA_SEED
    init_seed: int = INIT_SEED
    checkpoint_load_threads: int = 8
    router_lb_loss_weight: Optional[float] = 0.015
    required_run_name: str = ""


def _joint_vision_is_frozen(config: Any) -> bool:
    """Return whether this config selects the joint frozen-vision control."""
    return (
        config.phase is VisionAlignmentPhase.joint
        and getattr(
            config,
            "joint_trainability_arm",
            JointTrainabilityArm.treatment,
        )
        is JointTrainabilityArm.frozen_vision_control
    )


def _apply_joint_trainability_arm(config: ExperimentConfig) -> None:
    """Derive frozen-vision train-module fields from the selected joint arm."""
    if not _joint_vision_is_frozen(config):
        return
    config.train_module.freeze_params = [
        "vision.*",
        *(config.train_module.freeze_params or []),
    ]
    vision_groups = [
        group
        for group in (config.train_module.optim.group_overrides or [])
        if group.params == ["*vision.*"]
    ]
    if len(vision_groups) != 1:
        raise ValueError("Joint frozen-vision control requires one exact vision optimizer group")
    vision_groups[0].opts["lr"] = 0.0


def _validate_joint_trainability_arm(config: ExperimentConfig) -> None:
    """Reject a joint-only trainability arm on another data phase."""
    if (
        config.phase is not VisionAlignmentPhase.joint
        and config.joint_trainability_arm is not JointTrainabilityArm.treatment
    ):
        raise ValueError("Joint trainability arms are defined only for the joint phase")


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


def _strict_json_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object repeats key {key!r}")
        result[key] = value
    return result


def _extract_phase(overrides: Sequence[str]) -> VisionAlignmentPhase:
    selectors = [value.split("=", 1)[1] for value in overrides if value.startswith("--phase=")]
    if len(selectors) != 1:
        raise ValueError(
            "Vision alignment requires exactly one --phase=<bridge|perception|joint> selector"
        )
    try:
        return VisionAlignmentPhase(selectors[0])
    except ValueError as error:
        raise ValueError(
            f"Unknown vision-alignment phase {selectors[0]!r}; "
            f"expected one of {[phase.value for phase in VisionAlignmentPhase]}"
        ) from error


def _validate_run_name(run_name: str) -> None:
    if _RUN_NAME_RE.fullmatch(run_name) is None:
        raise ValueError(
            "Vision-alignment run names must match [a-z0-9][a-z0-9_-]{0,127}; " f"got {run_name!r}"
        )


def _image_token_ids(token_ids: Molmo2TokenIds) -> List[int]:
    return [
        token_ids.im_start_id,
        token_ids.im_end_id,
        token_ids.im_patch_id,
        token_ids.im_col_id,
        token_ids.low_res_im_start_id,
        token_ids.image_placeholder_id,
    ]


def _load_tokenizer(artifacts: ArtifactConfig):
    return load_pinned_vision_alignment_tokenizer(
        identifier=artifacts.tokenizer_id,
        revision=artifacts.tokenizer_revision,
        expected_fingerprint=artifacts.tokenizer_fingerprint,
        cache_dir=artifacts.hf_cache_dir,
    )


def _build_model_config(token_ids: Molmo2TokenIds, artifacts: ArtifactConfig) -> MultimodalLMConfig:
    """Compose the pinned bare s002 LM with the Molmo2 connector/SigLIP architecture."""
    from olmo_core.nn.attention import AttentionConfig
    from olmo_core.nn.attention.backend import AttentionBackendName
    from olmo_core.nn.ddp.block import OLMoDDPTransformerBlockConfig
    from olmo_core.nn.moe.v2.ep_config import ExpertParallelPath
    from olmo_core.nn.transformer import OLMoDDPModelConfig
    from olmo_core.nn.vision import (
        load_molmo2_hf_vision_config,
        multimodal_config_from_molmo2_vision,
    )

    config_path = Path(artifacts.base_checkpoint) / "config.json"
    if _sha256_file(config_path) != artifacts.base_config_sha256:
        raise ValueError(f"Bare s002 config fingerprint mismatch for {config_path}")
    data_paths_path = Path(artifacts.base_checkpoint) / "data_paths.txt"
    if _sha256_file(data_paths_path) != artifacts.base_data_paths_sha256:
        raise ValueError(f"Bare s002 data-path fingerprint mismatch for {data_paths_path}")
    checkpoint_marker = Path(artifacts.base_checkpoint) / ".metadata.json"
    if _sha256_file(checkpoint_marker) != artifacts.base_checkpoint_marker_sha256:
        raise ValueError(f"Bare s002 checkpoint marker mismatch for {checkpoint_marker}")
    checkpoint_metadata = Path(artifacts.base_checkpoint) / "model_and_optim" / ".metadata"
    if _sha256_file(checkpoint_metadata) != artifacts.base_checkpoint_metadata_sha256:
        raise ValueError(f"Bare s002 DCP metadata mismatch for {checkpoint_metadata}")
    with config_path.open() as file_handle:
        lm_config = OLMoDDPModelConfig.from_dict(json.load(file_handle)["model"])

    for block_config in [lm_config.block, *(lm_config.block_overrides or {}).values()]:
        if not isinstance(block_config, OLMoDDPTransformerBlockConfig):
            raise TypeError("The pinned s002 LM must use OLMoDDP transformer blocks")
        if isinstance(block_config.sequence_mixer, AttentionConfig):
            block_config.sequence_mixer.backend = AttentionBackendName.flex
        if block_config.ep is not None:
            block_config.ep.path = ExpertParallelPath.rowwise_nvshmem
    lm_config.recompute_each_block = True
    lm_config.recompute_all_blocks_by_chunk = False
    lm_config.two_batch_overlap = False

    hf_config = load_molmo2_hf_vision_config(
        artifacts.molmo2_config_model_id,
        revision=artifacts.molmo2_config_revision,
        cache_dir=artifacts.hf_cache_dir,
    )
    return multimodal_config_from_molmo2_vision(
        hf_config, lm_config, image_patch_token_id=token_ids.im_patch_id
    )


def _build_train_module_config(
    policy: _PhasePolicy, image_token_ids: List[int]
) -> MultimodalOLMoDDPTrainModuleConfig:
    # OLMoDDPOptimizer requires a positive default LR even when every trainable parameter in a
    # frozen-LM phase is assigned to an explicit override group. The runtime trainability check
    # below proves that bridge/perception have no live fallback parameters.
    default_lr = policy.lm_lr if policy.lm_lr > 0 else policy.connector_lr
    group_overrides = [
        OptimGroupOverride(
            params=["*lm.embeddings.weight"],
            opts={
                "lr": policy.connector_lr,
                "weight_decay": 0.0,
                "scheduler_name": "connector",
            },
        ),
        OptimGroupOverride(
            params=["*connector.*"],
            opts={
                "lr": policy.connector_lr,
                "weight_decay": 0.0,
                "scheduler_name": "connector",
            },
        ),
        OptimGroupOverride(
            params=["*vision.*"],
            opts={
                "lr": policy.vision_lr,
                "weight_decay": 0.0,
                "scheduler_name": "vision",
            },
        ),
    ]
    return MultimodalOLMoDDPTrainModuleConfig(
        rank_microbatch_size=policy.rank_microbatch_instances * policy.sequence_length,
        max_sequence_length=policy.sequence_length,
        optim=OLMoDDPOptimizerConfig(
            lr=default_lr,
            betas=(0.9, 0.95),
            eps=1e-6,
            weight_decay=0.0,
            group_overrides=group_overrides,
            compile=False,
            foreach_chunk_size=50_000_000,
            sigma_factor=12,
            max_grad_norm=1.0,
            clip_grad_norm_by_scheduler_group=True,
            check_nan_inf_grad=True,
            use_distributed=True,
        ),
        freeze_params=list(policy.freeze_params),
        train_embedding_rows=image_token_ids,
        vision_activation_checkpointing=policy.phase is not VisionAlignmentPhase.bridge,
        connector_activation_checkpointing=True,
        response_logits_only=True,
        diagnostics_interval=100,
        z_loss_multiplier=1e-4,
        max_grad_norm=1.0,
        compile_model=True,
        scheduler=PerGroupScheduler(
            schedulers={
                "connector": CosWithWarmup(
                    warmup=policy.connector_warmup,
                    alpha_f=0.1,
                    t_max=policy.connector_t_max,
                ),
                "vision": CosWithWarmup(warmup=policy.vision_warmup, alpha_f=0.1, t_max=None),
            },
            default=CosWithWarmup(warmup=policy.lm_warmup, alpha_f=0.1, t_max=None),
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


def _configure_router_load_balancing(lm_config, weight: Optional[float]) -> None:
    if weight is not None and weight < 0:
        raise ValueError("router_lb_loss_weight must be non-negative or None")
    configured = 0
    for block_config in [lm_config.block, *(lm_config.block_overrides or {}).values()]:
        router = getattr(block_config, "routed_experts_router", None)
        if router is not None:
            router.lb_loss_weight = weight
            configured += 1
    if configured == 0:
        raise ValueError("The pinned s002 LM has no routed-expert router configs")


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


def _build_console_logger() -> ConsoleLoggerCallback:
    callback = ConsoleLoggerCallback()
    callback.metrics.extend(
        [
            "data/*",
            "multimodal/*",
            "optim/* grad norm",
            "optim/* clip coefficient",
        ]
    )
    return callback


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects ambiguous duplicate mapping keys."""

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> Dict[Any, Any]:
        """Construct one mapping after rejecting duplicate or unhashable keys."""
        if not isinstance(node, MappingNode):
            raise ConstructorError(
                None,
                None,
                f"expected a mapping node, got {node.id}",
                node.start_mark,
            )
        self.flatten_mapping(node)
        seen: set[Any] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in seen
            except TypeError as error:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found an unhashable mapping key",
                    key_node.start_mark,
                ) from error
            if duplicate:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _load_profile_yaml(raw: bytes, *, path: Path) -> Mapping[str, Any]:
    """Parse one profile with strict duplicate-key and root-schema handling."""
    try:
        profile = yaml.load(raw, Loader=_UniqueKeySafeLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"Could not parse vision-alignment profile {path}: {error}") from error
    if not isinstance(profile, Mapping):
        raise ValueError(f"Invalid vision-alignment profile {path}: expected a mapping")
    return profile


def _load_profile(overrides: List[str]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    prefix = "--profile="
    paths = [value[len(prefix) :] for value in overrides if value.startswith(prefix)]
    if len(paths) > 1:
        raise ValueError("At most one --profile may be supplied")
    if not paths:
        return None, overrides
    repository_root = Path(__file__).resolve().parents[3]
    profile_path = Path(paths[0]).expanduser().resolve()
    try:
        raw_profile = profile_path.read_bytes()
    except OSError as error:
        raise ValueError(
            f"Could not read vision-alignment profile {profile_path}: {error}"
        ) from error
    profile = dict(_load_profile_yaml(raw_profile, path=profile_path))
    if (
        isinstance(profile.get("version"), bool)
        or not isinstance(profile.get("version"), int)
        or profile.get("version") != 1
    ):
        raise ValueError(f"Invalid vision-alignment profile {profile_path}: expected version: 1")
    unknown = set(profile) - {"version", "name", "description", "phase", "launch", "overrides"}
    if unknown:
        raise ValueError(f"Unknown profile fields in {profile_path}: {sorted(unknown)}")
    phase = profile.get("phase")
    if not isinstance(phase, str):
        raise ValueError(f"{profile_path}: phase must be a string")
    profile_overrides = profile.get("overrides", [])
    if not isinstance(profile_overrides, list) or not all(
        isinstance(value, str) and value.startswith("--") for value in profile_overrides
    ):
        raise ValueError(f"{profile_path}: overrides must be '--key=value' strings")
    override_destinations: list[str] = []
    for value in profile_overrides:
        destination, separator, _ = value[2:].partition("=")
        if not separator or not destination:
            raise ValueError(f"{profile_path}: every profile override must be '--key=value'")
        override_destinations.append(destination)
    if len(set(override_destinations)) != len(override_destinations):
        raise ValueError(f"{profile_path}: profile overrides repeat a destination")
    cli = [value for value in overrides if not value.startswith(prefix)]
    if any(value.startswith("--phase=") for value in [*profile_overrides, *cli]):
        raise ValueError("Set phase in the profile or on the CLI, not both")
    production_phase = (
        VisionAlignmentPhase(phase)
        if phase in (VisionAlignmentPhase.perception.value, VisionAlignmentPhase.joint.value)
        else None
    )
    if production_phase is not None:
        profile_root = (
            PERCEPTION_PROFILE_ROOT
            if production_phase is VisionAlignmentPhase.perception
            else JOINT_PROFILE_ROOT
        )
        approved_root = (repository_root / profile_root).resolve()
        if (
            profile_path.parent != approved_root
            or profile_path.suffix != ".yaml"
            or profile_path.name.endswith(".yaml.template")
        ):
            raise ValueError(
                f"Production {production_phase.value} requires a checked-in .yaml profile directly "
                "under "
                f"{approved_root}"
            )
        if cli:
            raise ValueError(
                f"Production {production_phase.value} profiles own the complete configuration; "
                "additional "
                "CLI overrides are forbidden"
            )
    return profile, [f"--phase={phase}", *profile_overrides, *cli]


def _apply_profile_launch(
    config: ExperimentConfig,
    profile: Optional[Dict[str, Any]],
    *,
    run_name: Optional[str] = None,
) -> ExperimentConfig:
    if profile is None:
        return config
    profile_name = profile.get("name")
    if not isinstance(profile_name, str) or not profile_name:
        raise ValueError("Vision-alignment profiles must declare a non-empty name")
    if run_name is not None and profile_name != run_name:
        raise ValueError(
            f"Profile name {profile_name!r} must match positional run name {run_name!r}"
        )
    launch = profile.get("launch", {})
    if not isinstance(launch, dict):
        raise ValueError("Profile launch must be a mapping")
    unknown = set(launch) - {
        "num_nodes",
        "num_gpus",
        "workspace",
        "cluster",
        "budget",
        "priority",
        "min_runtime",
    }
    if unknown:
        raise ValueError(f"Unknown profile launch fields: {sorted(unknown)}")
    cluster = str(launch.get("cluster", BEAKER_CLUSTER))
    if cluster != BEAKER_CLUSTER:
        raise ValueError(f"Vision alignment may launch only on {BEAKER_CLUSTER!r}")
    config.launch.clusters = [cluster]
    config.launch.hostnames = None
    config.launch.num_nodes = int(launch.get("num_nodes", config.launch.num_nodes))
    config.launch.num_gpus = int(launch.get("num_gpus", config.launch.num_gpus))
    config.launch.workspace = launch.get("workspace", config.launch.workspace)
    config.launch.budget = launch.get("budget", config.launch.budget)
    config.launch.priority = str(launch.get("priority", config.launch.priority))
    config.launch.min_runtime = launch.get("min_runtime", config.launch.min_runtime)
    config.launch.description = profile.get("description")
    return config


def _checkpoint_state_dir(checkpoint: str) -> str:
    nested = Path(checkpoint) / "model_and_optim"
    return str(nested if nested.is_dir() else Path(checkpoint))


def _latest_output_checkpoint(config: ExperimentConfig) -> Optional[str]:
    try:
        return Checkpointer.latest_checkpoint(config.trainer.save_folder)
    except FileNotFoundError:
        return None


def _checkpoint_config(checkpoint: str) -> Tuple[Dict[str, Any], str]:
    config_path = Path(checkpoint) / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Vision-alignment checkpoint lacks required config: {config_path}")
    raw = config_path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"Checkpoint config must be a JSON object: {config_path}")
    return data, hashlib.sha256(raw).hexdigest()


def _validate_permanent_checkpoint(checkpoint: str) -> None:
    """Require a completed, non-ephemeral checkpoint at a numbered training step."""
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if re.fullmatch(r"step[1-9][0-9]*", checkpoint_path.name) is None:
        raise ValueError(f"Phase parent must be a positive numbered checkpoint: {checkpoint}")
    marker_path = checkpoint_path / ".metadata.json"
    try:
        marker = json.loads(marker_path.read_bytes(), object_pairs_hook=_strict_json_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Invalid parent checkpoint marker {marker_path}: {error}") from error
    if not isinstance(marker, Mapping) or marker.get("ephemeral") is not False:
        raise ValueError("A phase transition requires a permanent parent checkpoint")


def _parent_is_inside_output(parent: str, output: str) -> bool:
    parent_path = Path(parent).resolve()
    output_path = Path(output).resolve()
    return (
        parent_path == output_path
        or output_path in parent_path.parents
        or parent_path in output_path.parents
    )


class _AuditedDataset:
    """Bind a live map-style dataset to a pinned source-audit identity.

    The wrapper keeps the original content fingerprint so data-loader state from the
    completed phase recipes remains compatible. Runtime validation is intentionally limited
    to fields that affect training examples; producer implementation metadata is treated as
    opaque, checksum-bound artifact metadata.
    """

    content_fingerprint_version = "vision-alignment-source-audit-v2"

    def __init__(self, dataset: Any, source_name: str, audit: Mapping[str, Any]):
        self._dataset = dataset
        self.source_name = source_name
        inputs = audit.get("inputs")
        if not isinstance(inputs, Mapping) or not isinstance(inputs.get(source_name), Mapping):
            raise ValueError(f"Source audit does not describe {source_name!r}")
        source = cast(Mapping[str, Any], inputs[source_name])

        runtime_fingerprint = _runtime_dataset_fingerprint(dataset)
        if runtime_fingerprint != source.get("dataset_fingerprint"):
            raise ValueError(
                f"Live dataset fingerprint for {source_name!r} differs from its source audit"
            )
        if len(dataset) != source.get("dataset_size"):
            raise ValueError(
                f"Live dataset length for {source_name!r} differs from its source audit"
            )

        probe_indices = source.get("probe_indices")
        row_hashes = source.get("serialized_row_hashes")
        if (
            not isinstance(probe_indices, list)
            or any(
                type(index) is not int or index < 0 or index >= len(dataset)
                for index in probe_indices
            )
            or len(set(probe_indices)) != len(probe_indices)
            or source.get("probe_indices_sha256") != _canonical_sha256(probe_indices)
            or not isinstance(row_hashes, list)
            or any(
                not isinstance(row_hash, str) or re.fullmatch(r"[0-9a-f]{64}", row_hash) is None
                for row_hash in row_hashes
            )
            or source.get("serialized_row_hashes_sha256") != _canonical_sha256(row_hashes)
        ):
            raise ValueError(f"Source audit has an invalid runtime probe for {source_name!r}")

        raw_epochs = source.get("probe_epochs")
        epochs: Tuple[int, ...]
        if raw_epochs is None:
            epochs = (0,)
        elif type(raw_epochs) is int and raw_epochs > 0:
            epochs = tuple(range(raw_epochs))
        elif (
            isinstance(raw_epochs, list)
            and raw_epochs
            and all(type(epoch) is int and epoch >= 0 for epoch in raw_epochs)
            and len(set(raw_epochs)) == len(raw_epochs)
        ):
            epochs = tuple(raw_epochs)
        else:
            raise ValueError(f"Source audit has invalid probe epochs for {source_name!r}")

        probe_pairs = tuple(
            (index, epoch) for epoch in epochs for index in cast(List[int], probe_indices)
        )
        if len(probe_pairs) != len(row_hashes):
            raise ValueError(f"Source audit probe panel is inconsistent for {source_name!r}")
        get = getattr(dataset, "get", None)
        for (index, epoch), expected_hash in zip(probe_pairs, row_hashes):
            example = get(index, epoch) if callable(get) else dataset[index]
            if serialized_example_sha256(example) != expected_hash:
                raise ValueError(
                    f"Live serialized row differs for {source_name!r} at index {index}, "
                    f"epoch {epoch}"
                )

        image_digest = source.get("probe_image_content_sha256")
        if image_digest is not None and source_name != "native_text_replay":
            validate_image_content = getattr(dataset, "validate_image_content", None)
            if not callable(validate_image_content):
                raise ValueError(f"Live dataset {source_name!r} lacks image-content validation")
            if validate_image_content(probe_indices) != image_digest:
                raise ValueError(
                    f"Live probe image bytes differ from the source audit for {source_name!r}"
                )

        fingerprint_payload = {
            "audit_fingerprint": audit["fingerprint"],
            "source_registry_sha256": audit["source_registry_sha256"],
            "input_content_sha256": audit["input_content_sha256"],
            "source": source_name,
            "source_sha256": source["sha256"],
            "probe_indices_sha256": source["probe_indices_sha256"],
            "serialized_row_hashes_sha256": source["serialized_row_hashes_sha256"],
            "runtime_dataset_fingerprint": runtime_fingerprint,
            "runtime_dataset_length": len(dataset),
        }
        if audit.get("format") == _JOINT_AUDIT_FORMAT:
            fingerprint_payload["exporter_implementation"] = audit["exporter_implementation"]
        else:
            fingerprint_payload["exporter_sha256"] = audit["exporter_sha256"]
        if image_digest is not None:
            fingerprint_payload["probe_image_content_sha256"] = image_digest
        if raw_epochs is not None:
            fingerprint_payload["probe_epochs"] = raw_epochs
        self.content_fingerprint = _canonical_sha256(fingerprint_payload)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.get(index, 0)

    def get(self, index: int, epoch: int = 0) -> Dict[str, Any]:
        """Return one audited dataset row for the requested index and epoch."""
        get = getattr(self._dataset, "get", None)
        return get(index, epoch) if callable(get) else self._dataset[index]


def _perception_provenance(config: ExperimentConfig) -> PerceptionProvenanceManifest:
    """Load the externally SHA-pinned perception train/validation provenance."""
    data = config.data
    if data.perception_provenance_path is None or data.perception_provenance_sha256 is None:
        raise ValueError("Perception requires a pinned image-provenance artifact")
    cache_key = (
        str(Path(data.perception_provenance_path).expanduser().resolve()),
        data.perception_provenance_sha256,
    )
    manifest = _PERCEPTION_PROVENANCE_RUNTIME_CACHE.get(cache_key)
    if manifest is None:
        import torch.distributed as dist

        distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if distributed else 0
        error_message: Optional[str] = None
        if rank == 0:
            try:
                manifest = load_perception_provenance_manifest(
                    data.perception_provenance_path,
                    expected_sha256=data.perception_provenance_sha256,
                )
                manifest.validate_image_path_signatures()
                # Close the long path-restat window by rechecking the FineVision receipt and
                # output signatures after it. The validator's exact-stat cache makes the
                # unchanged case metadata-only, but any drift forces a full byte revalidation.
                load_perception_provenance_manifest(
                    data.perception_provenance_path,
                    expected_sha256=data.perception_provenance_sha256,
                    load_image_path_signatures=False,
                )
            except Exception as error:
                error_message = f"{type(error).__name__}: {error}"
        if distributed:
            result = [error_message]
            dist.broadcast_object_list(result, src=0)
            error_message = cast(Optional[str], result[0])
        if error_message is not None:
            raise ValueError(
                f"Perception provenance runtime snapshot validation failed: {error_message}"
            )
        if rank != 0:
            try:
                manifest = load_perception_provenance_manifest(
                    data.perception_provenance_path,
                    expected_sha256=data.perception_provenance_sha256,
                    verify_finevision_materialization=False,
                    load_image_path_signatures=False,
                )
            except Exception as error:
                error_message = f"rank {rank}: {type(error).__name__}: {error}"
        if distributed:
            rank_errors: list[Optional[str]] = [None] * dist.get_world_size()
            dist.all_gather_object(rank_errors, error_message)
            failures = [value for value in rank_errors if value is not None]
            if failures:
                raise ValueError(
                    "Perception provenance rank-local validation failed: " + "; ".join(failures)
                )
        assert manifest is not None
        _PERCEPTION_PROVENANCE_RUNTIME_CACHE[cache_key] = manifest
    if manifest.source_spec.pixmo_cap_path != str(Path(data.pixmo_cap_path).resolve()):
        raise ValueError("Perception provenance PixMoCap path differs from the training config")
    if (
        manifest.source_spec.sequence_length != data.sequence_length
        or manifest.source_spec.max_crops != data.max_crops
        or manifest.source_spec.message_format != data.message_format
        or manifest.source_spec.loss_token_weighting != data.loss_token_weighting
        or manifest.source_spec.caption_prompt != data.caption_prompt
        or manifest.source_spec.transcript_prompt != data.transcript_prompt
        or manifest.source_spec.require_transcript != data.require_transcript
    ):
        raise ValueError("Perception provenance serialization differs from the training config")
    return manifest


def _joint_visual_projection(
    config: ExperimentConfig,
    token_ids: Optional[Molmo2TokenIds] = None,
) -> JointVisualProjectionManifest:
    """Load the exact 8,192-token visual projection inherited from perception."""
    data = config.data
    if data.joint_visual_projection_path is None or data.joint_visual_projection_sha256 is None:
        raise ValueError("Joint alignment requires a pinned visual-projection artifact")
    if token_ids is None:
        _, token_ids = _load_tokenizer(config.artifacts)
    if type(token_ids) is not Molmo2TokenIds:
        raise ValueError("Joint projection requires exact tokenizer-adapted token IDs")
    token_ids_sha256 = _canonical_sha256(asdict(token_ids))
    cache_key = (
        str(Path(data.joint_visual_projection_path).expanduser().resolve()),
        data.joint_visual_projection_sha256,
        token_ids_sha256,
    )
    manifest = _JOINT_PROJECTION_RUNTIME_CACHE.get(cache_key)
    if manifest is None:
        import torch.distributed as dist

        distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if distributed else 0
        error_message: Optional[str] = None
        if rank == 0:
            try:
                manifest = load_joint_visual_projection_manifest(
                    data.joint_visual_projection_path,
                    expected_token_ids=token_ids,
                    expected_sha256=data.joint_visual_projection_sha256,
                )
                manifest.parent_provenance.validate_image_path_signatures()
                # Close the long path-restat window with a second artifact check.
                load_joint_visual_projection_manifest(
                    data.joint_visual_projection_path,
                    expected_token_ids=token_ids,
                    expected_sha256=data.joint_visual_projection_sha256,
                    verify_finevision_materialization=False,
                    load_image_path_signatures=False,
                )
            except Exception as error:
                error_message = f"{type(error).__name__}: {error}"
        if distributed:
            result = [error_message]
            dist.broadcast_object_list(result, src=0)
            error_message = cast(Optional[str], result[0])
        if error_message is not None:
            raise ValueError(
                f"Joint visual-projection runtime snapshot validation failed: {error_message}"
            )
        if rank != 0:
            try:
                manifest = load_joint_visual_projection_manifest(
                    data.joint_visual_projection_path,
                    expected_token_ids=token_ids,
                    expected_sha256=data.joint_visual_projection_sha256,
                    verify_finevision_materialization=False,
                    load_image_path_signatures=False,
                )
            except Exception as error:
                error_message = f"rank {rank}: {type(error).__name__}: {error}"
        if distributed:
            rank_errors: list[Optional[str]] = [None] * dist.get_world_size()
            dist.all_gather_object(rank_errors, error_message)
            failures = [value for value in rank_errors if value is not None]
            if failures:
                raise ValueError(
                    "Joint visual-projection rank-local validation failed: " + "; ".join(failures)
                )
        assert manifest is not None
        _JOINT_PROJECTION_RUNTIME_CACHE[cache_key] = manifest
    parent_spec = manifest.source_spec.perception_spec
    if parent_spec.pixmo_cap_path != str(Path(data.pixmo_cap_path).resolve()):
        raise ValueError("Joint visual projection PixMoCap path differs from the training config")
    if (
        manifest.source_spec.sequence_length != data.sequence_length
        or parent_spec.max_crops != data.max_crops
        or parent_spec.message_format != data.message_format
        or parent_spec.loss_token_weighting != data.loss_token_weighting
        or parent_spec.caption_prompt != data.caption_prompt
        or parent_spec.transcript_prompt != data.transcript_prompt
        or parent_spec.require_transcript != data.require_transcript
    ):
        raise ValueError("Joint visual projection serialization differs from the training config")
    return manifest


def _source_spec(config: ExperimentConfig) -> VisionAlignmentSourceSpec:
    """Return the exact source specification shared by all phase source builders."""
    data = config.data
    artifacts = getattr(config, "artifacts", None)
    phase = (
        config.phase.value if isinstance(config.phase, VisionAlignmentPhase) else str(config.phase)
    )
    return VisionAlignmentSourceSpec(
        phase=phase,
        pixmo_cap_path=data.pixmo_cap_path,
        sequence_length=data.sequence_length,
        max_crops=data.max_crops,
        message_format=data.message_format,
        loss_token_weighting=data.loss_token_weighting,
        caption_prompt=data.caption_prompt,
        transcript_prompt=data.transcript_prompt,
        require_transcript=data.require_transcript,
        tokenizer_id=getattr(artifacts, "tokenizer_id", TOKENIZER_ID),
        tokenizer_revision=getattr(artifacts, "tokenizer_revision", TOKENIZER_REVISION),
        tokenizer_fingerprint=getattr(artifacts, "tokenizer_fingerprint", TOKENIZER_FINGERPRINT),
        native_text_replay_fingerprint=data.native_text_replay_fingerprint,
    )


def _runtime_dataset_fingerprint(dataset: Any) -> Optional[str]:
    """Resolve a stable identity from a live source or its selected Arrow split."""
    return runtime_dataset_fingerprint(dataset)


def _preprocessing_config_sha256(config: ExperimentConfig) -> str:
    """Hash every recipe field that changes serialized source examples."""
    if config.phase is VisionAlignmentPhase.perception:
        return _perception_provenance(config).source_spec_sha256
    if config.phase is VisionAlignmentPhase.joint:
        return _canonical_sha256(
            {
                "visual": _joint_visual_projection(config).source_spec.as_canonical_dict(),
                "native_text_replay_fingerprint": config.data.native_text_replay_fingerprint,
            }
        )
    return _source_spec(config).preprocessing_sha256


def _validated_source_audit(config: ExperimentConfig) -> Optional[Mapping[str, Any]]:
    """Load the checksum-pinned source audit consumed by the data loader.

    The audit producer owns exhaustive schema and metric validation. The recipe verifies the
    immutable artifact identity plus the fields that bind it to the live training config and
    serialized runtime probes.
    """
    data = config.data
    if data.allow_unpinned_synthetic_smoke:
        if (
            config.phase is not VisionAlignmentPhase.bridge
            or data.pixmo_cap_path != "synthetic"
            or data.source_audit_path is not None
            or data.source_audit_fingerprint is not None
            or set(data.mixture.resolved_targets()) != {"pixmo_caption", "pixmo_transcript"}
            or config.trainer.max_duration.value != 1
            or config.evaluation.interval is not None
        ):
            raise ValueError(
                "The unpinned audit bypass is restricted to the one-step synthetic bridge smoke"
            )
        return None
    if data.source_audit_path is None or data.source_audit_fingerprint is None:
        raise ValueError("Production vision alignment requires a pinned source audit")

    path = Path(data.source_audit_path).expanduser().resolve()
    try:
        audit = json.loads(path.read_bytes(), object_pairs_hook=_strict_json_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Invalid vision-alignment source audit {path}: {error}") from error
    if not isinstance(audit, Mapping):
        raise ValueError(f"Vision-alignment source audit must be an object: {path}")

    unsigned = dict(audit)
    recorded_fingerprint = unsigned.pop("fingerprint", None)
    computed_fingerprint = _canonical_sha256(unsigned)
    if (
        recorded_fingerprint != computed_fingerprint
        or data.source_audit_fingerprint != computed_fingerprint
    ):
        raise ValueError("Vision-alignment source audit fingerprint differs")

    expected_identity = {
        VisionAlignmentPhase.bridge: ("vision_alignment_source_audit", 2),
        VisionAlignmentPhase.perception: ("vision_alignment_perception_source_audit", 2),
        VisionAlignmentPhase.joint: (_JOINT_AUDIT_FORMAT, 1),
    }[config.phase]
    if (
        (audit.get("format"), audit.get("version")) != expected_identity
        or audit.get("status") != "ok"
        or audit.get("phase") != config.phase.value
        or audit.get("recipe_version") != RECIPE_VERSION
        or audit.get("formatter_version") != FORMATTER_VERSION
        or audit.get("failures") != []
    ):
        raise ValueError("Vision-alignment source audit identity or status differs")

    preprocessing_field = (
        "preprocessing_sha256"
        if config.phase is VisionAlignmentPhase.joint
        else "preprocessing_config_sha256"
    )
    if audit.get(preprocessing_field) != _preprocessing_config_sha256(config):
        raise ValueError("Vision-alignment source audit preprocessing differs from training")

    targets = data.mixture.resolved_targets()
    means = data.mixture.mean_loss_weight
    sampling = data.mixture.sampling_weights()
    for field_name, expected in (
        ("target_loss_mass", targets),
        ("mean_loss_weight", means),
        ("sampling_probabilities", sampling),
    ):
        value = audit.get(field_name)
        if not isinstance(value, Mapping) or _canonical_sha256(value) != _canonical_sha256(
            expected
        ):
            raise ValueError(f"Vision-alignment source audit {field_name} differs from training")

    delivered = audit.get("expected_loss_mass")
    expected_delivered = expected_loss_mass(sampling, means)
    if (
        not isinstance(delivered, Mapping)
        or set(delivered) != set(targets)
        or any(
            isinstance(delivered[name], bool)
            or not isinstance(delivered[name], (int, float))
            or not math.isclose(
                float(delivered[name]),
                float(expected_delivered[name]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for name in targets
        )
    ):
        raise ValueError("Vision-alignment source audit loss-mass calibration differs")

    inputs = audit.get("inputs")
    summaries = audit.get("sources")
    if (
        not isinstance(inputs, Mapping)
        or not isinstance(summaries, Mapping)
        or set(inputs) != set(targets)
        or set(summaries) != set(targets)
    ):
        raise ValueError("Vision-alignment source audit source set differs")
    for source_name in targets:
        source = inputs[source_name]
        summary = summaries[source_name]
        if not isinstance(source, Mapping) or not isinstance(summary, Mapping):
            raise ValueError(f"Vision-alignment source audit {source_name!r} is invalid")
        probe_indices = source.get("probe_indices")
        row_hashes = source.get("serialized_row_hashes")
        dataset_size = source.get("dataset_size")
        if (
            not isinstance(source.get("dataset_fingerprint"), str)
            or not source["dataset_fingerprint"]
            or type(dataset_size) is not int
            or dataset_size < 1
            or not isinstance(probe_indices, list)
            or not probe_indices
            or any(
                type(index) is not int or index < 0 or index >= dataset_size
                for index in probe_indices
            )
            or len(set(probe_indices)) != len(probe_indices)
            or source.get("probe_indices_sha256") != _canonical_sha256(probe_indices)
            or not isinstance(row_hashes, list)
            or not row_hashes
            or any(
                not isinstance(row_hash, str) or re.fullmatch(r"[0-9a-f]{64}", row_hash) is None
                for row_hash in row_hashes
            )
            or source.get("serialized_row_hashes_sha256") != _canonical_sha256(row_hashes)
            or re.fullmatch(r"[0-9a-f]{64}", str(source.get("sha256", ""))) is None
            or summary.get("mean_sum_loss_masks") != means[source_name]
            or summary.get("error_samples") != []
        ):
            raise ValueError(f"Vision-alignment source audit probe for {source_name!r} is invalid")
    if (
        re.fullmatch(r"[0-9a-f]{64}", str(audit.get("source_registry_sha256", ""))) is None
        or re.fullmatch(r"[0-9a-f]{64}", str(audit.get("input_content_sha256", ""))) is None
    ):
        raise ValueError("Vision-alignment source audit content identity is invalid")
    if config.phase is VisionAlignmentPhase.joint:
        exporter = audit.get("exporter_implementation")
        if not isinstance(exporter, Mapping):
            raise ValueError("Joint source audit lacks exporter metadata")
    elif re.fullmatch(r"[0-9a-f]{64}", str(audit.get("exporter_sha256", ""))) is None:
        raise ValueError("Vision-alignment source audit lacks exporter metadata")
    return audit


def _validate_validation_manifest(
    config: ExperimentConfig,
    source_audit: Optional[Mapping[str, Any]],
) -> Optional[Mapping[str, Any]]:
    """Validate the checksum-pinned PixMo train/validation split manifest."""
    if config.data.allow_unpinned_synthetic_smoke:
        if (
            config.evaluation.validation_manifest_path is not None
            or config.evaluation.validation_manifest_sha256 is not None
        ):
            raise ValueError("Synthetic smoke must not claim a production validation manifest")
        return None
    path_value = config.evaluation.validation_manifest_path
    expected_sha = config.evaluation.validation_manifest_sha256
    if path_value is None or expected_sha is None or source_audit is None:
        raise ValueError("Production bridge training requires a pinned validation manifest")

    path = Path(path_value).expanduser().resolve()
    try:
        raw = path.read_bytes()
        manifest = json.loads(raw, object_pairs_hook=_strict_json_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Invalid vision-alignment validation manifest {path}: {error}") from error
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("Vision-alignment validation manifest SHA differs")
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("format") != "vision_alignment_validation_manifest"
        or manifest.get("version") != 3
    ):
        raise ValueError("Vision-alignment validation manifest identity is incompatible")

    output = manifest.get("output")
    source = manifest.get("source")
    if not isinstance(output, Mapping) or not isinstance(source, Mapping):
        raise ValueError("Validation manifest lacks source/output dataset identities")
    output_value = output.get("dataset_path")
    if not isinstance(output_value, str) or not output_value:
        raise ValueError("Validation manifest output path is invalid")
    relative_output = Path(output_value)
    if relative_output.is_absolute() or ".." in relative_output.parts:
        raise ValueError("Validation manifest output path must stay inside its artifact directory")
    output_path = (path.parent / relative_output).resolve()
    if (
        not output_path.is_relative_to(path.parent)
        or output_path != Path(config.data.pixmo_cap_path).expanduser().resolve()
    ):
        raise ValueError("Configured PixMoCap path differs from the validation manifest")

    output_splits = output.get("splits")
    source_splits = source.get("splits")
    if (
        not isinstance(output_splits, Mapping)
        or not isinstance(source_splits, Mapping)
        or set(output_splits) != {"train", "validation"}
        or set(source_splits) != {"train", "validation"}
    ):
        raise ValueError("Validation manifest must identify train and validation splits")

    for split_name in ("train", "validation"):
        split = output_splits[split_name]
        if (
            not isinstance(split, Mapping)
            or not isinstance(split.get("dataset_fingerprint"), str)
            or not split["dataset_fingerprint"]
            or type(split.get("examples")) is not int
            or split["examples"] < 1
        ):
            raise ValueError(f"Validation manifest {split_name} split identity is invalid")
    if output_splits["validation"]["examples"] < config.evaluation.examples_per_source:
        raise ValueError("Validation manifest does not contain enough held-out examples")

    source_validation = source_splits["validation"]
    output_validation = output_splits["validation"]
    preserved_fields = (
        "examples",
        "row_image_paths_sha256",
        "row_image_content_sha256",
        "unique_image_paths",
        "unique_image_content",
    )
    if not isinstance(source_validation, Mapping) or any(
        source_validation.get(field) != output_validation.get(field) for field in preserved_fields
    ):
        raise ValueError("Validation manifest does not preserve the source validation split")
    filtering = manifest.get("filtering")
    if not isinstance(filtering, Mapping) or filtering.get("output_overlap_unique_images") != 0:
        raise ValueError("Validation manifest train and validation images are not disjoint")

    inventories = manifest.get("inventories")
    train_inventory = inventories.get("train") if isinstance(inventories, Mapping) else None
    if not isinstance(train_inventory, Mapping) or train_inventory.get(
        "sha256"
    ) != source_audit.get("image_manifest_sha256"):
        raise ValueError("Bridge source audit and validation manifest identify different images")

    inputs = source_audit.get("inputs")
    output_train = output_splits["train"]
    if not isinstance(inputs, Mapping):
        raise ValueError("Bridge source audit lacks runtime input identities")
    for source_name in ("pixmo_caption", "pixmo_transcript"):
        audit_input = inputs.get(source_name)
        if (
            not isinstance(audit_input, Mapping)
            or audit_input.get("dataset_fingerprint") != output_train["dataset_fingerprint"]
            or audit_input.get("dataset_size") != output_train["examples"]
        ):
            raise ValueError(
                f"Bridge source {source_name!r} differs from the pinned PixMo train split"
            )
    return manifest


def _validate_live_validation_dataset(dataset: Any, manifest: Optional[Mapping[str, Any]]) -> None:
    """Bind an evaluator's live dataset object to its pinned validation manifest."""
    if manifest is None:
        return
    live_fingerprint = _runtime_dataset_fingerprint(dataset)
    split = cast(Mapping[str, Any], cast(Mapping[str, Any], manifest["output"])["splits"])[
        "validation"
    ]
    if live_fingerprint != split["dataset_fingerprint"]:
        raise ValueError(
            "Live validation dataset fingerprint differs from the pinned validation manifest: "
            f"expected {split['dataset_fingerprint']!r}, got {live_fingerprint!r}"
        )
    if len(dataset) != split["examples"]:
        raise ValueError(
            "Live validation dataset length differs from the pinned validation manifest: "
            f"expected {split['examples']}, got {len(dataset)}"
        )
    dataset.validate_required_annotations()


def _set_contract_hashes(config: ExperimentConfig) -> None:
    data_contract: Dict[str, Any] = {
        "phase": config.phase.value,
        "formatter_version": FORMATTER_VERSION,
        "data": config.data.as_config_dict(),
        "evaluation": config.evaluation.as_config_dict(),
        "collator": config.collator.as_config_dict(),
        "global_batch_size": config.global_batch_size,
        "data_seed": config.data_seed,
    }
    config.vision_alignment.data_contract_sha256 = _canonical_sha256(data_contract)
    trainable_contract: Dict[str, Any] = {
        "model": config.model.as_config_dict(),
        "train_module": config.train_module.as_config_dict(),
        "router_lb_loss_weight": config.router_lb_loss_weight,
        "max_duration": {
            "value": config.trainer.max_duration.value,
            "unit": config.trainer.max_duration.unit.value,
        },
    }
    if config.phase is not VisionAlignmentPhase.bridge:
        # Preserve the completed perception/joint checkpoints' legacy contract identity.
        trainable_contract["perception_trainability_arm"] = "treatment"
    if _joint_vision_is_frozen(config):
        # Keep the completed treatment checkpoint's hash byte-for-byte compatible while giving
        # the new, non-default arm an explicit identity in addition to its derived train module.
        trainable_contract["joint_trainability_arm"] = config.joint_trainability_arm.value
    config.vision_alignment.trainable_contract_sha256 = _canonical_sha256(trainable_contract)


def _load_pinned_native_parent_paths(artifacts: ArtifactConfig) -> Tuple[str, ...]:
    """Load the exact expanded s002 parent-path inventory behind a replay receipt."""
    path = Path(artifacts.base_checkpoint) / "data_paths.txt"
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError(
            f"Could not read pinned native replay parent paths {path}: {error}"
        ) from error
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != artifacts.base_data_paths_sha256:
        raise ValueError(
            "Native replay parent data-path bytes differ from the pinned SHA-256: "
            f"expected {artifacts.base_data_paths_sha256}, got {actual_sha256}"
        )
    try:
        parent_paths = tuple(raw.decode("utf-8").splitlines())
    except UnicodeDecodeError as error:
        raise ValueError("Native replay parent data paths must be UTF-8") from error
    if (
        len(parent_paths) != _JOINT_NATIVE_REPLAY_PARENT_OBJECTS
        or len(set(parent_paths)) != len(parent_paths)
        or any(not value or value.strip() != value for value in parent_paths)
    ):
        raise ValueError(
            "Native replay parent data paths must contain exactly "
            f"{_JOINT_NATIVE_REPLAY_PARENT_OBJECTS} unique canonical rows"
        )
    return parent_paths


def _native_parent_dataset_fingerprint(
    remote_sources: Sequence[Mapping[str, Any]],
) -> str:
    """Reconstruct the pinned s002 NumpyFSLDataset fingerprint from remote metadata."""
    digest = hashlib.sha256()
    digest.update(b"class=NumpyFSLDataset")
    for field_name, field_value in (
        ("vocab_size", 100_278),
        ("pad_token_id", 100_277),
        ("eos_token_id", 100_257),
        ("dtype", np.uint32),
        ("max_target_sequence_length", _JOINT_SEQUENCE_LENGTH),
        ("bos_token_id", None),
    ):
        digest.update(f"{field_name}={field_value},".encode())
    for index, source in enumerate(remote_sources):
        parent_path = source.get("parent_path")
        size_bytes = source.get("size_bytes")
        if (
            not isinstance(parent_path, str)
            or not parent_path
            or isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes <= 0
        ):
            raise ValueError(f"Native replay remote source {index} has invalid path or size")
        digest.update(f"path={os.path.basename(parent_path)},size={size_bytes},".encode())
    return digest.hexdigest()


def _validate_native_replay_pair(config: ExperimentConfig) -> None:
    """Validate pinned, disjoint native train/holdout replay manifests for joint CPT."""
    train_config = config.data.native_text_replay
    holdout_config = config.evaluation.native_text_holdout
    if train_config is None or holdout_config is None:
        raise ValueError(
            "Joint vision alignment requires native replay train and holdout manifests"
        )
    expected = {
        "expected_parent_checkpoint": config.artifacts.base_checkpoint,
        "expected_parent_mix": config.artifacts.parent_text_mix,
        "expected_parent_paths_sha256": config.artifacts.base_data_paths_sha256,
    }
    for split, replay_config, pinned_fingerprint in (
        ("train", train_config, config.data.native_text_replay_fingerprint),
        ("holdout", holdout_config, config.evaluation.native_text_holdout_fingerprint),
    ):
        for field_name, expected_value in expected.items():
            if getattr(replay_config, field_name) != expected_value:
                raise ValueError(f"Native {split} replay has incompatible {field_name}")
        if (
            replay_config.expected_fingerprint is None
            or pinned_fingerprint != replay_config.expected_fingerprint
            or not replay_config.validate_source_files
            or replay_config.verify_source_hashes
            or replay_config.verification_receipt_path is None
            or replay_config.expected_verification_receipt_sha256 is None
        ):
            raise ValueError(
                f"Native {split} replay must pin its fingerprint and offline verification "
                "receipt, validate current file sizes, and avoid runtime full-corpus hashing"
            )
    if (
        train_config.verification_receipt_path != holdout_config.verification_receipt_path
        or train_config.expected_verification_receipt_sha256
        != holdout_config.expected_verification_receipt_sha256
    ):
        raise ValueError("Native train and holdout must use the same verification receipt")

    assert train_config.verification_receipt_path is not None
    assert train_config.expected_verification_receipt_sha256 is not None
    receipt = NativeTextReplayVerificationReceipt.load(
        train_config.verification_receipt_path,
        expected_sha256=train_config.expected_verification_receipt_sha256,
    )
    if receipt.version != 3:
        raise ValueError("Joint native replay requires a compact v3 verification receipt")
    expected_receipt_lineage = {
        "parent_paths_sha256": config.artifacts.base_data_paths_sha256,
        "parent_mix_sha256": BASE_PARENT_MIX_SHA256,
        "parent_config_sha256": config.artifacts.base_config_sha256,
        "parent_trainer_state_sha256": BASE_TRAINER_STATE_SHA256,
        "parent_dataset_fingerprint": BASE_DATASET_FINGERPRINT,
    }
    for field_name, expected_value in expected_receipt_lineage.items():
        if getattr(receipt, field_name) != expected_value:
            raise ValueError(f"Native replay verification receipt has incompatible {field_name}")
    parent_paths = _load_pinned_native_parent_paths(config.artifacts)
    receipt_parent_paths = tuple(
        cast(str, remote_source["parent_path"]) for remote_source in receipt.remote_sources
    )
    if receipt_parent_paths != parent_paths:
        raise ValueError(
            "Native replay verification receipt remote_sources must exactly enumerate the "
            "pinned parent data paths"
        )
    if _native_parent_dataset_fingerprint(receipt.remote_sources) != BASE_DATASET_FINGERPRINT:
        raise ValueError(
            "Native replay verification receipt remote sizes do not reconstruct the pinned "
            "parent dataset fingerprint"
        )

    train = train_config.build().manifest
    holdout = holdout_config.build().manifest
    if train.version != 3 or holdout.version != 3:
        raise ValueError("Joint native replay requires compact v3 train and holdout manifests")
    if (
        train.sequence_length != config.data.sequence_length
        or holdout.sequence_length != config.data.sequence_length
        or holdout.num_windows < config.evaluation.examples_per_source
        or train.provenance.get("split") != "train"
        or holdout.provenance.get("split") != "holdout"
    ):
        raise ValueError("Native replay train/holdout lineage, size, or sequence contract differs")

    expected_manifest_lineage: Dict[str, Any] = {
        "parent_checkpoint": config.artifacts.base_checkpoint,
        "parent_mix": config.artifacts.parent_text_mix,
        "parent_paths_sha256": config.artifacts.base_data_paths_sha256,
        "parent_config_sha256": config.artifacts.base_config_sha256,
        "parent_trainer_state_sha256": BASE_TRAINER_STATE_SHA256,
        "parent_dataset_fingerprint": BASE_DATASET_FINGERPRINT,
        "remote_snapshot_sha256": receipt.remote_snapshot_sha256,
        "compact_materialization_sha256": receipt.compact_materialization_sha256,
        "selection_algorithm": _JOINT_NATIVE_REPLAY_SELECTION_ALGORITHM,
        "selection_seed": _JOINT_NATIVE_REPLAY_SELECTION_SEED,
    }
    for split, manifest in (("train", train), ("holdout", holdout)):
        for field_name, expected_value in expected_manifest_lineage.items():
            if manifest.provenance.get(field_name) != expected_value:
                raise ValueError(f"Native {split} replay manifest has incompatible {field_name}")
    receipt.validate_pair(train, holdout)


def _validate_joint_parent_projection_lineage(
    config: ExperimentConfig, parent_config: Mapping[str, Any]
) -> None:
    """Bind the joint projection to the provenance saved by its perception parent."""
    if config.phase is not VisionAlignmentPhase.joint:
        return
    parent_data = parent_config.get("data")
    if not isinstance(parent_data, Mapping):
        raise ValueError("Joint parent checkpoint lacks its saved perception data config")
    parent_provenance_path = parent_data.get("perception_provenance_path")
    parent_provenance_sha256 = parent_data.get("perception_provenance_sha256")
    if (
        not isinstance(parent_provenance_path, str)
        or not parent_provenance_path
        or not isinstance(parent_provenance_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", parent_provenance_sha256) is None
    ):
        raise ValueError(
            "Joint parent checkpoint lacks a pinned perception provenance path and raw SHA-256"
        )
    unresolved_parent_path = Path(parent_provenance_path).expanduser()
    resolved_parent_path = unresolved_parent_path.resolve()
    if (
        not unresolved_parent_path.is_absolute()
        or ".." in unresolved_parent_path.parts
        or str(resolved_parent_path) != parent_provenance_path
    ):
        raise ValueError("Joint parent checkpoint perception provenance path is not normalized")
    projection_parent = _joint_visual_projection(config).parent_provenance
    if (
        resolved_parent_path != projection_parent.path
        or parent_provenance_sha256 != projection_parent.raw_sha256
    ):
        raise ValueError(
            "Joint visual projection parent provenance differs from the approved perception "
            "checkpoint data config"
        )


def _validate_joint_parent_trainability_lineage(
    config: ExperimentConfig, parent_config: Mapping[str, Any]
) -> None:
    """Prove that a frozen-vision joint arm never inherited an unfrozen vision encoder."""
    if not _joint_vision_is_frozen(config):
        return
    if parent_config.get("perception_trainability_arm") != "frozen_vision_control":
        raise ValueError("Joint frozen-vision control requires the frozen-vision perception parent")
    train_module = parent_config.get("train_module")
    if not isinstance(train_module, Mapping):
        raise ValueError("Frozen-vision perception parent lacks its train-module config")
    freeze_params = train_module.get("freeze_params")
    if not isinstance(freeze_params, list) or "vision.*" not in freeze_params:
        raise ValueError("Frozen-vision perception parent did not freeze the vision encoder")
    optim = train_module.get("optim")
    groups = optim.get("group_overrides") if isinstance(optim, Mapping) else None
    if not isinstance(groups, list):
        raise ValueError("Frozen-vision perception parent lacks optimizer-group config")
    vision_groups = [
        group
        for group in groups
        if isinstance(group, Mapping) and group.get("params") == ["*vision.*"]
    ]
    if len(vision_groups) != 1:
        raise ValueError(
            "Frozen-vision perception parent must contain one exact vision optimizer group"
        )
    opts = vision_groups[0].get("opts")
    vision_lr = opts.get("lr") if isinstance(opts, Mapping) else None
    if (
        isinstance(vision_lr, bool)
        or not isinstance(vision_lr, (int, float))
        or not math.isclose(float(vision_lr), 0.0, abs_tol=0.0)
    ):
        raise ValueError("Frozen-vision perception parent must use zero vision learning rate")


def _validate_parent_or_resume(config: ExperimentConfig) -> None:
    existing = _latest_output_checkpoint(config)
    if existing is not None:
        saved, _ = _checkpoint_config(existing)
        saved_meta = saved.get("vision_alignment")
        if not isinstance(saved_meta, dict):
            raise ValueError(
                f"Existing output checkpoint {existing} is not a vision-alignment checkpoint"
            )
        for field_name, expected in {
            "recipe_version": RECIPE_VERSION,
            "formatter_version": FORMATTER_VERSION,
            "phase": config.phase.value,
            "lineage_id": config.vision_alignment.lineage_id,
            "parent_checkpoint": config.vision_alignment.parent_checkpoint,
            "data_contract_sha256": config.vision_alignment.data_contract_sha256,
            "trainable_contract_sha256": config.vision_alignment.trainable_contract_sha256,
        }.items():
            if saved_meta.get(field_name) != expected:
                raise ValueError(
                    f"Existing output checkpoint {existing} has incompatible {field_name}: "
                    f"{saved_meta.get(field_name)!r} != {expected!r}"
                )
        saved_parent_sha = saved_meta.get("parent_config_sha256")
        configured_parent_sha = config.initialization.parent_config_sha256
        if config.phase is VisionAlignmentPhase.bridge:
            if saved_parent_sha is not None:
                raise ValueError(
                    f"Existing bridge checkpoint {existing} unexpectedly records a parent"
                )
        elif not isinstance(saved_parent_sha, str) or len(saved_parent_sha) != 64:
            raise ValueError(
                f"Existing output checkpoint {existing} lacks its parent config fingerprint"
            )
        elif configured_parent_sha is not None and configured_parent_sha != saved_parent_sha:
            raise ValueError(
                f"Existing output checkpoint {existing} has parent config SHA "
                f"{saved_parent_sha}, expected {configured_parent_sha}"
            )
        config.initialization.parent_config_sha256 = saved_parent_sha
        config.vision_alignment.parent_config_sha256 = saved_parent_sha
        if config.phase is VisionAlignmentPhase.joint:
            parent = config.initialization.checkpoint
            if not isinstance(parent, str) or not parent:
                raise ValueError("A joint resume lacks its perception parent checkpoint")
            parent_config, parent_sha = _checkpoint_config(parent)
            if parent_sha != saved_parent_sha:
                raise ValueError(
                    "Joint resume perception parent config differs from its saved lineage SHA"
                )
            _validate_joint_parent_projection_lineage(config, parent_config)
            _validate_joint_parent_trainability_lineage(config, parent_config)
        return

    if config.phase is VisionAlignmentPhase.bridge:
        return
    parent = config.initialization.checkpoint
    assert parent is not None
    parent_config, parent_sha = _checkpoint_config(parent)
    parent_meta = parent_config.get("vision_alignment")
    if not isinstance(parent_meta, dict):
        raise ValueError(f"Parent {parent} is not a vision-alignment checkpoint")
    expected_phase = config.initialization.expected_parent_phase
    if expected_phase is None or parent_meta.get("phase") != expected_phase.value:
        raise ValueError(
            f"Parent {parent} phase is {parent_meta.get('phase')!r}; "
            f"expected {expected_phase.value if expected_phase is not None else None!r}"
        )
    if parent_meta.get("recipe_version") != RECIPE_VERSION:
        raise ValueError(f"Parent {parent} has an incompatible recipe version")
    configured_sha = config.initialization.parent_config_sha256
    if configured_sha is not None and configured_sha != parent_sha:
        raise ValueError(
            f"Parent config SHA mismatch: configured {configured_sha}, actual {parent_sha}"
        )
    config.initialization.parent_config_sha256 = parent_sha
    config.vision_alignment.parent_config_sha256 = parent_sha
    _validate_permanent_checkpoint(parent)
    _validate_joint_parent_projection_lineage(config, parent_config)
    _validate_joint_parent_trainability_lineage(config, parent_config)


def _validate_canonical_data_policy(config: ExperimentConfig) -> None:
    """Reject structural data-policy drift while allowing pinned artifact overrides."""
    expected_targets = VisionAlignmentMixtureConfig(phase=config.phase.value).resolved_targets()
    actual_targets = config.data.mixture.resolved_targets()
    expected_sources = set(expected_targets)
    actual_sources = set(actual_targets)
    if actual_sources != expected_sources:
        raise ValueError(
            f"Phase {config.phase.value} requires canonical mixture sources "
            f"{sorted(expected_sources)}; got {sorted(actual_sources)}"
        )
    if actual_targets != expected_targets:
        raise ValueError(
            f"Phase {config.phase.value} requires canonical target loss-mass ratios "
            f"{expected_targets}; got {actual_targets}"
        )

    calibrated_sources = set(config.data.mixture.mean_loss_weight)
    if calibrated_sources != expected_sources:
        raise ValueError(
            "Loss-mass calibration must contain exactly the canonical phase sources; "
            f"expected {sorted(expected_sources)}, got {sorted(calibrated_sources)}"
        )

    expected_fields: Tuple[Tuple[str, Any], ...] = (
        ("message_format", "document"),
        ("loss_token_weighting", "root_subsegments_root_tokens"),
        ("caption_prompt", "Description:"),
        ("transcript_prompt", "Transcript:"),
        ("max_crops", MAX_CROPS),
        ("pack_sequences", True),
        ("pack_buffer_size", PACK_BUFFER_SIZE),
        ("pack_max_crops", PACK_MAX_CROPS),
    )
    for field_name, expected_value in expected_fields:
        actual_value = getattr(config.data, field_name)
        if actual_value != expected_value:
            raise ValueError(
                f"Vision-alignment data policy requires {field_name}={expected_value!r}; "
                f"got {actual_value!r}"
            )


def _validate_optimizer_scheduler_contract(config: ExperimentConfig, policy: _PhasePolicy) -> None:
    """Reject unsafe optimizer topology or scheduler routing changes."""
    optim = config.train_module.optim
    expected_default_lr = policy.lm_lr if policy.lm_lr > 0 else policy.connector_lr
    if not math.isclose(float(optim.lr), expected_default_lr) or optim.lr <= 0:
        raise ValueError("Optimizer default LR must remain positive and phase-derived")
    if (
        optim.use_distributed is not True
        or optim.check_nan_inf_grad is not True
        or optim.clip_grad_norm_by_scheduler_group is not True
        or optim.compile is not False
    ):
        raise ValueError("Vision alignment requires the pinned distributed optimizer safeguards")

    effective_vision_lr = 0.0 if _joint_vision_is_frozen(config) else policy.vision_lr
    expected_groups = (
        ("*lm.embeddings.weight", policy.connector_lr, "connector"),
        ("*connector.*", policy.connector_lr, "connector"),
        ("*vision.*", effective_vision_lr, "vision"),
    )
    groups = optim.group_overrides or []
    if len(groups) != len(expected_groups):
        raise ValueError("Vision alignment requires exactly three optimizer override groups")
    for group, (pattern, expected_lr, scheduler_name) in zip(groups, expected_groups):
        if group.params != [pattern] or set(group.opts) != {
            "lr",
            "weight_decay",
            "scheduler_name",
        }:
            raise ValueError("Vision-alignment optimizer group topology was overridden")
        if (
            not math.isclose(float(group.opts["lr"]), expected_lr)
            or group.opts["weight_decay"] != 0.0
            or group.opts["scheduler_name"] != scheduler_name
        ):
            raise ValueError("Vision-alignment optimizer LR or scheduler routing was overridden")
    if policy.connector_lr <= 0 or (
        policy.phase is not VisionAlignmentPhase.bridge
        and not _joint_vision_is_frozen(config)
        and effective_vision_lr <= 0
    ):
        raise ValueError("Every trainable visual component requires a positive LR")
    if (policy.phase is VisionAlignmentPhase.joint) != (policy.lm_lr > 0):
        raise ValueError("The LM LR must be positive only in the joint phase")

    scheduler = config.train_module.scheduler
    if (
        not isinstance(scheduler, PerGroupScheduler)
        or scheduler.group_name_field != "scheduler_name"
        or set(scheduler.schedulers) != {"connector", "vision"}
        or not isinstance(scheduler.default, CosWithWarmup)
    ):
        raise ValueError("Vision alignment requires the pinned per-group scheduler topology")
    expected_schedulers = (
        (scheduler.schedulers["connector"], policy.connector_warmup, policy.connector_t_max),
        (scheduler.schedulers["vision"], policy.vision_warmup, None),
        (scheduler.default, policy.lm_warmup, None),
    )
    for component_scheduler, expected_warmup, expected_t_max in expected_schedulers:
        if (
            not isinstance(component_scheduler, CosWithWarmup)
            or component_scheduler.warmup != expected_warmup
            or component_scheduler.t_max != expected_t_max
            or not math.isclose(component_scheduler.alpha_f, 0.1)
        ):
            raise ValueError("Vision-alignment scheduler shape or warmup was overridden")

    duration = config.trainer.max_duration
    if duration.unit.value != "steps" or duration.value <= 0:
        raise ValueError("Vision-alignment phase duration must be a positive step count")
    active_warmups = [policy.connector_warmup]
    if policy.phase is not VisionAlignmentPhase.bridge and not _joint_vision_is_frozen(config):
        active_warmups.append(policy.vision_warmup)
    if policy.phase is VisionAlignmentPhase.joint:
        active_warmups.append(policy.lm_warmup)
    if not config.data.allow_unpinned_synthetic_smoke and max(active_warmups) >= duration.value:
        raise ValueError("Every active warmup must be shorter than the production phase duration")
    if (
        not config.data.allow_unpinned_synthetic_smoke
        and policy.connector_t_max is not None
        and not (policy.connector_warmup < policy.connector_t_max <= duration.value)
    ):
        raise ValueError(
            "The connector decay horizon must follow warmup and fit inside the phase duration"
        )


def _validate_native_artifact_phase(config: ExperimentConfig) -> None:
    """Forbid every native replay artifact outside the joint phase."""
    if config.phase is VisionAlignmentPhase.joint:
        return
    if any(
        value is not None
        for value in (
            config.data.native_text_replay,
            config.data.native_text_replay_fingerprint,
            config.evaluation.native_text_holdout,
            config.evaluation.native_text_holdout_fingerprint,
        )
    ):
        raise ValueError(
            f"Native replay configs and fingerprints are forbidden outside joint ({config.phase})"
        )


def _validate_phase_contract(config: ExperimentConfig, run_name: str) -> None:
    policy = _PHASE_POLICIES[config.phase]
    if config.artifacts != ArtifactConfig():
        raise ValueError("Pinned vision-alignment artifact identities may not be overridden")
    if config.required_run_name != run_name or config.vision_alignment.lineage_id != run_name:
        raise ValueError(
            "Positional run name, required_run_name, and lineage_id must match exactly"
        )
    expected_save_folder = f"{VISION_ALIGNMENT_ROOT}/checkpoints/{run_name}"
    if config.trainer.save_folder != expected_save_folder:
        raise ValueError(
            f"Vision alignment save folder must be {expected_save_folder!r}, "
            f"got {config.trainer.save_folder!r}"
        )
    checkpoint_root = (Path(VISION_ALIGNMENT_ROOT) / "checkpoints").resolve()
    resolved_save_folder = Path(config.trainer.save_folder).resolve()
    if resolved_save_folder.parent != checkpoint_root:
        raise ValueError(f"Vision-alignment output must be one direct child of {checkpoint_root}")
    if config.trainer.save_overwrite is not True:
        raise ValueError("Vision alignment requires save_overwrite=True for exact resumes")
    if config.trainer.no_checkpoints is not False:
        raise ValueError("Vision alignment may not disable checkpoints or checkpoint loading")
    if config.trainer.no_evals is not False:
        raise ValueError("Vision alignment may not disable its configured intrinsic evaluators")
    wandb = config.trainer.callbacks.get("wandb")
    if (
        not isinstance(wandb, WandBCallback)
        or wandb.name != run_name
        or wandb.project != WANDB_PROJECT
        or wandb.entity != WANDB_ENTITY
        or wandb.enabled is not True
        or wandb.auto_resume is not True
    ):
        raise ValueError("W&B identity must match the positional run name")
    checkpointer = config.trainer.callbacks.get("checkpointer")
    if (
        not isinstance(checkpointer, CheckpointerCallback)
        or checkpointer.enabled is not True
        or checkpointer.pre_train_checkpoint is not True
    ):
        raise ValueError("Vision alignment requires its pre-train checkpoint callback")
    if not config.launch.name.startswith(f"{run_name}-"):
        raise ValueError("Beaker launch identity must be derived from the positional run name")
    if config.launch.hostnames:
        raise ValueError("Vision alignment profiles may select Holmes only, not exact hosts")
    if config.launch.clusters != [BEAKER_CLUSTER]:
        raise ValueError(f"Vision alignment must target only {BEAKER_CLUSTER}")
    if config.launch.workspace != BEAKER_WORKSPACE or config.launch.budget != BEAKER_BUDGET:
        raise ValueError("Vision alignment workspace and budget are pinned by the recipe")
    if config.launch.num_gpus != 8 or config.launch.num_nodes < 1:
        raise ValueError("Vision alignment requires one or more complete 8-GPU Holmes nodes")
    if any(
        secret is not None
        for secret in (
            config.launch.aws_config_secret,
            config.launch.aws_credentials_secret,
            config.launch.google_credentials_secret,
        )
    ):
        raise ValueError("Vision alignment training must not receive cloud credentials")
    if {secret.name for secret in config.launch.env_secrets} != {
        "BEAKER_TOKEN",
        "WANDB_API_KEY",
    }:
        raise ValueError("Vision alignment launch has an unexpected secret surface")
    if not any(
        bucket.bucket == "oe-training-default" and bucket.mount == "/weka/oe-training-default"
        for bucket in (config.launch.weka_buckets or [])
    ):
        raise ValueError("Vision alignment requires the approved training Weka mount")
    if not config.launch.beaker_image or not config.launch.post_setup:
        raise ValueError("Vision alignment requires the pinned runtime image and setup hook")

    if config.initialization.mode is not policy.initialization_mode:
        raise ValueError(
            f"Phase {config.phase.value} requires initialization mode "
            f"{policy.initialization_mode.value}"
        )
    if config.initialization.expected_parent_phase is not policy.expected_parent_phase:
        raise ValueError("Initialization parent-phase contract was overridden")
    _validate_joint_trainability_arm(config)
    expected_freeze_params = list(policy.freeze_params)
    if _joint_vision_is_frozen(config):
        expected_freeze_params.insert(0, "vision.*")
    if expected_freeze_params != (config.train_module.freeze_params or []):
        raise ValueError("Phase freeze patterns are derived and may not be overridden")
    if (
        config.train_module.train_embedding_rows is None
        or len(config.train_module.train_embedding_rows) != 6
    ):
        raise ValueError("Exactly six input-only image-token rows must be trainable")
    _validate_optimizer_scheduler_contract(config, policy)

    _validate_canonical_data_policy(config)
    if config.data.message_format != "document":
        raise ValueError("Vision alignment requires native document serialization")
    if not config.data.caption_prompt or not config.data.transcript_prompt:
        raise ValueError("Caption and transcript prompts must be explicit and non-empty")
    if not config.data.require_transcript:
        raise ValueError("Vision alignment forbids caption fallback for the transcript source")
    if config.data.sequence_length != config.collator.pad_sequence_length:
        raise ValueError("Data and collator sequence lengths must match")
    if config.data.sequence_length != config.train_module.max_sequence_length:
        raise ValueError("Data and train-module sequence lengths must match")
    if config.global_batch_size % config.data.sequence_length != 0:
        raise ValueError("global_batch_size must be divisible by sequence_length")
    if config.data.mixture.phase != config.phase.value:
        raise ValueError("Mixture phase must match the experiment phase")
    if not config.data.pack_sequences:
        raise ValueError("Vision alignment requires packing for exact source delivery telemetry")
    if config.train_module.source_loss_mass_targets != config.data.mixture.resolved_targets():
        raise ValueError("Train-module source telemetry targets differ from the data mixture")
    if not config.data.allow_unpinned_synthetic_smoke and (
        config.evaluation.interval is None
        or config.evaluation.interval <= 0
        or not config.evaluation.eval_on_startup
        or not config.evaluation.eval_on_finish
    ):
        raise ValueError(
            "Production vision alignment requires positive-cadence intrinsic evaluation "
            "at startup and finish"
        )
    if config.evaluation.rank_batch_instances != policy.rank_microbatch_instances:
        raise ValueError(
            "Intrinsic evaluation rank batch must match the phase's supported train/eval "
            f"capacity ({policy.rank_microbatch_instances} instances)"
        )

    replay = config.data.native_text_replay
    if config.phase is VisionAlignmentPhase.joint:
        if replay is None:
            raise ValueError("The joint phase requires an exact native-text replay manifest")
        if replay.expected_parent_checkpoint != config.artifacts.base_checkpoint:
            raise ValueError("Native replay must pin the bare model's exact parent checkpoint")
        if replay.expected_parent_mix != config.artifacts.parent_text_mix:
            raise ValueError("Native replay must pin the bare model's exact parent text mix")
        if replay.expected_parent_paths_sha256 != config.artifacts.base_data_paths_sha256:
            raise ValueError(
                "Native replay must pin the bare model's exact expanded parent path manifest"
            )
        if replay.expected_fingerprint is None:
            raise ValueError("Joint native replay must pin expected_fingerprint")
        if config.data.native_text_replay_fingerprint != replay.expected_fingerprint:
            raise ValueError("Native replay data fingerprint does not match its config pin")
        if (
            not replay.validate_source_files
            or replay.verify_source_hashes
            or replay.verification_receipt_path is None
            or replay.expected_verification_receipt_sha256 is None
        ):
            raise ValueError(
                "Joint native replay requires a pinned offline verification receipt and "
                "runtime size checks without re-hashing the full corpus"
            )
        _validate_native_replay_pair(config)
    else:
        _validate_native_artifact_phase(config)

    source_audit = _validated_source_audit(config)
    if config.phase is VisionAlignmentPhase.perception:
        provenance = _perception_provenance(config)
        if any(
            len(provenance.selection(source_name, "validation").indices)
            != config.evaluation.examples_per_source
            for source_name in PERCEPTION_SOURCE_NAMES
        ):
            raise ValueError(
                "Perception provenance must provide exactly the configured held-out rows "
                "for every source"
            )
        if (
            config.evaluation.validation_manifest_path is not None
            or config.evaluation.validation_manifest_sha256 is not None
        ):
            raise ValueError("Perception uses its union provenance, not the PixMo-only manifest")
        if (
            config.data.joint_visual_projection_path is not None
            or config.data.joint_visual_projection_sha256 is not None
        ):
            raise ValueError("Joint visual projection is forbidden outside the joint phase")
    elif config.phase is VisionAlignmentPhase.joint:
        projection = _joint_visual_projection(config)
        if any(
            len(projection.selection(source_name, "validation").indices)
            != config.evaluation.examples_per_source
            for source_name in JOINT_VISUAL_SOURCE_NAMES
        ):
            raise ValueError(
                "Joint visual projection must provide exactly the configured held-out rows "
                "for every visual source"
            )
        if (
            config.data.perception_provenance_path is not None
            or config.data.perception_provenance_sha256 is not None
        ):
            raise ValueError("Perception provenance is forbidden outside the perception phase")
        if (
            config.evaluation.validation_manifest_path is not None
            or config.evaluation.validation_manifest_sha256 is not None
        ):
            raise ValueError("Joint uses its visual projection, not the PixMo-only manifest")
    else:
        if (
            config.data.perception_provenance_path is not None
            or config.data.perception_provenance_sha256 is not None
        ):
            raise ValueError("Perception provenance is forbidden outside the perception phase")
        if (
            config.data.joint_visual_projection_path is not None
            or config.data.joint_visual_projection_sha256 is not None
        ):
            raise ValueError("Joint visual projection is forbidden outside the joint phase")
        _validate_validation_manifest(config, source_audit)

    parent = config.initialization.checkpoint
    if config.phase is VisionAlignmentPhase.bridge:
        if parent is not None or config.trainer.load_path is not None:
            raise ValueError("A fresh bridge must initialize from the pinned bare artifacts")
        if config.trainer.load_strategy is not LoadStrategy.if_available:
            raise ValueError("Bridge must use if_available for exact same-phase resume")
        if config.trainer.load_optim_state is not None:
            raise ValueError("Bridge load_optim_state is owned by same-folder resume semantics")
        if config.trainer.load_trainer_state is not None:
            raise ValueError("Bridge load_trainer_state is owned by same-folder resume semantics")
    else:
        if parent is None or config.trainer.load_path != parent:
            raise ValueError("A new phase requires one exact parent checkpoint as load_path")
        if _parent_is_inside_output(parent, config.trainer.save_folder):
            raise ValueError(
                "Parent checkpoint must not be the output folder or one of its children"
            )
        if config.trainer.load_strategy is not LoadStrategy.always:
            raise ValueError("A phase transition must require its parent checkpoint")
        if config.trainer.load_optim_state is not False:
            raise ValueError("A phase transition must start with a fresh optimizer")
        if config.trainer.load_trainer_state is not False:
            raise ValueError("A phase transition must start with a fresh trainer/data cursor")

    _set_contract_hashes(config)
    _validate_parent_or_resume(config)


def build_config(
    script: str,
    run_name: str,
    overrides: List[str],
) -> ExperimentConfig:
    """Build and validate one phase config after resolving its phase selector first."""
    _validate_run_name(run_name)
    phase = _extract_phase(overrides)
    policy = _PHASE_POLICIES[phase]
    artifacts = ArtifactConfig()
    tokenizer, token_ids = _load_tokenizer(artifacts)
    if tokenizer.pad_token_id is None:
        raise ValueError(f"Tokenizer {artifacts.tokenizer_id!r} has no pad token")
    image_ids = _image_token_ids(token_ids)

    initialization = InitializationConfig(
        mode=policy.initialization_mode,
        expected_parent_phase=policy.expected_parent_phase,
    )
    data = VisionAlignmentDataConfig(
        sequence_length=policy.sequence_length,
        mixture=VisionAlignmentMixtureConfig(phase=phase.value),
    )
    collator = MultimodalCollatorConfig(
        pad_token_id=int(tokenizer.pad_token_id),
        label_ignore_index=-100,
        pad_sequence_length=policy.sequence_length,
    )
    train_module = _build_train_module_config(policy, image_ids)
    trainer = (
        TrainerConfig(
            save_folder=f"{VISION_ALIGNMENT_ROOT}/checkpoints/{run_name}",
            save_overwrite=True,
            load_path=None,
            load_strategy=(
                LoadStrategy.if_available
                if phase is VisionAlignmentPhase.bridge
                else LoadStrategy.always
            ),
            load_optim_state=None if phase is VisionAlignmentPhase.bridge else False,
            load_trainer_state=None if phase is VisionAlignmentPhase.bridge else False,
            metrics_collect_interval=5,
            cancel_check_interval=5,
            max_duration=Duration.steps(policy.max_steps),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=max(policy.max_steps // 4, 1),
                ephemeral_save_interval=max(policy.max_steps // 10, 1),
                save_async=False,
                pre_train_checkpoint=True,
                max_checkpoints=6,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                entity=WANDB_ENTITY,
                project=WANDB_PROJECT,
                enabled=WANDB_PROJECT is not None,
                cancel_check_interval=10,
                auto_resume=True,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("garbage_collector", GarbageCollectorCallback())
        .with_callback("beaker", BeakerCallback())
        .with_callback("console_logger", _build_console_logger())
    )
    launch_config = build_launch_config(
        name=run_name,
        root_dir=get_root_dir(BEAKER_CLUSTER),
        cmd=[script, "train", run_name, *overrides],
        cluster=BEAKER_CLUSTER,
        workspace=BEAKER_WORKSPACE,
        budget=BEAKER_BUDGET,
        num_nodes=2,
    )
    launch_config.aws_config_secret = None
    launch_config.aws_credentials_secret = None
    launch_config.google_credentials_secret = None
    launch_config.env_secrets = [
        secret
        for secret in launch_config.env_secrets
        if secret.name in ("BEAKER_TOKEN", "WANDB_API_KEY")
    ]
    _configure_launch_runtime(launch_config)

    config = ExperimentConfig(
        launch=launch_config,
        model=_build_model_config(token_ids, artifacts),
        collator=collator,
        train_module=train_module,
        trainer=trainer,
        phase=phase,
        artifacts=artifacts,
        initialization=initialization,
        data=data,
        evaluation=_build_evaluation_config(policy),
        vision_alignment=VisionAlignmentMetadataConfig(
            phase=phase,
            lineage_id=run_name,
            parent_checkpoint=initialization.checkpoint,
        ),
        global_batch_size=GLOBAL_BATCH_INSTANCES * policy.sequence_length,
        joint_trainability_arm=JointTrainabilityArm.treatment,
        required_run_name=run_name,
    ).merge(overrides)
    _apply_joint_trainability_arm(config)
    if (
        config.phase is not VisionAlignmentPhase.bridge
        and config.trainer.load_path is None
        and config.initialization.checkpoint is not None
    ):
        config.trainer.load_path = config.initialization.checkpoint
    config.vision_alignment.phase = config.phase
    config.vision_alignment.parent_checkpoint = config.initialization.checkpoint

    if config.data.native_text_replay is not None:
        manifest = config.data.native_text_replay.build(tokenizer).manifest
        config.data.native_text_replay_fingerprint = manifest.content_fingerprint
    if config.evaluation.native_text_holdout is not None:
        holdout_manifest = config.evaluation.native_text_holdout.build(tokenizer).manifest
        config.evaluation.native_text_holdout_fingerprint = holdout_manifest.content_fingerprint
    config.train_module.source_loss_mass_targets = config.data.mixture.resolved_targets()
    _configure_router_load_balancing(config.model.lm, config.router_lb_loss_weight)
    _validate_phase_contract(config, run_name)
    return config


def _visual_dataset_config(
    config: ExperimentConfig,
    token_ids: Molmo2TokenIds,
    source_name: str,
    *,
    split: str = "train",
) -> Any:
    return build_vision_alignment_dataset_config(
        _source_spec(config), token_ids, source_name, split=split
    )


def _build_mixture_sources(
    tokenizer, token_ids: Molmo2TokenIds, config: ExperimentConfig
) -> Tuple[List[Any], List[float], List[str]]:
    targets = config.data.mixture.resolved_targets()
    supported = {
        "pixmo_caption",
        "pixmo_transcript",
        "pixmo_points_basic",
        "pixmo_points_high_frequency",
        "cosyn_point",
        "native_text_replay",
        "ocr_document",
        "scalar_count",
        "count_numeric",
        "audited_alignment",
    }
    missing = sorted(set(targets) - supported)
    if missing:
        raise ValueError(
            "Vision-alignment phase contains source contracts without audited adapters: "
            f"{missing}. Implement and audit them; do not substitute QA/SFT sources."
        )
    audit = _validated_source_audit(config)
    perception_provenance = (
        _perception_provenance(config) if config.phase is VisionAlignmentPhase.perception else None
    )
    joint_projection = (
        _joint_visual_projection(config, token_ids)
        if config.phase is VisionAlignmentPhase.joint
        else None
    )
    weights = config.data.mixture.sampling_weights()
    sources: List[Tuple[str, Any, float]] = []
    for name in targets:
        dataset: Any
        if name == "native_text_replay":
            replay = config.data.native_text_replay
            if replay is None:
                raise ValueError("native_text_replay target requires its manifest config")
            dataset = replay.build(tokenizer)
            if dataset.sequence_length != config.data.sequence_length:
                raise ValueError("Native replay and joint training sequence lengths must match")
        else:
            if joint_projection is not None:
                dataset = build_selected_joint_dataset(
                    joint_projection,
                    tokenizer,
                    token_ids,
                    name,
                    logical_split="train",
                    validate_required_annotations=True,
                )
            elif perception_provenance is not None:
                dataset = build_selected_perception_dataset(
                    perception_provenance,
                    tokenizer,
                    token_ids,
                    name,
                    logical_split="train",
                    validate_required_annotations=True,
                    verify_finevision_materialization=False,
                )
            else:
                dataset = build_vision_alignment_dataset(
                    _source_spec(config),
                    tokenizer,
                    token_ids,
                    name,
                    split="train",
                    validate_required_annotations=True,
                )
        if audit is not None:
            dataset = _AuditedDataset(dataset, name, audit)
        sources.append((name, dataset, weights[name]))
    sources.sort(key=lambda item: item[0])
    names = [item[0] for item in sources]
    datasets = [item[1] for item in sources]
    sampling = [item[2] for item in sources]
    delivered = expected_loss_mass(dict(zip(names, sampling)), config.data.mixture.mean_loss_weight)
    log.info("Vision-alignment sampling probabilities: %s", dict(zip(names, sampling)))
    log.info("Calibrated expected supervised-loss mass: %s", delivered)
    return datasets, sampling, names


def _add_intrinsic_visual_evaluators(
    trainer,
    tokenizer,
    config: ExperimentConfig,
    collator,
    token_ids: Molmo2TokenIds,
    *,
    dp_world_size: int,
    dp_rank: int,
) -> None:
    eval_config = config.evaluation
    if eval_config.interval is None:
        return
    if eval_config.interval <= 0 or eval_config.examples_per_source <= 0:
        raise ValueError("Evaluation interval and example count must be positive")
    global_instances = eval_config.rank_batch_instances * dp_world_size
    if eval_config.examples_per_source % global_instances:
        raise ValueError("examples_per_source must divide the global evaluation batch")
    eval_batches = eval_config.examples_per_source // global_instances
    evaluators: List[Evaluator] = []
    validation_manifest = None
    perception_provenance = None
    joint_projection = None
    validation_sources: Tuple[str, ...]
    if config.phase is VisionAlignmentPhase.perception:
        perception_provenance = _perception_provenance(config)
        validation_sources = PERCEPTION_SOURCE_NAMES
    elif config.phase is VisionAlignmentPhase.joint:
        joint_projection = _joint_visual_projection(config, token_ids)
        validation_sources = JOINT_VISUAL_SOURCE_NAMES
    else:
        validation_manifest = _validate_validation_manifest(config, _validated_source_audit(config))
        validation_sources = ("pixmo_caption", "pixmo_transcript")
    for source_name in validation_sources:
        dataset: Any
        if joint_projection is not None:
            dataset = build_selected_joint_dataset(
                joint_projection,
                tokenizer,
                token_ids,
                source_name,
                logical_split="validation",
                validate_required_annotations=True,
            )
            if len(dataset) != eval_config.examples_per_source:
                raise ValueError(
                    f"Joint validation {source_name!r} must contain exactly "
                    f"{eval_config.examples_per_source} projection-selected rows"
                )
            dataset.validate_image_content()
        elif perception_provenance is not None:
            dataset = build_selected_perception_dataset(
                perception_provenance,
                tokenizer,
                token_ids,
                source_name,
                logical_split="validation",
                validate_required_annotations=True,
                verify_finevision_materialization=False,
            )
            if len(dataset) != eval_config.examples_per_source:
                raise ValueError(
                    f"Perception validation {source_name!r} must contain exactly "
                    f"{eval_config.examples_per_source} provenance-selected rows"
                )
            # Rehash the complete pinned held-out population before constructing the
            # evaluator. This is small (512 rows/source) and prevents mutable image paths
            # from silently invalidating train/eval disjointness after provenance build.
            dataset.validate_image_content()
        else:
            dataset = _visual_dataset_config(
                config, token_ids, source_name, split="validation"
            ).build(tokenizer)
            _validate_live_validation_dataset(dataset, validation_manifest)
        loader = MultimodalDataLoader(
            dataset,
            collator,
            work_dir=trainer.work_dir / f"{source_name}_validation",
            global_batch_size=global_instances * config.data.sequence_length,
            seed=eval_config.seed,
            shuffle=False,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
        )
        evaluators.append(
            MultimodalLMEvaluator(
                name=f"vision-alignment-{source_name}-validation",
                batches=loader,
                device=trainer.device,
                process_group=trainer.dp_process_group,
                deterministic=True,
            )
        )
        if source_name in ("pixmo_caption", "pixmo_transcript"):
            evaluators.append(
                MultimodalBlankImageEvaluator(
                    name=f"vision-alignment-{source_name}-blank-image",
                    batches=loader,
                    device=trainer.device,
                    process_group=trainer.dp_process_group,
                    deterministic=True,
                )
            )
    if config.phase is VisionAlignmentPhase.joint:
        holdout_config = config.evaluation.native_text_holdout
        assert holdout_config is not None
        holdout_dataset = holdout_config.build(tokenizer)
        holdout_loader = MultimodalDataLoader(
            holdout_dataset,
            collator,
            work_dir=trainer.work_dir / "native_text_holdout",
            global_batch_size=global_instances * config.data.sequence_length,
            seed=eval_config.seed,
            # Replay manifests are grouped by source and source file. A deterministic
            # permutation is required when the evaluator intentionally consumes only a
            # fixed prefix of the holdout; otherwise that prefix is not representative of
            # the parent pretraining mixture.
            shuffle=True,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
        )
        evaluators.append(
            MultimodalLMEvaluator(
                name="vision-alignment-native-text-holdout",
                batches=holdout_loader,
                device=trainer.device,
                process_group=trainer.dp_process_group,
                deterministic=True,
            )
        )
    expected_evaluators = {
        VisionAlignmentPhase.bridge: 4,
        VisionAlignmentPhase.perception: 10,
        VisionAlignmentPhase.joint: 11,
    }[config.phase]
    if len(evaluators) != expected_evaluators:
        raise ValueError(
            f"Phase {config.phase.value} requires exactly {expected_evaluators} intrinsic "
            f"evaluators, got {len(evaluators)}"
        )
    trainer.add_callback(
        "vision_alignment_intrinsic_validation",
        EvaluatorCallback(
            evaluators=evaluators,
            eval_interval=eval_config.interval,
            eval_duration=Duration.steps(eval_batches),
            eval_on_startup=eval_config.eval_on_startup,
            eval_on_finish=eval_config.eval_on_finish,
            log_interval=max(eval_batches // 4, 1),
        ),
    )


def _validate_runtime_trainability(train_module, config: ExperimentConfig) -> None:
    model = train_module.multimodal_model
    patterns = config.train_module.freeze_params or []
    parameter_names = [name for name, _ in model.named_parameters()]
    unmatched_patterns = [
        pattern
        for pattern in patterns
        if not any(fnmatch.fnmatch(name, pattern) for name in parameter_names)
    ]
    if unmatched_patterns:
        raise RuntimeError(
            "Vision-alignment freeze patterns did not match model parameters: "
            f"{unmatched_patterns}"
        )
    mismatches = []
    for name, parameter in model.named_parameters():
        expected_trainable = not any(fnmatch.fnmatch(name, pattern) for pattern in patterns)
        if parameter.requires_grad != expected_trainable:
            mismatches.append((name, parameter.requires_grad, expected_trainable))
    if mismatches:
        raise RuntimeError(
            f"Vision-alignment trainable-parameter contract mismatch: {mismatches[:8]}"
        )
    override_patterns = [
        pattern
        for override in (config.train_module.optim.group_overrides or [])
        for pattern in override.params
    ]
    fallback_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not any(fnmatch.fnmatch(name, pattern) for pattern in override_patterns)
    ]
    if config.phase is not VisionAlignmentPhase.joint:
        if fallback_names:
            raise RuntimeError(
                "Frozen-LM vision-alignment phases may not optimize fallback/default-group "
                f"parameters: {fallback_names[:8]}"
            )
    elif not fallback_names or any(not name.startswith("lm.") for name in fallback_names):
        raise RuntimeError(
            "Joint vision alignment requires a non-empty LM-only default optimizer group; "
            f"got {fallback_names[:8]}"
        )
    if tuple(sorted(train_module.train_embedding_rows)) != tuple(
        sorted(config.train_module.train_embedding_rows or [])
    ):
        raise RuntimeError("Runtime image-row mask differs from the phase config")


def _initialize_fresh_bridge(train_module, config: ExperimentConfig, token_ids) -> None:
    import torch.distributed as dist

    artifacts = config.artifacts
    state_dir = _checkpoint_state_dir(artifacts.base_checkpoint)
    log.info("Loading bare s002 language-model weights from %s", state_dir)
    train_module.load_state_dict_direct(
        state_dir,
        process_group=dist.group.WORLD,
        thread_count=config.checkpoint_load_threads,
        load_optim_state=False,
    )

    from olmo_core.nn.vision import (
        load_siglip_hf_vision_state_dict,
        siglip_hf_state_dict_to_vision,
        vision_state_fingerprint,
    )

    hf_state = load_siglip_hf_vision_state_dict(
        artifacts.vision_model_id,
        revision=artifacts.vision_revision,
        cache_dir=artifacts.hf_cache_dir,
    )
    vision_state = siglip_hf_state_dict_to_vision(
        hf_state, train_module.multimodal_model.cfg.vision
    )
    fingerprint = vision_state_fingerprint(vision_state)
    if fingerprint != artifacts.vision_fingerprint:
        raise ValueError(
            f"SigLIP fingerprint mismatch: expected {artifacts.vision_fingerprint}, got {fingerprint}"
        )
    train_module.load_vision_state_dict(vision_state)
    train_module.assert_vision_optimizer_state_synced()
    del hf_state, vision_state
    train_module.reset_image_token_rows(
        _image_token_ids(token_ids), seed=config.init_seed, reset_output_rows=False
    )


def train(config: ExperimentConfig) -> None:
    """Run one validated vision-alignment phase under torchrun."""
    os.environ.setdefault("OLMO_USE_OWN_SYMM_MEM", "1")
    os.environ.setdefault("OLMO_EP_MP_HIGH_PRIORITY_GROUP", "1")
    os.environ.setdefault("OLMO_OWN_SYMM_PREWARM", "1")
    seed_all(config.init_seed)

    tokenizer, token_ids = _load_tokenizer(config.artifacts)
    if _image_token_ids(token_ids) != config.train_module.train_embedding_rows:
        raise ValueError("Tokenizer image-token IDs differ from the pinned phase row mask")
    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    _validate_runtime_trainability(train_module, config)

    existing = _latest_output_checkpoint(config)
    if existing is None and config.phase is VisionAlignmentPhase.bridge:
        _initialize_fresh_bridge(train_module, config, token_ids)
    elif existing is not None:
        log.info(
            "Same-phase checkpoint %s exists; deferring exact full-state load to Trainer", existing
        )
    else:
        log.info(
            "Fresh %s phase will model-only load parent %s in Trainer.fit",
            config.phase.value,
            config.initialization.checkpoint,
        )

    collator = config.collator.build()
    dp_group = train_module.dp_process_group
    dp_world_size, dp_rank = get_world_size(dp_group), get_rank(dp_group)
    datasets, weights, names = _build_mixture_sources(tokenizer, token_ids, config)
    data_loader: DataLoaderBase = MixtureDataLoader(
        datasets,
        weights,
        collator,
        work_dir=config.trainer.save_folder,
        global_batch_size=config.global_batch_size,
        seed=config.data_seed,
        pack=config.data.pack_sequences,
        pack_max_crops=(config.data.pack_max_crops if config.data.pack_sequences else None),
        pack_buffer_size=(config.data.pack_buffer_size if config.data.pack_sequences else 0),
        prefetch_workers=config.data.prefetch_workers,
        dataset_names=names,
        allow_legacy_state_without_dataset_fingerprints=False,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
    )

    # Dataset preparation can have large rank-local skew on cold hosts. Synchronize on the
    # 60-minute main process group before Trainer creates shorter-lived bookkeeping groups.
    import torch.distributed as dist

    dist.barrier()
    log.info("All ranks finished vision-alignment dataset setup")
    trainer = config.trainer.build(train_module, data_loader)
    _add_intrinsic_visual_evaluators(
        trainer,
        tokenizer,
        config,
        collator,
        token_ids,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
    )
    config_dict = config.as_config_dict()
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict
    cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict
    trainer.fit()


def launch(config: ExperimentConfig) -> None:
    """Submit the already validated recipe to the Holmes cluster."""
    if not config.launch.workspace or config.launch.clusters != [BEAKER_CLUSTER]:
        raise RuntimeError("Vision alignment launch requires the approved Holmes workspace")
    if config.launch.hostnames:
        raise RuntimeError("Exact node selection is forbidden; request the Holmes cluster only")
    config.launch.launch(follow=True)


def main() -> None:
    """Run the vision-alignment recipe command-line interface."""
    usage = f"""
Usage
=====

python {sys.argv[0]} [dry_run|launch|train] RUN_NAME --phase=PHASE [OVERRIDES...]
python {sys.argv[0]} dry_run RUN_NAME --profile=PATH

PHASE is one of bridge, perception, or joint. A checked-in profile owns the phase selector,
so do not also pass --phase when --profile is used.
""".strip()
    if len(sys.argv) < 3:
        print(usage)
        sys.exit(1)

    script, command, run_name, *raw_overrides = sys.argv
    if command not in {"dry_run", "launch", "train"}:
        print(usage)
        sys.exit(1)
    if command == "train":
        prepare_training_environment(timeout=timedelta(minutes=60))
    else:
        prepare_cli_environment()

    profile, overrides = _load_profile(raw_overrides)
    experiment = build_config(script, run_name, overrides)
    experiment = _apply_profile_launch(experiment, profile, run_name=run_name)
    _validate_phase_contract(experiment, run_name)
    log.info("%s", experiment)

    if command == "train":
        train(experiment)
        teardown_training_environment()
    elif command == "launch":
        launch(experiment)


if __name__ == "__main__":
    main()
