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

The bridge phase is executable today.  The production perception and joint defaults include
source contracts whose audited adapters/manifests are still required; training fails closed
when one is absent instead of silently substituting SFT/QA data.

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
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, cast

import yaml

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
from olmo_core.data.multimodal.paths import PIXMO_DATASETS
from olmo_core.data.multimodal.vision_alignment_sources import (
    VISION_ALIGNMENT_FORMATTER_VERSION,
    VISION_ALIGNMENT_PIXMO_ROW_PATH_INVENTORY_ALGORITHM,
    VISION_ALIGNMENT_PROBE_FORMAT,
    VISION_ALIGNMENT_PROBE_SELECTION_ALGORITHM,
    VISION_ALIGNMENT_PROBE_VERSION,
    VISION_ALIGNMENT_RECIPE_VERSION,
    VISION_ALIGNMENT_SOURCE_CATALOG_VERSION,
    VISION_ALIGNMENT_SOURCE_REGISTRY_VERSION,
    VISION_ALIGNMENT_TOKENIZER_FINGERPRINT,
    VISION_ALIGNMENT_TOKENIZER_ID,
    VISION_ALIGNMENT_TOKENIZER_REVISION,
    VisionAlignmentSourceSpec,
    build_vision_alignment_dataset,
    build_vision_alignment_dataset_config,
    load_pinned_vision_alignment_tokenizer,
    pixmo_row_path_inventory,
    runtime_dataset_fingerprint,
    select_deterministic_probe_indices,
    validate_serialized_runtime_probe,
    vision_alignment_source_registry_sha256,
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
MIN_SOURCE_AUDIT_EXAMPLES = 1024

_SOURCE_AUDIT_FIELDS = frozenset(
    {
        "format",
        "version",
        "auditor_sha256",
        "status",
        "phase",
        "recipe_version",
        "formatter_version",
        "source_catalog_version",
        "source_registry_version",
        "source_registry_sha256",
        "exporter_sha256",
        "image_manifest_sha256",
        "preprocessing_config",
        "preprocessing_config_sha256",
        "probe",
        "catalog_sha256",
        "input_content_sha256",
        "inputs",
        "target_loss_mass",
        "sources",
        "mean_loss_weight",
        "sampling_probabilities",
        "expected_loss_mass",
        "failures",
        "fingerprint",
    }
)
_SOURCE_AUDIT_INPUT_FIELDS = frozenset(
    {
        "format",
        "path",
        "sha256",
        "dataset_fingerprint",
        "dataset_size",
        "probe_indices",
        "probe_indices_sha256",
        "serialized_row_hashes",
        "serialized_row_hashes_sha256",
    }
)
_SOURCE_AUDIT_PROBE_FIELDS = frozenset(
    {"format", "version", "selection_algorithm", "seed", "epoch", "examples_per_source"}
)

_PIXMO_MANIFEST_FIELDS = frozenset(
    {"format", "version", "builder", "source", "output", "inventories", "filtering"}
)
_PIXMO_BUILDER_FIELDS = frozenset(
    {
        "format",
        "version",
        "script",
        "script_sha256",
        "filter_algorithm",
        "image_hash_algorithm",
        "row_image_paths_algorithm",
        "row_image_content_algorithm",
    }
)
_PIXMO_DATASET_FIELDS = frozenset({"dataset_path", "splits"})
_PIXMO_SOURCE_SPLIT_FIELDS = frozenset(
    {
        "dataset_fingerprint",
        "examples",
        "row_image_paths_sha256",
        "row_image_content_sha256",
        "unique_image_paths",
        "unique_image_content",
    }
)
_PIXMO_OUTPUT_SPLIT_FIELDS = _PIXMO_SOURCE_SPLIT_FIELDS | {"row_image_content_path"}
_PIXMO_INVENTORY_FIELDS = frozenset({"path", "sha256", "count"})
_PIXMO_FILTERING_FIELDS = frozenset(
    {
        "source_overlap_unique_images",
        "removed_train_examples",
        "validation_duplicate_examples",
        "output_overlap_unique_images",
    }
)
_PIXMO_SPLITS = frozenset({"train", "validation"})
_PIXMO_BUILDER_SCRIPT = "src/scripts/data/build_vision_alignment_pixmo_cap.py"
_PIXMO_FILTER_ALGORITHM = "preserve-validation-drop-train-content-overlap-v1"
_PIXMO_ROW_CONTENT_ALGORITHM = "sha256-lines-v1"
_CANONICAL_PIXMO_SOURCE_DATASET = (
    "/weka/oe-training-default/mm-olmo/torch_datasets/pixmo_datasets/cap"
)
_CANONICAL_PIXMO_SOURCE_SPLITS: Mapping[str, Tuple[str, int]] = {
    "train": ("db8d55b1f2bbb62e", 714_985),
    "validation": ("502dc5bb570bab20", 2_048),
}

_RUN_NAME_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,127}")
_ALLOWED_OVERRIDE_PREFIXES = (
    "--phase=",
    "--data.",
    "--evaluation.",
    "--initialization.checkpoint=",
    "--initialization.parent_config_sha256=",
    "--initialization.parent_gate_path=",
    "--initialization.parent_gate_sha256=",
    "--trainer.max_duration.value=",
    "--trainer.callbacks.checkpointer.save_interval=",
    "--trainer.callbacks.checkpointer.ephemeral_save_interval=",
    "--trainer.callbacks.checkpointer.fixed_steps=",
    "--trainer.callbacks.checkpointer.max_checkpoints=",
    "--train_module.vision_activation_checkpointing=",
    "--train_module.connector_activation_checkpointing=",
    "--global_batch_size=",
    "--data_seed=",
    "--init_seed=",
    "--checkpoint_load_threads=",
    "--router_lb_loss_weight=",
)


class VisionAlignmentPhase(StrEnum):
    """The three optimizer/data phases of vision alignment."""

    bridge = "bridge"
    perception = "perception"
    joint = "joint"


class InitializationMode(StrEnum):
    """How a fresh vision-alignment phase obtains its model weights."""

    bare = "bare"
    checkpoint = "checkpoint"


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
    parent_gate_path: Optional[str] = None
    parent_gate_sha256: Optional[str] = None


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
    parent_gate_sha256: Optional[str] = None
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
    data_seed: int = DATA_SEED
    init_seed: int = INIT_SEED
    checkpoint_load_threads: int = 8
    router_lb_loss_weight: Optional[float] = 0.015
    required_run_name: str = ""
    expected_launch_command: List[str] = field(default_factory=list)


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


def _validate_override_surface(overrides: Sequence[str]) -> None:
    forbidden = [
        value
        for value in overrides
        if not any(value.startswith(prefix) for prefix in _ALLOWED_OVERRIDE_PREFIXES)
    ]
    if forbidden:
        raise ValueError(
            "Vision alignment accepts only the audited override surface; rejected "
            f"{forbidden}. Change the checked-in recipe for structural modifications."
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
                "connector": CosWithWarmup(warmup=policy.connector_warmup, alpha_f=0.1, t_max=None),
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


def _load_profile(overrides: List[str]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    prefix = "--profile="
    paths = [value[len(prefix) :] for value in overrides if value.startswith(prefix)]
    if len(paths) > 1:
        raise ValueError("At most one --profile may be supplied")
    if not paths:
        return None, overrides
    profile_path = Path(paths[0])
    with profile_path.open() as file_handle:
        profile = yaml.safe_load(file_handle)
    if not isinstance(profile, dict) or profile.get("version") != 1:
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
    cli = [value for value in overrides if not value.startswith(prefix)]
    if any(value.startswith("--phase=") for value in [*profile_overrides, *cli]):
        raise ValueError("Set phase in the profile or on the CLI, not both")
    return profile, [f"--phase={phase}", *profile_overrides, *cli]


def _apply_profile_launch(
    config: ExperimentConfig, profile: Optional[Dict[str, Any]]
) -> ExperimentConfig:
    if profile is None:
        return config
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


def _validate_parent_gate(
    config: ExperimentConfig,
    parent: str,
    parent_config: Mapping[str, Any],
    parent_config_sha256: str,
) -> str:
    """Validate an explicit evaluation approval for a cross-phase parent checkpoint."""
    parent_data = parent_config.get("data")
    if isinstance(parent_data, Mapping) and parent_data.get("allow_unpinned_synthetic_smoke"):
        raise ValueError("A synthetic-smoke checkpoint may not parent a production phase")
    gate_path_value = config.initialization.parent_gate_path
    expected_gate_sha = config.initialization.parent_gate_sha256
    if gate_path_value is None or expected_gate_sha is None:
        raise ValueError("A phase transition requires a pinned approved parent-quality gate")
    gate_path = Path(gate_path_value).expanduser().resolve()
    try:
        raw = gate_path.read_bytes()
        gate = json.loads(raw, object_pairs_hook=_strict_json_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Invalid parent-quality gate {gate_path}: {error}") from error
    actual_gate_sha = hashlib.sha256(raw).hexdigest()
    if expected_gate_sha != actual_gate_sha:
        raise ValueError(
            f"Parent-quality gate SHA mismatch: configured {expected_gate_sha}, "
            f"actual {actual_gate_sha}"
        )
    if not isinstance(gate, dict):
        raise ValueError(f"Parent-quality gate must be an object: {gate_path}")
    allowed_fields = {
        "format",
        "version",
        "status",
        "recipe_version",
        "formatter_version",
        "phase",
        "checkpoint",
        "checkpoint_config_sha256",
        "data_contract_sha256",
        "trainable_contract_sha256",
        "global_step",
        "metrics_artifact_sha256",
    }
    if set(gate) != allowed_fields:
        raise ValueError(
            "Parent-quality gate fields differ from the locked schema: "
            f"missing={sorted(allowed_fields - set(gate))}, "
            f"extra={sorted(set(gate) - allowed_fields)}"
        )
    parent_meta = parent_config.get("vision_alignment")
    expected_parent_phase = config.initialization.expected_parent_phase
    assert isinstance(parent_meta, Mapping)
    if (
        gate["format"] != "vision_alignment_parent_gate"
        or gate["version"] != 1
        or gate["status"] != "approved"
        or gate["recipe_version"] != RECIPE_VERSION
        or gate["formatter_version"] != FORMATTER_VERSION
        or expected_parent_phase is None
        or gate["phase"] != expected_parent_phase.value
        or Path(str(gate["checkpoint"])).expanduser().resolve() != Path(parent).resolve()
        or gate["checkpoint_config_sha256"] != parent_config_sha256
        or gate["data_contract_sha256"] != parent_meta.get("data_contract_sha256")
        or gate["trainable_contract_sha256"] != parent_meta.get("trainable_contract_sha256")
    ):
        raise ValueError("Parent-quality gate is incompatible with the selected parent checkpoint")
    global_step = gate["global_step"]
    if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step <= 0:
        raise ValueError("Parent-quality gate global_step must be a positive integer")
    step_match = re.fullmatch(r"step(\d+)", Path(parent).name)
    if step_match is None or int(step_match.group(1)) != global_step:
        raise ValueError("Parent-quality gate global_step must match the checkpoint directory")
    if re.fullmatch(r"[0-9a-f]{64}", str(gate["metrics_artifact_sha256"])) is None:
        raise ValueError("Parent-quality gate must pin its evaluation artifact SHA-256")
    marker_path = Path(parent) / ".metadata.json"
    try:
        marker = json.loads(marker_path.read_bytes(), object_pairs_hook=_strict_json_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Invalid parent checkpoint marker {marker_path}: {error}") from error
    if not isinstance(marker, Mapping) or marker.get("ephemeral") is not False:
        raise ValueError("A phase transition requires a permanent parent checkpoint")
    return actual_gate_sha


def _parent_is_inside_output(parent: str, output: str) -> bool:
    parent_path = Path(parent).resolve()
    output_path = Path(output).resolve()
    return (
        parent_path == output_path
        or output_path in parent_path.parents
        or parent_path in output_path.parents
    )


class _AuditedDataset:
    """Attach a pinned serialized-source audit identity to a map-style dataset."""

    content_fingerprint_version = "vision-alignment-source-audit-v2"

    def __init__(self, dataset: Any, source_name: str, audit: Mapping[str, Any]):
        self._dataset = dataset
        self.source_name = source_name
        source = cast(Mapping[str, Any], cast(Mapping[str, Any], audit["inputs"])[source_name])
        runtime_fingerprint = _runtime_dataset_fingerprint(dataset)
        if runtime_fingerprint != source.get("dataset_fingerprint"):
            raise ValueError(
                f"Live dataset fingerprint for {source_name!r} is {runtime_fingerprint!r}, "
                f"but its pinned audit records {source.get('dataset_fingerprint')!r}"
            )
        if len(dataset) != source.get("dataset_size"):
            raise ValueError(
                f"Live dataset length for {source_name!r} is {len(dataset)}, but its pinned "
                f"audit identifies {source.get('dataset_size')!r} examples"
            )
        probe_indices = cast(Sequence[int], source["probe_indices"])
        row_hashes = cast(Sequence[str], source["serialized_row_hashes"])
        validate_serialized_runtime_probe(dataset, probe_indices, row_hashes, epoch=0)
        self.content_fingerprint = _canonical_sha256(
            {
                "audit_fingerprint": audit["fingerprint"],
                "source_registry_sha256": audit["source_registry_sha256"],
                "exporter_sha256": audit["exporter_sha256"],
                "input_content_sha256": audit["input_content_sha256"],
                "source": source_name,
                "source_sha256": source["sha256"],
                "probe_indices_sha256": source["probe_indices_sha256"],
                "serialized_row_hashes_sha256": source["serialized_row_hashes_sha256"],
                "runtime_dataset_fingerprint": runtime_fingerprint,
                "runtime_dataset_length": len(dataset),
            }
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.get(index, 0)

    def get(self, index: int, epoch: int = 0) -> Dict[str, Any]:
        get = getattr(self._dataset, "get", None)
        return get(index, epoch) if get is not None else self._dataset[index]


def _source_spec(config: ExperimentConfig) -> VisionAlignmentSourceSpec:
    """Return the exact source specification shared with the canonical exporter."""
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
    return _source_spec(config).preprocessing_sha256


def _validated_source_audit(config: ExperimentConfig) -> Optional[Mapping[str, Any]]:
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
        raise ValueError(
            "Real vision alignment requires a pinned successful serialized-source audit"
        )
    audit_path = Path(data.source_audit_path).expanduser().resolve()
    try:
        audit = json.loads(audit_path.read_bytes(), object_pairs_hook=_strict_json_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Invalid vision-alignment source audit {audit_path}: {error}") from error
    if not isinstance(audit, dict):
        raise ValueError(f"Vision-alignment source audit must be an object: {audit_path}")
    if set(audit) != _SOURCE_AUDIT_FIELDS:
        raise ValueError("Vision-alignment source audit fields differ from the locked schema")
    recorded_fingerprint = audit.get("fingerprint")
    unsigned_audit = dict(audit)
    unsigned_audit.pop("fingerprint", None)
    computed_fingerprint = _canonical_sha256(unsigned_audit)
    if (
        recorded_fingerprint != computed_fingerprint
        or data.source_audit_fingerprint != computed_fingerprint
    ):
        raise ValueError(
            "Vision-alignment source audit fingerprint mismatch: "
            f"recorded={recorded_fingerprint!r}, configured={data.source_audit_fingerprint!r}, "
            f"computed={computed_fingerprint!r}"
        )
    if (
        audit.get("format") != "vision_alignment_source_audit"
        or audit.get("version") != 2
        or audit.get("status") != "ok"
        or audit.get("recipe_version") != RECIPE_VERSION
        or audit.get("formatter_version") != FORMATTER_VERSION
        or audit.get("source_catalog_version") != VISION_ALIGNMENT_SOURCE_CATALOG_VERSION
        or audit.get("source_registry_version") != VISION_ALIGNMENT_SOURCE_REGISTRY_VERSION
        or audit.get("source_registry_sha256") != vision_alignment_source_registry_sha256()
        or audit.get("phase") != config.phase.value
    ):
        raise ValueError("Vision-alignment source audit identity or status is incompatible")
    auditor_path = Path(__file__).resolve().parents[1] / "data" / "audit_vision_alignment_mix.py"
    if audit.get("auditor_sha256") != _sha256_file(auditor_path):
        raise ValueError("Vision-alignment source audit was not produced by this pinned auditor")
    exporter_path = (
        Path(__file__).resolve().parents[1] / "data" / "export_vision_alignment_probe.py"
    )
    if audit.get("exporter_sha256") != _sha256_file(exporter_path):
        raise ValueError("Vision-alignment source audit was not produced by this pinned exporter")
    for field_name in (
        "image_manifest_sha256",
        "input_content_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", str(audit.get(field_name, ""))) is None:
            raise ValueError(f"Vision-alignment source audit has invalid {field_name}")
    expected_preprocessing_sha = _preprocessing_config_sha256(config)
    expected_preprocessing_config = _source_spec(config).as_canonical_dict()
    if (
        audit.get("preprocessing_config_sha256") != expected_preprocessing_sha
        or not isinstance(audit.get("preprocessing_config"), Mapping)
        or _canonical_sha256(audit["preprocessing_config"])
        != _canonical_sha256(expected_preprocessing_config)
    ):
        raise ValueError(
            "Vision-alignment source audit preprocessing config differs from training: "
            f"expected {expected_preprocessing_sha}, got "
            f"{audit.get('preprocessing_config_sha256')!r}"
        )
    probe = audit.get("probe")
    if not isinstance(probe, Mapping) or set(probe) != _SOURCE_AUDIT_PROBE_FIELDS:
        raise ValueError("Vision-alignment source audit probe fields differ from the schema")
    probe_version = probe.get("version")
    probe_epoch = probe.get("epoch")
    probe_seed = probe.get("seed")
    examples_per_source = probe.get("examples_per_source")
    if (
        probe.get("format") != VISION_ALIGNMENT_PROBE_FORMAT
        or isinstance(probe_version, bool)
        or not isinstance(probe_version, int)
        or probe_version != VISION_ALIGNMENT_PROBE_VERSION
        or probe.get("selection_algorithm") != VISION_ALIGNMENT_PROBE_SELECTION_ALGORITHM
        or isinstance(probe_epoch, bool)
        or not isinstance(probe_epoch, int)
        or probe_epoch != 0
        or isinstance(probe_seed, bool)
        or not isinstance(probe_seed, int)
        or probe_seed < 0
        or isinstance(examples_per_source, bool)
        or not isinstance(examples_per_source, int)
        or examples_per_source < MIN_SOURCE_AUDIT_EXAMPLES
    ):
        raise ValueError("Vision-alignment source audit probe identity or size is incompatible")
    targets = config.data.mixture.resolved_targets()
    means = config.data.mixture.mean_loss_weight
    sampling = config.data.mixture.sampling_weights()
    for field_name, expected in (
        ("target_loss_mass", targets),
        ("mean_loss_weight", means),
        ("sampling_probabilities", sampling),
    ):
        actual = audit.get(field_name)
        if not isinstance(actual, Mapping) or _canonical_sha256(actual) != _canonical_sha256(
            expected
        ):
            raise ValueError(
                f"Vision-alignment source audit {field_name} differs from the training config"
            )
    inputs = audit.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != set(targets):
        raise ValueError("Vision-alignment source audit inputs differ from mixture sources")
    for source_name, source in inputs.items():
        if (
            not isinstance(source, Mapping)
            or set(source) != _SOURCE_AUDIT_INPUT_FIELDS
            or source.get("format") != "jsonl"
            or not isinstance(source.get("path"), str)
            or not source.get("path")
            or re.fullmatch(r"[0-9a-f]{64}", str(source.get("sha256", ""))) is None
            or not isinstance(source.get("dataset_fingerprint"), str)
            or not source.get("dataset_fingerprint")
            or isinstance(source.get("dataset_size"), bool)
            or not isinstance(source.get("dataset_size"), int)
            or source["dataset_size"] < 1
        ):
            raise ValueError(
                f"Vision-alignment audit source {source_name!r} lacks content/runtime identity"
            )
        probe_indices = source.get("probe_indices")
        row_hashes = source.get("serialized_row_hashes")
        assert isinstance(examples_per_source, int)
        if (
            not isinstance(probe_indices, list)
            or len(probe_indices) != examples_per_source
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= source["dataset_size"]
                for index in probe_indices
            )
            or len(set(probe_indices)) != len(probe_indices)
            or not isinstance(row_hashes, list)
            or len(row_hashes) != examples_per_source
            or any(re.fullmatch(r"[0-9a-f]{64}", str(value)) is None for value in row_hashes)
            or _canonical_sha256(probe_indices) != source.get("probe_indices_sha256")
            or _canonical_sha256(row_hashes) != source.get("serialized_row_hashes_sha256")
        ):
            raise ValueError(
                f"Vision-alignment audit source {source_name!r} has an invalid runtime probe"
            )
        expected_indices = select_deterministic_probe_indices(
            source["dataset_size"],
            examples_per_source,
            seed=probe_seed,
            dataset_fingerprint=source["dataset_fingerprint"],
        )
        if tuple(probe_indices) != expected_indices:
            raise ValueError(
                f"Vision-alignment audit source {source_name!r} uses non-canonical probe indices"
            )
    source_reports = audit.get("sources")
    if not isinstance(source_reports, Mapping) or set(source_reports) != set(targets):
        raise ValueError("Vision-alignment source audit summaries differ from mixture sources")
    if audit.get("failures") != []:
        raise ValueError("Vision-alignment source audit records preprocessing failures")
    expected_loss_mass = audit.get("expected_loss_mass")
    if not isinstance(expected_loss_mass, Mapping) or _canonical_sha256(
        expected_loss_mass
    ) != _canonical_sha256(targets):
        raise ValueError("Vision-alignment source audit expected loss mass differs from targets")
    for source_name, source_report in source_reports.items():
        if not isinstance(source_report, Mapping):
            raise ValueError(f"Vision-alignment audit summary for {source_name!r} is invalid")
        examples = source_report.get("examples")
        mean_loss_weight = source_report.get("mean_sum_loss_masks")
        if (
            not isinstance(examples, Mapping)
            or set(examples) != {"seen", "valid", "errors"}
            or isinstance(examples.get("valid"), bool)
            or not isinstance(examples.get("valid"), int)
            or examples["valid"] != examples_per_source
            or examples.get("seen") != examples["valid"]
            or examples.get("errors") != 0
            or source_report.get("zero_loss_examples") != 0
            or source_report.get("error_samples") != []
            or isinstance(mean_loss_weight, bool)
            or not isinstance(mean_loss_weight, (int, float))
            or not math.isfinite(float(mean_loss_weight))
            or float(mean_loss_weight) != float(means[source_name])
        ):
            raise ValueError(
                f"Vision-alignment audit summary for {source_name!r} is incomplete, too "
                "small, or inconsistent with its calibrated loss weight"
            )
    return audit


def _validate_validation_manifest(
    config: ExperimentConfig,
    source_audit: Optional[Mapping[str, Any]],
    *,
    validate_live_datasets: bool = True,
) -> Optional[Mapping[str, Any]]:
    """Require the pinned canonical PixMoCap builder artifact for production runs.

    Version 3 is an output of the checked-in split builder, not a caller-authored claim. It
    binds the filtered train and preserved validation Arrow splits to builder-time byte hashes,
    then cheaply recomputes their ordered image-path inventories at runtime.
    """
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
        raise ValueError("Production vision alignment requires a pinned validation manifest")
    path = Path(path_value).expanduser().resolve()
    try:
        raw = path.read_bytes()
        manifest = json.loads(raw, object_pairs_hook=_strict_json_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Invalid vision-alignment validation manifest {path}: {error}") from error
    actual_sha = hashlib.sha256(raw).hexdigest()
    if expected_sha != actual_sha:
        raise ValueError(
            f"Validation-manifest SHA mismatch: configured {expected_sha}, actual {actual_sha}"
        )
    if not isinstance(manifest, Mapping):
        raise ValueError(f"Validation manifest must be an object: {path}")
    if set(manifest) != _PIXMO_MANIFEST_FIELDS:
        raise ValueError("Validation manifest fields differ from the locked schema")
    if manifest["format"] != "vision_alignment_validation_manifest" or manifest["version"] != 3:
        raise ValueError("Validation manifest identity is incompatible")

    def exact_mapping(value: Any, fields: frozenset[str], name: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError(f"Validation manifest {name} fields differ from the locked schema")
        return value

    def resolve_artifact_path(
        path_value: Any, name: str, *, within_manifest_root: bool = False
    ) -> Path:
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"Validation manifest {name} path must be non-empty")
        artifact_path = Path(path_value).expanduser()
        if within_manifest_root and artifact_path.is_absolute():
            raise ValueError(f"Validation manifest {name} must be relative to the artifact root")
        if not artifact_path.is_absolute():
            artifact_path = path.parent / artifact_path
        artifact_path = artifact_path.resolve()
        if within_manifest_root and not artifact_path.is_relative_to(path.parent):
            raise ValueError(f"Validation manifest {name} escapes the artifact root")
        return artifact_path

    def require_sha256(value: Any, name: str) -> str:
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"Validation manifest {name} must be a lowercase SHA-256")
        return value

    def require_count(value: Any, name: str, *, positive: bool = False) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < (1 if positive else 0):
            qualifier = "positive" if positive else "non-negative"
            raise ValueError(f"Validation manifest {name} must be a {qualifier} integer")
        return value

    def load_hash_rows(
        path_value: Any,
        expected_digest: Any,
        expected_count: Any,
        name: str,
        *,
        unique: bool,
    ) -> Tuple[List[str], Path]:
        hash_path = resolve_artifact_path(path_value, name, within_manifest_root=True)
        try:
            raw_hashes = hash_path.read_bytes()
        except OSError as error:
            raise ValueError(f"Could not read image-hash manifest {hash_path}: {error}") from error
        actual_digest = hashlib.sha256(raw_hashes).hexdigest()
        if require_sha256(expected_digest, f"{name}.sha256") != actual_digest:
            raise ValueError(
                f"Image-hash manifest SHA mismatch for {hash_path}: "
                f"configured {expected_digest!r}, actual {actual_digest}"
            )
        try:
            hashes = raw_hashes.decode("utf-8").splitlines()
        except UnicodeDecodeError as error:
            raise ValueError(f"Image-hash manifest {hash_path} is not UTF-8") from error
        if (
            not hashes
            or not raw_hashes.endswith(b"\n")
            or any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in hashes)
        ):
            raise ValueError(f"Image-hash manifest {hash_path} contains invalid SHA-256 rows")
        if unique and hashes != sorted(set(hashes)):
            raise ValueError(
                f"Image-hash manifest {hash_path} must contain sorted unique SHA-256 rows"
            )
        if require_count(expected_count, f"{name}.count", positive=True) != len(hashes):
            raise ValueError(f"Image-hash manifest count mismatch for {hash_path}")
        return hashes, hash_path

    builder = exact_mapping(manifest["builder"], _PIXMO_BUILDER_FIELDS, "builder")
    if (
        builder["format"] != "vision_alignment_pixmo_cap_builder"
        or builder["version"] != 1
        or builder["script"] != _PIXMO_BUILDER_SCRIPT
        or builder["filter_algorithm"] != _PIXMO_FILTER_ALGORITHM
        or builder["image_hash_algorithm"] != "sha256"
        or builder["row_image_paths_algorithm"]
        != VISION_ALIGNMENT_PIXMO_ROW_PATH_INVENTORY_ALGORITHM
        or builder["row_image_content_algorithm"] != _PIXMO_ROW_CONTENT_ALGORITHM
    ):
        raise ValueError("Validation manifest builder identity or algorithms are incompatible")
    builder_path = Path(__file__).resolve().parents[3] / _PIXMO_BUILDER_SCRIPT
    if builder.get("script_sha256") != _sha256_file(builder_path):
        raise ValueError("Validation manifest was not produced by this pinned PixMoCap builder")

    datasets: Dict[str, Mapping[str, Any]] = {}
    for dataset_kind, split_fields in (
        ("source", _PIXMO_SOURCE_SPLIT_FIELDS),
        ("output", _PIXMO_OUTPUT_SPLIT_FIELDS),
    ):
        dataset_entry = exact_mapping(manifest[dataset_kind], _PIXMO_DATASET_FIELDS, dataset_kind)
        splits = dataset_entry["splits"]
        if not isinstance(splits, Mapping) or set(splits) != _PIXMO_SPLITS:
            raise ValueError(
                f"Validation manifest {dataset_kind}.splits must be exactly train/validation"
            )
        for split in sorted(_PIXMO_SPLITS):
            split_entry = exact_mapping(
                splits[split], split_fields, f"{dataset_kind}.splits.{split}"
            )
            if (
                not isinstance(split_entry["dataset_fingerprint"], str)
                or not split_entry["dataset_fingerprint"]
            ):
                raise ValueError(
                    f"Validation manifest {dataset_kind}.{split} fingerprint is invalid"
                )
            require_count(
                split_entry["examples"], f"{dataset_kind}.{split}.examples", positive=True
            )
            require_sha256(
                split_entry["row_image_paths_sha256"],
                f"{dataset_kind}.{split}.row_image_paths_sha256",
            )
            require_sha256(
                split_entry["row_image_content_sha256"],
                f"{dataset_kind}.{split}.row_image_content_sha256",
            )
            require_count(
                split_entry["unique_image_paths"],
                f"{dataset_kind}.{split}.unique_image_paths",
                positive=True,
            )
            require_count(
                split_entry["unique_image_content"],
                f"{dataset_kind}.{split}.unique_image_content",
                positive=True,
            )
            if (
                split_entry["unique_image_paths"] > split_entry["examples"]
                or split_entry["unique_image_content"] > split_entry["examples"]
            ):
                raise ValueError(
                    f"Validation manifest {dataset_kind}.{split} unique counts exceed rows"
                )
        datasets[dataset_kind] = dataset_entry

    source_splits = cast(Mapping[str, Mapping[str, Any]], datasets["source"]["splits"])
    output_splits = cast(Mapping[str, Mapping[str, Any]], datasets["output"]["splits"])
    if datasets["output"]["dataset_path"] != "dataset":
        raise ValueError("Canonical PixMoCap output path in the manifest must be 'dataset'")
    output_path = resolve_artifact_path(
        datasets["output"]["dataset_path"],
        "output.dataset",
        within_manifest_root=True,
    )
    source_path = resolve_artifact_path(datasets["source"]["dataset_path"], "source.dataset")
    if output_path != Path(config.data.pixmo_cap_path).expanduser().resolve():
        raise ValueError("Configured PixMoCap path is not the manifest's canonical output")
    if output_path == source_path:
        raise ValueError("Canonical PixMoCap output must be separate from its source dataset")
    if source_path != Path(_CANONICAL_PIXMO_SOURCE_DATASET).resolve():
        raise ValueError(
            "Validation manifest was not built from the canonical PixMoCap source dataset"
        )
    if set(_CANONICAL_PIXMO_SOURCE_SPLITS) != _PIXMO_SPLITS:
        raise RuntimeError("Canonical PixMoCap source split policy is internally inconsistent")
    for split, (expected_fingerprint, expected_examples) in _CANONICAL_PIXMO_SOURCE_SPLITS.items():
        source_split = source_splits[split]
        if (
            source_split["dataset_fingerprint"] != expected_fingerprint
            or source_split["examples"] != expected_examples
        ):
            raise ValueError(
                f"Validation manifest {split} split does not match the canonical PixMoCap "
                "source fingerprint and row count"
            )

    inventories = manifest["inventories"]
    if not isinstance(inventories, Mapping) or set(inventories) != _PIXMO_SPLITS:
        raise ValueError("Validation manifest inventories must be exactly train/validation")
    inventory_hashes: Dict[str, set[str]] = {}
    for split in sorted(_PIXMO_SPLITS):
        inventory = exact_mapping(
            inventories[split], _PIXMO_INVENTORY_FIELDS, f"inventories.{split}"
        )
        rows, _ = load_hash_rows(
            inventory["path"],
            inventory["sha256"],
            inventory["count"],
            f"inventories.{split}",
            unique=True,
        )
        inventory_hashes[split] = set(rows)
        if len(rows) != output_splits[split]["unique_image_content"]:
            raise ValueError(
                f"Validation manifest {split} content inventory count differs from output"
            )
    if inventory_hashes["train"] & inventory_hashes["validation"]:
        raise ValueError("Training and validation image-hash manifests are not disjoint")

    for split in sorted(_PIXMO_SPLITS):
        output = output_splits[split]
        row_hashes, _ = load_hash_rows(
            output["row_image_content_path"],
            output["row_image_content_sha256"],
            output["examples"],
            f"output.splits.{split}.row_image_content",
            unique=False,
        )
        if set(row_hashes) != inventory_hashes[split]:
            raise ValueError(
                f"Validation manifest {split} row-content hashes differ from its inventory"
            )

    filtering = exact_mapping(manifest["filtering"], _PIXMO_FILTERING_FIELDS, "filtering")
    for field_name in sorted(_PIXMO_FILTERING_FIELDS):
        require_count(filtering[field_name], f"filtering.{field_name}")
    if (
        filtering["output_overlap_unique_images"] != 0
        or source_splits["train"]["examples"] - output_splits["train"]["examples"]
        != filtering["removed_train_examples"]
        or source_splits["validation"]["examples"] != output_splits["validation"]["examples"]
        or source_splits["validation"]["row_image_paths_sha256"]
        != output_splits["validation"]["row_image_paths_sha256"]
        or source_splits["validation"]["row_image_content_sha256"]
        != output_splits["validation"]["row_image_content_sha256"]
        or source_splits["validation"]["unique_image_paths"]
        != output_splits["validation"]["unique_image_paths"]
        or source_splits["validation"]["unique_image_content"]
        != output_splits["validation"]["unique_image_content"]
        or source_splits["train"]["unique_image_content"]
        - output_splits["train"]["unique_image_content"]
        != filtering["source_overlap_unique_images"]
        or filtering["removed_train_examples"] < filtering["source_overlap_unique_images"]
        or output_splits["validation"]["examples"]
        - output_splits["validation"]["unique_image_content"]
        != filtering["validation_duplicate_examples"]
        or output_splits["validation"]["examples"] < config.evaluation.examples_per_source
    ):
        raise ValueError("Validation manifest filtering or held-out split contract is invalid")

    train_inventory_sha = cast(Mapping[str, Any], inventories["train"])["sha256"]
    if train_inventory_sha != source_audit.get("image_manifest_sha256"):
        raise ValueError("Source audit was not calibrated against this canonical train inventory")
    audit_inputs = source_audit.get("inputs")
    if not isinstance(audit_inputs, Mapping):
        raise ValueError("Source audit lacks its runtime input identities")
    for source_name in ("pixmo_caption", "pixmo_transcript"):
        audit_input = audit_inputs.get(source_name)
        if (
            not isinstance(audit_input, Mapping)
            or audit_input.get("dataset_fingerprint")
            != output_splits["train"]["dataset_fingerprint"]
            or audit_input.get("dataset_size") != output_splits["train"]["examples"]
        ):
            raise ValueError(
                f"Source audit {source_name!r} is not bound to the canonical train split"
            )

    if validate_live_datasets:
        from olmo_core.data.multimodal.dataset_compat import load_from_disk_compat

        for dataset_kind, dataset_path, expected_splits in (
            ("source", source_path, source_splits),
            ("output", output_path, output_splits),
        ):
            try:
                live = load_from_disk_compat(str(dataset_path))
            except Exception as error:
                raise ValueError(
                    f"Could not load canonical PixMoCap {dataset_kind} {dataset_path}: {error}"
                ) from error
            if not hasattr(live, "keys") or set(live.keys()) != _PIXMO_SPLITS:
                raise ValueError(
                    f"Canonical PixMoCap {dataset_kind} must contain exactly train/validation"
                )
            for split in sorted(_PIXMO_SPLITS):
                live_split = live[split]
                expected = expected_splits[split]
                inventory = pixmo_row_path_inventory(live_split)
                if (
                    _runtime_dataset_fingerprint(live_split) != expected["dataset_fingerprint"]
                    or len(live_split) != expected["examples"]
                    or inventory["algorithm"] != builder["row_image_paths_algorithm"]
                    or inventory["rows"] != expected["examples"]
                    or inventory["unique_paths"] != expected["unique_image_paths"]
                    or inventory["sha256"] != expected["row_image_paths_sha256"]
                ):
                    raise ValueError(
                        f"Live canonical PixMoCap {dataset_kind} {split} split differs "
                        "from the builder manifest"
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
    config.vision_alignment.trainable_contract_sha256 = _canonical_sha256(
        {
            "model": config.model.as_config_dict(),
            "train_module": config.train_module.as_config_dict(),
            "router_lb_loss_weight": config.router_lb_loss_weight,
            "max_duration": {
                "value": config.trainer.max_duration.value,
                "unit": config.trainer.max_duration.unit.value,
            },
        }
    )


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
    train = train_config.build().manifest
    holdout = holdout_config.build().manifest
    if (
        train.sequence_length != config.data.sequence_length
        or holdout.sequence_length != config.data.sequence_length
        or holdout.num_windows < config.evaluation.examples_per_source
        or train.provenance.get("split") != "train"
        or holdout.provenance.get("split") != "holdout"
        or train.provenance.get("materialized_sources_sha256")
        != holdout.provenance.get("materialized_sources_sha256")
        or train.provenance.get("source_catalog_sha256")
        != holdout.provenance.get("source_catalog_sha256")
        or train.provenance.get("selection_seed") != holdout.provenance.get("selection_seed")
    ):
        raise ValueError("Native replay train/holdout lineage, size, or sequence contract differs")
    holdout_sources = {source.source_id: source for source in holdout.sources}
    for train_source in train.sources:
        holdout_source = holdout_sources.get(train_source.source_id)
        if holdout_source is None:
            continue
        train_starts = train_source.window_starts
        holdout_starts = holdout_source.window_starts
        train_index = holdout_index = 0
        while train_index < len(train_starts) and holdout_index < len(holdout_starts):
            train_start = train_starts[train_index]
            holdout_start = holdout_starts[holdout_index]
            if train_start == holdout_start:
                raise ValueError(
                    f"Native train and holdout replay overlap for source {train_source.source_id!r}"
                )
            if train_start < holdout_start:
                train_index += 1
            else:
                holdout_index += 1


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
        saved_gate_sha = saved_meta.get("parent_gate_sha256")
        configured_gate_sha = config.initialization.parent_gate_sha256
        if config.phase is VisionAlignmentPhase.bridge:
            if saved_gate_sha is not None:
                raise ValueError(
                    f"Existing bridge checkpoint {existing} unexpectedly records a parent gate"
                )
        elif not isinstance(saved_gate_sha, str) or len(saved_gate_sha) != 64:
            raise ValueError(f"Existing output checkpoint {existing} lacks its parent gate SHA")
        elif configured_gate_sha is not None and configured_gate_sha != saved_gate_sha:
            raise ValueError(
                f"Existing output checkpoint {existing} has parent gate SHA {saved_gate_sha}, "
                f"expected {configured_gate_sha}"
            )
        config.initialization.parent_gate_sha256 = saved_gate_sha
        config.vision_alignment.parent_gate_sha256 = saved_gate_sha
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
    parent_gate_sha = _validate_parent_gate(config, parent, parent_config, parent_sha)
    config.initialization.parent_gate_sha256 = parent_gate_sha
    config.vision_alignment.parent_gate_sha256 = parent_gate_sha


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

    expected_groups = (
        ("*lm.embeddings.weight", policy.connector_lr, "connector"),
        ("*connector.*", policy.connector_lr, "connector"),
        ("*vision.*", policy.vision_lr, "vision"),
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
        policy.phase is not VisionAlignmentPhase.bridge and policy.vision_lr <= 0
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
        (scheduler.schedulers["connector"], policy.connector_warmup),
        (scheduler.schedulers["vision"], policy.vision_warmup),
        (scheduler.default, policy.lm_warmup),
    )
    for component_scheduler, expected_warmup in expected_schedulers:
        if (
            not isinstance(component_scheduler, CosWithWarmup)
            or component_scheduler.warmup != expected_warmup
            or component_scheduler.t_max is not None
            or not math.isclose(component_scheduler.alpha_f, 0.1)
        ):
            raise ValueError("Vision-alignment scheduler shape or warmup was overridden")

    duration = config.trainer.max_duration
    if duration.unit.value != "steps" or duration.value <= 0:
        raise ValueError("Vision-alignment phase duration must be a positive step count")
    active_warmups = [policy.connector_warmup]
    if policy.phase is not VisionAlignmentPhase.bridge:
        active_warmups.append(policy.vision_warmup)
    if policy.phase is VisionAlignmentPhase.joint:
        active_warmups.append(policy.lm_warmup)
    if not config.data.allow_unpinned_synthetic_smoke and max(active_warmups) >= duration.value:
        raise ValueError("Every active warmup must be shorter than the production phase duration")


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
    if config.launch.cmd != config.expected_launch_command:
        raise ValueError("Beaker launch command differs from the validated training command")
    if config.launch.hostnames:
        raise ValueError("Vision alignment profiles may select Holmes only, not exact hosts")
    if config.launch.clusters != [BEAKER_CLUSTER]:
        raise ValueError(f"Vision alignment must target only {BEAKER_CLUSTER}")
    if config.launch.workspace != BEAKER_WORKSPACE or config.launch.budget != BEAKER_BUDGET:
        raise ValueError("Vision alignment workspace and budget are pinned by the recipe")
    if config.launch.num_gpus != 8 or config.launch.num_nodes < 1:
        raise ValueError("Vision alignment requires one or more complete 8-GPU Holmes nodes")
    if config.launch.allow_dirty:
        raise ValueError("Vision alignment may launch only a clean committed revision")
    if config.launch.git is None or config.launch.git.branch != "vision-moe":
        raise ValueError("Vision alignment may launch only from the user-owned vision-moe branch")
    if re.fullmatch(r"[0-9a-f]{40}", config.launch.git.ref or "") is None:
        raise ValueError("Vision alignment launch must pin an exact 40-character git revision")
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
    if list(policy.freeze_params) != (config.train_module.freeze_params or []):
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
    elif replay is not None:
        raise ValueError(f"Native text replay is forbidden while the LM is frozen ({config.phase})")

    source_audit = _validated_source_audit(config)
    _validate_validation_manifest(config, source_audit)

    parent = config.initialization.checkpoint
    if config.phase is VisionAlignmentPhase.bridge:
        if parent is not None or config.trainer.load_path is not None:
            raise ValueError("A fresh bridge must initialize from the pinned bare artifacts")
        if (
            config.initialization.parent_gate_path is not None
            or config.initialization.parent_gate_sha256 is not None
        ):
            raise ValueError("A bridge phase may not name a parent-quality gate")
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


def build_config(script: str, run_name: str, overrides: List[str]) -> ExperimentConfig:
    """Build and validate one phase config after resolving its phase selector first."""
    _validate_run_name(run_name)
    _validate_override_surface(overrides)
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
            parent_gate_sha256=initialization.parent_gate_sha256,
        ),
        global_batch_size=GLOBAL_BATCH_INSTANCES * policy.sequence_length,
        required_run_name=run_name,
        expected_launch_command=[script, "train", run_name, *overrides],
    ).merge(overrides)
    if (
        config.phase is not VisionAlignmentPhase.bridge
        and config.trainer.load_path is None
        and config.initialization.checkpoint is not None
    ):
        config.trainer.load_path = config.initialization.checkpoint
    config.vision_alignment.phase = config.phase
    config.vision_alignment.parent_checkpoint = config.initialization.checkpoint
    config.vision_alignment.parent_gate_sha256 = config.initialization.parent_gate_sha256

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
    }
    missing = sorted(set(targets) - supported)
    if missing:
        raise ValueError(
            "Vision-alignment phase contains source contracts without audited adapters: "
            f"{missing}. Implement and audit them; do not substitute QA/SFT sources."
        )
    audit = _validated_source_audit(config)
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
    validation_manifest = _validate_validation_manifest(
        config, _validated_source_audit(config), validate_live_datasets=False
    )
    for source_name in ("pixmo_caption", "pixmo_transcript"):
        dataset = _visual_dataset_config(config, token_ids, source_name, split="validation").build(
            tokenizer
        )
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


if __name__ == "__main__":
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
    if command == "train":
        prepare_training_environment(timeout=timedelta(minutes=60))
    else:
        prepare_cli_environment()
    profile, overrides = _load_profile(raw_overrides)
    experiment = build_config(script, run_name, overrides)
    experiment = _apply_profile_launch(experiment, profile)
    _validate_phase_contract(experiment, run_name)
    log.info(experiment)
    if command == "train":
        train(experiment)
        teardown_training_environment()
    elif command == "launch":
        launch(experiment)
    elif command != "dry_run":
        print(usage)
        sys.exit(1)
