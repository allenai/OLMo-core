"""Vision alignment continued pretraining for pinned bare language-model lineages.

This is a new recipe, intentionally independent from :mod:`Molmo2-Stage1`.  It treats
multimodal adaptation as continued pretraining rather than instruction tuning and exposes
three explicit, model-only phase boundaries:

``bridge``
    Load one pinned bare-LM checkpoint and pristine pinned SigLIP, then train only the
    connector and the six input-only image-token rows on document-formatted
    captions/transcripts.
``perception``
    Fork from a bridge checkpoint with a fresh optimizer and data cursor, unfreeze the vision
    encoder, and add audited perception sources while keeping the language model frozen.
``joint``
    Fork from a perception checkpoint, unfreeze the language model at a low learning rate,
    and begin exact native pretraining-data replay.

The default ``s002`` lineage retains its exact OLMoDDP/EP8 implementation. The paired 1.4B Cx8
Scalable-Softmax variants use the generic dense-HSDP path and differ only in per-head QK RMSNorm
on their four global-attention layers. Their recurrent GatedDeltaNet blocks require unpacked,
single-response examples; packed metadata is rejected instead of allowing recurrent-state
leakage across examples.

Run without arguments for usage.  No command in this file launches automatically.
"""

from __future__ import annotations

import fnmatch
import hashlib
import io
import json
import logging
import math
import os
import re
import stat
import subprocess
import sys
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union, cast

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
    NativeTextReplayVerificationReceipt,
)
from olmo_core.data.multimodal.mixtures.vision_alignment import (
    VisionAlignmentMixtureConfig,
    expected_loss_mass,
    sampling_weights_from_loss_mass,
)
from olmo_core.data.multimodal.paths import PIXMO_DATASETS
from olmo_core.data.multimodal.ssmax_single_response import (
    SSMaxSingleResponseDataset,
    SSMaxSingleResponseProjectionConfig,
    ssmax_single_response_calibration_summary,
    validate_ssmax_single_response_calibration,
)
from olmo_core.data.multimodal.vision_alignment_joint_provenance import (
    JointVisualProjectionManifest,
    build_selected_joint_dataset,
    joint_alignment_runtime_implementation_inventory,
    joint_alignment_runtime_registry_sha256,
    load_joint_visual_projection_manifest,
    validate_joint_live_example,
)
from olmo_core.data.multimodal.vision_alignment_joint_sources import (
    JOINT_VISUAL_SOURCE_NAMES,
    VISION_ALIGNMENT_JOINT_SOURCE_CATALOG_VERSION,
    VISION_ALIGNMENT_JOINT_SOURCE_REGISTRY_VERSION,
)
from olmo_core.data.multimodal.vision_alignment_perception_provenance import (
    PERCEPTION_SOURCE_NAMES,
    PerceptionPathSignature,
    PerceptionProvenanceManifest,
    build_selected_perception_dataset,
    load_perception_provenance_manifest,
)
from olmo_core.data.multimodal.vision_alignment_perception_sources import (
    VISION_ALIGNMENT_PERCEPTION_PROBE_EPOCHS,
    VISION_ALIGNMENT_PERCEPTION_PROBE_EXAMPLES,
    VISION_ALIGNMENT_PERCEPTION_PROBE_FORMAT,
    VISION_ALIGNMENT_PERCEPTION_PROBE_SEED,
    VISION_ALIGNMENT_PERCEPTION_PROBE_SELECTION_ALGORITHM,
    VISION_ALIGNMENT_PERCEPTION_PROBE_VERSION,
    VISION_ALIGNMENT_PERCEPTION_SOURCE_CATALOG_VERSION,
    VISION_ALIGNMENT_PERCEPTION_SOURCE_REGISTRY_VERSION,
    vision_alignment_perception_implementation_inventory,
    vision_alignment_perception_source_registry_sha256,
)
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
    serialized_example_sha256,
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
from olmo_core.launch.beaker import (
    BeakerEnvVar,
    BeakerLaunchConfig,
    BeakerWekaBucket,
    is_running_in_beaker_batch_job,
)
from olmo_core.nn.transformer import TransformerDataParallelWrappingStrategy
from olmo_core.nn.vision import Molmo2TokenIds, MultimodalLMConfig
from olmo_core.optim import (
    CosWithWarmup,
    OLMoDDPOptimizerConfig,
    OptimGroupOverride,
    PerGroupScheduler,
    SkipStepAdamWConfig,
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
    SSMaxHealthLedgerCallback,
    WandBCallback,
)
from olmo_core.train.checkpoint import Checkpointer
from olmo_core.train.train_module import (
    MultimodalOLMoDDPTrainModuleConfig,
    MultimodalTransformerTrainModuleConfig,
    TransformerActivationCheckpointingConfig,
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

SSMAX_HEAD_QKNORM_CHECKPOINT = (
    "/weka/oe-training-default/ai2-llm/scaling-ladders/mainline/yashasbls/"
    "v0.0.1-ssmax-a04f0e8e7236/1.4B-Cx8/pretrain/step65799"
)
SSMAX_HEAD_QKNORM_CONFIG_SHA256 = "34505b2d2738be361fc879722210bb5d17f4621dc4a3d22440177d2ff7ab5545"
SSMAX_HEAD_QKNORM_TRAINER_STATE_SHA256 = (
    "7beea032e59affa5f7fff7a7eeede2834e89362c19beef7c3115b665d789f445"
)
SSMAX_HEAD_QKNORM_DCP_METADATA_SHA256 = (
    "c302ab461188b8e708b751ea1721fa8d043a9144155e04c32181e112b51870f2"
)
SSMAX_HEAD_QKNORM_SOURCE_COMMIT = "fc048bf86746ba8eb97d9bf02bb8a54a59f98581"
SSMAX_HEAD_QKNORM_PARAMETER_COUNT = 1_422_110_784
SSMAX_HEAD_QKNORM_TENSOR_COUNT = 384
SSMAX_HEAD_QKNORM_KEYSET_SHA256 = "0d7834e0612209f80d2fe075e9816bf363aba85c507c60cf67ed59bad85ac597"
SSMAX_HEAD_QKNORM_INVENTORY_SHA256 = (
    "08100697841f2ac39074d3fb2938176f2c88ca4976e63e29a05cb9ea71eb21b4"
)
SSMAX_HEAD_QKNORM_CHECKPOINT_IDENTITY_SHA256 = (
    "4ec8641183f87e2d73b2779dec58ea9c11ffe919fa4ac1e01f6aec0c84028748"
)
SSMAX_HEAD_QKNORM_STATE_FILE_COUNT = 1025
SSMAX_HEAD_QKNORM_STATE_FILE_INVENTORY_SHA256 = (
    "b9f8ef60fd81bf84ae5827246190253bb0352420ff2d6121f31252a6565d02a3"
)
SSMAX_HEAD_QKNORM_TRAINER_STATE_COUNT = 64
SSMAX_HEAD_QKNORM_TRAINER_STATE_INVENTORY_SHA256 = (
    "e72a816e2278f3a1ef39d1c2cc4507af06b07180376adbee500f5b8cb8892042"
)

SSMAX_NO_QKNORM_CHECKPOINT = (
    "/weka/oe-training-default/ai2-llm/scaling-ladders/mainline/yashasbls/"
    "v0.0.1-ssmax_wo_qknorm-66759252e944/1.4B-Cx8/pretrain/step65799"
)
SSMAX_NO_QKNORM_CONFIG_SHA256 = "fea0962eda65fd3e26745be8d05fb799089a744f94aac4ea6a1a3af1608619be"
SSMAX_NO_QKNORM_TRAINER_STATE_SHA256 = (
    "d326aef32f4c8639b492f6a29703126078bc1bcd5cdd2dce5665c2b96c2829ee"
)
SSMAX_NO_QKNORM_DCP_METADATA_SHA256 = (
    "c17b8d5b491989ac8432df3e13b0b90d9e437789e8d18d17ced114158508b7d2"
)
SSMAX_NO_QKNORM_SOURCE_COMMIT = "9a4dc59b8f83749fb919388c6b77a29a1abf5ad5"
SSMAX_NO_QKNORM_PARAMETER_COUNT = 1_422_109_760
SSMAX_NO_QKNORM_TENSOR_COUNT = 376
SSMAX_NO_QKNORM_KEYSET_SHA256 = "9f94f6da44d60380570861125e5f9d09f85c3286beeb98c704b953d23b9a12e4"
SSMAX_NO_QKNORM_INVENTORY_SHA256 = (
    "97cd17d4a610efeaddfb1fe867f396dadb6e3ee4d6f60eb1da98bcfe7c8c5f85"
)
SSMAX_NO_QKNORM_CHECKPOINT_IDENTITY_SHA256 = (
    "66d38252ea86d000f92a2fe4aef1d0b8b52d8fc6865601ba911d19d68c68750b"
)
SSMAX_NO_QKNORM_STATE_FILE_COUNT = 1025
SSMAX_NO_QKNORM_STATE_FILE_INVENTORY_SHA256 = (
    "7d32cb4a511c0485bdb8f7d806424dd17779af04f80db460144580e9727f2c5b"
)
SSMAX_NO_QKNORM_TRAINER_STATE_COUNT = 64
SSMAX_NO_QKNORM_TRAINER_STATE_INVENTORY_SHA256 = (
    "1e75c405f566ee709ef3e77cde949f493a4edaa2194f2ce399f3b2993fc34a9e"
)

SSMAX_DATA_PATHS_SHA256 = "852491e33d2fb27ddd00619e500871d23429e00c382e70593079a4dc5f983139"
SSMAX_DATASET_FINGERPRINT = BASE_DATASET_FINGERPRINT
SSMAX_CHECKPOINT_MARKER_SHA256 = BASE_CHECKPOINT_MARKER_SHA256
SSMAX_OLMO_CORE_COMMIT = "1ca6f05c8061c260223e8dc65496f18167071c6c"
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
SSMAX_EXPERIMENT_ROOT = "/weka/oe-training-default/rustin/experiments/vision-ssmax-molmofication"
SSMAX_VISION_ALIGNMENT_ROOT = f"{SSMAX_EXPERIMENT_ROOT}/vision-alignment"
BEAKER_CLUSTER = "ai2/holmes"
BEAKER_WORKSPACE = "ai2/molmofication"
SSMAX_BEAKER_WORKSPACE = "ai2/scaling-ladders"
BEAKER_BUDGET = "ai2/oe-other"
SSMAX_DIRECT_JOINT_GIT_HISTORY_POST_SETUP = 'git fetch --no-tags --depth 4 origin "$GIT_REF"'
SSMAX_EXPLORATORY_JOINT_GIT_HISTORY_POST_SETUP = 'git fetch --no-tags --depth 4 origin "$GIT_REF"'
SSMAX_EXPLORATORY_WAIVER_JOINT_GIT_HISTORY_POST_SETUP = (
    'git fetch --no-tags --depth 5 origin "$GIT_REF"'
)
WANDB_PROJECT: Optional[str] = "vision-alignment"
SSMAX_WANDB_PROJECT: Optional[str] = "vision-ssmax-molmofication"
WANDB_ENTITY: Optional[str] = None

EP_DEGREE = 8
GLOBAL_BATCH_INSTANCES = 128
DATA_PREFETCH_WORKERS = 8
IMAGE_PATH_SIGNATURE_WORKERS = 64
IMAGE_PATH_SIGNATURE_MAX_PENDING = 256
PACK_BUFFER_SIZE = 48
PACK_MAX_CROPS = 9
MAX_CROPS = 8
DATA_SEED = 95818
INIT_SEED = 6198
EVAL_SEED = 6198
MIN_SOURCE_AUDIT_EXAMPLES = 1024
_PERCEPTION_PROVENANCE_RUNTIME_CACHE: Dict[Tuple[str, str], PerceptionProvenanceManifest] = {}
_JOINT_PROJECTION_RUNTIME_CACHE: Dict[Tuple[str, str, str], JointVisualProjectionManifest] = {}
PERCEPTION_PROFILE_ROOT = "configs/vision_moe/vision_alignment/perception"
PERCEPTION_PROFILE_ALLOWLIST = f"{PERCEPTION_PROFILE_ROOT}/approved_profiles.json"
PERCEPTION_PROFILE_ALLOWLIST_FORMAT = "vision_alignment_perception_profile_allowlist"
JOINT_PROFILE_ROOT = "configs/vision_moe/vision_alignment/joint"
JOINT_PROFILE_ALLOWLIST = f"{JOINT_PROFILE_ROOT}/approved_profiles.json"
JOINT_PROFILE_ALLOWLIST_FORMAT = "vision_alignment_joint_profile_allowlist"

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
_PERCEPTION_AUDIT_PROBE_FIELDS = frozenset(
    {"format", "version", "selection_algorithm", "seed", "epochs", "examples_per_source"}
)
_PERCEPTION_AUDIT_FIELDS = frozenset(
    {
        "format",
        "version",
        "status",
        "phase",
        "recipe_version",
        "formatter_version",
        "source_catalog_version",
        "auditor_sha256",
        "shared_auditor_sha256",
        "catalog_path",
        "catalog_sha256",
        "input_content_sha256",
        "source_registry_version",
        "source_registry_sha256",
        "source_implementation_inventory",
        "exporter_sha256",
        "image_provenance",
        "preprocessing_config",
        "preprocessing_config_sha256",
        "probe",
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
_PERCEPTION_AUDIT_INPUT_FIELDS = frozenset(
    {
        "name",
        "format",
        "path",
        "dataset_fingerprint",
        "dataset_size",
        "sha256",
        "probe_indices",
        "probe_indices_sha256",
        "probe_epochs",
        "serialized_row_hashes_sha256",
        "probe_image_content_sha256",
        "serialized_row_hashes",
    }
)

_JOINT_AUDIT_FORMAT = "vision_alignment_joint_source_audit"
_JOINT_AUDIT_VERSION = 1
_JOINT_AUDITOR_IMPLEMENTATION = "src/scripts/data/audit_vision_alignment_joint_mix.py"
_JOINT_SHARED_AUDITOR_IMPLEMENTATION = "src/scripts/data/audit_vision_alignment_mix.py"
_JOINT_EXPORTER_IMPLEMENTATION = "src/scripts/data/export_vision_alignment_joint_probe.py"
_JOINT_PROBE_FORMAT = "vision_alignment_joint_runtime_probe"
_JOINT_PROBE_VERSION = 1
_JOINT_PROBE_SEED = 6198
_JOINT_VISUAL_PROBE_INDICES = 256
_JOINT_VISUAL_PROBE_EPOCHS = (0, 1, 2, 3)
_JOINT_NATIVE_PROBE_INDICES = 1024
_JOINT_NATIVE_PROBE_EPOCHS = (0,)
_JOINT_SEQUENCE_LENGTH = 8192
_JOINT_NATIVE_REPLAY_PARENT_OBJECTS = 950
_JOINT_NATIVE_REPLAY_SELECTION_ALGORITHM = "affine-grid-v1"
_JOINT_NATIVE_REPLAY_SELECTION_SEED = 6198
_JOINT_SOURCE_NAMES = tuple(sorted((*JOINT_VISUAL_SOURCE_NAMES, "native_text_replay")))
_JOINT_AUDIT_FIELDS = frozenset(
    {
        "format",
        "version",
        "status",
        "phase",
        "recipe_version",
        "formatter_version",
        "source_catalog_version",
        "auditor_implementation",
        "shared_auditor_sha256",
        "catalog_path",
        "catalog_sha256",
        "catalog_content_sha256",
        "input_content_sha256",
        "source_registry_version",
        "source_registry_sha256",
        "source_implementation_inventory",
        "exporter_implementation",
        "visual_projection",
        "native_train_manifest",
        "native_verification_receipt",
        "preprocessing",
        "preprocessing_sha256",
        "probe",
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
_JOINT_IMPLEMENTATION_FIELDS = frozenset({"path", "sha256"})
_JOINT_VISUAL_PROJECTION_FIELDS = frozenset({"path", "raw_sha256", "content_sha256"})
_JOINT_NATIVE_MANIFEST_FIELDS = frozenset({"path", "raw_sha256", "content_fingerprint"})
_JOINT_RECEIPT_FIELDS = frozenset({"path", "sha256"})
_JOINT_PREPROCESSING_FIELDS = frozenset({"visual", "native_text_replay_fingerprint"})
_JOINT_PROBE_FIELDS = frozenset(
    {
        "format",
        "version",
        "selection_algorithm",
        "seed",
        "visual",
        "native_text_replay",
        "sequence_length",
        "truncation_policy",
    }
)
_JOINT_PROBE_KIND_FIELDS = frozenset({"unique_indices", "epochs", "rows_per_source"})
_JOINT_AUDIT_INPUT_FIELDS = frozenset(
    {
        "name",
        "kind",
        "format",
        "path",
        "dataset_fingerprint",
        "dataset_size",
        "sha256",
        "probe_epochs",
        "probe_indices",
        "probe_indices_sha256",
        "serialized_row_hashes_sha256",
        "probe_image_content_sha256",
        "max_observed_sequence_length",
        "truncated_rows",
        "serialized_row_hashes",
    }
)
_JOINT_SOURCE_SUMMARY_FIELDS = frozenset(
    {
        "examples",
        "raw_input_tokens",
        "positive_supervised_tokens",
        "summed_loss_weight",
        "mean_sum_loss_masks",
        "image_crops",
        "truncated_examples",
        "zero_loss_examples",
        "error_samples",
    }
)
_JOINT_METRIC_SUMMARY_FIELDS = frozenset({"total", "mean", "min", "max"})

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
    "--model_variant=",
    "--perception_trainability_arm=",
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


class VisionAlignmentModelVariant(StrEnum):
    """Pinned language-model lineage used by a vision-alignment experiment."""

    s002 = "s002"
    ssmax_head_qknorm = "ssmax_head_qknorm"
    ssmax_no_qknorm = "ssmax_no_qknorm"


_SSMAX_JOINT_DATA_ERROR_QUARANTINE_RUNS = {
    VisionAlignmentModelVariant.ssmax_head_qknorm: ("vision-ssmax-head-qknorm-1p4b-cx8-joint-v1"),
    VisionAlignmentModelVariant.ssmax_no_qknorm: ("vision-ssmax-no-qknorm-1p4b-cx8-joint-v1"),
}
_SSMAX_JOINT_DATA_ERROR_QUARANTINE_PROJECTION_SHA256 = (
    "11c1df56d7fbc270a9eff999193476c0c578c6964017d217a320b3d39305a730"
)
_SSMAX_JOINT_DATA_ERROR_QUARANTINE = {
    ("audited_alignment", 46333, 0): (
        ValueError,
        "no usable (user, assistant) turn in row",
    ),
    ("audited_alignment", 86346, 0): (
        ValueError,
        "no usable (user, assistant) turn in row",
    ),
    ("audited_alignment", 100000, 0): (
        ValueError,
        "no usable (user, assistant) turn in row",
    ),
}


class VisionAlignmentPhase(StrEnum):
    """The three optimizer/data phases of vision alignment."""

    bridge = "bridge"
    perception = "perception"
    joint = "joint"


class InitializationMode(StrEnum):
    """How a fresh vision-alignment phase obtains its model weights."""

    bare = "bare"
    checkpoint = "checkpoint"


class PerceptionTrainabilityArm(StrEnum):
    """Causal perception comparison with an identical data/provenance contract."""

    treatment = "treatment"
    frozen_vision_control = "frozen_vision_control"


_SECRETLESS_SSMAX_SMOKE_PROFILES = {
    VisionAlignmentModelVariant.ssmax_head_qknorm: (
        "vision-ssmax-head-qknorm-1p4b-cx8-bridge-smoke",
        "configs/vision_moe/vision_alignment/bridge/ssmax_head_qknorm_1p4b_cx8_smoke.yaml",
    ),
    VisionAlignmentModelVariant.ssmax_no_qknorm: (
        "vision-ssmax-no-qknorm-1p4b-cx8-bridge-smoke",
        "configs/vision_moe/vision_alignment/bridge/ssmax_no_qknorm_1p4b_cx8_smoke.yaml",
    ),
}
_SECRETLESS_SSMAX_SMOKE_REQUIRED_OVERRIDES = frozenset(
    {
        "--data.pixmo_cap_path=synthetic",
        "--data.allow_unpinned_synthetic_smoke=true",
        "--trainer.max_duration.value=1",
    }
)


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
    base_trainer_state_sha256: str = BASE_TRAINER_STATE_SHA256
    base_dataset_fingerprint: str = BASE_DATASET_FINGERPRINT
    base_parent_mix_sha256: str = BASE_PARENT_MIX_SHA256
    source_commit: Optional[str] = None
    source_olmo_core_commit: Optional[str] = None
    expected_lm_parameter_count: Optional[int] = None
    expected_lm_tensor_count: Optional[int] = None
    base_model_keyset_sha256: Optional[str] = None
    base_model_inventory_sha256: Optional[str] = None
    base_checkpoint_identity_sha256: Optional[str] = None
    base_checkpoint_state_file_count: Optional[int] = None
    base_checkpoint_state_file_inventory_sha256: Optional[str] = None
    base_checkpoint_trainer_state_count: Optional[int] = None
    base_checkpoint_trainer_state_inventory_sha256: Optional[str] = None
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

    @classmethod
    def for_model_variant(cls, variant: VisionAlignmentModelVariant) -> "ArtifactConfig":
        """Return the immutable parent-artifact contract for ``variant``."""

        if variant is VisionAlignmentModelVariant.s002:
            return cls()
        common = dict(
            base_data_paths_sha256=SSMAX_DATA_PATHS_SHA256,
            base_checkpoint_marker_sha256=SSMAX_CHECKPOINT_MARKER_SHA256,
            base_dataset_fingerprint=SSMAX_DATASET_FINGERPRINT,
            source_olmo_core_commit=SSMAX_OLMO_CORE_COMMIT,
        )
        if variant is VisionAlignmentModelVariant.ssmax_head_qknorm:
            return cls(
                base_checkpoint=SSMAX_HEAD_QKNORM_CHECKPOINT,
                base_config_sha256=SSMAX_HEAD_QKNORM_CONFIG_SHA256,
                base_checkpoint_metadata_sha256=SSMAX_HEAD_QKNORM_DCP_METADATA_SHA256,
                base_trainer_state_sha256=SSMAX_HEAD_QKNORM_TRAINER_STATE_SHA256,
                source_commit=SSMAX_HEAD_QKNORM_SOURCE_COMMIT,
                expected_lm_parameter_count=SSMAX_HEAD_QKNORM_PARAMETER_COUNT,
                expected_lm_tensor_count=SSMAX_HEAD_QKNORM_TENSOR_COUNT,
                base_model_keyset_sha256=SSMAX_HEAD_QKNORM_KEYSET_SHA256,
                base_model_inventory_sha256=SSMAX_HEAD_QKNORM_INVENTORY_SHA256,
                base_checkpoint_identity_sha256=SSMAX_HEAD_QKNORM_CHECKPOINT_IDENTITY_SHA256,
                base_checkpoint_state_file_count=SSMAX_HEAD_QKNORM_STATE_FILE_COUNT,
                base_checkpoint_state_file_inventory_sha256=(
                    SSMAX_HEAD_QKNORM_STATE_FILE_INVENTORY_SHA256
                ),
                base_checkpoint_trainer_state_count=SSMAX_HEAD_QKNORM_TRAINER_STATE_COUNT,
                base_checkpoint_trainer_state_inventory_sha256=(
                    SSMAX_HEAD_QKNORM_TRAINER_STATE_INVENTORY_SHA256
                ),
                **common,
            )
        if variant is VisionAlignmentModelVariant.ssmax_no_qknorm:
            return cls(
                base_checkpoint=SSMAX_NO_QKNORM_CHECKPOINT,
                base_config_sha256=SSMAX_NO_QKNORM_CONFIG_SHA256,
                base_checkpoint_metadata_sha256=SSMAX_NO_QKNORM_DCP_METADATA_SHA256,
                base_trainer_state_sha256=SSMAX_NO_QKNORM_TRAINER_STATE_SHA256,
                source_commit=SSMAX_NO_QKNORM_SOURCE_COMMIT,
                expected_lm_parameter_count=SSMAX_NO_QKNORM_PARAMETER_COUNT,
                expected_lm_tensor_count=SSMAX_NO_QKNORM_TENSOR_COUNT,
                base_model_keyset_sha256=SSMAX_NO_QKNORM_KEYSET_SHA256,
                base_model_inventory_sha256=SSMAX_NO_QKNORM_INVENTORY_SHA256,
                base_checkpoint_identity_sha256=SSMAX_NO_QKNORM_CHECKPOINT_IDENTITY_SHA256,
                base_checkpoint_state_file_count=SSMAX_NO_QKNORM_STATE_FILE_COUNT,
                base_checkpoint_state_file_inventory_sha256=(
                    SSMAX_NO_QKNORM_STATE_FILE_INVENTORY_SHA256
                ),
                base_checkpoint_trainer_state_count=SSMAX_NO_QKNORM_TRAINER_STATE_COUNT,
                base_checkpoint_trainer_state_inventory_sha256=(
                    SSMAX_NO_QKNORM_TRAINER_STATE_INVENTORY_SHA256
                ),
                **common,
            )
        raise AssertionError(f"Unhandled vision-alignment model variant: {variant}")


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
    perception_provenance_path: Optional[str] = None
    perception_provenance_sha256: Optional[str] = None
    joint_visual_projection_path: Optional[str] = None
    joint_visual_projection_sha256: Optional[str] = None
    ssmax_single_response_projection: Optional[SSMaxSingleResponseProjectionConfig] = None
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
    model_variant: VisionAlignmentModelVariant = VisionAlignmentModelVariant.s002
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
    train_module: Union[MultimodalOLMoDDPTrainModuleConfig, MultimodalTransformerTrainModuleConfig]
    trainer: TrainerConfig
    model_variant: VisionAlignmentModelVariant
    phase: VisionAlignmentPhase
    perception_trainability_arm: PerceptionTrainabilityArm
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
    reviewed_profile_path: Optional[str] = None
    reviewed_profile_sha256: Optional[str] = None
    reviewed_profile_allowlist_path: Optional[str] = None
    reviewed_profile_allowlist_sha256: Optional[str] = None


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


def _extract_model_variant(overrides: Sequence[str]) -> VisionAlignmentModelVariant:
    selectors = [
        value.split("=", 1)[1] for value in overrides if value.startswith("--model_variant=")
    ]
    if len(selectors) > 1:
        raise ValueError("Vision alignment accepts at most one --model_variant selector")
    if not selectors:
        return VisionAlignmentModelVariant.s002
    try:
        return VisionAlignmentModelVariant(selectors[0])
    except ValueError as error:
        raise ValueError(
            f"Unknown vision-alignment model variant {selectors[0]!r}; expected one of "
            f"{[variant.value for variant in VisionAlignmentModelVariant]}"
        ) from error


def _is_ssmax_variant(variant: VisionAlignmentModelVariant) -> bool:
    return variant in (
        VisionAlignmentModelVariant.ssmax_head_qknorm,
        VisionAlignmentModelVariant.ssmax_no_qknorm,
    )


def _experiment_root(variant: VisionAlignmentModelVariant) -> str:
    return SSMAX_VISION_ALIGNMENT_ROOT if _is_ssmax_variant(variant) else VISION_ALIGNMENT_ROOT


def _beaker_workspace(variant: VisionAlignmentModelVariant) -> str:
    return SSMAX_BEAKER_WORKSPACE if _is_ssmax_variant(variant) else BEAKER_WORKSPACE


def _wandb_project(variant: VisionAlignmentModelVariant) -> Optional[str]:
    return SSMAX_WANDB_PROJECT if _is_ssmax_variant(variant) else WANDB_PROJECT


def _expected_git_branch(variant: VisionAlignmentModelVariant) -> str:
    return "rustin/vision-ssmax-molmofication" if _is_ssmax_variant(variant) else "vision-moe"


def _pin_launch_git_branch(config: ExperimentConfig) -> None:
    """Ensure Gantry exports the exact reviewed branch as worker metadata."""

    if config.launch.git is None:
        raise ValueError("Vision alignment launch must include Git provenance")
    config.launch.git.branch = _expected_git_branch(config.model_variant)


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


def _build_model_config(
    token_ids: Molmo2TokenIds,
    artifacts: ArtifactConfig,
    model_variant: VisionAlignmentModelVariant = VisionAlignmentModelVariant.s002,
) -> MultimodalLMConfig:
    """Compose one pinned bare LM with the Molmo2 connector/SigLIP architecture."""
    from olmo_core.nn.attention import AttentionConfig, GatedDeltaNetConfig
    from olmo_core.nn.attention.backend import AttentionBackendName
    from olmo_core.nn.ddp.block import OLMoDDPTransformerBlockConfig
    from olmo_core.nn.moe.v2.ep_config import ExpertParallelPath
    from olmo_core.nn.transformer import OLMoDDPModelConfig, TransformerConfig
    from olmo_core.nn.vision import (
        load_molmo2_hf_vision_config,
        multimodal_config_from_molmo2_vision,
    )

    config_path = Path(artifacts.base_checkpoint) / "config.json"
    if _sha256_file(config_path) != artifacts.base_config_sha256:
        raise ValueError(f"Bare LM config fingerprint mismatch for {config_path}")
    data_paths_path = Path(artifacts.base_checkpoint) / "data_paths.txt"
    if _sha256_file(data_paths_path) != artifacts.base_data_paths_sha256:
        raise ValueError(f"Bare LM data-path fingerprint mismatch for {data_paths_path}")
    checkpoint_marker = Path(artifacts.base_checkpoint) / ".metadata.json"
    if _sha256_file(checkpoint_marker) != artifacts.base_checkpoint_marker_sha256:
        raise ValueError(f"Bare LM checkpoint marker mismatch for {checkpoint_marker}")
    checkpoint_metadata = Path(artifacts.base_checkpoint) / "model_and_optim" / ".metadata"
    if _sha256_file(checkpoint_metadata) != artifacts.base_checkpoint_metadata_sha256:
        raise ValueError(f"Bare LM DCP metadata mismatch for {checkpoint_metadata}")
    with config_path.open() as file_handle:
        raw_lm_config = json.load(file_handle)["model"]

    if model_variant is VisionAlignmentModelVariant.s002:
        lm_config = OLMoDDPModelConfig.from_dict(raw_lm_config)
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
    else:
        lm_config = TransformerConfig.from_dict(raw_lm_config)
        if not _is_ssmax_variant(model_variant):
            raise AssertionError(f"Unhandled model variant {model_variant}")
        if (
            lm_config.d_model != 1280
            or lm_config.n_layers != 20
            or lm_config.vocab_size != 100_352
            or not isinstance(lm_config.block.sequence_mixer, GatedDeltaNetConfig)
        ):
            raise ValueError("Pinned SSMax parent does not match the reviewed 1.4B hybrid layout")
        attention_layers: List[int] = []
        for layer_idx, block_config in enumerate(lm_config.resolved_block_configs):
            mixer = block_config.sequence_mixer
            if isinstance(mixer, AttentionConfig):
                attention_layers.append(layer_idx)
                mixer.backend = AttentionBackendName.flex
                if not mixer.scalable_softmax:
                    raise ValueError(f"SSMax is disabled in global-attention layer {layer_idx}")
                has_qk_norm = mixer.qk_norm is not None and mixer.use_head_qk_norm is True
                expected_qk_norm = model_variant is VisionAlignmentModelVariant.ssmax_head_qknorm
                if has_qk_norm != expected_qk_norm:
                    raise ValueError(
                        f"QK-norm contract differs in global-attention layer {layer_idx}: "
                        f"expected {expected_qk_norm}, got {has_qk_norm}"
                    )
            elif not isinstance(mixer, GatedDeltaNetConfig):
                raise TypeError(
                    f"SSMax layer {layer_idx} has unsupported mixer {type(mixer).__name__}"
                )
        if attention_layers != [4, 9, 14, 19]:
            raise ValueError(f"SSMax global-attention layers differ: {attention_layers}")
        if (
            artifacts.expected_lm_parameter_count is None
            or lm_config.num_params != artifacts.expected_lm_parameter_count
        ):
            raise ValueError(
                "SSMax LM parameter count differs from the parent checkpoint contract: "
                f"config={lm_config.num_params:,d}, "
                f"expected={artifacts.expected_lm_parameter_count}"
            )
        if (
            artifacts.expected_lm_tensor_count is None
            or artifacts.base_model_keyset_sha256 is None
            or artifacts.base_model_inventory_sha256 is None
        ):
            raise ValueError("SSMax parent model-state inventory pins are incomplete")
        from olmo_core.eval.checkpoint_model_state import (
            CheckpointModelStateContract,
            verify_checkpoint_model_state,
        )

        verified_parent = verify_checkpoint_model_state(
            artifacts.base_checkpoint,
            contract=CheckpointModelStateContract(
                config_sha256=artifacts.base_config_sha256,
                data_paths_sha256=artifacts.base_data_paths_sha256,
                marker_sha256=artifacts.base_checkpoint_marker_sha256,
                dcp_metadata_sha256=artifacts.base_checkpoint_metadata_sha256,
                model_keyset_sha256=artifacts.base_model_keyset_sha256,
                model_inventory_sha256=artifacts.base_model_inventory_sha256,
                model_tensor_count=artifacts.expected_lm_tensor_count,
                model_parameter_count=artifacts.expected_lm_parameter_count,
                model_parameter_tensor_count=artifacts.expected_lm_tensor_count,
            ),
            expected_model=lm_config.build(init_device="meta"),
            trainer_state_relative_path=None,
        )
        if verified_parent.buffer_keys:
            raise ValueError(
                f"Pinned SSMax parent unexpectedly contains buffers: {verified_parent.buffer_keys}"
            )
        log.info(
            "Verified %s parent DCP metadata: %d FP32 parameter tensors, %d parameters",
            model_variant.value,
            verified_parent.model_parameter_tensor_count,
            verified_parent.model_parameter_count,
        )

    hf_config = load_molmo2_hf_vision_config(
        artifacts.molmo2_config_model_id,
        revision=artifacts.molmo2_config_revision,
        cache_dir=artifacts.hf_cache_dir,
    )
    return multimodal_config_from_molmo2_vision(
        hf_config, lm_config, image_patch_token_id=token_ids.im_patch_id
    )


def _build_train_module_config(
    policy: _PhasePolicy,
    image_token_ids: List[int],
    model_variant: VisionAlignmentModelVariant = VisionAlignmentModelVariant.s002,
) -> Union[MultimodalOLMoDDPTrainModuleConfig, MultimodalTransformerTrainModuleConfig]:
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
    scheduler = PerGroupScheduler(
        schedulers={
            "connector": CosWithWarmup(
                warmup=policy.connector_warmup,
                alpha_f=0.1,
                t_max=policy.connector_t_max,
            ),
            "vision": CosWithWarmup(warmup=policy.vision_warmup, alpha_f=0.1, t_max=None),
        },
        default=CosWithWarmup(warmup=policy.lm_warmup, alpha_f=0.1, t_max=None),
    )
    common = dict(
        rank_microbatch_size=policy.rank_microbatch_instances * policy.sequence_length,
        max_sequence_length=policy.sequence_length,
        freeze_params=list(policy.freeze_params),
        train_embedding_rows=image_token_ids,
        vision_activation_checkpointing=policy.phase is not VisionAlignmentPhase.bridge,
        connector_activation_checkpointing=True,
        response_logits_only=True,
        diagnostics_interval=100,
        source_loss_mass_targets={},
        max_grad_norm=1.0,
        scheduler=scheduler,
    )
    if model_variant is VisionAlignmentModelVariant.s002:
        return MultimodalOLMoDDPTrainModuleConfig(
            **common,
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
            z_loss_multiplier=1e-4,
            compile_model=True,
            dp_config=TransformerDataParallelConfig(
                name=DataParallelType.ddp,
                reduce_dtype=DType.float32,
                only_allreduce_last_microbatch=True,
                reduce_grads_in_fp32=True,
                accumulate_grads_in_fp32=True,
            ),
            ep_config=TransformerExpertParallelConfig(degree=EP_DEGREE),
        )
    if not _is_ssmax_variant(model_variant):
        raise AssertionError(f"Unhandled model variant {model_variant}")
    return MultimodalTransformerTrainModuleConfig(
        **common,
        new_component_init_seed=INIT_SEED,
        optim=SkipStepAdamWConfig(
            lr=default_lr,
            betas=(0.9, 0.95),
            eps=1e-6,
            weight_decay=0.0,
            group_overrides=group_overrides,
            compile=False,
            foreach=True,
            sigma_factor=12,
        ),
        ac_config=TransformerActivationCheckpointingConfig(),
        z_loss_multiplier=1e-4,
        compile_model=True,
        autocast_precision=DType.bfloat16,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
            prefetch_factor=0,
        ),
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


def _configure_launch_runtime(
    launch_config: BeakerLaunchConfig,
    model_variant: VisionAlignmentModelVariant = VisionAlignmentModelVariant.s002,
) -> None:
    from olmo_core.launch.beaker_presets import get_preset

    preset = get_preset("olmo-ddp")
    if preset.beaker_image is not None:
        launch_config.beaker_image = preset.beaker_image
    env = {item.name: item.value for item in launch_config.env_vars}
    preset_env = dict(preset.env_vars)
    if _is_ssmax_variant(model_variant):
        preset_env.pop("OLMO_SYMM_VDEV2D_AUTO_BUILD", None)
    env.update(preset_env)
    env["TORCHINDUCTOR_COMPILE_THREADS"] = "8"
    if model_variant is VisionAlignmentModelVariant.s002:
        env.update(
            {
                "OLMO_USE_OWN_SYMM_MEM": "1",
                "OLMO_EP_MP_HIGH_PRIORITY_GROUP": "1",
                "OLMO_OWN_SYMM_PREWARM": "1",
                "TORCH_LOGS": "-dynamo",
            }
        )
    launch_config.env_vars = [BeakerEnvVar(name=name, value=value) for name, value in env.items()]
    launch_config.post_setup = (
        preset.post_setup if model_variant is VisionAlignmentModelVariant.s002 else None
    )


def _ssmax_joint_parent_gate_version(config: ExperimentConfig) -> Optional[int]:
    """Read the pinned SSMax joint parent-gate version used to configure launch history."""

    if config.phase is not VisionAlignmentPhase.joint or not _is_ssmax_variant(
        config.model_variant
    ):
        return None
    path_value = config.initialization.parent_gate_path
    expected_sha256 = config.initialization.parent_gate_sha256
    if path_value is None or expected_sha256 is None:
        return None
    gate_path = Path(path_value).expanduser().resolve()
    try:
        raw = gate_path.read_bytes()
        gate = json.loads(
            raw,
            object_pairs_hook=_strict_json_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Invalid parent-quality gate {gate_path}: {error}") from error
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Parent-quality gate SHA mismatch while configuring Git history")
    if not isinstance(gate, Mapping) or type(gate.get("version")) is not int:
        raise ValueError("Parent-quality gate version is not a canonical integer")
    return gate["version"]


def _configure_ssmax_direct_joint_git_history(config: ExperimentConfig) -> None:
    """Materialize the exact ancestry needed by a direct or exploratory joint consumer."""

    gate_version = _ssmax_joint_parent_gate_version(config)
    if gate_version == 7:
        config.launch.post_setup = SSMAX_DIRECT_JOINT_GIT_HISTORY_POST_SETUP
    elif gate_version == 8:
        config.launch.post_setup = SSMAX_EXPLORATORY_JOINT_GIT_HISTORY_POST_SETUP
    elif gate_version == 9:
        config.launch.post_setup = SSMAX_EXPLORATORY_WAIVER_JOINT_GIT_HISTORY_POST_SETUP


def _is_secretless_ssmax_smoke_request(
    *,
    model_variant: VisionAlignmentModelVariant,
    phase: VisionAlignmentPhase,
    run_name: str,
    reviewed_profile_path: Optional[str],
    overrides: Sequence[str],
) -> bool:
    expected_identity = _SECRETLESS_SSMAX_SMOKE_PROFILES.get(model_variant)
    return bool(
        expected_identity is not None
        and phase is VisionAlignmentPhase.bridge
        and run_name == expected_identity[0]
        and reviewed_profile_path == expected_identity[1]
        and _SECRETLESS_SSMAX_SMOKE_REQUIRED_OVERRIDES.issubset(overrides)
    )


def _build_vision_alignment_launch_config(
    *,
    script: str,
    run_name: str,
    overrides: Sequence[str],
    model_variant: VisionAlignmentModelVariant,
    secretless_runtime_smoke: bool,
) -> BeakerLaunchConfig:
    if secretless_runtime_smoke:
        return BeakerLaunchConfig(
            name=f"{run_name}-runtime",
            cmd=[script, "train", run_name, *overrides],
            budget=BEAKER_BUDGET,
            workspace=_beaker_workspace(model_variant),
            clusters=[BEAKER_CLUSTER],
            num_nodes=2,
            num_gpus=8,
            shared_filesystem=True,
            allow_dirty=False,
            env_vars=[BeakerEnvVar(name="NCCL_DEBUG", value="WARN")],
            env_secrets=[],
            google_credentials_secret=None,
            aws_config_secret=None,
            aws_credentials_secret=None,
            weka_buckets=[
                BeakerWekaBucket(
                    bucket="oe-training-default",
                    mount="/weka/oe-training-default",
                )
            ],
            step_soft_timeout=10 * 60,
        )
    return build_launch_config(
        name=run_name,
        root_dir=get_root_dir(BEAKER_CLUSTER),
        cmd=[script, "train", run_name, *overrides],
        cluster=BEAKER_CLUSTER,
        workspace=_beaker_workspace(model_variant),
        budget=BEAKER_BUDGET,
        num_nodes=2,
    )


def _is_secretless_ssmax_smoke(config: ExperimentConfig) -> bool:
    if not config.data.allow_unpinned_synthetic_smoke:
        return False
    expected_identity = _SECRETLESS_SSMAX_SMOKE_PROFILES.get(config.model_variant)
    duration = config.trainer.max_duration
    return bool(
        expected_identity is not None
        and config.phase is VisionAlignmentPhase.bridge
        and config.required_run_name == expected_identity[0]
        and config.reviewed_profile_path == expected_identity[1]
        and config.data.pixmo_cap_path == "synthetic"
        and duration.unit.value == "steps"
        and duration.value == 1
    )


def _configure_synthetic_smoke_observability(config: ExperimentConfig) -> None:
    """Keep the two reviewed one-step smokes independent of user-scoped secrets."""

    if not _is_secretless_ssmax_smoke(config):
        return
    wandb = config.trainer.callbacks.get("wandb")
    beaker = config.trainer.callbacks.get("beaker")
    if not isinstance(wandb, WandBCallback) or not isinstance(beaker, BeakerCallback):
        raise ValueError("Synthetic smoke requires the standard W&B and Beaker callbacks")
    wandb.enabled = False
    beaker.enabled = False
    config.launch.env_secrets = []


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


def _reviewed_profile_policy(phase: VisionAlignmentPhase) -> Tuple[str, str, str]:
    if phase is VisionAlignmentPhase.perception:
        return (
            PERCEPTION_PROFILE_ROOT,
            PERCEPTION_PROFILE_ALLOWLIST,
            PERCEPTION_PROFILE_ALLOWLIST_FORMAT,
        )
    if phase is VisionAlignmentPhase.joint:
        return JOINT_PROFILE_ROOT, JOINT_PROFILE_ALLOWLIST, JOINT_PROFILE_ALLOWLIST_FORMAT
    raise ValueError(f"Phase {phase.value!r} does not use a reviewed profile allowlist")


def _load_approved_profiles(
    repository_root: Path, phase: VisionAlignmentPhase
) -> Tuple[Mapping[str, str], str]:
    """Load one exact code-reviewed production-profile SHA allowlist."""
    profile_root, allowlist_name, allowlist_format = _reviewed_profile_policy(phase)
    allowlist_path = (repository_root / allowlist_name).resolve()
    try:
        raw = allowlist_path.read_bytes()
        value = json.loads(raw, object_pairs_hook=_strict_json_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(
            f"Invalid {phase.value} profile allowlist {allowlist_path}: {error}"
        ) from error
    if not isinstance(value, Mapping) or set(value) != {"format", "version", "profiles"}:
        raise ValueError(f"{phase.value.capitalize()} profile allowlist schema differs")
    if (
        value["format"] != allowlist_format
        or isinstance(value["version"], bool)
        or not isinstance(value["version"], int)
        or value["version"] != 1
        or not isinstance(value["profiles"], Mapping)
    ):
        raise ValueError(f"{phase.value.capitalize()} profile allowlist identity differs")
    expected_raw = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    if raw != expected_raw:
        raise ValueError(
            f"{phase.value.capitalize()} profile allowlist must use canonical JSON bytes"
        )
    profiles: Dict[str, str] = {}
    for profile_path, profile_sha256 in value["profiles"].items():
        if (
            not isinstance(profile_path, str)
            or not profile_path.startswith(f"{profile_root}/")
            or Path(profile_path).parent.as_posix() != profile_root
            or Path(profile_path).suffix != ".yaml"
            or Path(profile_path).name.endswith(".yaml.template")
            or not isinstance(profile_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", profile_sha256) is None
        ):
            raise ValueError(
                f"{phase.value.capitalize()} profile allowlist contains an invalid path or SHA-256"
            )
        profiles[profile_path] = profile_sha256
    return profiles, hashlib.sha256(raw).hexdigest()


def _load_approved_perception_profiles(repository_root: Path) -> Tuple[Mapping[str, str], str]:
    """Load the exact code-reviewed perception-profile SHA allowlist."""
    return _load_approved_profiles(repository_root, VisionAlignmentPhase.perception)


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
    reviewed_phase = (
        VisionAlignmentPhase(phase)
        if phase in (VisionAlignmentPhase.perception.value, VisionAlignmentPhase.joint.value)
        else None
    )
    ssmax_bridge_profile = phase == VisionAlignmentPhase.bridge.value and any(
        value
        in (
            "--model_variant=ssmax_head_qknorm",
            "--model_variant=ssmax_no_qknorm",
        )
        for value in profile_overrides
    )
    if reviewed_phase is not None:
        profile_root, _, _ = _reviewed_profile_policy(reviewed_phase)
        approved_root = (repository_root / profile_root).resolve()
        if (
            profile_path.parent != approved_root
            or profile_path.suffix != ".yaml"
            or profile_path.name.endswith(".yaml.template")
        ):
            raise ValueError(
                f"Production {reviewed_phase.value} requires a checked-in .yaml profile directly "
                "under "
                f"{approved_root}"
            )
        if cli:
            raise ValueError(
                f"Production {reviewed_phase.value} profiles own the complete configuration; "
                "additional "
                "CLI overrides are forbidden"
            )
    if ssmax_bridge_profile:
        approved_root = (repository_root / "configs/vision_moe/vision_alignment/bridge").resolve()
        if (
            profile_path.parent != approved_root
            or profile_path.suffix != ".yaml"
            or profile_path.name.endswith(".yaml.template")
        ):
            raise ValueError(
                "Production SSMax bridge requires a checked-in .yaml profile directly under "
                f"{approved_root}"
            )
        if cli:
            raise ValueError(
                "Production SSMax bridge profiles own the complete configuration; additional "
                "CLI overrides are forbidden"
            )
    try:
        relative_profile_path = profile_path.relative_to(repository_root).as_posix()
    except ValueError:
        if reviewed_phase is not None:
            raise ValueError(
                f"Production {reviewed_phase.value} profiles must live inside the repository"
            )
        relative_profile_path = str(profile_path)
    raw_profile_sha256 = hashlib.sha256(raw_profile).hexdigest()
    if reviewed_phase is not None:
        _, allowlist_path, _ = _reviewed_profile_policy(reviewed_phase)
        approved_profiles, allowlist_sha256 = _load_approved_profiles(
            repository_root, reviewed_phase
        )
        approved_sha256 = approved_profiles.get(relative_profile_path)
        if approved_sha256 != raw_profile_sha256:
            raise ValueError(
                f"Production {reviewed_phase.value} profile bytes are not in the reviewed "
                "SHA-256 allowlist"
            )
        profile["__reviewed_allowlist_path__"] = allowlist_path
        profile["__reviewed_allowlist_sha256__"] = allowlist_sha256
    profile["__reviewed_path__"] = relative_profile_path
    profile["__reviewed_sha256__"] = raw_profile_sha256
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
    reviewed_path = profile.get("__reviewed_path__")
    reviewed_sha256 = profile.get("__reviewed_sha256__")
    if reviewed_path is not None or reviewed_sha256 is not None:
        if (
            not isinstance(reviewed_path, str)
            or not reviewed_path
            or not isinstance(reviewed_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", reviewed_sha256) is None
        ):
            raise ValueError("Vision-alignment profile review identity is malformed")
        config.reviewed_profile_path = reviewed_path
        config.reviewed_profile_sha256 = reviewed_sha256
        profile_overrides = profile.get("overrides", [])
        if not isinstance(profile_overrides, list) or not all(
            isinstance(value, str) for value in profile_overrides
        ):
            raise ValueError("Vision-alignment profile overrides are malformed")
        expanded_profile_command = [
            *config.expected_launch_command[:3],
            f"--phase={profile.get('phase')}",
            *profile_overrides,
        ]
        if (
            config.expected_launch_command[: len(expanded_profile_command)]
            != expanded_profile_command
        ):
            raise ValueError("Vision-alignment launch command does not match its profile")
        profile_command = [
            *config.expected_launch_command[:3],
            f"--profile={reviewed_path}",
            *config.expected_launch_command[len(expanded_profile_command) :],
        ]
        config.expected_launch_command = profile_command
        config.launch.cmd = list(profile_command)
        if config.phase in (VisionAlignmentPhase.perception, VisionAlignmentPhase.joint):
            _, expected_allowlist_path, _ = _reviewed_profile_policy(config.phase)
            allowlist_path = profile.get("__reviewed_allowlist_path__")
            allowlist_sha256 = profile.get("__reviewed_allowlist_sha256__")
            if (
                allowlist_path != expected_allowlist_path
                or not isinstance(allowlist_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", allowlist_sha256) is None
            ):
                raise ValueError(
                    f"{config.phase.value.capitalize()} profile allowlist identity is malformed"
                )
            config.reviewed_profile_allowlist_path = allowlist_path
            config.reviewed_profile_allowlist_sha256 = allowlist_sha256
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


_LEGACY_S002_ARTIFACT_FIELDS = frozenset(
    {
        "base_trainer_state_sha256",
        "base_dataset_fingerprint",
        "base_parent_mix_sha256",
        "source_commit",
        "source_olmo_core_commit",
        "expected_lm_parameter_count",
        "expected_lm_tensor_count",
        "base_model_keyset_sha256",
        "base_model_inventory_sha256",
    }
)


def _checkpoint_model_variant(checkpoint_config: Mapping[str, Any]) -> VisionAlignmentModelVariant:
    """Resolve a saved alignment checkpoint's model lineage without guessing dense variants.

    Checkpoints produced before model variants were introduced have neither root nor metadata
    selectors. Those are accepted only when their saved artifact contract identifies the exact
    historical s002 parent. Every newer checkpoint, and every dense SSMax checkpoint, must carry
    an explicit and internally consistent selector.
    """

    metadata = checkpoint_config.get("vision_alignment")
    metadata_variant = metadata.get("model_variant") if isinstance(metadata, Mapping) else None
    root_variant = checkpoint_config.get("model_variant")
    selectors = [value for value in (root_variant, metadata_variant) if value is not None]
    if selectors:
        if any(not isinstance(value, str) for value in selectors):
            raise ValueError("Checkpoint model_variant selectors must be strings")
        if len(set(selectors)) != 1:
            raise ValueError("Checkpoint root and vision-alignment model variants differ")
        try:
            return VisionAlignmentModelVariant(selectors[0])
        except ValueError as error:
            raise ValueError(
                f"Checkpoint names an unknown model variant {selectors[0]!r}"
            ) from error

    artifacts = checkpoint_config.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Checkpoint lacks an explicit model variant and saved artifact lineage")
    expected = ArtifactConfig.for_model_variant(VisionAlignmentModelVariant.s002)
    legacy_identity = {
        "base_checkpoint": expected.base_checkpoint,
        "base_config_sha256": expected.base_config_sha256,
        "base_data_paths_sha256": expected.base_data_paths_sha256,
        "base_checkpoint_marker_sha256": expected.base_checkpoint_marker_sha256,
        "base_checkpoint_metadata_sha256": expected.base_checkpoint_metadata_sha256,
    }
    if any(artifacts.get(name) != value for name, value in legacy_identity.items()):
        raise ValueError("Only the exact historical s002 artifact lineage may omit model_variant")
    return VisionAlignmentModelVariant.s002


def _validate_checkpoint_model_lineage(
    config: ExperimentConfig,
    checkpoint_config: Mapping[str, Any],
    *,
    checkpoint: str,
) -> None:
    """Require a phase parent/resume to retain this run's exact bare-model lineage."""

    checkpoint_variant = _checkpoint_model_variant(checkpoint_config)
    if checkpoint_variant is not config.model_variant:
        raise ValueError(
            f"Checkpoint {checkpoint} uses model variant {checkpoint_variant.value!r}; "
            f"expected {config.model_variant.value!r}"
        )
    raw_artifacts = checkpoint_config.get("artifacts")
    if not isinstance(raw_artifacts, Mapping):
        raise ValueError(f"Checkpoint {checkpoint} lacks its saved artifact contract")
    expected_artifacts = asdict(ArtifactConfig.for_model_variant(config.model_variant))
    metadata = checkpoint_config.get("vision_alignment")
    legacy_s002 = (
        config.model_variant is VisionAlignmentModelVariant.s002
        and checkpoint_config.get("model_variant") is None
        and (not isinstance(metadata, Mapping) or metadata.get("model_variant") is None)
    )
    missing = set(expected_artifacts) - set(raw_artifacts)
    optional_missing = {
        name for name, expected_value in expected_artifacts.items() if expected_value is None
    }
    allowed_missing = set(optional_missing)
    if legacy_s002:
        allowed_missing.update(_LEGACY_S002_ARTIFACT_FIELDS)
    if not missing <= allowed_missing:
        raise ValueError(
            f"Checkpoint {checkpoint} artifact contract is incomplete: {sorted(missing)}"
        )
    mismatches = {
        name: (raw_artifacts.get(name), expected_value)
        for name, expected_value in expected_artifacts.items()
        if name in raw_artifacts and raw_artifacts.get(name) != expected_value
    }
    if mismatches:
        raise ValueError(
            f"Checkpoint {checkpoint} bare-model artifact lineage differs: {mismatches}"
        )


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
    version_1_fields = {
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
    version_2_fields = version_1_fields | {
        "promotion_bundle_path",
        "promotion_bundle_sha256",
        "checkpoint_identity_sha256",
        "approved_by",
        "approved_at",
        "waivers",
    }
    version_3_fields = version_2_fields | {
        "promotion_kind",
        "promotion_policy",
    }
    version_4_fields = {
        "format",
        "version",
        "status",
        "recipe_version",
        "formatter_version",
        "phase",
        "model_variant",
        "arm",
        "checkpoint",
        "checkpoint_config_sha256",
        "checkpoint_identity_sha256",
        "data_contract_sha256",
        "trainable_contract_sha256",
        "global_step",
        "metrics_artifact_sha256",
        "promotion_report_path",
        "promotion_report_sha256",
        "manifest_path",
        "manifest_sha256",
        "manifest_content_sha256",
        "approved_by",
        "approved_at",
        "waivers",
    }
    version_5_fields = version_4_fields | {"promotion_report_content_sha256"}
    # SSMax perception protocol v2 uses the same exact approval payload shape as v5 while
    # assigning a new version so the referenced v2 manifest/report cannot be confused with
    # historical zero-skip evidence.
    version_6_fields = version_5_fields
    # The direct single-lineage protocol has a distinct, transitively bound approval payload.
    # It intentionally has no paired-arm field and cannot be confused with a v5/v6 gate.
    version_7_fields = {
        "format",
        "version",
        "status",
        "recipe_version",
        "formatter_version",
        "phase",
        "model_variant",
        "lineage_kind",
        "run_id",
        "checkpoint",
        "checkpoint_config_sha256",
        "checkpoint_identity_sha256",
        "data_contract_sha256",
        "trainable_contract_sha256",
        "global_step",
        "metrics_artifact_sha256",
        "promotion_report_path",
        "promotion_report_sha256",
        "promotion_report_content_sha256",
        "manifest_path",
        "manifest_sha256",
        "manifest_content_sha256",
        "protocol_amendment_path",
        "protocol_amendment_sha256",
        "protocol_amendment_content_sha256",
        "training_git_ref",
        "evidence_git_ref",
        "approved_by",
        "approved_at",
        "waivers",
    }
    # Exploratory admission preserves the strict direct-evidence lineage while recording the
    # exact failed strict report and an explicit, content-addressed authorization. It is a
    # separate schema and may authorize only the SSMax joint phase.
    version_8_fields = {
        "format",
        "version",
        "status",
        "scope",
        "recipe_version",
        "formatter_version",
        "phase",
        "model_variant",
        "lineage_kind",
        "run_id",
        "checkpoint",
        "checkpoint_config_sha256",
        "checkpoint_identity_sha256",
        "data_contract_sha256",
        "trainable_contract_sha256",
        "global_step",
        "metrics_artifact_sha256",
        "strict_report_path",
        "strict_report_sha256",
        "strict_report_content_sha256",
        "strict_report_status",
        "strict_receipts",
        "acknowledged_deviations",
        "manifest_path",
        "manifest_sha256",
        "manifest_content_sha256",
        "protocol_amendment_path",
        "protocol_amendment_sha256",
        "protocol_amendment_content_sha256",
        "authorization",
        "training_git_ref",
        "evidence_git_ref",
        "approved_by",
        "approved_at",
        "waivers",
    }
    # The evaluation-complete exploratory waiver preserves all direct evaluation evidence while
    # explicitly limiting run-health evidence to step 0. Its distinct schema cannot be confused
    # with the six-receipt v8 exploratory admission and may authorize only SSMax joint training.
    version_9_fields = {
        "format",
        "version",
        "status",
        "scope",
        "recipe_version",
        "formatter_version",
        "phase",
        "model_variant",
        "lineage_kind",
        "run_id",
        "checkpoint",
        "checkpoint_config_sha256",
        "checkpoint_identity_sha256",
        "data_contract_sha256",
        "trainable_contract_sha256",
        "global_step",
        "metrics_artifact_sha256",
        "evidence_report_path",
        "evidence_report_sha256",
        "evidence_report_content_sha256",
        "evidence_report_status",
        "evidence_receipts",
        "acknowledged_deviations",
        "manifest_path",
        "manifest_sha256",
        "manifest_content_sha256",
        "protocol_amendment_path",
        "protocol_amendment_sha256",
        "protocol_amendment_content_sha256",
        "authorization",
        "training_git_ref",
        "evidence_git_ref",
        "admission_git_ref",
        "approved_by",
        "approved_at",
        "waivers",
        "promotion_decision",
        "winner_selection",
    }
    gate_schemas = {
        1: version_1_fields,
        2: version_2_fields,
        3: version_3_fields,
        4: version_4_fields,
        5: version_5_fields,
        6: version_6_fields,
        7: version_7_fields,
        8: version_8_fields,
        9: version_9_fields,
    }
    gate_version = gate.get("version")
    allowed_fields = gate_schemas.get(gate_version) if type(gate_version) is int else None
    if allowed_fields is None:
        raise ValueError(
            "Parent-quality gate version must be exactly integer 1, 2, 3, 4, 5, 6, 7, 8, or 9"
        )
    if set(gate) != allowed_fields:
        raise ValueError(
            "Parent-quality gate fields differ from the locked schema: "
            f"missing={sorted(allowed_fields - set(gate))}, "
            f"extra={sorted(set(gate) - allowed_fields)}"
        )
    parent_meta = parent_config.get("vision_alignment")
    expected_parent_phase = config.initialization.expected_parent_phase
    assert isinstance(parent_meta, Mapping)
    if gate_version in (3, 4, 5, 6, 7, 8, 9):
        expected_recipe_version = parent_meta.get("recipe_version")
        expected_formatter_version = parent_meta.get("formatter_version")
        if type(expected_recipe_version) is not int or not isinstance(
            expected_formatter_version, str
        ):
            raise ValueError(
                "Perception parent checkpoint recipe and formatter metadata are malformed"
            )
    else:
        expected_recipe_version = RECIPE_VERSION
        expected_formatter_version = FORMATTER_VERSION
    if (
        gate["format"] != "vision_alignment_parent_gate"
        or gate["status"] != "approved"
        or gate["recipe_version"] != expected_recipe_version
        or gate["formatter_version"] != expected_formatter_version
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
    production_perception = getattr(
        config, "phase", None
    ) is VisionAlignmentPhase.perception and not bool(
        getattr(getattr(config, "data", None), "allow_unpinned_synthetic_smoke", False)
    )
    production_joint = getattr(config, "phase", None) is VisionAlignmentPhase.joint and not bool(
        getattr(getattr(config, "data", None), "allow_unpinned_synthetic_smoke", False)
    )
    model_variant = getattr(config, "model_variant", VisionAlignmentModelVariant.s002)
    if production_perception:
        if _is_ssmax_variant(model_variant):
            if gate_version != 4:
                raise ValueError(
                    "Production SSMax perception requires a deviation-free v4 bridge parent gate"
                )
        elif gate_version != 2:
            raise ValueError(
                "Production perception requires a v2 parent gate with an audited promotion bundle"
            )
    if gate_version == 3 and getattr(config, "phase", None) is not VisionAlignmentPhase.joint:
        raise ValueError("A v3 perception parent gate may only authorize the joint phase")
    if gate_version == 4 and (
        getattr(config, "phase", None) is not VisionAlignmentPhase.perception
        or not _is_ssmax_variant(model_variant)
        or expected_parent_phase is not VisionAlignmentPhase.bridge
    ):
        raise ValueError("A v4 SSMax bridge parent gate may only authorize SSMax perception")
    if gate_version in (5, 6) and (
        getattr(config, "phase", None) is not VisionAlignmentPhase.joint
        or not _is_ssmax_variant(model_variant)
        or expected_parent_phase is not VisionAlignmentPhase.perception
    ):
        raise ValueError("A v5/v6 SSMax perception parent gate may only authorize SSMax joint")
    if gate_version == 7 and (
        getattr(config, "phase", None) is not VisionAlignmentPhase.joint
        or not _is_ssmax_variant(model_variant)
        or expected_parent_phase is not VisionAlignmentPhase.perception
    ):
        raise ValueError("A v7 direct SSMax perception parent gate may only authorize SSMax joint")
    if gate_version == 8 and (
        getattr(config, "phase", None) is not VisionAlignmentPhase.joint
        or not _is_ssmax_variant(model_variant)
        or expected_parent_phase is not VisionAlignmentPhase.perception
    ):
        raise ValueError(
            "A v8 exploratory SSMax perception parent gate may only authorize SSMax joint"
        )
    if gate_version == 9 and (
        getattr(config, "phase", None) is not VisionAlignmentPhase.joint
        or not _is_ssmax_variant(model_variant)
        or expected_parent_phase is not VisionAlignmentPhase.perception
    ):
        raise ValueError(
            "A v9 exploratory-waiver SSMax perception parent gate may only authorize SSMax joint"
        )
    expected_joint_gates = {5, 6, 7, 8, 9} if _is_ssmax_variant(model_variant) else {3}
    if production_joint and (
        gate_version not in expected_joint_gates
        or expected_parent_phase is not VisionAlignmentPhase.perception
    ):
        expected_joint_gate_text = (
            "v5, v6, v7, v8, or v9" if _is_ssmax_variant(model_variant) else "v3"
        )
        raise ValueError(
            f"Production joint requires a {expected_joint_gate_text} perception parent gate and "
            "perception parent phase"
        )
    if gate_version == 2:
        from olmo_core.eval import vision_alignment_promotion as promotion

        bundle_path = Path(str(gate["promotion_bundle_path"])).expanduser().resolve()
        expected_bundle_sha = gate["promotion_bundle_sha256"]
        if re.fullmatch(r"[0-9a-f]{64}", str(expected_bundle_sha)) is None:
            raise ValueError("Parent-quality gate promotion bundle SHA-256 is invalid")
        if gate["metrics_artifact_sha256"] != expected_bundle_sha:
            raise ValueError(
                "Parent-quality gate metrics artifact must be the pinned promotion bundle"
            )
        try:
            actual_bundle_sha = promotion.sha256_file(bundle_path)
            bundle = promotion.load_json(bundle_path)
        except (OSError, promotion.PromotionValidationError) as error:
            raise ValueError(f"Invalid parent promotion bundle {bundle_path}: {error}") from error
        if actual_bundle_sha != expected_bundle_sha:
            raise ValueError(
                f"Parent promotion bundle SHA mismatch: configured {expected_bundle_sha}, "
                f"actual {actual_bundle_sha}"
            )
        if not isinstance(bundle, Mapping):
            raise ValueError("Parent promotion bundle must be a JSON object")
        try:
            summary = promotion.validate_promotion_bundle(
                bundle,
                expected_checkpoint=Path(parent).resolve(),
                expected_checkpoint_config_sha256=parent_config_sha256,
            )
        except promotion.PromotionValidationError as error:
            raise ValueError(f"Parent promotion bundle failed validation: {error}") from error
        candidate = summary["candidate"]
        checkpoint_identity_sha = gate["checkpoint_identity_sha256"]
        if (
            re.fullmatch(r"[0-9a-f]{64}", str(checkpoint_identity_sha)) is None
            or checkpoint_identity_sha != candidate["checkpoint_identity_sha256"]
        ):
            raise ValueError("Parent gate checkpoint identity differs from the promotion bundle")
        approved_by = gate["approved_by"]
        if (
            not isinstance(approved_by, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._@:/+\-]{2,127}", approved_by) is None
        ):
            raise ValueError("Parent-quality gate approved_by is not a durable human identity")
        approved_at = gate["approved_at"]
        if not isinstance(approved_at, str):
            raise ValueError("Parent-quality gate approved_at must be an ISO-8601 timestamp")
        try:
            parsed_approval = datetime.fromisoformat(approved_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(
                "Parent-quality gate approved_at must be an ISO-8601 timestamp"
            ) from error
        if parsed_approval.tzinfo is None or parsed_approval.utcoffset() is None:
            raise ValueError("Parent-quality gate approved_at must include a timezone")
        bundle_created_at = bundle.get("created_at")
        if not isinstance(bundle_created_at, str):
            raise ValueError("Parent promotion bundle lacks its creation timestamp")
        try:
            parsed_bundle_created_at = datetime.fromisoformat(
                bundle_created_at.replace("Z", "+00:00")
            )
        except ValueError as error:
            raise ValueError("Parent promotion bundle creation timestamp is invalid") from error
        if (
            parsed_bundle_created_at.tzinfo is None
            or parsed_bundle_created_at.utcoffset() is None
            or parsed_approval < parsed_bundle_created_at
        ):
            raise ValueError("Parent approval must occur after the promotion bundle was created")

        waivers = gate["waivers"]
        if not isinstance(waivers, list):
            raise ValueError("Parent-quality gate waivers must be a list")
        expected_deviation_sha = summary["deviation_sha256"]
        observed_waiver_ids: list[str] = []
        for waiver in waivers:
            if not isinstance(waiver, Mapping) or set(waiver) != {
                "id",
                "decision",
                "deviation_sha256",
            }:
                raise ValueError("Parent-quality gate waiver fields differ from the v2 schema")
            waiver_id = waiver["id"]
            if waiver_id not in promotion.REQUIRED_WAIVER_IDS or waiver["decision"] != "approved":
                raise ValueError("Parent-quality gate contains an unknown or unapproved waiver")
            if waiver["deviation_sha256"] != expected_deviation_sha.get(waiver_id):
                raise ValueError("Parent-quality gate waiver is not bound to its deviation")
            observed_waiver_ids.append(waiver_id)
        if observed_waiver_ids != sorted(promotion.REQUIRED_WAIVER_IDS):
            raise ValueError(
                "Parent-quality gate must explicitly approve exactly the two locked deviations"
            )
    elif gate_version == 3:
        from olmo_core.eval import (
            vision_alignment_perception_promotion as perception_promotion,
        )

        if actual_gate_sha != (
            perception_promotion.EXPECTED_APPROVED_PERCEPTION_PARENT_GATE_RAW_SHA256
        ):
            raise ValueError("Parent-quality gate is not the exact approved perception gate")
        bundle_path = Path(str(gate["promotion_bundle_path"])).expanduser().resolve()
        expected_bundle_sha = gate["promotion_bundle_sha256"]
        if (
            not isinstance(expected_bundle_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_bundle_sha) is None
        ):
            raise ValueError("Perception parent gate promotion bundle SHA-256 is invalid")
        if gate["metrics_artifact_sha256"] != expected_bundle_sha:
            raise ValueError(
                "Perception parent gate metrics artifact must be the pinned promotion bundle"
            )
        try:
            bundle_raw = bundle_path.read_bytes()
            actual_bundle_sha = hashlib.sha256(bundle_raw).hexdigest()
            bundle = json.loads(bundle_raw, object_pairs_hook=_strict_json_object)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError(
                f"Invalid perception parent promotion bundle {bundle_path}: {error}"
            ) from error
        if actual_bundle_sha != expected_bundle_sha:
            raise ValueError(
                f"Perception parent promotion bundle SHA mismatch: configured "
                f"{expected_bundle_sha}, actual {actual_bundle_sha}"
            )
        if actual_bundle_sha != (
            perception_promotion.EXPECTED_APPROVED_PERCEPTION_PROMOTION_BUNDLE_RAW_SHA256
        ):
            raise ValueError("Perception parent promotion bundle is not the exact approved bundle")
        if not isinstance(bundle, Mapping):
            raise ValueError("Perception parent promotion bundle must be a JSON object")
        try:
            perception_promotion.validate_approved_perception_parent_gate_bundle(
                bundle,
                gate=gate,
                expected_checkpoint=Path(parent).resolve(),
                expected_checkpoint_config_sha256=parent_config_sha256,
            )
        except perception_promotion.PromotionValidationError as error:
            raise ValueError(
                f"Perception parent promotion bundle failed validation: {error}"
            ) from error
    elif gate_version == 4:
        from olmo_core.eval import vision_alignment_ssmax_bridge as ssmax_bridge

        try:
            ssmax_bridge.validate_ssmax_bridge_parent_gate(
                gate,
                expected_checkpoint=Path(parent).resolve(),
                expected_checkpoint_config_sha256=parent_config_sha256,
                expected_model_variant=model_variant.value,
                expected_data_contract_sha256=str(parent_meta.get("data_contract_sha256")),
                expected_trainable_contract_sha256=str(
                    parent_meta.get("trainable_contract_sha256")
                ),
            )
        except ssmax_bridge.SSMaxBridgeEvidenceError as error:
            raise ValueError(f"SSMax bridge parent gate failed validation: {error}") from error
    elif gate_version in (5, 6):
        from olmo_core.eval import vision_alignment_ssmax_perception as ssmax_perception

        try:
            ssmax_perception.validate_ssmax_perception_parent_gate(
                gate,
                expected_checkpoint=Path(parent).resolve(),
                expected_checkpoint_config_sha256=parent_config_sha256,
                expected_model_variant=model_variant.value,
                expected_data_contract_sha256=str(parent_meta.get("data_contract_sha256")),
                expected_trainable_contract_sha256=str(
                    parent_meta.get("trainable_contract_sha256")
                ),
            )
        except ssmax_perception.SSMaxPerceptionEvidenceError as error:
            raise ValueError(f"SSMax perception parent gate failed validation: {error}") from error
    elif gate_version == 7:
        from olmo_core.eval import (
            vision_alignment_ssmax_perception_direct as ssmax_perception_direct,
        )

        try:
            ssmax_perception_direct.validate_ssmax_perception_direct_parent_gate(
                gate,
                expected_checkpoint=Path(parent).resolve(),
                expected_checkpoint_config_sha256=parent_config_sha256,
                expected_model_variant=model_variant.value,
                expected_data_contract_sha256=str(parent_meta.get("data_contract_sha256")),
                expected_trainable_contract_sha256=str(
                    parent_meta.get("trainable_contract_sha256")
                ),
            )
        except ssmax_perception_direct.SSMaxPerceptionDirectEvidenceError as error:
            raise ValueError(
                f"SSMax direct perception parent gate failed validation: {error}"
            ) from error
    elif gate_version == 8:
        from olmo_core.eval import (
            vision_alignment_ssmax_perception_exploratory as ssmax_perception_exploratory,
        )

        try:
            ssmax_perception_exploratory.validate_ssmax_perception_exploratory_parent_gate(
                gate,
                expected_checkpoint=Path(parent).resolve(),
                expected_checkpoint_config_sha256=parent_config_sha256,
                expected_model_variant=model_variant.value,
                expected_data_contract_sha256=str(parent_meta.get("data_contract_sha256")),
                expected_trainable_contract_sha256=str(
                    parent_meta.get("trainable_contract_sha256")
                ),
            )
        except ssmax_perception_exploratory.SSMaxPerceptionExploratoryEvidenceError as error:
            raise ValueError(
                f"SSMax exploratory perception parent gate failed validation: {error}"
            ) from error
    elif gate_version == 9:
        from olmo_core.eval import (
            vision_alignment_ssmax_perception_exploratory_waiver as ssmax_perception_exploratory_waiver,
        )

        try:
            ssmax_perception_exploratory_waiver.validate_ssmax_perception_exploratory_waiver_parent_gate(
                gate,
                expected_checkpoint=Path(parent).resolve(),
                expected_checkpoint_config_sha256=parent_config_sha256,
                expected_model_variant=model_variant.value,
                expected_data_contract_sha256=str(parent_meta.get("data_contract_sha256")),
                expected_trainable_contract_sha256=str(
                    parent_meta.get("trainable_contract_sha256")
                ),
            )
        except (
            ssmax_perception_exploratory_waiver.SSMaxPerceptionExploratoryWaiverEvidenceError
        ) as error:
            raise ValueError(
                f"SSMax exploratory-waiver perception parent gate failed validation: {error}"
            ) from error
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

    def __init__(
        self,
        dataset: Any,
        source_name: str,
        audit: Mapping[str, Any],
        token_ids: Optional[Molmo2TokenIds] = None,
    ):
        self._dataset = dataset
        self.source_name = source_name
        source = cast(Mapping[str, Any], cast(Mapping[str, Any], audit["inputs"])[source_name])
        runtime_fingerprint = _runtime_dataset_fingerprint(dataset)
        if runtime_fingerprint != source.get("dataset_fingerprint"):
            raise ValueError(
                f"Live dataset fingerprint for {source_name!r} is {runtime_fingerprint!r}, "
                f"but its pinned audit records {source.get('dataset_fingerprint')!r}"
            )
        # Projection identity is defined at the immutable provenance/selection boundary.  The
        # audit remains a separately bound and validated wrapper, but must not make the exact
        # same selected rows hash differently from the offline calibration producer.
        self.ssmax_projection_base_content_fingerprint = runtime_fingerprint
        if len(dataset) != source.get("dataset_size"):
            raise ValueError(
                f"Live dataset length for {source_name!r} is {len(dataset)}, but its pinned "
                f"audit identifies {source.get('dataset_size')!r} examples"
            )
        probe_indices = cast(Sequence[int], source["probe_indices"])
        row_hashes = cast(Sequence[str], source["serialized_row_hashes"])
        probe_image_content_sha256 = source.get("probe_image_content_sha256")
        audit_format = audit.get("format")
        if audit_format == "vision_alignment_perception_source_audit":
            probe_epochs = cast(int, source["probe_epochs"])
            validate_image_content = getattr(dataset, "validate_image_content", None)
            if not callable(validate_image_content):
                raise ValueError(
                    f"Live perception dataset {source_name!r} lacks image-content validation"
                )
            if validate_image_content(probe_indices) != probe_image_content_sha256:
                raise ValueError(
                    f"Live perception probe image bytes for {source_name!r} differ from "
                    "the pinned source audit"
                )
            probe_pairs = tuple(
                (index, epoch) for epoch in range(probe_epochs) for index in probe_indices
            )
            if len(probe_pairs) != len(row_hashes):
                raise ValueError(
                    f"Perception probe epoch panel for {source_name!r} has inconsistent rows"
                )
            get = getattr(dataset, "get", None)
            live_loss_weight_total = 0.0
            for (probe_index, probe_epoch), row_hash in zip(probe_pairs, row_hashes):
                example = get(probe_index, probe_epoch) if callable(get) else dataset[probe_index]
                if serialized_example_sha256(example) != row_hash:
                    raise ValueError(
                        f"Live perception probe {source_name!r} serialized row differs at "
                        f"index {probe_index}, epoch {probe_epoch}"
                    )
                loss_masks = example.get("loss_masks")
                if loss_masks is None:
                    raise ValueError(f"Live perception probe {source_name!r} lacks loss masks")
                row_loss_weight = math.fsum(float(value) for value in loss_masks)
                if not math.isfinite(row_loss_weight) or row_loss_weight < 0:
                    raise ValueError(f"Live perception probe {source_name!r} has invalid loss mass")
                live_loss_weight_total = math.fsum((live_loss_weight_total, row_loss_weight))
            live_mean_loss_weight = live_loss_weight_total / len(probe_pairs)
            source_summary = cast(Mapping[str, Any], audit["sources"])[source_name]
            pinned_mean_loss_weight = source_summary.get("mean_sum_loss_masks")
            if (
                isinstance(pinned_mean_loss_weight, bool)
                or not isinstance(pinned_mean_loss_weight, (int, float))
                or not math.isclose(
                    live_mean_loss_weight,
                    float(pinned_mean_loss_weight),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    f"Live perception probe loss mass for {source_name!r} differs from "
                    "the pinned source audit"
                )
        elif audit_format == _JOINT_AUDIT_FORMAT:
            raw_joint_probe_epochs = source["probe_epochs"]
            expected_epochs = (
                _JOINT_NATIVE_PROBE_EPOCHS
                if source_name == "native_text_replay"
                else _JOINT_VISUAL_PROBE_EPOCHS
            )
            if (
                not isinstance(raw_joint_probe_epochs, list)
                or any(type(epoch) is not int for epoch in raw_joint_probe_epochs)
                or tuple(raw_joint_probe_epochs) != expected_epochs
            ):
                raise ValueError(
                    f"Joint probe epoch panel for {source_name!r} differs from the runtime policy"
                )
            joint_probe_epochs = cast(Sequence[int], raw_joint_probe_epochs)
            if source_name != "native_text_replay":
                validate_image_content = getattr(dataset, "validate_image_content", None)
                if not callable(validate_image_content):
                    raise ValueError(
                        f"Live joint visual dataset {source_name!r} lacks image validation"
                    )
                if validate_image_content(probe_indices) != probe_image_content_sha256:
                    raise ValueError(
                        f"Live joint probe image bytes for {source_name!r} differ from "
                        "the pinned source audit"
                    )
            elif probe_image_content_sha256 != _canonical_sha256([]):
                raise ValueError("Native joint replay has an invalid image-content digest")
            probe_pairs = tuple(
                (index, epoch) for epoch in joint_probe_epochs for index in probe_indices
            )
            if len(probe_pairs) != len(row_hashes):
                raise ValueError(
                    f"Joint probe epoch panel for {source_name!r} has inconsistent rows"
                )
            get = getattr(dataset, "get", None)
            live_loss_weight_total = 0.0
            live_zero_loss_examples = 0
            maximum_length = 0
            for (probe_index, probe_epoch), row_hash in zip(probe_pairs, row_hashes):
                example = get(probe_index, probe_epoch) if callable(get) else dataset[probe_index]
                validate_joint_live_example(
                    example,
                    source_name=source_name,
                    source_kind=(
                        "native_text_replay" if source_name == "native_text_replay" else "visual"
                    ),
                    token_ids=token_ids,
                )
                if serialized_example_sha256(example) != row_hash:
                    raise ValueError(
                        f"Live joint probe {source_name!r} serialized row differs at "
                        f"index {probe_index}, epoch {probe_epoch}"
                    )
                input_ids = example.get("input_ids")
                loss_masks = example.get("loss_masks")
                if input_ids is None or loss_masks is None:
                    raise ValueError(f"Live joint probe {source_name!r} lacks token/loss arrays")
                try:
                    sequence_length = len(input_ids)
                except TypeError as error:
                    raise ValueError(
                        f"Live joint probe {source_name!r} input_ids are not array-like"
                    ) from error
                if sequence_length < 1 or sequence_length > _JOINT_SEQUENCE_LENGTH:
                    raise ValueError(
                        f"Live joint probe {source_name!r} has invalid sequence length "
                        f"{sequence_length}"
                    )
                if source_name == "native_text_replay" and sequence_length != (
                    _JOINT_SEQUENCE_LENGTH
                ):
                    raise ValueError("Live native joint replay row is not exactly 8,192 tokens")
                maximum_length = max(maximum_length, sequence_length)
                row_loss_weight = math.fsum(float(value) for value in loss_masks)
                if not math.isfinite(row_loss_weight) or row_loss_weight < 0:
                    raise ValueError(f"Live joint probe {source_name!r} has invalid loss mass")
                live_zero_loss_examples += int(row_loss_weight == 0)
                live_loss_weight_total = math.fsum((live_loss_weight_total, row_loss_weight))
            if live_loss_weight_total <= 0:
                raise ValueError(
                    f"Live joint probe {source_name!r} has no aggregate supervised loss mass"
                )
            if maximum_length != source.get("max_observed_sequence_length"):
                raise ValueError(
                    f"Live joint probe sequence bound for {source_name!r} differs from "
                    "the pinned source audit"
                )
            live_mean_loss_weight = live_loss_weight_total / len(probe_pairs)
            source_summary = cast(Mapping[str, Any], audit["sources"])[source_name]
            pinned_mean_loss_weight = source_summary.get("mean_sum_loss_masks")
            pinned_zero_loss_examples = source_summary.get("zero_loss_examples")
            if (
                isinstance(pinned_mean_loss_weight, bool)
                or not isinstance(pinned_mean_loss_weight, (int, float))
                or isinstance(pinned_zero_loss_examples, bool)
                or not isinstance(pinned_zero_loss_examples, int)
                or live_zero_loss_examples != pinned_zero_loss_examples
                or not math.isclose(
                    live_mean_loss_weight,
                    float(pinned_mean_loss_weight),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    f"Live joint probe loss mass for {source_name!r} differs from "
                    "the pinned source audit"
                )
        else:
            validate_serialized_runtime_probe(dataset, probe_indices, row_hashes, epoch=0)
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
        if audit_format == _JOINT_AUDIT_FORMAT:
            fingerprint_payload["exporter_implementation"] = audit["exporter_implementation"]
        else:
            fingerprint_payload["exporter_sha256"] = audit["exporter_sha256"]
        if probe_image_content_sha256 is not None:
            fingerprint_payload["probe_image_content_sha256"] = probe_image_content_sha256
        if audit_format in {
            "vision_alignment_perception_source_audit",
            _JOINT_AUDIT_FORMAT,
        }:
            fingerprint_payload["probe_epochs"] = source["probe_epochs"]
        self.content_fingerprint = _canonical_sha256(fingerprint_payload)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.get(index, 0)

    def get(self, index: int, epoch: int = 0) -> Dict[str, Any]:
        get = getattr(self._dataset, "get", None)
        return get(index, epoch) if get is not None else self._dataset[index]


def _validate_image_path_signatures_parallel(
    manifest: PerceptionProvenanceManifest,
    *,
    workers: int = IMAGE_PATH_SIGNATURE_WORKERS,
    max_pending: int = IMAGE_PATH_SIGNATURE_MAX_PENDING,
) -> None:
    """Exhaustively restat pinned images with bounded, ordered concurrency.

    This preserves :meth:`PerceptionProvenanceManifest.validate_image_path_signatures`
    exactly while avoiding a serial metadata walk over more than one million Weka paths.
    Results are consumed in manifest order so the same earliest bad path wins regardless of
    completion order. Any failure cancels queued work and waits for running calls to quiesce
    before returning control to the caller.
    """
    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or workers < 1
        or workers > IMAGE_PATH_SIGNATURE_WORKERS
    ):
        raise ValueError(
            f"Image path-signature workers must be an integer in [1, {IMAGE_PATH_SIGNATURE_WORKERS}]"
        )
    if (
        isinstance(max_pending, bool)
        or not isinstance(max_pending, int)
        or max_pending < workers
        or max_pending > IMAGE_PATH_SIGNATURE_MAX_PENDING
    ):
        raise ValueError(
            "Image path-signature pending work must be an integer between workers and "
            f"{IMAGE_PATH_SIGNATURE_MAX_PENDING}"
        )

    def restat(
        record: PerceptionPathSignature,
    ) -> tuple[Path, Optional[OSError], bool]:
        path = Path(record.path)
        try:
            info = path.stat()
        except OSError as error:
            return path, error, False
        actual = (
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
            info.st_ino,
            info.st_dev,
        )
        expected = (
            record.size_bytes,
            record.mtime_ns,
            record.ctime_ns,
            record.inode,
            record.device,
        )
        return path, None, not stat.S_ISREG(info.st_mode) or actual != expected

    records = iter(manifest.image_path_signatures)
    pending: deque[
        tuple[
            PerceptionPathSignature,
            Future[tuple[Path, Optional[OSError], bool]],
        ]
    ] = deque()
    executor = ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="vision-image-signature",
    )
    try:
        for _ in range(max_pending):
            try:
                record = next(records)
            except StopIteration:
                break
            pending.append((record, executor.submit(restat, record)))
        while pending:
            _, future = pending.popleft()
            path, unavailable, changed = future.result()
            if unavailable is not None:
                raise ValueError(f"Pinned perception image is unavailable: {path}") from unavailable
            if changed:
                raise ValueError(f"Pinned perception image signature changed: {path}")
            try:
                record = next(records)
            except StopIteration:
                continue
            pending.append((record, executor.submit(restat, record)))
    except BaseException:
        for _, future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)


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
                _validate_image_path_signatures_parallel(manifest.parent_provenance)
                # Close the long path-restat window with a second exact artifact/code check.
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
    if config.phase is VisionAlignmentPhase.perception:
        return _perception_provenance(config).source_spec_sha256
    if config.phase is VisionAlignmentPhase.joint:
        return _joint_visual_projection(config).source_spec_sha256
    return _source_spec(config).preprocessing_sha256


def _ssmax_single_response_calibration(
    config: ExperimentConfig,
    source_audit: Mapping[str, Any],
) -> Optional[Mapping[str, Any]]:
    """Load and semantically validate the immutable projected loss-mass receipt."""

    projection = config.data.ssmax_single_response_projection
    if projection is None:
        return None
    assert projection.calibration_path is not None
    assert projection.calibration_sha256 is not None
    calibration_path = Path(projection.calibration_path).expanduser().resolve()
    try:
        raw = calibration_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != projection.calibration_sha256:
            raise ValueError("raw SHA-256 differs")
        payload = json.loads(raw, object_pairs_hook=_strict_json_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(
            f"Invalid SSMax single-response calibration {calibration_path}: {error}"
        ) from error
    source_audit_path = Path(str(config.data.source_audit_path)).expanduser().resolve()
    source_audit_ref = {
        "path": str(source_audit_path),
        "raw_sha256": _sha256_file(source_audit_path),
        "content_sha256": source_audit["fingerprint"],
    }
    if config.phase is VisionAlignmentPhase.perception:
        selection = _perception_provenance(config)
        selection_ref = {
            "path": str(selection.path),
            "raw_sha256": selection.raw_sha256,
            "content_sha256": selection.content_sha256,
        }
        visual_sources = tuple(PERCEPTION_SOURCE_NAMES)
        unprojected_sources: Tuple[str, ...] = ()
    elif config.phase is VisionAlignmentPhase.joint:
        selection = _joint_visual_projection(config)
        selection_ref = {
            "path": str(selection.path),
            "raw_sha256": selection.raw_sha256,
            "content_sha256": selection.content_sha256,
        }
        visual_sources = tuple(JOINT_VISUAL_SOURCE_NAMES)
        unprojected_sources = ("native_text_replay",)
    else:
        raise ValueError("SSMax single-response calibration is forbidden for the bridge")
    try:
        validated = validate_ssmax_single_response_calibration(
            payload,
            expected_phase=config.phase.value,
            expected_contract=projection.contract(
                loss_token_weighting=config.data.loss_token_weighting
            ),
            expected_source_audit=source_audit_ref,
            expected_selection_manifest=selection_ref,
            expected_visual_sources=visual_sources,
            expected_unprojected_sources=unprojected_sources,
            expected_mean_loss_weight=projection.projected_mean_loss_weight,
            expected_validation_rows_per_source={
                source: config.evaluation.examples_per_source for source in visual_sources
            },
        )
    except ValueError as error:
        raise ValueError(f"SSMax projection calibration failed validation: {error}") from error
    repository_root = Path(__file__).resolve().parents[3]
    for reference_name in ("producer", "projection_implementation"):
        reference = cast(Mapping[str, Any], validated[reference_name])
        implementation_path = (repository_root / str(reference["path"])).resolve()
        if (
            not implementation_path.is_relative_to(repository_root)
            or not implementation_path.is_file()
            or _sha256_file(implementation_path) != reference["sha256"]
        ):
            raise ValueError(
                f"SSMax projection calibration {reference_name} implementation differs"
            )
    return validated


def _ssmax_projection_audit_panel(
    audit: Mapping[str, Any], source_name: str
) -> Tuple[Tuple[int, int], ...]:
    source = cast(Mapping[str, Any], cast(Mapping[str, Any], audit["inputs"])[source_name])
    indices = cast(Sequence[int], source["probe_indices"])
    raw_epochs = source["probe_epochs"]
    if audit.get("format") == "vision_alignment_perception_source_audit":
        epochs = tuple(range(cast(int, raw_epochs)))
    elif audit.get("format") == _JOINT_AUDIT_FORMAT:
        epochs = tuple(cast(Sequence[int], raw_epochs))
    else:
        raise ValueError("SSMax projection calibration requires perception/joint source audit")
    return tuple((int(index), int(epoch)) for epoch in epochs for index in indices)


def _validate_live_ssmax_projection_summary(
    calibration: Mapping[str, Any],
    dataset: Any,
    source_name: str,
    *,
    section: str,
    logical_split: str,
    panel: Optional[Sequence[Tuple[int, int]]] = None,
) -> None:
    """Rebuild one complete panel on rank 0 after cheap identity checks on every rank."""

    import torch.distributed as dist

    sources = cast(Mapping[str, Any], calibration[section])
    if source_name not in sources:
        if source_name in calibration["unprojected_sources"]:
            return
        raise ValueError(f"SSMax {section} lacks source {source_name!r}")
    expected = cast(Mapping[str, Any], sources[source_name])
    local_error: Optional[str] = None
    try:
        if not isinstance(dataset, SSMaxSingleResponseDataset):
            raise TypeError("runtime dataset is not SSMaxSingleResponseDataset")
        if dataset.source_name != source_name or dataset.logical_split != logical_split:
            raise ValueError("runtime source/split identity differs")
        if dict(dataset.contract) != dict(calibration["projection_contract"]):
            raise ValueError("runtime projection contract differs")
        if dataset.content_fingerprint != expected["dataset_content_fingerprint"]:
            raise ValueError("runtime projected dataset fingerprint differs")
        if logical_split != "train" and len(dataset) != expected["rows"]:
            raise ValueError("runtime validation row count differs")
    except Exception as error:  # noqa: BLE001 - every rank must reach the collective
        local_error = f"rank {get_rank()}: {type(error).__name__}: {error}"

    distributed = dist.is_available() and dist.is_initialized()
    if distributed:
        rank_errors: List[Optional[str]] = [None] * dist.get_world_size()
        dist.all_gather_object(rank_errors, local_error)
        failures = [error for error in rank_errors if error is not None]
    else:
        failures = [local_error] if local_error is not None else []
    if failures:
        raise ValueError("SSMax projection rank-local identity failed: " + "; ".join(failures))

    replay_error: Optional[str] = None
    if get_rank() == 0:
        try:
            if panel is None:
                raise ValueError("rank-zero projection replay lacks its immutable panel")
            summary = ssmax_single_response_calibration_summary(dataset, panel)
            if summary != expected:
                differing = sorted(
                    key
                    for key in set(summary) | set(expected)
                    if summary.get(key) != expected.get(key)
                )
                raise ValueError(f"summary differs in {differing}")
        except Exception as error:  # noqa: BLE001 - broadcast instead of stranding peers
            replay_error = f"{type(error).__name__}: {error}"
    if distributed:
        result = [replay_error]
        dist.broadcast_object_list(result, src=0)
        replay_error = cast(Optional[str], result[0])
    if replay_error is not None:
        raise ValueError(
            f"Live SSMax single-response {section} for {source_name!r} failed: {replay_error}"
        )


def _validate_live_ssmax_projection_calibration(
    calibration: Mapping[str, Any],
    audit: Mapping[str, Any],
    dataset: Any,
    source_name: str,
) -> None:
    """Rebuild one complete projected training audit panel once globally."""

    panel = _ssmax_projection_audit_panel(audit, source_name) if get_rank() == 0 else None
    _validate_live_ssmax_projection_summary(
        calibration,
        dataset,
        source_name,
        section="sources",
        logical_split="train",
        panel=panel,
    )


def _validate_live_ssmax_projection_validation(
    calibration: Mapping[str, Any], dataset: Any, source_name: str
) -> None:
    """Rebuild the complete fixed held-out projection once globally."""

    panel = tuple((index, 0) for index in range(len(dataset))) if get_rank() == 0 else None
    _validate_live_ssmax_projection_summary(
        calibration,
        dataset,
        source_name,
        section="validation_preflight",
        logical_split="validation",
        panel=panel,
    )


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
        audit_raw = audit_path.read_bytes()
        audit = json.loads(audit_raw, object_pairs_hook=_strict_json_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Invalid vision-alignment source audit {audit_path}: {error}") from error
    if not isinstance(audit, dict):
        raise ValueError(f"Vision-alignment source audit must be an object: {audit_path}")
    if config.phase is VisionAlignmentPhase.perception:
        return _validate_perception_source_audit(config, audit_path, audit)
    if config.phase is VisionAlignmentPhase.joint:
        return _validate_joint_source_audit(
            config,
            audit_path,
            audit,
            expected_audit_raw_sha256=hashlib.sha256(audit_raw).hexdigest(),
        )
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
    if (
        not isinstance(expected_loss_mass, Mapping)
        or set(expected_loss_mass) != set(targets)
        or any(
            isinstance(expected_loss_mass[source_name], bool)
            or not isinstance(expected_loss_mass[source_name], (int, float))
            or not math.isfinite(float(expected_loss_mass[source_name]))
            or not math.isclose(
                float(expected_loss_mass[source_name]),
                float(target_weight),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for source_name, target_weight in targets.items()
        )
    ):
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


def _joint_audit_mapping(value: Any, fields: frozenset[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    actual = set(value)
    if actual != fields:
        raise ValueError(
            f"{name} fields differ: missing={sorted(fields - actual)}, "
            f"extra={sorted(actual - fields)}"
        )
    return value


def _joint_audit_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _joint_audit_count(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _joint_audit_absolute_file(value: Any, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty absolute path")
    unresolved = Path(value).expanduser()
    if not unresolved.is_absolute() or ".." in unresolved.parts:
        raise ValueError(f"{name} must be an absolute normalized path without traversal")
    path = unresolved.resolve()
    if str(path) != value or not path.is_file():
        raise ValueError(f"{name} must identify a normalized existing file")
    return path


def _validate_joint_source_audit(
    config: ExperimentConfig,
    audit_path: Path,
    audit: Mapping[str, Any],
    *,
    expected_audit_raw_sha256: str,
) -> Mapping[str, Any]:
    """Validate the joint auditor's exact nine-source version-1 report."""
    if set(audit) != _JOINT_AUDIT_FIELDS:
        raise ValueError("Joint source-audit fields differ from the locked schema")
    unsigned = dict(audit)
    recorded_fingerprint = unsigned.pop("fingerprint", None)
    computed_fingerprint = _canonical_sha256(unsigned)
    if (
        recorded_fingerprint != computed_fingerprint
        or config.data.source_audit_fingerprint != computed_fingerprint
    ):
        raise ValueError("Joint source-audit fingerprint differs")

    auditor = _joint_audit_mapping(
        audit["auditor_implementation"],
        _JOINT_IMPLEMENTATION_FIELDS,
        name="joint audit auditor_implementation",
    )
    exporter = _joint_audit_mapping(
        audit["exporter_implementation"],
        _JOINT_IMPLEMENTATION_FIELDS,
        name="joint audit exporter_implementation",
    )
    repo_root = Path(__file__).resolve().parents[3]
    auditor_path = repo_root / _JOINT_AUDITOR_IMPLEMENTATION
    exporter_path = repo_root / _JOINT_EXPORTER_IMPLEMENTATION
    shared_auditor_path = repo_root / _JOINT_SHARED_AUDITOR_IMPLEMENTATION
    if (
        audit.get("format") != _JOINT_AUDIT_FORMAT
        or type(audit.get("version")) is not int
        or audit["version"] != _JOINT_AUDIT_VERSION
        or audit.get("status") != "ok"
        or audit.get("phase") != "joint"
        or type(audit.get("recipe_version")) is not int
        or audit.get("recipe_version") != RECIPE_VERSION
        or audit.get("formatter_version") != FORMATTER_VERSION
        or type(audit.get("source_catalog_version")) is not int
        or audit.get("source_catalog_version") != VISION_ALIGNMENT_JOINT_SOURCE_CATALOG_VERSION
        or auditor
        != {
            "path": _JOINT_AUDITOR_IMPLEMENTATION,
            "sha256": _sha256_file(auditor_path),
        }
        or audit.get("shared_auditor_sha256") != _sha256_file(shared_auditor_path)
        or exporter
        != {
            "path": _JOINT_EXPORTER_IMPLEMENTATION,
            "sha256": _sha256_file(exporter_path),
        }
        or type(audit.get("source_registry_version")) is not int
        or audit.get("source_registry_version") != VISION_ALIGNMENT_JOINT_SOURCE_REGISTRY_VERSION
        or audit.get("source_registry_sha256") != joint_alignment_runtime_registry_sha256()
        or _canonical_sha256(audit.get("source_implementation_inventory"))
        != _canonical_sha256(joint_alignment_runtime_implementation_inventory())
        or audit.get("failures") != []
    ):
        raise ValueError("Joint source-audit implementation, identity, or status differs")

    catalog_path = _joint_audit_absolute_file(
        audit.get("catalog_path"), name="joint audit catalog_path"
    )
    if audit.get("catalog_sha256") != _sha256_file(catalog_path):
        raise ValueError("Joint source-audit catalog bytes differ")
    _joint_audit_sha256(
        audit.get("catalog_content_sha256"), name="joint audit catalog_content_sha256"
    )

    projection = _joint_visual_projection(config)
    projection_ref = _joint_audit_mapping(
        audit["visual_projection"],
        _JOINT_VISUAL_PROJECTION_FIELDS,
        name="joint audit visual_projection",
    )
    if projection_ref != {
        "path": str(projection.path),
        "raw_sha256": projection.raw_sha256,
        "content_sha256": projection.content_sha256,
    }:
        raise ValueError("Joint source audit identifies a different visual projection")

    replay = config.data.native_text_replay
    if replay is None or config.data.native_text_replay_fingerprint is None:
        raise ValueError("Joint source audit requires configured native train replay")
    native_path = _joint_audit_absolute_file(
        replay.manifest_path, name="native train manifest path"
    )
    native_ref = _joint_audit_mapping(
        audit["native_train_manifest"],
        _JOINT_NATIVE_MANIFEST_FIELDS,
        name="joint audit native_train_manifest",
    )
    expected_native_ref = {
        "path": str(native_path),
        "raw_sha256": _sha256_file(native_path),
        "content_fingerprint": config.data.native_text_replay_fingerprint,
    }
    if (
        native_ref != expected_native_ref
        or replay.expected_fingerprint != expected_native_ref["content_fingerprint"]
    ):
        raise ValueError("Joint source audit identifies a different native train manifest")
    if (
        replay.verification_receipt_path is None
        or replay.expected_verification_receipt_sha256 is None
    ):
        raise ValueError("Joint source audit requires a pinned native verification receipt")
    receipt_path = _joint_audit_absolute_file(
        replay.verification_receipt_path, name="native verification receipt path"
    )
    receipt_ref = _joint_audit_mapping(
        audit["native_verification_receipt"],
        _JOINT_RECEIPT_FIELDS,
        name="joint audit native_verification_receipt",
    )
    expected_receipt_ref = {
        "path": str(receipt_path),
        "sha256": replay.expected_verification_receipt_sha256,
    }
    if (
        receipt_ref != expected_receipt_ref
        or _sha256_file(receipt_path) != replay.expected_verification_receipt_sha256
    ):
        raise ValueError("Joint source audit identifies a different native verification receipt")

    preprocessing = _joint_audit_mapping(
        audit["preprocessing"],
        _JOINT_PREPROCESSING_FIELDS,
        name="joint audit preprocessing",
    )
    expected_preprocessing = {
        "visual": projection.source_spec.as_canonical_dict(),
        "native_text_replay_fingerprint": config.data.native_text_replay_fingerprint,
    }
    if _canonical_sha256(preprocessing) != _canonical_sha256(expected_preprocessing) or audit.get(
        "preprocessing_sha256"
    ) != _canonical_sha256(expected_preprocessing):
        raise ValueError("Joint source-audit preprocessing identity differs from training")

    probe = _joint_audit_mapping(audit["probe"], _JOINT_PROBE_FIELDS, name="joint audit probe")
    visual_probe = _joint_audit_mapping(
        probe["visual"], _JOINT_PROBE_KIND_FIELDS, name="joint audit probe.visual"
    )
    native_probe = _joint_audit_mapping(
        probe["native_text_replay"],
        _JOINT_PROBE_KIND_FIELDS,
        name="joint audit probe.native_text_replay",
    )
    expected_probe = {
        "format": _JOINT_PROBE_FORMAT,
        "version": _JOINT_PROBE_VERSION,
        "selection_algorithm": VISION_ALIGNMENT_PROBE_SELECTION_ALGORITHM,
        "seed": _JOINT_PROBE_SEED,
        "visual": {
            "unique_indices": _JOINT_VISUAL_PROBE_INDICES,
            "epochs": list(_JOINT_VISUAL_PROBE_EPOCHS),
            "rows_per_source": _JOINT_VISUAL_PROBE_INDICES * len(_JOINT_VISUAL_PROBE_EPOCHS),
        },
        "native_text_replay": {
            "unique_indices": _JOINT_NATIVE_PROBE_INDICES,
            "epochs": list(_JOINT_NATIVE_PROBE_EPOCHS),
            "rows_per_source": _JOINT_NATIVE_PROBE_INDICES * len(_JOINT_NATIVE_PROBE_EPOCHS),
        },
        "sequence_length": _JOINT_SEQUENCE_LENGTH,
        "truncation_policy": "forbid-raw-length-above-sequence-length-v1",
    }
    if (
        _canonical_sha256(probe) != _canonical_sha256(expected_probe)
        or _canonical_sha256(visual_probe) != _canonical_sha256(expected_probe["visual"])
        or _canonical_sha256(native_probe)
        != _canonical_sha256(expected_probe["native_text_replay"])
    ):
        raise ValueError("Joint source-audit probe identity or epoch panel differs")

    targets = config.data.mixture.resolved_targets()
    means = config.data.mixture.mean_loss_weight
    sampling = config.data.mixture.sampling_weights()
    if tuple(sorted(targets)) != _JOINT_SOURCE_NAMES or set(means) != set(targets):
        raise ValueError("Joint training mixture does not contain the exact nine sources")
    for field_name, expected in (
        ("target_loss_mass", targets),
        ("mean_loss_weight", means),
        ("sampling_probabilities", sampling),
    ):
        actual = audit.get(field_name)
        if not isinstance(actual, Mapping) or _canonical_sha256(actual) != _canonical_sha256(
            expected
        ):
            raise ValueError(f"Joint source-audit {field_name} differs from training")
    expected_mass = expected_loss_mass(sampling, means)
    actual_mass = audit.get("expected_loss_mass")
    if (
        not isinstance(actual_mass, Mapping)
        or set(actual_mass) != set(expected_mass)
        or any(
            isinstance(actual_mass[name], bool)
            or not isinstance(actual_mass[name], (int, float))
            or not math.isfinite(float(actual_mass[name]))
            or not math.isclose(
                float(actual_mass[name]), float(expected_mass[name]), rel_tol=0.0, abs_tol=1e-12
            )
            for name in expected_mass
        )
    ):
        raise ValueError("Joint source-audit expected loss mass differs from targets")

    inputs = audit.get("inputs")
    summaries = audit.get("sources")
    if (
        not isinstance(inputs, Mapping)
        or not isinstance(summaries, Mapping)
        or tuple(sorted(inputs)) != _JOINT_SOURCE_NAMES
        or tuple(sorted(summaries)) != _JOINT_SOURCE_NAMES
    ):
        raise ValueError("Joint source-audit source set differs")
    input_descriptors: List[Dict[str, Any]] = []
    pinned_probe_files: List[Tuple[Path, str]] = []
    for source_name in _JOINT_SOURCE_NAMES:
        source = _joint_audit_mapping(
            inputs[source_name],
            _JOINT_AUDIT_INPUT_FIELDS,
            name=f"joint audit input {source_name!r}",
        )
        kind = "native_text_replay" if source_name == "native_text_replay" else "visual"
        expected_epochs = (
            _JOINT_NATIVE_PROBE_EPOCHS
            if kind == "native_text_replay"
            else _JOINT_VISUAL_PROBE_EPOCHS
        )
        expected_indices_count = (
            _JOINT_NATIVE_PROBE_INDICES
            if kind == "native_text_replay"
            else _JOINT_VISUAL_PROBE_INDICES
        )
        expected_rows = expected_indices_count * len(expected_epochs)
        dataset_size = _joint_audit_count(
            source.get("dataset_size"), name=f"{source_name}.dataset_size", minimum=1
        )
        fingerprint = _joint_audit_sha256(
            source.get("dataset_fingerprint"), name=f"{source_name}.dataset_fingerprint"
        )
        raw_probe_epochs = source.get("probe_epochs")
        probe_indices = source.get("probe_indices")
        row_hashes = source.get("serialized_row_hashes")
        if (
            source.get("name") != source_name
            or source.get("kind") != kind
            or source.get("format") != "jsonl"
            or source.get("path") != f"{source_name}.jsonl"
            or not isinstance(raw_probe_epochs, list)
            or any(type(epoch) is not int for epoch in raw_probe_epochs)
            or tuple(raw_probe_epochs) != expected_epochs
            or not isinstance(probe_indices, list)
            or len(probe_indices) != expected_indices_count
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= dataset_size
                for index in probe_indices
            )
            or len(set(probe_indices)) != len(probe_indices)
            or tuple(probe_indices)
            != select_deterministic_probe_indices(
                dataset_size,
                expected_indices_count,
                seed=_JOINT_PROBE_SEED,
                dataset_fingerprint=fingerprint,
            )
            or source.get("probe_indices_sha256") != _canonical_sha256(probe_indices)
            or not isinstance(row_hashes, list)
            or len(row_hashes) != expected_rows
            or any(
                not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
                for value in row_hashes
            )
            or source.get("serialized_row_hashes_sha256") != _canonical_sha256(row_hashes)
        ):
            raise ValueError(f"Joint audit source {source_name!r} runtime probe differs")
        source_sha = _joint_audit_sha256(source.get("sha256"), name=f"{source_name}.sha256")
        image_sha = _joint_audit_sha256(
            source.get("probe_image_content_sha256"),
            name=f"{source_name}.probe_image_content_sha256",
        )
        maximum_length = _joint_audit_count(
            source.get("max_observed_sequence_length"),
            name=f"{source_name}.max_observed_sequence_length",
            minimum=1,
        )
        truncated_rows = _joint_audit_count(
            source.get("truncated_rows"), name=f"{source_name}.truncated_rows"
        )
        if (
            maximum_length > _JOINT_SEQUENCE_LENGTH
            or (kind == "native_text_replay" and maximum_length != _JOINT_SEQUENCE_LENGTH)
            or truncated_rows != 0
            or (kind == "native_text_replay" and image_sha != _canonical_sha256([]))
        ):
            raise ValueError(
                f"Joint audit source {source_name!r} truncation/image evidence differs"
            )
        source_path = (catalog_path.parent / str(source["path"])).resolve()
        if (
            source_path.parent != catalog_path.parent
            or not source_path.is_file()
            or _sha256_file(source_path) != source_sha
        ):
            raise ValueError(f"Joint audit source {source_name!r} probe bytes differ")
        pinned_probe_files.append((source_path, source_sha))

        summary = _joint_audit_mapping(
            summaries[source_name],
            _JOINT_SOURCE_SUMMARY_FIELDS,
            name=f"joint audit summary {source_name!r}",
        )
        if summary.get("examples") != {
            "seen": expected_rows,
            "valid": expected_rows,
            "errors": 0,
        }:
            raise ValueError(f"Joint audit summary {source_name!r} example counts differ")
        metric_summaries: Dict[str, Mapping[str, Any]] = {}
        for metric_name in (
            "raw_input_tokens",
            "positive_supervised_tokens",
            "summed_loss_weight",
            "image_crops",
        ):
            metric = _joint_audit_mapping(
                summary[metric_name],
                _JOINT_METRIC_SUMMARY_FIELDS,
                name=f"joint audit summary {source_name!r}.{metric_name}",
            )
            if any(
                isinstance(metric[field_name], bool)
                or not isinstance(metric[field_name], (int, float))
                or not math.isfinite(float(metric[field_name]))
                or float(metric[field_name]) < 0
                for field_name in _JOINT_METRIC_SUMMARY_FIELDS
            ):
                raise ValueError(f"Joint audit summary {source_name!r}.{metric_name} is not finite")
            metric_summaries[metric_name] = metric
        calibrated_mean = summary.get("mean_sum_loss_masks")
        truncated_examples = _joint_audit_count(
            summary.get("truncated_examples"),
            name=f"{source_name}.truncated_examples",
        )
        zero_loss_examples = _joint_audit_count(
            summary.get("zero_loss_examples"),
            name=f"{source_name}.zero_loss_examples",
        )
        loss_metric = metric_summaries["summed_loss_weight"]
        if (
            truncated_examples != 0
            or zero_loss_examples > expected_rows
            or (kind == "visual" and zero_loss_examples != 0)
            or (zero_loss_examples > 0) != (float(loss_metric["min"]) == 0.0)
            or summary.get("error_samples") != []
            or isinstance(calibrated_mean, bool)
            or not isinstance(calibrated_mean, (int, float))
            or not math.isfinite(float(calibrated_mean))
            or float(calibrated_mean) <= 0
            or not math.isclose(
                float(loss_metric["mean"]),
                float(calibrated_mean),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(loss_metric["total"]),
                float(calibrated_mean) * expected_rows,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                float(calibrated_mean),
                float(means[source_name]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(f"Joint audit summary {source_name!r} calibration differs")
        input_descriptors.append(
            {
                "name": source_name,
                "kind": kind,
                "sha256": source_sha,
                "dataset_fingerprint": fingerprint,
                "probe_indices_sha256": source["probe_indices_sha256"],
                "probe_epochs": list(expected_epochs),
                "serialized_row_hashes_sha256": source["serialized_row_hashes_sha256"],
                "probe_image_content_sha256": image_sha,
                "max_observed_sequence_length": maximum_length,
                "truncated_rows": 0,
            }
        )
    if audit.get("input_content_sha256") != _canonical_sha256(input_descriptors):
        raise ValueError("Joint source-audit input-content identity differs")

    # Close the path-restat window after every dependent probe file has been checked.
    if (
        _sha256_file(audit_path) != expected_audit_raw_sha256
        or _sha256_file(catalog_path) != audit["catalog_sha256"]
        or _sha256_file(auditor_path) != auditor["sha256"]
        or _sha256_file(exporter_path) != exporter["sha256"]
        or _sha256_file(shared_auditor_path) != audit["shared_auditor_sha256"]
        or _sha256_file(projection.path) != projection.raw_sha256
        or _sha256_file(native_path) != native_ref["raw_sha256"]
        or _sha256_file(receipt_path) != receipt_ref["sha256"]
        or any(_sha256_file(path) != sha256 for path, sha256 in pinned_probe_files)
        or joint_alignment_runtime_registry_sha256() != audit["source_registry_sha256"]
        or _canonical_sha256(joint_alignment_runtime_implementation_inventory())
        != _canonical_sha256(audit["source_implementation_inventory"])
    ):
        raise ValueError("Joint source-audit inputs changed during launcher validation")
    return audit


def _validate_perception_source_audit(
    config: ExperimentConfig,
    audit_path: Path,
    audit: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate the separate, provenance-bound eight-source perception audit."""
    if set(audit) != _PERCEPTION_AUDIT_FIELDS:
        raise ValueError("Perception source-audit fields differ from the locked schema")
    unsigned = dict(audit)
    recorded_fingerprint = unsigned.pop("fingerprint", None)
    computed_fingerprint = _canonical_sha256(unsigned)
    if (
        recorded_fingerprint != computed_fingerprint
        or config.data.source_audit_fingerprint != computed_fingerprint
    ):
        raise ValueError("Perception source-audit fingerprint differs")
    auditor_path = (
        Path(__file__).resolve().parents[1] / "data" / "audit_vision_alignment_perception_mix.py"
    )
    shared_auditor_path = (
        Path(__file__).resolve().parents[1] / "data" / "audit_vision_alignment_mix.py"
    )
    exporter_path = (
        Path(__file__).resolve().parents[1] / "data" / "export_vision_alignment_perception_probe.py"
    )
    if (
        audit.get("format") != "vision_alignment_perception_source_audit"
        or audit.get("version") != 2
        or audit.get("status") != "ok"
        or audit.get("phase") != "perception"
        or audit.get("recipe_version") != RECIPE_VERSION
        or audit.get("formatter_version") != FORMATTER_VERSION
        or audit.get("source_catalog_version") != VISION_ALIGNMENT_PERCEPTION_SOURCE_CATALOG_VERSION
        or audit.get("auditor_sha256") != _sha256_file(auditor_path)
        or audit.get("shared_auditor_sha256") != _sha256_file(shared_auditor_path)
        or audit.get("exporter_sha256") != _sha256_file(exporter_path)
        or audit.get("source_registry_version")
        != VISION_ALIGNMENT_PERCEPTION_SOURCE_REGISTRY_VERSION
        or audit.get("source_registry_sha256")
        != vision_alignment_perception_source_registry_sha256()
        or audit.get("source_implementation_inventory")
        != vision_alignment_perception_implementation_inventory()
        or audit.get("failures") != []
    ):
        raise ValueError("Perception source-audit implementation, identity, or status differs")
    catalog_path = Path(str(audit.get("catalog_path", ""))).expanduser().resolve()
    if not catalog_path.is_file() or audit.get("catalog_sha256") != _sha256_file(catalog_path):
        raise ValueError("Perception source-audit catalog bytes differ")

    provenance = _perception_provenance(config)
    provenance_ref = audit.get("image_provenance")
    if not isinstance(provenance_ref, Mapping) or provenance_ref != {
        "path": str(provenance.path),
        "sha256": provenance.raw_sha256,
        "content_sha256": provenance.content_sha256,
        "source_spec_sha256": provenance.source_spec_sha256,
    }:
        raise ValueError("Perception source audit identifies different image provenance")
    if (
        audit.get("preprocessing_config") != provenance.source_spec.as_canonical_dict()
        or audit.get("preprocessing_config_sha256") != provenance.source_spec_sha256
    ):
        raise ValueError("Perception source-audit preprocessing identity differs")
    probe = audit.get("probe")
    if not isinstance(probe, Mapping) or set(probe) != _PERCEPTION_AUDIT_PROBE_FIELDS:
        raise ValueError("Perception source-audit probe fields differ")
    examples_per_source = probe.get("examples_per_source")
    probe_seed = probe.get("seed")
    probe_epochs = probe.get("epochs")
    if (
        probe.get("format") != VISION_ALIGNMENT_PERCEPTION_PROBE_FORMAT
        or probe.get("version") != VISION_ALIGNMENT_PERCEPTION_PROBE_VERSION
        or probe.get("selection_algorithm") != VISION_ALIGNMENT_PERCEPTION_PROBE_SELECTION_ALGORITHM
        or isinstance(probe_epochs, bool)
        or not isinstance(probe_epochs, int)
        or probe_epochs != VISION_ALIGNMENT_PERCEPTION_PROBE_EPOCHS
        or isinstance(probe_seed, bool)
        or not isinstance(probe_seed, int)
        or probe_seed != VISION_ALIGNMENT_PERCEPTION_PROBE_SEED
        or isinstance(examples_per_source, bool)
        or not isinstance(examples_per_source, int)
        or examples_per_source != VISION_ALIGNMENT_PERCEPTION_PROBE_EXAMPLES
    ):
        raise ValueError("Perception source-audit probe identity or size differs")

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
            raise ValueError(f"Perception source-audit {field_name} differs from training")
    expected_mass = audit.get("expected_loss_mass")
    if (
        not isinstance(expected_mass, Mapping)
        or set(expected_mass) != set(targets)
        or any(
            isinstance(expected_mass[source_name], bool)
            or not isinstance(expected_mass[source_name], (int, float))
            or not math.isfinite(float(expected_mass[source_name]))
            or not math.isclose(
                float(expected_mass[source_name]),
                float(targets[source_name]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for source_name in targets
        )
    ):
        raise ValueError("Perception source-audit expected loss mass differs from targets")
    inputs = audit.get("inputs")
    summaries = audit.get("sources")
    if (
        not isinstance(inputs, Mapping)
        or not isinstance(summaries, Mapping)
        or set(inputs) != set(PERCEPTION_SOURCE_NAMES)
        or set(summaries) != set(PERCEPTION_SOURCE_NAMES)
        or set(targets) != set(PERCEPTION_SOURCE_NAMES)
    ):
        raise ValueError("Perception source-audit source set differs")
    input_descriptors: List[Dict[str, Any]] = []
    for source_name in PERCEPTION_SOURCE_NAMES:
        source = inputs[source_name]
        summary = summaries[source_name]
        selection = provenance.selection(source_name, "train")
        if (
            not isinstance(source, Mapping)
            or set(source) != _PERCEPTION_AUDIT_INPUT_FIELDS
            or source.get("name") != source_name
            or source.get("format") != "jsonl"
            or source.get("dataset_fingerprint") != selection.runtime_dataset_fingerprint
            or source.get("dataset_size") != len(selection.indices)
            or not isinstance(summary, Mapping)
        ):
            raise ValueError(f"Perception audit source {source_name!r} identity differs")
        probe_indices = source.get("probe_indices")
        row_hashes = source.get("serialized_row_hashes")
        if (
            not isinstance(probe_indices, list)
            or len(probe_indices) != examples_per_source // probe_epochs
            or tuple(probe_indices)
            != select_deterministic_probe_indices(
                len(selection.indices),
                examples_per_source // probe_epochs,
                seed=probe_seed,
                dataset_fingerprint=selection.runtime_dataset_fingerprint,
            )
            or _canonical_sha256(probe_indices) != source.get("probe_indices_sha256")
            or source.get("probe_epochs") != probe_epochs
            or not isinstance(row_hashes, list)
            or len(row_hashes) != examples_per_source
            or _canonical_sha256(row_hashes) != source.get("serialized_row_hashes_sha256")
        ):
            raise ValueError(f"Perception audit source {source_name!r} probe differs")
        expected_probe_image_content_sha256 = _canonical_sha256(
            [
                {
                    "index": index,
                    "image_sha256": selection.row_image_content_sha256[index],
                }
                for index in probe_indices
            ]
        )
        if source.get("probe_image_content_sha256") != expected_probe_image_content_sha256:
            raise ValueError(f"Perception audit source {source_name!r} image-content probe differs")
        input_descriptors.append(
            {
                "name": source_name,
                "sha256": source["sha256"],
                "dataset_fingerprint": selection.runtime_dataset_fingerprint,
                "probe_indices_sha256": source["probe_indices_sha256"],
                "probe_epochs": probe_epochs,
                "serialized_row_hashes_sha256": source["serialized_row_hashes_sha256"],
                "probe_image_content_sha256": expected_probe_image_content_sha256,
            }
        )
        source_path = (catalog_path.parent / str(source["path"])).resolve()
        if (
            not source_path.is_relative_to(catalog_path.parent)
            or not source_path.is_file()
            or source.get("sha256") != _sha256_file(source_path)
        ):
            raise ValueError(f"Perception audit source {source_name!r} bytes differ")
        examples = summary.get("examples")
        if (
            not isinstance(examples, Mapping)
            or examples
            != {
                "seen": examples_per_source,
                "valid": examples_per_source,
                "errors": 0,
            }
            or summary.get("zero_loss_examples") != 0
            or summary.get("truncated_examples") != 0
            or summary.get("error_samples") != []
            or summary.get("mean_sum_loss_masks") != means[source_name]
        ):
            raise ValueError(f"Perception audit source {source_name!r} summary differs")
    if audit.get("input_content_sha256") != _canonical_sha256(input_descriptors):
        raise ValueError("Perception source-audit input-content identity differs")
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
            "perception_trainability_arm": config.perception_trainability_arm.value,
            "model": config.model.as_config_dict(),
            "train_module": config.train_module.as_config_dict(),
            "router_lb_loss_weight": config.router_lb_loss_weight,
            "max_duration": {
                "value": config.trainer.max_duration.value,
                "unit": config.trainer.max_duration.unit.value,
            },
        }
    )


def _load_pinned_native_parent_paths(artifacts: ArtifactConfig) -> Tuple[str, ...]:
    """Load the selected bare model's exact expanded parent-path replay inventory."""
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
    """Reconstruct the shared native NumpyFSLDataset fingerprint from remote metadata."""
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


def _normalized_native_parent_paths(parent_paths: Sequence[str]) -> Tuple[str, ...]:
    """Normalize storage-scheme aliases while retaining exact bucket/key order."""

    normalized = []
    for index, parent_path in enumerate(parent_paths):
        match = re.fullmatch(r"(?:s3|gs)://ai2-llm/(.+)", parent_path)
        if match is None or not match.group(1):
            raise ValueError(
                f"Native replay parent path {index} is not a canonical ai2-llm object URI"
            )
        normalized.append(f"ai2-llm/{match.group(1)}")
    return tuple(normalized)


def _native_dataset_semantic_contract(raw_config: bytes, *, name: str) -> Mapping[str, Any]:
    """Extract only fields that determine the native FSL dataset's token corpus."""

    try:
        root = json.loads(raw_config, object_pairs_hook=_strict_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Pinned {name} config is invalid JSON: {error}") from error
    if not isinstance(root, Mapping) or not isinstance(root.get("dataset"), Mapping):
        raise ValueError(f"Pinned {name} config lacks its native dataset contract")
    dataset = cast(Mapping[str, Any], root["dataset"])
    fields = (
        "_CLASS_",
        "tokenizer",
        "mix",
        "expand_glob",
        "include_instance_metadata",
        "instance_filter_config",
        "ignore_fingerprint_mismatch",
        "sequence_length",
        "max_target_sequence_length",
        "generate_doc_lengths",
    )
    missing = [field_name for field_name in fields if field_name not in dataset]
    if missing:
        raise ValueError(f"Pinned {name} native dataset contract lacks {missing}")
    mix_base_dir = dataset.get("mix_base_dir")
    if mix_base_dir not in ("s3://ai2-llm", "gs://ai2-llm"):
        raise ValueError(f"Pinned {name} native mix_base_dir is not an ai2-llm root")
    return {field_name: dataset[field_name] for field_name in fields}


def _safe_native_loader_contract(raw_trainer_state: bytes, *, name: str) -> Mapping[str, Any]:
    """Read the restricted dataset identity from one pinned rank-0 trainer state."""

    import torch

    allowed_globals = [
        np._core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        type(np.dtype("uint32")),
        type(np.dtype("int64")),
        type(np.dtype("float64")),
        type(np.dtype("bool")),
    ]
    try:
        with torch.serialization.safe_globals(allowed_globals):
            state = torch.load(io.BytesIO(raw_trainer_state), map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"Could not safely load pinned {name} trainer state: {error}") from error
    if not isinstance(state, Mapping) or not isinstance(state.get("data_loader"), Mapping):
        raise ValueError(f"Pinned {name} trainer state lacks its data-loader identity")
    loader = cast(Mapping[str, Any], state["data_loader"])
    fields = (
        "dataset_fingerprint_version",
        "dataset_fingerprint",
        "dataset_type",
        "sequence_length",
        "max_target_sequence_length",
    )
    if any(field_name not in loader for field_name in fields):
        raise ValueError(f"Pinned {name} trainer state has an incomplete dataset identity")
    return {field_name: loader[field_name] for field_name in fields}


def _native_replay_lineage_artifacts(artifacts: ArtifactConfig) -> ArtifactConfig:
    """Select the reviewed replay anchor after proving SSMax corpus equivalence.

    The two SSMax parents and historical s002 parent use the same ordered OLMo-mix-0925 token
    objects; only the object-store scheme differs (``gs`` versus ``s3``). This proof permits all
    three lineages to consume one immutable compact materialization while their model checkpoints
    remain distinct. It deliberately ignores loader cursor, batch size, and seed: joint replay is
    a new deterministic panel of the parent corpus, not a continuation of pretraining's cursor.
    """

    anchor = ArtifactConfig.for_model_variant(VisionAlignmentModelVariant.s002)
    if artifacts == anchor:
        return anchor
    if artifacts not in (
        ArtifactConfig.for_model_variant(VisionAlignmentModelVariant.ssmax_head_qknorm),
        ArtifactConfig.for_model_variant(VisionAlignmentModelVariant.ssmax_no_qknorm),
    ):
        raise ValueError("Native replay lineage is not one of the reviewed bare-model artifacts")
    if (
        artifacts.parent_text_mix != anchor.parent_text_mix
        or artifacts.base_parent_mix_sha256 != anchor.base_parent_mix_sha256
        or artifacts.base_dataset_fingerprint != anchor.base_dataset_fingerprint
    ):
        raise ValueError("SSMax native corpus identity differs from the shared replay anchor")

    def pinned_bytes(path: Path, expected_sha256: str, *, name: str) -> bytes:
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise ValueError(f"Could not read pinned {name} at {path}: {error}") from error
        actual_sha256 = hashlib.sha256(raw).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Pinned {name} SHA-256 differs: expected {expected_sha256}, "
                f"got {actual_sha256}"
            )
        return raw

    selected_config = pinned_bytes(
        Path(artifacts.base_checkpoint) / "config.json",
        artifacts.base_config_sha256,
        name="selected native-parent config",
    )
    anchor_config = pinned_bytes(
        Path(anchor.base_checkpoint) / "config.json",
        anchor.base_config_sha256,
        name="shared replay-anchor config",
    )
    if _native_dataset_semantic_contract(
        selected_config, name="selected native parent"
    ) != _native_dataset_semantic_contract(anchor_config, name="shared replay anchor"):
        raise ValueError("SSMax native dataset semantics differ from the shared replay anchor")

    selected_paths = _load_pinned_native_parent_paths(artifacts)
    anchor_paths = _load_pinned_native_parent_paths(anchor)
    if _normalized_native_parent_paths(selected_paths) != _normalized_native_parent_paths(
        anchor_paths
    ):
        raise ValueError("SSMax native object inventory differs from the shared replay anchor")

    selected_trainer = pinned_bytes(
        Path(artifacts.base_checkpoint) / "train" / "rank0.pt",
        artifacts.base_trainer_state_sha256,
        name="selected native-parent trainer state",
    )
    anchor_trainer = pinned_bytes(
        Path(anchor.base_checkpoint) / "train" / "rank0.pt",
        anchor.base_trainer_state_sha256,
        name="shared replay-anchor trainer state",
    )
    selected_loader = _safe_native_loader_contract(selected_trainer, name="selected native parent")
    anchor_loader = _safe_native_loader_contract(anchor_trainer, name="shared replay anchor")
    if selected_loader != anchor_loader or selected_loader != {
        "dataset_fingerprint_version": "v2.0",
        "dataset_fingerprint": artifacts.base_dataset_fingerprint,
        "dataset_type": "fsl",
        "sequence_length": _JOINT_SEQUENCE_LENGTH,
        "max_target_sequence_length": _JOINT_SEQUENCE_LENGTH,
    }:
        raise ValueError("SSMax trainer dataset identity differs from the shared replay anchor")
    return anchor


def _validate_native_replay_pair(config: ExperimentConfig) -> None:
    """Validate pinned, disjoint native train/holdout replay manifests for joint CPT."""
    train_config = config.data.native_text_replay
    holdout_config = config.evaluation.native_text_holdout
    if train_config is None or holdout_config is None:
        raise ValueError(
            "Joint vision alignment requires native replay train and holdout manifests"
        )
    replay_artifacts = _native_replay_lineage_artifacts(config.artifacts)
    expected = {
        "expected_parent_checkpoint": replay_artifacts.base_checkpoint,
        "expected_parent_mix": replay_artifacts.parent_text_mix,
        "expected_parent_paths_sha256": replay_artifacts.base_data_paths_sha256,
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

    receipt = NativeTextReplayVerificationReceipt.load(
        train_config.verification_receipt_path,
        expected_sha256=train_config.expected_verification_receipt_sha256,
    )
    if receipt.version != 3:
        raise ValueError("Joint native replay requires a compact v3 verification receipt")
    expected_receipt_lineage = {
        "parent_paths_sha256": replay_artifacts.base_data_paths_sha256,
        "parent_mix_sha256": replay_artifacts.base_parent_mix_sha256,
        "parent_config_sha256": replay_artifacts.base_config_sha256,
        "parent_trainer_state_sha256": replay_artifacts.base_trainer_state_sha256,
        "parent_dataset_fingerprint": replay_artifacts.base_dataset_fingerprint,
    }
    for field_name, expected_value in expected_receipt_lineage.items():
        if getattr(receipt, field_name) != expected_value:
            raise ValueError(f"Native replay verification receipt has incompatible {field_name}")
    parent_paths = _load_pinned_native_parent_paths(replay_artifacts)
    receipt_parent_paths = tuple(
        cast(str, remote_source["parent_path"]) for remote_source in receipt.remote_sources
    )
    if receipt_parent_paths != parent_paths:
        raise ValueError(
            "Native replay verification receipt remote_sources must exactly enumerate the "
            "pinned parent data paths"
        )
    if (
        _native_parent_dataset_fingerprint(receipt.remote_sources)
        != replay_artifacts.base_dataset_fingerprint
    ):
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

    expected_manifest_lineage = {
        "parent_checkpoint": replay_artifacts.base_checkpoint,
        "parent_mix": replay_artifacts.parent_text_mix,
        "parent_paths_sha256": replay_artifacts.base_data_paths_sha256,
        "parent_config_sha256": replay_artifacts.base_config_sha256,
        "parent_trainer_state_sha256": replay_artifacts.base_trainer_state_sha256,
        "parent_dataset_fingerprint": replay_artifacts.base_dataset_fingerprint,
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


def _validate_parent_or_resume(config: ExperimentConfig) -> None:
    existing = _latest_output_checkpoint(config)
    if existing is not None:
        saved, _ = _checkpoint_config(existing)
        _validate_checkpoint_model_lineage(config, saved, checkpoint=existing)
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
        return

    if config.phase is VisionAlignmentPhase.bridge:
        return
    parent = config.initialization.checkpoint
    assert parent is not None
    parent_config, parent_sha = _checkpoint_config(parent)
    _validate_checkpoint_model_lineage(config, parent_config, checkpoint=parent)
    parent_meta = parent_config.get("vision_alignment")
    if not isinstance(parent_meta, dict):
        raise ValueError(f"Parent {parent} is not a vision-alignment checkpoint")
    expected_phase = config.initialization.expected_parent_phase
    if expected_phase is None or parent_meta.get("phase") != expected_phase.value:
        raise ValueError(
            f"Parent {parent} phase is {parent_meta.get('phase')!r}; "
            f"expected {expected_phase.value if expected_phase is not None else None!r}"
        )
    if (
        config.phase is not VisionAlignmentPhase.joint
        and parent_meta.get("recipe_version") != RECIPE_VERSION
    ):
        raise ValueError(f"Parent {parent} has an incompatible recipe version")
    configured_sha = config.initialization.parent_config_sha256
    if configured_sha is not None and configured_sha != parent_sha:
        raise ValueError(
            f"Parent config SHA mismatch: configured {configured_sha}, actual {parent_sha}"
        )
    config.initialization.parent_config_sha256 = parent_sha
    config.vision_alignment.parent_config_sha256 = parent_sha
    _validate_joint_parent_projection_lineage(config, parent_config)
    parent_gate_sha = _validate_parent_gate(config, parent, parent_config, parent_sha)
    config.initialization.parent_gate_sha256 = parent_gate_sha
    config.vision_alignment.parent_gate_sha256 = parent_gate_sha


def _validate_canonical_data_policy(config: ExperimentConfig) -> None:
    """Reject structural data-policy drift while allowing pinned artifact overrides."""
    model_variant = getattr(config, "model_variant", VisionAlignmentModelVariant.s002)
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

    single_response = config.data.ssmax_single_response_projection
    requires_single_response = _is_ssmax_variant(model_variant) and config.phase in (
        VisionAlignmentPhase.perception,
        VisionAlignmentPhase.joint,
    )
    if not requires_single_response:
        if single_response is not None:
            raise ValueError(
                "The deterministic single-response projection is SSMax perception/joint only"
            )
    elif single_response is None:
        raise ValueError("SSMax perception/joint requires deterministic single-response projection")
    else:
        single_response.contract(loss_token_weighting=config.data.loss_token_weighting)
        if single_response.seed != DATA_SEED or single_response.seed != config.data_seed:
            raise ValueError(
                f"SSMax single-response projection seed must equal data_seed={DATA_SEED}"
            )
        projected_means = single_response.projected_mean_loss_weight
        if set(projected_means) != expected_sources or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
            for value in projected_means.values()
        ):
            raise ValueError(
                "SSMax projected loss-mass calibration must contain one positive finite mean "
                f"for every phase source {sorted(expected_sources)}"
            )
        if (
            not isinstance(single_response.calibration_path, str)
            or not single_response.calibration_path
            or re.fullmatch(r"[0-9a-f]{64}", str(single_response.calibration_sha256 or "")) is None
        ):
            raise ValueError(
                "SSMax perception/joint requires a raw-SHA-pinned projection calibration receipt"
            )

    expected_fields: Tuple[Tuple[str, Any], ...] = (
        ("message_format", "document"),
        ("loss_token_weighting", "root_subsegments_root_tokens"),
        ("caption_prompt", "Description:"),
        ("transcript_prompt", "Transcript:"),
        ("max_crops", MAX_CROPS),
        # GatedDeltaNet state cannot currently be reset at multimodal pack boundaries.
        # The dense SSMax arms therefore use one serialized example per sequence and fail
        # closed if packed metadata reaches the model.
        ("pack_sequences", not _is_ssmax_variant(model_variant)),
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
    model_variant = getattr(config, "model_variant", VisionAlignmentModelVariant.s002)
    expected_default_lr = policy.lm_lr if policy.lm_lr > 0 else policy.connector_lr
    if not math.isclose(float(optim.lr), expected_default_lr) or optim.lr <= 0:
        raise ValueError("Optimizer default LR must remain positive and phase-derived")
    if model_variant is VisionAlignmentModelVariant.s002:
        if not isinstance(optim, OLMoDDPOptimizerConfig) or (
            optim.use_distributed is not True
            or optim.check_nan_inf_grad is not True
            or optim.clip_grad_norm_by_scheduler_group is not True
            or optim.compile is not False
        ):
            raise ValueError(
                "s002 vision alignment requires the pinned distributed optimizer safeguards"
            )
    elif not isinstance(optim, SkipStepAdamWConfig) or (
        optim.compile is not False
        or optim.foreach is not True
        or optim.step_increment_bugfix is not True
        or optim.sigma_factor != 12
        or optim.betas != (0.9, 0.95)
        or not math.isclose(optim.eps, 1e-6)
        or not math.isclose(optim.weight_decay, 0.0)
    ):
        raise ValueError("Dense SSMax alignment requires the pinned SkipStep AdamW contract")

    effective_vision_lr = (
        0.0
        if config.phase is VisionAlignmentPhase.perception
        and config.perception_trainability_arm is PerceptionTrainabilityArm.frozen_vision_control
        else policy.vision_lr
    )
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
        and config.perception_trainability_arm is PerceptionTrainabilityArm.treatment
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
    if policy.phase is not VisionAlignmentPhase.bridge:
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


def _validate_git_provenance(config: ExperimentConfig, *, runtime: bool) -> None:
    git = config.launch.git
    model_variant = getattr(config, "model_variant", VisionAlignmentModelVariant.s002)
    expected_branch = _expected_git_branch(model_variant)
    if git is None or re.fullmatch(r"[0-9a-f]{40}", git.ref or "") is None:
        raise ValueError("Vision alignment launch must pin an exact 40-character git revision")

    if not runtime:
        if git.branch != expected_branch:
            if model_variant is VisionAlignmentModelVariant.s002:
                raise ValueError(
                    "Vision alignment may launch only from the user-owned vision-moe branch"
                )
            raise ValueError(f"Vision alignment may launch only from {expected_branch!r}")
        return

    if not is_running_in_beaker_batch_job():
        raise ValueError("Vision alignment train workers must run inside a Beaker batch job")

    # Gantry checks out the submitted SHA in detached-HEAD state. Its reconstructed
    # GitRepoState therefore has no active branch, even though the authoritative job
    # metadata contains GIT_BRANCH and GIT_REF. Validate both the submitted metadata and
    # the actual detached checkout instead of requiring an active worker-side branch.
    runtime_branch = os.environ.get("GIT_BRANCH")
    runtime_ref = os.environ.get("GIT_REF")
    if runtime_branch != expected_branch:
        raise ValueError(f"Vision alignment runtime metadata must identify {expected_branch!r}")
    if re.fullmatch(r"[0-9a-f]{40}", runtime_ref or "") is None:
        raise ValueError("Vision alignment runtime must include an exact GIT_REF")
    if git.ref != runtime_ref:
        raise ValueError("Vision alignment detached checkout does not match the submitted GIT_REF")
    if git.branch not in (None, expected_branch):
        raise ValueError("Vision alignment runtime checkout reports an unexpected active branch")


def _validate_remote_git_ref(config: ExperimentConfig) -> None:
    """Prove the exact submitted commit is present at the reviewed remote branch."""

    git = config.launch.git
    expected_branch = _expected_git_branch(config.model_variant)
    if git is None or git.branch != expected_branch:
        raise RuntimeError("Vision alignment launch Git branch is not pinned")
    remote_ref = f"refs/heads/{expected_branch}"
    try:
        output = subprocess.check_output(
            ["git", "ls-remote", "--heads", git.repo_url, remote_ref],
            text=True,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Could not verify the reviewed remote Git branch") from error
    records = [line.split("\t", 1) for line in output.splitlines() if line]
    if records != [[git.ref, remote_ref]]:
        raise RuntimeError(
            "The exact vision-alignment Git revision is not pushed at the reviewed branch"
        )


def _validate_reviewed_profile(config: ExperimentConfig) -> None:
    """Bind each production trainable phase to a checked-in, override-free profile."""
    if config.phase is VisionAlignmentPhase.bridge:
        return
    profile_root, allowlist_name, _ = _reviewed_profile_policy(config.phase)
    path_value = config.reviewed_profile_path
    expected_sha256 = config.reviewed_profile_sha256
    allowlist_path_value = config.reviewed_profile_allowlist_path
    expected_allowlist_sha256 = config.reviewed_profile_allowlist_sha256
    if (
        not isinstance(path_value, str)
        or not path_value
        or Path(path_value).is_absolute()
        or ".." in Path(path_value).parts
        or not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        raise ValueError(
            f"Production {config.phase.value} requires an exact reviewed profile identity"
        )
    repository_root = Path(__file__).resolve().parents[3]
    if (
        allowlist_path_value != allowlist_name
        or not isinstance(expected_allowlist_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_allowlist_sha256) is None
    ):
        raise ValueError(
            f"Production {config.phase.value} requires an exact profile-allowlist identity"
        )
    approved_profiles, actual_allowlist_sha256 = _load_approved_profiles(
        repository_root, config.phase
    )
    if actual_allowlist_sha256 != expected_allowlist_sha256:
        raise ValueError(
            f"Production {config.phase.value} profile allowlist bytes differ from their pin"
        )
    approved_root = (repository_root / profile_root).resolve()
    profile_path = (repository_root / path_value).resolve()
    if (
        profile_path.parent != approved_root
        or profile_path.suffix != ".yaml"
        or profile_path.name.endswith(".yaml.template")
        or not profile_path.is_file()
    ):
        raise ValueError(
            f"Production {config.phase.value} profile path is not a checked-in profile"
        )
    try:
        raw = profile_path.read_bytes()
    except OSError as error:
        raise ValueError(
            f"Could not verify {config.phase.value} profile {profile_path}: {error}"
        ) from error
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    if raw_sha256 != expected_sha256:
        raise ValueError(
            f"Production {config.phase.value} profile bytes differ from their review pin"
        )
    if approved_profiles.get(path_value) != raw_sha256:
        raise ValueError(
            f"Production {config.phase.value} profile is not in the reviewed SHA-256 allowlist"
        )
    profile = _load_profile_yaml(raw, path=profile_path)
    if (
        set(profile) - {"version", "name", "description", "phase", "launch", "overrides"}
        or isinstance(profile.get("version"), bool)
        or not isinstance(profile.get("version"), int)
        or profile.get("version") != 1
        or profile.get("name") != config.required_run_name
        or profile.get("phase") != config.phase.value
    ):
        raise ValueError(f"Production {config.phase.value} profile identity or schema differs")
    profile_overrides = profile.get("overrides")
    if not isinstance(profile_overrides, list) or not all(
        isinstance(value, str) for value in profile_overrides
    ):
        raise ValueError(f"Production {config.phase.value} profile overrides are malformed")
    variant_selectors = [
        value for value in profile_overrides if value.startswith("--model_variant=")
    ]
    expected_variant_selector = f"--model_variant={config.model_variant.value}"
    if _is_ssmax_variant(config.model_variant):
        if variant_selectors != [expected_variant_selector]:
            raise ValueError(
                f"Production SSMax {config.phase.value} profile must explicitly select "
                f"{config.model_variant.value!r} exactly once"
            )
    elif variant_selectors not in ([], [expected_variant_selector]):
        raise ValueError("Production profile model variant differs from the experiment lineage")
    profile_launch = profile.get("launch")
    required_launch = {
        "num_nodes": 2,
        "num_gpus": 8,
        "workspace": _beaker_workspace(config.model_variant),
        "cluster": BEAKER_CLUSTER,
        "budget": BEAKER_BUDGET,
        "priority": "urgent",
        "min_runtime": "8h",
    }
    if not isinstance(profile_launch, Mapping) or any(
        type(profile_launch.get(field_name)) is not type(expected_value)
        or profile_launch.get(field_name) != expected_value
        for field_name, expected_value in required_launch.items()
    ):
        raise ValueError(
            f"Production {config.phase.value} requires the reviewed 2x8 Holmes "
            "urgent/eight-hour launch"
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


def _validate_phase_contract(
    config: ExperimentConfig, run_name: str, *, runtime: bool = False
) -> None:
    policy = _PHASE_POLICIES[config.phase]
    expected_artifacts = ArtifactConfig.for_model_variant(config.model_variant)
    if config.artifacts != expected_artifacts:
        raise ValueError("Pinned vision-alignment artifact identities may not be overridden")
    if config.vision_alignment.model_variant is not config.model_variant:
        raise ValueError("Checkpoint metadata and experiment model variants must match")
    if config.required_run_name != run_name or config.vision_alignment.lineage_id != run_name:
        raise ValueError(
            "Positional run name, required_run_name, and lineage_id must match exactly"
        )
    _validate_reviewed_profile(config)
    experiment_root = _experiment_root(config.model_variant)
    expected_save_folder = f"{experiment_root}/checkpoints/{run_name}"
    if config.trainer.save_folder != expected_save_folder:
        raise ValueError(
            f"Vision alignment save folder must be {expected_save_folder!r}, "
            f"got {config.trainer.save_folder!r}"
        )
    checkpoint_root = (Path(experiment_root) / "checkpoints").resolve()
    resolved_save_folder = Path(config.trainer.save_folder).resolve()
    if resolved_save_folder.parent != checkpoint_root:
        raise ValueError(f"Vision-alignment output must be one direct child of {checkpoint_root}")
    if config.trainer.save_overwrite is not True:
        raise ValueError("Vision alignment requires save_overwrite=True for exact resumes")
    if config.trainer.no_checkpoints is not False:
        raise ValueError("Vision alignment may not disable checkpoints or checkpoint loading")
    if config.trainer.no_evals is not False:
        raise ValueError("Vision alignment may not disable its configured intrinsic evaluators")
    secretless_smoke = _is_secretless_ssmax_smoke(config)
    expected_wandb_enabled = not secretless_smoke
    wandb = config.trainer.callbacks.get("wandb")
    if (
        not isinstance(wandb, WandBCallback)
        or wandb.name != run_name
        or wandb.project != _wandb_project(config.model_variant)
        or wandb.entity != WANDB_ENTITY
        or wandb.enabled is not expected_wandb_enabled
        or wandb.auto_resume is not True
    ):
        raise ValueError("W&B identity must match the positional run name")
    beaker = config.trainer.callbacks.get("beaker")
    expected_beaker_enabled = False if secretless_smoke else None
    if not isinstance(beaker, BeakerCallback) or beaker.enabled is not expected_beaker_enabled:
        raise ValueError("Beaker callback observability differs from the phase contract")
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
    if (
        config.launch.workspace != _beaker_workspace(config.model_variant)
        or config.launch.budget != BEAKER_BUDGET
    ):
        raise ValueError("Vision alignment workspace and budget are pinned by the recipe")
    if config.launch.num_gpus != 8 or config.launch.num_nodes < 1:
        raise ValueError("Vision alignment requires one or more complete 8-GPU Holmes nodes")
    if config.launch.allow_dirty:
        raise ValueError("Vision alignment may launch only a clean committed revision")
    _validate_git_provenance(config, runtime=runtime)
    if any(
        secret is not None
        for secret in (
            config.launch.aws_config_secret,
            config.launch.aws_credentials_secret,
            config.launch.google_credentials_secret,
        )
    ):
        raise ValueError("Vision alignment training must not receive cloud credentials")
    expected_env_secret_names = set() if secretless_smoke else {"BEAKER_TOKEN", "WANDB_API_KEY"}
    if {secret.name for secret in config.launch.env_secrets} != expected_env_secret_names:
        raise ValueError("Vision alignment launch has an unexpected secret surface")
    if not any(
        bucket.bucket == "oe-training-default" and bucket.mount == "/weka/oe-training-default"
        for bucket in (config.launch.weka_buckets or [])
    ):
        raise ValueError("Vision alignment requires the approved training Weka mount")
    if not config.launch.beaker_image:
        raise ValueError("Vision alignment requires the pinned runtime image")
    if config.model_variant is VisionAlignmentModelVariant.s002 and not config.launch.post_setup:
        raise ValueError("s002 vision alignment requires its OLMoDDP runtime setup hook")
    ssmax_parent_gate_version = _ssmax_joint_parent_gate_version(config)
    if ssmax_parent_gate_version == 7:
        expected_ssmax_post_setup = SSMAX_DIRECT_JOINT_GIT_HISTORY_POST_SETUP
    elif ssmax_parent_gate_version == 8:
        expected_ssmax_post_setup = SSMAX_EXPLORATORY_JOINT_GIT_HISTORY_POST_SETUP
    elif ssmax_parent_gate_version == 9:
        expected_ssmax_post_setup = SSMAX_EXPLORATORY_WAIVER_JOINT_GIT_HISTORY_POST_SETUP
    else:
        expected_ssmax_post_setup = None
    if _is_ssmax_variant(config.model_variant) and (
        config.launch.post_setup != expected_ssmax_post_setup
    ):
        raise ValueError("Dense SSMax launch Git-history setup differs from its parent protocol")

    if config.initialization.mode is not policy.initialization_mode:
        raise ValueError(
            f"Phase {config.phase.value} requires initialization mode "
            f"{policy.initialization_mode.value}"
        )
    if config.initialization.expected_parent_phase is not policy.expected_parent_phase:
        raise ValueError("Initialization parent-phase contract was overridden")
    expected_freeze_params = list(policy.freeze_params)
    if (
        config.phase is VisionAlignmentPhase.perception
        and config.perception_trainability_arm is PerceptionTrainabilityArm.frozen_vision_control
    ):
        expected_freeze_params.insert(0, "vision.*")
    if expected_freeze_params != (config.train_module.freeze_params or []):
        raise ValueError("Phase freeze patterns are derived and may not be overridden")
    if (
        config.phase is not VisionAlignmentPhase.perception
        and config.perception_trainability_arm is not PerceptionTrainabilityArm.treatment
    ):
        raise ValueError("The frozen-vision control is defined only for perception")
    if (
        config.train_module.train_embedding_rows is None
        or len(config.train_module.train_embedding_rows) != 6
    ):
        raise ValueError("Exactly six input-only image-token rows must be trainable")
    if config.model_variant is VisionAlignmentModelVariant.s002:
        if "ssmax_health_ledger" in config.trainer.callbacks:
            raise ValueError("The SSMax health ledger must not alter the historical s002 recipe")
        if not isinstance(config.train_module, MultimodalOLMoDDPTrainModuleConfig):
            raise ValueError("s002 requires the audited OLMoDDP multimodal train module")
        if (
            config.train_module.dp_config is None
            or config.train_module.dp_config.name is not DataParallelType.ddp
            or config.train_module.ep_config is None
            or config.train_module.ep_config.degree != EP_DEGREE
        ):
            raise ValueError("s002 requires the pinned DDP/EP8 topology")
        if config.router_lb_loss_weight is None:
            raise ValueError("s002 requires its routed-expert load-balancing loss")
    else:
        ledger = config.trainer.callbacks.get("ssmax_health_ledger")
        if (
            not isinstance(ledger, SSMaxHealthLedgerCallback)
            or ledger.enabled is not True
            or ledger.model_variant != config.model_variant.value
            or ledger.phase != config.phase.value
            or ledger.run_name != run_name
        ):
            raise ValueError(
                "Dense SSMax phases require their exact checkpoint-native health ledger"
            )
        if not isinstance(config.train_module, MultimodalTransformerTrainModuleConfig):
            raise ValueError("Dense SSMax variants require the generic multimodal train module")
        if (
            config.train_module.dp_config is None
            or config.train_module.dp_config.name is not DataParallelType.hsdp
            or config.train_module.dp_config.param_dtype is not DType.bfloat16
            or config.train_module.dp_config.reduce_dtype is not DType.float32
        ):
            raise ValueError("Dense SSMax variants require the pinned BF16/FP32 HSDP topology")
        if config.router_lb_loss_weight is not None:
            raise ValueError("Dense SSMax variants do not have a router loss")
        if config.train_module.new_component_init_seed != config.init_seed:
            raise ValueError("SSMax connector initialization seed must match init_seed")
        if (
            config.train_module.state_dict_load_opts is not None
            or config.train_module.load_key_mapping is not None
        ):
            raise ValueError(
                "SSMax alignment requires the generic checkpointer's default strict, "
                "identity-key model load"
            )
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
    if config.data.pack_sequences == _is_ssmax_variant(config.model_variant):
        raise ValueError(
            "s002 requires audited sequence packing, while recurrent SSMax variants must "
            "disable it"
        )
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
        replay_artifacts = _native_replay_lineage_artifacts(config.artifacts)
        if replay.expected_parent_checkpoint != replay_artifacts.base_checkpoint:
            raise ValueError("Native replay must pin the reviewed shared corpus anchor")
        if replay.expected_parent_mix != replay_artifacts.parent_text_mix:
            raise ValueError("Native replay must pin the shared corpus's exact parent text mix")
        if replay.expected_parent_paths_sha256 != replay_artifacts.base_data_paths_sha256:
            raise ValueError(
                "Native replay must pin the shared corpus's exact expanded parent path manifest"
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
    if source_audit is not None:
        _ssmax_single_response_calibration(config, source_audit)
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


def build_config(
    script: str,
    run_name: str,
    overrides: List[str],
    *,
    runtime: bool = False,
    reviewed_profile_path: Optional[str] = None,
    reviewed_profile_sha256: Optional[str] = None,
    reviewed_profile_allowlist_path: Optional[str] = None,
    reviewed_profile_allowlist_sha256: Optional[str] = None,
) -> ExperimentConfig:
    """Build and validate one phase config after resolving its phase selector first."""
    _validate_run_name(run_name)
    _validate_override_surface(overrides)
    phase = _extract_phase(overrides)
    model_variant = _extract_model_variant(overrides)
    policy = _PHASE_POLICIES[phase]
    artifacts = ArtifactConfig.for_model_variant(model_variant)
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
        pack_sequences=not _is_ssmax_variant(model_variant),
        ssmax_single_response_projection=(
            SSMaxSingleResponseProjectionConfig(seed=DATA_SEED)
            if _is_ssmax_variant(model_variant)
            and phase in (VisionAlignmentPhase.perception, VisionAlignmentPhase.joint)
            else None
        ),
    )
    collator = MultimodalCollatorConfig(
        pad_token_id=int(tokenizer.pad_token_id),
        label_ignore_index=-100,
        pad_sequence_length=policy.sequence_length,
    )
    train_module = _build_train_module_config(policy, image_ids, model_variant)
    experiment_root = _experiment_root(model_variant)
    wandb_project = _wandb_project(model_variant)
    trainer = (
        TrainerConfig(
            save_folder=f"{experiment_root}/checkpoints/{run_name}",
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
                project=wandb_project,
                enabled=wandb_project is not None,
                cancel_check_interval=10,
                auto_resume=True,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("garbage_collector", GarbageCollectorCallback())
        .with_callback("beaker", BeakerCallback())
        .with_callback("console_logger", _build_console_logger())
    )
    if _is_ssmax_variant(model_variant):
        trainer = trainer.with_callback(
            "ssmax_health_ledger",
            SSMaxHealthLedgerCallback(
                model_variant=model_variant.value,
                phase=phase.value,
                run_name=run_name,
            ),
        )
    launch_config = _build_vision_alignment_launch_config(
        script=script,
        run_name=run_name,
        overrides=overrides,
        model_variant=model_variant,
        secretless_runtime_smoke=(
            runtime
            and _is_secretless_ssmax_smoke_request(
                model_variant=model_variant,
                phase=phase,
                run_name=run_name,
                reviewed_profile_path=reviewed_profile_path,
                overrides=overrides,
            )
        ),
    )
    launch_config.aws_config_secret = None
    launch_config.aws_credentials_secret = None
    launch_config.google_credentials_secret = None
    launch_config.env_secrets = [
        secret
        for secret in launch_config.env_secrets
        if secret.name in ("BEAKER_TOKEN", "WANDB_API_KEY")
    ]
    _configure_launch_runtime(launch_config, model_variant)

    config = ExperimentConfig(
        launch=launch_config,
        model=_build_model_config(token_ids, artifacts, model_variant),
        collator=collator,
        train_module=train_module,
        trainer=trainer,
        model_variant=model_variant,
        phase=phase,
        perception_trainability_arm=PerceptionTrainabilityArm.treatment,
        artifacts=artifacts,
        initialization=initialization,
        data=data,
        evaluation=_build_evaluation_config(policy),
        vision_alignment=VisionAlignmentMetadataConfig(
            model_variant=model_variant,
            phase=phase,
            lineage_id=run_name,
            parent_checkpoint=initialization.checkpoint,
            parent_gate_sha256=initialization.parent_gate_sha256,
        ),
        global_batch_size=GLOBAL_BATCH_INSTANCES * policy.sequence_length,
        router_lb_loss_weight=(
            0.015 if model_variant is VisionAlignmentModelVariant.s002 else None
        ),
        required_run_name=run_name,
        expected_launch_command=[script, "train", run_name, *overrides],
        reviewed_profile_path=reviewed_profile_path,
        reviewed_profile_sha256=reviewed_profile_sha256,
        reviewed_profile_allowlist_path=reviewed_profile_allowlist_path,
        reviewed_profile_allowlist_sha256=reviewed_profile_allowlist_sha256,
    ).merge(overrides)
    _configure_ssmax_direct_joint_git_history(config)
    _pin_launch_git_branch(config)
    _configure_synthetic_smoke_observability(config)
    if (
        config.phase is VisionAlignmentPhase.perception
        and config.perception_trainability_arm is PerceptionTrainabilityArm.frozen_vision_control
    ):
        config.train_module.freeze_params = [
            "vision.*",
            *(config.train_module.freeze_params or []),
        ]
        vision_override = (config.train_module.optim.group_overrides or [])[2]
        vision_override.opts["lr"] = 0.0
    if (
        config.phase is not VisionAlignmentPhase.bridge
        and config.trainer.load_path is None
        and config.initialization.checkpoint is not None
    ):
        config.trainer.load_path = config.initialization.checkpoint
    config.vision_alignment.phase = config.phase
    config.vision_alignment.model_variant = config.model_variant
    config.vision_alignment.parent_checkpoint = config.initialization.checkpoint
    config.vision_alignment.parent_gate_sha256 = config.initialization.parent_gate_sha256
    if isinstance(config.train_module, MultimodalTransformerTrainModuleConfig):
        config.train_module.new_component_init_seed = config.init_seed

    if config.data.native_text_replay is not None:
        manifest = config.data.native_text_replay.build(tokenizer).manifest
        config.data.native_text_replay_fingerprint = manifest.content_fingerprint
    if config.evaluation.native_text_holdout is not None:
        holdout_manifest = config.evaluation.native_text_holdout.build(tokenizer).manifest
        config.evaluation.native_text_holdout_fingerprint = holdout_manifest.content_fingerprint
    config.train_module.source_loss_mass_targets = config.data.mixture.resolved_targets()
    if config.model_variant is VisionAlignmentModelVariant.s002:
        _configure_router_load_balancing(config.model.lm, config.router_lb_loss_weight)
    elif config.router_lb_loss_weight is not None:
        raise ValueError("Dense SSMax variants do not have a routed-expert load-balancing loss")
    _validate_phase_contract(config, run_name, runtime=runtime)
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


def _ssmax_single_response_dataset(
    config: ExperimentConfig,
    dataset: Any,
    source_name: str,
    *,
    logical_split: str,
) -> Any:
    """Apply the SSMax-only annotation projection at the selected-dataset boundary."""

    projection = config.data.ssmax_single_response_projection
    required = _is_ssmax_variant(config.model_variant) and config.phase in (
        VisionAlignmentPhase.perception,
        VisionAlignmentPhase.joint,
    )
    if not required:
        if projection is not None:
            raise ValueError("Single-response projection is configured outside its SSMax phases")
        return dataset
    if source_name == "native_text_replay":
        return dataset
    if projection is None:
        raise ValueError("SSMax visual source reached runtime without a projection contract")
    return SSMaxSingleResponseDataset(
        dataset,
        source_name=source_name,
        logical_split=logical_split,
        seed=projection.seed,
        loss_token_weighting=config.data.loss_token_weighting,
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
    calibration = _ssmax_single_response_calibration(config, audit) if audit is not None else None
    perception_provenance = (
        _perception_provenance(config) if config.phase is VisionAlignmentPhase.perception else None
    )
    joint_projection = (
        _joint_visual_projection(config, token_ids)
        if config.phase is VisionAlignmentPhase.joint
        else None
    )
    single_response = config.data.ssmax_single_response_projection
    effective_means = (
        single_response.projected_mean_loss_weight
        if single_response is not None
        else config.data.mixture.mean_loss_weight
    )
    weights = sampling_weights_from_loss_mass(targets, effective_means)
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
            dataset = _AuditedDataset(dataset, name, audit, token_ids=token_ids)
        dataset = _ssmax_single_response_dataset(
            config,
            dataset,
            name,
            logical_split="train",
        )
        if calibration is not None:
            if audit is None:
                raise ValueError("SSMax projection calibration requires the pinned source audit")
            _validate_live_ssmax_projection_calibration(
                calibration,
                audit,
                dataset,
                name,
            )
        sources.append((name, dataset, weights[name]))
    sources.sort(key=lambda item: item[0])
    names = [item[0] for item in sources]
    datasets = [item[1] for item in sources]
    sampling = [item[2] for item in sources]
    delivered = expected_loss_mass(dict(zip(names, sampling)), effective_means)
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
    calibration = None
    if getattr(config.data, "ssmax_single_response_projection", None) is not None:
        source_audit = _validated_source_audit(config)
        assert source_audit is not None
        calibration = _ssmax_single_response_calibration(config, source_audit)
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
        validation_manifest = _validate_validation_manifest(
            config, _validated_source_audit(config), validate_live_datasets=False
        )
        validation_sources = ("pixmo_caption", "pixmo_transcript")
    for source_name in validation_sources:
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
        dataset = _ssmax_single_response_dataset(
            config,
            dataset,
            source_name,
            logical_split="validation",
        )
        if calibration is not None:
            _validate_live_ssmax_projection_validation(calibration, dataset, source_name)
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


def _mixture_data_error_policy(
    model_variant: VisionAlignmentModelVariant,
) -> Dict[str, int]:
    """Fail recurrent SSMax runs on the first non-reproducible data row."""

    if _is_ssmax_variant(model_variant):
        return {"max_consecutive_data_errors": 0, "max_total_data_errors": 0}
    return {}


def _mixture_allowed_data_error_signatures(
    config: ExperimentConfig,
) -> Dict[Tuple[str, int, int], Tuple[type[Exception], str]]:
    """Return the one audited source defect quarantined for the matched SSMax comparison."""

    expected_run_name = _SSMAX_JOINT_DATA_ERROR_QUARANTINE_RUNS.get(config.model_variant)
    if (
        expected_run_name is not None
        and config.phase is VisionAlignmentPhase.joint
        and config.required_run_name == expected_run_name
        and config.data.joint_visual_projection_sha256
        == _SSMAX_JOINT_DATA_ERROR_QUARANTINE_PROJECTION_SHA256
    ):
        return dict(_SSMAX_JOINT_DATA_ERROR_QUARANTINE)
    return {}


def _write_immutable_json_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    """Create one canonical receipt, accepting only a byte-identical existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        if path.read_bytes() != raw:
            raise RuntimeError(f"Existing immutable receipt differs at {path}")
        return
    with os.fdopen(descriptor, "wb") as file_handle:
        file_handle.write(raw)
        file_handle.flush()
        os.fsync(file_handle.fileno())


def _validate_ssmax_phase_parent_model_state(train_module, config: ExperimentConfig) -> None:
    """Preflight an SSMax cross-phase checkpoint before Trainer performs its model-only load.

    The ordinary generic checkpointer remains the sole loader. This metadata-only pass makes the
    intended contract explicit: every current multimodal tensor must exist exactly once under the
    native ``model.`` prefix with the same shape, dtype, and layout. Optimizer entries may be
    present in the parent DCP, but the phase config separately requires ``load_optim_state=False``
    and ``load_trainer_state=False``.
    """

    if not _is_ssmax_variant(config.model_variant) or config.phase is VisionAlignmentPhase.bridge:
        return
    parent = config.initialization.checkpoint
    if not isinstance(parent, str) or not parent:
        raise RuntimeError("SSMax phase transition lacks its validated parent checkpoint")

    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    from olmo_core.distributed.checkpoint import get_checkpoint_metadata

    state_dir = _checkpoint_state_dir(parent)
    metadata = get_checkpoint_metadata(state_dir)
    checkpoint_model = {
        key.removeprefix("model."): value
        for key, value in metadata.state_dict_metadata.items()
        if key.startswith("model.")
    }
    current_model = train_module.multimodal_model.state_dict()
    current_keys = set(current_model)
    checkpoint_keys = set(checkpoint_model)
    if checkpoint_keys != current_keys:
        raise RuntimeError(
            "SSMax phase parent model inventory differs from the current multimodal model; "
            f"missing={sorted(current_keys - checkpoint_keys)[:16]}, "
            f"unexpected={sorted(checkpoint_keys - current_keys)[:16]}"
        )
    for key, current_value in current_model.items():
        tensor_metadata = checkpoint_model[key]
        if not isinstance(tensor_metadata, TensorStorageMetadata):
            raise RuntimeError(f"SSMax phase parent model entry {key!r} is not a tensor")
        if (
            tuple(tensor_metadata.size) != tuple(current_value.shape)
            or tensor_metadata.properties.dtype != current_value.dtype
            or tensor_metadata.properties.layout != current_value.layout
        ):
            raise RuntimeError(
                f"SSMax phase parent tensor contract differs for {key!r}: "
                f"checkpoint=(shape={tuple(tensor_metadata.size)}, "
                f"dtype={tensor_metadata.properties.dtype}, "
                f"layout={tensor_metadata.properties.layout}), "
                f"current=(shape={tuple(current_value.shape)}, dtype={current_value.dtype}, "
                f"layout={current_value.layout})"
            )
    log.info(
        "Verified strict model-only %s parent inventory: %d tensors from %s",
        config.phase.value,
        len(current_model),
        state_dir,
    )


def _verify_ssmax_parent_checkpoint_bytes(config: ExperimentConfig) -> Mapping[str, Any]:
    """Rehash every native parent shard on rank 0 and broadcast its exact identity."""
    import torch.distributed as dist

    from olmo_core.eval.vision_alignment_ssmax_bridge import checkpoint_identity

    artifacts = config.artifacts
    required = {
        "identity_sha256": artifacts.base_checkpoint_identity_sha256,
        "state_file_count": artifacts.base_checkpoint_state_file_count,
        "state_file_inventory_sha256": artifacts.base_checkpoint_state_file_inventory_sha256,
        "trainer_state_count": artifacts.base_checkpoint_trainer_state_count,
        "trainer_state_inventory_sha256": (
            artifacts.base_checkpoint_trainer_state_inventory_sha256
        ),
    }
    if any(value is None for value in required.values()):
        raise RuntimeError("SSMax parent full-checkpoint byte pins are incomplete")
    result: list[Any] = [None, None]
    if get_rank() == 0:
        try:
            result[0] = checkpoint_identity(
                Path(artifacts.base_checkpoint), workers=config.checkpoint_load_threads
            )
        # Rank 0 must convert every failure into a broadcast or its peers will deadlock here.
        except Exception as error:  # noqa: BLE001
            result[1] = f"{type(error).__name__}: {error}"
    dist.broadcast_object_list(result, src=0)
    if result[1] is not None:
        raise RuntimeError(f"SSMax parent full-checkpoint verification failed: {result[1]}")
    identity = cast(Mapping[str, Any], result[0])
    expected = {
        "path": str(Path(artifacts.base_checkpoint).expanduser().resolve()),
        "global_step": 65_799,
        "config_sha256": artifacts.base_config_sha256,
        "marker_sha256": artifacts.base_checkpoint_marker_sha256,
        "dcp_metadata_sha256": artifacts.base_checkpoint_metadata_sha256,
        **required,
    }
    differences = sorted(name for name, value in expected.items() if identity.get(name) != value)
    unexpected = sorted(set(identity) - set(expected))
    if differences or unexpected:
        raise RuntimeError(
            "SSMax parent full-checkpoint identity differs from its pinned bytes: "
            f"differing={differences}, unexpected={unexpected}"
        )
    return identity


def _initialize_ssmax_parent(train_module, config: ExperimentConfig) -> None:
    """Strictly load every and only LM tensor from one byte-pinned SSMax parent DCP."""
    import torch.distributed as dist

    parent_checkpoint_identity = _verify_ssmax_parent_checkpoint_bytes(config)
    model = train_module.multimodal_model
    model_state = model.state_dict()
    named_parameters = dict(model.named_parameters())
    loaded_model_keys = {key for key in model_state if key.startswith("lm.")}
    missing_model_keys = set(model_state) - loaded_model_keys
    loaded_parameter_keys = {key for key in named_parameters if key.startswith("lm.")}
    artifacts = config.artifacts
    if len(loaded_model_keys) != artifacts.expected_lm_tensor_count:
        raise RuntimeError(
            "SSMax runtime LM tensor inventory differs from its pinned parent: "
            f"runtime={len(loaded_model_keys)}, expected={artifacts.expected_lm_tensor_count}"
        )
    parameter_count = sum(named_parameters[key].numel() for key in loaded_parameter_keys)
    if parameter_count != artifacts.expected_lm_parameter_count:
        raise RuntimeError(
            "SSMax runtime LM parameter inventory differs from its pinned parent: "
            f"runtime={parameter_count:,d}, expected={artifacts.expected_lm_parameter_count}"
        )
    key_mapping = {key: key.removeprefix("lm.") for key in loaded_model_keys}
    receipt = train_module.load_parent_model_state_dict(
        _checkpoint_state_dir(artifacts.base_checkpoint),
        current_to_checkpoint_key_mapping=key_mapping,
        expected_loaded_model_keys=loaded_model_keys,
        expected_missing_model_keys=missing_model_keys,
        expected_loaded_parameter_keys=loaded_parameter_keys,
        process_group=dist.group.WORLD,
        thread_count=config.checkpoint_load_threads,
    )
    receipt.update(
        {
            "format": "vision_alignment_ssmax_parent_load_receipt",
            "version": 1,
            "model_variant": config.model_variant.value,
            "parent_checkpoint": artifacts.base_checkpoint,
            "parent_config_sha256": artifacts.base_config_sha256,
            "parent_data_paths_sha256": artifacts.base_data_paths_sha256,
            "parent_checkpoint_marker_sha256": artifacts.base_checkpoint_marker_sha256,
            "parent_dcp_metadata_sha256": artifacts.base_checkpoint_metadata_sha256,
            "parent_trainer_state_sha256": artifacts.base_trainer_state_sha256,
            "parent_source_commit": artifacts.source_commit,
            "parent_olmo_core_commit": artifacts.source_olmo_core_commit,
            "parent_model_keyset_sha256": artifacts.base_model_keyset_sha256,
            "parent_model_inventory_sha256": artifacts.base_model_inventory_sha256,
            "parent_checkpoint_identity_sha256": parent_checkpoint_identity["identity_sha256"],
            "parent_state_file_count": parent_checkpoint_identity["state_file_count"],
            "parent_state_file_inventory_sha256": parent_checkpoint_identity[
                "state_file_inventory_sha256"
            ],
            "parent_trainer_state_count": parent_checkpoint_identity["trainer_state_count"],
            "parent_trainer_state_inventory_sha256": parent_checkpoint_identity[
                "trainer_state_inventory_sha256"
            ],
            "loaded_parameter_numel": parameter_count,
        }
    )
    receipt["fingerprint"] = _canonical_sha256(receipt)
    if get_rank() == 0:
        _write_immutable_json_receipt(
            Path(config.trainer.save_folder) / "bridge-parent-load-receipt.json",
            receipt,
        )
    dist.barrier()


def _initialize_fresh_bridge(train_module, config: ExperimentConfig, token_ids) -> None:
    import torch.distributed as dist

    artifacts = config.artifacts
    if config.model_variant is VisionAlignmentModelVariant.s002:
        state_dir = _checkpoint_state_dir(artifacts.base_checkpoint)
        log.info("Loading bare s002 language-model weights from %s", state_dir)
        train_module.load_state_dict_direct(
            state_dir,
            process_group=dist.group.WORLD,
            thread_count=config.checkpoint_load_threads,
            load_optim_state=False,
        )
    else:
        log.info("Loading pinned bare %s language-model weights", config.model_variant.value)
        _initialize_ssmax_parent(train_module, config)

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
    if config.model_variant is VisionAlignmentModelVariant.s002:
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
    if existing is None:
        _validate_ssmax_phase_parent_model_state(train_module, config)
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
    data_error_policy = _mixture_data_error_policy(config.model_variant)
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
        allowed_data_error_signatures=_mixture_allowed_data_error_signatures(config),
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        **data_error_policy,
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
    _validate_remote_git_ref(config)
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
    runtime = command == "train"
    experiment = build_config(
        script,
        run_name,
        overrides,
        runtime=runtime,
        reviewed_profile_path=(
            cast(str, profile["__reviewed_path__"]) if profile is not None else None
        ),
        reviewed_profile_sha256=(
            cast(str, profile["__reviewed_sha256__"]) if profile is not None else None
        ),
        reviewed_profile_allowlist_path=(
            cast(str, profile["__reviewed_allowlist_path__"])
            if profile is not None and "__reviewed_allowlist_path__" in profile
            else None
        ),
        reviewed_profile_allowlist_sha256=(
            cast(str, profile["__reviewed_allowlist_sha256__"])
            if profile is not None and "__reviewed_allowlist_sha256__" in profile
            else None
        ),
    )
    experiment = _apply_profile_launch(experiment, profile, run_name=run_name)
    _validate_phase_contract(experiment, run_name, runtime=runtime)
    log.info(experiment)
    if command == "train":
        train(experiment)
        teardown_training_environment()
    elif command == "launch":
        launch(experiment)
    elif command != "dry_run":
        print(usage)
        sys.exit(1)
