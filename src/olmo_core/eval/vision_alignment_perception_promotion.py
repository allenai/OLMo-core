"""Fail-closed evidence boundary for promoting a perception checkpoint.

This module is deliberately separate from :mod:`vision_alignment_promotion`.  That module
encodes the already-published bridge-step500 policy; loosening its bridge-specific schemas to
cover perception would silently weaken an existing phase boundary.  This module validates
immutable evidence for the paired frozen-vision-control versus vision-unfrozen treatment and
builds a compact bundle for explicit human approval.

The module never runs training, invents measurements, or approves a parent.  Every component is
an immutable raw-SHA-pinned receipt, and bundle audit re-opens those receipts and re-derives the
scientific and operational gates.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import math
import os
import re
import stat
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import numpy as np

from olmo_core.eval import vision_alignment_promotion as bridge
from olmo_core.eval.vision_alignment_promotion import (
    IMAGE_TOKEN_ROWS,
    PromotionValidationError,
    artifact_reference,
    canonical_sha256,
    load_json,
    sha256_file,
)

PERCEPTION_PROMOTION_BUNDLE_FORMAT = "vision_alignment_perception_promotion_bundle"
PERCEPTION_PROMOTION_BUNDLE_VERSION = 1
PERCEPTION_PROMOTION_POLICY = "vision-alignment-perception-paired-step4000-promotion-v1"

OUTCOME_RECEIPT_FORMAT = "vision_alignment_perception_outcome_receipt"
COUNTERFACTUAL_OUTCOME_RECEIPT_FORMAT = OUTCOME_RECEIPT_FORMAT
PAIR_CONTRACT_RECEIPT_FORMAT = "vision_alignment_perception_profile_pair_audit"
PAIR_CONTRACT_RECEIPT_VERSION = 2
INITIALIZATION_PARITY_RECEIPT_FORMAT = "vision_alignment_perception_initialization_parity_receipt"
PERCEPTION_FROZEN_STATE_RECEIPT_FORMAT = "vision_alignment_perception_frozen_state_receipt"
PERCEPTION_TEXT_RETENTION_RECEIPT_FORMAT = "vision_alignment_perception_text_retention_receipt"
RUN_HEALTH_RECEIPT_FORMAT = "vision_alignment_perception_run_health_receipt"
LOSS_MASS_PAIR_RECEIPT_FORMAT = "vision_alignment_perception_loss_mass_pair_receipt"
RECEIPT_VERSION = 1

TREATMENT_GUARD_WAIVER_ID = "perception_treatment_seven_optimizer_guard_skips"
REQUIRED_WAIVER_IDS = frozenset({TREATMENT_GUARD_WAIVER_ID})

CONTROL_ARM = "frozen_vision_control"
TREATMENT_ARM = "treatment"
ARMS = (CONTROL_ARM, TREATMENT_ARM)
SOURCES = (
    "audited_alignment",
    "cosyn_point",
    "ocr_document",
    "pixmo_caption",
    "pixmo_points_basic",
    "pixmo_points_high_frequency",
    "pixmo_transcript",
    "scalar_count",
)
LOSS_MASS_TARGETS = {
    "audited_alignment": 0.05,
    "cosyn_point": 0.03,
    "ocr_document": 0.10,
    "pixmo_caption": 0.45,
    "pixmo_points_basic": 0.10,
    "pixmo_points_high_frequency": 0.02,
    "pixmo_transcript": 0.20,
    "scalar_count": 0.05,
}
PRIMARY_STEP = 4000
DURABILITY_STEP = 3000
EXPECTED_TREATMENT_SKIP_STEPS = (294, 1209, 1545, 1826, 2586, 3359, 3610)
EXPECTED_EXPERIMENT_IDS = {
    CONTROL_ARM: "01KZWCJV3AYHS0VWX8B7HPHN2D",
    TREATMENT_ARM: "01KZWCJWDEMPZ67QVWQB20HNWG",
}
EXPECTED_FROZEN_TENSOR_COUNTS = {CONTROL_ARM: 806, TREATMENT_ARM: 403}
EXPECTED_INITIALIZATION_PARAMETER_COUNT = 818
EXPECTED_INITIALIZATION_BUFFER_COUNT = 0
EXPECTED_PAIR_CONTRACT_RAW_SHA256 = (
    "5c7d9f3b2a882ed3147ca239eaaf00e9089d8e47c552a5cd19c351fdd806ea04"
)
EXPECTED_PAIR_CONTRACT_CONTENT_SHA256 = (
    "52e2cac1ac8b45eda2daa4a54422f9767f7eb97dbf6e5c2dd77e97b5e4dc8d7f"
)
EXPECTED_GIT_REF = "d8ec4f57cf026424ccd13f20452365b6b1df34e5"
EXPECTED_PARENT_CHECKPOINT = (
    "/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/checkpoints/"
    "vision-alignment-bridge-real-v1/step500"
)
EXPECTED_PARENT_CONFIG_SHA256 = "41df40c299f4f3101c3ef58d657d99fb624194beaee7321ea456727212be1dad"
EXPECTED_PARENT_GATE_PATH = (
    "/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/evals/"
    "bridge-real-v1-promotion-v1/parent-gate-v2.json"
)
EXPECTED_PARENT_GATE_SHA256 = "e6dea8f8f1fd52c2b008e5460854169a893a814bd19da77b1567330116282b6a"
EXPECTED_TEXT_SENTINEL_RAW_SHA256 = (
    "f5f5c6d6771e376648e9ddeff105d1bd6440cebfd78536643278e63f40f7f4db"
)
EXPECTED_TEXT_SENTINEL_FINGERPRINT = (
    "7ff86cb7fbd34c1b5681e9ca27b7e3c8b5d69e48f9d83873387a31f56625f640"
)
EXPECTED_EVALUATION_SEED = 6198
EXPECTED_PROFILE_CONTRACTS = {
    CONTROL_ARM: {
        "name": "vision-alignment-perception-frozen-vision-control-v1",
        "repository_path": (
            "configs/vision_moe/vision_alignment/perception/frozen_vision_control_v1.yaml"
        ),
        "sha256": "0c304f74c5565edfe8675ee97b940bb7b60a175946d22ef43ce877b5b5721601",
        "trainable_contract_sha256": (
            "dad1bcf6de11527d4e5df8e7ec9901e6211336c28a0a348171479904109b08b7"
        ),
        "vision_lr": 0.0,
        "save_folder": (
            "/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/"
            "checkpoints/vision-alignment-perception-frozen-vision-control-v1"
        ),
    },
    TREATMENT_ARM: {
        "name": "vision-alignment-perception-treatment-v1",
        "repository_path": "configs/vision_moe/vision_alignment/perception/treatment_v1.yaml",
        "sha256": "9a813d50fd1f5cf3d4bb4fee49ca694f60d89fd872e7d71751169f1dd334807f",
        "trainable_contract_sha256": (
            "b8721acb806dbf023f1554917a82df4c31d61eb38a172ebf59ea6241b203fa8e"
        ),
        "vision_lr": 3e-6,
        "save_folder": (
            "/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/"
            "checkpoints/vision-alignment-perception-treatment-v1"
        ),
    },
}
ROLLING_INTERVAL_LENGTH = 128
SIGMA_FACTOR = 12
MIN_EXAMPLES_PER_SOURCE = 256
EXAMPLES_DIVISOR = 32
CORRECT_CE_MAX_RELATIVE_INCREASE = 0.02
LATE_GAP_RETENTION_FRACTION = 0.8
LOSS_MASS_ABSOLUTE_TOLERANCE = 0.02

EXPECTED_PERCEPTION_LOSS_RECIPE_GIT_COMMIT = "bfaa560362854b2b4518641b327df5f7eacfddd7"
EXPECTED_PERCEPTION_LOSS_RECIPE_REPOSITORY_PATH = "src/scripts/train/Vision-Alignment.py"
EXPECTED_PERCEPTION_LOSS_RECIPE_RECORDED_PATH = (
    "/weka/oe-training-default/rustin/OLMo-core/src/scripts/train/Vision-Alignment.py"
)
EXPECTED_PERCEPTION_LOSS_RECIPE_SHA256 = (
    "b8a96d946224e42cd0cb6422d27081da09265ea4d0e963f8e7509ac6f39267a5"
)
EXPECTED_PERCEPTION_LOSS_RECIPE_MANIFEST_RAW_SHA256 = (
    "f9fee688f10aac1b71948a74c820f7cc5cf09f6f8bc0384e208737635e93b708"
)
EXPECTED_APPROVED_PERCEPTION_PARENT_GATE_RAW_SHA256 = (
    "6f110f00becd2f6360fcb0dd8f85fd78e4bcba787087ef44f3159c5f8d486316"
)
EXPECTED_APPROVED_PERCEPTION_PROMOTION_BUNDLE_PATH = (
    "/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/evals/"
    "perception-v1-promotion-v1/promotion-bundle-bfaa56036.json"
)
EXPECTED_APPROVED_PERCEPTION_PROMOTION_BUNDLE_RAW_SHA256 = (
    "6d06a99fb9cbe5e6941689d413c8f832d8c222b28063551a2923000950cadfff"
)
EXPECTED_APPROVED_PERCEPTION_PROMOTION_BUNDLE_CONTENT_SHA256 = (
    "776cb9a4577c385a60c02f6596a39872763bbe6a8064c8a08f894bc79e38e22d"
)
EXPECTED_APPROVED_PERCEPTION_PROMOTION_POLICY_SHA256 = (
    "39a3905e89926e42f7bc10ce2b9f16ef335d28a168b9612347b3e14b67ad52ff"
)
EXPECTED_APPROVED_PERCEPTION_OUTCOME_RAW_SHA256 = (
    "b62ebe1e90a12d5204972e5697cebd65a6484a52e7a806d3d2a0be742d92a6a8"
)
EXPECTED_APPROVED_PERCEPTION_DEVIATION_SHA256 = (
    "19170e165b2407b2c3533ef5240db5e88fea7104319c36112b056eb549e90afc"
)
EXPECTED_APPROVED_PERCEPTION_APPROVED_BY = "rustins"
EXPECTED_APPROVED_PERCEPTION_APPROVED_AT = "2026-08-13T21:47:16Z"

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
_OUTCOME_EVALUATOR_PATH = (
    _SOURCE_ROOT / "scripts" / "eval" / ("vision_alignment_perception_outcome.py")
)
_PERCEPTION_STATE_TEXT_EVALUATOR_PATH = (
    _SOURCE_ROOT / "scripts" / "eval" / ("vision_alignment_perception_state_text.py")
)
_PERCEPTION_MATCHED_EVALUATOR_PATH = (
    _SOURCE_ROOT / "scripts" / "eval" / "vision_alignment_perception_matched_wrong.py"
)
_RUN_HEALTH_PRODUCER_PATH = (
    _SOURCE_ROOT / "scripts" / "eval" / "vision_alignment_perception_run_health.py"
)
_LOSS_MASS_PRODUCER_PATH = (
    _SOURCE_ROOT / "scripts" / "eval" / "vision_alignment_perception_loss_mass.py"
)
_PERCEPTION_LOSS_RECIPE_MANIFEST_PATH = (
    _SOURCE_ROOT
    / ".."
    / "configs"
    / "vision_moe"
    / "vision_alignment"
    / "promotion"
    / "perception_loss_mass_historical_recipe_v1.json"
).resolve()
_HISTORICAL_RECIPE_MANIFEST_FIELDS = frozenset(
    {
        "format",
        "version",
        "purpose",
        "repository_commit",
        "repository_path",
        "recorded_path",
        "sha256",
        "content_sha256",
    }
)
_SHA_FIELDS = (
    "checkpoint_config_sha256",
    "checkpoint_identity_sha256",
    "checkpoint_marker_sha256",
    "dcp_metadata_sha256",
    "state_file_inventory_sha256",
    "data_contract_sha256",
    "trainable_contract_sha256",
)
_CANDIDATE_FIELDS = frozenset(
    {
        "checkpoint",
        "global_step",
        "phase",
        "lineage_id",
        *_SHA_FIELDS,
        "vocab_size",
        "image_embedding_rows",
    }
)
_RECEIPT_CANDIDATE_FIELDS = frozenset(
    {"checkpoint", "global_step", "checkpoint_config_sha256", "checkpoint_identity_sha256"}
)
_ARTIFACT_REF_FIELDS = frozenset({"path", "sha256"})
_ARM_MAP_FIELDS = frozenset(ARMS)
_BUNDLE_FIELDS = frozenset(
    {
        "format",
        "version",
        "status",
        "created_at",
        "policy",
        "candidate",
        "comparator",
        "receipts",
        "deviations",
        "content_sha256",
    }
)
_BUNDLE_RECEIPT_FIELDS = frozenset(
    {
        "pair_contract",
        "initialization_parity",
        "counterfactual_outcome",
        "frozen_state",
        "text_retention",
        "run_health",
        "loss_mass_pair",
    }
)
_APPROVED_PARENT_GATE_V3_FIELDS = frozenset(
    {
        "format",
        "version",
        "status",
        "promotion_kind",
        "promotion_policy",
        "recipe_version",
        "formatter_version",
        "phase",
        "checkpoint",
        "checkpoint_config_sha256",
        "data_contract_sha256",
        "trainable_contract_sha256",
        "global_step",
        "metrics_artifact_sha256",
        "promotion_bundle_path",
        "promotion_bundle_sha256",
        "checkpoint_identity_sha256",
        "approved_by",
        "approved_at",
        "waivers",
    }
)
_APPROVED_WAIVER_FIELDS = frozenset({"id", "decision", "deviation_sha256"})
_APPROVED_DEVIATION_FIELDS = frozenset(
    {
        "id",
        "arm",
        "criterion",
        "waiver_required",
        "reason_code",
        "steps",
        "count",
        "rate",
        "minimum_spacing",
        "clean_final_steps",
        "rolling_interval_length",
        "run_id",
        "evidence_receipt_sha256",
        "sha256",
    }
)
_APPROVED_OUTCOME_FIELDS = frozenset(
    {
        "format",
        "version",
        "status",
        "created_at",
        "producer",
        "inputs",
        "protocol",
        "checkpoints",
        "sources",
        "summary",
        "content_sha256",
    }
)
_APPROVED_OUTCOME_CHECKPOINT_FIELDS = frozenset(
    {
        "root",
        "state_dir",
        "config_sha256",
        "checkpoint_marker_sha256",
        "dcp_metadata_sha256",
        "state_file_hash_algorithm",
        "state_file_inventory_sha256",
        "state_file_inventory",
        "identity_sha256",
    }
)
_APPROVED_OUTCOME_INVENTORY_FIELDS = frozenset({"path", "size", "sha256"})


def _exact_fields(value: Any, expected: frozenset[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PromotionValidationError(f"{name} must be an object")
    actual = set(value)
    if actual != expected:
        raise PromotionValidationError(
            f"{name} fields differ from the locked schema: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return value


def _sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or bridge.SHA256_RE.fullmatch(value) is None:
        raise PromotionValidationError(f"{name} must be a lowercase SHA-256")
    return value


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise PromotionValidationError(f"{name} must be a finite number")
    return float(value)


def _int(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PromotionValidationError(f"{name} must be an integer >= {minimum}")
    return value


def _strict_json_bytes(raw: bytes, *, name: str) -> Any:
    def reject_constant(value: str) -> Any:
        raise PromotionValidationError(f"{name} contains non-finite constant {value}")

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=bridge._strict_json_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PromotionValidationError(f"Could not parse {name}: {error}") from error


def load_json_pinned(path: Path, expected_sha256: str, *, name: str) -> Any:
    """Read, hash, and strictly parse one immutable JSON byte snapshot."""
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise PromotionValidationError(f"Could not read {name} {path}: {error}") from error
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise PromotionValidationError(
            f"{name} raw SHA-256 differs: expected {expected_sha256}, got {actual}"
        )
    return _strict_json_bytes(raw, name=name)


def load_trainer_state(path: Path, *, expected_sha256: str | None = None) -> Mapping[str, Any]:
    """Safely load a native trainer state without permitting arbitrary pickle globals.

    The completed perception checkpoints contain NumPy's legacy RNG-state array in addition to
    tensors and primitive containers. ``weights_only=True`` rejects arbitrary pickle code while
    this narrow allowlist admits only the NumPy reconstruction and dtype classes needed by those
    RNG arrays.

    :param path: Exact SHA-pinned trainer-state path.
    :returns: The decoded trainer-state mapping.
    """
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise PromotionValidationError(f"Could not read trainer state {path}: {error}") from error
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise PromotionValidationError(f"Trainer state {path} SHA-256 differs")
    try:
        import torch
    except ImportError as error:  # pragma: no cover - OLMo-core always requires torch.
        raise PromotionValidationError("PyTorch is required to inspect trainer state") from error
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
            value = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as error:
        raise PromotionValidationError(
            f"Could not safely load trainer state {path}: {error}"
        ) from error
    if not isinstance(value, Mapping):
        raise PromotionValidationError(f"Trainer state {path} must be an object")
    return value


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _stable_file_record(path: Path, *, root: Path) -> tuple[dict[str, Any], tuple[int, ...]]:
    """Hash one non-symlink regular file and retain its immutable-entry signature."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise PromotionValidationError(f"Could not open immutable file {path}: {error}") from error
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PromotionValidationError(f"Immutable evidence is not a regular file: {path}")
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        current = path.lstat()
    except OSError as error:
        raise PromotionValidationError(f"Immutable file disappeared: {path}") from error
    signature = _stat_identity(before)
    if (
        signature != _stat_identity(after)
        or signature != _stat_identity(current)
        or size != before.st_size
    ):
        raise PromotionValidationError(f"Immutable file changed while hashing: {path}")
    return (
        {
            "path": path.relative_to(root).as_posix(),
            "size": size,
            "sha256": digest.hexdigest(),
        },
        signature,
    )


def _direct_existing_path(path: Path, *, name: str) -> Path:
    """Return an absolute path while rejecting every symlinked component."""
    absolute = Path(os.path.abspath(path.expanduser()))
    for component in (*reversed(absolute.parents), absolute):
        if component == Path(component.anchor):
            continue
        try:
            info = component.lstat()
        except OSError as error:
            raise PromotionValidationError(
                f"{name} component is unavailable: {component}: {error}"
            ) from error
        if stat.S_ISLNK(info.st_mode):
            raise PromotionValidationError(f"{name} contains a symlinked component: {component}")
    return absolute


def _validate_live_checkpoint_identity_stable(
    identity: Mapping[str, Any], *, name: str, hash_workers: int = 16
) -> None:
    """Re-hash a checkpoint as one stable non-symlinked filesystem snapshot."""
    root = _direct_existing_path(Path(str(identity["root"])), name=f"{name} checkpoint root")
    state_dir = _direct_existing_path(
        Path(str(identity["state_dir"])), name=f"{name} checkpoint state directory"
    )
    expected_inventory = identity["state_file_inventory"]
    if not isinstance(expected_inventory, list) or not expected_inventory:
        raise PromotionValidationError(f"{name} checkpoint inventory is empty")
    expected_paths = [root / str(item["path"]) for item in expected_inventory]
    actual_entries = sorted(state_dir.iterdir())
    if actual_entries != sorted(expected_paths):
        raise PromotionValidationError(f"Live {name} checkpoint entries differ")
    support_paths = [root / "config.json", root / ".metadata.json"]
    all_paths = [*expected_paths, *support_paths]
    initial: dict[Path, tuple[int, ...]] = {}
    for path in all_paths:
        try:
            info = path.lstat()
        except OSError as error:
            raise PromotionValidationError(f"Live {name} file is unavailable: {path}") from error
        if not stat.S_ISREG(info.st_mode):
            raise PromotionValidationError(f"Live {name} contains a symlink/non-file: {path}")
        initial[path] = _stat_identity(info)
    with ThreadPoolExecutor(max_workers=min(hash_workers, len(expected_paths))) as executor:
        records = list(
            executor.map(lambda path: _stable_file_record(path, root=root)[0], expected_paths)
        )
    config_record, _ = _stable_file_record(support_paths[0], root=root)
    marker_record, _ = _stable_file_record(support_paths[1], root=root)
    if actual_entries != sorted(state_dir.iterdir()):
        raise PromotionValidationError(f"Live {name} checkpoint entries changed during hashing")
    for path, signature in initial.items():
        if _stat_identity(path.lstat()) != signature:
            raise PromotionValidationError(f"Live {name} file changed during snapshot: {path}")
    if records != expected_inventory:
        raise PromotionValidationError(f"Live {name} DCP inventory differs")
    metadata_path = (state_dir / ".metadata").relative_to(root).as_posix()
    metadata = next((record for record in records if record["path"] == metadata_path), None)
    if (
        config_record["sha256"] != identity["config_sha256"]
        or marker_record["sha256"] != identity["checkpoint_marker_sha256"]
        or not isinstance(metadata, Mapping)
        or metadata["sha256"] != identity["dcp_metadata_sha256"]
        or canonical_sha256(records) != identity["state_file_inventory_sha256"]
    ):
        raise PromotionValidationError(f"Live {name} stable checkpoint identity differs")


def _timestamp(value: Any, *, name: str) -> datetime:
    if not isinstance(value, str):
        raise PromotionValidationError(f"{name} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise PromotionValidationError(f"{name} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PromotionValidationError(f"{name} must include a timezone")
    return parsed


def _resolved_path(value: Any, *, name: str, must_exist: bool = True) -> Path:
    if not isinstance(value, str) or not value:
        raise PromotionValidationError(f"{name} must be a non-empty path")
    path = Path(value).expanduser().resolve()
    if must_exist and not path.exists():
        raise PromotionValidationError(f"{name} does not exist: {path}")
    return path


def _validate_content_sha(payload: Mapping[str, Any], *, name: str) -> None:
    expected = _sha256(payload.get("content_sha256"), name=f"{name} content SHA-256")
    actual = canonical_sha256(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    if actual != expected:
        raise PromotionValidationError(f"{name} content SHA-256 differs")


def _load_reference(reference: Any, *, name: str) -> tuple[Path, Mapping[str, Any]]:
    path, raw = _load_raw_reference(reference, name=name)
    payload = _strict_json_bytes(raw, name=name)
    if not isinstance(payload, Mapping):
        raise PromotionValidationError(f"{name} must be a JSON object")
    return path, payload


def _validate_raw_reference(reference: Any, *, name: str) -> Path:
    path, _ = _load_raw_reference(reference, name=name)
    return path


def _load_raw_reference(reference: Any, *, name: str) -> tuple[Path, bytes]:
    """Read and hash one raw artifact once so semantic checks consume the pinned bytes."""
    ref = _exact_fields(reference, _ARTIFACT_REF_FIELDS, name=f"{name} reference")
    raw_path = Path(str(ref["path"])).expanduser()
    path = _direct_existing_path(raw_path, name=f"{name} path")
    expected = _sha256(ref["sha256"], name=f"{name} raw SHA-256")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise PromotionValidationError(f"Could not read {name} {path}: {error}") from error
    if hashlib.sha256(raw).hexdigest() != expected:
        raise PromotionValidationError(f"{name} raw SHA-256 differs")
    return path, raw


def _validate_implementation_reference(
    reference: Any, *, name: str, basename: str, canonical_path: Path
) -> Path:
    """Validate a source pin while allowing an unavailable Gantry-recorded checkout path."""
    return bridge._validate_implementation_reference(
        reference,
        name=name,
        expected_basename=basename,
        canonical_path=canonical_path,
    )


def _validate_historical_loss_recipe_reference(reference: Any, *, name: str) -> Path:
    """Validate the one reviewed recipe snapshot without re-hashing the evolving launcher.

    The loss-mass receipt records the exact recipe that constructed the saved loader states. The
    training launcher is expected to evolve after promotion, so re-hashing its live canonical path
    would make approved evidence self-invalidating. This exception is deliberately limited to the
    exact path and SHA recorded in a raw-SHA-pinned reviewed manifest; all executable evidence
    producers continue to use :func:`_validate_implementation_reference` and live validation.
    """
    manifest = load_json_pinned(
        _PERCEPTION_LOSS_RECIPE_MANIFEST_PATH,
        EXPECTED_PERCEPTION_LOSS_RECIPE_MANIFEST_RAW_SHA256,
        name="reviewed historical perception loss recipe manifest",
    )
    manifest = _exact_fields(
        manifest,
        _HISTORICAL_RECIPE_MANIFEST_FIELDS,
        name="reviewed historical perception loss recipe manifest",
    )
    _validate_content_sha(manifest, name="reviewed historical perception loss recipe manifest")
    if (
        manifest["format"] != "vision_alignment_reviewed_historical_recipe"
        or type(manifest["version"]) is not int
        or manifest["version"] != 1
        or manifest["purpose"] != "perception_loss_mass_evidence"
        or manifest["repository_commit"] != EXPECTED_PERCEPTION_LOSS_RECIPE_GIT_COMMIT
        or manifest["repository_path"] != EXPECTED_PERCEPTION_LOSS_RECIPE_REPOSITORY_PATH
        or manifest["recorded_path"] != EXPECTED_PERCEPTION_LOSS_RECIPE_RECORDED_PATH
        or manifest["sha256"] != EXPECTED_PERCEPTION_LOSS_RECIPE_SHA256
    ):
        raise PromotionValidationError(
            "Reviewed historical perception loss recipe manifest differs from its allowlist"
        )

    ref = _exact_fields(reference, _ARTIFACT_REF_FIELDS, name=f"{name} reference")
    if ref["path"] != manifest["recorded_path"]:
        raise PromotionValidationError(f"{name} names an incompatible historical path")
    expected_sha = _sha256(ref["sha256"], name=f"{name} reference SHA-256")
    if expected_sha != manifest["sha256"]:
        raise PromotionValidationError(f"{name} differs from the reviewed historical pin")
    return Path(str(ref["path"]))


def _validate_receipt_header(
    receipt: Mapping[str, Any],
    *,
    expected_format: str,
    expected_fields: frozenset[str],
    name: str,
) -> None:
    _exact_fields(receipt, expected_fields, name=name)
    if (
        receipt["format"] != expected_format
        or type(receipt["version"]) is not int
        or receipt["version"] != RECEIPT_VERSION
        or receipt["status"] != "passed"
    ):
        raise PromotionValidationError(f"{name} identity or status is incompatible")
    _timestamp(receipt["created_at"], name=f"{name} created_at")
    _validate_content_sha(receipt, name=name)


def _receipt_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint": candidate["checkpoint"],
        "global_step": candidate["global_step"],
        "checkpoint_config_sha256": candidate["checkpoint_config_sha256"],
        "checkpoint_identity_sha256": candidate["checkpoint_identity_sha256"],
    }


def _validate_receipt_candidate(
    value: Any, *, expected: Mapping[str, Any], name: str
) -> Mapping[str, Any]:
    candidate = _exact_fields(value, _RECEIPT_CANDIDATE_FIELDS, name=name)
    _int(candidate["global_step"], name=f"{name} global step")
    expected_candidate = _receipt_candidate(expected)
    if dict(candidate) != expected_candidate:
        raise PromotionValidationError(f"{name} differs from the selected checkpoint")
    return candidate


def _validate_arm(value: Any, *, name: str) -> str:
    if value not in ARMS:
        raise PromotionValidationError(f"{name} must be one of {list(ARMS)}")
    return str(value)


def _outcome_role(role: str) -> str:
    if role == "control":
        return CONTROL_ARM
    if role == "treatment":
        return TREATMENT_ARM
    raise PromotionValidationError("Outcome role must be 'treatment' or 'control'")


def _read_candidate_config(
    checkpoint: Path,
    identity: Mapping[str, Any],
    *,
    arm: str,
    verify_live_contents: bool,
) -> dict[str, Any]:
    root = Path(str(identity["root"])).expanduser().resolve()
    if root != checkpoint or root.name != f"step{PRIMARY_STEP}":
        raise PromotionValidationError(f"{arm} candidate must be the evaluated step4000")
    expected = EXPECTED_PROFILE_CONTRACTS[arm]
    expected_root = Path(str(expected["save_folder"])).resolve() / f"step{PRIMARY_STEP}"
    if root != expected_root:
        raise PromotionValidationError(f"{arm} candidate is outside the locked production run")
    if verify_live_contents:
        _validate_live_checkpoint_identity_stable(identity, name=f"perception {arm} step4000")
    config_path = root / "config.json"
    config = load_json_pinned(
        config_path, str(identity["config_sha256"]), name=f"{arm} candidate config"
    )
    if not isinstance(config, Mapping):
        raise PromotionValidationError(f"{arm} candidate config must be an object")
    metadata = config.get("vision_alignment")
    model = config.get("model")
    train_module = config.get("train_module")
    if (
        config.get("phase") != "perception"
        or config.get("perception_trainability_arm") != arm
        or not isinstance(metadata, Mapping)
        or metadata.get("phase") != "perception"
        or not isinstance(model, Mapping)
        or not isinstance(model.get("lm"), Mapping)
        or not isinstance(train_module, Mapping)
    ):
        raise PromotionValidationError(f"{arm} candidate is not the expected perception arm")
    for field in ("data_contract_sha256", "trainable_contract_sha256"):
        _sha256(metadata.get(field), name=f"{arm} {field}")
    lineage = metadata.get("lineage_id")
    trainer = config.get("trainer")
    launch = config.get("launch")
    initialization = config.get("initialization")
    if (
        not isinstance(trainer, Mapping)
        or not isinstance(launch, Mapping)
        or not isinstance(initialization, Mapping)
    ):
        raise PromotionValidationError(f"{arm} runtime identity sections are missing")
    callbacks = trainer.get("callbacks")
    launch_git = launch.get("git")
    if not isinstance(callbacks, Mapping) or not isinstance(launch_git, Mapping):
        raise PromotionValidationError(f"{arm} callback or Git identity is missing")
    wandb = callbacks.get("wandb")
    expected_name = expected["name"]
    expected_profile_path = expected["repository_path"]
    if (
        lineage != expected_name
        or config.get("required_run_name") != expected_name
        or config.get("reviewed_profile_path") != expected_profile_path
        or config.get("reviewed_profile_sha256") != expected["sha256"]
        or metadata.get("trainable_contract_sha256") != expected["trainable_contract_sha256"]
        or Path(str(trainer.get("save_folder", ""))).resolve()
        != Path(str(expected["save_folder"])).resolve()
        or not isinstance(wandb, Mapping)
        or wandb.get("name") != expected_name
        or config.get("expected_launch_command")
        != [
            "src/scripts/train/Vision-Alignment.py",
            "train",
            expected_name,
            f"--profile={expected_profile_path}",
        ]
        or launch_git.get("ref") != EXPECTED_GIT_REF
        or Path(str(initialization.get("checkpoint", ""))).resolve()
        != Path(EXPECTED_PARENT_CHECKPOINT).resolve()
        or initialization.get("expected_parent_phase") != "bridge"
        or initialization.get("parent_config_sha256") != EXPECTED_PARENT_CONFIG_SHA256
        or Path(str(initialization.get("parent_gate_path", ""))).resolve()
        != Path(EXPECTED_PARENT_GATE_PATH).resolve()
        or initialization.get("parent_gate_sha256") != EXPECTED_PARENT_GATE_SHA256
    ):
        raise PromotionValidationError(f"{arm} reviewed identity differs from policy")
    vocab_size = _int(model["lm"].get("vocab_size"), name=f"{arm} vocab size", minimum=1)
    rows = train_module.get("train_embedding_rows")
    if rows != list(IMAGE_TOKEN_ROWS) or any(row >= vocab_size for row in rows):
        raise PromotionValidationError(f"{arm} image embedding rows differ from policy")
    freeze = train_module.get("freeze_params")
    expected_freeze = (
        ["vision.*", "lm.embedding_norm.*", "lm.blocks.*", "lm.lm_head.*"]
        if arm == CONTROL_ARM
        else ["lm.embedding_norm.*", "lm.blocks.*", "lm.lm_head.*"]
    )
    if freeze != expected_freeze:
        raise PromotionValidationError(f"{arm} freeze surface differs from the causal contract")
    optim = train_module.get("optim")
    group_overrides = optim.get("group_overrides") if isinstance(optim, Mapping) else None
    if not isinstance(group_overrides, list):
        raise PromotionValidationError(f"{arm} optimizer groups are missing")
    vision_groups = [
        item
        for item in group_overrides
        if isinstance(item, Mapping) and item.get("params") == ["*vision.*"]
    ]
    if len(vision_groups) != 1 or not isinstance(vision_groups[0].get("opts"), Mapping):
        raise PromotionValidationError(f"{arm} vision optimizer group differs from policy")
    if vision_groups[0]["opts"].get("lr") != expected["vision_lr"]:
        raise PromotionValidationError(f"{arm} vision learning rate differs from policy")
    return {
        "checkpoint": str(root),
        "global_step": PRIMARY_STEP,
        "phase": "perception",
        "lineage_id": lineage,
        "checkpoint_config_sha256": identity["config_sha256"],
        "checkpoint_identity_sha256": identity["identity_sha256"],
        "checkpoint_marker_sha256": identity["checkpoint_marker_sha256"],
        "dcp_metadata_sha256": identity["dcp_metadata_sha256"],
        "state_file_inventory_sha256": identity["state_file_inventory_sha256"],
        "data_contract_sha256": metadata["data_contract_sha256"],
        "trainable_contract_sha256": metadata["trainable_contract_sha256"],
        "vocab_size": vocab_size,
        "image_embedding_rows": list(rows),
    }


def candidate_from_outcome_receipt(
    checkpoint: Path,
    receipt: Mapping[str, Any],
    *,
    role: Literal["treatment", "control"] = "treatment",
    verify_live_contents: bool = True,
) -> dict[str, Any]:
    """Derive a bridge-compatible perception candidate from an outcome receipt.

    :param checkpoint: Exact live step-4000 checkpoint selected for ``role``.
    :param receipt: Parsed perception outcome receipt.
    :param role: ``"treatment"`` or ``"control"``.
    :param verify_live_contents: Re-hash all DCP files when true.
    :returns: The normalized candidate identity used by existing state/text receipt validators.
    """
    arm = _outcome_role(role)
    if not isinstance(receipt, Mapping) or receipt.get("format") != OUTCOME_RECEIPT_FORMAT:
        raise PromotionValidationError("Perception outcome receipt identity is incompatible")
    if (
        type(receipt.get("version")) is not int
        or receipt.get("version") != RECEIPT_VERSION
        or receipt.get("status") != "passed"
    ):
        raise PromotionValidationError("Perception outcome receipt did not pass")
    checkpoints = _exact_fields(
        receipt.get("checkpoints"), _ARM_MAP_FIELDS, name="outcome checkpoints"
    )
    arm_checkpoints = _exact_fields(
        checkpoints[arm], frozenset({"step3000", "step4000"}), name=f"{arm} checkpoints"
    )
    identity = bridge._validate_checkpoint_identity(
        arm_checkpoints["step4000"], name=f"outcome {arm} step4000"
    )
    return _read_candidate_config(
        checkpoint.expanduser().resolve(),
        identity,
        arm=arm,
        verify_live_contents=verify_live_contents,
    )


def validate_pair_contract_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the published v2 causal profile-pair receipt and its live code/profile pins."""
    fields = frozenset(
        {
            "comparison",
            "data",
            "format",
            "git",
            "initialization",
            "launch_contract",
            "perception_contract",
            "producer",
            "profiles",
            "recipe",
            "recipe_execution_module",
            "review_allowlist",
            "save_folders",
            "status",
            "version",
        }
    )
    _exact_fields(receipt, fields, name="perception pair contract")
    if canonical_sha256(receipt) != EXPECTED_PAIR_CONTRACT_CONTENT_SHA256:
        raise PromotionValidationError("Perception pair contract differs from the published v2")
    if (
        receipt["format"] != PAIR_CONTRACT_RECEIPT_FORMAT
        or type(receipt["version"]) is not int
        or receipt["version"] != PAIR_CONTRACT_RECEIPT_VERSION
        or receipt["status"] != "passed"
        or receipt["recipe_execution_module"] != "__main__"
    ):
        raise PromotionValidationError("Perception pair contract identity is incompatible")
    for name in ("producer", "recipe", "review_allowlist"):
        value = receipt[name]
        if not isinstance(value, Mapping) or not {"path", "sha256"} <= set(value):
            raise PromotionValidationError(f"Pair-contract {name} reference is malformed")
        _validate_raw_reference(
            {"path": value["path"], "sha256": value["sha256"]}, name=f"pair-contract {name}"
        )
    profiles = _exact_fields(receipt["profiles"], _ARM_MAP_FIELDS, name="pair profiles")
    for arm in ARMS:
        profile = profiles[arm]
        if not isinstance(profile, Mapping) or not {"name", "path", "sha256"} <= set(profile):
            raise PromotionValidationError(f"Pair-contract {arm} profile is malformed")
        _validate_raw_reference(
            {"path": profile["path"], "sha256": profile["sha256"]},
            name=f"pair-contract {arm} profile",
        )
    launch = receipt["launch_contract"]
    if not isinstance(launch, Mapping) or launch.get("workspace") != "ai2/molmofication":
        raise PromotionValidationError("Perception pair contract uses the wrong Beaker workspace")
    if (
        _int(launch.get("num_nodes"), name="pair launch nodes") != 2
        or _int(launch.get("num_gpus"), name="pair launch GPUs") != 8
    ):
        raise PromotionValidationError("Perception pair topology differs from 2x8 GPUs")
    comparison = receipt["comparison"]
    data = receipt["data"]
    initialization = receipt["initialization"]
    contract = receipt["perception_contract"]
    if not all(
        isinstance(value, Mapping) for value in (comparison, data, initialization, contract)
    ):
        raise PromotionValidationError("Perception pair contract sections must be objects")
    trainable = _exact_fields(
        comparison.get("trainable_contract_sha256"), _ARM_MAP_FIELDS, name="pair trainability"
    )
    for arm in ARMS:
        _sha256(trainable[arm], name=f"pair {arm} trainable contract")
    data_sha = _sha256(data.get("data_contract_sha256"), name="pair data contract")
    parent_config_sha = _sha256(
        initialization.get("parent_config_sha256"), name="pair parent config"
    )
    parent_gate_sha = _sha256(initialization.get("parent_gate_sha256"), name="pair parent gate")
    parent_checkpoint = _resolved_path(
        initialization.get("checkpoint"), name="pair parent checkpoint"
    )
    if (
        parent_checkpoint != Path(EXPECTED_PARENT_CHECKPOINT).resolve()
        or parent_config_sha != EXPECTED_PARENT_CONFIG_SHA256
        or parent_gate_sha != EXPECTED_PARENT_GATE_SHA256
        or Path(str(initialization.get("parent_gate_path", ""))).resolve()
        != Path(EXPECTED_PARENT_GATE_PATH).resolve()
    ):
        raise PromotionValidationError("Perception parent must be bridge step500")
    duration = contract.get("duration")
    evaluation = contract.get("evaluation")
    if (
        not isinstance(duration, Mapping)
        or duration.get("unit") != "steps"
        or duration.get("value") != PRIMARY_STEP
        or not isinstance(evaluation, Mapping)
        or evaluation.get("eval_on_finish") is not True
        or evaluation.get("eval_on_startup") is not True
        or evaluation.get("seed") != EXPECTED_EVALUATION_SEED
        or evaluation.get("examples_per_source") != 512
    ):
        raise PromotionValidationError("Perception pair duration/evaluation contract differs")
    return {
        "data_contract_sha256": data_sha,
        "trainable_contract_sha256": dict(trainable),
        "parent_checkpoint": str(parent_checkpoint),
        "parent_config_sha256": parent_config_sha,
        "parent_gate_sha256": parent_gate_sha,
        "profiles": {
            arm: {"name": profiles[arm]["name"], "sha256": profiles[arm]["sha256"]} for arm in ARMS
        },
        "evaluation_seed": evaluation.get("seed"),
        "examples_per_source": evaluation.get("examples_per_source"),
    }


def _load_published_pair_contract(reference: Any) -> tuple[Mapping[str, Any], dict[str, Any]]:
    ref = _exact_fields(reference, _ARTIFACT_REF_FIELDS, name="pair-contract reference")
    if ref["sha256"] != EXPECTED_PAIR_CONTRACT_RAW_SHA256:
        raise PromotionValidationError("Pair contract is not the published v2 receipt")
    _, payload = _load_reference(ref, name="pair contract")
    summary = validate_pair_contract_receipt(payload)

    gate_path = Path(EXPECTED_PARENT_GATE_PATH).resolve()
    gate = load_json_pinned(
        gate_path, EXPECTED_PARENT_GATE_SHA256, name="published bridge parent gate"
    )
    if not isinstance(gate, Mapping):
        raise PromotionValidationError("Published bridge parent gate must be an object")
    if (
        gate.get("format") != "vision_alignment_parent_gate"
        or type(gate.get("version")) is not int
        or gate.get("version") != 2
        or gate.get("status") != "approved"
        or gate.get("phase") != "bridge"
        or _int(gate.get("global_step"), name="bridge gate global step") != 500
        or Path(str(gate.get("checkpoint", ""))).resolve()
        != Path(EXPECTED_PARENT_CHECKPOINT).resolve()
        or gate.get("checkpoint_config_sha256") != EXPECTED_PARENT_CONFIG_SHA256
        or not isinstance(gate.get("checkpoint_identity_sha256"), str)
    ):
        raise PromotionValidationError("Published bridge parent gate identity differs")
    bundle_path = _resolved_path(
        gate.get("promotion_bundle_path"), name="published bridge promotion bundle"
    )
    bundle_sha = _sha256(
        gate.get("promotion_bundle_sha256"), name="published bridge promotion bundle SHA-256"
    )
    if gate.get("metrics_artifact_sha256") != bundle_sha:
        raise PromotionValidationError("Bridge gate metrics and bundle identities differ")
    bundle = load_json_pinned(bundle_path, bundle_sha, name="published bridge promotion bundle")
    if not isinstance(bundle, Mapping):
        raise PromotionValidationError("Published bridge promotion bundle must be an object")
    parent_identity = _validate_published_bridge_bundle(bundle, gate=gate)
    summary["parent_checkpoint_identity_sha256"] = parent_identity
    return payload, summary


def _validate_published_bridge_bundle(bundle: Mapping[str, Any], *, gate: Mapping[str, Any]) -> str:
    """Validate the immutable approved bridge evidence without unsafe legacy pickle loads.

    The historical bridge validator predates same-buffer safe deserialization. Re-executing it
    here would reopen trainer-state pickle paths after hashing them. The human-approved v2 gate's
    hard-coded raw SHA is the trust anchor; this adapter validates its bundle, candidate, waiver,
    and every transitive receipt byte identity without interpreting historical pickle payloads.
    The perception initialization receipt then re-hashes the complete live parent checkpoint and
    binds that exact identity back to this gate.
    """
    expected_bundle_fields = frozenset(
        {
            "format",
            "version",
            "status",
            "created_at",
            "policy",
            "candidate",
            "receipts",
            "deviations",
            "content_sha256",
        }
    )
    _exact_fields(bundle, expected_bundle_fields, name="published bridge promotion bundle")
    if (
        bundle["format"] != bridge.PROMOTION_BUNDLE_FORMAT
        or type(bundle["version"]) is not int
        or bundle["version"] != bridge.PROMOTION_BUNDLE_VERSION
        or bundle["status"] != "ready_for_human_approval"
        or bundle["policy"] != bridge._promotion_policy()
    ):
        raise PromotionValidationError("Published bridge promotion bundle identity differs")
    _timestamp(bundle["created_at"], name="published bridge bundle created_at")
    _validate_content_sha(bundle, name="published bridge promotion bundle")

    candidate_fields = frozenset(
        {
            "checkpoint",
            "global_step",
            "phase",
            "lineage_id",
            "checkpoint_config_sha256",
            "checkpoint_identity_sha256",
            "checkpoint_marker_sha256",
            "dcp_metadata_sha256",
            "state_file_inventory_sha256",
            "data_contract_sha256",
            "trainable_contract_sha256",
            "vocab_size",
            "image_embedding_rows",
        }
    )
    candidate = _exact_fields(
        bundle["candidate"], candidate_fields, name="published bridge candidate"
    )
    parent_identity = _sha256(
        gate["checkpoint_identity_sha256"], name="published bridge checkpoint identity"
    )
    if (
        Path(str(candidate["checkpoint"])).resolve() != Path(EXPECTED_PARENT_CHECKPOINT).resolve()
        or _int(candidate["global_step"], name="published bridge candidate step") != 500
        or candidate["phase"] != "bridge"
        or candidate["checkpoint_config_sha256"] != EXPECTED_PARENT_CONFIG_SHA256
        or candidate["checkpoint_identity_sha256"] != parent_identity
        or candidate["data_contract_sha256"] != gate.get("data_contract_sha256")
        or candidate["trainable_contract_sha256"] != gate.get("trainable_contract_sha256")
        or candidate["image_embedding_rows"] != list(IMAGE_TOKEN_ROWS)
    ):
        raise PromotionValidationError("Published bridge bundle candidate differs from its gate")
    for field in (
        "checkpoint_marker_sha256",
        "dcp_metadata_sha256",
        "state_file_inventory_sha256",
        "data_contract_sha256",
        "trainable_contract_sha256",
    ):
        _sha256(candidate[field], name=f"published bridge candidate {field}")
    _int(candidate["vocab_size"], name="published bridge vocabulary size", minimum=1)
    if not isinstance(candidate["lineage_id"], str) or not candidate["lineage_id"]:
        raise PromotionValidationError("Published bridge candidate lineage is malformed")

    receipts = _exact_fields(
        bundle["receipts"],
        frozenset(
            {
                "frozen_state",
                "text_retention",
                "cumulative_loss_mass",
                "optimizer_guard",
                "matched_wrong",
            }
        ),
        name="published bridge receipt references",
    )
    for name in ("frozen_state", "text_retention", "cumulative_loss_mass", "optimizer_guard"):
        _load_raw_reference(receipts[name], name=f"published bridge {name}")
    matched = _exact_fields(
        receipts["matched_wrong"],
        frozenset(
            {
                "canary_step250",
                "bridge_step250",
                "bridge_step500",
                "independent_step0",
                "independent_step500",
            }
        ),
        name="published bridge matched-wrong references",
    )
    for role, receipt_reference in matched.items():
        _load_raw_reference(receipt_reference, name=f"published bridge {role}")

    deviations = bundle["deviations"]
    waivers = gate.get("waivers")
    if not isinstance(deviations, list) or not isinstance(waivers, list):
        raise PromotionValidationError("Published bridge deviation evidence is malformed")
    deviation_by_id: dict[str, Mapping[str, Any]] = {}
    for index, deviation in enumerate(deviations):
        if not isinstance(deviation, Mapping):
            raise PromotionValidationError(f"Published bridge deviation {index} is malformed")
        deviation_id = deviation.get("id")
        if not isinstance(deviation_id, str) or deviation_id in deviation_by_id:
            raise PromotionValidationError("Published bridge deviation IDs are malformed")
        unsigned = {key: value for key, value in deviation.items() if key != "sha256"}
        if _sha256(
            deviation.get("sha256"), name=f"{deviation_id} deviation SHA-256"
        ) != canonical_sha256(unsigned):
            raise PromotionValidationError(f"Published bridge deviation {deviation_id} differs")
        deviation_by_id[deviation_id] = deviation
    if set(deviation_by_id) != bridge.REQUIRED_WAIVER_IDS or len(waivers) != len(
        bridge.REQUIRED_WAIVER_IDS
    ):
        raise PromotionValidationError("Published bridge waiver set differs")
    for waiver in waivers:
        if not isinstance(waiver, Mapping) or set(waiver) != {
            "id",
            "decision",
            "deviation_sha256",
        }:
            raise PromotionValidationError("Published bridge waiver is malformed")
        deviation = deviation_by_id.get(str(waiver["id"]))
        if (
            deviation is None
            or waiver["decision"] != "approved"
            or waiver["deviation_sha256"] != deviation["sha256"]
        ):
            raise PromotionValidationError("Published bridge waiver differs from its deviation")
    return parent_identity


def _load_outcome_module() -> ModuleType:
    if not _OUTCOME_EVALUATOR_PATH.is_file():
        raise PromotionValidationError(
            f"Canonical perception outcome evaluator is unavailable: {_OUTCOME_EVALUATOR_PATH}"
        )
    spec = importlib.util.spec_from_file_location(
        "_vision_alignment_perception_outcome_for_promotion", _OUTCOME_EVALUATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise PromotionValidationError("Could not load the perception outcome evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_run_health_module() -> ModuleType:
    if not _RUN_HEALTH_PRODUCER_PATH.is_file():
        raise PromotionValidationError(
            f"Canonical run-health producer is unavailable: {_RUN_HEALTH_PRODUCER_PATH}"
        )
    spec = importlib.util.spec_from_file_location(
        "_vision_alignment_perception_run_health_for_promotion", _RUN_HEALTH_PRODUCER_PATH
    )
    if spec is None or spec.loader is None:
        raise PromotionValidationError("Could not load the perception run-health producer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_loss_mass_module() -> ModuleType:
    if not _LOSS_MASS_PRODUCER_PATH.is_file():
        raise PromotionValidationError(
            f"Canonical loss-mass producer is unavailable: {_LOSS_MASS_PRODUCER_PATH}"
        )
    spec = importlib.util.spec_from_file_location(
        "_vision_alignment_perception_loss_mass_for_promotion", _LOSS_MASS_PRODUCER_PATH
    )
    if spec is None or spec.loader is None:
        raise PromotionValidationError("Could not load the perception loss-mass producer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_counterfactual_outcome_receipt(
    receipt: Mapping[str, Any], *, verify_live_inputs: bool = True
) -> dict[str, Any]:
    """Validate the four-receipt paired outcome and enforce the locked promotion policy."""
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("format") != OUTCOME_RECEIPT_FORMAT
        or type(receipt.get("version")) is not int
        or receipt.get("version") != RECEIPT_VERSION
        or receipt.get("status") != "passed"
    ):
        raise PromotionValidationError("Perception outcome receipt identity is incompatible")
    module = _load_outcome_module()
    validator = getattr(module, "validate_outcome_receipt", None)
    if not callable(validator):
        raise PromotionValidationError("Perception outcome evaluator lacks its validator")
    try:
        normalized = validator(receipt, verify_live_inputs=verify_live_inputs)
    except (OSError, TypeError, ValueError) as error:
        raise PromotionValidationError(
            f"Perception outcome receipt failed validation: {error}"
        ) from error
    if not isinstance(normalized, Mapping):
        raise PromotionValidationError("Perception outcome validator returned no summary")
    protocol = receipt.get("protocol")
    if not isinstance(protocol, Mapping):
        raise PromotionValidationError("Perception outcome protocol is missing")
    examples = _int(
        protocol.get("examples_per_source"), name="outcome examples per source", minimum=1
    )
    if examples < MIN_EXAMPLES_PER_SOURCE or examples % EXAMPLES_DIVISOR:
        raise PromotionValidationError(
            f"Outcome examples/source must be >= {MIN_EXAMPLES_PER_SOURCE} and divisible by "
            f"{EXAMPLES_DIVISOR}"
        )
    if (
        _int(protocol.get("primary_step"), name="outcome primary step") != PRIMARY_STEP
        or _int(protocol.get("durability_step"), name="outcome durability step") != DURABILITY_STEP
        or _int(protocol.get("pairing_seed"), name="outcome pairing seed")
        != EXPECTED_EVALUATION_SEED
    ):
        raise PromotionValidationError("Outcome endpoint or pairing seed differs from policy")
    sources = protocol.get("sources")
    if sources != list(SOURCES):
        raise PromotionValidationError("Outcome source order differs from the locked eight sources")
    policy = _outcome_policy_metrics(receipt, normalized)
    _enforce_outcome_policy(policy)
    return {
        "normalized": dict(normalized),
        "policy_metrics": policy,
        "examples_per_source": examples,
    }


def _metric_path(value: Mapping[str, Any], paths: Sequence[Sequence[str]], *, name: str) -> float:
    for path in paths:
        current: Any = value
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                break
            current = current[key]
        else:
            return _finite(current, name=name)
    raise PromotionValidationError(f"Outcome validator summary lacks {name}")


def _outcome_policy_metrics(
    receipt: Mapping[str, Any], normalized: Mapping[str, Any]
) -> dict[str, Any]:
    """Normalize the comparator's re-derived summary without trusting stored receipt claims."""
    source_value = normalized.get("sources")
    macro_value = normalized.get("macro") or normalized.get("summary")
    if not isinstance(source_value, Mapping) or not isinstance(macro_value, Mapping):
        # The outcome producer's receipt summary is allowed only if its validator explicitly
        # returns the same object under ``policy_metrics``.
        policy = normalized.get("policy_metrics")
        if isinstance(policy, Mapping):
            return dict(policy)
        raise PromotionValidationError("Outcome validator lacks normalized source/macro metrics")
    step4000 = macro_value.get("step4000") or macro_value.get(str(PRIMARY_STEP))
    step3000 = macro_value.get("step3000") or macro_value.get(str(DURABILITY_STEP))
    if not isinstance(step4000, Mapping) or not isinstance(step3000, Mapping):
        raise PromotionValidationError("Outcome validator lacks both matched endpoints")
    macro = {
        "did_ci_low": _metric_path(
            step4000,
            (("did", "ci", "low"), ("gap_improvement", "ci", "low"), ("did_ci_low",)),
            name="step4000 source-balanced DID lower CI",
        ),
        "treatment_gap_ci_low": _metric_path(
            step4000,
            (("treatment", "gap", "ci", "low"), ("treatment_gap_ci_low",)),
            name="step4000 treatment gap lower CI",
        ),
        "control_correct_ce": _metric_path(
            step4000,
            (("control", "correct_ce"), ("control_correct_ce",)),
            name="step4000 control correct CE",
        ),
        "treatment_correct_ce": _metric_path(
            step4000,
            (("treatment", "correct_ce"), ("treatment_correct_ce",)),
            name="step4000 treatment correct CE",
        ),
        "treatment_gap": _metric_path(
            step4000,
            (("treatment", "gap", "mean"), ("treatment_gap",)),
            name="step4000 treatment gap",
        ),
        "step3000_treatment_gap": _metric_path(
            step3000,
            (("treatment", "gap", "mean"), ("treatment_gap",)),
            name="step3000 treatment gap",
        ),
    }
    sources: dict[str, Any] = {}
    for source in SOURCES:
        raw = source_value.get(source)
        if not isinstance(raw, Mapping):
            raise PromotionValidationError(f"Outcome validator lacks source {source}")
        endpoint = raw.get("step4000") or raw.get(str(PRIMARY_STEP)) or raw
        if not isinstance(endpoint, Mapping):
            raise PromotionValidationError(f"Outcome source {source} lacks step4000")
        sources[source] = {
            "control_correct_ce": _metric_path(
                endpoint,
                (("control", "correct_ce"), ("control_correct_ce",)),
                name=f"{source} control correct CE",
            ),
            "treatment_correct_ce": _metric_path(
                endpoint,
                (("treatment", "correct_ce"), ("treatment_correct_ce",)),
                name=f"{source} treatment correct CE",
            ),
        }
    return {"macro": macro, "sources": sources}


def _enforce_outcome_policy(policy: Mapping[str, Any]) -> None:
    macro = policy.get("macro")
    sources = policy.get("sources")
    if not isinstance(macro, Mapping) or not isinstance(sources, Mapping):
        raise PromotionValidationError("Outcome policy metrics are malformed")
    did_low = _finite(macro.get("did_ci_low"), name="outcome DID lower CI")
    gap_low = _finite(macro.get("treatment_gap_ci_low"), name="outcome treatment gap lower CI")
    control_ce = _finite(macro.get("control_correct_ce"), name="control macro correct CE")
    treatment_ce = _finite(macro.get("treatment_correct_ce"), name="treatment macro correct CE")
    gap4000 = _finite(macro.get("treatment_gap"), name="treatment step4000 gap")
    gap3000 = _finite(macro.get("step3000_treatment_gap"), name="treatment step3000 gap")
    if did_low <= 0:
        raise PromotionValidationError("Treatment DID lower confidence bound is not positive")
    if gap_low <= 0:
        raise PromotionValidationError(
            "Treatment absolute-gap lower confidence bound is not positive"
        )
    if treatment_ce > (1 + CORRECT_CE_MAX_RELATIVE_INCREASE) * control_ce:
        raise PromotionValidationError("Treatment macro correct CE exceeds the +2% bound")
    if gap4000 < LATE_GAP_RETENTION_FRACTION * gap3000:
        raise PromotionValidationError("Treatment step4000 gap fails late durability")
    _exact_fields(sources, frozenset(SOURCES), name="outcome source policy metrics")
    for source in SOURCES:
        value = sources[source]
        if not isinstance(value, Mapping):
            raise PromotionValidationError(f"Outcome source {source} metrics are malformed")
        source_control = _finite(
            value.get("control_correct_ce"), name=f"{source} control correct CE"
        )
        source_treatment = _finite(
            value.get("treatment_correct_ce"), name=f"{source} treatment correct CE"
        )
        if source_treatment > (1 + CORRECT_CE_MAX_RELATIVE_INCREASE) * source_control:
            raise PromotionValidationError(f"Treatment {source} correct CE exceeds the +2% bound")


_INIT_FIELDS = frozenset(
    {
        "format",
        "version",
        "status",
        "created_at",
        "producer",
        "native_helper",
        "snapshot_helper",
        "arm",
        "reference_checkpoint",
        "perception_step0",
        "protocol",
        "comparisons",
        "summary",
        "content_sha256",
    }
)
_INIT_PROTOCOL_FIELDS = frozenset({"name", "hash_algorithm", "tensor_encoding"})
_INIT_COMPARISON_FIELDS = frozenset(
    {"name", "kind", "dtype", "shape", "numel", "reference_sha256", "step0_sha256"}
)
_INIT_SUMMARY_FIELDS = frozenset(
    {
        "complete",
        "expected_tensor_count",
        "compared_tensor_count",
        "mismatch_count",
        "comparison_inventory_sha256",
    }
)


def _validate_perception_native_helper(value: Any, *, name: str) -> None:
    bridge._validate_implementation_reference(
        value,
        name=f"{name} native helper",
        expected_basename="vision_alignment_matched_wrong.py",
        canonical_path=_SOURCE_ROOT / "scripts" / "eval" / "vision_alignment_matched_wrong.py",
    )


def _validate_perception_snapshot_helper(value: Any, *, name: str) -> None:
    bridge._validate_implementation_reference(
        value,
        name=f"{name} checkpoint snapshot helper",
        expected_basename="vision_alignment_perception_matched_wrong.py",
        canonical_path=_PERCEPTION_MATCHED_EVALUATOR_PATH,
    )


def validate_initialization_parity_receipt(
    receipt: Mapping[str, Any], *, candidate: Mapping[str, Any], expected_arm: str
) -> dict[str, Any]:
    """Validate exact model-parameter/buffer equality from bridge parent to arm step0."""
    _validate_receipt_header(
        receipt,
        expected_format=INITIALIZATION_PARITY_RECEIPT_FORMAT,
        expected_fields=_INIT_FIELDS,
        name=f"{expected_arm} initialization-parity receipt",
    )
    _validate_arm(receipt["arm"], name="initialization-parity arm")
    if receipt["arm"] != expected_arm:
        raise PromotionValidationError("Initialization-parity arm differs from its bundle role")
    _validate_implementation_reference(
        receipt["producer"],
        name="initialization-parity producer",
        basename="vision_alignment_perception_state_text.py",
        canonical_path=_PERCEPTION_STATE_TEXT_EVALUATOR_PATH,
    )
    _validate_perception_native_helper(
        receipt["native_helper"], name=f"{expected_arm} initialization parity"
    )
    _validate_perception_snapshot_helper(
        receipt["snapshot_helper"], name=f"{expected_arm} initialization parity"
    )
    reference = bridge._validate_checkpoint_identity(
        receipt["reference_checkpoint"], name=f"{expected_arm} bridge reference"
    )
    step0 = bridge._validate_checkpoint_identity(
        receipt["perception_step0"], name=f"{expected_arm} perception step0"
    )
    reference_root = Path(str(reference["root"])).resolve()
    step0_root = Path(str(step0["root"])).resolve()
    if reference_root.name != "step500" or step0_root.name != "step0":
        raise PromotionValidationError(
            "Initialization parity requires bridge step500 and arm step0"
        )
    if step0_root.parent != Path(str(candidate["checkpoint"])).resolve().parent:
        raise PromotionValidationError(
            "Initialization-parity step0 is not from the candidate lineage"
        )
    if step0["config_sha256"] != candidate["checkpoint_config_sha256"]:
        raise PromotionValidationError("Initialization-parity step0 config differs from candidate")
    _validate_live_checkpoint_identity_stable(reference, name=f"{expected_arm} bridge reference")
    _validate_live_checkpoint_identity_stable(step0, name=f"{expected_arm} perception step0")
    protocol = _exact_fields(
        receipt["protocol"], _INIT_PROTOCOL_FIELDS, name="initialization-parity protocol"
    )
    if protocol != {
        "name": "logical-all-model-tensor-sha256-v1",
        "hash_algorithm": "sha256",
        "tensor_encoding": "dtype-shape-contiguous-little-endian-v1",
    }:
        raise PromotionValidationError("Initialization-parity protocol is incompatible")
    comparisons = receipt["comparisons"]
    if not isinstance(comparisons, list) or not comparisons:
        raise PromotionValidationError("Initialization-parity comparisons are empty")
    names: set[str] = set()
    normalized: list[Mapping[str, Any]] = []
    for index, raw in enumerate(comparisons):
        comparison = _exact_fields(
            raw, _INIT_COMPARISON_FIELDS, name=f"initialization comparison {index}"
        )
        name = comparison["name"]
        if not isinstance(name, str) or not name or name in names:
            raise PromotionValidationError("Initialization comparison names must be unique")
        names.add(name)
        if comparison["kind"] not in ("parameter", "persistent_buffer"):
            raise PromotionValidationError(f"Initialization comparison {name!r} has unknown kind")
        if not isinstance(comparison["dtype"], str) or not comparison["dtype"]:
            raise PromotionValidationError(f"Initialization comparison {name!r} has no dtype")
        shape = comparison["shape"]
        if not isinstance(shape, list) or any(
            _int(size, name=f"{name} shape") < 0 for size in shape
        ):
            raise PromotionValidationError(f"Initialization comparison {name!r} shape is invalid")
        _int(comparison["numel"], name=f"{name} numel", minimum=1)
        left = _sha256(comparison["reference_sha256"], name=f"{name} reference SHA-256")
        right = _sha256(comparison["step0_sha256"], name=f"{name} step0 SHA-256")
        if left != right:
            raise PromotionValidationError(f"Initialization comparison {name!r} differs")
        normalized.append(comparison)
    summary = _exact_fields(
        receipt["summary"], _INIT_SUMMARY_FIELDS, name="initialization-parity summary"
    )
    expected_count = _int(summary["expected_tensor_count"], name="expected tensor count", minimum=1)
    compared_count = _int(summary["compared_tensor_count"], name="compared tensor count", minimum=1)
    parameter_count = sum(item["kind"] == "parameter" for item in normalized)
    buffer_count = sum(item["kind"] == "persistent_buffer" for item in normalized)
    if (
        summary["complete"] is not True
        or _int(summary["mismatch_count"], name="initialization mismatch count") != 0
        or expected_count != EXPECTED_INITIALIZATION_PARAMETER_COUNT
        or parameter_count != EXPECTED_INITIALIZATION_PARAMETER_COUNT
        or buffer_count != EXPECTED_INITIALIZATION_BUFFER_COUNT
        or expected_count != compared_count
        or compared_count != len(comparisons)
        or summary["comparison_inventory_sha256"]
        != canonical_sha256(
            sorted(normalized, key=lambda item: (str(item["kind"]), str(item["name"])))
        )
    ):
        raise PromotionValidationError("Initialization-parity receipt is incomplete")
    return {
        "arm": expected_arm,
        "reference_checkpoint": dict(reference),
        "step0_checkpoint": dict(step0),
        "comparison_inventory_sha256": summary["comparison_inventory_sha256"],
        "tensor_count": compared_count,
    }


def _perception_state_text_bridge_copy(
    receipt: Mapping[str, Any], *, expected_format: str, bridge_format: str, name: str
) -> dict[str, Any]:
    """Validate the perception evaluator pin, then adapt only its header for generic checks.

    The frozen/text schemas and numerical policies are intentionally identical to the bridge
    receipts, but the executable must be a new file so the historical bridge evaluator SHA stays
    immutable.  We validate that new executable first and then delegate every remaining field to
    the already-tested generic validator using a transient in-memory header.
    """
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("format") != expected_format
        or type(receipt.get("version")) is not int
        or receipt.get("version") != RECEIPT_VERSION
        or receipt.get("status") != "passed"
    ):
        raise PromotionValidationError(f"{name} format is incompatible")
    receipt_candidate = receipt.get("candidate")
    receipt_reference = receipt.get("reference_checkpoint")
    if not isinstance(receipt_candidate, Mapping) or not isinstance(receipt_reference, Mapping):
        raise PromotionValidationError(f"{name} checkpoint fields are malformed")
    _int(receipt_candidate.get("global_step"), name=f"{name} candidate global step")
    _int(receipt_reference.get("global_step"), name=f"{name} reference global step")
    if expected_format == PERCEPTION_FROZEN_STATE_RECEIPT_FORMAT:
        summary = receipt.get("summary")
        if not isinstance(summary, Mapping):
            raise PromotionValidationError(f"{name} summary is malformed")
        _int(summary.get("mismatch_count"), name=f"{name} mismatch count")
    elif expected_format == PERCEPTION_TEXT_RETENTION_RECEIPT_FORMAT:
        dataset = receipt.get("dataset")
        if not isinstance(dataset, Mapping):
            raise PromotionValidationError(f"{name} dataset is malformed")
        _int(dataset.get("image_token_count"), name=f"{name} image-token count")
        _int(dataset.get("image_tensor_count"), name=f"{name} image-tensor count")
    evaluator = receipt.get("evaluator")
    bridge._validate_implementation_reference(
        evaluator,
        name=f"{name} evaluator",
        expected_basename="vision_alignment_perception_state_text.py",
        canonical_path=_PERCEPTION_STATE_TEXT_EVALUATOR_PATH,
    )
    _validate_perception_native_helper(receipt.get("native_helper"), name=name)
    _validate_perception_snapshot_helper(receipt.get("snapshot_helper"), name=name)
    bridge_evaluator = _SOURCE_ROOT / "scripts" / "eval" / "vision_alignment_state_text.py"
    adapted = dict(receipt)
    adapted.pop("native_helper")
    adapted.pop("snapshot_helper")
    adapted["format"] = bridge_format
    adapted["evaluator"] = {
        "path": str(bridge_evaluator),
        "sha256": sha256_file(bridge_evaluator),
    }
    return adapted


def validate_perception_frozen_state_receipt(
    receipt: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    expected_frozen_tensor_count: int | None = None,
) -> dict[str, Any]:
    """Validate exact perception-frozen tensors with the new evaluator identity."""
    adapted = _perception_state_text_bridge_copy(
        receipt,
        expected_format=PERCEPTION_FROZEN_STATE_RECEIPT_FORMAT,
        bridge_format=bridge.FROZEN_STATE_RECEIPT_FORMAT,
        name="perception frozen-state receipt",
    )
    return bridge.validate_frozen_state_receipt(
        adapted,
        candidate=candidate,
        expected_frozen_tensor_count=expected_frozen_tensor_count,
    )


def validate_perception_text_retention_receipt(
    receipt: Mapping[str, Any], *, candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate exact image-free text retention with the new evaluator identity."""
    adapted = _perception_state_text_bridge_copy(
        receipt,
        expected_format=PERCEPTION_TEXT_RETENTION_RECEIPT_FORMAT,
        bridge_format=bridge.TEXT_RETENTION_RECEIPT_FORMAT,
        name="perception text-retention receipt",
    )
    bridge._validate_receipt_candidate(
        adapted["candidate"], expected=candidate, name="perception text-retention candidate"
    )
    reference = _exact_fields(
        adapted["reference_checkpoint"],
        bridge._FROZEN_REFERENCE_FIELDS,
        name="perception text-retention reference checkpoint",
    )
    reference_path = _resolved_path(
        reference["checkpoint"],
        name="perception text-retention reference checkpoint",
        must_exist=False,
    )
    if (
        reference_path.name != "step0"
        or _int(reference["global_step"], name="text reference global step") != 0
        or reference_path.parent != Path(str(candidate["checkpoint"])).resolve().parent
        or reference["checkpoint_config_sha256"] != candidate["checkpoint_config_sha256"]
    ):
        raise PromotionValidationError("Perception text-retention reference is incompatible")
    _sha256(reference["checkpoint_config_sha256"], name="text reference config SHA-256")
    _sha256(reference["checkpoint_identity_sha256"], name="text reference identity SHA-256")

    dataset = _exact_fields(
        adapted["dataset"], bridge._TEXT_DATASET_FIELDS, name="perception text dataset"
    )
    if (
        dataset["sha256"] != EXPECTED_TEXT_SENTINEL_RAW_SHA256
        or dataset["fingerprint"] != EXPECTED_TEXT_SENTINEL_FINGERPRINT
    ):
        raise PromotionValidationError("Text retention does not use the locked sentinel")
    dataset_path, sentinel_raw = _load_raw_reference(
        {"path": dataset["path"], "sha256": dataset["sha256"]},
        name="perception text sentinel",
    )
    sentinel = _strict_json_bytes(sentinel_raw, name="perception text sentinel")
    if not isinstance(sentinel, Mapping):
        raise PromotionValidationError("Perception text sentinel must be an object")
    sentinel_summary = bridge.validate_text_sentinel(sentinel)
    examples = _int(dataset["examples"], name="text dataset examples", minimum=128)
    supervised_tokens = _int(
        dataset["supervised_tokens"], name="text supervised tokens", minimum=32_768
    )
    for field in ("input_ids_sha256", "labels_sha256"):
        _sha256(dataset[field], name=f"text dataset {field}")
    if (
        _int(dataset["image_token_count"], name="text image-token count") != 0
        or _int(dataset["image_tensor_count"], name="text image-tensor count") != 0
        or examples != sentinel_summary["examples"]
        or supervised_tokens != sentinel_summary["supervised_tokens"]
        or dataset["input_ids_sha256"] != sentinel_summary["input_ids_sha256"]
        or dataset["labels_sha256"] != sentinel_summary["labels_sha256"]
        or dataset["fingerprint"] != sentinel_summary["fingerprint"]
    ):
        raise PromotionValidationError("Perception text dataset claims differ from the sentinel")

    protocol = _exact_fields(
        adapted["protocol"], bridge._TEXT_PROTOCOL_FIELDS, name="perception text protocol"
    )
    atol = _finite(protocol["atol"], name="perception text atol")
    rtol = _finite(protocol["rtol"], name="perception text rtol")
    if (
        protocol["name"] != "per-token-nll-and-argmax-v1"
        or not 0 <= atol <= 1e-6
        or not 0 <= rtol <= 1e-6
        or protocol["same_topology"] is not True
        or protocol["same_backend"] is not True
        or protocol["image_free"] is not True
    ):
        raise PromotionValidationError("Perception text-retention protocol is incompatible")
    metrics = _exact_fields(
        adapted["metrics"], bridge._TEXT_METRIC_FIELDS, name="perception text metrics"
    )
    for field in (
        "reference_mean_ce",
        "candidate_mean_ce",
        "max_abs_token_ce_delta",
        "max_rel_token_ce_delta",
    ):
        _finite(metrics[field], name=f"perception text {field}")
    matches = _int(metrics["argmax_matches"], name="text argmax matches")
    total = _int(metrics["argmax_total"], name="text argmax total", minimum=1)
    if (
        metrics["all_finite"] is not True
        or metrics["max_abs_token_ce_delta"] > atol
        or metrics["max_rel_token_ce_delta"] > rtol
        or matches != total
        or total != supervised_tokens
    ):
        raise PromotionValidationError("Perception text metrics fail the locked tolerance")
    return {
        "examples": examples,
        "supervised_tokens": supervised_tokens,
        "max_abs_token_ce_delta": float(metrics["max_abs_token_ce_delta"]),
        "argmax_match_rate": 1.0,
        "reference_checkpoint": str(reference_path),
        "reference_checkpoint_config_sha256": reference["checkpoint_config_sha256"],
        "reference_checkpoint_identity_sha256": reference["checkpoint_identity_sha256"],
        "dataset_path": str(dataset_path),
    }


_RUN_FIELDS = frozenset(
    {
        "format",
        "version",
        "status",
        "created_at",
        "producer",
        "checkpoint_identity_helper",
        "arm",
        "candidate",
        "launch",
        "run",
        "rank_state_inventory",
        "permanent_checkpoints",
        "optimizer_guard",
        "evidence",
        "content_sha256",
    }
)
_RUN_LAUNCH_FIELDS = frozenset(
    {"workspace", "experiment_id", "successful_jobs", "prestart_failures"}
)
_RUN_SUCCESS_JOB_FIELDS = frozenset(
    {"job_id", "replica_rank", "exit_code", "started_training", "completed_training", "log"}
)
_RUN_PRESTART_FIELDS = frozenset(
    {"job_id", "replica_rank", "exit_code", "started_training", "reason", "evidence"}
)
_RUN_SUMMARY_FIELDS = frozenset(
    {
        "run_id",
        "global_steps",
        "exit_code",
        "rank_state_count",
        "permanent_checkpoint_steps",
        "metric_step_count",
        "numeric_metric_count",
        "nonfinite_metric_count",
        "unexpected_anomaly_count",
        "total_data_errors",
        "successful_terminal_marker",
    }
)
_RUN_RANK_FIELDS = frozenset(
    {"rank", "path", "sha256", "global_step", "batches_processed", "total_data_errors", "run_id"}
)
_RUN_CHECKPOINT_FIELDS = frozenset({"step", "identity"})
_RUN_GUARD_FIELDS = frozenset(
    {
        "rolling_interval_length",
        "sigma_factor",
        "observed_steps",
        "count",
        "rate",
        "minimum_spacing",
        "clean_final_steps",
        "every_next_step_finite",
    }
)
_RUN_EVIDENCE_FIELDS = frozenset({"beaker_experiment", "wandb_output", "wandb_summary"})


def validate_perception_run_health_receipt(
    receipt: Mapping[str, Any], *, candidate: Mapping[str, Any], expected_arm: str
) -> dict[str, Any]:
    """Validate one completed perception run and the exact seven-skip deviation, if any."""
    _validate_receipt_header(
        receipt,
        expected_format=RUN_HEALTH_RECEIPT_FORMAT,
        expected_fields=_RUN_FIELDS,
        name=f"{expected_arm} run-health receipt",
    )
    _validate_implementation_reference(
        receipt["producer"],
        name=f"{expected_arm} run-health producer",
        basename="vision_alignment_perception_run_health.py",
        canonical_path=_RUN_HEALTH_PRODUCER_PATH,
    )
    _validate_perception_snapshot_helper(
        receipt["checkpoint_identity_helper"], name=f"{expected_arm} run health"
    )
    run_health_module = _load_run_health_module()
    if _validate_arm(receipt["arm"], name="run-health arm") != expected_arm:
        raise PromotionValidationError("Run-health arm differs from its bundle role")
    _validate_receipt_candidate(
        receipt["candidate"], expected=candidate, name=f"{expected_arm} run-health candidate"
    )
    launch = _exact_fields(receipt["launch"], _RUN_LAUNCH_FIELDS, name="run-health launch")
    if launch["workspace"] != "ai2/molmofication":
        raise PromotionValidationError("Run-health receipt is outside ai2/molmofication")
    if launch["experiment_id"] != EXPECTED_EXPERIMENT_IDS[expected_arm]:
        raise PromotionValidationError("Run-health experiment ID differs from the locked run")
    successful = launch["successful_jobs"]
    if not isinstance(successful, list) or len(successful) != 2:
        raise PromotionValidationError("Run-health requires exactly two successful replicas")
    job_ids: set[str] = set()
    for index, raw_job in enumerate(successful):
        job = _exact_fields(raw_job, _RUN_SUCCESS_JOB_FIELDS, name=f"successful job {index}")
        if (
            _int(job["replica_rank"], name=f"successful job {index} replica rank") != index
            or _int(job["exit_code"], name=f"successful job {index} exit code") != 0
            or job["started_training"] is not True
            or job["completed_training"] is not True
            or not isinstance(job["job_id"], str)
            or not job["job_id"]
            or job["job_id"] in job_ids
        ):
            raise PromotionValidationError("Run-health successful job is incompatible")
        job_ids.add(job["job_id"])
        log_path, log_raw = _load_raw_reference(job["log"], name=f"successful replica{index} log")
        started, completed = run_health_module._audit_job_log_text(
            log_raw.decode("utf-8", errors="replace"), name=str(log_path)
        )
        if started is not True or completed is not True:
            raise PromotionValidationError("Successful job log does not prove completion")
    if {
        int(job["replica_rank"]): str(job["job_id"]) for job in successful
    } != run_health_module.EXPECTED_SUCCESSFUL_JOBS[expected_arm]:
        raise PromotionValidationError("Successful jobs differ from the locked arm run")
    prestart = launch["prestart_failures"]
    expected_prestart_count = 1 if expected_arm == CONTROL_ARM else 0
    if not isinstance(prestart, list) or len(prestart) != expected_prestart_count:
        raise PromotionValidationError("Unexpected pre-start failure inventory")
    for index, raw_failure in enumerate(prestart):
        failure = _exact_fields(
            raw_failure, _RUN_PRESTART_FIELDS, name=f"pre-start failure {index}"
        )
        if (
            failure["started_training"] is not False
            or _int(failure["replica_rank"], name=f"pre-start failure {index} rank") != 1
            or _int(failure["exit_code"], name=f"pre-start failure {index} exit code") == 0
            or not isinstance(failure["job_id"], str)
            or not failure["job_id"]
            or failure["job_id"] in job_ids
            or not isinstance(failure["reason"], str)
            or not failure["reason"]
        ):
            raise PromotionValidationError("Pre-start failure reached user training")
        job_ids.add(failure["job_id"])
        _failure_path, failure_raw = _load_raw_reference(
            failure["evidence"], name=f"pre-start failure {index} evidence"
        )
        failure_text = failure_raw.decode("utf-8", errors="replace")
        if "[step=" in failure_text or "Finalizing successful W&B run" in failure_text:
            raise PromotionValidationError("Pre-start failure evidence reached user training")
    if expected_arm == CONTROL_ARM:
        locked_failure = run_health_module.CONTROL_PRESTART_FAILURE
        observed_failure = prestart[0]
        if (
            observed_failure["job_id"] != locked_failure["job_id"]
            or observed_failure["replica_rank"] != locked_failure["replica_rank"]
            or observed_failure["exit_code"] != locked_failure["canceled_code"]
            or observed_failure["reason"] != locked_failure["reason"]
        ):
            raise PromotionValidationError("Control pre-start failure differs from the locked job")
    run = _exact_fields(receipt["run"], _RUN_SUMMARY_FIELDS, name="run-health run")
    for field in (
        "global_steps",
        "exit_code",
        "rank_state_count",
        "metric_step_count",
        "numeric_metric_count",
        "nonfinite_metric_count",
        "unexpected_anomaly_count",
        "total_data_errors",
    ):
        _int(run[field], name=f"run-health {field}")
    permanent_steps = run["permanent_checkpoint_steps"]
    if not isinstance(permanent_steps, list) or any(
        type(step) is not int for step in permanent_steps
    ):
        raise PromotionValidationError("Run-health permanent checkpoint steps are malformed")
    if (
        not isinstance(run["run_id"], str)
        or not run["run_id"]
        or run["global_steps"] != PRIMARY_STEP
        or run["exit_code"] != 0
        or run["rank_state_count"] != 16
        or run["permanent_checkpoint_steps"] != [0, 1000, 2000, 3000, 4000]
        or run["metric_step_count"] != PRIMARY_STEP
        or run["numeric_metric_count"] <= 0
        or run["nonfinite_metric_count"] != 0
        or run["unexpected_anomaly_count"] != 0
        or run["total_data_errors"] != 0
        or run["successful_terminal_marker"] is not True
    ):
        raise PromotionValidationError("Perception run did not finish cleanly at step4000")
    rank_state_inventory = _validate_rank_states(
        receipt["rank_state_inventory"],
        run=run,
        candidate=candidate,
        expected_arm=expected_arm,
    )
    permanent_checkpoints = _validate_permanent_checkpoints(
        receipt["permanent_checkpoints"], candidate=candidate
    )
    guard_summary = _validate_optimizer_guard(receipt["optimizer_guard"], expected_arm=expected_arm)
    evidence = _exact_fields(receipt["evidence"], _RUN_EVIDENCE_FIELDS, name="run evidence")
    _beaker_path, beaker_raw = _load_raw_reference(
        evidence["beaker_experiment"], name="Beaker experiment evidence"
    )
    beaker_payload = _strict_json_bytes(beaker_raw, name="Beaker experiment evidence")
    run_health_module._verify_locked_experiment_snapshot(beaker_payload, arm=expected_arm)
    log_path, log_raw = _load_raw_reference(evidence["wandb_output"], name="W&B output")
    log_audit = run_health_module._audit_log_text(log_raw.decode("utf-8", errors="replace"))
    observed_log_steps = tuple(int(step) for step in log_audit["guarded_skip_steps"])
    if observed_log_steps != tuple(guard_summary["steps"]):
        raise PromotionValidationError("Run log guarded skips differ from the receipt")
    for field in (
        "metric_step_count",
        "numeric_metric_count",
        "nonfinite_metric_count",
        "unexpected_anomaly_count",
        "successful_terminal_marker",
    ):
        if run[field] != log_audit[field]:
            raise PromotionValidationError(f"Run summary {field} differs from the strict log audit")
    if log_audit["every_next_step_finite"] is not guard_summary["every_next_step_finite"]:
        raise PromotionValidationError("Run guarded-skip recovery evidence differs")
    summary_path, summary_raw = _load_raw_reference(evidence["wandb_summary"], name="W&B summary")
    candidate_root = Path(str(candidate["checkpoint"])).resolve().parent
    expected_wandb_root = candidate_root / "wandb" / "wandb"
    if (
        log_path.parent != summary_path.parent
        or log_path.parent.name != "files"
        or not log_path.parent.parent.name.endswith(f"-{run['run_id']}")
        or not log_path.is_relative_to(expected_wandb_root)
        or not summary_path.is_relative_to(expected_wandb_root)
    ):
        raise PromotionValidationError("W&B evidence is outside the candidate run directory")
    run_health_module._audit_summary_value(
        _strict_json_bytes(summary_raw, name="W&B summary"), expected_run_id=run["run_id"]
    )
    return {
        "arm": expected_arm,
        "run_id": run["run_id"],
        "guarded_skip_steps": list(guard_summary["steps"]),
        "guarded_skip_rate": guard_summary["rate"],
        "rank_state_inventory": rank_state_inventory,
        "permanent_checkpoints": permanent_checkpoints,
        "evidence_sha256": receipt["content_sha256"],
    }


def _validate_rank_states(
    value: Any,
    *,
    run: Mapping[str, Any],
    candidate: Mapping[str, Any],
    expected_arm: str,
) -> list[dict[str, Any]]:
    states = value
    if not isinstance(states, list) or len(states) != 16:
        raise PromotionValidationError("Run-health rank-state inventory is incomplete")
    candidate_train = Path(str(candidate["checkpoint"])).resolve() / "train"
    expected_name = str(EXPECTED_PROFILE_CONTRACTS[expected_arm]["name"])
    normalized: list[dict[str, Any]] = []
    for expected_rank, raw in enumerate(states):
        state = _exact_fields(raw, _RUN_RANK_FIELDS, name=f"run rank{expected_rank} state")
        for field in ("rank", "global_step", "batches_processed", "total_data_errors"):
            _int(state[field], name=f"run rank{expected_rank} {field}")
        path = _resolved_path(state["path"], name=f"run rank{expected_rank} state")
        expected_sha = _sha256(state["sha256"], name=f"run rank{expected_rank} SHA-256")
        if (
            state["rank"] != expected_rank
            or path != candidate_train / f"rank{expected_rank}.pt"
            or path.is_symlink()
        ):
            raise PromotionValidationError("Run-health rank-state ordering differs")
        if not path.is_file():
            raise PromotionValidationError(f"Run rank{expected_rank} state is missing")
        loaded = load_trainer_state(path, expected_sha256=expected_sha)
        loader = loaded.get("data_loader") if isinstance(loaded, Mapping) else None
        callbacks = loaded.get("callbacks") if isinstance(loaded, Mapping) else None
        wandb = callbacks.get("wandb") if isinstance(callbacks, Mapping) else None
        if (
            not isinstance(loader, Mapping)
            or not isinstance(wandb, Mapping)
            or type(loaded.get("global_step")) is not int
            or loaded.get("global_step") != state["global_step"]
            or type(loader.get("batches_processed")) is not int
            or loader.get("batches_processed") != state["batches_processed"]
            or type(loader.get("total_data_errors")) is not int
            or loader.get("total_data_errors") != state["total_data_errors"]
            or wandb.get("run_id") != state["run_id"]
            or type(wandb.get("step")) is not int
            or wandb.get("step") != PRIMARY_STEP
            or wandb.get("name") != expected_name
            or wandb.get("project") != "vision-alignment"
            or state["global_step"] != PRIMARY_STEP
            or state["batches_processed"] != PRIMARY_STEP
            or state["total_data_errors"] != 0
            or (expected_rank == 0 and state["run_id"] != run["run_id"])
            or (expected_rank != 0 and state["run_id"] is not None)
        ):
            raise PromotionValidationError(f"Run rank{expected_rank} trainer state differs")
        normalized.append({"rank": expected_rank, "path": str(path), "sha256": expected_sha})
    return normalized


def _validate_permanent_checkpoints(
    value: Any, *, candidate: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    checkpoints = value
    expected_steps = [0, 1000, 2000, 3000, 4000]
    if not isinstance(checkpoints, list) or len(checkpoints) != len(expected_steps):
        raise PromotionValidationError("Permanent checkpoint inventory differs")
    candidate_root = Path(str(candidate["checkpoint"])).resolve()
    normalized: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(checkpoints):
        checkpoint = _exact_fields(
            raw, _RUN_CHECKPOINT_FIELDS, name=f"permanent checkpoint {index}"
        )
        step = _int(checkpoint["step"], name=f"permanent checkpoint {index} step")
        if step != expected_steps[index]:
            raise PromotionValidationError("Permanent checkpoint sequence differs")
        identity = bridge._validate_checkpoint_identity(
            checkpoint["identity"], name=f"permanent step{step}"
        )
        root = _resolved_path(identity["root"], name=f"permanent step{step}")
        if root != candidate_root.parent / f"step{step}":
            raise PromotionValidationError(f"Permanent step{step} is outside candidate lineage")
        _validate_live_checkpoint_identity_stable(identity, name=f"permanent step{step}")
        marker = load_json_pinned(
            root / ".metadata.json",
            identity["checkpoint_marker_sha256"],
            name=f"permanent step{step} marker",
        )
        if not isinstance(marker, Mapping) or marker.get("ephemeral") is not False:
            raise PromotionValidationError(f"Step{step} is not permanent")
        normalized[f"step{step}"] = dict(identity)
    final = normalized[f"step{PRIMARY_STEP}"]
    if (
        final["root"] != str(candidate_root)
        or final["config_sha256"] != candidate["checkpoint_config_sha256"]
        or final["checkpoint_marker_sha256"] != candidate["checkpoint_marker_sha256"]
        or final["dcp_metadata_sha256"] != candidate["dcp_metadata_sha256"]
        or final["state_file_inventory_sha256"] != candidate["state_file_inventory_sha256"]
        or final["identity_sha256"] != candidate["checkpoint_identity_sha256"]
    ):
        raise PromotionValidationError("Run-health final checkpoint differs from candidate")
    return normalized


def _validate_optimizer_guard(value: Any, *, expected_arm: str) -> dict[str, Any]:
    guard = _exact_fields(value, _RUN_GUARD_FIELDS, name="optimizer guard")
    steps = guard["observed_steps"]
    if not isinstance(steps, list) or any(
        isinstance(step, bool) or not isinstance(step, int) for step in steps
    ):
        raise PromotionValidationError("Optimizer guarded-skip steps are malformed")
    expected_steps = [] if expected_arm == CONTROL_ARM else list(EXPECTED_TREATMENT_SKIP_STEPS)
    spacings = [right - left for left, right in pairwise(steps)]
    minimum_spacing = min(spacings) if spacings else PRIMARY_STEP
    clean_final = PRIMARY_STEP - steps[-1] if steps else PRIMARY_STEP
    if (
        _int(guard["rolling_interval_length"], name="guard rolling interval")
        != ROLLING_INTERVAL_LENGTH
        or _int(guard["sigma_factor"], name="guard sigma factor") != SIGMA_FACTOR
        or steps != expected_steps
        or _int(guard["count"], name="guard count") != len(steps)
        or not math.isclose(
            _finite(guard["rate"], name="optimizer skip rate"),
            len(steps) / PRIMARY_STEP,
            rel_tol=0,
            abs_tol=1e-15,
        )
        or _int(guard["minimum_spacing"], name="guard minimum spacing") != minimum_spacing
        or _int(guard["clean_final_steps"], name="guard clean final steps") != clean_final
        or guard["every_next_step_finite"] is not True
        or minimum_spacing < ROLLING_INTERVAL_LENGTH
        or clean_final < ROLLING_INTERVAL_LENGTH
    ):
        raise PromotionValidationError("Optimizer guarded skips differ from the exact evidence")
    return {
        "steps": tuple(steps),
        "rate": len(steps) / PRIMARY_STEP,
        "every_next_step_finite": True,
    }


_LOSS_FIELDS = frozenset(
    {
        "format",
        "version",
        "status",
        "created_at",
        "producer",
        "pair_contract",
        "candidate",
        "comparator",
        "protocol",
        "loader",
        "evidence",
        "sources",
        "summary",
        "content_sha256",
    }
)
_LOSS_PROTOCOL_FIELDS = frozenset(
    {
        "name",
        "start_step",
        "end_step",
        "exact_packing_cursor",
        "share_tolerance",
        "arm_cursor_equality",
    }
)
_LOSS_LOADER_FIELDS = frozenset(
    {
        "dp_world_size",
        "batches_replayed",
        "rank_state_count",
        "total_data_errors",
        "dataset_fingerprints_sha256",
        "replayed_final_state_sha256",
        "arms",
    }
)
_LOSS_ARM_FIELDS = frozenset(
    {
        "rank_states_global_step",
        "rank_states_batches_processed",
        "rank_state_inventory_sha256",
        "checkpoint_final_state_sha256",
        "replayed_final_state_sha256",
    }
)
_LOSS_EVIDENCE_FIELDS = frozenset({"recipe", "producer", "rank_state_inventory"})
_LOSS_INVENTORY_FIELDS = frozenset({"rank", "path", "sha256"})
_LOSS_SOURCE_FIELDS = frozenset(
    {
        "examples",
        "tokens",
        "positive_tokens",
        "loss_weight",
        "active_loss_weight",
        "target_loss_mass",
        "loss_mass_share",
        "active_loss_mass_share",
        "absolute_error",
        "active_absolute_error",
    }
)
_LOSS_SUMMARY_FIELDS = frozenset(
    {
        "total_loss_weight",
        "total_active_loss_weight",
        "share_sum",
        "active_share_sum",
        "within_tolerance",
        "arm_final_cursor_equal",
    }
)


def validate_loss_mass_pair_receipt(
    receipt: Mapping[str, Any], *, candidate: Mapping[str, Any], comparator: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate exhaustive shared-data replay and exact final cursor equality across arms."""
    _validate_receipt_header(
        receipt,
        expected_format=LOSS_MASS_PAIR_RECEIPT_FORMAT,
        expected_fields=_LOSS_FIELDS,
        name="loss-mass-pair receipt",
    )
    _validate_implementation_reference(
        receipt["producer"],
        name="loss-mass pair producer",
        basename="vision_alignment_perception_loss_mass.py",
        canonical_path=_LOSS_MASS_PRODUCER_PATH,
    )
    pair_ref = _exact_fields(
        receipt["pair_contract"], _ARTIFACT_REF_FIELDS, name="loss-mass pair-contract reference"
    )
    if pair_ref["sha256"] != EXPECTED_PAIR_CONTRACT_RAW_SHA256:
        raise PromotionValidationError("Loss-mass receipt uses a different pair contract")
    _validate_raw_reference(pair_ref, name="loss-mass pair contract")
    _validate_receipt_candidate(receipt["candidate"], expected=candidate, name="loss candidate")
    _validate_receipt_candidate(receipt["comparator"], expected=comparator, name="loss comparator")
    protocol = _exact_fields(receipt["protocol"], _LOSS_PROTOCOL_FIELDS, name="loss protocol")
    if (
        protocol["name"] != "exact-packed-loader-paired-cumulative-loss-mass-v1"
        or _int(protocol["start_step"], name="loss start step") != 0
        or _int(protocol["end_step"], name="loss end step") != PRIMARY_STEP
        or protocol["exact_packing_cursor"] is not True
        or not math.isclose(
            _finite(protocol["share_tolerance"], name="loss share tolerance"),
            LOSS_MASS_ABSOLUTE_TOLERANCE,
            rel_tol=0,
            abs_tol=0,
        )
        or protocol["arm_cursor_equality"] is not True
    ):
        raise PromotionValidationError("Loss-mass pair protocol is incompatible")
    loader = _exact_fields(receipt["loader"], _LOSS_LOADER_FIELDS, name="loss loader")
    for field in ("dp_world_size", "batches_replayed", "rank_state_count", "total_data_errors"):
        _int(loader[field], name=f"loss loader {field}")
    if (
        loader["dp_world_size"] != 16
        or loader["batches_replayed"] != PRIMARY_STEP
        or loader["rank_state_count"] != 16
        or loader["total_data_errors"] != 0
    ):
        raise PromotionValidationError("Loss-mass replay topology or completion differs")
    for field in ("dataset_fingerprints_sha256", "replayed_final_state_sha256"):
        _sha256(loader[field], name=f"loss loader {field}")
    arms = _exact_fields(loader["arms"], _ARM_MAP_FIELDS, name="loss loader arms")
    for arm in ARMS:
        arm_value = _exact_fields(arms[arm], _LOSS_ARM_FIELDS, name=f"loss loader {arm}")
        _int(arm_value["rank_states_global_step"], name=f"loss {arm} global step")
        _int(
            arm_value["rank_states_batches_processed"],
            name=f"loss {arm} batches processed",
        )
        if (
            arm_value["rank_states_global_step"] != PRIMARY_STEP
            or arm_value["rank_states_batches_processed"] != PRIMARY_STEP
        ):
            raise PromotionValidationError(f"Loss loader {arm} is not at step4000")
        for field in (
            "rank_state_inventory_sha256",
            "checkpoint_final_state_sha256",
            "replayed_final_state_sha256",
        ):
            _sha256(arm_value[field], name=f"loss loader {arm} {field}")
        if (
            arm_value["checkpoint_final_state_sha256"] != loader["replayed_final_state_sha256"]
            or arm_value["replayed_final_state_sha256"] != loader["replayed_final_state_sha256"]
        ):
            raise PromotionValidationError(f"Loss loader {arm} final cursor differs from replay")
    if (
        arms[CONTROL_ARM]["checkpoint_final_state_sha256"]
        != arms[TREATMENT_ARM]["checkpoint_final_state_sha256"]
    ):
        raise PromotionValidationError("Control and treatment final loader cursors differ")
    rank_state_inventory = _validate_loss_evidence(
        receipt["evidence"],
        arms=arms,
        candidates={CONTROL_ARM: comparator, TREATMENT_ARM: candidate},
        expected_final_cursor_sha256=loader["replayed_final_state_sha256"],
        expected_dataset_fingerprints_sha256=loader["dataset_fingerprints_sha256"],
    )
    source_summary = _validate_loss_sources(receipt["sources"], receipt["summary"])
    return {
        "batches_replayed": PRIMARY_STEP,
        "final_cursor_sha256": loader["replayed_final_state_sha256"],
        "rank_state_inventory": rank_state_inventory,
        **source_summary,
    }


def _validate_loss_evidence(
    value: Any,
    *,
    arms: Mapping[str, Any],
    candidates: Mapping[str, Mapping[str, Any]],
    expected_final_cursor_sha256: str,
    expected_dataset_fingerprints_sha256: str,
) -> dict[str, list[dict[str, Any]]]:
    evidence = _exact_fields(value, _LOSS_EVIDENCE_FIELDS, name="loss evidence")
    _validate_historical_loss_recipe_reference(
        evidence["recipe"],
        name="loss recipe",
    )
    _validate_implementation_reference(
        evidence["producer"],
        name="loss producer evidence",
        basename="vision_alignment_perception_loss_mass.py",
        canonical_path=_LOSS_MASS_PRODUCER_PATH,
    )
    inventories = _exact_fields(
        evidence["rank_state_inventory"], _ARM_MAP_FIELDS, name="loss rank inventories"
    )
    loss_module = _load_loss_mass_module()
    cursor_inventories: dict[str, Any] = {}
    validated_rank_inventories: dict[str, list[dict[str, Any]]] = {}
    observed_dataset_fingerprints: Any = None
    for arm in ARMS:
        inventory = inventories[arm]
        if not isinstance(inventory, list) or len(inventory) != 16:
            raise PromotionValidationError(f"Loss {arm} rank-state inventory is incomplete")
        normalized: list[Mapping[str, Any]] = []
        loaded_states: list[Mapping[str, Any]] = []
        expected_train = Path(str(candidates[arm]["checkpoint"])).resolve() / "train"
        for expected_rank, raw in enumerate(inventory):
            item = _exact_fields(
                raw, _LOSS_INVENTORY_FIELDS, name=f"loss {arm} rank{expected_rank}"
            )
            _int(item["rank"], name=f"loss {arm} rank index")
            path = _resolved_path(item["path"], name=f"loss {arm} rank{expected_rank}")
            digest = _sha256(item["sha256"], name=f"loss {arm} rank{expected_rank} SHA-256")
            if (
                item["rank"] != expected_rank
                or path != expected_train / f"rank{expected_rank}.pt"
                or path.is_symlink()
                or not path.is_file()
            ):
                raise PromotionValidationError(f"Loss {arm} rank-state identity differs")
            loaded = load_trainer_state(path, expected_sha256=digest)
            loader = loaded.get("data_loader")
            packing = loader.get("packing_state") if isinstance(loader, Mapping) else None
            if (
                type(loaded.get("global_step")) is not int
                or loaded.get("global_step") != PRIMARY_STEP
                or type(loaded.get("world_size")) is not int
                or loaded.get("world_size") != 16
                or not isinstance(loader, Mapping)
                or type(loader.get("batches_processed")) is not int
                or loader.get("batches_processed") != PRIMARY_STEP
                or type(loader.get("total_data_errors")) is not int
                or loader.get("total_data_errors") != 0
                or not isinstance(packing, Mapping)
                or type(packing.get("version")) is not int
                or packing.get("version") != 5
                or type(packing.get("dp_world_size")) is not int
                or packing.get("dp_world_size") != 16
                or type(packing.get("dp_rank")) is not int
                or packing.get("dp_rank") != expected_rank
                or type(packing.get("packs_emitted")) is not int
                or packing.get("packs_emitted") != 32_000
            ):
                raise PromotionValidationError(f"Loss {arm} rank{expected_rank} cursor differs")
            dataset_fingerprints = loss_module._jsonable(packing.get("dataset_fingerprints"))
            if not isinstance(dataset_fingerprints, (list, Mapping)) or not dataset_fingerprints:
                raise PromotionValidationError(
                    f"Loss {arm} rank{expected_rank} dataset fingerprints are missing"
                )
            if observed_dataset_fingerprints is None:
                observed_dataset_fingerprints = dataset_fingerprints
            elif dataset_fingerprints != observed_dataset_fingerprints:
                raise PromotionValidationError("Saved loader dataset fingerprints differ")
            normalized.append(item)
            loaded_states.append(loaded)
        if canonical_sha256(normalized) != arms[arm]["rank_state_inventory_sha256"]:
            raise PromotionValidationError(f"Loss {arm} rank-state inventory SHA-256 differs")
        validated_rank_inventories[arm] = [dict(item) for item in normalized]
        cursor_inventory = loss_module._loader_state_inventory(loaded_states)
        cursor_sha256 = canonical_sha256(cursor_inventory)
        if (
            cursor_sha256 != arms[arm]["checkpoint_final_state_sha256"]
            or cursor_sha256 != expected_final_cursor_sha256
        ):
            raise PromotionValidationError(f"Loss {arm} saved loader cursor differs")
        cursor_inventories[arm] = cursor_inventory
    if cursor_inventories[CONTROL_ARM] != cursor_inventories[TREATMENT_ARM]:
        raise PromotionValidationError("Control and treatment saved loader cursors differ")
    if canonical_sha256(observed_dataset_fingerprints) != expected_dataset_fingerprints_sha256:
        raise PromotionValidationError("Saved loader dataset fingerprint identity differs")
    return validated_rank_inventories


def _validate_loss_sources(value: Any, raw_summary: Any) -> dict[str, Any]:
    sources = _exact_fields(value, frozenset(SOURCES), name="loss sources")
    totals = {"loss": 0.0, "active": 0.0}
    normalized: dict[str, dict[str, float]] = {}
    for source in SOURCES:
        item = _exact_fields(sources[source], _LOSS_SOURCE_FIELDS, name=f"loss source {source}")
        for field in ("examples", "tokens", "positive_tokens"):
            _int(item[field], name=f"{source} {field}", minimum=1)
        loss_weight = _finite(item["loss_weight"], name=f"{source} loss weight")
        active_weight = _finite(item["active_loss_weight"], name=f"{source} active loss weight")
        if loss_weight <= 0 or active_weight <= 0:
            raise PromotionValidationError(f"Loss source {source} has no supervised mass")
        target = _finite(item["target_loss_mass"], name=f"{source} target")
        if not math.isclose(target, LOSS_MASS_TARGETS[source], rel_tol=0, abs_tol=1e-15):
            raise PromotionValidationError(f"Loss source {source} target differs")
        totals["loss"] += loss_weight
        totals["active"] += active_weight
        normalized[source] = {"loss_weight": loss_weight, "active_loss_weight": active_weight}
    shares: dict[str, float] = {}
    active_shares: dict[str, float] = {}
    for source in SOURCES:
        item = sources[source]
        share = normalized[source]["loss_weight"] / totals["loss"]
        active_share = normalized[source]["active_loss_weight"] / totals["active"]
        target = LOSS_MASS_TARGETS[source]
        for observed, expected, label in (
            (item["loss_mass_share"], share, "share"),
            (item["active_loss_mass_share"], active_share, "active share"),
            (item["absolute_error"], abs(share - target), "absolute error"),
            (item["active_absolute_error"], abs(active_share - target), "active absolute error"),
        ):
            if not math.isclose(
                _finite(observed, name=f"{source} {label}"), expected, rel_tol=0, abs_tol=1e-12
            ):
                raise PromotionValidationError(f"Loss source {source} {label} is inconsistent")
        if (
            abs(share - target) > LOSS_MASS_ABSOLUTE_TOLERANCE
            or abs(active_share - target) > LOSS_MASS_ABSOLUTE_TOLERANCE
        ):
            raise PromotionValidationError(f"Loss source {source} exceeds the 2% tolerance")
        shares[source] = share
        active_shares[source] = active_share
    summary = _exact_fields(raw_summary, _LOSS_SUMMARY_FIELDS, name="loss summary")
    expected_summary = {
        "total_loss_weight": totals["loss"],
        "total_active_loss_weight": totals["active"],
        "share_sum": sum(shares.values()),
        "active_share_sum": sum(active_shares.values()),
        "within_tolerance": True,
        "arm_final_cursor_equal": True,
    }
    for field in ("total_loss_weight", "total_active_loss_weight", "share_sum", "active_share_sum"):
        if not math.isclose(
            _finite(summary[field], name=f"loss summary {field}"),
            expected_summary[field],
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise PromotionValidationError(f"Loss summary {field} is inconsistent")
    if summary["within_tolerance"] is not True or summary["arm_final_cursor_equal"] is not True:
        raise PromotionValidationError("Loss summary did not pass")
    return {"loss_mass_share": shares, "active_loss_mass_share": active_shares}


def _promotion_policy() -> dict[str, Any]:
    return {
        "name": PERCEPTION_PROMOTION_POLICY,
        "primary_step": PRIMARY_STEP,
        "durability_step": DURABILITY_STEP,
        "sources": list(SOURCES),
        "primary_window": "all",
        "minimum_examples_per_source": MIN_EXAMPLES_PER_SOURCE,
        "examples_per_source_divisor": EXAMPLES_DIVISOR,
        "confidence": 0.95,
        "source_balanced_macro": True,
        "did_lower_confidence_bound_strictly_positive": True,
        "treatment_gap_lower_confidence_bound_strictly_positive": True,
        "correct_ce_max_relative_increase": CORRECT_CE_MAX_RELATIVE_INCREASE,
        "late_gap_retention_fraction": LATE_GAP_RETENTION_FRACTION,
        "loss_mass_targets": dict(LOSS_MASS_TARGETS),
        "loss_mass_absolute_tolerance": LOSS_MASS_ABSOLUTE_TOLERANCE,
        "optimizer_guard_criterion": "no_guarded_optimizer_skip",
        "required_waiver_ids": sorted(REQUIRED_WAIVER_IDS),
    }


def _guard_deviation(
    treatment_health: Mapping[str, Any], treatment_reference: Mapping[str, Any]
) -> dict[str, Any]:
    deviation: dict[str, Any] = {
        "id": TREATMENT_GUARD_WAIVER_ID,
        "waiver_required": True,
        "arm": TREATMENT_ARM,
        "criterion": "no_guarded_optimizer_skip",
        "reason_code": "optimizer_safety_guard",
        "count": len(EXPECTED_TREATMENT_SKIP_STEPS),
        "steps": list(EXPECTED_TREATMENT_SKIP_STEPS),
        "rate": len(EXPECTED_TREATMENT_SKIP_STEPS) / PRIMARY_STEP,
        "rolling_interval_length": ROLLING_INTERVAL_LENGTH,
        "minimum_spacing": min(
            right - left for left, right in pairwise(EXPECTED_TREATMENT_SKIP_STEPS)
        ),
        "clean_final_steps": PRIMARY_STEP - EXPECTED_TREATMENT_SKIP_STEPS[-1],
        "evidence_receipt_sha256": treatment_reference["sha256"],
        "run_id": treatment_health["run_id"],
    }
    deviation["sha256"] = canonical_sha256(deviation)
    return deviation


def _arm_references(control: Path, treatment: Path) -> dict[str, dict[str, str]]:
    return {
        CONTROL_ARM: artifact_reference(control),
        TREATMENT_ARM: artifact_reference(treatment),
    }


def build_perception_promotion_bundle(
    *,
    checkpoint: Path,
    comparator_checkpoint: Path,
    pair_contract: Path,
    counterfactual_outcome: Path,
    control_initialization_parity: Path,
    treatment_initialization_parity: Path,
    control_frozen_state: Path,
    treatment_frozen_state: Path,
    control_text_retention: Path,
    treatment_text_retention: Path,
    control_run_health: Path,
    treatment_run_health: Path,
    loss_mass_pair: Path,
    created_at: str,
) -> dict[str, Any]:
    """Validate all component receipts and build one immutable approval-ready bundle."""
    _timestamp(created_at, name="promotion bundle created_at")
    _, outcome = _load_reference(
        artifact_reference(counterfactual_outcome), name="counterfactual outcome"
    )
    candidate = candidate_from_outcome_receipt(
        checkpoint, outcome, role="treatment", verify_live_contents=True
    )
    comparator = candidate_from_outcome_receipt(
        comparator_checkpoint, outcome, role="control", verify_live_contents=True
    )
    references: dict[str, Any] = {
        "pair_contract": artifact_reference(pair_contract),
        "initialization_parity": _arm_references(
            control_initialization_parity, treatment_initialization_parity
        ),
        "counterfactual_outcome": artifact_reference(counterfactual_outcome),
        "frozen_state": _arm_references(control_frozen_state, treatment_frozen_state),
        "text_retention": _arm_references(control_text_retention, treatment_text_retention),
        "run_health": _arm_references(control_run_health, treatment_run_health),
        "loss_mass_pair": artifact_reference(loss_mass_pair),
    }
    component = _validate_component_references(
        references, candidate=candidate, comparator=comparator
    )
    deviation = _guard_deviation(
        component["run_health"][TREATMENT_ARM], references["run_health"][TREATMENT_ARM]
    )
    bundle: dict[str, Any] = {
        "format": PERCEPTION_PROMOTION_BUNDLE_FORMAT,
        "version": PERCEPTION_PROMOTION_BUNDLE_VERSION,
        "status": "ready_for_human_approval",
        "created_at": created_at,
        "policy": _promotion_policy(),
        "candidate": candidate,
        "comparator": comparator,
        "receipts": references,
        "deviations": [deviation],
        "content_sha256": "",
    }
    bundle["content_sha256"] = canonical_sha256(
        {key: value for key, value in bundle.items() if key != "content_sha256"}
    )
    return bundle


def _validate_component_references(
    references: Mapping[str, Any], *, candidate: Mapping[str, Any], comparator: Mapping[str, Any]
) -> dict[str, Any]:
    _exact_fields(references, _BUNDLE_RECEIPT_FIELDS, name="bundle receipts")
    _pair_payload, pair = _load_published_pair_contract(references["pair_contract"])
    if (
        candidate["data_contract_sha256"] != pair["data_contract_sha256"]
        or comparator["data_contract_sha256"] != pair["data_contract_sha256"]
        or candidate["trainable_contract_sha256"]
        != pair["trainable_contract_sha256"][TREATMENT_ARM]
        or comparator["trainable_contract_sha256"] != pair["trainable_contract_sha256"][CONTROL_ARM]
    ):
        raise PromotionValidationError("Bundle arm/data contracts differ from pair audit")
    if candidate["data_contract_sha256"] != comparator["data_contract_sha256"]:
        raise PromotionValidationError("Candidate and comparator data contracts differ")
    _, outcome_payload = _load_reference(
        references["counterfactual_outcome"], name="counterfactual outcome"
    )
    outcome = validate_counterfactual_outcome_receipt(outcome_payload, verify_live_inputs=True)
    outcome_candidate = candidate_from_outcome_receipt(
        Path(str(candidate["checkpoint"])),
        outcome_payload,
        role="treatment",
        verify_live_contents=False,
    )
    outcome_comparator = candidate_from_outcome_receipt(
        Path(str(comparator["checkpoint"])),
        outcome_payload,
        role="control",
        verify_live_contents=False,
    )
    if dict(candidate) != outcome_candidate or dict(comparator) != outcome_comparator:
        raise PromotionValidationError("Bundle checkpoints differ from the paired outcome")
    if (
        Path(str(candidate["checkpoint"])).resolve()
        == Path(str(comparator["checkpoint"])).resolve()
    ):
        raise PromotionValidationError("Treatment and control checkpoints must be distinct")
    outcome_pair = outcome_payload.get("inputs", {}).get("profile_pair_receipt")
    if outcome_pair != references["pair_contract"]:
        raise PromotionValidationError("Outcome does not bind the bundle pair contract")
    result: dict[str, Any] = {"pair_contract": pair, "outcome": outcome}
    result["initialization_parity"] = _validate_arm_receipts(
        references["initialization_parity"],
        candidates={CONTROL_ARM: comparator, TREATMENT_ARM: candidate},
        validator=lambda payload, arm, selected: validate_initialization_parity_receipt(
            payload, candidate=selected, expected_arm=arm
        ),
        name="initialization parity",
    )
    control_init = result["initialization_parity"][CONTROL_ARM]
    treatment_init = result["initialization_parity"][TREATMENT_ARM]
    if (
        control_init["reference_checkpoint"]["identity_sha256"]
        != treatment_init["reference_checkpoint"]["identity_sha256"]
        or Path(control_init["reference_checkpoint"]["root"]).resolve()
        != Path(pair["parent_checkpoint"]).resolve()
        or control_init["reference_checkpoint"]["config_sha256"] != pair["parent_config_sha256"]
        or control_init["reference_checkpoint"]["identity_sha256"]
        != pair["parent_checkpoint_identity_sha256"]
        or control_init["comparison_inventory_sha256"]
        != treatment_init["comparison_inventory_sha256"]
    ):
        raise PromotionValidationError("Arm initialization parity does not share one parent state")
    result["frozen_state"] = _validate_arm_receipts(
        references["frozen_state"],
        candidates={CONTROL_ARM: comparator, TREATMENT_ARM: candidate},
        validator=lambda payload, arm, selected: validate_perception_frozen_state_receipt(
            payload,
            candidate=selected,
            expected_frozen_tensor_count=EXPECTED_FROZEN_TENSOR_COUNTS[arm],
        ),
        name="frozen state",
    )
    result["text_retention"] = _validate_arm_receipts(
        references["text_retention"],
        candidates={CONTROL_ARM: comparator, TREATMENT_ARM: candidate},
        validator=lambda payload, _arm, selected: validate_perception_text_retention_receipt(
            payload, candidate=selected
        ),
        name="text retention",
    )
    for evidence_name in ("frozen_state", "text_retention"):
        for arm in ARMS:
            summary = result[evidence_name][arm]
            init = result["initialization_parity"][arm]
            if (
                Path(summary["reference_checkpoint"]).resolve()
                != Path(init["step0_checkpoint"]["root"]).resolve()
                or summary["reference_checkpoint_config_sha256"]
                != init["step0_checkpoint"]["config_sha256"]
                or summary["reference_checkpoint_identity_sha256"]
                != init["step0_checkpoint"]["identity_sha256"]
            ):
                raise PromotionValidationError(f"{evidence_name} {arm} does not bind audited step0")
    result["run_health"] = _validate_arm_receipts(
        references["run_health"],
        candidates={CONTROL_ARM: comparator, TREATMENT_ARM: candidate},
        validator=lambda payload, arm, selected: validate_perception_run_health_receipt(
            payload, candidate=selected, expected_arm=arm
        ),
        name="run health",
    )
    _bind_run_health_to_outcome(result["run_health"], outcome["normalized"])
    _bind_run_health_to_initialization(result["run_health"], result["initialization_parity"])
    if result["run_health"][CONTROL_ARM]["guarded_skip_steps"]:
        raise PromotionValidationError("Control run contains a guarded optimizer skip")
    _, loss_payload = _load_reference(references["loss_mass_pair"], name="loss mass pair")
    if loss_payload.get("pair_contract") != references["pair_contract"]:
        raise PromotionValidationError("Loss-mass receipt does not bind the pair contract")
    result["loss_mass_pair"] = validate_loss_mass_pair_receipt(
        loss_payload, candidate=candidate, comparator=comparator
    )
    _bind_run_health_to_loss(result["run_health"], result["loss_mass_pair"])
    return result


def _bind_run_health_to_initialization(
    run_health: Mapping[str, Any], initialization: Mapping[str, Any]
) -> None:
    """Require each logical step-0 audit to name the completed run's exact checkpoint bytes."""
    for arm in ARMS:
        health_arm = run_health.get(arm)
        initialization_arm = initialization.get(arm)
        health_checkpoints = (
            health_arm.get("permanent_checkpoints") if isinstance(health_arm, Mapping) else None
        )
        health_step0 = (
            health_checkpoints.get("step0") if isinstance(health_checkpoints, Mapping) else None
        )
        initialization_step0 = (
            initialization_arm.get("step0_checkpoint")
            if isinstance(initialization_arm, Mapping)
            else None
        )
        if not isinstance(health_step0, Mapping) or not isinstance(initialization_step0, Mapping):
            raise PromotionValidationError(f"{arm} lacks step0 cross-binding evidence")
        if canonical_sha256(health_step0) != canonical_sha256(initialization_step0):
            raise PromotionValidationError(
                f"{arm} run-health step0 differs from initialization evidence"
            )


def _bind_run_health_to_loss(run_health: Mapping[str, Any], loss_mass: Mapping[str, Any]) -> None:
    """Require cumulative replay to consume the trainer states attested by run health."""
    loss_inventories = loss_mass.get("rank_state_inventory")
    if not isinstance(loss_inventories, Mapping):
        raise PromotionValidationError("Loss replay lacks trainer-state cross-binding evidence")
    for arm in ARMS:
        health_arm = run_health.get(arm)
        health_inventory = (
            health_arm.get("rank_state_inventory") if isinstance(health_arm, Mapping) else None
        )
        loss_inventory = loss_inventories.get(arm)
        if not isinstance(health_inventory, list) or not isinstance(loss_inventory, list):
            raise PromotionValidationError(f"{arm} lacks trainer-state cross-binding evidence")
        if canonical_sha256(health_inventory) != canonical_sha256(loss_inventory):
            raise PromotionValidationError(
                f"{arm} loss replay trainer states differ from run-health evidence"
            )


def _bind_run_health_to_outcome(
    run_health: Mapping[str, Any], outcome_normalized: Mapping[str, Any]
) -> None:
    """Require run-health milestone identities to equal the paired outcome endpoints."""
    outcome_checkpoints = outcome_normalized.get("checkpoints")
    if not isinstance(outcome_checkpoints, Mapping):
        raise PromotionValidationError("Outcome lacks checkpoint identities for run-health binding")
    for arm in ARMS:
        arm_outcome = outcome_checkpoints.get(arm)
        if not isinstance(arm_outcome, Mapping):
            raise PromotionValidationError(f"Outcome lacks {arm} checkpoint identities")
        for step in (DURABILITY_STEP, PRIMARY_STEP):
            key = f"step{step}"
            outcome_identity = arm_outcome.get(key)
            arm_health = run_health.get(arm)
            health_checkpoints = (
                arm_health.get("permanent_checkpoints") if isinstance(arm_health, Mapping) else None
            )
            health_identity = (
                health_checkpoints.get(key) if isinstance(health_checkpoints, Mapping) else None
            )
            if not isinstance(outcome_identity, Mapping) or not isinstance(
                health_identity, Mapping
            ):
                raise PromotionValidationError(
                    f"{arm} step{step} is absent from outcome/run-health evidence"
                )
            if canonical_sha256(health_identity) != canonical_sha256(outcome_identity):
                raise PromotionValidationError(
                    f"{arm} step{step} run-health identity differs from paired outcome"
                )


def _validate_arm_receipts(
    value: Any,
    *,
    candidates: Mapping[str, Mapping[str, Any]],
    validator: Callable[[Mapping[str, Any], str, Mapping[str, Any]], dict[str, Any]],
    name: str,
) -> dict[str, dict[str, Any]]:
    refs = _exact_fields(value, _ARM_MAP_FIELDS, name=f"{name} references")
    summaries: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        _, payload = _load_reference(refs[arm], name=f"{arm} {name}")
        summaries[arm] = validator(payload, arm, candidates[arm])
    return summaries


def validate_perception_promotion_bundle(
    bundle: Mapping[str, Any],
    *,
    expected_checkpoint: Path | None = None,
    expected_checkpoint_config_sha256: str | None = None,
) -> dict[str, Any]:
    """Re-open and validate a perception bundle and all raw-SHA-pinned receipts."""
    _exact_fields(bundle, _BUNDLE_FIELDS, name="perception promotion bundle")
    if (
        bundle["format"] != PERCEPTION_PROMOTION_BUNDLE_FORMAT
        or type(bundle["version"]) is not int
        or bundle["version"] != PERCEPTION_PROMOTION_BUNDLE_VERSION
        or bundle["status"] != "ready_for_human_approval"
        or not isinstance(bundle["policy"], Mapping)
        or canonical_sha256(bundle["policy"]) != canonical_sha256(_promotion_policy())
    ):
        raise PromotionValidationError(
            "Perception promotion bundle identity/policy is incompatible"
        )
    _timestamp(bundle["created_at"], name="promotion bundle created_at")
    _validate_content_sha(bundle, name="perception promotion bundle")
    candidate = _validate_bundle_candidate(bundle["candidate"], name="bundle candidate")
    comparator = _validate_bundle_candidate(bundle["comparator"], name="bundle comparator")
    if (
        expected_checkpoint is not None
        and Path(candidate["checkpoint"]).resolve() != expected_checkpoint.resolve()
    ):
        raise PromotionValidationError("Bundle candidate differs from the selected parent")
    if (
        expected_checkpoint_config_sha256 is not None
        and candidate["checkpoint_config_sha256"] != expected_checkpoint_config_sha256
    ):
        raise PromotionValidationError("Bundle candidate config SHA-256 differs from parent")
    component = _validate_component_references(
        bundle["receipts"], candidate=candidate, comparator=comparator
    )
    expected_deviation = _guard_deviation(
        component["run_health"][TREATMENT_ARM],
        bundle["receipts"]["run_health"][TREATMENT_ARM],
    )
    if canonical_sha256(bundle["deviations"]) != canonical_sha256([expected_deviation]):
        raise PromotionValidationError(
            "Bundle deviations differ from the exact seven-skip evidence"
        )
    return {
        "status": "ready_for_human_approval",
        "candidate": dict(candidate),
        "comparator": dict(comparator),
        "component_summaries": component,
        "deviation_sha256": {TREATMENT_GUARD_WAIVER_ID: expected_deviation["sha256"]},
    }


def validate_approved_perception_parent_gate_bundle(
    bundle: Mapping[str, Any],
    *,
    gate: Mapping[str, Any],
    expected_checkpoint: Path,
    expected_checkpoint_config_sha256: str,
) -> dict[str, Any]:
    """Authenticate the immutable, human-approved perception parent evidence.

    The caller must first raw-SHA authenticate the v3 gate against
    :data:`EXPECTED_APPROVED_PERCEPTION_PARENT_GATE_RAW_SHA256`. The adapter deliberately does not
    re-run historical semantic evaluators: it authenticates the one approved bundle and every
    transitive receipt by raw SHA, extracts the full treatment identity from the strict outcome
    JSON, and re-hashes the complete live treatment checkpoint. The comparator checkpoint is not
    required to remain live after approval.
    """
    gate = _exact_fields(gate, _APPROVED_PARENT_GATE_V3_FIELDS, name="approved perception gate")
    selected_checkpoint = expected_checkpoint.expanduser().resolve()
    selected_config_sha = _sha256(
        expected_checkpoint_config_sha256, name="selected perception parent config SHA-256"
    )
    approved_at = _timestamp(gate["approved_at"], name="approved perception gate approved_at")
    approved_by = gate["approved_by"]
    if (
        gate["format"] != "vision_alignment_parent_gate"
        or type(gate["version"]) is not int
        or gate["version"] != 3
        or gate["status"] != "approved"
        or gate["promotion_kind"] != "perception"
        or gate["promotion_policy"] != PERCEPTION_PROMOTION_POLICY
        or type(gate["recipe_version"]) is not int
        or gate["recipe_version"] != 1
        or gate["formatter_version"] != "vision-alignment-document-v1"
        or gate["phase"] != "perception"
        or Path(str(gate["checkpoint"])).expanduser().resolve() != selected_checkpoint
        or gate["checkpoint_config_sha256"] != selected_config_sha
        or _int(gate["global_step"], name="approved perception gate global step") != PRIMARY_STEP
        or gate["metrics_artifact_sha256"]
        != EXPECTED_APPROVED_PERCEPTION_PROMOTION_BUNDLE_RAW_SHA256
        or gate["promotion_bundle_path"] != EXPECTED_APPROVED_PERCEPTION_PROMOTION_BUNDLE_PATH
        or gate["promotion_bundle_sha256"]
        != EXPECTED_APPROVED_PERCEPTION_PROMOTION_BUNDLE_RAW_SHA256
        or approved_by != EXPECTED_APPROVED_PERCEPTION_APPROVED_BY
        or gate["approved_at"] != EXPECTED_APPROVED_PERCEPTION_APPROVED_AT
        or approved_at.isoformat()
        != _timestamp(
            EXPECTED_APPROVED_PERCEPTION_APPROVED_AT,
            name="allowlisted perception approval timestamp",
        ).isoformat()
        or not isinstance(approved_by, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._@:/+\-]{2,127}", approved_by) is None
    ):
        raise PromotionValidationError("Approved perception gate differs from its allowlist")
    for field in (
        "checkpoint_config_sha256",
        "checkpoint_identity_sha256",
        "data_contract_sha256",
        "trainable_contract_sha256",
        "metrics_artifact_sha256",
        "promotion_bundle_sha256",
    ):
        _sha256(gate[field], name=f"approved perception gate {field}")

    bundle_path, bundle_raw = _load_raw_reference(
        {
            "path": gate["promotion_bundle_path"],
            "sha256": EXPECTED_APPROVED_PERCEPTION_PROMOTION_BUNDLE_RAW_SHA256,
        },
        name="approved perception promotion bundle",
    )
    pinned_bundle = _strict_json_bytes(bundle_raw, name="approved perception promotion bundle")
    if not isinstance(pinned_bundle, Mapping) or pinned_bundle != bundle:
        raise PromotionValidationError(
            "Supplied perception promotion bundle differs from the approved raw artifact"
        )
    bundle = _exact_fields(bundle, _BUNDLE_FIELDS, name="approved perception promotion bundle")
    if (
        bundle_path != Path(EXPECTED_APPROVED_PERCEPTION_PROMOTION_BUNDLE_PATH)
        or bundle["format"] != PERCEPTION_PROMOTION_BUNDLE_FORMAT
        or type(bundle["version"]) is not int
        or bundle["version"] != PERCEPTION_PROMOTION_BUNDLE_VERSION
        or bundle["status"] != "ready_for_human_approval"
        or bundle["content_sha256"] != EXPECTED_APPROVED_PERCEPTION_PROMOTION_BUNDLE_CONTENT_SHA256
        or not isinstance(bundle["policy"], Mapping)
        or canonical_sha256(bundle["policy"])
        != EXPECTED_APPROVED_PERCEPTION_PROMOTION_POLICY_SHA256
        or bundle["policy"].get("name") != PERCEPTION_PROMOTION_POLICY
    ):
        raise PromotionValidationError("Approved perception promotion bundle differs")
    bundle_created_at = _timestamp(
        bundle["created_at"], name="approved perception bundle created_at"
    )
    if approved_at < bundle_created_at:
        raise PromotionValidationError("Perception approval predates its promotion bundle")
    _validate_content_sha(bundle, name="approved perception promotion bundle")

    candidate = _validate_approved_bundle_candidate(
        bundle["candidate"], arm=TREATMENT_ARM, name="approved perception candidate"
    )
    comparator = _validate_approved_bundle_candidate(
        bundle["comparator"], arm=CONTROL_ARM, name="approved perception comparator"
    )
    if (
        Path(str(candidate["checkpoint"])).expanduser().resolve() != selected_checkpoint
        or candidate["checkpoint_config_sha256"] != selected_config_sha
        or candidate["checkpoint"] != gate["checkpoint"]
        or candidate["checkpoint_config_sha256"] != gate["checkpoint_config_sha256"]
        or candidate["checkpoint_identity_sha256"] != gate["checkpoint_identity_sha256"]
        or candidate["data_contract_sha256"] != gate["data_contract_sha256"]
        or candidate["trainable_contract_sha256"] != gate["trainable_contract_sha256"]
        or candidate["global_step"] != gate["global_step"]
        or candidate["phase"] != gate["phase"]
        or candidate["data_contract_sha256"] != comparator["data_contract_sha256"]
    ):
        raise PromotionValidationError(
            "Approved perception candidate, comparator, gate, or selected parent differs"
        )

    outcome = _validate_approved_perception_receipt_references(bundle["receipts"])
    identity = _approved_treatment_identity_from_outcome(
        outcome, expected_checkpoint=selected_checkpoint
    )
    if (
        identity["config_sha256"] != candidate["checkpoint_config_sha256"]
        or identity["identity_sha256"] != candidate["checkpoint_identity_sha256"]
        or identity["checkpoint_marker_sha256"] != candidate["checkpoint_marker_sha256"]
        or identity["dcp_metadata_sha256"] != candidate["dcp_metadata_sha256"]
        or identity["state_file_inventory_sha256"] != candidate["state_file_inventory_sha256"]
    ):
        raise PromotionValidationError(
            "Approved outcome treatment identity differs from the bundle candidate"
        )

    deviation = _validate_approved_perception_deviation(
        bundle["deviations"], receipt_references=bundle["receipts"]
    )
    waivers = gate["waivers"]
    if not isinstance(waivers, list) or len(waivers) != 1:
        raise PromotionValidationError("Approved perception gate must contain exactly one waiver")
    waiver = _exact_fields(
        waivers[0], _APPROVED_WAIVER_FIELDS, name="approved perception gate waiver"
    )
    if (
        waiver["id"] != TREATMENT_GUARD_WAIVER_ID
        or waiver["decision"] != "approved"
        or waiver["deviation_sha256"] != deviation["sha256"]
    ):
        raise PromotionValidationError(
            "Approved perception waiver differs from the exact seven-skip deviation"
        )

    _validate_live_checkpoint_identity_stable(identity, name="approved perception parent")
    marker = load_json_pinned(
        selected_checkpoint / ".metadata.json",
        identity["checkpoint_marker_sha256"],
        name="approved perception parent checkpoint marker",
    )
    if not isinstance(marker, Mapping) or marker.get("ephemeral") is not False:
        raise PromotionValidationError("Approved perception parent must be a permanent checkpoint")
    return {
        "status": "approved",
        "candidate": dict(candidate),
        "comparator": dict(comparator),
        "deviation_sha256": {TREATMENT_GUARD_WAIVER_ID: deviation["sha256"]},
        "approved_by": approved_by,
        "approved_at": gate["approved_at"],
        "bundle_raw_sha256": EXPECTED_APPROVED_PERCEPTION_PROMOTION_BUNDLE_RAW_SHA256,
        "bundle_content_sha256": bundle["content_sha256"],
    }


def _validate_approved_bundle_candidate(value: Any, *, arm: str, name: str) -> Mapping[str, Any]:
    candidate = _exact_fields(value, _CANDIDATE_FIELDS, name=name)
    checkpoint_value = candidate["checkpoint"]
    if not isinstance(checkpoint_value, str) or not checkpoint_value:
        raise PromotionValidationError(f"{name} checkpoint must be a non-empty path")
    checkpoint = Path(checkpoint_value).expanduser()
    expected_contract = EXPECTED_PROFILE_CONTRACTS[arm]
    if (
        checkpoint.name != f"step{PRIMARY_STEP}"
        or _int(candidate["global_step"], name=f"{name} global step") != PRIMARY_STEP
        or candidate["phase"] != "perception"
        or candidate["lineage_id"] != expected_contract["name"]
        or candidate["trainable_contract_sha256"] != expected_contract["trainable_contract_sha256"]
    ):
        raise PromotionValidationError(f"{name} identity differs")
    for field in _SHA_FIELDS:
        _sha256(candidate[field], name=f"{name} {field}")
    vocab_size = _int(candidate["vocab_size"], name=f"{name} vocab size", minimum=1)
    rows = candidate["image_embedding_rows"]
    if (
        not isinstance(rows, list)
        or any(type(row) is not int for row in rows)
        or rows != list(IMAGE_TOKEN_ROWS)
        or any(row >= vocab_size for row in rows)
    ):
        raise PromotionValidationError(f"{name} image embedding rows differ")
    return candidate


def _validate_approved_perception_receipt_references(value: Any) -> Mapping[str, Any]:
    references = _exact_fields(
        value, _BUNDLE_RECEIPT_FIELDS, name="approved perception receipt references"
    )
    raw_by_name: dict[str, bytes] = {}
    for name in ("pair_contract", "counterfactual_outcome", "loss_mass_pair"):
        _, raw_by_name[name] = _load_raw_reference(
            references[name], name=f"approved perception {name} receipt"
        )
    for category in ("initialization_parity", "frozen_state", "text_retention", "run_health"):
        arm_references = _exact_fields(
            references[category], _ARM_MAP_FIELDS, name=f"approved perception {category} references"
        )
        for arm in ARMS:
            _load_raw_reference(
                arm_references[arm], name=f"approved perception {category} {arm} receipt"
            )
    outcome_reference = _exact_fields(
        references["counterfactual_outcome"],
        _ARTIFACT_REF_FIELDS,
        name="approved perception outcome reference",
    )
    if outcome_reference["sha256"] != EXPECTED_APPROVED_PERCEPTION_OUTCOME_RAW_SHA256:
        raise PromotionValidationError("Approved perception outcome differs from its allowlist")
    outcome = _strict_json_bytes(
        raw_by_name["counterfactual_outcome"], name="approved perception outcome"
    )
    if not isinstance(outcome, Mapping):
        raise PromotionValidationError("Approved perception outcome must be an object")
    outcome = _exact_fields(outcome, _APPROVED_OUTCOME_FIELDS, name="approved perception outcome")
    if (
        outcome["format"] != OUTCOME_RECEIPT_FORMAT
        or type(outcome["version"]) is not int
        or outcome["version"] != RECEIPT_VERSION
        or outcome["status"] != "passed"
    ):
        raise PromotionValidationError("Approved perception outcome identity differs")
    _timestamp(outcome["created_at"], name="approved perception outcome created_at")
    _validate_content_sha(outcome, name="approved perception outcome")
    return outcome


def _approved_treatment_identity_from_outcome(
    outcome: Mapping[str, Any], *, expected_checkpoint: Path
) -> Mapping[str, Any]:
    checkpoints = _exact_fields(
        outcome["checkpoints"], _ARM_MAP_FIELDS, name="approved outcome checkpoints"
    )
    for arm in ARMS:
        _exact_fields(
            checkpoints[arm],
            frozenset({f"step{DURABILITY_STEP}", f"step{PRIMARY_STEP}"}),
            name=f"approved outcome {arm} checkpoints",
        )
    identity = _exact_fields(
        checkpoints[TREATMENT_ARM][f"step{PRIMARY_STEP}"],
        _APPROVED_OUTCOME_CHECKPOINT_FIELDS,
        name="approved outcome treatment step4000 identity",
    )
    root_value = identity["root"]
    state_dir_value = identity["state_dir"]
    if (
        not isinstance(root_value, str)
        or not root_value
        or not isinstance(state_dir_value, str)
        or not state_dir_value
    ):
        raise PromotionValidationError("Approved outcome treatment checkpoint paths are malformed")
    root = Path(root_value).expanduser().resolve()
    state_dir = Path(state_dir_value).expanduser().resolve()
    if root != expected_checkpoint or state_dir != root / "model_and_optim":
        raise PromotionValidationError("Approved outcome treatment checkpoint path differs")
    if identity["state_file_hash_algorithm"] != "sha256":
        raise PromotionValidationError("Approved outcome treatment hash algorithm differs")
    for field in (
        "config_sha256",
        "checkpoint_marker_sha256",
        "dcp_metadata_sha256",
        "state_file_inventory_sha256",
        "identity_sha256",
    ):
        _sha256(identity[field], name=f"approved outcome treatment {field}")
    inventory = identity["state_file_inventory"]
    if not isinstance(inventory, list) or not inventory:
        raise PromotionValidationError("Approved outcome treatment inventory is empty")
    normalized_paths: list[str] = []
    for index, raw in enumerate(inventory):
        item = _exact_fields(
            raw,
            _APPROVED_OUTCOME_INVENTORY_FIELDS,
            name=f"approved outcome treatment inventory item {index}",
        )
        relative_value = item["path"]
        if not isinstance(relative_value, str) or not relative_value:
            raise PromotionValidationError(
                f"Approved outcome treatment inventory item {index} path is malformed"
            )
        relative = Path(relative_value)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.parts[:1] != ("model_and_optim",)
            or relative.as_posix() != relative_value
        ):
            raise PromotionValidationError(
                f"Approved outcome treatment inventory item {index} path differs"
            )
        _int(
            item["size"],
            name=f"approved outcome treatment inventory item {index} size",
            minimum=1,
        )
        _sha256(
            item["sha256"],
            name=f"approved outcome treatment inventory item {index} SHA-256",
        )
        normalized_paths.append(relative_value)
    if (
        normalized_paths != sorted(normalized_paths)
        or len(set(normalized_paths)) != len(normalized_paths)
        or "model_and_optim/.metadata" not in normalized_paths
        or canonical_sha256(inventory) != identity["state_file_inventory_sha256"]
        or canonical_sha256(
            {key: value for key, value in identity.items() if key != "identity_sha256"}
        )
        != identity["identity_sha256"]
    ):
        raise PromotionValidationError("Approved outcome treatment inventory identity differs")
    return identity


def _validate_approved_perception_deviation(
    value: Any, *, receipt_references: Mapping[str, Any]
) -> Mapping[str, Any]:
    if not isinstance(value, list) or len(value) != 1:
        raise PromotionValidationError(
            "Approved perception bundle must contain exactly one deviation"
        )
    deviation = _exact_fields(
        value[0], _APPROVED_DEVIATION_FIELDS, name="approved perception deviation"
    )
    treatment_run_health = _exact_fields(
        _exact_fields(
            receipt_references["run_health"],
            _ARM_MAP_FIELDS,
            name="approved perception run-health references",
        )[TREATMENT_ARM],
        _ARTIFACT_REF_FIELDS,
        name="approved perception treatment run-health reference",
    )
    unsigned = {key: item for key, item in deviation.items() if key != "sha256"}
    steps = list(EXPECTED_TREATMENT_SKIP_STEPS)
    if (
        deviation["sha256"] != EXPECTED_APPROVED_PERCEPTION_DEVIATION_SHA256
        or canonical_sha256(unsigned) != EXPECTED_APPROVED_PERCEPTION_DEVIATION_SHA256
        or deviation["id"] != TREATMENT_GUARD_WAIVER_ID
        or deviation["arm"] != TREATMENT_ARM
        or deviation["criterion"] != "no_guarded_optimizer_skip"
        or deviation["waiver_required"] is not True
        or deviation["reason_code"] != "optimizer_safety_guard"
        or deviation["steps"] != steps
        or _int(deviation["count"], name="approved perception deviation count") != len(steps)
        or not math.isclose(
            _finite(deviation["rate"], name="approved perception deviation rate"),
            len(steps) / PRIMARY_STEP,
            rel_tol=0,
            abs_tol=0,
        )
        or _int(
            deviation["minimum_spacing"],
            name="approved perception deviation minimum spacing",
        )
        != min(right - left for left, right in pairwise(steps))
        or _int(
            deviation["clean_final_steps"],
            name="approved perception deviation clean final steps",
        )
        != PRIMARY_STEP - steps[-1]
        or _int(
            deviation["rolling_interval_length"],
            name="approved perception deviation rolling interval",
        )
        != ROLLING_INTERVAL_LENGTH
        or deviation["run_id"] != "4eggnrzc"
        or deviation["evidence_receipt_sha256"] != treatment_run_health["sha256"]
    ):
        raise PromotionValidationError(
            "Approved perception deviation differs from the exact seven-skip evidence"
        )
    return deviation


def _validate_bundle_candidate(value: Any, *, name: str) -> Mapping[str, Any]:
    candidate = _exact_fields(value, _CANDIDATE_FIELDS, name=name)
    checkpoint = _resolved_path(candidate["checkpoint"], name=f"{name} checkpoint")
    if (
        checkpoint.name != "step4000"
        or _int(candidate["global_step"], name=f"{name} global step") != PRIMARY_STEP
    ):
        raise PromotionValidationError(f"{name} must be perception step4000")
    if candidate["phase"] != "perception":
        raise PromotionValidationError(f"{name} phase differs")
    if not isinstance(candidate["lineage_id"], str) or not candidate["lineage_id"]:
        raise PromotionValidationError(f"{name} lineage is empty")
    for field in _SHA_FIELDS:
        _sha256(candidate[field], name=f"{name} {field}")
    vocab_size = _int(candidate["vocab_size"], name=f"{name} vocab size", minimum=1)
    rows = candidate["image_embedding_rows"]
    if (
        not isinstance(rows, list)
        or any(type(row) is not int for row in rows)
        or rows != list(IMAGE_TOKEN_ROWS)
        or any(row >= vocab_size for row in rows)
    ):
        raise PromotionValidationError(f"{name} image embedding rows differ")
    for path, expected, label in (
        (checkpoint / "config.json", candidate["checkpoint_config_sha256"], "config"),
        (checkpoint / ".metadata.json", candidate["checkpoint_marker_sha256"], "marker"),
        (
            checkpoint / "model_and_optim" / ".metadata",
            candidate["dcp_metadata_sha256"],
            "DCP metadata",
        ),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise PromotionValidationError(f"Live {name} {label} differs")
    return candidate


__all__ = [
    "ARMS",
    "CONTROL_ARM",
    "CORRECT_CE_MAX_RELATIVE_INCREASE",
    "COUNTERFACTUAL_OUTCOME_RECEIPT_FORMAT",
    "DURABILITY_STEP",
    "EXPECTED_APPROVED_PERCEPTION_PARENT_GATE_RAW_SHA256",
    "EXPECTED_APPROVED_PERCEPTION_PROMOTION_BUNDLE_CONTENT_SHA256",
    "EXPECTED_APPROVED_PERCEPTION_PROMOTION_BUNDLE_PATH",
    "EXPECTED_APPROVED_PERCEPTION_PROMOTION_BUNDLE_RAW_SHA256",
    "EXPECTED_EXPERIMENT_IDS",
    "EXPECTED_FROZEN_TENSOR_COUNTS",
    "EXPECTED_PAIR_CONTRACT_CONTENT_SHA256",
    "EXPECTED_PAIR_CONTRACT_RAW_SHA256",
    "EXPECTED_TREATMENT_SKIP_STEPS",
    "IMAGE_TOKEN_ROWS",
    "INITIALIZATION_PARITY_RECEIPT_FORMAT",
    "LOSS_MASS_PAIR_RECEIPT_FORMAT",
    "LOSS_MASS_TARGETS",
    "OUTCOME_RECEIPT_FORMAT",
    "PAIR_CONTRACT_RECEIPT_FORMAT",
    "PERCEPTION_FROZEN_STATE_RECEIPT_FORMAT",
    "PERCEPTION_PROMOTION_BUNDLE_FORMAT",
    "PERCEPTION_PROMOTION_BUNDLE_VERSION",
    "PERCEPTION_PROMOTION_POLICY",
    "PERCEPTION_TEXT_RETENTION_RECEIPT_FORMAT",
    "PRIMARY_STEP",
    "RECEIPT_VERSION",
    "REQUIRED_WAIVER_IDS",
    "RUN_HEALTH_RECEIPT_FORMAT",
    "SOURCES",
    "TREATMENT_ARM",
    "TREATMENT_GUARD_WAIVER_ID",
    "PromotionValidationError",
    "artifact_reference",
    "build_perception_promotion_bundle",
    "candidate_from_outcome_receipt",
    "canonical_sha256",
    "load_json",
    "load_json_pinned",
    "load_trainer_state",
    "sha256_file",
    "validate_counterfactual_outcome_receipt",
    "validate_approved_perception_parent_gate_bundle",
    "validate_initialization_parity_receipt",
    "validate_loss_mass_pair_receipt",
    "validate_pair_contract_receipt",
    "validate_perception_frozen_state_receipt",
    "validate_perception_promotion_bundle",
    "validate_perception_run_health_receipt",
    "validate_perception_text_retention_receipt",
]
