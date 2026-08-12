"""Fail-closed evidence bundle for promoting a Vision Alignment bridge checkpoint.

This module deliberately does not run training or manufacture scientific measurements. It
validates immutable receipts emitted by the corresponding distributed audits, checks their
cross-artifact identities and preregistered acceptance criteria, and assembles one compact
promotion bundle for a human to approve. The parent-quality gate pins the raw bundle bytes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from olmo_core.eval.matched_wrong_image import (
    matched_wrong_image_pairing_sha256,
    validate_matched_wrong_image_pairing,
)

PROMOTION_BUNDLE_FORMAT = "vision_alignment_bridge_promotion_bundle"
PROMOTION_BUNDLE_VERSION = 1
PROMOTION_POLICY = "vision-alignment-bridge-step500-promotion-v1"

FROZEN_STATE_RECEIPT_FORMAT = "vision_alignment_frozen_state_receipt"
TEXT_RETENTION_RECEIPT_FORMAT = "vision_alignment_text_retention_receipt"
TEXT_SENTINEL_FORMAT = "vision_alignment_text_sentinel"
LOSS_MASS_RECEIPT_FORMAT = "vision_alignment_cumulative_loss_mass_receipt"
OPTIMIZER_GUARD_RECEIPT_FORMAT = "vision_alignment_optimizer_guard_receipt"
RECEIPT_VERSION = 1

STEP250_WAIVER_ID = "step250_caption_first32_90pct_canary"
STEP356_WAIVER_ID = "step356_optimizer_guard_skip"
REQUIRED_WAIVER_IDS = frozenset({STEP250_WAIVER_ID, STEP356_WAIVER_ID})

SOURCES = ("pixmo_caption", "pixmo_transcript")
PRIMARY_WINDOWS = ("first_8", "first_32", "all")
RETENTION_WINDOWS = ("first_8", "first_32")
INDEPENDENT_PAIRING_SEED_OFFSET = 1_000_003
IMAGE_TOKEN_ROWS = (100278, 100279, 100280, 100281, 100282, 100283)
SHA256_RE = re.compile(r"[0-9a-f]{64}")

_ARTIFACT_REF_FIELDS = frozenset({"path", "sha256"})
_CANDIDATE_FIELDS = frozenset(
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
_BUNDLE_FIELDS = frozenset(
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
_RECEIPTS_FIELDS = frozenset(
    {
        "frozen_state",
        "text_retention",
        "cumulative_loss_mass",
        "optimizer_guard",
        "matched_wrong",
    }
)
_MATCHED_RECEIPTS_FIELDS = frozenset(
    {
        "canary_step250",
        "bridge_step250",
        "bridge_step500",
        "independent_step0",
        "independent_step500",
    }
)


class PromotionValidationError(ValueError):
    """Raised when a bridge-promotion artifact fails its locked evidence contract."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the canonical JSON representation used for semantic SHA-256 identities."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible value in canonical form."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file without loading it entirely into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PromotionValidationError(f"JSON repeats key {key!r}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    """Load strict finite JSON from ``path``."""

    def reject_constant(value: str) -> Any:
        raise PromotionValidationError(f"JSON contains non-finite constant {value}")

    try:
        return json.loads(
            path.read_text(),
            object_pairs_hook=_strict_json_object,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PromotionValidationError(f"Could not read JSON artifact {path}: {error}") from error


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
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise PromotionValidationError(f"{name} must be a lowercase SHA-256")
    return value


def _positive_int(value: Any, *, name: str, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise PromotionValidationError(f"{name} must be a {qualifier} integer")
    return value


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PromotionValidationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PromotionValidationError(f"{name} must be finite")
    return result


def _timestamp(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise PromotionValidationError(f"{name} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise PromotionValidationError(f"{name} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PromotionValidationError(f"{name} must include a timezone")
    return value


def _resolved_path(value: Any, *, name: str, must_exist: bool = True) -> Path:
    if not isinstance(value, str) or not value:
        raise PromotionValidationError(f"{name} must be a non-empty path")
    path = Path(value).expanduser().resolve()
    if must_exist and not path.is_file() and not path.is_dir():
        raise PromotionValidationError(f"{name} does not exist: {path}")
    return path


def artifact_reference(path: Path) -> dict[str, str]:
    """Return an absolute, raw-byte SHA-pinned artifact reference."""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise PromotionValidationError(f"Receipt does not exist: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _load_reference(reference: Any, *, name: str) -> tuple[Path, Mapping[str, Any]]:
    ref = _exact_fields(reference, _ARTIFACT_REF_FIELDS, name=f"{name} reference")
    expected_sha = _sha256(ref["sha256"], name=f"{name} reference SHA-256")
    path = _resolved_path(ref["path"], name=f"{name} reference path")
    if not path.is_file():
        raise PromotionValidationError(f"{name} reference is not a file: {path}")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise PromotionValidationError(
            f"{name} receipt SHA-256 differs: expected {expected_sha}, got {actual_sha}"
        )
    payload = load_json(path)
    if not isinstance(payload, Mapping):
        raise PromotionValidationError(f"{name} receipt must contain a JSON object")
    return path, payload


def _validate_raw_reference(reference: Any, *, name: str) -> Path:
    ref = _exact_fields(reference, _ARTIFACT_REF_FIELDS, name=f"{name} reference")
    expected_sha = _sha256(ref["sha256"], name=f"{name} reference SHA-256")
    path = _resolved_path(ref["path"], name=f"{name} reference path")
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise PromotionValidationError(f"{name} artifact SHA-256 differs")
    return path


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
        or receipt["version"] != RECEIPT_VERSION
        or receipt["status"] != "passed"
    ):
        raise PromotionValidationError(f"{name} identity or status is incompatible")
    _timestamp(receipt["created_at"], name=f"{name} created_at")


def _validate_receipt_candidate(value: Any, *, expected: Mapping[str, Any], name: str) -> None:
    fields = frozenset(
        {"checkpoint", "global_step", "checkpoint_config_sha256", "checkpoint_identity_sha256"}
    )
    candidate = _exact_fields(value, fields, name=name)
    if (
        _resolved_path(candidate["checkpoint"], name=f"{name} checkpoint", must_exist=False)
        != Path(str(expected["checkpoint"])).resolve()
    ):
        raise PromotionValidationError(f"{name} checkpoint differs from the promotion candidate")
    if candidate["global_step"] != expected["global_step"]:
        raise PromotionValidationError(f"{name} global_step differs from the candidate")
    for field in ("checkpoint_config_sha256", "checkpoint_identity_sha256"):
        _sha256(candidate[field], name=f"{name} {field}")
        if candidate[field] != expected[field]:
            raise PromotionValidationError(f"{name} {field} differs from the candidate")


_FROZEN_FIELDS = frozenset(
    {
        "format",
        "version",
        "status",
        "created_at",
        "evaluator",
        "candidate",
        "reference_checkpoint",
        "protocol",
        "comparisons",
        "summary",
    }
)
_FROZEN_REFERENCE_FIELDS = frozenset(
    {"checkpoint", "global_step", "checkpoint_config_sha256", "checkpoint_identity_sha256"}
)
_FROZEN_PROTOCOL_FIELDS = frozenset(
    {"name", "hash_algorithm", "tensor_encoding", "image_embedding_rows"}
)
_FROZEN_COMPARISON_FIELDS = frozenset(
    {
        "name",
        "kind",
        "dtype",
        "shape",
        "numel",
        "reference_sha256",
        "candidate_sha256",
    }
)
_FROZEN_SUMMARY_FIELDS = frozenset(
    {
        "complete",
        "expected_frozen_tensor_count",
        "compared_frozen_tensor_count",
        "non_image_embedding_row_count",
        "mismatch_count",
        "comparison_inventory_sha256",
    }
)


def validate_frozen_state_receipt(
    receipt: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    expected_frozen_tensor_count: int | None = None,
) -> dict[str, Any]:
    """Validate exact frozen tensors and non-image input-embedding rows."""
    _validate_receipt_header(
        receipt,
        expected_format=FROZEN_STATE_RECEIPT_FORMAT,
        expected_fields=_FROZEN_FIELDS,
        name="frozen-state receipt",
    )
    evaluator_path = _validate_raw_reference(receipt["evaluator"], name="frozen-state evaluator")
    if evaluator_path.name != "vision_alignment_state_text.py":
        raise PromotionValidationError("Frozen-state receipt names an incompatible evaluator")
    _validate_receipt_candidate(
        receipt["candidate"], expected=candidate, name="frozen-state candidate"
    )
    reference = _exact_fields(
        receipt["reference_checkpoint"],
        _FROZEN_REFERENCE_FIELDS,
        name="frozen-state reference checkpoint",
    )
    reference_path = _resolved_path(
        reference["checkpoint"],
        name="frozen-state reference checkpoint",
        must_exist=False,
    )
    if reference_path.name != "step0" or reference["global_step"] != 0:
        raise PromotionValidationError("Frozen-state reference must be the bridge step0 checkpoint")
    if reference_path.parent != Path(str(candidate["checkpoint"])).resolve().parent:
        raise PromotionValidationError("Frozen-state reference and candidate must share a lineage")
    _sha256(reference["checkpoint_config_sha256"], name="frozen reference config SHA-256")
    _sha256(reference["checkpoint_identity_sha256"], name="frozen reference identity SHA-256")
    if reference["checkpoint_config_sha256"] != candidate["checkpoint_config_sha256"]:
        raise PromotionValidationError("Frozen-state step0 config differs from the candidate")

    protocol = _exact_fields(
        receipt["protocol"], _FROZEN_PROTOCOL_FIELDS, name="frozen-state protocol"
    )
    if (
        protocol["name"] != "logical-tensor-sha256-v1"
        or protocol["hash_algorithm"] != "sha256"
        or protocol["tensor_encoding"] != "dtype-shape-contiguous-little-endian-v1"
        or protocol["image_embedding_rows"] != candidate["image_embedding_rows"]
    ):
        raise PromotionValidationError("Frozen-state comparison protocol is incompatible")

    comparisons = receipt["comparisons"]
    if not isinstance(comparisons, list) or not comparisons:
        raise PromotionValidationError("Frozen-state receipt must include tensor comparisons")
    names: set[str] = set()
    frozen_count = 0
    non_image_count = 0
    normalized: list[Mapping[str, Any]] = []
    for index, raw in enumerate(comparisons):
        comparison = _exact_fields(
            raw, _FROZEN_COMPARISON_FIELDS, name=f"frozen comparison {index}"
        )
        name = comparison["name"]
        if not isinstance(name, str) or not name or name in names:
            raise PromotionValidationError("Frozen comparison names must be non-empty and unique")
        names.add(name)
        kind = comparison["kind"]
        if kind == "frozen_tensor":
            frozen_count += 1
        elif kind == "non_image_embedding_rows":
            non_image_count += 1
        else:
            raise PromotionValidationError(f"Unknown frozen comparison kind {kind!r}")
        if not isinstance(comparison["dtype"], str) or not comparison["dtype"]:
            raise PromotionValidationError(f"Frozen comparison {name!r} has an invalid dtype")
        shape = comparison["shape"]
        if not isinstance(shape, list) or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in shape
        ):
            raise PromotionValidationError(f"Frozen comparison {name!r} has an invalid shape")
        _positive_int(comparison["numel"], name=f"frozen comparison {name!r} numel")
        reference_sha = _sha256(
            comparison["reference_sha256"], name=f"frozen comparison {name!r} reference"
        )
        candidate_sha = _sha256(
            comparison["candidate_sha256"], name=f"frozen comparison {name!r} candidate"
        )
        if reference_sha != candidate_sha:
            raise PromotionValidationError(f"Frozen comparison {name!r} differs")
        normalized.append(comparison)
    if non_image_count != 1:
        raise PromotionValidationError(
            "Frozen-state receipt must contain exactly one non-image embedding-row comparison"
        )

    summary = _exact_fields(receipt["summary"], _FROZEN_SUMMARY_FIELDS, name="frozen-state summary")
    expected_count = _positive_int(
        summary["expected_frozen_tensor_count"], name="expected frozen tensor count"
    )
    compared_count = _positive_int(
        summary["compared_frozen_tensor_count"], name="compared frozen tensor count"
    )
    non_image_rows = _positive_int(
        summary["non_image_embedding_row_count"], name="non-image embedding row count"
    )
    if (
        summary["complete"] is not True
        or summary["mismatch_count"] != 0
        or expected_count != compared_count
        or compared_count != frozen_count
        or non_image_rows != candidate["vocab_size"] - len(candidate["image_embedding_rows"])
        or (
            expected_frozen_tensor_count is not None
            and expected_count != expected_frozen_tensor_count
        )
    ):
        raise PromotionValidationError("Frozen-state receipt is incomplete or reports differences")
    inventory = sorted(normalized, key=lambda item: (str(item["kind"]), str(item["name"])))
    if summary["comparison_inventory_sha256"] != canonical_sha256(inventory):
        raise PromotionValidationError("Frozen-state comparison inventory SHA-256 differs")
    return {
        "frozen_tensor_count": frozen_count,
        "non_image_embedding_row_count": non_image_rows,
        "reference_checkpoint": str(reference_path),
        "reference_checkpoint_config_sha256": reference["checkpoint_config_sha256"],
        "reference_checkpoint_identity_sha256": reference["checkpoint_identity_sha256"],
    }


_TEXT_FIELDS = frozenset(
    {
        "format",
        "version",
        "status",
        "created_at",
        "evaluator",
        "candidate",
        "reference_checkpoint",
        "dataset",
        "protocol",
        "metrics",
    }
)
_TEXT_DATASET_FIELDS = frozenset(
    {
        "path",
        "sha256",
        "fingerprint",
        "examples",
        "supervised_tokens",
        "input_ids_sha256",
        "labels_sha256",
        "image_token_count",
        "image_tensor_count",
    }
)
_TEXT_PROTOCOL_FIELDS = frozenset(
    {"name", "atol", "rtol", "same_topology", "same_backend", "image_free"}
)
_TEXT_METRIC_FIELDS = frozenset(
    {
        "all_finite",
        "reference_mean_ce",
        "candidate_mean_ce",
        "max_abs_token_ce_delta",
        "max_rel_token_ce_delta",
        "argmax_matches",
        "argmax_total",
    }
)
_TEXT_SENTINEL_FIELDS = frozenset(
    {
        "format",
        "version",
        "parent_checkpoint",
        "parent_checkpoint_config_sha256",
        "parent_data_paths",
        "selection",
        "rows",
        "content_sha256",
    }
)
_TEXT_SENTINEL_PARENT_PATHS_FIELDS = frozenset({"path", "sha256", "count"})
_TEXT_SENTINEL_SELECTION_FIELDS = frozenset(
    {"algorithm", "examples", "sequence_length", "dtype", "source_indices"}
)
_TEXT_SENTINEL_ROW_FIELDS = frozenset({"source_index", "source_path", "start", "tokens"})


def validate_text_sentinel(
    sentinel: Mapping[str, Any], *, expected_raw_sha256: str | None = None, path: Path | None = None
) -> dict[str, Any]:
    """Validate a deterministic image-free slice of the bare parent's pretraining corpus."""
    _exact_fields(sentinel, _TEXT_SENTINEL_FIELDS, name="text sentinel")
    if sentinel["format"] != TEXT_SENTINEL_FORMAT or sentinel["version"] != RECEIPT_VERSION:
        raise PromotionValidationError("Text sentinel identity is incompatible")
    _resolved_path(
        sentinel["parent_checkpoint"], name="text sentinel parent checkpoint", must_exist=False
    )
    _sha256(sentinel["parent_checkpoint_config_sha256"], name="text sentinel parent config SHA-256")
    parent_paths = _exact_fields(
        sentinel["parent_data_paths"],
        _TEXT_SENTINEL_PARENT_PATHS_FIELDS,
        name="text sentinel parent data paths",
    )
    parent_path = _resolved_path(parent_paths["path"], name="text sentinel parent data paths")
    parent_sha = _sha256(parent_paths["sha256"], name="text sentinel data paths SHA-256")
    if not parent_path.is_file() or sha256_file(parent_path) != parent_sha:
        raise PromotionValidationError("Text sentinel parent data-path manifest differs")
    source_paths = parent_path.read_text().splitlines()
    if len(source_paths) != parent_paths["count"] or not source_paths:
        raise PromotionValidationError("Text sentinel parent data-path count differs")

    selection = _exact_fields(
        sentinel["selection"], _TEXT_SENTINEL_SELECTION_FIELDS, name="text sentinel selection"
    )
    examples = _positive_int(selection["examples"], name="text sentinel examples")
    sequence_length = _positive_int(
        selection["sequence_length"], name="text sentinel sequence length"
    )
    if (
        selection["algorithm"] != "evenly-spaced-parent-path-first-window-v1"
        or selection["dtype"] != "uint32-little-endian"
        or examples < 128
        or examples * sequence_length < 32_768
    ):
        raise PromotionValidationError("Text sentinel selection policy is incompatible")
    expected_indices = [(index * len(source_paths)) // examples for index in range(examples)]
    if selection["source_indices"] != expected_indices:
        raise PromotionValidationError("Text sentinel source selection is not deterministic")

    rows = sentinel["rows"]
    if not isinstance(rows, list) or len(rows) != examples:
        raise PromotionValidationError("Text sentinel row count differs")
    input_ids: list[list[int]] = []
    labels: list[list[int]] = []
    for position, raw_row in enumerate(rows):
        row = _exact_fields(
            raw_row, _TEXT_SENTINEL_ROW_FIELDS, name=f"text sentinel row {position}"
        )
        source_index = row["source_index"]
        if (
            source_index != expected_indices[position]
            or row["source_path"] != source_paths[source_index]
        ):
            raise PromotionValidationError("Text sentinel row source differs from its manifest")
        if row["start"] != 0:
            raise PromotionValidationError("Text sentinel rows must use the first source window")
        tokens = row["tokens"]
        if not isinstance(tokens, list) or len(tokens) != sequence_length + 1:
            raise PromotionValidationError("Text sentinel token window length differs")
        if any(
            isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < 100278
            for token in tokens
        ):
            raise PromotionValidationError("Text sentinel contains an invalid or image token ID")
        input_ids.append(tokens[:-1])
        labels.append(tokens[1:])
    without_sha = {key: value for key, value in sentinel.items() if key != "content_sha256"}
    content_sha = _sha256(sentinel["content_sha256"], name="text sentinel content SHA-256")
    if canonical_sha256(without_sha) != content_sha:
        raise PromotionValidationError("Text sentinel content SHA-256 differs")
    if expected_raw_sha256 is not None and (
        path is None or sha256_file(path) != expected_raw_sha256
    ):
        raise PromotionValidationError("Text sentinel raw SHA-256 differs")
    return {
        "examples": examples,
        "sequence_length": sequence_length,
        "supervised_tokens": examples * sequence_length,
        "input_ids": input_ids,
        "labels": labels,
        "input_ids_sha256": canonical_sha256(input_ids),
        "labels_sha256": canonical_sha256(labels),
        "fingerprint": content_sha,
    }


def build_text_sentinel(
    *,
    parent_checkpoint: Path,
    parent_checkpoint_config_sha256: str,
    parent_data_paths: Path,
    expected_parent_data_paths_sha256: str,
    sequence_length: int = 256,
    examples: int = 128,
) -> dict[str, Any]:
    """Read deterministic tiny ranges from the bare parent's exact expanded data manifest."""
    from olmo_core.data.utils import load_array_slice

    parent_checkpoint = parent_checkpoint.expanduser().resolve()
    config_path = parent_checkpoint / "config.json"
    if not config_path.is_file() or sha256_file(config_path) != parent_checkpoint_config_sha256:
        raise PromotionValidationError("Text sentinel parent checkpoint config differs")
    parent_data_paths = parent_data_paths.expanduser().resolve()
    if (
        not parent_data_paths.is_file()
        or sha256_file(parent_data_paths) != expected_parent_data_paths_sha256
    ):
        raise PromotionValidationError("Text sentinel parent data-path manifest differs")
    source_paths = parent_data_paths.read_text().splitlines()
    if not source_paths:
        raise PromotionValidationError("Text sentinel parent data-path manifest is empty")
    _positive_int(sequence_length, name="text sentinel sequence length")
    _positive_int(examples, name="text sentinel examples")
    if examples < 128 or examples * sequence_length < 32_768:
        raise PromotionValidationError("Text sentinel is smaller than the locked minimum")
    source_indices = [(index * len(source_paths)) // examples for index in range(examples)]
    rows = []
    for source_index in source_indices:
        source_path = source_paths[source_index]
        values = load_array_slice(source_path, 0, sequence_length + 1, dtype=np.uint32)
        tokens = [int(value) for value in values]
        if len(tokens) != sequence_length + 1:
            raise PromotionValidationError(
                f"Could not read a complete sentinel row from {source_path}"
            )
        rows.append(
            {
                "source_index": source_index,
                "source_path": source_path,
                "start": 0,
                "tokens": tokens,
            }
        )
    sentinel: dict[str, Any] = {
        "format": TEXT_SENTINEL_FORMAT,
        "version": RECEIPT_VERSION,
        "parent_checkpoint": str(parent_checkpoint),
        "parent_checkpoint_config_sha256": parent_checkpoint_config_sha256,
        "parent_data_paths": {
            "path": str(parent_data_paths),
            "sha256": expected_parent_data_paths_sha256,
            "count": len(source_paths),
        },
        "selection": {
            "algorithm": "evenly-spaced-parent-path-first-window-v1",
            "examples": examples,
            "sequence_length": sequence_length,
            "dtype": "uint32-little-endian",
            "source_indices": source_indices,
        },
        "rows": rows,
        "content_sha256": "",
    }
    sentinel["content_sha256"] = canonical_sha256(
        {key: value for key, value in sentinel.items() if key != "content_sha256"}
    )
    validate_text_sentinel(sentinel)
    return sentinel


def validate_text_retention_receipt(
    receipt: Mapping[str, Any], *, candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate a pinned image-free parent-versus-candidate language sentinel."""
    _validate_receipt_header(
        receipt,
        expected_format=TEXT_RETENTION_RECEIPT_FORMAT,
        expected_fields=_TEXT_FIELDS,
        name="text-retention receipt",
    )
    evaluator_path = _validate_raw_reference(receipt["evaluator"], name="text-retention evaluator")
    if evaluator_path.name != "vision_alignment_state_text.py":
        raise PromotionValidationError("Text-retention receipt names an incompatible evaluator")
    _validate_receipt_candidate(
        receipt["candidate"], expected=candidate, name="text-retention candidate"
    )
    reference = _exact_fields(
        receipt["reference_checkpoint"],
        _FROZEN_REFERENCE_FIELDS,
        name="text-retention reference checkpoint",
    )
    reference_path = _resolved_path(
        reference["checkpoint"], name="text-retention reference checkpoint", must_exist=False
    )
    if reference_path.name != "step0" or reference["global_step"] != 0:
        raise PromotionValidationError("Text-retention reference must be bridge step0")
    if reference_path.parent != Path(str(candidate["checkpoint"])).resolve().parent:
        raise PromotionValidationError(
            "Text-retention reference and candidate must share a lineage"
        )
    _sha256(reference["checkpoint_config_sha256"], name="text reference config SHA-256")
    _sha256(reference["checkpoint_identity_sha256"], name="text reference identity SHA-256")
    if reference["checkpoint_config_sha256"] != candidate["checkpoint_config_sha256"]:
        raise PromotionValidationError("Text-retention step0 config differs from the candidate")

    dataset = _exact_fields(receipt["dataset"], _TEXT_DATASET_FIELDS, name="text dataset")
    dataset_path = _resolved_path(dataset["path"], name="text dataset")
    for field in ("sha256", "input_ids_sha256", "labels_sha256"):
        _sha256(dataset[field], name=f"text dataset {field}")
    if not isinstance(dataset["fingerprint"], str) or not dataset["fingerprint"]:
        raise PromotionValidationError("Text dataset fingerprint must be non-empty")
    if sha256_file(dataset_path) != dataset["sha256"]:
        raise PromotionValidationError("Text sentinel raw SHA-256 differs")
    sentinel = load_json(dataset_path)
    if not isinstance(sentinel, Mapping):
        raise PromotionValidationError("Text sentinel must be a JSON object")
    sentinel_summary = validate_text_sentinel(
        sentinel, expected_raw_sha256=dataset["sha256"], path=dataset_path
    )
    examples = _positive_int(dataset["examples"], name="text dataset examples")
    supervised_tokens = _positive_int(
        dataset["supervised_tokens"], name="text supervised token count"
    )
    if examples < 128 or supervised_tokens < 32_768:
        raise PromotionValidationError(
            "Text-retention sentinel must contain at least 128 examples and 32,768 tokens"
        )
    if dataset["image_token_count"] != 0 or dataset["image_tensor_count"] != 0:
        raise PromotionValidationError("Text-retention sentinel must be image-free")
    if (
        dataset["examples"] != sentinel_summary["examples"]
        or dataset["supervised_tokens"] != sentinel_summary["supervised_tokens"]
        or dataset["input_ids_sha256"] != sentinel_summary["input_ids_sha256"]
        or dataset["labels_sha256"] != sentinel_summary["labels_sha256"]
        or dataset["fingerprint"] != sentinel_summary["fingerprint"]
    ):
        raise PromotionValidationError("Text-retention dataset claims differ from the sentinel")

    protocol = _exact_fields(
        receipt["protocol"], _TEXT_PROTOCOL_FIELDS, name="text-retention protocol"
    )
    atol = _finite(protocol["atol"], name="text-retention atol")
    rtol = _finite(protocol["rtol"], name="text-retention rtol")
    if (
        protocol["name"] != "per-token-nll-and-argmax-v1"
        or not 0 <= atol <= 1e-6
        or not 0 <= rtol <= 1e-6
        or protocol["same_topology"] is not True
        or protocol["same_backend"] is not True
        or protocol["image_free"] is not True
    ):
        raise PromotionValidationError("Text-retention protocol is weaker than the locked policy")

    metrics = _exact_fields(receipt["metrics"], _TEXT_METRIC_FIELDS, name="text-retention metrics")
    for field in (
        "reference_mean_ce",
        "candidate_mean_ce",
        "max_abs_token_ce_delta",
        "max_rel_token_ce_delta",
    ):
        _finite(metrics[field], name=f"text-retention {field}")
    matches = _positive_int(metrics["argmax_matches"], name="text argmax matches")
    total = _positive_int(metrics["argmax_total"], name="text argmax total")
    if (
        metrics["all_finite"] is not True
        or metrics["max_abs_token_ce_delta"] > atol
        or metrics["max_rel_token_ce_delta"] > rtol
        or matches != total
        or total != supervised_tokens
    ):
        raise PromotionValidationError("Text-retention metrics fail the locked tolerance")
    return {
        "examples": examples,
        "supervised_tokens": supervised_tokens,
        "max_abs_token_ce_delta": float(metrics["max_abs_token_ce_delta"]),
        "argmax_match_rate": 1.0,
        "reference_checkpoint": str(reference_path),
        "reference_checkpoint_config_sha256": reference["checkpoint_config_sha256"],
        "reference_checkpoint_identity_sha256": reference["checkpoint_identity_sha256"],
    }


_LOSS_FIELDS = frozenset(
    {
        "format",
        "version",
        "status",
        "created_at",
        "candidate",
        "protocol",
        "loader",
        "evidence",
        "sources",
        "summary",
    }
)
_LOSS_PROTOCOL_FIELDS = frozenset(
    {"name", "start_step", "end_step", "share_tolerance", "exact_packing_cursor"}
)
_LOSS_LOADER_FIELDS = frozenset(
    {
        "data_contract_sha256",
        "dataset_fingerprints_sha256",
        "initial_state_sha256",
        "checkpoint_final_state_sha256",
        "replayed_final_state_sha256",
        "rank_state_inventory_sha256",
        "rank_state_count",
        "rank_states_global_step",
        "rank_states_batches_processed",
        "dp_world_size",
        "batches_replayed",
        "total_data_errors",
    }
)
_LOSS_SOURCE_FIELDS = frozenset(
    {
        "examples",
        "tokens",
        "positive_tokens",
        "loss_weight",
        "active_loss_weight",
        "loss_mass_share",
        "active_loss_mass_share",
        "target_loss_mass",
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
    }
)
_LOSS_EVIDENCE_FIELDS = frozenset({"recipe", "producer", "rank_state_inventory"})
_LOSS_RANK_STATE_FIELDS = frozenset({"rank", "path", "sha256"})


def validate_loss_mass_receipt(
    receipt: Mapping[str, Any], *, candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate an exact cumulative replay of all bridge loader batches."""
    _validate_receipt_header(
        receipt,
        expected_format=LOSS_MASS_RECEIPT_FORMAT,
        expected_fields=_LOSS_FIELDS,
        name="loss-mass receipt",
    )
    _validate_receipt_candidate(
        receipt["candidate"], expected=candidate, name="loss-mass candidate"
    )
    protocol = _exact_fields(receipt["protocol"], _LOSS_PROTOCOL_FIELDS, name="loss-mass protocol")
    tolerance = _finite(protocol["share_tolerance"], name="loss-mass tolerance")
    if (
        protocol["name"] != "exact-packed-loader-cumulative-loss-mass-v1"
        or protocol["start_step"] != 0
        or protocol["end_step"] != candidate["global_step"]
        or tolerance != 0.02
        or protocol["exact_packing_cursor"] is not True
    ):
        raise PromotionValidationError("Cumulative loss-mass protocol is incompatible")

    loader = _exact_fields(receipt["loader"], _LOSS_LOADER_FIELDS, name="loss-mass loader")
    for field in (
        "data_contract_sha256",
        "dataset_fingerprints_sha256",
        "initial_state_sha256",
        "checkpoint_final_state_sha256",
        "replayed_final_state_sha256",
        "rank_state_inventory_sha256",
    ):
        _sha256(loader[field], name=f"loss-mass loader {field}")
    if loader["data_contract_sha256"] != candidate["data_contract_sha256"]:
        raise PromotionValidationError("Loss-mass receipt uses a different data contract")
    dp_world_size = _positive_int(loader["dp_world_size"], name="loss-mass DP world size")
    rank_state_count = _positive_int(loader["rank_state_count"], name="loss-mass rank state count")
    if (
        loader["batches_replayed"] != candidate["global_step"]
        or loader["rank_states_global_step"] != candidate["global_step"]
        or loader["rank_states_batches_processed"] != candidate["global_step"]
        or loader["total_data_errors"] != 0
        or loader["checkpoint_final_state_sha256"] != loader["replayed_final_state_sha256"]
    ):
        raise PromotionValidationError("Loss-mass replay is incomplete or contains data errors")

    evidence = _exact_fields(receipt["evidence"], _LOSS_EVIDENCE_FIELDS, name="loss-mass evidence")
    recipe_path = _validate_raw_reference(evidence["recipe"], name="loss-mass recipe")
    producer_path = _validate_raw_reference(evidence["producer"], name="loss-mass producer")
    if recipe_path.name != "Vision-Alignment.py" or producer_path.name != (
        "vision_alignment_loss_mass.py"
    ):
        raise PromotionValidationError("Loss-mass evidence names incompatible implementations")
    rank_inventory = evidence["rank_state_inventory"]
    if (
        not isinstance(rank_inventory, list)
        or len(rank_inventory) != rank_state_count
        or rank_state_count != dp_world_size
        or canonical_sha256(rank_inventory) != loader["rank_state_inventory_sha256"]
    ):
        raise PromotionValidationError("Loss-mass rank-state evidence is incomplete")
    try:
        import torch
    except ImportError as error:  # pragma: no cover - OLMo-core always requires torch.
        raise PromotionValidationError("PyTorch is required to audit loader rank states") from error
    checkpoint_states: list[dict[str, Any]] = []
    dataset_fingerprints: Any = None
    observed_data_errors = 0
    for expected_rank, raw_rank in enumerate(rank_inventory):
        rank_entry = _exact_fields(
            raw_rank, _LOSS_RANK_STATE_FIELDS, name=f"loss-mass rank state {expected_rank}"
        )
        rank = _positive_int(rank_entry["rank"], name="loss-mass rank", allow_zero=True)
        path = _resolved_path(rank_entry["path"], name=f"loss-mass rank{rank} state")
        expected_sha = _sha256(rank_entry["sha256"], name=f"loss-mass rank{rank} SHA-256")
        if (
            rank != expected_rank
            or not path.is_file()
            or path.name != f"rank{rank}.pt"
            or sha256_file(path) != expected_sha
        ):
            raise PromotionValidationError(f"Loss-mass rank{rank} state identity differs")
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        data_loader = loaded.get("data_loader") if isinstance(loaded, Mapping) else None
        if not isinstance(data_loader, Mapping):
            raise PromotionValidationError(f"Loss-mass rank{rank} data-loader state is missing")
        packing = data_loader.get("packing_state")
        if (
            loaded.get("global_step") != candidate["global_step"]
            or loaded.get("world_size") != dp_world_size
            or data_loader.get("batches_processed") != candidate["global_step"]
            or data_loader.get("total_data_errors") != 0
            or not isinstance(packing, Mapping)
            or packing.get("version") != 5
            or packing.get("dp_rank") != rank
            or packing.get("dp_world_size") != dp_world_size
        ):
            raise PromotionValidationError(f"Loss-mass rank{rank} cursor is incompatible")
        current_fingerprints = packing.get("dataset_fingerprints")
        if rank == 0:
            dataset_fingerprints = current_fingerprints
        elif current_fingerprints != dataset_fingerprints:
            raise PromotionValidationError("Loss-mass dataset fingerprints differ across ranks")
        checkpoint_states.append({"rank": rank, "state": data_loader})
        observed_data_errors += int(data_loader["total_data_errors"])
    if (
        canonical_sha256(dataset_fingerprints) != loader["dataset_fingerprints_sha256"]
        or canonical_sha256(checkpoint_states) != loader["checkpoint_final_state_sha256"]
        or observed_data_errors != loader["total_data_errors"]
    ):
        raise PromotionValidationError("Loss-mass checkpoint cursor evidence differs")

    sources = _exact_fields(receipt["sources"], frozenset(SOURCES), name="loss-mass sources")
    targets = {"pixmo_caption": 0.7, "pixmo_transcript": 0.3}
    shares: dict[str, float] = {}
    active_shares: dict[str, float] = {}
    loss_weights: dict[str, float] = {}
    active_loss_weights: dict[str, float] = {}
    for source in SOURCES:
        metrics = _exact_fields(
            sources[source], _LOSS_SOURCE_FIELDS, name=f"loss-mass source {source}"
        )
        for count_field in ("examples", "tokens", "positive_tokens"):
            _positive_int(metrics[count_field], name=f"{source} {count_field}")
        loss_weight = _finite(metrics["loss_weight"], name=f"{source} loss weight")
        active_loss_weight = _finite(
            metrics["active_loss_weight"], name=f"{source} active loss weight"
        )
        share = _finite(metrics["loss_mass_share"], name=f"{source} loss-mass share")
        active_share = _finite(
            metrics["active_loss_mass_share"], name=f"{source} active loss-mass share"
        )
        target = _finite(metrics["target_loss_mass"], name=f"{source} target")
        absolute_error = _finite(metrics["absolute_error"], name=f"{source} absolute error")
        active_error = _finite(
            metrics["active_absolute_error"], name=f"{source} active absolute error"
        )
        if (
            loss_weight <= 0
            or active_loss_weight <= 0
            or not 0 <= share <= 1
            or not 0 <= active_share <= 1
            or target != targets[source]
            or not math.isclose(abs(share - target), absolute_error, rel_tol=0, abs_tol=1e-12)
            or not math.isclose(abs(active_share - target), active_error, rel_tol=0, abs_tol=1e-12)
            or absolute_error > tolerance
            or active_error > tolerance
        ):
            raise PromotionValidationError(f"Loss-mass source {source} fails its target")
        shares[source] = share
        active_shares[source] = active_share
        loss_weights[source] = loss_weight
        active_loss_weights[source] = active_loss_weight

    summary = _exact_fields(receipt["summary"], _LOSS_SUMMARY_FIELDS, name="loss-mass summary")
    total = _finite(summary["total_loss_weight"], name="total loss weight")
    active_total = _finite(summary["total_active_loss_weight"], name="total active loss weight")
    share_sum = _finite(summary["share_sum"], name="loss-mass share sum")
    active_share_sum = _finite(summary["active_share_sum"], name="active loss-mass share sum")
    if (
        summary["within_tolerance"] is not True
        or not math.isclose(total, sum(loss_weights.values()), rel_tol=1e-12, abs_tol=1e-9)
        or not math.isclose(
            active_total, sum(active_loss_weights.values()), rel_tol=1e-12, abs_tol=1e-9
        )
        or not math.isclose(share_sum, sum(shares.values()), rel_tol=0, abs_tol=1e-12)
        or not math.isclose(active_share_sum, sum(active_shares.values()), rel_tol=0, abs_tol=1e-12)
        or not math.isclose(share_sum, 1.0, rel_tol=0, abs_tol=1e-12)
        or not math.isclose(active_share_sum, 1.0, rel_tol=0, abs_tol=1e-12)
    ):
        raise PromotionValidationError("Loss-mass summary is inconsistent")
    return {
        "batches_replayed": loader["batches_replayed"],
        "loss_mass_share": shares,
        "active_loss_mass_share": active_shares,
    }


_GUARD_FIELDS = frozenset(
    {
        "format",
        "version",
        "status",
        "created_at",
        "candidate",
        "run",
        "rank_state_inventory",
        "permanent_checkpoints",
        "guarded_skips",
        "unexpected_guarded_skip_count",
        "evidence_artifact",
    }
)
_GUARD_RUN_FIELDS = frozenset(
    {
        "run_id",
        "global_steps",
        "exit_code",
        "rank_state_count",
        "permanent_checkpoint_steps",
        "nonfinite_metric_count",
        "unexpected_anomaly_count",
    }
)
_GUARD_SKIP_FIELDS = frozenset({"step", "count", "reason_code", "waiver_required"})
_GUARD_RANK_STATE_FIELDS = frozenset(
    {
        "rank",
        "path",
        "sha256",
        "global_step",
        "batches_processed",
        "total_data_errors",
        "run_id",
    }
)
_GUARD_CHECKPOINT_FIELDS = frozenset({"step", "path", "marker_sha256"})


def audit_optimizer_run_log(path: Path, *, expected_steps: int) -> dict[str, Any]:
    """Recompute finite-metric and guarded-skip evidence from a raw trainer output log."""
    try:
        text = path.read_text()
    except (OSError, UnicodeDecodeError) as error:
        raise PromotionValidationError(
            f"Could not read optimizer run log {path}: {error}"
        ) from error
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    current_step: int | None = None
    observed_steps: set[int] = set()
    guarded_skips: list[dict[str, Any]] = []
    numeric_metric_count = 0
    nonfinite_metric_count = 0
    for line in text.splitlines():
        match = re.search(rf"\[step=(\d+)/{expected_steps},", line)
        if match is not None:
            current_step = int(match.group(1))
            observed_steps.add(current_step)
        metric = re.match(r"\s+([^=]+)=(\S+)", line)
        if metric is None or current_step is None:
            continue
        try:
            value = float(metric.group(2).replace(",", ""))
        except ValueError:
            continue
        numeric_metric_count += 1
        if not math.isfinite(value):
            nonfinite_metric_count += 1
        if metric.group(1).strip() == "optim/step skipped" and value != 0:
            guarded_skips.append({"step": current_step, "value": value})
    if observed_steps != set(range(1, expected_steps + 1)):
        missing = sorted(set(range(1, expected_steps + 1)) - observed_steps)
        raise PromotionValidationError(
            f"Optimizer run log does not contain every step; missing {missing[:10]}"
        )
    if numeric_metric_count <= 0 or nonfinite_metric_count:
        raise PromotionValidationError(
            "Optimizer run log contains no metrics or non-finite metrics"
        )
    anomaly_patterns = (
        "Traceback (most recent call last):",
        " CRITICAL ",
        "Unhandled exception",
        "NCCL watchdog caught collective operation timeout",
    )
    unexpected_anomaly_count = sum(text.count(pattern) for pattern in anomaly_patterns)
    if unexpected_anomaly_count:
        raise PromotionValidationError("Optimizer run log contains unexpected fatal anomalies")
    if "Finalizing successful W&B run" not in text:
        raise PromotionValidationError("Optimizer run log lacks the successful-run terminal marker")
    return {
        "numeric_metric_count": numeric_metric_count,
        "nonfinite_metric_count": 0,
        "unexpected_anomaly_count": 0,
        "guarded_skips": guarded_skips,
    }


def validate_optimizer_guard_receipt(
    receipt: Mapping[str, Any], *, candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the one known safety-guard skip and reject any unrecorded skip."""
    _validate_receipt_header(
        receipt,
        expected_format=OPTIMIZER_GUARD_RECEIPT_FORMAT,
        expected_fields=_GUARD_FIELDS,
        name="optimizer-guard receipt",
    )
    _validate_receipt_candidate(
        receipt["candidate"], expected=candidate, name="optimizer-guard candidate"
    )
    run = _exact_fields(receipt["run"], _GUARD_RUN_FIELDS, name="optimizer-guard run")
    if not isinstance(run["run_id"], str) or not run["run_id"]:
        raise PromotionValidationError("Optimizer-guard run ID must be non-empty")
    _positive_int(run["rank_state_count"], name="optimizer-guard rank state count")
    if (
        run["global_steps"] != candidate["global_step"]
        or run["exit_code"] != 0
        or run["permanent_checkpoint_steps"] != [0, 100, 200, 250, 300, 400, 500]
        or run["nonfinite_metric_count"] != 0
        or run["unexpected_anomaly_count"] != 0
    ):
        raise PromotionValidationError("Optimizer-guard run did not complete the candidate step")
    rank_states = receipt["rank_state_inventory"]
    if not isinstance(rank_states, list) or len(rank_states) != run["rank_state_count"]:
        raise PromotionValidationError("Optimizer-guard rank-state inventory is incomplete")
    try:
        import torch
    except ImportError as error:  # pragma: no cover - OLMo-core always requires torch.
        raise PromotionValidationError(
            "PyTorch is required to audit trainer rank states"
        ) from error
    observed_ranks: list[int] = []
    for index, raw_state in enumerate(rank_states):
        state = _exact_fields(
            raw_state, _GUARD_RANK_STATE_FIELDS, name=f"optimizer rank state {index}"
        )
        rank = _positive_int(state["rank"], name="optimizer rank", allow_zero=True)
        path = _resolved_path(state["path"], name=f"optimizer rank{rank} state")
        expected_sha = _sha256(state["sha256"], name=f"optimizer rank{rank} SHA-256")
        if not path.is_file() or path.name != f"rank{rank}.pt" or sha256_file(path) != expected_sha:
            raise PromotionValidationError(f"Optimizer rank{rank} state identity differs")
        # These tiny files are authored by the selected, locally trusted parent checkpoint. Their
        # RNG payload predates weights_only serialization, so an exact audit necessarily uses the
        # native trusted checkpoint loader.
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        data_loader = loaded.get("data_loader") if isinstance(loaded, Mapping) else None
        callbacks = loaded.get("callbacks") if isinstance(loaded, Mapping) else None
        wandb = callbacks.get("wandb") if isinstance(callbacks, Mapping) else None
        if (
            not isinstance(data_loader, Mapping)
            or not isinstance(wandb, Mapping)
            or loaded.get("global_step") != state["global_step"]
            or data_loader.get("batches_processed") != state["batches_processed"]
            or data_loader.get("total_data_errors") != state["total_data_errors"]
            or wandb.get("run_id") != state["run_id"]
            or state["global_step"] != candidate["global_step"]
            or state["batches_processed"] != candidate["global_step"]
            or state["total_data_errors"] != 0
            or (rank == 0 and state["run_id"] != run["run_id"])
            or (rank != 0 and state["run_id"] is not None)
        ):
            raise PromotionValidationError(f"Optimizer rank{rank} state is incompatible")
        observed_ranks.append(rank)
    if observed_ranks != list(range(run["rank_state_count"])):
        raise PromotionValidationError("Optimizer rank-state inventory is not contiguous")

    checkpoints = receipt["permanent_checkpoints"]
    if (
        not isinstance(checkpoints, list)
        or [entry.get("step") for entry in checkpoints] != run["permanent_checkpoint_steps"]
    ):
        raise PromotionValidationError("Optimizer permanent checkpoint inventory differs")
    for index, raw_checkpoint in enumerate(checkpoints):
        checkpoint = _exact_fields(
            raw_checkpoint,
            _GUARD_CHECKPOINT_FIELDS,
            name=f"optimizer permanent checkpoint {index}",
        )
        step = checkpoint["step"]
        path = _resolved_path(checkpoint["path"], name=f"permanent step{step}")
        marker_path = path / ".metadata.json"
        marker_sha = _sha256(
            checkpoint["marker_sha256"], name=f"permanent step{step} marker SHA-256"
        )
        marker = load_json(marker_path)
        if (
            not path.is_dir()
            or path.name != f"step{step}"
            or sha256_file(marker_path) != marker_sha
            or not isinstance(marker, Mapping)
            or marker.get("ephemeral") is not False
        ):
            raise PromotionValidationError(f"Permanent step{step} checkpoint identity differs")
    skips = receipt["guarded_skips"]
    if not isinstance(skips, list) or len(skips) != 1:
        raise PromotionValidationError("Exactly one optimizer guard skip must be recorded")
    skip = _exact_fields(skips[0], _GUARD_SKIP_FIELDS, name="optimizer guard skip")
    if (
        skip["step"] != 356
        or skip["count"] != 1
        or skip["reason_code"] != "optimizer_safety_guard"
        or skip["waiver_required"] is not True
        or receipt["unexpected_guarded_skip_count"] != 0
    ):
        raise PromotionValidationError(
            "Optimizer guard receipt differs from the known step356 skip"
        )
    evidence_path = _validate_raw_reference(
        receipt["evidence_artifact"], name="optimizer-guard source evidence"
    )
    log_audit = audit_optimizer_run_log(evidence_path, expected_steps=candidate["global_step"])
    if log_audit["guarded_skips"] != [{"step": 356, "value": 1.0}]:
        raise PromotionValidationError("Optimizer run log differs from the one guarded skip")
    if (
        log_audit["nonfinite_metric_count"] != run["nonfinite_metric_count"]
        or log_audit["unexpected_anomaly_count"] != run["unexpected_anomaly_count"]
    ):
        raise PromotionValidationError("Optimizer run-health claims differ from the pinned log")
    return {"run_id": run["run_id"], "guarded_skip_step": 356, "evidence": str(evidence_path)}


def build_optimizer_guard_receipt(
    *,
    candidate: Mapping[str, Any],
    output_log: Path,
    expected_output_log_sha256: str,
    created_at: str,
) -> dict[str, Any]:
    """Build the run-health receipt from checkpoint rank states and the pinned trainer log."""
    _timestamp(created_at, name="optimizer-guard receipt created_at")
    checkpoint = _resolved_path(candidate["checkpoint"], name="optimizer candidate checkpoint")
    expected_log_sha = _sha256(expected_output_log_sha256, name="optimizer output log SHA-256")
    output_log = output_log.expanduser().resolve()
    if not output_log.is_file() or sha256_file(output_log) != expected_log_sha:
        raise PromotionValidationError("Optimizer output log differs from its explicit pin")
    log_audit = audit_optimizer_run_log(output_log, expected_steps=candidate["global_step"])
    if log_audit["guarded_skips"] != [{"step": 356, "value": 1.0}]:
        raise PromotionValidationError("Run health requires exactly one guarded skip at step356")

    train_dir = checkpoint / "train"
    rank_paths: list[tuple[int, Path]] = []
    for path in train_dir.glob("rank*.pt"):
        match = re.fullmatch(r"rank(\d+)\.pt", path.name)
        if match is not None:
            rank_paths.append((int(match.group(1)), path.resolve()))
    rank_paths.sort()
    if not rank_paths or [rank for rank, _ in rank_paths] != list(range(len(rank_paths))):
        raise PromotionValidationError("Candidate rank-state files are missing or non-contiguous")
    try:
        import torch
    except ImportError as error:  # pragma: no cover
        raise PromotionValidationError("PyTorch is required to read trainer rank states") from error
    rank_inventory: list[dict[str, Any]] = []
    run_ids: set[str] = set()
    for rank, path in rank_paths:
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        data_loader = loaded.get("data_loader") if isinstance(loaded, Mapping) else None
        callbacks = loaded.get("callbacks") if isinstance(loaded, Mapping) else None
        wandb = callbacks.get("wandb") if isinstance(callbacks, Mapping) else None
        if not isinstance(data_loader, Mapping) or not isinstance(wandb, Mapping):
            raise PromotionValidationError(f"Rank{rank} trainer state is incomplete")
        run_id = wandb.get("run_id")
        if rank == 0:
            if not isinstance(run_id, str) or not run_id:
                raise PromotionValidationError("Rank0 trainer state lacks its run ID")
            run_ids.add(run_id)
        elif run_id is not None:
            raise PromotionValidationError(f"Non-leader rank{rank} unexpectedly owns a run ID")
        rank_inventory.append(
            {
                "rank": rank,
                "path": str(path),
                "sha256": sha256_file(path),
                "global_step": loaded.get("global_step"),
                "batches_processed": data_loader.get("batches_processed"),
                "total_data_errors": data_loader.get("total_data_errors"),
                "run_id": run_id,
            }
        )
    if len(run_ids) != 1:
        raise PromotionValidationError("Candidate trainer rank states disagree on run ID")

    permanent_steps = [0, 100, 200, 250, 300, 400, 500]
    permanent_checkpoints: list[dict[str, Any]] = []
    for step in permanent_steps:
        path = checkpoint.parent / f"step{step}"
        marker_path = path / ".metadata.json"
        if not marker_path.is_file():
            raise PromotionValidationError(f"Permanent step{step} marker is missing")
        marker = load_json(marker_path)
        if not isinstance(marker, Mapping) or marker.get("ephemeral") is not False:
            raise PromotionValidationError(f"Step{step} is not a permanent checkpoint")
        permanent_checkpoints.append(
            {
                "step": step,
                "path": str(path.resolve()),
                "marker_sha256": sha256_file(marker_path),
            }
        )

    receipt: dict[str, Any] = {
        "format": OPTIMIZER_GUARD_RECEIPT_FORMAT,
        "version": RECEIPT_VERSION,
        "status": "passed",
        "created_at": created_at,
        "candidate": {
            "checkpoint": candidate["checkpoint"],
            "global_step": candidate["global_step"],
            "checkpoint_config_sha256": candidate["checkpoint_config_sha256"],
            "checkpoint_identity_sha256": candidate["checkpoint_identity_sha256"],
        },
        "run": {
            "run_id": next(iter(run_ids)),
            "global_steps": candidate["global_step"],
            "exit_code": 0,
            "rank_state_count": len(rank_inventory),
            "permanent_checkpoint_steps": permanent_steps,
            "nonfinite_metric_count": log_audit["nonfinite_metric_count"],
            "unexpected_anomaly_count": log_audit["unexpected_anomaly_count"],
        },
        "rank_state_inventory": rank_inventory,
        "permanent_checkpoints": permanent_checkpoints,
        "guarded_skips": [
            {
                "step": 356,
                "count": 1,
                "reason_code": "optimizer_safety_guard",
                "waiver_required": True,
            }
        ],
        "unexpected_guarded_skip_count": 0,
        "evidence_artifact": artifact_reference(output_log),
    }
    validate_optimizer_guard_receipt(receipt, candidate=candidate)
    return receipt


def _validate_checkpoint_identity(checkpoint: Any, *, name: str) -> Mapping[str, Any]:
    required = {
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
    identity = _exact_fields(checkpoint, frozenset(required), name=f"{name} checkpoint")
    for field in (
        "config_sha256",
        "checkpoint_marker_sha256",
        "dcp_metadata_sha256",
        "state_file_inventory_sha256",
        "identity_sha256",
    ):
        _sha256(identity[field], name=f"{name} checkpoint {field}")
    if identity["state_file_hash_algorithm"] != "sha256":
        raise PromotionValidationError(f"{name} checkpoint does not use SHA-256 inventory")
    root = _resolved_path(identity["root"], name=f"{name} checkpoint root", must_exist=False)
    state_dir = _resolved_path(
        identity["state_dir"], name=f"{name} checkpoint state directory", must_exist=False
    )
    if state_dir != root / "model_and_optim":
        raise PromotionValidationError(f"{name} checkpoint state directory is incompatible")
    inventory = identity["state_file_inventory"]
    if not isinstance(inventory, list) or not inventory:
        raise PromotionValidationError(f"{name} checkpoint state inventory is empty")
    inventory_paths: list[str] = []
    for index, raw_item in enumerate(inventory):
        item = _exact_fields(
            raw_item,
            frozenset({"path", "size", "sha256"}),
            name=f"{name} checkpoint state file {index}",
        )
        relative = item["path"]
        if not isinstance(relative, str) or not relative:
            raise PromotionValidationError(f"{name} checkpoint state path is invalid")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise PromotionValidationError(f"{name} checkpoint state path escapes its root")
        if relative_path.parts[:1] != ("model_and_optim",):
            raise PromotionValidationError(f"{name} checkpoint state path is outside DCP state")
        _positive_int(item["size"], name=f"{name} checkpoint state size", allow_zero=True)
        _sha256(item["sha256"], name=f"{name} checkpoint state SHA-256")
        inventory_paths.append(relative_path.as_posix())
    if inventory_paths != sorted(inventory_paths) or len(set(inventory_paths)) != len(
        inventory_paths
    ):
        raise PromotionValidationError(f"{name} checkpoint state inventory is not canonical")
    if canonical_sha256(inventory) != identity["state_file_inventory_sha256"]:
        raise PromotionValidationError(f"{name} checkpoint state inventory SHA-256 differs")
    without_identity = {key: value for key, value in identity.items() if key != "identity_sha256"}
    if canonical_sha256(without_identity) != identity["identity_sha256"]:
        raise PromotionValidationError(f"{name} checkpoint identity SHA-256 differs")
    return identity


def _validate_live_checkpoint_identity(
    identity: Mapping[str, Any], *, name: str, hash_workers: int = 16
) -> None:
    """Re-hash the live DCP inventory bound by a matched-wrong checkpoint receipt."""
    root = Path(str(identity["root"])).expanduser().resolve()
    state_dir = Path(str(identity["state_dir"])).expanduser().resolve()
    config_path = root / "config.json"
    marker_path = root / ".metadata.json"
    dcp_metadata_path = state_dir / ".metadata"
    for path, expected, label in (
        (config_path, identity["config_sha256"], "config"),
        (marker_path, identity["checkpoint_marker_sha256"], "checkpoint marker"),
        (dcp_metadata_path, identity["dcp_metadata_sha256"], "DCP metadata"),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise PromotionValidationError(f"Live {name} {label} differs from its receipt")
    state_files = sorted(path for path in state_dir.iterdir() if path.is_file())
    if not state_files:
        raise PromotionValidationError(f"Live {name} DCP state is empty")
    with ThreadPoolExecutor(max_workers=min(hash_workers, len(state_files))) as executor:
        hashes = list(executor.map(sha256_file, state_files))
    actual_inventory = [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": digest,
        }
        for path, digest in zip(state_files, hashes, strict=True)
    ]
    if actual_inventory != identity["state_file_inventory"]:
        raise PromotionValidationError(f"Live {name} DCP shard inventory differs from its receipt")
    if canonical_sha256(actual_inventory) != identity["state_file_inventory_sha256"]:
        raise PromotionValidationError(f"Live {name} DCP inventory identity differs")


def _validate_metric_window(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PromotionValidationError(f"{name} must be an object")
    required = {
        "correct_ce_mean",
        "wrong_ce_mean",
        "gap_wrong_minus_correct_mean",
        "win_rate",
        "mean_gap_bootstrap_ci",
    }
    if not required <= set(value):
        raise PromotionValidationError(f"{name} omits required metrics")
    for field in ("correct_ce_mean", "wrong_ce_mean", "gap_wrong_minus_correct_mean", "win_rate"):
        _finite(value[field], name=f"{name} {field}")
    ci = value["mean_gap_bootstrap_ci"]
    if not isinstance(ci, Mapping) or not {"low", "high"} <= set(ci):
        raise PromotionValidationError(f"{name} confidence interval is incomplete")
    low = _finite(ci["low"], name=f"{name} CI low")
    high = _finite(ci["high"], name=f"{name} CI high")
    if low > high:
        raise PromotionValidationError(f"{name} confidence interval is reversed")
    return value


def _validate_matched_evaluator(value: Any, *, name: str, verify_live: bool) -> dict[str, str]:
    evaluator = _exact_fields(
        value,
        frozenset(
            {
                "path",
                "sha256",
                "pairing_implementation_path",
                "pairing_implementation_sha256",
            }
        ),
        name=f"{name} evaluator",
    )
    evaluator_path = _resolved_path(
        evaluator["path"], name=f"{name} evaluator path", must_exist=verify_live
    )
    pairing_path = _resolved_path(
        evaluator["pairing_implementation_path"],
        name=f"{name} pairing implementation path",
        must_exist=verify_live,
    )
    evaluator_sha = _sha256(evaluator["sha256"], name=f"{name} evaluator SHA-256")
    pairing_sha = _sha256(
        evaluator["pairing_implementation_sha256"],
        name=f"{name} pairing implementation SHA-256",
    )
    if (
        evaluator_path.name != "vision_alignment_matched_wrong.py"
        or pairing_path.name != "matched_wrong_image.py"
    ):
        raise PromotionValidationError(f"{name} names incompatible evaluator implementations")
    if verify_live and (
        sha256_file(evaluator_path) != evaluator_sha or sha256_file(pairing_path) != pairing_sha
    ):
        raise PromotionValidationError(f"{name} evaluator implementation differs from its pin")
    return {
        "path": str(evaluator_path),
        "sha256": evaluator_sha,
        "pairing_implementation_path": str(pairing_path),
        "pairing_implementation_sha256": pairing_sha,
    }


def _validate_matched_receipt(
    receipt: Mapping[str, Any], *, name: str, verify_live_evaluator: bool = False
) -> dict[str, Any]:
    required_top = {
        "schema_version",
        "checkpoint",
        "native_checkpoint_load",
        "validation",
        "pairings",
        "artifact_policy",
        "evaluator",
        "protocol",
        "results",
    }
    if not required_top <= set(receipt):
        raise PromotionValidationError(f"{name} omits required v3 matched-wrong fields")
    if receipt["schema_version"] != 3:
        raise PromotionValidationError(f"{name} is not a v3 matched-wrong receipt")
    _validate_matched_evaluator(receipt["evaluator"], name=name, verify_live=verify_live_evaluator)
    checkpoint = _validate_checkpoint_identity(receipt["checkpoint"], name=name)
    native = receipt["native_checkpoint_load"]
    if (
        not isinstance(native, Mapping)
        or native.get("complete") is not True
        or native.get("load_completed") is not True
    ):
        raise PromotionValidationError(f"{name} lacks a completed native checkpoint load")
    if (
        native.get("model_parameter_count") != native.get("model_parameter_checkpoint_key_count")
        or not isinstance(native.get("model_parameter_count"), int)
        or native["model_parameter_count"] <= 0
    ):
        raise PromotionValidationError(f"{name} native checkpoint load is incomplete")
    protocol = receipt["protocol"]
    if not isinstance(protocol, Mapping) or protocol.get("name") != (
        "vision-alignment-native-matched-wrong-image-v3"
    ):
        raise PromotionValidationError(f"{name} protocol is incompatible")
    if protocol.get("evaluation_population") != "matched_eligible_validation_subset":
        raise PromotionValidationError(f"{name} does not label the matched-eligible population")
    policy = receipt["artifact_policy"]
    if not isinstance(policy, Mapping) or policy.get("output_overwrite_enabled") is not False:
        raise PromotionValidationError(f"{name} is not an immutable evaluation receipt")

    pairings = _exact_fields(receipt["pairings"], frozenset(SOURCES), name=f"{name} pairings")
    results = _exact_fields(receipt["results"], frozenset(SOURCES), name=f"{name} results")
    pairing_payloads: dict[str, Mapping[str, Any]] = {}
    for source in SOURCES:
        pairing_meta = pairings[source]
        if not isinstance(pairing_meta, Mapping):
            raise PromotionValidationError(f"{name} {source} pairing metadata is invalid")
        path = _resolved_path(pairing_meta.get("path"), name=f"{name} {source} pairing path")
        expected_sha = _sha256(pairing_meta.get("sha256"), name=f"{name} {source} pairing SHA-256")
        if sha256_file(path) != expected_sha:
            raise PromotionValidationError(f"{name} {source} pairing raw SHA-256 differs")
        pairing = load_json(path)
        if not isinstance(pairing, Mapping):
            raise PromotionValidationError(f"{name} {source} pairing must be an object")
        validate_matched_wrong_image_pairing(pairing)
        if matched_wrong_image_pairing_sha256(pairing) != expected_sha:
            raise PromotionValidationError(f"{name} {source} canonical pairing SHA-256 differs")
        if protocol.get("pairing_sha256", {}).get(source) != expected_sha:
            raise PromotionValidationError(f"{name} protocol does not bind {source} pairing")
        result = results[source]
        if not isinstance(result, Mapping) or result.get("pairing_sha256") != expected_sha:
            raise PromotionValidationError(f"{name} result does not bind {source} pairing")
        if result.get("population") != "matched_eligible_validation_subset":
            raise PromotionValidationError(f"{name} {source} result population is incompatible")
        if result.get("coverage") != pairing.get("coverage"):
            raise PromotionValidationError(f"{name} {source} pairing coverage differs")
        metrics = result.get("metrics")
        if not isinstance(metrics, Mapping):
            raise PromotionValidationError(f"{name} {source} metrics are missing")
        for window in PRIMARY_WINDOWS:
            _validate_metric_window(metrics.get(window), name=f"{name} {source} {window}")
        pairing_payloads[source] = pairing
    return {"checkpoint": checkpoint, "pairings": pairing_payloads, "receipt": receipt}


def _window(receipt: Mapping[str, Any], source: str, window: str) -> Mapping[str, Any]:
    return receipt["results"][source]["metrics"][window]


def _checkpoint_step(identity: Mapping[str, Any], *, expected: int, name: str) -> None:
    root = Path(str(identity["root"])).resolve()
    if root.name != f"step{expected}":
        raise PromotionValidationError(f"{name} checkpoint must be step{expected}, got {root.name}")


def _pairing_population(pairing: Mapping[str, Any]) -> tuple[set[int], set[str]]:
    indices = {
        int(index) for pair in pairing["pairs"] for index in (pair["recipient"], pair["donor"])
    }
    by_index = {int(row["index"]): str(row["content_id"]) for row in pairing["rows"]}
    return indices, {by_index[index] for index in indices}


def _validate_matched_set(
    matched: Mapping[str, dict[str, Any]], *, checkpoint: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    canary250 = matched["canary_step250"]
    bridge250 = matched["bridge_step250"]
    bridge500 = matched["bridge_step500"]
    independent0 = matched["independent_step0"]
    independent500 = matched["independent_step500"]
    _checkpoint_step(canary250["checkpoint"], expected=250, name="canary step250")
    _checkpoint_step(bridge250["checkpoint"], expected=250, name="bridge step250")
    _checkpoint_step(bridge500["checkpoint"], expected=500, name="bridge step500")
    _checkpoint_step(independent0["checkpoint"], expected=0, name="independent step0")
    _checkpoint_step(independent500["checkpoint"], expected=500, name="independent step500")
    if Path(str(bridge500["checkpoint"]["root"])).resolve() != checkpoint:
        raise PromotionValidationError("Primary step500 receipt names a different checkpoint")
    if (
        independent500["checkpoint"]["identity_sha256"]
        != bridge500["checkpoint"]["identity_sha256"]
    ):
        raise PromotionValidationError(
            "Independent step500 did not evaluate the candidate identity"
        )
    if Path(str(independent0["checkpoint"]["root"])).resolve().parent != checkpoint.parent:
        raise PromotionValidationError("Independent step0 is not from the candidate lineage")

    primary_pairing_sha = bridge500["receipt"]["protocol"]["pairing_sha256"]
    if bridge250["receipt"]["protocol"]["pairing_sha256"] != primary_pairing_sha:
        raise PromotionValidationError("Primary step250 and step500 pairings differ")
    independent_pairing_sha = independent500["receipt"]["protocol"]["pairing_sha256"]
    if independent0["receipt"]["protocol"]["pairing_sha256"] != independent_pairing_sha:
        raise PromotionValidationError("Independent step0 and step500 pairings differ")
    primary_seed = bridge500["receipt"]["protocol"].get("pairing_seed")
    independent_seed = independent0["receipt"]["protocol"].get("pairing_seed")
    if (
        isinstance(primary_seed, bool)
        or not isinstance(primary_seed, int)
        or independent_seed != primary_seed + INDEPENDENT_PAIRING_SEED_OFFSET
        or independent500["receipt"]["protocol"].get("pairing_seed") != independent_seed
    ):
        raise PromotionValidationError("Independent pairing seed differs from the locked offset")
    primary_bootstrap_seed = bridge500["receipt"]["protocol"].get("bootstrap", {}).get("seed")
    independent_bootstrap_seed = (
        independent0["receipt"]["protocol"].get("bootstrap", {}).get("seed")
    )
    if (
        primary_bootstrap_seed != primary_seed + INDEPENDENT_PAIRING_SEED_OFFSET
        or any(
            matched[role]["receipt"]["protocol"].get("bootstrap", {}).get("seed")
            != primary_bootstrap_seed
            for role in ("canary_step250", "bridge_step250")
        )
        or independent_bootstrap_seed != independent_seed + INDEPENDENT_PAIRING_SEED_OFFSET
        or independent500["receipt"]["protocol"].get("bootstrap", {}).get("seed")
        != independent_bootstrap_seed
    ):
        raise PromotionValidationError(
            "Matched-wrong bootstrap seeds differ from the locked offsets"
        )

    for role in ("bridge_step250", "bridge_step500", "independent_step0", "independent_step500"):
        validation = matched[role]["receipt"]["validation"]
        reference_validation = bridge500["receipt"]["validation"]
        for field in ("manifest_sha256", "row_content_sha256"):
            if not isinstance(validation, Mapping) or validation.get(
                field
            ) != reference_validation.get(field):
                raise PromotionValidationError(
                    f"{role} does not use the primary candidate validation population"
                )

    disjointness: dict[str, Any] = {}
    for source in SOURCES:
        primary_indices, primary_content = _pairing_population(bridge500["pairings"][source])
        independent_indices, independent_content = _pairing_population(
            independent500["pairings"][source]
        )
        index_overlap = primary_indices & independent_indices
        content_overlap = primary_content & independent_content
        if index_overlap or content_overlap:
            raise PromotionValidationError(
                f"Independent {source} pairing overlaps the primary recipient/donor population"
            )
        primary_metadata = bridge500["receipt"]["pairings"][source]
        independent0_metadata = independent0["receipt"]["pairings"][source]
        independent500_metadata = independent500["receipt"]["pairings"][source]
        exclusions = [
            independent0_metadata.get("excluded_primary_pairing"),
            independent500_metadata.get("excluded_primary_pairing"),
        ]
        expected_exclusion_pins = [
            independent0["receipt"]["artifact_policy"].get("expected_excluded_pairing_sha256"),
            independent500["receipt"]["artifact_policy"].get("expected_excluded_pairing_sha256"),
        ]
        if (
            independent0_metadata.get("provenance") != "built"
            or independent500_metadata.get("provenance") != "loaded"
            or any(not isinstance(exclusion, Mapping) for exclusion in exclusions)
            or any(
                Path(str(exclusion.get("path"))).expanduser().resolve()
                != Path(str(primary_metadata.get("path"))).expanduser().resolve()
                or exclusion.get("sha256") != primary_pairing_sha[source]
                or exclusion.get("excluded_recipient_and_donor_count") != len(primary_indices)
                for exclusion in exclusions
            )
            or any(
                not isinstance(pins, Mapping) or pins.get(source) != primary_pairing_sha[source]
                for pins in expected_exclusion_pins
            )
        ):
            raise PromotionValidationError(
                f"Independent {source} pairing lacks pinned primary-exclusion provenance"
            )
        disjointness[source] = {
            "primary_row_count": len(primary_indices),
            "independent_row_count": len(independent_indices),
            "row_index_overlap_count": 0,
            "content_id_overlap_count": 0,
            "primary_pairing_sha256": primary_pairing_sha[source],
            "independent_pairing_sha256": independent_pairing_sha[source],
        }

    failed_step250: list[tuple[str, str, float, float]] = []
    for source in SOURCES:
        for window in PRIMARY_WINDOWS:
            observed = _finite(
                _window(bridge250["receipt"], source, window)["gap_wrong_minus_correct_mean"],
                name=f"bridge step250 {source} {window} gap",
            )
            required = 0.9 * _finite(
                _window(canary250["receipt"], source, window)["gap_wrong_minus_correct_mean"],
                name=f"canary step250 {source} {window} gap",
            )
            if observed < required:
                failed_step250.append((source, window, observed, required))
            if (
                _finite(
                    _window(bridge250["receipt"], source, window)["mean_gap_bootstrap_ci"]["low"],
                    name=f"bridge step250 {source} {window} CI low",
                )
                <= 0
            ):
                raise PromotionValidationError("Bridge step250 has a non-positive primary lower CI")
        bridge_ce = _finite(
            _window(bridge250["receipt"], source, "all")["correct_ce_mean"],
            name=f"bridge step250 {source} CE",
        )
        canary_ce = _finite(
            _window(canary250["receipt"], source, "all")["correct_ce_mean"],
            name=f"canary step250 {source} CE",
        )
        if bridge_ce > 1.02 * canary_ce:
            raise PromotionValidationError("Bridge step250 correct CE exceeds the canary bound")
    if len(failed_step250) != 1 or failed_step250[0][:2] != (
        "pixmo_caption",
        "first_32",
    ):
        raise PromotionValidationError(
            "Bridge step250 deviations differ from the one preregistered threshold miss"
        )
    source, window, observed, required = failed_step250[0]
    step250_deviation: dict[str, Any] = {
        "id": STEP250_WAIVER_ID,
        "waiver_required": True,
        "source": source,
        "window": window,
        "criterion": "bridge_step250_gap_at_least_90pct_canary_step250",
        "observed": observed,
        "required": required,
        "deficit": required - observed,
        "retention_fraction": observed / (required / 0.9),
    }
    step250_deviation["sha256"] = canonical_sha256(step250_deviation)

    for source in SOURCES:
        for window in PRIMARY_WINDOWS:
            primary500_window = _window(bridge500["receipt"], source, window)
            if (
                _finite(
                    primary500_window["mean_gap_bootstrap_ci"]["low"],
                    name=f"bridge step500 {source} {window} CI low",
                )
                <= 0
            ):
                raise PromotionValidationError("Bridge step500 has a non-positive primary lower CI")
        for window in RETENTION_WINDOWS:
            observed = _finite(
                _window(bridge500["receipt"], source, window)["gap_wrong_minus_correct_mean"],
                name=f"bridge step500 {source} {window} gap",
            )
            reference = _finite(
                _window(bridge250["receipt"], source, window)["gap_wrong_minus_correct_mean"],
                name=f"bridge step250 {source} {window} gap",
            )
            if observed < 0.8 * reference:
                raise PromotionValidationError("Bridge step500 fails the extension retention gate")
        step500_ce = _finite(
            _window(bridge500["receipt"], source, "all")["correct_ce_mean"],
            name=f"bridge step500 {source} CE",
        )
        canary_ce = _finite(
            _window(canary250["receipt"], source, "all")["correct_ce_mean"],
            name=f"canary step250 {source} CE",
        )
        if step500_ce > 1.02 * canary_ce:
            raise PromotionValidationError("Bridge step500 correct CE exceeds the canary bound")

        for window in PRIMARY_WINDOWS:
            independent0_window = _window(independent0["receipt"], source, window)
            low = _finite(
                independent0_window["mean_gap_bootstrap_ci"]["low"],
                name=f"independent step0 {source} {window} CI low",
            )
            high = _finite(
                independent0_window["mean_gap_bootstrap_ci"]["high"],
                name=f"independent step0 {source} {window} CI high",
            )
            if not low <= 0 <= high:
                raise PromotionValidationError("Independent step0 does not reproduce the null")
            independent500_window = _window(independent500["receipt"], source, window)
            if (
                _finite(
                    independent500_window["mean_gap_bootstrap_ci"]["low"],
                    name=f"independent step500 {source} {window} CI low",
                )
                <= 0
            ):
                raise PromotionValidationError("Independent step500 has a non-positive lower CI")
        for window in RETENTION_WINDOWS:
            independent_gap = _finite(
                _window(independent500["receipt"], source, window)["gap_wrong_minus_correct_mean"],
                name=f"independent step500 {source} {window} gap",
            )
            primary_gap = _finite(
                _window(bridge500["receipt"], source, window)["gap_wrong_minus_correct_mean"],
                name=f"primary step500 {source} {window} gap",
            )
            if independent_gap < 0.8 * primary_gap:
                raise PromotionValidationError("Independent step500 fails the 80% replication gate")
        independent_ce = _finite(
            _window(independent500["receipt"], source, "all")["correct_ce_mean"],
            name=f"independent step500 {source} CE",
        )
        primary_ce = _finite(
            _window(bridge500["receipt"], source, "all")["correct_ce_mean"],
            name=f"primary step500 {source} CE",
        )
        if independent_ce > 1.02 * primary_ce:
            raise PromotionValidationError("Independent step500 correct CE exceeds the +2% bound")

    summary = {
        "primary_checkpoint_identity_sha256": bridge500["checkpoint"]["identity_sha256"],
        "pairing_disjointness": disjointness,
        "step500_primary_lower_ci_positive": True,
        "independent_step0_null_reproduced": True,
        "independent_step500_lower_ci_positive": True,
        "independent_step500_replication_fraction": 0.8,
    }
    return summary, [step250_deviation]


def _candidate_from_primary(
    checkpoint: Path,
    receipt: Mapping[str, Any],
    *,
    verify_live_contents: bool = True,
) -> dict[str, Any]:
    identity = _validate_checkpoint_identity(receipt["checkpoint"], name="primary step500")
    root = Path(str(identity["root"])).expanduser().resolve()
    if root != checkpoint or root.name != "step500":
        raise PromotionValidationError("Promotion candidate must be the primary step500 checkpoint")
    if verify_live_contents:
        _validate_live_checkpoint_identity(identity, name="primary step500")
    config_path = root / "config.json"
    marker_path = root / ".metadata.json"
    dcp_metadata_path = root / "model_and_optim" / ".metadata"
    for path, expected, label in (
        (config_path, identity["config_sha256"], "config"),
        (marker_path, identity["checkpoint_marker_sha256"], "marker"),
        (dcp_metadata_path, identity["dcp_metadata_sha256"], "DCP metadata"),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise PromotionValidationError(f"Live candidate {label} differs from the v3 receipt")
    config = load_json(config_path)
    if not isinstance(config, Mapping) or not isinstance(config.get("vision_alignment"), Mapping):
        raise PromotionValidationError("Candidate config lacks vision_alignment metadata")
    metadata = config["vision_alignment"]
    if metadata.get("phase") != "bridge":
        raise PromotionValidationError("Promotion candidate must be a bridge checkpoint")
    for field in ("data_contract_sha256", "trainable_contract_sha256"):
        _sha256(metadata.get(field), name=f"candidate {field}")
    lineage = metadata.get("lineage_id")
    if not isinstance(lineage, str) or not lineage:
        raise PromotionValidationError("Candidate lineage_id must be non-empty")
    model = config.get("model")
    train_module = config.get("train_module")
    if not isinstance(model, Mapping) or not isinstance(model.get("lm"), Mapping):
        raise PromotionValidationError("Candidate config lacks its language-model vocabulary")
    vocab_size = _positive_int(model["lm"].get("vocab_size"), name="candidate vocab size")
    if not isinstance(train_module, Mapping):
        raise PromotionValidationError("Candidate config lacks its train-module contract")
    image_embedding_rows = train_module.get("train_embedding_rows")
    if image_embedding_rows != list(IMAGE_TOKEN_ROWS) or any(
        row >= vocab_size for row in image_embedding_rows
    ):
        raise PromotionValidationError("Candidate image embedding rows differ from the policy")
    return {
        "checkpoint": str(root),
        "global_step": 500,
        "phase": "bridge",
        "lineage_id": lineage,
        "checkpoint_config_sha256": identity["config_sha256"],
        "checkpoint_identity_sha256": identity["identity_sha256"],
        "checkpoint_marker_sha256": identity["checkpoint_marker_sha256"],
        "dcp_metadata_sha256": identity["dcp_metadata_sha256"],
        "state_file_inventory_sha256": identity["state_file_inventory_sha256"],
        "data_contract_sha256": metadata["data_contract_sha256"],
        "trainable_contract_sha256": metadata["trainable_contract_sha256"],
        "vocab_size": vocab_size,
        "image_embedding_rows": list(image_embedding_rows),
    }


def candidate_from_matched_receipt(
    checkpoint: Path,
    receipt: Mapping[str, Any],
    *,
    verify_live_contents: bool = True,
) -> dict[str, Any]:
    """Derive a live promotion candidate from a validated primary matched-wrong receipt.

    :param verify_live_contents: Re-hash all DCP shards. A distributed caller may disable this
        only when it immediately performs one rank-zero hash and broadcasts the exact identity.
    """
    validated = _validate_matched_receipt(receipt, name="primary step500")
    return _candidate_from_primary(
        checkpoint.expanduser().resolve(),
        validated["receipt"],
        verify_live_contents=verify_live_contents,
    )


def _guard_deviation(
    guard_summary: Mapping[str, Any], guard_reference: Mapping[str, Any]
) -> dict[str, Any]:
    deviation: dict[str, Any] = {
        "id": STEP356_WAIVER_ID,
        "waiver_required": True,
        "step": 356,
        "count": 1,
        "criterion": "no_guarded_optimizer_skip",
        "reason_code": "optimizer_safety_guard",
        "run_id": guard_summary["run_id"],
        "evidence_receipt_sha256": guard_reference["sha256"],
    }
    deviation["sha256"] = canonical_sha256(deviation)
    return deviation


def _promotion_policy() -> dict[str, Any]:
    return {
        "name": PROMOTION_POLICY,
        "primary_windows": list(PRIMARY_WINDOWS),
        "step250_canary_retention_fraction": 0.9,
        "step500_extension_retention_fraction": 0.8,
        "independent_replication_fraction": 0.8,
        "independent_pairing_seed_offset": INDEPENDENT_PAIRING_SEED_OFFSET,
        "correct_ce_max_relative_increase": 0.02,
        "loss_mass_targets": {"pixmo_caption": 0.7, "pixmo_transcript": 0.3},
        "loss_mass_absolute_tolerance": 0.02,
        "required_waiver_ids": sorted(REQUIRED_WAIVER_IDS),
    }


def build_promotion_bundle(
    *,
    checkpoint: Path,
    frozen_state: Path,
    text_retention: Path,
    cumulative_loss_mass: Path,
    optimizer_guard: Path,
    canary_step250: Path,
    bridge_step250: Path,
    bridge_step500: Path,
    independent_step0: Path,
    independent_step500: Path,
    created_at: str,
) -> dict[str, Any]:
    """Validate component receipts and build a ready-for-human-approval bundle."""
    _timestamp(created_at, name="promotion bundle created_at")
    matched_paths = {
        "canary_step250": canary_step250,
        "bridge_step250": bridge_step250,
        "bridge_step500": bridge_step500,
        "independent_step0": independent_step0,
        "independent_step500": independent_step500,
    }
    matched: dict[str, dict[str, Any]] = {}
    matched_references: dict[str, dict[str, str]] = {}
    for role, path in matched_paths.items():
        reference = artifact_reference(path)
        _, receipt_payload = _load_reference(reference, name=role)
        matched[role] = _validate_matched_receipt(
            receipt_payload, name=role, verify_live_evaluator=role.startswith("independent_")
        )
        matched_references[role] = reference

    candidate = _candidate_from_primary(
        checkpoint.expanduser().resolve(), matched["bridge_step500"]["receipt"]
    )
    matched_summary, deviations = _validate_matched_set(
        matched, checkpoint=Path(candidate["checkpoint"])
    )
    expected_frozen_count = _positive_int(
        matched["bridge_step500"]["receipt"]["native_checkpoint_load"]["frozen_state_key_count"],
        name="primary step500 frozen state key count",
    ) + _positive_int(
        matched["bridge_step500"]["receipt"]["native_checkpoint_load"]["persistent_buffer_count"],
        name="primary step500 persistent buffer count",
        allow_zero=True,
    )

    component_specs: tuple[tuple[str, Path, Callable[[Mapping[str, Any]], dict[str, Any]]], ...] = (
        (
            "frozen_state",
            frozen_state,
            lambda payload: validate_frozen_state_receipt(
                payload,
                candidate=candidate,
                expected_frozen_tensor_count=expected_frozen_count,
            ),
        ),
        (
            "text_retention",
            text_retention,
            lambda payload: validate_text_retention_receipt(payload, candidate=candidate),
        ),
        (
            "cumulative_loss_mass",
            cumulative_loss_mass,
            lambda payload: validate_loss_mass_receipt(payload, candidate=candidate),
        ),
        (
            "optimizer_guard",
            optimizer_guard,
            lambda payload: validate_optimizer_guard_receipt(payload, candidate=candidate),
        ),
    )
    component_references: dict[str, dict[str, str]] = {}
    component_summaries: dict[str, dict[str, Any]] = {}
    for name, path, validator in component_specs:
        reference = artifact_reference(path)
        _, receipt_payload = _load_reference(reference, name=name)
        component_summaries[name] = validator(receipt_payload)
        component_references[name] = reference
    step0_identity = matched["independent_step0"]["checkpoint"]
    for name in ("frozen_state", "text_retention"):
        summary = component_summaries[name]
        if (
            Path(summary["reference_checkpoint"]).resolve()
            != Path(str(step0_identity["root"])).resolve()
            or summary["reference_checkpoint_config_sha256"] != step0_identity["config_sha256"]
            or summary["reference_checkpoint_identity_sha256"] != step0_identity["identity_sha256"]
        ):
            raise PromotionValidationError(
                f"{name} reference does not match the independently evaluated step0"
            )
    deviations.append(
        _guard_deviation(
            component_summaries["optimizer_guard"], component_references["optimizer_guard"]
        )
    )
    deviations.sort(key=lambda deviation: deviation["id"])
    if {deviation["id"] for deviation in deviations} != REQUIRED_WAIVER_IDS:
        raise PromotionValidationError("Promotion bundle deviations differ from the locked set")

    bundle_payload: dict[str, Any] = {
        "format": PROMOTION_BUNDLE_FORMAT,
        "version": PROMOTION_BUNDLE_VERSION,
        "status": "ready_for_human_approval",
        "created_at": created_at,
        "policy": _promotion_policy(),
        "candidate": candidate,
        "receipts": {
            **component_references,
            "matched_wrong": matched_references,
        },
        "deviations": deviations,
        "content_sha256": "",
    }
    # Summaries are deliberately re-derived by every audit instead of becoming unaudited claims
    # inside the signed bundle. Keep the local variable alive to make that design explicit.
    _ = matched_summary, component_summaries
    bundle_payload["content_sha256"] = canonical_sha256(
        {key: value for key, value in bundle_payload.items() if key != "content_sha256"}
    )
    return bundle_payload


def validate_promotion_bundle(
    bundle: Mapping[str, Any],
    *,
    expected_checkpoint: Path | None = None,
    expected_checkpoint_config_sha256: str | None = None,
) -> dict[str, Any]:
    """Re-open and revalidate every receipt referenced by a promotion bundle.

    :param bundle: Parsed promotion bundle.
    :param expected_checkpoint: Optional live parent checkpoint required by a phase transition.
    :param expected_checkpoint_config_sha256: Optional exact live parent config SHA-256.
    :returns: A compact, re-derived audit summary including deviation identities.
    :raises PromotionValidationError: If any artifact, identity, or scientific gate differs.
    """
    _exact_fields(bundle, _BUNDLE_FIELDS, name="promotion bundle")
    if (
        bundle["format"] != PROMOTION_BUNDLE_FORMAT
        or bundle["version"] != PROMOTION_BUNDLE_VERSION
        or bundle["status"] != "ready_for_human_approval"
    ):
        raise PromotionValidationError("Promotion bundle identity or status is incompatible")
    _timestamp(bundle["created_at"], name="promotion bundle created_at")
    expected_content_sha = _sha256(bundle["content_sha256"], name="bundle content SHA-256")
    actual_content_sha = canonical_sha256(
        {key: value for key, value in bundle.items() if key != "content_sha256"}
    )
    if actual_content_sha != expected_content_sha:
        raise PromotionValidationError("Promotion bundle content SHA-256 differs")
    if bundle["policy"] != _promotion_policy():
        raise PromotionValidationError("Promotion bundle policy is incompatible")

    candidate = _exact_fields(bundle["candidate"], _CANDIDATE_FIELDS, name="bundle candidate")
    checkpoint = _resolved_path(candidate["checkpoint"], name="bundle candidate checkpoint")
    if not checkpoint.is_dir() or checkpoint.name != "step500":
        raise PromotionValidationError("Bundle candidate must be a live step500 checkpoint")
    if candidate["global_step"] != 500 or candidate["phase"] != "bridge":
        raise PromotionValidationError("Bundle candidate is not bridge step500")
    for field in (
        "checkpoint_config_sha256",
        "checkpoint_identity_sha256",
        "checkpoint_marker_sha256",
        "dcp_metadata_sha256",
        "state_file_inventory_sha256",
        "data_contract_sha256",
        "trainable_contract_sha256",
    ):
        _sha256(candidate[field], name=f"bundle candidate {field}")
    if not isinstance(candidate["lineage_id"], str) or not candidate["lineage_id"]:
        raise PromotionValidationError("Bundle candidate lineage_id must be non-empty")
    vocab_size = _positive_int(candidate["vocab_size"], name="bundle candidate vocab size")
    if candidate["image_embedding_rows"] != list(IMAGE_TOKEN_ROWS) or any(
        row >= vocab_size for row in candidate["image_embedding_rows"]
    ):
        raise PromotionValidationError("Bundle candidate image embedding rows differ")
    if expected_checkpoint is not None and checkpoint != expected_checkpoint.resolve():
        raise PromotionValidationError("Bundle candidate differs from the selected parent")
    if (
        expected_checkpoint_config_sha256 is not None
        and candidate["checkpoint_config_sha256"] != expected_checkpoint_config_sha256
    ):
        raise PromotionValidationError("Bundle candidate config SHA-256 differs from the parent")
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
            raise PromotionValidationError(f"Live candidate {label} differs from the bundle")

    receipts = _exact_fields(bundle["receipts"], _RECEIPTS_FIELDS, name="bundle receipts")
    matched_refs = _exact_fields(
        receipts["matched_wrong"], _MATCHED_RECEIPTS_FIELDS, name="matched-wrong receipts"
    )
    matched: dict[str, dict[str, Any]] = {}
    for role in sorted(_MATCHED_RECEIPTS_FIELDS):
        _, payload = _load_reference(matched_refs[role], name=role)
        matched[role] = _validate_matched_receipt(
            payload, name=role, verify_live_evaluator=role.startswith("independent_")
        )
    primary_identity = matched["bridge_step500"]["checkpoint"]["identity_sha256"]
    if primary_identity != candidate["checkpoint_identity_sha256"]:
        raise PromotionValidationError("Bundle candidate identity differs from primary receipt")
    _validate_live_checkpoint_identity(
        matched["bridge_step500"]["checkpoint"], name="primary step500"
    )
    matched_summary, expected_deviations = _validate_matched_set(matched, checkpoint=checkpoint)
    expected_frozen_count = _positive_int(
        matched["bridge_step500"]["receipt"]["native_checkpoint_load"]["frozen_state_key_count"],
        name="primary step500 frozen state key count",
    ) + _positive_int(
        matched["bridge_step500"]["receipt"]["native_checkpoint_load"]["persistent_buffer_count"],
        name="primary step500 persistent buffer count",
        allow_zero=True,
    )

    component_summaries: dict[str, dict[str, Any]] = {}
    validators: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
        "frozen_state": lambda payload: validate_frozen_state_receipt(
            payload,
            candidate=candidate,
            expected_frozen_tensor_count=expected_frozen_count,
        ),
        "text_retention": lambda payload: validate_text_retention_receipt(
            payload, candidate=candidate
        ),
        "cumulative_loss_mass": lambda payload: validate_loss_mass_receipt(
            payload, candidate=candidate
        ),
        "optimizer_guard": lambda payload: validate_optimizer_guard_receipt(
            payload, candidate=candidate
        ),
    }
    for name, validator in validators.items():
        _, payload = _load_reference(receipts[name], name=name)
        component_summaries[name] = validator(payload)
    step0_identity = matched["independent_step0"]["checkpoint"]
    for name in ("frozen_state", "text_retention"):
        summary = component_summaries[name]
        if (
            Path(summary["reference_checkpoint"]).resolve()
            != Path(str(step0_identity["root"])).resolve()
            or summary["reference_checkpoint_config_sha256"] != step0_identity["config_sha256"]
            or summary["reference_checkpoint_identity_sha256"] != step0_identity["identity_sha256"]
        ):
            raise PromotionValidationError(
                f"{name} reference does not match the independently evaluated step0"
            )
    expected_deviations.append(
        _guard_deviation(component_summaries["optimizer_guard"], receipts["optimizer_guard"])
    )
    expected_deviations.sort(key=lambda deviation: deviation["id"])
    if bundle["deviations"] != expected_deviations:
        raise PromotionValidationError("Bundle deviations do not match re-derived evidence")
    return {
        "status": "ready_for_human_approval",
        "candidate": dict(candidate),
        "component_summaries": component_summaries,
        "matched_wrong_summary": matched_summary,
        "deviation_sha256": {
            deviation["id"]: deviation["sha256"] for deviation in expected_deviations
        },
    }


__all__ = [
    "FROZEN_STATE_RECEIPT_FORMAT",
    "IMAGE_TOKEN_ROWS",
    "INDEPENDENT_PAIRING_SEED_OFFSET",
    "LOSS_MASS_RECEIPT_FORMAT",
    "OPTIMIZER_GUARD_RECEIPT_FORMAT",
    "PROMOTION_BUNDLE_FORMAT",
    "PROMOTION_BUNDLE_VERSION",
    "REQUIRED_WAIVER_IDS",
    "STEP250_WAIVER_ID",
    "STEP356_WAIVER_ID",
    "TEXT_RETENTION_RECEIPT_FORMAT",
    "TEXT_SENTINEL_FORMAT",
    "PromotionValidationError",
    "artifact_reference",
    "audit_optimizer_run_log",
    "build_optimizer_guard_receipt",
    "build_promotion_bundle",
    "build_text_sentinel",
    "candidate_from_matched_receipt",
    "canonical_sha256",
    "load_json",
    "sha256_file",
    "validate_frozen_state_receipt",
    "validate_loss_mass_receipt",
    "validate_optimizer_guard_receipt",
    "validate_promotion_bundle",
    "validate_text_retention_receipt",
    "validate_text_sentinel",
]
