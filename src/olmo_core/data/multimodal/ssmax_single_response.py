"""Deterministic single-response projection for recurrent SSMax vision phases.

Some canonical Molmo vision sources serialize several annotations for one image as isolated
``subsegment_ids`` branches.  Dense attention can consume that representation directly, but a
recurrent mixer cannot reset its state between those branches.  This module projects one branch
*after* the source/provenance selection boundary and before collation.  The original serializer
and the historical s002 path are intentionally left unchanged.

Training chooses a branch from a stable hash of source, logical sample index, and epoch so that
annotation diversity remains available across epochs.  Validation and immutable evidence always
use epoch zero, making their branch choice fixed.  The projected loss weights are rebuilt using
the exact single-annotation convention rather than rescaling the multi-branch weights.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

from olmo_core.config import Config

from .sequence_builder import ATTEND_ALL_SUBSEGMENT_ID, LOSS_TOKEN_WEIGHTINGS
from .vision_alignment_sources import (
    runtime_dataset_fingerprint,
    serialized_example_sha256,
)

__all__ = [
    "SSMAX_SINGLE_RESPONSE_PROJECTION_ALGORITHM",
    "SSMAX_SINGLE_RESPONSE_CALIBRATION_FORMAT",
    "SSMAX_SINGLE_RESPONSE_CALIBRATION_VERSION",
    "SSMAX_SINGLE_RESPONSE_PROJECTION_FORMAT",
    "SSMAX_SINGLE_RESPONSE_PROJECTION_SEED",
    "SSMAX_SINGLE_RESPONSE_PROJECTION_VERSION",
    "SSMaxSingleResponseDataset",
    "SSMaxSingleResponseProjectionConfig",
    "project_ssmax_single_response",
    "resolve_ssmax_stable_sample_index",
    "ssmax_single_response_calibration_summary",
    "ssmax_single_response_projection_contract",
    "validate_ssmax_single_response_calibration",
]


SSMAX_SINGLE_RESPONSE_PROJECTION_FORMAT = "ssmax_single_response_projection"
SSMAX_SINGLE_RESPONSE_PROJECTION_VERSION = 1
SSMAX_SINGLE_RESPONSE_PROJECTION_ALGORITHM = "sha256-source-sample-epoch-mod-branches-v1"
SSMAX_SINGLE_RESPONSE_PROJECTION_SEED = 95818
SSMAX_SINGLE_RESPONSE_CALIBRATION_FORMAT = "ssmax_single_response_loss_mass_calibration"
SSMAX_SINGLE_RESPONSE_CALIBRATION_VERSION = 1

_TOKEN_FIELDS = (
    "input_ids",
    "labels",
    "loss_masks",
    "position_ids",
    "token_type_ids",
)
_EVAL_SPLITS = frozenset({"validation", "eval", "evaluation", "test", "evidence"})


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass
class SSMaxSingleResponseProjectionConfig(Config):
    """Versioned SSMax branch-selection and recalibration contract.

    ``projected_mean_loss_weight`` is deliberately empty by default.  A production perception
    or joint profile must populate it from a deterministic projection preflight over the exact
    source population; reusing the multi-branch source-audit means is rejected by the launcher.

    :param seed: Non-negative branch-selection seed shared by paired runs.
    :param projected_mean_loss_weight: Projected mean ``sum(loss_masks)`` for every phase source.
    :param calibration_path: Immutable projection-calibration receipt path.
    :param calibration_sha256: Raw SHA-256 of the calibration receipt.
    :param format: Projection contract format.
    :param version: Projection contract schema version.
    :param algorithm: Exact deterministic branch-selection algorithm.
    """

    seed: int = SSMAX_SINGLE_RESPONSE_PROJECTION_SEED
    projected_mean_loss_weight: Dict[str, float] = field(default_factory=dict)
    calibration_path: Optional[str] = None
    calibration_sha256: Optional[str] = None
    format: str = SSMAX_SINGLE_RESPONSE_PROJECTION_FORMAT
    version: int = SSMAX_SINGLE_RESPONSE_PROJECTION_VERSION
    algorithm: str = SSMAX_SINGLE_RESPONSE_PROJECTION_ALGORITHM

    def contract(self, *, loss_token_weighting: str) -> Dict[str, Any]:
        """Return the canonical semantics that every runtime wrapper must enforce."""

        return ssmax_single_response_projection_contract(
            seed=self.seed,
            loss_token_weighting=loss_token_weighting,
            format=self.format,
            version=self.version,
            algorithm=self.algorithm,
        )


def ssmax_single_response_projection_contract(
    *,
    seed: int,
    loss_token_weighting: str,
    format: str = SSMAX_SINGLE_RESPONSE_PROJECTION_FORMAT,
    version: int = SSMAX_SINGLE_RESPONSE_PROJECTION_VERSION,
    algorithm: str = SSMAX_SINGLE_RESPONSE_PROJECTION_ALGORITHM,
) -> Dict[str, Any]:
    """Build and validate the canonical projection contract.

    :returns: A JSON-canonical mapping including its own semantic SHA-256.
    :raises ValueError: If any semantic selector differs from the supported contract.
    """

    if format != SSMAX_SINGLE_RESPONSE_PROJECTION_FORMAT:
        raise ValueError("Unsupported SSMax single-response projection format")
    if version != SSMAX_SINGLE_RESPONSE_PROJECTION_VERSION:
        raise ValueError("Unsupported SSMax single-response projection version")
    if algorithm != SSMAX_SINGLE_RESPONSE_PROJECTION_ALGORITHM:
        raise ValueError("Unsupported SSMax single-response selection algorithm")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("SSMax single-response projection seed must be non-negative")
    if loss_token_weighting not in LOSS_TOKEN_WEIGHTINGS:
        raise ValueError(f"Unsupported loss-token weighting {loss_token_weighting!r}")
    payload = {
        "format": format,
        "version": version,
        "algorithm": algorithm,
        "seed": seed,
        "training_epoch_policy": "requested-epoch",
        "evaluation_epoch_policy": "fixed-zero",
        "shared_subsegment_id": ATTEND_ALL_SUBSEGMENT_ID,
        "loss_token_weighting": loss_token_weighting,
        "positive_weight_policy": {
            "none": "one",
            "root_subsegments": "one",
            "root_subsegments_root_tokens": "two-over-sqrt-positive-target-count",
        },
    }
    return {**payload, "content_sha256": _canonical_sha256(payload)}


def _require_token_arrays(example: Mapping[str, Any]) -> int:
    missing = sorted(set(_TOKEN_FIELDS) - set(example))
    if missing:
        raise ValueError(f"SSMax single-response input lacks token fields {missing}")
    length: Optional[int] = None
    for name in _TOKEN_FIELDS:
        value = example[name]
        if not isinstance(value, np.ndarray) or value.ndim != 1:
            raise ValueError(f"SSMax single-response field {name!r} must be a rank-1 NumPy array")
        if length is None:
            length = len(value)
        elif len(value) != length:
            raise ValueError("SSMax single-response token fields have different lengths")
    assert length is not None
    if length < 1:
        raise ValueError("SSMax single-response input must contain at least one token")
    loss_masks = example["loss_masks"]
    if not np.issubdtype(loss_masks.dtype, np.floating):
        raise ValueError("SSMax single-response loss_masks must have floating dtype")
    if not np.isfinite(loss_masks).all() or np.any(loss_masks < 0):
        raise ValueError("SSMax single-response loss_masks must be finite and non-negative")
    return length


def _branch_ids(example: Mapping[str, Any], length: int) -> tuple[int, ...]:
    subsegments = example.get("subsegment_ids")
    if subsegments is None:
        return ()
    if (
        not isinstance(subsegments, np.ndarray)
        or subsegments.ndim != 1
        or len(subsegments) != length
        or not np.issubdtype(subsegments.dtype, np.integer)
    ):
        raise ValueError("SSMax subsegment_ids must be a rank-1 integer array matching tokens")
    if np.any(subsegments < 0):
        raise ValueError("SSMax subsegment_ids must be non-negative")
    supervised = example["loss_masks"] > 0
    branches = tuple(
        sorted(
            int(branch_id)
            for branch_id in np.unique(subsegments[supervised])
            if int(branch_id) != ATTEND_ALL_SUBSEGMENT_ID
        )
    )
    if not branches:
        raise ValueError("Branched SSMax example contains no supervised response branch")
    if np.any(supervised & (subsegments == ATTEND_ALL_SUBSEGMENT_ID)):
        raise ValueError("Shared SSMax prefix unexpectedly contains supervised targets")
    unexpected = set(int(value) for value in np.unique(subsegments)) - {
        ATTEND_ALL_SUBSEGMENT_ID,
        *branches,
    }
    if unexpected:
        raise ValueError(
            "SSMax example contains response branches with no surviving supervised target: "
            f"{sorted(unexpected)}"
        )
    return branches


def _selection_ordinal(
    *,
    seed: int,
    source_name: str,
    stable_sample_index: int,
    selection_epoch: int,
    branch_count: int,
) -> int:
    if branch_count <= 0:
        raise ValueError("branch_count must be positive")
    digest = hashlib.sha256(
        _canonical_bytes(
            {
                "algorithm": SSMAX_SINGLE_RESPONSE_PROJECTION_ALGORITHM,
                "seed": seed,
                "source": source_name,
                "stable_sample_index": stable_sample_index,
                "epoch": selection_epoch,
            }
        )
    ).digest()
    return int.from_bytes(digest, "big") % branch_count


def resolve_ssmax_stable_sample_index(dataset: Any, index: int) -> int:
    """Resolve a selected/wrapped dataset index to its stable underlying source index.

    Perception and joint provenance datasets expose an immutable ``indices`` selection.  Runtime
    audit wrappers may sit outside that selected dataset, so this walks only the conventional
    transparent ``dataset``/``_dataset`` links and applies every explicit index mapping it sees.

    :param dataset: Dataset at the SSMax projection boundary.
    :param index: Logical outer index requested by the data loader.
    :returns: Stable non-negative source index used by branch selection and receipts.
    """

    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("SSMax projection index must be a non-negative integer")
    stable_index = index
    current = dataset
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        indices = getattr(current, "indices", None)
        if indices is not None:
            if isinstance(indices, (str, bytes)) or not isinstance(indices, Sequence):
                raise ValueError("SSMax selected dataset indices must be a sequence")
            if stable_index >= len(indices):
                raise IndexError(stable_index)
            mapped = indices[stable_index]
            if isinstance(mapped, bool) or not isinstance(mapped, (int, np.integer)):
                raise ValueError("SSMax selected dataset index mapping must contain integers")
            stable_index = int(mapped)
            if stable_index < 0:
                raise ValueError("SSMax selected dataset index mapping must be non-negative")
        inner = getattr(current, "_dataset", None)
        if inner is None:
            inner = getattr(current, "dataset", None)
        if inner is current:
            break
        current = inner
    return stable_index


def _projected_positive_weight(loss_token_weighting: str, positive_targets: int) -> float:
    if positive_targets <= 0:
        raise ValueError("A projected SSMax response must contain a positive loss target")
    if loss_token_weighting == "root_subsegments_root_tokens":
        return 2.0 / math.sqrt(positive_targets)
    if loss_token_weighting in ("root_subsegments", "none"):
        return 1.0
    raise ValueError(f"Unsupported loss-token weighting {loss_token_weighting!r}")


def project_ssmax_single_response(
    example: Mapping[str, Any],
    *,
    source_name: str,
    logical_split: str,
    sample_index: int,
    epoch: int,
    seed: int,
    loss_token_weighting: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Project an already serialized example onto exactly one response annotation.

    :returns: The model-consumed projected example and an auditable selection receipt.
    :raises ValueError: If the input is malformed, packed across examples, or has no response.
    """

    if not isinstance(example, Mapping):
        raise TypeError("SSMax single-response input must be a mapping")
    if not isinstance(source_name, str) or not source_name:
        raise ValueError("source_name must be non-empty")
    if not isinstance(logical_split, str) or not logical_split:
        raise ValueError("logical_split must be non-empty")
    if isinstance(sample_index, bool) or not isinstance(sample_index, int) or sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("epoch must be non-negative")
    contract = ssmax_single_response_projection_contract(
        seed=seed, loss_token_weighting=loss_token_weighting
    )
    if "example_ids" in example:
        raise ValueError("SSMax projection must run before cross-example sequence packing")
    length = _require_token_arrays(example)
    branches = _branch_ids(example, length)
    if not branches:
        if not np.any(example["loss_masks"] > 0):
            raise ValueError("Unbranched SSMax example contains no supervised targets")
        output = dict(example)
        output.pop("subsegment_ids", None)
        receipt = {
            "projection_contract_sha256": contract["content_sha256"],
            "source": source_name,
            "logical_split": logical_split,
            "stable_sample_index": sample_index,
            "requested_epoch": epoch,
            "selection_epoch": epoch if logical_split == "train" else 0,
            "materialization_epoch": epoch if logical_split == "train" else 0,
            "branch_count": 1,
            "selected_branch_id": None,
            "input_tokens": length,
            "projected_tokens": length,
            "positive_targets": int(np.count_nonzero(example["loss_masks"] > 0)),
        }
        return output, {**receipt, "content_sha256": _canonical_sha256(receipt)}

    selection_epoch = epoch if logical_split == "train" else 0
    ordinal = _selection_ordinal(
        seed=seed,
        source_name=source_name,
        stable_sample_index=sample_index,
        selection_epoch=selection_epoch,
        branch_count=len(branches),
    )
    selected_branch = branches[ordinal]
    subsegments = example["subsegment_ids"]
    keep = (subsegments == ATTEND_ALL_SUBSEGMENT_ID) | (subsegments == selected_branch)
    retained_indices = np.flatnonzero(keep)
    if retained_indices.size == 0:
        raise ValueError("SSMax response projection unexpectedly retained no tokens")
    output: Dict[str, Any] = {}
    for name, value in example.items():
        if name == "subsegment_ids":
            continue
        if isinstance(value, np.ndarray) and value.ndim == 1 and len(value) == length:
            output[name] = value[keep].copy()
        else:
            output[name] = value

    # A chosen branch need not be the first serialized branch.  Repair only the labels that
    # crossed removed tokens; labels inside the branch are retained so repetition filtering and
    # other source-specific label masks remain byte exact.
    labels = output["labels"].copy()
    input_ids = output["input_ids"]
    discontinuities = np.flatnonzero(np.diff(retained_indices) != 1)
    for position in discontinuities:
        labels[position] = input_ids[position + 1]
    output["labels"] = labels
    output["position_ids"] = np.arange(len(input_ids), dtype=example["position_ids"].dtype)

    positive = output["loss_masks"] > 0
    positive_targets = int(np.count_nonzero(positive))
    weight = _projected_positive_weight(loss_token_weighting, positive_targets)
    loss_masks = np.zeros_like(output["loss_masks"])
    loss_masks[positive] = weight
    output["loss_masks"] = loss_masks
    if "subsegment_ids" in output or "example_ids" in output:
        raise AssertionError("Projected SSMax example still contains packed metadata")

    receipt = {
        "projection_contract_sha256": contract["content_sha256"],
        "source": source_name,
        "logical_split": logical_split,
        "stable_sample_index": sample_index,
        "requested_epoch": epoch,
        "selection_epoch": selection_epoch,
        "materialization_epoch": selection_epoch,
        "branch_count": len(branches),
        "selected_branch_id": selected_branch,
        "input_tokens": length,
        "projected_tokens": len(input_ids),
        "positive_targets": positive_targets,
    }
    return output, {**receipt, "content_sha256": _canonical_sha256(receipt)}


class SSMaxSingleResponseDataset:
    """Map-style dataset wrapper implementing the SSMax projection contract.

    The wrapper proxies source validation methods but gives the projected view a distinct content
    fingerprint.  Call :meth:`projection_receipt` to retain the exact branch choice in offline
    calibration/evidence artifacts.
    """

    content_fingerprint_version = "ssmax-single-response-dataset-v1"

    def __init__(
        self,
        dataset: Any,
        *,
        source_name: str,
        logical_split: str,
        seed: int,
        loss_token_weighting: str,
    ):
        if logical_split != "train" and logical_split not in _EVAL_SPLITS:
            raise ValueError(
                "SSMax logical_split must be 'train' or an explicit fixed evaluation split"
            )
        self.dataset = dataset
        self.source_name = source_name
        self.logical_split = logical_split
        self.seed = seed
        self.loss_token_weighting = loss_token_weighting
        self.contract = ssmax_single_response_projection_contract(
            seed=seed, loss_token_weighting=loss_token_weighting
        )
        runtime_fingerprint = runtime_dataset_fingerprint(dataset)
        base_fingerprint = getattr(
            dataset, "ssmax_projection_base_content_fingerprint", runtime_fingerprint
        )
        if not isinstance(base_fingerprint, str) or not base_fingerprint:
            raise ValueError("SSMax projection requires a stable source dataset fingerprint")
        if base_fingerprint != runtime_fingerprint:
            inner = getattr(dataset, "_dataset", None)
            if inner is None:
                inner = getattr(dataset, "dataset", None)
            if inner is None or runtime_dataset_fingerprint(inner) != base_fingerprint:
                raise ValueError(
                    "SSMax projection base fingerprint differs from its transparent wrapper"
                )
        self.base_content_fingerprint = base_fingerprint
        self.content_fingerprint = _canonical_sha256(
            {
                "version": self.content_fingerprint_version,
                "base_content_fingerprint": base_fingerprint,
                "source": source_name,
                "logical_split": logical_split,
                "projection_contract_sha256": self.contract["content_sha256"],
            }
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.get(index, 0)

    def _raw(self, index: int, epoch: int) -> Mapping[str, Any]:
        getter = getattr(self.dataset, "get", None)
        row = getter(index, epoch) if callable(getter) else self.dataset[index]
        if not isinstance(row, Mapping):
            raise TypeError(f"SSMax source {self.source_name!r} row {index} is not a mapping")
        return row

    def get(self, index: int, epoch: int = 0) -> Dict[str, Any]:
        """Return the deterministic single-response view of one logical sample."""

        projected, _ = self.get_with_receipt(index, epoch)
        return projected

    def get_with_receipt(self, index: int, epoch: int = 0) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Materialize a row once and return both projection and selection receipt."""

        stable_index = resolve_ssmax_stable_sample_index(self.dataset, index)
        materialization_epoch = epoch if self.logical_split == "train" else 0
        return project_ssmax_single_response(
            self._raw(index, materialization_epoch),
            source_name=self.source_name,
            logical_split=self.logical_split,
            sample_index=stable_index,
            epoch=epoch,
            seed=self.seed,
            loss_token_weighting=self.loss_token_weighting,
        )

    def projection_receipt(self, index: int, epoch: int = 0) -> Dict[str, Any]:
        """Return the canonical receipt for one deterministic branch decision."""

        _, receipt = self.get_with_receipt(index, epoch)
        return receipt

    def validate_image_content(self, indices: Optional[Sequence[int]] = None) -> str:
        """Delegate immutable image-byte validation to the selected source dataset."""

        validate = self._find_dataset_method("validate_image_content")
        return validate(indices) if indices is not None else validate()

    def validate_required_annotations(self) -> Any:
        """Delegate the source-wide annotation validator when it exists."""

        try:
            validate = self._find_dataset_method("validate_required_annotations")
        except ValueError:
            return None
        return validate()

    def _find_dataset_method(self, name: str) -> Any:
        """Find a validation method through transparent audit/selection wrappers."""

        current = self.dataset
        visited: set[int] = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            method = getattr(current, name, None)
            if callable(method):
                return method
            inner = getattr(current, "_dataset", None)
            if inner is None:
                inner = getattr(current, "dataset", None)
            if inner is current:
                break
            current = inner
        raise ValueError(f"Wrapped SSMax source lacks {name.replace('_', '-')} validation")


def ssmax_single_response_calibration_summary(
    dataset: Any,
    panel: Sequence[tuple[int, int]],
) -> Dict[str, Any]:
    """Recompute a raw, deterministic calibration panel from a projected dataset.

    :param dataset: A :class:`SSMaxSingleResponseDataset` for a visual source.
    :param panel: Ordered ``(outer selected index, epoch)`` pairs from the pinned source audit.
    :returns: Compact hashes and the exact mean projected loss mass.
    """

    if not isinstance(dataset, SSMaxSingleResponseDataset):
        raise TypeError("Visual projection calibration requires SSMaxSingleResponseDataset")
    if not panel:
        raise ValueError("SSMax projection calibration panel must not be empty")
    panel_rows = []
    projection_receipts = []
    serialized_rows = []
    loss_masses = []
    projected_lengths = []
    branch_counts: Dict[str, int] = {}
    for outer_index, epoch in panel:
        if (
            isinstance(outer_index, bool)
            or not isinstance(outer_index, int)
            or outer_index < 0
            or isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or epoch < 0
        ):
            raise ValueError("SSMax projection calibration panel entries must be non-negative ints")
        example, receipt = dataset.get_with_receipt(outer_index, epoch)
        if "subsegment_ids" in example or "example_ids" in example:
            raise ValueError("Projected calibration row retains packed metadata")
        loss_mass = math.fsum(float(value) for value in example["loss_masks"])
        if not math.isfinite(loss_mass) or loss_mass <= 0:
            raise ValueError("Projected calibration row has non-positive loss mass")
        panel_rows.append(
            {
                "outer_index": outer_index,
                "stable_sample_index": receipt["stable_sample_index"],
                "requested_epoch": epoch,
                "selection_epoch": receipt["selection_epoch"],
                "materialization_epoch": receipt["materialization_epoch"],
            }
        )
        projection_receipts.append(receipt["content_sha256"])
        serialized_rows.append(serialized_example_sha256(example))
        loss_masses.append(loss_mass)
        projected_lengths.append(len(example["input_ids"]))
        branch_key = str(receipt["branch_count"])
        branch_counts[branch_key] = branch_counts.get(branch_key, 0) + 1
    return {
        "dataset_content_fingerprint": dataset.content_fingerprint,
        "rows": len(panel),
        "panel_sha256": _canonical_sha256(panel_rows),
        "projection_receipts_sha256": _canonical_sha256(projection_receipts),
        "serialized_rows_sha256": _canonical_sha256(serialized_rows),
        "mean_sum_loss_masks": math.fsum(loss_masses) / len(loss_masses),
        "minimum_sum_loss_masks": min(loss_masses),
        "maximum_sum_loss_masks": max(loss_masses),
        "minimum_projected_tokens": min(projected_lengths),
        "maximum_projected_tokens": max(projected_lengths),
        "branch_count_histogram": branch_counts,
        "zero_loss_examples": 0,
        "errors": 0,
    }


def validate_ssmax_single_response_calibration(
    value: Any,
    *,
    expected_phase: str,
    expected_contract: Mapping[str, Any],
    expected_source_audit: Mapping[str, Any],
    expected_selection_manifest: Mapping[str, Any],
    expected_visual_sources: Sequence[str],
    expected_unprojected_sources: Sequence[str],
    expected_mean_loss_weight: Mapping[str, float],
    expected_validation_rows_per_source: Optional[Mapping[str, int]] = None,
) -> Mapping[str, Any]:
    """Strictly validate a projection-calibration receipt's semantic bindings.

    Raw file hashes and checked-in producer/implementation blobs are filesystem concerns and are
    verified by the launcher.  This validator closes the JSON schema and recomputes every
    semantic self-hash so callers cannot accept a hand-edited receipt.
    """

    fields = {
        "format",
        "version",
        "status",
        "created_at",
        "phase",
        "producer",
        "projection_implementation",
        "projection_contract",
        "source_audit",
        "selection_manifest",
        "sources",
        "validation_preflight",
        "unprojected_sources",
        "projected_mean_loss_weight",
        "errors",
        "content_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("SSMax single-response calibration fields differ from schema")
    if (
        value["format"] != SSMAX_SINGLE_RESPONSE_CALIBRATION_FORMAT
        or value["version"] != SSMAX_SINGLE_RESPONSE_CALIBRATION_VERSION
        or value["status"] != "ok"
        or value["phase"] != expected_phase
        or value["errors"] != []
    ):
        raise ValueError("SSMax single-response calibration identity or status differs")
    created_at = value["created_at"]
    try:
        parsed_created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise ValueError("SSMax calibration created_at is invalid") from error
    if not created_at or parsed_created_at.tzinfo is None:
        raise ValueError("SSMax calibration created_at must be a timezone-aware timestamp")
    for name in ("producer", "projection_implementation"):
        reference = value[name]
        if (
            not isinstance(reference, Mapping)
            or set(reference) != {"path", "sha256"}
            or not isinstance(reference["path"], str)
            or not reference["path"]
            or not _is_sha256(reference["sha256"])
        ):
            raise ValueError(f"SSMax calibration {name} reference is invalid")
    for name, expected in (
        ("source_audit", expected_source_audit),
        ("selection_manifest", expected_selection_manifest),
    ):
        reference = value[name]
        if (
            not isinstance(reference, Mapping)
            or set(reference) != {"path", "raw_sha256", "content_sha256"}
            or not isinstance(reference["path"], str)
            or not reference["path"]
            or not _is_sha256(reference["raw_sha256"])
            or not _is_sha256(reference["content_sha256"])
            or dict(reference) != dict(expected)
        ):
            raise ValueError(f"SSMax calibration {name} reference differs")
    if not isinstance(value["projection_contract"], Mapping) or dict(
        value["projection_contract"]
    ) != dict(expected_contract):
        raise ValueError("SSMax calibration projection contract differs")
    visual_sources = tuple(sorted(str(source) for source in expected_visual_sources))
    unprojected_sources = tuple(sorted(str(source) for source in expected_unprojected_sources))
    if value["unprojected_sources"] != list(unprojected_sources):
        raise ValueError("SSMax calibration unprojected source list differs")
    sources = value["sources"]
    validation_preflight = value["validation_preflight"]
    for label, summaries in (
        ("calibration", sources),
        ("validation preflight", validation_preflight),
    ):
        if not isinstance(summaries, Mapping) or tuple(sorted(summaries)) != visual_sources:
            raise ValueError(f"SSMax {label} visual source set differs")
    summary_fields = {
        "dataset_content_fingerprint",
        "rows",
        "panel_sha256",
        "projection_receipts_sha256",
        "serialized_rows_sha256",
        "mean_sum_loss_masks",
        "minimum_sum_loss_masks",
        "maximum_sum_loss_masks",
        "minimum_projected_tokens",
        "maximum_projected_tokens",
        "branch_count_histogram",
        "zero_loss_examples",
        "errors",
    }
    for label, summaries in (
        ("calibration", sources),
        ("validation preflight", validation_preflight),
    ):
        for source in visual_sources:
            summary = summaries[source]
            if not isinstance(summary, Mapping) or set(summary) != summary_fields:
                raise ValueError(f"SSMax {label} summary {source!r} fields differ")
            if (
                not isinstance(summary["dataset_content_fingerprint"], str)
                or not summary["dataset_content_fingerprint"]
                or isinstance(summary["rows"], bool)
                or not isinstance(summary["rows"], int)
                or summary["rows"] <= 0
                or summary["zero_loss_examples"] != 0
                or summary["errors"] != 0
            ):
                raise ValueError(f"SSMax {label} summary {source!r} is incomplete")
            if expected_validation_rows_per_source is not None and label == "validation preflight":
                expected_rows = expected_validation_rows_per_source.get(source)
                if (
                    isinstance(expected_rows, bool)
                    or not isinstance(expected_rows, int)
                    or expected_rows <= 0
                    or summary["rows"] != expected_rows
                ):
                    raise ValueError(f"SSMax validation preflight {source!r} row count differs")
            for sha_name in (
                "panel_sha256",
                "projection_receipts_sha256",
                "serialized_rows_sha256",
            ):
                sha = summary[sha_name]
                if not _is_sha256(sha):
                    raise ValueError(f"SSMax {label} summary {source!r} has invalid SHA")
            for numeric_name in (
                "mean_sum_loss_masks",
                "minimum_sum_loss_masks",
                "maximum_sum_loss_masks",
            ):
                number = summary[numeric_name]
                if (
                    isinstance(number, bool)
                    or not isinstance(number, (int, float))
                    or not math.isfinite(float(number))
                    or float(number) <= 0
                ):
                    raise ValueError(f"SSMax {label} {source!r} {numeric_name} is invalid")
            for count_name in ("minimum_projected_tokens", "maximum_projected_tokens"):
                count = summary[count_name]
                if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                    raise ValueError(f"SSMax {label} {source!r} projected length is invalid")
            histogram = summary["branch_count_histogram"]
            if (
                not isinstance(histogram, Mapping)
                or not histogram
                or any(
                    not isinstance(key, str)
                    or not key.isdigit()
                    or int(key) <= 0
                    or isinstance(count, bool)
                    or not isinstance(count, int)
                    or count <= 0
                    for key, count in histogram.items()
                )
                or sum(histogram.values()) != summary["rows"]
            ):
                raise ValueError(f"SSMax {label} {source!r} branch histogram is invalid")
    means = value["projected_mean_loss_weight"]
    if not isinstance(means, Mapping) or set(means) != set(expected_mean_loss_weight):
        raise ValueError("SSMax calibration projected mean source set differs")
    for source, expected in expected_mean_loss_weight.items():
        actual = means[source]
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not math.isfinite(float(actual))
            or float(actual) != float(expected)
        ):
            raise ValueError(f"SSMax calibration projected mean {source!r} differs")
        if source in sources and float(actual) != float(sources[source]["mean_sum_loss_masks"]):
            raise ValueError(f"SSMax calibration source summary mean {source!r} differs")
    unsigned = dict(value)
    content_sha256 = unsigned.pop("content_sha256")
    if not _is_sha256(content_sha256) or content_sha256 != _canonical_sha256(unsigned):
        raise ValueError("SSMax single-response calibration content SHA-256 differs")
    return value
