"""Deterministic, replayable same-geometry wrong-image evaluation datasets."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from ..exceptions import OLMoConfigurationError

_PAIRING_FORMAT = "multimodal_matched_wrong_image_pairing"
_PAIRING_VERSION = 2
_PAIRING_FIELDS = frozenset(
    {
        "format",
        "version",
        "dataset_size",
        "recipient_count",
        "seed",
        "epoch",
        "content_ids_sha256",
        "coverage",
        "rows",
        "pairs",
    }
)
_ROW_FIELDS = frozenset({"index", "content_id", "example"})
_PAIR_FIELDS = frozenset({"recipient", "donor"})
_ARRAY_FIELDS = frozenset({"kind", "dtype", "shape", "sha256"})
_COVERAGE_FIELDS = frozenset(
    {
        "dataset_count",
        "eligible_count",
        "excluded_count",
        "selected_recipient_count",
        "geometry_count",
        "eligible_geometry_count",
        "selected_geometry_count",
        "geometry_histogram",
    }
)
_GEOMETRY_COVERAGE_FIELDS = frozenset(
    {
        "geometry",
        "dataset_count",
        "distinct_count",
        "eligible_count",
        "excluded_count",
        "selected_recipient_count",
        "selected_donor_count",
    }
)
_GEOMETRY_FIELDS = frozenset({"images", "pooled_patches_idx"})
_IMAGE_GEOMETRY_FIELDS = frozenset({"kind", "dtype", "shape"})
_POOLED_GEOMETRY_FIELDS = frozenset({"kind", "dtype", "shape", "sha256"})


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("utf-8")


def serialize_matched_wrong_image_pairing(payload: Mapping[str, Any]) -> bytes:
    """Serialize an explicit pairing payload to canonical JSON bytes.

    :param payload: Payload returned by :func:`build_matched_wrong_image_pairing`.
    :returns: Canonical UTF-8 JSON, including one trailing newline.
    """
    return _canonical_json_bytes(payload)


def matched_wrong_image_pairing_sha256(payload: Mapping[str, Any]) -> str:
    """Return the exact canonical-file SHA-256 for a pairing payload."""
    return hashlib.sha256(serialize_matched_wrong_image_pairing(payload)).hexdigest()


def _array_descriptor(value: Any, *, field_name: str) -> Dict[str, Any]:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        try:
            raw = tensor.view(torch.uint8).reshape(-1).numpy().tobytes(order="C")
        except RuntimeError as error:
            raise OLMoConfigurationError(
                f"Could not describe tensor field {field_name!r} for wrong-image evaluation"
            ) from error
        return {
            "kind": "torch",
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    array = np.asarray(value)
    if array.dtype.hasobject:
        raise OLMoConfigurationError(
            f"Wrong-image evaluation field {field_name!r} has an object dtype"
        )
    dtype = array.dtype
    if dtype.itemsize > 1:
        little_dtype = dtype.newbyteorder("<")
        if dtype.byteorder == ">" or (dtype.byteorder == "=" and sys.byteorder == "big"):
            array = array.byteswap().view(little_dtype)
        else:
            array = array.astype(little_dtype, copy=False)
    array = np.ascontiguousarray(array)
    return {
        "kind": "numpy",
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _example_descriptor(example: Mapping[str, Any], *, index: int) -> Dict[str, Any]:
    missing = sorted({"images", "pooled_patches_idx"} - set(example))
    if missing:
        raise OLMoConfigurationError(
            f"Wrong-image validation row {index} is missing fields {missing}"
        )
    for field_name in ("images", "pooled_patches_idx"):
        value = example[field_name]
        if not isinstance(value, (np.ndarray, torch.Tensor)):
            raise OLMoConfigurationError(
                f"Wrong-image validation row {index} has a non-tensor {field_name!r} field"
            )
        elements = value.numel() if isinstance(value, torch.Tensor) else value.size
        if elements == 0:
            raise OLMoConfigurationError(
                f"Wrong-image validation row {index} has an empty {field_name!r} field"
            )
    return {
        str(field_name): _array_descriptor(value, field_name=str(field_name))
        for field_name, value in sorted(example.items(), key=lambda item: str(item[0]))
    }


def _get_example(dataset: Any, index: int, epoch: int) -> Mapping[str, Any]:
    get = getattr(dataset, "get", None)
    example = get(index, epoch) if callable(get) else dataset[index]
    if not isinstance(example, Mapping):
        raise OLMoConfigurationError(f"Wrong-image validation row {index} is not a mapping")
    return example


def _geometry(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    example = row["example"]
    images = example["images"]
    pooled = example["pooled_patches_idx"]
    return (
        images["kind"],
        images["dtype"],
        tuple(images["shape"]),
        pooled["kind"],
        pooled["dtype"],
        tuple(pooled["shape"]),
        pooled["sha256"],
    )


def _geometry_descriptor(row: Mapping[str, Any]) -> Dict[str, Any]:
    example = row["example"]
    images = example["images"]
    pooled = example["pooled_patches_idx"]
    return {
        "images": {
            "kind": images["kind"],
            "dtype": images["dtype"],
            "shape": list(images["shape"]),
        },
        "pooled_patches_idx": {
            "kind": pooled["kind"],
            "dtype": pooled["dtype"],
            "shape": list(pooled["shape"]),
            "sha256": pooled["sha256"],
        },
    }


def _order_digest(seed: int, purpose: str, index: int, content_id: str) -> str:
    return hashlib.sha256(
        f"multimodal-wrong-image-v1\0{seed}\0{purpose}\0{index}\0{content_id}".encode()
    ).hexdigest()


def _content_ids_sha256(content_ids: Sequence[str]) -> str:
    return hashlib.sha256("".join(f"{value}\n" for value in content_ids).encode()).hexdigest()


def build_matched_wrong_image_pairing(
    dataset: Any,
    *,
    recipient_count: int,
    seed: int,
    content_ids: Sequence[str],
    epoch: int = 0,
) -> Dict[str, Any]:
    """Build a deterministic explicit pairing over one fixed validation dataset.

    All rows are materialized only by this builder. It first groups them by exact image tensor
    shape and byte-identical ``pooled_patches_idx``. Within each group it removes duplicate source
    content and duplicate materialized image tensors, then deterministically selects the requested
    number of eligible recipients across the complete validation split. Every selected recipient
    receives a unique donor from its group. Singleton and otherwise unusable geometry groups are
    skipped; construction fails if the full requested recipient set cannot be matched.

    The returned JSON-compatible payload contains the selected global recipient indices, donor
    indices, exact runtime descriptors needed for rank-local replay and drift detection, and a
    deterministic geometry histogram that makes selection coverage and exclusions explicit.

    :param dataset: Fixed map-style validation dataset.
    :param recipient_count: Number of paired validation examples to select.
    :param seed: Non-negative deterministic selection and matching seed.
    :param content_ids: Row-aligned lowercase source-image SHA-256 identities. This argument is
        required; callers must not infer source identity from a batch-local image tensor.
    :param epoch: Dataset source epoch used to materialize every descriptor.
    :returns: Canonicalizable explicit pairing payload.
    :raises OLMoConfigurationError: If inputs are invalid or a full pairing is impossible.
    """
    dataset_size = len(dataset)
    if (
        isinstance(recipient_count, bool)
        or not isinstance(recipient_count, int)
        or recipient_count < 1
        or recipient_count > dataset_size
    ):
        raise OLMoConfigurationError(
            "Wrong-image recipient_count must be a positive integer no larger than the "
            f"validation dataset ({dataset_size}), got {recipient_count!r}"
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise OLMoConfigurationError("Wrong-image pairing seed must be a non-negative integer")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise OLMoConfigurationError("Wrong-image pairing epoch must be a non-negative integer")
    if len(content_ids) != dataset_size:
        raise OLMoConfigurationError(
            "Wrong-image content identities must have exactly one entry per validation row "
            f"({dataset_size}), got {len(content_ids)}"
        )
    for index, content_id in enumerate(content_ids):
        if (
            not isinstance(content_id, str)
            or len(content_id) != 64
            or any(character not in "0123456789abcdef" for character in content_id)
        ):
            raise OLMoConfigurationError(
                f"Wrong-image content identity at validation index {index} is not a lowercase "
                "SHA-256"
            )

    rows: list[Dict[str, Any]] = []
    for index, content_id in enumerate(content_ids):
        example = _get_example(dataset, index, epoch)
        rows.append(
            {
                "index": index,
                "content_id": content_id,
                "example": _example_descriptor(example, index=index),
            }
        )

    rows_by_geometry: Dict[Tuple[Any, ...], list[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_geometry[_geometry(row)].append(row)

    eligible_by_geometry: Dict[Tuple[Any, ...], list[Dict[str, Any]]] = {}
    distinct_count_by_geometry: Dict[Tuple[Any, ...], int] = {}
    all_eligible: list[Dict[str, Any]] = []
    for geometry, group in rows_by_geometry.items():
        ordered = sorted(
            group,
            key=lambda row: (
                _order_digest(seed, "eligible", row["index"], row["content_id"]),
                row["index"],
            ),
        )
        distinct: list[Dict[str, Any]] = []
        seen_content: set[str] = set()
        seen_tensors: set[str] = set()
        for row in ordered:
            image_sha = row["example"]["images"]["sha256"]
            if row["content_id"] in seen_content or image_sha in seen_tensors:
                continue
            seen_content.add(row["content_id"])
            seen_tensors.add(image_sha)
            distinct.append(row)
        distinct_count_by_geometry[geometry] = len(distinct)
        if len(distinct) >= 2:
            eligible_by_geometry[geometry] = distinct
            all_eligible.extend(distinct)

    selected = sorted(
        all_eligible,
        key=lambda row: (
            _order_digest(seed, "recipient-selection", row["index"], row["content_id"]),
            row["index"],
        ),
    )[:recipient_count]
    if len(selected) != recipient_count:
        raise OLMoConfigurationError(
            "Could not select enough validation rows with a distinct exact-geometry image donor: "
            f"requested {recipient_count}, found {len(selected)} across {dataset_size} rows"
        )

    selected_by_geometry: Dict[Tuple[Any, ...], list[Dict[str, Any]]] = defaultdict(list)
    for row in selected:
        selected_by_geometry[_geometry(row)].append(row)

    donor_by_recipient: Dict[int, int] = {}
    for geometry, recipients in selected_by_geometry.items():
        donors = sorted(
            eligible_by_geometry[geometry],
            key=lambda row: (
                _order_digest(seed, "donor", row["index"], row["content_id"]),
                row["index"],
            ),
        )
        ordered_recipients = sorted(
            recipients,
            key=lambda row: (
                _order_digest(seed, "recipient", row["index"], row["content_id"]),
                row["index"],
            ),
        )
        donor_to_recipient: Dict[int, int] = {}

        def augment(recipient: Mapping[str, Any], seen: set[int]) -> bool:
            offset = int(
                _order_digest(seed, "rotation", recipient["index"], recipient["content_id"])[:16],
                16,
            ) % len(donors)
            for donor in donors[offset:] + donors[:offset]:
                donor_index = donor["index"]
                if donor_index in seen or donor["content_id"] == recipient["content_id"]:
                    continue
                seen.add(donor_index)
                previous_index = donor_to_recipient.get(donor_index)
                previous = None
                if previous_index is not None:
                    previous = next(
                        row for row in ordered_recipients if row["index"] == previous_index
                    )
                if previous is None or augment(previous, seen):
                    donor_to_recipient[donor_index] = recipient["index"]
                    donor_by_recipient[recipient["index"]] = donor_index
                    return True
            return False

        for recipient in ordered_recipients:
            if not augment(recipient, set()):
                raise OLMoConfigurationError(
                    "Could not construct a complete unique-donor wrong-image pairing for exact "
                    f"geometry group containing validation row {recipient['index']}"
                )

    pairs = [
        {"recipient": row["index"], "donor": donor_by_recipient[row["index"]]} for row in selected
    ]
    selected_recipient_count_by_geometry: Dict[Tuple[Any, ...], int] = defaultdict(int)
    selected_donor_count_by_geometry: Dict[Tuple[Any, ...], int] = defaultdict(int)
    row_by_index = {row["index"]: row for row in rows}
    for pair in pairs:
        selected_recipient_count_by_geometry[_geometry(row_by_index[pair["recipient"]])] += 1
        selected_donor_count_by_geometry[_geometry(row_by_index[pair["donor"]])] += 1

    geometry_histogram = []
    for geometry, group in rows_by_geometry.items():
        eligible_count = len(eligible_by_geometry.get(geometry, ()))
        geometry_histogram.append(
            {
                "geometry": _geometry_descriptor(group[0]),
                "dataset_count": len(group),
                "distinct_count": distinct_count_by_geometry[geometry],
                "eligible_count": eligible_count,
                "excluded_count": len(group) - eligible_count,
                "selected_recipient_count": selected_recipient_count_by_geometry[geometry],
                "selected_donor_count": selected_donor_count_by_geometry[geometry],
            }
        )
    geometry_histogram.sort(key=lambda entry: _canonical_json_bytes(entry["geometry"]))

    used_indices = {pair[field] for pair in pairs for field in ("recipient", "donor")}
    payload = {
        "format": _PAIRING_FORMAT,
        "version": _PAIRING_VERSION,
        "dataset_size": dataset_size,
        "recipient_count": recipient_count,
        "seed": seed,
        "epoch": epoch,
        "content_ids_sha256": _content_ids_sha256(content_ids),
        "coverage": {
            "dataset_count": dataset_size,
            "eligible_count": len(all_eligible),
            "excluded_count": dataset_size - len(all_eligible),
            "selected_recipient_count": recipient_count,
            "geometry_count": len(rows_by_geometry),
            "eligible_geometry_count": len(eligible_by_geometry),
            "selected_geometry_count": len(selected_by_geometry),
            "geometry_histogram": geometry_histogram,
        },
        "rows": [row for row in rows if row["index"] in used_indices],
        "pairs": pairs,
    }
    validate_matched_wrong_image_pairing(
        payload,
        dataset_size=dataset_size,
        recipient_count=recipient_count,
        seed=seed,
        epoch=epoch,
        content_ids_sha256=_content_ids_sha256(content_ids),
    )
    return payload


def _validate_array_descriptor(value: Any, *, name: str) -> None:
    if not isinstance(value, Mapping) or set(value) != _ARRAY_FIELDS:
        raise OLMoConfigurationError(f"Wrong-image pairing {name} descriptor fields are invalid")
    if value["kind"] not in ("numpy", "torch") or not isinstance(value["dtype"], str):
        raise OLMoConfigurationError(f"Wrong-image pairing {name} dtype is invalid")
    shape = value["shape"]
    if not isinstance(shape, list) or any(
        isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in shape
    ):
        raise OLMoConfigurationError(f"Wrong-image pairing {name} shape is invalid")
    digest = value["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise OLMoConfigurationError(f"Wrong-image pairing {name} digest is invalid")


def _validate_geometry_tensor_descriptor(
    value: Any,
    *,
    name: str,
    expected_fields: frozenset[str],
) -> None:
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise OLMoConfigurationError(
            f"Wrong-image pairing coverage {name} descriptor fields are invalid"
        )
    if value["kind"] not in ("numpy", "torch") or not isinstance(value["dtype"], str):
        raise OLMoConfigurationError(f"Wrong-image pairing coverage {name} dtype is invalid")
    shape = value["shape"]
    if not isinstance(shape, list) or any(
        isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in shape
    ):
        raise OLMoConfigurationError(f"Wrong-image pairing coverage {name} shape is invalid")
    if "sha256" in expected_fields:
        digest = value["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise OLMoConfigurationError(f"Wrong-image pairing coverage {name} digest is invalid")


def _validate_geometry_descriptor(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != _GEOMETRY_FIELDS:
        raise OLMoConfigurationError(
            "Wrong-image pairing coverage geometry descriptor fields are invalid"
        )
    _validate_geometry_tensor_descriptor(
        value["images"],
        name="image geometry",
        expected_fields=_IMAGE_GEOMETRY_FIELDS,
    )
    _validate_geometry_tensor_descriptor(
        value["pooled_patches_idx"],
        name="pooling geometry",
        expected_fields=_POOLED_GEOMETRY_FIELDS,
    )


def _coverage_count(value: Any, *, name: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise OLMoConfigurationError(
            f"Wrong-image pairing coverage {name} must be a {qualifier} integer"
        )
    return value


def _validate_pairing_coverage(
    payload: Mapping[str, Any],
    *,
    row_by_index: Mapping[int, Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
) -> None:
    coverage = payload["coverage"]
    if not isinstance(coverage, Mapping) or set(coverage) != _COVERAGE_FIELDS:
        raise OLMoConfigurationError("Wrong-image pairing coverage fields are invalid")

    dataset_count = _coverage_count(coverage["dataset_count"], name="dataset_count", positive=True)
    eligible_count = _coverage_count(coverage["eligible_count"], name="eligible_count")
    excluded_count = _coverage_count(coverage["excluded_count"], name="excluded_count")
    selected_recipient_count = _coverage_count(
        coverage["selected_recipient_count"],
        name="selected_recipient_count",
        positive=True,
    )
    geometry_count = _coverage_count(
        coverage["geometry_count"], name="geometry_count", positive=True
    )
    eligible_geometry_count = _coverage_count(
        coverage["eligible_geometry_count"], name="eligible_geometry_count"
    )
    selected_geometry_count = _coverage_count(
        coverage["selected_geometry_count"], name="selected_geometry_count", positive=True
    )
    if (
        dataset_count != payload["dataset_size"]
        or selected_recipient_count != payload["recipient_count"]
        or eligible_count + excluded_count != dataset_count
        or selected_recipient_count > eligible_count
        or selected_geometry_count > eligible_geometry_count
        or eligible_geometry_count > geometry_count
    ):
        raise OLMoConfigurationError("Wrong-image pairing coverage totals are inconsistent")

    histogram = coverage["geometry_histogram"]
    if not isinstance(histogram, list) or len(histogram) != geometry_count:
        raise OLMoConfigurationError(
            "Wrong-image pairing coverage geometry histogram length is inconsistent"
        )

    histogram_by_geometry: Dict[bytes, Mapping[str, Any]] = {}
    previous_geometry_key: Optional[bytes] = None
    totals: Dict[str, int] = defaultdict(int)
    for entry in histogram:
        if not isinstance(entry, Mapping) or set(entry) != _GEOMETRY_COVERAGE_FIELDS:
            raise OLMoConfigurationError(
                "Wrong-image pairing coverage geometry histogram fields are invalid"
            )
        geometry = entry["geometry"]
        _validate_geometry_descriptor(geometry)
        geometry_key = _canonical_json_bytes(geometry)
        if previous_geometry_key is not None and geometry_key <= previous_geometry_key:
            raise OLMoConfigurationError(
                "Wrong-image pairing coverage geometry histogram is not canonically ordered"
            )
        previous_geometry_key = geometry_key
        histogram_by_geometry[geometry_key] = entry

        entry_dataset_count = _coverage_count(
            entry["dataset_count"], name="geometry dataset_count", positive=True
        )
        distinct_count = _coverage_count(
            entry["distinct_count"], name="geometry distinct_count", positive=True
        )
        entry_eligible_count = _coverage_count(
            entry["eligible_count"], name="geometry eligible_count"
        )
        entry_excluded_count = _coverage_count(
            entry["excluded_count"], name="geometry excluded_count"
        )
        entry_selected_recipient_count = _coverage_count(
            entry["selected_recipient_count"], name="geometry selected_recipient_count"
        )
        entry_selected_donor_count = _coverage_count(
            entry["selected_donor_count"], name="geometry selected_donor_count"
        )
        if (
            distinct_count > entry_dataset_count
            or entry_eligible_count not in (0, distinct_count)
            or (entry_eligible_count == 0 and distinct_count >= 2)
            or (entry_eligible_count > 0 and entry_eligible_count < 2)
            or entry_excluded_count != entry_dataset_count - entry_eligible_count
            or entry_selected_recipient_count > entry_eligible_count
            or entry_selected_donor_count != entry_selected_recipient_count
        ):
            raise OLMoConfigurationError(
                "Wrong-image pairing coverage geometry counts are inconsistent"
            )
        totals["dataset_count"] += entry_dataset_count
        totals["eligible_count"] += entry_eligible_count
        totals["excluded_count"] += entry_excluded_count
        totals["selected_recipient_count"] += entry_selected_recipient_count
        totals["selected_donor_count"] += entry_selected_donor_count
        totals["eligible_geometry_count"] += int(entry_eligible_count > 0)
        totals["selected_geometry_count"] += int(entry_selected_recipient_count > 0)

    if (
        totals["dataset_count"] != dataset_count
        or totals["eligible_count"] != eligible_count
        or totals["excluded_count"] != excluded_count
        or totals["selected_recipient_count"] != selected_recipient_count
        or totals["selected_donor_count"] != selected_recipient_count
        or totals["eligible_geometry_count"] != eligible_geometry_count
        or totals["selected_geometry_count"] != selected_geometry_count
    ):
        raise OLMoConfigurationError(
            "Wrong-image pairing coverage histogram totals are inconsistent"
        )

    observed_recipients: Dict[bytes, int] = defaultdict(int)
    observed_donors: Dict[bytes, int] = defaultdict(int)
    for pair in pairs:
        recipient_key = _canonical_json_bytes(_geometry_descriptor(row_by_index[pair["recipient"]]))
        donor_key = _canonical_json_bytes(_geometry_descriptor(row_by_index[pair["donor"]]))
        if recipient_key not in histogram_by_geometry or donor_key not in histogram_by_geometry:
            raise OLMoConfigurationError(
                "Wrong-image pairing coverage omits a selected recipient or donor geometry"
            )
        observed_recipients[recipient_key] += 1
        observed_donors[donor_key] += 1
    for geometry_key, entry in histogram_by_geometry.items():
        if (
            entry["selected_recipient_count"] != observed_recipients[geometry_key]
            or entry["selected_donor_count"] != observed_donors[geometry_key]
        ):
            raise OLMoConfigurationError(
                "Wrong-image pairing coverage disagrees with the explicit selected pairs"
            )


def validate_matched_wrong_image_pairing(
    payload: Mapping[str, Any],
    *,
    dataset_size: Optional[int] = None,
    recipient_count: Optional[int] = None,
    seed: Optional[int] = None,
    epoch: Optional[int] = None,
    content_ids_sha256: Optional[str] = None,
) -> None:
    """Validate a serialized pairing payload and optional expected identities."""
    if not isinstance(payload, Mapping) or set(payload) != _PAIRING_FIELDS:
        raise OLMoConfigurationError(
            f"Wrong-image pairing payload fields differ from version {_PAIRING_VERSION}"
        )
    if payload["format"] != _PAIRING_FORMAT or payload["version"] != _PAIRING_VERSION:
        raise OLMoConfigurationError("Wrong-image pairing payload identity is incompatible")
    for name, expected in (
        ("dataset_size", dataset_size),
        ("recipient_count", recipient_count),
        ("seed", seed),
        ("epoch", epoch),
    ):
        actual = payload[name]
        if isinstance(actual, bool) or not isinstance(actual, int) or actual < 0:
            raise OLMoConfigurationError(f"Wrong-image pairing {name} is invalid")
        if expected is not None and actual != expected:
            raise OLMoConfigurationError(
                f"Wrong-image pairing {name} differs: expected {expected}, got {actual}"
            )
    actual_content_sha = payload["content_ids_sha256"]
    if (
        not isinstance(actual_content_sha, str)
        or len(actual_content_sha) != 64
        or any(character not in "0123456789abcdef" for character in actual_content_sha)
        or (content_ids_sha256 is not None and actual_content_sha != content_ids_sha256)
    ):
        raise OLMoConfigurationError("Wrong-image pairing content identity differs or is invalid")

    rows = payload["rows"]
    pairs = payload["pairs"]
    if not isinstance(rows, list) or not isinstance(pairs, list):
        raise OLMoConfigurationError("Wrong-image pairing rows and pairs must be lists")
    if len(pairs) != payload["recipient_count"]:
        raise OLMoConfigurationError("Wrong-image pairing does not contain every recipient")
    row_by_index: Dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _ROW_FIELDS:
            raise OLMoConfigurationError("Wrong-image pairing row fields are invalid")
        index = row["index"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < payload["dataset_size"]
            or index in row_by_index
        ):
            raise OLMoConfigurationError("Wrong-image pairing row index is invalid or duplicated")
        content_id = row["content_id"]
        if (
            not isinstance(content_id, str)
            or len(content_id) != 64
            or any(character not in "0123456789abcdef" for character in content_id)
        ):
            raise OLMoConfigurationError("Wrong-image pairing row content identity is invalid")
        example = row["example"]
        if not isinstance(example, Mapping) or not {"images", "pooled_patches_idx"} <= set(example):
            raise OLMoConfigurationError("Wrong-image pairing row example is incomplete")
        for field_name, descriptor in example.items():
            _validate_array_descriptor(descriptor, name=f"row {index} field {field_name}")
        row_by_index[index] = row

    recipient_indices: set[int] = set()
    donor_indices: set[int] = set()
    for pair in pairs:
        if not isinstance(pair, Mapping) or set(pair) != _PAIR_FIELDS:
            raise OLMoConfigurationError("Wrong-image pairing pair fields are invalid")
        recipient = pair["recipient"]
        donor = pair["donor"]
        if (
            isinstance(recipient, bool)
            or not isinstance(recipient, int)
            or isinstance(donor, bool)
            or not isinstance(donor, int)
            or recipient not in row_by_index
            or donor not in row_by_index
            or recipient in recipient_indices
            or donor in donor_indices
        ):
            raise OLMoConfigurationError(
                "Wrong-image pairing uses an unknown, duplicate, or malformed recipient/donor"
            )
        recipient_row = row_by_index[recipient]
        donor_row = row_by_index[donor]
        if (
            recipient_row["content_id"] == donor_row["content_id"]
            or recipient_row["example"]["images"]["sha256"]
            == donor_row["example"]["images"]["sha256"]
            or _geometry(recipient_row) != _geometry(donor_row)
        ):
            raise OLMoConfigurationError(
                "Wrong-image pairing contains same-content or geometry-incompatible images"
            )
        recipient_indices.add(recipient)
        donor_indices.add(donor)
    if set(row_by_index) != recipient_indices | donor_indices:
        raise OLMoConfigurationError(
            "Wrong-image pairing contains unused or missing row identities"
        )
    _validate_pairing_coverage(payload, row_by_index=row_by_index, pairs=pairs)


class MultimodalFixedValidationDataset:
    """Replay the exact recipient subset from an explicit wrong-image pairing payload."""

    def __init__(
        self,
        dataset: Any,
        *,
        pairing: Mapping[str, Any],
        pairing_sha256: str,
    ):
        validate_matched_wrong_image_pairing(pairing, dataset_size=len(dataset))
        actual_sha = matched_wrong_image_pairing_sha256(pairing)
        if actual_sha != pairing_sha256:
            raise OLMoConfigurationError(
                f"Wrong-image pairing SHA mismatch: expected {pairing_sha256}, got {actual_sha}"
            )
        self.dataset = dataset
        self.pairing = pairing
        self.pairing_sha256 = actual_sha
        self.epoch = int(pairing["epoch"])
        self._pairs = tuple(pairing["pairs"])
        self._rows = {row["index"]: row for row in pairing["rows"]}

    def __len__(self) -> int:
        return len(self._pairs)

    @property
    def recipient_indices(self) -> Tuple[int, ...]:
        """Return selected global validation indices in evaluator order."""
        return tuple(pair["recipient"] for pair in self._pairs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.get(index, self.epoch)

    def get(self, index: int, epoch: int = 0) -> Dict[str, Any]:
        """Return and verify one fixed recipient without changing any field."""
        if epoch != self.epoch:
            raise OLMoConfigurationError(
                f"Matched validation evaluation is pinned to source epoch {self.epoch}, got {epoch}"
            )
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self):
            raise IndexError(index)
        recipient_index = self._pairs[index]["recipient"]
        return dict(self._verified_example(recipient_index))

    def _verified_example(self, base_index: int) -> Mapping[str, Any]:
        example = _get_example(self.dataset, base_index, self.epoch)
        actual = _example_descriptor(example, index=base_index)
        expected = self._rows[base_index]["example"]
        if actual != expected:
            raise OLMoConfigurationError(
                f"Wrong-image validation row {base_index} drifted from its explicit pairing"
            )
        return example


class MultimodalMatchedWrongImageDataset(MultimodalFixedValidationDataset):
    """Replay fixed recipients while replacing only images with their matched wrong donors."""

    @property
    def donor_indices(self) -> Tuple[int, ...]:
        """Return matched global donor indices in evaluator order."""
        return tuple(pair["donor"] for pair in self._pairs)

    def get(self, index: int, epoch: int = 0) -> Dict[str, Any]:
        """Return a verified recipient with only its image tensor replaced."""
        if epoch != self.epoch:
            raise OLMoConfigurationError(
                f"Wrong-image evaluation is pinned to source epoch {self.epoch}, got {epoch}"
            )
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self):
            raise IndexError(index)
        pair = self._pairs[index]
        recipient = self._verified_example(pair["recipient"])
        donor = self._verified_example(pair["donor"])
        recipient_images = recipient["images"]
        donor_images = donor["images"]
        if tuple(recipient_images.shape) != tuple(donor_images.shape) or not self._arrays_equal(
            recipient["pooled_patches_idx"], donor["pooled_patches_idx"]
        ):
            raise OLMoConfigurationError(
                f"Wrong-image geometry drifted for recipient {pair['recipient']} and donor "
                f"{pair['donor']}"
            )
        transformed = dict(recipient)
        if isinstance(donor_images, torch.Tensor):
            transformed["images"] = donor_images.clone()
        else:
            transformed["images"] = np.array(donor_images, copy=True)
        return transformed

    @staticmethod
    def _arrays_equal(left: Any, right: Any) -> bool:
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            return left.dtype == right.dtype and torch.equal(left, right)
        if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
            return left.dtype == right.dtype and np.array_equal(left, right)
        return False
