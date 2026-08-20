"""Auditable, bounded attention diagnostics for Scalable-Softmax models.

The collector in this module observes :class:`~olmo_core.nn.attention.Attention`
modules through ordinary PyTorch hooks. It does not replace the attention implementation and it
does not request attention matrices from the backend. Instead, it reconstructs logits for a fixed,
manifest-pinned set of query positions in small chunks. This keeps peak diagnostic memory linear
in sequence length while retaining the model's exact causal, bidirectional-image, branch,
example-isolation, padding, and grouped-query-attention semantics. The reconstructed probabilities
also expose how much attention routes to image, prompt, and response keys, including argmax-key
shares and exact normalization/partition checks.

The intended checkpoint integration is::

    manifest = SSMaxProbeManifest.load(path, expected_sha256=sha256)
    collector = SSMaxAttentionDiagnosticsCollector(model.lm, manifest)
    with collector.capture_batch(
        sample_ids=sample_ids,
        input_ids=batch["input_ids"],
        token_type_ids=batch["token_type_ids"],
        loss_masks=batch["loss_masks"],
        valid_tokens=batch["router_token_mask"],
        subsegment_ids=batch.get("subsegment_ids"),
        example_ids=batch.get("example_ids"),
    ):
        model(**batch_without_diagnostic_metadata)
    report = collector.finalize(checkpoint_identity=checkpoint_identity)

Every rank can call :meth:`export_state`; rank zero can merge those bounded states with
:meth:`finalize_states`. No KV cache is supported, deliberately matching SSMax training.
"""

from __future__ import annotations

import base64
import hashlib
import heapq
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

from olmo_core.aliases import PathOrStr
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import Attention

__all__ = [
    "KEY_CATEGORIES",
    "PROBE_CATEGORIES",
    "ProbeSequence",
    "SSMaxProbeBatch",
    "SSMaxProbeManifest",
    "SSMaxAttentionDiagnosticsCollector",
    "build_attention_allow_vector",
    "build_probe_manifest",
    "capture_ssmax_probe_batches",
    "compare_ssmax_attention_reports",
    "iter_ssmax_probe_batches",
    "probe_manifest_sha256",
    "serialize_probe_manifest",
    "validate_ssmax_attention_report",
]

PROBE_CATEGORIES = ("all", "image", "prompt", "response")
KEY_CATEGORIES = ("image", "prompt", "response")
_MANIFEST_FORMAT = "ssmax_multimodal_attention_probe"
_MANIFEST_VERSION = 1
_REPORT_FORMAT = "ssmax_attention_diagnostics"
_REPORT_VERSION = 1
_STATE_FORMAT = "ssmax_attention_diagnostics_state"
_STATE_VERSION = 1
_HEX = frozenset("0123456789abcdef")
_DISTRIBUTION_QUANTILES = (0.5, 0.9, 0.99, 0.999)
_ROUTING_MASS_METRICS = tuple(
    f"attention_mass_to_{key_category}_keys" for key_category in KEY_CATEGORIES
)
_ROUTING_CHECK_METRICS = (
    "attention_mass_to_allowed_keys",
    "attention_mass_normalization_error",
    "attention_mass_category_partition_error",
)
_ROUTING_ARGMAX_METRICS = tuple(f"argmax_key_is_{key_category}" for key_category in KEY_CATEGORIES)


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in _HEX for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _tensor_bytes(value: torch.Tensor) -> bytes:
    tensor = value.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.view(torch.uint16)
    return tensor.numpy().tobytes(order="C")


def _sequence_tensor_sha256(value: torch.Tensor, valid_length: int) -> str:
    value = value.detach().cpu()
    if value.ndim != 1:
        raise ValueError("Probe sequence tensors must be one-dimensional")
    if not 0 < valid_length <= value.shape[0]:
        raise ValueError("valid_length must be within the probe sequence")
    prefix = value[:valid_length].contiguous()
    descriptor = f"{prefix.dtype}\0{tuple(prefix.shape)}\0".encode()
    return _sha256_bytes(descriptor + _tensor_bytes(prefix))


def _category_masks(
    token_type_ids: torch.Tensor,
    loss_masks: torch.Tensor,
    valid_tokens: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    if not (
        token_type_ids.ndim == loss_masks.ndim == valid_tokens.ndim == 1
        and token_type_ids.shape == loss_masks.shape == valid_tokens.shape
    ):
        raise ValueError("Token-type, loss, and validity vectors must have identical 1D shapes")
    valid = valid_tokens.to(dtype=torch.bool)
    image = valid & (token_type_ids != 0)
    response = valid & (token_type_ids == 0) & (loss_masks > 0)
    prompt = valid & (token_type_ids == 0) & ~(loss_masks > 0)
    if not torch.equal(image | response | prompt, valid):
        raise RuntimeError("Probe token categories do not partition valid tokens")
    return {"all": valid, "image": image, "prompt": prompt, "response": response}


def build_attention_allow_vector(
    *,
    query_position: int,
    valid_tokens: torch.Tensor,
    token_type_ids: torch.Tensor,
    subsegment_ids: Optional[torch.Tensor] = None,
    example_ids: Optional[torch.Tensor] = None,
    window_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Build the exact allowed-key vector for one multimodal attention query.

    The rule is ``(causal_or_window | image_query_and_key) & subsegment & example & padding``.
    This matches the dense and FlexAttention paths used by :class:`MultimodalLM`. Invalid query
    positions return an all-false vector. Padding must be a contiguous suffix, which is the only
    layout supported by the fixed unpacked probe protocol.

    :param query_position: Zero-based sequence position of the query.
    :param valid_tokens: Boolean vector marking non-padding tokens.
    :param token_type_ids: Non-zero values mark image tokens.
    :param subsegment_ids: Optional branch IDs using ``query_id <= key_id`` semantics.
    :param example_ids: Optional packed-example IDs requiring equality between query and key.
    :param window_size: Backend window tuple ``(left, right)``; ``(-1, -1)`` means full causal.
    :returns: Boolean vector with one value per key position.
    """

    if valid_tokens.ndim != 1 or token_type_ids.shape != valid_tokens.shape:
        raise ValueError("valid_tokens and token_type_ids must be same-shaped 1D tensors")
    sequence_length = int(valid_tokens.shape[0])
    if not 0 <= query_position < sequence_length:
        raise ValueError("query_position is outside the sequence")
    valid = valid_tokens.to(dtype=torch.bool)
    valid_length = int(valid.sum().item())
    expected_valid = torch.arange(sequence_length, device=valid.device) < valid_length
    if not torch.equal(valid, expected_valid):
        raise ValueError("The SSMax probe requires right-padding with no holes")
    if not bool(valid[query_position]):
        return torch.zeros_like(valid)

    key_positions = torch.arange(sequence_length, device=valid.device)
    allowed = key_positions <= query_position
    if window_size is not None and window_size != (-1, -1):
        left, right = window_size
        if left < 0 or right < 0:
            raise ValueError("A finite attention window must contain non-negative bounds")
        allowed &= key_positions >= query_position - left
        allowed &= key_positions <= query_position + right

    is_image = token_type_ids != 0
    allowed |= is_image[query_position] & is_image

    if subsegment_ids is not None:
        if subsegment_ids.shape != valid.shape:
            raise ValueError("subsegment_ids must match valid_tokens")
        allowed &= subsegment_ids[query_position] <= subsegment_ids
    if example_ids is not None:
        if example_ids.shape != valid.shape:
            raise ValueError("example_ids must match valid_tokens")
        allowed &= example_ids[query_position] == example_ids
    return allowed & valid


@dataclass(frozen=True)
class ProbeSequence:
    """One tokenized validation row used to build a fixed probe manifest."""

    sample_id: str
    dataset_index: int
    input_ids: torch.Tensor
    token_type_ids: torch.Tensor
    loss_masks: torch.Tensor
    valid_tokens: torch.Tensor


@dataclass(frozen=True)
class SSMaxProbeBatch:
    """One rank-local collated batch reconstructed from a fixed probe manifest."""

    sample_ids: Tuple[str, ...]
    dataset_indices: Tuple[int, ...]
    batch: Mapping[str, Any]


def _select_positions(
    positions: Iterable[int],
    *,
    sample_id: str,
    category: str,
    seed: int,
    maximum: int,
) -> Tuple[int, ...]:
    ranked = []
    for position in positions:
        digest = hashlib.sha256(
            f"ssmax-probe-v1\0{seed}\0{sample_id}\0{category}\0{position}".encode()
        ).digest()
        ranked.append((digest, position))
    ranked.sort()
    return tuple(sorted(position for _, position in ranked[:maximum]))


def build_probe_manifest(
    sequences: Sequence[ProbeSequence],
    *,
    validation_manifest_path: PathOrStr,
    validation_manifest_sha256: str,
    seed: int,
    max_queries_per_category_per_row: int,
) -> "SSMaxProbeManifest":
    """Build a deterministic probe manifest from already-tokenized validation rows.

    The caller chooses rows from the separately pinned validation population. This function pins
    each valid token prefix and deterministically selects query positions inside each token type.

    :param sequences: Fixed, uniquely identified tokenized validation rows.
    :param validation_manifest_path: Path to the upstream validation-population manifest.
    :param validation_manifest_sha256: Exact byte hash of that manifest.
    :param seed: Non-negative query-selection seed.
    :param max_queries_per_category_per_row: Per-row bound for each query category.
    :returns: A validated :class:`SSMaxProbeManifest`.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if (
        isinstance(max_queries_per_category_per_row, bool)
        or not isinstance(max_queries_per_category_per_row, int)
        or max_queries_per_category_per_row <= 0
    ):
        raise ValueError("max_queries_per_category_per_row must be positive")
    if not sequences:
        raise ValueError("At least one probe sequence is required")
    _validate_sha256(validation_manifest_sha256, "validation_manifest_sha256")

    rows = []
    seen_ids = set()
    seen_indices = set()
    for sequence in sequences:
        if not sequence.sample_id or sequence.sample_id in seen_ids:
            raise ValueError("Probe sample IDs must be non-empty and unique")
        if sequence.dataset_index < 0 or sequence.dataset_index in seen_indices:
            raise ValueError("Probe dataset indices must be non-negative and unique")
        seen_ids.add(sequence.sample_id)
        seen_indices.add(sequence.dataset_index)
        tensors = (
            sequence.input_ids,
            sequence.token_type_ids,
            sequence.loss_masks,
            sequence.valid_tokens,
        )
        if (
            any(tensor.ndim != 1 for tensor in tensors)
            or len({tensor.shape for tensor in tensors}) != 1
        ):
            raise ValueError("Every probe row tensor must have the same one-dimensional shape")
        valid = sequence.valid_tokens.to(dtype=torch.bool)
        valid_length = int(valid.sum().item())
        expected_valid = torch.arange(valid.shape[0], device=valid.device) < valid_length
        if valid_length <= 0 or not torch.equal(valid, expected_valid):
            raise ValueError("Probe rows must be non-empty and right-padded without holes")
        masks = _category_masks(sequence.token_type_ids, sequence.loss_masks, valid)
        selected = {
            category: list(
                _select_positions(
                    torch.where(mask)[0].tolist(),
                    sample_id=sequence.sample_id,
                    category=category,
                    seed=seed,
                    maximum=max_queries_per_category_per_row,
                )
            )
            for category, mask in masks.items()
        }
        rows.append(
            {
                "sample_id": sequence.sample_id,
                "dataset_index": sequence.dataset_index,
                "valid_length": valid_length,
                "input_ids_sha256": _sequence_tensor_sha256(sequence.input_ids, valid_length),
                "token_type_ids_sha256": _sequence_tensor_sha256(
                    sequence.token_type_ids, valid_length
                ),
                "loss_masks_sha256": _sequence_tensor_sha256(sequence.loss_masks, valid_length),
                "query_positions": selected,
            }
        )
    rows.sort(key=lambda row: (row["dataset_index"], row["sample_id"]))
    return SSMaxProbeManifest.from_dict(
        {
            "format": _MANIFEST_FORMAT,
            "version": _MANIFEST_VERSION,
            "validation_manifest": {
                "path": str(Path(validation_manifest_path).expanduser().resolve()),
                "sha256": validation_manifest_sha256,
            },
            "selection": {
                "algorithm": "sha256-priority-per-row-category-v1",
                "seed": seed,
                "max_queries_per_category_per_row": max_queries_per_category_per_row,
                "categories": list(PROBE_CATEGORIES),
            },
            "rows": rows,
        }
    )


@dataclass(frozen=True)
class SSMaxProbeManifest:
    """Strict, JSON-serializable identity for fixed multimodal diagnostic queries."""

    payload: Mapping[str, Any]
    rows_by_sample_id: Mapping[str, Mapping[str, Any]] = field(repr=False)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SSMaxProbeManifest":
        """Validate and construct a manifest from a decoded JSON mapping."""

        if payload.get("format") != _MANIFEST_FORMAT or payload.get("version") != _MANIFEST_VERSION:
            raise ValueError("Unsupported SSMax probe manifest format or version")
        validation = payload.get("validation_manifest")
        selection = payload.get("selection")
        rows = payload.get("rows")
        if not isinstance(validation, Mapping) or not isinstance(selection, Mapping):
            raise ValueError("Probe manifest lacks validation or selection metadata")
        if not isinstance(rows, list) or not rows:
            raise ValueError("Probe manifest rows must be a non-empty list")
        validation_path = validation.get("path")
        validation_sha = validation.get("sha256")
        if not isinstance(validation_path, str) or not validation_path:
            raise ValueError("Probe validation-manifest path must be non-empty")
        if not isinstance(validation_sha, str):
            raise ValueError("Probe validation-manifest SHA-256 must be a string")
        _validate_sha256(validation_sha, "validation_manifest.sha256")
        if selection.get("algorithm") != "sha256-priority-per-row-category-v1":
            raise ValueError("Unsupported probe query-selection algorithm")
        if selection.get("categories") != list(PROBE_CATEGORIES):
            raise ValueError("Probe categories differ from the version-1 protocol")
        seed = selection.get("seed")
        maximum = selection.get("max_queries_per_category_per_row")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("Probe selection seed must be a non-negative integer")
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
            raise ValueError("Probe per-category query bound must be positive")

        rows_by_id: Dict[str, Mapping[str, Any]] = {}
        indices = set()
        canonical_rows = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("Every probe row must be a mapping")
            sample_id = row.get("sample_id")
            dataset_index = row.get("dataset_index")
            valid_length = row.get("valid_length")
            if not isinstance(sample_id, str) or not sample_id or sample_id in rows_by_id:
                raise ValueError("Probe row sample IDs must be non-empty and unique")
            if (
                isinstance(dataset_index, bool)
                or not isinstance(dataset_index, int)
                or dataset_index < 0
                or dataset_index in indices
            ):
                raise ValueError("Probe dataset indices must be non-negative and unique")
            if (
                isinstance(valid_length, bool)
                or not isinstance(valid_length, int)
                or valid_length <= 0
            ):
                raise ValueError("Probe valid lengths must be positive integers")
            for name in ("input_ids_sha256", "token_type_ids_sha256", "loss_masks_sha256"):
                value = row.get(name)
                if not isinstance(value, str):
                    raise ValueError(f"Probe row {name} must be a string")
                _validate_sha256(value, name)
            query_positions = row.get("query_positions")
            if not isinstance(query_positions, Mapping) or set(query_positions) != set(
                PROBE_CATEGORIES
            ):
                raise ValueError("Every row must contain exactly the version-1 query categories")
            for category in PROBE_CATEGORIES:
                positions = query_positions[category]
                if (
                    not isinstance(positions, list)
                    or len(positions) > maximum
                    or positions != sorted(set(positions))
                    or any(
                        isinstance(position, bool)
                        or not isinstance(position, int)
                        or not 0 <= position < valid_length
                        for position in positions
                    )
                ):
                    raise ValueError(f"Malformed {category!r} query positions for {sample_id}")
            indices.add(dataset_index)
            canonical_row = dict(row)
            rows_by_id[sample_id] = canonical_row
            canonical_rows.append(canonical_row)
        expected_order = sorted(
            canonical_rows, key=lambda row: (row["dataset_index"], row["sample_id"])
        )
        if canonical_rows != expected_order:
            raise ValueError("Probe rows must be sorted by dataset index and sample ID")
        canonical_payload = json.loads(_canonical_json_bytes(payload))
        return cls(payload=canonical_payload, rows_by_sample_id=rows_by_id)

    @classmethod
    def load(
        cls,
        path: PathOrStr,
        *,
        expected_sha256: str,
        verify_validation_manifest: bool = True,
    ) -> "SSMaxProbeManifest":
        """Load a manifest and verify both it and its upstream validation manifest."""

        _validate_sha256(expected_sha256, "expected_sha256")
        manifest_path = Path(path).expanduser().resolve()
        actual_sha = _sha256_file(manifest_path)
        if actual_sha != expected_sha256:
            raise ValueError(
                f"SSMax probe manifest SHA mismatch: expected {expected_sha256}, got {actual_sha}"
            )
        try:
            payload = json.loads(manifest_path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not decode SSMax probe manifest {manifest_path}") from error
        if not isinstance(payload, Mapping):
            raise ValueError("SSMax probe manifest root must be a mapping")
        manifest = cls.from_dict(payload)
        if verify_validation_manifest:
            validation = manifest.payload["validation_manifest"]
            validation_path = Path(validation["path"]).expanduser().resolve()
            actual_validation_sha = _sha256_file(validation_path)
            if actual_validation_sha != validation["sha256"]:
                raise ValueError(
                    "Upstream validation manifest SHA mismatch: "
                    f"expected {validation['sha256']}, got {actual_validation_sha}"
                )
        return manifest

    def as_dict(self) -> Dict[str, Any]:
        """Return a detached JSON-compatible manifest mapping."""

        return json.loads(_canonical_json_bytes(self.payload))

    @property
    def sha256(self) -> str:
        """Return the canonical serialized manifest hash."""

        return probe_manifest_sha256(self)


def serialize_probe_manifest(manifest: SSMaxProbeManifest) -> bytes:
    """Serialize a probe manifest to canonical JSON bytes with a trailing newline."""

    return _canonical_json_bytes(manifest.payload)


def probe_manifest_sha256(manifest: SSMaxProbeManifest) -> str:
    """Return the canonical probe-manifest SHA-256."""

    return _sha256_bytes(serialize_probe_manifest(manifest))


def _manifest_population_rows(
    manifest: SSMaxProbeManifest,
) -> Tuple[Mapping[str, Any], ...]:
    population = manifest.payload.get("population")
    if not isinstance(population, Mapping):
        raise ValueError("Operational probe manifests must contain population metadata")
    source = population.get("source")
    selected_indices = population.get("selected_dataset_indices")
    selected_content_ids = population.get("selected_content_ids")
    if not isinstance(source, str) or not source:
        raise ValueError("Probe population source must be non-empty")
    if not isinstance(selected_indices, list) or not isinstance(selected_content_ids, list):
        raise ValueError("Probe population must pin selected indices and content IDs")
    if len(selected_indices) != len(selected_content_ids) or len(selected_indices) != len(
        manifest.rows_by_sample_id
    ):
        raise ValueError("Probe population identities and manifest rows have different lengths")
    rows = tuple(
        sorted(
            manifest.rows_by_sample_id.values(),
            key=lambda row: (row["dataset_index"], row["sample_id"]),
        )
    )
    if [row["dataset_index"] for row in rows] != selected_indices:
        raise ValueError("Probe population indices differ from tokenized manifest rows")
    for row, content_id in zip(rows, selected_content_ids, strict=True):
        expected_sample_id = f"{source}:{row['dataset_index']}:{content_id}"
        if row["sample_id"] != expected_sample_id:
            raise ValueError(
                f"Probe sample ID {row['sample_id']!r} differs from {expected_sample_id!r}"
            )
    return rows


def iter_ssmax_probe_batches(
    dataset: Any,
    manifest: SSMaxProbeManifest,
    *,
    content_ids: Sequence[str],
    collate: Callable[[list[Dict[str, Any]]], Mapping[str, Any]],
    rank: int,
    world_size: int,
    batch_size: int,
) -> Iterator[SSMaxProbeBatch]:
    """Reconstruct this rank's exact manifest rows and collate bounded batches.

    The caller must first validate the live dataset and pass its independently pinned ordered
    image-content IDs from the upstream validation/provenance artifact. Before collation, this
    helper verifies each content-derived sample ID and all three tokenized prefix hashes. Rows are
    partitioned by stable manifest ordinal (``ordinal % world_size``), making every rank's sample
    set deterministic and disjoint at any supported DP topology.

    :param dataset: Map-style source wrapper supporting ``get(index, epoch)`` or ``__getitem__``.
    :param manifest: Operational fixed probe manifest with population identities.
    :param content_ids: Independently verified content SHA-256 values in dataset logical order.
    :param collate: Multimodal collator callable.
    :param rank: Rank within the data-parallel process group.
    :param world_size: Data-parallel process-group size.
    :param batch_size: Maximum examples per rank-local forward.
    :returns: An iterator of collated batches with their exact sample identities.
    """

    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("rank must be within a positive world_size")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rows = _manifest_population_rows(manifest)
    if not isinstance(content_ids, Sequence) or isinstance(content_ids, (str, bytes)):
        raise TypeError("Probe content_ids must be an ordered sequence")
    local_rows = [row for ordinal, row in enumerate(rows) if ordinal % world_size == rank]
    get = getattr(dataset, "get", None)
    population = manifest.payload["population"]
    source = str(population["source"])
    for start in range(0, len(local_rows), batch_size):
        row_chunk = local_rows[start : start + batch_size]
        examples = []
        for row in row_chunk:
            index = int(row["dataset_index"])
            if index >= len(content_ids):
                raise ValueError(f"Probe content identity index {index} is out of bounds")
            content_id = content_ids[index]
            if not isinstance(content_id, str):
                raise TypeError(f"Probe content identity {index} must be a string")
            _validate_sha256(content_id, f"content_ids[{index}]")
            expected_sample_id = f"{source}:{index}:{content_id}"
            if row["sample_id"] != expected_sample_id:
                raise ValueError(
                    f"Probe sample/content identity {row['sample_id']!r} differs from "
                    f"{expected_sample_id!r}"
                )
            example = get(index, 0) if callable(get) else dataset[index]
            if not isinstance(example, Mapping):
                raise ValueError(f"Probe validation row {index} did not produce a mapping")
            valid_length = int(row["valid_length"])
            for name in ("input_ids", "token_type_ids", "loss_masks"):
                if name not in example:
                    raise ValueError(f"Probe validation row {index} lacks {name!r}")
                tensor = torch.as_tensor(example[name])
                expected_sha = row[f"{name}_sha256"]
                try:
                    actual_sha = _sequence_tensor_sha256(tensor, valid_length)
                except ValueError as error:
                    raise ValueError(
                        f"Probe validation row {index} has an invalid {name} sequence"
                    ) from error
                if actual_sha != expected_sha:
                    raise ValueError(
                        f"Probe validation row {index} {name} prefix differs from its manifest"
                    )
            examples.append(dict(example))
        batch = collate(examples)
        if not isinstance(batch, Mapping):
            raise TypeError("Probe collator must return a mapping")
        yield SSMaxProbeBatch(
            sample_ids=tuple(str(row["sample_id"]) for row in row_chunk),
            dataset_indices=tuple(int(row["dataset_index"]) for row in row_chunk),
            batch=batch,
        )


def capture_ssmax_probe_batches(
    collector: "SSMaxAttentionDiagnosticsCollector",
    batches: Iterable[SSMaxProbeBatch],
    *,
    forward_batch: Callable[[Mapping[str, Any]], Any],
) -> Dict[str, Any]:
    """Capture exactly one native no-cache model forward for every reconstructed batch.

    ``forward_batch`` normally calls the already-loaded train module's ``eval_batch`` method. It
    must not generate autoregressively or issue multiple model forwards. Token-prefix and category
    identities are revalidated by :meth:`SSMaxAttentionDiagnosticsCollector.capture_batch` before
    each forward, and every SSMax layer must execute exactly once.

    :param collector: Collector attached to the loaded model's language model.
    :param batches: Output from :func:`iter_ssmax_probe_batches`.
    :param forward_batch: Callable performing one evaluation forward on the supplied batch.
    :returns: Bounded rank-local state suitable for distributed object gathering.
    """

    required = {"input_ids", "token_type_ids", "loss_masks", "router_token_mask"}
    for probe_batch in batches:
        batch = probe_batch.batch
        missing = required - set(batch)
        if missing:
            raise ValueError(f"Collated SSMax probe batch lacks fields {sorted(missing)}")
        with collector.capture_batch(
            sample_ids=probe_batch.sample_ids,
            input_ids=batch["input_ids"],
            token_type_ids=batch["token_type_ids"],
            loss_masks=batch["loss_masks"],
            valid_tokens=batch["router_token_mask"],
            subsegment_ids=batch.get("subsegment_ids"),
            example_ids=batch.get("example_ids"),
        ):
            forward_batch(batch)
    return collector.export_state()


class _StreamingDistribution:
    """Mergeable exact moments with a deterministic bounded priority sample."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.count = 0
        self.total = 0.0
        self.total_squares = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf
        self._sample: list[tuple[int, float]] = []

    @staticmethod
    def _priorities(identity: str, count: int) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little")
        values = np.arange(count, dtype=np.uint64) + np.uint64(seed)
        values ^= values >> np.uint64(30)
        values *= np.uint64(0xBF58476D1CE4E5B9)
        values ^= values >> np.uint64(27)
        values *= np.uint64(0x94D049BB133111EB)
        values ^= values >> np.uint64(31)
        return values

    def add(self, values: torch.Tensor, *, identity: str) -> None:
        flat = values.detach().float().reshape(-1)
        if flat.numel() == 0:
            return
        if not bool(torch.isfinite(flat).all()):
            raise RuntimeError(f"Non-finite diagnostic values for {identity}")
        self.count += int(flat.numel())
        self.total += float(flat.double().sum().item())
        self.total_squares += float(flat.double().square().sum().item())
        self.minimum = min(self.minimum, float(flat.min().item()))
        self.maximum = max(self.maximum, float(flat.max().item()))
        cpu_values = flat.cpu().numpy().astype(np.float64, copy=False)
        priorities = self._priorities(identity, len(cpu_values))
        if len(cpu_values) > self.capacity:
            keep = np.argpartition(priorities, self.capacity - 1)[: self.capacity]
            priorities = priorities[keep]
            cpu_values = cpu_values[keep]
        for priority, value in zip(priorities.tolist(), cpu_values.tolist()):
            item = (-int(priority), float(value))
            if len(self._sample) < self.capacity:
                heapq.heappush(self._sample, item)
            elif item > self._sample[0]:
                heapq.heapreplace(self._sample, item)

    def merge(self, payload: Mapping[str, Any]) -> None:
        if payload.get("capacity") != self.capacity:
            raise ValueError("Cannot merge distributions with different sample capacities")
        count = payload.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Malformed distribution count")
        self.count += count
        self.total += float(payload["total"])
        self.total_squares += float(payload["total_squares"])
        if count:
            self.minimum = min(self.minimum, float(payload["minimum"]))
            self.maximum = max(self.maximum, float(payload["maximum"]))
        sample = payload.get("sample")
        if not isinstance(sample, Mapping) or sample.get("encoding") != "base64-u64-f64-v1":
            raise ValueError("Malformed encoded diagnostic priority sample")
        sample_count = sample.get("count")
        if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 0:
            raise ValueError("Malformed encoded diagnostic sample count")
        try:
            priorities = np.frombuffer(
                base64.b64decode(sample["priorities"], validate=True), dtype="<u8"
            )
            values = np.frombuffer(base64.b64decode(sample["values"], validate=True), dtype="<f8")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Could not decode diagnostic priority sample") from error
        if len(priorities) != sample_count or len(values) != sample_count:
            raise ValueError("Encoded diagnostic sample length differs from its count")
        for priority, value in zip(priorities.tolist(), values.tolist(), strict=True):
            item = (-int(priority), float(value))
            if len(self._sample) < self.capacity:
                heapq.heappush(self._sample, item)
            elif item > self._sample[0]:
                heapq.heapreplace(self._sample, item)

    def export(self) -> Dict[str, Any]:
        ordered = sorted([(-priority, value) for priority, value in self._sample])
        priorities = np.asarray([priority for priority, _ in ordered], dtype="<u8")
        values = np.asarray([value for _, value in ordered], dtype="<f8")
        return {
            "capacity": self.capacity,
            "count": self.count,
            "total": self.total,
            "total_squares": self.total_squares,
            "minimum": None if not self.count else self.minimum,
            "maximum": None if not self.count else self.maximum,
            "sample": {
                "encoding": "base64-u64-f64-v1",
                "count": len(ordered),
                "priorities": base64.b64encode(priorities.tobytes()).decode("ascii"),
                "values": base64.b64encode(values.tobytes()).decode("ascii"),
            },
        }

    def summary(self) -> Dict[str, Any]:
        if not self.count:
            return {
                "count": 0,
                "mean": None,
                "rms": None,
                "std": None,
                "min": None,
                "max": None,
                "sample_count": 0,
                "quantiles": {str(value): None for value in _DISTRIBUTION_QUANTILES},
            }
        mean = self.total / self.count
        variance = max(0.0, self.total_squares / self.count - mean * mean)
        sample_values = np.asarray([value for _, value in self._sample], dtype=np.float64)
        quantiles = np.quantile(sample_values, _DISTRIBUTION_QUANTILES, method="linear")
        return {
            "count": self.count,
            "mean": mean,
            "rms": math.sqrt(self.total_squares / self.count),
            "std": math.sqrt(variance),
            "min": self.minimum,
            "max": self.maximum,
            "sample_count": len(sample_values),
            "quantiles": {
                str(level): float(value)
                for level, value in zip(_DISTRIBUTION_QUANTILES, quantiles.tolist())
            },
        }


@dataclass
class _BatchState:
    sample_ids: Tuple[str, ...]
    token_type_ids: torch.Tensor
    loss_masks: torch.Tensor
    valid_tokens: torch.Tensor
    subsegment_ids: Optional[torch.Tensor]
    example_ids: Optional[torch.Tensor]
    rows: Tuple[Mapping[str, Any], ...]
    seen_layers: set[str] = field(default_factory=set)


class SSMaxAttentionDiagnosticsCollector:
    """Hook-based, memory-bounded collector for all SSMax attention layers in a model."""

    _MAGNITUDE_METRICS = (
        "q_pre_ssmax_rms",
        "q_post_ssmax_rms",
        "k_rms",
        "ssmax_effective_multiplier",
    )
    _BASE_ATTENTION_METRICS = (
        "logit",
        "absolute_logit",
        "normalized_entropy",
        "effective_context",
        "effective_context_fraction",
        "max_attention_probability",
        "visible_key_count",
    )
    _ATTENTION_METRICS = (
        *_BASE_ATTENTION_METRICS,
        *_ROUTING_MASS_METRICS,
        *_ROUTING_CHECK_METRICS,
        *_ROUTING_ARGMAX_METRICS,
    )

    def __init__(
        self,
        model: nn.Module,
        manifest: SSMaxProbeManifest,
        *,
        distribution_sample_capacity: int = 512,
        query_chunk_size: int = 8,
    ):
        """Attach diagnostics hooks to every Scalable-Softmax attention layer.

        :param model: Language model or enclosing multimodal model.
        :param manifest: Fixed validation/query manifest.
        :param distribution_sample_capacity: Deterministic per-metric/head/category reservoir.
        :param query_chunk_size: Maximum queries materialized in one logit matmul.
        """

        if distribution_sample_capacity <= 0 or query_chunk_size <= 0:
            raise ValueError("Diagnostic sample capacity and query chunk size must be positive")
        layers = {
            (name or "attention"): module
            for name, module in model.named_modules()
            if isinstance(module, Attention) and module.scalable_softmax
        }
        if not layers:
            raise OLMoConfigurationError("The model contains no Scalable-Softmax attention layers")
        self.model = model
        self.manifest = manifest
        self.layers = layers
        self.distribution_sample_capacity = distribution_sample_capacity
        self.query_chunk_size = query_chunk_size
        self._active_batch: Optional[_BatchState] = None
        self._closed = False
        self._seen_sample_ids: set[str] = set()
        self._layer_metadata: Dict[str, Dict[str, Any]] = {}
        self._distributions: Dict[Tuple[str, int, str, str], _StreamingDistribution] = {}
        self._base_q: Dict[str, torch.Tensor] = {}
        self._base_k: Dict[str, torch.Tensor] = {}
        self._handles = []
        for layer_name, layer in self.layers.items():
            self._handles.extend(self._register_layer_hooks(layer_name, layer))

    def _register_layer_hooks(self, layer_name: str, layer: Attention) -> list[Any]:
        q_source = layer.q_norm if layer.q_norm is not None else layer.w_q
        k_source = layer.k_norm if layer.k_norm is not None else layer.w_k

        def save_q(_module: nn.Module, _inputs: Tuple[Any, ...], output: torch.Tensor) -> None:
            self._base_q[layer_name] = output.detach()

        def save_k(_module: nn.Module, _inputs: Tuple[Any, ...], output: torch.Tensor) -> None:
            self._base_k[layer_name] = output.detach()

        def inspect_backend(
            _module: nn.Module, args: Tuple[Any, ...], kwargs: Mapping[str, Any]
        ) -> None:
            self._collect_layer(layer_name, layer, args, kwargs)

        return [
            q_source.register_forward_hook(save_q),
            k_source.register_forward_hook(save_k),
            layer.backend.register_forward_pre_hook(inspect_backend, with_kwargs=True),
        ]

    def close(self) -> None:
        """Remove all hooks. The collector cannot be reused after closing."""

        if self._active_batch is not None:
            raise RuntimeError("Cannot close diagnostics during an active batch")
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._closed = True

    def __enter__(self) -> "SSMaxAttentionDiagnosticsCollector":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    @staticmethod
    def _validate_batch_shapes(tensors: Mapping[str, Optional[torch.Tensor]]) -> Tuple[int, int]:
        present = {name: tensor for name, tensor in tensors.items() if tensor is not None}
        first = next(iter(present.values()), None)
        if first is None or first.ndim != 2:
            raise ValueError("Probe batch tensors must be two-dimensional")
        shape = tuple(first.shape)
        if any(tensor.ndim != 2 or tuple(tensor.shape) != shape for tensor in present.values()):
            raise ValueError("All probe batch tensors must have the same two-dimensional shape")
        return int(shape[0]), int(shape[1])

    def _validate_manifest_row(
        self,
        row: Mapping[str, Any],
        *,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        loss_masks: torch.Tensor,
        valid_tokens: torch.Tensor,
    ) -> None:
        valid = valid_tokens.detach().cpu().to(dtype=torch.bool)
        valid_length = int(valid.sum().item())
        expected_valid = torch.arange(valid.shape[0]) < valid_length
        if valid_length != row["valid_length"] or not torch.equal(valid, expected_valid):
            raise ValueError(f"Valid-token layout differs for probe sample {row['sample_id']}")
        actual_hashes = {
            "input_ids_sha256": _sequence_tensor_sha256(input_ids, valid_length),
            "token_type_ids_sha256": _sequence_tensor_sha256(token_type_ids, valid_length),
            "loss_masks_sha256": _sequence_tensor_sha256(loss_masks, valid_length),
        }
        differing = [name for name, value in actual_hashes.items() if value != row[name]]
        if differing:
            raise ValueError(
                f"Tokenized probe sample {row['sample_id']} differs in fields {differing}"
            )
        masks = _category_masks(token_type_ids, loss_masks, valid_tokens)
        for category, positions in row["query_positions"].items():
            if any(not bool(masks[category][position]) for position in positions):
                raise ValueError(f"Manifest query category drift for {row['sample_id']}:{category}")

    @contextmanager
    def capture_batch(
        self,
        *,
        sample_ids: Sequence[str],
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        loss_masks: torch.Tensor,
        valid_tokens: torch.Tensor,
        subsegment_ids: Optional[torch.Tensor] = None,
        example_ids: Optional[torch.Tensor] = None,
    ) -> Generator[None, None, None]:
        """Validate one manifest batch and capture its next model forward.

        The context must contain exactly one complete no-cache forward through every SSMax layer.
        Each manifest sample may be observed only once across the collector lifetime.
        """

        if self._closed:
            raise RuntimeError("The diagnostics collector is closed")
        if self._active_batch is not None:
            raise RuntimeError("Nested diagnostic batches are not supported")
        batch_size, _ = self._validate_batch_shapes(
            {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "loss_masks": loss_masks,
                "valid_tokens": valid_tokens,
                "subsegment_ids": subsegment_ids,
                "example_ids": example_ids,
            }
        )
        if len(sample_ids) != batch_size or len(set(sample_ids)) != batch_size:
            raise ValueError("sample_ids must uniquely identify every batch row")
        duplicate = set(sample_ids) & self._seen_sample_ids
        if duplicate:
            raise ValueError(f"Probe samples were evaluated more than once: {sorted(duplicate)}")
        unknown = set(sample_ids) - set(self.manifest.rows_by_sample_id)
        if unknown:
            raise ValueError(
                "Batch contains samples absent from the probe manifest: " f"{sorted(unknown)}"
            )
        rows = tuple(self.manifest.rows_by_sample_id[sample_id] for sample_id in sample_ids)
        for batch_index, row in enumerate(rows):
            self._validate_manifest_row(
                row,
                input_ids=input_ids[batch_index],
                token_type_ids=token_type_ids[batch_index],
                loss_masks=loss_masks[batch_index],
                valid_tokens=valid_tokens[batch_index],
            )
        for layer_name, layer in self.layers.items():
            if layer.kv_cache_manager is not None:
                raise OLMoConfigurationError(
                    f"SSMax diagnostics forbid KV caching, but {layer_name} has a cache manager"
                )
        self._active_batch = _BatchState(
            sample_ids=tuple(sample_ids),
            token_type_ids=token_type_ids,
            loss_masks=loss_masks,
            valid_tokens=valid_tokens,
            subsegment_ids=subsegment_ids,
            example_ids=example_ids,
            rows=rows,
        )
        self._base_q.clear()
        self._base_k.clear()
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            state = self._active_batch
            self._active_batch = None
            self._base_q.clear()
            self._base_k.clear()
            if not failed and state is not None:
                missing_layers = set(self.layers) - state.seen_layers
                if missing_layers:
                    raise RuntimeError(
                        f"Diagnostic forward did not visit SSMax layers {sorted(missing_layers)}"
                    )
                self._seen_sample_ids.update(state.sample_ids)

    def _distribution(
        self, layer_name: str, head: int, category: str, metric: str
    ) -> _StreamingDistribution:
        key = (layer_name, head, category, metric)
        if key not in self._distributions:
            self._distributions[key] = _StreamingDistribution(self.distribution_sample_capacity)
        return self._distributions[key]

    @staticmethod
    def _as_head_tensor(value: torch.Tensor, *, head_dim: int) -> torch.Tensor:
        if value.ndim == 4:
            return value
        if value.ndim != 3 or value.shape[-1] % head_dim:
            raise RuntimeError(f"Cannot reshape diagnostic Q/K tensor with shape {value.shape}")
        return value.view(value.shape[0], value.shape[1], -1, head_dim)

    @staticmethod
    def _ssmax_scale(layer: Attention, n_heads: int) -> torch.Tensor:
        if layer.ssmax_scale is None:
            raise RuntimeError("A scalable-softmax layer has no ssmax_scale parameter")
        scale: torch.Tensor = layer.ssmax_scale.detach()
        if isinstance(scale, DTensor):
            local = scale.to_local()
            scale = local if local.numel() == n_heads else scale.full_tensor()
        scale = scale.reshape(-1).float()
        if scale.numel() != n_heads:
            raise RuntimeError(f"SSMax scale has {scale.numel()} values for {n_heads} query heads")
        return scale

    def _record_magnitudes(
        self,
        *,
        layer_name: str,
        q_base: torch.Tensor,
        q_scaled: torch.Tensor,
        k_base: torch.Tensor,
        kv_head_by_query_head: torch.Tensor,
        ssmax_scale: torch.Tensor,
        state: _BatchState,
    ) -> None:
        head_dim = q_base.shape[-1]
        q_rms = q_base.float().square().mean(dim=-1).sqrt()
        q_scaled_rms = q_scaled.float().square().mean(dim=-1).sqrt()
        k_rms = k_base.float().square().mean(dim=-1).sqrt()
        for batch_index, row in enumerate(state.rows):
            for category in PROBE_CATEGORIES:
                positions = row["query_positions"][category]
                if not positions:
                    continue
                position_index = torch.tensor(positions, device=q_base.device, dtype=torch.long)
                for head in range(q_base.shape[2]):
                    kv_head = int(kv_head_by_query_head[head])
                    identity = f"{layer_name}\0{row['sample_id']}\0{category}\0{head}"
                    self._distribution(layer_name, head, category, "q_pre_ssmax_rms").add(
                        q_rms[batch_index, position_index, head], identity=identity + "\0q-base"
                    )
                    self._distribution(layer_name, head, category, "q_post_ssmax_rms").add(
                        q_scaled_rms[batch_index, position_index, head],
                        identity=identity + "\0q-scaled",
                    )
                    self._distribution(layer_name, head, category, "k_rms").add(
                        k_rms[batch_index, position_index, kv_head], identity=identity + "\0k"
                    )
                    visible_lengths = position_index + 1
                    multiplier = visible_lengths.log().float() * ssmax_scale[head].to(
                        position_index.device
                    )
                    self._distribution(
                        layer_name, head, category, "ssmax_effective_multiplier"
                    ).add(multiplier, identity=identity + "\0multiplier")
        if head_dim <= 0:
            raise RuntimeError("Attention head dimension must be positive")

    def _record_attention(
        self,
        *,
        layer_name: str,
        layer: Attention,
        q_scaled: torch.Tensor,
        k: torch.Tensor,
        kv_head_by_query_head: torch.Tensor,
        state: _BatchState,
    ) -> None:
        softmax_scale = layer.backend.scale
        if softmax_scale is None:
            softmax_scale = layer.head_dim**-0.5
        window_size = getattr(layer.backend, "window_size", (-1, -1))
        for batch_index, row in enumerate(state.rows):
            key_category_masks = {
                key_category: mask.to(device=q_scaled.device)
                for key_category, mask in _category_masks(
                    state.token_type_ids[batch_index],
                    state.loss_masks[batch_index],
                    state.valid_tokens[batch_index],
                ).items()
                if key_category in KEY_CATEGORIES
            }
            for category in PROBE_CATEGORIES:
                positions = row["query_positions"][category]
                for chunk_start in range(0, len(positions), self.query_chunk_size):
                    chunk = positions[chunk_start : chunk_start + self.query_chunk_size]
                    for query_position in chunk:
                        allowed = build_attention_allow_vector(
                            query_position=query_position,
                            valid_tokens=state.valid_tokens[batch_index],
                            token_type_ids=state.token_type_ids[batch_index],
                            subsegment_ids=(
                                None
                                if state.subsegment_ids is None
                                else state.subsegment_ids[batch_index]
                            ),
                            example_ids=(
                                None
                                if state.example_ids is None
                                else state.example_ids[batch_index]
                            ),
                            window_size=window_size,
                        ).to(device=q_scaled.device)
                        key_positions = torch.where(allowed)[0]
                        if key_positions.numel() == 0:
                            raise RuntimeError("A valid diagnostic query has no visible keys")
                        query = q_scaled[batch_index, query_position].float()
                        keys = k[batch_index, key_positions].float()
                        keys = keys[:, kv_head_by_query_head]
                        logits = torch.einsum("hd,khd->hk", query, keys) * softmax_scale
                        probabilities = torch.softmax(logits, dim=-1)
                        category_by_key = torch.stack(
                            [
                                key_category_masks[key_category][key_positions]
                                for key_category in KEY_CATEGORIES
                            ]
                        )
                        if not bool((category_by_key.sum(dim=0) == 1).all()):
                            raise RuntimeError(
                                "Image, prompt, and response categories do not exactly partition "
                                "the allowed diagnostic keys"
                            )
                        attention_mass_by_key_category = (
                            probabilities @ category_by_key.to(dtype=probabilities.dtype).T
                        )
                        attention_mass_to_allowed_keys = probabilities.sum(dim=-1)
                        attention_mass_category_total = attention_mass_by_key_category.sum(dim=-1)
                        argmax_key_by_category = category_by_key[
                            :, probabilities.argmax(dim=-1)
                        ].T.to(dtype=probabilities.dtype)
                        entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1)
                        visible_count = int(key_positions.numel())
                        if visible_count == 1:
                            normalized_entropy = torch.ones_like(entropy)
                        else:
                            normalized_entropy = entropy / math.log(visible_count)
                        effective_context = entropy.exp()
                        identity_base = (
                            f"{layer_name}\0{row['sample_id']}\0{category}\0q{query_position}"
                        )
                        for head in range(q_scaled.shape[2]):
                            metric_values = {
                                "logit": logits[head],
                                "absolute_logit": logits[head].abs(),
                                "normalized_entropy": normalized_entropy[head : head + 1],
                                "effective_context": effective_context[head : head + 1],
                                "effective_context_fraction": (
                                    effective_context[head : head + 1] / visible_count
                                ),
                                "max_attention_probability": probabilities[head].max().reshape(1),
                                "visible_key_count": logits.new_tensor([visible_count]),
                                "attention_mass_to_allowed_keys": (
                                    attention_mass_to_allowed_keys[head : head + 1]
                                ),
                                "attention_mass_normalization_error": (
                                    attention_mass_to_allowed_keys[head : head + 1] - 1.0
                                ).abs(),
                                "attention_mass_category_partition_error": (
                                    attention_mass_category_total[head : head + 1]
                                    - attention_mass_to_allowed_keys[head : head + 1]
                                ).abs(),
                            }
                            for key_category_index, key_category in enumerate(KEY_CATEGORIES):
                                metric_values[
                                    f"attention_mass_to_{key_category}_keys"
                                ] = attention_mass_by_key_category[
                                    head : head + 1, key_category_index
                                ]
                                metric_values[
                                    f"argmax_key_is_{key_category}"
                                ] = argmax_key_by_category[head : head + 1, key_category_index]
                            for metric, values in metric_values.items():
                                self._distribution(layer_name, head, category, metric).add(
                                    values,
                                    identity=f"{identity_base}\0h{head}\0{metric}",
                                )

    def _collect_layer(
        self,
        layer_name: str,
        layer: Attention,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> None:
        state = self._active_batch
        if state is None:
            return
        if layer_name in state.seen_layers:
            raise RuntimeError(f"SSMax layer {layer_name} ran more than once in one probe forward")
        if layer_name not in self._base_q or layer_name not in self._base_k:
            raise RuntimeError(f"Could not observe pre-SSMax Q/K for layer {layer_name}")
        if kwargs.get("kv_cache_manager") is not None or layer.kv_cache_manager is not None:
            raise OLMoConfigurationError("SSMax diagnostics do not support KV caching")
        if kwargs.get("cu_doc_lens") is not None:
            raise OLMoConfigurationError(
                "The fixed multimodal SSMax probe requires unpacked sequences without cu_doc_lens"
            )
        if not args or not isinstance(args[0], tuple) or len(args[0]) != 3:
            raise RuntimeError("Unexpected attention backend input; expected a Q/K/V tuple")
        q_scaled, k, _ = args[0]
        if not isinstance(q_scaled, torch.Tensor) or not isinstance(k, torch.Tensor):
            raise RuntimeError("Attention backend Q/K inputs must be tensors")
        q_base = self._as_head_tensor(self._base_q.pop(layer_name), head_dim=layer.head_dim)
        k_base = self._as_head_tensor(self._base_k.pop(layer_name), head_dim=layer.head_dim)
        if q_scaled.ndim != 4 or k.ndim != 4 or q_scaled.shape[:2] != k.shape[:2]:
            raise RuntimeError("SSMax diagnostics require same-length rank-local 4D Q/K tensors")
        if q_base.shape != q_scaled.shape or k_base.shape != k.shape:
            raise RuntimeError(
                "Observed pre/post Q/K shapes differ; tensor/context parallel probes are "
                "unsupported"
            )
        n_heads = int(q_scaled.shape[2])
        n_kv_heads = int(k.shape[2])
        if n_heads % n_kv_heads:
            raise RuntimeError("Query-head count must be divisible by KV-head count for GQA")
        group_size = n_heads // n_kv_heads
        kv_head_by_query_head = torch.arange(n_heads, device=q_scaled.device) // group_size
        ssmax_scale = self._ssmax_scale(layer, n_heads)
        metadata = {
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "gqa_group_size": group_size,
            "head_dim": layer.head_dim,
            "softmax_scale": (
                float(layer.backend.scale)
                if layer.backend.scale is not None
                else layer.head_dim**-0.5
            ),
            "ssmax_scale": [float(value) for value in ssmax_scale.cpu().tolist()],
            "qk_norm": layer.q_norm is not None and layer.k_norm is not None,
            "headwise_qk_norm": bool(layer.use_head_qk_norm),
        }
        previous_metadata = self._layer_metadata.setdefault(layer_name, metadata)
        if previous_metadata != metadata:
            raise RuntimeError(f"SSMax layer metadata changed during evaluation: {layer_name}")
        self._record_magnitudes(
            layer_name=layer_name,
            q_base=q_base,
            q_scaled=q_scaled,
            k_base=k_base,
            kv_head_by_query_head=kv_head_by_query_head,
            ssmax_scale=ssmax_scale,
            state=state,
        )
        self._record_attention(
            layer_name=layer_name,
            layer=layer,
            q_scaled=q_scaled,
            k=k,
            kv_head_by_query_head=kv_head_by_query_head,
            state=state,
        )
        state.seen_layers.add(layer_name)

    def export_state(self) -> Dict[str, Any]:
        """Export bounded mergeable state for rank-zero aggregation."""

        if self._active_batch is not None:
            raise RuntimeError("Cannot export diagnostics during an active batch")
        return {
            "format": _STATE_FORMAT,
            "version": _STATE_VERSION,
            "manifest_sha256": self.manifest.sha256,
            "distribution_sample_capacity": self.distribution_sample_capacity,
            "query_chunk_size": self.query_chunk_size,
            "metric_schema": {
                "magnitude": list(self._MAGNITUDE_METRICS),
                "attention": list(self._ATTENTION_METRICS),
            },
            "seen_sample_ids": sorted(self._seen_sample_ids),
            "layers": self._layer_metadata,
            "distributions": {
                "\0".join((layer, str(head), category, metric)): distribution.export()
                for (layer, head, category, metric), distribution in self._distributions.items()
            },
        }

    def finalize(self, *, checkpoint_identity: Mapping[str, Any]) -> Dict[str, Any]:
        """Finalize this collector's local state into a report."""

        return self.finalize_states(
            self.manifest,
            [self.export_state()],
            checkpoint_identity=checkpoint_identity,
        )

    @classmethod
    def finalize_states(
        cls,
        manifest: SSMaxProbeManifest,
        states: Sequence[Mapping[str, Any]],
        *,
        checkpoint_identity: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Merge disjoint rank-local states and create one canonical diagnostic report."""

        if not states:
            raise ValueError("At least one diagnostic state is required")
        manifest_ids = set(manifest.rows_by_sample_id)
        seen: set[str] = set()
        layer_metadata: Optional[Mapping[str, Any]] = None
        capacity: Optional[int] = None
        query_chunk_size: Optional[int] = None
        magnitude_metrics: Optional[Tuple[str, ...]] = None
        attention_metrics: Optional[Tuple[str, ...]] = None
        merged: Dict[str, _StreamingDistribution] = {}
        for state in states:
            if state.get("format") != _STATE_FORMAT or state.get("version") != _STATE_VERSION:
                raise ValueError("Unsupported SSMax diagnostics state")
            if state.get("manifest_sha256") != manifest.sha256:
                raise ValueError("Diagnostic state was collected from a different manifest")
            state_ids = set(state["seen_sample_ids"])
            overlap = seen & state_ids
            if overlap:
                raise ValueError(f"Diagnostic states overlap on samples {sorted(overlap)}")
            seen |= state_ids
            current_layers = state["layers"]
            raw_metric_schema = state.get("metric_schema")
            if raw_metric_schema is None:
                current_magnitude_metrics = cls._MAGNITUDE_METRICS
                current_attention_metrics = cls._BASE_ATTENTION_METRICS
            else:
                if not isinstance(raw_metric_schema, Mapping) or set(raw_metric_schema) != {
                    "magnitude",
                    "attention",
                }:
                    raise ValueError("Malformed SSMax diagnostics metric schema")
                raw_magnitude_metrics = raw_metric_schema["magnitude"]
                raw_attention_metrics = raw_metric_schema["attention"]
                if not isinstance(raw_magnitude_metrics, list) or not all(
                    isinstance(metric, str) for metric in raw_magnitude_metrics
                ):
                    raise ValueError("Malformed SSMax magnitude metric schema")
                if not isinstance(raw_attention_metrics, list) or not all(
                    isinstance(metric, str) for metric in raw_attention_metrics
                ):
                    raise ValueError("Malformed SSMax attention metric schema")
                current_magnitude_metrics = tuple(raw_magnitude_metrics)
                current_attention_metrics = tuple(raw_attention_metrics)
                supported_schemas = {
                    (cls._MAGNITUDE_METRICS, cls._BASE_ATTENTION_METRICS),
                    (cls._MAGNITUDE_METRICS, cls._ATTENTION_METRICS),
                }
                if (current_magnitude_metrics, current_attention_metrics) not in supported_schemas:
                    raise ValueError("Unsupported SSMax diagnostics metric schema")
            if layer_metadata is None:
                layer_metadata = current_layers
                capacity = int(state["distribution_sample_capacity"])
                query_chunk_size = int(state["query_chunk_size"])
                magnitude_metrics = current_magnitude_metrics
                attention_metrics = current_attention_metrics
            elif current_layers != layer_metadata:
                raise ValueError("Diagnostic states contain different layer metadata")
            elif int(state["distribution_sample_capacity"]) != capacity:
                raise ValueError("Diagnostic states use different distribution capacities")
            elif (
                current_magnitude_metrics != magnitude_metrics
                or current_attention_metrics != attention_metrics
            ):
                raise ValueError("Diagnostic states use different metric schemas")
            for key, payload in state["distributions"].items():
                parts = key.split("\0")
                if len(parts) != 4:
                    raise ValueError("Malformed SSMax diagnostic distribution key")
                _, _, category, metric = parts
                if category not in PROBE_CATEGORIES or metric not in {
                    *current_magnitude_metrics,
                    *current_attention_metrics,
                }:
                    raise ValueError("Diagnostic state contains an undeclared metric")
                distribution = merged.setdefault(key, _StreamingDistribution(int(capacity)))
                distribution.merge(payload)
        if seen != manifest_ids:
            raise ValueError(
                "Diagnostic states do not exactly cover the probe manifest; "
                f"missing={sorted(manifest_ids - seen)}, unexpected={sorted(seen - manifest_ids)}"
            )
        assert layer_metadata is not None and capacity is not None and query_chunk_size is not None
        assert magnitude_metrics is not None and attention_metrics is not None
        layers: Dict[str, Any] = {}
        for layer_name, metadata in sorted(layer_metadata.items()):
            heads = {}
            group_size = int(metadata["gqa_group_size"])
            for head in range(int(metadata["n_heads"])):
                categories = {}
                for category in PROBE_CATEGORIES:
                    metrics = {}
                    for metric in (*magnitude_metrics, *attention_metrics):
                        key = "\0".join((layer_name, str(head), category, metric))
                        distribution = merged.get(key, _StreamingDistribution(capacity))
                        metrics[metric] = distribution.summary()
                    categories[category] = metrics
                heads[str(head)] = {
                    "kv_head": head // group_size,
                    "ssmax_scale": metadata["ssmax_scale"][head],
                    "categories": categories,
                }
            layers[layer_name] = {**metadata, "heads": heads}
        has_key_routing = attention_metrics == cls._ATTENTION_METRICS
        protocol = {
            "name": (
                "fixed-multimodal-ssmax-attention-diagnostics-v2"
                if has_key_routing
                else "fixed-multimodal-ssmax-attention-diagnostics-v1"
            ),
            "manifest_sha256": manifest.sha256,
            "manifest_rows": len(manifest_ids),
            "categories": list(PROBE_CATEGORIES),
            "mask_rule": (
                "(causal_or_window | bidirectional_image) & subsegment & example & valid_padding"
            ),
            "gqa_rule": "query_head // (n_heads / n_kv_heads)",
            "kv_cache": False,
            "packed_cu_doc_lens": False,
            "query_chunk_size": query_chunk_size,
            "distribution_sample_capacity_per_layer_head_category_metric": capacity,
            "quantile_sampling": "deterministic-smallest-splitmix64-priority-v1",
            "moments": "exact-over-all-observations",
            "quantiles": "bounded-deterministic-sample",
            "singleton_normalized_entropy": 1.0,
        }
        if has_key_routing:
            protocol.update(
                {
                    "key_categories": list(KEY_CATEGORIES),
                    "routing_metrics": {
                        "probability_mass": list(_ROUTING_MASS_METRICS),
                        "integrity_checks": list(_ROUTING_CHECK_METRICS),
                        "argmax_indicators": list(_ROUTING_ARGMAX_METRICS),
                    },
                    "routing": (
                        "post-softmax mass and argmax destination over the exact allowed keys; "
                        "image/prompt/response form a partition"
                    ),
                }
            )
        protocol["sha256"] = _sha256_bytes(_canonical_json_bytes(protocol))
        report = {
            "format": _REPORT_FORMAT,
            "version": _REPORT_VERSION,
            "checkpoint": dict(checkpoint_identity),
            "manifest": manifest.as_dict(),
            "protocol": protocol,
            "coverage": {"sample_ids": sorted(seen), "count": len(seen)},
            "layers": layers,
        }
        report["report_sha256"] = _sha256_bytes(_canonical_json_bytes(report))
        return report


def _metric_value(
    report: Mapping[str, Any],
    layer: str,
    head: str,
    category: str,
    metric: str,
    statistic: str,
) -> Optional[float]:
    metrics = report["layers"][layer]["heads"][head]["categories"][category]
    metric_payload = metrics.get(metric)
    if not isinstance(metric_payload, Mapping):
        return None
    value = metric_payload.get(statistic)
    return None if value is None else float(value)


def _routing_comparison(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    layer: str,
    head: str,
    query_category: str,
) -> Optional[Dict[str, Any]]:
    destinations: Dict[str, Any] = {}
    for key_category in KEY_CATEGORIES:
        baseline_mass = _metric_value(
            baseline,
            layer,
            head,
            query_category,
            f"attention_mass_to_{key_category}_keys",
            "mean",
        )
        candidate_mass = _metric_value(
            candidate,
            layer,
            head,
            query_category,
            f"attention_mass_to_{key_category}_keys",
            "mean",
        )
        baseline_argmax = _metric_value(
            baseline,
            layer,
            head,
            query_category,
            f"argmax_key_is_{key_category}",
            "mean",
        )
        candidate_argmax = _metric_value(
            candidate,
            layer,
            head,
            query_category,
            f"argmax_key_is_{key_category}",
            "mean",
        )
        if any(
            value is None
            for value in (baseline_mass, candidate_mass, baseline_argmax, candidate_argmax)
        ):
            return None
        assert baseline_mass is not None and candidate_mass is not None
        assert baseline_argmax is not None and candidate_argmax is not None
        destinations[key_category] = {
            "attention_mass": {
                "baseline_mean": baseline_mass,
                "candidate_mean": candidate_mass,
                "delta": candidate_mass - baseline_mass,
            },
            "argmax_share": {
                "baseline": baseline_argmax,
                "candidate": candidate_argmax,
                "delta": candidate_argmax - baseline_argmax,
            },
        }

    check_statistics = {
        "attention_mass_to_allowed_keys": "mean",
        "attention_mass_normalization_error": "max",
        "attention_mass_category_partition_error": "max",
    }
    checks: Dict[str, Any] = {}
    for metric, statistic in check_statistics.items():
        baseline_value = _metric_value(baseline, layer, head, query_category, metric, statistic)
        candidate_value = _metric_value(candidate, layer, head, query_category, metric, statistic)
        if baseline_value is None or candidate_value is None:
            return None
        checks[metric] = {
            "statistic": statistic,
            "baseline": baseline_value,
            "candidate": candidate_value,
            "delta": candidate_value - baseline_value,
        }
    return {"destinations": destinations, "checks": checks}


def _validate_report(report: Mapping[str, Any], label: str) -> None:
    if report.get("format") != _REPORT_FORMAT or report.get("version") != _REPORT_VERSION:
        raise ValueError(f"{label} is not an SSMax attention diagnostics report")
    expected_sha = report.get("report_sha256")
    if not isinstance(expected_sha, str):
        raise ValueError(f"{label} lacks a report SHA-256")
    _validate_sha256(expected_sha, f"{label}.report_sha256")
    actual_sha = _sha256_bytes(
        _canonical_json_bytes(
            {key: value for key, value in report.items() if key != "report_sha256"}
        )
    )
    if actual_sha != expected_sha:
        raise ValueError(
            f"{label} report SHA mismatch: expected {expected_sha}, computed {actual_sha}"
        )


def validate_ssmax_attention_report(
    report: Mapping[str, Any], *, label: str = "attention diagnostics"
) -> None:
    """Validate the format and self-hash of an SSMax attention-diagnostics report.

    This public boundary lets phase-specific evidence validators reject a modified embedded
    report without manufacturing a comparison against an unrelated checkpoint.

    :param report: The report produced by
        :meth:`SSMaxAttentionDiagnosticsCollector.finalize_states`.
    :param label: A human-readable artifact label used in validation errors.
    """

    _validate_report(report, label)


def compare_ssmax_attention_reports(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    entropy_drop_threshold: float = 0.10,
    effective_context_fraction_ratio_threshold: float = 0.50,
    absolute_logit_q99_ratio_threshold: float = 2.0,
    q_magnitude_ratio_threshold: float = 2.0,
) -> Dict[str, Any]:
    """Compare matched reports and flag candidate attention-collapse signatures.

    Thresholds are triage heuristics, not promotion gates. The raw per-head deltas are retained so
    trajectories can be inspected across bridge, perception, joint, and later mid-training saves.
    """

    for report, label in ((baseline, "baseline"), (candidate, "candidate")):
        _validate_report(report, label)
    if baseline["protocol"]["sha256"] != candidate["protocol"]["sha256"]:
        raise ValueError("Reports use different diagnostic protocols")
    if set(baseline["layers"]) != set(candidate["layers"]):
        raise ValueError("Reports contain different attention layers")
    comparisons = []
    flags = []
    for layer in sorted(baseline["layers"]):
        baseline_heads = baseline["layers"][layer]["heads"]
        candidate_heads = candidate["layers"][layer]["heads"]
        if set(baseline_heads) != set(candidate_heads):
            raise ValueError(f"Reports contain different heads in {layer}")
        for head in sorted(baseline_heads, key=int):
            for category in PROBE_CATEGORIES:
                base_entropy = _metric_value(
                    baseline, layer, head, category, "normalized_entropy", "mean"
                )
                cand_entropy = _metric_value(
                    candidate, layer, head, category, "normalized_entropy", "mean"
                )
                base_effective = _metric_value(
                    baseline,
                    layer,
                    head,
                    category,
                    "effective_context_fraction",
                    "mean",
                )
                cand_effective = _metric_value(
                    candidate,
                    layer,
                    head,
                    category,
                    "effective_context_fraction",
                    "mean",
                )
                base_logit = baseline["layers"][layer]["heads"][head]["categories"][category][
                    "absolute_logit"
                ]["quantiles"]["0.99"]
                cand_logit = candidate["layers"][layer]["heads"][head]["categories"][category][
                    "absolute_logit"
                ]["quantiles"]["0.99"]
                base_q = _metric_value(baseline, layer, head, category, "q_post_ssmax_rms", "mean")
                cand_q = _metric_value(candidate, layer, head, category, "q_post_ssmax_rms", "mean")
                optional_values = (
                    base_entropy,
                    cand_entropy,
                    base_effective,
                    cand_effective,
                    base_logit,
                    cand_logit,
                    base_q,
                    cand_q,
                )
                if any(value is None for value in optional_values):
                    continue
                assert base_entropy is not None and cand_entropy is not None
                assert base_effective is not None and cand_effective is not None
                assert base_q is not None and cand_q is not None
                entropy_delta = float(cand_entropy) - float(base_entropy)
                effective_ratio = float(cand_effective) / max(float(base_effective), 1e-30)
                logit_ratio = float(cand_logit) / max(float(base_logit), 1e-30)
                q_ratio = float(cand_q) / max(float(base_q), 1e-30)
                record = {
                    "layer": layer,
                    "head": int(head),
                    "category": category,
                    "normalized_entropy_delta": entropy_delta,
                    "effective_context_fraction_ratio": effective_ratio,
                    "absolute_logit_q99_ratio": logit_ratio,
                    "q_post_ssmax_rms_ratio": q_ratio,
                    "ssmax_scale_delta": (
                        float(candidate_heads[head]["ssmax_scale"])
                        - float(baseline_heads[head]["ssmax_scale"])
                    ),
                }
                routing = _routing_comparison(
                    baseline,
                    candidate,
                    layer=layer,
                    head=head,
                    query_category=category,
                )
                if routing is not None:
                    record["key_routing"] = routing
                comparisons.append(record)
                reasons = []
                if entropy_delta <= -entropy_drop_threshold:
                    reasons.append("normalized_entropy_drop")
                if effective_ratio <= effective_context_fraction_ratio_threshold:
                    reasons.append("effective_context_fraction_contraction")
                if logit_ratio >= absolute_logit_q99_ratio_threshold:
                    reasons.append("absolute_logit_q99_growth")
                if q_ratio >= q_magnitude_ratio_threshold:
                    reasons.append("post_ssmax_query_magnitude_growth")
                if reasons:
                    flags.append({**record, "reasons": reasons})
    result = {
        "format": "ssmax_attention_diagnostics_comparison",
        "version": 1,
        "baseline_report_sha256": baseline["report_sha256"],
        "candidate_report_sha256": candidate["report_sha256"],
        "thresholds": {
            "entropy_drop": entropy_drop_threshold,
            "effective_context_fraction_ratio": effective_context_fraction_ratio_threshold,
            "absolute_logit_q99_ratio": absolute_logit_q99_ratio_threshold,
            "q_post_ssmax_rms_ratio": q_magnitude_ratio_threshold,
        },
        "comparisons": comparisons,
        "flags": flags,
        "flag_count": len(flags),
        "interpretation": (
            "Heuristic attention-collapse triage and descriptive key-category routing only; "
            "downstream BLINK jigsaw and MathVista geometry trajectories remain the outcome "
            "measures."
        ),
    }
    result["comparison_sha256"] = _sha256_bytes(_canonical_json_bytes(result))
    return result
