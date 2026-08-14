#!/usr/bin/env python
"""Independently audit the nine-source Vision Alignment joint probe catalog.

The auditor raw-pins and strict-parses every referenced artifact, rebuilds all eight selected
visual adapters plus native replay, and re-derives every probe row and image digest.  Only a
zero-error, zero-truncation panel with positive supervised loss mass is eligible for conversion
from the checked-in nine-way loss-mass targets to example-sampling probabilities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
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
)

from olmo_core.data.multimodal.mixtures.vision_alignment import (
    VisionAlignmentMixtureConfig,
    expected_loss_mass,
    sampling_weights_from_loss_mass,
)
from olmo_core.data.multimodal.native_text_replay import (
    NativeTextReplayDataset,
    NativeTextReplayManifest,
    NativeTextReplayVerificationReceipt,
)
from olmo_core.data.multimodal.vision_alignment_joint_provenance import (
    JOINT_VISUAL_UNBOUNDED_SEQUENCE_LENGTH,
    JointVisualProjectionManifest,
    build_selected_joint_dataset,
    joint_alignment_runtime_implementation_inventory,
    joint_alignment_runtime_registry_sha256,
    load_joint_visual_projection_manifest,
    validate_joint_live_example,
    validate_joint_probe_record,
    validate_joint_unbounded_dataset_identity,
)
from olmo_core.data.multimodal.vision_alignment_joint_sources import (
    JOINT_VISUAL_SOURCE_NAMES,
    VISION_ALIGNMENT_JOINT_SOURCE_CATALOG_VERSION,
    VISION_ALIGNMENT_JOINT_SOURCE_REGISTRY_VERSION,
    build_vision_alignment_joint_dataset_config,
    vision_alignment_joint_adapter_projection_sha256,
)
from olmo_core.data.multimodal.vision_alignment_sources import (
    VISION_ALIGNMENT_FORMATTER_VERSION,
    VISION_ALIGNMENT_PROBE_SELECTION_ALGORITHM,
    VISION_ALIGNMENT_RECIPE_VERSION,
    VISION_ALIGNMENT_TOKENIZER_FINGERPRINT,
    VISION_ALIGNMENT_TOKENIZER_ID,
    VISION_ALIGNMENT_TOKENIZER_REVISION,
    load_pinned_vision_alignment_tokenizer,
    runtime_dataset_fingerprint,
    select_deterministic_probe_indices,
    serialized_probe_record,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.vision import Molmo2TokenIds
from scripts.data import audit_vision_alignment_mix as shared_audit

CATALOG_FORMAT = "vision_alignment_joint_preprocessed_source_catalog"
CATALOG_VERSION = VISION_ALIGNMENT_JOINT_SOURCE_CATALOG_VERSION
AUDIT_FORMAT = "vision_alignment_joint_source_audit"
AUDIT_VERSION = 1
JOINT_PROBE_FORMAT = "vision_alignment_joint_runtime_probe"
JOINT_PROBE_VERSION = 1
JOINT_PROBE_SEED = 6198
JOINT_VISUAL_PROBE_INDICES = 256
JOINT_VISUAL_PROBE_EPOCHS = (0, 1, 2, 3)
JOINT_NATIVE_PROBE_INDICES = 1024
JOINT_NATIVE_PROBE_EPOCHS = (0,)
JOINT_SEQUENCE_LENGTH = 8192
UNBOUNDED_SEQUENCE_LENGTH = JOINT_VISUAL_UNBOUNDED_SEQUENCE_LENGTH
DEFAULT_HF_CACHE_DIR = "/weka/oe-training-default/rustin/hf-cache/hub"
DEFAULT_GUARD_WORKERS = 4
MAX_GUARD_WORKERS = 16
PROGRESS_INTERVAL_ROWS = 128
EXPORTER_IMPLEMENTATION_PATH = "src/scripts/data/export_vision_alignment_joint_probe.py"
AUDITOR_IMPLEMENTATION_PATH = "src/scripts/data/audit_vision_alignment_joint_mix.py"
JOINT_SOURCE_NAMES = tuple(sorted((*JOINT_VISUAL_SOURCE_NAMES, "native_text_replay")))

ProgressCallback = Callable[[Mapping[str, Any]], None]

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ROOT_FIELDS = frozenset(
    {
        "format",
        "version",
        "phase",
        "recipe_version",
        "formatter_version",
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
        "sources",
        "content_sha256",
    }
)
_IMPLEMENTATION_FIELDS = frozenset({"path", "sha256"})
_VISUAL_PROJECTION_FIELDS = frozenset({"path", "raw_sha256", "content_sha256"})
_NATIVE_MANIFEST_FIELDS = frozenset({"path", "raw_sha256", "content_fingerprint"})
_RECEIPT_FIELDS = frozenset({"path", "sha256"})
_PREPROCESSING_FIELDS = frozenset({"visual", "native_text_replay_fingerprint"})
_PROBE_FIELDS = frozenset(
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
_PROBE_KIND_FIELDS = frozenset({"unique_indices", "epochs", "rows_per_source"})
_SOURCE_FIELDS = frozenset(
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
    }
)


@dataclass(frozen=True)
class _RuntimeProbeSource:
    dataset: Any
    unbounded_dataset: Optional[Any]
    token_ids: Molmo2TokenIds


def _validate_workers(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_GUARD_WORKERS:
        raise ValueError(f"workers must be an integer in [1, {MAX_GUARD_WORKERS}]")
    return value


def _emit_progress(
    callback: Optional[ProgressCallback],
    *,
    event: str,
    started_at: float,
    source_name: Optional[str] = None,
    **fields: Any,
) -> None:
    if callback is None:
        return
    payload: Dict[str, Any] = {
        "phase": "joint_mix_audit",
        "event": event,
        "elapsed_seconds": round(time.monotonic() - started_at, 3),
    }
    if source_name is not None:
        payload["source"] = source_name
    payload.update(fields)
    try:
        callback(payload)
    except Exception:
        # Telemetry must never alter immutable evidence or audit behavior.
        pass


def _stderr_progress(payload: Mapping[str, Any]) -> None:
    try:
        print(json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)
    except OSError:
        # Progress is operational telemetry, never part of the immutable audit bytes.
        pass


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file_handle:
            while chunk := file_handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"Could not hash joint audit input {path}: {error}") from error
    return digest.hexdigest()


def _strict_json_object(pairs: Sequence[tuple[str, Any]]) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"JSON repeats key {key!r}")
        value[key] = item
    return value


def _exact_mapping(value: Any, fields: frozenset[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    actual = set(value)
    if actual != fields:
        raise ValueError(
            f"{name} fields differ: missing={sorted(fields - actual)}, "
            f"extra={sorted(actual - fields)}"
        )
    return value


def _integer(value: Any, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _absolute_file(value: Any, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty absolute path")
    unresolved = Path(value).expanduser()
    if not unresolved.is_absolute() or ".." in unresolved.parts:
        raise ValueError(f"{name} must be an absolute normalized path without traversal")
    path = unresolved.resolve()
    if str(path) != value or not path.is_file():
        raise ValueError(f"{name} is not a normalized existing file: {value!r}")
    return path


def _catalog_local_path(catalog: Path, value: Any, *, name: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute() or "://" in value:
        raise ValueError(f"{name} must be a non-empty local relative path")
    unresolved = Path(value)
    if ".." in unresolved.parts:
        raise ValueError(f"{name} must not contain path traversal")
    path = (catalog.parent / unresolved).resolve()
    if path.parent != catalog.parent or not path.is_file():
        raise ValueError(f"{name} is not a catalog-local file")
    return path


def _load_catalog(path: Path) -> tuple[bytes, Mapping[str, Any]]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw, object_pairs_hook=_strict_json_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Invalid joint source catalog {path}: {error}") from error
    return raw, _exact_mapping(value, _ROOT_FIELDS, name="joint source catalog")


def _load_jsonl(path: Path, *, expected_sha256: str) -> tuple[bytes, list[Mapping[str, Any]]]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError(f"Could not read joint probe {path}: {error}") from error
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"Joint source {path.stem!r} probe-file SHA-256 differs")
    records = []
    for ordinal, line in enumerate(raw.splitlines()):
        try:
            value = json.loads(line, object_pairs_hook=_strict_json_object)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"Joint probe {path} row {ordinal} is malformed: {error}") from error
        if not isinstance(value, Mapping):
            raise ValueError(f"Joint probe {path} row {ordinal} must be an object")
        records.append(value)
    canonical_raw = b"".join(_canonical_bytes(record) + b"\n" for record in records)
    if raw != canonical_raw:
        raise ValueError(f"Joint source {path.stem!r} probe is not exact canonical JSONL")
    return raw, records


def _expected_probe_policy() -> Dict[str, Any]:
    return {
        "format": JOINT_PROBE_FORMAT,
        "version": JOINT_PROBE_VERSION,
        "selection_algorithm": VISION_ALIGNMENT_PROBE_SELECTION_ALGORITHM,
        "seed": JOINT_PROBE_SEED,
        "visual": {
            "unique_indices": JOINT_VISUAL_PROBE_INDICES,
            "epochs": list(JOINT_VISUAL_PROBE_EPOCHS),
            "rows_per_source": JOINT_VISUAL_PROBE_INDICES * len(JOINT_VISUAL_PROBE_EPOCHS),
        },
        "native_text_replay": {
            "unique_indices": JOINT_NATIVE_PROBE_INDICES,
            "epochs": list(JOINT_NATIVE_PROBE_EPOCHS),
            "rows_per_source": JOINT_NATIVE_PROBE_INDICES * len(JOINT_NATIVE_PROBE_EPOCHS),
        },
        "sequence_length": JOINT_SEQUENCE_LENGTH,
        "truncation_policy": "forbid-raw-length-above-sequence-length-v1",
    }


def _validate_probe_policy(value: Any) -> Mapping[str, Any]:
    probe = _exact_mapping(value, _PROBE_FIELDS, name="catalog.probe")
    for kind in ("visual", "native_text_replay"):
        _exact_mapping(probe[kind], _PROBE_KIND_FIELDS, name=f"catalog.probe.{kind}")
    if _canonical_bytes(probe) != _canonical_bytes(_expected_probe_policy()):
        raise ValueError("Joint probe policy differs from the production version-1 policy")
    return probe


def _dataset_get(dataset: Any, index: int, epoch: int) -> Mapping[str, Any]:
    get = getattr(dataset, "get", None)
    value = get(index, epoch) if callable(get) else dataset[index]
    if not isinstance(value, Mapping):
        raise ValueError(f"Live joint row {index} at epoch {epoch} must be an object")
    return value


def _sequence_length(example: Mapping[str, Any], *, name: str) -> int:
    input_ids = example.get("input_ids")
    if input_ids is None:
        raise ValueError(f"{name} is missing input_ids")
    try:
        length = len(input_ids)
    except TypeError as error:
        raise ValueError(f"{name} input_ids must be one-dimensional array-like data") from error
    if isinstance(length, bool) or length < 1:
        raise ValueError(f"{name} input_ids must not be empty")
    return length


def _probe_image_digest(dataset: Any, indices: Sequence[int], *, source_name: str) -> str:
    validate = getattr(dataset, "validate_image_content", None)
    if not callable(validate):
        raise ValueError(f"Joint visual source {source_name!r} lacks image-content validation")
    digest = validate(indices)
    return _sha256(digest, name=f"{source_name} live image-content digest")


def _live_probe_record(
    runtime: _RuntimeProbeSource,
    *,
    source_name: str,
    kind: str,
    dataset_index: int,
    epoch: int,
) -> Dict[str, Any]:
    example = _dataset_get(runtime.dataset, dataset_index, epoch)
    validate_joint_live_example(
        example,
        source_name=source_name,
        source_kind=kind,
        token_ids=runtime.token_ids,
    )
    record = serialized_probe_record(
        example,
        source_name=source_name,
        dataset_index=dataset_index,
        epoch=epoch,
    )
    if kind == "visual":
        if runtime.unbounded_dataset is None:
            raise ValueError(f"Joint visual source {source_name!r} lacks an unbounded adapter")
        raw_indices = getattr(runtime.dataset, "indices", None)
        if not isinstance(raw_indices, tuple) or len(raw_indices) != len(runtime.dataset):
            raise ValueError(f"Joint visual source {source_name!r} lacks exact projected indices")
        raw_index = raw_indices[dataset_index]
        if isinstance(raw_index, bool) or not isinstance(raw_index, int) or raw_index < 0:
            raise ValueError(f"Joint visual source {source_name!r} has an invalid projected row")
        raw_example = _dataset_get(runtime.unbounded_dataset, raw_index, epoch)
        validate_joint_live_example(
            raw_example,
            source_name=source_name,
            source_kind=kind,
            token_ids=runtime.token_ids,
        )
        raw_length = _sequence_length(
            raw_example, name=f"unbounded {source_name}/{dataset_index}/{epoch}"
        )
        if raw_length > JOINT_SEQUENCE_LENGTH:
            raise ValueError(
                f"Joint visual source {source_name!r} row {dataset_index} epoch {epoch} "
                f"requires {raw_length} tokens; truncation is forbidden"
            )
        raw_record = serialized_probe_record(
            raw_example,
            source_name=source_name,
            dataset_index=dataset_index,
            epoch=epoch,
        )
        if _canonical_bytes(raw_record) != _canonical_bytes(record):
            raise ValueError(
                f"Joint visual source {source_name!r} row {dataset_index} epoch {epoch} "
                "differs from its unbounded serialization"
            )
    elif kind == "native_text_replay":
        if runtime.unbounded_dataset is not None:
            raise ValueError("Native replay must not name an unbounded visual adapter")
        raw_length = _sequence_length(example, name=f"native_text_replay/{dataset_index}/{epoch}")
        if raw_length != JOINT_SEQUENCE_LENGTH:
            raise ValueError(
                f"Native replay row {dataset_index} has {raw_length} tokens, expected "
                f"{JOINT_SEQUENCE_LENGTH}"
            )
        if record["image_crops"] != 0 or record["pooled_tokens"] != 0:
            raise ValueError("Native replay rows must not contain visual inputs")
    else:
        raise ValueError(f"Unknown joint source kind {kind!r}")
    record["raw_sequence_length"] = raw_length
    record["truncated"] = False
    validate_joint_probe_record(
        record,
        source_name=source_name,
        source_kind=kind,
        expected_index=dataset_index,
        expected_epoch=epoch,
        sequence_length=JOINT_SEQUENCE_LENGTH,
        token_ids=runtime.token_ids,
    )
    return record


def _ordered_live_records(
    runtime: _RuntimeProbeSource,
    *,
    source_name: str,
    kind: str,
    work_items: Iterable[tuple[int, int]],
    workers: int,
) -> Generator[Dict[str, Any], None, None]:
    """Rebuild independent rows concurrently and yield exact catalog order."""

    def build(item: tuple[int, int]) -> Dict[str, Any]:
        dataset_index, epoch = item
        return _live_probe_record(
            runtime,
            source_name=source_name,
            kind=kind,
            dataset_index=dataset_index,
            epoch=epoch,
        )

    if workers == 1:
        yield from map(build, work_items)
        return

    # Ordered yielding preserves floating-point accumulation and report bytes exactly. Waiting
    # during shutdown ensures no runtime source can remain active past an audit failure boundary.
    items = iter(work_items)
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="joint-mix-audit")
    futures: deque[Future[Dict[str, Any]]] = deque()
    try:
        for _ in range(workers):
            try:
                futures.append(executor.submit(build, next(items)))
            except StopIteration:
                break
        while futures:
            result = futures.popleft().result()
            try:
                futures.append(executor.submit(build, next(items)))
            except StopIteration:
                pass
            yield result
    finally:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def _build_unbounded_visual_dataset(
    projection: JointVisualProjectionManifest,
    tokenizer: Any,
    token_ids: Any,
    source_name: str,
) -> Any:
    selection = projection.selection(source_name, "train")
    config = build_vision_alignment_joint_dataset_config(
        projection.source_spec,
        token_ids,
        source_name,
        split=selection.physical_split,
    )
    try:
        unbounded_config = replace(config, max_sequence_length=UNBOUNDED_SEQUENCE_LENGTH)
    except TypeError as error:
        raise ValueError(f"Joint visual config {source_name!r} is not replaceable") from error
    if (
        vision_alignment_joint_adapter_projection_sha256(unbounded_config)
        != selection.adapter_projection_sha256
    ):
        raise ValueError(f"Unbounded joint adapter projection differs for {source_name!r}")
    dataset = unbounded_config.build(tokenizer)
    validate = getattr(dataset, "validate_required_annotations", None)
    if not callable(validate):
        raise ValueError(f"Unbounded joint source {source_name!r} lacks annotation validation")
    validate()
    validate_joint_unbounded_dataset_identity(
        dataset,
        source_name=source_name,
        selection=selection,
        max_sequence_length=UNBOUNDED_SEQUENCE_LENGTH,
    )
    return dataset


class _LazyRuntimeProbeSources(Mapping[str, _RuntimeProbeSource]):
    """Build one runtime source per lookup without retaining earlier heavy adapters."""

    def __init__(
        self,
        projection: JointVisualProjectionManifest,
        native_manifest: NativeTextReplayManifest,
        receipt: NativeTextReplayVerificationReceipt,
        *,
        tokenizer: Any,
        token_ids: Molmo2TokenIds,
    ):
        self.projection = projection
        self.native_manifest = native_manifest
        self.receipt = receipt
        self.tokenizer = tokenizer
        self.token_ids = token_ids

    def __len__(self) -> int:
        return len(JOINT_SOURCE_NAMES)

    def __iter__(self) -> Iterator[str]:
        return iter(JOINT_SOURCE_NAMES)

    def __getitem__(self, source_name: str) -> _RuntimeProbeSource:
        if source_name not in JOINT_SOURCE_NAMES:
            raise KeyError(source_name)
        if source_name == "native_text_replay":
            native = NativeTextReplayDataset(
                self.native_manifest.path,
                expected_fingerprint=self.native_manifest.content_fingerprint,
                verification_receipt_path=self.receipt.path,
                expected_verification_receipt_sha256=self.receipt.receipt_sha256,
            )
            native.validate_tokenizer(self.tokenizer)
            return _RuntimeProbeSource(native, None, self.token_ids)

        selected = build_selected_joint_dataset(
            self.projection,
            self.tokenizer,
            self.token_ids,
            source_name,
            logical_split="train",
            validate_required_annotations=False,
        )
        unbounded = _build_unbounded_visual_dataset(
            self.projection,
            self.tokenizer,
            self.token_ids,
            source_name,
        )
        return _RuntimeProbeSource(selected, unbounded, self.token_ids)


def _build_runtime_sources(
    projection: JointVisualProjectionManifest,
    native_manifest: NativeTextReplayManifest,
    receipt: NativeTextReplayVerificationReceipt,
    *,
    tokenizer: Any,
    token_ids: Molmo2TokenIds,
) -> Mapping[str, _RuntimeProbeSource]:
    parent_spec = projection.source_spec.perception_spec
    if (
        parent_spec.tokenizer_id != VISION_ALIGNMENT_TOKENIZER_ID
        or parent_spec.tokenizer_revision != VISION_ALIGNMENT_TOKENIZER_REVISION
        or parent_spec.tokenizer_fingerprint != VISION_ALIGNMENT_TOKENIZER_FINGERPRINT
    ):
        raise ValueError("Joint visual projection does not bind the pinned runtime tokenizer")
    return _LazyRuntimeProbeSources(
        projection,
        native_manifest,
        receipt,
        tokenizer=tokenizer,
        token_ids=token_ids,
    )


def _load_native_manifest(path: Path) -> NativeTextReplayManifest:
    return NativeTextReplayManifest.load(path)


def _load_native_receipt(
    path: Path, *, expected_sha256: str
) -> NativeTextReplayVerificationReceipt:
    return NativeTextReplayVerificationReceipt.load(path, expected_sha256=expected_sha256)


def _fresh_native_runtime_evidence(
    native_manifest: NativeTextReplayManifest,
    receipt: NativeTextReplayVerificationReceipt,
    *,
    expected_size: int,
) -> NativeTextReplayDataset:
    """Rebuild native replay so closing validation repeats every source-file stat check."""
    dataset = NativeTextReplayDataset(
        native_manifest.path,
        expected_fingerprint=native_manifest.content_fingerprint,
        verification_receipt_path=receipt.path,
        expected_verification_receipt_sha256=receipt.receipt_sha256,
        validate_source_files=True,
    )
    if (
        len(dataset) != expected_size
        or len(dataset) != native_manifest.num_windows
        or dataset.sequence_length != JOINT_SEQUENCE_LENGTH
        or runtime_dataset_fingerprint(dataset) != native_manifest.content_fingerprint
        or sum(dataset.source_counts.values()) != expected_size
    ):
        raise ValueError("Fresh native replay runtime size or identity evidence differs")
    return dataset


def _closing_validate_inputs(
    *,
    projection: JointVisualProjectionManifest,
    native_manifest: NativeTextReplayManifest,
    receipt: NativeTextReplayVerificationReceipt,
    expected_native_size: int,
    token_ids: Molmo2TokenIds,
) -> None:
    """Re-load every external identity and fresh-stat native replay before returning."""
    if _sha256_file(native_manifest.path) != native_manifest.manifest_sha256:
        raise ValueError("Native train manifest changed during closing validation")
    closing_native = _load_native_manifest(native_manifest.path)
    if (
        closing_native.manifest_sha256 != native_manifest.manifest_sha256
        or closing_native.content_fingerprint != native_manifest.content_fingerprint
    ):
        raise ValueError("Native train manifest identity changed during closing validation")
    closing_receipt = _load_native_receipt(receipt.path, expected_sha256=receipt.receipt_sha256)
    closing_receipt.validate_manifest(closing_native)
    if (
        closing_native.provenance.get("verification_receipt_sha256")
        != closing_receipt.receipt_sha256
    ):
        raise ValueError("Closing native manifest does not bind its verification receipt")
    _fresh_native_runtime_evidence(
        closing_native,
        closing_receipt,
        expected_size=expected_native_size,
    )
    # Make reviewed producer identities the final dependencies observed after the potentially
    # long native source-stat pass.
    closing_receipt = _load_native_receipt(receipt.path, expected_sha256=receipt.receipt_sha256)
    closing_receipt.validate_manifest(closing_native)
    closing_projection = load_joint_visual_projection_manifest(
        projection.path,
        expected_token_ids=token_ids,
        expected_sha256=projection.raw_sha256,
    )
    if (
        closing_projection.raw_sha256 != projection.raw_sha256
        or closing_projection.content_sha256 != projection.content_sha256
        or _canonical_bytes(closing_projection.source_spec.as_canonical_dict())
        != _canonical_bytes(projection.source_spec.as_canonical_dict())
    ):
        raise ValueError("Joint visual projection identity changed during closing validation")


def audit_joint_catalog(
    path: str | Path,
    *,
    expected_catalog_sha256: Optional[str] = None,
    hf_cache_dir: str = DEFAULT_HF_CACHE_DIR,
    workers: int = 1,
    progress: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """Audit a strict joint catalog and return its canonical version-1 report.

    :param path: Exact catalog path.
    :param expected_catalog_sha256: Optional external raw-byte catalog pin.
    :param hf_cache_dir: Pinned-tokenizer cache directory.
    :param workers: Bounded row-rebuild concurrency. Reduction always follows catalog order.
    :param progress: Optional structured timing callback excluded from report evidence.
    :returns: Canonical version-1 audit report.
    :raises ValueError: If any catalog, runtime, row, image, or implementation evidence differs.
    """
    workers = _validate_workers(workers)
    audit_started_at = time.monotonic()
    _emit_progress(
        progress,
        event="phase_start",
        started_at=audit_started_at,
        sources_total=len(JOINT_SOURCE_NAMES),
        workers=workers,
    )
    catalog_path = Path(path).expanduser().resolve()
    catalog_raw, catalog = _load_catalog(catalog_path)
    catalog_sha256 = hashlib.sha256(catalog_raw).hexdigest()
    if expected_catalog_sha256 is not None and catalog_sha256 != _sha256(
        expected_catalog_sha256, name="expected catalog SHA-256"
    ):
        raise ValueError("Joint catalog differs from its external raw SHA-256 pin")
    if (
        catalog["format"] != CATALOG_FORMAT
        or _integer(catalog["version"], name="catalog.version", minimum=1) != CATALOG_VERSION
        or catalog["phase"] != "joint"
        or _integer(catalog["recipe_version"], name="catalog.recipe_version", minimum=1)
        != VISION_ALIGNMENT_RECIPE_VERSION
        or catalog["formatter_version"] != VISION_ALIGNMENT_FORMATTER_VERSION
    ):
        raise ValueError("Joint catalog phase, recipe, or formatter identity differs")
    declared_content_sha = _sha256(catalog["content_sha256"], name="catalog.content_sha256")
    unsigned_catalog = dict(catalog)
    unsigned_catalog.pop("content_sha256")
    if _canonical_sha256(unsigned_catalog) != declared_content_sha:
        raise ValueError("Joint catalog content SHA-256 differs")

    registry_sha256 = joint_alignment_runtime_registry_sha256()
    implementation_inventory = joint_alignment_runtime_implementation_inventory()
    if (
        _integer(
            catalog["source_registry_version"],
            name="catalog.source_registry_version",
            minimum=1,
        )
        != VISION_ALIGNMENT_JOINT_SOURCE_REGISTRY_VERSION
        or catalog["source_registry_sha256"] != registry_sha256
        or _canonical_bytes(catalog["source_implementation_inventory"])
        != _canonical_bytes(implementation_inventory)
    ):
        raise ValueError("Joint catalog source implementation identity differs")

    exporter_ref = _exact_mapping(
        catalog["exporter_implementation"],
        _IMPLEMENTATION_FIELDS,
        name="catalog.exporter_implementation",
    )
    if exporter_ref["path"] != EXPORTER_IMPLEMENTATION_PATH:
        raise ValueError("Joint catalog names an unreviewed exporter implementation")
    repo_root = Path(__file__).resolve().parents[3]
    exporter_path = repo_root / EXPORTER_IMPLEMENTATION_PATH
    exporter_sha256 = _sha256(exporter_ref["sha256"], name="exporter SHA-256")
    if not exporter_path.is_file() or _sha256_file(exporter_path) != exporter_sha256:
        raise ValueError("Joint catalog exporter bytes differ")

    projection_ref = _exact_mapping(
        catalog["visual_projection"],
        _VISUAL_PROJECTION_FIELDS,
        name="catalog.visual_projection",
    )
    projection_path = _absolute_file(projection_ref["path"], name="visual_projection.path")
    projection_sha256 = _sha256(projection_ref["raw_sha256"], name="visual_projection.raw_sha256")
    if _sha256_file(projection_path) != projection_sha256:
        raise ValueError("Joint visual projection differs from its raw SHA-256 pin")
    tokenizer, token_ids = load_pinned_vision_alignment_tokenizer(
        identifier=VISION_ALIGNMENT_TOKENIZER_ID,
        revision=VISION_ALIGNMENT_TOKENIZER_REVISION,
        expected_fingerprint=VISION_ALIGNMENT_TOKENIZER_FINGERPRINT,
        cache_dir=hf_cache_dir,
    )
    projection = load_joint_visual_projection_manifest(
        projection_path,
        expected_token_ids=token_ids,
        expected_sha256=projection_sha256,
    )
    if projection.raw_sha256 != projection_sha256 or projection.content_sha256 != _sha256(
        projection_ref["content_sha256"], name="visual_projection.content_sha256"
    ):
        raise ValueError("Joint visual projection identity differs")

    native_ref = _exact_mapping(
        catalog["native_train_manifest"],
        _NATIVE_MANIFEST_FIELDS,
        name="catalog.native_train_manifest",
    )
    native_path = _absolute_file(native_ref["path"], name="native_train_manifest.path")
    native_raw_sha256 = _sha256(native_ref["raw_sha256"], name="native_train_manifest.raw_sha256")
    if _sha256_file(native_path) != native_raw_sha256:
        raise ValueError("Native train manifest differs from its raw SHA-256 pin")
    native_manifest = _load_native_manifest(native_path)
    native_fingerprint = _sha256(
        native_ref["content_fingerprint"], name="native_train_manifest.content_fingerprint"
    )
    if (
        native_manifest.manifest_sha256 != native_raw_sha256
        or native_manifest.content_fingerprint != native_fingerprint
        or native_manifest.sequence_length != JOINT_SEQUENCE_LENGTH
        or native_manifest.provenance.get("split") != "train"
    ):
        raise ValueError("Native train manifest runtime identity differs")

    receipt_ref = _exact_mapping(
        catalog["native_verification_receipt"],
        _RECEIPT_FIELDS,
        name="catalog.native_verification_receipt",
    )
    receipt_path = _absolute_file(receipt_ref["path"], name="native_verification_receipt.path")
    receipt_sha256 = _sha256(receipt_ref["sha256"], name="native_verification_receipt.sha256")
    if _sha256_file(receipt_path) != receipt_sha256:
        raise ValueError("Native verification receipt differs from its raw SHA-256 pin")
    receipt = _load_native_receipt(receipt_path, expected_sha256=receipt_sha256)
    if receipt.receipt_sha256 != receipt_sha256:
        raise ValueError("Native verification receipt runtime identity differs")
    receipt.validate_manifest(native_manifest)
    if native_manifest.provenance.get("verification_receipt_sha256") != receipt_sha256:
        raise ValueError("Native train manifest does not bind the supplied verification receipt")

    preprocessing = _exact_mapping(
        catalog["preprocessing"], _PREPROCESSING_FIELDS, name="catalog.preprocessing"
    )
    expected_preprocessing = {
        "visual": projection.source_spec.as_canonical_dict(),
        "native_text_replay_fingerprint": native_fingerprint,
    }
    if _canonical_bytes(preprocessing) != _canonical_bytes(expected_preprocessing) or catalog[
        "preprocessing_sha256"
    ] != _canonical_sha256(expected_preprocessing):
        raise ValueError("Joint aggregate preprocessing identity differs")
    probe = _validate_probe_policy(catalog["probe"])

    raw_sources = catalog["sources"]
    if not isinstance(raw_sources, list) or len(raw_sources) != len(JOINT_SOURCE_NAMES):
        raise ValueError("Joint catalog must contain exactly nine sources")
    names = tuple(
        value.get("name") if isinstance(value, Mapping) else None for value in raw_sources
    )
    if names != JOINT_SOURCE_NAMES:
        raise ValueError("Joint catalog sources are not in exact canonical order")

    runtime_sources = _build_runtime_sources(
        projection,
        native_manifest,
        receipt,
        tokenizer=tokenizer,
        token_ids=token_ids,
    )
    if tuple(sorted(runtime_sources)) != JOINT_SOURCE_NAMES:
        raise ValueError("Joint runtime rebuild did not produce the exact nine-source set")
    targets = VisionAlignmentMixtureConfig(phase="joint").resolved_targets()
    if tuple(sorted(targets)) != JOINT_SOURCE_NAMES:
        raise ValueError("Checked-in joint target loss mass is not the exact nine-source set")

    source_reports: Dict[str, Any] = {}
    source_inputs: Dict[str, Any] = {}
    mean_loss_weight: Dict[str, float] = {}
    failures = []
    input_descriptor = []
    pinned_probe_bytes: Dict[Path, bytes] = {}
    for ordinal, raw_source in enumerate(raw_sources):
        source_name = JOINT_SOURCE_NAMES[ordinal]
        source_started_at = time.monotonic()
        source = _exact_mapping(raw_source, _SOURCE_FIELDS, name=f"catalog.sources[{ordinal}]")
        kind = "native_text_replay" if source_name == "native_text_replay" else "visual"
        if (
            source["name"] != source_name
            or source["kind"] != kind
            or source["format"] != "jsonl"
            or source["path"] != f"{source_name}.jsonl"
        ):
            raise ValueError(f"Joint source {source_name!r} catalog identity differs")
        expected_epochs = (
            JOINT_NATIVE_PROBE_EPOCHS if kind == "native_text_replay" else JOINT_VISUAL_PROBE_EPOCHS
        )
        expected_unique = (
            JOINT_NATIVE_PROBE_INDICES
            if kind == "native_text_replay"
            else JOINT_VISUAL_PROBE_INDICES
        )
        rows_total = expected_unique * len(expected_epochs)
        _emit_progress(
            progress,
            event="source_start",
            started_at=source_started_at,
            source_name=source_name,
            rows_completed=0,
            rows_total=rows_total,
            workers=workers,
        )
        runtime = runtime_sources[source_name]
        dataset = runtime.dataset
        fingerprint = runtime_dataset_fingerprint(dataset)
        if (
            not isinstance(fingerprint, str)
            or _SHA256_RE.fullmatch(fingerprint) is None
            or source["dataset_fingerprint"] != fingerprint
            or _integer(source["dataset_size"], name=f"{source_name}.dataset_size", minimum=1)
            != len(dataset)
        ):
            raise ValueError(f"Joint source {source_name!r} runtime identity differs")
        raw_epochs = source["probe_epochs"]
        if (
            not isinstance(raw_epochs, list)
            or any(type(epoch) is not int for epoch in raw_epochs)
            or tuple(raw_epochs) != expected_epochs
        ):
            raise ValueError(f"Joint source {source_name!r} epoch panel differs")
        raw_indices = source["probe_indices"]
        if not isinstance(raw_indices, list):
            raise ValueError(f"Joint source {source_name!r} probe_indices must be a list")
        indices = tuple(
            _integer(value, name=f"{source_name}.probe_indices", minimum=0) for value in raw_indices
        )
        expected_indices = select_deterministic_probe_indices(
            len(dataset),
            expected_unique,
            seed=JOINT_PROBE_SEED,
            dataset_fingerprint=fingerprint,
        )
        if indices != expected_indices:
            raise ValueError(f"Joint source {source_name!r} deterministic selection differs")
        indices_sha256 = _sha256(
            source["probe_indices_sha256"], name=f"{source_name}.probe_indices_sha256"
        )
        if _canonical_sha256(list(indices)) != indices_sha256:
            raise ValueError(f"Joint source {source_name!r} probe-index digest differs")
        expected_image_digest = (
            _canonical_sha256([])
            if kind == "native_text_replay"
            else _probe_image_digest(dataset, indices, source_name=source_name)
        )
        if source["probe_image_content_sha256"] != expected_image_digest:
            raise ValueError(f"Joint source {source_name!r} image-content digest differs")

        source_path = _catalog_local_path(catalog_path, source["path"], name=f"{source_name}.path")
        declared_source_sha256 = _sha256(source["sha256"], name=f"{source_name}.sha256")
        source_raw, records = _load_jsonl(source_path, expected_sha256=declared_source_sha256)
        pinned_probe_bytes[source_path] = source_raw
        source_sha256 = hashlib.sha256(source_raw).hexdigest()
        expected_pairs = tuple(
            (dataset_index, epoch) for epoch in expected_epochs for dataset_index in indices
        )
        if len(records) != len(expected_pairs):
            raise ValueError(
                f"Joint source {source_name!r} has {len(records)} rows, expected "
                f"{len(expected_pairs)}"
            )
        accumulator = shared_audit.SourceAccumulator()
        row_hashes = []
        maximum_length = 0
        live_records = _ordered_live_records(
            runtime,
            source_name=source_name,
            kind=kind,
            work_items=expected_pairs,
            workers=workers,
        )
        try:
            for row_ordinal, ((dataset_index, epoch), stored_record, live_record) in enumerate(
                zip(expected_pairs, records, live_records),
                start=1,
            ):
                validate_joint_probe_record(
                    stored_record,
                    source_name=source_name,
                    source_kind=kind,
                    expected_index=dataset_index,
                    expected_epoch=epoch,
                    sequence_length=JOINT_SEQUENCE_LENGTH,
                    token_ids=runtime.token_ids,
                )
                if _canonical_bytes(stored_record) != _canonical_bytes(live_record):
                    raise ValueError(
                        f"Joint source {source_name!r} serialized row drifted at ordinal "
                        f"{row_ordinal - 1}"
                    )
                row_hashes.append(live_record["serialized_row_sha256"])
                maximum_length = max(maximum_length, int(live_record["raw_sequence_length"]))
                accumulator.add_example(live_record)
                if row_ordinal % PROGRESS_INTERVAL_ROWS == 0 or row_ordinal == len(expected_pairs):
                    _emit_progress(
                        progress,
                        event="source_progress",
                        started_at=source_started_at,
                        source_name=source_name,
                        rows_completed=row_ordinal,
                        rows_total=len(expected_pairs),
                        workers=workers,
                    )
        finally:
            live_records.close()
        declared_row_hashes = _sha256(
            source["serialized_row_hashes_sha256"],
            name=f"{source_name}.serialized_row_hashes_sha256",
        )
        if _canonical_sha256(row_hashes) != declared_row_hashes:
            raise ValueError(f"Joint source {source_name!r} serialized-row digest differs")
        if (
            _integer(
                source["max_observed_sequence_length"],
                name=f"{source_name}.max_observed_sequence_length",
                minimum=1,
            )
            != maximum_length
            or _integer(source["truncated_rows"], name=f"{source_name}.truncated_rows", minimum=0)
            != 0
        ):
            raise ValueError(f"Joint source {source_name!r} truncation evidence differs")
        if kind == "visual" and (
            _probe_image_digest(dataset, indices, source_name=source_name) != expected_image_digest
        ):
            raise ValueError(f"Joint source {source_name!r} image bytes changed during audit")
        source_reports[source_name] = accumulator.as_dict()
        if accumulator.seen != len(expected_pairs) or accumulator.valid != len(expected_pairs):
            failures.append(f"{source_name}: malformed or missing probe rows")
        if accumulator.truncated:
            failures.append(f"{source_name}: contains truncated rows")
        if kind == "visual" and accumulator.zero_loss:
            failures.append(
                f"{source_name}: contains {accumulator.zero_loss} zero-loss visual probe rows"
            )
        if accumulator.loss_weight.total <= 0:
            failures.append(f"{source_name}: has non-positive supervised loss mass")
        else:
            mean_loss_weight[source_name] = accumulator.loss_weight.total / accumulator.valid
        source_inputs[source_name] = {
            **dict(source),
            "serialized_row_hashes": row_hashes,
        }
        input_descriptor.append(
            {
                "name": source_name,
                "kind": kind,
                "sha256": source_sha256,
                "dataset_fingerprint": fingerprint,
                "probe_indices_sha256": indices_sha256,
                "probe_epochs": list(expected_epochs),
                "serialized_row_hashes_sha256": declared_row_hashes,
                "probe_image_content_sha256": expected_image_digest,
                "max_observed_sequence_length": maximum_length,
                "truncated_rows": 0,
            }
        )
        _emit_progress(
            progress,
            event="source_complete",
            started_at=source_started_at,
            source_name=source_name,
            rows_completed=len(expected_pairs),
            rows_total=len(expected_pairs),
            workers=workers,
        )
        # The production runtime mapping is lazy; dropping these locals releases both the
        # selected and unbounded adapters before the next source is constructed.
        del dataset, live_records, records, runtime

    sampling_probabilities = None
    expected_mass = None
    if not failures:
        sampling_probabilities = sampling_weights_from_loss_mass(targets, mean_loss_weight)
        expected_mass = expected_loss_mass(sampling_probabilities, mean_loss_weight)
        if any(
            not math.isclose(expected_mass[name], targets[name], rel_tol=0.0, abs_tol=1e-12)
            for name in targets
        ):
            failures.append("Calibrated joint expected loss mass differs from exact targets")

    auditor_path = Path(__file__).resolve()
    auditor_sha256 = _sha256_file(auditor_path)
    shared_auditor_path = Path(shared_audit.__file__).resolve()
    shared_auditor_sha256 = _sha256_file(shared_auditor_path)
    report: Dict[str, Any] = {
        "format": AUDIT_FORMAT,
        "version": AUDIT_VERSION,
        "status": "ok" if not failures else "failed",
        "phase": "joint",
        "recipe_version": VISION_ALIGNMENT_RECIPE_VERSION,
        "formatter_version": VISION_ALIGNMENT_FORMATTER_VERSION,
        "source_catalog_version": CATALOG_VERSION,
        "auditor_implementation": {
            "path": AUDITOR_IMPLEMENTATION_PATH,
            "sha256": auditor_sha256,
        },
        "shared_auditor_sha256": shared_auditor_sha256,
        "catalog_path": str(catalog_path),
        "catalog_sha256": catalog_sha256,
        "catalog_content_sha256": declared_content_sha,
        "input_content_sha256": _canonical_sha256(input_descriptor),
        "source_registry_version": VISION_ALIGNMENT_JOINT_SOURCE_REGISTRY_VERSION,
        "source_registry_sha256": registry_sha256,
        "source_implementation_inventory": implementation_inventory,
        "exporter_implementation": dict(exporter_ref),
        "visual_projection": dict(projection_ref),
        "native_train_manifest": dict(native_ref),
        "native_verification_receipt": dict(receipt_ref),
        "preprocessing": expected_preprocessing,
        "preprocessing_sha256": catalog["preprocessing_sha256"],
        "probe": dict(probe),
        "inputs": source_inputs,
        "target_loss_mass": targets,
        "sources": source_reports,
        "mean_loss_weight": mean_loss_weight,
        "sampling_probabilities": sampling_probabilities,
        "expected_loss_mass": expected_mass,
        "failures": failures,
    }
    # Keep the launcher-facing audit identity consistent with the established bridge and
    # perception contracts: ``fingerprint`` hashes the entire unsigned canonical report.
    report["fingerprint"] = _canonical_sha256(report)
    _emit_progress(
        progress,
        event="closing_validation_start",
        started_at=audit_started_at,
        workers=workers,
    )
    _closing_validate_inputs(
        projection=projection,
        native_manifest=native_manifest,
        receipt=receipt,
        expected_native_size=int(source_inputs["native_text_replay"]["dataset_size"]),
        token_ids=token_ids,
    )
    if (
        _sha256_file(catalog_path) != catalog_sha256
        or _sha256_file(exporter_path) != exporter_sha256
        or _sha256_file(projection_path) != projection_sha256
        or _sha256_file(native_path) != native_raw_sha256
        or _sha256_file(receipt_path) != receipt_sha256
        or _sha256_file(auditor_path) != auditor_sha256
        or _sha256_file(shared_auditor_path) != shared_auditor_sha256
        or joint_alignment_runtime_registry_sha256() != registry_sha256
        or joint_alignment_runtime_implementation_inventory() != implementation_inventory
        or any(source_path.read_bytes() != raw for source_path, raw in pinned_probe_bytes.items())
    ):
        raise ValueError("Joint audit input or implementation changed during audit")
    _emit_progress(
        progress,
        event="phase_complete",
        started_at=audit_started_at,
        sources_completed=len(source_reports),
        workers=workers,
    )
    return report


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable joint audit {path}")
    raw = _canonical_bytes(value) + b"\n"
    descriptor, temporary_value = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_value)
    try:
        with os.fdopen(descriptor, "wb") as file_handle:
            file_handle.write(raw)
            file_handle.flush()
            os.fsync(file_handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog")
    parser.add_argument("--expected-catalog-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=DEFAULT_GUARD_WORKERS)
    parser.add_argument("--hf-cache-dir", default=DEFAULT_HF_CACHE_DIR)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Audit the exact joint source catalog and return a process exit code."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        report = audit_joint_catalog(
            args.catalog,
            expected_catalog_sha256=args.expected_catalog_sha256,
            hf_cache_dir=args.hf_cache_dir,
            workers=args.workers,
            progress=_stderr_progress,
        )
        _write_once(Path(args.output), report)
    except (FileExistsError, OLMoConfigurationError, OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    if report["status"] != "ok":
        print("error: joint audit failed; inspect failures", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
