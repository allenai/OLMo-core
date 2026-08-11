"""Canonical source construction and probe identities for vision alignment.

The training launcher and the offline source exporter both use this module. Keeping source
configuration here prevents an audit tool from approximating the runtime serializer with a
second implementation. A deterministic probe records exact runtime dataset indices and hashes
the model-consumed arrays returned by each dataset's :meth:`get` method.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from olmo_core.nn.vision import Molmo2TokenIds

from .pixmo_cap import PixMoCapDatasetConfig
from .pixmo_points import CoSynPointDatasetConfig, PixMoPointsDatasetConfig

__all__ = [
    "VISION_ALIGNMENT_FORMATTER_VERSION",
    "VISION_ALIGNMENT_PROBE_SELECTION_ALGORITHM",
    "VISION_ALIGNMENT_RECIPE_VERSION",
    "VISION_ALIGNMENT_SOURCE_CATALOG_VERSION",
    "VISION_ALIGNMENT_SOURCE_REGISTRY_VERSION",
    "VISION_ALIGNMENT_PROBE_FORMAT",
    "VISION_ALIGNMENT_PROBE_VERSION",
    "VISION_ALIGNMENT_PIXMO_ROW_PATH_INVENTORY_ALGORITHM",
    "VISION_ALIGNMENT_TOKENIZER_FILES",
    "VISION_ALIGNMENT_TOKENIZER_FINGERPRINT",
    "VISION_ALIGNMENT_TOKENIZER_ID",
    "VISION_ALIGNMENT_TOKENIZER_REVISION",
    "VisionAlignmentSourceSpec",
    "array_content_descriptor",
    "build_vision_alignment_dataset",
    "build_vision_alignment_dataset_config",
    "load_pinned_vision_alignment_tokenizer",
    "pixmo_row_path_inventory",
    "runtime_dataset_fingerprint",
    "select_deterministic_probe_indices",
    "serialized_example_descriptor",
    "serialized_example_sha256",
    "serialized_descriptor_sha256",
    "serialized_probe_record",
    "validate_serialized_runtime_probe",
    "vision_alignment_source_registry_sha256",
]

VISION_ALIGNMENT_RECIPE_VERSION = 1
VISION_ALIGNMENT_FORMATTER_VERSION = "vision-alignment-document-v1"
VISION_ALIGNMENT_SOURCE_REGISTRY_VERSION = 1
VISION_ALIGNMENT_SOURCE_CATALOG_VERSION = 2
VISION_ALIGNMENT_PROBE_FORMAT = "vision_alignment_serialized_probe"
VISION_ALIGNMENT_PROBE_VERSION = 1
VISION_ALIGNMENT_PROBE_SELECTION_ALGORITHM = "sha256-affine-permutation-v1"
VISION_ALIGNMENT_PIXMO_ROW_PATH_INVENTORY_ALGORITHM = "sha256-jsonl-index-image-path-v1"

VISION_ALIGNMENT_TOKENIZER_ID = "allenai/dolma2-tokenizer"
VISION_ALIGNMENT_TOKENIZER_REVISION = "5292e5d6c0f40b67cc765fe41bec991cf4345b5c"
VISION_ALIGNMENT_TOKENIZER_FINGERPRINT = (
    "8fec2af8c372f4c72a1a665ad8e70517625f94f041dbfcb7db4932071380f9a7"
)
VISION_ALIGNMENT_TOKENIZER_FILES = (
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)

_SERIALIZED_REQUIRED_FIELDS = (
    "input_ids",
    "labels",
    "loss_masks",
    "position_ids",
    "token_type_ids",
    "images",
    "pooled_patches_idx",
)
_SERIALIZED_OPTIONAL_FIELDS = ("subsegment_ids",)
_SERIALIZED_IGNORED_FIELDS = frozenset({"metadata"})
_INLINE_PROBE_FIELDS = (
    "input_ids",
    "labels",
    "loss_masks",
    "position_ids",
    "token_type_ids",
    "subsegment_ids",
)
_SERIALIZED_EXAMPLE_DOMAIN = b"vision-alignment-serialized-example-v1\0"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class VisionAlignmentSourceSpec:
    """All checked fields that determine visual-source serialization.

    :param phase: Vision-alignment phase name.
    :param pixmo_cap_path: Runtime PixMoCap dataset path.
    :param sequence_length: Maximum serialized token length.
    :param max_crops: Maximum image crops per example.
    :param message_format: Message layout; production vision alignment uses ``document``.
    :param loss_token_weighting: Per-response token weighting policy.
    :param caption_prompt: Fixed caption document prompt.
    :param transcript_prompt: Fixed transcript document prompt.
    :param require_transcript: Require every transcript row to contain a non-blank transcript.
    :param tokenizer_id: Tokenizer repository identifier.
    :param tokenizer_revision: Immutable tokenizer repository revision.
    :param tokenizer_fingerprint: Digest of the pinned tokenizer files.
    :param native_text_replay_fingerprint: Optional joint-phase replay identity.
    :param recipe_version: Vision-alignment recipe schema version.
    :param formatter_version: Document formatter identity.
    """

    phase: str
    pixmo_cap_path: str
    sequence_length: int
    max_crops: int
    message_format: str
    loss_token_weighting: str
    caption_prompt: str
    transcript_prompt: str
    require_transcript: bool
    tokenizer_id: str = VISION_ALIGNMENT_TOKENIZER_ID
    tokenizer_revision: str = VISION_ALIGNMENT_TOKENIZER_REVISION
    tokenizer_fingerprint: str = VISION_ALIGNMENT_TOKENIZER_FINGERPRINT
    native_text_replay_fingerprint: Optional[str] = None
    recipe_version: int = VISION_ALIGNMENT_RECIPE_VERSION
    formatter_version: str = VISION_ALIGNMENT_FORMATTER_VERSION

    def as_canonical_dict(self) -> Dict[str, Any]:
        """Return the versioned canonical preprocessing descriptor."""
        values = asdict(self)
        if self.pixmo_cap_path != "synthetic":
            values["pixmo_cap_path"] = str(Path(self.pixmo_cap_path).expanduser().resolve())
        return {
            "source_registry_version": VISION_ALIGNMENT_SOURCE_REGISTRY_VERSION,
            **values,
        }

    @property
    def preprocessing_sha256(self) -> str:
        """SHA-256 of every checked serialization field."""
        return hashlib.sha256(_canonical_bytes(self.as_canonical_dict())).hexdigest()


def vision_alignment_source_registry_sha256() -> str:
    """Return the SHA-256 of this exact source-registry implementation."""
    return _sha256_file(Path(__file__).resolve())


def load_pinned_vision_alignment_tokenizer(
    *,
    identifier: str,
    revision: str,
    expected_fingerprint: str,
    cache_dir: str,
    model_vocab_size: int = 100352,
):
    """Load and verify the tokenizer used by both training and probe export.

    :param identifier: Hugging Face tokenizer identifier.
    :param revision: Immutable repository revision.
    :param expected_fingerprint: Digest of the pinned tokenizer file inventory.
    :param cache_dir: Local Hugging Face cache directory.
    :param model_vocab_size: Padded model vocabulary size after adding image tokens.
    :returns: The prepared tokenizer and its :class:`Molmo2TokenIds`.
    :raises ValueError: If the local snapshot differs from the pinned fingerprint.
    """
    from huggingface_hub import snapshot_download
    from transformers import GPT2Tokenizer

    from olmo_core.nn.vision import prepare_molmo2_tokenizer

    snapshot = Path(
        snapshot_download(
            identifier,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=True,
        )
    )
    digest = hashlib.sha256()
    for name in VISION_ALIGNMENT_TOKENIZER_FILES:
        file_hash = _sha256_file(snapshot / name)
        digest.update(f"{file_hash}  {name}\n".encode())
    actual_fingerprint = digest.hexdigest()
    if actual_fingerprint != expected_fingerprint:
        raise ValueError(
            f"Tokenizer snapshot fingerprint mismatch for {identifier}@{revision}: "
            f"expected {expected_fingerprint}, got {actual_fingerprint}"
        )
    tokenizer = GPT2Tokenizer.from_pretrained(snapshot, local_files_only=True)
    token_ids = prepare_molmo2_tokenizer(tokenizer, model_vocab_size=model_vocab_size)
    return tokenizer, token_ids


def build_vision_alignment_dataset_config(
    spec: VisionAlignmentSourceSpec,
    token_ids: Molmo2TokenIds,
    source_name: str,
    *,
    split: str = "train",
) -> Any:
    """Build the canonical runtime config for one visual source.

    This is the sole source-config registry used by training and canonical probe export.
    Unsupported future sources fail instead of being approximated with another dataset.

    :param spec: Exact preprocessing specification.
    :param token_ids: Prepared image-token identities.
    :param source_name: Canonical mixture source name.
    :param split: Dataset split.
    :returns: The matching multimodal dataset configuration.
    :raises KeyError: If no audited adapter exists for ``source_name``.
    :raises ValueError: If a source does not support the requested split.
    """
    common: Dict[str, Any] = {
        "max_crops": spec.max_crops,
        "max_sequence_length": spec.sequence_length,
        "loss_token_weighting": spec.loss_token_weighting,
        "token_ids": token_ids,
        "message_format": spec.message_format,
        "seed": 0,
    }
    if source_name == "pixmo_caption":
        return PixMoCapDatasetConfig(
            dataset_path=spec.pixmo_cap_path,
            split=split,
            require_split=spec.pixmo_cap_path != "synthetic",
            mode="caption",
            fixed_prompt=spec.caption_prompt,
            style_length_conditioning=False,
            **common,
        )
    if source_name == "pixmo_transcript":
        return PixMoCapDatasetConfig(
            dataset_path=spec.pixmo_cap_path,
            split=split,
            require_split=spec.pixmo_cap_path != "synthetic",
            mode="transcript",
            require_transcript=spec.require_transcript,
            fixed_prompt=spec.transcript_prompt,
            style_length_conditioning=False,
            **common,
        )
    point_common = {key: value for key, value in common.items() if key != "max_sequence_length"}
    if source_name == "pixmo_points_basic":
        return PixMoPointsDatasetConfig(
            split=split,
            kind="basic",
            counting=False,
            both_mode="per_annotation",
            **point_common,
        )
    if source_name == "pixmo_points_high_frequency":
        return PixMoPointsDatasetConfig(
            split=split,
            kind="high_frequency",
            counting=False,
            both_mode="per_annotation",
            **point_common,
        )
    if source_name == "cosyn_point":
        if split != "train":
            raise ValueError("CoSyn Point has no approved vision-alignment validation split")
        return CoSynPointDatasetConfig(**point_common)
    raise KeyError(source_name)


def build_vision_alignment_dataset(
    spec: VisionAlignmentSourceSpec,
    tokenizer: Any,
    token_ids: Molmo2TokenIds,
    source_name: str,
    *,
    split: str = "train",
    validate_required_annotations: bool = True,
) -> Any:
    """Build one canonical source and enforce its strict annotation contract.

    :param spec: Exact preprocessing specification.
    :param tokenizer: Prepared runtime tokenizer.
    :param token_ids: Prepared image-token identities.
    :param source_name: Canonical mixture source name.
    :param split: Dataset split.
    :param validate_required_annotations: Run any dataset-wide strict annotation check.
    :returns: The built map-style dataset.
    """
    dataset = build_vision_alignment_dataset_config(
        spec, token_ids, source_name, split=split
    ).build(tokenizer)
    if validate_required_annotations:
        validate = getattr(dataset, "validate_required_annotations", None)
        if callable(validate):
            validate()
    return dataset


def runtime_dataset_fingerprint(dataset: Any) -> Optional[str]:
    """Resolve a stable identity from a live source or its selected Arrow split."""
    for candidate in (dataset, getattr(dataset, "_hf", None)):
        if candidate is None:
            continue
        for attribute in ("content_fingerprint", "fingerprint", "_fingerprint"):
            value = getattr(candidate, attribute, None)
            if callable(value):
                value = value()
            if isinstance(value, str) and value:
                return value
    return None


def pixmo_row_path_inventory(dataset: Any) -> Dict[str, Any]:
    """Hash the ordered image-path column without reading image contents.

    This is the inexpensive runtime half of the canonical PixMoCap split contract. The
    offline split builder hashes every image's bytes and records those content inventories;
    training recomputes this ordered path digest from the live Arrow split so a manifest
    cannot be attached to a different row/path layout without re-hashing the images.

    :param dataset: A Hugging Face Arrow dataset or a runtime wrapper exposing one as ``_hf``.
    :returns: The versioned row count, unique-path count, and ordered path digest.
    :raises ValueError: If the live source is not an Arrow split with a non-empty string
        ``image`` column.
    """
    arrow_dataset = getattr(dataset, "_hf", dataset)
    table = getattr(arrow_dataset, "data", None)
    if table is None:
        raise ValueError("PixMoCap row/path inventory requires a live Arrow dataset split")
    try:
        image_column = table.column("image")
    except (KeyError, ValueError) as error:
        raise ValueError("PixMoCap Arrow split lacks the required 'image' path column") from error

    digest = hashlib.sha256()
    unique_paths = set()
    row_count = 0
    for index, scalar in enumerate(image_column):
        image_path = scalar.as_py()
        if not isinstance(image_path, str) or not image_path:
            raise ValueError(f"PixMoCap image path at row {index} must be a non-empty string")
        record = _canonical_bytes({"image": image_path, "index": index}) + b"\n"
        digest.update(record)
        unique_paths.add(image_path)
        row_count += 1
    if row_count == 0:
        raise ValueError("PixMoCap Arrow split must contain at least one row")
    return {
        "algorithm": VISION_ALIGNMENT_PIXMO_ROW_PATH_INVENTORY_ALGORITHM,
        "rows": row_count,
        "unique_paths": len(unique_paths),
        "sha256": digest.hexdigest(),
    }


def select_deterministic_probe_indices(
    dataset_size: int,
    num_examples: int,
    *,
    seed: int,
    dataset_fingerprint: str,
) -> Tuple[int, ...]:
    """Select a portable no-replacement prefix of a deterministic affine permutation.

    Sources that share the same live dataset fingerprint and size receive the same probe rows,
    which keeps caption/transcript calibration paired without depending on NumPy RNG behavior.

    :param dataset_size: Number of rows in the live source.
    :param num_examples: Number of probe rows to select.
    :param seed: Non-negative selection seed.
    :param dataset_fingerprint: Stable live dataset identity.
    :returns: Ordered unique dataset indices.
    :raises ValueError: If counts, seed, or fingerprint are invalid.
    """
    if isinstance(dataset_size, bool) or not isinstance(dataset_size, int) or dataset_size <= 0:
        raise ValueError("dataset_size must be a positive integer")
    if (
        isinstance(num_examples, bool)
        or not isinstance(num_examples, int)
        or num_examples <= 0
        or num_examples > dataset_size
    ):
        raise ValueError("num_examples must be in [1, dataset_size]")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("probe seed must be a non-negative integer")
    if not isinstance(dataset_fingerprint, str) or not dataset_fingerprint:
        raise ValueError("dataset_fingerprint must be a non-empty string")

    digest = hashlib.sha256(
        _canonical_bytes(
            {
                "algorithm": VISION_ALIGNMENT_PROBE_SELECTION_ALGORITHM,
                "dataset_fingerprint": dataset_fingerprint,
                "dataset_size": dataset_size,
                "seed": seed,
            }
        )
    ).digest()
    offset = int.from_bytes(digest[:16], "big") % dataset_size
    step = int.from_bytes(digest[16:], "big") % dataset_size
    if step == 0:
        step = 1
    while math.gcd(step, dataset_size) != 1:
        step = 1 if step + 1 == dataset_size else step + 1
    return tuple((offset + ordinal * step) % dataset_size for ordinal in range(num_examples))


def _as_numpy(value: Any, field_name: str) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError(f"Serialized example field {field_name!r} has object dtype")
    return array


def array_content_descriptor(value: Any, *, field_name: str) -> Dict[str, Any]:
    """Return a canonical dtype/shape/content digest for an array-like field.

    :param value: NumPy-, tensor-, or list-like value.
    :param field_name: Field name used in validation errors.
    :returns: Canonical array descriptor.
    """
    array = _as_numpy(value, field_name)
    dtype = array.dtype
    if dtype.itemsize > 1:
        little_dtype = dtype.newbyteorder("<")
        if dtype.byteorder == ">" or (dtype.byteorder == "=" and sys.byteorder == "big"):
            array = array.byteswap().view(little_dtype)
        else:
            array = array.astype(little_dtype, copy=False)
    array = np.ascontiguousarray(array)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def serialized_example_descriptor(example: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Describe every model-consumed array in one uncollated runtime example.

    :param example: Dataset example returned by the canonical adapter.
    :returns: Per-field dtype, shape, and content digests.
    :raises ValueError: If required fields are absent or an unversioned field is present.
    """
    missing = [field for field in _SERIALIZED_REQUIRED_FIELDS if field not in example]
    if missing:
        raise ValueError(f"Serialized vision-alignment example is missing fields {missing}")
    supported = set(_SERIALIZED_REQUIRED_FIELDS) | set(_SERIALIZED_OPTIONAL_FIELDS)
    unknown = sorted(set(example) - supported - _SERIALIZED_IGNORED_FIELDS)
    if unknown:
        raise ValueError(
            "Serialized vision-alignment example contains unversioned fields " f"{unknown}"
        )
    fields = [*_SERIALIZED_REQUIRED_FIELDS]
    fields.extend(field for field in _SERIALIZED_OPTIONAL_FIELDS if field in example)
    return {field: array_content_descriptor(example[field], field_name=field) for field in fields}


def serialized_example_sha256(example: Mapping[str, Any]) -> str:
    """Hash the canonical descriptor of one uncollated runtime example."""
    return serialized_descriptor_sha256(serialized_example_descriptor(example))


def serialized_descriptor_sha256(descriptor: Mapping[str, Any]) -> str:
    """Hash a canonical serialized-example field descriptor."""
    return hashlib.sha256(_SERIALIZED_EXAMPLE_DOMAIN + _canonical_bytes(descriptor)).hexdigest()


def serialized_probe_record(
    example: Mapping[str, Any],
    *,
    source_name: str,
    dataset_index: int,
    epoch: int = 0,
) -> Dict[str, Any]:
    """Build the compact audited JSON record for one exact runtime example.

    Large image arrays are represented by canonical content descriptors, while token and label
    arrays are also included inline for structural and loss-mass auditing.

    :param example: Exact example returned by the canonical runtime dataset.
    :param source_name: Canonical mixture source name.
    :param dataset_index: Live map-style dataset index.
    :param epoch: Source epoch used to materialize the example.
    :returns: Canonicalizable probe record.
    """
    descriptor = serialized_example_descriptor(example)
    record: Dict[str, Any] = {
        "source": source_name,
        "probe_index": dataset_index,
        "probe_epoch": epoch,
        "serialized_fields": descriptor,
        "serialized_row_sha256": serialized_descriptor_sha256(descriptor),
        "image_crops": descriptor["images"]["shape"][0],
        "pooled_tokens": descriptor["pooled_patches_idx"]["shape"][0],
    }
    for field in _INLINE_PROBE_FIELDS:
        if field in example:
            record[field] = _as_numpy(example[field], field).tolist()
    return record


def validate_serialized_runtime_probe(
    dataset: Any,
    probe_indices: Sequence[int],
    expected_row_hashes: Sequence[str],
    *,
    epoch: int = 0,
) -> None:
    """Verify pinned serialized rows against a live runtime dataset.

    :param dataset: Exact built map-style runtime dataset.
    :param probe_indices: Ordered live indices pinned by the canonical exporter.
    :param expected_row_hashes: Expected model-input hashes at those indices.
    :param epoch: Dataset epoch used by the exporter.
    :raises ValueError: If the binding is malformed or any live row has drifted.
    """
    if len(probe_indices) != len(expected_row_hashes) or not probe_indices:
        raise ValueError("Runtime probe indices and row hashes must have the same non-zero length")
    get = getattr(dataset, "get", None)
    for ordinal, (index, expected_hash) in enumerate(zip(probe_indices, expected_row_hashes)):
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= len(dataset)
        ):
            raise ValueError(f"Runtime probe index {index!r} at ordinal {ordinal} is invalid")
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(character not in "0123456789abcdef" for character in expected_hash)
        ):
            raise ValueError(f"Runtime probe hash at ordinal {ordinal} is invalid")
        example = get(index, epoch) if callable(get) else dataset[index]
        actual_hash = serialized_example_sha256(example)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Live serialized runtime row drifted at probe ordinal {ordinal}, "
                f"dataset index {index}: expected {expected_hash}, got {actual_hash}"
            )
