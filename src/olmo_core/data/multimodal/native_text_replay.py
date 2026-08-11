"""Native next-token replay for multimodal continued pretraining.

This module adapts a finite, pretokenized text replay manifest to the example contract used
by :mod:`olmo_core.data.multimodal`.  It deliberately does not tokenize, concatenate, or add
document markers: the IDs (including EOS document boundaries) are read directly from the
parent language model's pretraining arrays. A manifest window contains exactly
``sequence_length`` tokens on the parent's fixed-length grid. Labels shift within that window,
the final label is ignored, and the parent's repetition filter can mask the whole instance.

The manifest is a JSON object with this schema::

    {
      "format": "olmo_native_text_replay",
      "version": 1,
      "sequence_length": 2560,
      "dtype": "uint32",
      "tokenizer": {
        "identifier": "allenai/dolma2-tokenizer",
        "vocab_size": 100278,
        "eos_token_id": 100257,
        "pad_token_id": 100277
      },
      "provenance": {
        "parent_checkpoint": "/path/to/s002-step125500",
        "parent_mix": "OLMo-mix-0925",
        "parent_paths_sha256": "...",
        "instance_filter": {
          "repetition_min_period": 1,
          "repetition_max_period": 13,
          "repetition_max_count": 32
        }
      },
      "num_windows": 2,
      "sources": [
        {
          "id": "web-000000",
          "source": "web",
          "parent_path_index": 0,
          "parent_path": "s3://ai2-llm/.../000000.npy",
          "path": "tokens/web-000000.npy",
          "num_tokens": 16384,
          "size_bytes": 65536,
          "sha256": "...",
          "window_starts": [0, 8192]
        }
      ]
    }

Token files are headerless numpy memmaps, matching OLMo's preprocessed ``.npy`` arrays.
Relative paths are resolved from the manifest directory.  Window starts must be ordered,
non-overlapping, in bounds, and explicitly enumerated.  These constraints make the replay
set bounded and deterministic; stochastic ordering belongs in the data loader.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from olmo_core.config import Config
from olmo_core.data.utils import find_periodic_sequences
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.vision.molmo2_tokens import N_PATCHES_SQ, PATCH_DIM, POOL_H, POOL_W

__all__ = [
    "NATIVE_TEXT_REPLAY_FORMAT",
    "NATIVE_TEXT_REPLAY_VERIFICATION_FORMAT",
    "NATIVE_TEXT_REPLAY_VERIFICATION_VERSION",
    "NATIVE_TEXT_REPLAY_VERSION",
    "NativeTextReplayDataset",
    "NativeTextReplayDatasetConfig",
    "NativeTextReplayManifest",
    "NativeTextReplaySource",
    "NativeTextReplayVerificationReceipt",
]

NATIVE_TEXT_REPLAY_FORMAT = "olmo_native_text_replay"
NATIVE_TEXT_REPLAY_VERSION = 1
NATIVE_TEXT_REPLAY_VERIFICATION_FORMAT = "olmo_native_text_replay_verification_receipt"
NATIVE_TEXT_REPLAY_VERIFICATION_VERSION = 1
S002_INSTANCE_FILTER = {
    "repetition_min_period": 1,
    "repetition_max_period": 13,
    "repetition_max_count": 32,
}
_FINGERPRINT_DOMAIN = b"olmo-native-text-replay-v1\0"
_SUPPORTED_DTYPES = {
    "uint8": np.dtype(np.uint8),
    "uint16": np.dtype(np.uint16),
    "uint32": np.dtype(np.uint32),
    "uint64": np.dtype(np.uint64),
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OLMoConfigurationError(
            f"Native text replay manifest field {name!r} must be an object"
        )
    return value


def _require_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise OLMoConfigurationError(
            f"Native text replay manifest field {name!r} must be an integer >= {minimum}"
        )
    return value


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise OLMoConfigurationError(
            f"Native text replay manifest field {name!r} must be a non-empty string"
        )
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OLMoConfigurationError(f"Native text replay JSON repeats key {key!r}")
        result[key] = value
    return result


def _require_exact_fields(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        raise OLMoConfigurationError(
            f"Native text replay {name} is missing required fields: {missing}"
        )
    if unknown:
        raise OLMoConfigurationError(f"Native text replay {name} has unknown fields: {unknown}")


@dataclass(frozen=True)
class NativeTextReplaySource:
    """One pretokenized source file and its finite list of selected replay windows.

    ``token_path`` is the path exactly as recorded in the manifest and therefore part of
    the semantic fingerprint. ``resolved_path`` is the local path used for reads.
    """

    source_id: str
    source_name: str
    parent_path_index: int
    parent_path: str
    token_path: str
    resolved_path: Path
    num_tokens: int
    size_bytes: int
    sha256: str
    window_starts: Tuple[int, ...]


@dataclass(frozen=True)
class NativeTextReplayManifest:
    """Validated replay-manifest contents and stable content identifiers."""

    path: Path
    sequence_length: int
    dtype: np.dtype
    tokenizer: Mapping[str, Any]
    provenance: Mapping[str, Any]
    sources: Tuple[NativeTextReplaySource, ...]
    num_windows: int
    manifest_sha256: str
    content_fingerprint: str

    @classmethod
    def load(cls, path: os.PathLike[str] | str) -> "NativeTextReplayManifest":
        """Load and strictly validate a finite native-text replay manifest.

        The raw-file SHA-256 is exposed as :attr:`manifest_sha256`. The semantic
        :attr:`content_fingerprint` hashes canonical JSON, so harmless whitespace or key
        ordering changes do not invalidate a loader resume.

        :param path: Path to the JSON manifest.
        :returns: The validated manifest.
        :raises OLMoConfigurationError: If the manifest is malformed or inconsistent.
        """
        manifest_path = Path(path).expanduser().resolve()
        try:
            raw = manifest_path.read_bytes()
        except OSError as error:
            raise OLMoConfigurationError(
                f"Could not read native text replay manifest {manifest_path}: {error}"
            ) from error
        try:
            data = json.loads(raw, object_pairs_hook=_strict_json_object)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise OLMoConfigurationError(
                f"Invalid native text replay JSON in {manifest_path}: {error}"
            ) from error
        data = _require_mapping(data, "root")

        if data.get("format") != NATIVE_TEXT_REPLAY_FORMAT:
            raise OLMoConfigurationError(
                f"Native text replay manifest format must be {NATIVE_TEXT_REPLAY_FORMAT!r}"
            )
        version = _require_int(data.get("version"), "version", minimum=1)
        if version != NATIVE_TEXT_REPLAY_VERSION:
            raise OLMoConfigurationError(
                f"Unsupported native text replay manifest version {version}; "
                f"expected {NATIVE_TEXT_REPLAY_VERSION}"
            )

        sequence_length = _require_int(data.get("sequence_length"), "sequence_length", minimum=2)
        window_length = sequence_length
        dtype_name = _require_string(data.get("dtype"), "dtype")
        if dtype_name not in _SUPPORTED_DTYPES:
            raise OLMoConfigurationError(
                f"Unsupported native text replay dtype {dtype_name!r}; "
                f"expected one of {sorted(_SUPPORTED_DTYPES)}"
            )
        dtype = _SUPPORTED_DTYPES[dtype_name]

        tokenizer = dict(_require_mapping(data.get("tokenizer"), "tokenizer"))
        _require_string(tokenizer.get("identifier"), "tokenizer.identifier")
        vocab_size = _require_int(tokenizer.get("vocab_size"), "tokenizer.vocab_size", minimum=1)
        eos_token_id = _require_int(tokenizer.get("eos_token_id"), "tokenizer.eos_token_id")
        pad_token_id = _require_int(tokenizer.get("pad_token_id"), "tokenizer.pad_token_id")
        if eos_token_id >= vocab_size or pad_token_id >= vocab_size:
            raise OLMoConfigurationError(
                "Native text replay tokenizer EOS and pad IDs must be below vocab_size"
            )

        provenance = dict(_require_mapping(data.get("provenance"), "provenance"))
        _require_string(provenance.get("parent_checkpoint"), "provenance.parent_checkpoint")
        _require_string(provenance.get("parent_mix"), "provenance.parent_mix")
        parent_paths_sha256 = _require_string(
            provenance.get("parent_paths_sha256"), "provenance.parent_paths_sha256"
        ).lower()
        if _SHA256_RE.fullmatch(parent_paths_sha256) is None:
            raise OLMoConfigurationError(
                "Native text replay provenance.parent_paths_sha256 must be a SHA-256"
            )
        provenance["parent_paths_sha256"] = parent_paths_sha256
        raw_instance_filter = _require_mapping(
            provenance.get("instance_filter"), "provenance.instance_filter"
        )
        instance_filter = {
            name: _require_int(
                raw_instance_filter.get(name), f"provenance.instance_filter.{name}", minimum=1
            )
            for name in S002_INSTANCE_FILTER
        }
        if set(raw_instance_filter) != set(S002_INSTANCE_FILTER):
            raise OLMoConfigurationError(
                "Native text replay provenance.instance_filter must contain exactly "
                f"{sorted(S002_INSTANCE_FILTER)}"
            )
        if instance_filter != S002_INSTANCE_FILTER:
            raise OLMoConfigurationError(
                "Native text replay must preserve the pinned s002 repetition filter: "
                f"expected {S002_INSTANCE_FILTER}, got {instance_filter}"
            )
        provenance["instance_filter"] = instance_filter

        raw_sources = data.get("sources")
        if not isinstance(raw_sources, list) or not raw_sources:
            raise OLMoConfigurationError(
                "Native text replay manifest field 'sources' must be a non-empty list"
            )

        sources = []
        source_ids = set()
        parent_path_indices = set()
        resolved_paths = set()
        total_windows = 0
        for source_index, raw_source in enumerate(raw_sources):
            source = _require_mapping(raw_source, f"sources[{source_index}]")
            prefix = f"sources[{source_index}]"
            source_id = _require_string(source.get("id"), f"{prefix}.id")
            if source_id in source_ids:
                raise OLMoConfigurationError(
                    f"Duplicate native text replay source id {source_id!r}"
                )
            source_ids.add(source_id)
            source_name = _require_string(source.get("source"), f"{prefix}.source")
            parent_path_index = _require_int(
                source.get("parent_path_index"), f"{prefix}.parent_path_index"
            )
            if parent_path_index in parent_path_indices:
                raise OLMoConfigurationError(
                    f"Duplicate native replay parent_path_index {parent_path_index}"
                )
            parent_path_indices.add(parent_path_index)
            parent_path = _require_string(source.get("parent_path"), f"{prefix}.parent_path")
            token_path = _require_string(source.get("path"), f"{prefix}.path")
            if "://" in token_path:
                raise OLMoConfigurationError(
                    f"Native text replay source {source_id!r} must be materialized locally; "
                    f"URL paths are not supported ({token_path!r})"
                )
            unresolved_path = Path(token_path).expanduser()
            resolved_path = (
                unresolved_path
                if unresolved_path.is_absolute()
                else manifest_path.parent / unresolved_path
            ).resolve()
            if resolved_path in resolved_paths:
                raise OLMoConfigurationError(
                    f"Native text replay token path is listed more than once: {resolved_path}"
                )
            resolved_paths.add(resolved_path)

            num_tokens = _require_int(source.get("num_tokens"), f"{prefix}.num_tokens", minimum=1)
            size_bytes = _require_int(source.get("size_bytes"), f"{prefix}.size_bytes", minimum=1)
            expected_size = num_tokens * dtype.itemsize
            if size_bytes != expected_size:
                raise OLMoConfigurationError(
                    f"Native text replay source {source_id!r} declares size_bytes={size_bytes}, "
                    f"but num_tokens * dtype.itemsize is {expected_size}"
                )
            source_sha256 = _require_string(source.get("sha256"), f"{prefix}.sha256").lower()
            if _SHA256_RE.fullmatch(source_sha256) is None:
                raise OLMoConfigurationError(
                    f"Native text replay source {source_id!r} has an invalid SHA-256"
                )

            raw_starts = source.get("window_starts")
            if not isinstance(raw_starts, list) or not raw_starts:
                raise OLMoConfigurationError(
                    f"Native text replay source {source_id!r} must select at least one window"
                )
            starts = tuple(
                _require_int(value, f"{prefix}.window_starts[{i}]")
                for i, value in enumerate(raw_starts)
            )
            previous_stop = -1
            for start in starts:
                if start % sequence_length:
                    raise OLMoConfigurationError(
                        f"Native text replay window start {start} for source {source_id!r} "
                        f"is not aligned to the parent {sequence_length}-token FSL grid"
                    )
                stop = start + window_length
                if stop > num_tokens:
                    raise OLMoConfigurationError(
                        f"Native text replay window [{start}, {stop}) is out of bounds for "
                        f"source {source_id!r} with {num_tokens} tokens"
                    )
                if start < previous_stop:
                    raise OLMoConfigurationError(
                        f"Native text replay windows for source {source_id!r} must be ordered "
                        "and non-overlapping"
                    )
                previous_stop = stop

            sources.append(
                NativeTextReplaySource(
                    source_id=source_id,
                    source_name=source_name,
                    parent_path_index=parent_path_index,
                    parent_path=parent_path,
                    token_path=token_path,
                    resolved_path=resolved_path,
                    num_tokens=num_tokens,
                    size_bytes=size_bytes,
                    sha256=source_sha256,
                    window_starts=starts,
                )
            )
            total_windows += len(starts)

        declared_num_windows = _require_int(data.get("num_windows"), "num_windows", minimum=1)
        if declared_num_windows != total_windows:
            raise OLMoConfigurationError(
                f"Native text replay manifest declares {declared_num_windows} windows but "
                f"contains {total_windows}"
            )

        try:
            canonical = json.dumps(
                data,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise OLMoConfigurationError(
                f"Native text replay manifest is not canonicalizable: {error}"
            ) from error
        content_fingerprint = hashlib.sha256(_FINGERPRINT_DOMAIN + canonical).hexdigest()

        return cls(
            path=manifest_path,
            sequence_length=sequence_length,
            dtype=dtype,
            tokenizer=tokenizer,
            provenance=provenance,
            sources=tuple(sources),
            num_windows=total_windows,
            manifest_sha256=hashlib.sha256(raw).hexdigest(),
            content_fingerprint=content_fingerprint,
        )


@dataclass(frozen=True)
class NativeTextReplayVerificationReceipt:
    """Pinned record of a completed offline source-integrity pass.

    A receipt is emitted only after the replay-manifest builder has streamed and verified
    every materialized source file. Runtime construction verifies the small receipt bytes,
    exact source metadata, and current file sizes without re-reading the full replay corpus.
    The receipt's SHA-256 must be pinned by the training configuration.
    """

    path: Path
    receipt_sha256: str
    source_catalog_sha256: str
    parent_paths_sha256: str
    materialized_sources_sha256: str
    sources: Mapping[str, Mapping[str, Any]]

    @classmethod
    def load(
        cls,
        path: os.PathLike[str] | str,
        *,
        expected_sha256: Optional[str] = None,
    ) -> "NativeTextReplayVerificationReceipt":
        """Load and strictly validate an offline verification receipt.

        :param path: Path to the receipt JSON.
        :param expected_sha256: Required SHA-256 of the exact receipt bytes.
        :returns: The validated receipt.
        :raises OLMoConfigurationError: If the receipt or its digest is invalid.
        """
        receipt_path = Path(path).expanduser().resolve()
        try:
            raw = receipt_path.read_bytes()
        except OSError as error:
            raise OLMoConfigurationError(
                f"Could not read native replay verification receipt {receipt_path}: {error}"
            ) from error
        receipt_sha256 = hashlib.sha256(raw).hexdigest()
        if expected_sha256 is not None and receipt_sha256 != expected_sha256:
            raise OLMoConfigurationError(
                "Native replay verification receipt SHA-256 does not match the pinned value: "
                f"expected {expected_sha256}, got {receipt_sha256}"
            )
        try:
            root = _require_mapping(
                json.loads(raw, object_pairs_hook=_strict_json_object),
                "verification_receipt",
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise OLMoConfigurationError(
                f"Invalid native replay verification receipt {receipt_path}: {error}"
            ) from error
        _require_exact_fields(
            root,
            {
                "format",
                "version",
                "hash_algorithm",
                "source_catalog_sha256",
                "parent_paths_sha256",
                "materialized_sources_sha256",
                "sources",
            },
            "verification receipt",
        )
        if root["format"] != NATIVE_TEXT_REPLAY_VERIFICATION_FORMAT:
            raise OLMoConfigurationError("Native replay verification receipt has the wrong format")
        version = _require_int(root["version"], "verification_receipt.version", minimum=1)
        if version != NATIVE_TEXT_REPLAY_VERIFICATION_VERSION:
            raise OLMoConfigurationError(
                f"Unsupported native replay verification receipt version {version}"
            )
        if root["hash_algorithm"] != "sha256":
            raise OLMoConfigurationError(
                "Native replay verification receipt hash_algorithm must be 'sha256'"
            )
        digests = {}
        for field_name in (
            "source_catalog_sha256",
            "parent_paths_sha256",
            "materialized_sources_sha256",
        ):
            digest = _require_string(root[field_name], f"verification_receipt.{field_name}")
            if _SHA256_RE.fullmatch(digest) is None:
                raise OLMoConfigurationError(
                    f"Native replay verification receipt {field_name} must be a SHA-256"
                )
            digests[field_name] = digest

        raw_sources = root["sources"]
        if not isinstance(raw_sources, list) or not raw_sources:
            raise OLMoConfigurationError(
                "Native replay verification receipt sources must be a non-empty list"
            )
        sources: Dict[str, Mapping[str, Any]] = {}
        parent_indices = set()
        resolved_paths = set()
        source_fields = {
            "id",
            "source",
            "parent_path_index",
            "parent_path",
            "resolved_path",
            "num_tokens",
            "size_bytes",
            "sha256",
        }
        for index, raw_source in enumerate(raw_sources):
            source = dict(_require_mapping(raw_source, f"verification_receipt.sources[{index}]"))
            _require_exact_fields(
                source,
                source_fields,
                f"verification receipt source {index}",
            )
            source_id = _require_string(source["id"], f"verification_receipt.sources[{index}].id")
            if source_id in sources:
                raise OLMoConfigurationError(
                    f"Native replay verification receipt repeats source {source_id!r}"
                )
            _require_string(source["source"], f"verification_receipt.sources[{index}].source")
            parent_index = _require_int(
                source["parent_path_index"],
                f"verification_receipt.sources[{index}].parent_path_index",
            )
            if parent_index in parent_indices:
                raise OLMoConfigurationError(
                    "Native replay verification receipt repeats a parent path index"
                )
            parent_indices.add(parent_index)
            _require_string(
                source["parent_path"],
                f"verification_receipt.sources[{index}].parent_path",
            )
            resolved_path = Path(
                _require_string(
                    source["resolved_path"],
                    f"verification_receipt.sources[{index}].resolved_path",
                )
            )
            if not resolved_path.is_absolute() or resolved_path in resolved_paths:
                raise OLMoConfigurationError(
                    "Native replay verification receipt paths must be unique and absolute"
                )
            resolved_paths.add(resolved_path)
            num_tokens = _require_int(
                source["num_tokens"],
                f"verification_receipt.sources[{index}].num_tokens",
                minimum=1,
            )
            size_bytes = _require_int(
                source["size_bytes"],
                f"verification_receipt.sources[{index}].size_bytes",
                minimum=1,
            )
            if size_bytes != num_tokens * np.dtype(np.uint32).itemsize:
                raise OLMoConfigurationError(
                    f"Native replay verification receipt source {source_id!r} has an "
                    "inconsistent token count and byte size"
                )
            source_sha256 = _require_string(
                source["sha256"], f"verification_receipt.sources[{index}].sha256"
            )
            if _SHA256_RE.fullmatch(source_sha256) is None:
                raise OLMoConfigurationError(
                    f"Native replay verification receipt source {source_id!r} has an "
                    "invalid SHA-256"
                )
            sources[source_id] = source

        return cls(
            path=receipt_path,
            receipt_sha256=receipt_sha256,
            source_catalog_sha256=digests["source_catalog_sha256"],
            parent_paths_sha256=digests["parent_paths_sha256"],
            materialized_sources_sha256=digests["materialized_sources_sha256"],
            sources=sources,
        )

    def validate_manifest(self, manifest: NativeTextReplayManifest) -> None:
        """Require every selected manifest source to match this verified receipt."""
        provenance = manifest.provenance
        expected = {
            "source_catalog_sha256": self.source_catalog_sha256,
            "parent_paths_sha256": self.parent_paths_sha256,
            "materialized_sources_sha256": self.materialized_sources_sha256,
            "verification_receipt_sha256": self.receipt_sha256,
        }
        for field_name, expected_value in expected.items():
            if provenance.get(field_name) != expected_value:
                raise OLMoConfigurationError(
                    f"Native replay manifest {field_name} does not match its verification receipt"
                )
        for source in manifest.sources:
            receipt_source = self.sources.get(source.source_id)
            actual = {
                "id": source.source_id,
                "source": source.source_name,
                "parent_path_index": source.parent_path_index,
                "parent_path": source.parent_path,
                "resolved_path": str(source.resolved_path),
                "num_tokens": source.num_tokens,
                "size_bytes": source.size_bytes,
                "sha256": source.sha256,
            }
            if receipt_source != actual:
                raise OLMoConfigurationError(
                    f"Native replay source {source.source_id!r} differs from the pinned "
                    "offline verification receipt"
                )


@dataclass
class NativeTextReplayDatasetConfig(Config):
    """Configuration for :class:`NativeTextReplayDataset`.

    :param manifest_path: Path to a finite replay manifest.
    :param expected_fingerprint: Optional semantic content fingerprint to require.
    :param expected_parent_checkpoint: Optional parent checkpoint identity to require.
    :param expected_parent_mix: Optional parent-mix name to require, such as
        ``"OLMo-mix-0925"``.
    :param expected_parent_paths_sha256: Optional digest of the exact expanded parent path
        manifest to require.
    :param verification_receipt_path: Optional pinned receipt from the builder's completed
        offline full-file integrity pass.
    :param expected_verification_receipt_sha256: SHA-256 of the exact receipt bytes.
    :param validate_source_files: Check each materialized file's size at construction.
    :param verify_source_hashes: Also stream and verify every source SHA-256 at construction.
        This is intentionally opt-in because a production replay corpus can be large.
    :param validate_token_ids: Check each loaded window against the manifest vocabulary.
    :param max_open_files: Per-process LRU bound for source memmaps.
    """

    manifest_path: str
    expected_fingerprint: Optional[str] = None
    expected_parent_checkpoint: Optional[str] = None
    expected_parent_mix: Optional[str] = None
    expected_parent_paths_sha256: Optional[str] = None
    verification_receipt_path: Optional[str] = None
    expected_verification_receipt_sha256: Optional[str] = None
    validate_source_files: bool = True
    verify_source_hashes: bool = False
    validate_token_ids: bool = True
    max_open_files: int = 8

    def build(self, tokenizer=None) -> "NativeTextReplayDataset":
        """Build the dataset and optionally validate a runtime tokenizer's native IDs."""
        dataset = NativeTextReplayDataset(
            self.manifest_path,
            expected_fingerprint=self.expected_fingerprint,
            expected_parent_checkpoint=self.expected_parent_checkpoint,
            expected_parent_mix=self.expected_parent_mix,
            expected_parent_paths_sha256=self.expected_parent_paths_sha256,
            verification_receipt_path=self.verification_receipt_path,
            expected_verification_receipt_sha256=self.expected_verification_receipt_sha256,
            validate_source_files=self.validate_source_files,
            verify_source_hashes=self.verify_source_hashes,
            validate_token_ids=self.validate_token_ids,
            max_open_files=self.max_open_files,
        )
        if tokenizer is not None:
            dataset.validate_tokenizer(tokenizer)
        return dataset


class NativeTextReplayDataset:
    """Finite map-style native-text replay dataset for the multimodal training stack.

    The dataset is invariant to source epoch: :meth:`get` intentionally ignores its
    ``epoch`` argument so the same manifest index always denotes the same token window.
    Data-loader shuffling may change order, but never window contents.
    """

    def __init__(
        self,
        manifest_path: os.PathLike[str] | str,
        *,
        expected_fingerprint: Optional[str] = None,
        expected_parent_checkpoint: Optional[str] = None,
        expected_parent_mix: Optional[str] = None,
        expected_parent_paths_sha256: Optional[str] = None,
        verification_receipt_path: Optional[os.PathLike[str] | str] = None,
        expected_verification_receipt_sha256: Optional[str] = None,
        validate_source_files: bool = True,
        verify_source_hashes: bool = False,
        validate_token_ids: bool = True,
        max_open_files: int = 8,
    ):
        if max_open_files <= 0:
            raise OLMoConfigurationError("max_open_files must be positive")
        if verify_source_hashes and not validate_source_files:
            raise OLMoConfigurationError(
                "verify_source_hashes=True requires validate_source_files=True"
            )
        if (verification_receipt_path is None) != (expected_verification_receipt_sha256 is None):
            raise OLMoConfigurationError(
                "Native replay verification receipt path and SHA-256 must be provided together"
            )
        self.manifest = NativeTextReplayManifest.load(manifest_path)
        if (
            expected_fingerprint is not None
            and self.manifest.content_fingerprint != expected_fingerprint
        ):
            raise OLMoConfigurationError(
                "Native text replay content fingerprint does not match the expected value: "
                f"expected {expected_fingerprint}, got {self.manifest.content_fingerprint}"
            )
        parent_checkpoint = self.manifest.provenance["parent_checkpoint"]
        if (
            expected_parent_checkpoint is not None
            and parent_checkpoint != expected_parent_checkpoint
        ):
            raise OLMoConfigurationError(
                f"Native text replay parent checkpoint is {parent_checkpoint!r}, expected "
                f"{expected_parent_checkpoint!r}"
            )
        parent_mix = self.manifest.provenance["parent_mix"]
        if expected_parent_mix is not None and parent_mix != expected_parent_mix:
            raise OLMoConfigurationError(
                f"Native text replay parent mix is {parent_mix!r}, expected "
                f"{expected_parent_mix!r}"
            )
        parent_paths_sha256 = self.manifest.provenance["parent_paths_sha256"]
        if (
            expected_parent_paths_sha256 is not None
            and parent_paths_sha256 != expected_parent_paths_sha256
        ):
            raise OLMoConfigurationError(
                "Native text replay parent path-list fingerprint does not match the expected "
                f"value: expected {expected_parent_paths_sha256}, got {parent_paths_sha256}"
            )
        if expected_parent_checkpoint is not None and expected_parent_paths_sha256 is not None:
            parent_paths_file = Path(expected_parent_checkpoint) / "data_paths.txt"
            try:
                parent_paths_raw = parent_paths_file.read_bytes()
            except OSError as error:
                raise OLMoConfigurationError(
                    f"Could not read parent path manifest {parent_paths_file}: {error}"
                ) from error
            actual_parent_paths_sha = hashlib.sha256(parent_paths_raw).hexdigest()
            if actual_parent_paths_sha != expected_parent_paths_sha256:
                raise OLMoConfigurationError(
                    f"Parent path manifest {parent_paths_file} has SHA-256 "
                    f"{actual_parent_paths_sha}, expected {expected_parent_paths_sha256}"
                )
            parent_paths = parent_paths_raw.decode("utf-8").splitlines()
            for source in self.manifest.sources:
                if (
                    source.parent_path_index >= len(parent_paths)
                    or parent_paths[source.parent_path_index] != source.parent_path
                ):
                    raise OLMoConfigurationError(
                        f"Native replay source {source.source_id!r} does not map to pinned "
                        f"parent path index {source.parent_path_index}"
                    )

        self.verification_receipt: Optional[NativeTextReplayVerificationReceipt] = None
        if verification_receipt_path is not None:
            self.verification_receipt = NativeTextReplayVerificationReceipt.load(
                verification_receipt_path,
                expected_sha256=expected_verification_receipt_sha256,
            )
            self.verification_receipt.validate_manifest(self.manifest)

        self.validate_token_ids = validate_token_ids
        self.max_open_files = max_open_files
        self._cumulative_windows = tuple(
            np.cumsum([len(source.window_starts) for source in self.manifest.sources]).tolist()
        )
        self._mmap_cache: "OrderedDict[int, np.memmap]" = OrderedDict()
        self._cache_lock = threading.RLock()

        if validate_source_files:
            self._validate_source_files(verify_hashes=verify_source_hashes)

    @property
    def fingerprint(self) -> str:
        """Semantic content fingerprint suitable for loader-state validation."""
        return self.manifest.content_fingerprint

    @property
    def fingerprint_version(self) -> str:
        """Version of :attr:`fingerprint`."""
        return f"native-text-replay-v{NATIVE_TEXT_REPLAY_VERSION}"

    @property
    def sequence_length(self) -> int:
        """Number of input and label tokens in every example."""
        return self.manifest.sequence_length

    @property
    def source_counts(self) -> Mapping[str, int]:
        """Number of selected windows by manifest source label."""
        counts: Dict[str, int] = {}
        for source in self.manifest.sources:
            counts[source.source_name] = counts.get(source.source_name, 0) + len(
                source.window_starts
            )
        return counts

    def _validate_source_files(self, *, verify_hashes: bool) -> None:
        for source in self.manifest.sources:
            try:
                actual_size = source.resolved_path.stat().st_size
            except OSError as error:
                raise OLMoConfigurationError(
                    f"Could not stat native text replay source {source.resolved_path}: {error}"
                ) from error
            if actual_size != source.size_bytes:
                raise OLMoConfigurationError(
                    f"Native text replay source {source.resolved_path} has {actual_size} bytes, "
                    f"expected {source.size_bytes}"
                )
            if verify_hashes:
                actual_hash = _sha256_file(source.resolved_path)
                if actual_hash != source.sha256:
                    raise OLMoConfigurationError(
                        f"Native text replay source {source.resolved_path} has SHA-256 "
                        f"{actual_hash}, expected {source.sha256}"
                    )

    def validate_tokenizer(self, tokenizer) -> None:
        """Verify native EOS and padding IDs against a runtime tokenizer.

        Vocabulary length is not compared because vision setup appends reserved image tokens
        after loading the native tokenizer. The native EOS and padding IDs must remain fixed.

        :param tokenizer: A tokenizer exposing ``eos_token_id`` and ``pad_token_id``.
        :raises OLMoConfigurationError: If either ID differs from the manifest.
        """
        expected_eos = int(self.manifest.tokenizer["eos_token_id"])
        expected_pad = int(self.manifest.tokenizer["pad_token_id"])
        actual_eos = getattr(tokenizer, "eos_token_id", None)
        actual_pad = getattr(tokenizer, "pad_token_id", None)
        if actual_eos != expected_eos or actual_pad != expected_pad:
            raise OLMoConfigurationError(
                "Runtime tokenizer does not preserve native replay token IDs: "
                f"EOS {actual_eos!r} != {expected_eos}, or pad {actual_pad!r} != {expected_pad}"
            )

    def __len__(self) -> int:
        return self.manifest.num_windows

    def _locate(self, index: int) -> Tuple[int, int, int]:
        normalized = int(index)
        if normalized < 0:
            normalized += len(self)
        if normalized < 0 or normalized >= len(self):
            raise IndexError(f"{index} is out of bounds for replay dataset of size {len(self)}")
        source_index = bisect.bisect_right(self._cumulative_windows, normalized)
        source_start = self._cumulative_windows[source_index - 1] if source_index else 0
        return normalized, source_index, normalized - source_start

    def provenance_for(self, index: int) -> Mapping[str, Any]:
        """Return immutable-window provenance for a dataset index.

        :param index: Dataset index, including ordinary negative indexing.
        :returns: Source identity, content hash, and exact half-open token interval.
        """
        normalized, source_index, source_window_index = self._locate(index)
        source = self.manifest.sources[source_index]
        start = source.window_starts[source_window_index]
        return {
            "dataset_fingerprint": self.fingerprint,
            "manifest_index": normalized,
            "source_id": source.source_id,
            "source": source.source_name,
            "path": source.token_path,
            "source_sha256": source.sha256,
            "start": start,
            "stop": start + self.sequence_length,
        }

    def _read_tokens(self, source_index: int, start: int) -> np.ndarray:
        source = self.manifest.sources[source_index]
        stop = start + self.sequence_length
        # Copy under the cache lock so another prefetch thread cannot evict and close the
        # memmap while this thread is materializing its view.
        with self._cache_lock:
            mmap = self._mmap_cache.pop(source_index, None)
            if mmap is None:
                mmap = np.memmap(source.resolved_path, mode="r", dtype=self.manifest.dtype)
            self._mmap_cache[source_index] = mmap
            tokens = np.array(mmap[start:stop], dtype=np.int64, copy=True)
            while len(self._mmap_cache) > self.max_open_files:
                _, evicted = self._mmap_cache.popitem(last=False)
                evicted._mmap.close()

        if len(tokens) != self.sequence_length:
            raise RuntimeError(
                f"Native text replay short read from {source.resolved_path}: expected "
                f"{self.sequence_length} tokens, got {len(tokens)}"
            )
        if self.validate_token_ids and tokens.size:
            max_token_id = int(tokens.max())
            vocab_size = int(self.manifest.tokenizer["vocab_size"])
            if max_token_id >= vocab_size:
                raise RuntimeError(
                    f"Native text replay source {source.source_id!r} contains token ID "
                    f"{max_token_id}, outside native vocabulary size {vocab_size}"
                )
        return tokens

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.get(index, 0)

    def get(self, index: int, epoch: int = 0) -> Dict[str, Any]:
        """Materialize one parent-equivalent fixed-length pretraining instance.

        ``epoch`` is accepted for compatibility with the multimodal mixture loader and is
        intentionally ignored. No token is injected, removed, or retokenized.

        :param index: Manifest-window index.
        :param epoch: Ignored source epoch.
        :returns: A multimodal-stack example with empty image arrays.
        """
        del epoch
        normalized, source_index, source_window_index = self._locate(index)
        source = self.manifest.sources[source_index]
        start = source.window_starts[source_window_index]
        tokens = self._read_tokens(source_index, start)
        labels = np.full(self.sequence_length, -100, dtype=np.int64)
        labels[:-1] = tokens[1:]
        loss_masks = np.ones(self.sequence_length, dtype=np.float32)
        loss_masks[-1] = 0.0
        instance_filter = self.manifest.provenance["instance_filter"]
        valid_instance = not any(
            match.times >= instance_filter["repetition_max_count"]
            for match in find_periodic_sequences(
                tokens,
                min_period=instance_filter["repetition_min_period"],
                max_period=instance_filter["repetition_max_period"],
            )
        )
        if not valid_instance:
            # The parent text train module masks the numerator via ignore-index labels but
            # deliberately adds the filtered instance's L-1 next-token positions back to the
            # batch divisor. Preserve that behavior in the multimodal weighted-loss path while
            # making every target ignored.
            labels.fill(-100)
        metadata = dict(self.provenance_for(normalized))
        metadata["instance_filter_valid"] = valid_instance
        return {
            "input_ids": tokens,
            "labels": labels,
            "loss_masks": loss_masks,
            "position_ids": np.arange(self.sequence_length, dtype=np.int64),
            "token_type_ids": np.zeros(self.sequence_length, dtype=np.int64),
            "images": np.zeros((0, N_PATCHES_SQ, PATCH_DIM), dtype=np.float32),
            "pooled_patches_idx": np.full((0, POOL_H * POOL_W), -1, dtype=np.int64),
            "metadata": metadata,
        }

    def close(self) -> None:
        """Close all memmaps currently held by this process."""
        with self._cache_lock:
            for mmap in self._mmap_cache.values():
                mmap._mmap.close()
            self._mmap_cache.clear()

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_mmap_cache"] = OrderedDict()
        state["_cache_lock"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._mmap_cache = OrderedDict()
        self._cache_lock = threading.RLock()

    def __del__(self):
        # Interpreter shutdown can leave partially initialized attributes or modules.
        try:
            self.close()
        except (AttributeError, OSError):
            pass
