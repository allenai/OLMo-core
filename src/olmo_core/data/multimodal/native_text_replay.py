"""Native next-token replay for multimodal continued pretraining.

This module adapts a finite, pretokenized text replay manifest to the example contract used
by :mod:`olmo_core.data.multimodal`.  It deliberately does not tokenize, concatenate, or add
document markers: the IDs (including EOS document boundaries) are read directly from the
parent language model's pretraining arrays. A manifest window contains exactly
``sequence_length`` tokens on the parent's fixed-length grid. Labels shift within that window,
the final label is ignored, and the parent's repetition filter can mask the whole instance.

The compact manifest is a JSON object with this schema::

    {
      "format": "olmo_native_text_replay",
      "version": 3,
      "sequence_length": 8192,
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
        "parent_mix_sha256": "...",
        "parent_config_sha256": "...",
        "parent_trainer_state_sha256": "...",
        "parent_dataset_fingerprint": "...",
        "remote_snapshot_sha256": "...",
        "compact_materialization_sha256": "...",
        "builder_implementation": "offline-compact-replay-builder-v1",
        "builder_sha256": "...",
        "instance_filter": {
          "repetition_min_period": 1,
          "repetition_max_period": 13,
          "repetition_max_count": 32
        },
        "selection_algorithm": "affine-grid-v1",
        "selection_seed": 6198,
        "split": "train",
        "usable_tokens": 16382,
        "source_usable_tokens": {"web": 16382},
        "minimum_source_usable_tokens": {},
        "raw_tokens_per_window": 8192,
        "loss_tokens_per_window": 8191,
        "verification_receipt_sha256": "..."
      },
      "num_windows": 2,
      "sources": [
        {
          "id": "web-000000",
          "source": "web",
          "parent_path_index": 0,
          "parent_path": "s3://ai2-llm/.../000000.npy",
          "path": "tokens/web-000000.npy",
          "parent_num_tokens": 16384,
          "num_tokens": 16384,
          "size_bytes": 65536,
          "sha256": "...",
          "window_starts": [0, 8192],
          "parent_window_starts": [0, 8192]
        }
      ]
    }

Token files are headerless numpy memmaps, matching OLMo's preprocessed ``.npy`` arrays.
Compact paths must be normalized relative paths beneath the manifest directory and are opened
without following symbolic links. Each source records consecutive compact-local window starts
and the corresponding parent window starts. A shared receipt binds the complete train and
holdout materializations. These constraints make the replay set bounded and deterministic;
stochastic ordering belongs in the data loader.
"""

from __future__ import annotations

import base64
import binascii
import bisect
import hashlib
import json
import os
import re
import stat
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any

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
NATIVE_TEXT_REPLAY_VERSION = 3
NATIVE_TEXT_REPLAY_VERIFICATION_FORMAT = "olmo_native_text_replay_verification_receipt"
NATIVE_TEXT_REPLAY_VERIFICATION_VERSION = 3
S002_INSTANCE_FILTER = {
    "repetition_min_period": 1,
    "repetition_max_period": 13,
    "repetition_max_count": 32,
}
_FINGERPRINT_DOMAIN = b"olmo-native-text-replay-v3\0"
_SUPPORTED_DTYPES = {
    "uint8": np.dtype(np.uint8),
    "uint16": np.dtype(np.uint16),
    "uint32": np.dtype(np.uint32),
    "uint64": np.dtype(np.uint64),
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ROOT_FIELDS = {
    "format",
    "version",
    "sequence_length",
    "dtype",
    "tokenizer",
    "provenance",
    "num_windows",
    "sources",
}
_TOKENIZER_FIELDS = {"identifier", "vocab_size", "eos_token_id", "pad_token_id"}
_PROVENANCE_FIELDS = {
    "parent_checkpoint",
    "parent_mix",
    "parent_paths_sha256",
    "parent_mix_sha256",
    "parent_config_sha256",
    "parent_trainer_state_sha256",
    "parent_dataset_fingerprint",
    "remote_snapshot_sha256",
    "compact_materialization_sha256",
    "builder_implementation",
    "builder_sha256",
    "instance_filter",
    "selection_algorithm",
    "selection_seed",
    "split",
    "usable_tokens",
    "source_usable_tokens",
    "minimum_source_usable_tokens",
    "raw_tokens_per_window",
    "loss_tokens_per_window",
    "verification_receipt_sha256",
}
_SOURCE_FIELDS = {
    "id",
    "source",
    "parent_path_index",
    "parent_path",
    "path",
    "parent_num_tokens",
    "num_tokens",
    "size_bytes",
    "sha256",
    "window_starts",
    "parent_window_starts",
}
_RECEIPT_FIELDS = {
    "format",
    "version",
    "hash_algorithm",
    "builder_implementation",
    "builder_sha256",
    "parent_paths_sha256",
    "parent_mix_sha256",
    "parent_config_sha256",
    "parent_trainer_state_sha256",
    "parent_dataset_fingerprint",
    "remote_snapshot_sha256",
    "compact_materialization_sha256",
    "manifest_contract_sha256",
    "mirror_policy",
    "remote_sources",
    "splits",
}
_MIRROR_POLICY = "s3-to-gs-same-bucket-key-v1"
_REMOTE_SOURCE_FIELDS = {
    "parent_path_index",
    "parent_path",
    "mirror_uri",
    "size_bytes",
    "num_tokens",
    "generation",
    "etag",
    "md5_hash",
    "crc32c",
    "source_etag",
}
_RECEIPT_SOURCE_FIELDS = {
    "id",
    "source",
    "parent_path_index",
    "parent_path",
    "path",
    "resolved_path",
    "parent_num_tokens",
    "num_tokens",
    "size_bytes",
    "mtime_ns",
    "ctime_ns",
    "inode",
    "device",
    "sha256",
    "num_windows",
    "parent_window_starts_sha256",
}


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


def _strict_json_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
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


def _require_sha256(value: Any, name: str) -> str:
    digest = _require_string(value, name)
    if _SHA256_RE.fullmatch(digest) is None:
        raise OLMoConfigurationError(f"Native text replay field {name!r} must be a SHA-256")
    return digest


def _require_optional_string(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, name)


def _canonical_json(value: Any, name: str) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise OLMoConfigurationError(
            f"Native text replay {name} is not canonicalizable: {error}"
        ) from error


def _canonical_sha256(value: Any, name: str) -> str:
    return hashlib.sha256(_canonical_json(value, name)).hexdigest()


def _require_base64_bytes(value: Any, name: str, *, length: int) -> str:
    encoded = _require_string(value, name)
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise OLMoConfigurationError(
            f"Native text replay field {name!r} must be canonical base64"
        ) from error
    if len(decoded) != length or base64.b64encode(decoded).decode("ascii") != encoded:
        raise OLMoConfigurationError(
            f"Native text replay field {name!r} must encode exactly {length} bytes"
        )
    return encoded


def _require_compact_relative_path(value: Any, name: str) -> str:
    token_path = _require_string(value, name)
    pure_path = PurePosixPath(token_path)
    if (
        "://" in token_path
        or "\\" in token_path
        or pure_path.is_absolute()
        or token_path != pure_path.as_posix()
        or any(part in {"", ".", ".."} for part in pure_path.parts)
    ):
        raise OLMoConfigurationError(
            f"Native text replay field {name!r} must be a normalized relative POSIX path"
        )
    return token_path


def _resolve_compact_path(manifest_path: Path, token_path: str, name: str) -> Path:
    root = manifest_path.parent.resolve()
    compact_path = root / token_path
    resolved = compact_path.resolve()
    if not resolved.is_relative_to(root):
        raise OLMoConfigurationError(
            f"Native text replay field {name!r} resolves outside the manifest directory"
        )
    if resolved != compact_path:
        raise OLMoConfigurationError(
            f"Native text replay field {name!r} must not traverse symbolic links"
        )
    return compact_path


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
    parent_num_tokens: int
    num_tokens: int
    size_bytes: int
    sha256: str
    window_starts: tuple[int, ...]
    parent_window_starts: tuple[int, ...]


@dataclass(frozen=True)
class NativeTextReplayManifest:
    """Validated replay-manifest contents and stable content identifiers."""

    path: Path
    version: int
    sequence_length: int
    dtype: np.dtype
    tokenizer: Mapping[str, Any]
    provenance: Mapping[str, Any]
    sources: tuple[NativeTextReplaySource, ...]
    num_windows: int
    manifest_sha256: str
    content_fingerprint: str
    manifest_contract_sha256: str

    @classmethod
    def load(cls, path: os.PathLike[str] | str) -> NativeTextReplayManifest:
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
        _require_exact_fields(data, _ROOT_FIELDS, "manifest root")

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
        if dtype != np.dtype(np.uint32):
            raise OLMoConfigurationError("Native text replay v3 dtype must be 'uint32'")

        tokenizer = dict(_require_mapping(data.get("tokenizer"), "tokenizer"))
        _require_exact_fields(tokenizer, _TOKENIZER_FIELDS, "tokenizer")
        _require_string(tokenizer.get("identifier"), "tokenizer.identifier")
        vocab_size = _require_int(tokenizer.get("vocab_size"), "tokenizer.vocab_size", minimum=1)
        eos_token_id = _require_int(tokenizer.get("eos_token_id"), "tokenizer.eos_token_id")
        pad_token_id = _require_int(tokenizer.get("pad_token_id"), "tokenizer.pad_token_id")
        if eos_token_id >= vocab_size or pad_token_id >= vocab_size:
            raise OLMoConfigurationError(
                "Native text replay tokenizer EOS and pad IDs must be below vocab_size"
            )

        provenance = dict(_require_mapping(data.get("provenance"), "provenance"))
        _require_exact_fields(provenance, _PROVENANCE_FIELDS, "provenance")
        _require_string(provenance.get("parent_checkpoint"), "provenance.parent_checkpoint")
        _require_string(provenance.get("parent_mix"), "provenance.parent_mix")
        digest_fields = (
            "parent_paths_sha256",
            "parent_mix_sha256",
            "parent_config_sha256",
            "parent_trainer_state_sha256",
            "parent_dataset_fingerprint",
            "remote_snapshot_sha256",
            "compact_materialization_sha256",
            "builder_sha256",
            "verification_receipt_sha256",
        )
        for field_name in digest_fields:
            provenance[field_name] = _require_sha256(
                provenance.get(field_name), f"provenance.{field_name}"
            )
        _require_string(
            provenance.get("builder_implementation"), "provenance.builder_implementation"
        )
        selection_algorithm = _require_string(
            provenance.get("selection_algorithm"), "provenance.selection_algorithm"
        )
        if selection_algorithm != "affine-grid-v1":
            raise OLMoConfigurationError(
                "Native text replay provenance.selection_algorithm must be 'affine-grid-v1'"
            )
        _require_int(provenance.get("selection_seed"), "provenance.selection_seed")
        split = _require_string(provenance.get("split"), "provenance.split")
        if split not in {"train", "holdout"}:
            raise OLMoConfigurationError(
                "Native text replay provenance.split must be 'train' or 'holdout'"
            )
        usable_tokens = _require_int(
            provenance.get("usable_tokens"), "provenance.usable_tokens", minimum=1
        )
        raw_tokens_per_window = _require_int(
            provenance.get("raw_tokens_per_window"),
            "provenance.raw_tokens_per_window",
            minimum=2,
        )
        loss_tokens_per_window = _require_int(
            provenance.get("loss_tokens_per_window"),
            "provenance.loss_tokens_per_window",
            minimum=1,
        )
        if (
            raw_tokens_per_window != sequence_length
            or loss_tokens_per_window != sequence_length - 1
        ):
            raise OLMoConfigurationError(
                "Native text replay provenance window-token counts do not match sequence_length"
            )

        raw_source_usable_tokens = _require_mapping(
            provenance.get("source_usable_tokens"), "provenance.source_usable_tokens"
        )
        if not raw_source_usable_tokens:
            raise OLMoConfigurationError(
                "Native text replay provenance.source_usable_tokens must be non-empty"
            )
        source_usable_tokens = {
            _require_string(source_name, "provenance.source_usable_tokens key"): _require_int(
                tokens,
                f"provenance.source_usable_tokens.{source_name}",
                minimum=1,
            )
            for source_name, tokens in raw_source_usable_tokens.items()
        }
        provenance["source_usable_tokens"] = source_usable_tokens
        raw_minimum_source_usable_tokens = _require_mapping(
            provenance.get("minimum_source_usable_tokens"),
            "provenance.minimum_source_usable_tokens",
        )
        minimum_source_usable_tokens = {
            _require_string(
                source_name, "provenance.minimum_source_usable_tokens key"
            ): _require_int(
                tokens,
                f"provenance.minimum_source_usable_tokens.{source_name}",
                minimum=1,
            )
            for source_name, tokens in raw_minimum_source_usable_tokens.items()
        }
        provenance["minimum_source_usable_tokens"] = minimum_source_usable_tokens
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

        sources: list[NativeTextReplaySource] = []
        source_ids = set()
        parent_path_indices = set()
        resolved_paths = set()
        total_windows = 0
        for source_index, raw_source in enumerate(raw_sources):
            source = _require_mapping(raw_source, f"sources[{source_index}]")
            prefix = f"sources[{source_index}]"
            _require_exact_fields(source, _SOURCE_FIELDS, prefix)
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
            token_path = _require_compact_relative_path(source.get("path"), f"{prefix}.path")
            resolved_path = _resolve_compact_path(manifest_path, token_path, f"{prefix}.path")
            if resolved_path in resolved_paths:
                raise OLMoConfigurationError(
                    f"Native text replay token path is listed more than once: {resolved_path}"
                )
            resolved_paths.add(resolved_path)

            num_tokens = _require_int(source.get("num_tokens"), f"{prefix}.num_tokens", minimum=1)
            parent_num_tokens = _require_int(
                source.get("parent_num_tokens"),
                f"{prefix}.parent_num_tokens",
                minimum=1,
            )
            size_bytes = _require_int(source.get("size_bytes"), f"{prefix}.size_bytes", minimum=1)
            expected_size = num_tokens * dtype.itemsize
            if size_bytes != expected_size:
                raise OLMoConfigurationError(
                    f"Native text replay source {source_id!r} declares size_bytes={size_bytes}, "
                    f"but num_tokens * dtype.itemsize is {expected_size}"
                )
            source_sha256 = _require_sha256(source.get("sha256"), f"{prefix}.sha256")

            raw_starts = source.get("window_starts")
            if not isinstance(raw_starts, list) or not raw_starts:
                raise OLMoConfigurationError(
                    f"Native text replay source {source_id!r} must select at least one window"
                )
            starts = tuple(
                _require_int(value, f"{prefix}.window_starts[{i}]")
                for i, value in enumerate(raw_starts)
            )
            expected_local_starts = tuple(range(0, len(starts) * sequence_length, sequence_length))
            if starts != expected_local_starts:
                raise OLMoConfigurationError(
                    f"Native text replay v3 source {source_id!r} local window_starts must "
                    "enumerate every compact window consecutively from zero"
                )
            if num_tokens != len(starts) * sequence_length:
                raise OLMoConfigurationError(
                    f"Native text replay v3 source {source_id!r} num_tokens must equal its "
                    "window count times sequence_length"
                )
            raw_parent_starts = source.get("parent_window_starts")
            if not isinstance(raw_parent_starts, list) or not raw_parent_starts:
                raise OLMoConfigurationError(
                    f"Native text replay v3 source {source_id!r} must enumerate parent windows"
                )
            parent_starts = tuple(
                _require_int(value, f"{prefix}.parent_window_starts[{i}]")
                for i, value in enumerate(raw_parent_starts)
            )
            if len(parent_starts) != len(starts):
                raise OLMoConfigurationError(
                    f"Native text replay v3 source {source_id!r} local and parent window "
                    "counts must match"
                )
            previous_parent_stop = -1
            for parent_start in parent_starts:
                if parent_start % sequence_length:
                    raise OLMoConfigurationError(
                        f"Native text replay parent window start {parent_start} for source "
                        f"{source_id!r} is not aligned to the {sequence_length}-token grid"
                    )
                parent_stop = parent_start + window_length
                if parent_stop > parent_num_tokens:
                    raise OLMoConfigurationError(
                        f"Native text replay parent window [{parent_start}, {parent_stop}) is "
                        f"out of bounds for source {source_id!r} with "
                        f"{parent_num_tokens} parent tokens"
                    )
                if parent_start < previous_parent_stop:
                    raise OLMoConfigurationError(
                        f"Native text replay parent windows for source {source_id!r} must be "
                        "ordered and non-overlapping"
                    )
                previous_parent_stop = parent_stop

            sources.append(
                NativeTextReplaySource(
                    source_id=source_id,
                    source_name=source_name,
                    parent_path_index=parent_path_index,
                    parent_path=parent_path,
                    token_path=token_path,
                    resolved_path=resolved_path,
                    parent_num_tokens=parent_num_tokens,
                    num_tokens=num_tokens,
                    size_bytes=size_bytes,
                    sha256=source_sha256,
                    window_starts=starts,
                    parent_window_starts=parent_starts,
                )
            )
            total_windows += len(starts)

        declared_num_windows = _require_int(data.get("num_windows"), "num_windows", minimum=1)
        if declared_num_windows != total_windows:
            raise OLMoConfigurationError(
                f"Native text replay manifest declares {declared_num_windows} windows but "
                f"contains {total_windows}"
            )
        expected_usable_tokens = total_windows * loss_tokens_per_window
        if usable_tokens != expected_usable_tokens:
            raise OLMoConfigurationError(
                f"Native text replay provenance declares {usable_tokens} usable tokens but "
                f"contains {expected_usable_tokens}"
            )
        computed_source_usable_tokens: dict[str, int] = {}
        for manifest_source in sources:
            computed_source_usable_tokens[manifest_source.source_name] = (
                computed_source_usable_tokens.get(manifest_source.source_name, 0)
                + len(manifest_source.window_starts) * loss_tokens_per_window
            )
        if source_usable_tokens != computed_source_usable_tokens:
            raise OLMoConfigurationError(
                "Native text replay provenance.source_usable_tokens does not match its sources"
            )
        for source_name, minimum_tokens in minimum_source_usable_tokens.items():
            if minimum_tokens % loss_tokens_per_window:
                raise OLMoConfigurationError(
                    "Native text replay provenance.minimum_source_usable_tokens must use whole "
                    f"windows for source {source_name!r}"
                )
            if minimum_tokens > source_usable_tokens.get(source_name, 0):
                raise OLMoConfigurationError(
                    "Native text replay provenance.minimum_source_usable_tokens exceeds the "
                    f"selected tokens for source {source_name!r}"
                )

        canonical = _canonical_json(data, "manifest")
        content_fingerprint = hashlib.sha256(_FINGERPRINT_DOMAIN + canonical).hexdigest()
        contract = dict(data)
        contract_provenance = dict(_require_mapping(contract["provenance"], "provenance"))
        del contract_provenance["verification_receipt_sha256"]
        contract["provenance"] = contract_provenance
        manifest_contract_sha256 = _canonical_sha256(contract, "manifest contract")

        return cls(
            path=manifest_path,
            version=version,
            sequence_length=sequence_length,
            dtype=dtype,
            tokenizer=tokenizer,
            provenance=provenance,
            sources=tuple(sources),
            num_windows=total_windows,
            manifest_sha256=hashlib.sha256(raw).hexdigest(),
            content_fingerprint=content_fingerprint,
            manifest_contract_sha256=manifest_contract_sha256,
        )


@dataclass(frozen=True)
class NativeTextReplayVerificationReceipt:
    """Pinned record of a completed offline source-integrity pass.

    A receipt is emitted only after the replay-manifest builder has streamed and verified
    every materialized source file. Runtime construction verifies the small receipt bytes,
    exact builder and parent-lineage identities, exact source metadata, and current file stat
    signatures without re-reading the full replay corpus. The receipt's SHA-256 must be pinned
    by the training configuration.
    """

    path: Path
    version: int
    receipt_sha256: str
    builder_implementation: str
    builder_sha256: str
    parent_paths_sha256: str
    parent_mix_sha256: str
    parent_config_sha256: str
    parent_trainer_state_sha256: str
    parent_dataset_fingerprint: str
    remote_snapshot_sha256: str
    compact_materialization_sha256: str
    manifest_contract_sha256: Mapping[str, str]
    mirror_policy: str
    remote_sources: tuple[Mapping[str, Any], ...]
    split_sources: Mapping[str, tuple[Mapping[str, Any], ...]]

    @classmethod
    def load(
        cls,
        path: os.PathLike[str] | str,
        *,
        expected_sha256: str | None = None,
    ) -> NativeTextReplayVerificationReceipt:
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
        version = _require_int(root.get("version"), "verification_receipt.version", minimum=1)
        if version != NATIVE_TEXT_REPLAY_VERIFICATION_VERSION:
            raise OLMoConfigurationError(
                f"Unsupported native replay verification receipt version {version}; "
                f"expected {NATIVE_TEXT_REPLAY_VERIFICATION_VERSION}"
            )
        _require_exact_fields(root, _RECEIPT_FIELDS, "verification receipt")
        if root["format"] != NATIVE_TEXT_REPLAY_VERIFICATION_FORMAT:
            raise OLMoConfigurationError("Native replay verification receipt has the wrong format")
        if root["hash_algorithm"] != "sha256":
            raise OLMoConfigurationError(
                "Native replay verification receipt hash_algorithm must be 'sha256'"
            )
        builder_implementation = _require_string(
            root["builder_implementation"], "verification_receipt.builder_implementation"
        )
        digests = {}
        digest_fields = (
            "builder_sha256",
            "parent_paths_sha256",
            "parent_mix_sha256",
            "parent_config_sha256",
            "parent_trainer_state_sha256",
            "parent_dataset_fingerprint",
            "remote_snapshot_sha256",
            "compact_materialization_sha256",
        )
        for field_name in digest_fields:
            digests[field_name] = _require_sha256(
                root[field_name], f"verification_receipt.{field_name}"
            )
        mirror_policy = _require_string(root["mirror_policy"], "verification_receipt.mirror_policy")
        if mirror_policy != _MIRROR_POLICY:
            raise OLMoConfigurationError(
                f"Native replay v3 mirror_policy must be {_MIRROR_POLICY!r}"
            )
        raw_manifest_contracts = _require_mapping(
            root["manifest_contract_sha256"],
            "verification_receipt.manifest_contract_sha256",
        )
        _require_exact_fields(
            raw_manifest_contracts,
            {"train", "holdout"},
            "verification receipt manifest_contract_sha256",
        )
        manifest_contract_sha256 = {
            split: _require_sha256(
                raw_manifest_contracts[split],
                f"verification_receipt.manifest_contract_sha256.{split}",
            )
            for split in ("train", "holdout")
        }

        remote_sources: tuple[Mapping[str, Any], ...] = ()
        split_sources: dict[str, tuple[Mapping[str, Any], ...]] = {}
        if version == NATIVE_TEXT_REPLAY_VERIFICATION_VERSION:
            raw_remote_sources = root["remote_sources"]
            if not isinstance(raw_remote_sources, list) or not raw_remote_sources:
                raise OLMoConfigurationError(
                    "Native replay v3 verification receipt remote_sources must be non-empty"
                )
            normalized_remote_sources: list[Mapping[str, Any]] = []
            parent_paths = set()
            mirror_uris = set()
            for index, raw_remote_source in enumerate(raw_remote_sources):
                prefix = f"verification_receipt.remote_sources[{index}]"
                remote_source = dict(_require_mapping(raw_remote_source, prefix))
                _require_exact_fields(remote_source, _REMOTE_SOURCE_FIELDS, prefix)
                parent_index = _require_int(
                    remote_source["parent_path_index"], f"{prefix}.parent_path_index"
                )
                if parent_index != index:
                    raise OLMoConfigurationError(
                        "Native replay v3 remote_sources must use canonical contiguous parent "
                        "path indices"
                    )
                parent_path = _require_string(remote_source["parent_path"], f"{prefix}.parent_path")
                if not parent_path.startswith("s3://ai2-llm/") or parent_path in parent_paths:
                    raise OLMoConfigurationError(
                        "Native replay v3 remote parent paths must be unique s3://ai2-llm/ URIs"
                    )
                parent_paths.add(parent_path)
                mirror_uri = _require_string(remote_source["mirror_uri"], f"{prefix}.mirror_uri")
                expected_mirror_uri = "gs://ai2-llm/" + parent_path.removeprefix("s3://ai2-llm/")
                if mirror_uri != expected_mirror_uri or mirror_uri in mirror_uris:
                    raise OLMoConfigurationError(
                        f"Native replay v3 remote source {index} has the wrong mirror URI"
                    )
                mirror_uris.add(mirror_uri)
                num_tokens = _require_int(
                    remote_source["num_tokens"], f"{prefix}.num_tokens", minimum=1
                )
                size_bytes = _require_int(
                    remote_source["size_bytes"], f"{prefix}.size_bytes", minimum=1
                )
                if size_bytes != num_tokens * np.dtype(np.uint32).itemsize:
                    raise OLMoConfigurationError(
                        f"Native replay v3 remote source {index} has an inconsistent token "
                        "count and byte size"
                    )
                generation = _require_string(remote_source["generation"], f"{prefix}.generation")
                if not generation.isascii() or not generation.isdigit() or int(generation) <= 0:
                    raise OLMoConfigurationError(
                        f"Native replay field {prefix + '.generation'!r} must contain positive "
                        "decimal digits"
                    )
                _require_string(remote_source["etag"], f"{prefix}.etag")
                md5_hash = _require_optional_string(remote_source["md5_hash"], f"{prefix}.md5_hash")
                if md5_hash is not None:
                    _require_base64_bytes(md5_hash, f"{prefix}.md5_hash", length=16)
                _require_base64_bytes(remote_source["crc32c"], f"{prefix}.crc32c", length=4)
                _require_optional_string(remote_source["source_etag"], f"{prefix}.source_etag")
                normalized_remote_sources.append(remote_source)
            if (
                _canonical_sha256(normalized_remote_sources, "remote_sources")
                != digests["remote_snapshot_sha256"]
            ):
                raise OLMoConfigurationError(
                    "Native replay v3 remote_snapshot_sha256 does not match remote_sources"
                )
            remote_sources = tuple(normalized_remote_sources)

            raw_splits = _require_mapping(root["splits"], "verification_receipt.splits")
            _require_exact_fields(raw_splits, {"train", "holdout"}, "verification receipt splits")
            all_resolved_paths = set()
            for split in ("train", "holdout"):
                raw_split_sources = raw_splits[split]
                if not isinstance(raw_split_sources, list) or not raw_split_sources:
                    raise OLMoConfigurationError(
                        f"Native replay v3 verification receipt split {split!r} must be non-empty"
                    )
                normalized_split_sources: list[Mapping[str, Any]] = []
                split_source_ids = set()
                split_parent_indices = set()
                previous_parent_index = -1
                for index, raw_split_source in enumerate(raw_split_sources):
                    prefix = f"verification_receipt.splits.{split}[{index}]"
                    split_source = dict(_require_mapping(raw_split_source, prefix))
                    _require_exact_fields(split_source, _RECEIPT_SOURCE_FIELDS, prefix)
                    source_id = _require_string(split_source["id"], f"{prefix}.id")
                    if source_id in split_source_ids:
                        raise OLMoConfigurationError(
                            f"Native replay v3 receipt split {split!r} repeats source id "
                            f"{source_id!r}"
                        )
                    split_source_ids.add(source_id)
                    _require_string(split_source["source"], f"{prefix}.source")
                    parent_index = _require_int(
                        split_source["parent_path_index"], f"{prefix}.parent_path_index"
                    )
                    if (
                        parent_index in split_parent_indices
                        or parent_index <= previous_parent_index
                    ):
                        raise OLMoConfigurationError(
                            f"Native replay v3 receipt split {split!r} parent indices must be "
                            "unique and strictly ordered"
                        )
                    split_parent_indices.add(parent_index)
                    previous_parent_index = parent_index
                    if parent_index >= len(remote_sources):
                        raise OLMoConfigurationError(
                            f"Native replay v3 receipt split {split!r} references an unknown "
                            "remote parent"
                        )
                    parent_remote_source = remote_sources[parent_index]
                    parent_path = _require_string(
                        split_source["parent_path"], f"{prefix}.parent_path"
                    )
                    if parent_path != parent_remote_source["parent_path"]:
                        raise OLMoConfigurationError(
                            f"Native replay v3 receipt split {split!r} parent path differs from "
                            "remote_sources"
                        )
                    token_path = _require_compact_relative_path(
                        split_source["path"], f"{prefix}.path"
                    )
                    split_source["path"] = token_path
                    raw_resolved_path = _require_string(
                        split_source["resolved_path"], f"{prefix}.resolved_path"
                    )
                    resolved_path = Path(raw_resolved_path)
                    if (
                        not resolved_path.is_absolute()
                        or str(resolved_path) != raw_resolved_path
                        or resolved_path.resolve() != resolved_path
                        or resolved_path in all_resolved_paths
                    ):
                        raise OLMoConfigurationError(
                            "Native replay v3 receipt resolved paths must be normalized, unique, "
                            "and absolute"
                        )
                    all_resolved_paths.add(resolved_path)
                    parent_num_tokens = _require_int(
                        split_source["parent_num_tokens"],
                        f"{prefix}.parent_num_tokens",
                        minimum=1,
                    )
                    if parent_num_tokens != parent_remote_source["num_tokens"]:
                        raise OLMoConfigurationError(
                            f"Native replay v3 receipt split {split!r} parent token count differs "
                            "from remote_sources"
                        )
                    num_tokens = _require_int(
                        split_source["num_tokens"], f"{prefix}.num_tokens", minimum=1
                    )
                    size_bytes = _require_int(
                        split_source["size_bytes"], f"{prefix}.size_bytes", minimum=1
                    )
                    if size_bytes != num_tokens * np.dtype(np.uint32).itemsize:
                        raise OLMoConfigurationError(
                            f"Native replay v3 receipt source {source_id!r} has an inconsistent "
                            "token count and byte size"
                        )
                    for stat_field in ("mtime_ns", "ctime_ns", "inode", "device"):
                        split_source[stat_field] = _require_int(
                            split_source[stat_field], f"{prefix}.{stat_field}"
                        )
                    _require_sha256(split_source["sha256"], f"{prefix}.sha256")
                    num_windows = _require_int(
                        split_source["num_windows"], f"{prefix}.num_windows", minimum=1
                    )
                    if num_tokens % num_windows:
                        raise OLMoConfigurationError(
                            f"Native replay v3 receipt source {source_id!r} token count must be "
                            "divisible by its window count"
                        )
                    _require_sha256(
                        split_source["parent_window_starts_sha256"],
                        f"{prefix}.parent_window_starts_sha256",
                    )
                    normalized_split_sources.append(split_source)
                split_sources[split] = tuple(normalized_split_sources)
            canonical_splits = {split: list(split_sources[split]) for split in ("train", "holdout")}
            if (
                _canonical_sha256(canonical_splits, "compact materialization")
                != digests["compact_materialization_sha256"]
            ):
                raise OLMoConfigurationError(
                    "Native replay v3 compact_materialization_sha256 does not match splits"
                )

        return cls(
            path=receipt_path,
            version=version,
            receipt_sha256=receipt_sha256,
            builder_implementation=builder_implementation,
            builder_sha256=digests["builder_sha256"],
            parent_paths_sha256=digests["parent_paths_sha256"],
            parent_mix_sha256=digests["parent_mix_sha256"],
            parent_config_sha256=digests["parent_config_sha256"],
            parent_trainer_state_sha256=digests["parent_trainer_state_sha256"],
            parent_dataset_fingerprint=digests["parent_dataset_fingerprint"],
            remote_snapshot_sha256=digests["remote_snapshot_sha256"],
            compact_materialization_sha256=digests["compact_materialization_sha256"],
            manifest_contract_sha256=manifest_contract_sha256,
            mirror_policy=mirror_policy,
            remote_sources=remote_sources,
            split_sources=split_sources,
        )

    def validate_manifest(self, manifest: NativeTextReplayManifest) -> None:
        """Require every selected manifest source to match this verified receipt."""
        if manifest.version != self.version:
            raise OLMoConfigurationError(
                "Native replay manifest and verification receipt versions do not match"
            )
        provenance = manifest.provenance
        expected = {
            "builder_implementation": self.builder_implementation,
            "builder_sha256": self.builder_sha256,
            "parent_paths_sha256": self.parent_paths_sha256,
            "parent_mix_sha256": self.parent_mix_sha256,
            "parent_config_sha256": self.parent_config_sha256,
            "parent_trainer_state_sha256": self.parent_trainer_state_sha256,
            "parent_dataset_fingerprint": self.parent_dataset_fingerprint,
            "remote_snapshot_sha256": self.remote_snapshot_sha256,
            "compact_materialization_sha256": self.compact_materialization_sha256,
            "verification_receipt_sha256": self.receipt_sha256,
        }
        for field_name, expected_value in expected.items():
            if provenance.get(field_name) != expected_value:
                raise OLMoConfigurationError(
                    f"Native replay manifest {field_name} does not match its verification receipt"
                )
        split = str(provenance["split"])
        if self.manifest_contract_sha256.get(split) != manifest.manifest_contract_sha256:
            raise OLMoConfigurationError(
                f"Native replay v3 manifest contract for split {split!r} does not match its "
                "verification receipt"
            )
        actual_split_sources = tuple(
            {
                "id": source.source_id,
                "source": source.source_name,
                "parent_path_index": source.parent_path_index,
                "parent_path": source.parent_path,
                "path": source.token_path,
                "resolved_path": str(source.resolved_path),
                "parent_num_tokens": source.parent_num_tokens,
                "num_tokens": source.num_tokens,
                "size_bytes": source.size_bytes,
                "sha256": source.sha256,
                "num_windows": len(source.window_starts),
                "parent_window_starts_sha256": _canonical_sha256(
                    list(source.parent_window_starts), "parent window starts"
                ),
            }
            for source in manifest.sources
        )
        receipt_split_sources = self.split_sources.get(split)
        if receipt_split_sources is None or len(receipt_split_sources) != len(actual_split_sources):
            receipt_manifest_sources = None
        else:
            receipt_manifest_sources = tuple(
                {field_name: source[field_name] for field_name in actual_split_sources[index]}
                for index, source in enumerate(receipt_split_sources)
            )
        if receipt_manifest_sources != actual_split_sources:
            raise OLMoConfigurationError(
                f"Native replay v3 manifest split {split!r} differs from the pinned offline "
                "verification receipt"
            )

    def validate_pair(
        self,
        train_manifest: NativeTextReplayManifest,
        holdout_manifest: NativeTextReplayManifest,
    ) -> None:
        """Validate an exact v3 train/holdout pair and reject parent-window overlap."""
        if train_manifest.provenance.get("split") != "train":
            raise OLMoConfigurationError("Native replay train manifest has the wrong split")
        if holdout_manifest.provenance.get("split") != "holdout":
            raise OLMoConfigurationError("Native replay holdout manifest has the wrong split")
        if train_manifest.sequence_length != holdout_manifest.sequence_length:
            raise OLMoConfigurationError(
                "Native replay v3 train and holdout sequence lengths must match"
            )
        self.validate_manifest(train_manifest)
        self.validate_manifest(holdout_manifest)
        train_intervals = {
            (source.parent_path_index, start, start + train_manifest.sequence_length)
            for source in train_manifest.sources
            for start in source.parent_window_starts
        }
        holdout_intervals = {
            (source.parent_path_index, start, start + holdout_manifest.sequence_length)
            for source in holdout_manifest.sources
            for start in source.parent_window_starts
        }
        by_parent: dict[int, list[tuple[int, int, str]]] = {}
        for parent_index, start, stop in train_intervals:
            by_parent.setdefault(parent_index, []).append((start, stop, "train"))
        for parent_index, start, stop in holdout_intervals:
            by_parent.setdefault(parent_index, []).append((start, stop, "holdout"))
        for parent_index, intervals in by_parent.items():
            ordered = sorted(intervals)
            for left, right in pairwise(ordered):
                if left[2] != right[2] and right[0] < left[1]:
                    raise OLMoConfigurationError(
                        "Native replay v3 train and holdout manifests overlap in parent path "
                        f"index {parent_index}"
                    )


@dataclass
class NativeTextReplayDatasetConfig(Config):
    """Configuration for :class:`NativeTextReplayDataset`.

    :param manifest_path: Path to a finite replay manifest.
    :param expected_fingerprint: Mandatory semantic content fingerprint to require.
    :param expected_parent_checkpoint: Optional parent checkpoint identity to require.
    :param expected_parent_mix: Optional parent-mix name to require, such as
        ``"OLMo-mix-0925"``.
    :param expected_parent_paths_sha256: Optional digest of the exact expanded parent path
        manifest to require.
    :param verification_receipt_path: Pinned receipt from the builder's completed
        offline full-file integrity pass.
    :param expected_verification_receipt_sha256: SHA-256 of the exact receipt bytes.
    :param validate_source_files: Check each materialized file's size at construction.
    :param verify_source_hashes: Also stream and verify every source SHA-256 at construction.
        This is intentionally opt-in because a production replay corpus can be large.
    :param validate_token_ids: Check each loaded window against the manifest vocabulary.
    :param max_open_files: Per-process LRU bound for source memmaps.
    """

    manifest_path: str
    expected_fingerprint: str | None = None
    expected_parent_checkpoint: str | None = None
    expected_parent_mix: str | None = None
    expected_parent_paths_sha256: str | None = None
    verification_receipt_path: str | None = None
    expected_verification_receipt_sha256: str | None = None
    validate_source_files: bool = True
    verify_source_hashes: bool = False
    validate_token_ids: bool = True
    max_open_files: int = 8

    def build(self, tokenizer=None) -> NativeTextReplayDataset:
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
        expected_fingerprint: str | None = None,
        expected_parent_checkpoint: str | None = None,
        expected_parent_mix: str | None = None,
        expected_parent_paths_sha256: str | None = None,
        verification_receipt_path: os.PathLike[str] | str | None = None,
        expected_verification_receipt_sha256: str | None = None,
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
        if expected_fingerprint is None:
            raise OLMoConfigurationError(
                "Native text replay v3 requires an explicit expected_fingerprint"
            )
        if verification_receipt_path is None:
            raise OLMoConfigurationError(
                "Native text replay v3 requires a pinned verification receipt path and SHA-256"
            )
        if not validate_source_files:
            raise OLMoConfigurationError(
                "Native text replay v3 requires validate_source_files=True for live file "
                "identity evidence"
            )
        if self.manifest.content_fingerprint != expected_fingerprint:
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

        self.verification_receipt = NativeTextReplayVerificationReceipt.load(
            verification_receipt_path,
            expected_sha256=expected_verification_receipt_sha256,
        )
        self.verification_receipt.validate_manifest(self.manifest)
        split = str(self.manifest.provenance["split"])
        self._receipt_source_stats: Mapping[str, tuple[int, int, int, int, int]] = {
            str(source["id"]): (
                int(source["size_bytes"]),
                int(source["mtime_ns"]),
                int(source["ctime_ns"]),
                int(source["inode"]),
                int(source["device"]),
            )
            for source in self.verification_receipt.split_sources[split]
        }

        self.validate_token_ids = validate_token_ids
        self.max_open_files = max_open_files
        self._cumulative_windows = tuple(
            np.cumsum([len(source.window_starts) for source in self.manifest.sources]).tolist()
        )
        self._mmap_cache: OrderedDict[int, np.memmap] = OrderedDict()
        self._cache_lock = threading.RLock()

        self._validate_source_files(verify_hashes=verify_source_hashes)

    @property
    def fingerprint(self) -> str:
        """Semantic content fingerprint suitable for loader-state validation."""
        return self.manifest.content_fingerprint

    @property
    def fingerprint_version(self) -> str:
        """Version of :attr:`fingerprint`."""
        return f"native-text-replay-v{self.manifest.version}"

    @property
    def sequence_length(self) -> int:
        """Number of input and label tokens in every example."""
        return self.manifest.sequence_length

    @property
    def source_counts(self) -> Mapping[str, int]:
        """Number of selected windows by manifest source label."""
        counts: dict[str, int] = {}
        for source in self.manifest.sources:
            counts[source.source_name] = counts.get(source.source_name, 0) + len(
                source.window_starts
            )
        return counts

    def _validate_source_files(self, *, verify_hashes: bool) -> None:
        for source in self.manifest.sources:
            try:
                descriptor, source_stat = self._open_source(source)
            except OSError as error:
                raise OLMoConfigurationError(
                    f"Could not open native text replay source {source.resolved_path}: {error}"
                ) from error
            try:
                expected_stat = self._receipt_source_stats[source.source_id]
                actual_stat = self._source_stat_signature(source_stat)
                if actual_stat != expected_stat:
                    raise OLMoConfigurationError(
                        f"Native text replay source {source.resolved_path} has stat "
                        f"signature {actual_stat}, expected {expected_stat}"
                    )
                if verify_hashes:
                    digest = hashlib.sha256()
                    with os.fdopen(descriptor, "rb") as file_handle:
                        descriptor = -1
                        while chunk := file_handle.read(8 * 1024 * 1024):
                            digest.update(chunk)
                    actual_hash = digest.hexdigest()
                    if actual_hash != source.sha256:
                        raise OLMoConfigurationError(
                            f"Native text replay source {source.resolved_path} has SHA-256 "
                            f"{actual_hash}, expected {source.sha256}"
                        )
            finally:
                if descriptor >= 0:
                    os.close(descriptor)

    @staticmethod
    def _source_stat_signature(source_stat: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            source_stat.st_size,
            source_stat.st_mtime_ns,
            source_stat.st_ctime_ns,
            source_stat.st_ino,
            source_stat.st_dev,
        )

    def _open_source(self, source: NativeTextReplaySource) -> tuple[int, os.stat_result]:
        descriptor = -1
        if not all(
            hasattr(os, flag_name) for flag_name in ("O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK")
        ):
            raise OLMoConfigurationError(
                "Native text replay v3 requires O_NOFOLLOW, O_DIRECTORY, and O_NONBLOCK "
                "for compact source validation"
            )
        compact_root = self.manifest.path.parent.resolve()
        try:
            relative_path = source.resolved_path.relative_to(compact_root)
        except ValueError as error:
            raise OLMoConfigurationError(
                f"Native text replay compact source escapes its manifest: {source.resolved_path}"
            ) from error
        directory_descriptor = os.open(
            compact_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        try:
            for part in relative_path.parts[:-1]:
                next_descriptor = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory_descriptor,
                )
                os.close(directory_descriptor)
                directory_descriptor = next_descriptor
            descriptor = os.open(
                relative_path.name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=directory_descriptor,
            )
        finally:
            os.close(directory_descriptor)
        try:
            source_stat = os.fstat(descriptor)
            if not stat.S_ISREG(source_stat.st_mode):
                raise OLMoConfigurationError(
                    f"Native text replay source is not a regular file: {source.resolved_path}"
                )
            return descriptor, source_stat
        except BaseException:
            os.close(descriptor)
            raise

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

    def _locate(self, index: int) -> tuple[int, int, int]:
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
        provenance = {
            "dataset_fingerprint": self.fingerprint,
            "manifest_index": normalized,
            "source_id": source.source_id,
            "source": source.source_name,
            "path": source.token_path,
            "source_sha256": source.sha256,
            "start": start,
            "stop": start + self.sequence_length,
        }
        parent_start = source.parent_window_starts[source_window_index]
        provenance.update(
            {
                "parent_path_index": source.parent_path_index,
                "parent_path": source.parent_path,
                "parent_start": parent_start,
                "parent_stop": parent_start + self.sequence_length,
                "parent_num_tokens": source.parent_num_tokens,
                "compact_path": source.token_path,
                "compact_start": start,
                "compact_stop": start + self.sequence_length,
            }
        )
        return provenance

    def _read_tokens(self, source_index: int, start: int) -> np.ndarray:
        source = self.manifest.sources[source_index]
        stop = start + self.sequence_length
        # Copy under the cache lock so another prefetch thread cannot evict and close the
        # memmap while this thread is materializing its view.
        with self._cache_lock:
            mmap = self._mmap_cache.pop(source_index, None)
            if mmap is None:
                try:
                    descriptor, source_stat = self._open_source(source)
                except OSError as error:
                    raise RuntimeError(
                        f"Could not open native text replay source {source.resolved_path}: {error}"
                    ) from error
                expected_stat = self._receipt_source_stats[source.source_id]
                actual_stat = self._source_stat_signature(source_stat)
                if actual_stat != expected_stat:
                    os.close(descriptor)
                    raise RuntimeError(
                        f"Native text replay source {source.resolved_path} changed stat signature "
                        f"before read: expected {expected_stat}, got {actual_stat}"
                    )
                with os.fdopen(descriptor, "rb") as file_handle:
                    mmap = np.memmap(file_handle, mode="r", dtype=self.manifest.dtype)
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

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.get(index, 0)

    def get(self, index: int, epoch: int = 0) -> dict[str, Any]:
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

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_mmap_cache"] = OrderedDict()
        state["_cache_lock"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._mmap_cache = OrderedDict()
        self._cache_lock = threading.RLock()

    def __del__(self):
        # Interpreter shutdown can leave partially initialized attributes or modules.
        try:
            self.close()
        except (AttributeError, OSError):
            pass
