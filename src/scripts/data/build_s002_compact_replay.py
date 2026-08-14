#!/usr/bin/env python
"""Materialize a compact, immutable v3 replay of the pinned s002 text dataset.

The builder reads only the 950 exact object names recorded by the s002 checkpoint. It maps
each ``s3://ai2-llm/...`` parent URI to the same bucket and key in GCS, snapshots object
metadata without listing, selects a deterministic affine-grid panel, and downloads only the
selected 8,192-token byte ranges. Generation preconditions and closing metadata checks make
remote mutation fail closed. The resulting local files are headerless ``uint32`` arrays.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import ctypes
import errno
import fcntl
import hashlib
import io
import json
import math
import os
import re
import stat
import subprocess
import sys
import threading
import time
import urllib.parse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
)

import numpy as np

NATIVE_TEXT_REPLAY_FORMAT = "olmo_native_text_replay"
NATIVE_TEXT_REPLAY_VERSION = 3
VERIFICATION_RECEIPT_FORMAT = "olmo_native_text_replay_verification_receipt"
VERIFICATION_RECEIPT_VERSION = 3
VERIFICATION_RECEIPT_FILENAME = "native-text-replay-verification.json"
TRAIN_MANIFEST_FILENAME = "native-text-replay-train.json"
HOLDOUT_MANIFEST_FILENAME = "native-text-replay-holdout.json"
BUILDER_IMPLEMENTATION_REFERENCE = "src/scripts/data/build_s002_compact_replay.py"
BUILDER_IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

S002_PARENT_CHECKPOINT = "/weka/oe-training-default/robertb/s002-step125500"
S002_PARENT_CONFIG_FILE = f"{S002_PARENT_CHECKPOINT}/config.json"
S002_PARENT_PATHS_FILE = f"{S002_PARENT_CHECKPOINT}/data_paths.txt"
S002_PARENT_TRAINER_STATE_FILE = f"{S002_PARENT_CHECKPOINT}/train/rank0.pt"
S002_PARENT_MIX = "OLMo-mix-0925"
S002_PARENT_MIX_FILE = str(
    Path(__file__).resolve().parents[2] / "olmo_core" / "data" / "mixes" / "OLMo-mix-0925.txt"
)
S002_PARENT_CONFIG_SHA256 = "35ce23db053dd2204bc37783546f1b2f98eafb742488903773dd0ef3e5741146"
S002_PARENT_PATHS_SHA256 = "f1155957f4f249fc17e1c7067512e7d881ce6675c6b854d5ce089c649cec1c2d"
S002_PARENT_MIX_SHA256 = "fcc6a82b9a5e868885decfbc30486967644c7ca482a7d687102f7ff597dbd7c9"
S002_PARENT_TRAINER_STATE_SHA256 = (
    "451a536f6483b5347837251ab931c38c70434854c001d74456737592750170d3"
)
S002_PARENT_DATASET_FINGERPRINT = "37e1ae62dccee1f0cb5c3e416572e6e48218a6c644580fa5034f575880e08c11"
S002_PARENT_DATASET_FINGERPRINT_VERSION = "v2.0"
S002_EXPECTED_OBJECTS = 950
S002_TOKENIZER = {
    "identifier": "allenai/dolma2-tokenizer",
    "vocab_size": 100_278,
    "eos_token_id": 100_257,
    "pad_token_id": 100_277,
}
S002_INSTANCE_FILTER = {
    "repetition_min_period": 1,
    "repetition_max_period": 13,
    "repetition_max_count": 32,
}

TOKEN_DTYPE = "uint32"
TOKEN_ITEM_SIZE = 4
SEQUENCE_LENGTH = 8192
WINDOW_SIZE_BYTES = SEQUENCE_LENGTH * TOKEN_ITEM_SIZE
TRAIN_WINDOWS = 100_000
HOLDOUT_WINDOWS = 1_000
SELECTION_SEED = 6198
SELECTION_ALGORITHM = "affine-grid-v1"
MIRROR_POLICY = "s3-to-gs-same-bucket-key-v1"
GCS_PROJECT = "ai2-llm"
DEFAULT_WORKERS = 16
MAX_WORKERS = 64
PLAN_FILENAME = "compact-replay-plan.json"
PLAN_FORMAT = "olmo_native_text_replay_compact_plan"
PLAN_VERSION = 1

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_T = TypeVar("_T")
_U = TypeVar("_U")
ProgressCallback = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class ParentSource:
    """One exact source row reconstructed from the pinned checkpoint manifests."""

    source_id: str
    source_name: str
    parent_path_index: int
    parent_path: str


@dataclass(frozen=True)
class RemoteObject:
    """Generation-pinned metadata for one exact mirrored parent object."""

    parent_path_index: int
    parent_path: str
    mirror_uri: str
    size_bytes: int
    num_tokens: int
    generation: str
    etag: str
    md5_hash: Optional[str]
    crc32c: str
    source_etag: Optional[str]

    def as_dict(self) -> Dict[str, Any]:
        """Return the strict receipt representation."""
        return {
            "parent_path_index": self.parent_path_index,
            "parent_path": self.parent_path,
            "mirror_uri": self.mirror_uri,
            "size_bytes": self.size_bytes,
            "num_tokens": self.num_tokens,
            "generation": self.generation,
            "etag": self.etag,
            "md5_hash": self.md5_hash,
            "crc32c": self.crc32c,
            "source_etag": self.source_etag,
        }


@dataclass(frozen=True)
class RangeResult:
    """Exact bytes and response identity returned by one remote range request."""

    data: bytes
    content_range: str
    content_length: int
    generation: str
    etag: str


class ObjectStore(Protocol):
    """Minimal exact-object interface used by the compact replay builder."""

    def head(self, parent_path: str, parent_path_index: int) -> RemoteObject:
        """Return metadata for one exact URI without listing a bucket."""

    def get_range(self, snapshot: RemoteObject, start: int, stop: int) -> RangeResult:
        """Read the half-open byte range ``[start, stop)`` under a version precondition."""


@dataclass(frozen=True)
class CompactSelection:
    """One split/source selection ordered by parent-token start."""

    split: str
    source: ParentSource
    remote: RemoteObject
    parent_window_starts: Tuple[int, ...]


@dataclass(frozen=True)
class ParentIdentity:
    """Validated pinned parent bytes, source rows, and saved loader identity."""

    sources: Tuple[ParentSource, ...]
    config_sha256: str
    paths_sha256: str
    mix_sha256: str
    trainer_state_sha256: str
    dataset_fingerprint: str


@dataclass(frozen=True)
class BuildResult:
    """Published compact artifact paths and immutable identities."""

    output_dir: Path
    train_manifest: Path
    holdout_manifest: Path
    verification_receipt: Path
    train_manifest_sha256: str
    holdout_manifest_sha256: str
    verification_receipt_sha256: str
    remote_snapshot_sha256: str
    compact_materialization_sha256: str


@dataclass(frozen=True)
class _ClosingFileExpectation:
    size_bytes: int
    sha256: str
    raw: Optional[bytes] = None
    stat_identity: Optional[Tuple[int, int, int, int]] = None


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


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def _strict_json_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON repeats object key {key!r}")
        result[key] = value
    return result


def _validate_workers(workers: int) -> int:
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be an integer in [1, {MAX_WORKERS}]")
    return workers


def _emit_progress(
    callback: Optional[ProgressCallback],
    *,
    event: str,
    started_at: float,
    **fields: Any,
) -> None:
    if callback is None:
        return
    payload: Dict[str, Any] = {
        "phase": "s002_compact_replay",
        "event": event,
        "elapsed_seconds": round(time.monotonic() - started_at, 3),
    }
    payload.update(fields)
    try:
        callback(payload)
    except Exception:
        pass


def _stderr_progress(payload: Mapping[str, Any]) -> None:
    try:
        print(json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)
    except OSError:
        pass


def _ordered_bounded_map(
    items: Iterable[_T],
    function: Callable[[_T], _U],
    *,
    workers: int,
    thread_name_prefix: str,
) -> Tuple[_U, ...]:
    workers = _validate_workers(workers)
    if workers == 1:
        return tuple(map(function, items))
    iterator = iter(items)
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=thread_name_prefix)
    futures: deque[Future[_U]] = deque()
    output = []
    try:
        for _ in range(workers):
            try:
                futures.append(executor.submit(function, next(iterator)))
            except StopIteration:
                break
        while futures:
            output.append(futures.popleft().result())
            try:
                futures.append(executor.submit(function, next(iterator)))
            except StopIteration:
                pass
    finally:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
    return tuple(output)


def _read_pinned_file(path_value: str, expected_sha256: str, name: str) -> bytes:
    if _SHA256_RE.fullmatch(expected_sha256) is None:
        raise ValueError(f"Pinned {name} SHA-256 is invalid")
    path = Path(path_value).expanduser().resolve()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"Could not open pinned {name} {path}: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Pinned {name} is not a regular file: {path}")
        with os.fdopen(descriptor, "rb") as file_handle:
            descriptor = -1
            raw = file_handle.read()
            after = os.fstat(file_handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    def signature(value: os.stat_result) -> Tuple[int, int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    if signature(before) != signature(after):
        raise ValueError(f"Pinned {name} changed while it was read")
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"Pinned {name} has SHA-256 {actual}, expected {expected_sha256}")
    return raw


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _safe_load_trainer_state(raw: bytes) -> Mapping[str, Any]:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - torch is an OLMo-core dependency.
        raise ValueError("PyTorch is required to inspect the pinned trainer state") from error
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
        raise ValueError(f"Could not safely load pinned trainer state: {error}") from error
    return _require_mapping(value, "Pinned trainer state")


def _load_parent_identity(*, expected_objects: int = S002_EXPECTED_OBJECTS) -> ParentIdentity:
    config_raw = _read_pinned_file(
        S002_PARENT_CONFIG_FILE, S002_PARENT_CONFIG_SHA256, "s002 config"
    )
    paths_raw = _read_pinned_file(
        S002_PARENT_PATHS_FILE, S002_PARENT_PATHS_SHA256, "s002 expanded path manifest"
    )
    mix_raw = _read_pinned_file(
        S002_PARENT_MIX_FILE, S002_PARENT_MIX_SHA256, "OLMo-mix-0925 manifest"
    )
    trainer_raw = _read_pinned_file(
        S002_PARENT_TRAINER_STATE_FILE,
        S002_PARENT_TRAINER_STATE_SHA256,
        "s002 rank0 trainer state",
    )
    try:
        config = _require_mapping(
            json.loads(config_raw, object_pairs_hook=_strict_json_object), "Pinned s002 config"
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Pinned s002 config is invalid JSON: {error}") from error
    dataset = _require_mapping(config.get("dataset"), "Pinned s002 config dataset")
    tokenizer = _require_mapping(dataset.get("tokenizer"), "Pinned s002 tokenizer")
    data_loader = _require_mapping(config.get("data_loader"), "Pinned s002 data_loader")
    instance_filter = _require_mapping(
        dataset.get("instance_filter_config"), "Pinned s002 instance filter"
    )
    expected_instance_filter: Dict[str, Any] = dict(S002_INSTANCE_FILTER)
    expected_instance_filter["_CLASS_"] = "olmo_core.data.numpy_dataset.InstanceFilterConfig"
    if (
        config.get("_CLASS_") != "olmo_core.internal.experiment.ExperimentConfig"
        or dataset.get("_CLASS_") != "olmo_core.data.numpy_dataset.NumpyFSLDatasetConfig"
        or dataset.get("mix") != S002_PARENT_MIX
        or dataset.get("mix_base_dir") != "s3://ai2-llm"
        or dataset.get("expand_glob") is not False
        or dataset.get("sequence_length") != SEQUENCE_LENGTH
        or dataset.get("max_target_sequence_length") != SEQUENCE_LENGTH
        or dataset.get("ignore_fingerprint_mismatch") is not False
        or dict(instance_filter) != expected_instance_filter
        or tokenizer.get("_CLASS_") != "olmo_core.data.tokenizer.TokenizerConfig"
        or any(tokenizer.get(key) != value for key, value in S002_TOKENIZER.items())
        or data_loader.get("_CLASS_") != "olmo_core.data.data_loader.NumpyDataLoaderConfig"
        or data_loader.get("type") != "numpy"
        or data_loader.get("ignore_fingerprint_mismatch") is not False
    ):
        raise ValueError("Pinned s002 config does not preserve the native replay contract")

    try:
        parent_paths = paths_raw.decode("utf-8").splitlines()
        mix_lines = mix_raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("Pinned parent path and mix manifests must be UTF-8") from error
    if (
        len(parent_paths) != expected_objects
        or len(mix_lines) != expected_objects
        or len(set(parent_paths)) != expected_objects
    ):
        raise ValueError(
            f"Pinned s002 parent identity must contain exactly {expected_objects} unique rows"
        )
    sources = []
    for index, (parent_path, mix_line) in enumerate(zip(parent_paths, mix_lines)):
        source_name, separator, relative_path = mix_line.partition(",")
        if (
            not separator
            or not source_name
            or not relative_path
            or "," in relative_path
            or source_name.strip() != source_name
            or relative_path.strip() != relative_path
        ):
            raise ValueError(f"Pinned parent mix row {index} is not canonical LABEL,PATH")
        expanded = "s3://ai2-llm/" + relative_path.replace(
            "{TOKENIZER}", str(S002_TOKENIZER["identifier"])
        )
        if expanded != parent_path or _mirror_uri(parent_path) != parent_path.replace(
            "s3://", "gs://", 1
        ):
            raise ValueError(f"Pinned parent path {index} does not match its mix row")
        path_digest = hashlib.sha256(parent_path.encode("utf-8")).hexdigest()
        sources.append(
            ParentSource(
                source_id=f"s002-{index:06d}-{path_digest[:16]}",
                source_name=source_name,
                parent_path_index=index,
                parent_path=parent_path,
            )
        )

    trainer = _safe_load_trainer_state(trainer_raw)
    loader_state = _require_mapping(trainer.get("data_loader"), "Pinned trainer data_loader")
    if (
        loader_state.get("dataset_fingerprint_version") != S002_PARENT_DATASET_FINGERPRINT_VERSION
        or loader_state.get("dataset_fingerprint") != S002_PARENT_DATASET_FINGERPRINT
        or loader_state.get("dataset_type") != "fsl"
        or loader_state.get("sequence_length") != SEQUENCE_LENGTH
        or loader_state.get("max_target_sequence_length") != SEQUENCE_LENGTH
    ):
        raise ValueError("Pinned trainer state does not contain the reviewed dataset identity")
    return ParentIdentity(
        sources=tuple(sources),
        config_sha256=S002_PARENT_CONFIG_SHA256,
        paths_sha256=S002_PARENT_PATHS_SHA256,
        mix_sha256=S002_PARENT_MIX_SHA256,
        trainer_state_sha256=S002_PARENT_TRAINER_STATE_SHA256,
        dataset_fingerprint=S002_PARENT_DATASET_FINGERPRINT,
    )


def _mirror_uri(parent_path: str) -> str:
    parsed = urllib.parse.urlsplit(parent_path)
    if (
        parsed.scheme != "s3"
        or parsed.netloc != "ai2-llm"
        or not parsed.path.startswith("/")
        or parsed.path == "/"
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"Parent URI is not a canonical s3://ai2-llm object: {parent_path!r}")
    return urllib.parse.urlunsplit(("gs", parsed.netloc, parsed.path, "", ""))


def _validate_base64(value: Optional[str], *, name: str, decoded_size: int) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"Remote {name} must be a non-empty base64 string or null")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError(f"Remote {name} is not canonical base64") from error
    if len(decoded) != decoded_size or base64.b64encode(decoded).decode("ascii") != value:
        raise ValueError(f"Remote {name} must encode exactly {decoded_size} bytes")
    return value


class GCSMirrorStore:
    """Exact-object GCS mirror reader with generation-predicated range requests.

    :param project: Google Cloud project used to construct the storage client.
    :param use_gcloud_access_token: Use ``gcloud auth print-access-token`` for local runs.
        The token is captured in memory and never logged.
    :param client: Optional compatible storage client, primarily for controlled tests.
    :param request_timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        *,
        project: str = GCS_PROJECT,
        use_gcloud_access_token: bool = False,
        client: Optional[Any] = None,
        request_timeout: float = 120.0,
    ):
        if client is None:
            try:
                from google.cloud import storage
            except ImportError as error:
                raise ValueError("google-cloud-storage is required for compact replay") from error
            credentials = None
            if use_gcloud_access_token:
                try:
                    from google.oauth2.credentials import Credentials

                    completed = subprocess.run(
                        ["gcloud", "auth", "print-access-token"],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                except (OSError, subprocess.CalledProcessError) as error:
                    raise ValueError("Could not obtain a local gcloud access token") from error
                token = completed.stdout.strip()
                if not token or any(character.isspace() for character in token):
                    raise ValueError("gcloud returned an invalid access token")
                credentials = Credentials(token)
            client = storage.Client(project=project, credentials=credentials)
        self.client = client
        self.request_timeout = request_timeout

    @staticmethod
    def _bucket_key(parent_path: str) -> Tuple[str, str]:
        mirror = urllib.parse.urlsplit(_mirror_uri(parent_path))
        return mirror.netloc, mirror.path.removeprefix("/")

    def head(self, parent_path: str, parent_path_index: int) -> RemoteObject:
        """Read one exact GCS object's metadata without any list operation."""
        bucket_name, key = self._bucket_key(parent_path)
        blob = self.client.bucket(bucket_name).blob(key)
        try:
            blob.reload(timeout=self.request_timeout)
        except TypeError:
            blob.reload()
        metadata = blob.metadata or {}
        source_etag = metadata.get("x-goog-source-etag")
        return RemoteObject(
            parent_path_index=parent_path_index,
            parent_path=parent_path,
            mirror_uri=_mirror_uri(parent_path),
            size_bytes=int(blob.size) if blob.size is not None else -1,
            num_tokens=(int(blob.size) // TOKEN_ITEM_SIZE if blob.size is not None else -1),
            generation=str(blob.generation) if blob.generation is not None else "",
            etag=str(blob.etag) if blob.etag is not None else "",
            md5_hash=str(blob.md5_hash) if blob.md5_hash is not None else None,
            crc32c=str(blob.crc32c) if blob.crc32c is not None else "",
            source_etag=str(source_etag) if source_etag is not None else None,
        )

    def get_range(self, snapshot: RemoteObject, start: int, stop: int) -> RangeResult:
        """Download one half-open range with exact GCS generation and ETag preconditions."""
        if start < 0 or stop <= start or stop > snapshot.size_bytes:
            raise ValueError(f"Invalid range [{start}, {stop}) for {snapshot.mirror_uri}")
        bucket_name, key = self._bucket_key(snapshot.parent_path)
        base = (
            "https://storage.googleapis.com/download/storage/v1/b/"
            f"{urllib.parse.quote(bucket_name, safe='')}/o/"
            f"{urllib.parse.quote(key, safe='')}"
        )
        query = urllib.parse.urlencode({"alt": "media", "ifGenerationMatch": snapshot.generation})
        response = self.client._http.request(
            method="GET",
            url=f"{base}?{query}",
            headers={"Range": f"bytes={start}-{stop - 1}", "If-Match": snapshot.etag},
            timeout=self.request_timeout,
        )
        if response.status_code != 206:
            raise ValueError(
                f"Generation-pinned range GET for {snapshot.mirror_uri} returned "
                f"HTTP {response.status_code}, expected 206"
            )
        headers = {str(key).lower(): value for key, value in response.headers.items()}
        try:
            content_length = int(headers.get("content-length", ""))
        except ValueError as error:
            raise ValueError("Range GET returned an invalid Content-Length") from error
        return RangeResult(
            data=bytes(response.content),
            content_range=headers.get("content-range", ""),
            content_length=content_length,
            generation=headers.get("x-goog-generation", ""),
            etag=headers.get("etag", ""),
        )


def snapshot_remote_objects(
    identity: ParentIdentity,
    store: ObjectStore,
    *,
    workers: int,
    progress: Optional[ProgressCallback] = None,
) -> Tuple[RemoteObject, ...]:
    """HEAD every exact parent URI and return records in checkpoint order.

    :param identity: Validated pinned parent source identity.
    :param store: Exact-object remote store; no listing method is used or required.
    :param workers: Bounded HEAD concurrency.
    :param progress: Optional operational progress callback.
    :returns: Strict remote records ordered by parent path index.
    """
    started_at = time.monotonic()
    completed = 0
    completion_lock = threading.Lock()

    def head(source: ParentSource) -> RemoteObject:
        nonlocal completed
        result = _validate_remote_object(
            store.head(source.parent_path, source.parent_path_index), source
        )
        with completion_lock:
            completed += 1
            current = completed
        if current % 50 == 0 or current == len(identity.sources):
            _emit_progress(
                progress,
                event="remote_snapshot_progress",
                started_at=started_at,
                objects_completed=current,
                objects_total=len(identity.sources),
            )
        return result

    records = _ordered_bounded_map(
        identity.sources,
        head,
        workers=workers,
        thread_name_prefix="s002-compact-head",
    )
    if tuple(record.parent_path_index for record in records) != tuple(range(len(records))):
        raise ValueError("Remote snapshot is not in exact parent path order")
    return records


def reconstruct_parent_dataset_fingerprint(
    parent_paths: Sequence[str], object_sizes: Sequence[int]
) -> str:
    """Replay the exact NumpyFSLDataset v2.0 fingerprint over paths and remote sizes.

    :param parent_paths: Exact expanded paths in checkpoint order.
    :param object_sizes: Exact byte sizes in the same order.
    :returns: The reconstructed hexadecimal dataset fingerprint.
    """
    if len(parent_paths) != len(object_sizes) or not parent_paths:
        raise ValueError("Dataset fingerprint paths and sizes must be equally sized and nonempty")
    digest = hashlib.sha256()
    digest.update(b"class=NumpyFSLDataset")
    fields = (
        ("vocab_size", S002_TOKENIZER["vocab_size"]),
        ("pad_token_id", S002_TOKENIZER["pad_token_id"]),
        ("eos_token_id", S002_TOKENIZER["eos_token_id"]),
        ("dtype", np.uint32),
        ("max_target_sequence_length", SEQUENCE_LENGTH),
        ("bos_token_id", None),
    )
    for field_name, field_value in fields:
        digest.update(f"{field_name}={field_value},".encode())
    for path, size in zip(parent_paths, object_sizes):
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("Dataset fingerprint object sizes must be positive integers")
        digest.update(f"path={os.path.basename(path)},size={size},".encode())
    return digest.hexdigest()


def _apportion_exact(
    total: int,
    capacities: Mapping[str, int],
    minima: Optional[Mapping[str, int]] = None,
) -> Dict[str, int]:
    keys = sorted(capacities)
    if not keys:
        raise ValueError("Cannot apportion over an empty source set")
    allocations = {key: int((minima or {}).get(key, 0)) for key in keys}
    for key in keys:
        if capacities[key] < 0 or allocations[key] < 0 or allocations[key] > capacities[key]:
            raise ValueError(f"Invalid capacity or minimum for {key!r}")
    minimum_total = sum(allocations.values())
    if minimum_total > total or total > sum(capacities.values()):
        raise ValueError("Requested replay windows exceed exact source capacity")
    remaining = total - minimum_total
    residual = {key: capacities[key] - allocations[key] for key in keys}
    residual_total = sum(residual.values())
    if remaining == 0:
        return allocations
    assigned = 0
    remainders = []
    for key in keys:
        quotient, remainder = divmod(remaining * residual[key], residual_total)
        allocations[key] += quotient
        assigned += quotient
        remainders.append((remainder, key))
    for _, key in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if assigned == remaining:
            break
        if allocations[key] < capacities[key]:
            allocations[key] += 1
            assigned += 1
    if assigned != remaining:
        raise AssertionError("Exact source apportionment did not close")
    return allocations


def _permutation_prefix(capacity: int, count: int, *, seed: int, source_id: str) -> Tuple[int, ...]:
    if count < 0 or count > capacity:
        raise ValueError(f"Cannot select {count} slots from capacity {capacity}")
    if count == 0:
        return ()
    if capacity == 1:
        return (0,)
    key = json.dumps(
        [SELECTION_ALGORITHM, seed, source_id],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    hashed = hashlib.sha256(key).digest()
    multiplier = int.from_bytes(hashed[:16], "big") % capacity
    offset = int.from_bytes(hashed[16:], "big") % capacity
    if multiplier == 0:
        multiplier = 1
    while math.gcd(multiplier, capacity) != 1:
        multiplier = (multiplier + 1) % capacity
        if multiplier == 0:
            multiplier = 1
    return tuple((multiplier * index + offset) % capacity for index in range(count))


def select_compact_windows(
    identity: ParentIdentity,
    remote_objects: Sequence[RemoteObject],
    *,
    train_windows: int,
    holdout_windows: int,
    seed: int,
) -> Mapping[str, Tuple[CompactSelection, ...]]:
    """Replicate affine-grid-v1 source apportionment and disjoint window selection.

    :param identity: Validated parent source rows.
    :param remote_objects: Exact object snapshot in parent order.
    :param train_windows: Exact train-window count.
    :param holdout_windows: Exact holdout-window count.
    :param seed: Non-negative selection seed.
    :returns: Train and holdout selections ordered by parent path index.
    """
    if (
        isinstance(train_windows, bool)
        or not isinstance(train_windows, int)
        or train_windows <= 0
        or isinstance(holdout_windows, bool)
        or not isinstance(holdout_windows, int)
        or holdout_windows <= 0
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise ValueError("Window counts must be positive and seed must be non-negative")
    if len(identity.sources) != len(remote_objects):
        raise ValueError("Remote snapshot does not cover every parent source")
    by_index = {remote.parent_path_index: remote for remote in remote_objects}
    if set(by_index) != set(range(len(identity.sources))):
        raise ValueError("Remote snapshot parent indices differ")
    capacities = {
        source.source_id: by_index[source.parent_path_index].num_tokens // SEQUENCE_LENGTH
        for source in identity.sources
    }
    if any(capacity <= 0 for capacity in capacities.values()):
        raise ValueError("Every parent object must contain at least one full FSL window")
    source_names = sorted({source.source_name for source in identity.sources})
    label_capacities = {
        source_name: sum(
            capacities[source.source_id]
            for source in identity.sources
            if source.source_name == source_name
        )
        for source_name in source_names
    }
    union_by_label = _apportion_exact(train_windows + holdout_windows, label_capacities)
    holdout_by_label = _apportion_exact(holdout_windows, union_by_label)
    train: list[CompactSelection] = []
    holdout: list[CompactSelection] = []
    for source_name in source_names:
        label_sources = sorted(
            (source for source in identity.sources if source.source_name == source_name),
            key=lambda source: source.source_id,
        )
        file_capacities = {
            source.source_id: capacities[source.source_id] for source in label_sources
        }
        union_by_file = _apportion_exact(union_by_label[source_name], file_capacities)
        holdout_by_file = _apportion_exact(holdout_by_label[source_name], union_by_file)
        for source in label_sources:
            selected_slots = _permutation_prefix(
                capacities[source.source_id],
                union_by_file[source.source_id],
                seed=seed,
                source_id=source.source_id,
            )
            holdout_count = holdout_by_file[source.source_id]
            remote = by_index[source.parent_path_index]
            holdout_starts = tuple(
                sorted(slot * SEQUENCE_LENGTH for slot in selected_slots[:holdout_count])
            )
            train_starts = tuple(
                sorted(slot * SEQUENCE_LENGTH for slot in selected_slots[holdout_count:])
            )
            if train_starts:
                train.append(CompactSelection("train", source, remote, train_starts))
            if holdout_starts:
                holdout.append(CompactSelection("holdout", source, remote, holdout_starts))
    train.sort(key=lambda item: item.source.parent_path_index)
    holdout.sort(key=lambda item: item.source.parent_path_index)
    if sum(len(item.parent_window_starts) for item in train) != train_windows:
        raise AssertionError("Train affine selection did not meet its exact window count")
    if sum(len(item.parent_window_starts) for item in holdout) != holdout_windows:
        raise AssertionError("Holdout affine selection did not meet its exact window count")
    train_intervals = {
        (item.source.parent_path_index, start)
        for item in train
        for start in item.parent_window_starts
    }
    holdout_intervals = {
        (item.source.parent_path_index, start)
        for item in holdout
        for start in item.parent_window_starts
    }
    if train_intervals & holdout_intervals:
        raise AssertionError("Train and holdout affine selections overlap")
    return {"train": tuple(train), "holdout": tuple(holdout)}


def _entry_stat(path: Path) -> Optional[os.stat_result]:
    try:
        return os.lstat(path)
    except FileNotFoundError:
        return None


def _require_regular_entry(path: Path, *, name: str) -> Optional[os.stat_result]:
    value = _entry_stat(path)
    if value is not None and not stat.S_ISREG(value.st_mode):
        raise ValueError(f"{name} must be a regular non-symlink file: {path}")
    return value


def _unlink_recoverable_regular(path: Path, *, name: str) -> bool:
    if _require_regular_entry(path, name=name) is None:
        return False
    try:
        path.unlink()
    except OSError as error:
        raise ValueError(f"Could not remove recoverable {name} {path}: {error}") from error
    _fsync_directory(path.parent)
    return True


def _read_regular_no_follow(path: Path, *, name: str) -> Tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"Could not open {name} {path}: {error}") from error
    try:
        source_stat = os.fstat(descriptor)
        if not stat.S_ISREG(source_stat.st_mode):
            raise ValueError(f"{name} must be a regular file: {path}")
        with os.fdopen(descriptor, "rb") as file_handle:
            descriptor = -1
            raw = file_handle.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return raw, source_stat


def _write_temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.write.tmp")


def _write_once(path: Path, raw: bytes) -> None:
    temporary = _write_temporary_path(path)
    _unlink_recoverable_regular(temporary, name="stale metadata temporary")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(temporary, flags, 0o600)
    except OSError as error:
        raise ValueError(
            f"Could not create immutable metadata temporary {temporary}: {error}"
        ) from error
    try:
        with os.fdopen(descriptor, "wb") as file_handle:
            descriptor = -1
            file_handle.write(raw)
            file_handle.flush()
            os.fsync(file_handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise FileExistsError(f"Refusing to overwrite immutable path {path}") from error
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if _entry_stat(temporary) is not None:
            _unlink_recoverable_regular(temporary, name="metadata temporary")


def _write_or_verify(path: Path, raw: bytes) -> None:
    _unlink_recoverable_regular(_write_temporary_path(path), name="stale metadata temporary")
    if _entry_stat(path) is not None:
        actual, _ = _read_regular_no_follow(path, name="resumable staging metadata")
        if actual != raw:
            raise ValueError(f"Resumable staging metadata drifted: {path}")
        return
    _write_once(path, raw)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_directory_tree_no_follow(root: Path, relative_parts: Sequence[str]) -> None:
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(root, flags)
    try:
        for part in relative_parts:
            try:
                os.mkdir(part, mode=0o755, dir_fd=descriptor)
            except FileExistsError:
                pass
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except OSError as error:
                raise ValueError(
                    f"Resumable staging directory must be a non-symlink directory: "
                    f"{root.joinpath(*relative_parts)}"
                ) from error
            os.close(descriptor)
            descriptor = child
    finally:
        os.close(descriptor)


def _prepare_staging_tree(staging: Path) -> None:
    for relative_parts in (
        ("tokens",),
        ("tokens", "train"),
        ("tokens", "holdout"),
        ("resume",),
        ("resume", "train"),
        ("resume", "holdout"),
    ):
        _ensure_directory_tree_no_follow(staging, relative_parts)


def _descriptor_signature(source_stat: os.stat_result) -> Tuple[int, int, int, int, int, int]:
    return (
        source_stat.st_dev,
        source_stat.st_ino,
        source_stat.st_mode,
        source_stat.st_size,
        source_stat.st_mtime_ns,
        source_stat.st_ctime_ns,
    )


def _validate_file_descriptor(
    directory_descriptor: int,
    name: str,
    relative_path: str,
    expectation: _ClosingFileExpectation,
) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    except OSError as error:
        raise ValueError(
            f"Closing artifact file is not safely openable: {relative_path}"
        ) from error
    digest = hashlib.sha256()
    captured = bytearray() if expectation.raw is not None else None
    try:
        before = os.fstat(descriptor)
        entry_before = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or _descriptor_signature(entry_before) != _descriptor_signature(before)
            or before.st_size != expectation.size_bytes
            or (
                expectation.stat_identity is not None
                and _stat_identity(before) != expectation.stat_identity
            )
        ):
            raise ValueError(f"Closing artifact file identity differs: {relative_path}")
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if captured is not None:
                captured.extend(chunk)
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        entry_after = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if (
            _descriptor_signature(after) != _descriptor_signature(before)
            or _descriptor_signature(entry_after) != _descriptor_signature(before)
            or digest.hexdigest() != expectation.sha256
            or (captured is not None and bytes(captured) != expectation.raw)
        ):
            raise ValueError(f"Closing artifact file content differs: {relative_path}")
    finally:
        os.close(descriptor)


def _validate_and_fsync_staging(
    staging: Path, expected_files: Mapping[str, _ClosingFileExpectation]
) -> Tuple[int, int]:
    expected_parts: Dict[Tuple[str, ...], _ClosingFileExpectation] = {}
    expected_directories: set[Tuple[str, ...]] = {()}
    for relative_path, expectation in expected_files.items():
        parts = tuple(relative_path.split("/"))
        if not parts or any(not part or part in {".", ".."} for part in parts):
            raise ValueError(f"Invalid closing artifact relative path: {relative_path!r}")
        if parts in expected_parts:
            raise ValueError(f"Duplicate closing artifact relative path: {relative_path!r}")
        expected_parts[parts] = expectation
        for length in range(1, len(parts)):
            expected_directories.add(parts[:length])

    directory_children: Dict[Tuple[str, ...], set[str]] = {
        directory: set() for directory in expected_directories
    }
    file_children: Dict[Tuple[str, ...], set[str]] = {
        directory: set() for directory in expected_directories
    }
    for directory in expected_directories:
        if directory:
            directory_children[directory[:-1]].add(directory[-1])
    for parts in expected_parts:
        file_children[parts[:-1]].add(parts[-1])

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)

    def validate_directory(directory_descriptor: int, parts: Tuple[str, ...]) -> None:
        expected_names = directory_children[parts] | file_children[parts]
        actual_names = set(os.listdir(directory_descriptor))
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            extra = sorted(actual_names - expected_names)
            raise ValueError(
                f"Closing artifact tree differs at {'/'.join(parts) or '.'}: "
                f"missing={missing}, extra={extra}"
            )
        for child_name in sorted(directory_children[parts]):
            child_parts = parts + (child_name,)
            try:
                child_descriptor = os.open(child_name, directory_flags, dir_fd=directory_descriptor)
            except OSError as error:
                raise ValueError(
                    f"Closing artifact directory is not safe: {'/'.join(child_parts)}"
                ) from error
            try:
                child_stat = os.fstat(child_descriptor)
                entry_stat = os.stat(child_name, dir_fd=directory_descriptor, follow_symlinks=False)
                if (
                    not stat.S_ISDIR(child_stat.st_mode)
                    or child_stat.st_dev != entry_stat.st_dev
                    or child_stat.st_ino != entry_stat.st_ino
                ):
                    raise ValueError(
                        f"Closing artifact directory identity differs: {'/'.join(child_parts)}"
                    )
                validate_directory(child_descriptor, child_parts)
                closing_entry = os.stat(
                    child_name, dir_fd=directory_descriptor, follow_symlinks=False
                )
                if (
                    closing_entry.st_dev != child_stat.st_dev
                    or closing_entry.st_ino != child_stat.st_ino
                    or not stat.S_ISDIR(closing_entry.st_mode)
                ):
                    raise ValueError(f"Closing artifact directory changed: {'/'.join(child_parts)}")
            finally:
                os.close(child_descriptor)
        for child_name in sorted(file_children[parts]):
            child_parts = parts + (child_name,)
            _validate_file_descriptor(
                directory_descriptor,
                child_name,
                "/".join(child_parts),
                expected_parts[child_parts],
            )
        os.fsync(directory_descriptor)

    try:
        root_descriptor = os.open(staging, directory_flags)
    except OSError as error:
        raise ValueError(f"Closing artifact root is not a safe directory: {staging}") from error
    try:
        root_stat = os.fstat(root_descriptor)
        if not stat.S_ISDIR(root_stat.st_mode):
            raise ValueError(f"Closing artifact root is not a directory: {staging}")
        validate_directory(root_descriptor, ())
        closing_root = os.stat(staging, follow_symlinks=False)
        if (
            not stat.S_ISDIR(closing_root.st_mode)
            or closing_root.st_dev != root_stat.st_dev
            or closing_root.st_ino != root_stat.st_ino
        ):
            raise ValueError("Closing artifact root changed during descriptor validation")
        return root_stat.st_dev, root_stat.st_ino
    finally:
        os.close(root_descriptor)


def _publish_no_replace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2(RENAME_NOREPLACE) is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(f"Refusing to overwrite compact replay artifact {destination}")
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _validate_range_result(
    result: RangeResult,
    snapshot: RemoteObject,
    *,
    start: int,
    stop: int,
) -> bytes:
    expected_length = stop - start
    expected_range = f"bytes {start}-{stop - 1}/{snapshot.size_bytes}"
    if (
        result.content_range != expected_range
        or result.content_length != expected_length
        or len(result.data) != expected_length
        or result.generation != snapshot.generation
        or result.etag != snapshot.etag
    ):
        raise ValueError(
            f"Range response identity differs for {snapshot.mirror_uri} [{start}, {stop})"
        )
    return result.data


def _selection_relative_path(selection: CompactSelection) -> str:
    return f"tokens/{selection.split}/{selection.source.source_id}.npy"


def _selection_resume_relative_path(selection: CompactSelection) -> str:
    return f"resume/{selection.split}/{selection.source.source_id}.json"


def _selection_partial_path(token_path: Path, plan_sha256: str) -> Path:
    if _SHA256_RE.fullmatch(plan_sha256) is None:
        raise ValueError("Compact replay plan SHA-256 is invalid")
    return token_path.with_name(f".{token_path.name}.{plan_sha256}.partial")


def _stat_identity(source_stat: os.stat_result) -> Tuple[int, int, int, int]:
    return (
        source_stat.st_dev,
        source_stat.st_ino,
        source_stat.st_mtime_ns,
        source_stat.st_ctime_ns,
    )


def _stat_regular_no_follow(path: Path, *, name: str) -> os.stat_result:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"Could not open {name} {path}: {error}") from error
    try:
        source_stat = os.fstat(descriptor)
        if not stat.S_ISREG(source_stat.st_mode):
            raise ValueError(f"{name} must be a regular file: {path}")
        return source_stat
    finally:
        os.close(descriptor)


def _inspect_compact_file(
    path: Path, *, expected_size: int, expected_windows: int
) -> Tuple[str, list[str], os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"Could not open resumable compact source {path}: {error}") from error
    digest = hashlib.sha256()
    window_hashes = []
    try:
        source_stat = os.fstat(descriptor)
        if not stat.S_ISREG(source_stat.st_mode) or source_stat.st_size != expected_size:
            raise ValueError(f"Resumable compact source identity drifted: {path}")
        with os.fdopen(descriptor, "rb") as file_handle:
            descriptor = -1
            for _ in range(expected_windows):
                data = file_handle.read(WINDOW_SIZE_BYTES)
                if len(data) != WINDOW_SIZE_BYTES:
                    raise ValueError(f"Resumable compact source is truncated: {path}")
                digest.update(data)
                window_hashes.append(hashlib.sha256(data).hexdigest())
            if file_handle.read(1):
                raise ValueError(f"Resumable compact source contains trailing bytes: {path}")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return digest.hexdigest(), window_hashes, source_stat


def _materialize_selection(
    selection: CompactSelection,
    *,
    store: ObjectStore,
    staging_dir: Path,
    output_dir: Path,
    plan_sha256: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], bytes]:
    relative_path = _selection_relative_path(selection)
    token_path = staging_dir / relative_path
    resume_path = staging_dir / _selection_resume_relative_path(selection)
    partial_path = _selection_partial_path(token_path, plan_sha256)
    expected_prefix = {
        "format": "olmo_native_text_replay_compact_source_completion",
        "version": 1,
        "plan_sha256": plan_sha256,
        "split": selection.split,
        "source_id": selection.source.source_id,
        "parent_path_index": selection.source.parent_path_index,
        "parent_window_starts": list(selection.parent_window_starts),
        "remote_generation": selection.remote.generation,
        "remote_etag": selection.remote.etag,
    }
    _unlink_recoverable_regular(partial_path, name="stale compact partial")
    _unlink_recoverable_regular(
        _write_temporary_path(resume_path), name="stale completion temporary"
    )
    token_exists = _require_regular_entry(token_path, name="resumable compact source") is not None
    resume_exists = (
        _require_regular_entry(resume_path, name="resumable source completion") is not None
    )
    if token_exists != resume_exists:
        if token_exists:
            _unlink_recoverable_regular(token_path, name="orphan compact source")
        if resume_exists:
            _unlink_recoverable_regular(resume_path, name="orphan source completion")
        token_exists = False
        resume_exists = False

    size_bytes = len(selection.parent_window_starts) * WINDOW_SIZE_BYTES
    if token_exists and resume_exists:
        completion_raw, _ = _read_regular_no_follow(resume_path, name="resumable source completion")
        try:
            completion = _require_mapping(
                json.loads(completion_raw, object_pairs_hook=_strict_json_object),
                "source completion",
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Resumable source completion is invalid JSON") from error
        expected_completion_fields = set(expected_prefix) | {
            "size_bytes",
            "sha256",
            "window_sha256",
        }
        if set(completion) != expected_completion_fields or any(
            completion.get(key) != value for key, value in expected_prefix.items()
        ):
            raise ValueError(f"Resumable plan drift for {selection.source.source_id}")
        source_sha256, window_hashes, token_stat = _inspect_compact_file(
            token_path,
            expected_size=size_bytes,
            expected_windows=len(selection.parent_window_starts),
        )
        if (
            completion.get("size_bytes") != size_bytes
            or completion.get("sha256") != source_sha256
            or completion.get("window_sha256") != window_hashes
            or completion_raw != _json_bytes(completion)
        ):
            raise ValueError(f"Resumable compact bytes drifted for {selection.source.source_id}")
    else:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(partial_path, flags, 0o600)
        digest = hashlib.sha256()
        window_hashes = []
        try:
            with os.fdopen(descriptor, "wb") as file_handle:
                descriptor = -1
                for parent_start in selection.parent_window_starts:
                    if parent_start % SEQUENCE_LENGTH:
                        raise ValueError("Parent window is not on the exact 8192-token grid")
                    byte_start = parent_start * TOKEN_ITEM_SIZE
                    byte_stop = byte_start + WINDOW_SIZE_BYTES
                    if byte_stop > selection.remote.size_bytes:
                        raise ValueError("Selected parent window is out of remote bounds")
                    data = _validate_range_result(
                        store.get_range(selection.remote, byte_start, byte_stop),
                        selection.remote,
                        start=byte_start,
                        stop=byte_stop,
                    )
                    window_hashes.append(hashlib.sha256(data).hexdigest())
                    digest.update(data)
                    file_handle.write(data)
                file_handle.flush()
                os.fsync(file_handle.fileno())
            os.link(partial_path, token_path, follow_symlinks=False)
            _fsync_directory(token_path.parent)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if _entry_stat(partial_path) is not None:
                _unlink_recoverable_regular(partial_path, name="compact partial")
        source_sha256 = digest.hexdigest()
        token_stat = _stat_regular_no_follow(token_path, name="compact source")
        if token_stat.st_size != size_bytes:
            raise ValueError(f"New compact source has the wrong size: {token_path}")
        completion = {
            **expected_prefix,
            "size_bytes": size_bytes,
            "sha256": source_sha256,
            "window_sha256": window_hashes,
        }
        completion_raw = _json_bytes(completion)
        _write_once(resume_path, completion_raw)

    num_windows = len(selection.parent_window_starts)
    num_tokens = num_windows * SEQUENCE_LENGTH
    local_starts = [index * SEQUENCE_LENGTH for index in range(num_windows)]
    manifest_source = {
        "id": selection.source.source_id,
        "source": selection.source.source_name,
        "parent_path_index": selection.source.parent_path_index,
        "parent_path": selection.source.parent_path,
        "path": relative_path,
        "parent_num_tokens": selection.remote.num_tokens,
        "num_tokens": num_tokens,
        "size_bytes": num_tokens * TOKEN_ITEM_SIZE,
        "sha256": source_sha256,
        "window_starts": local_starts,
        "parent_window_starts": list(selection.parent_window_starts),
    }
    receipt_source = {
        "id": selection.source.source_id,
        "source": selection.source.source_name,
        "parent_path_index": selection.source.parent_path_index,
        "parent_path": selection.source.parent_path,
        "path": relative_path,
        "resolved_path": str((output_dir / relative_path).resolve()),
        "parent_num_tokens": selection.remote.num_tokens,
        "num_tokens": num_tokens,
        "size_bytes": num_tokens * TOKEN_ITEM_SIZE,
        "sha256": source_sha256,
        "num_windows": num_windows,
        "parent_window_starts_sha256": _canonical_sha256(list(selection.parent_window_starts)),
        "mtime_ns": token_stat.st_mtime_ns,
        "ctime_ns": token_stat.st_ctime_ns,
        "inode": token_stat.st_ino,
        "device": token_stat.st_dev,
    }
    return manifest_source, receipt_source, completion_raw


def _build_manifest(
    *,
    split: str,
    sources: Sequence[Mapping[str, Any]],
    identity: ParentIdentity,
    remote_snapshot_sha256: str,
    compact_materialization_sha256: str,
    verification_receipt_sha256: str,
    seed: int,
) -> Dict[str, Any]:
    num_windows = sum(len(source["window_starts"]) for source in sources)
    source_tokens: Dict[str, int] = {}
    for source in sources:
        source_name = str(source["source"])
        usable = len(source["window_starts"]) * (SEQUENCE_LENGTH - 1)
        source_tokens[source_name] = source_tokens.get(source_name, 0) + usable
    return {
        "format": NATIVE_TEXT_REPLAY_FORMAT,
        "version": NATIVE_TEXT_REPLAY_VERSION,
        "sequence_length": SEQUENCE_LENGTH,
        "dtype": TOKEN_DTYPE,
        "tokenizer": dict(S002_TOKENIZER),
        "provenance": {
            "parent_checkpoint": S002_PARENT_CHECKPOINT,
            "parent_mix": S002_PARENT_MIX,
            "parent_paths_sha256": identity.paths_sha256,
            "parent_mix_sha256": identity.mix_sha256,
            "parent_config_sha256": identity.config_sha256,
            "parent_trainer_state_sha256": identity.trainer_state_sha256,
            "parent_dataset_fingerprint": identity.dataset_fingerprint,
            "remote_snapshot_sha256": remote_snapshot_sha256,
            "compact_materialization_sha256": compact_materialization_sha256,
            "builder_implementation": BUILDER_IMPLEMENTATION_REFERENCE,
            "builder_sha256": BUILDER_IMPLEMENTATION_SHA256,
            "instance_filter": dict(S002_INSTANCE_FILTER),
            "selection_algorithm": SELECTION_ALGORITHM,
            "selection_seed": seed,
            "split": split,
            "usable_tokens": num_windows * (SEQUENCE_LENGTH - 1),
            "source_usable_tokens": source_tokens,
            "minimum_source_usable_tokens": {},
            "raw_tokens_per_window": SEQUENCE_LENGTH,
            "loss_tokens_per_window": SEQUENCE_LENGTH - 1,
            "verification_receipt_sha256": verification_receipt_sha256,
        },
        "num_windows": num_windows,
        "sources": [dict(source) for source in sources],
    }


def _manifest_contract_sha256(manifest: Mapping[str, Any]) -> str:
    provenance = dict(_require_mapping(manifest.get("provenance"), "manifest provenance"))
    if "verification_receipt_sha256" not in provenance:
        raise ValueError("Manifest contract is missing verification_receipt_sha256")
    provenance.pop("verification_receipt_sha256")
    contract = dict(manifest)
    contract["provenance"] = provenance
    return _canonical_sha256(contract)


def _build_plan(
    *,
    output_dir: Path,
    identity: ParentIdentity,
    remote_objects: Sequence[RemoteObject],
    selections: Mapping[str, Sequence[CompactSelection]],
    train_windows: int,
    holdout_windows: int,
    seed: int,
) -> Dict[str, Any]:
    plan: Dict[str, Any] = {
        "format": PLAN_FORMAT,
        "version": PLAN_VERSION,
        "builder_implementation": BUILDER_IMPLEMENTATION_REFERENCE,
        "builder_sha256": BUILDER_IMPLEMENTATION_SHA256,
        "output_dir": str(output_dir),
        "parent_config_sha256": identity.config_sha256,
        "parent_paths_sha256": identity.paths_sha256,
        "parent_mix_sha256": identity.mix_sha256,
        "parent_trainer_state_sha256": identity.trainer_state_sha256,
        "parent_dataset_fingerprint": identity.dataset_fingerprint,
        "mirror_policy": MIRROR_POLICY,
        "remote_snapshot_sha256": _canonical_sha256(
            [remote.as_dict() for remote in remote_objects]
        ),
        "selection_algorithm": SELECTION_ALGORITHM,
        "selection_seed": seed,
        "sequence_length": SEQUENCE_LENGTH,
        "train_windows": train_windows,
        "holdout_windows": holdout_windows,
        "selections": {
            split: [
                {
                    "id": selection.source.source_id,
                    "parent_path_index": selection.source.parent_path_index,
                    "parent_window_starts": list(selection.parent_window_starts),
                    "generation": selection.remote.generation,
                    "etag": selection.remote.etag,
                }
                for selection in selections[split]
            ]
            for split in ("train", "holdout")
        },
    }
    plan["content_sha256"] = _canonical_sha256(plan)
    return plan


def _validate_loaded_builder() -> None:
    actual = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
    if actual != BUILDER_IMPLEMENTATION_SHA256:
        raise ValueError("Compact replay builder changed after module load")


def _open_staging(output_dir: Path, plan: Mapping[str, Any]) -> Tuple[Path, int]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite compact replay artifact {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.with_name(f".{output_dir.name}.lock")
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(lock_fd)
        raise RuntimeError(f"Another compact replay builder holds {lock_path}") from error
    plan_sha256 = str(plan["content_sha256"])
    staging = output_dir.with_name(f".{output_dir.name}.{plan_sha256[:16]}.building")
    try:
        staging.mkdir(mode=0o755)
    except FileExistsError:
        if not staging.is_dir() or staging.is_symlink():
            os.close(lock_fd)
            raise ValueError(f"Resumable staging path is invalid: {staging}")
    try:
        _write_or_verify(staging / PLAN_FILENAME, _json_bytes(plan))
    except BaseException:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
        raise
    return staging, lock_fd


def _build_compact_replay(
    output_dir: os.PathLike[str] | str,
    *,
    store: ObjectStore,
    workers: int = DEFAULT_WORKERS,
    progress: Optional[ProgressCallback] = None,
    train_windows: int = TRAIN_WINDOWS,
    holdout_windows: int = HOLDOUT_WINDOWS,
    seed: int = SELECTION_SEED,
    expected_objects: int = S002_EXPECTED_OBJECTS,
    require_production_counts: bool = True,
) -> BuildResult:
    """Build and atomically publish one compact replay artifact.

    This internal entry point exposes counts for synthetic tests. Production callers should
    use :func:`build_compact_replay`, which fixes the reviewed 100,000/1,000 window panel.
    """
    _validate_loaded_builder()
    workers = _validate_workers(workers)
    if require_production_counts and (
        train_windows != TRAIN_WINDOWS
        or holdout_windows != HOLDOUT_WINDOWS
        or seed != SELECTION_SEED
        or expected_objects != S002_EXPECTED_OBJECTS
    ):
        raise ValueError("Production compact replay requires the exact reviewed selection panel")
    started_at = time.monotonic()
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite compact replay artifact {output}")
    identity = _load_parent_identity(expected_objects=expected_objects)
    _emit_progress(
        progress,
        event="remote_snapshot_start",
        started_at=started_at,
        objects_total=len(identity.sources),
        workers=workers,
    )
    remote_objects = snapshot_remote_objects(identity, store, workers=workers, progress=progress)
    remote_records = [remote.as_dict() for remote in remote_objects]
    remote_snapshot_sha256 = _canonical_sha256(remote_records)
    reconstructed = reconstruct_parent_dataset_fingerprint(
        [source.parent_path for source in identity.sources],
        [remote.size_bytes for remote in remote_objects],
    )
    if reconstructed != identity.dataset_fingerprint:
        raise ValueError(
            "Remote object sizes reconstruct dataset fingerprint "
            f"{reconstructed}, expected saved {identity.dataset_fingerprint}"
        )
    selections = select_compact_windows(
        identity,
        remote_objects,
        train_windows=train_windows,
        holdout_windows=holdout_windows,
        seed=seed,
    )
    plan = _build_plan(
        output_dir=output,
        identity=identity,
        remote_objects=remote_objects,
        selections=selections,
        train_windows=train_windows,
        holdout_windows=holdout_windows,
        seed=seed,
    )
    staging, lock_fd = _open_staging(output, plan)
    try:
        _prepare_staging_tree(staging)
        materialization_pairs: Dict[
            str, Tuple[Tuple[Dict[str, Any], Dict[str, Any], bytes], ...]
        ] = {}
        for split in ("train", "holdout"):
            split_started = time.monotonic()
            _emit_progress(
                progress,
                event="split_materialization_start",
                started_at=split_started,
                split=split,
                sources_total=len(selections[split]),
                workers=workers,
            )
            completed = 0
            completion_lock = threading.Lock()

            def materialize(selection: CompactSelection, *, split_name: str = split):
                nonlocal completed
                result = _materialize_selection(
                    selection,
                    store=store,
                    staging_dir=staging,
                    output_dir=output,
                    plan_sha256=str(plan["content_sha256"]),
                )
                with completion_lock:
                    completed += 1
                    current = completed
                if current % 25 == 0 or current == len(selections[split_name]):
                    _emit_progress(
                        progress,
                        event="split_materialization_progress",
                        started_at=split_started,
                        split=split_name,
                        sources_completed=current,
                        sources_total=len(selections[split_name]),
                    )
                return result

            materialization_pairs[split] = _ordered_bounded_map(
                selections[split],
                materialize,
                workers=workers,
                thread_name_prefix=f"s002-compact-{split}",
            )
        manifest_sources = {
            split: [pair[0] for pair in materialization_pairs[split]]
            for split in ("train", "holdout")
        }
        receipt_splits = {
            split: [pair[1] for pair in materialization_pairs[split]]
            for split in ("train", "holdout")
        }
        compact_materialization_sha256 = _canonical_sha256(receipt_splits)

        _emit_progress(
            progress,
            event="closing_remote_snapshot_start",
            started_at=started_at,
            objects_total=len(identity.sources),
        )
        closing_remote = snapshot_remote_objects(
            identity, store, workers=workers, progress=progress
        )
        if _canonical_bytes([remote.as_dict() for remote in closing_remote]) != _canonical_bytes(
            remote_records
        ):
            raise ValueError("Remote object snapshot changed during compact materialization")

        provisional_manifests = {
            split: _build_manifest(
                split=split,
                sources=manifest_sources[split],
                identity=identity,
                remote_snapshot_sha256=remote_snapshot_sha256,
                compact_materialization_sha256=compact_materialization_sha256,
                verification_receipt_sha256="0" * 64,
                seed=seed,
            )
            for split in ("train", "holdout")
        }
        manifest_contract_sha256 = {
            split: _manifest_contract_sha256(provisional_manifests[split])
            for split in ("train", "holdout")
        }
        receipt = {
            "format": VERIFICATION_RECEIPT_FORMAT,
            "version": VERIFICATION_RECEIPT_VERSION,
            "hash_algorithm": "sha256",
            "builder_implementation": BUILDER_IMPLEMENTATION_REFERENCE,
            "builder_sha256": BUILDER_IMPLEMENTATION_SHA256,
            "parent_paths_sha256": identity.paths_sha256,
            "parent_mix_sha256": identity.mix_sha256,
            "parent_config_sha256": identity.config_sha256,
            "parent_trainer_state_sha256": identity.trainer_state_sha256,
            "parent_dataset_fingerprint": identity.dataset_fingerprint,
            "mirror_policy": MIRROR_POLICY,
            "remote_snapshot_sha256": remote_snapshot_sha256,
            "compact_materialization_sha256": compact_materialization_sha256,
            "manifest_contract_sha256": manifest_contract_sha256,
            "remote_sources": remote_records,
            "splits": receipt_splits,
        }
        receipt_raw = _json_bytes(receipt)
        receipt_sha256 = hashlib.sha256(receipt_raw).hexdigest()
        train_manifest = _build_manifest(
            split="train",
            sources=manifest_sources["train"],
            identity=identity,
            remote_snapshot_sha256=remote_snapshot_sha256,
            compact_materialization_sha256=compact_materialization_sha256,
            verification_receipt_sha256=receipt_sha256,
            seed=seed,
        )
        holdout_manifest = _build_manifest(
            split="holdout",
            sources=manifest_sources["holdout"],
            identity=identity,
            remote_snapshot_sha256=remote_snapshot_sha256,
            compact_materialization_sha256=compact_materialization_sha256,
            verification_receipt_sha256=receipt_sha256,
            seed=seed,
        )
        if (
            _manifest_contract_sha256(train_manifest) != manifest_contract_sha256["train"]
            or _manifest_contract_sha256(holdout_manifest) != manifest_contract_sha256["holdout"]
        ):
            raise AssertionError("Final manifest contract changed after receipt binding")
        train_raw = _json_bytes(train_manifest)
        holdout_raw = _json_bytes(holdout_manifest)
        _write_or_verify(staging / VERIFICATION_RECEIPT_FILENAME, receipt_raw)
        _write_or_verify(staging / TRAIN_MANIFEST_FILENAME, train_raw)
        _write_or_verify(staging / HOLDOUT_MANIFEST_FILENAME, holdout_raw)

        closing_identity = _load_parent_identity(expected_objects=expected_objects)
        if closing_identity != identity:
            raise ValueError("Pinned parent identity changed during compact materialization")
        _validate_loaded_builder()
        plan_raw = _json_bytes(plan)
        expected_files: Dict[str, _ClosingFileExpectation] = {
            PLAN_FILENAME: _ClosingFileExpectation(
                len(plan_raw), hashlib.sha256(plan_raw).hexdigest(), plan_raw
            ),
            VERIFICATION_RECEIPT_FILENAME: _ClosingFileExpectation(
                len(receipt_raw), receipt_sha256, receipt_raw
            ),
            TRAIN_MANIFEST_FILENAME: _ClosingFileExpectation(
                len(train_raw), hashlib.sha256(train_raw).hexdigest(), train_raw
            ),
            HOLDOUT_MANIFEST_FILENAME: _ClosingFileExpectation(
                len(holdout_raw), hashlib.sha256(holdout_raw).hexdigest(), holdout_raw
            ),
        }
        for split in ("train", "holdout"):
            for manifest_source, receipt_source, completion_raw in materialization_pairs[split]:
                expected_files[str(manifest_source["path"])] = _ClosingFileExpectation(
                    int(manifest_source["size_bytes"]),
                    str(manifest_source["sha256"]),
                    stat_identity=(
                        int(receipt_source["device"]),
                        int(receipt_source["inode"]),
                        int(receipt_source["mtime_ns"]),
                        int(receipt_source["ctime_ns"]),
                    ),
                )
                resume_relative_path = f"resume/{split}/{manifest_source['id']}.json"
                expected_files[resume_relative_path] = _ClosingFileExpectation(
                    len(completion_raw),
                    hashlib.sha256(completion_raw).hexdigest(),
                    completion_raw,
                )
        staging_identity = _validate_and_fsync_staging(staging, expected_files)
        closing_staging = os.lstat(staging)
        if (
            not stat.S_ISDIR(closing_staging.st_mode)
            or (closing_staging.st_dev, closing_staging.st_ino) != staging_identity
        ):
            raise ValueError("Closing artifact root changed before publication")
        _publish_no_replace(staging, output)
        _fsync_directory(output.parent)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

    result = BuildResult(
        output_dir=output,
        train_manifest=output / TRAIN_MANIFEST_FILENAME,
        holdout_manifest=output / HOLDOUT_MANIFEST_FILENAME,
        verification_receipt=output / VERIFICATION_RECEIPT_FILENAME,
        train_manifest_sha256=hashlib.sha256(train_raw).hexdigest(),
        holdout_manifest_sha256=hashlib.sha256(holdout_raw).hexdigest(),
        verification_receipt_sha256=receipt_sha256,
        remote_snapshot_sha256=remote_snapshot_sha256,
        compact_materialization_sha256=compact_materialization_sha256,
    )
    _emit_progress(
        progress,
        event="phase_complete",
        started_at=started_at,
        train_windows=train_windows,
        holdout_windows=holdout_windows,
    )
    return result


def build_compact_replay(
    output_dir: os.PathLike[str] | str,
    *,
    store: Optional[ObjectStore] = None,
    workers: int = DEFAULT_WORKERS,
    progress: Optional[ProgressCallback] = None,
    use_gcloud_access_token: bool = False,
) -> BuildResult:
    """Build the production 100,000-train/1,000-holdout compact s002 replay.

    :param output_dir: New immutable artifact directory.
    :param store: Optional exact-object store implementation.
    :param workers: Bounded metadata and per-source materialization concurrency.
    :param progress: Optional operational progress callback excluded from evidence.
    :param use_gcloud_access_token: Use a locally obtained gcloud token for GCS access.
    :returns: Published paths and their immutable SHA-256 identities.
    """
    resolved_store = store or GCSMirrorStore(use_gcloud_access_token=use_gcloud_access_token)
    return _build_compact_replay(
        output_dir,
        store=resolved_store,
        workers=workers,
        progress=progress,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--use-gcloud-access-token",
        action="store_true",
        help="Use `gcloud auth print-access-token` locally; ambient ADC remains the default.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the production compact replay builder and return a process exit code."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = build_compact_replay(
            args.output_dir,
            workers=args.workers,
            progress=_stderr_progress,
            use_gcloud_access_token=args.use_gcloud_access_token,
        )
    except (FileExistsError, OSError, RuntimeError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "train_manifest": str(result.train_manifest),
                "train_manifest_sha256": result.train_manifest_sha256,
                "holdout_manifest": str(result.holdout_manifest),
                "holdout_manifest_sha256": result.holdout_manifest_sha256,
                "verification_receipt": str(result.verification_receipt),
                "verification_receipt_sha256": result.verification_receipt_sha256,
                "remote_snapshot_sha256": result.remote_snapshot_sha256,
                "compact_materialization_sha256": result.compact_materialization_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


def _validate_remote_object(value: RemoteObject, source: ParentSource) -> RemoteObject:
    if (
        isinstance(value.parent_path_index, bool)
        or not isinstance(value.parent_path_index, int)
        or value.parent_path_index != source.parent_path_index
        or value.parent_path != source.parent_path
        or value.mirror_uri != _mirror_uri(source.parent_path)
        or isinstance(value.size_bytes, bool)
        or not isinstance(value.size_bytes, int)
        or value.size_bytes <= 0
        or value.size_bytes % TOKEN_ITEM_SIZE
        or isinstance(value.num_tokens, bool)
        or not isinstance(value.num_tokens, int)
        or value.num_tokens != value.size_bytes // TOKEN_ITEM_SIZE
        or not isinstance(value.generation, str)
        or not value.generation.isdigit()
        or int(value.generation) <= 0
        or not isinstance(value.etag, str)
        or not value.etag
        or not isinstance(value.crc32c, str)
        or not value.crc32c
        or (
            value.source_etag is not None
            and (not isinstance(value.source_etag, str) or not value.source_etag)
        )
    ):
        raise ValueError(
            f"Remote object metadata differs for parent index {source.parent_path_index}"
        )
    _validate_base64(value.crc32c, name="crc32c", decoded_size=4)
    _validate_base64(value.md5_hash, name="md5_hash", decoded_size=16)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
