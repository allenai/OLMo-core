"""Read-only model-state inventory and verification for native DCP checkpoints.

This module deliberately reads only the checkpoint's small artifact files and DCP
``.metadata``. It never opens ``*.distcp`` tensor shards. A caller can compare the source
inventory with a model built on ``meta`` to prove the exact key, shape, dtype, parameter, and
buffer contract before beginning a distributed bridge load.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

from olmo_core.aliases import PathOrStr


class CheckpointModelStateVerificationError(ValueError):
    """Raised when a checkpoint differs from its immutable model-state contract."""


@dataclass(frozen=True, order=True)
class ModelTensorMetadata:
    """Canonical metadata for one source model tensor, excluding the outer ``model.`` prefix."""

    key: str
    shape: Tuple[int, ...]
    dtype: str

    @property
    def numel(self) -> int:
        """Return the number of scalar elements described by :attr:`shape`."""

        return math.prod(self.shape)

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation used by inventory hashing."""

        return {"key": self.key, "shape": list(self.shape), "dtype": self.dtype}


@dataclass(frozen=True)
class CheckpointModelStateInventory:
    """Read-only artifact hashes and exact model tensor metadata for one checkpoint."""

    checkpoint: Path
    dcp_directory: Path
    config_sha256: str
    data_paths_sha256: str
    marker_sha256: str
    dcp_metadata_sha256: str
    trainer_state_sha256: Optional[str]
    marker_ephemeral: bool
    tensors: Tuple[ModelTensorMetadata, ...]
    model_keyset_sha256: str
    model_inventory_sha256: str

    @property
    def model_keys(self) -> Tuple[str, ...]:
        """Return the exact sorted source model key set."""

        return tuple(tensor.key for tensor in self.tensors)

    @property
    def model_tensor_count(self) -> int:
        """Return the number of model tensor entries in DCP metadata."""

        return len(self.tensors)

    @property
    def model_state_numel(self) -> int:
        """Return total scalar elements across model tensors, including persistent buffers."""

        return sum(tensor.numel for tensor in self.tensors)


@dataclass(frozen=True)
class CheckpointModelStateContract:
    """Immutable expected artifact and model-state identity for bridge initialization.

    ``model_parameter_count`` counts scalar parameters, not parameter tensors. The latter can
    optionally be pinned independently through ``model_parameter_tensor_count``.
    """

    config_sha256: str
    data_paths_sha256: str
    marker_sha256: str
    dcp_metadata_sha256: str
    model_keyset_sha256: str
    model_inventory_sha256: str
    model_tensor_count: int
    model_parameter_count: int
    model_parameter_tensor_count: Optional[int] = None
    trainer_state_sha256: Optional[str] = None
    require_permanent: bool = True

    def __post_init__(self) -> None:
        hashes = {
            "config_sha256": self.config_sha256,
            "data_paths_sha256": self.data_paths_sha256,
            "marker_sha256": self.marker_sha256,
            "dcp_metadata_sha256": self.dcp_metadata_sha256,
            "model_keyset_sha256": self.model_keyset_sha256,
            "model_inventory_sha256": self.model_inventory_sha256,
        }
        if self.trainer_state_sha256 is not None:
            hashes["trainer_state_sha256"] = self.trainer_state_sha256
        for name, value in hashes.items():
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if self.model_tensor_count <= 0:
            raise ValueError("model_tensor_count must be positive")
        if self.model_parameter_count <= 0:
            raise ValueError("model_parameter_count must be positive")
        if self.model_parameter_tensor_count is not None and self.model_parameter_tensor_count <= 0:
            raise ValueError("model_parameter_tensor_count must be positive when provided")


@dataclass(frozen=True)
class VerifiedCheckpointModelState:
    """A checkpoint inventory proven to match an exact expected bare model."""

    inventory: CheckpointModelStateInventory
    parameter_keys: Tuple[str, ...]
    buffer_keys: Tuple[str, ...]
    model_parameter_count: int

    @property
    def model_parameter_tensor_count(self) -> int:
        """Return the number of named parameter entries in the source model state."""

        return len(self.parameter_keys)


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise CheckpointModelStateVerificationError(f"Required checkpoint file is absent: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _tensor_metadata(key: str, shape: torch.Size | Tuple[int, ...], dtype: torch.dtype):
    return ModelTensorMetadata(
        key=key,
        shape=tuple(int(dim) for dim in shape),
        dtype=_dtype_name(dtype),
    )


def inspect_checkpoint_model_state(
    checkpoint: PathOrStr,
    *,
    model_prefix: str = "model.",
    trainer_state_relative_path: Optional[str] = "train/rank0.pt",
) -> CheckpointModelStateInventory:
    """Read artifact hashes and the exact model inventory without loading tensor shards.

    :param checkpoint: Native checkpoint root containing ``config.json``, ``data_paths.txt``,
        ``.metadata.json``, and ``model_and_optim/.metadata``.
    :param model_prefix: DCP key prefix identifying model state. The prefix is stripped from
        returned source keys.
    :param trainer_state_relative_path: Optional checkpoint-relative trainer-state artifact to
        hash when present. Set to ``None`` to ignore trainer state.
    :returns: Canonical artifact hashes and sorted tensor metadata.
    :raises CheckpointModelStateVerificationError: If required artifacts or metadata are invalid.
    """

    if not model_prefix or not model_prefix.endswith("."):
        raise ValueError("model_prefix must be non-empty and end with '.'")
    root = Path(checkpoint).expanduser().resolve()
    if not root.is_dir():
        raise CheckpointModelStateVerificationError(f"Checkpoint directory is absent: {root}")

    config_path = root / "config.json"
    data_paths_path = root / "data_paths.txt"
    marker_path = root / ".metadata.json"
    dcp_directory = root / "model_and_optim"
    dcp_metadata_path = dcp_directory / ".metadata"

    config_sha256 = _sha256_file(config_path)
    data_paths_sha256 = _sha256_file(data_paths_path)
    marker_sha256 = _sha256_file(marker_path)
    dcp_metadata_sha256 = _sha256_file(dcp_metadata_path)

    try:
        config = json.loads(config_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointModelStateVerificationError(
            f"Checkpoint config is not valid JSON: {config_path}"
        ) from error
    if not isinstance(config, Mapping) or not isinstance(config.get("model"), Mapping):
        raise CheckpointModelStateVerificationError(
            f"Checkpoint config lacks a model mapping: {config_path}"
        )

    try:
        marker = json.loads(marker_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointModelStateVerificationError(
            f"Checkpoint marker is not valid JSON: {marker_path}"
        ) from error
    if not isinstance(marker, Mapping) or type(marker.get("ephemeral")) is not bool:
        raise CheckpointModelStateVerificationError(
            f"Checkpoint marker lacks a boolean 'ephemeral' field: {marker_path}"
        )

    trainer_state_sha256: Optional[str] = None
    if trainer_state_relative_path is not None:
        trainer_state_path = root / trainer_state_relative_path
        if trainer_state_path.exists():
            trainer_state_sha256 = _sha256_file(trainer_state_path)

    try:
        dcp_metadata = FileSystemReader(dcp_directory).read_metadata()
    except Exception as error:
        raise CheckpointModelStateVerificationError(
            f"Could not read DCP metadata from {dcp_directory}"
        ) from error

    tensors = []
    for full_key, metadata in dcp_metadata.state_dict_metadata.items():
        if not full_key.startswith(model_prefix):
            continue
        key = full_key[len(model_prefix) :]
        if not key:
            raise CheckpointModelStateVerificationError(
                f"Checkpoint contains an empty model key under prefix {model_prefix!r}"
            )
        if not isinstance(metadata, TensorStorageMetadata):
            raise CheckpointModelStateVerificationError(
                f"Checkpoint model entry {full_key!r} is not tensor metadata"
            )
        tensors.append(_tensor_metadata(key, metadata.size, metadata.properties.dtype))
    tensors.sort()
    if not tensors:
        raise CheckpointModelStateVerificationError(
            f"Checkpoint contains no tensor entries under prefix {model_prefix!r}"
        )
    keys = [tensor.key for tensor in tensors]
    records = [tensor.as_dict() for tensor in tensors]

    return CheckpointModelStateInventory(
        checkpoint=root,
        dcp_directory=dcp_directory,
        config_sha256=config_sha256,
        data_paths_sha256=data_paths_sha256,
        marker_sha256=marker_sha256,
        dcp_metadata_sha256=dcp_metadata_sha256,
        trainer_state_sha256=trainer_state_sha256,
        marker_ephemeral=marker["ephemeral"],
        tensors=tuple(tensors),
        model_keyset_sha256=_canonical_sha256(keys),
        model_inventory_sha256=_canonical_sha256(records),
    )


def _named_parameter_keys(model: nn.Module) -> Tuple[str, ...]:
    try:
        named_parameters = model.named_parameters(remove_duplicate=False)
    except TypeError:  # pragma: no cover - compatibility with older torch releases.
        named_parameters = model.named_parameters()
    return tuple(sorted(name for name, _ in named_parameters))


def _expected_model_tensors(model: nn.Module) -> Tuple[ModelTensorMetadata, ...]:
    tensors = []
    for key, value in model.state_dict().items():
        if not isinstance(value, torch.Tensor):
            raise CheckpointModelStateVerificationError(
                f"Expected model state entry {key!r} is not a tensor"
            )
        tensors.append(_tensor_metadata(key, value.shape, value.dtype))
    tensors.sort()
    return tuple(tensors)


def _raise_hash_mismatch(name: str, actual: str, expected: str) -> None:
    if actual != expected:
        raise CheckpointModelStateVerificationError(
            f"Checkpoint {name} mismatch: expected {expected}, got {actual}"
        )


def verify_checkpoint_model_state(
    checkpoint: PathOrStr,
    *,
    contract: CheckpointModelStateContract,
    expected_model: nn.Module,
    model_prefix: str = "model.",
    trainer_state_relative_path: Optional[str] = "train/rank0.pt",
) -> VerifiedCheckpointModelState:
    """Verify one native checkpoint against artifact pins and an exact bare model.

    The expected model should normally be built on ``meta`` from the already hash-pinned source
    config. Its state is not initialized or materialized. Verification is strictly read-only and
    must run before the bridge's distributed state load.

    :param checkpoint: Native checkpoint root.
    :param contract: Immutable artifact, inventory, tensor-count, and parameter-count pins.
    :param expected_model: Bare source model whose state keys, shapes, dtypes, parameters, and
        buffers must match the DCP ``model.*`` inventory exactly.
    :param model_prefix: DCP key prefix identifying model state.
    :param trainer_state_relative_path: Optional trainer-state file to hash.
    :returns: Verified inventory plus exact parameter and buffer key sets.
    :raises CheckpointModelStateVerificationError: If any part of the contract differs.
    """

    inventory = inspect_checkpoint_model_state(
        checkpoint,
        model_prefix=model_prefix,
        trainer_state_relative_path=trainer_state_relative_path,
    )
    for name in (
        "config_sha256",
        "data_paths_sha256",
        "marker_sha256",
        "dcp_metadata_sha256",
        "model_keyset_sha256",
        "model_inventory_sha256",
    ):
        _raise_hash_mismatch(name, getattr(inventory, name), getattr(contract, name))
    if contract.trainer_state_sha256 is not None:
        if inventory.trainer_state_sha256 is None:
            raise CheckpointModelStateVerificationError(
                "Checkpoint lacks the required trainer-state artifact"
            )
        _raise_hash_mismatch(
            "trainer_state_sha256",
            inventory.trainer_state_sha256,
            contract.trainer_state_sha256,
        )
    if contract.require_permanent and inventory.marker_ephemeral:
        raise CheckpointModelStateVerificationError(
            f"Bridge source checkpoint must be permanent: {inventory.checkpoint}"
        )
    if inventory.model_tensor_count != contract.model_tensor_count:
        raise CheckpointModelStateVerificationError(
            "Checkpoint model tensor count mismatch: "
            f"expected {contract.model_tensor_count}, got {inventory.model_tensor_count}"
        )

    expected_tensors = _expected_model_tensors(expected_model)
    actual_by_key = {tensor.key: tensor for tensor in inventory.tensors}
    expected_by_key = {tensor.key: tensor for tensor in expected_tensors}
    missing = sorted(expected_by_key.keys() - actual_by_key.keys())
    unexpected = sorted(actual_by_key.keys() - expected_by_key.keys())
    if missing or unexpected:
        raise CheckpointModelStateVerificationError(
            "Checkpoint source model key set differs from the expected bare model: "
            f"missing={missing[:16]}, unexpected={unexpected[:16]}"
        )
    mismatches = [
        (
            key,
            actual_by_key[key].shape,
            expected_by_key[key].shape,
            actual_by_key[key].dtype,
            expected_by_key[key].dtype,
        )
        for key in sorted(actual_by_key)
        if actual_by_key[key] != expected_by_key[key]
    ]
    if mismatches:
        raise CheckpointModelStateVerificationError(
            f"Checkpoint source model shape/dtype inventory differs: {mismatches[:16]}"
        )

    parameter_keys = _named_parameter_keys(expected_model)
    unknown_parameter_keys = sorted(set(parameter_keys) - expected_by_key.keys())
    if unknown_parameter_keys:
        raise CheckpointModelStateVerificationError(
            f"Expected model parameters are absent from its state: {unknown_parameter_keys[:16]}"
        )
    parameter_count = sum(parameter.numel() for parameter in expected_model.parameters())
    if parameter_count != contract.model_parameter_count:
        raise CheckpointModelStateVerificationError(
            "Expected model parameter count differs from the checkpoint contract: "
            f"expected {contract.model_parameter_count}, got {parameter_count}"
        )
    if (
        contract.model_parameter_tensor_count is not None
        and len(parameter_keys) != contract.model_parameter_tensor_count
    ):
        raise CheckpointModelStateVerificationError(
            "Expected model parameter tensor count differs from the checkpoint contract: "
            f"expected {contract.model_parameter_tensor_count}, got {len(parameter_keys)}"
        )
    buffer_keys = tuple(sorted(expected_by_key.keys() - set(parameter_keys)))

    return VerifiedCheckpointModelState(
        inventory=inventory,
        parameter_keys=parameter_keys,
        buffer_keys=buffer_keys,
        model_parameter_count=parameter_count,
    )
