from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn

from olmo_core.eval.checkpoint_model_state import (
    CheckpointModelStateContract,
    CheckpointModelStateVerificationError,
    inspect_checkpoint_model_state,
    verify_checkpoint_model_state,
)


class _ToyModel(nn.Module):
    def __init__(self, shape=(2, 3), dtype=torch.float32):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(6, dtype=dtype).reshape(shape))
        self.register_buffer("token_counts", torch.tensor([2, 4], dtype=torch.int64))


def _write_checkpoint(root: Path, model: nn.Module, *, ephemeral: bool = False) -> None:
    root.mkdir()
    (root / "config.json").write_text('{"model":{"name":"toy"}}')
    (root / "data_paths.txt").write_text("/data/train.npy\n")
    (root / ".metadata.json").write_text(
        '{"ephemeral":' + ("true" if ephemeral else "false") + ',"version":"test"}'
    )
    (root / "train").mkdir()
    (root / "train" / "rank0.pt").write_bytes(b"trainer state")
    dcp.save(
        {"model": model.state_dict(), "optim": {"step": torch.tensor(3)}},
        checkpoint_id=root / "model_and_optim",
    )


def _contract(root: Path, model: nn.Module) -> CheckpointModelStateContract:
    inventory = inspect_checkpoint_model_state(root)
    return CheckpointModelStateContract(
        config_sha256=inventory.config_sha256,
        data_paths_sha256=inventory.data_paths_sha256,
        marker_sha256=inventory.marker_sha256,
        dcp_metadata_sha256=inventory.dcp_metadata_sha256,
        trainer_state_sha256=inventory.trainer_state_sha256,
        model_keyset_sha256=inventory.model_keyset_sha256,
        model_inventory_sha256=inventory.model_inventory_sha256,
        model_tensor_count=len(model.state_dict()),
        model_parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        model_parameter_tensor_count=len(dict(model.named_parameters())),
    )


def _small_file_identity(root: Path):
    paths = (
        root / "config.json",
        root / "data_paths.txt",
        root / ".metadata.json",
        root / "model_and_optim" / ".metadata",
        root / "train" / "rank0.pt",
    )
    return {
        path: (
            path.stat().st_size,
            path.stat().st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in paths
    }


def test_verifies_exact_model_inventory_without_mutating_checkpoint(tmp_path):
    root = tmp_path / "step1"
    model = _ToyModel()
    _write_checkpoint(root, model)
    contract = _contract(root, model)
    before = _small_file_identity(root)

    verified = verify_checkpoint_model_state(root, contract=contract, expected_model=model)

    assert verified.inventory.model_keys == ("token_counts", "weight")
    assert verified.inventory.model_tensor_count == 2
    assert verified.inventory.model_state_numel == 8
    assert verified.parameter_keys == ("weight",)
    assert verified.buffer_keys == ("token_counts",)
    assert verified.model_parameter_count == 6
    assert _small_file_identity(root) == before


def test_rejects_missing_source_model_key(tmp_path):
    root = tmp_path / "step1"
    source = _ToyModel()
    _write_checkpoint(root, source)
    contract = _contract(root, source)
    expected = _ToyModel()
    expected.extra = nn.Parameter(torch.ones(1))

    with pytest.raises(CheckpointModelStateVerificationError, match="missing=.*extra"):
        verify_checkpoint_model_state(root, contract=contract, expected_model=expected)


@pytest.mark.parametrize(
    "expected",
    [
        _ToyModel(shape=(3, 2)),
        _ToyModel(dtype=torch.float64),
    ],
)
def test_rejects_source_shape_or_dtype_difference(tmp_path, expected):
    root = tmp_path / "step1"
    source = _ToyModel()
    _write_checkpoint(root, source)
    contract = _contract(root, source)

    with pytest.raises(CheckpointModelStateVerificationError, match="shape/dtype inventory"):
        verify_checkpoint_model_state(root, contract=contract, expected_model=expected)


def test_rejects_artifact_hash_difference(tmp_path):
    root = tmp_path / "step1"
    model = _ToyModel()
    _write_checkpoint(root, model)
    contract = _contract(root, model)
    (root / "config.json").write_text('{"model":{"name":"tampered"}}')

    with pytest.raises(CheckpointModelStateVerificationError, match="config_sha256 mismatch"):
        verify_checkpoint_model_state(root, contract=contract, expected_model=model)


def test_rejects_ephemeral_bridge_source_even_when_hash_is_pinned(tmp_path):
    root = tmp_path / "step1"
    model = _ToyModel()
    _write_checkpoint(root, model, ephemeral=True)
    contract = _contract(root, model)

    with pytest.raises(CheckpointModelStateVerificationError, match="must be permanent"):
        verify_checkpoint_model_state(root, contract=contract, expected_model=model)
