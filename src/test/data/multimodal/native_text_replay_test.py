import hashlib
import json
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from olmo_core.data.multimodal.native_text_replay import (
    NATIVE_TEXT_REPLAY_FORMAT,
    NATIVE_TEXT_REPLAY_VERIFICATION_FORMAT,
    NATIVE_TEXT_REPLAY_VERIFICATION_VERSION,
    NATIVE_TEXT_REPLAY_VERSION,
    NativeTextReplayDataset,
    NativeTextReplayDatasetConfig,
    NativeTextReplayManifest,
    NativeTextReplayVerificationReceipt,
)
from olmo_core.exceptions import OLMoConfigurationError

_BUILDER_SHA256 = "d" * 64
_PARENT_PATH = "s3://ai2-llm/tokens/web-000000.npy"


@dataclass
class _Artifacts:
    manifest_paths: dict[str, Path]
    manifests: dict[str, dict[str, Any]]
    receipt_path: Path
    receipt: dict[str, Any]
    receipt_sha256: str
    compact_paths: dict[str, Path]
    parent_checkpoint: Path
    parent_paths_sha256: str


def _write_json(path: Path, value: Any, *, indent: int | None = 2) -> Path:
    path.write_text(json.dumps(value, sort_keys=True, indent=indent) + "\n")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _manifest_contract_sha256(manifest: dict[str, Any]) -> str:
    contract = deepcopy(manifest)
    del contract["provenance"]["verification_receipt_sha256"]
    return _canonical_sha256(contract)


def _make_artifacts(
    root: Path,
    *,
    sequence_length: int = 3,
    train_parent_starts: tuple[int, ...] = (0, 6),
    holdout_parent_starts: tuple[int, ...] = (3, 9),
    train_values: list[int] | None = None,
) -> _Artifacts:
    root.mkdir(parents=True, exist_ok=True)
    parent_checkpoint = root / "parent-checkpoint"
    parent_checkpoint.mkdir()
    parent_paths_file = parent_checkpoint / "data_paths.txt"
    parent_paths_file.write_text(_PARENT_PATH + "\n")
    parent_paths_sha256 = _sha256(parent_paths_file)

    starts_by_split = {
        "train": train_parent_starts,
        "holdout": holdout_parent_starts,
    }
    parent_num_tokens = max(start for starts in starts_by_split.values() for start in starts)
    parent_num_tokens += sequence_length
    compact_paths = {split: root / f"{split}-tokens.npy" for split in ("train", "holdout")}
    for split, compact_path in compact_paths.items():
        num_tokens = len(starts_by_split[split]) * sequence_length
        values = train_values if split == "train" and train_values is not None else None
        tokens = np.asarray(values or range(1, num_tokens + 1), dtype=np.uint32)
        assert len(tokens) == num_tokens
        tokens.tofile(compact_path)

    remote_sources = [
        {
            "parent_path_index": 0,
            "parent_path": _PARENT_PATH,
            "mirror_uri": "gs://ai2-llm/tokens/web-000000.npy",
            "size_bytes": parent_num_tokens * 4,
            "num_tokens": parent_num_tokens,
            "generation": "123456789",
            "etag": "gcs-etag",
            "md5_hash": None,
            "crc32c": "AAAAAA==",
            "source_etag": None,
        }
    ]
    remote_snapshot_sha256 = _canonical_sha256(remote_sources)
    manifests: dict[str, dict[str, Any]] = {}
    split_receipt_sources: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "holdout"):
        parent_starts = list(starts_by_split[split])
        compact_path = compact_paths[split]
        num_windows = len(parent_starts)
        num_tokens = num_windows * sequence_length
        source_id = f"web-000000-{split}"
        compact_stat = compact_path.stat()
        source = {
            "id": source_id,
            "source": "web",
            "parent_path_index": 0,
            "parent_path": _PARENT_PATH,
            "path": compact_path.name,
            "parent_num_tokens": parent_num_tokens,
            "num_tokens": num_tokens,
            "size_bytes": num_tokens * 4,
            "sha256": _sha256(compact_path),
            "window_starts": [index * sequence_length for index in range(num_windows)],
            "parent_window_starts": parent_starts,
        }
        split_receipt_sources[split] = [
            {
                "id": source_id,
                "source": "web",
                "parent_path_index": 0,
                "parent_path": _PARENT_PATH,
                "path": compact_path.name,
                "resolved_path": str(compact_path.resolve()),
                "parent_num_tokens": parent_num_tokens,
                "num_tokens": num_tokens,
                "size_bytes": num_tokens * 4,
                "mtime_ns": compact_stat.st_mtime_ns,
                "ctime_ns": compact_stat.st_ctime_ns,
                "inode": compact_stat.st_ino,
                "device": compact_stat.st_dev,
                "sha256": source["sha256"],
                "num_windows": num_windows,
                "parent_window_starts_sha256": _canonical_sha256(parent_starts),
            }
        ]
        manifests[split] = {
            "format": NATIVE_TEXT_REPLAY_FORMAT,
            "version": NATIVE_TEXT_REPLAY_VERSION,
            "sequence_length": sequence_length,
            "dtype": "uint32",
            "tokenizer": {
                "identifier": "allenai/dolma2-tokenizer",
                "vocab_size": 100_300,
                "eos_token_id": 100_257,
                "pad_token_id": 100_277,
            },
            "provenance": {
                "parent_checkpoint": str(parent_checkpoint),
                "parent_mix": "OLMo-mix-0925",
                "parent_paths_sha256": parent_paths_sha256,
                "parent_mix_sha256": "b" * 64,
                "parent_config_sha256": "c" * 64,
                "parent_trainer_state_sha256": "d" * 64,
                "parent_dataset_fingerprint": "e" * 64,
                "remote_snapshot_sha256": remote_snapshot_sha256,
                "compact_materialization_sha256": "0" * 64,
                "builder_implementation": "offline/compact-replay-builder.py",
                "builder_sha256": _BUILDER_SHA256,
                "instance_filter": {
                    "repetition_min_period": 1,
                    "repetition_max_period": 13,
                    "repetition_max_count": 32,
                },
                "selection_algorithm": "affine-grid-v1",
                "selection_seed": 17,
                "split": split,
                "usable_tokens": num_windows * (sequence_length - 1),
                "source_usable_tokens": {"web": num_windows * (sequence_length - 1)},
                "minimum_source_usable_tokens": {},
                "raw_tokens_per_window": sequence_length,
                "loss_tokens_per_window": sequence_length - 1,
                "verification_receipt_sha256": "0" * 64,
            },
            "num_windows": num_windows,
            "sources": [source],
        }

    compact_materialization_sha256 = _canonical_sha256(split_receipt_sources)
    for manifest in manifests.values():
        manifest["provenance"]["compact_materialization_sha256"] = compact_materialization_sha256
    receipt = {
        "format": NATIVE_TEXT_REPLAY_VERIFICATION_FORMAT,
        "version": NATIVE_TEXT_REPLAY_VERIFICATION_VERSION,
        "hash_algorithm": "sha256",
        "builder_implementation": "offline/compact-replay-builder.py",
        "builder_sha256": _BUILDER_SHA256,
        "parent_paths_sha256": parent_paths_sha256,
        "parent_mix_sha256": "b" * 64,
        "parent_config_sha256": "c" * 64,
        "parent_trainer_state_sha256": "d" * 64,
        "parent_dataset_fingerprint": "e" * 64,
        "remote_snapshot_sha256": remote_snapshot_sha256,
        "compact_materialization_sha256": compact_materialization_sha256,
        "manifest_contract_sha256": {
            split: _manifest_contract_sha256(manifest) for split, manifest in manifests.items()
        },
        "mirror_policy": "s3-to-gs-same-bucket-key-v1",
        "remote_sources": remote_sources,
        "splits": split_receipt_sources,
    }
    receipt_path = _write_json(root / "receipt.json", receipt)
    receipt_sha256 = _sha256(receipt_path)
    manifest_paths = {}
    for split, manifest in manifests.items():
        manifest["provenance"]["verification_receipt_sha256"] = receipt_sha256
        manifest_paths[split] = _write_json(root / f"{split}-manifest.json", manifest)
    return _Artifacts(
        manifest_paths,
        manifests,
        receipt_path,
        receipt,
        receipt_sha256,
        compact_paths,
        parent_checkpoint,
        parent_paths_sha256,
    )


def _republish(artifacts: _Artifacts) -> None:
    compact_sha256 = _canonical_sha256(artifacts.receipt["splits"])
    artifacts.receipt["compact_materialization_sha256"] = compact_sha256
    for manifest in artifacts.manifests.values():
        manifest["provenance"]["compact_materialization_sha256"] = compact_sha256
    artifacts.receipt["manifest_contract_sha256"] = {
        split: _manifest_contract_sha256(manifest)
        for split, manifest in artifacts.manifests.items()
    }
    _write_json(artifacts.receipt_path, artifacts.receipt)
    artifacts.receipt_sha256 = _sha256(artifacts.receipt_path)
    for split, manifest in artifacts.manifests.items():
        manifest["provenance"]["verification_receipt_sha256"] = artifacts.receipt_sha256
        _write_json(artifacts.manifest_paths[split], manifest)


def _fingerprint(path: Path) -> str:
    return NativeTextReplayManifest.load(path).content_fingerprint


def _dataset(artifacts: _Artifacts, split: str = "train", **kwargs: Any):
    return NativeTextReplayDataset(
        artifacts.manifest_paths[split],
        expected_fingerprint=_fingerprint(artifacts.manifest_paths[split]),
        verification_receipt_path=artifacts.receipt_path,
        expected_verification_receipt_sha256=artifacts.receipt_sha256,
        **kwargs,
    )


def test_dataset_preserves_native_next_token_semantics_and_provenance(tmp_path: Path):
    artifacts = _make_artifacts(tmp_path)
    config = NativeTextReplayDatasetConfig(
        manifest_path=str(artifacts.manifest_paths["train"]),
        expected_fingerprint=_fingerprint(artifacts.manifest_paths["train"]),
        expected_parent_checkpoint=str(artifacts.parent_checkpoint),
        expected_parent_mix="OLMo-mix-0925",
        expected_parent_paths_sha256=artifacts.parent_paths_sha256,
        verification_receipt_path=str(artifacts.receipt_path),
        expected_verification_receipt_sha256=artifacts.receipt_sha256,
    )
    dataset = config.build(SimpleNamespace(eos_token_id=100_257, pad_token_id=100_277))

    example = dataset.get(1, epoch=99)
    np.testing.assert_array_equal(example["input_ids"], [4, 5, 6])
    np.testing.assert_array_equal(example["labels"], [5, 6, -100])
    np.testing.assert_array_equal(example["loss_masks"], [1.0, 1.0, 0.0])
    assert example["images"].shape[0] == 0
    assert dataset.source_counts == {"web": 2}
    assert dataset.fingerprint_version == "native-text-replay-v3"
    assert example["metadata"]["instance_filter_valid"]
    assert example["metadata"]["parent_start"] == 6
    assert example["metadata"]["compact_start"] == 3
    np.testing.assert_array_equal(dataset[-1]["input_ids"], example["input_ids"])

    with pytest.raises(OLMoConfigurationError, match="does not preserve native replay token IDs"):
        dataset.validate_tokenizer(SimpleNamespace(eos_token_id=1, pad_token_id=100_277))


def test_runtime_repetition_filter_and_token_id_validation(tmp_path: Path):
    repeated = _make_artifacts(
        tmp_path / "repeated",
        sequence_length=40,
        train_parent_starts=(0,),
        holdout_parent_starts=(40,),
        train_values=[7] * 40,
    )
    example = _dataset(repeated)[0]
    assert not example["metadata"]["instance_filter_valid"]
    assert np.all(example["labels"] == -100)

    invalid = _make_artifacts(
        tmp_path / "invalid-id",
        train_parent_starts=(0,),
        holdout_parent_starts=(3,),
        train_values=[1, 100_300, 2],
    )
    with pytest.raises(RuntimeError, match="outside native vocabulary"):
        _ = _dataset(invalid)[0]
    np.testing.assert_array_equal(
        _dataset(invalid, validate_token_ids=False)[0]["input_ids"], [1, 100_300, 2]
    )


def test_manifest_fingerprint_is_canonical_and_content_sensitive(tmp_path: Path):
    artifacts = _make_artifacts(tmp_path)
    path = artifacts.manifest_paths["train"]
    original = _fingerprint(path)
    expected = hashlib.sha256(
        b"olmo-native-text-replay-v3\0" + _canonical_bytes(artifacts.manifests["train"])
    ).hexdigest()
    assert original == expected

    _write_json(path, artifacts.manifests["train"], indent=None)
    assert _fingerprint(path) == original
    artifacts.manifests["train"]["provenance"]["selection_seed"] = 18
    _write_json(path, artifacts.manifests["train"])
    assert _fingerprint(path) != original


def test_legacy_manifest_and_receipt_versions_are_rejected(tmp_path: Path):
    artifacts = _make_artifacts(tmp_path)
    manifest = deepcopy(artifacts.manifests["train"])
    manifest["version"] = 2
    with pytest.raises(OLMoConfigurationError, match="expected 3"):
        NativeTextReplayManifest.load(_write_json(tmp_path / "v2-manifest.json", manifest))

    receipt = deepcopy(artifacts.receipt)
    receipt["version"] = 2
    path = _write_json(tmp_path / "v2-receipt.json", receipt)
    with pytest.raises(OLMoConfigurationError, match="expected 3"):
        NativeTextReplayVerificationReceipt.load(path, expected_sha256=_sha256(path))


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda data: data.update(unexpected=True), "unknown fields"),
        (lambda data: data.update(dtype="uint16"), "dtype must be 'uint32'"),
        (lambda data: data["sources"][0].update(window_starts=[0, 6]), "local window_starts"),
        (
            lambda data: data["sources"][0].update(parent_window_starts=[0, 4]),
            "not aligned",
        ),
        (
            lambda data: data["sources"][0].update(parent_window_starts=[0, 0]),
            "ordered and non-overlapping",
        ),
        (lambda data: data["sources"][0].update(parent_num_tokens=8), "out of bounds"),
        (lambda data: data["sources"][0].update(path="../outside.npy"), "relative POSIX path"),
        (
            lambda data: data["sources"][0].update(parent_window_starts=[0, True]),
            "must be an integer",
        ),
    ],
)
def test_manifest_rejects_schema_window_and_path_attacks(tmp_path: Path, mutate, match: str):
    artifacts = _make_artifacts(tmp_path)
    attacked = deepcopy(artifacts.manifests["train"])
    mutate(attacked)
    with pytest.raises(OLMoConfigurationError, match=match):
        NativeTextReplayManifest.load(_write_json(tmp_path / "attacked-manifest.json", attacked))


def test_manifest_rejects_compact_symlinks(tmp_path: Path):
    artifacts = _make_artifacts(tmp_path)
    link = tmp_path / "linked.npy"
    link.symlink_to(artifacts.compact_paths["train"])
    attacked = deepcopy(artifacts.manifests["train"])
    attacked["sources"][0]["path"] = link.name
    with pytest.raises(OLMoConfigurationError, match="must not traverse symbolic links"):
        NativeTextReplayManifest.load(_write_json(tmp_path / "symlink-manifest.json", attacked))


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda receipt: receipt.update(unexpected=True), "unknown fields"),
        (lambda receipt: receipt.update(mirror_policy="other"), "mirror_policy"),
        (
            lambda receipt: receipt["remote_sources"][0].update(mirror_uri="gs://other/x"),
            "wrong mirror URI",
        ),
        (
            lambda receipt: receipt["remote_sources"][0].update(generation="12x"),
            "generation.*digits",
        ),
        (lambda receipt: receipt["splits"].pop("holdout"), "missing required fields"),
        (
            lambda receipt: receipt["splits"]["train"][0].update(num_windows=True),
            "must be an integer",
        ),
        (
            lambda receipt: receipt["splits"]["train"][0].update(
                resolved_path="/tmp/../tmp/not-normalized.npy"
            ),
            "resolved paths must be normalized",
        ),
        (
            lambda receipt: receipt["splits"]["train"][0].update(parent_num_tokens=1),
            "parent token count differs",
        ),
    ],
)
def test_receipt_rejects_schema_remote_and_materialization_attacks(
    tmp_path: Path, mutate, match: str
):
    artifacts = _make_artifacts(tmp_path)
    attacked = deepcopy(artifacts.receipt)
    mutate(attacked)
    path = _write_json(tmp_path / "attacked-receipt.json", attacked)
    with pytest.raises(OLMoConfigurationError, match=match):
        NativeTextReplayVerificationReceipt.load(path, expected_sha256=_sha256(path))


def test_dataset_requires_exact_manifest_and_receipt_pins(tmp_path: Path):
    artifacts = _make_artifacts(tmp_path)
    path = artifacts.manifest_paths["train"]
    with pytest.raises(OLMoConfigurationError, match="explicit expected_fingerprint"):
        NativeTextReplayDataset(
            path,
            verification_receipt_path=artifacts.receipt_path,
            expected_verification_receipt_sha256=artifacts.receipt_sha256,
        )
    with pytest.raises(OLMoConfigurationError, match="requires a pinned verification receipt"):
        NativeTextReplayDataset(path, expected_fingerprint=_fingerprint(path))
    with pytest.raises(OLMoConfigurationError, match="content fingerprint"):
        NativeTextReplayDataset(
            path,
            expected_fingerprint="f" * 64,
            verification_receipt_path=artifacts.receipt_path,
            expected_verification_receipt_sha256=artifacts.receipt_sha256,
        )
    with pytest.raises(OLMoConfigurationError, match="receipt SHA-256"):
        NativeTextReplayDataset(
            path,
            expected_fingerprint=_fingerprint(path),
            verification_receipt_path=artifacts.receipt_path,
            expected_verification_receipt_sha256="f" * 64,
        )


@pytest.mark.parametrize("attack", ["lineage", "contract", "source"])
def test_receipt_cross_binds_lineage_manifest_contract_and_sources(tmp_path: Path, attack: str):
    artifacts = _make_artifacts(tmp_path)
    if attack == "lineage":
        artifacts.manifests["train"]["provenance"]["parent_config_sha256"] = "f" * 64
        _write_json(artifacts.manifest_paths["train"], artifacts.manifests["train"])
        match = "parent_config_sha256"
    elif attack == "contract":
        artifacts.manifests["train"]["tokenizer"]["identifier"] = "attacker/tokenizer"
        _write_json(artifacts.manifest_paths["train"], artifacts.manifests["train"])
        match = "manifest contract"
    else:
        artifacts.receipt["splits"]["train"][0]["parent_window_starts_sha256"] = "f" * 64
        _republish(artifacts)
        match = "differs from the pinned"

    with pytest.raises(OLMoConfigurationError, match=match):
        _dataset(artifacts)


def test_pair_validation_accepts_disjoint_windows_and_rejects_overlap(tmp_path: Path):
    valid = _make_artifacts(tmp_path / "valid")
    receipt = NativeTextReplayVerificationReceipt.load(
        valid.receipt_path, expected_sha256=valid.receipt_sha256
    )
    receipt.validate_pair(
        NativeTextReplayManifest.load(valid.manifest_paths["train"]),
        NativeTextReplayManifest.load(valid.manifest_paths["holdout"]),
    )

    overlap = _make_artifacts(
        tmp_path / "overlap",
        train_parent_starts=(0, 6),
        holdout_parent_starts=(0, 9),
    )
    receipt = NativeTextReplayVerificationReceipt.load(
        overlap.receipt_path, expected_sha256=overlap.receipt_sha256
    )
    with pytest.raises(OLMoConfigurationError, match="overlap in parent path"):
        receipt.validate_pair(
            NativeTextReplayManifest.load(overlap.manifest_paths["train"]),
            NativeTextReplayManifest.load(overlap.manifest_paths["holdout"]),
        )


def test_runtime_stat_and_optional_full_hash_validation(tmp_path: Path):
    changed = _make_artifacts(tmp_path / "changed")
    tokens = np.fromfile(changed.compact_paths["train"], dtype=np.uint32)
    tokens[0] = 42
    tokens.tofile(changed.compact_paths["train"])
    changed_stat = changed.compact_paths["train"].stat()
    os.utime(
        changed.compact_paths["train"],
        ns=(changed_stat.st_atime_ns, changed_stat.st_mtime_ns + 1_000_000_000),
    )
    with pytest.raises(OLMoConfigurationError, match="stat signature"):
        _dataset(changed)

    hash_check = _make_artifacts(tmp_path / "hash-check")
    tokens = np.fromfile(hash_check.compact_paths["train"], dtype=np.uint32)
    tokens[0] = 42
    tokens.tofile(hash_check.compact_paths["train"])
    source_stat = hash_check.compact_paths["train"].stat()
    hash_check.receipt["splits"]["train"][0].update(
        mtime_ns=source_stat.st_mtime_ns,
        ctime_ns=source_stat.st_ctime_ns,
        inode=source_stat.st_ino,
        device=source_stat.st_dev,
    )
    _republish(hash_check)
    _dataset(hash_check, verify_source_hashes=False)
    with pytest.raises(OLMoConfigurationError, match="has SHA-256"):
        _dataset(hash_check, verify_source_hashes=True)

    replaced = _make_artifacts(tmp_path / "replaced")
    dataset = _dataset(replaced)
    replacement = replaced.compact_paths["train"].with_suffix(".replacement")
    replacement.write_bytes(replaced.compact_paths["train"].read_bytes())
    os.replace(replacement, replaced.compact_paths["train"])
    with pytest.raises(RuntimeError, match="changed stat signature before read"):
        _ = dataset[0]


@pytest.mark.parametrize("replacement_kind", ["directory", "fifo"])
def test_runtime_rejects_non_regular_compact_sources(tmp_path: Path, replacement_kind: str):
    artifacts = _make_artifacts(tmp_path)
    compact_path = artifacts.compact_paths["train"]
    compact_path.unlink()
    if replacement_kind == "directory":
        compact_path.mkdir()
    else:
        os.mkfifo(compact_path)

    with pytest.raises(OLMoConfigurationError, match="not a regular file"):
        _dataset(artifacts)
