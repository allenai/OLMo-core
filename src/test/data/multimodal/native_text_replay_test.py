import hashlib
import json
import os
import pickle
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from olmo_core.data.multimodal.collator import MultimodalCollator
from olmo_core.data.multimodal.native_text_replay import (
    NATIVE_TEXT_REPLAY_BUILDER_IMPLEMENTATION_REFERENCE,
    NATIVE_TEXT_REPLAY_COMPACT_BUILDER_IMPLEMENTATION_REFERENCE,
    NATIVE_TEXT_REPLAY_FORMAT,
    NATIVE_TEXT_REPLAY_VERIFICATION_FORMAT,
    NATIVE_TEXT_REPLAY_VERIFICATION_VERSION,
    NATIVE_TEXT_REPLAY_VERSION,
    NativeTextReplayDataset,
    NativeTextReplayDatasetConfig,
    NativeTextReplayManifest,
    NativeTextReplayVerificationReceipt,
    _reviewed_builder_sha256,
)
from olmo_core.exceptions import OLMoConfigurationError


def _write_raw_tokens(path: Path, values) -> np.ndarray:
    tokens = np.asarray(values, dtype=np.uint32)
    tokens.tofile(path)
    return tokens


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_data(token_path: Path, tokens: np.ndarray, *, sequence_length: int = 3):
    return {
        "format": NATIVE_TEXT_REPLAY_FORMAT,
        "version": 2,
        "sequence_length": sequence_length,
        "dtype": "uint32",
        "tokenizer": {
            "identifier": "allenai/dolma2-tokenizer",
            "vocab_size": 100_300,
            "eos_token_id": 100_257,
            "pad_token_id": 100_277,
        },
        "provenance": {
            "parent_checkpoint": "/checkpoints/s002-step125500",
            "parent_mix": "OLMo-mix-0925",
            "parent_paths_sha256": "a" * 64,
            "parent_mix_sha256": "b" * 64,
            "upstream_provenance_sha256": "c" * 64,
            "builder_implementation": NATIVE_TEXT_REPLAY_BUILDER_IMPLEMENTATION_REFERENCE,
            "builder_sha256": _reviewed_builder_sha256(),
            "instance_filter": {
                "repetition_min_period": 1,
                "repetition_max_period": 13,
                "repetition_max_count": 32,
            },
            "materialized_sources_sha256": "e" * 64,
            "source_catalog_sha256": "f" * 64,
            "source_catalog_format": "olmo_native_text_replay_source_catalog",
            "source_catalog_version": 2,
            "selection_algorithm": "affine-grid-v1",
            "selection_seed": 17,
            "split": "train",
            "usable_tokens": 2 * (sequence_length - 1),
            "source_usable_tokens": {"web": 2 * (sequence_length - 1)},
            "minimum_source_usable_tokens": {},
            "raw_tokens_per_window": sequence_length,
            "loss_tokens_per_window": sequence_length - 1,
        },
        "num_windows": 2,
        "sources": [
            {
                "id": "web-000000",
                "source": "web",
                "parent_path_index": 0,
                "parent_path": "s3://ai2-llm/tokens/web-000000.npy",
                "path": token_path.name,
                "num_tokens": len(tokens),
                "size_bytes": tokens.nbytes,
                "sha256": _sha256(token_path),
                "window_starts": [0, 3],
            }
        ],
    }


def _write_manifest(path: Path, data, *, indent=None) -> Path:
    path.write_text(json.dumps(data, indent=indent))
    return path


def _write_verification_receipt(path: Path, manifest_data, token_path: Path) -> tuple[Path, str]:
    source = manifest_data["sources"][0]
    receipt = {
        "format": NATIVE_TEXT_REPLAY_VERIFICATION_FORMAT,
        "version": 2,
        "hash_algorithm": "sha256",
        "builder_implementation": manifest_data["provenance"]["builder_implementation"],
        "builder_sha256": manifest_data["provenance"]["builder_sha256"],
        "source_catalog_sha256": "b" * 64,
        "parent_paths_sha256": manifest_data["provenance"]["parent_paths_sha256"],
        "parent_mix_sha256": manifest_data["provenance"]["parent_mix_sha256"],
        "upstream_provenance_sha256": manifest_data["provenance"]["upstream_provenance_sha256"],
        "materialized_sources_sha256": "c" * 64,
        "sources": [
            {
                "id": source["id"],
                "source": source["source"],
                "parent_path_index": source["parent_path_index"],
                "parent_path": source["parent_path"],
                "resolved_path": str(token_path.resolve()),
                "num_tokens": source["num_tokens"],
                "size_bytes": source["size_bytes"],
                "sha256": source["sha256"],
            }
        ],
    }
    path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    receipt_sha256 = _sha256(path)
    manifest_data["provenance"].update(
        {
            "source_catalog_sha256": receipt["source_catalog_sha256"],
            "materialized_sources_sha256": receipt["materialized_sources_sha256"],
            "verification_receipt_sha256": receipt_sha256,
        }
    )
    return path, receipt_sha256


def _rewrite_json_with_sha256(path: Path, data) -> str:
    path.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
    return _sha256(path)


def _canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _manifest_contract_sha256(manifest) -> str:
    contract = deepcopy(manifest)
    del contract["provenance"]["verification_receipt_sha256"]
    return _canonical_sha256(contract)


def _write_v3_artifacts(
    tmp_path: Path,
    *,
    train_parent_starts=(0, 6),
    holdout_parent_starts=(3, 9),
):
    sequence_length = 3
    parent_num_tokens = 12
    compact_paths = {
        "train": tmp_path / "train-tokens.npy",
        "holdout": tmp_path / "holdout-tokens.npy",
    }
    for split, compact_path in compact_paths.items():
        starts = train_parent_starts if split == "train" else holdout_parent_starts
        _write_raw_tokens(compact_path, list(range(1, len(starts) * sequence_length + 1)))

    remote_sources = [
        {
            "parent_path_index": 0,
            "parent_path": "s3://ai2-llm/tokens/web-000000.npy",
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
    builder_sha256 = _reviewed_builder_sha256(
        NATIVE_TEXT_REPLAY_COMPACT_BUILDER_IMPLEMENTATION_REFERENCE
    )
    manifests: dict[str, dict[str, Any]] = {}
    split_receipt_sources: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "holdout"):
        starts = list(train_parent_starts if split == "train" else holdout_parent_starts)
        compact_path = compact_paths[split]
        num_windows = len(starts)
        num_tokens = num_windows * sequence_length
        source_id = f"web-000000-{split}"
        compact_stat = compact_path.stat()
        source = {
            "id": source_id,
            "source": "web",
            "parent_path_index": 0,
            "parent_path": remote_sources[0]["parent_path"],
            "path": compact_path.name,
            "parent_num_tokens": parent_num_tokens,
            "num_tokens": num_tokens,
            "size_bytes": num_tokens * 4,
            "sha256": _sha256(compact_path),
            "window_starts": [index * sequence_length for index in range(num_windows)],
            "parent_window_starts": starts,
        }
        split_receipt_sources[split] = [
            {
                "id": source_id,
                "source": "web",
                "parent_path_index": 0,
                "parent_path": remote_sources[0]["parent_path"],
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
                "parent_window_starts_sha256": _canonical_sha256(starts),
            }
        ]
        manifests[split] = {
            "format": NATIVE_TEXT_REPLAY_FORMAT,
            "version": 3,
            "sequence_length": sequence_length,
            "dtype": "uint32",
            "tokenizer": {
                "identifier": "allenai/dolma2-tokenizer",
                "vocab_size": 100_300,
                "eos_token_id": 100_257,
                "pad_token_id": 100_277,
            },
            "provenance": {
                "parent_checkpoint": "/checkpoints/s002-step125500",
                "parent_mix": "OLMo-mix-0925",
                "parent_paths_sha256": "a" * 64,
                "parent_mix_sha256": "b" * 64,
                "parent_config_sha256": "c" * 64,
                "parent_trainer_state_sha256": "d" * 64,
                "parent_dataset_fingerprint": "e" * 64,
                "remote_snapshot_sha256": remote_snapshot_sha256,
                "compact_materialization_sha256": "0" * 64,
                "builder_implementation": NATIVE_TEXT_REPLAY_COMPACT_BUILDER_IMPLEMENTATION_REFERENCE,
                "builder_sha256": builder_sha256,
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
    manifest_contract_sha256 = {
        split: _manifest_contract_sha256(manifest) for split, manifest in manifests.items()
    }
    receipt: dict[str, Any] = {
        "format": NATIVE_TEXT_REPLAY_VERIFICATION_FORMAT,
        "version": 3,
        "hash_algorithm": "sha256",
        "builder_implementation": NATIVE_TEXT_REPLAY_COMPACT_BUILDER_IMPLEMENTATION_REFERENCE,
        "builder_sha256": builder_sha256,
        "parent_paths_sha256": "a" * 64,
        "parent_mix_sha256": "b" * 64,
        "parent_config_sha256": "c" * 64,
        "parent_trainer_state_sha256": "d" * 64,
        "parent_dataset_fingerprint": "e" * 64,
        "remote_snapshot_sha256": remote_snapshot_sha256,
        "compact_materialization_sha256": compact_materialization_sha256,
        "manifest_contract_sha256": manifest_contract_sha256,
        "mirror_policy": "s3-to-gs-same-bucket-key-v1",
        "remote_sources": remote_sources,
        "splits": split_receipt_sources,
    }
    receipt_path = tmp_path / "v3-receipt.json"
    receipt_sha256 = _rewrite_json_with_sha256(receipt_path, receipt)
    manifest_paths: dict[str, Path] = {}
    for split, manifest in manifests.items():
        manifest["provenance"]["verification_receipt_sha256"] = receipt_sha256
        manifest_paths[split] = _write_manifest(tmp_path / f"v3-{split}.json", manifest, indent=2)
    return manifest_paths, manifests, receipt_path, receipt, receipt_sha256, compact_paths


def _v3_fingerprint(manifest_path: Path) -> str:
    return NativeTextReplayManifest.load(manifest_path).content_fingerprint


def _republish_v3_artifacts(manifest_paths, manifests, receipt_path, receipt) -> str:
    compact_materialization_sha256 = _canonical_sha256(receipt["splits"])
    receipt["compact_materialization_sha256"] = compact_materialization_sha256
    for manifest in manifests.values():
        manifest["provenance"]["compact_materialization_sha256"] = compact_materialization_sha256
    receipt["manifest_contract_sha256"] = {
        split: _manifest_contract_sha256(manifest) for split, manifest in manifests.items()
    }
    receipt_sha256 = _rewrite_json_with_sha256(receipt_path, receipt)
    for split, manifest in manifests.items():
        manifest["provenance"]["verification_receipt_sha256"] = receipt_sha256
        _write_manifest(manifest_paths[split], manifest, indent=2)
    return receipt_sha256


@pytest.fixture
def replay_files(tmp_path: Path):
    # Include native Dolma IDs above uint16 to catch accidental narrowing or retokenization.
    tokens = _write_raw_tokens(
        tmp_path / "tokens.npy",
        [100_257, 70_001, 70_002, 100_257, 90_001, 90_002, 90_003, 100_257],
    )
    data = _manifest_data(tmp_path / "tokens.npy", tokens)
    manifest_path = _write_manifest(tmp_path / "manifest.json", data, indent=2)
    return manifest_path, data, tokens


def test_native_replay_matches_parent_fsl_shift_and_final_label_mask(replay_files):
    manifest_path, _, tokens = replay_files
    dataset = NativeTextReplayDataset(
        manifest_path,
        expected_parent_checkpoint="/checkpoints/s002-step125500",
        expected_parent_mix="OLMo-mix-0925",
        verify_source_hashes=True,
    )

    assert len(dataset) == 2
    assert dataset.sequence_length == 3
    assert dataset.source_counts == {"web": 2}

    example = dataset[0]
    np.testing.assert_array_equal(example["input_ids"], tokens[0:3].astype(np.int64))
    np.testing.assert_array_equal(example["labels"], [70_001, 70_002, -100])
    np.testing.assert_array_equal(example["loss_masks"], [1.0, 1.0, 0.0])
    np.testing.assert_array_equal(example["position_ids"], np.arange(3, dtype=np.int64))
    np.testing.assert_array_equal(example["token_type_ids"], np.zeros(3, dtype=np.int64))
    assert example["input_ids"].dtype == np.int64
    assert example["images"].dtype == np.float32
    assert example["images"].shape[0] == 0
    assert example["pooled_patches_idx"].dtype == np.int64
    assert example["pooled_patches_idx"].shape[0] == 0
    assert example["metadata"] == {
        **dataset.provenance_for(0),
        "instance_filter_valid": True,
    }
    assert example["metadata"]["instance_filter_valid"] is True
    assert example["metadata"]["start"] == 0
    assert example["metadata"]["stop"] == 3


def test_native_replay_is_epoch_invariant_bounded_and_pickleable(replay_files):
    manifest_path, _, tokens = replay_files
    dataset = NativeTextReplayDataset(manifest_path)

    for epoch in (0, 1, 19_283):
        np.testing.assert_array_equal(dataset.get(1, epoch)["input_ids"], tokens[3:6])
    np.testing.assert_array_equal(dataset[-1]["labels"], [90_001, 90_002, -100])
    with pytest.raises(IndexError):
        _ = dataset[2]
    with pytest.raises(IndexError):
        _ = dataset[-3]

    restored = pickle.loads(pickle.dumps(dataset))
    np.testing.assert_array_equal(restored[0]["labels"], [70_001, 70_002, -100])


def test_native_replay_applies_parent_repetition_filter(tmp_path):
    tokens = _write_raw_tokens(tmp_path / "repetitive.npy", [7] * 80)
    data = _manifest_data(tmp_path / "repetitive.npy", tokens, sequence_length=40)
    data["sources"][0]["window_starts"] = [0, 40]
    manifest_path = _write_manifest(tmp_path / "repetitive.json", data)

    example = NativeTextReplayDataset(manifest_path)[0]

    assert example["metadata"]["instance_filter_valid"] is False
    np.testing.assert_array_equal(example["labels"], np.full(40, -100, dtype=np.int64))
    np.testing.assert_array_equal(
        example["loss_masks"], np.array([1.0] * 39 + [0.0], dtype=np.float32)
    )


def test_native_replay_collates_as_text_only_without_changing_targets(replay_files):
    manifest_path, _, tokens = replay_files
    dataset = NativeTextReplayDataset(manifest_path)
    collator = MultimodalCollator(pad_token_id=100_277, pad_sequence_length=3)

    batch = collator([dataset[0], dataset[1]])
    np.testing.assert_array_equal(batch["input_ids"].numpy()[0], tokens[0:3])
    np.testing.assert_array_equal(batch["labels"].numpy()[1], [90_001, 90_002, -100])
    np.testing.assert_array_equal(batch["loss_masks"].numpy(), [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    # The source examples have no image. The collator's single zero crop is only the
    # collective-safe placeholder required by the multimodal forward pass.
    assert batch["image_crop_counts"].tolist() == [0, 0]
    assert batch["pooled_token_counts"].tolist() == [0, 0]
    assert batch["images"].shape[1] == 1
    assert not batch["images"].any()
    assert (batch["pooled_patches_idx"] == -1).all()


def test_semantic_fingerprint_is_formatting_invariant_and_tracks_content(replay_files, tmp_path):
    _, data, _ = replay_files
    # Use the same absolute token path so the two manifest objects have identical semantics.
    token_path = tmp_path / data["sources"][0]["path"]
    data["sources"][0]["path"] = str(token_path)
    compact_path = _write_manifest(tmp_path / "compact.json", data)
    pretty_path = _write_manifest(tmp_path / "pretty.json", data, indent=4)

    compact = NativeTextReplayManifest.load(compact_path)
    pretty = NativeTextReplayManifest.load(pretty_path)
    assert compact.content_fingerprint == pretty.content_fingerprint
    assert compact.manifest_sha256 != pretty.manifest_sha256

    changed_data = deepcopy(data)
    changed_data["sources"][0]["source"] = "code"
    changed_data["provenance"]["source_usable_tokens"] = {
        "code": changed_data["provenance"]["usable_tokens"]
    }
    changed_path = _write_manifest(tmp_path / "changed.json", changed_data)
    assert NativeTextReplayManifest.load(changed_path).content_fingerprint != (
        compact.content_fingerprint
    )

    NativeTextReplayDatasetConfig(
        manifest_path=str(compact_path),
        expected_fingerprint=compact.content_fingerprint,
    ).build()
    with pytest.raises(OLMoConfigurationError, match="fingerprint"):
        NativeTextReplayDataset(
            compact_path,
            expected_fingerprint="0" * 64,
        )
    with pytest.raises(OLMoConfigurationError, match="path-list fingerprint"):
        NativeTextReplayDataset(
            compact_path,
            expected_parent_paths_sha256="0" * 64,
        )
    with pytest.raises(OLMoConfigurationError, match="parent checkpoint"):
        NativeTextReplayDataset(
            compact_path,
            expected_parent_checkpoint="/weka/not-s002",
        )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda data: data["sources"][0].update(window_starts=[0, 0]),
            "ordered and non-overlapping",
        ),
        (
            lambda data: data["sources"][0].update(window_starts=[0, 4]),
            "not aligned",
        ),
        (
            lambda data: data["sources"][0].update(window_starts=[0, 6]),
            "out of bounds",
        ),
        (lambda data: data.update(num_windows=3), "declares 3 windows"),
        (lambda data: data["sources"][0].update(size_bytes=1), "num_tokens.*dtype"),
        (lambda data: data.update(dtype="int64"), "Unsupported.*dtype"),
        (
            lambda data: data["provenance"]["instance_filter"].update(repetition_max_count=31),
            "pinned s002 repetition filter",
        ),
    ],
)
def test_manifest_rejects_unbounded_or_inconsistent_windows(replay_files, tmp_path, mutate, match):
    _, data, _ = replay_files
    mutate(data)
    manifest_path = _write_manifest(tmp_path / "invalid.json", data)
    with pytest.raises(OLMoConfigurationError, match=match):
        NativeTextReplayManifest.load(manifest_path)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda data: data.update(unexpected=True), "manifest root has unknown fields"),
        (
            lambda data: data["tokenizer"].update(unexpected=True),
            "tokenizer has unknown fields",
        ),
        (
            lambda data: data["provenance"].update(unexpected=True),
            "provenance has unknown fields",
        ),
        (
            lambda data: data["sources"][0].update(unexpected=True),
            r"sources\[0\] has unknown fields",
        ),
        (lambda data: data.update(version=True), "version.*integer"),
        (
            lambda data: data["provenance"].update(selection_seed=True),
            "selection_seed.*integer",
        ),
        (
            lambda data: data["sources"][0].update(parent_path_index=True),
            "parent_path_index.*integer",
        ),
        (
            lambda data: data["provenance"].update(builder_sha256=7),
            "builder_sha256.*non-empty string",
        ),
        (
            lambda data: data["provenance"].update(builder_sha256="D" * 64),
            "builder_sha256.*SHA-256",
        ),
    ],
)
def test_manifest_schema_rejects_unknown_fields_bool_int_and_noncanonical_hashes(
    replay_files, tmp_path, mutate, match
):
    _, data, _ = replay_files
    mutate(data)
    manifest_path = _write_manifest(tmp_path / "invalid-schema.json", data)
    with pytest.raises(OLMoConfigurationError, match=match):
        NativeTextReplayManifest.load(manifest_path)


def test_source_size_hash_and_runtime_tokenizer_validation(replay_files, tmp_path):
    manifest_path, data, tokens = replay_files
    dataset = NativeTextReplayDataset(manifest_path, verify_source_hashes=True)
    dataset.validate_tokenizer(SimpleNamespace(eos_token_id=100_257, pad_token_id=100_277))
    with pytest.raises(OLMoConfigurationError, match="Runtime tokenizer"):
        dataset.validate_tokenizer(SimpleNamespace(eos_token_id=1, pad_token_id=100_277))

    wrong_hash_data = deepcopy(data)
    wrong_hash_data["sources"][0]["sha256"] = "0" * 64
    wrong_hash_path = _write_manifest(tmp_path / "wrong-hash.json", wrong_hash_data)
    # The declared content hash always participates in the fingerprint; full-byte
    # verification is separately opt-in for large production corpora.
    NativeTextReplayDataset(wrong_hash_path, verify_source_hashes=False)
    with pytest.raises(OLMoConfigurationError, match="SHA-256"):
        NativeTextReplayDataset(wrong_hash_path, verify_source_hashes=True)

    too_large = _write_raw_tokens(tmp_path / "too-large.npy", [*tokens, 1])
    wrong_size_data = _manifest_data(tmp_path / "too-large.npy", too_large)
    wrong_size_data["sources"][0]["num_tokens"] = len(tokens)
    wrong_size_data["sources"][0]["size_bytes"] = tokens.nbytes
    wrong_size_path = _write_manifest(tmp_path / "wrong-size.json", wrong_size_data)
    with pytest.raises(OLMoConfigurationError, match="has 36 bytes, expected 32"):
        NativeTextReplayDataset(wrong_size_path)


def test_pinned_offline_verification_receipt_avoids_runtime_rehash(replay_files, tmp_path):
    _, data, _ = replay_files
    token_path = tmp_path / data["sources"][0]["path"]
    receipt_path, receipt_sha256 = _write_verification_receipt(
        tmp_path / "receipt.json", data, token_path
    )
    manifest_path = _write_manifest(tmp_path / "with-receipt.json", data)

    dataset = NativeTextReplayDatasetConfig(
        manifest_path=str(manifest_path),
        verification_receipt_path=str(receipt_path),
        expected_verification_receipt_sha256=receipt_sha256,
        verify_source_hashes=False,
    ).build()

    assert dataset.verification_receipt is not None
    assert dataset.verification_receipt.receipt_sha256 == receipt_sha256
    with pytest.raises(OLMoConfigurationError, match="receipt SHA-256"):
        NativeTextReplayVerificationReceipt.load(
            receipt_path,
            expected_sha256="0" * 64,
        )

    changed = deepcopy(data)
    changed["sources"][0]["source"] = "code"
    changed["provenance"]["source_usable_tokens"] = {"code": changed["provenance"]["usable_tokens"]}
    changed_path = _write_manifest(tmp_path / "changed-source.json", changed)
    with pytest.raises(OLMoConfigurationError, match="differs from the pinned"):
        NativeTextReplayDataset(
            changed_path,
            verification_receipt_path=receipt_path,
            expected_verification_receipt_sha256=receipt_sha256,
        )

    with pytest.raises(OLMoConfigurationError, match="provided together"):
        NativeTextReplayDataset(
            manifest_path,
            verification_receipt_path=receipt_path,
        )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda receipt: receipt.update(unexpected=True), "unknown fields"),
        (lambda receipt: receipt.update(version=True), "version.*integer"),
        (
            lambda receipt: receipt.update(builder_implementation="some/other/builder.py"),
            "unreviewed builder",
        ),
        (
            lambda receipt: receipt.update(parent_mix_sha256=True),
            "parent_mix_sha256.*non-empty string",
        ),
        (
            lambda receipt: receipt["sources"][0].update(num_tokens=True),
            "num_tokens.*integer",
        ),
        (
            lambda receipt: receipt["sources"][0].update(unexpected=True),
            "source 0 has unknown fields",
        ),
    ],
)
def test_verification_receipt_rejects_unknown_fields_and_type_confusion(
    replay_files, tmp_path, mutate, match
):
    _, data, _ = replay_files
    token_path = tmp_path / data["sources"][0]["path"]
    receipt_path, _ = _write_verification_receipt(tmp_path / "receipt.json", data, token_path)
    receipt = json.loads(receipt_path.read_text())
    mutate(receipt)
    receipt_sha256 = _rewrite_json_with_sha256(receipt_path, receipt)

    with pytest.raises(OLMoConfigurationError, match=match):
        NativeTextReplayVerificationReceipt.load(
            receipt_path,
            expected_sha256=receipt_sha256,
        )


@pytest.mark.parametrize(
    ("field_name", "match"),
    [
        ("builder_sha256", "differs from the reviewed implementation"),
        ("parent_mix_sha256", "parent_mix_sha256 does not match"),
        ("upstream_provenance_sha256", "upstream_provenance_sha256 does not match"),
    ],
)
def test_verification_receipt_cross_binds_builder_and_parent_lineage(
    replay_files, tmp_path, field_name, match
):
    _, data, _ = replay_files
    token_path = tmp_path / data["sources"][0]["path"]
    receipt_path, _ = _write_verification_receipt(tmp_path / "receipt.json", data, token_path)
    receipt = json.loads(receipt_path.read_text())
    receipt[field_name] = "0" * 64
    receipt_sha256 = _rewrite_json_with_sha256(receipt_path, receipt)
    data["provenance"]["verification_receipt_sha256"] = receipt_sha256
    manifest_path = _write_manifest(tmp_path / "manifest-with-tampered-receipt.json", data)

    with pytest.raises(OLMoConfigurationError, match=match):
        NativeTextReplayDataset(
            manifest_path,
            verification_receipt_path=receipt_path,
            expected_verification_receipt_sha256=receipt_sha256,
        )


def test_loaded_windows_reject_ids_outside_native_vocabulary(tmp_path: Path):
    tokens = _write_raw_tokens(tmp_path / "tokens.npy", [1, 2, 100_300, 4])
    data = _manifest_data(tmp_path / "tokens.npy", tokens)
    data["num_windows"] = 1
    data["sources"][0]["window_starts"] = [0]
    data["provenance"]["usable_tokens"] = 2
    data["provenance"]["source_usable_tokens"] = {"web": 2}
    manifest_path = _write_manifest(tmp_path / "manifest.json", data)

    with pytest.raises(RuntimeError, match="outside native vocabulary"):
        _ = NativeTextReplayDataset(manifest_path)[0]
    np.testing.assert_array_equal(
        NativeTextReplayDataset(manifest_path, validate_token_ids=False)[0]["labels"],
        np.array([tokens[1], tokens[2], -100], dtype=np.int64),
    )


def test_v3_compact_runtime_cross_binds_shared_receipt_and_parent_provenance(tmp_path: Path):
    manifest_paths, _, receipt_path, _, receipt_sha256, _ = _write_v3_artifacts(tmp_path)

    assert NATIVE_TEXT_REPLAY_VERSION == 3
    assert NATIVE_TEXT_REPLAY_VERIFICATION_VERSION == 3
    with pytest.raises(OLMoConfigurationError, match="v3 requires a pinned verification receipt"):
        NativeTextReplayDataset(
            manifest_paths["train"],
            expected_fingerprint=_v3_fingerprint(manifest_paths["train"]),
        )
    with pytest.raises(OLMoConfigurationError, match="explicit expected_fingerprint"):
        NativeTextReplayDataset(
            manifest_paths["train"],
            verification_receipt_path=receipt_path,
            expected_verification_receipt_sha256=receipt_sha256,
        )

    dataset = NativeTextReplayDataset(
        manifest_paths["train"],
        expected_fingerprint=_v3_fingerprint(manifest_paths["train"]),
        verification_receipt_path=receipt_path,
        expected_verification_receipt_sha256=receipt_sha256,
    )
    receipt = dataset.verification_receipt
    assert receipt is not None
    assert receipt.version == 3
    assert receipt.mirror_policy == "s3-to-gs-same-bucket-key-v1"
    assert len(receipt.remote_sources) == 1
    assert set(receipt.split_sources) == {"train", "holdout"}
    assert dataset.fingerprint_version == "native-text-replay-v3"
    np.testing.assert_array_equal(dataset[1]["input_ids"], [4, 5, 6])
    assert dataset.provenance_for(1) == {
        "dataset_fingerprint": dataset.fingerprint,
        "manifest_index": 1,
        "source_id": "web-000000-train",
        "source": "web",
        "path": "train-tokens.npy",
        "source_sha256": dataset.manifest.sources[0].sha256,
        "start": 3,
        "stop": 6,
        "parent_path_index": 0,
        "parent_path": "s3://ai2-llm/tokens/web-000000.npy",
        "parent_start": 6,
        "parent_stop": 9,
        "parent_num_tokens": 12,
        "compact_path": "train-tokens.npy",
        "compact_start": 3,
        "compact_stop": 6,
    }

    holdout_manifest = NativeTextReplayManifest.load(manifest_paths["holdout"])
    receipt.validate_pair(dataset.manifest, holdout_manifest)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda data: data.update(dtype="uint16"), "v3 dtype must be 'uint32'"),
        (
            lambda data: data["sources"][0].update(window_starts=[0, 6]),
            "local window_starts",
        ),
        (
            lambda data: data["sources"][0].update(
                num_tokens=9, size_bytes=36, window_starts=[0, 3]
            ),
            "num_tokens must equal",
        ),
        (
            lambda data: data["sources"][0].update(parent_window_starts=[0]),
            "local and parent window counts",
        ),
        (
            lambda data: data["sources"][0].update(parent_window_starts=[0, 4]),
            "not aligned",
        ),
        (
            lambda data: data["sources"][0].update(parent_window_starts=[0, 0]),
            "ordered and non-overlapping",
        ),
        (
            lambda data: data["sources"][0].update(parent_window_starts=[0, 12]),
            "out of bounds",
        ),
        (
            lambda data: data["sources"][0].update(parent_window_starts=[0, True]),
            r"parent_window_starts\[1\].*integer",
        ),
        (
            lambda data: data["sources"][0].update(path="../outside.npy"),
            "normalized relative POSIX path",
        ),
        (
            lambda data: data["sources"][0].update(path="/tmp/outside.npy"),
            "normalized relative POSIX path",
        ),
    ],
)
def test_v3_manifest_rejects_compact_and_parent_window_attacks(tmp_path, mutate, match):
    _, manifests, _, _, _, _ = _write_v3_artifacts(tmp_path)
    train = deepcopy(manifests["train"])
    mutate(train)
    path = _write_manifest(tmp_path / "attacked-v3.json", train)
    with pytest.raises(OLMoConfigurationError, match=match):
        NativeTextReplayManifest.load(path)


def test_v3_manifest_rejects_compact_symlink(tmp_path: Path):
    _, manifests, _, _, _, compact_paths = _write_v3_artifacts(tmp_path)
    symlink = tmp_path / "compact-link.npy"
    symlink.symlink_to(compact_paths["train"])
    train = deepcopy(manifests["train"])
    train["sources"][0]["path"] = symlink.name

    with pytest.raises(OLMoConfigurationError, match="must not traverse symbolic links"):
        NativeTextReplayManifest.load(_write_manifest(tmp_path / "symlink.json", train))


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda receipt: receipt.update(unexpected=True), "unknown fields"),
        (
            lambda receipt: receipt.update(mirror_policy="some-other-policy"),
            "mirror_policy",
        ),
        (
            lambda receipt: receipt["remote_sources"][0].update(
                mirror_uri="gs://other-bucket/tokens.npy"
            ),
            "wrong mirror URI",
        ),
        (
            lambda receipt: receipt["remote_sources"][0].update(generation="12x"),
            "generation.*digits",
        ),
        (
            lambda receipt: receipt["remote_sources"][0].update(crc32c="AAAA"),
            "encode exactly 4 bytes",
        ),
        (
            lambda receipt: receipt["remote_sources"][0].update(md5_hash=True),
            "md5_hash.*non-empty string",
        ),
        (
            lambda receipt: receipt["remote_sources"][0].update(
                md5_hash="AAAAAAAAAAAAAAAAAAAAAA=="
            ),
            "remote_snapshot_sha256",
        ),
        (
            lambda receipt: receipt["remote_sources"][0].update(source_etag=True),
            "source_etag.*non-empty string",
        ),
        (
            lambda receipt: receipt["remote_sources"][0].update(parent_path_index=True),
            "parent_path_index.*integer",
        ),
        (lambda receipt: receipt["splits"].pop("holdout"), "missing required fields"),
        (lambda receipt: receipt["splits"].update(holdout=[]), "holdout.*non-empty"),
        (
            lambda receipt: receipt["manifest_contract_sha256"].pop("holdout"),
            "manifest_contract_sha256.*missing required fields",
        ),
        (
            lambda receipt: receipt["manifest_contract_sha256"].update(train=True),
            "manifest_contract_sha256.train.*non-empty string",
        ),
        (
            lambda receipt: receipt["splits"]["train"][0].update(num_windows=True),
            "num_windows.*integer",
        ),
        (
            lambda receipt: receipt["splits"]["train"][0].update(mtime_ns=True),
            "mtime_ns.*integer",
        ),
        (
            lambda receipt: receipt["splits"]["train"][0].update(
                resolved_path="/tmp/../tmp/not-normalized.npy"
            ),
            "resolved paths must be normalized",
        ),
        (
            lambda receipt: receipt["splits"]["train"][0].update(parent_num_tokens=11),
            "parent token count differs",
        ),
    ],
)
def test_v3_receipt_rejects_remote_and_materialization_attacks(tmp_path, mutate, match):
    _, _, receipt_path, receipt, _, _ = _write_v3_artifacts(tmp_path)
    attacked = deepcopy(receipt)
    mutate(attacked)
    attacked_sha256 = _rewrite_json_with_sha256(receipt_path, attacked)

    with pytest.raises(OLMoConfigurationError, match=match):
        NativeTextReplayVerificationReceipt.load(
            receipt_path,
            expected_sha256=attacked_sha256,
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda manifest: manifest["tokenizer"].update(identifier="attacker/tokenizer"),
        lambda manifest: manifest["provenance"].update(parent_checkpoint="/checkpoints/attacker"),
        lambda manifest: manifest["provenance"].update(parent_mix="attacker-mix"),
        lambda manifest: manifest["provenance"].update(selection_seed=18),
    ],
)
def test_v3_manifest_contract_binds_fields_outside_receipt_lineage(tmp_path, mutate):
    manifest_paths, manifests, receipt_path, _, receipt_sha256, _ = _write_v3_artifacts(tmp_path)
    attacked = deepcopy(manifests["train"])
    mutate(attacked)
    attacked_path = _write_manifest(manifest_paths["train"], attacked, indent=2)

    with pytest.raises(OLMoConfigurationError, match="manifest contract"):
        NativeTextReplayDataset(
            attacked_path,
            expected_fingerprint=_v3_fingerprint(attacked_path),
            verification_receipt_path=receipt_path,
            expected_verification_receipt_sha256=receipt_sha256,
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "parent_config_sha256",
        "parent_trainer_state_sha256",
        "parent_dataset_fingerprint",
        "remote_snapshot_sha256",
        "compact_materialization_sha256",
    ],
)
def test_v3_receipt_cross_binds_all_parent_and_materialization_lineage(tmp_path, field_name):
    manifest_paths, _, receipt_path, receipt, _, _ = _write_v3_artifacts(tmp_path)
    attacked = deepcopy(receipt)
    attacked[field_name] = "f" * 64
    attacked_sha256 = _rewrite_json_with_sha256(receipt_path, attacked)

    with pytest.raises(OLMoConfigurationError, match=field_name):
        NativeTextReplayDataset(
            manifest_paths["train"],
            expected_fingerprint=_v3_fingerprint(manifest_paths["train"]),
            verification_receipt_path=receipt_path,
            expected_verification_receipt_sha256=attacked_sha256,
        )


def test_v3_receipt_cross_binds_exact_source_and_parent_starts_digest(tmp_path: Path):
    manifest_paths, manifests, receipt_path, receipt, _, _ = _write_v3_artifacts(tmp_path)
    attacked = deepcopy(receipt)
    attacked["splits"]["train"][0]["parent_window_starts_sha256"] = "f" * 64
    attacked_manifest = deepcopy(manifests["train"])
    attacked_manifests = {**manifests, "train": attacked_manifest}
    attacked_manifest_paths = {
        **manifest_paths,
        "train": tmp_path / "attacked-parent-starts-manifest.json",
    }
    attacked_sha256 = _republish_v3_artifacts(
        attacked_manifest_paths,
        attacked_manifests,
        receipt_path,
        attacked,
    )
    attacked_manifest_path = attacked_manifest_paths["train"]

    with pytest.raises(OLMoConfigurationError, match="differs from the pinned"):
        NativeTextReplayDataset(
            attacked_manifest_path,
            expected_fingerprint=_v3_fingerprint(attacked_manifest_path),
            verification_receipt_path=receipt_path,
            expected_verification_receipt_sha256=attacked_sha256,
        )


def test_v3_pair_validator_rejects_parent_interval_overlap(tmp_path: Path):
    manifest_paths, _, receipt_path, _, receipt_sha256, _ = _write_v3_artifacts(
        tmp_path,
        holdout_parent_starts=(0, 9),
    )
    receipt = NativeTextReplayVerificationReceipt.load(
        receipt_path,
        expected_sha256=receipt_sha256,
    )

    with pytest.raises(OLMoConfigurationError, match="overlap in parent path"):
        receipt.validate_pair(
            NativeTextReplayManifest.load(manifest_paths["train"]),
            NativeTextReplayManifest.load(manifest_paths["holdout"]),
        )


def test_v3_runtime_rejects_same_size_compact_edit_from_stat_signature(tmp_path: Path):
    manifest_paths, _, receipt_path, _, receipt_sha256, compact_paths = _write_v3_artifacts(
        tmp_path
    )
    compact_path = compact_paths["train"]
    original = np.fromfile(compact_path, dtype=np.uint32)
    original[0] = 42
    original.tofile(compact_path)

    with pytest.raises(OLMoConfigurationError, match="stat signature"):
        NativeTextReplayDataset(
            manifest_paths["train"],
            expected_fingerprint=_v3_fingerprint(manifest_paths["train"]),
            verification_receipt_path=receipt_path,
            expected_verification_receipt_sha256=receipt_sha256,
        )


def test_v3_offline_verify_hashes_full_compact_file(tmp_path: Path):
    manifest_paths, manifests, receipt_path, receipt, _, compact_paths = _write_v3_artifacts(
        tmp_path
    )
    compact_path = compact_paths["train"]
    original = np.fromfile(compact_path, dtype=np.uint32)
    original[0] = 42
    original.tofile(compact_path)
    compact_stat = compact_path.stat()
    receipt["splits"]["train"][0].update(
        mtime_ns=compact_stat.st_mtime_ns,
        ctime_ns=compact_stat.st_ctime_ns,
        inode=compact_stat.st_ino,
        device=compact_stat.st_dev,
    )
    receipt_sha256 = _republish_v3_artifacts(
        manifest_paths,
        manifests,
        receipt_path,
        receipt,
    )

    NativeTextReplayDataset(
        manifest_paths["train"],
        expected_fingerprint=_v3_fingerprint(manifest_paths["train"]),
        verification_receipt_path=receipt_path,
        expected_verification_receipt_sha256=receipt_sha256,
        verify_source_hashes=False,
    )
    with pytest.raises(OLMoConfigurationError, match="has SHA-256"):
        NativeTextReplayDataset(
            manifest_paths["train"],
            expected_fingerprint=_v3_fingerprint(manifest_paths["train"]),
            verification_receipt_path=receipt_path,
            expected_verification_receipt_sha256=receipt_sha256,
            verify_source_hashes=True,
        )


def test_v3_reopen_rejects_same_size_path_replacement(tmp_path: Path):
    manifest_paths, _, receipt_path, _, receipt_sha256, compact_paths = _write_v3_artifacts(
        tmp_path
    )
    dataset = NativeTextReplayDataset(
        manifest_paths["train"],
        expected_fingerprint=_v3_fingerprint(manifest_paths["train"]),
        verification_receipt_path=receipt_path,
        expected_verification_receipt_sha256=receipt_sha256,
    )
    replacement = tmp_path / "replacement.npy"
    replacement.write_bytes(compact_paths["train"].read_bytes())
    os.replace(replacement, compact_paths["train"])

    with pytest.raises(RuntimeError, match="changed stat signature before read"):
        _ = dataset[0]


def test_v3_runtime_rejects_non_regular_compact_source(tmp_path: Path):
    manifest_paths, _, receipt_path, _, receipt_sha256, compact_paths = _write_v3_artifacts(
        tmp_path
    )
    compact_path = compact_paths["train"]
    compact_path.unlink()
    compact_path.mkdir()

    with pytest.raises(OLMoConfigurationError, match="not a regular file"):
        NativeTextReplayDataset(
            manifest_paths["train"],
            expected_fingerprint=_v3_fingerprint(manifest_paths["train"]),
            verification_receipt_path=receipt_path,
            expected_verification_receipt_sha256=receipt_sha256,
        )


def test_v3_runtime_rejects_fifo_without_blocking(tmp_path: Path):
    manifest_paths, _, receipt_path, _, receipt_sha256, compact_paths = _write_v3_artifacts(
        tmp_path
    )
    compact_paths["train"].unlink()
    os.mkfifo(compact_paths["train"])

    with pytest.raises(OLMoConfigurationError, match="not a regular file"):
        NativeTextReplayDataset(
            manifest_paths["train"],
            expected_fingerprint=_v3_fingerprint(manifest_paths["train"]),
            verification_receipt_path=receipt_path,
            expected_verification_receipt_sha256=receipt_sha256,
        )
