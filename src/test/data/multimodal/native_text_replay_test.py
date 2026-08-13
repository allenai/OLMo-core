import hashlib
import json
import pickle
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from olmo_core.data.multimodal.collator import MultimodalCollator
from olmo_core.data.multimodal.native_text_replay import (
    NATIVE_TEXT_REPLAY_BUILDER_IMPLEMENTATION_REFERENCE,
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
        "version": NATIVE_TEXT_REPLAY_VERIFICATION_VERSION,
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
