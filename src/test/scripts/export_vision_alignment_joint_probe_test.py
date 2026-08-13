from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.data import export_vision_alignment_joint_probe as exporter


def _canonical_sha256(value) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _example(length: int, *, row: int, epoch: int, zero_loss: bool = False):
    input_ids = np.arange(length, dtype=np.int64) + row * 100 + epoch
    return {
        "input_ids": input_ids,
        "labels": np.array(input_ids, copy=True),
        "loss_masks": (
            np.zeros(length, dtype=np.float32) if zero_loss else np.ones(length, dtype=np.float32)
        ),
        "position_ids": np.arange(length, dtype=np.int64),
        "token_type_ids": np.zeros(length, dtype=np.int64),
        "images": np.zeros((1, 2, 3), dtype=np.float32),
        "pooled_patches_idx": np.zeros((1, 2), dtype=np.int64),
    }


class _RawDataset:
    def __init__(self, length: int, *, rows: int = 6):
        self.length = length
        self.rows = rows

    def __len__(self):
        return self.rows

    def get(self, index: int, epoch: int):
        return _example(self.length, row=index, epoch=epoch)


class _SelectedDataset:
    def __init__(self, raw: _RawDataset, *, source_name: str):
        self.raw = raw
        self.indices = tuple(reversed(range(len(raw))))
        self.content_fingerprint = hashlib.sha256(source_name.encode()).hexdigest()
        self.annotation_validations = 0

    def __len__(self):
        return len(self.indices)

    def get(self, index: int, epoch: int):
        return self.raw.get(self.indices[index], epoch)

    def validate_required_annotations(self):
        self.annotation_validations += 1

    def validate_image_content(self, indices):
        return _canonical_sha256(
            [
                {
                    "index": index,
                    "image_sha256": hashlib.sha256(
                        f"image-{self.indices[index]}".encode()
                    ).hexdigest(),
                }
                for index in indices
            ]
        )


class _NativeDataset:
    def __init__(self, length: int, *, rows: int = 6):
        self.length = length
        self.rows = rows
        self.content_fingerprint = "9" * 64

    def __len__(self):
        return self.rows

    def get(self, index: int, epoch: int):
        example = _example(self.length, row=index, epoch=epoch)
        example["images"] = np.zeros((0, 2, 3), dtype=np.float32)
        example["pooled_patches_idx"] = np.zeros((0, 2), dtype=np.int64)
        return example


def test_visual_export_proves_unbounded_parity_and_rehashes_images(tmp_path, monkeypatch):
    monkeypatch.setattr(exporter, "JOINT_SEQUENCE_LENGTH", 8)
    raw = _RawDataset(8)
    selected = _SelectedDataset(raw, source_name="pixmo_caption")
    output = tmp_path / "pixmo_caption.jsonl"

    entry = exporter.export_source_probe(
        selected,
        source_name="pixmo_caption",
        kind="visual",
        output_path=output,
        unique_indices=2,
        epochs=(0, 1),
        seed=13,
        unbounded_dataset=raw,
    )

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(records) == 4
    assert [(row["probe_epoch"], row["probe_index"]) for row in records] == [
        (epoch, index) for epoch in (0, 1) for index in entry["probe_indices"]
    ]
    assert all(row["raw_sequence_length"] == 8 for row in records)
    assert all(row["truncated"] is False for row in records)
    assert entry["probe_epochs"] == [0, 1]
    assert entry["truncated_rows"] == 0
    assert entry["max_observed_sequence_length"] == 8
    assert entry["probe_image_content_sha256"] == selected.validate_image_content(
        entry["probe_indices"]
    )
    assert selected.annotation_validations == 1


def test_visual_export_rejects_any_raw_row_above_joint_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(exporter, "JOINT_SEQUENCE_LENGTH", 8)
    raw = _RawDataset(9)
    selected = _SelectedDataset(raw, source_name="pixmo_caption")

    with pytest.raises(ValueError, match="truncation is forbidden"):
        exporter.export_source_probe(
            selected,
            source_name="pixmo_caption",
            kind="visual",
            output_path=tmp_path / "pixmo_caption.jsonl",
            unique_indices=1,
            epochs=(0,),
            seed=13,
            unbounded_dataset=raw,
        )
    assert not (tmp_path / "pixmo_caption.jsonl").exists()


def test_visual_export_rejects_selected_unbounded_serialization_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(exporter, "JOINT_SEQUENCE_LENGTH", 8)
    raw = _RawDataset(8)
    selected = _SelectedDataset(raw, source_name="pixmo_caption")
    original_get = selected.get

    def drifted(index, epoch):
        example = original_get(index, epoch)
        example["input_ids"] = np.array(example["input_ids"], copy=True)
        example["input_ids"][0] += 1
        return example

    selected.get = drifted
    with pytest.raises(ValueError, match="unbounded serialization"):
        exporter.export_source_probe(
            selected,
            source_name="pixmo_caption",
            kind="visual",
            output_path=tmp_path / "pixmo_caption.jsonl",
            unique_indices=1,
            epochs=(0,),
            seed=13,
            unbounded_dataset=raw,
        )


def test_native_export_requires_exact_sequence_length_and_empty_image_evidence(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(exporter, "JOINT_SEQUENCE_LENGTH", 8)
    native = _NativeDataset(8)
    output = tmp_path / "native_text_replay.jsonl"
    entry = exporter.export_source_probe(
        native,
        source_name="native_text_replay",
        kind="native_text_replay",
        output_path=output,
        unique_indices=2,
        epochs=(0,),
        seed=13,
    )
    assert entry["probe_image_content_sha256"] == _canonical_sha256([])
    assert entry["max_observed_sequence_length"] == 8
    assert len(output.read_text().splitlines()) == 2

    with pytest.raises(ValueError, match="exactly 8 tokens"):
        exporter.export_source_probe(
            _NativeDataset(7),
            source_name="native_text_replay",
            kind="native_text_replay",
            output_path=tmp_path / "short.jsonl",
            unique_indices=1,
            epochs=(0,),
            seed=13,
        )


def test_catalog_binds_all_provenance_and_has_canonical_content_sha(tmp_path, monkeypatch):
    monkeypatch.setattr(exporter, "JOINT_VISUAL_PROBE_INDICES", 2)
    monkeypatch.setattr(exporter, "JOINT_VISUAL_PROBE_EPOCHS", (0, 1))
    monkeypatch.setattr(exporter, "JOINT_NATIVE_PROBE_INDICES", 4)
    monkeypatch.setattr(exporter, "JOINT_NATIVE_PROBE_EPOCHS", (0,))
    projection_path = tmp_path / "projection.json"
    projection_path.write_text("projection")
    native_path = tmp_path / "native.json"
    native_path.write_text("native")
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text("receipt")
    spec = SimpleNamespace(as_canonical_dict=lambda: {"phase": "joint", "sequence_length": 8192})
    projection = SimpleNamespace(
        path=projection_path.resolve(),
        raw_sha256=hashlib.sha256(projection_path.read_bytes()).hexdigest(),
        content_sha256="1" * 64,
        source_spec=spec,
    )
    native = SimpleNamespace(
        path=native_path.resolve(),
        manifest_sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(),
        content_fingerprint="2" * 64,
    )
    entries = []
    for name in exporter.JOINT_SOURCE_NAMES:
        kind = "native_text_replay" if name == "native_text_replay" else "visual"
        count = 4 if kind == "native_text_replay" else 2
        epochs = [0] if kind == "native_text_replay" else [0, 1]
        fingerprint = hashlib.sha256(name.encode()).hexdigest()
        indices = list(
            exporter.select_deterministic_probe_indices(
                6,
                count,
                seed=exporter.JOINT_PROBE_SEED,
                dataset_fingerprint=fingerprint,
            )
        )
        entries.append(
            {
                "name": name,
                "kind": kind,
                "format": "jsonl",
                "path": f"{name}.jsonl",
                "dataset_fingerprint": fingerprint,
                "dataset_size": 6,
                "sha256": "4" * 64,
                "probe_epochs": epochs,
                "probe_indices": indices,
                "probe_indices_sha256": _canonical_sha256(indices),
                "serialized_row_hashes_sha256": "5" * 64,
                "probe_image_content_sha256": "6" * 64,
                "max_observed_sequence_length": (8192 if kind == "native_text_replay" else 8),
                "truncated_rows": 0,
            }
        )
    catalog = exporter.build_probe_catalog(
        projection=projection,
        native_manifest=native,
        verification_receipt_path=receipt_path.resolve(),
        verification_receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        source_entries=entries,
        exporter_sha256="3" * 64,
        probe_seed=exporter.JOINT_PROBE_SEED,
    )

    unsigned = dict(catalog)
    content_sha256 = unsigned.pop("content_sha256")
    assert content_sha256 == _canonical_sha256(unsigned)
    assert catalog["phase"] == "joint"
    assert catalog["visual_projection"]["raw_sha256"] == projection.raw_sha256
    assert catalog["native_train_manifest"] == {
        "path": str(native.path),
        "raw_sha256": native.manifest_sha256,
        "content_fingerprint": native.content_fingerprint,
    }
    assert catalog["native_verification_receipt"]["path"] == str(receipt_path.resolve())
    assert catalog["preprocessing"] == {
        "visual": spec.as_canonical_dict(),
        "native_text_replay_fingerprint": native.content_fingerprint,
    }
    assert catalog["probe"]["visual"]["rows_per_source"] == 4
    assert catalog["probe"]["native_text_replay"]["rows_per_source"] == 4


def test_catalog_rejects_noncanonical_or_incomplete_source_set(tmp_path):
    projection = SimpleNamespace(
        path=tmp_path / "projection",
        raw_sha256="1" * 64,
        content_sha256="2" * 64,
        source_spec=SimpleNamespace(as_canonical_dict=lambda: {}),
    )
    native = SimpleNamespace(
        path=tmp_path / "native",
        manifest_sha256="3" * 64,
        content_fingerprint="4" * 64,
    )
    with pytest.raises(ValueError, match="exact canonical nine-source ordering"):
        exporter.build_probe_catalog(
            projection=projection,
            native_manifest=native,
            verification_receipt_path=tmp_path / "receipt",
            verification_receipt_sha256="5" * 64,
            source_entries=[{"name": "native_text_replay"}],
            exporter_sha256="6" * 64,
            probe_seed=exporter.JOINT_PROBE_SEED,
        )


def test_production_probe_panel_is_exactly_1024_rows_per_source():
    assert exporter.JOINT_VISUAL_PROBE_INDICES == 256
    assert exporter.JOINT_VISUAL_PROBE_EPOCHS == (0, 1, 2, 3)
    assert exporter.JOINT_VISUAL_PROBE_INDICES * len(exporter.JOINT_VISUAL_PROBE_EPOCHS) == 1024
    assert exporter.JOINT_NATIVE_PROBE_INDICES == 1024
    assert exporter.JOINT_NATIVE_PROBE_EPOCHS == (0,)
