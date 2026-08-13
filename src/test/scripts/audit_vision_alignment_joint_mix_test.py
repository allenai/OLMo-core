from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.data import audit_vision_alignment_joint_mix as auditor
from scripts.data import export_vision_alignment_joint_probe as exporter


def _canonical_bytes(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


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


class _RawVisualDataset:
    def __init__(self, length: int = 8, *, rows: int = 4, zero_loss: bool = False):
        self.length = length
        self.rows = rows
        self.zero_loss = zero_loss

    def __len__(self):
        return self.rows

    def get(self, index: int, epoch: int):
        return _example(self.length, row=index, epoch=epoch, zero_loss=self.zero_loss)


class _SelectedVisualDataset:
    def __init__(self, raw: _RawVisualDataset, *, source_name: str):
        self.raw = raw
        self.indices = tuple(reversed(range(len(raw))))
        self.content_fingerprint = hashlib.sha256(f"selected-{source_name}".encode()).hexdigest()
        self.image_salt = ""

    def __len__(self):
        return len(self.indices)

    def get(self, index: int, epoch: int):
        return self.raw.get(self.indices[index], epoch)

    def validate_required_annotations(self):
        return None

    def validate_image_content(self, indices):
        return _canonical_sha256(
            [
                {
                    "index": index,
                    "image_sha256": hashlib.sha256(
                        f"image-{self.indices[index]}-{self.image_salt}".encode()
                    ).hexdigest(),
                }
                for index in indices
            ]
        )


class _NativeDataset:
    def __init__(self, *, length: int = 8, rows: int = 4, zero_loss: bool = False):
        self.length = length
        self.rows = rows
        self.zero_loss = zero_loss
        self.drift = False
        self.content_fingerprint = "a" * 64

    def __len__(self):
        return self.rows

    def get(self, index: int, epoch: int):
        example = _example(self.length, row=index, epoch=epoch, zero_loss=self.zero_loss)
        example["images"] = np.zeros((0, 2, 3), dtype=np.float32)
        example["pooled_patches_idx"] = np.zeros((0, 2), dtype=np.int64)
        if self.drift:
            example["input_ids"] = np.array(example["input_ids"], copy=True)
            example["input_ids"][0] += 1
        return example


class _SourceSpec:
    def as_canonical_dict(self):
        return {
            "phase": "joint",
            "sequence_length": 8192,
            "parent_perception_preprocessing_sha256": "b" * 64,
        }


@dataclass
class _Receipt:
    path: Path
    receipt_sha256: str

    def validate_manifest(self, manifest):
        if manifest.provenance["verification_receipt_sha256"] != self.receipt_sha256:
            raise ValueError("receipt mismatch")


@dataclass
class _Fixture:
    catalog_path: Path
    catalog_sha256: str
    projection: object
    native_manifest: object
    receipt: _Receipt
    runtimes: dict
    visual_selected: dict
    visual_raw: dict
    native_dataset: _NativeDataset


def _patch_small_probe(monkeypatch):
    for module in (exporter, auditor):
        monkeypatch.setattr(module, "JOINT_SEQUENCE_LENGTH", 8)
        monkeypatch.setattr(module, "JOINT_VISUAL_PROBE_INDICES", 2)
        monkeypatch.setattr(module, "JOINT_VISUAL_PROBE_EPOCHS", (0, 1))
        monkeypatch.setattr(module, "JOINT_NATIVE_PROBE_INDICES", 2)
        monkeypatch.setattr(module, "JOINT_NATIVE_PROBE_EPOCHS", (0,))


def _build_fixture(tmp_path, monkeypatch, *, zero_source=None) -> _Fixture:
    _patch_small_probe(monkeypatch)
    projection_path = (tmp_path / "vision-alignment-joint-visual-projection.json").resolve()
    projection_path.write_bytes(b"synthetic joint projection\n")
    receipt_path = (tmp_path / "native-replay-verification.json").resolve()
    receipt_path.write_bytes(b"synthetic native receipt\n")
    receipt_sha256 = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    native_path = (tmp_path / "native-train.json").resolve()
    native_path.write_bytes(b"synthetic native manifest\n")

    projection = SimpleNamespace(
        path=projection_path,
        raw_sha256=hashlib.sha256(projection_path.read_bytes()).hexdigest(),
        content_sha256="c" * 64,
        source_spec=_SourceSpec(),
    )
    native_manifest = SimpleNamespace(
        path=native_path,
        manifest_sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(),
        content_fingerprint="a" * 64,
        sequence_length=8,
        provenance={"split": "train", "verification_receipt_sha256": receipt_sha256},
    )
    receipt = _Receipt(receipt_path, receipt_sha256)

    visual_selected = {}
    visual_raw = {}
    runtimes = {}
    entries = []
    for source_name in auditor.JOINT_SOURCE_NAMES:
        output_path = tmp_path / f"{source_name}.jsonl"
        if source_name == "native_text_replay":
            native = _NativeDataset(zero_loss=zero_source == source_name)
            entry = exporter.export_source_probe(
                native,
                source_name=source_name,
                kind="native_text_replay",
                output_path=output_path,
                unique_indices=2,
                epochs=(0,),
                seed=exporter.JOINT_PROBE_SEED,
            )
            runtimes[source_name] = auditor._RuntimeProbeSource(native, None)
            native_dataset = native
        else:
            raw = _RawVisualDataset(zero_loss=zero_source == source_name)
            selected = _SelectedVisualDataset(raw, source_name=source_name)
            entry = exporter.export_source_probe(
                selected,
                source_name=source_name,
                kind="visual",
                output_path=output_path,
                unique_indices=2,
                epochs=(0, 1),
                seed=exporter.JOINT_PROBE_SEED,
                unbounded_dataset=raw,
            )
            visual_selected[source_name] = selected
            visual_raw[source_name] = raw
            runtimes[source_name] = auditor._RuntimeProbeSource(selected, raw)
        entries.append(entry)

    exporter_path = Path(exporter.__file__).resolve()
    catalog = exporter.build_probe_catalog(
        projection=projection,
        native_manifest=native_manifest,
        verification_receipt_path=receipt_path,
        verification_receipt_sha256=receipt_sha256,
        source_entries=entries,
        exporter_sha256=hashlib.sha256(exporter_path.read_bytes()).hexdigest(),
        probe_seed=exporter.JOINT_PROBE_SEED,
    )
    catalog_path = (tmp_path / "vision-alignment-joint-source-catalog.json").resolve()
    catalog_path.write_bytes(_canonical_bytes(catalog) + b"\n")
    catalog_sha256 = hashlib.sha256(catalog_path.read_bytes()).hexdigest()

    monkeypatch.setattr(
        auditor,
        "load_joint_visual_projection_manifest",
        lambda path, *, expected_sha256: projection,
    )
    monkeypatch.setattr(auditor, "_load_native_manifest", lambda path: native_manifest)
    monkeypatch.setattr(
        auditor,
        "_load_native_receipt",
        lambda path, *, expected_sha256: receipt,
    )

    def build_runtime_sources(projection_arg, native_arg, receipt_arg, *, hf_cache_dir):
        assert projection_arg is projection
        assert native_arg is native_manifest
        assert receipt_arg is receipt
        assert hf_cache_dir
        return runtimes

    monkeypatch.setattr(auditor, "_build_runtime_sources", build_runtime_sources)
    return _Fixture(
        catalog_path=catalog_path,
        catalog_sha256=catalog_sha256,
        projection=projection,
        native_manifest=native_manifest,
        receipt=receipt,
        runtimes=runtimes,
        visual_selected=visual_selected,
        visual_raw=visual_raw,
        native_dataset=native_dataset,
    )


def test_audit_rebuilds_exact_nine_sources_and_calibrates_joint_targets(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, monkeypatch)
    report = auditor.audit_joint_catalog(
        fixture.catalog_path,
        expected_catalog_sha256=fixture.catalog_sha256,
        hf_cache_dir="unused-synthetic-cache",
    )

    assert report["format"] == "vision_alignment_joint_source_audit"
    assert report["version"] == 1
    assert report["status"] == "ok"
    assert tuple(sorted(report["sources"])) == auditor.JOINT_SOURCE_NAMES
    assert tuple(sorted(report["target_loss_mass"])) == auditor.JOINT_SOURCE_NAMES
    assert tuple(sorted(report["mean_loss_weight"])) == auditor.JOINT_SOURCE_NAMES
    assert tuple(sorted(report["sampling_probabilities"])) == auditor.JOINT_SOURCE_NAMES
    assert sum(report["sampling_probabilities"].values()) == pytest.approx(1.0)
    assert report["expected_loss_mass"] == pytest.approx(report["target_loss_mass"])
    assert all(value["truncated_examples"] == 0 for value in report["sources"].values())
    unsigned = dict(report)
    fingerprint = unsigned.pop("fingerprint")
    assert fingerprint == _canonical_sha256(unsigned)


def test_audit_rejects_probe_file_byte_tamper(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, monkeypatch)
    source_path = tmp_path / "pixmo_caption.jsonl"
    source_path.write_bytes(source_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="probe-file SHA-256 differs"):
        auditor.audit_joint_catalog(fixture.catalog_path)


def test_audit_rejects_live_serialized_row_drift(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, monkeypatch)
    fixture.native_dataset.drift = True

    with pytest.raises(ValueError, match="serialized row drifted"):
        auditor.audit_joint_catalog(fixture.catalog_path)


def test_audit_rejects_live_visual_truncation(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, monkeypatch)
    fixture.visual_raw["pixmo_caption"].length = 9

    with pytest.raises(ValueError, match="truncation is forbidden"):
        auditor.audit_joint_catalog(fixture.catalog_path)


def test_audit_rejects_live_image_byte_drift(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, monkeypatch)
    fixture.visual_selected["pixmo_caption"].image_salt = "drift"

    with pytest.raises(ValueError, match="image-content digest differs"):
        auditor.audit_joint_catalog(fixture.catalog_path)


def test_audit_returns_failed_report_for_zero_supervised_loss(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, monkeypatch, zero_source="native_text_replay")
    report = auditor.audit_joint_catalog(fixture.catalog_path)

    assert report["status"] == "failed"
    assert report["sampling_probabilities"] is None
    assert report["expected_loss_mass"] is None
    assert report["mean_loss_weight"].keys() == set(auditor.JOINT_SOURCE_NAMES) - {
        "native_text_replay"
    }
    assert report["failures"] == ["native_text_replay: has non-positive supervised loss mass"]


def test_audit_strictly_rejects_catalog_schema_extension(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, monkeypatch)
    catalog = json.loads(fixture.catalog_path.read_text())
    catalog["unreviewed"] = True
    fixture.catalog_path.write_bytes(_canonical_bytes(catalog) + b"\n")

    with pytest.raises(ValueError, match="fields differ"):
        auditor.audit_joint_catalog(fixture.catalog_path)


def test_audit_requires_external_catalog_raw_pin(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="external raw SHA-256 pin"):
        auditor.audit_joint_catalog(
            fixture.catalog_path,
            expected_catalog_sha256="0" * 64,
        )


def test_audit_rejects_catalog_content_digest_tamper(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, monkeypatch)
    catalog = json.loads(fixture.catalog_path.read_text())
    catalog["preprocessing"]["native_text_replay_fingerprint"] = "f" * 64
    fixture.catalog_path.write_bytes(_canonical_bytes(catalog) + b"\n")

    with pytest.raises(ValueError, match="content SHA-256 differs"):
        auditor.audit_joint_catalog(fixture.catalog_path)
