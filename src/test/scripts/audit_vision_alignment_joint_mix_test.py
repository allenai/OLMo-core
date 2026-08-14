from __future__ import annotations

import gc
import hashlib
import json
import threading
import time
import weakref
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from olmo_core.data.multimodal.vision_alignment_sources import (
    array_content_descriptor,
    serialized_descriptor_sha256,
)
from olmo_core.nn.vision import Molmo2TokenIds
from scripts.data import audit_vision_alignment_joint_mix as auditor
from scripts.data import export_vision_alignment_joint_probe as exporter

_TEST_TOKEN_IDS = Molmo2TokenIds(
    im_start_id=100278,
    im_end_id=100279,
    im_patch_id=100280,
    im_col_id=100281,
    low_res_im_start_id=100282,
    image_placeholder_id=100283,
    im_end_turn_id=100265,
)


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
    input_ids[0] = _TEST_TOKEN_IDS.im_patch_id
    token_type_ids = np.isin(
        input_ids,
        np.fromiter(_TEST_TOKEN_IDS.image_token_ids, dtype=np.int64),
    ).astype(np.int64)
    return {
        "input_ids": input_ids,
        "labels": np.array(input_ids, copy=True),
        "loss_masks": (
            np.zeros(length, dtype=np.float32) if zero_loss else np.ones(length, dtype=np.float32)
        ),
        "position_ids": np.arange(length, dtype=np.int64),
        "token_type_ids": token_type_ids,
        "images": np.zeros((1, 2, 3), dtype=np.float32),
        "pooled_patches_idx": np.zeros((1, 2), dtype=np.int64),
    }


class _RawVisualDataset:
    def __init__(self, length: int = 8, *, rows: int = 4, zero_loss: bool = False):
        self.length = length
        self.rows = rows
        self.zero_loss = zero_loss
        self.zero_rows: set[int] = set()

    def __len__(self):
        return self.rows

    def get(self, index: int, epoch: int):
        return _example(
            self.length,
            row=index,
            epoch=epoch,
            zero_loss=self.zero_loss or index in self.zero_rows,
        )


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
        self.zero_rows: set[int] = set()
        self.content_fingerprint = "a" * 64

    def __len__(self):
        return self.rows

    def get(self, index: int, epoch: int):
        example = _example(
            self.length,
            row=index,
            epoch=epoch,
            zero_loss=self.zero_loss or index in self.zero_rows,
        )
        example["input_ids"][0] = 0
        example["token_type_ids"][0] = 0
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
    from olmo_core.data.multimodal import (
        vision_alignment_joint_provenance as provenance,
    )

    monkeypatch.setattr(provenance, "N_PATCHES_SQ", 2)
    monkeypatch.setattr(provenance, "PATCH_DIM", 3)
    monkeypatch.setattr(provenance, "POOL_H", 1)
    monkeypatch.setattr(provenance, "POOL_W", 2)
    for module in (exporter, auditor):
        monkeypatch.setattr(module, "JOINT_SEQUENCE_LENGTH", 8)
        monkeypatch.setattr(module, "JOINT_VISUAL_PROBE_INDICES", 2)
        monkeypatch.setattr(module, "JOINT_VISUAL_PROBE_EPOCHS", (0, 1))
        monkeypatch.setattr(module, "JOINT_NATIVE_PROBE_INDICES", 2)
        monkeypatch.setattr(module, "JOINT_NATIVE_PROBE_EPOCHS", (0,))


def _build_fixture(
    tmp_path, monkeypatch, *, zero_source=None, partial_zero_source=None
) -> _Fixture:
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
            if partial_zero_source == source_name:
                selected_indices = exporter.select_deterministic_probe_indices(
                    len(native),
                    2,
                    seed=exporter.JOINT_PROBE_SEED,
                    dataset_fingerprint=native.content_fingerprint,
                )
                native.zero_rows.add(selected_indices[0])
            entry = exporter.export_source_probe(
                native,
                source_name=source_name,
                kind="native_text_replay",
                output_path=output_path,
                unique_indices=2,
                epochs=(0,),
                seed=exporter.JOINT_PROBE_SEED,
                token_ids=_TEST_TOKEN_IDS,
            )
            runtimes[source_name] = auditor._RuntimeProbeSource(native, None, _TEST_TOKEN_IDS)
            native_dataset = native
        else:
            raw = _RawVisualDataset(zero_loss=zero_source == source_name)
            selected = _SelectedVisualDataset(raw, source_name=source_name)
            if partial_zero_source == source_name:
                selected_indices = exporter.select_deterministic_probe_indices(
                    len(selected),
                    2,
                    seed=exporter.JOINT_PROBE_SEED,
                    dataset_fingerprint=selected.content_fingerprint,
                )
                raw.zero_rows.add(selected.indices[selected_indices[0]])
            entry = exporter.export_source_probe(
                selected,
                source_name=source_name,
                kind="visual",
                output_path=output_path,
                unique_indices=2,
                epochs=(0, 1),
                seed=exporter.JOINT_PROBE_SEED,
                unbounded_dataset=raw,
                token_ids=_TEST_TOKEN_IDS,
            )
            visual_selected[source_name] = selected
            visual_raw[source_name] = raw
            runtimes[source_name] = auditor._RuntimeProbeSource(selected, raw, _TEST_TOKEN_IDS)
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
        lambda path, *, expected_token_ids, expected_sha256: (
            projection if expected_token_ids == _TEST_TOKEN_IDS else None
        ),
    )
    monkeypatch.setattr(auditor, "_load_native_manifest", lambda path: native_manifest)
    monkeypatch.setattr(
        auditor,
        "load_pinned_vision_alignment_tokenizer",
        lambda **_kwargs: (object(), _TEST_TOKEN_IDS),
    )
    monkeypatch.setattr(
        auditor,
        "_load_native_receipt",
        lambda path, *, expected_sha256: receipt,
    )

    def build_runtime_sources(projection_arg, native_arg, receipt_arg, *, tokenizer, token_ids):
        assert projection_arg is projection
        assert native_arg is native_manifest
        assert receipt_arg is receipt
        assert tokenizer is not None
        assert token_ids == _TEST_TOKEN_IDS
        return runtimes

    monkeypatch.setattr(auditor, "_build_runtime_sources", build_runtime_sources)
    monkeypatch.setattr(
        auditor,
        "_fresh_native_runtime_evidence",
        lambda manifest, receipt_arg, *, expected_size: native_dataset,
    )
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


def _rewrite_probe_and_catalog(
    fixture: _Fixture,
    source_name: str,
    records: list[dict],
    *,
    canonical: bool = True,
) -> None:
    source_path = fixture.catalog_path.parent / f"{source_name}.jsonl"
    if canonical:
        source_raw = b"".join(_canonical_bytes(record) + b"\n" for record in records)
    else:
        source_raw = b"".join(json.dumps(record).encode("utf-8") + b"\n" for record in records)
    source_path.write_bytes(source_raw)
    catalog = json.loads(fixture.catalog_path.read_text())
    source = next(item for item in catalog["sources"] if item["name"] == source_name)
    source["sha256"] = hashlib.sha256(source_raw).hexdigest()
    catalog.pop("content_sha256")
    catalog["content_sha256"] = _canonical_sha256(catalog)
    fixture.catalog_path.write_bytes(_canonical_bytes(catalog) + b"\n")


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


def test_parallel_audit_report_is_byte_identical_to_serial(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, monkeypatch)
    serial = auditor.audit_joint_catalog(fixture.catalog_path, workers=1)
    progress = []
    parallel = auditor.audit_joint_catalog(
        fixture.catalog_path,
        workers=4,
        progress=progress.append,
    )

    assert _canonical_bytes(parallel) == _canonical_bytes(serial)
    assert progress[0]["event"] == "phase_start"
    assert progress[-1]["event"] == "phase_complete"
    assert [event["source"] for event in progress if event["event"] == "source_complete"] == list(
        auditor.JOINT_SOURCE_NAMES
    )
    assert all(event["phase"] == "joint_mix_audit" for event in progress)
    assert all(isinstance(event["elapsed_seconds"], float) for event in progress)


def test_runtime_sources_are_built_lazily_and_released_per_source(monkeypatch):
    parent_spec = SimpleNamespace(
        tokenizer_id=auditor.VISION_ALIGNMENT_TOKENIZER_ID,
        tokenizer_revision=auditor.VISION_ALIGNMENT_TOKENIZER_REVISION,
        tokenizer_fingerprint=auditor.VISION_ALIGNMENT_TOKENIZER_FINGERPRINT,
    )
    projection = SimpleNamespace(source_spec=SimpleNamespace(perception_spec=parent_spec))
    native_manifest = SimpleNamespace()
    receipt = SimpleNamespace()
    built = []
    references = []

    class RuntimeDataset:
        pass

    def new_dataset(label):
        dataset = RuntimeDataset()
        built.append(label)
        references.append(weakref.ref(dataset))
        return dataset

    monkeypatch.setattr(
        auditor,
        "build_selected_joint_dataset",
        lambda *_args, **kwargs: new_dataset(f"selected:{kwargs['logical_split']}"),
    )
    monkeypatch.setattr(
        auditor,
        "_build_unbounded_visual_dataset",
        lambda _projection, _tokenizer, _token_ids, source_name: new_dataset(
            f"unbounded:{source_name}"
        ),
    )
    sources = auditor._build_runtime_sources(
        projection,
        native_manifest,
        receipt,
        tokenizer=object(),
        token_ids=_TEST_TOKEN_IDS,
    )

    assert tuple(sorted(sources)) == auditor.JOINT_SOURCE_NAMES
    assert built == []
    runtime = sources["pixmo_caption"]
    assert built == ["selected:train", "unbounded:pixmo_caption"]
    assert all(reference() is not None for reference in references)
    del runtime
    gc.collect()
    assert all(reference() is None for reference in references)


def test_audit_calibration_uses_zero_relative_tolerance(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, monkeypatch)
    real_isclose = auditor.math.isclose
    calls = []

    def strict_isclose(left, right, *, rel_tol, abs_tol):
        calls.append((rel_tol, abs_tol))
        return real_isclose(left, right, rel_tol=rel_tol, abs_tol=abs_tol)

    monkeypatch.setattr(auditor.math, "isclose", strict_isclose)
    report = auditor.audit_joint_catalog(fixture.catalog_path)

    assert report["status"] == "ok"
    assert len(calls) == len(auditor.JOINT_SOURCE_NAMES)
    assert all(rel_tol == 0.0 and abs_tol == 1e-12 for rel_tol, abs_tol in calls)


def test_audit_rejects_probe_file_byte_tamper(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, monkeypatch)
    source_path = tmp_path / "pixmo_caption.jsonl"
    source_path.write_bytes(source_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="probe-file SHA-256 differs"):
        auditor.audit_joint_catalog(fixture.catalog_path)


def test_audit_rejects_noncanonical_jsonl_even_when_catalog_rehashes_it(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, monkeypatch)
    source_path = tmp_path / "pixmo_caption.jsonl"
    records = [json.loads(line) for line in source_path.read_text().splitlines()]
    _rewrite_probe_and_catalog(fixture, "pixmo_caption", records, canonical=False)

    with pytest.raises(ValueError, match="not exact canonical JSONL"):
        auditor.audit_joint_catalog(fixture.catalog_path)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("probe_epoch", False, "probe_epoch must be an integer"),
        ("serialized_row_sha256", "0" * 64, "does not match serialized_fields"),
    ],
)
def test_audit_rejects_malformed_stored_record_structure(
    tmp_path, monkeypatch, field, value, match
):
    fixture = _build_fixture(tmp_path, monkeypatch)
    source_path = tmp_path / "pixmo_caption.jsonl"
    records = [json.loads(line) for line in source_path.read_text().splitlines()]
    records[0][field] = value
    _rewrite_probe_and_catalog(fixture, "pixmo_caption", records)

    with pytest.raises(ValueError, match=match):
        auditor.audit_joint_catalog(fixture.catalog_path)


def test_audit_rejects_resigned_out_of_range_position_id(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, monkeypatch)
    source_path = tmp_path / "pixmo_caption.jsonl"
    records = [json.loads(line) for line in source_path.read_text().splitlines()]
    record = records[0]
    record["position_ids"][0] = len(record["input_ids"])
    record["serialized_fields"]["position_ids"] = array_content_descriptor(
        np.asarray(record["position_ids"], dtype=np.int64),
        field_name="position_ids",
    )
    record["serialized_row_sha256"] = serialized_descriptor_sha256(record["serialized_fields"])
    _rewrite_probe_and_catalog(fixture, "pixmo_caption", records)

    with pytest.raises(ValueError, match="position_ids must be within the token sequence"):
        auditor.audit_joint_catalog(fixture.catalog_path)


def test_audit_rejects_resigned_positive_loss_on_ignored_visual_label(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, monkeypatch)
    source_path = tmp_path / "pixmo_caption.jsonl"
    records = [json.loads(line) for line in source_path.read_text().splitlines()]
    record = records[0]
    record["labels"][1] = -100
    record["serialized_fields"]["labels"] = array_content_descriptor(
        np.asarray(record["labels"], dtype=np.int64),
        field_name="labels",
    )
    record["serialized_row_sha256"] = serialized_descriptor_sha256(record["serialized_fields"])
    _rewrite_probe_and_catalog(fixture, "pixmo_caption", records)

    with pytest.raises(ValueError, match="positive loss weights require non-ignored labels"):
        auditor.audit_joint_catalog(fixture.catalog_path)


@pytest.mark.parametrize(
    ("descriptor_field", "member", "value", "match"),
    [
        ("images", "dtype", "<f8", "dtype differs from model input"),
        ("pooled_patches_idx", "dtype", "<i4", "dtype differs from model input"),
        ("images", "shape", [1, 0, 0], "invalid geometry"),
        ("pooled_patches_idx", "shape", [1, 0], "invalid geometry"),
    ],
)
def test_audit_rejects_resigned_malformed_model_geometry(
    tmp_path, monkeypatch, descriptor_field, member, value, match
):
    fixture = _build_fixture(tmp_path, monkeypatch)
    source_path = tmp_path / "pixmo_caption.jsonl"
    records = [json.loads(line) for line in source_path.read_text().splitlines()]
    records[0]["serialized_fields"][descriptor_field][member] = value
    records[0]["serialized_row_sha256"] = serialized_descriptor_sha256(
        records[0]["serialized_fields"]
    )
    _rewrite_probe_and_catalog(fixture, "pixmo_caption", records)

    with pytest.raises(ValueError, match=match):
        auditor.audit_joint_catalog(fixture.catalog_path)


def test_audit_rejects_resigned_tenth_visual_crop(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, monkeypatch)
    source_path = tmp_path / "pixmo_caption.jsonl"
    records = [json.loads(line) for line in source_path.read_text().splitlines()]
    record = records[0]
    record["serialized_fields"]["images"]["shape"][0] = 10
    record["image_crops"] = 10
    record["serialized_row_sha256"] = serialized_descriptor_sha256(record["serialized_fields"])
    _rewrite_probe_and_catalog(fixture, "pixmo_caption", records)

    with pytest.raises(ValueError, match="valid image and pooled-token counts"):
        auditor.audit_joint_catalog(fixture.catalog_path)


def test_auditor_fresh_native_evidence_rechecks_size_and_source_stats(tmp_path, monkeypatch):
    manifest = SimpleNamespace(
        path=(tmp_path / "native.json").resolve(),
        content_fingerprint="7" * 64,
        num_windows=3,
    )
    receipt = SimpleNamespace(path=(tmp_path / "receipt.json").resolve(), receipt_sha256="8" * 64)
    calls = []

    class FreshDataset:
        content_fingerprint = manifest.content_fingerprint
        sequence_length = auditor.JOINT_SEQUENCE_LENGTH
        source_counts = {"one": 1, "two": 2}

        def __init__(self, path, **kwargs):
            calls.append((path, kwargs))

        def __len__(self):
            return 3

    monkeypatch.setattr(auditor, "NativeTextReplayDataset", FreshDataset)
    result = auditor._fresh_native_runtime_evidence(manifest, receipt, expected_size=3)

    assert isinstance(result, FreshDataset)
    assert calls == [
        (
            manifest.path,
            {
                "expected_fingerprint": manifest.content_fingerprint,
                "verification_receipt_path": receipt.path,
                "expected_verification_receipt_sha256": receipt.receipt_sha256,
                "validate_source_files": True,
            },
        )
    ]


def test_audit_rejects_live_serialized_row_drift(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, monkeypatch)
    fixture.native_dataset.drift = True
    native_source = next(
        source
        for source in json.loads(fixture.catalog_path.read_text())["sources"]
        if source["name"] == "native_text_replay"
    )
    first_pair = (native_source["probe_indices"][0], 0)
    another_worker_started = threading.Event()
    lock = threading.Lock()
    active = 0
    original = auditor._live_probe_record

    def slow_parallel_native(runtime, **kwargs):
        nonlocal active
        if kwargs["source_name"] != "native_text_replay":
            return original(runtime, **kwargs)
        pair = (kwargs["dataset_index"], kwargs["epoch"])
        with lock:
            active += 1
        try:
            if pair == first_pair:
                if not another_worker_started.wait(timeout=2):
                    raise AssertionError("parallel auditor did not start another worker")
            else:
                another_worker_started.set()
                time.sleep(0.05)
            return original(runtime, **kwargs)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(auditor, "_live_probe_record", slow_parallel_native)

    with pytest.raises(ValueError, match="serialized row drifted"):
        auditor.audit_joint_catalog(fixture.catalog_path, workers=4)
    assert active == 0


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


def test_audit_allows_individual_zero_loss_native_row_when_source_total_is_positive(
    tmp_path, monkeypatch
):
    fixture = _build_fixture(
        tmp_path,
        monkeypatch,
        partial_zero_source="native_text_replay",
    )
    report = auditor.audit_joint_catalog(fixture.catalog_path)

    assert report["status"] == "ok"
    assert report["sources"]["native_text_replay"]["zero_loss_examples"] == 1
    assert report["mean_loss_weight"]["native_text_replay"] > 0


def test_audit_fails_on_individual_zero_loss_visual_row(tmp_path, monkeypatch):
    fixture = _build_fixture(
        tmp_path,
        monkeypatch,
        partial_zero_source="pixmo_caption",
    )
    report = auditor.audit_joint_catalog(fixture.catalog_path)

    assert report["status"] == "failed"
    assert report["sources"]["pixmo_caption"]["zero_loss_examples"] == 2
    assert report["failures"] == ["pixmo_caption: contains 2 zero-loss visual probe rows"]


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


def test_audit_rejects_boolean_source_epoch_even_with_recomputed_catalog(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, monkeypatch)
    catalog = json.loads(fixture.catalog_path.read_text())
    source = next(item for item in catalog["sources"] if item["name"] == "native_text_replay")
    source["probe_epochs"] = [False]
    catalog.pop("content_sha256")
    catalog["content_sha256"] = _canonical_sha256(catalog)
    fixture.catalog_path.write_bytes(_canonical_bytes(catalog) + b"\n")

    with pytest.raises(ValueError, match="epoch panel differs"):
        auditor.audit_joint_catalog(fixture.catalog_path)


@pytest.mark.parametrize("closing_drift", ["projection", "native", "receipt", "native_stat"])
def test_audit_closing_pass_reloads_and_rejects_late_drift(tmp_path, monkeypatch, closing_drift):
    fixture = _build_fixture(tmp_path, monkeypatch)
    if closing_drift == "projection":
        calls = 0

        def load_projection(path, *, expected_token_ids, expected_sha256):
            nonlocal calls
            assert expected_token_ids == _TEST_TOKEN_IDS
            calls += 1
            if calls == 1:
                return fixture.projection
            return SimpleNamespace(
                **{
                    **vars(fixture.projection),
                    "content_sha256": "0" * 64,
                }
            )

        monkeypatch.setattr(auditor, "load_joint_visual_projection_manifest", load_projection)
        match = "projection identity changed"
    elif closing_drift == "native":
        calls = 0

        def load_native(path):
            nonlocal calls
            calls += 1
            if calls == 1:
                return fixture.native_manifest
            return SimpleNamespace(
                **{
                    **vars(fixture.native_manifest),
                    "content_fingerprint": "0" * 64,
                }
            )

        monkeypatch.setattr(auditor, "_load_native_manifest", load_native)
        match = "manifest identity changed"
    elif closing_drift == "receipt":
        calls = 0

        def load_receipt(path, *, expected_sha256):
            nonlocal calls
            calls += 1
            if calls == 1:
                return fixture.receipt
            raise ValueError("late receipt/build pin drift")

        monkeypatch.setattr(auditor, "_load_native_receipt", load_receipt)
        match = "receipt/build pin drift"
    else:
        monkeypatch.setattr(
            auditor,
            "_fresh_native_runtime_evidence",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("late native stat drift")),
        )
        match = "native stat drift"

    with pytest.raises(ValueError, match=match):
        auditor.audit_joint_catalog(fixture.catalog_path)
