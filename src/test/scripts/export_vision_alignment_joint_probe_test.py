from __future__ import annotations

import hashlib
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from olmo_core.nn.vision import Molmo2TokenIds
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


@pytest.fixture(autouse=True)
def _small_joint_geometry(monkeypatch):
    from olmo_core.data.multimodal import (
        vision_alignment_joint_provenance as provenance,
    )

    monkeypatch.setattr(provenance, "N_PATCHES_SQ", 2)
    monkeypatch.setattr(provenance, "PATCH_DIM", 3)
    monkeypatch.setattr(provenance, "POOL_H", 1)
    monkeypatch.setattr(provenance, "POOL_W", 2)


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
        example["input_ids"][0] = 0
        example["token_type_ids"][0] = 0
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
        token_ids=_TEST_TOKEN_IDS,
    )

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert output.read_bytes() == b"".join(
        json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
        for row in records
    )
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


def test_parallel_visual_export_is_byte_identical_under_out_of_order_completion(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(exporter, "JOINT_SEQUENCE_LENGTH", 8)
    raw = _RawDataset(8, rows=8)
    selected = _SelectedDataset(raw, source_name="pixmo_caption")
    serial_path = tmp_path / "serial" / "pixmo_caption.jsonl"
    parallel_path = tmp_path / "parallel" / "pixmo_caption.jsonl"
    serial_entry = exporter.export_source_probe(
        selected,
        source_name="pixmo_caption",
        kind="visual",
        output_path=serial_path,
        unique_indices=4,
        epochs=(0, 1),
        seed=13,
        unbounded_dataset=raw,
        token_ids=_TEST_TOKEN_IDS,
        workers=1,
    )

    first_pair = (serial_entry["probe_indices"][0], 0)
    release_first = threading.Event()
    lock = threading.Lock()
    completion_order = []
    active = 0
    maximum_active = 0
    original = exporter._live_probe_record

    def adversarial_completion(*args, **kwargs):
        nonlocal active, maximum_active
        pair = (kwargs["dataset_index"], kwargs["epoch"])
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            if pair == first_pair:
                if not release_first.wait(timeout=2):
                    raise AssertionError("parallel exporter did not overlap independent rows")
            result = original(*args, **kwargs)
            if pair != first_pair:
                release_first.set()
            with lock:
                completion_order.append(pair)
            return result
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(exporter, "_live_probe_record", adversarial_completion)
    progress = []
    parallel_entry = exporter.export_source_probe(
        selected,
        source_name="pixmo_caption",
        kind="visual",
        output_path=parallel_path,
        unique_indices=4,
        epochs=(0, 1),
        seed=13,
        unbounded_dataset=raw,
        token_ids=_TEST_TOKEN_IDS,
        workers=3,
        progress=progress.append,
    )

    assert completion_order[0] != first_pair
    assert 2 <= maximum_active <= 3
    assert parallel_path.read_bytes() == serial_path.read_bytes()
    assert parallel_entry == serial_entry
    assert [event["event"] for event in progress] == [
        "source_start",
        "source_progress",
        "source_complete",
    ]
    assert all(event["phase"] == "joint_probe_export" for event in progress)
    assert all(isinstance(event["elapsed_seconds"], float) for event in progress)


def test_parallel_export_is_quiescent_before_row_failure_returns(tmp_path, monkeypatch):
    monkeypatch.setattr(exporter, "JOINT_SEQUENCE_LENGTH", 8)
    native = _NativeDataset(8, rows=8)
    indices = exporter.select_deterministic_probe_indices(
        len(native),
        4,
        seed=13,
        dataset_fingerprint=native.content_fingerprint,
    )
    first_pair = (indices[0], 0)
    another_worker_started = threading.Event()
    lock = threading.Lock()
    active = 0
    original = exporter._live_probe_record

    def fail_with_slow_in_flight_work(*args, **kwargs):
        nonlocal active
        pair = (kwargs["dataset_index"], kwargs["epoch"])
        with lock:
            active += 1
        try:
            if pair == first_pair:
                if not another_worker_started.wait(timeout=2):
                    raise AssertionError("parallel exporter did not start another worker")
                raise ValueError("synthetic row failure")
            another_worker_started.set()
            time.sleep(0.05)
            return original(*args, **kwargs)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(exporter, "_live_probe_record", fail_with_slow_in_flight_work)
    output = tmp_path / "native_text_replay.jsonl"
    with pytest.raises(ValueError, match="synthetic row failure"):
        exporter.export_source_probe(
            native,
            source_name="native_text_replay",
            kind="native_text_replay",
            output_path=output,
            unique_indices=4,
            epochs=(0,),
            seed=13,
            token_ids=_TEST_TOKEN_IDS,
            workers=3,
        )

    assert active == 0
    assert not output.exists()


def test_visual_export_allows_global_plus_eight_tiles_and_rejects_tenth(tmp_path, monkeypatch):
    monkeypatch.setattr(exporter, "JOINT_SEQUENCE_LENGTH", 8)
    raw = _RawDataset(8)
    original_get = raw.get

    def with_crops(index, epoch):
        example = original_get(index, epoch)
        example["images"] = np.zeros((9, 2, 3), dtype=np.float32)
        return example

    raw.get = with_crops
    selected = _SelectedDataset(raw, source_name="pixmo_caption")
    exporter.export_source_probe(
        selected,
        source_name="pixmo_caption",
        kind="visual",
        output_path=tmp_path / "nine-crops.jsonl",
        unique_indices=1,
        epochs=(0,),
        seed=13,
        unbounded_dataset=raw,
        token_ids=_TEST_TOKEN_IDS,
    )

    def with_too_many_crops(index, epoch):
        example = original_get(index, epoch)
        example["images"] = np.zeros((10, 2, 3), dtype=np.float32)
        return example

    raw.get = with_too_many_crops
    with pytest.raises(ValueError, match="positive image and pooled-token counts"):
        exporter.export_source_probe(
            selected,
            source_name="pixmo_caption",
            kind="visual",
            output_path=tmp_path / "ten-crops.jsonl",
            unique_indices=1,
            epochs=(0,),
            seed=13,
            unbounded_dataset=raw,
            token_ids=_TEST_TOKEN_IDS,
        )


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
            token_ids=_TEST_TOKEN_IDS,
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
        example["input_ids"][1] += 1
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
            token_ids=_TEST_TOKEN_IDS,
            workers=4,
        )


@pytest.mark.parametrize(
    "defect",
    [
        "empty_visual",
        "empty_pooled",
        "misaligned_tokens",
        "invalid_image_rank",
        "invalid_image_geometry",
        "image_dtype",
        "pooled_dtype",
        "pooled_out_of_range",
        "image_token_mismatch",
        "negative_position",
        "position_out_of_range",
        "negative_subsegment",
        "token_type_mismatch",
        "ignored_weighted_label",
        "input_out_of_vocab",
        "label_invalid_negative",
        "label_out_of_vocab",
    ],
)
def test_visual_export_rejects_malformed_complete_record(tmp_path, monkeypatch, defect):
    monkeypatch.setattr(exporter, "JOINT_SEQUENCE_LENGTH", 8)
    raw = _RawDataset(8)
    original_get = raw.get

    def malformed(index, epoch):
        example = original_get(index, epoch)
        if defect == "empty_visual":
            example["images"] = np.zeros((0, 2, 3), dtype=np.float32)
            example["pooled_patches_idx"] = np.zeros((0, 2), dtype=np.int64)
        elif defect == "empty_pooled":
            example["pooled_patches_idx"] = np.zeros((0, 2), dtype=np.int64)
        elif defect == "misaligned_tokens":
            example["labels"] = example["labels"][:-1]
        elif defect == "invalid_image_rank":
            example["images"] = np.zeros((1, 2), dtype=np.float32)
        elif defect == "invalid_image_geometry":
            example["images"] = np.zeros((1, 1, 3), dtype=np.float32)
        elif defect == "image_dtype":
            example["images"] = example["images"].astype(np.float64)
        elif defect == "pooled_dtype":
            example["pooled_patches_idx"] = example["pooled_patches_idx"].astype(np.int32)
        elif defect == "pooled_out_of_range":
            example["pooled_patches_idx"][0, 0] = 2
        elif defect == "image_token_mismatch":
            example["input_ids"][0] = 0
        elif defect == "negative_position":
            example["position_ids"][0] = -1
        elif defect == "position_out_of_range":
            example["position_ids"][0] = len(example["input_ids"])
        elif defect == "negative_subsegment":
            example["subsegment_ids"] = np.arange(len(example["input_ids"]), dtype=np.int64)
            example["subsegment_ids"][0] = -1
        elif defect == "token_type_mismatch":
            example["token_type_ids"][0] = 0
        elif defect == "ignored_weighted_label":
            example["labels"][1] = -100
        elif defect == "input_out_of_vocab":
            example["input_ids"][1] = 100352
        elif defect == "label_invalid_negative":
            example["labels"][1] = -1
        else:
            example["labels"][1] = 100352
        return example

    raw.get = malformed
    selected = _SelectedDataset(raw, source_name="pixmo_caption")
    with pytest.raises(
        ValueError,
        match=(
            "positive image|must align|shape|exact float32|exact int64|out-of-range|"
            "image-token|token sequence|subsegment_ids|exactly mark|non-ignored|model-vocabulary"
        ),
    ):
        exporter.export_source_probe(
            selected,
            source_name="pixmo_caption",
            kind="visual",
            output_path=tmp_path / "pixmo_caption.jsonl",
            unique_indices=1,
            epochs=(0,),
            seed=13,
            unbounded_dataset=raw,
            token_ids=_TEST_TOKEN_IDS,
        )


def test_export_rejects_boolean_epoch(tmp_path):
    with pytest.raises(ValueError, match="non-negative integers"):
        exporter.export_source_probe(
            _NativeDataset(8),
            source_name="native_text_replay",
            kind="native_text_replay",
            output_path=tmp_path / "native_text_replay.jsonl",
            unique_indices=1,
            epochs=(False,),
            seed=13,
            token_ids=_TEST_TOKEN_IDS,
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
        token_ids=_TEST_TOKEN_IDS,
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
            token_ids=_TEST_TOKEN_IDS,
        )


def test_native_manifest_parser_bytes_must_match_external_raw_pin(tmp_path, monkeypatch):
    native_path = tmp_path / "native.json"
    native_path.write_text("pinned bytes")
    raw_sha256 = hashlib.sha256(native_path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        exporter.NativeTextReplayManifest,
        "load",
        lambda path: SimpleNamespace(
            manifest_sha256="0" * 64,
            content_fingerprint="1" * 64,
        ),
    )

    with pytest.raises(ValueError, match="runtime identity differs"):
        exporter._load_native_manifest_pinned(
            native_path,
            expected_raw_sha256=raw_sha256,
            expected_content_fingerprint="1" * 64,
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

    native_ordinal = exporter.JOINT_SOURCE_NAMES.index("native_text_replay")
    for field, value in (
        ("probe_epochs", [False]),
        ("probe_indices", [False, *entries[native_ordinal]["probe_indices"][1:]]),
        ("truncated_rows", False),
    ):
        mutated = json.loads(json.dumps(entries))
        mutated[native_ordinal][field] = value
        with pytest.raises(ValueError, match="identity differs"):
            exporter.build_probe_catalog(
                projection=projection,
                native_manifest=native,
                verification_receipt_path=receipt_path.resolve(),
                verification_receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                source_entries=mutated,
                exporter_sha256="3" * 64,
                probe_seed=exporter.JOINT_PROBE_SEED,
            )


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


def test_publish_fails_closed_without_renameat2(tmp_path, monkeypatch):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    monkeypatch.setattr(exporter.ctypes, "CDLL", lambda *_args, **_kwargs: object())

    with pytest.raises(RuntimeError, match="renameat2.*unavailable"):
        exporter._publish_no_replace(source, destination)
    assert source.is_dir()
    assert not destination.exists()


def test_closing_validation_reloads_all_inputs_and_fresh_stats_native(tmp_path, monkeypatch):
    projection_path = (tmp_path / "projection.json").resolve()
    native_path = (tmp_path / "native.json").resolve()
    receipt_path = (tmp_path / "receipt.json").resolve()
    for path in (projection_path, native_path, receipt_path):
        path.write_text(path.name)
    spec = SimpleNamespace(as_canonical_dict=lambda: {"phase": "joint"})
    projection = SimpleNamespace(
        path=projection_path,
        raw_sha256=hashlib.sha256(projection_path.read_bytes()).hexdigest(),
        content_sha256="1" * 64,
        source_spec=spec,
    )
    native = SimpleNamespace(
        path=native_path,
        manifest_sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(),
        content_fingerprint="2" * 64,
        provenance={"verification_receipt_sha256": "3" * 64},
    )

    class Receipt:
        path = receipt_path
        receipt_sha256 = "3" * 64
        validations = 0

        def validate_manifest(self, manifest):
            assert manifest is native
            self.validations += 1

    receipt = Receipt()
    calls = []
    monkeypatch.setattr(
        exporter,
        "load_joint_visual_projection_manifest",
        lambda path, *, expected_token_ids, expected_sha256: (
            projection if expected_token_ids == _TEST_TOKEN_IDS else None
        ),
    )
    monkeypatch.setattr(
        exporter,
        "_load_native_manifest_pinned",
        lambda path, *, expected_raw_sha256, expected_content_fingerprint: native,
    )
    monkeypatch.setattr(
        exporter.NativeTextReplayVerificationReceipt,
        "load",
        lambda path, *, expected_sha256: receipt,
    )
    monkeypatch.setattr(
        exporter,
        "_fresh_native_runtime_evidence",
        lambda manifest, receipt_arg, *, expected_size, tokenizer: calls.append(
            (manifest, receipt_arg, expected_size, tokenizer)
        ),
    )
    tokenizer = object()
    exporter._closing_validate_inputs(
        projection=projection,
        native_manifest=native,
        receipt=receipt,
        expected_native_size=17,
        tokenizer=tokenizer,
        token_ids=_TEST_TOKEN_IDS,
    )

    assert receipt.validations == 2
    assert calls == [(native, receipt, 17, tokenizer)]


def test_fresh_native_runtime_evidence_uses_strict_dataset_constructor(tmp_path, monkeypatch):
    calls = []
    manifest = SimpleNamespace(
        path=(tmp_path / "native.json").resolve(),
        content_fingerprint="4" * 64,
        num_windows=3,
    )
    receipt = SimpleNamespace(path=(tmp_path / "receipt.json").resolve(), receipt_sha256="5" * 64)

    class FreshDataset:
        content_fingerprint = manifest.content_fingerprint
        sequence_length = exporter.JOINT_SEQUENCE_LENGTH
        source_counts = {"a": 1, "b": 2}

        def __init__(self, path, **kwargs):
            calls.append((path, kwargs))

        def __len__(self):
            return 3

        def validate_tokenizer(self, tokenizer):
            calls.append(("tokenizer", tokenizer))

    monkeypatch.setattr(exporter, "NativeTextReplayDataset", FreshDataset)
    tokenizer = object()
    result = exporter._fresh_native_runtime_evidence(
        manifest,
        receipt,
        expected_size=3,
        tokenizer=tokenizer,
    )

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
        ),
        ("tokenizer", tokenizer),
    ]
