from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pytest

from olmo_core.data.multimodal.sequence_builder import (
    ATTEND_ALL_SUBSEGMENT_ID,
    build_branched_sequence,
    build_packed_sequence,
)
from olmo_core.data.multimodal.ssmax_single_response import (
    SSMAX_SINGLE_RESPONSE_CALIBRATION_FORMAT,
    SSMAX_SINGLE_RESPONSE_CALIBRATION_VERSION,
    SSMaxSingleResponseDataset,
    project_ssmax_single_response,
    ssmax_single_response_calibration_summary,
    validate_ssmax_single_response_calibration,
)


def _with_visual_fields(example: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        **example,
        "images": np.zeros((1, 4, 3), dtype=np.float32),
        "pooled_patches_idx": np.zeros((1, 4), dtype=np.int64),
        "metadata": {"kept": True},
    }


class _Dataset:
    content_fingerprint = "synthetic-selected-source-v1"

    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def get(self, index: int, epoch: int = 0) -> dict[str, Any]:
        del epoch
        return self.rows[index]


class _SelectedDataset:
    def __init__(self, dataset: _Dataset, indices: tuple[int, ...]):
        self._dataset = dataset
        self.indices = indices
        self.content_fingerprint = f"selected-{dataset.content_fingerprint}"

    def __len__(self) -> int:
        return len(self.indices)

    def get(self, index: int, epoch: int = 0) -> dict[str, Any]:
        return self._dataset.get(self.indices[index], epoch)


class _AuditedDataset:
    content_fingerprint = "synthetic-source-audit-v1"

    def __init__(self, dataset: _SelectedDataset, *, base_fingerprint: str | None = None):
        self._dataset = dataset
        self.ssmax_projection_base_content_fingerprint = (
            dataset.content_fingerprint if base_fingerprint is None else base_fingerprint
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def get(self, index: int, epoch: int = 0) -> dict[str, Any]:
        return self._dataset.get(index, epoch)


class _EpochVaryingDataset(_Dataset):
    def get(self, index: int, epoch: int = 0) -> dict[str, Any]:
        row = super().get(index, epoch)
        output = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in row.items()
        }
        output["metadata"] = {"materialized_epoch": epoch}
        return output


def _assert_exact_model_fields(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    assert set(actual) == set(expected)
    for field, expected_value in expected.items():
        if isinstance(expected_value, np.ndarray):
            np.testing.assert_array_equal(actual[field], expected_value)
            assert actual[field].dtype == expected_value.dtype
        else:
            assert actual[field] == expected_value


@pytest.mark.parametrize(
    "loss_token_weighting",
    ["none", "root_subsegments", "root_subsegments_root_tokens"],
)
def test_project_packed_sequence_is_exact_single_branch(loss_token_weighting: str) -> None:
    prefix = [10, 11, 12]
    responses = [[21, 22], [31, 32, 33, 34], [41]]
    multi = _with_visual_fields(
        build_packed_sequence(
            prefix,
            responses,
            eos_id=2,
            image_token_ids=frozenset({11}),
            loss_token_weighting=loss_token_weighting,
        )
    )
    projected, receipt = project_ssmax_single_response(
        multi,
        source_name="pixmo_caption",
        logical_split="validation",
        sample_index=17,
        epoch=9,
        seed=6198,
        loss_token_weighting=loss_token_weighting,
    )
    selected = receipt["selected_branch_id"]
    assert isinstance(selected, int)
    expected = _with_visual_fields(
        build_packed_sequence(
            prefix,
            [responses[selected]],
            eos_id=2,
            image_token_ids=frozenset({11}),
            loss_token_weighting=loss_token_weighting,
        )
    )
    _assert_exact_model_fields(projected, expected)
    assert receipt["positive_targets"] == int(np.count_nonzero(expected["loss_masks"] > 0))
    assert receipt["selection_epoch"] == 0


@pytest.mark.parametrize(
    "loss_token_weighting",
    ["none", "root_subsegments", "root_subsegments_root_tokens"],
)
def test_project_branched_sequence_is_exact_single_branch(loss_token_weighting: str) -> None:
    prefix = [10, 11]
    branches = [
        ([20, 21], [22, 23]),
        ([30], [31, 32, 33]),
        [([40], [41, 42]), ([43, 44], [45])],
    ]
    multi = _with_visual_fields(
        build_branched_sequence(
            prefix,
            branches,
            eos_id=2,
            image_token_ids=frozenset({11}),
            loss_token_weighting=loss_token_weighting,
        )
    )
    projected, receipt = project_ssmax_single_response(
        multi,
        source_name="cosyn_point",
        logical_split="validation",
        sample_index=0,
        epoch=27,
        seed=6198,
        loss_token_weighting=loss_token_weighting,
    )
    selected = receipt["selected_branch_id"]
    assert isinstance(selected, int)
    expected = _with_visual_fields(
        build_branched_sequence(
            prefix,
            [branches[selected]],
            eos_id=2,
            image_token_ids=frozenset({11}),
            loss_token_weighting=loss_token_weighting,
        )
    )
    _assert_exact_model_fields(projected, expected)
    assert receipt["positive_targets"] == int(np.count_nonzero(expected["loss_masks"] > 0))
    assert receipt["selection_epoch"] == 0


def test_validation_is_epoch_fixed_and_training_is_epoch_addressed() -> None:
    row = _with_visual_fields(
        build_branched_sequence(
            [10],
            [([20], [21]), ([30], [31]), ([40], [41])],
            eos_id=2,
            loss_token_weighting="root_subsegments_root_tokens",
        )
    )
    validation = SSMaxSingleResponseDataset(
        _Dataset([row]),
        source_name="cosyn_point",
        logical_split="validation",
        seed=6198,
        loss_token_weighting="root_subsegments_root_tokens",
    )
    first = validation.projection_receipt(0, 0)
    later = validation.projection_receipt(0, 123)
    assert first["selected_branch_id"] == later["selected_branch_id"]
    assert first["selection_epoch"] == later["selection_epoch"] == 0
    np.testing.assert_array_equal(
        validation.get(0, 0)["input_ids"], validation.get(0, 123)["input_ids"]
    )

    training = SSMaxSingleResponseDataset(
        _Dataset([row]),
        source_name="cosyn_point",
        logical_split="train",
        seed=6198,
        loss_token_weighting="root_subsegments_root_tokens",
    )
    receipts = [training.projection_receipt(0, epoch) for epoch in range(32)]
    assert {receipt["selection_epoch"] for receipt in receipts} == set(range(32))
    assert len({receipt["selected_branch_id"] for receipt in receipts}) > 1
    assert training.projection_receipt(0, 7) == training.projection_receipt(0, 7)


def test_validation_materializes_epoch_zero_from_epoch_varying_source() -> None:
    row = _with_visual_fields(
        build_branched_sequence(
            [10],
            [([20], [21]), ([30], [31]), ([40], [41])],
            eos_id=2,
        )
    )
    validation = SSMaxSingleResponseDataset(
        _EpochVaryingDataset([row]),
        source_name="cosyn_point",
        logical_split="evidence",
        seed=95818,
        loss_token_weighting="root_subsegments",
    )
    zero, zero_receipt = validation.get_with_receipt(0, 0)
    later, later_receipt = validation.get_with_receipt(0, 99)
    _assert_exact_model_fields(later, zero)
    assert later["metadata"] == {"materialized_epoch": 0}
    assert zero_receipt["requested_epoch"] == 0
    assert later_receipt["requested_epoch"] == 99
    assert zero_receipt["materialization_epoch"] == 0
    assert later_receipt["materialization_epoch"] == 0
    assert zero_receipt["selected_branch_id"] == later_receipt["selected_branch_id"]


def test_selection_hash_uses_stable_underlying_index() -> None:
    rows = [
        _with_visual_fields(
            build_branched_sequence(
                [10],
                [([20 + index], [30]), ([40 + index], [50]), ([60 + index], [70])],
                eos_id=2,
            )
        )
        for index in range(8)
    ]
    base = _Dataset(rows)
    first = SSMaxSingleResponseDataset(
        _SelectedDataset(base, (7,)),
        source_name="cosyn_point",
        logical_split="validation",
        seed=6198,
        loss_token_weighting="root_subsegments",
    )
    second = SSMaxSingleResponseDataset(
        _SelectedDataset(base, (3, 7)),
        source_name="cosyn_point",
        logical_split="validation",
        seed=6198,
        loss_token_weighting="root_subsegments",
    )
    first_receipt = first.projection_receipt(0)
    second_receipt = second.projection_receipt(1)
    assert first_receipt["stable_sample_index"] == second_receipt["stable_sample_index"] == 7
    assert first_receipt["selected_branch_id"] == second_receipt["selected_branch_id"]
    np.testing.assert_array_equal(first.get(0)["input_ids"], second.get(1)["input_ids"])


def test_offline_selected_and_runtime_audited_projection_fingerprints_match() -> None:
    row = _with_visual_fields(
        build_branched_sequence(
            [10],
            [([20], [21]), ([30], [31])],
            eos_id=2,
        )
    )
    selected = _SelectedDataset(_Dataset([row]), (0,))
    offline = SSMaxSingleResponseDataset(
        selected,
        source_name="cosyn_point",
        logical_split="train",
        seed=95818,
        loss_token_weighting="root_subsegments",
    )
    runtime = SSMaxSingleResponseDataset(
        _AuditedDataset(selected),
        source_name="cosyn_point",
        logical_split="train",
        seed=95818,
        loss_token_weighting="root_subsegments",
    )
    assert offline.base_content_fingerprint == selected.content_fingerprint
    assert runtime.base_content_fingerprint == selected.content_fingerprint
    assert runtime.content_fingerprint == offline.content_fingerprint
    assert runtime.projection_receipt(0, 7) == offline.projection_receipt(0, 7)

    with pytest.raises(ValueError, match="base fingerprint differs"):
        SSMaxSingleResponseDataset(
            _AuditedDataset(selected, base_fingerprint="drift"),
            source_name="cosyn_point",
            logical_split="train",
            seed=95818,
            loss_token_weighting="root_subsegments",
        )


def test_projection_rejects_packing_and_dead_branches() -> None:
    row = _with_visual_fields(
        build_branched_sequence(
            [10],
            [([20], [21]), ([30], [31])],
            eos_id=2,
        )
    )
    with pytest.raises(ValueError, match="before cross-example"):
        project_ssmax_single_response(
            {**row, "example_ids": np.zeros(len(row["input_ids"]), dtype=np.int64)},
            source_name="cosyn_point",
            logical_split="train",
            sample_index=0,
            epoch=0,
            seed=1,
            loss_token_weighting="root_subsegments",
        )

    dead = {
        key: value.copy() if isinstance(value, np.ndarray) else value for key, value in row.items()
    }
    dead["loss_masks"][dead["subsegment_ids"] == 1] = 0
    with pytest.raises(ValueError, match="no surviving supervised target"):
        project_ssmax_single_response(
            dead,
            source_name="cosyn_point",
            logical_split="train",
            sample_index=0,
            epoch=0,
            seed=1,
            loss_token_weighting="root_subsegments",
        )


def test_unbranched_row_is_preserved_without_metadata_mutation() -> None:
    row = _with_visual_fields(
        build_packed_sequence(
            [10, 11],
            [[20, 21]],
            eos_id=2,
            loss_token_weighting="root_subsegments_root_tokens",
        )
    )
    projected, receipt = project_ssmax_single_response(
        row,
        source_name="pixmo_caption",
        logical_split="validation",
        sample_index=0,
        epoch=8,
        seed=3,
        loss_token_weighting="root_subsegments_root_tokens",
    )
    _assert_exact_model_fields(projected, row)
    assert receipt["branch_count"] == 1
    assert receipt["selected_branch_id"] is None
    assert receipt["selection_epoch"] == 0


def test_projection_uses_attend_all_prefix_and_removes_subsegments() -> None:
    row = _with_visual_fields(build_branched_sequence([10], [([20], [21]), ([30], [31])], eos_id=2))
    assert ATTEND_ALL_SUBSEGMENT_ID in row["subsegment_ids"]
    projected, _ = project_ssmax_single_response(
        row,
        source_name="cosyn_point",
        logical_split="validation",
        sample_index=0,
        epoch=0,
        seed=1,
        loss_token_weighting="root_subsegments",
    )
    assert "subsegment_ids" not in projected
    np.testing.assert_array_equal(
        projected["position_ids"], np.arange(len(projected["input_ids"]), dtype=np.int64)
    )


def test_calibration_rebuild_and_tamper_validation() -> None:
    row = _with_visual_fields(
        build_branched_sequence(
            [10],
            [([20], [21, 22]), ([30], [31, 32, 33])],
            eos_id=2,
            loss_token_weighting="root_subsegments_root_tokens",
        )
    )
    dataset = SSMaxSingleResponseDataset(
        _Dataset([row]),
        source_name="cosyn_point",
        logical_split="train",
        seed=95818,
        loss_token_weighting="root_subsegments_root_tokens",
    )
    summary = ssmax_single_response_calibration_summary(dataset, ((0, 0), (0, 1)))
    source_audit = {"path": "/audit.json", "raw_sha256": "a" * 64, "content_sha256": "b" * 64}
    selection = {
        "path": "/selection.json",
        "raw_sha256": "c" * 64,
        "content_sha256": "d" * 64,
    }
    payload: dict[str, Any] = {
        "format": SSMAX_SINGLE_RESPONSE_CALIBRATION_FORMAT,
        "version": SSMAX_SINGLE_RESPONSE_CALIBRATION_VERSION,
        "status": "ok",
        "created_at": "2026-08-20T00:00:00+00:00",
        "phase": "perception",
        "producer": {"path": "producer.py", "sha256": "e" * 64},
        "projection_implementation": {"path": "projection.py", "sha256": "f" * 64},
        "projection_contract": dataset.contract,
        "source_audit": source_audit,
        "selection_manifest": selection,
        "sources": {"cosyn_point": summary},
        "validation_preflight": {"cosyn_point": summary},
        "unprojected_sources": [],
        "projected_mean_loss_weight": {"cosyn_point": summary["mean_sum_loss_masks"]},
        "errors": [],
    }

    def canonical(value: Any) -> str:
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    payload["content_sha256"] = canonical(payload)
    validate_ssmax_single_response_calibration(
        payload,
        expected_phase="perception",
        expected_contract=dataset.contract,
        expected_source_audit=source_audit,
        expected_selection_manifest=selection,
        expected_visual_sources=("cosyn_point",),
        expected_unprojected_sources=(),
        expected_mean_loss_weight={"cosyn_point": summary["mean_sum_loss_masks"]},
    )

    payload["producer"]["sha256"] = "G" * 64
    payload["content_sha256"] = canonical(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    with pytest.raises(ValueError, match="producer reference"):
        validate_ssmax_single_response_calibration(
            payload,
            expected_phase="perception",
            expected_contract=dataset.contract,
            expected_source_audit=source_audit,
            expected_selection_manifest=selection,
            expected_visual_sources=("cosyn_point",),
            expected_unprojected_sources=(),
            expected_mean_loss_weight={"cosyn_point": summary["mean_sum_loss_masks"]},
        )
    payload["producer"]["sha256"] = "e" * 64
    payload["content_sha256"] = canonical(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    payload["sources"]["cosyn_point"]["serialized_rows_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="content SHA-256"):
        validate_ssmax_single_response_calibration(
            payload,
            expected_phase="perception",
            expected_contract=dataset.contract,
            expected_source_audit=source_audit,
            expected_selection_manifest=selection,
            expected_visual_sources=("cosyn_point",),
            expected_unprojected_sources=(),
            expected_mean_loss_weight={"cosyn_point": summary["mean_sum_loss_masks"]},
        )


@pytest.mark.parametrize(
    ("value", "error"),
    [
        ("not-a-timestamp", "created_at"),
        ("2026-08-20T00:00:00", "timezone-aware"),
    ],
)
def test_calibration_rejects_malformed_created_at(value: str, error: str) -> None:
    row = _with_visual_fields(build_packed_sequence([10], [[20]], eos_id=2))
    dataset = SSMaxSingleResponseDataset(
        _Dataset([row]),
        source_name="pixmo_caption",
        logical_split="validation",
        seed=95818,
        loss_token_weighting="root_subsegments",
    )
    summary = ssmax_single_response_calibration_summary(dataset, ((0, 0),))
    audit = {"path": "/a", "raw_sha256": "a" * 64, "content_sha256": "b" * 64}
    selection = {"path": "/s", "raw_sha256": "c" * 64, "content_sha256": "d" * 64}
    payload: dict[str, Any] = {
        "format": SSMAX_SINGLE_RESPONSE_CALIBRATION_FORMAT,
        "version": SSMAX_SINGLE_RESPONSE_CALIBRATION_VERSION,
        "status": "ok",
        "created_at": value,
        "phase": "perception",
        "producer": {"path": "p", "sha256": "e" * 64},
        "projection_implementation": {"path": "i", "sha256": "f" * 64},
        "projection_contract": dataset.contract,
        "source_audit": audit,
        "selection_manifest": selection,
        "sources": {"pixmo_caption": summary},
        "validation_preflight": {"pixmo_caption": summary},
        "unprojected_sources": [],
        "projected_mean_loss_weight": {"pixmo_caption": summary["mean_sum_loss_masks"]},
        "errors": [],
    }

    def canonical(item: Any) -> str:
        return hashlib.sha256(
            json.dumps(item, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    payload["content_sha256"] = canonical(payload)
    with pytest.raises(ValueError, match=error):
        validate_ssmax_single_response_calibration(
            payload,
            expected_phase="perception",
            expected_contract=dataset.contract,
            expected_source_audit=audit,
            expected_selection_manifest=selection,
            expected_visual_sources=("pixmo_caption",),
            expected_unprojected_sources=(),
            expected_mean_loss_weight={"pixmo_caption": summary["mean_sum_loss_masks"]},
        )
