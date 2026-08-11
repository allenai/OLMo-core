import copy
import hashlib

import numpy as np
import pytest
import torch

from olmo_core.eval import (
    MultimodalBlankImageEvaluator,
    MultimodalFixedValidationDataset,
    MultimodalLMEvaluator,
    MultimodalMatchedWrongImageDataset,
    build_matched_wrong_image_pairing,
    matched_wrong_image_pairing_sha256,
    validate_matched_wrong_image_pairing,
)
from olmo_core.exceptions import OLMoConfigurationError


def _make_evaluator() -> MultimodalLMEvaluator:
    return MultimodalLMEvaluator(
        name="pixmo-cap-validation",
        batches=iter([]),
        device=torch.device("cpu"),
    )


def test_multimodal_lm_evaluator_normalizes_summed_weighted_loss():
    evaluator = _make_evaluator()
    batch = {
        "labels": torch.tensor([[1, 2, 3, -100]]),
        "loss_masks": torch.tensor([[0.0, 1.0, 0.5, 1.0]]),
    }
    # Weighted token losses: 2 * 1.0 + 4 * 0.5 = 4.0. The ignored final label
    # contributes neither loss nor denominator weight.
    evaluator.update_metrics(batch, ce_loss=torch.tensor(4.0), logits=None)

    metrics = evaluator.compute_metrics()

    torch.testing.assert_close(metrics["CE loss"], torch.tensor(4.0 / 1.5))
    torch.testing.assert_close(metrics["PPL"], torch.exp(torch.tensor(4.0 / 1.5)))


def test_multimodal_lm_evaluator_accumulates_by_loss_weight():
    evaluator = _make_evaluator()
    evaluator.update_metrics(
        {"labels": torch.tensor([[1]]), "loss_masks": torch.tensor([[1.0]])},
        ce_loss=torch.tensor(2.0),
        logits=None,
    )
    evaluator.update_metrics(
        {"labels": torch.tensor([[1]]), "loss_masks": torch.tensor([[3.0]])},
        ce_loss=torch.tensor(12.0),
        logits=None,
    )

    metrics = evaluator.compute_metrics()

    torch.testing.assert_close(metrics["CE loss"], torch.tensor(3.5))


def test_multimodal_blank_image_control_changes_only_images():
    images = torch.arange(6, dtype=torch.float32).reshape(1, 1, 2, 3)
    batch = {
        "images": images,
        "input_ids": torch.tensor([[1, 2]]),
        "labels": torch.tensor([[2, -100]]),
        "loss_masks": torch.tensor([[1.0, 0.0]]),
    }
    evaluator = MultimodalBlankImageEvaluator(
        name="blank-image",
        batches=[batch],
        device=torch.device("cpu"),
    )

    [transformed] = list(evaluator)

    torch.testing.assert_close(transformed["images"], torch.zeros_like(images))
    torch.testing.assert_close(transformed["input_ids"], batch["input_ids"])
    torch.testing.assert_close(batch["images"], images)


class _MutableMultimodalDataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def get(self, index, epoch=0):
        assert epoch == 0
        return self.rows[index]


def _row(index: int, *, pooled=(0, 1), image_value=None):
    value = float(index if image_value is None else image_value)
    return {
        "images": np.full((1, 2, 3), value, dtype=np.float32),
        "pooled_patches_idx": np.asarray([pooled], dtype=np.int64),
        "input_ids": np.asarray([10, 20, 30 + index], dtype=np.int64),
        "labels": np.asarray([-100, 20, 30 + index], dtype=np.int64),
        "loss_masks": np.asarray([0.0, 1.0, 1.0], dtype=np.float32),
    }


def _content_ids(count: int):
    return tuple(hashlib.sha256(f"image-{index}".encode()).hexdigest() for index in range(count))


def test_matched_wrong_image_pairing_changes_only_images_and_is_order_independent():
    rows = [_row(index) for index in range(8)]
    dataset = _MutableMultimodalDataset(rows)
    payload = build_matched_wrong_image_pairing(
        dataset,
        recipient_count=6,
        seed=6198,
        content_ids=_content_ids(len(rows)),
    )
    pairing_sha256 = matched_wrong_image_pairing_sha256(payload)
    first = MultimodalMatchedWrongImageDataset(
        dataset, pairing=payload, pairing_sha256=pairing_sha256
    )
    second = MultimodalMatchedWrongImageDataset(
        dataset, pairing=payload, pairing_sha256=pairing_sha256
    )
    fixed = MultimodalFixedValidationDataset(
        dataset, pairing=payload, pairing_sha256=pairing_sha256
    )

    assert first.donor_indices == second.donor_indices
    assert first.pairing_sha256 == second.pairing_sha256
    assert len(set(first.donor_indices)) == len(first)

    # Accessing rank-like disjoint slices, and accessing them in reverse, cannot alter pairing.
    for local_index in [4, 2, 0, 5, 3, 1]:
        recipient_index = first.recipient_indices[local_index]
        donor_index = first.donor_indices[local_index]
        assert donor_index != recipient_index
        transformed = first.get(local_index, epoch=0)
        recipient = rows[recipient_index]
        donor = rows[donor_index]
        np.testing.assert_array_equal(transformed["images"], donor["images"])
        assert not np.array_equal(transformed["images"], recipient["images"])
        for field_name in set(recipient) - {"images"}:
            np.testing.assert_array_equal(transformed[field_name], recipient[field_name])
        for field_name in recipient:
            np.testing.assert_array_equal(fixed.get(local_index)[field_name], recipient[field_name])


def test_matched_wrong_image_pairing_requires_exact_pooling_geometry_and_distinct_content():
    rows = [
        _row(0, pooled=(0, 1)),
        _row(1, pooled=(0, 1)),
        _row(2, pooled=(1, 0)),
        _row(3, pooled=(1, 0)),
        _row(4, pooled=(0, 1)),
        _row(5, pooled=(0, 1)),
        _row(6, pooled=(1, 0)),
        _row(7, pooled=(1, 0)),
    ]
    content_ids = list(_content_ids(len(rows)))
    # Rows 0 and 1 deliberately claim the same underlying source image. They may not pair.
    content_ids[1] = content_ids[0]
    dataset = _MutableMultimodalDataset(rows)
    payload = build_matched_wrong_image_pairing(
        dataset, recipient_count=4, seed=7, content_ids=content_ids
    )
    paired = MultimodalMatchedWrongImageDataset(
        dataset,
        pairing=payload,
        pairing_sha256=matched_wrong_image_pairing_sha256(payload),
    )

    for recipient_index, donor_index in zip(paired.recipient_indices, paired.donor_indices):
        assert content_ids[recipient_index] != content_ids[donor_index]
        np.testing.assert_array_equal(
            rows[recipient_index]["pooled_patches_idx"],
            rows[donor_index]["pooled_patches_idx"],
        )
        assert rows[recipient_index]["images"].shape == rows[donor_index]["images"].shape


def test_matched_wrong_image_pairing_reports_deterministic_selection_coverage():
    rows = [
        _row(0, pooled=(0, 1), image_value=0),
        _row(1, pooled=(0, 1), image_value=0),
        _row(2, pooled=(0, 1), image_value=2),
        _row(3, pooled=(1, 0), image_value=3),
        _row(4, pooled=(2, 0), image_value=4),
        _row(5, pooled=(2, 0), image_value=5),
        _row(6, pooled=(2, 0), image_value=6),
    ]
    content_ids = list(_content_ids(len(rows)))
    content_ids[1] = content_ids[0]
    dataset = _MutableMultimodalDataset(rows)

    payload = build_matched_wrong_image_pairing(
        dataset,
        recipient_count=4,
        seed=11,
        content_ids=content_ids,
    )
    repeated = build_matched_wrong_image_pairing(
        dataset,
        recipient_count=4,
        seed=11,
        content_ids=content_ids,
    )

    assert payload == repeated
    assert payload["version"] == 2
    coverage = payload["coverage"]
    assert coverage == repeated["coverage"]
    assert coverage["dataset_count"] == 7
    assert coverage["eligible_count"] == 5
    assert coverage["excluded_count"] == 2
    assert coverage["selected_recipient_count"] == 4
    assert coverage["geometry_count"] == 3
    assert coverage["eligible_geometry_count"] == 2
    assert coverage["selected_geometry_count"] == 2
    histogram = coverage["geometry_histogram"]
    assert sorted(
        (
            entry["dataset_count"],
            entry["distinct_count"],
            entry["eligible_count"],
            entry["excluded_count"],
        )
        for entry in histogram
    ) == [(1, 1, 0, 1), (3, 2, 2, 1), (3, 3, 3, 0)]
    assert sum(entry["selected_recipient_count"] for entry in histogram) == 4
    assert all(
        entry["selected_recipient_count"] == entry["selected_donor_count"] for entry in histogram
    )

    inconsistent = copy.deepcopy(payload)
    inconsistent["coverage"]["eligible_count"] += 1
    with pytest.raises(OLMoConfigurationError, match="coverage totals are inconsistent"):
        validate_matched_wrong_image_pairing(inconsistent)

    unordered = copy.deepcopy(payload)
    unordered["coverage"]["geometry_histogram"].reverse()
    with pytest.raises(OLMoConfigurationError, match="not canonically ordered"):
        validate_matched_wrong_image_pairing(unordered)


def test_matched_wrong_image_pairing_fails_closed_on_insufficient_pairs_and_drift():
    singleton_geometry = _MutableMultimodalDataset([_row(0, pooled=(0, 1)), _row(1, pooled=(1, 0))])
    with pytest.raises(OLMoConfigurationError, match="select enough validation rows"):
        build_matched_wrong_image_pairing(
            singleton_geometry,
            recipient_count=2,
            seed=0,
            content_ids=_content_ids(2),
        )

    rows = [_row(index) for index in range(4)]
    dataset = _MutableMultimodalDataset(rows)
    payload = build_matched_wrong_image_pairing(
        dataset, recipient_count=2, seed=0, content_ids=_content_ids(4)
    )
    paired = MultimodalMatchedWrongImageDataset(
        dataset,
        pairing=payload,
        pairing_sha256=matched_wrong_image_pairing_sha256(payload),
    )
    drift_index = paired.recipient_indices[0]
    rows[drift_index]["labels"][1] = 999
    with pytest.raises(OLMoConfigurationError, match=f"row {drift_index} drifted"):
        paired.get(0)
    with pytest.raises(OLMoConfigurationError, match="pinned to source epoch 0"):
        paired.get(1, epoch=1)
