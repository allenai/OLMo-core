from types import SimpleNamespace

import numpy as np
import pytest

from olmo_core.data.multimodal import pixmo_points
from olmo_core.nn.vision.molmo2_tokens import N_PATCHES_SQ, PATCH_DIM, Molmo2TokenIds


class _RowsDataset:
    def __init__(self, rows=(), fingerprint="fake-arrow-v1"):
        self.rows = list(rows)
        self._fingerprint = fingerprint
        self.column_names = sorted({key for row in self.rows for key in row})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self.rows]
        return self.rows[key]

    def select_columns(self, columns):
        return _RowsDataset(
            [{column: row[column] for column in columns} for row in self.rows],
            self._fingerprint,
        )


def _concatenate_rows(sources):
    return _RowsDataset(
        [row for source in sources for row in source.rows],
        "+".join(source._fingerprint for source in sources),
    )


def test_pointing_dataset_configs_forward_the_requested_split(monkeypatch):
    loaded = []

    def load_split(path, split, *, require_split):
        loaded.append((path, split, require_split))
        return _RowsDataset()

    monkeypatch.setattr(pixmo_points, "_load_split", load_split)

    import datasets

    monkeypatch.setattr(datasets, "concatenate_datasets", _concatenate_rows)
    tokenizer = object()

    pixmo_points.PixMoPointsDatasetConfig(
        kind="basic", split="validation", require_split=True
    ).build(tokenizer)
    pixmo_points.PixMoCountDatasetConfig(split="validation", require_split=True).build(tokenizer)
    pixmo_points.CoSynPointDatasetConfig(split="validation", require_split=True).build(tokenizer)

    assert loaded == [
        (f"{pixmo_points.PIXMO_DATASETS}/points-pointing", "validation", True),
        (f"{pixmo_points.PIXMO_DATASETS}/count", "validation", True),
        (f"{pixmo_points.PIXMO_DATASETS}/cosyn-point", "validation", True),
    ]


def test_load_split_fails_closed_when_the_named_split_is_missing(monkeypatch):
    from olmo_core.data.multimodal import dataset_compat

    unsplit = _RowsDataset()
    monkeypatch.setattr(dataset_compat, "load_from_disk_compat", lambda _path: {"train": unsplit})

    with pytest.raises(ValueError, match="lacks required split 'validation'"):
        pixmo_points._load_split("unused", "validation", require_split=True)

    assert pixmo_points._load_split("unused", "validation", require_split=False) == {
        "train": unsplit
    }


def test_pointing_dataset_configs_preserve_legacy_non_strict_defaults():
    assert not pixmo_points.PixMoPointsDatasetConfig().require_split
    assert not pixmo_points.PixMoCountDatasetConfig().require_split
    assert not pixmo_points.CoSynPointDatasetConfig().require_split


def test_points_both_mode_controls_dataset_expansion():
    dataset = object.__new__(pixmo_points.PixMoPointsDataset)
    dataset._index = [(0, [0]), (1, [0]), (2, [0])]

    dataset.config = pixmo_points.PixMoPointsDatasetConfig(both_mode="per_annotation")
    assert len(dataset) == 3

    dataset.config = pixmo_points.PixMoPointsDatasetConfig(both_mode="duplicate")
    assert len(dataset) == 6


def test_points_blank_label_filter_is_strict_perception_only():
    dataset = object.__new__(pixmo_points.PixMoPointsDataset)
    dataset._data = {
        "count": [[1, 1]],
        "label": [["", "cow"]],
    }

    dataset.config = pixmo_points.PixMoPointsDatasetConfig(require_split=False)
    assert dataset._build_sub_index() == [(0, [0, 1])]

    dataset.config = pixmo_points.PixMoPointsDatasetConfig(require_split=True)
    assert dataset._build_sub_index() == [(0, [1])]
    assert dataset.annotation_filter_stats["blank_labels"] == 1


def _build_count_example(monkeypatch, *, row, counting="both", mode="grounded", index=0):
    dataset = object.__new__(pixmo_points.PixMoCountDataset)
    dataset.config = pixmo_points.PixMoCountDatasetConfig(
        mode=mode,
        counting=counting,
        message_format="document",
    )
    dataset.tokenizer = object()
    dataset._data = [row]

    monkeypatch.setattr(
        pixmo_points,
        "_open_image",
        lambda _path: SimpleNamespace(size=(100, 100)),
    )

    def capture_branches(_tokenizer, _image, build_branches, *, rng, **_kwargs):
        return build_branches(rng)

    monkeypatch.setattr(pixmo_points, "_build_example", capture_branches)
    return dataset.get(index)


def test_scalar_count_is_one_example_per_row_and_uses_the_same_declared_target(monkeypatch):
    train = _build_count_example(
        monkeypatch,
        row={
            "image": "unused",
            "label": "cows",
            "count": 9,
            # Scalar mode deliberately ignores grounding annotations.
            "points": {"x": [10.0, 20.0], "y": [30.0, 40.0]},
        },
        mode="scalar_count",
    )
    validation = _build_count_example(
        monkeypatch,
        row={
            "image": "unused",
            "label": "cows",
            "count": 9,
            "points": {"x": [], "y": []},
        },
        mode="scalar_count",
    )

    assert train == validation == [("How many cows are there?", "9")]

    dataset = object.__new__(pixmo_points.PixMoCountDataset)
    dataset.config = pixmo_points.PixMoCountDatasetConfig(
        mode="scalar_count",
        message_format="document",
    )
    dataset._n = 3
    assert len(dataset) == 3


def test_scalar_count_rejects_chat_serialization():
    with pytest.raises(ValueError, match="requires message_format='document'"):
        pixmo_points.PixMoCountDatasetConfig(mode="scalar_count").build(object())


@pytest.mark.parametrize(
    ("counting", "index"),
    [(True, 0), (False, 0), ("both", 0), ("both", 1)],
)
def test_pixmo_count_uses_declared_count_when_validation_points_are_absent(
    monkeypatch, counting, index
):
    branches = _build_count_example(
        monkeypatch,
        row={
            "image": "unused",
            "label": "cows",
            "count": 8,
            "points": {"x": [], "y": []},
        },
        counting=counting,
        index=index,
    )

    assert branches == [("How many cows are there?", "8")]


@pytest.mark.parametrize(
    ("counting", "index", "expected_style"),
    [
        (True, 0, "Counting the "),
        (False, 0, '<points coords="'),
        ("both", 0, "Counting the "),
        ("both", 1, '<points coords="'),
    ],
)
def test_pixmo_count_keeps_point_derived_answers(monkeypatch, counting, index, expected_style):
    branches = _build_count_example(
        monkeypatch,
        row={
            "image": "unused",
            "label": "cows",
            # Deliberately disagree with the annotations: valid point-derived modes
            # must continue to use the two points, not this declared count.
            "count": 9,
            "points": {"x": [10.0, 20.0], "y": [30.0, 40.0]},
        },
        counting=counting,
        index=index,
    )

    answer = branches[0][1]
    assert answer.startswith(expected_style)
    if counting is True or (counting == "both" and index == 0):
        assert answer.endswith("shows a total of 2.")
    assert "total of 9" not in answer


def test_pixmo_count_keeps_zero_point_training_answer(monkeypatch):
    branches = _build_count_example(
        monkeypatch,
        row={
            "image": "unused",
            "label": "ties",
            "count": 0,
            "points": {"x": [], "y": []},
        },
        counting=True,
    )

    assert branches[0][1] == "There are none."


def test_points_content_fingerprint_binds_raw_source_config_and_derived_index(monkeypatch):
    row = {
        "image": "unused",
        "label": ["cow"],
        "points": [[{"x": 10.0, "y": 20.0}]],
        "count": [1],
        "collection_method": ["pointing"],
    }
    raw_fingerprint = ["raw-points-v1"]

    def load_split(_path, _split, *, require_split):
        assert require_split
        return _RowsDataset([row], raw_fingerprint[0])

    monkeypatch.setattr(pixmo_points, "_load_split", load_split)
    import datasets

    monkeypatch.setattr(datasets, "concatenate_datasets", _concatenate_rows)
    config = pixmo_points.PixMoPointsDatasetConfig(
        kind="basic", max_sequence_length=2560, require_split=True
    )
    first = config.build(object())
    second = config.build(object())
    assert first.content_fingerprint == second.content_fingerprint
    assert len(first.content_fingerprint) == 64
    assert first.content_fingerprint_version == "pixmo-perception-adapter-v1"

    changed_index = pixmo_points.PixMoPointsDatasetConfig(
        kind="basic",
        max_points=0,
        max_sequence_length=2560,
        require_split=True,
    ).build(object())
    assert changed_index.content_fingerprint != first.content_fingerprint

    raw_fingerprint[0] = "raw-points-v2"
    changed_raw = config.build(object())
    assert changed_raw.content_fingerprint != first.content_fingerprint


def _dataset_for_validation(dataset_type, rows):
    dataset = object.__new__(dataset_type)
    dataset._data = _RowsDataset(rows)
    dataset._annotations_validated = False
    if dataset_type is pixmo_points.PixMoPointsDataset:
        dataset._index = [
            (row_index, list(range(len(row["label"])))) for row_index, row in enumerate(rows)
        ]
    return dataset


def test_required_annotation_validators_accept_well_formed_rows():
    points = _dataset_for_validation(
        pixmo_points.PixMoPointsDataset,
        [
            {
                "image": "unused",
                "label": ["cow"],
                "points": [[{"x": 10.0, "y": 20.0}]],
                "count": [1],
                "collection_method": ["pointing"],
            }
        ],
    )
    count = _dataset_for_validation(
        pixmo_points.PixMoCountDataset,
        [
            {
                "image": "unused",
                "label": "cows",
                "count": 8,
                "points": {"x": [], "y": []},
            }
        ],
    )
    cosyn = _dataset_for_validation(
        pixmo_points.CoSynPointDataset,
        [
            {
                "image": "unused",
                "questions": ["Where is the cow?"],
                "answer_points": [{"x": [10.0], "y": [20.0]}],
                "names": ["Cow"],
            }
        ],
    )

    points.validate_required_annotations()
    count.validate_required_annotations()
    cosyn.validate_required_annotations()
    assert points._annotations_validated
    assert count._annotations_validated
    assert cosyn._annotations_validated


@pytest.mark.parametrize(
    ("dataset_type", "row", "error"),
    [
        (
            pixmo_points.PixMoPointsDataset,
            {
                "image": "unused",
                "label": ["cow"],
                "points": [[{"x": 103.0, "y": 20.0}]],
                "count": [1],
                "collection_method": ["pointing"],
            },
            "clamp tolerance \\[-2, 102\\]",
        ),
        (
            pixmo_points.PixMoCountDataset,
            {
                "image": "unused",
                "label": "cows",
                "count": 8,
                "points": {"x": [1.0], "y": []},
            },
            "same length",
        ),
        (
            pixmo_points.CoSynPointDataset,
            {
                "image": "unused",
                "questions": ["Where?"],
                "answer_points": [],
                "names": ["Cow"],
            },
            "equally sized",
        ),
    ],
)
def test_required_annotation_validators_reject_malformed_rows(dataset_type, row, error):
    dataset = _dataset_for_validation(dataset_type, [row])

    with pytest.raises(ValueError, match=error):
        dataset.validate_required_annotations()


def _serialized_example(length=6):
    token_ids = Molmo2TokenIds()
    input_ids = np.array([token_ids.im_patch_id, *range(10, 10 + length - 1)], dtype=np.int64)
    token_type_ids = np.isin(input_ids, list(token_ids.image_token_ids)).astype(np.int64)
    return {
        "input_ids": input_ids,
        "labels": np.arange(length, dtype=np.int64),
        "loss_masks": np.array([0.0, *([1.0] * (length - 1))], dtype=np.float32),
        "position_ids": np.arange(length, dtype=np.int64),
        "token_type_ids": token_type_ids,
        "images": np.zeros((1, N_PATCHES_SQ, PATCH_DIM), dtype=np.float32),
        "pooled_patches_idx": np.array([[0, 1, 2, 3]], dtype=np.int64),
    }


def test_serialized_examples_are_safely_bounded_before_packing():
    bounded = pixmo_points._finalize_example(
        _serialized_example(),
        strict_validation=True,
        max_sequence_length=4,
        max_crops=8,
        high_res_max_crops=24,
        p_high_res=0.0,
        loss_token_weighting="root_subsegments",
        token_ids=Molmo2TokenIds(),
    )

    assert len(bounded["input_ids"]) == 4
    assert len(bounded["labels"]) == 4
    assert float(bounded["loss_masks"].sum()) == 3.0
    assert bounded["metadata"] == {"original_length": 6, "truncated": True}


def test_serialized_examples_preserve_legacy_non_strict_behavior():
    example = _serialized_example()
    finalized = pixmo_points._finalize_example(
        example,
        strict_validation=False,
        max_sequence_length=4,
        max_crops=8,
        high_res_max_crops=24,
        p_high_res=0.0,
        loss_token_weighting="root_subsegments",
        token_ids=Molmo2TokenIds(),
    )

    assert finalized is example
    assert len(finalized["input_ids"]) == 6
    assert "metadata" not in finalized


@pytest.mark.parametrize("failure", ["pooled_index", "image_token_parity", "token_types"])
def test_serialized_geometry_validation_fails_before_packing(failure):
    example = _serialized_example()
    if failure == "pooled_index":
        example["pooled_patches_idx"][0, 0] = N_PATCHES_SQ
        error = "out-of-range"
    elif failure == "image_token_parity":
        example["input_ids"][0] = 5
        example["token_type_ids"][0] = 0
        error = "pooled rows"
    else:
        example["token_type_ids"][0] = 0
        error = "do not exactly mark"

    with pytest.raises(ValueError, match=error):
        pixmo_points._finalize_example(
            example,
            strict_validation=True,
            max_sequence_length=2560,
            max_crops=8,
            high_res_max_crops=24,
            p_high_res=0.0,
            loss_token_weighting="root_subsegments",
            token_ids=Molmo2TokenIds(),
        )
