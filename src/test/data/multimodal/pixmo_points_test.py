from types import SimpleNamespace

import pytest

from olmo_core.data.multimodal import pixmo_points


class _EmptyDataset:
    def __len__(self):
        return 0

    def __getitem__(self, key):
        if key == "count":
            return []
        raise KeyError(key)


def test_pointing_dataset_configs_forward_the_requested_split(monkeypatch):
    loaded = []

    def load_split(path, split):
        loaded.append((path, split))
        return _EmptyDataset()

    monkeypatch.setattr(pixmo_points, "_load_split", load_split)

    import datasets

    monkeypatch.setattr(datasets, "concatenate_datasets", lambda sources: _EmptyDataset())
    tokenizer = object()

    pixmo_points.PixMoPointsDatasetConfig(kind="basic", split="validation").build(tokenizer)
    pixmo_points.PixMoCountDatasetConfig(split="validation").build(tokenizer)

    assert loaded == [
        (f"{pixmo_points.PIXMO_DATASETS}/points-pointing", "validation"),
        (f"{pixmo_points.PIXMO_DATASETS}/count", "validation"),
    ]


def test_points_both_mode_controls_dataset_expansion():
    dataset = object.__new__(pixmo_points.PixMoPointsDataset)
    dataset._index = [(0, [0]), (1, [0]), (2, [0])]

    dataset.config = pixmo_points.PixMoPointsDatasetConfig(both_mode="per_annotation")
    assert len(dataset) == 3

    dataset.config = pixmo_points.PixMoPointsDatasetConfig(both_mode="duplicate")
    assert len(dataset) == 6


def _build_count_example(monkeypatch, *, row, counting, index=0):
    dataset = object.__new__(pixmo_points.PixMoCountDataset)
    dataset.config = pixmo_points.PixMoCountDatasetConfig(
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
