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
