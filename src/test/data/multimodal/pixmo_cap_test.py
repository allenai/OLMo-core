"""
Tests for the local-disk PixMo-Cap adapter.

Auto-skips if the local dataset path isn't present (e.g. running outside the
AI2 cluster). Successful runs read a couple of examples from disk.
"""

import os

import pytest

from olmo_core.data.multimodal.pixmo_cap import PixmoCapDatasetConfig


def _local_path_present() -> bool:
    """Whether the on-disk PixMo-Cap dataset is available."""
    return os.path.isdir(PixmoCapDatasetConfig().resolve_data_dir())


pytestmark = pytest.mark.skipif(
    not _local_path_present(),
    reason="local PixMo-Cap dataset directory not present",
)


def test_yields_one_example():
    pytest.importorskip("datasets")
    pytest.importorskip("PIL")
    ds = PixmoCapDatasetConfig(limit=1).build()
    ex = next(iter(ds))
    prompt, caption, image = ex
    assert isinstance(prompt, str) and len(prompt) > 0
    assert isinstance(caption, str) and len(caption) > 0
    assert hasattr(image, "size")
    assert len(image.size) == 2


def test_iteration_stops_at_limit():
    pytest.importorskip("datasets")
    pytest.importorskip("PIL")
    ds = PixmoCapDatasetConfig(limit=3).build()
    items = list(ds)
    assert len(items) == 3


# Note: a test_feeds_preprocessor_end_to_end end-to-end test (PixmoCapDataset →
# MultimodalPreprocessor) lives in src/test/data/multimodal/preprocessor_test.py,
# added with the training data pipeline in the next PR.
