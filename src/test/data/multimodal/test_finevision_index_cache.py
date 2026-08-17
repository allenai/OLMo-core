"""Tests for FineVision filtered-row index caching."""

from __future__ import annotations

import numpy as np

from olmo_core.data.multimodal.finevision import (
    FineVisionDatasetConfig,
    finevision_index_cache_path,
    load_finevision_index_cache,
    save_finevision_index_cache,
)


def test_finevision_index_cache_roundtrip_filtered(tmp_path):
    config = FineVisionDatasetConfig(
        config_name="test_config",
        root=str(tmp_path),
        index_cache_dir=str(tmp_path / "cache"),
        require_single_image=True,
        max_rows=10,
    )
    table_rows = 100
    positions = np.array([1, 3, 5, 7], dtype=np.int64)

    assert finevision_index_cache_path(config, table_rows) is not None
    assert load_finevision_index_cache(config, table_rows) == (False, None)

    save_finevision_index_cache(config, table_rows, positions)
    hit, cached = load_finevision_index_cache(config, table_rows)
    assert hit
    np.testing.assert_array_equal(cached, positions)


def test_finevision_index_cache_roundtrip_full_table(tmp_path):
    config = FineVisionDatasetConfig(
        config_name="test_config",
        root=str(tmp_path),
        index_cache_dir=str(tmp_path / "cache"),
    )
    table_rows = 42

    save_finevision_index_cache(config, table_rows, None)
    hit, cached = load_finevision_index_cache(config, table_rows)
    assert hit
    assert cached is None


def test_finevision_index_cache_miss_on_row_count_change(tmp_path):
    config = FineVisionDatasetConfig(
        config_name="test_config",
        root=str(tmp_path),
        index_cache_dir=str(tmp_path / "cache"),
    )
    save_finevision_index_cache(config, 10, np.array([0, 2], dtype=np.int64))
    assert load_finevision_index_cache(config, 11) == (False, None)


def test_finevision_index_cache_disabled(tmp_path):
    config = FineVisionDatasetConfig(
        config_name="test_config",
        root=str(tmp_path),
        index_cache_dir="",
    )
    assert finevision_index_cache_path(config, 10) is None
