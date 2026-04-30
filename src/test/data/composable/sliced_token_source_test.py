from pathlib import Path

import pytest

from olmo_core.data.composable import *
from olmo_core.exceptions import OLMoConfigurationError


def test_sliced_token_source(tmp_path: Path):
    source = InMemoryTokenSource(list(range(10)), work_dir=tmp_path)
    sliced = SlicedTokenSource(source, slice(2, 7), work_dir=tmp_path)
    assert len(sliced) == 5
    assert list(sliced[:]["input_ids"]) == list(range(2, 7))


def test_sliced_token_source_negative_start(tmp_path: Path):
    source = InMemoryTokenSource(list(range(10)), work_dir=tmp_path)
    sliced = SlicedTokenSource(source, slice(-3, None), work_dir=tmp_path)
    assert len(sliced) == 3
    assert list(sliced[:]["input_ids"]) == list(range(7, 10))


def test_sliced_token_source_negative_end(tmp_path: Path):
    source = InMemoryTokenSource(list(range(10)), work_dir=tmp_path)
    sliced = SlicedTokenSource(source, slice(2, -1), work_dir=tmp_path)
    assert len(sliced) == 7
    assert list(sliced[:]["input_ids"]) == list(range(2, 9))


def test_sliced_token_source_both_negative(tmp_path: Path):
    source = InMemoryTokenSource(list(range(10)), work_dir=tmp_path)
    sliced = SlicedTokenSource(source, slice(-5, -2), work_dir=tmp_path)
    assert len(sliced) == 3
    assert list(sliced[:]["input_ids"]) == list(range(5, 8))


def test_sliced_token_source_none_start(tmp_path: Path):
    source = InMemoryTokenSource(list(range(10)), work_dir=tmp_path)
    sliced = SlicedTokenSource(source, slice(None, 5), work_dir=tmp_path)
    assert len(sliced) == 5
    assert list(sliced[:]["input_ids"]) == list(range(0, 5))


def test_sliced_token_source_none_end(tmp_path: Path):
    source = InMemoryTokenSource(list(range(10)), work_dir=tmp_path)
    sliced = SlicedTokenSource(source, slice(5, None), work_dir=tmp_path)
    assert len(sliced) == 5
    assert list(sliced[:]["input_ids"]) == list(range(5, 10))


def test_sliced_token_source_full_slice(tmp_path: Path):
    source = InMemoryTokenSource(list(range(10)), work_dir=tmp_path)
    sliced = SlicedTokenSource(source, slice(None, None), work_dir=tmp_path)
    assert len(sliced) == 10
    assert list(sliced[:]["input_ids"]) == list(range(10))


def test_sliced_token_source_empty_slice(tmp_path: Path):
    source = InMemoryTokenSource(list(range(10)), work_dir=tmp_path)
    with pytest.raises(OLMoConfigurationError, match="empty slice"):
        SlicedTokenSource(source, slice(5, 5), work_dir=tmp_path)


def test_sliced_token_source_start_at_end(tmp_path: Path):
    source = InMemoryTokenSource(list(range(10)), work_dir=tmp_path)
    with pytest.raises(OLMoConfigurationError, match="empty slice"):
        SlicedTokenSource(source, slice(10, None), work_dir=tmp_path)


def test_sliced_token_source_start_beyond_end(tmp_path: Path):
    source = InMemoryTokenSource(list(range(10)), work_dir=tmp_path)
    with pytest.raises(OLMoConfigurationError, match="empty slice"):
        SlicedTokenSource(source, slice(15, 20), work_dir=tmp_path)


def test_sliced_token_source_stop_beyond_end(tmp_path: Path):
    source = InMemoryTokenSource(list(range(10)), work_dir=tmp_path)
    sliced = SlicedTokenSource(source, slice(5, 20), work_dir=tmp_path)
    assert len(sliced) == 5
    assert list(sliced[:]["input_ids"]) == list(range(5, 10))


def test_sliced_token_source_negative_start_out_of_bounds(tmp_path: Path):
    source = InMemoryTokenSource(list(range(10)), work_dir=tmp_path)
    with pytest.raises(OLMoConfigurationError, match="out of bounds"):
        SlicedTokenSource(source, slice(-15, None), work_dir=tmp_path)


def test_sliced_token_source_step_not_one(tmp_path: Path):
    source = InMemoryTokenSource(list(range(10)), work_dir=tmp_path)
    with pytest.raises(OLMoConfigurationError, match="does not support slices with a step"):
        SlicedTokenSource(source, slice(0, 10, 2), work_dir=tmp_path)


def test_sliced_token_source_get_token_range(tmp_path: Path):
    source = InMemoryTokenSource(list(range(10)), work_dir=tmp_path)
    sliced = SlicedTokenSource(source, slice(2, 7), work_dir=tmp_path)
    # Get a range from the sliced source
    token_range = sliced.get_token_range(1, 3)
    assert list(token_range["input_ids"]) == [
        3,
        4,
    ]  # Indices 1-3 in sliced = indices 3-4 in original
