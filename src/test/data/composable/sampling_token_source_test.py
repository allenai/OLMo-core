from pathlib import Path

import pytest

from olmo_core.data.composable import InMemoryTokenSource
from olmo_core.data.composable.sampling_token_source import SamplingTokenSource


def test_sampling_token_source(tmp_path: Path):
    source = SamplingTokenSource(
        InMemoryTokenSource(list(range(10)), work_dir=tmp_path),
        InMemoryTokenSource(list(range(10, 20)), work_dir=tmp_path),
        max_tokens=16,
        work_dir=tmp_path,
    )
    assert source.num_tokens == 16
    assert list(source[:]["input_ids"]) == list(range(8)) + list(range(10, 18))
    assert list(source[6:10]["input_ids"]) == [6, 7, 10, 11]


def test_sampling_token_source_with_repetition(tmp_path: Path):
    source = SamplingTokenSource(
        InMemoryTokenSource(list(range(10)), work_dir=tmp_path),
        InMemoryTokenSource(list(range(10, 20)), work_dir=tmp_path),
        max_tokens=24,
        work_dir=tmp_path,
        allow_repetition=True,
    )
    assert source.num_tokens == 24
    assert list(source[:]["input_ids"]) == list(range(10)) + list(range(0, 2)) + list(
        range(10, 20)
    ) + list(range(10, 12))


@pytest.mark.parametrize("seed", [0, 542, 1234])
def test_sampling_token_source_with_random_sampling(tmp_path: Path, seed: int):
    source = SamplingTokenSource(
        InMemoryTokenSource(list(range(10)), work_dir=tmp_path),
        InMemoryTokenSource(list(range(10, 20)), work_dir=tmp_path),
        max_tokens=16,
        work_dir=tmp_path,
        seed=seed,
    )
    assert source.num_tokens == 16
    assert 0 <= list(source[:8]["input_ids"])[0] < 4
    assert 7 <= list(source[:8]["input_ids"])[-1] < 10
    assert 10 <= list(source[8:]["input_ids"])[0] < 14
    assert 17 <= list(source[8:]["input_ids"])[-1] < 20
