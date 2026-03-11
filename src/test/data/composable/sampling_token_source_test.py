from pathlib import Path

import pytest

from olmo_core.data.composable import *


def test_sampling_token_source(tmp_path: Path):
    source = SamplingTokenSource(
        InMemoryTokenSource(list(range(10)), work_dir=tmp_path),
        InMemoryTokenSource(list(range(10, 20)), work_dir=tmp_path),
        max_tokens=16,
        work_dir=tmp_path,
        seed=None,
    )
    assert source.num_tokens == 16
    assert list(source[:]["input_ids"]) == list(range(8)) + list(range(10, 18))
    assert list(source[6:10]["input_ids"]) == [6, 7, 10, 11]


def test_sampling_token_source_with_mixing_source(tmp_path: Path):
    # Down-sampled mixing source.
    mixing_source = MixingTokenSource(
        MixingTokenSource.Spec(
            source=InMemoryTokenSource(list(range(12)), work_dir=tmp_path),
            ratio=0.25,
        ),
        MixingTokenSource.Spec(
            source=InMemoryTokenSource(list(range(12, 24)), work_dir=tmp_path),
            ratio=0.75,
        ),
        work_dir=tmp_path,
        num_tokens=16,
        seed=None,
    )

    # Then down-sample again.
    source = SamplingTokenSource(
        mixing_source,
        max_tokens=8,
        work_dir=tmp_path,
        seed=None,
    )
    assert source.num_tokens == 8
    assert list(source[:]["input_ids"]) == [0, 1, 12, 13, 14, 15, 16, 17]

    # Or up-sample.
    source = SamplingTokenSource(
        mixing_source,
        max_tokens=32,
        work_dir=tmp_path,
        seed=None,
    )
    assert source.num_tokens == 32
    assert list(source[:]["input_ids"]) == list(range(4)) * 2 + list(range(12, 24)) * 2


def test_sampling_token_source_with_sampling_token_source(tmp_path):
    # Create an up-sampled source, 8->12 tokens.
    up_sampled_source = InMemoryTokenSource(list(range(8)), work_dir=tmp_path).resize(
        1.5, seed=None
    )
    assert len(up_sampled_source) == 12
    assert list(up_sampled_source[:]["input_ids"]) == list(range(8)) + list(range(4))

    # Then down-sample from the up-sampled source.
    source = SamplingTokenSource(
        up_sampled_source,
        max_tokens=9,
        work_dir=tmp_path,
        seed=None,
    )
    assert source.num_tokens == 9
    assert list(source[:]["input_ids"]) == list(range(6)) + list(range(3))

    # Or up-sample again.
    source = SamplingTokenSource(
        up_sampled_source,
        max_tokens=18,
        work_dir=tmp_path,
        seed=None,
    )
    assert source.num_tokens == 18
    assert list(source[:]["input_ids"]) == list(range(8)) + list(range(4)) + list(range(4)) + list(
        range(2)
    )


def test_sampling_token_source_with_repetition(tmp_path: Path):
    source = SamplingTokenSource(
        InMemoryTokenSource(list(range(10)), work_dir=tmp_path),
        InMemoryTokenSource(list(range(10, 20)), work_dir=tmp_path),
        max_tokens=24,
        work_dir=tmp_path,
        seed=None,
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
