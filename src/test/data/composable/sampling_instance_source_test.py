from pathlib import Path

from olmo_core.data.composable import ConcatAndChunkInstanceSource, InMemoryTokenSource
from olmo_core.data.composable.sampling_instance_source import SamplingInstanceSource


def test_sampling_instance_source(tmp_path: Path):
    sequence_length = 8
    source = SamplingInstanceSource(
        ConcatAndChunkInstanceSource(
            InMemoryTokenSource(list(range(64)), work_dir=tmp_path),
            sequence_length=sequence_length,
            work_dir=tmp_path,
        ),
        ConcatAndChunkInstanceSource(
            InMemoryTokenSource(list(range(64, 128)), work_dir=tmp_path),
            sequence_length=sequence_length,
            work_dir=tmp_path,
        ),
        max_instances=8,
        work_dir=tmp_path,
        seed=None,
    )
    assert source.source_sample_sizes == (4, 4)
    assert len(source) == 8

    assert list(source[0]["input_ids"]) == list(range(0, 8))
    assert list(source[4]["input_ids"]) == list(range(64, 64 + 8))


def test_sampling_instance_source_with_oversampling(tmp_path: Path):
    sequence_length = 8
    source = SamplingInstanceSource(
        ConcatAndChunkInstanceSource(
            InMemoryTokenSource(list(range(64)), work_dir=tmp_path),
            sequence_length=sequence_length,
            work_dir=tmp_path,
        ),
        ConcatAndChunkInstanceSource(
            InMemoryTokenSource(list(range(64, 128)), work_dir=tmp_path),
            sequence_length=sequence_length,
            work_dir=tmp_path,
        ),
        max_instances=24,
        work_dir=tmp_path,
        seed=None,
    )
    assert source.source_sample_sizes == (12, 12)
    assert len(source) == 24
    assert list(source[0]["input_ids"]) == list(range(0, 8))
    assert list(source[12]["input_ids"]) == list(range(64, 64 + 8))


def test_sampling_instance_source_with_random_sampling(tmp_path: Path, seed: int = 42):
    sequence_length = 8
    source = SamplingInstanceSource(
        ConcatAndChunkInstanceSource(
            InMemoryTokenSource(list(range(64)), work_dir=tmp_path),
            sequence_length=sequence_length,
            work_dir=tmp_path,
        ),
        ConcatAndChunkInstanceSource(
            InMemoryTokenSource(list(range(64, 128)), work_dir=tmp_path),
            sequence_length=sequence_length,
            work_dir=tmp_path,
        ),
        max_instances=8,
        seed=seed,
        work_dir=tmp_path,
    )
    assert source.source_sample_sizes == (4, 4)
    assert len(source) == 8

    for idx in range(4):
        for token in source[idx]["input_ids"]:
            assert 0 <= token < 64
    for idx in range(4, 8):
        for token in source[idx]["input_ids"]:
            assert 64 <= token < 128


def test_sampling_instance_source_with_random_oversampling(tmp_path: Path, seed: int = 42):
    sequence_length = 8
    source = SamplingInstanceSource(
        ConcatAndChunkInstanceSource(
            InMemoryTokenSource(list(range(64)), work_dir=tmp_path),
            sequence_length=sequence_length,
            work_dir=tmp_path,
        ),
        ConcatAndChunkInstanceSource(
            InMemoryTokenSource(list(range(64, 128)), work_dir=tmp_path),
            sequence_length=sequence_length,
            work_dir=tmp_path,
        ),
        max_instances=24,
        seed=seed,
        work_dir=tmp_path,
    )
    assert source.source_sample_sizes == (12, 12)
    assert len(source) == 24

    for idx in range(12):
        for token in source[idx]["input_ids"]:
            assert 0 <= token < 64
    for idx in range(12, 24):
        for token in source[idx]["input_ids"]:
            assert 64 <= token < 128
