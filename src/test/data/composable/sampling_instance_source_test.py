from pathlib import Path

from olmo_core.data.composable import *


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


def test_sampling_instance_source_with_mixing_source(tmp_path: Path):
    sequence_length = 8
    source = SamplingInstanceSource(
        MixingInstanceSource(
            MixingInstanceSourceSpec(
                source=ConcatAndChunkInstanceSource(
                    InMemoryTokenSource(list(range(64)), work_dir=tmp_path),
                    sequence_length=sequence_length,
                    work_dir=tmp_path,
                ),
                ratio=0.25,
            ),
            MixingInstanceSourceSpec(
                source=ConcatAndChunkInstanceSource(
                    InMemoryTokenSource(list(range(64, 128)), work_dir=tmp_path),
                    sequence_length=sequence_length,
                    work_dir=tmp_path,
                ),
                ratio=0.75,
            ),
            work_dir=tmp_path,
            num_instances=8,
        ),
        max_instances=4,
        work_dir=tmp_path,
        seed=None,
    )
    # Should unwind the mix's sources and maintain the sampling ratios.
    assert source.source_sample_sizes == (1, 3)
    assert len(source) == 4


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
