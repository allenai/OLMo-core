from pathlib import Path

import pytest

from olmo_core.data.composable import *
from olmo_core.exceptions import OLMoConfigurationError


def test_mixing_instance_source(tmp_path: Path):
    sequence_length = 8
    source = MixingInstanceSource(
        MixingInstanceSource.Spec(
            source=ConcatAndChunkInstanceSource(
                InMemoryTokenSource(list(range(64)), work_dir=tmp_path),
                sequence_length=sequence_length,
                work_dir=tmp_path,
            ),
            ratio=0.50,
        ),
        MixingInstanceSource.Spec(
            source=ConcatAndChunkInstanceSource(
                InMemoryTokenSource(list(range(64, 256)), work_dir=tmp_path),
                sequence_length=sequence_length,
                work_dir=tmp_path,
            ),
            ratio=0.50,
        ),
        work_dir=tmp_path,
        seed=None,
    )
    assert len(source) == 16
    assert list(source[0]["input_ids"]) == list(range(0, 8))
    assert list(source[8]["input_ids"]) == list(range(64, 64 + 8))


def test_mixing_instance_source_with_bad_constraints(tmp_path: Path):
    sequence_length = 8

    with pytest.raises(OLMoConfigurationError, match="Unable to meet target size"):
        MixingInstanceSource(
            MixingInstanceSource.Spec(
                source=ConcatAndChunkInstanceSource(
                    InMemoryTokenSource(list(range(64)), work_dir=tmp_path),
                    sequence_length=sequence_length,
                    work_dir=tmp_path,
                ),
                ratio=0.50,
            ),
            MixingInstanceSource.Spec(
                source=ConcatAndChunkInstanceSource(
                    InMemoryTokenSource(list(range(64, 128)), work_dir=tmp_path),
                    sequence_length=sequence_length,
                    work_dir=tmp_path,
                ),
                ratio=0.50,
            ),
            work_dir=tmp_path,
            num_tokens=150,
        )
