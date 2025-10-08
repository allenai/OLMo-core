from pathlib import Path

from olmo_core.data.composable import ConcatAndChunkInstanceSource, InMemoryTokenSource
from olmo_core.data.composable.mixing_instance_source import (
    MixingInstanceSource,
    MixingInstanceSourceSpec,
)


def test_mixing_instance_source(tmp_path: Path):
    sequence_length = 8
    source = MixingInstanceSource(
        MixingInstanceSourceSpec(
            source=ConcatAndChunkInstanceSource(
                InMemoryTokenSource(list(range(64)), work_dir=tmp_path),
                sequence_length=sequence_length,
                work_dir=tmp_path,
            ),
            ratio=0.50,
        ),
        MixingInstanceSourceSpec(
            source=ConcatAndChunkInstanceSource(
                InMemoryTokenSource(list(range(64, 256)), work_dir=tmp_path),
                sequence_length=sequence_length,
                work_dir=tmp_path,
            ),
            ratio=0.50,
        ),
        work_dir=tmp_path,
    )
    assert len(source) == 16
    assert list(source[0]["input_ids"]) == list(range(0, 8))
    assert list(source[8]["input_ids"]) == list(range(64, 64 + 8))
