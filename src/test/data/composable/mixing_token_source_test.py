from pathlib import Path

from olmo_core.data.composable import InMemoryTokenSource
from olmo_core.data.composable.mixing_token_source import MixingTokenSource


def test_mixing_token_source(tmp_path: Path):
    source = MixingTokenSource(
        MixingTokenSource.Spec(
            source=InMemoryTokenSource(list(range(8)), work_dir=tmp_path),
            ratio=0.50,
        ),
        MixingTokenSource.Spec(
            source=InMemoryTokenSource(list(range(8, 24)), work_dir=tmp_path),
            ratio=0.50,
        ),
        work_dir=tmp_path,
        seed=None,
    )
    assert len(source) == 16
    assert list(source[:]["input_ids"]) == list(range(0, 8)) + list(range(8, 16))


def test_mixing_token_source_with_repetition(tmp_path: Path):
    source = MixingTokenSource(
        MixingTokenSource.Spec(
            source=InMemoryTokenSource(list(range(8)), work_dir=tmp_path),
            ratio=0.50,
            max_repetition_factor=2.0,
        ),
        MixingTokenSource.Spec(
            source=InMemoryTokenSource(list(range(8, 24)), work_dir=tmp_path),
            ratio=0.50,
        ),
        work_dir=tmp_path,
        seed=None,
    )
    assert len(source) == 24
    assert list(source[:]["input_ids"]) == list(range(0, 8)) + list(range(0, 4)) + list(
        range(8, 20)
    )
