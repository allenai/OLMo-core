from pathlib import Path

from olmo_core.data import TokenizerConfig
from olmo_core.data.composable.token_source import (
    ConcatenatedDocumentSource,
    InMemoryDocumentSource,
    InMemoryTokenSource,
)


def test_in_memory_token_source(tmp_path: Path):
    source = InMemoryTokenSource(list(range(10)), work_dir=tmp_path)
    assert len(source) == 10
    assert isinstance(source.fingerprint, str)
    assert list(source[:]["input_ids"]) == list(range(10))


def test_in_memory_document_source(tmp_path: Path):
    source = InMemoryDocumentSource(
        [1, 1, 0, 2, 2, 0, 3, 3, 0, 4, 4],
        tokenizer=TokenizerConfig(vocab_size=32_000, eos_token_id=0, pad_token_id=-1),
        work_dir=tmp_path,
    )
    assert len(source) == 11
    assert isinstance(source.fingerprint, str)
    assert list(source.get_document_offsets()) == [(0, 3), (3, 6), (6, 9), (9, 11)]


def test_concatenated_document_source(tmp_path: Path):
    source1 = ConcatenatedDocumentSource(
        InMemoryDocumentSource(
            [1, 1, 0, 2, 2, 0, 3, 3, 0, 4, 4],
            tokenizer=TokenizerConfig(vocab_size=32_000, eos_token_id=0, pad_token_id=-1),
            work_dir=tmp_path,
        ),
        InMemoryDocumentSource(
            [5, 5, 5, 0, 6, 6, 0],
            tokenizer=TokenizerConfig(vocab_size=32_000, eos_token_id=0, pad_token_id=-1),
            work_dir=tmp_path,
        ),
        work_dir=tmp_path,
    )

    source2 = source1.sources[0] + source1.sources[1]
    assert source1.fingerprint == source2.fingerprint
    assert isinstance(source2, ConcatenatedDocumentSource)

    for source in (source1, source2):
        assert source.work_dir == tmp_path / "ConcatenatedDocumentSource"
        assert list(source.get_document_offsets()) == [
            (0, 3),
            (3, 6),
            (6, 9),
            (9, 11),
            (11, 15),
            (15, 18),
        ]
        assert list(source[15:18]["input_ids"]) == [6, 6, 0]
        assert list(source[13:18]["input_ids"]) == [5, 0, 6, 6, 0]
        assert list(source[13:17]["input_ids"]) == [5, 0, 6, 6]

    source = source1 + source2
    assert isinstance(source, ConcatenatedDocumentSource)
    assert len(source.sources) == 4
