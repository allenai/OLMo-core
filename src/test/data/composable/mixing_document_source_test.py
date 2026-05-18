from pathlib import Path

from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import InMemoryDocumentSource
from olmo_core.data.composable.mixing_document_source import MixingDocumentSource


def test_mixing_token_source(tmp_path: Path):
    tokenizer = TokenizerConfig(vocab_size=32_000, eos_token_id=0, pad_token_id=-1)
    source = MixingDocumentSource(
        MixingDocumentSource.Spec(
            source=InMemoryDocumentSource(
                [1, 1, 1, 0, 2, 2, 2, 0], tokenizer=tokenizer, work_dir=tmp_path
            ),
            ratio=0.50,
        ),
        MixingDocumentSource.Spec(
            source=InMemoryDocumentSource(
                [3, 3, 3, 0, 4, 4, 4, 0, 5, 5, 5, 0, 6, 6, 6, 0],
                tokenizer=tokenizer,
                work_dir=tmp_path,
            ),
            ratio=0.50,
        ),
        work_dir=tmp_path,
        seed=None,
    )
    assert len(source) == 16
    assert list(source[:]["input_ids"]) == [1, 1, 1, 0, 2, 2, 2, 0, 3, 3, 3, 0, 4, 4, 4, 0]


def test_mixing_token_source_with_repetition(tmp_path: Path):
    tokenizer = TokenizerConfig(vocab_size=32_000, eos_token_id=0, pad_token_id=-1)
    source = MixingDocumentSource(
        MixingDocumentSource.Spec(
            source=InMemoryDocumentSource(
                [1, 1, 1, 0, 2, 2, 2, 0], tokenizer=tokenizer, work_dir=tmp_path
            ),
            ratio=0.50,
            max_repetition_factor=2.0,
        ),
        MixingDocumentSource.Spec(
            source=InMemoryDocumentSource(
                [3, 3, 3, 0, 4, 4, 4, 0, 5, 5, 5, 0, 6, 6, 6, 0],
                tokenizer=tokenizer,
                work_dir=tmp_path,
            ),
            ratio=0.50,
        ),
        work_dir=tmp_path,
        seed=None,
    )
    assert len(source) == 24
    assert list(source[:]["input_ids"]) == (
        [1, 1, 1, 0, 2, 2, 2, 0, 1, 1, 1, 0, 3, 3, 3, 0, 4, 4, 4, 0, 5, 5, 5, 0]
    )
