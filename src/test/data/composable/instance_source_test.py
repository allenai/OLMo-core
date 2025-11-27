from pathlib import Path

from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import *
from olmo_core.data.composable.utils import SEED_NOT_SET


def test_split(tmp_path: Path):
    tokenizer = TokenizerConfig.dolma2()

    source1 = RandomInstanceSource(
        tokenizer=tokenizer,
        sequence_length=16,
        seed=0,
        max_sequence_length=32,
        avg_document_length=8,
        num_tokens=512,
        work_dir=tmp_path,
    )
    assert len(source1) == 32

    source1a, source1b = source1.split(0.75)
    assert len(source1a) == 24
    assert len(source1b) == 8

    source2 = RandomInstanceSource(
        tokenizer=tokenizer,
        sequence_length=32,
        seed=0,
        max_sequence_length=32,
        avg_document_length=8,
        num_tokens=512,
        work_dir=tmp_path,
    )
    assert len(source2) == 16

    source2a, source2b = source2.split(0.75)
    assert len(source2a) == 12
    assert len(source2b) == 4

    assert source2a.fingerprint == source1a.fingerprint
    assert source2b.fingerprint == source1b.fingerprint


def test_config():
    tokenizer = TokenizerConfig.dolma2()

    config = RandomInstanceSourceConfig(
        tokenizer=tokenizer,
        sequence_length=16,
        max_sequence_length=32,
        avg_document_length=8,
        num_tokens=512,
    )

    # Upon creating the config, the seed should be concrete.
    assert config.seed is not SEED_NOT_SET
    assert config.seed == 0
