from pathlib import Path

from olmo_core.data.composable.random_instance_source import RandomInstanceSource
from olmo_core.data.tokenizer import TokenizerConfig


def test_random_instance_source(tmp_path: Path):
    tokenizer = TokenizerConfig(vocab_size=32_000, eos_token_id=1, pad_token_id=1, bos_token_id=0)

    # Make two sources that are the same except for sequence length.
    source1 = RandomInstanceSource(
        tokenizer=tokenizer,
        sequence_length=16,
        seed=0,
        avg_document_length=8,
        max_sequence_length=32,
        num_tokens=512,
        work_dir=tmp_path,
    )
    source2 = RandomInstanceSource(
        tokenizer=tokenizer,
        sequence_length=32,
        seed=0,
        avg_document_length=8,
        max_sequence_length=32,
        num_tokens=512,
        work_dir=tmp_path,
    )

    assert source1.fingerprint == source2.fingerprint
    assert len(source1) == 32
    assert len(source2) == 16

    assert list(source1[0]["input_ids"]) + list(source1[1]["input_ids"]) == list(
        source2[0]["input_ids"]
    )
    assert list(source1[-2]["input_ids"]) + list(source1[-1]["input_ids"]) == list(
        source2[-1]["input_ids"]
    )
