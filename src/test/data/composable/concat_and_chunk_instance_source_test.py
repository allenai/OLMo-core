from pathlib import Path

from olmo_core.data.composable.concat_and_chunk_instance_source import (
    ConcatAndChunkInstanceSource,
)
from olmo_core.data.composable.token_source import InMemoryTokenSource


def test_concat_and_chunk_instance_source(tmp_path: Path):
    tokens1 = InMemoryTokenSource(tokens=list(range(17)), work_dir=tmp_path)
    assert tokens1.num_tokens == 17
    tokens2 = InMemoryTokenSource(tokens=list(range(17, 32)), work_dir=tmp_path)
    assert tokens2.num_tokens == 15
    instances = ConcatAndChunkInstanceSource(tokens1, tokens2, sequence_length=4, work_dir=tmp_path)
    assert isinstance(instances.fingerprint, str)
    assert len(instances) == 7
    assert list(instances[0]["input_ids"]) == [0, 1, 2, 3]
    assert list(instances[4]["input_ids"]) == [17, 18, 19, 20]


def test_concat_and_chunk_instance_source_varying_seq_len(tmp_path: Path):
    tokens1 = InMemoryTokenSource(tokens=list(range(17)), work_dir=tmp_path)
    assert tokens1.num_tokens == 17
    tokens2 = InMemoryTokenSource(tokens=list(range(17, 32)), work_dir=tmp_path)
    assert tokens2.num_tokens == 15

    instances1 = ConcatAndChunkInstanceSource(
        tokens1, tokens2, sequence_length=8, max_sequence_length=8, work_dir=tmp_path
    )
    instances2 = ConcatAndChunkInstanceSource(
        tokens1, tokens2, sequence_length=4, max_sequence_length=8, work_dir=tmp_path
    )
    assert instances1.fingerprint == instances2.fingerprint
    assert len(instances1) == 3
    assert len(instances2) == 6
    assert set([x for instance in instances1 for x in instance["input_ids"]]) == set(
        [x for instance in instances2 for x in instance["input_ids"]]
    )
