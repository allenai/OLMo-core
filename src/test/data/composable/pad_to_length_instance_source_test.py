from pathlib import Path

import pytest

from olmo_core.data.composable.pad_to_length_instance_source import PadToLengthInstanceSource
from olmo_core.data.composable.token_source import InMemoryDocumentSource
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.exceptions import OLMoConfigurationError


@pytest.fixture
def tokenizer() -> TokenizerConfig:
    # No BOS token, so EOS alone marks document boundaries (like the Qwen3 tokenizer).
    return TokenizerConfig(vocab_size=100, eos_token_id=99, pad_token_id=99)


def test_pad_to_length_instance_source(tmp_path: Path, tokenizer: TokenizerConfig):
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    # Three EOS-terminated documents of lengths 4, 8, and 2 (including the EOS).
    tokens = [1, 2, 3, eos] + [4, 5, 6, 7, 8, 9, 10, eos] + [11, eos]
    docs = InMemoryDocumentSource(tokens=tokens, tokenizer=tokenizer, work_dir=tmp_path)

    instances = PadToLengthInstanceSource(
        docs, sequence_length=8, tokenizer=tokenizer, work_dir=tmp_path
    )
    assert isinstance(instances.fingerprint, str)
    assert len(instances) == 3

    # One document per instance, right-padded to the sequence length.
    first = instances[0]
    assert list(first["input_ids"]) == [1, 2, 3, eos, pad, pad, pad, pad]
    assert list(first["label_mask"]) == [True] * 4 + [False] * 4

    # An exact-length document gets no padding.
    second = instances[1]
    assert list(second["input_ids"]) == [4, 5, 6, 7, 8, 9, 10, eos]
    assert list(second["label_mask"]) == [True] * 8

    third = instances[2]
    assert list(third["input_ids"]) == [11, eos, pad, pad, pad, pad, pad, pad]
    assert list(third["label_mask"]) == [True] * 2 + [False] * 6


def test_pad_to_length_skips_long_documents(tmp_path: Path, tokenizer: TokenizerConfig):
    eos = tokenizer.eos_token_id
    tokens = [1, 2, 3, eos] + list(range(4, 14)) + [eos] + [20, eos]
    docs = InMemoryDocumentSource(tokens=tokens, tokenizer=tokenizer, work_dir=tmp_path)

    instances = PadToLengthInstanceSource(
        docs, sequence_length=8, tokenizer=tokenizer, work_dir=tmp_path
    )
    # The middle 11-token document is too long and gets skipped.
    assert len(instances) == 2
    assert list(instances[1]["input_ids"])[:2] == [20, eos]


def test_pad_to_length_preserves_upstream_label_mask(tmp_path: Path, tokenizer: TokenizerConfig):
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    tokens = [1, 2, 3, eos]
    # Loss only on the "response" portion of the document.
    label_mask = [False, False, True, True]
    docs = InMemoryDocumentSource(
        tokens=tokens, tokenizer=tokenizer, label_mask=label_mask, work_dir=tmp_path
    )

    instances = PadToLengthInstanceSource(
        docs, sequence_length=6, tokenizer=tokenizer, work_dir=tmp_path
    )
    assert len(instances) == 1
    instance = instances[0]
    assert list(instance["input_ids"]) == [1, 2, 3, eos, pad, pad]
    assert list(instance["label_mask"]) == [False, False, True, True, False, False]


def test_pad_to_length_rejects_sequence_length_ramp(tmp_path: Path, tokenizer: TokenizerConfig):
    eos = tokenizer.eos_token_id
    docs = InMemoryDocumentSource(tokens=[1, eos], tokenizer=tokenizer, work_dir=tmp_path)
    with pytest.raises(OLMoConfigurationError):
        PadToLengthInstanceSource(
            docs, sequence_length=4, max_sequence_length=8, tokenizer=tokenizer, work_dir=tmp_path
        )
