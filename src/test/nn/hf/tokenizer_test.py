"""Tokenizer export must preserve the backend, not merely vocabulary size."""

import json
from types import SimpleNamespace

import pytest
from tokenizers import Regex, Tokenizer, models, pre_tokenizers, trainers
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from olmo_core.nn.hf.tokenizer import (
    PROBES,
    export_checkpoint_tokenizer,
    load_tokenizer_losslessly,
    tokenizer_source,
    validate_tokenizer_backend,
)


def source(path):
    raw = Tokenizer(models.BPE(unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(r"\p{N}{1,3}"), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    raw.train_from_iterator(
        PROBES, trainers.BpeTrainer(vocab_size=512, special_tokens=["<unk>", "<eos>", "<pad>"])
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        unk_token="<unk>",
        eos_token="<eos>",
        pad_token="<pad>",
        clean_up_tokenization_spaces=False,
        model_max_length=8192,
        chat_template="{% for m in messages %}{{ m['content'] }}{% endfor %}",
    )
    tokenizer.save_pretrained(path)
    return tokenizer, Tokenizer.from_file(str(path / "tokenizer.json"))


def test_explicit_override_wins(tmp_path):
    (tmp_path / "tokenizer").mkdir()
    assert tokenizer_source(tmp_path / "step1", "fallback", "explicit") == "explicit"
    assert tokenizer_source(tmp_path / "step1", "fallback") == str(tmp_path / "tokenizer")
    assert tokenizer_source(None, "fallback") == "fallback"


def test_legacy_fallback_requires_choice():
    with pytest.raises(ValueError, match="ambiguous"):
        tokenizer_source(None, "allenai/dolma2-tokenizer")
    assert (
        tokenizer_source(None, "unused", "allenai/dolma2-tokenizer") == "allenai/dolma2-tokenizer"
    )


def test_generic_load_preserves_backend_and_metadata(tmp_path):
    src, dest = tmp_path / "source", tmp_path / "dest"
    tok, raw = source(src)
    metadata = json.loads((src / "tokenizer_config.json").read_text())
    metadata["tokenizer_class"] = "GPT2Tokenizer"
    (src / "tokenizer_config.json").write_text(json.dumps(metadata))
    cfg = SimpleNamespace(
        identifier="unused", vocab_size=len(tok), eos_token_id=1, pad_token_id=2, bos_token_id=None
    )
    exported = export_checkpoint_tokenizer(
        None, dest, cfg, override=str(src), max_sequence_length=65536
    )
    validate_tokenizer_backend(exported, raw)
    reloaded = AutoTokenizer.from_pretrained(dest, local_files_only=True)
    assert reloaded.chat_template == tok.chat_template
    assert reloaded.model_max_length == 65536
    assert json.loads((dest / "tokenizer-export-audit.json").read_text())["passed"]


def test_same_vocabulary_wrong_segmentation_rejected(tmp_path):
    tok, raw = source(tmp_path / "source")
    assert tok.get_vocab() == raw.get_vocab()
    tok.backend_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    assert tok.get_vocab() == raw.get_vocab()
    with pytest.raises(ValueError, match="backend changed"):
        validate_tokenizer_backend(tok, raw)


def test_missing_backend_fails_closed(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text("{}")
    with pytest.raises((ValueError, OSError)):
        load_tokenizer_losslessly(tmp_path)


def test_wrong_vocab_fails_closed(tmp_path):
    source(tmp_path / "source")
    cfg = SimpleNamespace(identifier=str(tmp_path / "source"), vocab_size=100278)
    with pytest.raises(ValueError, match="vocabulary size"):
        export_checkpoint_tokenizer(None, tmp_path / "dest", cfg)
