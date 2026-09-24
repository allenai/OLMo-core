"""Tokenizer export must preserve the backend, not merely vocabulary size."""

import json
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Regex, Tokenizer, models, pre_tokenizers, trainers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedTokenizerFast,
)

from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.hf.checkpoint import save_hf_model
from olmo_core.nn.hf.convert_checkpoint import convert_checkpoint_to_hf
from olmo_core.nn.hf.tokenizer import (
    PROBES,
    export_checkpoint_tokenizer,
    load_tokenizer_losslessly,
    tokenizer_source,
    validate_tokenizer_backend,
)
from olmo_core.nn.transformer.config import TransformerBlockType, TransformerConfig


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


def test_saved_tokenizer_identifier_remains_a_supported_fallback():
    assert tokenizer_source(None, "allenai/dolma2-tokenizer") == "allenai/dolma2-tokenizer"
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


@pytest.mark.parametrize("bos_override", [None, 1])
def test_export_preserves_unspecified_bos_and_honors_explicit_ids(tmp_path, bos_override):
    src = tmp_path / "source"
    tok, _ = source(src)
    tok.bos_token = "<unk>"
    tok.save_pretrained(src)
    config = SimpleNamespace(
        identifier=str(src),
        vocab_size=len(tok),
        eos_token_id=1,
        pad_token_id=2,
        bos_token_id=bos_override,
    )
    exported = export_checkpoint_tokenizer(None, tmp_path / "dest", config)
    assert exported.bos_token_id == (0 if bos_override is None else bos_override)
    assert exported.eos_token_id == 1 and exported.pad_token_id == 2


@pytest.mark.parametrize("entry_point", ["convert", "save"])
@pytest.mark.parametrize("source_bos,bos_override", [(0, None), (0, 1), (1, 0), (None, None)])
def test_exported_model_generates_with_resolved_bos(
    tmp_path, entry_point, source_bos, bos_override
):
    src, output = tmp_path / "source", tmp_path / "hf"
    tokenizer, _ = source(src)
    tokenizer.bos_token_id = source_bos
    tokenizer.save_pretrained(src)
    tokenizer_config = TokenizerConfig(
        identifier=str(src),
        vocab_size=len(tokenizer),
        eos_token_id=1,
        pad_token_id=2,
        bos_token_id=bos_override,
    )
    model_config = TransformerConfig.llama_like(
        d_model=32,
        n_layers=1,
        n_heads=4,
        vocab_size=len(tokenizer),
        block_name=TransformerBlockType.reordered_norm,
        qk_norm=True,
        attn_backend=AttentionBackendName.torch,
        init_seed=42,
    )
    model = model_config.build(init_device="cpu")
    model.init_weights(max_seq_len=8)
    if entry_point == "convert":
        convert_checkpoint_to_hf(
            None,
            output,
            model_config.as_config_dict(),
            tokenizer_config.as_config_dict(),
            model_state_dict=model.state_dict(),
            max_sequence_length=32,
            validate=False,
        )
    else:
        resolved = export_checkpoint_tokenizer(None, output, tokenizer_config)
        save_hf_model(
            output, model.state_dict(), model, huggingface_tokenizer=resolved, save_overwrite=True
        )

    expected_bos = source_bos if bos_override is None else bos_override
    reloaded = AutoModelForCausalLM.from_pretrained(output).eval()
    if expected_bos is None:
        with pytest.raises(ValueError, match="bos_token_id"):
            reloaded.generate(max_new_tokens=1, do_sample=False)
    else:
        # No input_ids or explicit BOS override: generation must use the saved metadata.
        with torch.no_grad():
            generated = reloaded.generate(max_new_tokens=1, do_sample=False)
        assert generated.shape == (1, 2)
        assert generated[0, 0].item() == expected_bos
    assert AutoTokenizer.from_pretrained(output).bos_token_id == expected_bos
    assert AutoConfig.from_pretrained(output).bos_token_id == expected_bos
    assert GenerationConfig.from_pretrained(output).bos_token_id == expected_bos


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
