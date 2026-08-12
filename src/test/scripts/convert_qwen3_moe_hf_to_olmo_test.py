"""Tests for the Qwen MoE Hugging Face-to-OLMo converter."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location(
    "convert_qwen3_moe_hf_to_olmo",
    Path(__file__).resolve().parents[3] / "src/scripts/convert_qwen3_moe_hf_to_olmo.py",
)
if spec is None or spec.loader is None:
    raise ImportError("Could not load convert_qwen3_moe_hf_to_olmo.py")
convert_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(convert_module)


def test_tokenizer_metadata_can_differ_from_base_model(monkeypatch) -> None:
    tokenizer = SimpleNamespace(
        eos_token_id=248_046,
        pad_token_id=248_044,
        bos_token_id=None,
    )
    calls = []

    def from_pretrained(identifier: str, **kwargs):
        calls.append((identifier, kwargs))
        return tokenizer

    monkeypatch.setattr(convert_module.AutoTokenizer, "from_pretrained", from_pretrained)
    config = convert_module._tokenizer_config_from_qwen_hf(
        "Qwen/Qwen3.5-35B-A3B-Base",
        {
            "text_config": {
                "vocab_size": 248_320,
                "eos_token_id": 248_044,
                "pad_token_id": 248_044,
            }
        },
        tokenizer_name="Qwen/Qwen3.5-35B-A3B",
    )

    assert calls == [("Qwen/Qwen3.5-35B-A3B", {"trust_remote_code": False})]
    assert config.vocab_size == 248_320
    assert config.eos_token_id == 248_046
    assert config.pad_token_id == 248_044
    assert config.bos_token_id is None
    assert config.identifier == "Qwen/Qwen3.5-35B-A3B"
