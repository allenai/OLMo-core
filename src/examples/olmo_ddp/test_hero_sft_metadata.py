"""CPU-only export-format checks against the exact tokenizer used for the SFT data."""

import json

from olmoe3_hero_sft_metadata import (
    GENERATION,
    check_tokenizer,
    inference_template,
    install_metadata,
)
from olmoe3_hero_sft_plan import DATA


def test_round_trip(tmp_path):
    from transformers import AutoTokenizer

    training = AutoTokenizer.from_pretrained(DATA / "train/tokenizer", local_files_only=True)
    output = tmp_path / "hf.partial"
    training.save_pretrained(output)
    (output / "config.json").write_text(
        json.dumps(
            {
                "max_position_embeddings": 65536,
                "vocab_size": 100278,
                "eos_token_id": 100257,
                "pad_token_id": 100277,
            }
        )
    )
    result = install_metadata(output, DATA / "train/tokenizer")
    assert result["passed"]
    assert (output / "chat_template.jinja").is_file()
    assert (output / "tokenizer.json").is_file()
    saved = json.loads((output / "generation_config.json").read_text())
    assert saved["eos_token_id"] == GENERATION["eos_token_id"]
    assert saved["do_sample"] and saved["top_k"] == 0
    check_tokenizer(AutoTokenizer.from_pretrained(output), training)


def test_reject_double_think():
    import pytest
    from transformers import AutoTokenizer

    training = AutoTokenizer.from_pretrained(DATA / "train/tokenizer", local_files_only=True)
    with pytest.raises(AssertionError):
        inference_template(inference_template(training.chat_template))
