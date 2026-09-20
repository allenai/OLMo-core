"""Export the trained chat format with a Think generation prefix, without changing IDs."""

import hashlib
import json
from pathlib import Path

REFERENCES = {
    "allenai/Olmo-3-7B-Think-SFT": "6ff857587e040d6d523a3d5f3a56e918f5401d66",
    "allenai/Olmo-Hybrid-Think-SFT-7B": "79d1f7f613a9f98169e6c9f00880ab2df4383860",
}
GENERATION = {
    "do_sample": True,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 0,
    "max_new_tokens": 32768,
    "eos_token_id": [100265, 100257],
    "pad_token_id": 100277,
    "bos_token_id": None,
}


def inference_template(training_template):
    """Only prefill the opening think tag at inference; keep all training turns unchanged."""
    old = "{% if loop.last and add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}"
    # Saved Jinja contains an actual newline in the quoted string.
    old = old.replace("\\n", "\n")
    if training_template.count(old) != 1:
        # Dolci's open-instruct template uses whitespace-trimmed Jinja tags.
        old = "{%- if loop.last and add_generation_prompt -%}{{- '<|im_start|>assistant\n' -}}"
    assert training_template.count(old) == 1
    return training_template.replace(old, old.replace("assistant\n'", "assistant\n<think>'"))


def check_tokenizer(tokenizer, training):
    """Fail on changed vocab, EOS/PAD/BOS, double formatting, or altered completed turns."""
    assert tokenizer.get_vocab() == training.get_vocab()
    assert len(tokenizer) == tokenizer.vocab_size == 100278
    assert tokenizer.eos_token_id == 100257 and tokenizer.pad_token_id == 100277
    assert tokenizer.bos_token_id is None and tokenizer.model_max_length == 65536
    assert tokenizer.convert_tokens_to_ids("<|im_start|>") == 100264
    assert tokenizer.convert_tokens_to_ids("<|im_end|>") == 100265
    assert tokenizer.chat_template == inference_template(training.chat_template)
    for messages in (
        [{"role": "user", "content": "What is 2+2?"}],
        [{"role": "system", "content": "Be helpful."}, {"role": "user", "content": "Hi"}],
        [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "<think>Think.</think>Answer."},
            {"role": "user", "content": "Next question"},
        ],
    ):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        original = training.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        assert prompt == original + "<think>"
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=False
        )
        assert ids == tokenizer.encode(prompt, add_special_tokens=False)
        assert ids[0] == 100264
        completed = messages + [{"role": "assistant", "content": "<think>Think.</think>Answer."}]
        assert tokenizer.apply_chat_template(
            completed, tokenize=True, return_dict=False
        ) == training.apply_chat_template(completed, tokenize=True, return_dict=False)


def install_metadata(output, training_path):
    """Write and reload metadata inside an unpublished HF export before its receipt is hashed."""
    from transformers import AutoTokenizer, GenerationConfig

    output, training_path = Path(output), Path(training_path)
    assert output.name == "hf.partial" and not output.is_symlink()
    training = AutoTokenizer.from_pretrained(training_path, local_files_only=True)
    exported = AutoTokenizer.from_pretrained(output, local_files_only=True)
    assert exported.get_vocab() == training.get_vocab()
    exported.chat_template = inference_template(training.chat_template)
    exported.bos_token = None
    exported.eos_token_id = 100257
    exported.pad_token_id = 100277
    exported.model_max_length = 65536
    exported.save_pretrained(output)
    GenerationConfig(**GENERATION).save_pretrained(output)
    check_tokenizer(AutoTokenizer.from_pretrained(output, local_files_only=True), training)
    generation = GenerationConfig.from_pretrained(output, local_files_only=True)
    assert all(getattr(generation, key) == value for key, value in GENERATION.items())
    config = json.loads((output / "config.json").read_text())
    assert config["max_position_embeddings"] == 65536 and config["vocab_size"] == 100278
    assert config["eos_token_id"] == 100257 and config["pad_token_id"] == 100277
    report = {
        "passed": True,
        "references": REFERENCES,
        "generation": GENERATION,
        "vocabulary_unchanged": True,
        "training_template_sha256": hashlib.sha256(training.chat_template.encode()).hexdigest(),
        "inference_template_sha256": hashlib.sha256(exported.chat_template.encode()).hexdigest(),
        "change": "Append <think> only for add_generation_prompt; preserve the trained system prompt.",
        "reference_differences": [
            "Preserve our trained generic system prompt instead of either reference model identity.",
            "Preserve our no-BOS behavior; apply_chat_template adds no extra special tokens.",
            "Use canonical pad_token_id and explicit do_sample/top_k, not legacy pad_token typo.",
        ],
    }
    (output / "sft-metadata-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
