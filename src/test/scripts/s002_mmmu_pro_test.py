import importlib.util
import sys
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "eval" / "s002_mmmu_pro.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("_s002_mmmu_pro_test_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Tokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(character) for character in text]


@pytest.fixture(scope="module")
def mmmu():
    return _load_module()


@pytest.mark.parametrize(
    ("layout", "expected"),
    [
        ("document", [0, 91, 81]),
        ("text_sft", [0, 91, *map(ord, "text_sft: Q")]),
        ("answer_cue", [0, 91, *map(ord, "Q\nAnswer:")]),
        (
            "bare_chat",
            [
                *map(ord, "<|im_start|>user\n"),
                91,
                *map(ord, "Q<|im_end|>\n<|im_start|>assistant\n"),
            ],
        ),
    ],
)
def test_prompt_layouts(mmmu, layout, expected):
    assert mmmu._prompt_ids_for_layout(_Tokenizer(), "Q", [91], layout=layout) == expected


def test_candidate_separator_tracks_layout(mmmu):
    tokenizer = _Tokenizer()
    assert mmmu._candidate_ids_for_layout(tokenizer, "A", layout="document") == [32, 65]
    assert mmmu._candidate_ids_for_layout(tokenizer, "A", layout="bare_chat") == [65]


def test_parse_option_texts(mmmu):
    assert mmmu._parse_option_texts({"options": "['one', 'two']"}) == ["one", "two"]
    assert mmmu._parse_option_texts({"options": ["one", "two"]}) == ["one", "two"]
    with pytest.raises(ValueError, match="between 2 and 10"):
        mmmu._parse_option_texts({"options": "['only']"})
    with pytest.raises(ValueError, match="non-empty string"):
        mmmu._parse_option_texts({"options": ["one", ""]})


def test_model_parts_are_put_in_eval_mode(mmmu):
    class ModelPart:
        training = True

        def eval(self):
            self.training = False

    parts = [ModelPart(), ModelPart()]
    train_module = type("TrainModule", (), {"model_parts": parts})()

    mmmu._set_model_parts_eval(train_module)

    assert all(not part.training for part in parts)
