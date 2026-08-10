import importlib.util
import sys
from pathlib import Path

import numpy as np
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
    image_placeholder_id = 999

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        marker = "<|image|>"
        if marker not in text:
            return [ord(character) for character in text]
        before, after = text.split(marker)
        return [*map(ord, before), self.image_placeholder_id, *map(ord, after)]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert not tokenize and add_generation_prompt
        return f"S{messages[0]['content']}A"


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
        ("olmo3_chat", [ord("S"), 91, ord("Q"), ord("A")]),
    ],
)
def test_prompt_layouts(mmmu, layout, expected):
    token_ids = mmmu.Molmo2TokenIds(image_placeholder_id=_Tokenizer.image_placeholder_id)
    assert (
        mmmu._prompt_ids_for_layout(_Tokenizer(), "Q", [91], layout=layout, token_ids=token_ids)
        == expected
    )


def test_candidate_separator_tracks_layout(mmmu):
    tokenizer = _Tokenizer()
    assert mmmu._candidate_ids_for_layout(tokenizer, "A", layout="document") == [32, 65]
    assert mmmu._candidate_ids_for_layout(tokenizer, "A", layout="bare_chat") == [65]
    assert mmmu._candidate_ids_for_layout(tokenizer, "A", layout="olmo3_chat") == [65]


def test_single_image_and_document_image_ids_are_unchanged(mmmu):
    tokenizer = _Tokenizer()
    assert mmmu._image_ids_for_prompt(tokenizer, [[91, 92]], layout="olmo3_chat") == [91, 92]
    assert mmmu._image_ids_for_prompt(
        tokenizer,
        [[91], [92]],
        layout="document",
    ) == [91, 92]


def test_stage2_multi_image_ids_include_training_prefixes(mmmu):
    tokenizer = _Tokenizer()
    assert mmmu._image_ids_for_prompt(
        tokenizer,
        [[91], [92]],
        layout="olmo3_chat",
    ) == [*map(ord, "Image 1"), 91, *map(ord, "Image 2"), 92]


def test_crop_budgets_keep_legacy_total_and_stage2_per_image_semantics(mmmu):
    assert mmmu._resolve_max_crops_per_image("document", None) is None
    assert mmmu._resolve_max_crops_per_image("olmo3_chat", None) == 8
    assert mmmu._resolve_max_crops_per_image("olmo3_chat", 4) == 4
    assert mmmu._image_crop_budgets(
        3,
        max_crops_total=8,
        max_crops_per_image=None,
    ) == [3, 3, 2]
    assert mmmu._image_crop_budgets(
        3,
        max_crops_total=8,
        max_crops_per_image=8,
    ) == [8, 8, 8]
    with pytest.raises(ValueError, match="at least one high-resolution crop"):
        mmmu._image_crop_budgets(
            9,
            max_crops_total=8,
            max_crops_per_image=None,
        )


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


_SFT_TOKENIZER = Path(
    "/weka/oe-training-default/robertb/olmo3moe-post-training/checkpoints/"
    "s002-olmo3moe-instruct-sft-resume-to1000-fused-20260727-hf"
)


@pytest.mark.skipif(
    not _SFT_TOKENIZER.is_dir(),
    reason="local s002 SFT tokenizer assets are unavailable",
)
def test_real_olmo3_multi_image_prompt_matches_stage2_training_layout(mmmu):
    transformers = pytest.importorskip("transformers")
    from olmo_core.data.multimodal.message_sequence import _multi_image_prefix_ids
    from olmo_core.data.multimodal.olmo3_layout import branch_context_ids
    from olmo_core.nn.vision import prepare_molmo2_tokenizer
    from olmo_core.nn.vision.molmo2_tokens import build_image_token_ids

    tokenizer = transformers.GPT2Tokenizer.from_pretrained(
        str(_SFT_TOKENIZER),
        local_files_only=True,
    )
    token_ids = prepare_molmo2_tokenizer(tokenizer, model_vocab_size=100352)
    image_grids = [np.asarray([1, 2, 1, 2]), np.asarray([2, 1, 2, 1])]
    prompt = "Which image contains the triangle?\nA. Image 1\nB. Image 2"
    image_id_groups = [
        build_image_token_ids(*(int(value) for value in grid), token_ids=token_ids)
        for grid in image_grids
    ]
    eval_image_ids = mmmu._image_ids_for_prompt(
        tokenizer,
        image_id_groups,
        layout="olmo3_chat",
    )
    eval_prompt_ids = mmmu._prompt_ids_for_layout(
        tokenizer,
        prompt,
        eval_image_ids,
        layout="olmo3_chat",
        token_ids=token_ids,
    )
    training_prompt_ids = [
        *_multi_image_prefix_ids(
            tokenizer,
            image_grids,
            message_format="olmo3_chat",
            token_ids=token_ids,
        ),
        *branch_context_ids(tokenizer, prompt, token_ids=token_ids),
    ]

    assert eval_prompt_ids == training_prompt_ids
