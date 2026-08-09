"""Tests for the exact s002 OLMo 3 chat serialization arm."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from olmo_core.data.multimodal.document_layout import (
    branch_context_ids as document_branch_context_ids,
)
from olmo_core.data.multimodal.document_layout import (
    image_prefix_ids as document_image_prefix_ids,
)
from olmo_core.data.multimodal.document_layout import (
    response_ids as document_response_ids,
)
from olmo_core.data.multimodal.olmo3_layout import (
    DEFAULT_SYSTEM_PREFIX,
    branch_context_ids,
    conversation_ids_and_assistant_mask,
    image_prefix_ids,
    validate_olmo3_chat_tokenizer,
)
from olmo_core.data.multimodal.sequence_builder import (
    ATTEND_ALL_SUBSEGMENT_ID,
    build_branched_sequence,
)
from olmo_core.data.multimodal.tulu import (
    Tulu4Dataset,
    Tulu4DatasetConfig,
    _format_messages,
)
from olmo_core.nn.vision import Molmo2TokenIds, prepare_molmo2_tokenizer
from olmo_core.nn.vision.molmo2_tokens import (
    IMAGE_PLACEHOLDER_TOKEN,
    build_image_token_ids,
)

_SPECIAL_IDS = {
    "<|endoftext|>": 100257,
    "<|im_start|>": 100264,
    "<|im_end|>": 100265,
    "<functions>": 100266,
    "</functions>": 100267,
    "<function_calls>": 100268,
    "</function_calls>": 100269,
    "<|pad|>": 100277,
    "<im_start>": 100278,
    "<im_end>": 100279,
    "<im_patch>": 100280,
    "<im_col>": 100281,
    "<low_res_im_start>": 100282,
    IMAGE_PLACEHOLDER_TOKEN: 100283,
}
_TOKEN_IDS = Molmo2TokenIds(
    im_start_id=100278,
    im_end_id=100279,
    im_patch_id=100280,
    im_col_id=100281,
    low_res_im_start_id=100282,
    image_placeholder_id=100283,
    im_end_turn_id=100265,
)


class _ExactChatTokenizer:
    """Small character tokenizer implementing the pinned Jinja behavior."""

    eos_token_id = 100257
    pad_token_id = 100277

    def get_vocab(self):
        return dict(_SPECIAL_IDS)

    def convert_tokens_to_ids(self, token):
        return _SPECIAL_IDS[token]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert not tokenize
        has_system = any(message["role"] == "system" for message in messages)
        text = "" if has_system else DEFAULT_SYSTEM_PREFIX
        for index, message in enumerate(messages):
            role = message["role"]
            content = message.get("content") or ""
            if role == "system":
                text += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                text += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                text += f"<|im_start|>assistant\n{content}"
                text += "<|endoftext|>" if index == len(messages) - 1 else "<|im_end|>\n"
            else:
                raise ValueError(role)
            if index == len(messages) - 1 and add_generation_prompt:
                text += "<|im_start|>assistant\n"
        return text

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        ids = []
        index = 0
        specials = sorted(_SPECIAL_IDS, key=len, reverse=True)
        while index < len(text):
            token = next((token for token in specials if text.startswith(token, index)), None)
            if token is not None:
                ids.append(_SPECIAL_IDS[token])
                index += len(token)
            else:
                ids.append(ord(text[index]) + 1000)
                index += 1
        return ids


def _replace_image_placeholder(ids, image_ids):
    location = ids.index(_TOKEN_IDS.image_placeholder_id)
    return ids[:location] + list(image_ids) + ids[location + 1 :]


def test_exact_chat_single_branch_reconstructs_template_and_masks_only_answer():
    tokenizer = _ExactChatTokenizer()
    validate_olmo3_chat_tokenizer(tokenizer, token_ids=_TOKEN_IDS)
    grid = np.asarray([1, 1, 1, 1])
    image_ids = build_image_token_ids(1, 1, 1, 1, token_ids=_TOKEN_IDS)
    question, answer = "Describe this image.", "A small cat."

    prefix = image_prefix_ids(tokenizer, grid, token_ids=_TOKEN_IDS)
    context = branch_context_ids(tokenizer, question, token_ids=_TOKEN_IDS)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    sequence = build_branched_sequence(
        prefix,
        [(context, answer_ids)],
        eos_id=tokenizer.eos_token_id,
        image_token_ids=_TOKEN_IDS.image_token_ids,
        loss_token_weighting="none",
    )

    rendered = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": IMAGE_PLACEHOLDER_TOKEN + question},
            {"role": "assistant", "content": answer},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    expected = _replace_image_placeholder(
        tokenizer.encode(rendered, add_special_tokens=False), image_ids
    )
    actual = sequence["input_ids"].tolist() + [int(sequence["labels"][-1])]
    assert actual == expected
    assert sequence["labels"][sequence["loss_masks"] > 0].tolist() == [
        *answer_ids,
        tokenizer.eos_token_id,
    ]


def test_exact_chat_multi_branch_is_independent_one_turn_chats():
    tokenizer = _ExactChatTokenizer()
    grid = np.asarray([1, 1, 1, 1])
    prefix = image_prefix_ids(tokenizer, grid, token_ids=_TOKEN_IDS)
    image_ids = build_image_token_ids(1, 1, 1, 1, token_ids=_TOKEN_IDS)
    turns = [("Count the cats.", "Two."), ("Locate the dog.", "At left.")]
    branches = [
        (
            branch_context_ids(tokenizer, question, token_ids=_TOKEN_IDS),
            tokenizer.encode(answer, add_special_tokens=False),
        )
        for question, answer in turns
    ]
    sequence = build_branched_sequence(
        prefix,
        branches,
        eos_id=tokenizer.eos_token_id,
        image_token_ids=_TOKEN_IDS.image_token_ids,
        loss_token_weighting="none",
    )

    subsegments = sequence["subsegment_ids"]
    branch_positions = [
        sequence["position_ids"][subsegments == branch_index] for branch_index in range(2)
    ]
    assert branch_positions[0][0] == branch_positions[1][0] == len(prefix)
    for positions in branch_positions:
        np.testing.assert_array_equal(
            positions, np.arange(len(prefix), len(prefix) + len(positions))
        )
    assert np.all(subsegments[: len(prefix)] == ATTEND_ALL_SUBSEGMENT_ID)
    for branch_index, (question, answer) in enumerate(turns):
        branch_locations = np.flatnonzero(subsegments == branch_index)
        branch_stream = [
            *sequence["input_ids"][: len(prefix)].tolist(),
            *sequence["input_ids"][branch_locations].tolist(),
            int(sequence["labels"][branch_locations[-1]]),
        ]
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": IMAGE_PLACEHOLDER_TOKEN + question},
                {"role": "assistant", "content": answer},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        expected = _replace_image_placeholder(
            tokenizer.encode(rendered, add_special_tokens=False), image_ids
        )
        assert branch_stream == expected


def test_conversation_mask_matches_cached_sft_header_and_end_token_semantics():
    tokenizer = _ExactChatTokenizer()
    messages = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Second question"},
        {"role": "assistant", "content": "Second answer"},
    ]
    ids, assistant_mask = conversation_ids_and_assistant_mask(tokenizer, messages)

    expected_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    assert ids.tolist() == tokenizer.encode(expected_text, add_special_tokens=False)
    expected_loss_ids = [
        *tokenizer.encode("First answer", add_special_tokens=False),
        _SPECIAL_IDS["<|im_end|>"],
        *tokenizer.encode("Second answer", add_special_tokens=False),
        tokenizer.eos_token_id,
    ]
    assert ids[assistant_mask > 0].tolist() == expected_loss_ids


def test_tulu_chat_preserves_explicit_system_and_styles_first_user():
    tokenizer = _ExactChatTokenizer()
    dataset = object.__new__(Tulu4Dataset)
    dataset.config = Tulu4DatasetConfig(
        max_sequence_length=512,
        loss_token_weighting="none",
        token_ids=_TOKEN_IDS,
        message_format="olmo3_chat",
    )
    dataset.tokenizer = tokenizer
    raw = [
        {"role": "system", "content": "Follow this policy."},
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
    ]
    messages = _format_messages(raw, preserve_system=True)
    assert messages == raw
    sequence = dataset._chat_sequence(messages, np.random.RandomState(0))
    stream = sequence["input_ids"].tolist() + [int(sequence["labels"][-1])]
    explicit_system = tokenizer.encode(
        "<|im_start|>system\nFollow this policy.<|im_end|>\n",
        add_special_tokens=False,
    )

    assert stream[: len(explicit_system)] == explicit_system
    assert stream[: len(tokenizer.encode(DEFAULT_SYSTEM_PREFIX, add_special_tokens=False))] != (
        tokenizer.encode(DEFAULT_SYSTEM_PREFIX, add_special_tokens=False)
    )
    assert tokenizer.encode("text_sft", add_special_tokens=False) in [
        stream[i : i + len(tokenizer.encode("text_sft", add_special_tokens=False))]
        for i in range(len(stream))
    ]
    assert sequence["labels"][sequence["loss_masks"] > 0].tolist() == [
        *tokenizer.encode("Answer", add_special_tokens=False),
        tokenizer.eos_token_id,
    ]


_BASE_TOKENIZER = Path(
    "/weka/oe-training-default/rustin/hf-cache/hub/models--allenai--dolma2-tokenizer/"
    "snapshots/5292e5d6c0f40b67cc765fe41bec991cf4345b5c"
)
_SFT_TOKENIZER = Path(
    "/weka/oe-training-default/robertb/olmo3moe-post-training/checkpoints/"
    "s002-olmo3moe-instruct-sft-resume-to1000-fused-20260727-hf"
)


@pytest.mark.skipif(
    not _BASE_TOKENIZER.is_dir() or not _SFT_TOKENIZER.is_dir(),
    reason="local s002 base/SFT tokenizer assets are unavailable",
)
def test_document_tensors_match_between_base_and_sft_tokenizers_offline():
    transformers = pytest.importorskip("transformers")
    base = transformers.GPT2Tokenizer.from_pretrained(str(_BASE_TOKENIZER), local_files_only=True)
    sft = transformers.GPT2Tokenizer.from_pretrained(str(_SFT_TOKENIZER), local_files_only=True)
    base_token_ids = prepare_molmo2_tokenizer(base, model_vocab_size=100352)
    sft_token_ids = prepare_molmo2_tokenizer(sft, model_vocab_size=100352)
    assert base_token_ids == sft_token_ids
    validate_olmo3_chat_tokenizer(sft, token_ids=sft_token_ids)

    grid = np.asarray([3, 4, 2, 3])
    branches_text = [
        ("long_caption 12: Describe this image.", "A café patio beneath a blue sky."),
        (
            "point_count: How many red cars are visible?",
            'Counting the <points coords="1 1 123 456">cars</points> shows a total of 1.',
        ),
    ]

    def build(tokenizer, token_ids):
        return build_branched_sequence(
            document_image_prefix_ids(tokenizer, grid, token_ids=token_ids),
            [
                (
                    document_branch_context_ids(tokenizer, question),
                    document_response_ids(tokenizer, answer),
                )
                for question, answer in branches_text
            ],
            eos_id=tokenizer.eos_token_id,
            image_token_ids=token_ids.image_token_ids,
            loss_token_weighting="root_subsegments",
        )

    base_sequence = build(base, base_token_ids)
    sft_sequence = build(sft, sft_token_ids)
    assert base_sequence.keys() == sft_sequence.keys()
    for key in base_sequence:
        np.testing.assert_array_equal(base_sequence[key], sft_sequence[key])

    messages = [
        {"role": "user", "content": "text_sft 8: Explain why the sky is blue."},
        {"role": "assistant", "content": "Rayleigh scattering."},
    ]
    base_dataset = object.__new__(Tulu4Dataset)
    base_dataset.config = Tulu4DatasetConfig(loss_token_weighting="none")
    base_dataset.tokenizer = base
    sft_dataset = object.__new__(Tulu4Dataset)
    sft_dataset.config = Tulu4DatasetConfig(loss_token_weighting="none")
    sft_dataset.tokenizer = sft
    base_text = base_dataset._text_sequence(messages, np.random.RandomState(7))
    sft_text = sft_dataset._text_sequence(messages, np.random.RandomState(7))
    for key in base_text:
        np.testing.assert_array_equal(base_text[key], sft_text[key])

    # Also prove that separately assembled multimodal parts reconstruct the actual SFT Jinja
    # rendering exactly under the real tokenizer, including the no-leading-space answer.
    question, answer = "Describe this image.", "A small café patio."
    image_ids = build_image_token_ids(1, 1, 1, 1, token_ids=sft_token_ids)
    chat_sequence = build_branched_sequence(
        image_prefix_ids(sft, np.asarray([1, 1, 1, 1]), token_ids=sft_token_ids),
        [
            (
                branch_context_ids(sft, question, token_ids=sft_token_ids),
                sft.encode(answer, add_special_tokens=False),
            )
        ],
        eos_id=sft.eos_token_id,
        image_token_ids=sft_token_ids.image_token_ids,
        loss_token_weighting="none",
    )
    rendered_chat = sft.apply_chat_template(
        [
            {"role": "user", "content": IMAGE_PLACEHOLDER_TOKEN + question},
            {"role": "assistant", "content": answer},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    expected_chat = _replace_image_placeholder(
        sft.encode(rendered_chat, add_special_tokens=False), image_ids
    )
    actual_chat = chat_sequence["input_ids"].tolist() + [int(chat_sequence["labels"][-1])]
    assert actual_chat == expected_chat
