"""Native OLMo 3 instruction-chat layout helpers.

The helpers in this module intentionally derive the serialized text from the tokenizer's
``chat_template`` instead of spelling role headers by hand.  The behavioral validation below
pins the template and token IDs to the s002 SFT export used by the OLMo 3 MoE post-training run.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from olmo_core.nn.vision.molmo2_tokens import (
    IMAGE_PLACEHOLDER_TOKEN,
    Molmo2TokenIds,
    build_image_token_ids,
)

MessageFormat = Literal["document", "olmo3_chat"]
MESSAGE_FORMATS: Tuple[MessageFormat, ...] = ("document", "olmo3_chat")

DEFAULT_SYSTEM_MESSAGE = (
    "You are a helpful function-calling AI assistant. You do not currently have access to any "
    "functions. <functions></functions>"
)
DEFAULT_SYSTEM_PREFIX = f"<|im_start|>system\n{DEFAULT_SYSTEM_MESSAGE}<|im_end|>\n"

_EXPECTED_SPECIAL_TOKEN_IDS = {
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

__all__ = [
    "DEFAULT_SYSTEM_MESSAGE",
    "DEFAULT_SYSTEM_PREFIX",
    "MESSAGE_FORMATS",
    "MessageFormat",
    "branch_context_ids",
    "conversation_ids_and_assistant_mask",
    "image_prefix_ids",
    "validate_message_format",
    "validate_olmo3_chat_tokenizer",
]


def validate_message_format(message_format: str) -> MessageFormat:
    """Validate and narrow a multimodal message-format value.

    :param message_format: Requested serialization mode.

    :returns: The validated value.

    :raises ValueError: If the mode is unknown.
    """
    if message_format not in MESSAGE_FORMATS:
        raise ValueError(
            f"Unknown message_format {message_format!r}; expected one of {MESSAGE_FORMATS}"
        )
    return message_format  # type: ignore[return-value]


def _render(tokenizer, messages: Sequence[Dict[str, str]], *, generation: bool) -> str:
    return tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=generation,
    )


def _encode(tokenizer, text: str) -> List[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _require_token_id(tokenizer, token: str) -> int:
    vocab = tokenizer.get_vocab()
    if token not in vocab:
        raise ValueError(f"OLMo 3 chat tokenizer is missing required token {token!r}")
    return int(tokenizer.convert_tokens_to_ids(token))


def validate_olmo3_chat_tokenizer(tokenizer, *, token_ids: Optional[Molmo2TokenIds] = None) -> None:
    """Validate the exact s002 SFT chat-template and vocabulary contract.

    This rejects the base Dolma2 tokenizer in ``olmo3_chat`` mode even though most ordinary
    text tokenizes identically.  Its function-tag IDs have different meanings and it does not
    carry the SFT template, so accepting it would make the ablation ambiguous.

    :param tokenizer: Prepared s002 SFT tokenizer.
    :param token_ids: Resolved Molmo2 image token IDs, when image tokens are required.

    :raises ValueError: If IDs or rendered template behavior differ from the SFT export.
    """
    for token, expected_id in _EXPECTED_SPECIAL_TOKEN_IDS.items():
        actual_id = _require_token_id(tokenizer, token)
        if actual_id != expected_id:
            raise ValueError(
                f"OLMo 3 chat token {token!r} has ID {actual_id}, expected {expected_id}"
            )

    if tokenizer.eos_token_id != _EXPECTED_SPECIAL_TOKEN_IDS["<|endoftext|>"]:
        raise ValueError(f"OLMo 3 chat EOS ID is {tokenizer.eos_token_id}, expected 100257")
    if tokenizer.pad_token_id != _EXPECTED_SPECIAL_TOKEN_IDS["<|pad|>"]:
        raise ValueError(f"OLMo 3 chat pad ID is {tokenizer.pad_token_id}, expected 100277")

    if token_ids is not None:
        expected_image_ids = Molmo2TokenIds(
            im_start_id=100278,
            im_end_id=100279,
            im_patch_id=100280,
            im_col_id=100281,
            low_res_im_start_id=100282,
            image_placeholder_id=100283,
            im_end_turn_id=100265,
        )
        if token_ids != expected_image_ids:
            raise ValueError(
                f"OLMo 3 chat image token IDs are {token_ids}, expected {expected_image_ids}"
            )

    rendered = _render(
        tokenizer,
        [{"role": "user", "content": IMAGE_PLACEHOLDER_TOKEN}],
        generation=True,
    )
    expected = (
        DEFAULT_SYSTEM_PREFIX
        + f"<|im_start|>user\n{IMAGE_PLACEHOLDER_TOKEN}<|im_end|>\n"
        + "<|im_start|>assistant\n"
    )
    if rendered != expected:
        raise ValueError(
            "Tokenizer chat_template does not match the s002 SFT one-turn contract: "
            f"got {rendered!r}"
        )


def _one_turn_parts(
    tokenizer,
    prompt: str,
    *,
    token_ids: Molmo2TokenIds,
) -> Tuple[List[int], List[int]]:
    """Return template IDs before and after a unique image placeholder."""
    if IMAGE_PLACEHOLDER_TOKEN in prompt:
        raise ValueError(f"Prompt must not contain reserved token {IMAGE_PLACEHOLDER_TOKEN!r}")
    rendered = _render(
        tokenizer,
        [{"role": "user", "content": IMAGE_PLACEHOLDER_TOKEN + prompt}],
        generation=True,
    )
    ids = _encode(tokenizer, rendered)
    locations = [i for i, token_id in enumerate(ids) if token_id == token_ids.image_placeholder_id]
    if len(locations) != 1:
        raise ValueError(
            "Expected exactly one image placeholder in rendered OLMo 3 user turn, "
            f"found {len(locations)}"
        )
    location = locations[0]
    return ids[:location], ids[location + 1 :]


def image_prefix_ids(
    tokenizer,
    image_grid: np.ndarray,
    *,
    token_ids: Molmo2TokenIds,
) -> List[int]:
    """Build the shared default-system, user-header, and expanded-image prefix."""
    resized_h, resized_w, height, width = (int(image_grid[i]) for i in range(4))
    before_image, _ = _one_turn_parts(tokenizer, "", token_ids=token_ids)
    return before_image + build_image_token_ids(
        resized_h,
        resized_w,
        height,
        width,
        token_ids=token_ids,
    )


def branch_context_ids(
    tokenizer,
    prompt: str,
    *,
    token_ids: Molmo2TokenIds,
) -> List[int]:
    """Build prompt, user terminator, and masked assistant-header IDs for one branch."""
    _, after_image = _one_turn_parts(tokenizer, prompt, token_ids=token_ids)
    return after_image


def conversation_ids_and_assistant_mask(
    tokenizer,
    messages: Sequence[Dict[str, str]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Render a conversation and mark authentic s002 SFT loss-bearing tokens.

    The cached Open-Instruct preprocessing used by the s002 SFT run masks system/user tokens
    and assistant generation headers.  It trains assistant content plus ``<|im_end|>`` for
    intermediate responses and content plus ``<|endoftext|>`` for the final response; the
    newline following an intermediate end marker remains masked.

    :param tokenizer: Validated s002 SFT tokenizer.
    :param messages: Complete system/user/assistant conversation ending in assistant.

    :returns: Full unshifted token IDs and a float32 mask over those same token positions.

    :raises ValueError: If incremental template renderings do not compose exactly.
    """
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("OLMo 3 SFT conversation must end with an assistant message")

    full_ids = _encode(tokenizer, _render(tokenizer, messages, generation=False))
    assistant_mask = np.zeros(len(full_ids), dtype=np.float32)
    eos_id = int(tokenizer.eos_token_id)
    im_end_id = _require_token_id(tokenizer, "<|im_end|>")

    for message_index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue

        header_ids = _encode(
            tokenizer,
            _render(tokenizer, messages[:message_index], generation=True),
        )
        assistant_as_last_ids = _encode(
            tokenizer,
            _render(tokenizer, messages[: message_index + 1], generation=False),
        )
        if assistant_as_last_ids[: len(header_ids)] != header_ids:
            raise ValueError("Assistant rendering does not begin with its generation header")
        if full_ids[: len(header_ids)] != header_ids:
            raise ValueError("Full conversation diverges before an assistant response")

        end_index = len(assistant_as_last_ids) - 1
        if assistant_as_last_ids[end_index] != eos_id:
            raise ValueError("Final assistant rendering does not end with the tokenizer EOS")
        if message_index == len(messages) - 1:
            if assistant_as_last_ids != full_ids:
                raise ValueError("Final assistant rendering does not match the full conversation")
        else:
            if full_ids[:end_index] != assistant_as_last_ids[:end_index]:
                raise ValueError("Intermediate assistant content tokenization is not stable")
            if full_ids[end_index] != im_end_id:
                raise ValueError("Intermediate assistant response does not end with <|im_end|>")

        # Header tokens are excluded; assistant content and its one-token terminator are trained.
        assistant_mask[len(header_ids) : end_index + 1] = 1.0

    return np.asarray(full_ids, dtype=np.int64), assistant_mask
