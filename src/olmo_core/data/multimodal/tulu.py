"""Tulu4 text-only NLP SFT dataset for the Molmo2 stage-1 mixture.

Ports ``Tulu4FilteredConfig`` (``mm_olmo/olmo/data/academic_datasets.py``): multi-turn,
**text-only** instruction data (no image). Examples use the same unlabelled Dolma2 document
layout as the s002 pretraining checkpoint, with loss on response turns only and no image block.

The training pipeline treats these as image-less examples: ``images`` is an empty
``(0, n_patches, patch_dim)`` array. The collator supplies a dummy zero crop so every
rank still executes the same vision/connector collectives, but no image features are
spliced into the token sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from olmo_core.config import Config
from olmo_core.nn.vision.molmo2_tokens import (
    N_PATCHES_SQ,
    PATCH_DIM,
    POOL_H,
    POOL_W,
    Molmo2TokenIds,
)

from .message_weight import MessageWeight
from .olmo3_layout import (
    MessageFormat,
    conversation_ids_and_assistant_mask,
    validate_message_format,
    validate_olmo3_chat_tokenizer,
)
from .paths import TULU4_DATA
from .rng import make_random_state

__all__ = ["Tulu4DatasetConfig", "Tulu4Dataset"]


def _format_messages(
    parts: List[Dict[str, str]], *, preserve_system: bool = False
) -> Optional[List[Dict[str, str]]]:
    """Validate/normalize raw messages into an alternating user/assistant list, folding a
    leading system message into the first user turn in document mode."""
    if not parts:
        return None
    out: List[Dict[str, str]] = []
    if parts[0]["role"] == "system":
        if len(parts) < 2 or parts[1]["role"] != "user":
            return None
        if preserve_system:
            out.extend([dict(parts[0]), dict(parts[1])])
        else:
            out.append(
                {
                    "role": "user",
                    "content": f"System: {parts[0]['content']}\n{parts[1]['content']}",
                }
            )
        parts = parts[2:]
    elif parts[0]["role"] == "assistant":
        return None
    else:
        out.append({"role": "user", "content": parts[0]["content"]})
        parts = parts[1:]
    for ix, m in enumerate(parts):
        expected = "assistant" if ix % 2 == 0 else "user"
        if m["role"] != expected:
            return None
        out.append({"role": m["role"], "content": m["content"]})
    # need at least one user + one assistant turn
    if len(out) <= 1 or out[-1]["role"] != "assistant":
        return None
    return out


@dataclass
class Tulu4DatasetConfig(Config):
    """Tulu4 filtered text SFT (matches mm_olmo ``get_dataset('tulu4')`` filter)."""

    max_first_msg_len: int = 2304
    max_sequence_length: int = 4096
    """Maximum length of a complete tokenized conversation."""
    loss_token_weighting: Optional[str] = "root_subsegments_root_tokens"
    """Assistant-token weighting mode; the default preserves the existing SFT behavior."""
    token_ids: Molmo2TokenIds = field(default_factory=Molmo2TokenIds)
    """Image token IDs reserved by the selected language-model tokenizer."""
    message_format: MessageFormat = "document"
    """Use native pretraining documents or the exact s002 SFT chat template."""
    use_code: bool = False
    use_non_english: bool = False
    use_reasoning: bool = False
    use_puzzles: bool = False
    seed: int = 0

    def build(self, tokenizer) -> "Tulu4Dataset":
        return Tulu4Dataset(self, tokenizer)


class Tulu4Dataset:
    def __init__(self, config: Tulu4DatasetConfig, tokenizer):
        validate_message_format(config.message_format)
        if config.message_format == "olmo3_chat":
            validate_olmo3_chat_tokenizer(tokenizer, token_ids=config.token_ids)
        self.config = config
        self.tokenizer = tokenizer
        self._data = self._load_filtered()

    def _load_filtered(self):
        from .dataset_compat import load_from_disk_compat

        ds = load_from_disk_compat(TULU4_DATA)
        ds = ds["train"] if hasattr(ds, "keys") and "train" in ds else ds
        cfg = self.config

        def _keep(category, source, n_tokens, empty_messages, has_special_token):
            if empty_messages or has_special_token:
                return False
            if source in ("allenai/dino-hardcodes", "allenai/hardcoded-olmo"):
                return False
            if not cfg.use_puzzles and source == "allenai/puzzle_data_160k-ngram-filtered":
                return False
            if not cfg.use_reasoning and source in (
                "faezeb/verifiable-reasoning-v3-o4-mini-length-filtered-verified",
                "allenai/verifiable-reasoning-filtered-o4-mini-filtered",
            ):
                return False
            if not cfg.use_code and category == "code":
                return False
            if not cfg.use_non_english and category == "non-english":
                return False
            if cfg.max_first_msg_len and n_tokens is not None and n_tokens > cfg.max_first_msg_len:
                return False
            return True

        cols = [
            "category",
            "source",
            "first_message_qwen3_tokens",
            "empty_messages",
            "has_special_token",
        ]
        return ds.filter(_keep, input_columns=cols)

    def __len__(self) -> int:
        return len(self._data)

    def _text_sequence(
        self, messages: List[Dict[str, str]], rng: np.random.RandomState
    ) -> Dict[str, np.ndarray]:
        """Tokenize a role-free Dolma2 document with loss on response messages only.

        This is Molmo's ``message_format=none`` path: the document begins with EOS, the first
        message is unchanged, and each subsequent message gets one leading space. Assistant
        messages also end in EOS so the final response predicts the native document boundary.
        """
        tok = self.tokenizer
        if tok.eos_token_id is None:
            raise ValueError("The s002 document layout requires an EOS token")

        messages = [dict(message) for message in messages]
        if rng.random() > 0.10:
            noisy_length = len(messages[-1]["content"]) + int(rng.normal(scale=25.0))
            style_prefix = f"text_sft {noisy_length // 15}:"
        else:
            style_prefix = "text_sft:"
        first_content = messages[0]["content"]
        messages[0]["content"] = (
            f"{style_prefix} {first_content}" if first_content else style_prefix
        )

        ids: List[int] = [int(tok.eos_token_id)]
        asst: List[float] = [0.0]
        for message_index, message in enumerate(messages):
            text = message["content"] if message_index == 0 else " " + message["content"]
            message_ids = tok.encode(text, add_special_tokens=False)
            is_assistant = message["role"] == "assistant"
            if is_assistant:
                message_ids.append(int(tok.eos_token_id))
            ids.extend(message_ids)
            asst.extend([float(is_assistant)] * len(message_ids))

        all_ids = np.asarray(ids, dtype=np.int64)
        asst_mask = np.asarray(asst, dtype=np.float32)
        n_assistant = int(asst_mask.sum())
        message_weight = MessageWeight.from_string(self.config.loss_token_weighting)
        if n_assistant and message_weight.root_length:
            asst_mask *= 2.0 / np.sqrt(n_assistant)
        if message_weight.weight is not None:
            asst_mask *= message_weight.weight

        input_ids = all_ids[:-1]
        labels = all_ids[1:]
        loss_masks = asst_mask[1:]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_masks": loss_masks.astype(np.float32),
            "position_ids": np.arange(len(input_ids), dtype=np.int64),
            "token_type_ids": np.zeros(len(input_ids), dtype=np.int64),
            # text-only: zero image crops / pooled rows -> no image for this example.
            "images": np.zeros((0, N_PATCHES_SQ, PATCH_DIM), dtype=np.float32),
            "pooled_patches_idx": np.full((0, POOL_H * POOL_W), -1, dtype=np.int64),
        }

    def _chat_sequence(
        self, messages: List[Dict[str, str]], rng: np.random.RandomState
    ) -> Dict[str, np.ndarray]:
        """Tokenize exact s002 SFT chat with loss on assistant content and end tokens."""
        messages = [dict(message) for message in messages]
        first_user_index = next(
            (i for i, message in enumerate(messages) if message["role"] == "user"), None
        )
        if first_user_index is None:
            raise ValueError("Tulu4 chat conversation has no user message")

        if rng.random() > 0.10:
            noisy_length = len(messages[-1]["content"]) + int(rng.normal(scale=25.0))
            style_prefix = f"text_sft {noisy_length // 15}:"
        else:
            style_prefix = "text_sft:"
        first_content = messages[first_user_index]["content"]
        messages[first_user_index]["content"] = (
            f"{style_prefix} {first_content}" if first_content else style_prefix
        )

        all_ids, assistant_mask = conversation_ids_and_assistant_mask(self.tokenizer, messages)
        n_assistant = int(assistant_mask.sum())
        message_weight = MessageWeight.from_string(self.config.loss_token_weighting)
        if n_assistant and message_weight.root_length:
            assistant_mask *= 2.0 / np.sqrt(n_assistant)
        if message_weight.weight is not None:
            assistant_mask *= message_weight.weight

        input_ids = all_ids[:-1]
        return {
            "input_ids": input_ids,
            "labels": all_ids[1:],
            "loss_masks": assistant_mask[1:].astype(np.float32),
            "position_ids": np.arange(len(input_ids), dtype=np.int64),
            "token_type_ids": np.zeros(len(input_ids), dtype=np.int64),
            "images": np.zeros((0, N_PATCHES_SQ, PATCH_DIM), dtype=np.float32),
            "pooled_patches_idx": np.full((0, POOL_H * POOL_W), -1, dtype=np.int64),
        }

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        return self.get(i, 0)

    def get(self, i: int, epoch: int = 0) -> Dict[str, np.ndarray]:
        """Build one deterministically formatted conversation for a source epoch."""
        chat = self.config.message_format == "olmo3_chat"
        messages = _format_messages(list(self._data[i]["messages"]), preserve_system=chat)
        if messages is None:
            # Filtered datasets should not contain these, but guard: fall back to next.
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi."},
            ]
        rng = make_random_state(self.config.seed + i, epoch)
        seq = self._chat_sequence(messages, rng) if chat else self._text_sequence(messages, rng)
        max_len = self.config.max_sequence_length
        if max_len and len(seq["input_ids"]) > max_len:
            seq = {
                key: value[:max_len] if value.ndim == 1 and len(value) > max_len else value
                for key, value in seq.items()
            }
        return seq
