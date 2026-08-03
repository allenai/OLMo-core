"""Tulu4 text-only NLP SFT dataset for the Molmo2 stage-1 mixture.

Ports ``Tulu4FilteredConfig`` (``mm_olmo/olmo/data/academic_datasets.py``): multi-turn,
**text-only** instruction data (no image). Examples produce a standard chat sequence with
the loss on assistant turns only — no image block, no ``<im_patch>`` tokens.

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

from .paths import TULU4_DATA

__all__ = ["Tulu4DatasetConfig", "Tulu4Dataset"]


def _format_messages(parts: List[Dict[str, str]]) -> Optional[List[Dict[str, str]]]:
    """Validate/normalize raw messages into an alternating user/assistant list, folding a
    leading system message into the first user turn (matches ``format_messages``)."""
    if not parts:
        return None
    out: List[Dict[str, str]] = []
    if parts[0]["role"] == "system":
        if len(parts) < 2 or parts[1]["role"] != "user":
            return None
        out.append(
            {"role": "user", "content": f"System: {parts[0]['content']}\n{parts[1]['content']}"}
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
    token_ids: Molmo2TokenIds = field(default_factory=Molmo2TokenIds)
    """Image and chat token IDs for the selected language-model tokenizer."""
    use_code: bool = False
    use_non_english: bool = False
    use_reasoning: bool = False
    use_puzzles: bool = False
    seed: int = 0

    def build(self, tokenizer) -> "Tulu4Dataset":
        return Tulu4Dataset(self, tokenizer)


class Tulu4Dataset:
    def __init__(self, config: Tulu4DatasetConfig, tokenizer):
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

    def _text_sequence(self, messages: List[Dict[str, str]]) -> Dict[str, np.ndarray]:
        """Tokenize the conversation turn-by-turn, with loss on assistant turns only.

        Each turn is the molmo2 chat layout: ``<|im_start|>user\\n{u}<|im_end|>\\n
        <|im_start|>assistant\\n`` (non-loss) followed by ``{a}`` (loss). We build
        it explicitly because the Molmo2/Qwen chat template doesn't emit an
        ``assistant_masks`` (no ``{% generation %}`` marker)."""
        tok = self.tokenizer
        ids: List[int] = []
        asst: List[float] = []
        segment_ends: List[bool] = []
        n_turns = (len(messages) - 1) // 2
        for turn_ix, ix in enumerate(range(0, len(messages) - 1, 2)):
            u, a = messages[ix]["content"], messages[ix + 1]["content"]
            u_ids = tok.encode(
                tok.apply_chat_template(
                    [{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True
                ),
                add_special_tokens=False,
            )
            a_ids = tok.encode(a, add_special_tokens=False)
            if turn_ix < n_turns - 1:
                a_ids.append(self.config.token_ids.im_end_turn_id)
            ids += u_ids + a_ids
            asst += [0.0] * len(u_ids) + [1.0] * len(a_ids)
            seg_end = [False] * (len(u_ids) + len(a_ids))
            seg_end[-1] = True
            segment_ends += seg_end

        input_ids = np.array(ids, dtype=np.int64)
        asst_mask = np.array(asst, dtype=np.float32)
        seg_ends = np.array(segment_ends, dtype=bool)
        n_assistant = int(asst_mask.sum())
        if n_assistant:
            asst_mask *= 2.0 / np.sqrt(n_assistant + 1)

        labels = np.zeros_like(input_ids)
        labels[:-1] = input_ids[1:]
        labels[seg_ends] = tok.eos_token_id
        loss_masks = np.zeros_like(asst_mask)
        loss_masks[:-1] = asst_mask[1:]
        loss_masks[seg_ends] = asst_mask[seg_ends]
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

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        messages = _format_messages(list(self._data[i]["messages"]))
        if messages is None:
            # Filtered datasets should not contain these, but guard: fall back to next.
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi."},
            ]
        seq = self._text_sequence(messages)
        max_len = self.config.max_sequence_length
        if max_len and len(seq["input_ids"]) > max_len:
            seq = {
                key: value[:max_len] if value.ndim == 1 and len(value) > max_len else value
                for key, value in seq.items()
            }
        return seq
