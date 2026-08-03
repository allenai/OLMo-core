"""PixMo Cap QA as user QA demo dataset for SFT."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from olmo_core.config import Config
from olmo_core.nn.vision.molmo2_tokens import Molmo2TokenIds

from .message_sequence import encode_sft_example
from .paths import PIXMO_DATASETS

__all__ = ["PixMoCapQaDatasetConfig", "PixMoCapQaDataset"]


def _parse_cap_qa_turns(question: str, answer: str) -> List[Tuple[str, str]]:
    parts = re.split(r"\s*(\[USER\]|\[ASSISTANT\])\s*", question)
    assert parts[0] == "" and parts[-1] == ""
    parts = parts[1:-1]
    messages: List[str] = []
    for part_ix, part in enumerate(parts):
        if part_ix % 4 == 1:
            messages.append(part)
        elif part_ix % 4 == 3:
            messages.append(part)
    messages.append(answer)
    turns: List[Tuple[str, str]] = []
    for u in range(0, len(messages) - 1, 2):
        turns.append((messages[u], messages[u + 1]))
    return turns


@dataclass
class PixMoCapQaDatasetConfig(Config):
    max_crops: int = 8
    loss_token_weighting: str = "root_subsegments_root_tokens"
    token_ids: Molmo2TokenIds = field(default_factory=Molmo2TokenIds)
    seed: int = 0

    def build(self, tokenizer) -> "PixMoCapQaDataset":
        return PixMoCapQaDataset(self, tokenizer)


class PixMoCapQaDataset:
    def __init__(self, config: PixMoCapQaDatasetConfig, tokenizer):
        from .dataset_compat import load_from_disk_compat

        self.config = config
        self.tokenizer = tokenizer
        ds = load_from_disk_compat(f"{PIXMO_DATASETS}/cap-qa")
        self._data = ds["train"] if "train" in ds else ds

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> Dict:
        row = self._data[index]
        turns: List[Tuple[str, str]] = []
        for qs, ans in zip(row["question"], row["answer"]):
            if not ans or not str(ans).strip():
                continue
            turns.extend(_parse_cap_qa_turns(qs, ans))
        if not turns:
            raise ValueError(f"No valid QA pairs at index {index}")
        image = row["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        return encode_sft_example(
            self.tokenizer,
            image,
            turns,
            max_crops=self.config.max_crops,
            loss_token_weighting=self.config.loss_token_weighting,
            token_ids=self.config.token_ids,
            shuffle_rng=np.random.RandomState(self.config.seed + index),
        )
