"""Academic VQA dataset wrapper for image-only-v9."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from olmo_core.config import Config
from olmo_core.nn.vision.molmo2_tokens import Molmo2TokenIds

from .academic.registry import (
    ACADEMIC_REGISTRY,
    build_academic_data,
    format_academic_example,
)
from .message_sequence import encode_sft_example
from .sft_common import SftMessageFormat, sft_example_rng, validate_sft_message_format
from .sft_formatter import SftFormatter

__all__ = ["AcademicDatasetConfig", "AcademicDataset", "ACADEMIC_DATASET_NAMES"]

ACADEMIC_DATASET_NAMES = sorted(ACADEMIC_REGISTRY.keys())


@dataclass
class AcademicDatasetConfig(Config):
    name: str
    max_crops: int = 8
    loss_token_weighting: str = "root_subsegments_root_tokens"
    token_ids: Molmo2TokenIds = field(default_factory=Molmo2TokenIds)
    message_format: SftMessageFormat = "qwen3"
    message_weight: float | None = None
    seed: int = 0

    def build(self, tokenizer) -> "AcademicDataset":
        return AcademicDataset(self, tokenizer)


class AcademicDataset:
    def __init__(self, config: AcademicDatasetConfig, tokenizer):
        validate_sft_message_format(
            config.message_format,
            tokenizer=tokenizer,
            token_ids=config.token_ids,
        )
        self.config = config
        self.tokenizer = tokenizer
        self._formatter = SftFormatter(seed=config.seed)
        self._data = build_academic_data(config.name, split="train")
        self._len = len(self._data)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.get(index, 0)

    def get(self, index: int, epoch: int = 0) -> Dict[str, Any]:
        """Build one example with the released epoch-aware Stage 2 RNG stream."""
        # One mm_olmo-derived rng threads the dataset formatter, prompt templating,
        # and branch shuffle (mm_olmo dataset.py:68) — including pixmo_clocks, whose
        # augmentation draws must vary per example.
        rng = sft_example_rng(
            self.config.seed,
            index,
            epoch,
            self.config.message_format,
        )
        row = self._data[index]
        formatted = format_academic_example(self.config.name, row, rng)
        turns = self._formatter.format_branches(formatted, index=index, rng=rng)
        example_weight = formatted.get("weight")
        message_weight = (
            example_weight if example_weight is not None else self.config.message_weight
        )
        return encode_sft_example(
            self.tokenizer,
            formatted["image"],
            turns,
            max_crops=self.config.max_crops,
            loss_token_weighting=self.config.loss_token_weighting,
            token_ids=self.config.token_ids,
            message_format=self.config.message_format,
            message_weight=message_weight,
            shuffle_rng=rng,
        )
