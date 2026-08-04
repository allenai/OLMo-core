"""Academic VQA dataset wrapper for image-only-v9."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


from olmo_core.config import Config

from .academic.registry import ACADEMIC_REGISTRY, build_academic_data, format_academic_example
from .message_sequence import encode_sft_example
from .sequence_builder import example_rng
from .sft_formatter import SftFormatter

__all__ = ["AcademicDatasetConfig", "AcademicDataset", "ACADEMIC_DATASET_NAMES"]

ACADEMIC_DATASET_NAMES = sorted(ACADEMIC_REGISTRY.keys())


@dataclass
class AcademicDatasetConfig(Config):
    name: str
    max_crops: int = 8
    loss_token_weighting: str = "root_subsegments_root_tokens"
    message_weight: float | None = None
    seed: int = 0

    def build(self, tokenizer) -> "AcademicDataset":
        return AcademicDataset(self, tokenizer)


class AcademicDataset:
    def __init__(self, config: AcademicDatasetConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self._formatter = SftFormatter(seed=config.seed)
        self._data = build_academic_data(config.name, split="train")
        self._len = len(self._data)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # One mm_olmo-derived rng threads the dataset formatter, prompt templating,
        # and branch shuffle (mm_olmo dataset.py:68) — including pixmo_clocks, whose
        # augmentation draws must vary per example.
        rng = example_rng(self.config.seed, index)
        row = self._data[index]
        formatted = format_academic_example(self.config.name, row, rng)
        turns = self._formatter.format_branches(formatted, index=index, rng=rng)
        example_weight = formatted.get("weight")
        message_weight = example_weight if example_weight is not None else self.config.message_weight
        return encode_sft_example(
            self.tokenizer,
            formatted["image"],
            turns,
            max_crops=self.config.max_crops,
            loss_token_weighting=self.config.loss_token_weighting,
            message_weight=message_weight,
            shuffle_rng=rng,
        )
