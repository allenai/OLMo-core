"""DynaMath-style programmatic visual math QA for stage-2 SFT.

Each training variant is a HuggingFace ``datasets`` directory saved under
``$MOLMO_EXPERIMENT_DATA_DIR/dynamath/<variant>/`` (see mm_olmo
``DynaMathConfig`` and ``molmo-experimental-data/dynamath-benchmark``).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from olmo_core.config import Config

from .message_sequence import encode_sft_example
from .paths import _DEFAULT_EXPERIMENT_DATA
from .sequence_builder import example_rng
from .sft_common import decode_pil_image, get_example_with_skip, truncate_example

__all__ = [
    "DynaMathDatasetConfig",
    "DynaMathDataset",
    "DYNAMATH_TRAINING_VARIANTS",
    "dynamath_variant_from_name",
]

log = logging.getLogger(__name__)

# Allowlisted training variants (339 programs each); equal weight in image-only-v10.
DYNAMATH_TRAINING_VARIANTS: Tuple[str, ...] = (
    "dynamath_seed_42_999",
    "dynamath_seed_100_1000",
    "dynamath_seed_101_1001",
    "dynamath_seed_102_1002",
    "dynamath_seed_103_1003",
    "dynamath_seed_104_1004",
)

_METADATA_FIELDS = (
    "answer_type",
    "subject",
    "level",
    "question_id",
    "variant",
)


def dynamath_variant_from_name(name: str) -> str:
    """Map a mixture dataset name (``dynamath_seed_42_999``) to on-disk variant dir."""
    prefix = "dynamath_"
    if not name.startswith(prefix):
        raise ValueError(f"Not a DynaMath dataset name: {name!r}")
    variant = name[len(prefix) :]
    if not variant:
        raise ValueError(f"Empty DynaMath variant in {name!r}")
    return variant


@dataclass
class DynaMathDatasetConfig(Config):
    """Configuration for :class:`DynaMathDataset`."""

    variant: str
    """On-disk subdirectory under ``<data_root>/dynamath/`` (e.g. ``seed_42_999``)."""

    data_root: Optional[str] = None
    """Root containing the ``dynamath/`` subdirectory (defaults to ``MOLMO_EXPERIMENT_DATA_DIR``)."""

    dataset_path: Optional[str] = None
    """Explicit ``save_to_disk`` directory; overrides :attr:`data_root` + :attr:`variant`."""

    split: str = "train"

    max_crops: int = 8
    max_sequence_length: int = 4096
    loss_token_weighting: str = "root_subsegments_root_tokens"
    seed: int = 0

    @property
    def name(self) -> str:
        return f"dynamath_{self.variant}"

    def _resolved_data_root(self) -> str:
        if self.data_root is not None:
            return self.data_root
        return os.environ.get("MOLMO_EXPERIMENT_DATA_DIR", _DEFAULT_EXPERIMENT_DATA)

    def resolved_path(self) -> str:
        if self.dataset_path is not None:
            return self.dataset_path
        return os.path.join(self._resolved_data_root(), "dynamath", self.variant)

    def build(self, tokenizer) -> "DynaMathDataset":
        return DynaMathDataset(self, tokenizer)


class DynaMathDataset:
    """Map-style dataset yielding packed DynaMath single-image QA examples."""

    def __init__(self, config: DynaMathDatasetConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self._warned = 0

        if config.split != "train":
            raise ValueError(
                f"DynaMath only provides the 'train' split, not {config.split!r}"
            )

        path = config.resolved_path()
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"DynaMath variant not found at {path}. "
                "Set MOLMO_EXPERIMENT_DATA_DIR or dataset_path, or generate variants "
                "via molmo-experimental-data/dynamath-benchmark."
            )

        from .dataset_compat import load_from_disk_compat

        loaded = load_from_disk_compat(path)
        if hasattr(loaded, "keys") and config.split in loaded:
            self._data = loaded[config.split]
        else:
            self._data = loaded

        log.info("DynaMath[%s]: loaded %d rows from %s", config.variant, len(self._data), path)

    def __len__(self) -> int:
        return len(self._data)

    def _build(self, i: int) -> Dict[str, np.ndarray]:
        cfg = self.config
        row = self._data[i]

        question = (row.get("question") or "").strip()
        answer = (row.get("answer") or "").strip()
        if not question or not answer:
            raise ValueError(
                f"DynaMath example {row.get('question_id')} has empty question or answer"
            )

        pil_image = decode_pil_image(row["image"])
        seq = encode_sft_example(
            self.tokenizer,
            [pil_image],
            [[(question, answer)]],
            max_crops=cfg.max_crops,
            max_images=1,
            loss_token_weighting=cfg.loss_token_weighting,
            shuffle_rng=example_rng(cfg.seed, i),
        )
        return truncate_example(seq, cfg.max_sequence_length)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        return get_example_with_skip(self, index, len(self))
