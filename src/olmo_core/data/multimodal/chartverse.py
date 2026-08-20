"""ChartVerse chart-reasoning QA for stage-2 SFT.

ChartVerse charts are rendered from generated plotting code, so the corpus covers
layouts the rest of the mixture never sees: 3D plots, hierarchical and Sankey/chord
diagrams, and multi-subplot figures. That is the gap ``charxiv_descriptive`` /
``charxiv_reasoning`` probe -- the existing chart sources (``chart_qa_weighted``,
``dv_qa``, ``figure_qa``, ``plot_qa``, ``cosyn_chart_exp``) are all simple
bar/line/pie.

Rows carry a short ``answer`` (1-22 chars) alongside a long ``cot_solution`` trace and
the generating ``code``/``code_solution``. Only ``answer`` is supervised: CharXiv and
MMMU-Pro are graded on short final answers, and training on the traces would shift
Molmo2's output length across every benchmark in the suite. This mirrors
:class:`MMFineReasonDataset`, which likewise keeps ``<answer>`` and drops ``<think>``.

Staged as a HuggingFace ``save_to_disk`` directory under
``$MOLMO_EXPERIMENT_DATA_DIR/chartverse/<subset>/`` (see
``mm_olmo/launch_scripts/donovan/dev/download_v11_datasets.py``).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from olmo_core.config import Config

from .message_sequence import encode_sft_example
from .paths import _DEFAULT_EXPERIMENT_DATA
from .sequence_builder import example_rng
from .sft_common import (
    decode_pil_image,
    get_example_with_skip,
    strip_image_placeholders,
    truncate_example,
)

__all__ = [
    "CHARTVERSE_DATASET_NAME",
    "CHARTVERSE_DEFAULT_SUBSET",
    "ChartVerseDatasetConfig",
    "ChartVerseDataset",
]

log = logging.getLogger(__name__)

#: Mixture source name (single source; the subset is a config field).
CHARTVERSE_DATASET_NAME = "chartverse"

#: On-disk subdirectory staged by ``download_v11_datasets.py``.
CHARTVERSE_DEFAULT_SUBSET = "sft_600k"

#: Columns the loader reads. ``code``/``code_solution``/``cot_solution`` are
#: deliberately absent -- see the module docstring.
CHARTVERSE_KEEP_COLUMNS = ("images", "question", "answer")


@dataclass
class ChartVerseDatasetConfig(Config):
    """Configuration for :class:`ChartVerseDataset`."""

    subset: str = CHARTVERSE_DEFAULT_SUBSET
    """On-disk subdirectory under ``<data_root>/chartverse/``."""

    data_root: Optional[str] = None
    """Root containing the ``chartverse/`` subdirectory (defaults to ``MOLMO_EXPERIMENT_DATA_DIR``)."""

    dataset_path: Optional[str] = None
    """Explicit ``save_to_disk`` directory; overrides :attr:`data_root` + :attr:`subset`."""

    split: str = "train"

    max_crops: int = 8
    max_sequence_length: int = 4096
    loss_token_weighting: str = "root_subsegments_root_tokens"
    seed: int = 0

    @property
    def name(self) -> str:
        return CHARTVERSE_DATASET_NAME

    def _resolved_data_root(self) -> str:
        if self.data_root is not None:
            return self.data_root
        return os.environ.get("MOLMO_EXPERIMENT_DATA_DIR", _DEFAULT_EXPERIMENT_DATA)

    def resolved_path(self) -> str:
        if self.dataset_path is not None:
            return self.dataset_path
        return os.path.join(self._resolved_data_root(), "chartverse", self.subset)

    def build(self, tokenizer) -> "ChartVerseDataset":
        return ChartVerseDataset(self, tokenizer)


class ChartVerseDataset:
    """Map-style dataset yielding packed ChartVerse single-image QA examples."""

    def __init__(self, config: ChartVerseDatasetConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self._warned = 0

        if config.split != "train":
            raise ValueError(f"ChartVerse only provides the 'train' split, not {config.split!r}")

        path = config.resolved_path()
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"ChartVerse subset not found at {path}. "
                "Set MOLMO_EXPERIMENT_DATA_DIR or dataset_path, or stage it via "
                "mm_olmo/launch_scripts/donovan/dev/download_v11_datasets.py."
            )

        from .dataset_compat import load_from_disk_compat

        loaded = load_from_disk_compat(path)
        if hasattr(loaded, "keys") and config.split in loaded:
            self._data = loaded[config.split]
        else:
            self._data = loaded

        log.info("ChartVerse[%s]: loaded %d rows from %s", config.subset, len(self._data), path)

    def __len__(self) -> int:
        return len(self._data)

    def _build(self, i: int) -> Dict[str, np.ndarray]:
        cfg = self.config
        row = self._data[i]

        # Any inline <image> marker is stripped: the image is supplied as an explicit
        # token block by encode_sft_example instead.
        question: str = strip_image_placeholders(row.get("question"))
        answer: str = str(row.get("answer") or "").strip()
        if not question or not answer:
            raise ValueError(f"ChartVerse row {i} has empty question or answer")

        raw_images = row.get("images") or []
        if not isinstance(raw_images, list):
            raw_images = [raw_images]
        pil_images = [decode_pil_image(im) for im in raw_images if im is not None]
        if not pil_images:
            raise ValueError(f"ChartVerse row {i} has no decodable image")

        seq = encode_sft_example(
            self.tokenizer,
            pil_images[:1],
            [[(question, answer)]],
            max_crops=cfg.max_crops,
            max_images=1,
            loss_token_weighting=cfg.loss_token_weighting,
            shuffle_rng=example_rng(cfg.seed, i),
        )
        return truncate_example(seq, cfg.max_sequence_length)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        return get_example_with_skip(self, index, len(self))
