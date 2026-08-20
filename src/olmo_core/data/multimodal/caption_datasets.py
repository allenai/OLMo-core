"""Image-to-caption sources (ArxivCap, OmniScience, VisText, Chart2Text) for stage-2 SFT.

All four are bare ``(image, caption)`` pairs, so they ride the ``long_caption`` /
``short_caption`` styles in :class:`SftFormatter`: a caption prompt is sampled per
example and no ``"style:"`` prefix is emitted (both are in ``DEMO_STYLES``). This is
the same path ``PixMoCapDataset._getitem_sft_demo`` takes for the caption source
already in the v9 mixture. Keeping one loader for all four means the per-corpus
schema differences are normalized once, in the staging script, rather than in four
places here.

* ``arxivcap`` -- real arXiv figures, i.e. the same source distribution CharXiv is
  sampled from. Complements ``finevision_arxivqa`` (figure *QA*) with figure captions.
* ``omniscience`` -- 300+ STEM sub-disciplines with MLLM-refined captions, aimed at
  MMMU-Pro's discipline breadth.
* ``vistext`` -- small (12k charts) but unusually well targeted: its L1 captions
  describe title, axes and encodings, which is close to a verbatim match for the
  ``charxiv_descriptive`` question templates.
* ``chart2text`` -- Statista/Pew charts, broadening chart provenance beyond academic
  figures. Captions are somewhat templated, so it is weighted low.

Weighting is ``root_subsegments_root_tokens`` (matching the ``sft_demo`` caption path
the mixture actually uses, not ``PixMoCapDatasetConfig``'s own default). The
per-token ``2/sqrt(n_response_tokens)`` factor matters more here than anywhere else:
OmniScience captions run ~400 words, and without it a few percent of caption data
would carry a wildly disproportionate share of the gradient mass.

Staged as ``save_to_disk`` directories under
``$MOLMO_EXPERIMENT_DATA_DIR/captions/<subset>/`` by
``mm_olmo/launch_scripts/donovan/dev/download_v11_datasets.py``, which also applies
the CharXiv/MMMU-Pro decontamination filter.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from olmo_core.config import Config

from .message_sequence import encode_sft_example
from .paths import _DEFAULT_EXPERIMENT_DATA
from .sequence_builder import example_rng
from .sft_common import decode_pil_image, get_example_with_skip, truncate_example
from .sft_formatter import SftFormatter

__all__ = [
    "CAPTION_DATASET_NAMES",
    "CAPTION_DATASET_STYLES",
    "CaptionDatasetConfig",
    "CaptionDataset",
]

log = logging.getLogger(__name__)

#: Mixture source names; each maps to a subdirectory of ``captions/``.
CAPTION_DATASET_NAMES: Tuple[str, ...] = (
    "arxivcap",
    "omniscience",
    "vistext",
    "chart2text",
)

#: Per-source caption style. Chart2Text captions are one or two sentences, so it uses
#: the short-caption prompt pool; the rest are long-form.
CAPTION_DATASET_STYLES: Dict[str, str] = {
    "arxivcap": "long_caption",
    "omniscience": "long_caption",
    "vistext": "long_caption",
    "chart2text": "short_caption",
}

_CAPTION_COLUMNS = ("caption", "text", "recaption")
_IMAGE_COLUMNS = ("image", "images")


@dataclass
class CaptionDatasetConfig(Config):
    """Configuration for :class:`CaptionDataset`."""

    subset: str
    """One of :data:`CAPTION_DATASET_NAMES`."""

    data_root: Optional[str] = None
    """Root containing ``captions/`` (defaults to ``MOLMO_EXPERIMENT_DATA_DIR``)."""

    dataset_path: Optional[str] = None
    """Explicit ``save_to_disk`` directory; overrides :attr:`data_root` + :attr:`subset`."""

    split: str = "train"

    style: Optional[str] = None
    """Caption style; defaults to :data:`CAPTION_DATASET_STYLES` for :attr:`subset`."""

    max_crops: int = 8
    max_sequence_length: int = 4096
    loss_token_weighting: str = "root_subsegments_root_tokens"
    seed: int = 0

    @property
    def name(self) -> str:
        return self.subset

    def resolved_style(self) -> str:
        if self.style is not None:
            return self.style
        return CAPTION_DATASET_STYLES.get(self.subset, "long_caption")

    def _resolved_data_root(self) -> str:
        if self.data_root is not None:
            return self.data_root
        return os.environ.get("MOLMO_EXPERIMENT_DATA_DIR", _DEFAULT_EXPERIMENT_DATA)

    def resolved_path(self) -> str:
        if self.dataset_path is not None:
            return self.dataset_path
        return os.path.join(self._resolved_data_root(), "captions", self.subset)

    def build(self, tokenizer) -> "CaptionDataset":
        return CaptionDataset(self, tokenizer)


class CaptionDataset:
    """Map-style dataset yielding packed image-caption examples."""

    def __init__(self, config: CaptionDatasetConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self._warned = 0
        self._formatter = SftFormatter(seed=config.seed)

        if config.split != "train":
            raise ValueError(
                f"{config.subset} only provides the 'train' split, not {config.split!r}"
            )

        path = config.resolved_path()
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Caption subset {config.subset!r} not found at {path}. "
                "Set MOLMO_EXPERIMENT_DATA_DIR or dataset_path, or stage it via "
                "mm_olmo/launch_scripts/donovan/dev/download_v11_datasets.py."
            )

        from .dataset_compat import load_from_disk_compat

        loaded = load_from_disk_compat(path)
        if hasattr(loaded, "keys") and config.split in loaded:
            self._data = loaded[config.split]
        else:
            self._data = loaded

        log.info(
            "Captions[%s]: loaded %d rows from %s",
            config.subset,
            len(self._data),
            path,
        )

    def __len__(self) -> int:
        return len(self._data)

    @staticmethod
    def _first_present(row, columns):
        for col in columns:
            value = row.get(col)
            if value:
                return value
        return None

    def _build(self, i: int) -> Dict[str, np.ndarray]:
        cfg = self.config
        row = self._data[i]

        caption = self._first_present(row, _CAPTION_COLUMNS)
        caption = (caption or "").strip() if isinstance(caption, str) else ""
        if not caption:
            raise ValueError(f"{cfg.subset} row {i} has no caption")

        raw_image = self._first_present(row, _IMAGE_COLUMNS)
        if isinstance(raw_image, list):
            raw_image = raw_image[0] if raw_image else None
        if raw_image is None:
            raise ValueError(f"{cfg.subset} row {i} has no image")
        pil_image = decode_pil_image(raw_image)

        rng = example_rng(cfg.seed, i)
        formatted = {"style": cfg.resolved_style(), "caption": caption, "text": caption}
        turns = self._formatter.format_turns(formatted, index=i, rng=rng)
        seq = encode_sft_example(
            self.tokenizer,
            [pil_image],
            turns,
            max_crops=cfg.max_crops,
            max_images=1,
            loss_token_weighting=cfg.loss_token_weighting,
            shuffle_rng=rng,
        )
        return truncate_example(seq, cfg.max_sequence_length)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        return get_example_with_skip(self, index, len(self))
