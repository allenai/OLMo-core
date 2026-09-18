"""ChartVerse chart-reasoning QA for stage-2 SFT.

ChartVerse charts are rendered from generated plotting code, so the corpus covers
layouts the rest of the mixture never sees: 3D plots, hierarchical and Sankey/chord
diagrams, and multi-subplot figures. That is the gap ``charxiv_descriptive`` /
``charxiv_reasoning`` probe -- the existing chart sources (``chart_qa_weighted``,
``dv_qa``, ``figure_qa``, ``plot_qa``, ``cosyn_chart_exp``) are all simple
bar/line/pie.

**Supervision target.** Rows carry a short ``answer`` alongside a long ``cot_solution``
trace and the generating ``code``/``code_solution``. By default only ``answer`` is
supervised, mirroring :class:`MMFineReasonDataset` (which keeps ``<answer>`` and drops
``<think>``): CharXiv and MMMU-Pro are graded on short final answers, and training on
traces shifts Molmo2's output length across every benchmark in the suite.

That default has a measured cost. Profiling 3,600 staged rows found the supervised
``answer`` is **8.0 chars on average** (median 5; 82.5% a single token; 52.2% purely
numeric), while the question is a 163-char multi-step aggregation over the chart
("the first year when the combined share of the two smallest segments surpassed 30%").
So answer-only supervision trains the model to emit a bare scalar for a question it
cannot plausibly answer in one forward pass, discarding ~4k tokens of derivation per
row. Setting :attr:`ChartVerseDatasetConfig.supervise_cot` supervises the derivation
instead, which is the hypothesis that ``cot_solution`` is the part of this corpus worth
paying for.

The trace is **not** in the staged Arrow directory (only ``id``/``images``/``question``/
``answer`` are). It lives in a row-aligned sidecar built by
``OLMo-core/launch_scripts/donovan/dev/stage_chartverse_cot.py`` from the retained
parquet; point :attr:`ChartVerseDatasetConfig.cot_sidecar` at it.

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
from .paths import require_experiment_data_dir
from .sequence_builder import example_rng
from .sft_common import (
    decode_pil_image,
    extract_reasoning_text,
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

#: Columns the loader reads from the staged directory. ``code``/``code_solution`` are
#: deliberately absent; ``cot_solution`` arrives via the sidecar -- see the module docstring.
CHARTVERSE_KEEP_COLUMNS = ("id", "images", "question", "answer")

#: Sidecar subdirectory suffix appended to the subset name by the staging script.
CHARTVERSE_COT_SIDECAR_SUFFIX = "-cot"


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

    supervise_cot: bool = False
    """Supervise the ``cot_solution`` derivation instead of the bare ``answer``.

    Requires :attr:`cot_sidecar` (or a staged directory that carries the column itself).
    Raises the mean supervised length from ~8 characters to ~4k tokens, so a
    ChartVerse row goes from ~3 to ~1 per 16,384-token pack -- compare arms at matched
    *rows consumed*, never at matched steps.
    """

    cot_sidecar: Optional[str] = None
    """Row-aligned ``save_to_disk`` directory holding ``id``/``cot_solution``.

    ``None`` with :attr:`supervise_cot` set defaults to ``<resolved_path()>-cot``. Row *i*
    of the sidecar must be row *i* of the staged subset; the loader asserts matching
    ``id`` on every row it reads, so a misaligned sidecar fails loudly rather than
    silently pairing a derivation with the wrong chart.
    """

    cot_column: str = "cot_solution"
    """Sidecar column holding the derivation."""

    max_crops: int = 8
    max_sequence_length: int = 4096
    loss_token_weighting: str = "root_subsegments_root_tokens"
    seed: int = 0

    skip_overlong: bool = False
    """Skip rows that exceed :attr:`max_sequence_length` instead of right-truncating them.

    :func:`truncate_example` right-truncates and only raises when *every* loss token is
    cut, so a derivation longer than the budget is silently trained without its final
    answer -- supervision of an unfinished thought, which is worse than no supervision.
    Defaults on whenever :attr:`supervise_cot` is set (see :meth:`__post_init__`).
    """

    def __post_init__(self):
        if self.supervise_cot:
            self.skip_overlong = True

    @property
    def name(self) -> str:
        return CHARTVERSE_DATASET_NAME

    def _resolved_data_root(self) -> str:
        if self.data_root is not None:
            return self.data_root
        return require_experiment_data_dir("the staged ChartVerse subsets")

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

        self._cot = None
        if config.supervise_cot:
            side = config.cot_sidecar or (path.rstrip("/") + CHARTVERSE_COT_SIDECAR_SUFFIX)
            if not os.path.isdir(side):
                raise FileNotFoundError(
                    f"supervise_cot is set but no derivation sidecar at {side}. Build it with "
                    "OLMo-core/launch_scripts/donovan/dev/stage_chartverse_cot.py, or point "
                    "cot_sidecar at an existing one."
                )
            # load_hf_dataset (not load_from_disk_compat) so the sidecar can be plain parquet
            # shards: at ~16 KB of text per row a save_to_disk copy would cost a second 31 GB
            # and an equally large HF cache, for a column we only ever read sequentially.
            from .sft_common import load_hf_dataset

            self._cot = load_hf_dataset(
                side, split=config.split, keep_columns=["id", config.cot_column]
            )
            if len(self._cot) != len(self._data):
                raise ValueError(
                    f"ChartVerse derivation sidecar {side} has {len(self._cot)} rows but the "
                    f"subset has {len(self._data)}; the sidecar must be row-aligned."
                )
            log.info(
                "ChartVerse[%s]: supervising %r from sidecar %s (%d rows)",
                config.subset,
                config.cot_column,
                side,
                len(self._cot),
            )

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

        target = self._supervision_target(i, row, answer) if self._cot is not None else answer

        seq = encode_sft_example(
            self.tokenizer,
            pil_images[:1],
            [[(question, target)]],
            max_crops=cfg.max_crops,
            max_images=1,
            loss_token_weighting=cfg.loss_token_weighting,
            shuffle_rng=example_rng(cfg.seed, i),
        )
        if cfg.skip_overlong and len(seq["input_ids"]) > cfg.max_sequence_length:
            # Raise rather than truncate: get_example_with_skip moves to the next row, so an
            # over-budget derivation is dropped whole instead of being supervised headless.
            raise ValueError(
                f"ChartVerse row {i}: {len(seq['input_ids'])} tokens exceeds "
                f"max_sequence_length={cfg.max_sequence_length}"
            )
        return truncate_example(seq, cfg.max_sequence_length)

    def _supervision_target(self, i: int, row: dict, answer: str) -> str:
        """Return the derivation for row ``i``, guaranteed to end in the final answer.

        The sidecar is row-aligned, so the ``id`` check is the only thing standing between a
        misbuilt sidecar and 1.8M chart/derivation mismatches -- keep it unconditional.
        """
        cfg = self.config
        side = self._cot[i]
        side_id, row_id = side.get("id"), row.get("id")
        if side_id is not None and row_id is not None and side_id != row_id:
            raise RuntimeError(
                f"ChartVerse derivation sidecar is misaligned at row {i}: "
                f"sidecar id {side_id!r} != subset id {row_id!r}"
            )
        cot = extract_reasoning_text(side.get(cfg.cot_column), final_answer=answer)
        if not cot:
            raise ValueError(f"ChartVerse row {i} has an empty {cfg.cot_column}")
        return cot

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        return get_example_with_skip(self, index, len(self))
