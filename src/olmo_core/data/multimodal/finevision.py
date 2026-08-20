"""FineVision instruction-tuning datasets (``HuggingFaceM4/FineVision``).

One loader for any FineVision config, since they all share the same row schema:

* ``images`` — a list of images.
* ``texts`` — a list of ``{"user": ..., "assistant": ...}`` turns; the user side holds the
  instruction, the assistant side the supervision target.
* ``source`` plus four quality signals with per-turn ratings and a per-row minimum:
  ``formatting``, ``visual_dependency``, ``image_correspondence`` and ``relevance``. The
  ``*_min`` columns are exposed as optional filters on :class:`FineVisionDatasetConfig`.

A row is assembled as ONE sequential conversation branch (loss on every assistant turn,
one EOS target at the end) by
:func:`~olmo_core.data.multimodal.message_sequence.encode_sft_example`, i.e. the same
qwen3 layout as every other stage-2 source: no BOS, the image token block(s) inside the
first user turn, ``Image {i+1}`` prefixes when a row carries several images.

**Loading.** Configs resolve to parquet shards or ``save_to_disk`` directories under
:data:`FINEVISION_ROOT`. image-only-v10 subsets are symlinked from mm_olmo's prepared
copies at ``$MOLMO_EXPERIMENT_DATA_DIR/finevision/<config>`` (see
``launch_scripts/donovan/env/setup-finevision-v10-symlinks.sh``). Set
:attr:`FineVisionDatasetConfig.hub_repo` only when you explicitly want a hub fetch.

Configs verified against local parquet copies on weka (see :data:`FINEVISION_ROOT`):

===========================  =========  ================  =========================
config                       rows       ``<image>`` mark  notes
===========================  =========  ================  =========================
``visualwebinstruct(filtered)``  263,581  leading, 1/row  web visual instruction
``mavis_math_rule_geo``           99,986  leading, 1/row  synthetic geometry w/ CoT
``mavis_math_metagen``            87,348  **none**        synthetic math w/ CoT
``geo170k(align)``                35,297  **none**        geometry caption/alignment
``geo170k(qa)``                   12,101  **none**        geometry multiple-choice
===========================  =========  ================  =========================

:image-only-v10 subsets (hub configs, see :data:`FINEVISION_V10_CONFIGS`):

* ``densefusion_1m``, ``objects365_qa``, ``arxivqa``, ``geomverse``, ``DoclingMatix``

Every one of those is single-turn with exactly one image per row. Note that three of them
carry **no** ``<image>`` marker at all: the image block is positioned from the ``images``
column, not from the marker, so both layouts work identically (any marker present is
stripped so it is never tokenized as literal text).

.. warning::
    ``image_correspondence_min`` is very low on the synthetic-geometry configs — 75% of
    ``geo170k(qa)`` and 63% of ``mavis_math_metagen`` rows score 1 — so a naive
    ``min_image_correspondence=4`` discards most of them. Prefer
    ``min_visual_dependency``, which is 4-5 for ~95% of those rows.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from olmo_core.config import Config

from .message_sequence import encode_sft_example
from .sequence_builder import example_rng
from .sft_common import (
    decode_pil_image,
    get_example_with_skip,
    load_hf_dataset,
    strip_image_placeholders,
    truncate_example,
)

__all__ = [
    "FineVisionDatasetConfig",
    "FineVisionDataset",
    "VisualWebInstructDatasetConfig",
    "FINEVISION_ROOT",
    "FINEVISION_HUB_REPO",
    "FINEVISION_V10_CONFIGS",
    "FINEVISION_V10_DATASET_NAMES",
    "FINEVISION_V10_SHUFFLE_SEED",
    "build_finevision_v10_config",
    "finevision_v10_hub_name",
    "finevision_index_cache_path",
    "load_finevision_index_cache",
    "save_finevision_index_cache",
]

log = logging.getLogger(__name__)

FINEVISION_ROOT = "/weka/oe-training-default/mm-olmo/hf_datasets/HuggingFaceM4___FineVision"
"""Directory holding one subdirectory of parquet shards per downloaded FineVision config."""

FINEVISION_HUB_REPO = "HuggingFaceM4/FineVision"
"""HuggingFace dataset repo for all FineVision configs."""

# mm_olmo image-only-v10: shuffle then cap (see download_finevision.py).
FINEVISION_V10_SHUFFLE_SEED = 6198

# Hub config name -> row cap used in mm_olmo image-only-v10 after single-image filtering.
FINEVISION_V10_CONFIGS: Dict[str, int] = {
    "densefusion_1m": 100_000,
    "objects365_qa": 100_000,
    "arxivqa": 50_000,
    "geomverse": 50_000,
    "DoclingMatix": 100_000,
}

# Mixture dataset name (``finevision_*``) -> hub config name.
FINEVISION_V10_HUB_ALIASES: Dict[str, str] = {
    "finevision_densefusion_1m": "densefusion_1m",
    "finevision_objects365_qa": "objects365_qa",
    "finevision_arxivqa": "arxivqa",
    "finevision_geomverse": "geomverse",
    "finevision_doclingmatix": "DoclingMatix",
}

FINEVISION_V10_DATASET_NAMES: Tuple[str, ...] = tuple(FINEVISION_V10_HUB_ALIASES)

_QUALITY_COLUMNS = {
    "min_formatting": "formatting_min",
    "min_visual_dependency": "visual_dependency_min",
    "min_image_correspondence": "image_correspondence_min",
    "min_relevance": "relevance_min",
}


def finevision_v10_hub_name(dataset_name: str) -> str:
    """Map a mixture source name (``finevision_densefusion_1m``) to a hub config."""
    if dataset_name not in FINEVISION_V10_HUB_ALIASES:
        raise KeyError(
            f"Unknown FineVision v10 dataset {dataset_name!r}; "
            f"expected one of: {', '.join(FINEVISION_V10_DATASET_NAMES)}"
        )
    return FINEVISION_V10_HUB_ALIASES[dataset_name]


def _finevision_index_cache_dir(config: "FineVisionDatasetConfig") -> Optional[str]:
    if config.index_cache_dir == "":
        return None
    if config.index_cache_dir is not None:
        return config.index_cache_dir
    return os.environ.get(
        "FINEVISION_INDEX_CACHE_DIR",
        os.path.join(FINEVISION_ROOT, ".index_cache"),
    )


def _finevision_index_source_key(config: "FineVisionDatasetConfig") -> str:
    if config.dataset_path is not None:
        source = config.dataset_path
    elif config.uses_hub():
        cache = config.cache_dir or os.environ.get("HF_DATASETS_CACHE", "")
        source = f"hub:{config.hub_repo}:{config.config_name}:{config.split}:{cache}"
    else:
        source = config.resolved_path()
    return hashlib.sha256(source.encode()).hexdigest()[:16]


def _finevision_index_filter_key(config: "FineVisionDatasetConfig") -> str:
    payload = {
        "split": config.split,
        "texts_column": config.texts_column,
        "images_column": config.images_column,
        "quality": {attr: getattr(config, attr) for attr in _QUALITY_COLUMNS},
        "require_single_image": config.require_single_image,
        "max_rows": config.max_rows,
        "shuffle_seed": config.shuffle_seed,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest[:16]


def finevision_index_cache_path(
    config: "FineVisionDatasetConfig", table_rows: int
) -> Optional[str]:
    """Return the on-disk cache path for a filtered row index, if caching is enabled."""
    cache_dir = _finevision_index_cache_dir(config)
    if cache_dir is None:
        return None
    fname = (
        f"{config.config_name}__{_finevision_index_source_key(config)}"
        f"__{_finevision_index_filter_key(config)}__rows{table_rows}.npz"
    )
    return os.path.join(cache_dir, fname)


def load_finevision_index_cache(
    config: "FineVisionDatasetConfig", table_rows: int
) -> tuple[bool, Optional[np.ndarray]]:
    """Load a cached filtered index.

    :returns: ``(hit, index)`` where ``index`` is ``None`` when all rows are kept.
    """
    path = finevision_index_cache_path(config, table_rows)
    if path is None or not os.path.isfile(path):
        return False, None
    try:
        with np.load(path, allow_pickle=False) as data:
            cached_rows = int(data["table_rows"])
            use_full = bool(int(data["use_full"]))
            if cached_rows != table_rows:
                return False, None
            if use_full:
                log.info(
                    "FineVision[%s]: loaded cached full-table index (%d rows)",
                    config.config_name,
                    table_rows,
                )
                return True, None
            positions = np.asarray(data["positions"], dtype=np.int64)
            log.info(
                "FineVision[%s]: loaded cached filtered index (%d / %d rows)",
                config.config_name,
                len(positions),
                table_rows,
            )
            return True, positions
    except (OSError, KeyError, ValueError) as exc:
        log.warning(
            "FineVision[%s]: ignoring corrupt index cache %s (%s)", config.config_name, path, exc
        )
        return False, None


def save_finevision_index_cache(
    config: "FineVisionDatasetConfig", table_rows: int, index: Optional[np.ndarray]
) -> None:
    """Persist a filtered row index to disk."""
    path = finevision_index_cache_path(config, table_rows)
    if path is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    use_full = index is None or len(index) == table_rows
    tmp_base = path.removesuffix(".npz") + ".tmp"
    if use_full:
        np.savez_compressed(
            tmp_base, table_rows=table_rows, use_full=1, positions=np.array([], dtype=np.int64)
        )
    else:
        np.savez_compressed(
            tmp_base,
            table_rows=table_rows,
            use_full=0,
            positions=np.asarray(index, dtype=np.int64),
        )
    os.replace(tmp_base + ".npz", path)
    log.info(
        "FineVision[%s]: wrote index cache (%d / %d rows) to %s",
        config.config_name,
        table_rows if use_full else len(index),
        table_rows,
        path,
    )


def build_finevision_v10_config(
    config_name: str,
    *,
    root: str = FINEVISION_ROOT,
    **kwargs,
) -> FineVisionDatasetConfig:
    """Return a config for one image-only-v10 FineVision subset.

    Reads from :data:`FINEVISION_ROOT` / ``root`` (local parquet or ``save_to_disk``
    directory). v10 subsets are symlinked from mm_olmo's prepared copies under
    ``$MOLMO_EXPERIMENT_DATA_DIR/finevision/<config>``.

    Applies mm_olmo v10 defaults: single-image rows only, shuffle seed
    :data:`FINEVISION_V10_SHUFFLE_SEED`, and the per-config row cap from
    :data:`FINEVISION_V10_CONFIGS`.

    :raises KeyError: If ``config_name`` is not in :data:`FINEVISION_V10_CONFIGS`.
    """
    if config_name not in FINEVISION_V10_CONFIGS:
        available = ", ".join(sorted(FINEVISION_V10_CONFIGS))
        raise KeyError(
            f"Unknown FineVision v10 config {config_name!r}; expected one of: {available}"
        )
    return FineVisionDatasetConfig(
        config_name=config_name,
        root=root,
        max_rows=FINEVISION_V10_CONFIGS[config_name],
        require_single_image=True,
        shuffle_seed=FINEVISION_V10_SHUFFLE_SEED,
        **kwargs,
    )


@dataclass
class FineVisionDatasetConfig(Config):
    """Configuration for :class:`FineVisionDataset`."""

    config_name: str = "visualwebinstruct(filtered)"
    """FineVision config to load, e.g. ``"mavis_math_rule_geo"`` or ``"geo170k(qa)"``.
    Resolved against :attr:`root` unless :attr:`dataset_path` or :attr:`hub_repo` is set."""

    root: str = FINEVISION_ROOT
    """Directory containing one subdirectory per FineVision config (local parquet layout)."""

    dataset_path: Optional[str] = None
    """Explicit parquet directory / glob / file, or a ``save_to_disk`` Arrow directory.
    Overrides :attr:`root` + :attr:`config_name` and :attr:`hub_repo` when set."""

    hub_repo: Optional[str] = None
    """When set (and :attr:`dataset_path` is unset), load via
    ``load_dataset(hub_repo, name=config_name, ...)`` instead of local parquet."""

    cache_dir: Optional[str] = None
    """HuggingFace datasets cache directory for hub loads (defaults to ``HF_DATASETS_CACHE``)."""

    index_cache_dir: Optional[str] = None
    """Directory for cached row-filter indexes. Defaults to ``FINEVISION_INDEX_CACHE_DIR``
    or ``$FINEVISION_ROOT/.index_cache``. Set to ``""`` to disable caching."""

    split: str = "train"

    texts_column: str = "texts"
    """Column holding the list of ``{"user", "assistant"}`` turns."""

    images_column: str = "images"

    min_formatting: Optional[int] = None
    """Keep rows with ``formatting_min >=`` this (1-5; how well-formed the answer is)."""

    min_visual_dependency: Optional[int] = None
    """Keep rows with ``visual_dependency_min >=`` this (1-5; does the answer need the
    image?). Usually the most effective filter for image-grounded training."""

    min_image_correspondence: Optional[int] = None
    """Keep rows with ``image_correspondence_min >=`` this (1-5). See the module-level
    warning before using this on the synthetic-geometry configs."""

    min_relevance: Optional[int] = None
    """Keep rows with ``relevance_min >=`` this (1-5)."""

    require_single_image: bool = False
    """Keep only rows with exactly one image (mm_olmo image-only-v10 uses this)."""

    max_rows: Optional[int] = None
    """Cap the number of rows after filtering. When set, rows are subsampled with
    :attr:`shuffle_seed` (mm_olmo v10 download script semantics)."""

    shuffle_seed: int = FINEVISION_V10_SHUFFLE_SEED
    """RNG seed for :attr:`max_rows` subsampling."""

    max_crops: int = 8
    """Max high-res crops *per image*. Rows with several images cost a multiple of this."""

    max_images: int = 5
    """Max images per row (extra images are dropped, matching the stage-2 budget)."""

    max_sequence_length: int = 4096
    loss_token_weighting: str = "root_subsegments_root_tokens"
    seed: int = 0

    def uses_hub(self) -> bool:
        """Whether this config loads from the HuggingFace hub."""
        return self.dataset_path is None and self.hub_repo is not None

    def resolved_path(self) -> str:
        """The directory this config will read for local parquet / save_to_disk layouts.

        :returns: :attr:`dataset_path` if set, else ``root/config_name``.
        """
        if self.dataset_path is not None:
            return self.dataset_path
        return os.path.join(self.root, self.config_name)

    def build(self, tokenizer) -> "FineVisionDataset":
        """Construct the dataset.

        :param tokenizer: A Molmo2 chat tokenizer.
        """
        return FineVisionDataset(self, tokenizer)


@dataclass
class VisualWebInstructDatasetConfig(FineVisionDatasetConfig):
    """:class:`FineVisionDatasetConfig` pinned to ``visualwebinstruct(filtered)``."""

    config_name: str = "visualwebinstruct(filtered)"


class FineVisionDataset:
    """Map-style dataset yielding packed FineVision instruction examples."""

    def __init__(self, config: FineVisionDatasetConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        self._data = self._load_table(config)
        self._index = self._build_index()
        self._warned = 0

    def _build_index(self) -> Optional[np.ndarray]:
        n = len(self._data)
        hit, cached = load_finevision_index_cache(self.config, n)
        if hit:
            return cached
        index = self._compute_index(n)
        save_finevision_index_cache(self.config, n, index)
        return index

    def _compute_index(self, n: int) -> Optional[np.ndarray]:
        """Row positions kept after quality / single-image filters and optional subsampling."""
        cfg = self.config
        keep = np.ones(n, dtype=bool)

        active_quality = {
            column: getattr(cfg, attr)
            for attr, column in _QUALITY_COLUMNS.items()
            if getattr(cfg, attr) is not None
        }
        for column, threshold in active_quality.items():
            if column not in self._data.column_names:
                log.warning("FineVision: no %r column; ignoring that filter", column)
                continue
            values = np.array(
                [-np.inf if v is None else float(v) for v in self._data[column]],
                dtype=np.float64,
            )
            keep &= values >= threshold

        if cfg.require_single_image:
            img_col = cfg.images_column
            # Vectorized check: count images per row for kept rows only
            for i in np.nonzero(keep)[0]:
                images = self._data[i].get(img_col) or []
                keep[i] = len(images) == 1

        positions = np.nonzero(keep)[0]
        if active_quality:
            log.info(
                "FineVision[%s]: kept %d / %d rows after quality filtering (%s)",
                cfg.config_name,
                len(positions),
                n,
                ", ".join(f"{c}>={t}" for c, t in active_quality.items()),
            )

        if cfg.require_single_image:
            log.info(
                "FineVision[%s]: %d / %d rows have exactly one image",
                cfg.config_name,
                len(positions),
                n,
            )

        if cfg.max_rows is not None and len(positions) > cfg.max_rows:
            rng = np.random.RandomState(cfg.shuffle_seed)
            positions = rng.permutation(positions)[: cfg.max_rows]
            log.info(
                "FineVision[%s]: subsampled to %d rows (shuffle_seed=%d)",
                cfg.config_name,
                len(positions),
                cfg.shuffle_seed,
            )

        if len(positions) == 0:
            log.warning("FineVision[%s]: no rows remain after filtering", cfg.config_name)
            return positions
        if len(positions) == n:
            return None
        return positions

    @staticmethod
    def _load_table(config: FineVisionDatasetConfig):
        keep_columns = [config.texts_column, config.images_column] + list(_QUALITY_COLUMNS.values())

        if config.dataset_path is not None:
            return load_hf_dataset(
                config.dataset_path,
                config.split,
                keep_columns=keep_columns,
            )

        if config.uses_hub():
            from datasets import load_dataset

            load_kwargs = {}
            if config.cache_dir is not None:
                load_kwargs["cache_dir"] = config.cache_dir
            log.info(
                "FineVision[%s]: loading from hub %s (cache_dir=%s)",
                config.config_name,
                config.hub_repo,
                config.cache_dir or os.environ.get("HF_DATASETS_CACHE", "(default)"),
            )
            ds = load_dataset(
                config.hub_repo,
                name=config.config_name,
                split=config.split,
                **load_kwargs,
            )
            extra = [c for c in ds.column_names if c not in keep_columns]
            if extra:
                ds = ds.remove_columns(extra)
            return ds

        return load_hf_dataset(
            config.resolved_path(),
            config.split,
            keep_columns=keep_columns,
        )

    def __len__(self) -> int:
        return len(self._data) if self._index is None else len(self._index)

    def _row(self, i: int) -> Dict:
        pos = int(i if self._index is None else self._index[i])
        return self._data[pos]

    def _build(self, i: int) -> Dict[str, np.ndarray]:
        cfg = self.config
        row = self._row(i)

        turns: List[Tuple[str, str]] = []
        for turn in row[cfg.texts_column] or []:
            # Any inline <image> marker is stripped: the image is supplied as an explicit
            # token block inside the first user turn instead. Several configs carry no
            # marker at all, which is equivalent here since the block comes from the
            # `images` column.
            user = strip_image_placeholders(turn.get("user"))
            assistant = (turn.get("assistant") or "").strip()
            if not user or not assistant:
                continue
            turns.append((user, assistant))
        if not turns:
            raise ValueError("no usable (user, assistant) turn in row")

        raw_images = row.get(cfg.images_column) or []
        pil_images = [decode_pil_image(im) for im in raw_images if im is not None]

        # mm_olmo DataFormatter puts each message_list row in its own branch (shared image
        # prefix); pass flat turns so encode_sft_example splits into [[(q,a)], ...].
        seq = encode_sft_example(
            self.tokenizer,
            pil_images,
            turns,
            max_crops=cfg.max_crops,
            max_images=cfg.max_images,
            loss_token_weighting=cfg.loss_token_weighting,
            shuffle_rng=example_rng(cfg.seed, i),
        )
        return truncate_example(seq, cfg.max_sequence_length)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """Build the example at ``index``, skipping ahead over unusable rows.

        See :func:`~olmo_core.data.multimodal.sft_common.get_example_with_skip`.
        """
        return get_example_with_skip(self, index, len(self))


# Backwards-compatible alias: the loader used to be VisualWebInstruct-specific.
VisualWebInstructDataset = FineVisionDataset
