"""image-only-v11 mixture registry for Molmo2 SFT.

Extends ``image-only-v10`` with three groups aimed at the two benchmarks v10 is
weakest on, ``charxiv_{descriptive,reasoning}`` and ``mmmu_pro``:

* ``chart_reasoning`` (6%) -- ChartVerse. Its charts are rendered from generated code,
  so it is the only source in the mixture covering 3D, hierarchical, Sankey/chord and
  multi-subplot layouts. Every pre-existing chart source (``chart_qa_weighted``,
  ``dv_qa``, ``figure_qa``, ``plot_qa``, ``cosyn_chart_exp``) is simple bar/line/pie.
* ``figure_captions`` (5%) -- ArxivCap, OmniScience, VisText, Chart2Text. ArxivCap is
  drawn from the same arXiv population CharXiv samples; VisText's L1 captions read
  almost like the ``charxiv_descriptive`` templates.
* ``web_reasoning`` (5%) -- the five FineVision configs already staged under
  ``FINEVISION_ROOT`` plus MMFineReason. ``visualwebinstruct(filtered)`` carries the
  strongest published MMMU-Pro evidence of any candidate source, and its authors'
  pHash/SimHash decontamination reports MMMU, MMMU-Pro, MMStar and CharXiv clean.

The six v10 groups are scaled by 0.84 to keep the total at 1.0. Holding new mass to
16% is deliberate: v10 is a validated recipe measured across 14 benchmarks, and
diluting it harder to chase two of them risks paying for those gains elsewhere.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from olmo_core.data.multimodal.caption_datasets import (
    CAPTION_DATASET_NAMES,
    CaptionDatasetConfig,
)
from olmo_core.data.multimodal.chartverse import (
    CHARTVERSE_DATASET_NAME,
    ChartVerseDatasetConfig,
)
from olmo_core.data.multimodal.finevision import (
    FINEVISION_V11_DATASET_NAMES,
    build_finevision_v11_config,
    finevision_v11_hub_name,
)
from olmo_core.data.multimodal.mixture_weights import (
    DatasetSource,
    SubMixture,
    compute_flat_mixture_weights,
)
from olmo_core.data.multimodal.mixtures.image_only_v9 import (
    filter_submixtures_single_image,
)
from olmo_core.data.multimodal.mixtures.image_only_v10 import (
    IMAGE_ONLY_V10_SUBMIXTURES,
    build_image_only_v10_dataset,
)
from olmo_core.data.multimodal.mmfinereason import MMFineReasonDatasetConfig

__all__ = [
    "IMAGE_ONLY_V11_BASE_SCALE",
    "IMAGE_ONLY_V11_SUBMIXTURES",
    "SINGLE_IMAGE_ONLY_V11_SUBMIXTURES",
    "VALIDATION_MIXTURES_V11",
    "MMFINEREASON_DATASET_NAME",
    "IMAGE_ONLY_V11_NEW_DATASET_NAMES",
    "build_image_only_v11_dataset",
    "build_image_only_v11_datasets",
    "build_image_only_v11_mixture",
    "build_single_image_only_v11_mixture",
    "image_only_v11_dataset_names",
    "image_only_v11_new_dataset_names",
]

MMFINEREASON_DATASET_NAME = "mmfinereason"

CHART_REASONING_V11_RATE = 0.06
FIGURE_CAPTIONS_V11_RATE = 0.05
WEB_REASONING_V11_RATE = 0.05

# 1 - chart_reasoning(0.06) - figure_captions(0.05) - web_reasoning(0.05)
IMAGE_ONLY_V11_BASE_SCALE = 0.84

_CHART_REASONING_SOURCES = [DatasetSource(CHARTVERSE_DATASET_NAME)]

# Within-group shares default to sqrt(len); sampling_rate nudges that. VisText is tiny
# but its L1 captions map almost directly onto the charxiv_descriptive templates, so it
# is boosted; Chart2Text's captions are noticeably templated, so it is damped.
_CAPTION_SAMPLING_RATES: Dict[str, float] = {
    "vistext": 2.0,
    "chart2text": 0.5,
}

_FIGURE_CAPTION_SOURCES = [
    DatasetSource(name, sampling_rate=_CAPTION_SAMPLING_RATES.get(name))
    for name in CAPTION_DATASET_NAMES
]

# MMFineReason is 586k rows, so plain sqrt(len) would hand it ~37% of the group. Cap it
# at sqrt(50k): it is a bonus source, and ~13% of its rows re-annotate
# visualwebinstruct(filtered) images, which are already in this same group.
_WEB_REASONING_SOURCES = [
    DatasetSource(
        "finevision_visualwebinstruct",
        sampling_rate=1.5,
    ),
    DatasetSource("finevision_mavis_math_rule_geo"),
    DatasetSource("finevision_mavis_math_metagen"),
    DatasetSource("finevision_geo170k_align"),
    DatasetSource("finevision_geo170k_qa"),
    DatasetSource(MMFINEREASON_DATASET_NAME, root_size_factor=50_000),
]

IMAGE_ONLY_V11_SUBMIXTURES: List[SubMixture] = [
    SubMixture(
        group.name,
        group.rate * IMAGE_ONLY_V11_BASE_SCALE,
        list(group.datasets),
    )
    for group in IMAGE_ONLY_V10_SUBMIXTURES
] + [
    SubMixture("chart_reasoning", CHART_REASONING_V11_RATE, _CHART_REASONING_SOURCES),
    SubMixture("figure_captions", FIGURE_CAPTIONS_V11_RATE, _FIGURE_CAPTION_SOURCES),
    SubMixture("web_reasoning", WEB_REASONING_V11_RATE, _WEB_REASONING_SOURCES),
]

# Every v11-only source is single-image, so the single-image tier drops exactly the
# same 11 multi-image v9 names as v10 does.
SINGLE_IMAGE_ONLY_V11_SUBMIXTURES: List[SubMixture] = filter_submixtures_single_image(
    IMAGE_ONLY_V11_SUBMIXTURES
)

IMAGE_ONLY_V11_NEW_DATASET_NAMES: Tuple[str, ...] = (
    (CHARTVERSE_DATASET_NAME,)
    + CAPTION_DATASET_NAMES
    + FINEVISION_V11_DATASET_NAMES
    + (MMFINEREASON_DATASET_NAME,)
)

VALIDATION_MIXTURES_V11: Dict[str, Optional[Tuple[str, ...]]] = {
    "image-only-v11": None,
    "single-image-only-v11": None,
    "chartverse": (CHARTVERSE_DATASET_NAME,),
    "figure-captions": CAPTION_DATASET_NAMES,
    "web-reasoning": FINEVISION_V11_DATASET_NAMES + (MMFINEREASON_DATASET_NAME,),
    "v11-new": IMAGE_ONLY_V11_NEW_DATASET_NAMES,
    # Per-source ablation tiers: train on exactly one v11-new source (see the matching
    # block in image_only_v10.py). "chartverse" is already a key above, so the dict
    # comprehension just re-states it identically. These must live in *this* dict, not
    # VALIDATION_MIXTURES: `_build_mixture` dispatches by dict membership, so a v11 name
    # registered in the v9 dict routes to build_image_only_v9_mixture and raises.
    **{name: (name,) for name in IMAGE_ONLY_V11_NEW_DATASET_NAMES},
}


def _source_lookup() -> Dict[str, DatasetSource]:
    out: Dict[str, DatasetSource] = {}
    for group in IMAGE_ONLY_V11_SUBMIXTURES:
        for src in group.datasets:
            out[src.name] = src
    return out


def image_only_v11_dataset_names() -> List[str]:
    """All dataset names in the image-only-v11 mixture."""
    return [src.name for group in IMAGE_ONLY_V11_SUBMIXTURES for src in group.datasets]


def image_only_v11_new_dataset_names() -> List[str]:
    """The v11-only sources (not present in image-only-v10)."""
    return list(IMAGE_ONLY_V11_NEW_DATASET_NAMES)


def build_image_only_v11_dataset(
    name: str,
    tokenizer,
    seed: int = 0,
    *,
    max_sequence_length: Optional[int] = None,
    finevision_cache_dir: Optional[str] = None,
):
    """Build a single image-only-v11 dataset by mixture source name."""
    if name == CHARTVERSE_DATASET_NAME:
        kw: Dict[str, Any] = dict(seed=seed)
        if max_sequence_length is not None:
            kw["max_sequence_length"] = max_sequence_length
        return ChartVerseDatasetConfig(**kw).build(tokenizer)

    if name in CAPTION_DATASET_NAMES:
        kw = dict(subset=name, seed=seed)
        if max_sequence_length is not None:
            kw["max_sequence_length"] = max_sequence_length
        return CaptionDatasetConfig(**kw).build(tokenizer)

    if name in FINEVISION_V11_DATASET_NAMES:
        hub_name = finevision_v11_hub_name(name)
        kw = dict(seed=seed, cache_dir=finevision_cache_dir)
        if max_sequence_length is not None:
            kw["max_sequence_length"] = max_sequence_length
        return build_finevision_v11_config(hub_name, **kw).build(tokenizer)

    if name == MMFINEREASON_DATASET_NAME:
        kw = dict(seed=seed)
        if max_sequence_length is not None:
            kw["max_sequence_length"] = max_sequence_length
        return MMFineReasonDatasetConfig(**kw).build(tokenizer)

    return build_image_only_v10_dataset(
        name,
        tokenizer,
        seed,
        max_sequence_length=max_sequence_length,
        finevision_cache_dir=finevision_cache_dir,
    )


class _LazyDatasetMap:
    """Lazy dataset registry: builds each dataset on first access."""

    def __init__(
        self,
        tokenizer,
        seed: int,
        *,
        max_sequence_length: Optional[int] = None,
        finevision_cache_dir: Optional[str] = None,
    ):
        self._tokenizer = tokenizer
        self._seed = seed
        self._max_sequence_length = max_sequence_length
        self._finevision_cache_dir = finevision_cache_dir
        self._cache: Dict[str, object] = {}
        self._source_map = _source_lookup()

    def keys(self):
        return self._source_map.keys()

    def __contains__(self, name: str) -> bool:
        return name in self._source_map

    def __getitem__(self, name: str):
        if name not in self._cache:
            self._cache[name] = build_image_only_v11_dataset(
                name,
                self._tokenizer,
                self._seed,
                max_sequence_length=self._max_sequence_length,
                finevision_cache_dir=self._finevision_cache_dir,
            )
        return self._cache[name]


def build_image_only_v11_datasets(
    tokenizer,
    seed: int = 0,
    *,
    max_sequence_length: Optional[int] = None,
    finevision_cache_dir: Optional[str] = None,
) -> _LazyDatasetMap:
    """Lazy registry of all image-only-v11 datasets keyed by mixture source name."""
    return _LazyDatasetMap(
        tokenizer,
        seed,
        max_sequence_length=max_sequence_length,
        finevision_cache_dir=finevision_cache_dir,
    )


def build_image_only_v11_mixture(
    tokenizer,
    seed: int = 0,
    *,
    dataset_names: Optional[Sequence[str]] = None,
    max_sequence_length: int = 16384,
    finevision_cache_dir: Optional[str] = None,
    submixtures: Optional[Sequence[SubMixture]] = None,
) -> Tuple[List, List[float], List[str]]:
    """Build weighted datasets for :class:`~olmo_core.data.multimodal.MixtureDataLoader`."""
    groups = list(IMAGE_ONLY_V11_SUBMIXTURES if submixtures is None else submixtures)
    datasets_map = build_image_only_v11_datasets(
        tokenizer,
        seed,
        max_sequence_length=max_sequence_length,
        finevision_cache_dir=finevision_cache_dir,
    )
    # Fast path for single-source ablation tiers. The surviving weight is renormalized to
    # 1.0 regardless of what the group math produces, so the general path below would build
    # *every* dataset in the mixture -- `lengths` calls len() on all of them, and the lazy
    # map constructs each one (minutes of Arrow/parquet index building) -- purely to compute
    # weights it then discards. Deliberately restricted to a single name: for a multi-source
    # filter, weights computed before vs after filtering genuinely differ whenever a group is
    # only partly kept, so the general path stays authoritative there.
    if dataset_names is not None and len(set(dataset_names)) == 1:
        only = next(iter(dataset_names))
        if only not in {src.name for group in groups for src in group.datasets}:
            raise ValueError(f"No mixture sources matched dataset_names={dataset_names!r}")
        return [datasets_map[only]], [1.0], [only]

    needed = {src.name for group in groups for src in group.datasets}
    lengths = {name: len(datasets_map[name]) for name in needed}
    flat = compute_flat_mixture_weights(groups, lengths)

    if dataset_names is not None:
        allowed = set(dataset_names)
        flat = [(name, w) for name, w in flat if name in allowed]
        if not flat:
            raise ValueError(f"No mixture sources matched dataset_names={dataset_names!r}")
        norm = sum(w for _, w in flat)
        flat = [(name, w / norm) for name, w in flat]

    out_names = [name for name, _ in flat]
    out_datasets = [datasets_map[name] for name in out_names]
    out_weights = [w for _, w in flat]
    return out_datasets, out_weights, out_names


def build_single_image_only_v11_mixture(
    tokenizer,
    seed: int = 0,
    *,
    dataset_names: Optional[Sequence[str]] = None,
    max_sequence_length: int = 16384,
    finevision_cache_dir: Optional[str] = None,
) -> Tuple[List, List[float], List[str]]:
    """Build image-only-v11 with multi-image v9 sources removed."""
    return build_image_only_v11_mixture(
        tokenizer,
        seed,
        dataset_names=dataset_names,
        max_sequence_length=max_sequence_length,
        finevision_cache_dir=finevision_cache_dir,
        submixtures=SINGLE_IMAGE_ONLY_V11_SUBMIXTURES,
    )
