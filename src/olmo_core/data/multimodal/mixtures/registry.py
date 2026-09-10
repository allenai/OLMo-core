"""Shared machinery for mixture registries (image-only-v9, image-only-v10, ...).

A mixture registry is a list of :class:`~olmo_core.data.multimodal.mixture_weights.SubMixture`
groups plus a function that builds one dataset by source name. Everything else — the
name→source lookup, lazily building each dataset once, restricting to a validation tier,
and turning group rates into flat per-source sampling weights — is identical across
mixtures and lives here.

This exists because ``image_only_v9`` and ``image_only_v10`` previously carried their own
copies of that plumbing, which had already drifted: v10's lazy map cached its source
lookup while v9 rebuilt it on every ``keys()`` / ``__contains__`` call. Sharing one
implementation means a fix lands once.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

from olmo_core.data.multimodal.mixture_weights import (
    DatasetSource,
    SubMixture,
    compute_flat_mixture_weights,
    restrict_submixtures,
)

__all__ = [
    "LazyDatasetMap",
    "build_mixture",
    "mixture_dataset_names",
    "source_lookup",
]


def source_lookup(submixtures: Sequence[SubMixture]) -> Dict[str, DatasetSource]:
    """Map source name → :class:`DatasetSource` across every group.

    Later groups win on duplicate names, matching the previous per-module behaviour.
    """
    out: Dict[str, DatasetSource] = {}
    for group in submixtures:
        for src in group.datasets:
            out[src.name] = src
    return out


def mixture_dataset_names(submixtures: Sequence[SubMixture]) -> List[str]:
    """Every source name in the mixture, in group order (duplicates preserved)."""
    return [src.name for group in submixtures for src in group.datasets]


class LazyDatasetMap:
    """Dataset registry that builds each dataset on first access, once.

    The key space is the *full* registry even when a run only draws a subset, so
    ``keys()`` / ``in`` still describe the whole mixture. Only ``__getitem__`` does work,
    so an unused source is never constructed.
    """

    def __init__(self, submixtures: Sequence[SubMixture], build_dataset: Callable[[str], object]):
        self._build_dataset = build_dataset
        # Cached rather than recomputed per call: `keys()` / `__contains__` are hit
        # repeatedly during mixture construction.
        self._source_map = source_lookup(submixtures)
        self._cache: Dict[str, object] = {}

    def keys(self):
        return self._source_map.keys()

    def __contains__(self, name: str) -> bool:
        return name in self._source_map

    def __len__(self) -> int:
        return len(self._source_map)

    def __getitem__(self, name: str):
        if name not in self._cache:
            self._cache[name] = self._build_dataset(name)
        return self._cache[name]


def build_mixture(
    registry_submixtures: Sequence[SubMixture],
    build_dataset: Callable[[str], object],
    *,
    submixtures: Optional[Sequence[SubMixture]] = None,
    dataset_names: Optional[Sequence[str]] = None,
) -> Tuple[List, List[float], List[str]]:
    """Build weighted datasets for :class:`~olmo_core.data.multimodal.MixtureDataLoader`.

    :param registry_submixtures: The full mixture, defining the lazy map's key space.
    :param build_dataset: Builds one dataset given a source name.
    :param submixtures: Groups to weight, if not the full registry (e.g. the
        single-image-only variant). Rates are taken from these groups.
    :param dataset_names: Restrict to these sources. Applied to the submixtures *before*
        anything is built, so excluded sources are never constructed and each surviving
        group keeps its nominal rate — see :func:`restrict_submixtures` for why the
        ordering matters to the resulting weights.

    :returns: ``(datasets, weights, names)`` in one consistent order.
    """
    groups = restrict_submixtures(
        registry_submixtures if submixtures is None else submixtures,
        dataset_names,
    )
    datasets_map = LazyDatasetMap(registry_submixtures, build_dataset)

    needed = {src.name for group in groups for src in group.datasets}
    lengths = {name: len(datasets_map[name]) for name in needed}  # type: ignore[arg-type]
    flat = compute_flat_mixture_weights(groups, lengths)

    out_names = [name for name, _ in flat]
    out_datasets = [datasets_map[name] for name in out_names]
    out_weights = [weight for _, weight in flat]
    return out_datasets, out_weights, out_names
