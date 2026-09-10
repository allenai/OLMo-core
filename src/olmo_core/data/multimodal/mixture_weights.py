"""Mixture sampling weights (port of mm_olmo SubMixture rate math)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "DatasetSource",
    "SubMixture",
    "compute_flat_mixture_weights",
    "restrict_submixtures",
]

log = logging.getLogger(__name__)


@dataclass
class DatasetSource:
    name: str
    sampling_rate: Optional[float] = None
    root_size_factor: Optional[Union[int, float]] = None
    message_weight: Optional[float] = None
    override_p_high_res: Optional[float] = None


@dataclass
class SubMixture:
    name: str
    rate: float
    datasets: Sequence[DatasetSource]


def _dataset_size_factor(source: DatasetSource, dataset_len: int) -> float:
    """mm_olmo root-size score (data_loader.py:264-271), all four branches."""
    if source.root_size_factor == 0:
        return 1.0
    if source.root_size_factor is None:
        return float(np.sqrt(max(dataset_len, 1)))
    if source.root_size_factor < 1:
        return float(np.sqrt(dataset_len * source.root_size_factor))
    return float(np.sqrt(source.root_size_factor))


def restrict_submixtures(
    groups: Sequence[SubMixture],
    dataset_names: Optional[Sequence[str]],
) -> List[SubMixture]:
    """Restrict each group to ``dataset_names``, dropping groups left empty.

    Applied *before* :func:`compute_flat_mixture_weights` so callers only need
    lengths for the sources they actually asked for — a validation tier such as
    ``--mixture=finevision`` therefore never constructs the sources it excludes.

    This also fixes what the weights mean. Restricting afterwards gives each group
    a share of ``group.rate * (surviving size-mass / total size-mass)``, so a tier
    that keeps only a slice of one group silently crushes that group relative to
    groups it keeps whole: ``debug`` (all of ``nlp``, 2 of 33 ``image_academic``)
    came out ~87% ``tulu4``. Restricting first gives each surviving group its
    nominal ``group.rate``. The two agree exactly whenever every group a tier
    touches is kept whole, which covers every tier except ``debug`` and
    ``multi-image``.

    :param dataset_names: allowlist of source names, or ``None`` to keep everything.
    :raises ValueError: If no source survives the filter.
    """
    if dataset_names is None:
        return list(groups)

    allowed = set(dataset_names)
    out: List[SubMixture] = []
    for group in groups:
        kept = [src for src in group.datasets if src.name in allowed]
        if kept:
            out.append(SubMixture(group.name, group.rate, kept))

    if not out:
        raise ValueError(f"No mixture sources matched dataset_names={list(dataset_names)!r}")

    absent = allowed - {src.name for group in out for src in group.datasets}
    if absent:
        # Not fatal: single-image variants legitimately drop the multi-image sources a
        # tier names. Logged so a typo or a renamed source doesn't just vanish.
        log.info(
            "Mixture filter: %d requested source(s) not present in this mixture: %s",
            len(absent),
            ", ".join(sorted(absent)),
        )
    return out


def compute_flat_mixture_weights(
    groups: Sequence[SubMixture],
    dataset_lengths: dict[str, int],
) -> List[Tuple[str, float]]:
    """Return normalized (dataset_name, global_rate) pairs."""
    flat: List[Tuple[str, float]] = []
    for group in groups:
        if group.rate <= 0 or not group.datasets:
            continue
        factors = []
        for src in group.datasets:
            frac = _dataset_size_factor(src, dataset_lengths[src.name])
            if src.sampling_rate is not None:
                frac *= src.sampling_rate
            factors.append(frac)
        total = sum(factors)
        for src, frac in zip(group.datasets, factors):
            flat.append((src.name, group.rate * (frac / total)))
    norm = sum(w for _, w in flat)
    return [(name, w / norm) for name, w in flat]
