"""Utilities for calculating multimodal mixture sampling weights."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "DatasetSource",
    "SubMixture",
    "compute_flat_mixture_weights",
    "expected_loss_mass",
    "sampling_weights_from_loss_mass",
]


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


def _validate_positive_mapping(values: Mapping[str, float], *, name: str) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty")
    invalid = {
        key: value for key, value in values.items() if not math.isfinite(float(value)) or value <= 0
    }
    if invalid:
        raise ValueError(f"{name} values must be positive, got {invalid}")


def _normalized(values: Mapping[str, float]) -> Dict[str, float]:
    total = float(sum(values.values()))
    if total <= 0:
        raise ValueError("Cannot normalize a mapping with non-positive total mass")
    return {key: float(value) / total for key, value in values.items()}


def sampling_weights_from_loss_mass(
    target_loss_mass: Mapping[str, float],
    mean_loss_weight: Mapping[str, float],
) -> Dict[str, float]:
    """Convert desired loss-mass ratios into dataset-example sampling probabilities.

    If source ``i`` has target mass :math:`t_i` and contributes an average supervised loss
    weight :math:`m_i` per sampled example, its unnormalized sampling probability is
    :math:`t_i / m_i`.

    :param target_loss_mass: Desired effective supervised-loss mass by source.
    :param mean_loss_weight: Preflight estimate of mean ``sum(loss_masks)`` by source.

    :returns: Normalized example-sampling probabilities with the same source keys.

    :raises ValueError: If mappings are empty, have different keys, or contain non-positive
        values.
    """
    _validate_positive_mapping(target_loss_mass, name="target_loss_mass")
    _validate_positive_mapping(mean_loss_weight, name="mean_loss_weight")
    if set(target_loss_mass) != set(mean_loss_weight):
        missing = sorted(set(target_loss_mass) - set(mean_loss_weight))
        extra = sorted(set(mean_loss_weight) - set(target_loss_mass))
        raise ValueError(
            "Loss-mass calibration source mismatch: "
            f"missing mean weights for {missing}, unexpected means for {extra}"
        )
    target = _normalized(target_loss_mass)
    return _normalized(
        {source: target[source] / float(mean_loss_weight[source]) for source in target}
    )


def expected_loss_mass(
    sampling_weights: Mapping[str, float],
    mean_loss_weight: Mapping[str, float],
) -> Dict[str, float]:
    """Calculate expected effective-loss ratios for a calibrated sampling distribution.

    :param sampling_weights: Dataset-example sampling probabilities by source.
    :param mean_loss_weight: Mean supervised loss weight per example by source.

    :returns: Normalized expected supervised-loss mass by source.

    :raises ValueError: If mappings are empty, have different keys, or contain non-positive
        values.
    """
    _validate_positive_mapping(sampling_weights, name="sampling_weights")
    _validate_positive_mapping(mean_loss_weight, name="mean_loss_weight")
    if set(sampling_weights) != set(mean_loss_weight):
        raise ValueError("sampling_weights and mean_loss_weight must contain identical sources")
    return _normalized(
        {
            source: float(sampling_weights[source]) * float(mean_loss_weight[source])
            for source in sampling_weights
        }
    )


def _dataset_size_factor(source: DatasetSource, dataset_len: int) -> float:
    """mm_olmo root-size score (data_loader.py:264-271), all four branches."""
    if source.root_size_factor == 0:
        return 1.0
    if source.root_size_factor is None:
        return float(np.sqrt(max(dataset_len, 1)))
    if source.root_size_factor < 1:
        return float(np.sqrt(dataset_len * source.root_size_factor))
    return float(np.sqrt(source.root_size_factor))


def compute_flat_mixture_weights(
    groups: Sequence[SubMixture],
    dataset_lengths: dict[str, int],
) -> List[Tuple[str, float]]:
    """Return normalized ``(dataset_name, global_rate)`` pairs.

    :param groups: Sub-mixtures and their relative sampling rates.
    :param dataset_lengths: Number of examples available from each dataset.
    :returns: Flattened dataset names and normalized global sampling rates.
    """
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
