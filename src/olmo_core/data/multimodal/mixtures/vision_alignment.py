"""Data-mixture contracts for vision alignment continued pretraining.

The ratios in this module are *effective supervised-loss mass* targets. They are not
dataset-example probabilities. A source that contributes many supervised tokens per example
must therefore be sampled less often than a short-response source with the same target mass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

from olmo_core.config import Config

__all__ = [
    "VISION_ALIGNMENT_PHASES",
    "VISION_ALIGNMENT_PHASE_TARGETS",
    "VisionAlignmentMixtureConfig",
    "expected_loss_mass",
    "sampling_weights_from_loss_mass",
]


VISION_ALIGNMENT_PHASES = ("bridge", "perception", "joint")

# These are initial experiment targets, not claims about a universally optimal mixture. They
# intentionally keep task-shaped sources small and reserve native text replay for the phase in
# which the language model is unfrozen.
VISION_ALIGNMENT_PHASE_TARGETS: Dict[str, Dict[str, float]] = {
    "bridge": {
        "pixmo_caption": 0.70,
        "pixmo_transcript": 0.30,
    },
    "perception": {
        "pixmo_caption": 0.45,
        "pixmo_transcript": 0.20,
        "pixmo_points_basic": 0.10,
        "pixmo_points_high_frequency": 0.02,
        "cosyn_point": 0.03,
        "ocr_document": 0.10,
        "scalar_count": 0.05,
        "audited_alignment": 0.05,
    },
    "joint": {
        "native_text_replay": 0.35,
        "pixmo_caption": 0.28,
        "pixmo_transcript": 0.12,
        "pixmo_points_basic": 0.05,
        "pixmo_points_high_frequency": 0.01,
        "cosyn_point": 0.02,
        "ocr_document": 0.08,
        "count_numeric": 0.04,
        "audited_alignment": 0.05,
    },
}


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


@dataclass
class VisionAlignmentMixtureConfig(Config):
    """Phase-specific vision-alignment mixture and its loss-mass calibration.

    ``mean_loss_weight`` must come from a deterministic preprocessing audit over the exact
    serialized sources. The training path intentionally refuses to reinterpret target masses as
    example probabilities when that calibration is absent.
    """

    phase: str = "bridge"
    target_loss_mass: Optional[Dict[str, float]] = None
    mean_loss_weight: Dict[str, float] = field(default_factory=dict)

    def resolved_targets(self) -> Dict[str, float]:
        """Return normalized targets for this phase, applying an explicit override if present."""
        if self.phase not in VISION_ALIGNMENT_PHASES:
            raise ValueError(
                f"Unknown vision-alignment phase {self.phase!r}; "
                f"expected one of {VISION_ALIGNMENT_PHASES}"
            )
        targets = self.target_loss_mass or VISION_ALIGNMENT_PHASE_TARGETS[self.phase]
        _validate_positive_mapping(targets, name="target_loss_mass")
        return _normalized(targets)

    def sampling_weights(self) -> Dict[str, float]:
        """Resolve example-sampling probabilities from the configured loss-mass targets."""
        return sampling_weights_from_loss_mass(self.resolved_targets(), self.mean_loss_weight)
