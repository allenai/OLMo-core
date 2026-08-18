"""Data-mixture contracts for vision alignment continued pretraining.

The ratios in this module are *effective supervised-loss mass* targets. They are not
dataset-example probabilities. A source that contributes many supervised tokens per example
must therefore be sampled less often than a short-response source with the same target mass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from olmo_core.config import Config
from olmo_core.data.multimodal.mixture_weights import (
    _normalized,
    _validate_positive_mapping,
    expected_loss_mass,
    sampling_weights_from_loss_mass,
)

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
