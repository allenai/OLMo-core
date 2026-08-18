"""Tests for vision-alignment loss-mass mixture calibration."""

from __future__ import annotations

import pytest

from olmo_core.data.multimodal.mixture_weights import (
    expected_loss_mass as generic_expected_loss_mass,
)
from olmo_core.data.multimodal.mixture_weights import (
    sampling_weights_from_loss_mass as generic_sampling_weights_from_loss_mass,
)
from olmo_core.data.multimodal.mixtures.vision_alignment import (
    VISION_ALIGNMENT_PHASE_TARGETS,
    VisionAlignmentMixtureConfig,
    expected_loss_mass,
    sampling_weights_from_loss_mass,
)


def test_phase_targets_are_normalized_and_keep_text_replay_joint_only():
    assert set(VISION_ALIGNMENT_PHASE_TARGETS) == {"bridge", "perception", "joint"}
    for targets in VISION_ALIGNMENT_PHASE_TARGETS.values():
        assert sum(targets.values()) == pytest.approx(1.0)
        assert all(value > 0 for value in targets.values())

    assert "native_text_replay" not in VISION_ALIGNMENT_PHASE_TARGETS["bridge"]
    assert "native_text_replay" not in VISION_ALIGNMENT_PHASE_TARGETS["perception"]
    assert VISION_ALIGNMENT_PHASE_TARGETS["joint"]["native_text_replay"] == 0.35


def test_loss_mass_helpers_remain_backward_compatible():
    assert expected_loss_mass is generic_expected_loss_mass
    assert sampling_weights_from_loss_mass is generic_sampling_weights_from_loss_mass


def test_mixture_config_supports_explicit_ablation_targets():
    config = VisionAlignmentMixtureConfig(
        phase="joint",
        target_loss_mass={"native_text_replay": 0.45, "pixmo_caption": 0.55},
        mean_loss_weight={"native_text_replay": 100.0, "pixmo_caption": 25.0},
    )

    assert config.resolved_targets() == pytest.approx(
        {"native_text_replay": 0.45, "pixmo_caption": 0.55}
    )
    assert expected_loss_mass(config.sampling_weights(), config.mean_loss_weight) == pytest.approx(
        config.resolved_targets()
    )


def test_mixture_config_rejects_unknown_phase():
    with pytest.raises(ValueError, match="Unknown vision-alignment phase"):
        VisionAlignmentMixtureConfig(phase="sft").resolved_targets()
