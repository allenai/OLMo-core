"""Tests for vision-alignment loss-mass mixture calibration."""

from __future__ import annotations

import pytest

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


def test_sampling_weights_recover_requested_effective_loss_mass():
    targets = {"long_caption": 0.4, "short_count": 0.2, "native_text": 0.4}
    means = {"long_caption": 100.0, "short_count": 2.0, "native_text": 512.0}

    sampling = sampling_weights_from_loss_mass(targets, means)

    assert sum(sampling.values()) == pytest.approx(1.0)
    assert sampling["short_count"] > sampling["long_caption"] > sampling["native_text"]
    assert expected_loss_mass(sampling, means) == pytest.approx(targets)


def test_sampling_weights_reject_incomplete_or_invalid_calibration():
    with pytest.raises(ValueError, match="source mismatch"):
        sampling_weights_from_loss_mass({"caption": 1.0}, {"text": 1.0})
    with pytest.raises(ValueError, match="must be positive"):
        sampling_weights_from_loss_mass({"caption": 1.0}, {"caption": 0.0})
    with pytest.raises(ValueError, match="must be positive"):
        sampling_weights_from_loss_mass({"caption": float("nan")}, {"caption": 1.0})


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
