"""Tests for generic loss-mass mixture calibration."""

from __future__ import annotations

import pytest

from olmo_core.data.multimodal.mixture_weights import (
    expected_loss_mass,
    sampling_weights_from_loss_mass,
)


def test_sampling_weights_recover_requested_effective_loss_mass():
    targets = {"long_caption": 4.0, "short_count": 2.0, "native_text": 4.0}
    means = {"long_caption": 100.0, "short_count": 2.0, "native_text": 512.0}

    sampling = sampling_weights_from_loss_mass(targets, means)

    assert list(sampling) == list(targets)
    assert sum(sampling.values()) == pytest.approx(1.0)
    assert sampling["short_count"] > sampling["long_caption"] > sampling["native_text"]
    assert expected_loss_mass(sampling, means) == pytest.approx(
        {"long_caption": 0.4, "short_count": 0.2, "native_text": 0.4}
    )


def test_sampling_weights_reject_incomplete_or_invalid_calibration():
    with pytest.raises(ValueError, match="source mismatch"):
        sampling_weights_from_loss_mass({"caption": 1.0}, {"text": 1.0})
    with pytest.raises(ValueError, match="must be positive"):
        sampling_weights_from_loss_mass({"caption": 1.0}, {"caption": 0.0})
    with pytest.raises(ValueError, match="must be positive"):
        sampling_weights_from_loss_mass({"caption": float("nan")}, {"caption": 1.0})


def test_expected_loss_mass_rejects_incomplete_calibration():
    with pytest.raises(ValueError, match="must contain identical sources"):
        expected_loss_mass({"caption": 1.0}, {"text": 1.0})
