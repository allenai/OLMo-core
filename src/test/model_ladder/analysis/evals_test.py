"""Tests for scaling law evaluation metrics."""

import numpy as np

from olmo_core.model_ladder.analysis.eval import (
    RolloutSplit,
    ScalingLawRollout,
    evaluate_rollout,
    evaluate_split,
    perplexity_ratio,
    relative_bpb_error,
)
from olmo_core.model_ladder.analysis.scaling_laws import ChinchillaParams


def test_perplexity_ratio_perfect_prediction():
    predicted = np.array([1.0, 2.0, 3.0])
    actual = np.array([1.0, 2.0, 3.0])
    ratios = perplexity_ratio(predicted, actual)
    assert np.allclose(ratios, 1.0)


def test_perplexity_ratio_overestimate():
    predicted = np.array([1.1, 2.1])
    actual = np.array([1.0, 2.0])
    ratios = perplexity_ratio(predicted, actual)
    # Should be > 1 (predicted higher loss)
    assert np.all(ratios > 1.0)
    # Check exact values: 2^(0.1) ≈ 1.0718
    assert np.allclose(ratios, 2**0.1)


def test_perplexity_ratio_underestimate():
    predicted = np.array([0.9, 1.9])
    actual = np.array([1.0, 2.0])
    ratios = perplexity_ratio(predicted, actual)
    # Should be < 1 (predicted lower loss)
    assert np.all(ratios < 1.0)
    # Check exact values: 2^(-0.1) ≈ 0.9330
    assert np.allclose(ratios, 2**-0.1)


def test_perplexity_ratio_symmetry():
    base = 1.0
    error = 0.1
    over_ratio = perplexity_ratio(np.array([base + error]), np.array([base]))[0]
    under_ratio = perplexity_ratio(np.array([base - error]), np.array([base]))[0]
    # Should be reciprocals
    assert np.allclose(over_ratio * under_ratio, 1.0)


def test_relative_bpb_error_scale_dependence():
    """Test that relative BPB error weights low-loss predictions more heavily."""
    # Same absolute error at different loss levels
    error = 0.01

    # At loss = 1.0: relative error = 1%
    rel_err_low = relative_bpb_error(np.array([1.0 + error]), np.array([1.0]))[0]
    assert np.isclose(rel_err_low, 0.01)

    # At loss = 2.0: relative error = 0.5%
    rel_err_high = relative_bpb_error(np.array([2.0 + error]), np.array([2.0]))[0]
    assert np.isclose(rel_err_high, 0.005)

    # Low-loss error should be weighted more heavily
    assert rel_err_low > rel_err_high


def test_evaluate_split_perfect_predictions():
    params = ChinchillaParams(E=1.0, A=100.0, alpha=0.5, B=200.0, beta=0.3)

    test_N = np.array([200e6, 400e6])
    test_D = np.array([2e9, 4e9])
    test_loss = params.predict_loss(test_N, test_D)

    split = RolloutSplit(
        cutoff_variable="N",
        cutoff_value=100e6,
        train_mask=np.array([True, False, False]),
        test_mask=np.array([False, True, True]),
        model=params,
        train_N=np.array([100e6]),
        train_D=np.array([1e9]),
        train_loss=np.array([params.predict_loss(100e6, 1e9)]),
        test_N=test_N,
        test_D=test_D,
        test_loss=test_loss,
    )

    eval_result = evaluate_split(split)

    assert eval_result.cutoff_value == 100e6
    assert eval_result.n_test_points == 2
    assert np.allclose(eval_result.perplexity_ratios, 1.0)
    assert np.allclose(eval_result.relative_errors, 0.0)
    assert np.isclose(eval_result.mean_ppl_error, 0.0)
    assert np.isclose(eval_result.weighted_mean_ppl_error, 0.0)
    assert np.isclose(eval_result.mean_relative_error, 0.0)
    assert np.isclose(eval_result.mean_signed_error, 0.0)


def test_evaluate_split_with_error():
    params = ChinchillaParams(E=1.0, A=100.0, alpha=0.5, B=200.0, beta=0.3)

    test_N = np.array([200e6])
    test_D = np.array([2e9])
    actual_loss = params.predict_loss(test_N, test_D)

    # Create a mock model that overpredicts by 0.01 BPB
    class OverpredictingModel:
        def predict_loss(self, N, D):
            return params.predict_loss(N, D) + 0.01

    split = RolloutSplit(
        cutoff_variable="N",
        cutoff_value=100e6,
        train_mask=np.array([True]),
        test_mask=np.array([False]),
        model=OverpredictingModel(),
        train_N=np.array([100e6]),
        train_D=np.array([1e9]),
        train_loss=np.array([params.predict_loss(100e6, 1e9)]),
        test_N=test_N,
        test_D=test_D,
        test_loss=actual_loss,
    )

    eval_result = evaluate_split(split)

    assert eval_result.n_test_points == 1
    # Perplexity ratio should be > 1 (overprediction)
    assert eval_result.perplexity_ratios[0] > 1.0
    # Errors should be positive
    assert eval_result.mean_ppl_error > 0.0
    assert eval_result.mean_relative_error > 0.0
    # Signed error should be positive (overprediction)
    assert eval_result.mean_signed_error > 0.0
    assert np.isclose(eval_result.bpb_errors[0], 0.01)


def test_evaluate_split_cutoff_by_D():
    params = ChinchillaParams(E=1.0, A=100.0, alpha=0.5, B=200.0, beta=0.3)

    test_N = np.array([200e6])
    test_D = np.array([2e9])
    test_loss = params.predict_loss(test_N, test_D)

    split = RolloutSplit(
        cutoff_variable="D",
        cutoff_value=1e9,
        train_mask=np.array([True]),
        test_mask=np.array([False]),
        model=params,
        train_N=np.array([100e6]),
        train_D=np.array([1e9]),
        train_loss=np.array([params.predict_loss(100e6, 1e9)]),
        test_N=test_N,
        test_D=test_D,
        test_loss=test_loss,
    )

    eval_result = evaluate_split(split)
    # Should use test_D for distance calculation
    assert eval_result.distances[0] > 0  # Should be positive since 2e9 > 1e9


def test_evaluate_rollout_splits():
    params = ChinchillaParams(E=1.0, A=100.0, alpha=0.5, B=200.0, beta=0.3)

    splits = []
    for cutoff_N in [100e6, 200e6]:
        test_N = np.array([cutoff_N * 2])
        test_D = np.array([1e9 * 2])
        test_loss = params.predict_loss(test_N, test_D)

        split = RolloutSplit(
            cutoff_variable="N",
            cutoff_value=cutoff_N,
            train_mask=np.array([True]),
            test_mask=np.array([False]),
            model=params,
            train_N=np.array([cutoff_N]),
            train_D=np.array([1e9]),
            train_loss=np.array([params.predict_loss(cutoff_N, 1e9)]),
            test_N=test_N,
            test_D=test_D,
            test_loss=test_loss,
        )
        splits.append(split)

    # Create a ScalingLawRollout object with the splits
    # Construct N, D, and loss arrays from the splits
    N = np.array([100e6, 200e6, 400e6])
    D = np.array([1e9, 2e9, 2e9])
    loss = params.predict_loss(N, D)
    rollout = ScalingLawRollout(N=N, D=D, loss=loss, splits=splits)

    eval_result = evaluate_rollout(rollout)

    assert len(eval_result.split_evaluations) == 2
    # Both error metrics should be non-negative
    assert eval_result.overall_weighted_mean_ppl_error >= 0.0
    assert eval_result.overall_mean_ppl_error >= 0.0
    assert eval_result.overall_weighted_mean_relative_error >= 0.0
    assert eval_result.overall_mean_relative_error >= 0.0
