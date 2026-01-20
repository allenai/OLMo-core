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
from olmo_core.model_ladder.analysis.scaling_laws import (
    ChinchillaParametricFit,
    ChinchillaParams,
)


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


def _create_test_rollout():
    """Helper to create a rollout with two splits for testing."""
    params = ChinchillaParams(E=1.0, A=100.0, alpha=0.5, B=200.0, beta=0.3)

    splits = []
    # First split: 1 training point, 1 test point
    # Second split: 3 training points, 1 test point
    train_configs = [
        (np.array([100e6]), np.array([1e9])),  # 1 training point
        (np.array([100e6, 150e6, 200e6]), np.array([1e9, 1.5e9, 2e9])),  # 3 training points
    ]
    for i, (train_N, train_D) in enumerate(train_configs):
        cutoff_N = train_N[-1]
        test_N = np.array([cutoff_N * 2])
        test_D = np.array([1e9 * 2])
        test_loss = params.predict_loss(test_N, test_D)

        split = RolloutSplit(
            cutoff_variable="N",
            cutoff_value=cutoff_N,
            train_mask=np.array([True] * len(train_N) + [False]),
            test_mask=np.array([False] * len(train_N) + [True]),
            model=params,
            train_N=train_N,
            train_D=train_D,
            train_loss=params.predict_loss(train_N, train_D),
            test_N=test_N,
            test_D=test_D,
            test_loss=test_loss,
        )
        splits.append(split)

    N = np.array([100e6, 150e6, 200e6, 400e6])
    D = np.array([1e9, 1.5e9, 2e9, 2e9])
    loss = params.predict_loss(N, D)
    return ScalingLawRollout(N=N, D=D, loss=loss, splits=splits)


def test_evaluate_rollout_splits():
    rollout = _create_test_rollout()

    eval_result = evaluate_rollout(rollout)

    assert len(eval_result.split_evaluations) == 2
    # Both error metrics should be non-negative
    assert eval_result.overall_weighted_mean_ppl_error >= 0.0
    assert eval_result.overall_mean_ppl_error >= 0.0
    assert eval_result.overall_weighted_mean_relative_error >= 0.0
    assert eval_result.overall_mean_relative_error >= 0.0


def test_evaluate_rollout_split_weights_fn_custom():
    rollout = _create_test_rollout()

    # Weight by total training compute
    eval_by_compute = evaluate_rollout(
        rollout, split_weights_fn=lambda s: np.sum(6.0 * s.train_N * s.train_D)
    )
    assert len(eval_by_compute.split_evaluations) == 2

    # Weight by cutoff value
    eval_by_cutoff = evaluate_rollout(rollout, split_weights_fn=lambda s: s.cutoff_value)
    assert len(eval_by_cutoff.split_evaluations) == 2

    # Constant weight (should be same as uniform)
    eval_const = evaluate_rollout(rollout, split_weights_fn=lambda s: 42.0)
    eval_uniform = evaluate_rollout(rollout)
    assert np.isclose(eval_const.overall_mean_ppl_error, eval_uniform.overall_mean_ppl_error)


def test_scaling_law_rollout_fit():
    """Test ScalingLawRollout.fit creates rollout splits."""
    # Generate synthetic data from known parameters
    true_params = ChinchillaParams(E=1.0, A=100.0, alpha=0.5, B=200.0, beta=0.3)

    # Create data with multiple unique N values for rollout splits
    # Need enough points so each split has >= 5 training points (required by ChinchillaParametricFit)
    N = np.array([1e6, 1e6, 1e6, 2e6, 2e6, 2e6, 5e6, 5e6, 5e6, 1e7, 1e7, 1e7])
    D = np.array([1e9, 2e9, 4e9, 1e9, 2e9, 4e9, 1e9, 2e9, 4e9, 1e9, 2e9, 4e9])
    loss = true_params.predict_loss(N, D)

    # Fit rollout with small number of slices for speed
    rollout = ScalingLawRollout.fit(
        N=N,
        D=D,
        loss=loss,
        fit_fn=ChinchillaParametricFit.fit,
        split_by="N",
        min_points_train=6,
        min_groups_train=2,
        num_slices=2,
    )

    # Verify rollout structure
    assert isinstance(rollout, ScalingLawRollout)
    assert (
        len(np.asarray(rollout.N))
        == len(np.asarray(rollout.D))
        == len(np.asarray(rollout.loss))
        == len(N)
    )
    assert len(rollout.splits) > 0

    # Verify each split has the expected structure
    for split in rollout.splits:
        assert split.cutoff_variable == "N"
        assert split.cutoff_value > 0
        assert hasattr(split.model, "predict_loss")
        assert len(split.train_N) == len(split.train_D) == len(split.train_loss)
        assert len(split.test_N) == len(split.test_D) == len(split.test_loss)
        assert split.train_mask.sum() >= 2  # min_points_train
        assert split.test_mask.sum() > 0  # Must have test points
