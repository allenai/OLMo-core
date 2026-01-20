from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numpy as np
from numpy.typing import ArrayLike

from olmo_core.model_ladder.analysis.scaling_laws import ScalingLawModel

# =============================================================================
# Metric Utilities
# =============================================================================


def perplexity_ratio(predicted_bpb: np.ndarray, actual_bpb: np.ndarray) -> np.ndarray:
    """
    Compute the perplexity ratio between predicted and actual loss.

    This metric is scale-invariant in terms of relative probability error:
    a 0.01 BPB error corresponds to ~0.7% perplexity change regardless of
    the base loss level.

    Mathematical relationship::

        perplexity = 2^BPB
        ratio = 2^predicted / 2^actual = 2^(predicted - actual)

    Interpretation:

    - ratio > 1: Overestimate (predicted higher loss than actual)
    - ratio < 1: Underestimate (predicted lower loss than actual)

    :param predicted_bpb: Predicted loss in bits-per-byte.
    :param actual_bpb: Actual loss in bits-per-byte.
    :returns: Perplexity ratio for each point.
    """
    return np.power(2.0, predicted_bpb - actual_bpb)


def relative_bpb_error(predicted_bpb: np.ndarray, actual_bpb: np.ndarray) -> np.ndarray:
    """
    Compute the relative BPB error between predicted and actual loss.

    This metric weights errors at low loss more heavily than errors at high loss.
    A 0.01 BPB error at loss=1.0 is a 1% relative error, while at loss=2.0 it's
    only 0.5%. This reflects the intuition that prediction accuracy matters more
    for better-performing (lower loss) models.

    Use this metric when:
    - You care more about accurate predictions for large/good models
    - You want errors measured relative to the log-probability scale
    - You're comparing scaling laws across different loss regimes

    :param predicted_bpb: Predicted loss in bits-per-byte.
    :param actual_bpb: Actual loss in bits-per-byte.
    :returns: Relative error |predicted - actual| / actual for each point.
    """
    return np.abs(predicted_bpb - actual_bpb) / actual_bpb


def rollout_distance(
    test_values: np.ndarray,
    cutoff_value: float,
    scale: Literal["log", "linear", "relative"] = "log",
) -> np.ndarray:
    """
    Compute the rollout distance from the training cutoff to each test point.

    For Chinchilla-style scaling laws, "log" scale is recommended because
    the scaling law is a power law that's linear in log-space.

    :param test_values: Test point values (N or D depending on split_by).
    :param cutoff_value: The cutoff value used for training.
    :param scale: How to measure distance:
        - "log": log(test/cutoff), natural for power-law relationships (recommended)
        - "linear": test - cutoff, raw difference
        - "relative": (test - cutoff) / cutoff, relative difference
    :returns: Distance for each test point (always positive for valid rollouts).
    """
    if scale == "log":
        return np.log(test_values / cutoff_value)
    elif scale == "linear":
        return test_values - cutoff_value
    elif scale == "relative":
        return (test_values - cutoff_value) / cutoff_value
    else:
        raise ValueError(f"Unknown scale: {scale}")


def distance_weights(
    distances: np.ndarray, decay: Literal["inverse", "exp"] = "inverse", decay_scale: float = 1.0
) -> np.ndarray:
    """
    Compute weights that decay with distance.

    Closer points (smaller distance) get higher weights, reflecting that
    scaling laws should be most accurate for nearby extrapolations.

    :param distances: Rollout distances (should be positive).
    :param decay: Decay function:
        - "inverse": 1 / (1 + distance/scale), moderate decay (recommended)
        - "exp": exp(-distance/scale), aggressive decay
    :param decay_scale: Scale parameter controlling decay rate.
    :returns: Normalized weights summing to 1.
    """
    d = distances / decay_scale

    if decay == "inverse":
        w = 1.0 / (1.0 + d)
    elif decay == "exp":
        w = np.exp(-d)
    else:
        raise ValueError(f"Unknown decay function: {decay}")

    return w / w.sum()


# =============================================================================
# Rollout / Cross-Validation Utilities
# =============================================================================


@dataclass
class RolloutSplit:
    """
    A single split in a rollout evaluation.

    This is a pure data container holding the train/test split information
    and the fitted scaling law model.
    """

    cutoff_variable: Literal["N", "D"]
    """Which variable was used for splitting ("N" or "D")."""

    cutoff_value: float
    """The cutoff value; training data has values <= this."""

    train_mask: np.ndarray
    """Boolean mask for training points."""

    test_mask: np.ndarray
    """Boolean mask for test points."""

    model: ScalingLawModel
    """The fitted scaling law model."""

    train_N: np.ndarray
    """Model sizes (parameters) for training points."""

    train_D: np.ndarray
    """Data sizes (tokens) for training points."""

    train_loss: np.ndarray
    """Measured loss for training points."""

    test_N: np.ndarray
    """Model sizes (parameters) for test points."""

    test_D: np.ndarray
    """Data sizes (tokens) for test points."""

    test_loss: np.ndarray
    """Measured loss for test points."""

    @property
    def train_predictions(self) -> np.ndarray:
        """Predictions for the train points."""
        return self.model.predict_loss(self.train_N, self.train_D)

    @property
    def test_predictions(self) -> np.ndarray:
        """Predictions for the test points."""
        return self.model.predict_loss(self.test_N, self.test_D)

    @property
    def residuals(self) -> np.ndarray:
        """Log-space residuals (log(actual) - log(predicted)). Scale-invariant."""
        return np.log(self.test_loss) - np.log(self.test_predictions)


@dataclass
class ScalingLawRollout:
    """
    Container for rollout cross-validation results.

    This class performs expanding window cross-validation on scaling law fits:
    1. Sort data by N (or D).
    2. Train on first k groups, test on remaining.
    3. Train on first k+1 groups, test on remaining.
    ...

    Example::

        from olmo_core.model_ladder.analysis import (
            ChinchillaParametricBootstrappedFit,
            ScalingLawRollout,
            evaluate_rollout_splits,
        )

        # Fit scaling laws
        rollout = ScalingLawRollout.fit(
            N=N, D=D, loss=loss, fit_fn=ChinchillaParametricBootstrappedFit.fit
        )

        # Evaluate predictions
        evaluation = evaluate_rollout_splits(rollout.splits)
        print(f"Relative error: {evaluation.overall_mean_relative_error:.2f}%")
        print(f"Perplexity error: {evaluation.overall_mean_ppl_error:.2f}%")
    """

    N: ArrayLike
    """Model sizes (parameters)."""

    D: ArrayLike
    """Data sizes (tokens)."""

    loss: ArrayLike
    """Measured loss values."""

    splits: list[RolloutSplit]
    """Rollout splits computed during fitting."""

    weights: Optional[ArrayLike] = None
    """Optional weights for each data point."""

    @classmethod
    def fit(
        cls,
        N: ArrayLike,
        D: ArrayLike,
        loss: ArrayLike,
        fit_fn: Callable[..., ScalingLawModel],
        weights: Optional[ArrayLike] = None,
        split_by: Literal["N", "D"] = "N",
        min_points_train: int = 5,
        min_groups_train: int = 3,
        **fit_kwargs,
    ) -> "ScalingLawRollout":
        """
        Fit scaling laws using rollout cross-validation.

        :param N: Model sizes (parameters).
        :param D: Data sizes (tokens).
        :param loss: Measured loss values.
        :param fit_fn: Callable that fits a scaling law model, e.g. ``ChinchillaParametricFit.fit``.
            Must accept ``N``, ``D``, ``loss`` keyword arguments and return a :class:`ScalingLawModel`.
        :param weights: Optional weights for each data point.
        :param split_by: Variable to split by ("N" or "D").
        :param min_points_train: Minimum number of training points required.
        :param min_groups_train: Minimum number of unique groups in training set.
        :param **fit_kwargs: Additional arguments passed to ``fit_fn``.
        :returns: ScalingLawRollout instance with computed splits.
        """
        N_arr = np.asarray(N)
        D_arr = np.asarray(D)
        loss_arr = np.asarray(loss)
        weights_arr = np.asarray(weights) if weights is not None else None

        if len(N_arr) != len(D_arr) != len(loss_arr):
            raise ValueError(
                f"Input arrays must have same length. Got N={len(N_arr)}, D={len(D_arr)}, loss={len(loss_arr)}"
            )
        if weights_arr is not None and len(weights_arr) != len(N_arr):
            raise ValueError(
                f"Weights must have same length as data. Got weights={len(weights_arr)}, N={len(N_arr)}"
            )

        split_var = N_arr if split_by == "N" else D_arr
        unique_vals = np.unique(split_var)
        unique_vals.sort()

        splits: list[RolloutSplit] = []

        for i in range(min_groups_train, len(unique_vals)):
            cutoff = unique_vals[i - 1]

            train_mask = split_var <= cutoff
            test_mask = split_var > cutoff

            if train_mask.sum() < min_points_train:
                continue

            if test_mask.sum() == 0:
                break

            current_kwargs = fit_kwargs.copy()
            if weights_arr is not None:
                current_kwargs["weights"] = weights_arr[train_mask]

            model = fit_fn(
                N=N_arr[train_mask],
                D=D_arr[train_mask],
                loss=loss_arr[train_mask],
                **current_kwargs,
            )

            split = RolloutSplit(
                cutoff_variable=split_by,
                cutoff_value=float(cutoff),
                train_mask=train_mask,
                test_mask=test_mask,
                model=model,
                train_N=N_arr[train_mask],
                train_D=D_arr[train_mask],
                train_loss=loss_arr[train_mask],
                test_N=N_arr[test_mask],
                test_D=D_arr[test_mask],
                test_loss=loss_arr[test_mask],
            )
            splits.append(split)

        return cls(N=N, D=D, loss=loss, splits=splits, weights=weights)


# =============================================================================
# Evaluation Functions and Result Containers
# =============================================================================


@dataclass
class SplitEvaluation:
    """
    Evaluation results for a single rollout split.

    Contains two complementary error metrics:

    - **Perplexity error**: Scale-invariant; same absolute BPB error gives same
      percentage error regardless of loss level. Use when comparing across
      very different loss regimes.

    - **Relative BPB error**: Weights low-loss predictions more heavily.
      A 0.01 BPB error is 1% at loss=1.0 but only 0.5% at loss=2.0. Use when
      accuracy on large model predictions matters most.
    """

    cutoff_value: float
    """The training cutoff value."""

    n_test_points: int
    """Number of test points."""

    # Per-point error metrics
    bpb_errors: np.ndarray
    """Signed BPB error (predicted - actual) for each test point."""

    perplexity_ratios: np.ndarray
    """Perplexity ratio 2^(predicted - actual) for each test point."""

    relative_errors: np.ndarray
    """Relative BPB error |predicted - actual| / actual for each test point."""

    # Distance weighting
    distances: np.ndarray
    """Rollout distance for each test point."""

    weights: np.ndarray
    """Distance-based weight for each test point (sums to 1)."""

    # Perplexity-based aggregates (scale-invariant)
    mean_ppl_error: float
    """Mean |perplexity_ratio - 1| * 100, in percent."""

    weighted_mean_ppl_error: float
    """Distance-weighted mean |perplexity_ratio - 1| * 100, in percent."""

    # Relative BPB aggregates (weights low-loss more)
    mean_relative_error: float
    """Mean relative BPB error * 100, in percent."""

    weighted_mean_relative_error: float
    """Distance-weighted mean relative BPB error * 100, in percent."""

    # Bias indicators
    mean_signed_error: float
    """Mean signed BPB error (positive = overestimate)."""

    weighted_mean_signed_error: float
    """Distance-weighted mean signed BPB error."""


@dataclass
class RolloutEvaluation:
    """
    Aggregated evaluation results across all rollout splits.

    Provides both perplexity-based and relative BPB error metrics.
    """

    split_evaluations: list[SplitEvaluation]
    """Per-split evaluation results."""

    # Perplexity-based aggregates
    overall_mean_ppl_error: float
    """Overall mean |perplexity_ratio - 1| * 100, in percent."""

    overall_weighted_mean_ppl_error: float
    """Overall distance-weighted mean perplexity error, in percent."""

    # Relative BPB aggregates
    overall_mean_relative_error: float
    """Overall mean relative BPB error * 100, in percent."""

    overall_weighted_mean_relative_error: float
    """Overall distance-weighted mean relative BPB error, in percent."""

    # Bias indicators
    overall_mean_signed_error: float
    """Overall mean signed BPB error."""

    overall_weighted_mean_signed_error: float
    """Overall distance-weighted mean signed BPB error."""


def evaluate_split(
    split: RolloutSplit,
    weight_decay: Literal["inverse", "exp"] = "inverse",
    weight_decay_scale: float = 1.0,
) -> SplitEvaluation:
    """
    Evaluate a single rollout split's prediction quality.

    Computes both perplexity-based and relative BPB error metrics:

    - **Perplexity error**: |2^(predicted - actual) - 1|, scale-invariant
    - **Relative BPB error**: |predicted - actual| / actual, weights low-loss more

    Also computes distance-based weights to emphasize nearby extrapolations.

    :param split: The RolloutSplit to evaluate.
    :param weight_decay: Distance decay function ("inverse", "exp").
    :param weight_decay_scale: Scale parameter for decay.
    :returns: SplitEvaluation with per-point and aggregate metrics.
    """
    test_loss = np.atleast_1d(split.test_loss)
    test_predictions = np.atleast_1d(split.test_predictions)

    if split.cutoff_variable == "N":
        test_values = np.atleast_1d(split.test_N)
    else:
        test_values = np.atleast_1d(split.test_D)

    # Compute per-point metrics
    bpb_errs = test_predictions - test_loss
    ppl_ratios = perplexity_ratio(test_predictions, test_loss)
    rel_errs = relative_bpb_error(test_predictions, test_loss)

    # Log-space distance
    distances = np.log(test_values / split.cutoff_value)
    # Weight the distances by the decay function to prioritize nearby predictions.
    weights = distance_weights(distances, decay=weight_decay, decay_scale=weight_decay_scale)

    # Perplexity-based aggregates
    abs_ppl_errors = np.abs(ppl_ratios - 1) * 100
    mean_ppl_error = float(np.mean(abs_ppl_errors))
    weighted_mean_ppl_error = float(np.sum(weights * abs_ppl_errors))

    # Relative BPB aggregates
    rel_errs_pct = rel_errs * 100
    mean_relative_error = float(np.mean(rel_errs_pct))
    weighted_mean_relative_error = float(np.sum(weights * rel_errs_pct))

    # Bias indicators
    mean_signed_error = float(np.mean(bpb_errs))
    weighted_mean_signed_error = float(np.sum(weights * bpb_errs))

    return SplitEvaluation(
        cutoff_value=split.cutoff_value,
        n_test_points=len(test_loss),
        bpb_errors=bpb_errs,
        perplexity_ratios=ppl_ratios,
        relative_errors=rel_errs,
        distances=distances,
        weights=weights,
        mean_ppl_error=mean_ppl_error,
        weighted_mean_ppl_error=weighted_mean_ppl_error,
        mean_relative_error=mean_relative_error,
        weighted_mean_relative_error=weighted_mean_relative_error,
        mean_signed_error=mean_signed_error,
        weighted_mean_signed_error=weighted_mean_signed_error,
    )


def evaluate_rollout(
    rollout: ScalingLawRollout,
    weight_decay: Literal["inverse", "exp"] = "inverse",
    weight_decay_scale: float = 1.0,
    split_weights_fn: Callable[[RolloutSplit], float] = lambda split: 1.0,
) -> RolloutEvaluation:
    """
    Evaluate scaling law predictions across multiple rollout splits.

    Computes both perplexity-based and relative BPB error metrics:

    - **Perplexity error**: Scale-invariant; a 0.01 BPB error is ~0.7% everywhere
    - **Relative BPB error**: Weights low-loss predictions more heavily

    :param rollout: ScalingLawRollout to evaluate.
    :param weight_decay: Distance decay function for within-split weighting.
    :param weight_decay_scale: Scale parameter for decay.
    :param split_weights_fn: Function that computes a weight for each split.
        Takes a RolloutSplit and returns a float. Weights are normalized to sum to 1.
        Default is uniform weighting (``lambda split: 1.0``).

        Common weighting functions::

            # Weight by number of training points
            lambda split: len(split.train_N)

            # Weight by total training compute
            lambda split: np.sum(6.0 * split.train_N * split.train_D)

    :returns: RolloutEvaluation with per-split and aggregate metrics.

    Example::

        rollout = ScalingLawRollout.fit(N=N, D=D, loss=loss, fit_fn=...)
        evaluation = evaluate_rollout(rollout)

        # Emphasize splits with more training data
        evaluation = evaluate_rollout(
            rollout, split_weights_fn=lambda s: len(s.train_N)
        )
    """
    if not rollout.splits:
        raise ValueError("rollout.splits list cannot be empty")

    split_evals = [
        evaluate_split(
            split,
            weight_decay=weight_decay,
            weight_decay_scale=weight_decay_scale,
        )
        for split in rollout.splits
    ]

    # Compute split weights by applying the weight function to each split
    computed_weights = np.array([split_weights_fn(split) for split in rollout.splits], dtype=float)
    computed_weights = computed_weights / computed_weights.sum()

    # Aggregate perplexity-based metrics
    overall_mean_ppl = sum(w * e.mean_ppl_error for w, e in zip(computed_weights, split_evals))
    overall_weighted_ppl = sum(
        w * e.weighted_mean_ppl_error for w, e in zip(computed_weights, split_evals)
    )

    # Aggregate relative BPB metrics
    overall_mean_rel = sum(w * e.mean_relative_error for w, e in zip(computed_weights, split_evals))
    overall_weighted_rel = sum(
        w * e.weighted_mean_relative_error for w, e in zip(computed_weights, split_evals)
    )

    # Aggregate bias indicators
    overall_mean_signed = sum(
        w * e.mean_signed_error for w, e in zip(computed_weights, split_evals)
    )
    overall_weighted_signed = sum(
        w * e.weighted_mean_signed_error for w, e in zip(computed_weights, split_evals)
    )

    return RolloutEvaluation(
        split_evaluations=split_evals,
        overall_mean_ppl_error=float(overall_mean_ppl),
        overall_weighted_mean_ppl_error=float(overall_weighted_ppl),
        overall_mean_relative_error=float(overall_mean_rel),
        overall_weighted_mean_relative_error=float(overall_weighted_rel),
        overall_mean_signed_error=float(overall_mean_signed),
        overall_weighted_mean_signed_error=float(overall_weighted_signed),
    )
