"""
Evaluation metrics for scaling law predictions.

This module provides principled metrics for evaluating how well a scaling law
extrapolates to held-out data points, with appropriate weighting by distance.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from .scaling_laws import RolloutSplit


def perplexity_ratio(predicted_bpb: np.ndarray, actual_bpb: np.ndarray) -> np.ndarray:
    """
    Compute the perplexity ratio between predicted and actual loss.

    Why perplexity ratio?
    ---------------------
    Perplexity ratio is a scale-invariant error metric for language model loss.
    Unlike absolute BPB differences, perplexity ratio has consistent interpretation
    across different loss scales:

    - A 0.01 BPB error at loss=1.0 is a 0.7% perplexity increase
    - A 0.01 BPB error at loss=2.0 is also a 0.7% perplexity increase

    This makes it suitable for comparing prediction quality across models with
    different baseline loss levels. It also has intuitive meaning: a 5% perplexity
    error means the predicted distribution is 5% less efficient at compressing the
    data than the actual model achieved.

    Mathematical relationship:
    --------------------------
    Since BPB = log2(perplexity), we have:
        perplexity = 2^BPB
        ratio = 2^predicted / 2^actual = 2^(predicted - actual)

    Interpretation:
    ---------------
    - ratio > 1: Overestimate (predicted higher loss/perplexity than actual)
    - ratio < 1: Underestimate (predicted lower loss/perplexity than actual)

    :param predicted_bpb: Predicted loss in bits-per-byte.
    :param actual_bpb: Actual loss in bits-per-byte.
    :returns: Perplexity ratio for each point.
    """
    return np.power(2.0, predicted_bpb - actual_bpb)


def perplexity_ratio_error(predicted_bpb: np.ndarray, actual_bpb: np.ndarray) -> np.ndarray:
    """
    Compute the signed perplexity ratio error as a percentage.

    This converts the perplexity ratio to a more intuitive percentage error:
        Error = (perplexity_ratio - 1) * 100

    :param predicted_bpb: Predicted loss in bits-per-byte.
    :param actual_bpb: Actual loss in bits-per-byte.
    :returns: Percentage error in perplexity for each point (signed).
    """
    return (perplexity_ratio(predicted_bpb, actual_bpb) - 1) * 100


def rollout_distance(
    test_values: np.ndarray,
    cutoff_value: float,
    scale: Literal["log", "linear", "relative"] = "log",
) -> np.ndarray:
    """
    Compute the rollout distance from the training cutoff to each test point.

    Which scale to use?
    -------------------
    For Chinchilla-style scaling laws, "log" is recommended because:

    1. The scaling law is a power law (L = E + A/N^α + B/D^β), which is linear
       in log-space. Extrapolation difficulty scales with log-distance, not
       linear distance.

    2. Model sizes and token counts span orders of magnitude. A 100M → 1B
       extrapolation (10x) is similar in difficulty to 1B → 10B (also 10x),
       but very different in linear distance (900M vs 9B).

    3. Log distance gives equal weight to each "doubling" of scale, which
       matches how we typically think about scaling (2x, 4x, 10x factors).

    The other scales may be useful for:
    - "linear": When you care about absolute parameter/token differences
    - "relative": Similar to log for small distances, but doesn't handle
      large extrapolations as naturally

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
    distances: np.ndarray,
    decay: Literal["inverse", "exp"] = "inverse",
    decay_scale: float = 1.0,
) -> np.ndarray:
    """
    Compute weights that decay with distance.

    Closer points (smaller distance) get higher weights. This reflects the
    principle that scaling laws should be most accurate for nearby extrapolations,
    and errors on distant points are more tolerable.

    Which decay function to use?
    ----------------------------
    For scaling law evaluation, "inverse" is recommended because:

    1. It provides a natural "half-weight" interpretation: at distance = scale,
       the weight is exactly half of the weight at distance = 0.

    2. It decays gracefully without over-penalizing distant points. Exponential
       decay can be too aggressive, making distant predictions almost irrelevant.

    3. It matches the intuition that prediction difficulty increases roughly
       proportionally with log-distance (when using log distance scale).

    Comparison of decay functions:
    - "inverse": 1/(1+d) - Moderate decay, good default. At d=1, weight=0.5.
    - "exp": exp(-d) - Aggressive decay. At d=1, weight≈0.37. At d=3, weight≈0.05.
    - "soft_inverse": 1/sqrt(1+d²) - Gentler decay. Stays high longer, then drops.

    The decay_scale parameter:
    --------------------------
    With log distance, decay_scale=1.0 means weight halves every e≈2.72x increase
    in model size. For example:
    - 100M → 270M: weight ≈ 0.5
    - 100M → 730M: weight ≈ 0.33
    - 100M → 2.7B: weight ≈ 0.25

    Increase decay_scale to weight distant points more heavily (useful if you
    care about long-range extrapolation accuracy).

    :param distances: Rollout distances (should be positive).
    :param decay: Decay function:
        - "inverse": 1 / (1 + distance/scale), simple hyperbolic decay (recommended)
        - "exp": exp(-distance/scale), exponential decay
        - "soft_inverse": 1 / sqrt(1 + (distance/scale)^2), softer decay
    :param decay_scale: Scale parameter controlling decay rate.
        Larger values = slower decay = more weight on distant points.
    :returns: Normalized weights summing to 1.
    """
    d = distances / decay_scale

    if decay == "inverse":
        w = 1.0 / (1.0 + d)
    elif decay == "exp":
        w = np.exp(-d)
    else:
        raise ValueError(f"Unknown decay function: {decay}")

    # Normalize to sum to 1
    return w / w.sum()


@dataclass
class SplitEvaluation:
    """Evaluation results for a single rollout split."""

    cutoff_value: float
    """The training cutoff value."""

    n_test_points: int
    """Number of test points."""

    # Raw metrics
    perplexity_ratios: np.ndarray
    """Perplexity ratio for each test point."""

    distances: np.ndarray
    """Rollout distance for each test point."""

    weights: np.ndarray
    """Distance-based weight for each test point."""

    # Aggregate metrics
    mean_abs_ppl_error: float
    """Mean absolute perplexity error (unweighted), in percent."""

    weighted_mean_abs_ppl_error: float
    """Distance-weighted mean absolute perplexity error, in percent."""

    mean_ppl_ratio: float
    """Geometric mean of perplexity ratios (unweighted)."""

    weighted_mean_ppl_ratio: float
    """Distance-weighted geometric mean of perplexity ratios."""


@dataclass
class RolloutEvaluation:
    """Aggregated evaluation results across all rollout splits."""

    split_evaluations: list[SplitEvaluation]
    """Per-split evaluation results."""

    # Aggregate across splits
    overall_weighted_mean_abs_ppl_error: float
    """Overall distance-weighted mean absolute perplexity error, in percent."""

    overall_mean_abs_ppl_error: float
    """Overall unweighted mean absolute perplexity error, in percent."""


def evaluate_split(
    split: RolloutSplit,
    distance_scale: Literal["log", "linear", "relative"] = "log",
    weight_decay: Literal["inverse", "exp"] = "inverse",
    weight_decay_scale: float = 1.0,
) -> SplitEvaluation:
    """
    Evaluate a single rollout split.

    :param split: The RolloutSplit to evaluate.
    :param distance_scale: How to measure rollout distance ("log", "linear", "relative").
    :param weight_decay: Distance decay function ("inverse", "exp", "soft_inverse").
    :param weight_decay_scale: Scale parameter for decay (larger = slower decay).
    :returns: SplitEvaluation with detailed metrics.
    """
    test_loss = np.atleast_1d(split.test_loss)
    predictions = np.atleast_1d(split.predictions)

    # Get the appropriate test values based on cutoff variable
    if split.cutoff_variable == "N":
        test_values = np.atleast_1d(split.test_N)
    else:
        test_values = np.atleast_1d(split.test_D)

    # Compute perplexity ratios
    ppl_ratios = perplexity_ratio(predictions, test_loss)

    # Compute distances and weights
    distances = rollout_distance(test_values, split.cutoff_value, scale=distance_scale)
    weights = distance_weights(distances, decay=weight_decay, decay_scale=weight_decay_scale)

    # Compute aggregate metrics
    abs_ppl_errors = np.abs(ppl_ratios - 1) * 100  # Convert to percentage

    mean_abs_ppl_error = float(np.mean(abs_ppl_errors))
    weighted_mean_abs_ppl_error = float(np.sum(weights * abs_ppl_errors))

    # Geometric mean of perplexity ratios (use log for numerical stability)
    log_ppl_ratios = np.log(ppl_ratios)
    mean_ppl_ratio = float(np.exp(np.mean(log_ppl_ratios)))
    weighted_mean_ppl_ratio = float(np.exp(np.sum(weights * log_ppl_ratios)))

    return SplitEvaluation(
        cutoff_value=split.cutoff_value,
        n_test_points=len(test_loss),
        perplexity_ratios=ppl_ratios,
        distances=distances,
        weights=weights,
        mean_abs_ppl_error=mean_abs_ppl_error,
        weighted_mean_abs_ppl_error=weighted_mean_abs_ppl_error,
        mean_ppl_ratio=mean_ppl_ratio,
        weighted_mean_ppl_ratio=weighted_mean_ppl_ratio,
    )


def evaluate_rollout(
    splits: list[RolloutSplit],
    distance_scale: Literal["log", "linear", "relative"] = "log",
    weight_decay: Literal["inverse", "exp"] = "inverse",
    weight_decay_scale: float = 1.0,
    split_weights: Optional[np.ndarray] = None,
) -> RolloutEvaluation:
    """
    Evaluate scaling law predictions across all rollout splits.

    This provides a principled aggregate metric that:
    1. Uses perplexity ratio as the pointwise error metric (scale-invariant for BPB)
    2. Weights errors by rollout distance (closer = more important)
    3. Aggregates across splits with optional weighting

    :param splits: List of RolloutSplit objects to evaluate.
    :param distance_scale: How to measure rollout distance ("log", "linear", "relative").
    :param weight_decay: Distance decay function ("inverse", "exp").
    :param weight_decay_scale: Scale parameter for decay. For "log" distance scale,
        a value of 1.0 means weight halves roughly every e-fold increase in N.
    :param split_weights: Optional weights for each split when aggregating.
        If None, weights splits by number of test points.
    :returns: RolloutEvaluation with per-split and aggregate metrics.

    Example::

        from olmo_core.model_ladder.analysis import (
            ChinchillaParametricBootstrappedFit,
            ScalingLawRollout,
        )
        from olmo_core.model_ladder.analysis.eval import evaluate_rollout

        rollout = ScalingLawRollout(N=N, D=D, loss=loss)
        splits = rollout.evaluate(ChinchillaParametricBootstrappedFit.fit)

        evaluation = evaluate_rollout(splits)
        print(f"Weighted mean |ppl error|: {evaluation.overall_weighted_mean_abs_ppl_error:.2f}%")
    """
    if not splits:
        raise ValueError("splits list cannot be empty")

    # Evaluate each split
    split_evals = [
        evaluate_split(
            split,
            distance_scale=distance_scale,
            weight_decay=weight_decay,
            weight_decay_scale=weight_decay_scale,
        )
        for split in splits
    ]

    # Compute split weights
    if split_weights is None:
        # Weight by number of test points
        split_weights = np.array([e.n_test_points for e in split_evals], dtype=float)
    else:
        split_weights = np.asarray(split_weights)

    split_weights = split_weights / split_weights.sum()

    # Aggregate across splits
    overall_weighted = sum(
        w * e.weighted_mean_abs_ppl_error for w, e in zip(split_weights, split_evals)
    )
    overall_unweighted = sum(w * e.mean_abs_ppl_error for w, e in zip(split_weights, split_evals))

    return RolloutEvaluation(
        split_evaluations=split_evals,
        overall_weighted_mean_abs_ppl_error=float(overall_weighted),
        overall_mean_abs_ppl_error=float(overall_unweighted),
    )
