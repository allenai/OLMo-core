import multiprocessing
import os
from dataclasses import dataclass
from functools import partial
from itertools import product
from typing import Literal, Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from scipy.optimize import minimize
from tqdm import tqdm

from olmo_core.data.composable.utils import format_token_count
from olmo_core.model_ladder.utils import format_count

# Type alias for supported loss functions
LossFunctionType = Literal["log_huber", "asymmetric_mae"]


def _huber_loss(residuals: np.ndarray, delta: float) -> np.ndarray:
    """Huber loss function applied to residuals."""
    abs_r = np.abs(residuals)
    quadratic = 0.5 * residuals**2
    linear = delta * (abs_r - 0.5 * delta)
    return np.where(abs_r <= delta, quadratic, linear)


def _asymmetric_mae_loss(residuals: np.ndarray, w: float = 10.0) -> np.ndarray:
    """
    Asymmetric Mean Absolute Error loss function.

    Penalizes overprediction (negative residuals) more heavily than underprediction.
    """
    # residuals = y_true - y_pred
    # If residual < 0: y_pred > y_true (overprediction) → penalize more
    loss = np.abs(residuals).copy()
    overprediction_mask = residuals < 0
    loss[overprediction_mask] *= w
    return loss


def _optimize_single_init(
    init_params: tuple[float, float, float, float, float],
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    loss_fn: LossFunctionType,
    loss_param: float,
    scipy_bounds: list[tuple[float, float]],
    weights: Optional[np.ndarray] = None,
) -> tuple[float, tuple[float, float, float, float, float]] | None:
    """
    Run a single optimization from given initial parameters.

    This is a module-level function to enable pickling for multiprocessing.

    Args:
        init_params: Tuple of (E, A, alpha, B, beta) initial values
        N: Parameter counts
        D: Token counts
        L: Loss values (raw, not log-transformed)
        loss_fn: Loss function type ("log_huber" or "asymmetric_mae")
        loss_param: Parameter for loss function (delta for huber, w for asymmetric_mae)
        scipy_bounds: Bounds for optimization in same order (E, A, alpha, B, beta)
        weights: Optional weights for each observation (applied to loss function)

    Returns:
        Tuple of (loss, (E, A, alpha, B, beta)) if successful, None otherwise.
    """
    # All parameters in consistent order: (E, A, alpha, B, beta)
    p0 = tuple(float(x) for x in init_params)

    # Default to uniform weights if not provided
    if weights is None:
        weights = np.ones_like(L)

    if loss_fn == "log_huber":
        log_L = np.log(L)

        def objective(params: np.ndarray) -> float:
            E_param, A, alpha, B, beta = params
            L_pred = chinchilla_parametric_scaling_law(N, D, E_param, A, alpha, B, beta)
            L_pred = np.maximum(L_pred, 1e-10)
            log_residuals = log_L - np.log(L_pred)
            # Weighted sum of losses
            return np.sum(weights * _huber_loss(log_residuals, loss_param))

    elif loss_fn == "asymmetric_mae":

        def objective(params: np.ndarray) -> float:
            E_param, A, alpha, B, beta = params
            L_pred = chinchilla_parametric_scaling_law(N, D, E_param, A, alpha, B, beta)
            L_pred = np.maximum(L_pred, 1e-10)
            residuals = L - L_pred
            return np.sum(weights * _asymmetric_mae_loss(residuals, loss_param))

    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")

    try:
        result = minimize(
            objective,
            p0,
            method="L-BFGS-B",
            bounds=scipy_bounds,
            options={"maxiter": 10000},
        )
        if np.isfinite(result.fun):
            return (result.fun, tuple(result.x))
    except Exception:
        pass
    return None


@dataclass
class ResidualDiagnostics:
    """
    Comprehensive residual diagnostics for validating scaling law fits.

    A well-specified model should have:
    - Normal residuals (shapiro_p > alpha)
    - Homoscedastic residuals (constant variance across N and D)
    - No systematic patterns (residuals uncorrelated with log(N), log(D), and predictions)
    """

    # Basic statistics
    n_observations: int
    """Number of observations used in diagnostics."""
    rmse: float
    """Root mean squared error."""
    mae: float
    """Mean absolute error."""
    mean_residual: float
    """Mean of residuals (should be ~0 for unbiased fit)."""
    std_residual: float
    """Standard deviation of residuals."""
    r_squared: Optional[float]
    """R-squared from the fit."""

    # Normality test (Shapiro-Wilk)
    shapiro_stat: Optional[float] = None
    """Shapiro-Wilk test statistic."""
    shapiro_p: Optional[float] = None
    """Shapiro-Wilk p-value (p > alpha means normal)."""
    normality_ok: Optional[bool] = None
    """True if residuals appear normally distributed."""

    # Homoscedasticity tests (constant variance)
    corr_abs_residual_N: Optional[float] = None
    """Spearman correlation of |residuals| with log(N)."""
    p_abs_residual_N: Optional[float] = None
    """P-value for correlation of |residuals| with log(N)."""
    homoscedastic_N: Optional[bool] = None
    """True if variance is constant across N."""

    corr_abs_residual_D: Optional[float] = None
    """Spearman correlation of |residuals| with log(D)."""
    p_abs_residual_D: Optional[float] = None
    """P-value for correlation of |residuals| with log(D)."""
    homoscedastic_D: Optional[bool] = None
    """True if variance is constant across D."""

    # Systematic pattern tests
    corr_residual_N: Optional[float] = None
    """Spearman correlation of residuals with log(N)."""
    p_residual_N: Optional[float] = None
    """P-value for correlation of residuals with log(N)."""
    no_pattern_N: Optional[bool] = None
    """True if no systematic pattern with N."""

    corr_residual_D: Optional[float] = None
    """Spearman correlation of residuals with log(D)."""
    p_residual_D: Optional[float] = None
    """P-value for correlation of residuals with log(D)."""
    no_pattern_D: Optional[bool] = None
    """True if no systematic pattern with D."""

    corr_residual_pred: Optional[float] = None
    """Spearman correlation of residuals with predictions."""
    p_residual_pred: Optional[float] = None
    """P-value for correlation of residuals with predictions."""
    no_pattern_pred: Optional[bool] = None
    """True if no systematic pattern with predictions."""

    alpha: float = 0.05
    """Significance level used for tests."""

    @property
    def all_tests_pass(self) -> bool:
        """True if all diagnostic tests pass."""
        tests = [
            self.normality_ok,
            self.homoscedastic_N,
            self.homoscedastic_D,
            self.no_pattern_N,
            self.no_pattern_D,
            self.no_pattern_pred,
        ]
        return all(t is True for t in tests if t is not None)

    @property
    def n_tests_passed(self) -> tuple[int, int]:
        """Returns (passed, total) count of diagnostic tests."""
        tests = [
            self.normality_ok,
            self.homoscedastic_N,
            self.homoscedastic_D,
            self.no_pattern_N,
            self.no_pattern_D,
            self.no_pattern_pred,
        ]
        valid_tests = [t for t in tests if t is not None]
        passed = sum(1 for t in valid_tests if t is True)
        return passed, len(valid_tests)

    def report(self) -> str:
        """Generate a human-readable diagnostic report."""
        lines = []
        passed, total = self.n_tests_passed

        lines.append("=" * 70)
        lines.append("RESIDUAL DIAGNOSTICS REPORT")
        lines.append("=" * 70)

        # Summary
        status = "✓ PASS" if self.all_tests_pass else "⚠ ISSUES DETECTED"
        lines.append(f"\nOverall: {status} ({passed}/{total} tests passed)")

        # Basic statistics
        lines.append("\nFIT QUALITY")
        lines.append("-" * 40)
        r2_str = f"{self.r_squared:.4f}" if self.r_squared else "N/A"
        lines.append(f"  R²:               {r2_str}")
        lines.append(f"  RMSE:             {self.rmse:.6f}")
        lines.append(f"  MAE:              {self.mae:.6f}")
        lines.append(f"  Mean residual:    {self.mean_residual:.6f}")
        lines.append(f"  Std residual:     {self.std_residual:.6f}")
        lines.append(f"  N observations:   {self.n_observations}")

        # Normality
        lines.append("\nNORMALITY (Shapiro-Wilk test)")
        lines.append("-" * 40)
        if self.shapiro_p is not None:
            status = "✓ PASS" if self.normality_ok else "✗ FAIL"
            lines.append(f"  {status}: p={self.shapiro_p:.4f} (α={self.alpha})")
            if not self.normality_ok:
                lines.append("  → Residuals are NOT normally distributed")
                lines.append("  → Consider: outliers, model misspecification, or heavy tails")
        else:
            lines.append("  Could not compute (insufficient data)")

        # Homoscedasticity
        lines.append("\nHOMOSCEDASTICITY (constant variance)")
        lines.append("-" * 40)

        if self.p_abs_residual_N is not None:
            status = "✓ PASS" if self.homoscedastic_N else "✗ FAIL"
            lines.append(
                f"  vs log(N): {status} (ρ={self.corr_abs_residual_N:.3f}, p={self.p_abs_residual_N:.4f})"
            )
            if not self.homoscedastic_N:
                if self.corr_abs_residual_N and self.corr_abs_residual_N > 0:
                    lines.append("  → Variance INCREASES with model size")
                else:
                    lines.append("  → Variance DECREASES with model size")

        if self.p_abs_residual_D is not None:
            status = "✓ PASS" if self.homoscedastic_D else "✗ FAIL"
            lines.append(
                f"  vs log(D): {status} (ρ={self.corr_abs_residual_D:.3f}, p={self.p_abs_residual_D:.4f})"
            )
            if not self.homoscedastic_D:
                if self.corr_abs_residual_D and self.corr_abs_residual_D > 0:
                    lines.append("  → Variance INCREASES with token count")
                else:
                    lines.append("  → Variance DECREASES with token count")

        # Systematic patterns
        lines.append("\nSYSTEMATIC PATTERNS (should be uncorrelated)")
        lines.append("-" * 40)

        if self.p_residual_N is not None:
            status = "✓ PASS" if self.no_pattern_N else "✗ FAIL"
            lines.append(
                f"  Residuals vs log(N): {status} (ρ={self.corr_residual_N:.3f}, p={self.p_residual_N:.4f})"
            )
            if not self.no_pattern_N:
                if self.corr_residual_N and self.corr_residual_N > 0:
                    lines.append("  → Model UNDERPREDICTS at large N")
                else:
                    lines.append("  → Model OVERPREDICTS at large N")

        if self.p_residual_D is not None:
            status = "✓ PASS" if self.no_pattern_D else "✗ FAIL"
            lines.append(
                f"  Residuals vs log(D): {status} (ρ={self.corr_residual_D:.3f}, p={self.p_residual_D:.4f})"
            )
            if not self.no_pattern_D:
                if self.corr_residual_D and self.corr_residual_D > 0:
                    lines.append("  → Model UNDERPREDICTS at large D")
                else:
                    lines.append("  → Model OVERPREDICTS at large D")

        if self.p_residual_pred is not None:
            status = "✓ PASS" if self.no_pattern_pred else "✗ FAIL"
            lines.append(
                f"  Residuals vs Predicted: {status} (ρ={self.corr_residual_pred:.3f}, p={self.p_residual_pred:.4f})"
            )
            if not self.no_pattern_pred:
                lines.append("  → Systematic bias in predictions")

        # Recommendations
        if not self.all_tests_pass:
            lines.append("\nRECOMMENDATIONS")
            lines.append("-" * 40)

            if self.normality_ok is False:
                lines.append("  • Check for outliers in training data")
                lines.append("  • Consider robust regression or outlier removal")

            if self.homoscedastic_N is False or self.homoscedastic_D is False:
                lines.append(
                    "  • Variance changes with scale - predictions less reliable at extremes"
                )
                lines.append("  • Consider weighted regression or log-space fitting")

            if self.no_pattern_N is False or self.no_pattern_D is False:
                lines.append("  • Systematic bias detected - model may be misspecified")
                lines.append("  • Consider adding interaction terms or different functional form")

        lines.append("=" * 70)
        return "\n".join(lines)

    def __repr__(self) -> str:
        passed, total = self.n_tests_passed
        status = "PASS" if self.all_tests_pass else "ISSUES"
        return f"ResidualDiagnostics({status}, {passed}/{total} tests, RMSE={self.rmse:.4f})"


def _extract_pareto_frontier_dominance(
    N: np.ndarray,
    D: np.ndarray,
    C: np.ndarray,
    L: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract the true Pareto frontier based on dominance in the (C, L) space.

    A point is on the Pareto frontier if no other point has both lower compute
    AND lower loss. This is more principled than binning because it doesn't
    require choosing arbitrary bin sizes and naturally adapts to the data density.

    Args:
        N: Array of parameter counts
        D: Array of token counts
        C: Array of compute values
        L: Array of loss values

    Returns:
        Tuple of (N_frontier, D_frontier, C_frontier, L_frontier) arrays containing
        only the Pareto-optimal points (not dominated by any other point)
    """
    n = len(C)
    if n == 0:
        raise ValueError("Empty input arrays")

    # A point i is dominated if there exists j such that C[j] <= C[i] and L[j] <= L[i]
    # with at least one strict inequality
    is_dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j is at least as good in both dimensions and strictly better in at least one
            if C[j] <= C[i] and L[j] <= L[i] and (C[j] < C[i] or L[j] < L[i]):
                is_dominated[i] = True
                break

    frontier_mask = ~is_dominated

    if not frontier_mask.any():
        raise ValueError("No Pareto-optimal points found")

    # Sort frontier by compute for consistency
    frontier_indices = np.where(frontier_mask)[0]
    sort_order = np.argsort(C[frontier_indices])
    idx = frontier_indices[sort_order]

    return N[idx], D[idx], C[idx], L[idx]


def chinchilla_parametric_scaling_law(
    N: np.ndarray, D: np.ndarray, E: float, A: float, alpha: float, B: float, beta: float
) -> np.ndarray:
    """
    Compute loss for given parameter count and token count using the Chinchilla scaling law.

    L(N, D) = E + A / N^α + B / D^β

    Uses logarithmic transformation for numerical stability with large N, D values:
        A / N^α  →  exp(log(A) - α*log(N))
        B / D^β  →  exp(log(B) - β*log(D))
    """
    # Use log-space computation to avoid underflow/overflow with large N, D
    param_term = np.exp(np.log(A) - alpha * np.log(N))
    data_term = np.exp(np.log(B) - beta * np.log(D))
    return E + param_term + data_term


@dataclass
class ChinchillaIsoParamFit:
    """Results from fitting the Chinchilla scaling law using Approach 1 (IsoParam).

    Input: Observations N_i, D_ij, L_ij, C_ij for model size i and data size j

    Output: Power Laws
        L(C) = E + A / C^alpha           (loss as function of compute)
        N_opt(C) = G * C^a               (optimal parameters as function of compute)
        D_opt(C) = H * C^b               (optimal tokens as function of compute)

    Method (Chinchilla Approach 1 with Pareto dominance):
    1. For each parameter count N, train models to different amounts of data D
    2. Extract the Pareto frontier: points where no other point has both lower
       compute AND lower loss (dominance-based, not binning)
    3. Fit power laws for L(C), N_opt(C), and D_opt(C) on these frontier points only

    This is inspired by the Chinchilla paper's approach of finding optimal (N, D)
    configurations, but uses true Pareto dominance instead of compute binning.
    This avoids arbitrary bin size choices and naturally adapts to data density.

    Where:
        N = number of parameters
        D = number of training tokens
        C = measured compute (FLOPs or petaFLOPs) - NOT the 6ND approximation
    """

    # Loss scaling law: L(C) = E + A / C^alpha
    E: float
    """Irreducible loss (entropy floor)"""
    A: float
    """Loss scaling coefficient"""
    alpha: float
    """Loss scaling exponent"""

    # Optimal N scaling law: N_opt(C) = G * C^a
    G: float
    """Optimal N coefficient"""
    a: float
    """Optimal N exponent (typically ~0.50)"""

    # Optimal D scaling law: D_opt(C) = H * C^b
    H: float
    """Optimal D coefficient"""
    b: float
    """Optimal D exponent (typically ~0.50)"""

    r_squared_loss: Optional[float] = None
    """R-squared of the loss fit L(C)"""
    r_squared_N: Optional[float] = None
    """R-squared of the N_opt(C) fit"""
    r_squared_D: Optional[float] = None
    """R-squared of the D_opt(C) fit"""

    # Stored data from fitting (set by fit())
    _N: Optional[np.ndarray] = None
    """Parameter counts used in fitting (stored automatically by fit())"""
    _D: Optional[np.ndarray] = None
    """Token counts used in fitting (stored automatically by fit())"""
    _C: Optional[np.ndarray] = None
    """Compute values used in fitting (stored automatically by fit())"""
    _loss: Optional[np.ndarray] = None
    """Loss values used in fitting (stored automatically by fit())"""

    def predict_loss(self, C: ArrayLike) -> np.ndarray:
        """Predict loss for given compute values using L(C) = E + A / C^alpha."""
        C = np.asarray(C)
        # Use log-space for numerical stability: A/C^α → exp(log(A) - α*log(C))
        return self.E + np.exp(np.log(self.A) - self.alpha * np.log(C))

    def predict_optimal_N(self, C: ArrayLike) -> np.ndarray:
        """Predict optimal parameter count for given compute using N_opt(C) = G * C^a."""
        C = np.asarray(C)
        return self.G * np.power(C, self.a)

    def predict_optimal_D(self, C: ArrayLike) -> np.ndarray:
        """Predict optimal token count for given compute using D_opt(C) = H * C^b."""
        C = np.asarray(C)
        return self.H * np.power(C, self.b)

    @classmethod
    def fit(
        cls,
        N: ArrayLike,
        D: ArrayLike,
        C: ArrayLike,
        loss: ArrayLike,
        huber_delta: float = 1e-3,
    ) -> "ChinchillaIsoParamFit":
        """
        Fit the IsoParam scaling laws using Chinchilla Approach 1 with Pareto dominance.

        This method first extracts the true Pareto frontier (points not dominated by any
        other point in compute-loss space), then fits power laws on those optimal points:

        1. Extract Pareto frontier: points where no other point has both lower compute
           AND lower loss
        2. Fit L(C) = E + A / C^alpha on frontier (C, L) pairs (Huber loss in log space)
        3. Fit N_opt(C) = G * C^a on frontier (C, N) pairs (linear regression in log-log space)
        4. Fit D_opt(C) = H * C^b on frontier (C, D) pairs (linear regression in log-log space)

        This is inspired by the Chinchilla paper's Approach 1 but uses true Pareto dominance
        instead of compute binning, avoiding arbitrary bin size choices.

        Args:
            N: Array of parameter counts (one per observation)
            D: Array of token counts (one per observation)
            C: Array of compute values (one per observation, e.g., FLOPs or petaFLOPs)
            loss: Array of loss values (one per observation)
            huber_delta: Delta parameter for Huber loss in L(C) fitting

        Returns:
            ChinchillaIsoParamFit with fitted parameters
        """
        N = np.asarray(N)
        D = np.asarray(D)
        C = np.asarray(C)
        loss = np.asarray(loss)

        # Clean data
        mask = (
            np.isfinite(N)
            & np.isfinite(D)
            & np.isfinite(C)
            & np.isfinite(loss)
            & (N > 0)
            & (D > 0)
            & (C > 0)
            & (loss > 0)
        )
        N_valid, D_valid, C_valid, L_valid = N[mask], D[mask], C[mask], loss[mask]

        if len(N_valid) < 3:
            raise ValueError(f"Need at least 3 valid data points, got {len(N_valid)}")

        # Extract Pareto frontier using dominance: points where no other point
        # has both lower compute AND lower loss
        N_frontier, D_frontier, C_frontier, L_frontier = _extract_pareto_frontier_dominance(
            N_valid, D_valid, C_valid, L_valid
        )

        if len(N_frontier) < 3:
            raise ValueError(
                f"Need at least 3 frontier points, got {len(N_frontier)}. "
                "Add more data with varied compute-loss trade-offs."
            )

        log_C = np.log(C_frontier)
        log_L = np.log(L_frontier)

        def huber_loss(residuals: np.ndarray, delta: float) -> np.ndarray:
            abs_r = np.abs(residuals)
            return np.where(abs_r <= delta, 0.5 * residuals**2, delta * (abs_r - 0.5 * delta))

        # Fit L(C) = E + A / C^alpha on frontier points
        def loss_objective(params: np.ndarray) -> float:
            E_param, A, alpha = params
            # Use log-space for numerical stability: A/C^α → exp(log(A) - α*log(C))
            L_pred = E_param + np.exp(np.log(A) - alpha * log_C)
            L_pred = np.maximum(L_pred, 1e-10)
            log_residuals = log_L - np.log(L_pred)
            return np.sum(huber_loss(log_residuals, huber_delta))

        E_init = L_frontier.min() * 0.8
        A_init = (L_frontier.max() - E_init) * C_frontier.min() ** 0.1
        alpha_init = 0.1

        loss_result = minimize(
            loss_objective,
            [E_init, A_init, alpha_init],
            method="L-BFGS-B",
            bounds=[(0.0, L_frontier.min()), (1e-10, 1e10), (0.01, 1.0)],
        )
        E_fit, A, alpha = loss_result.x

        # R-squared for loss fit (on frontier points)
        L_pred = E_fit + np.exp(np.log(A) - alpha * log_C)
        log_L_pred = np.log(np.maximum(L_pred, 1e-10))
        ss_res_L = np.sum((log_L - log_L_pred) ** 2)
        ss_tot_L = np.sum((log_L - log_L.mean()) ** 2)
        r2_loss = 1 - ss_res_L / ss_tot_L if ss_tot_L > 0 else 0.0

        # 2. Fit N_opt(C) = G * C^a on frontier points
        # log(N) = log(G) + a * log(C)
        log_N = np.log(N_frontier)
        a, log_G = np.polyfit(log_C, log_N, 1)
        G = np.exp(log_G)

        # R-squared for N fit
        log_N_pred = log_G + a * log_C
        ss_res_N = np.sum((log_N - log_N_pred) ** 2)
        ss_tot_N = np.sum((log_N - log_N.mean()) ** 2)
        r2_N = 1 - ss_res_N / ss_tot_N if ss_tot_N > 0 else 0.0

        # 3. Fit D_opt(C) = H * C^b on frontier points
        # log(D) = log(H) + b * log(C)
        log_D = np.log(D_frontier)
        b, log_H = np.polyfit(log_C, log_D, 1)
        H = np.exp(log_H)

        # R-squared for D fit
        log_D_pred = log_H + b * log_C
        ss_res_D = np.sum((log_D - log_D_pred) ** 2)
        ss_tot_D = np.sum((log_D - log_D.mean()) ** 2)
        r2_D = 1 - ss_res_D / ss_tot_D if ss_tot_D > 0 else 0.0

        return cls(
            E=E_fit,
            A=A,
            alpha=alpha,
            G=G,
            a=a,
            H=H,
            b=b,
            r_squared_loss=r2_loss,
            r_squared_N=r2_N,
            r_squared_D=r2_D,
            _N=N_valid,
            _D=D_valid,
            _C=C_valid,
            _loss=L_valid,
        )

    def report(self, compute_examples: Optional[list[float]] = None) -> str:
        if compute_examples is None:
            compute_examples = [
                1e4,  # 10k petaflops
                1e5,  # 100k petaflops (GPT-2 1.5B was ~80k petaflops)
                1e6,  # 1M petaflops
                1e7,  # 10M petaflops (Llama1 7B was ~40M petaflops)
                1e8,  # 100M petaflops (GPT-3 175B was ~300M petaflops)
            ]

        lines = []
        lines.append("=" * 70)
        lines.append("CHINCHILLA ISOPARAM SCALING LAW FIT")
        lines.append("=" * 70)

        # Fitted equations
        lines.append("\nFITTED SCALING LAWS")
        lines.append("-" * 40)

        r2_loss_str = f" (R²={self.r_squared_loss:.4f})" if self.r_squared_loss else ""
        r2_N_str = f" (R²={self.r_squared_N:.4f})" if self.r_squared_N else ""
        r2_D_str = f" (R²={self.r_squared_D:.4f})" if self.r_squared_D else ""

        lines.append(
            f"  Loss:      L(C) = {self.E:.4f} + {self.A:.4f} / C^{self.alpha:.4f}{r2_loss_str}"
        )
        lines.append(f"  Params:    N_opt(C) = {self.G:.4e} × C^{self.a:.4f}{r2_N_str}")
        lines.append(f"  Tokens:    D_opt(C) = {self.H:.4e} × C^{self.b:.4f}{r2_D_str}")

        # Key insights
        lines.append("-" * 40)
        lines.append(f"  Entropy floor (E):     {self.E:.4f}")
        lines.append(
            f"  Param exponent (a):    {self.a:.4f}  {'← Chinchilla: ~0.50' if abs(self.a - 0.5) < 0.1 else ''}"
        )
        lines.append(
            f"  Token exponent (b):    {self.b:.4f}  {'← Chinchilla: ~0.50' if abs(self.b - 0.5) < 0.1 else ''}"
        )

        # Compute-optimal ratio
        # If both a and b are ~0.5, then N and D scale equally with compute
        # D/N ratio at any compute C = (H/G) * C^(b-a)
        if abs(self.a - self.b) < 0.05:
            tokens_per_param = self.H / self.G
            lines.append(f"  ✓ Tokens per param:    {tokens_per_param:.1f}  (constant since a ≈ b)")
        else:
            lines.append("  ⚠ Tokens/param ratio varies with compute (a ≠ b):")
            # Show how ratio varies at example compute values
            for C in compute_examples:
                N_opt = self.predict_optimal_N(np.array([C]))[0]
                D_opt = self.predict_optimal_D(np.array([C]))[0]
                ratio = D_opt / N_opt
                lines.append(f"      C={C:,.0f} PF → {ratio:.1f} tok/param")

        # Example predictions
        lines.append("\nCOMPUTE-OPTIMAL PREDICTIONS")
        lines.append("-" * 40)
        lines.append(
            f"  {'Compute (PF)':>12}  {'Params':>10}  {'Tokens':>10}  {'Loss':>8}  {'Tok/Param':>10}"
        )
        lines.append(f"  {'-' * 12}  {'-' * 10}  {'-' * 10}  {'-' * 8}  {'-' * 10}")

        for C in compute_examples:
            N_opt = self.predict_optimal_N(np.array([C]))[0]
            D_opt = self.predict_optimal_D(np.array([C]))[0]
            L_pred = self.predict_loss(np.array([C]))[0]
            tok_per_param = D_opt / N_opt
            lines.append(
                f"  {C:>12,.0f}  {format_count(int(N_opt)):>10}  {format_token_count(int(D_opt)):>10}  {L_pred:>8.4f}  {tok_per_param:>10.1f}"
            )

        # Interpretation
        lines.append("\nINTERPRETATION")
        lines.append("-" * 40)
        if self.a > 0.45 and self.a < 0.55 and self.b > 0.45 and self.b < 0.55:
            lines.append("  ✓ Exponents ~0.5: Matches Chinchilla finding that params and tokens")
            lines.append("    should scale equally with compute (10× compute → ~3.2× each)")
        elif self.a > self.b:
            lines.append("  ⚠ a > b: Params scale faster than tokens with compute.")
            lines.append("    At higher compute, prefer larger models trained on fewer tokens.")
        else:
            lines.append("  ⚠ b > a: Tokens scale faster than params with compute.")
            lines.append("    At higher compute, prefer smaller models trained on more tokens.")

        if self.r_squared_loss and self.r_squared_loss < 0.95:
            lines.append(
                f"  ⚠ Loss R²={self.r_squared_loss:.3f} is below 0.95 - fit may be unreliable"
            )

        lines.append("=" * 70)
        return "\n".join(lines)

    def __repr__(self) -> str:
        parts = [
            f"L(C) = {self.E:.4f} + {self.A:.4f}/C^{self.alpha:.4f}",
            f"N_opt(C) = {self.G:.4e} * C^{self.a:.4f}",
            f"D_opt(C) = {self.H:.4e} * C^{self.b:.4f}",
        ]
        if self.r_squared_loss is not None:
            parts[0] += f" (R²={self.r_squared_loss:.4f})"
        if self.r_squared_N is not None:
            parts[1] += f" (R²={self.r_squared_N:.4f})"
        if self.r_squared_D is not None:
            parts[2] += f" (R²={self.r_squared_D:.4f})"
        return "\n".join(parts)


@dataclass
class ChinchillaParametricFit:
    """
    Results from fitting the full Chinchilla N, D parametric scaling law.

    Input: Observations N_i, D_ij, L_ij for model size i and data size j

    Output: Two-variable power law
    L(N, D) = E + A / N^alpha + B / D^beta

    Where:
        N = number of parameters
        D = number of training tokens
    """

    E: float
    """Irreducible loss (entropy floor)"""
    A: float
    """Parameter scale coefficient"""
    alpha: float
    """Parameter scaling exponent"""
    B: float
    """Data scale coefficient"""
    beta: float
    """Data scaling exponent"""

    # Goodness of fit metrics
    r_squared: Optional[float] = None
    """R-squared of the fit (computed on log scale to match optimization objective)"""
    huber_loss: Optional[float] = None
    """Huber loss of the fit (computed on log scale to match optimization objective)"""

    # Stored data from fitting (set by fit())
    _N: Optional[np.ndarray] = None
    """Parameter counts used in fitting (stored automatically by fit())"""
    _D: Optional[np.ndarray] = None
    """Token counts used in fitting (stored automatically by fit())"""
    _loss: Optional[np.ndarray] = None
    """Loss values used in fitting (stored automatically by fit())"""
    _C: Optional[np.ndarray] = None
    """Compute values (stored automatically by fit(), computed as 6*N*D)"""

    @property
    def a_opt(self) -> float:
        """Compute-optimal N exponent: N_opt ∝ C^a_opt where a_opt = β/(α+β)"""
        return self.beta / (self.alpha + self.beta)

    @property
    def b_opt(self) -> float:
        """Compute-optimal D exponent: D_opt ∝ C^b_opt where b_opt = α/(α+β)"""
        return self.alpha / (self.alpha + self.beta)

    def predict_loss(self, N: np.ndarray, D: np.ndarray) -> np.ndarray:
        """Predict loss for given parameter count and token count."""
        return chinchilla_parametric_scaling_law(
            N, D, E=self.E, A=self.A, alpha=self.alpha, B=self.B, beta=self.beta
        )

    def effective_data_multiplier(self, other: "ChinchillaParametricFit", D: float) -> float:
        """
        Compute the MARGINAL data scaling efficiency ratio at a given D (B/D^β term only).

        Returns k representing how the data terms compare. When β values differ,
        the ratio depends on D.

        WARNING: This IGNORES the entropy floor E and parameter term A/N^α!
        If E_self ≠ E_other, the actual losses will NOT be equal.
        Use predict_loss() for true loss comparisons.

        Args:
            other: The baseline model to compare against.
            D: Token count at which to evaluate.

        Returns:
            k > 1 means self is more data-efficient at this scale.
        """
        return (other.B / self.B) ** (1 / other.beta) * D ** ((self.beta - other.beta) / other.beta)

    def effective_param_multiplier(self, other: "ChinchillaParametricFit", N: float) -> float:
        """
        Compute the MARGINAL parameter scaling efficiency ratio at a given N (A/N^α term only).

        Returns k representing how the parameter terms compare. When α values differ,
        the ratio depends on N.

        WARNING: This IGNORES the entropy floor E and data term B/D^β!
        If E_self ≠ E_other, the actual losses will NOT be equal.
        Use predict_loss() for true loss comparisons.

        Args:
            other: The baseline model to compare against.
            N: Parameter count at which to evaluate.

        Returns:
            k > 1 means self is more parameter-efficient at this scale.
        """
        return (other.A / self.A) ** (1 / other.alpha) * N ** (
            (self.alpha - other.alpha) / other.alpha
        )

    def loss_advantage(
        self, other: "ChinchillaParametricFit", N: float, D: float, ppl_threshold: float = 1.005
    ) -> tuple[float, float, float, str]:
        """
        Compute the ACTUAL loss advantage of this model vs another at a given (N, D) scale.

        Unlike effective_data_multiplier and effective_param_multiplier which only compare
        marginal scaling terms, this compares the full predicted losses including E.

        Since CE loss is in log-space, we express differences as perplexity ratios:
        - ppl_ratio = exp(L_other) / exp(L_self) = exp(L_other - L_self)
        - ppl_ratio > 1 means self has lower perplexity (better)
        - ppl_ratio of 1.05 means baseline has 5% higher perplexity

        Args:
            other: The baseline model to compare against.
            N: Parameter count.
            D: Token count.
            ppl_threshold: Perplexity ratio threshold for significance (default 1.005 = 0.5%).

        Returns:
            Tuple of (loss_self, loss_other, ppl_ratio, interpretation) where:
            - loss_self: Predicted CE loss for this model
            - loss_other: Predicted CE loss for the baseline
            - ppl_ratio: exp(L_other - L_self), >1 means self is better
            - interpretation: Human-readable comparison string
        """
        L_self = self.predict_loss(np.array([N]), np.array([D]))[0]
        L_other = other.predict_loss(np.array([N]), np.array([D]))[0]
        diff = L_other - L_self  # positive = self is better
        ppl_ratio = np.exp(diff)  # >1 means self has lower perplexity
        ppl_pct = (ppl_ratio - 1) * 100  # percentage difference in perplexity

        if ppl_ratio > ppl_threshold:
            interp = f"This model wins: {ppl_pct:+.2f}% ppl (ΔCE={diff:.4f})"
        elif ppl_ratio < 1 / ppl_threshold:
            interp = f"Baseline wins: {-ppl_pct:+.2f}% ppl (ΔCE={-diff:.4f})"
        else:
            interp = f"Approximately equal ({ppl_pct:+.2f}% ppl)"

        return L_self, L_other, ppl_ratio, interp

    def residual_diagnostics(
        self,
        N: Optional[ArrayLike] = None,
        D: Optional[ArrayLike] = None,
        loss: Optional[ArrayLike] = None,
        alpha: float = 0.05,
    ) -> ResidualDiagnostics:
        """
        Compute comprehensive residual diagnostics to validate the scaling law fit. We fit our
        scaling law model using "best-effort" training runs and then examine the residuals. In practice,
        we can expect to observe some paterns in the residuals:

        1. One-sidedness (Positive Bias) - negative residuals exist but are rare and small in magnitude.
            this is because the model is implicitly a best-achievable frontier, and training inefficiency
            can only make losses worse.
        2. Heteroscedasticity with scale - larger models are more prone to instability and partial divergence.
            smaller models can be more sensitve to certain hparams, like warmup steps and optimizer hparams.
        3. Correlated residuals across runs - training inefficiency is systematic, not random.
        4. Regime-dependent bias - residuals are not flat across (N, D). It is typical to see underprediction
            in small-N/large-D corner, overprediction in large-N/small-D corner.
        5. "Good" vs "Bad" runs - you are likely to see a dense band of small residuals and a sparse set of large,
            positive outliers. This is because most runs converge normally, but a minority will suffer from
            partial divergence, subcritical batch / lr scaling.

        Tests for:
        1. Normality (Shapiro-Wilk test)
        2. Homoscedasticity (constant variance across N and D)
        3. Systematic patterns (residuals correlated with log(N) or log(D))
        4. Overall fit quality metrics

        Args:
            N: Array of parameter counts. If None, uses stored values from fit().
            D: Array of token counts. If None, uses stored values from fit().
            loss: Array of observed loss values. If None, uses stored values from fit().
            alpha: Significance level for statistical tests (default 0.05).

        Returns:
            ResidualDiagnostics object with test results and interpretation.
        """
        # Use stored values if not provided
        if N is None:
            if self._N is None:
                raise ValueError("N not provided and no stored values from fit()")
            N = self._N
        if D is None:
            if self._D is None:
                raise ValueError("D not provided and no stored values from fit()")
            D = self._D
        if loss is None:
            if self._loss is None:
                raise ValueError("loss not provided and no stored values from fit()")
            loss = self._loss

        N = np.asarray(N)
        D = np.asarray(D)
        loss = np.asarray(loss)

        # Compute predictions and residuals
        L_pred = self.predict_loss(N, D)
        residuals = loss - L_pred
        log_N = np.log(N)
        log_D = np.log(D)

        n = len(residuals)

        # Basic statistics
        rmse = float(np.sqrt(np.mean(residuals**2)))
        mae = float(np.mean(np.abs(residuals)))
        mean_residual = float(np.mean(residuals))
        std_residual = float(np.std(residuals, ddof=1))

        # 1. Normality test (Shapiro-Wilk)
        # Note: Shapiro-Wilk works best for n < 5000
        if n >= 3:
            shapiro_result = stats.shapiro(residuals)
            shapiro_stat = float(shapiro_result.statistic)
            shapiro_p = float(shapiro_result.pvalue)
        else:
            shapiro_stat, shapiro_p = float("nan"), float("nan")
        normality_ok = shapiro_p > alpha if not np.isnan(shapiro_p) else None

        # 2. Homoscedasticity tests
        # Test if |residuals| correlate with log(N) or log(D) (Breusch-Pagan-like)
        abs_residuals = np.abs(residuals)

        # Helper to extract correlation and p-value from spearmanr result
        def spearman_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
            result = stats.spearmanr(x, y)
            # Access correlation and pvalue - result is SpearmanrResult namedtuple
            corr: float = float(getattr(result, "statistic", result[0]))  # type: ignore[arg-type]
            pval: float = float(getattr(result, "pvalue", result[1]))  # type: ignore[arg-type]
            return corr, pval

        # Correlation of |residuals| with log(N)
        if n >= 3:
            corr_abs_N, p_abs_N = spearman_corr(log_N, abs_residuals)
        else:
            corr_abs_N, p_abs_N = float("nan"), float("nan")
        homoscedastic_N = p_abs_N > alpha if not np.isnan(p_abs_N) else None

        # Correlation of |residuals| with log(D)
        if n >= 3:
            corr_abs_D, p_abs_D = spearman_corr(log_D, abs_residuals)
        else:
            corr_abs_D, p_abs_D = float("nan"), float("nan")
        homoscedastic_D = p_abs_D > alpha if not np.isnan(p_abs_D) else None

        # 3. Systematic pattern tests (residuals should not correlate with predictors)
        # Correlation of residuals with log(N)
        if n >= 3:
            corr_N, p_corr_N = spearman_corr(log_N, residuals)
        else:
            corr_N, p_corr_N = float("nan"), float("nan")
        no_pattern_N = p_corr_N > alpha if not np.isnan(p_corr_N) else None

        # Correlation of residuals with log(D)
        if n >= 3:
            corr_D, p_corr_D = spearman_corr(log_D, residuals)
        else:
            corr_D, p_corr_D = float("nan"), float("nan")
        no_pattern_D = p_corr_D > alpha if not np.isnan(p_corr_D) else None

        # 4. Additional: test residuals vs predicted (should be uncorrelated)
        if n >= 3:
            corr_pred, p_corr_pred = spearman_corr(L_pred, residuals)
        else:
            corr_pred, p_corr_pred = float("nan"), float("nan")
        no_pattern_pred = p_corr_pred > alpha if not np.isnan(p_corr_pred) else None

        return ResidualDiagnostics(
            n_observations=n,
            rmse=rmse,
            mae=mae,
            mean_residual=mean_residual,
            std_residual=std_residual,
            r_squared=self.r_squared,
            # Normality
            shapiro_stat=float(shapiro_stat) if not np.isnan(shapiro_stat) else None,
            shapiro_p=float(shapiro_p) if not np.isnan(shapiro_p) else None,
            normality_ok=normality_ok,
            # Homoscedasticity
            corr_abs_residual_N=float(corr_abs_N) if not np.isnan(corr_abs_N) else None,
            p_abs_residual_N=float(p_abs_N) if not np.isnan(p_abs_N) else None,
            homoscedastic_N=homoscedastic_N,
            corr_abs_residual_D=float(corr_abs_D) if not np.isnan(corr_abs_D) else None,
            p_abs_residual_D=float(p_abs_D) if not np.isnan(p_abs_D) else None,
            homoscedastic_D=homoscedastic_D,
            # Systematic patterns
            corr_residual_N=float(corr_N) if not np.isnan(corr_N) else None,
            p_residual_N=float(p_corr_N) if not np.isnan(p_corr_N) else None,
            no_pattern_N=no_pattern_N,
            corr_residual_D=float(corr_D) if not np.isnan(corr_D) else None,
            p_residual_D=float(p_corr_D) if not np.isnan(p_corr_D) else None,
            no_pattern_D=no_pattern_D,
            corr_residual_pred=float(corr_pred) if not np.isnan(corr_pred) else None,
            p_residual_pred=float(p_corr_pred) if not np.isnan(p_corr_pred) else None,
            no_pattern_pred=no_pattern_pred,
            alpha=alpha,
        )

    def compare_to(
        self,
        other: "ChinchillaParametricFit",
        self_name: str = "This",
        other_name: str = "Baseline",
        model_sizes: list[float] | None = None,
        token_counts: list[float] | None = None,
    ) -> str:
        """
        Generate a detailed comparison report between this model and another.

        Args:
            other: The baseline model to compare against.
            self_name: Name for this model in the report.
            other_name: Name for the baseline model in the report.
            model_sizes: Model sizes (N) at which to compute comparisons.
            token_counts: Token counts (D) at which to compute comparisons.

        Returns:
            Formatted comparison report string.
        """
        if model_sizes is None:
            model_sizes = [190e6, 1e9, 7e9, 70e9]
        if token_counts is None:
            token_counts = [100e9, 1e12, 5e12, 15e12]

        lines = []
        lines.append("=" * 70)
        lines.append(f"SCALING LAW COMPARISON: {self_name} vs {other_name}")
        lines.append("=" * 70)

        # Parameter comparison
        lines.append("\nPARAMETER COMPARISON")
        lines.append("-" * 40)

        def delta_str(val: float, ref: float, higher_is_better: bool = True) -> str:
            pct = (val - ref) / abs(ref) * 100
            sign = "+" if pct > 0 else ""
            if higher_is_better:
                symbol = "✓" if pct > 0 else "✗" if pct < -1 else "≈"
            else:
                symbol = "✓" if pct < 0 else "✗" if pct > 1 else "≈"
            return f"{sign}{pct:.1f}% {symbol}"

        lines.append(f"  {'Parameter':<25} {self_name:>12} {other_name:>12} {'Δ':>12}")
        lines.append(f"  {'-' * 25} {'-' * 12} {'-' * 12} {'-' * 12}")
        lines.append(
            f"  {'Entropy floor (E)':<25} {self.E:>12.4f} {other.E:>12.4f} {delta_str(self.E, other.E, higher_is_better=False):>12}"
        )
        lines.append(
            f"  {'Param coeff (A)':<25} {self.A:>12.4f} {other.A:>12.4f} {delta_str(self.A, other.A, higher_is_better=False):>12}"
        )
        lines.append(
            f"  {'Param exponent (α)':<25} {self.alpha:>12.4f} {other.alpha:>12.4f} {delta_str(self.alpha, other.alpha, higher_is_better=True):>12}"
        )
        lines.append(
            f"  {'Data coeff (B)':<25} {self.B:>12.4f} {other.B:>12.4f} {delta_str(self.B, other.B, higher_is_better=False):>12}"
        )
        lines.append(
            f"  {'Data exponent (β)':<25} {self.beta:>12.4f} {other.beta:>12.4f} {delta_str(self.beta, other.beta, higher_is_better=True):>12}"
        )
        if self.r_squared and other.r_squared:
            lines.append(f"  {'R²':<25} {self.r_squared:>12.4f} {other.r_squared:>12.4f}")

        # Compute-optimal allocation comparison
        lines.append("\nCOMPUTE-OPTIMAL ALLOCATION")
        lines.append("-" * 40)
        lines.append(f"  {'Exponent':<25} {self_name:>12} {other_name:>12}")
        lines.append(f"  {'-' * 25} {'-' * 12} {'-' * 12}")
        lines.append(f"  {'N_opt ∝ C^a (a=β/(α+β))':<25} {self.a_opt:>12.3f} {other.a_opt:>12.3f}")
        lines.append(f"  {'D_opt ∝ C^b (b=α/(α+β))':<25} {self.b_opt:>12.3f} {other.b_opt:>12.3f}")

        # Efficiency comparison
        lines.append("\nEFFICIENCY MULTIPLIERS")
        lines.append("-" * 40)
        lines.append(
            "  Data efficiency: k such that this model with D tokens ≈ baseline with k×D tokens"
        )
        lines.append(
            "  Param efficiency: k such that this model with N params ≈ baseline with k×N params"
        )
        lines.append(
            f"\n  {'Scale':<15} {'Data Eff (k)':<15} {'Param Eff (k)':<15} {'Interpretation':<25}"
        )
        lines.append(f"  {'-' * 15} {'-' * 15} {'-' * 15} {'-' * 25}")

        for N, D in zip(model_sizes, token_counts):
            N_label = format_count(int(N))
            D_label = format_token_count(int(D))
            data_mult = self.effective_data_multiplier(other, D)
            param_mult = self.effective_param_multiplier(other, N)

            # Interpretation
            if data_mult > 1.05 and param_mult > 1.05:
                interp = f"✓ {self_name} wins both"
            elif data_mult < 0.95 and param_mult < 0.95:
                interp = f"✗ {other_name} wins both"
            elif data_mult > 1.05:
                interp = "↑ Better data efficiency"
            elif param_mult > 1.05:
                interp = "↑ Better param efficiency"
            elif data_mult < 0.95:
                interp = "↓ Worse data efficiency"
            elif param_mult < 0.95:
                interp = "↓ Worse param efficiency"
            else:
                interp = "≈ Similar"

            lines.append(
                f"  {N_label + '/' + D_label:<15} {data_mult:>15.2f}x {param_mult:>15.2f}x {interp:<25}"
            )

        # Loss prediction comparison
        lines.append("\nPREDICTED LOSS COMPARISON")
        lines.append("-" * 40)
        header = "  N \\ D          "
        for D in token_counts:
            header += f"  {format_token_count(int(D)):>10}"
        lines.append(header)

        for N in model_sizes:
            row_self = f"  {format_count(int(N)):<10} {self_name[:4]:>4}"
            row_other = f"  {'':<10} {other_name[:4]:>4}"
            row_delta = f"  {'':<10} {'Δ':>4}"
            for D in token_counts:
                L_self = self.predict_loss(np.array([N]), np.array([D]))[0]
                L_other = other.predict_loss(np.array([N]), np.array([D]))[0]
                delta = L_self - L_other
                row_self += f"  {L_self:>10.4f}"
                row_other += f"  {L_other:>10.4f}"
                sign = "+" if delta > 0 else ""
                row_delta += f"  {sign}{delta:>9.4f}"
            lines.append(row_self)
            lines.append(row_other)
            lines.append(row_delta)
            lines.append("")

        # Summary interpretation
        lines.append("SUMMARY")
        lines.append("-" * 40)

        # Check scaling behavior
        alpha_diff = self.alpha - other.alpha
        beta_diff = self.beta - other.beta
        A_ratio = self.A / other.A
        B_ratio = self.B / other.B

        if alpha_diff > 0.01 and beta_diff > 0.01:
            lines.append(f"  ✓ {self_name} has better scaling (higher α and β)")
            lines.append("    → Will increasingly outperform baseline at larger scales")
        elif alpha_diff < -0.01 and beta_diff < -0.01:
            lines.append(f"  ✗ {self_name} has worse scaling (lower α and β)")
            lines.append("    → Baseline will increasingly outperform at larger scales")
        elif abs(alpha_diff) < 0.01 and abs(beta_diff) < 0.01:
            lines.append(f"  ≈ {self_name} has similar scaling exponents")
        else:
            if alpha_diff > 0.01:
                lines.append(f"  ↑ {self_name} scales better with parameters (higher α)")
            elif alpha_diff < -0.01:
                lines.append(f"  ↓ {self_name} scales worse with parameters (lower α)")
            if beta_diff > 0.01:
                lines.append(f"  ↑ {self_name} scales better with data (higher β)")
            elif beta_diff < -0.01:
                lines.append(f"  ↓ {self_name} scales worse with data (lower β)")

        if A_ratio < 0.95 and B_ratio < 0.95:
            lines.append(f"  ✓ {self_name} is more efficient (lower A and B coefficients)")
        elif A_ratio > 1.05 and B_ratio > 1.05:
            lines.append(f"  ✗ {self_name} is less efficient (higher A and B coefficients)")
        elif A_ratio < 0.95:
            lines.append(f"  ↑ {self_name} is more parameter-efficient (lower A)")
        elif B_ratio < 0.95:
            lines.append(f"  ↑ {self_name} is more data-efficient (lower B)")

        # Crossover analysis if scaling differs
        if (alpha_diff < -0.01 or beta_diff < -0.01) and (A_ratio < 0.95 or B_ratio < 0.95):
            lines.append("")
            lines.append("  ⚠ CROSSOVER WARNING: Better efficiency but worse scaling")
            lines.append("    This model may win at small scales but lose at large scales.")

        lines.append("=" * 70)
        return "\n".join(lines)

    @staticmethod
    def _fit_with_grid_search(
        N: np.ndarray,
        D: np.ndarray,
        L: np.ndarray,
        loss_fn: LossFunctionType,
        loss_param: float,
        parallel: bool = True,
        weights: Optional[np.ndarray] = None,
    ) -> tuple[float, float, float, float, float]:
        """
        Fit Chinchilla scaling law parameters using grid search over initializations.

        Uses a grid of initializations and returns the fit with the lowest loss.

        Args:
            N: Array of parameter counts (cleaned)
            D: Array of token counts (cleaned)
            L: Array of loss values (cleaned)
            loss_fn: Loss function to use ("log_huber" or "asymmetric_mae")
            loss_param: Parameter for loss function:
                - For "log_huber": delta parameter (default 1e-3)
                - For "asymmetric_mae": overprediction weight w (default 10.0)
            parallel: If True, use multiprocessing for parallel optimization
            weights: Optional weights for each observation (applied to loss function)

        Returns:
            Tuple of (E, A, alpha, B, beta) fitted parameters
        """
        # E (entropy floor) must be <= minimum observed loss
        L_min = float(L.min())

        # Bounds in order: (E, A, alpha, B, beta)
        # E in [0, L_min], A in [1e-10, 1e20], alpha in [0.01, 2], B in [1e-10, 1e20], beta in [0.01, 2]
        scipy_bounds = [(0.0, L_min), (1e-10, 1e20), (0.01, 2.0), (1e-10, 1e20), (0.01, 2.0)]

        # Grid of initializations to find the best fit
        # Order: (E, A, alpha, B, beta)
        num_slices = 4
        E_grid = np.linspace(0.0, L_min, num_slices)
        A_grid = np.linspace(1, 20, num_slices)
        alpha_grid = np.linspace(0.2, 0.8, num_slices)
        B_grid = np.linspace(1, 20, num_slices)
        beta_grid = np.linspace(0.2, 0.8, num_slices)

        grid: list[tuple[float, float, float, float, float]] = [
            (float(E), float(A), float(alpha), float(B), float(beta))
            for E, A, alpha, B, beta in product(E_grid, A_grid, alpha_grid, B_grid, beta_grid)
        ]

        # Create partial function with fixed parameters for multiprocessing
        optimize_fn = partial(
            _optimize_single_init,
            N=N,
            D=D,
            L=L,
            loss_fn=loss_fn,
            loss_param=loss_param,
            scipy_bounds=scipy_bounds,
            weights=weights,
        )

        results: list[tuple[float, tuple[float, float, float, float, float]]] = []

        if parallel:
            n_workers = os.cpu_count() or 1
            ctx = multiprocessing.get_context("fork")
            with ctx.Pool(n_workers) as pool:
                for res in pool.imap_unordered(optimize_fn, grid):
                    if res is not None:
                        results.append(res)
        else:
            for init_params in grid:
                res = optimize_fn(init_params)
                if res is not None:
                    results.append(res)

        if not results:
            raise ValueError("All optimization attempts failed")

        # Find the result with the lowest loss
        best_loss, best_params = min(results, key=lambda x: x[0])
        E_fit, A, alpha, B, beta = best_params
        return float(E_fit), float(A), float(alpha), float(B), float(beta)

    @classmethod
    def fit(
        cls,
        N: ArrayLike,
        D: ArrayLike,
        loss: ArrayLike,
        loss_fn: LossFunctionType = "asymmetric_mae",
        loss_param: Optional[float] = None,
        parallel: bool = True,
        weights: Optional[ArrayLike] = None,
    ) -> "ChinchillaParametricFit":
        """
        Fit the full Chinchilla scaling law: L = E + A/N^alpha + B/D^beta

        Args:
            N: Array of parameter counts
            D: Array of token counts
            loss: Array of loss values
            loss_fn: Loss function to use:
                - "log_huber": Huber loss on log(L) (Chinchilla paper methodology)
                - "asymmetric_mae": Asymmetric MAE that penalizes overprediction more
            loss_param: Parameter for loss function:
                - For "log_huber": delta parameter (default 1e-3)
                - For "asymmetric_mae": overprediction weight w (default 10.0)
            parallel: If True, use multiprocessing for grid search optimization
            weights: Optional weights for each observation. Higher weights give more
                importance to those points during fitting. Common choices:
                - np.sqrt(6 * N * D): Weight by sqrt(compute) to emphasize large-scale points
                - N * D: Weight by compute (stronger emphasis on large scale)
                - None: Uniform weights (default)

        Returns:
            ChinchillaParametricFit with fitted parameters
        """
        # Set default loss_param based on loss function
        if loss_param is None:
            loss_param = 1e-3 if loss_fn == "log_huber" else 10.0

        N = np.asarray(N)
        D = np.asarray(D)
        loss = np.asarray(loss)

        # Clean data
        mask = np.isfinite(N) & np.isfinite(D) & np.isfinite(loss) & (N > 0) & (D > 0) & (loss > 0)
        N, D, L = N[mask], D[mask], loss[mask]

        # Apply mask to weights if provided
        weights_clean: Optional[np.ndarray] = None
        if weights is not None:
            weights = np.asarray(weights)
            weights_clean = weights[mask]
            # Normalize weights to sum to len(N) to preserve scale of loss function
            weights_clean = weights_clean * len(N) / weights_clean.sum()

        if len(N) < 5:
            raise ValueError(f"Need at least 5 valid data points, got {len(N)}")

        E_fit, A, alpha, B, beta = cls._fit_with_grid_search(
            N,
            D,
            L,
            loss_fn=loss_fn,
            loss_param=loss_param,
            parallel=parallel,
            weights=weights_clean,
        )

        # R-squared (computed on log scale for consistency)
        log_L = np.log(L)
        L_pred = chinchilla_parametric_scaling_law(N, D, E_fit, A, alpha, B, beta)
        log_L_pred = np.log(np.maximum(L_pred, 1e-10))
        ss_res = np.sum((log_L - log_L_pred) ** 2)
        ss_tot = np.sum((log_L - log_L.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        return cls(
            E=E_fit,
            A=A,
            alpha=alpha,
            B=B,
            beta=beta,
            r_squared=r_squared,
            _N=N,
            _D=D,
            _loss=L,
            _C=6 * N * D,
        )

    def report(
        self,
        model_sizes: Optional[list[float]] = None,
        token_counts: Optional[list[float]] = None,
    ) -> str:
        """
        Generate a rich report of the parametric scaling law fit for LLM practitioners.

        Args:
            model_sizes: List of parameter counts to show predictions for.
            token_counts: List of token counts to show predictions for.

        Returns:
            Formatted string report.
        """
        if model_sizes is None:
            model_sizes = [190e6, 1e9, 7e9, 70e9]  # 190M, 1B, 7B, 70B
        if token_counts is None:
            token_counts = [100e9, 1e12, 5e12, 6e12, 15e12]  # 100B, 1T, 5T, 15T

        lines = []
        lines.append("=" * 70)
        lines.append("CHINCHILLA PARAMETRIC SCALING LAW FIT")
        lines.append("=" * 70)

        # Fitted equation
        lines.append("\nFITTED SCALING LAW")
        lines.append("-" * 40)
        r2_str = f" (R²={self.r_squared:.4f})" if self.r_squared else ""
        lines.append(
            f"  L(N, D) = {self.E:.4f} + {self.A:.4f}/N^{self.alpha:.4f} + {self.B:.4f}/D^{self.beta:.4f}{r2_str}"
        )

        # Key parameters
        lines.append("-" * 40)
        lines.append(f"  Entropy floor (E):     {self.E:.4f}")
        lines.append(f"  Param coefficient (A): {self.A:.4f}")
        lines.append(
            f"  Param exponent (α):    {self.alpha:.4f}  {'← Chinchilla: ~0.34' if abs(self.alpha - 0.34) < 0.05 else ''}"
        )
        lines.append(f"  Data coefficient (B):  {self.B:.4f}")
        lines.append(
            f"  Data exponent (β):     {self.beta:.4f}  {'← Chinchilla: ~0.28' if abs(self.beta - 0.28) < 0.05 else ''}"
        )

        # Compute-optimal allocation (theoretical)
        # For L = E + A/N^α + B/D^β, optimal allocation: N ∝ C^(β/(α+β)), D ∝ C^(α/(α+β))
        lines.append("-" * 40)
        lines.append("  Compute-optimal allocation (theoretical):")
        lines.append(f"    N_opt ∝ C^{self.a_opt:.3f}  ← Chinchilla: 0.46")
        lines.append(f"    D_opt ∝ C^{self.b_opt:.3f}  ← Chinchilla: 0.54")
        if abs(self.a_opt - 0.5) < 0.05 and abs(self.b_opt - 0.5) < 0.05:
            lines.append("    ✓ Near equal scaling (~0.5 each) matches Chinchilla")
            # Optimal D/N = (βB/(αA))^(1/β) × N^((α-β)/β)
            # Show tokens/param at example model sizes
            base_ratio = (self.beta * self.B / (self.alpha * self.A)) ** (1 / self.beta)
            exp_factor = (self.alpha - self.beta) / self.beta
            lines.append("    Optimal tokens/param by model size:")
            for N_example, N_label in [(1e9, "1B"), (7e9, "7B"), (70e9, "70B")]:
                tok_per_param = base_ratio * (N_example**exp_factor)
                lines.append(f"      {N_label}: {tok_per_param:.1f} tokens/param")
        elif self.a_opt > self.b_opt:
            lines.append("    → Favor larger models over more tokens")
        else:
            lines.append("    → Favor more tokens over larger models")

        # Prediction table
        lines.append("\nPREDICTED LOSS (rows=params, cols=tokens)")
        lines.append("-" * 40)

        # Header row
        header = "  " + " " * 10
        for D in token_counts:
            header += f"  {format_token_count(int(D)):>10}"
        lines.append(header)
        lines.append("  " + "-" * (10 + len(token_counts) * 12))

        # Data rows
        for N in model_sizes:
            row = f"  {format_count(int(N)):>10}"
            for D in token_counts:
                L_pred = self.predict_loss(np.array([N]), np.array([D]))[0]
                row += f"  {L_pred:>10.4f}"
            lines.append(row)

        # Interpretation
        lines.append("\nINTERPRETATION")
        lines.append("-" * 40)
        if self.alpha > self.beta:
            lines.append("  α > β: Loss is more sensitive to model size than data.")
            lines.append("    → Scaling up parameters gives faster loss reduction per FLOP.")
        elif self.beta > self.alpha:
            lines.append("  β > α: Loss is more sensitive to data than model size.")
            lines.append("    → Scaling up tokens gives faster loss reduction per FLOP.")
        else:
            lines.append("  α ≈ β: Loss is equally sensitive to model size and data.")

        if self.r_squared and self.r_squared < 0.95:
            lines.append(f"  ⚠ R²={self.r_squared:.3f} is below 0.95 - fit may be unreliable")

        lines.append("=" * 70)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"L = {self.E:.4f} + {self.A:.4f}/N^{self.alpha:.4f} + {self.B:.4f}/D^{self.beta:.4f}"
            f" | R²={self.r_squared:.4f}"
            if self.r_squared
            else ""
        )

    @classmethod
    def rank_multiple(
        cls,
        fits: dict[str, "ChinchillaParametricFit"],
        model_sizes: Optional[list[float]] = None,
        token_counts: Optional[list[float]] = None,
        baseline_name: Optional[str] = None,
    ) -> str:
        """
        Compare and rank multiple ChinchillaParametricFit models across various criteria.

        Generates a comprehensive report including:
        - Parameter comparison table
        - Rankings by different criteria (entropy floor, scaling exponents, efficiency)
        - Loss predictions at various scales
        - Efficiency multipliers relative to a baseline
        - Pareto frontier identification

        Args:
            fits: Dictionary mapping model names to their ChinchillaParametricFit objects.
            model_sizes: Model sizes (N) at which to compute comparisons.
                Defaults to [190M, 1B, 7B, 70B].
            token_counts: Token counts (D) at which to compute comparisons.
                Defaults to [100B, 1T, 5T, 15T].
            baseline_name: Name of the baseline model for efficiency comparisons.
                If None, uses the first model in the dictionary.

        Returns:
            Formatted comparison and ranking report string.
        """
        if len(fits) < 2:
            raise ValueError("Need at least 2 fits to compare")

        if model_sizes is None:
            model_sizes = [190e6, 1e9, 7e9, 70e9]
        if token_counts is None:
            token_counts = [100e9, 1e12, 5e12, 15e12]

        names = list(fits.keys())
        if baseline_name is None:
            baseline_name = names[0]
        if baseline_name not in fits:
            raise ValueError(f"Baseline '{baseline_name}' not found in fits")

        baseline = fits[baseline_name]

        # Determine max name length for formatting (used throughout)
        max_name_len = max(len(name) for name in names)
        name_width = max(max_name_len, 12)

        # Calculate box width based on content
        # Parameter table: name + E(8) + A(10) + α(7) + B(10) + β(7) + a_opt(7) + b_opt(7) + R²(7) + separators
        box_width = name_width + 2 + 10 + 12 + 9 + 12 + 9 + 9 + 9 + 9 + 1
        box_width = max(box_width, 100)  # Minimum width for readability

        lines = []
        lines.append("=" * box_width)
        lines.append("MODEL LADDER SCALING LAW COMPARISON & RANKING")
        lines.append(f"Comparing {len(fits)} configurations | Baseline: {baseline_name}")
        lines.append("=" * box_width)

        # ──────────────────────────────────────────────────────────────────────
        # Parameter Comparison Table
        # ──────────────────────────────────────────────────────────────────────
        lines.append("\nPARAMETER COMPARISON")
        lines.append("-" * box_width)

        # Header
        header = (
            f"  {'Model':<{name_width}}  {'E':>8}  {'A':>10}  {'α':>7}  "
            f"{'B':>10}  {'β':>7}  {'a_opt':>7}  {'b_opt':>7}  {'R²':>7}"
        )
        lines.append(header)
        lines.append("-" * box_width)

        for name, fit in fits.items():
            r2_str = f"{fit.r_squared:.4f}" if fit.r_squared else "N/A"
            row = (
                f"  {name:<{name_width}}  {fit.E:>8.4f}  {fit.A:>10.2f}  {fit.alpha:>7.4f}  "
                f"{fit.B:>10.2f}  {fit.beta:>7.4f}  {fit.a_opt:>7.3f}  {fit.b_opt:>7.3f}  {r2_str:>7}"
            )
            lines.append(row)

        # ──────────────────────────────────────────────────────────────────────
        # Rankings by Different Criteria
        # ──────────────────────────────────────────────────────────────────────
        lines.append("\nRANKINGS BY CRITERIA (Best → Worst)")
        lines.append("-" * box_width)

        def rank_by(
            key_func, description: str, higher_is_better: bool = True
        ) -> list[tuple[str, float]]:
            ranked = sorted(fits.items(), key=lambda x: key_func(x[1]), reverse=higher_is_better)
            return [(name, key_func(fit)) for name, fit in ranked]

        def format_ranking(
            label: str, ranked: list[tuple[str, float]], fmt: str = ".4f"
        ) -> list[str]:
            """Format a ranking as multiple lines if needed."""
            result = [f"  {label}:"]
            items = [f"{name} ({val:{fmt}})" for name, val in ranked]
            # Join with arrows, wrap if too long
            ranking_line = "    " + " → ".join(items)
            if len(ranking_line) <= box_width:
                result.append(ranking_line)
            else:
                # Split into multiple lines
                current_line = "    "
                for i, item in enumerate(items):
                    addition = item if i == 0 else " → " + item
                    if len(current_line) + len(addition) > box_width - 4:
                        result.append(current_line)
                        current_line = "      → " + item
                    else:
                        current_line += addition
                if current_line.strip():
                    result.append(current_line)
            return result

        # Entropy floor (lower is better)
        ranked_E = rank_by(lambda f: f.E, "E", higher_is_better=False)
        lines.extend(format_ranking("Entropy Floor (E) ↓", ranked_E))

        # Parameter scaling (alpha, higher is better)
        ranked_alpha = rank_by(lambda f: f.alpha, "α", higher_is_better=True)
        lines.extend(format_ranking("Param Exponent (α) ↑", ranked_alpha))

        # Data scaling (beta, higher is better)
        ranked_beta = rank_by(lambda f: f.beta, "β", higher_is_better=True)
        lines.extend(format_ranking("Data Exponent (β) ↑", ranked_beta))

        # Param coefficient (A, lower is better)
        ranked_A = rank_by(lambda f: f.A, "A", higher_is_better=False)
        lines.extend(format_ranking("Param Coeff (A) ↓", ranked_A, fmt=".2f"))

        # Data coefficient (B, lower is better)
        ranked_B = rank_by(lambda f: f.B, "B", higher_is_better=False)
        lines.extend(format_ranking("Data Coeff (B) ↓", ranked_B, fmt=".2f"))

        # Combined scaling (alpha + beta, higher is better)
        ranked_combined = rank_by(lambda f: f.alpha + f.beta, "α+β", higher_is_better=True)
        lines.extend(format_ranking("Combined Scaling (α+β) ↑", ranked_combined))

        # Fit quality (R², higher is better)
        ranked_r2 = rank_by(
            lambda f: f.r_squared if f.r_squared else 0.0, "R²", higher_is_better=True
        )
        lines.extend(format_ranking("Fit Quality (R²) ↑", ranked_r2))

        # ──────────────────────────────────────────────────────────────────────
        # Loss Predictions at Key Scales
        # ──────────────────────────────────────────────────────────────────────
        lines.append("\nPREDICTED LOSS AT KEY SCALES")
        lines.append("-" * box_width)

        # Create header with token counts
        header = f"  {'Model':<{name_width}}  {'N':>8}"
        for D in token_counts:
            header += f"  {format_token_count(int(D)):>10}"
        lines.append(header)
        lines.append("-" * box_width)

        for N in model_sizes:
            N_label = format_count(int(N))
            for name in names:
                fit = fits[name]
                row = f"  {name:<{name_width}}  {N_label:>8}"
                for D in token_counts:
                    L_pred = fit.predict_loss(np.array([N]), np.array([D]))[0]
                    row += f"  {L_pred:>10.4f}"
                lines.append(row)
            lines.append("")  # Blank line between model sizes

        # ──────────────────────────────────────────────────────────────────────
        # Efficiency Multipliers vs Baseline
        # ──────────────────────────────────────────────────────────────────────
        lines.append(f"EFFICIENCY MULTIPLIERS vs BASELINE ({baseline_name})")
        lines.append("-" * box_width)
        lines.append("  Data eff:  k>1 means better B/D^β scaling (ignores E)")
        lines.append("  Param eff: k>1 means better A/N^α scaling (ignores E)")
        lines.append("  Result:    Based on ACTUAL predicted loss comparison (includes E)")
        lines.append("-" * box_width)

        # Header
        header = (
            f"  {'Model':<{name_width}}  {'Scale':>15}  "
            f"{'Data Eff':>10}  {'Param Eff':>10}  {'Actual Result':>18}"
        )
        lines.append(header)
        lines.append("-" * box_width)

        for name, fit in fits.items():
            if name == baseline_name:
                continue

            for i, (N, D) in enumerate(zip(model_sizes, token_counts)):
                N_label = format_count(int(N))
                D_label = format_token_count(int(D))
                scale_label = f"{N_label}/{D_label}"

                # Marginal efficiency multipliers (ignore E)
                data_mult = fit.effective_data_multiplier(baseline, D)
                param_mult = fit.effective_param_multiplier(baseline, N)

                # Actual loss comparison (includes E) - ppl_ratio > 1 means fit is better
                _, _, ppl_ratio, _ = fit.loss_advantage(baseline, N, D)
                ppl_pct = (ppl_ratio - 1) * 100  # percentage difference in perplexity

                # Compact interpretation for table display
                has_eff_advantage = data_mult > 1.0 or param_mult > 1.0
                if ppl_ratio > 1.0025:  # ~0.25% better perplexity
                    if has_eff_advantage:
                        interp = f"✓ Wins ({ppl_pct:+.1f}% ppl)"
                    else:
                        interp = f"✓ Lower ppl ({ppl_pct:+.1f}%)"
                elif ppl_ratio < 1 / 1.0025:  # ~0.25% worse perplexity
                    if has_eff_advantage:
                        interp = f"⚠ Eff but {ppl_pct:+.1f}% ppl"
                    else:
                        interp = f"✗ Higher ppl ({ppl_pct:+.1f}%)"
                else:
                    interp = "≈ Similar"

                # Only show model name on first row
                model_col = name if i == 0 else ""
                row = (
                    f"  {model_col:<{name_width}}  {scale_label:>15}  "
                    f"{data_mult:>9.2f}x  {param_mult:>9.2f}x  {interp:>18}"
                )
                lines.append(row)

            lines.append("")  # Blank line between models

        # ──────────────────────────────────────────────────────────────────────
        # Pareto Analysis: Which models are Pareto-optimal?
        # ──────────────────────────────────────────────────────────────────────
        lines.append("PARETO FRONTIER ANALYSIS")
        lines.append("-" * box_width)
        lines.append("  Metrics: E↓, A↓, α↑, B↓, β↑ (lower E/A/B better, higher α/β better)")
        lines.append("-" * box_width)

        # Check Pareto optimality based on all 5 parameters: E, A, alpha, B, beta
        # Lower E, A, B is better; higher alpha, beta is better
        def is_dominated(fit1: "ChinchillaParametricFit", fit2: "ChinchillaParametricFit") -> bool:
            """Returns True if fit1 is dominated by fit2 (fit2 is better on all metrics)."""
            # For each metric, check if fit2 is better or equal
            better_or_equal_E = fit2.E <= fit1.E
            better_or_equal_A = fit2.A <= fit1.A
            better_or_equal_alpha = fit2.alpha >= fit1.alpha
            better_or_equal_B = fit2.B <= fit1.B
            better_or_equal_beta = fit2.beta >= fit1.beta

            # Check if strictly better on at least one
            strictly_better_E = fit2.E < fit1.E - 1e-6
            strictly_better_A = fit2.A < fit1.A - 1e-6
            strictly_better_alpha = fit2.alpha > fit1.alpha + 1e-6
            strictly_better_B = fit2.B < fit1.B - 1e-6
            strictly_better_beta = fit2.beta > fit1.beta + 1e-6

            # Dominated if fit2 is better or equal on all, and strictly better on at least one
            all_better_or_equal = (
                better_or_equal_E
                and better_or_equal_A
                and better_or_equal_alpha
                and better_or_equal_B
                and better_or_equal_beta
            )
            at_least_one_strictly_better = (
                strictly_better_E
                or strictly_better_A
                or strictly_better_alpha
                or strictly_better_B
                or strictly_better_beta
            )

            return all_better_or_equal and at_least_one_strictly_better

        pareto_optimal = []
        dominated = []

        for name1, fit1 in fits.items():
            is_pareto = True
            dominated_by = []
            for name2, fit2 in fits.items():
                if name1 != name2 and is_dominated(fit1, fit2):
                    is_pareto = False
                    dominated_by.append(name2)

            if is_pareto:
                pareto_optimal.append(name1)
            else:
                dominated.append((name1, dominated_by))

        lines.append("  Pareto-optimal (no model strictly dominates on all 5 metrics):")
        for name in pareto_optimal:
            fit = fits[name]
            lines.append(
                f"    ✓ {name}: E={fit.E:.4f}, A={fit.A:.1f}, α={fit.alpha:.3f}, "
                f"B={fit.B:.1f}, β={fit.beta:.3f}"
            )

        if dominated:
            lines.append("")
            lines.append("  Dominated models:")
            for name, dominated_by in dominated:
                dom_str = ", ".join(dominated_by[:3])
                if len(dominated_by) > 3:
                    dom_str += "..."
                lines.append(f"    ✗ {name} (dominated by: {dom_str})")

        # ──────────────────────────────────────────────────────────────────────
        # Crossover Point Estimation
        # ──────────────────────────────────────────────────────────────────────
        lines.append("\nCROSSOVER POINT ANALYSIS")
        lines.append("-" * box_width)
        lines.append("  Where models with better coefficients (A,B) lose to better exponents (α,β)")
        lines.append("-" * box_width)

        # Find crossover points between baseline and other models
        # We search for N where L_baseline(N, r*N) ≈ L_other(N, r*N)
        # Using a fixed tokens-per-param ratio (e.g., 20 tokens/param as typical)
        tokens_per_param_ratio = 20.0

        def find_crossover_N(
            fit1: "ChinchillaParametricFit",
            fit2: "ChinchillaParametricFit",
            ratio: float,
            N_min: float = 1e6,
            N_max: float = 1e12,
        ) -> Optional[float]:
            """Find N where fit1 and fit2 have equal loss at D = ratio * N."""
            # Sample loss difference across log-spaced N values
            N_samples = np.logspace(np.log10(N_min), np.log10(N_max), 200)
            D_samples = ratio * N_samples

            L1 = fit1.predict_loss(N_samples, D_samples)
            L2 = fit2.predict_loss(N_samples, D_samples)
            diff = L1 - L2

            # Look for sign changes (crossover points)
            sign_changes = np.where(np.diff(np.sign(diff)))[0]

            if len(sign_changes) == 0:
                return None

            # Return the first crossover point (linear interpolation)
            idx = sign_changes[0]
            # Interpolate to find more precise crossover
            N_cross = N_samples[idx] + (N_samples[idx + 1] - N_samples[idx]) * (
                -diff[idx] / (diff[idx + 1] - diff[idx])
            )
            return float(N_cross)

        crossover_found = False
        for name, fit in fits.items():
            if name == baseline_name:
                continue

            # Check if there's a potential crossover (different tradeoffs)
            # Better coefficients but worse exponents, or vice versa
            coeff_better = (fit.A < baseline.A * 0.95) or (fit.B < baseline.B * 0.95)
            coeff_worse = (fit.A > baseline.A * 1.05) or (fit.B > baseline.B * 1.05)
            exp_better = (fit.alpha > baseline.alpha * 1.02) or (fit.beta > baseline.beta * 1.02)
            exp_worse = (fit.alpha < baseline.alpha * 0.98) or (fit.beta < baseline.beta * 0.98)

            potential_crossover = (coeff_better and exp_worse) or (coeff_worse and exp_better)

            if potential_crossover:
                crossover_N = find_crossover_N(baseline, fit, tokens_per_param_ratio)

                if crossover_N is not None:
                    crossover_found = True
                    crossover_D = crossover_N * tokens_per_param_ratio
                    crossover_C = 6 * crossover_N * crossover_D  # Approximate FLOPs

                    # Determine which model wins at small vs large scale
                    small_N, small_D = 100e6, 100e6 * tokens_per_param_ratio
                    L_baseline_small = baseline.predict_loss(
                        np.array([small_N]), np.array([small_D])
                    )[0]
                    L_fit_small = fit.predict_loss(np.array([small_N]), np.array([small_D]))[0]

                    if L_fit_small < L_baseline_small:
                        winner_small, winner_large = name, baseline_name
                    else:
                        winner_small, winner_large = baseline_name, name

                    # Format compute in petaFLOPs
                    crossover_pflops = crossover_C / 1e15

                    lines.append(f"  {name} vs {baseline_name}:")
                    lines.append(
                        f"    Crossover at N≈{format_count(int(crossover_N))}, "
                        f"D≈{format_token_count(int(crossover_D))} (~{crossover_pflops:.1e} PFLOPs)"
                    )
                    lines.append(f"    → {winner_small} wins below, {winner_large} wins above")
                    lines.append("")

        if not crossover_found:
            lines.append("  No significant crossovers detected (models maintain relative ordering)")

        # ──────────────────────────────────────────────────────────────────────
        # Summary & Recommendations
        # ──────────────────────────────────────────────────────────────────────
        lines.append("\nSUMMARY & RECOMMENDATIONS")
        lines.append("=" * box_width)

        # Best for large scale (highest combined scaling)
        best_scaling_name = ranked_combined[0][0]
        best_scaling_fit = fits[best_scaling_name]
        lines.append(
            f"  📈 Best for Scale:    {best_scaling_name} "
            f"(α+β={best_scaling_fit.alpha + best_scaling_fit.beta:.4f})"
        )

        # Best entropy floor
        best_E_name = ranked_E[0][0]
        best_E_fit = fits[best_E_name]
        lines.append(f"  🎯 Lowest E:          {best_E_name} (E={best_E_fit.E:.4f})")

        # Best coefficients (most efficient at small scale)
        best_A_name = ranked_A[0][0]
        best_A_fit = fits[best_A_name]
        best_B_name = ranked_B[0][0]
        best_B_fit = fits[best_B_name]
        lines.append(f"  ⚡ Best Param Coeff:  {best_A_name} (A={best_A_fit.A:.2f})")
        lines.append(f"  ⚡ Best Data Coeff:   {best_B_name} (B={best_B_fit.B:.2f})")

        # Warnings
        lines.append("")

        # Check for models with poor fit quality
        poor_fits = [
            (name, fit) for name, fit in fits.items() if fit.r_squared and fit.r_squared < 0.95
        ]
        if poor_fits:
            lines.append("  ⚠ Low R² Warning: Some fits have R² < 0.95:")
            for name, fit in poor_fits:
                lines.append(f"    • {name}: R²={fit.r_squared:.4f}")

        # Check for crossover behavior (including A/B vs α/β tradeoffs)
        crossovers = []
        for name, fit in fits.items():
            if name == baseline_name:
                continue
            # Check if better coefficients (A, B) but worse exponents (α, β)
            better_coeffs = (fit.A < baseline.A * 0.95) or (fit.B < baseline.B * 0.95)
            worse_exponents = (fit.alpha < baseline.alpha * 0.98) or (
                fit.beta < baseline.beta * 0.98
            )
            if better_coeffs and worse_exponents:
                crossovers.append((name, "better A/B but worse α/β → wins small, loses large"))
            # Check if worse coefficients but better exponents
            worse_coeffs = (fit.A > baseline.A * 1.05) or (fit.B > baseline.B * 1.05)
            better_exponents = (fit.alpha > baseline.alpha * 1.02) or (
                fit.beta > baseline.beta * 1.02
            )
            if worse_coeffs and better_exponents:
                crossovers.append((name, "worse A/B but better α/β → loses small, wins large"))

        if crossovers:
            lines.append("  ⚠ Crossover Risk: Models may trade off at different scales:")
            for name, reason in crossovers:
                lines.append(f"    • {name}: {reason}")

        lines.append("=" * box_width)

        return "\n".join(lines)


@dataclass
class ChinchillaParametricBootstrapFit:
    """
    Bootstrap ensemble of ChinchillaParametricFit models for uncertainty quantification.

    This class enables principled handling of noisy data with multiple observations per (N, D).
    By fitting scaling laws to bootstrap-resampled data, we obtain:
    - A distribution of parameter estimates (E, A, α, B, β)
    - Confidence intervals on loss predictions
    - Robust point estimates (median parameters)

    The bootstrap approach is particularly valuable when:
    - Multiple training runs exist at similar (N, D) scales
    - There is noise from hyperparameter variation, random seeds, or training instabilities
    - Uncertainty quantification is needed for downstream decisions

    Example:
        >>> # Fit with 100 bootstrap samples
        >>> boot_fit = ChinchillaParametricBootstrapFit.fit(N, D, loss, n_bootstrap=100)
        >>> # Get 90% confidence interval on predictions
        >>> mean, lower, upper = boot_fit.predict_loss_interval(N_new, D_new, confidence=0.90)
        >>> # Access parameter distributions
        >>> print(f"α = {boot_fit.alpha_mean:.3f} ± {boot_fit.alpha_std:.3f}")
    """

    fits: list[ChinchillaParametricFit]
    """List of bootstrap-fitted scaling laws."""

    point_estimate: Optional[ChinchillaParametricFit] = None
    """Single fit on full data (used as reference and for warm-starting bootstrap fits)."""

    # Original data used for fitting
    _N: Optional[np.ndarray] = None
    """Parameter counts used in fitting."""
    _D: Optional[np.ndarray] = None
    """Token counts used in fitting."""
    _loss: Optional[np.ndarray] = None
    """Loss values used in fitting."""

    @property
    def n_bootstrap(self) -> int:
        """Number of bootstrap samples."""
        return len(self.fits)

    # ──────────────────────────────────────────────────────────────────────────
    # Parameter Distributions
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def E_distribution(self) -> np.ndarray:
        """Distribution of entropy floor (E) across bootstrap samples."""
        return np.array([f.E for f in self.fits])

    @property
    def A_distribution(self) -> np.ndarray:
        """Distribution of parameter coefficient (A) across bootstrap samples."""
        return np.array([f.A for f in self.fits])

    @property
    def alpha_distribution(self) -> np.ndarray:
        """Distribution of parameter exponent (α) across bootstrap samples."""
        return np.array([f.alpha for f in self.fits])

    @property
    def B_distribution(self) -> np.ndarray:
        """Distribution of data coefficient (B) across bootstrap samples."""
        return np.array([f.B for f in self.fits])

    @property
    def beta_distribution(self) -> np.ndarray:
        """Distribution of data exponent (β) across bootstrap samples."""
        return np.array([f.beta for f in self.fits])

    @property
    def r_squared_distribution(self) -> np.ndarray:
        """Distribution of R² across bootstrap samples."""
        return np.array([f.r_squared if f.r_squared else np.nan for f in self.fits])

    # ──────────────────────────────────────────────────────────────────────────
    # Summary Statistics
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def E_mean(self) -> float:
        """Mean entropy floor across bootstrap samples."""
        return float(np.mean(self.E_distribution))

    @property
    def E_std(self) -> float:
        """Standard deviation of entropy floor across bootstrap samples."""
        return float(np.std(self.E_distribution))

    @property
    def A_mean(self) -> float:
        """Mean parameter coefficient across bootstrap samples."""
        return float(np.mean(self.A_distribution))

    @property
    def A_std(self) -> float:
        """Standard deviation of parameter coefficient across bootstrap samples."""
        return float(np.std(self.A_distribution))

    @property
    def alpha_mean(self) -> float:
        """Mean parameter exponent across bootstrap samples."""
        return float(np.mean(self.alpha_distribution))

    @property
    def alpha_std(self) -> float:
        """Standard deviation of parameter exponent across bootstrap samples."""
        return float(np.std(self.alpha_distribution))

    @property
    def B_mean(self) -> float:
        """Mean data coefficient across bootstrap samples."""
        return float(np.mean(self.B_distribution))

    @property
    def B_std(self) -> float:
        """Standard deviation of data coefficient across bootstrap samples."""
        return float(np.std(self.B_distribution))

    @property
    def beta_mean(self) -> float:
        """Mean data exponent across bootstrap samples."""
        return float(np.mean(self.beta_distribution))

    @property
    def beta_std(self) -> float:
        """Standard deviation of data exponent across bootstrap samples."""
        return float(np.std(self.beta_distribution))

    def parameter_summary(self) -> dict[str, dict[str, float]]:
        """
        Get comprehensive summary statistics for all parameters.

        Returns:
            Dictionary mapping parameter names to their summary statistics:
            - mean, std: Central tendency and spread
            - p5, p25, p50, p75, p95: Percentiles for understanding the distribution
        """

        def summarize(arr: np.ndarray) -> dict[str, float]:
            return {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "p5": float(np.percentile(arr, 5)),
                "p25": float(np.percentile(arr, 25)),
                "p50": float(np.percentile(arr, 50)),
                "p75": float(np.percentile(arr, 75)),
                "p95": float(np.percentile(arr, 95)),
            }

        return {
            "E": summarize(self.E_distribution),
            "A": summarize(self.A_distribution),
            "alpha": summarize(self.alpha_distribution),
            "B": summarize(self.B_distribution),
            "beta": summarize(self.beta_distribution),
            "r_squared": summarize(self.r_squared_distribution),
        }

    def parameter_interval(
        self, param: str, confidence: float = 0.90
    ) -> tuple[float, float, float]:
        """
        Get confidence interval for a specific parameter.

        Args:
            param: Parameter name ('E', 'A', 'alpha', 'B', 'beta')
            confidence: Confidence level (default 0.90 for 90% CI)

        Returns:
            Tuple of (median, lower_bound, upper_bound)
        """
        dist_map = {
            "E": self.E_distribution,
            "A": self.A_distribution,
            "alpha": self.alpha_distribution,
            "B": self.B_distribution,
            "beta": self.beta_distribution,
        }
        if param not in dist_map:
            raise ValueError(f"Unknown parameter: {param}. Expected one of {list(dist_map.keys())}")

        dist = dist_map[param]
        lower_pct = (1 - confidence) / 2 * 100
        upper_pct = (1 + confidence) / 2 * 100

        return (
            float(np.median(dist)),
            float(np.percentile(dist, lower_pct)),
            float(np.percentile(dist, upper_pct)),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Prediction Methods
    # ──────────────────────────────────────────────────────────────────────────

    def predict_loss(self, N: ArrayLike, D: ArrayLike) -> np.ndarray:
        """
        Predict mean loss across bootstrap samples.

        This is the expected loss under the bootstrap distribution of parameters.

        Args:
            N: Parameter counts
            D: Token counts

        Returns:
            Mean predicted loss at each (N, D) point.
        """
        N = np.asarray(N)
        D = np.asarray(D)
        predictions = np.array([f.predict_loss(N, D) for f in self.fits])
        return np.mean(predictions, axis=0)

    def predict_loss_distribution(self, N: ArrayLike, D: ArrayLike) -> np.ndarray:
        """
        Get full distribution of loss predictions across bootstrap samples.

        This enables custom analysis of prediction uncertainty beyond simple
        confidence intervals.

        Args:
            N: Parameter counts
            D: Token counts

        Returns:
            Array of shape (n_bootstrap, len(N)) with all predictions.
        """
        N = np.asarray(N)
        D = np.asarray(D)
        return np.array([f.predict_loss(N, D) for f in self.fits])

    def predict_loss_interval(
        self,
        N: ArrayLike,
        D: ArrayLike,
        confidence: float = 0.90,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict loss with confidence intervals.

        This is the primary method for uncertainty-aware predictions. The intervals
        reflect uncertainty from the training data, not irreducible noise.

        Args:
            N: Parameter counts
            D: Token counts
            confidence: Confidence level (default 0.90 for 90% CI)

        Returns:
            Tuple of (mean, lower_bound, upper_bound) arrays.
            - mean: Expected loss under bootstrap distribution
            - lower_bound: Lower percentile of predictions
            - upper_bound: Upper percentile of predictions
        """
        predictions = self.predict_loss_distribution(N, D)
        mean = np.mean(predictions, axis=0)
        lower_pct = (1 - confidence) / 2 * 100
        upper_pct = (1 + confidence) / 2 * 100
        lower = np.percentile(predictions, lower_pct, axis=0)
        upper = np.percentile(predictions, upper_pct, axis=0)
        return mean, lower, upper

    def predict_loss_std(self, N: ArrayLike, D: ArrayLike) -> np.ndarray:
        """
        Get standard deviation of loss predictions across bootstrap samples.

        This is useful for understanding prediction uncertainty at specific scales.

        Args:
            N: Parameter counts
            D: Token counts

        Returns:
            Standard deviation of predicted loss at each (N, D) point.
        """
        predictions = self.predict_loss_distribution(N, D)
        return np.std(predictions, axis=0)

    # ──────────────────────────────────────────────────────────────────────────
    # Fitting
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def fit(
        cls,
        N: ArrayLike,
        D: ArrayLike,
        loss: ArrayLike,
        n_bootstrap: int = 100,
        loss_fn: LossFunctionType = "asymmetric_mae",
        loss_param: Optional[float] = None,
        random_state: Optional[int] = None,
        parallel: bool = True,
        min_success_rate: float = 0.5,
    ) -> "ChinchillaParametricBootstrapFit":
        """
        Fit scaling laws using bootstrap resampling for uncertainty quantification.

        Each bootstrap sample is fitted using the full grid search procedure,
        ensuring each fit is independent and properly optimized. Bootstrap samples
        are processed sequentially, but the grid search within each sample is
        parallelized when parallel=True.

        Args:
            N: Array of parameter counts
            D: Array of token counts
            loss: Array of loss values
            n_bootstrap: Number of bootstrap samples (default 100).
                More samples give more precise uncertainty estimates but take longer.
                Note: Each bootstrap sample runs full grid search optimization.
            loss_fn: Loss function to use:
                - "log_huber": Huber loss on log(L) (Chinchilla paper methodology)
                - "asymmetric_mae": Asymmetric MAE that penalizes overprediction more
            loss_param: Parameter for loss function:
                - For "log_huber": delta parameter (default 1e-3)
                - For "asymmetric_mae": overprediction weight w (default 10.0)
            random_state: Random seed for reproducibility
            parallel: If True, use multiprocessing for grid search within each
                bootstrap sample (recommended for speed)
            min_success_rate: Minimum fraction of bootstrap fits that must succeed
                (default 0.5). Raises ValueError if too many fits fail.

        Returns:
            ChinchillaParametricBootstrapFit with n_bootstrap fitted models.

        Raises:
            ValueError: If insufficient data or too many bootstrap fits fail.

        Example:
            >>> boot_fit = ChinchillaParametricBootstrapFit.fit(N, D, loss, n_bootstrap=100)
            >>> mean, lo, hi = boot_fit.predict_loss_interval([7e9], [5e12], confidence=0.90)
            >>> print(f"7B @ 5T: {mean[0]:.4f} [{lo[0]:.4f}, {hi[0]:.4f}]")
        """
        N = np.asarray(N)
        D = np.asarray(D)
        loss = np.asarray(loss)

        # Clean data
        mask = np.isfinite(N) & np.isfinite(D) & np.isfinite(loss) & (N > 0) & (D > 0) & (loss > 0)
        N_clean, D_clean, L_clean = N[mask], D[mask], loss[mask]

        if len(N_clean) < 5:
            raise ValueError(f"Need at least 5 valid data points, got {len(N_clean)}")

        # Fit point estimate on full data
        point_estimate = ChinchillaParametricFit.fit(
            N_clean, D_clean, L_clean, loss_fn=loss_fn, loss_param=loss_param, parallel=parallel
        )

        # Bootstrap fitting - sequential over samples, parallel within each grid search
        rng = np.random.default_rng(random_state)
        n_samples = len(N_clean)

        fits: list[ChinchillaParametricFit] = []

        for i in tqdm(range(n_bootstrap), desc="Bootstrap fitting"):
            # Resample with replacement
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            N_boot = N_clean[indices]
            D_boot = D_clean[indices]
            L_boot = L_clean[indices]

            try:
                # Full fit with parallel grid search
                fit = ChinchillaParametricFit.fit(
                    N_boot,
                    D_boot,
                    L_boot,
                    loss_fn=loss_fn,
                    loss_param=loss_param,
                    parallel=parallel,
                )
                fits.append(fit)
            except Exception:
                # Skip failed fits (e.g., degenerate bootstrap samples)
                continue

        # Check success rate
        success_rate = len(fits) / n_bootstrap
        if success_rate < min_success_rate:
            raise ValueError(
                f"Too many bootstrap fits failed: {len(fits)}/{n_bootstrap} succeeded "
                f"({success_rate:.1%} < {min_success_rate:.1%} required). "
                "This may indicate poor data quality or ill-conditioning."
            )

        return cls(
            fits=fits,
            point_estimate=point_estimate,
            _N=N_clean,
            _D=D_clean,
            _loss=L_clean,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Reporting
    # ──────────────────────────────────────────────────────────────────────────

    def report(
        self,
        model_sizes: Optional[list[float]] = None,
        token_counts: Optional[list[float]] = None,
        confidence: float = 0.90,
    ) -> str:
        """
        Generate a comprehensive report with uncertainty estimates.

        Args:
            model_sizes: Model sizes to show predictions for (default: 190M, 1B, 7B, 70B)
            token_counts: Token counts to show predictions for (default: 100B, 1T, 5T, 15T)
            confidence: Confidence level for intervals (default 0.90)

        Returns:
            Formatted report string.
        """
        if model_sizes is None:
            model_sizes = [190e6, 1e9, 7e9, 70e9]
        if token_counts is None:
            token_counts = [100e9, 1e12, 5e12, 15e12]

        summary = self.parameter_summary()
        ci_pct = int(confidence * 100)

        lines = []
        lines.append("=" * 80)
        lines.append("CHINCHILLA PARAMETRIC SCALING LAW (BOOTSTRAP ENSEMBLE)")
        lines.append(f"Based on {self.n_bootstrap} bootstrap samples")
        lines.append("=" * 80)

        # Point estimate (if available)
        if self.point_estimate:
            lines.append("\nPOINT ESTIMATE (Full Data)")
            lines.append("-" * 50)
            r2_str = (
                f" (R²={self.point_estimate.r_squared:.4f})"
                if self.point_estimate.r_squared
                else ""
            )
            lines.append(
                f"  L(N, D) = {self.point_estimate.E:.4f} + "
                f"{self.point_estimate.A:.4f}/N^{self.point_estimate.alpha:.4f} + "
                f"{self.point_estimate.B:.4f}/D^{self.point_estimate.beta:.4f}{r2_str}"
            )

        # Parameter distributions
        lines.append(f"\nPARAMETER ESTIMATES (median [{ci_pct}% CI])")
        lines.append("-" * 50)
        lines.append(
            f"  {'Parameter':<12} {'Median':>10} {'Mean':>10} {'Std':>10} "
            f"{'[{0}% CI]'.format(ci_pct):>20}"
        )
        lines.append("-" * 50)

        for param_name in ["E", "A", "alpha", "B", "beta"]:
            stats = summary[param_name]
            lower_pct = (1 - confidence) / 2 * 100
            upper_pct = (1 + confidence) / 2 * 100
            lower = float(np.percentile(getattr(self, f"{param_name}_distribution"), lower_pct))
            upper = float(np.percentile(getattr(self, f"{param_name}_distribution"), upper_pct))

            # Format based on magnitude
            if param_name in ["E", "alpha", "beta"]:
                lines.append(
                    f"  {param_name:<12} {stats['p50']:>10.4f} {stats['mean']:>10.4f} "
                    f"{stats['std']:>10.4f} [{lower:.4f}, {upper:.4f}]"
                )
            else:
                lines.append(
                    f"  {param_name:<12} {stats['p50']:>10.2f} {stats['mean']:>10.2f} "
                    f"{stats['std']:>10.2f} [{lower:.2f}, {upper:.2f}]"
                )

        # Derived quantities
        lines.append("\nDERIVED QUANTITIES")
        lines.append("-" * 50)
        a_opt = self.beta_distribution / (self.alpha_distribution + self.beta_distribution)
        b_opt = self.alpha_distribution / (self.alpha_distribution + self.beta_distribution)
        lines.append(
            f"  a_opt (N∝C^a):  {np.median(a_opt):.3f} [{np.percentile(a_opt, 5):.3f}, "
            f"{np.percentile(a_opt, 95):.3f}]"
        )
        lines.append(
            f"  b_opt (D∝C^b):  {np.median(b_opt):.3f} [{np.percentile(b_opt, 5):.3f}, "
            f"{np.percentile(b_opt, 95):.3f}]"
        )

        # Prediction table with uncertainties
        lines.append(f"\nPREDICTED LOSS WITH {ci_pct}% CONFIDENCE INTERVALS")
        lines.append("-" * 80)

        # Header row
        n_d_label = "N \\ D"
        header = f"  {n_d_label:<10}"
        for D in token_counts:
            header += f"  {format_token_count(int(D)):>18}"
        lines.append(header)
        lines.append("-" * 80)

        # Data rows
        for N in model_sizes:
            row = f"  {format_count(int(N)):>10}"
            for D in token_counts:
                mean, lower, upper = self.predict_loss_interval(
                    np.array([N]), np.array([D]), confidence=confidence
                )
                # Format as "mean [lo, hi]"
                row += f"  {mean[0]:.4f} [{lower[0]:.4f},{upper[0]:.4f}]"
            lines.append(row)

        # Uncertainty assessment
        lines.append("\nUNCERTAINTY ASSESSMENT")
        lines.append("-" * 50)

        # Check coefficient of variation for parameters
        cv_alpha = self.alpha_std / self.alpha_mean * 100
        cv_beta = self.beta_std / self.beta_mean * 100

        if cv_alpha < 5 and cv_beta < 5:
            lines.append("  ✓ Low parameter uncertainty (CV < 5% for α, β)")
            lines.append("    → Predictions are well-constrained")
        elif cv_alpha < 15 and cv_beta < 15:
            lines.append("  ⚠ Moderate parameter uncertainty (5% < CV < 15%)")
            lines.append("    → Consider adding more data or reducing noise")
        else:
            lines.append("  ✗ High parameter uncertainty (CV > 15%)")
            lines.append("    → Predictions have wide confidence intervals")
            lines.append("    → Add more training runs or filter noisy observations")

        # Check for multimodality (rough heuristic)
        alpha_iqr = summary["alpha"]["p75"] - summary["alpha"]["p25"]
        alpha_range = summary["alpha"]["p95"] - summary["alpha"]["p5"]
        if alpha_range > 3 * alpha_iqr:
            lines.append("  ⚠ Possible multimodality in α distribution")
            lines.append("    → Examine bootstrap distribution for multiple modes")

        lines.append("=" * 80)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ChinchillaParametricBootstrapFit(n_bootstrap={self.n_bootstrap}, "
            f"α={self.alpha_mean:.3f}±{self.alpha_std:.3f}, "
            f"β={self.beta_mean:.3f}±{self.beta_std:.3f})"
        )


@dataclass
class ChinchillaParametricWildClusterBootstrapFit:
    """
    Wild cluster bootstrap for scaling law uncertainty with hierarchical data.

    This addresses two key issues with naive bootstrap on scaling law data:

    1. **Non-IID data**: Observations are clustered by (ladder, model_size). Points within
       a cluster share hyperparameters, architecture, and training trajectory. Naive bootstrap
       breaks this correlation structure.

    2. **Heteroscedasticity**: Variance typically decreases with model scale. Wild bootstrap
       handles non-constant variance by perturbing residuals rather than resampling.

    The wild cluster bootstrap works by:
    1. Fitting a point estimate to get residuals
    2. For each bootstrap iteration:
       - Assign a random weight (+1 or -1) to each *cluster*
       - All observations in a cluster get the same weight
       - Multiply residuals by cluster weights
       - Add perturbed residuals to fitted values
       - Refit the scaling law

    This preserves within-cluster correlation while properly estimating uncertainty.

    References:
        - Cameron, Gelbach, Miller (2008): Bootstrap-Based Improvements for Inference
          with Clustered Errors
        - Webb (2023): Reworking Wild Bootstrap Based Inference for Clustered Errors

    Example:
        >>> # Create cluster labels from ladder and model size
        >>> clusters = [f"{ladder}_{size}" for ladder, size in zip(ladders, sizes)]
        >>> fit = ChinchillaParametricWildClusterBootstrapFit.fit(
        ...     N, D, loss, clusters, n_bootstrap=200
        ... )
        >>> mean, lo, hi = fit.predict_loss_interval([7e9], [5e12])
    """

    fits: list[ChinchillaParametricFit]
    """List of wild bootstrap-fitted scaling laws."""

    point_estimate: ChinchillaParametricFit
    """Single fit on full data (used as base for residual perturbation)."""

    cluster_labels: np.ndarray
    """Cluster labels for each observation."""

    n_clusters: int
    """Number of unique clusters."""

    # Original data used for fitting
    _N: Optional[np.ndarray] = None
    """Parameter counts used in fitting."""
    _D: Optional[np.ndarray] = None
    """Token counts used in fitting."""
    _loss: Optional[np.ndarray] = None
    """Loss values used in fitting."""
    _residuals: Optional[np.ndarray] = None
    """Residuals from point estimate (loss - predicted)."""

    @property
    def n_bootstrap(self) -> int:
        """Number of bootstrap samples."""
        return len(self.fits)

    # ──────────────────────────────────────────────────────────────────────────
    # Parameter Distributions (same interface as ChinchillaParametricBootstrapFit)
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def E_distribution(self) -> np.ndarray:
        """Distribution of entropy floor (E) across bootstrap samples."""
        return np.array([f.E for f in self.fits])

    @property
    def A_distribution(self) -> np.ndarray:
        """Distribution of parameter coefficient (A) across bootstrap samples."""
        return np.array([f.A for f in self.fits])

    @property
    def alpha_distribution(self) -> np.ndarray:
        """Distribution of parameter exponent (α) across bootstrap samples."""
        return np.array([f.alpha for f in self.fits])

    @property
    def B_distribution(self) -> np.ndarray:
        """Distribution of data coefficient (B) across bootstrap samples."""
        return np.array([f.B for f in self.fits])

    @property
    def beta_distribution(self) -> np.ndarray:
        """Distribution of data exponent (β) across bootstrap samples."""
        return np.array([f.beta for f in self.fits])

    @property
    def r_squared_distribution(self) -> np.ndarray:
        """Distribution of R² across bootstrap samples."""
        return np.array([f.r_squared if f.r_squared else np.nan for f in self.fits])

    # ──────────────────────────────────────────────────────────────────────────
    # Summary Statistics
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def E_mean(self) -> float:
        """Mean entropy floor across bootstrap samples."""
        return float(np.mean(self.E_distribution))

    @property
    def E_std(self) -> float:
        """Standard deviation of entropy floor across bootstrap samples."""
        return float(np.std(self.E_distribution))

    @property
    def A_mean(self) -> float:
        """Mean parameter coefficient across bootstrap samples."""
        return float(np.mean(self.A_distribution))

    @property
    def A_std(self) -> float:
        """Standard deviation of parameter coefficient across bootstrap samples."""
        return float(np.std(self.A_distribution))

    @property
    def alpha_mean(self) -> float:
        """Mean parameter exponent across bootstrap samples."""
        return float(np.mean(self.alpha_distribution))

    @property
    def alpha_std(self) -> float:
        """Standard deviation of parameter exponent across bootstrap samples."""
        return float(np.std(self.alpha_distribution))

    @property
    def B_mean(self) -> float:
        """Mean data coefficient across bootstrap samples."""
        return float(np.mean(self.B_distribution))

    @property
    def B_std(self) -> float:
        """Standard deviation of data coefficient across bootstrap samples."""
        return float(np.std(self.B_distribution))

    @property
    def beta_mean(self) -> float:
        """Mean data exponent across bootstrap samples."""
        return float(np.mean(self.beta_distribution))

    @property
    def beta_std(self) -> float:
        """Standard deviation of data exponent across bootstrap samples."""
        return float(np.std(self.beta_distribution))

    def parameter_summary(self) -> dict[str, dict[str, float]]:
        """Get comprehensive summary statistics for all parameters."""

        def summarize(arr: np.ndarray) -> dict[str, float]:
            return {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "p5": float(np.percentile(arr, 5)),
                "p25": float(np.percentile(arr, 25)),
                "p50": float(np.percentile(arr, 50)),
                "p75": float(np.percentile(arr, 75)),
                "p95": float(np.percentile(arr, 95)),
            }

        return {
            "E": summarize(self.E_distribution),
            "A": summarize(self.A_distribution),
            "alpha": summarize(self.alpha_distribution),
            "B": summarize(self.B_distribution),
            "beta": summarize(self.beta_distribution),
            "r_squared": summarize(self.r_squared_distribution),
        }

    def parameter_interval(
        self, param: str, confidence: float = 0.90
    ) -> tuple[float, float, float]:
        """Get confidence interval for a specific parameter."""
        dist_map = {
            "E": self.E_distribution,
            "A": self.A_distribution,
            "alpha": self.alpha_distribution,
            "B": self.B_distribution,
            "beta": self.beta_distribution,
        }
        if param not in dist_map:
            raise ValueError(f"Unknown parameter: {param}. Expected one of {list(dist_map.keys())}")

        dist = dist_map[param]
        lower_pct = (1 - confidence) / 2 * 100
        upper_pct = (1 + confidence) / 2 * 100

        return (
            float(np.median(dist)),
            float(np.percentile(dist, lower_pct)),
            float(np.percentile(dist, upper_pct)),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Prediction Methods
    # ──────────────────────────────────────────────────────────────────────────

    def predict_loss(self, N: ArrayLike, D: ArrayLike) -> np.ndarray:
        """Predict mean loss across bootstrap samples."""
        N = np.asarray(N)
        D = np.asarray(D)
        predictions = np.array([f.predict_loss(N, D) for f in self.fits])
        return np.mean(predictions, axis=0)

    def predict_loss_distribution(self, N: ArrayLike, D: ArrayLike) -> np.ndarray:
        """Get full distribution of loss predictions across bootstrap samples."""
        N = np.asarray(N)
        D = np.asarray(D)
        return np.array([f.predict_loss(N, D) for f in self.fits])

    def predict_loss_interval(
        self,
        N: ArrayLike,
        D: ArrayLike,
        confidence: float = 0.90,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict loss with confidence intervals."""
        predictions = self.predict_loss_distribution(N, D)
        mean = np.mean(predictions, axis=0)
        lower_pct = (1 - confidence) / 2 * 100
        upper_pct = (1 + confidence) / 2 * 100
        lower = np.percentile(predictions, lower_pct, axis=0)
        upper = np.percentile(predictions, upper_pct, axis=0)
        return mean, lower, upper

    def predict_loss_std(self, N: ArrayLike, D: ArrayLike) -> np.ndarray:
        """Get standard deviation of loss predictions across bootstrap samples."""
        predictions = self.predict_loss_distribution(N, D)
        return np.std(predictions, axis=0)

    # ──────────────────────────────────────────────────────────────────────────
    # Wild Bootstrap Weight Distributions
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _rademacher_weights(rng: np.random.Generator, n: int) -> np.ndarray:
        """Rademacher distribution: {-1, +1} with equal probability."""
        return rng.choice([-1.0, 1.0], size=n)

    @staticmethod
    def _webb_six_point_weights(rng: np.random.Generator, n: int) -> np.ndarray:
        """
        Webb's six-point distribution for few clusters.

        Better finite-sample properties when number of clusters is small (< 10).
        Values: ±sqrt(3/2), ±sqrt(1/2), ±1 with equal probability.
        """
        sqrt_3_2 = np.sqrt(1.5)
        sqrt_1_2 = np.sqrt(0.5)
        values = np.array([-sqrt_3_2, -1.0, -sqrt_1_2, sqrt_1_2, 1.0, sqrt_3_2])
        return rng.choice(values, size=n)

    # ──────────────────────────────────────────────────────────────────────────
    # Fitting
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def fit(
        cls,
        N: ArrayLike,
        D: ArrayLike,
        loss: ArrayLike,
        clusters: ArrayLike,
        n_bootstrap: int = 200,
        loss_fn: LossFunctionType = "asymmetric_mae",
        loss_param: Optional[float] = None,
        random_state: Optional[int] = None,
        parallel: bool = True,
        weight_distribution: Literal["auto", "rademacher", "webb"] = "auto",
        weights: Optional[ArrayLike] = None,
    ) -> "ChinchillaParametricWildClusterBootstrapFit":
        """
        Fit scaling law with wild cluster bootstrap for uncertainty quantification.

        This method is appropriate when:
        - Data has hierarchical/clustered structure (e.g., by ladder and model size)
        - Observations within clusters are correlated
        - Variance may differ across scales (heteroscedasticity)

        The wild cluster bootstrap preserves within-cluster correlation by assigning
        the same random weight to all observations in each cluster.

        Args:
            N: Array of parameter counts
            D: Array of token counts
            loss: Array of loss values
            clusters: Array of cluster labels (e.g., "ladder_modelsize" strings).
                All observations with the same label are treated as a cluster.
            n_bootstrap: Number of bootstrap iterations (default 200).
                More samples give more precise estimates. Wild bootstrap is faster
                than naive bootstrap since we don't rerun full grid search.
            loss_fn: Loss function for fitting ("log_huber" or "asymmetric_mae")
            loss_param: Parameter for loss function
            random_state: Random seed for reproducibility
            parallel: If True, use parallel grid search for point estimate
            weight_distribution: Weight distribution for wild bootstrap:
                - "auto": Use Webb for < 10 clusters, Rademacher otherwise
                - "rademacher": Always use {-1, +1} weights
                - "webb": Always use Webb's 6-point distribution
            weights: Optional weights for each observation. Higher weights give more
                importance to those points during fitting. Common choices:
                - np.sqrt(6 * N * D): Weight by sqrt(compute) to emphasize large-scale points
                - N * D: Weight by compute (stronger emphasis on large scale)
                - None: Uniform weights (default)

        Returns:
            ChinchillaParametricWildClusterBootstrapFit with fitted models.

        Example:
            >>> # Create cluster labels from your data
            >>> clusters = [f"{row['ladder']}_{row['size']}" for _, row in df.iterrows()]
            >>> fit = ChinchillaParametricWildClusterBootstrapFit.fit(
            ...     N=df["num_params"].values,
            ...     D=df["throughput/total tokens"].values,
            ...     loss=df["train/CE loss"].values,
            ...     clusters=clusters,
            ...     n_bootstrap=200,
            ...     weights=np.sqrt(6 * df["num_params"].values * df["throughput/total tokens"].values),
            ... )
            >>> print(fit.report())
        """
        N = np.asarray(N)
        D = np.asarray(D)
        loss = np.asarray(loss)
        clusters = np.asarray(clusters)

        # Asymmetric MAE creates systematic bias - not appropriate for bootstrap uncertainty
        if loss_fn == "asymmetric_mae":
            raise ValueError(
                "asymmetric_mae loss is not recommended for bootstrap uncertainty estimation. "
                "This loss function penalizes overprediction more heavily, creating systematic "
                "downward bias in predictions. Use loss_fn='log_huber' for unbiased uncertainty "
                "estimates. If you specifically need asymmetric loss, use ChinchillaParametricFit "
                "directly for point estimation."
            )

        # Clean data
        mask = np.isfinite(N) & np.isfinite(D) & np.isfinite(loss) & (N > 0) & (D > 0) & (loss > 0)
        N_clean = N[mask]
        D_clean = D[mask]
        L_clean = loss[mask]
        clusters_clean = clusters[mask]

        # Apply mask to weights if provided
        weights_clean: Optional[np.ndarray] = None
        if weights is not None:
            weights_arr = np.asarray(weights)
            weights_masked = weights_arr[mask]
            # Normalize weights to sum to len(N_clean) to preserve scale of loss function
            weights_clean = weights_masked * len(N_clean) / weights_masked.sum()
            print(
                f"Using observation weights (range: {weights_masked.min():.2f} to {weights_masked.max():.2f})"
            )

        if len(N_clean) < 5:
            raise ValueError(f"Need at least 5 valid data points, got {len(N_clean)}")

        # Get unique clusters and create mapping
        unique_clusters = np.unique(clusters_clean)
        n_clusters = len(unique_clusters)
        cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
        cluster_indices = np.array([cluster_to_idx[c] for c in clusters_clean])

        print(f"Wild cluster bootstrap with {n_clusters} clusters:")
        for c in unique_clusters:
            n_obs = np.sum(clusters_clean == c)
            print(f"  {c}: {n_obs} observations")

        # Select weight distribution
        if weight_distribution == "auto":
            if n_clusters < 10:
                print("Using Webb's 6-point weights (recommended for < 10 clusters)")
                weight_fn = cls._webb_six_point_weights
            else:
                print("Using Rademacher weights")
                weight_fn = cls._rademacher_weights
        elif weight_distribution == "webb":
            weight_fn = cls._webb_six_point_weights
        else:
            weight_fn = cls._rademacher_weights

        # Fit point estimate on full data
        print("Fitting point estimate...")
        point_estimate = ChinchillaParametricFit.fit(
            N_clean,
            D_clean,
            L_clean,
            loss_fn=loss_fn,
            loss_param=loss_param,
            parallel=parallel,
            weights=weights_clean,
        )

        # Compute residuals
        L_predicted = point_estimate.predict_loss(N_clean, D_clean)
        residuals = L_clean - L_predicted

        # Wild cluster bootstrap
        rng = np.random.default_rng(random_state)
        fits: list[ChinchillaParametricFit] = []

        for _ in tqdm(range(n_bootstrap), desc="Wild cluster bootstrap"):
            # Generate one weight per cluster
            cluster_weights = weight_fn(rng, n_clusters)

            # Map cluster weights to observations
            obs_weights = cluster_weights[cluster_indices]

            # Perturb residuals and create bootstrap loss values
            L_boot = L_predicted + residuals * obs_weights

            # Ensure positive losses (can happen with large negative weights)
            L_boot = np.maximum(L_boot, 1e-6)

            try:
                # Fit on perturbed data
                # Note: We use the same N, D - only loss values change
                fit = ChinchillaParametricFit.fit(
                    N_clean,
                    D_clean,
                    L_boot,
                    loss_fn=loss_fn,
                    loss_param=loss_param,
                    parallel=parallel,
                    weights=weights_clean,
                )
                fits.append(fit)
            except Exception as e:
                # Skip failed fits
                print(f"Warning: Bootstrap iteration failed: {e}")
                continue

        if len(fits) < n_bootstrap * 0.5:
            raise ValueError(
                f"Too many bootstrap fits failed: {len(fits)}/{n_bootstrap} succeeded. "
                "This may indicate ill-conditioning or extreme residuals."
            )

        return cls(
            fits=fits,
            point_estimate=point_estimate,
            cluster_labels=clusters_clean,
            n_clusters=n_clusters,
            _N=N_clean,
            _D=D_clean,
            _loss=L_clean,
            _residuals=residuals,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Reporting
    # ──────────────────────────────────────────────────────────────────────────

    def report(
        self,
        model_sizes: Optional[list[float]] = None,
        token_counts: Optional[list[float]] = None,
        confidence: float = 0.90,
    ) -> str:
        """Generate a comprehensive report with uncertainty estimates."""
        if model_sizes is None:
            model_sizes = [190e6, 1e9, 7e9, 70e9]
        if token_counts is None:
            token_counts = [100e9, 1e12, 5e12, 15e12]

        summary = self.parameter_summary()
        ci_pct = int(confidence * 100)

        lines = []
        lines.append("=" * 80)
        lines.append("CHINCHILLA SCALING LAW (WILD CLUSTER BOOTSTRAP)")
        lines.append(
            f"Based on {self.n_bootstrap} bootstrap samples across {self.n_clusters} clusters"
        )
        lines.append("=" * 80)

        # Point estimate
        lines.append("\nPOINT ESTIMATE (Full Data)")
        lines.append("-" * 50)
        r2_str = (
            f" (R²={self.point_estimate.r_squared:.4f})" if self.point_estimate.r_squared else ""
        )
        lines.append(
            f"  L(N, D) = {self.point_estimate.E:.4f} + "
            f"{self.point_estimate.A:.4f}/N^{self.point_estimate.alpha:.4f} + "
            f"{self.point_estimate.B:.4f}/D^{self.point_estimate.beta:.4f}{r2_str}"
        )

        # Parameter distributions
        lines.append(f"\nPARAMETER ESTIMATES (median [{ci_pct}% CI])")
        lines.append("-" * 50)
        lines.append(
            f"  {'Parameter':<12} {'Median':>10} {'Mean':>10} {'Std':>10} "
            f"{'[{0}% CI]'.format(ci_pct):>20}"
        )
        lines.append("-" * 50)

        for param_name in ["E", "A", "alpha", "B", "beta"]:
            stats = summary[param_name]
            lower_pct = (1 - confidence) / 2 * 100
            upper_pct = (1 + confidence) / 2 * 100
            lower = float(np.percentile(getattr(self, f"{param_name}_distribution"), lower_pct))
            upper = float(np.percentile(getattr(self, f"{param_name}_distribution"), upper_pct))

            if param_name in ["E", "alpha", "beta"]:
                lines.append(
                    f"  {param_name:<12} {stats['p50']:>10.4f} {stats['mean']:>10.4f} "
                    f"{stats['std']:>10.4f} [{lower:.4f}, {upper:.4f}]"
                )
            else:
                lines.append(
                    f"  {param_name:<12} {stats['p50']:>10.2f} {stats['mean']:>10.2f} "
                    f"{stats['std']:>10.2f} [{lower:.2f}, {upper:.2f}]"
                )

        # Derived quantities
        lines.append("\nDERIVED QUANTITIES")
        lines.append("-" * 50)
        a_opt = self.beta_distribution / (self.alpha_distribution + self.beta_distribution)
        b_opt = self.alpha_distribution / (self.alpha_distribution + self.beta_distribution)
        lines.append(
            f"  a_opt (N∝C^a):  {np.median(a_opt):.3f} [{np.percentile(a_opt, 5):.3f}, "
            f"{np.percentile(a_opt, 95):.3f}]"
        )
        lines.append(
            f"  b_opt (D∝C^b):  {np.median(b_opt):.3f} [{np.percentile(b_opt, 5):.3f}, "
            f"{np.percentile(b_opt, 95):.3f}]"
        )

        # Cluster info
        lines.append("\nCLUSTER INFORMATION")
        lines.append("-" * 50)
        unique_clusters, counts = np.unique(self.cluster_labels, return_counts=True)
        lines.append(f"  Number of clusters: {self.n_clusters}")
        lines.append(
            f"  Observations per cluster: min={counts.min()}, max={counts.max()}, "
            f"mean={counts.mean():.1f}"
        )

        # Prediction table
        lines.append(f"\nPREDICTED LOSS WITH {ci_pct}% CONFIDENCE INTERVALS")
        lines.append("-" * 80)

        n_d_label = "N \\ D"
        header = f"  {n_d_label:<10}"
        for D in token_counts:
            header += f"  {format_token_count(int(D)):>18}"
        lines.append(header)
        lines.append("-" * 80)

        for N in model_sizes:
            row = f"  {format_count(int(N)):>10}"
            for D in token_counts:
                mean, lower, upper = self.predict_loss_interval(
                    np.array([N]), np.array([D]), confidence=confidence
                )
                row += f"  {mean[0]:.4f} [{lower[0]:.4f},{upper[0]:.4f}]"
            lines.append(row)

        # Uncertainty assessment
        lines.append("\nUNCERTAINTY ASSESSMENT")
        lines.append("-" * 50)

        cv_alpha = self.alpha_std / self.alpha_mean * 100
        cv_beta = self.beta_std / self.beta_mean * 100

        if cv_alpha < 5 and cv_beta < 5:
            lines.append("  ✓ Low parameter uncertainty (CV < 5% for α, β)")
            lines.append("    → Predictions are well-constrained")
        elif cv_alpha < 15 and cv_beta < 15:
            lines.append("  ⚠ Moderate parameter uncertainty (5% < CV < 15%)")
            lines.append("    → Consider adding more clusters or data")
        else:
            lines.append("  ✗ High parameter uncertainty (CV > 15%)")
            lines.append("    → Wide confidence intervals")
            lines.append("    → May need more clusters or cleaner data")

        if self.n_clusters < 10:
            lines.append(f"\n  ⚠ Few clusters ({self.n_clusters}): Using Webb's 6-point weights")
            lines.append("    → CIs may be conservative with < 10 clusters")

        lines.append("=" * 80)
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────────
    # Ranking and Comparison Methods
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def rank_fits(
        fits: dict[str, "ChinchillaParametricWildClusterBootstrapFit"],
        N_values: Optional[ArrayLike] = None,
        D_values: Optional[ArrayLike] = None,
        confidence: float = 0.90,
    ) -> str:
        """
        Rank multiple bootstrap fits by predicted loss at various scales.

        For each scale, models are ranked from best (lowest loss) to worst.
        Statistical significance is indicated when the confidence interval
        for the difference between adjacent ranks excludes zero.

        Args:
            fits: Dictionary mapping names to bootstrap fits.
            N_values: Parameter counts to evaluate at. Default: [1B, 7B, 32B, 70B].
            D_values: Token counts to evaluate at. Default: 20*N (20 tokens per param).
                If scalar, uses same D for all N values.
            confidence: Confidence level for significance tests (default 0.90).

        Returns:
            Formatted ranking report string.

        Example:
            >>> report = ChinchillaParametricWildClusterBootstrapFit.rank_fits(
            ...     fits={"baseline": bl, "muon": muon, "gnope": gnope},
            ... )
            >>> print(report)
        """
        if N_values is None:
            N_values = np.array([1e9, 7e9, 32e9, 70e9])
        else:
            N_values = np.asarray(N_values)

        if D_values is None:
            D_values = np.array([100e9, 1e12, 5e12, 10e12])
        else:
            D_values = np.asarray(D_values)
            if D_values.ndim == 0:
                D_values = np.full_like(N_values, float(D_values))

        ci_pct = int(confidence * 100)
        names = list(fits.keys())
        n_fits = len(fits)
        n_scales = len(N_values)

        # Compute predictions for all fits at all scales
        # predictions[fit_name][bootstrap_sample, scale_idx]
        n_samples = min(f.n_bootstrap for f in fits.values())
        predictions: dict[str, np.ndarray] = {}
        for name, fit in fits.items():
            predictions[name] = np.array(
                [fit.fits[i].predict_loss(N_values, D_values) for i in range(n_samples)]
            )

        # Mean predictions
        mean_preds = {name: np.mean(preds, axis=0) for name, preds in predictions.items()}

        lines = []
        lines.append("=" * 100)
        lines.append("BOOTSTRAP FIT RANKINGS")
        lines.append(
            f"Comparing {n_fits} models at {n_scales} scales | {ci_pct}% CI for significance"
        )
        lines.append("=" * 100)

        # Create scale labels
        scale_labels = []
        for N, D in zip(N_values, D_values):
            N_str = format_count(int(N))
            D_str = format_token_count(int(D))
            scale_labels.append(f"{N_str}/{D_str}")

        # Table: rows = models, cols = scales
        lines.append(f"\nPREDICTED LOSS BY SCALE (mean [{ci_pct}% CI])")
        lines.append("-" * 100)

        header = f"{'Model':<25}"
        for label in scale_labels:
            header += f" {label:>17}"
        lines.append(header)
        lines.append("-" * 100)

        for name in sorted(names):
            row = f"{name:<25}"
            preds = predictions[name]
            for i in range(n_scales):
                mean = np.mean(preds[:, i])
                lower = np.percentile(preds[:, i], (1 - confidence) / 2 * 100)
                upper = np.percentile(preds[:, i], (1 + confidence) / 2 * 100)
                row += f" {mean:.4f}[{lower:.4f},{upper:.4f}]"
            lines.append(row)

        # Rankings with significance
        lines.append("\nRANKINGS BY SCALE (Best → Worst)")
        lines.append("-" * 100)
        lines.append("Significance: '>>' means significantly better than next rank")
        lines.append("              '>'  means better but not statistically significant")
        lines.append("-" * 100)

        for scale_idx, label in enumerate(scale_labels):
            # Sort models by mean prediction (ascending = best first)
            sorted_names = sorted(names, key=lambda n: mean_preds[n][scale_idx])

            # Build ranking string with significance markers
            ranking_parts = []
            for rank, name in enumerate(sorted_names):
                if rank < len(sorted_names) - 1:
                    next_name = sorted_names[rank + 1]

                    # Pairwise test: is this model significantly better than next?
                    diff = predictions[next_name][:, scale_idx] - predictions[name][:, scale_idx]
                    lower = np.percentile(diff, (1 - confidence) / 2 * 100)
                    upper = np.percentile(diff, (1 + confidence) / 2 * 100)

                    # Significant if CI for (next - current) excludes 0 on the positive side
                    # i.e., next is significantly worse
                    if lower > 0:
                        sep = " >> "  # Significantly better
                    else:
                        sep = " > "  # Better but not significant

                    ranking_parts.append(f"{name}{sep}")
                else:
                    ranking_parts.append(name)

            lines.append(f"\n  {label}:")
            lines.append(f"    {''.join(ranking_parts)}")

        # Summary: how often each model is ranked #1, #2, etc.
        lines.append("\n" + "=" * 100)
        lines.append("RANK SUMMARY (how often each model places at each rank)")
        lines.append("-" * 100)

        # Count ranks across scales
        rank_counts: dict[str, list[int]] = {name: [0] * n_fits for name in names}
        for scale_idx in range(n_scales):
            sorted_names = sorted(names, key=lambda n: mean_preds[n][scale_idx])
            for rank, name in enumerate(sorted_names):
                rank_counts[name][rank] += 1

        header = f"{'Model':<25}"
        for r in range(min(n_fits, 5)):  # Show top 5 ranks
            header += f" {'#' + str(r + 1):>6}"
        header += "  Avg Rank"
        lines.append(header)
        lines.append("-" * 100)

        # Compute average rank
        avg_ranks = {}
        for name in names:
            avg_rank = sum((r + 1) * count for r, count in enumerate(rank_counts[name])) / n_scales
            avg_ranks[name] = avg_rank

        for name in sorted(names, key=lambda n: avg_ranks[n]):
            row = f"{name:<25}"
            for r in range(min(n_fits, 5)):
                row += f" {rank_counts[name][r]:>6}"
            row += f"  {avg_ranks[name]:>8.2f}"
            lines.append(row)

        # Pairwise significance matrix
        lines.append("\n" + "=" * 100)
        lines.append("PAIRWISE SIGNIFICANCE (at largest scale)")
        lines.append(f"Cell shows: P(row < col) | '**' if {ci_pct}% significant")
        lines.append("-" * 100)

        # Use largest scale for pairwise comparison
        largest_scale = n_scales - 1

        # Header row
        header = f"{'':15}"
        for name in sorted(names):
            header += f" {name[:12]:>12}"
        lines.append(header)

        for name_a in sorted(names):
            row = f"{name_a[:15]:<15}"
            for name_b in sorted(names):
                if name_a == name_b:
                    row += f" {'---':>12}"
                else:
                    # P(A < B) = P(B - A > 0)
                    diff = (
                        predictions[name_b][:, largest_scale]
                        - predictions[name_a][:, largest_scale]
                    )
                    prob_a_better = np.mean(diff > 0)
                    lower = np.percentile(diff, (1 - confidence) / 2 * 100)

                    sig = "**" if lower > 0 else ""
                    row += f" {prob_a_better:>5.0%}{sig:>5}"
            lines.append(row)

        lines.append("\n" + "=" * 100)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ChinchillaParametricWildClusterBootstrapFit("
            f"n_bootstrap={self.n_bootstrap}, n_clusters={self.n_clusters}, "
            f"α={self.alpha_mean:.3f}±{self.alpha_std:.3f}, "
            f"β={self.beta_mean:.3f}±{self.beta_std:.3f})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Extrapolation Validation
# ══════════════════════════════════════════════════════════════════════════════


def _compute_ndcg(predicted_ranking: list[str], actual_ranking: list[str]) -> float:
    """
    Compute NDCG (Normalized Discounted Cumulative Gain) for ranking evaluation.

    We assign relevance scores based on actual ranking:
    - Best ladder (rank 1) gets relevance = n
    - Worst ladder (rank n) gets relevance = 1

    NDCG measures if the predicted ranking puts high-relevance items at the top.

    Args:
        predicted_ranking: Ladders sorted by predicted loss (best first)
        actual_ranking: Ladders sorted by actual loss (best first)

    Returns:
        NDCG score in [0, 1], where 1 means perfect ranking match.
    """
    n = len(actual_ranking)
    if n == 0:
        return 1.0
    if n == 1:
        return 1.0  # Trivial case

    # Assign relevance scores: best ladder gets n, worst gets 1
    relevance = {ladder: n - rank for rank, ladder in enumerate(actual_ranking)}

    # DCG: sum of relevance / log2(position + 1)
    dcg = 0.0
    for pos, ladder in enumerate(predicted_ranking):
        rel = relevance.get(ladder, 0)
        dcg += rel / np.log2(pos + 2)  # +2 because positions are 0-indexed

    # Ideal DCG: best possible ordering (sorted by relevance)
    ideal_relevances = sorted(relevance.values(), reverse=True)
    idcg = sum(rel / np.log2(pos + 2) for pos, rel in enumerate(ideal_relevances))

    if idcg == 0:
        return 1.0

    return dcg / idcg


def _compute_kendall_tau(predicted_ranking: list[str], actual_ranking: list[str]) -> float:
    """
    Compute Kendall's tau rank correlation coefficient.

    Args:
        predicted_ranking: Ladders sorted by predicted loss (best first)
        actual_ranking: Ladders sorted by actual loss (best first)

    Returns:
        Kendall's tau in [-1, 1], where 1 means perfect agreement.
    """
    if len(predicted_ranking) < 2:
        return 1.0

    # Convert rankings to numeric ranks
    pred_rank = {ladder: i for i, ladder in enumerate(predicted_ranking)}
    actual_rank = {ladder: i for i, ladder in enumerate(actual_ranking)}

    # Get common ladders in consistent order
    common = [ldr for ldr in predicted_ranking if ldr in actual_rank]
    if len(common) < 2:
        return 1.0

    pred_ranks = [pred_rank[ldr] for ldr in common]
    actual_ranks = [actual_rank[ldr] for ldr in common]

    result = stats.kendalltau(pred_ranks, actual_ranks)
    # Handle both old and new scipy API - use getattr to avoid type errors
    tau = float(getattr(result, "statistic", getattr(result, "correlation", 0.0)))
    return tau if np.isfinite(tau) else 0.0


def _compute_spearman(predicted_ranking: list[str], actual_ranking: list[str]) -> float:
    """
    Compute Spearman's rank correlation coefficient.

    Args:
        predicted_ranking: Ladders sorted by predicted loss (best first)
        actual_ranking: Ladders sorted by actual loss (best first)

    Returns:
        Spearman's rho in [-1, 1], where 1 means perfect agreement.
    """
    if len(predicted_ranking) < 2:
        return 1.0

    # Convert rankings to numeric ranks
    pred_rank = {ladder: i for i, ladder in enumerate(predicted_ranking)}
    actual_rank = {ladder: i for i, ladder in enumerate(actual_ranking)}

    # Get common ladders in consistent order
    common = [ldr for ldr in predicted_ranking if ldr in actual_rank]
    if len(common) < 2:
        return 1.0

    pred_ranks = [pred_rank[ldr] for ldr in common]
    actual_ranks = [actual_rank[ldr] for ldr in common]

    result = stats.spearmanr(pred_ranks, actual_ranks)
    # Handle both old and new scipy API - use getattr to avoid type errors
    rho = float(getattr(result, "statistic", getattr(result, "correlation", 0.0)))
    return rho if np.isfinite(rho) else 0.0


@dataclass
class ExtrapolationPointResult:
    """Result for a single target compute level within a fold."""

    target_compute: float
    """Target compute value for this evaluation."""

    target_N: float
    """Target parameter count (representative, may vary by ladder)."""

    target_D: float
    """Target token count (representative, may vary by ladder)."""

    extrapolation_ratio: float
    """Ratio of target_compute / max_train_compute. Values > 1 indicate extrapolation."""

    log_extrapolation_distance: float
    """log10(target_compute / max_train_compute). Larger = further extrapolation."""

    predicted_losses: dict[str, float]
    """Ladder name -> predicted loss at target."""

    actual_losses: dict[str, float]
    """Ladder name -> actual loss at target."""

    predicted_ranking: list[str]
    """Ladders sorted by predicted loss (best/lowest first)."""

    actual_ranking: list[str]
    """Ladders sorted by actual loss (best/lowest first)."""

    ndcg: float
    """NDCG score at this target."""

    kendall_tau: float
    """Kendall's tau at this target."""

    spearman: float
    """Spearman's rho at this target."""

    n_ladders_evaluated: int
    """Number of ladders that had data at this target."""

    @property
    def top1_correct(self) -> bool:
        """Whether the top-1 prediction matches ground truth."""
        if not self.predicted_ranking or not self.actual_ranking:
            return False
        return self.predicted_ranking[0] == self.actual_ranking[0]

    @property
    def top2_correct(self) -> bool:
        """Whether the top-2 predictions match ground truth (order-insensitive)."""
        if len(self.predicted_ranking) < 2 or len(self.actual_ranking) < 2:
            return self.top1_correct
        return set(self.predicted_ranking[:2]) == set(self.actual_ranking[:2])


@dataclass
class ExtrapolationFoldResult:
    """Result for a single validation fold, with predictions at ALL higher compute levels."""

    fold_idx: int
    """Fold index (0-based)."""

    cutoff_compute: float
    """Compute threshold for this fold (training uses points below this)."""

    max_train_compute: float
    """Maximum compute in the training set for this fold."""

    point_results: list[ExtrapolationPointResult]
    """Results for each target compute level above the cutoff."""

    n_ladders_evaluated: int
    """Number of ladders that had enough training data for this fold."""

    ladders_skipped: list[str]
    """Ladders that were skipped (insufficient training data)."""

    ladder_fits: dict[str, ChinchillaParametricFit]
    """Fitted scaling laws for each ladder (for inspection/debugging)."""

    @property
    def n_target_points(self) -> int:
        """Number of target compute levels evaluated."""
        return len(self.point_results)

    @property
    def mean_ndcg(self) -> float:
        """Mean NDCG across all target points."""
        if not self.point_results:
            return 0.0
        return float(np.mean([pr.ndcg for pr in self.point_results]))

    @property
    def mean_kendall_tau(self) -> float:
        """Mean Kendall's tau across all target points."""
        if not self.point_results:
            return 0.0
        return float(np.mean([pr.kendall_tau for pr in self.point_results]))

    @property
    def mean_spearman(self) -> float:
        """Mean Spearman's rho across all target points."""
        if not self.point_results:
            return 0.0
        return float(np.mean([pr.spearman for pr in self.point_results]))

    @property
    def top1_accuracy(self) -> float:
        """Fraction of target points where top-1 prediction was correct."""
        if not self.point_results:
            return 0.0
        return sum(1 for pr in self.point_results if pr.top1_correct) / len(self.point_results)

    def report(self) -> str:
        """Generate a report for this fold."""
        lines = []
        lines.append(f"Fold {self.fold_idx}: Cutoff C={self.cutoff_compute:.2e}")
        lines.append(f"  Max train compute: {self.max_train_compute:.2e}")
        lines.append(f"  Ladders evaluated: {self.n_ladders_evaluated}")
        lines.append(f"  Target points: {self.n_target_points}")
        if self.ladders_skipped:
            lines.append(f"  Skipped: {', '.join(self.ladders_skipped)}")

        lines.append(f"\n  Overall: NDCG={self.mean_ndcg:.4f}, τ={self.mean_kendall_tau:.4f}")
        lines.append(f"  Top-1 accuracy: {self.top1_accuracy:.1%}")

        # Show metrics by extrapolation distance
        lines.append("\n  By extrapolation distance:")
        lines.append(f"    {'log₁₀(ratio)':>12} {'NDCG':>8} {'τ':>8} {'Top1':>6}")
        lines.append(f"    {'-' * 12} {'-' * 8} {'-' * 8} {'-' * 6}")
        for pr in self.point_results:
            top1 = "✓" if pr.top1_correct else "✗"
            lines.append(
                f"    {pr.log_extrapolation_distance:>12.2f} "
                f"{pr.ndcg:>8.4f} {pr.kendall_tau:>8.4f} {top1:>6}"
            )

        return "\n".join(lines)


@dataclass
class ExtrapolationValidationResult:
    """
    Results from scaling law extrapolation validation.

    This uses an expanding-window approach similar to time-series cross-validation:
    for each training cutoff, we fit scaling laws on data below the cutoff,
    then predict at ALL higher compute levels and compare rankings.

    Key insight: we track how accuracy degrades with extrapolation distance,
    measured as log10(target_compute / max_train_compute).
    """

    fold_results: list[ExtrapolationFoldResult]
    """Results for each validation fold."""

    n_folds: int
    """Total number of validation folds."""

    n_ladders: int
    """Total number of unique ladders in the dataset."""

    ladder_names: list[str]
    """Names of all ladders."""

    def _get_all_point_results(self) -> list[ExtrapolationPointResult]:
        """Get all point results across all folds."""
        return [pr for fr in self.fold_results for pr in fr.point_results]

    @property
    def total_evaluations(self) -> int:
        """Total number of (fold, target) evaluations."""
        return sum(fr.n_target_points for fr in self.fold_results)

    @property
    def mean_ndcg(self) -> float:
        """Mean NDCG across all evaluations."""
        all_points = self._get_all_point_results()
        return float(np.mean([pr.ndcg for pr in all_points])) if all_points else 0.0

    @property
    def std_ndcg(self) -> float:
        """Std of NDCG across all evaluations."""
        all_points = self._get_all_point_results()
        return float(np.std([pr.ndcg for pr in all_points])) if all_points else 0.0

    @property
    def mean_kendall_tau(self) -> float:
        """Mean Kendall's tau across all evaluations."""
        all_points = self._get_all_point_results()
        return float(np.mean([pr.kendall_tau for pr in all_points])) if all_points else 0.0

    @property
    def std_kendall_tau(self) -> float:
        """Std of Kendall's tau across all evaluations."""
        all_points = self._get_all_point_results()
        return float(np.std([pr.kendall_tau for pr in all_points])) if all_points else 0.0

    @property
    def mean_spearman(self) -> float:
        """Mean Spearman's rho across all evaluations."""
        all_points = self._get_all_point_results()
        return float(np.mean([pr.spearman for pr in all_points])) if all_points else 0.0

    @property
    def std_spearman(self) -> float:
        """Std of Spearman's rho across all evaluations."""
        all_points = self._get_all_point_results()
        return float(np.std([pr.spearman for pr in all_points])) if all_points else 0.0

    @property
    def top1_accuracy(self) -> float:
        """Fraction of evaluations where top-1 prediction was correct."""
        all_points = self._get_all_point_results()
        if not all_points:
            return 0.0
        return sum(1 for pr in all_points if pr.top1_correct) / len(all_points)

    @property
    def top2_accuracy(self) -> float:
        """Fraction of evaluations where top-2 prediction was correct."""
        all_points = self._get_all_point_results()
        if not all_points:
            return 0.0
        return sum(1 for pr in all_points if pr.top2_correct) / len(all_points)

    def metrics_by_extrapolation_distance(self, n_bins: int = 5) -> list[dict[str, float]]:
        """
        Compute metrics binned by extrapolation distance.

        This is the key analysis: how does ranking accuracy degrade as we
        extrapolate further from the training data?

        Args:
            n_bins: Number of bins for log extrapolation distance.

        Returns:
            List of dicts with keys: log_dist_min, log_dist_max, log_dist_mid,
            n_samples, ndcg, kendall_tau, spearman, top1_accuracy.
        """
        all_points = self._get_all_point_results()
        if not all_points:
            return []

        log_dists = np.array([pr.log_extrapolation_distance for pr in all_points])
        min_dist, max_dist = log_dists.min(), log_dists.max()

        if min_dist >= max_dist:
            # All same distance, return single bin
            return [
                {
                    "log_dist_min": float(min_dist),
                    "log_dist_max": float(max_dist),
                    "log_dist_mid": float(min_dist),
                    "n_samples": len(all_points),
                    "ndcg": float(np.mean([pr.ndcg for pr in all_points])),
                    "kendall_tau": float(np.mean([pr.kendall_tau for pr in all_points])),
                    "spearman": float(np.mean([pr.spearman for pr in all_points])),
                    "top1_accuracy": sum(1 for pr in all_points if pr.top1_correct)
                    / len(all_points),
                }
            ]

        bin_edges = np.linspace(min_dist, max_dist + 1e-9, n_bins + 1)
        results = []

        for i in range(n_bins):
            bin_min, bin_max = bin_edges[i], bin_edges[i + 1]
            bin_points = [
                pr for pr in all_points if bin_min <= pr.log_extrapolation_distance < bin_max
            ]

            if not bin_points:
                continue

            results.append(
                {
                    "log_dist_min": float(bin_min),
                    "log_dist_max": float(bin_max),
                    "log_dist_mid": float((bin_min + bin_max) / 2),
                    "n_samples": len(bin_points),
                    "ndcg": float(np.mean([pr.ndcg for pr in bin_points])),
                    "kendall_tau": float(np.mean([pr.kendall_tau for pr in bin_points])),
                    "spearman": float(np.mean([pr.spearman for pr in bin_points])),
                    "top1_accuracy": sum(1 for pr in bin_points if pr.top1_correct)
                    / len(bin_points),
                }
            )

        return results

    def report(self) -> str:
        """Generate a comprehensive validation report."""
        lines = []
        lines.append("=" * 90)
        lines.append("SCALING LAW EXTRAPOLATION VALIDATION REPORT")
        lines.append("=" * 90)

        lines.append(f"\nDataset: {self.n_ladders} ladders, {self.n_folds} training folds")
        lines.append(f"Total evaluations: {self.total_evaluations} (fold × target combinations)")
        lines.append(f"Ladders: {', '.join(self.ladder_names)}")

        # Overall metrics
        lines.append("\n" + "=" * 90)
        lines.append("OVERALL METRICS (across all extrapolation distances)")
        lines.append("-" * 50)

        lines.append(f"  {'Metric':<20} {'Mean':>10} {'Std':>10}")
        lines.append(f"  {'-' * 20} {'-' * 10} {'-' * 10}")
        lines.append(f"  {'NDCG':<20} {self.mean_ndcg:>10.4f} {self.std_ndcg:>10.4f}")
        lines.append(
            f"  {'Kendall τ':<20} {self.mean_kendall_tau:>10.4f} {self.std_kendall_tau:>10.4f}"
        )
        lines.append(f"  {'Spearman ρ':<20} {self.mean_spearman:>10.4f} {self.std_spearman:>10.4f}")

        lines.append(f"\n  Top-1 Accuracy: {self.top1_accuracy:.1%}")
        lines.append(f"  Top-2 Accuracy: {self.top2_accuracy:.1%}")

        # KEY: Metrics by extrapolation distance
        lines.append("\n" + "=" * 90)
        lines.append("METRICS BY EXTRAPOLATION DISTANCE")
        lines.append("(log₁₀(target_compute / max_train_compute))")
        lines.append("-" * 90)

        dist_metrics = self.metrics_by_extrapolation_distance(n_bins=6)
        if dist_metrics:
            header = f"  {'Distance Range':>20} {'N':>6} {'NDCG':>8} {'τ':>8} {'ρ':>8} {'Top1':>8}"
            lines.append(header)
            lines.append("-" * 90)

            for dm in dist_metrics:
                dist_range = f"[{dm['log_dist_min']:.2f}, {dm['log_dist_max']:.2f})"
                lines.append(
                    f"  {dist_range:>20} {dm['n_samples']:>6} "
                    f"{dm['ndcg']:>8.4f} {dm['kendall_tau']:>8.4f} "
                    f"{dm['spearman']:>8.4f} {dm['top1_accuracy']:>7.1%}"
                )

            # Interpretation
            lines.append("\n" + "-" * 50)
            if len(dist_metrics) >= 2:
                first_ndcg = dist_metrics[0]["ndcg"]
                last_ndcg = dist_metrics[-1]["ndcg"]
                degradation = first_ndcg - last_ndcg

                if degradation < 0.05:
                    lines.append("  ✓ Excellent: Ranking accuracy stable across distances")
                elif degradation < 0.15:
                    lines.append("  ○ Good: Moderate degradation with distance")
                else:
                    lines.append("  ⚠ Warning: Significant degradation at large distances")
                    lines.append(f"     NDCG drops from {first_ndcg:.3f} to {last_ndcg:.3f}")

        # Per-fold summary
        lines.append("\n" + "=" * 90)
        lines.append("PER-FOLD SUMMARY")
        lines.append("-" * 90)

        header = f"  {'Fold':>5} {'Cutoff C':>12} {'#Targets':>10} {'NDCG':>8} {'τ':>8} {'Top1':>8}"
        lines.append(header)
        lines.append("-" * 90)

        for fr in self.fold_results:
            row = (
                f"  {fr.fold_idx:>5} {fr.cutoff_compute:>12.2e} {fr.n_target_points:>10} "
                f"{fr.mean_ndcg:>8.4f} {fr.mean_kendall_tau:>8.4f} {fr.top1_accuracy:>7.1%}"
            )
            lines.append(row)

        # Detailed fold results (abbreviated)
        lines.append("\n" + "=" * 90)
        lines.append("DETAILED FOLD RESULTS")
        lines.append("=" * 90)

        for fr in self.fold_results:
            lines.append("\n" + fr.report())

        lines.append("\n" + "=" * 90)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ExtrapolationValidationResult("
            f"n_folds={self.n_folds}, total_evals={self.total_evaluations}, "
            f"NDCG={self.mean_ndcg:.3f}±{self.std_ndcg:.3f}, "
            f"τ={self.mean_kendall_tau:.3f}±{self.std_kendall_tau:.3f}, "
            f"top1_acc={self.top1_accuracy:.1%})"
        )


@dataclass
class ScalingLawExtrapolationValidator:
    """
    Validates scaling law extrapolation using expanding-window cross-validation.

    This validator assesses how well scaling laws fitted on smaller-scale data
    can predict relative performance (rankings) at larger scales. This is crucial
    for model ladder experiments where we want to know if early training signals
    reliably predict which configuration will be best at full scale.

    Key feature: For each training fold, we predict at ALL higher compute levels,
    not just the next one. This allows analysis of how ranking accuracy degrades
    with extrapolation distance (measured as log10(target_compute / max_train_compute)).

    The validation procedure:
    1. Sort all observations by compute (C = 6*N*D)
    2. Define validation folds based on compute thresholds
    3. For each fold:
       a. Fit a separate scaling law for each ladder using data below threshold
       b. Predict loss at ALL compute levels above threshold for each ladder
       c. At each target level, rank ladders by predicted loss
       d. Compare to actual ranking using NDCG, Kendall's tau, Spearman's rho
       e. Track metrics by extrapolation distance

    Output includes:
    - Overall metrics across all (fold, target) combinations
    - Metrics binned by extrapolation distance (to see degradation)
    - Per-fold details with all target points

    Example:
        >>> # Prepare data
        >>> ladders = np.array(["baseline", "baseline", "muon", "muon", ...])
        >>> N = np.array([190e6, 190e6, 190e6, 190e6, ...])
        >>> D = np.array([1e9, 10e9, 1e9, 10e9, ...])
        >>> loss = np.array([3.5, 3.0, 3.4, 2.9, ...])
        >>>
        >>> # Validate extrapolation
        >>> result = ScalingLawExtrapolationValidator.validate(
        ...     ladders=ladders,
        ...     N=N,
        ...     D=D,
        ...     loss=loss,
        ...     min_train_points=5,
        ... )
        >>> print(result.report())
        >>>
        >>> # Analyze by extrapolation distance
        >>> for bucket in result.metrics_by_extrapolation_distance():
        ...     print(f"Distance {bucket['log_dist_mid']:.2f}: NDCG={bucket['ndcg']:.3f}")
    """

    @staticmethod
    def validate(
        ladders: ArrayLike,
        N: ArrayLike,
        D: ArrayLike,
        loss: ArrayLike,
        min_train_points: int = 5,
        min_ladders_per_fold: int = 2,
        loss_fn: LossFunctionType = "log_huber",
        loss_param: Optional[float] = None,
        parallel: bool = True,
        validation_points: Optional[ArrayLike] = None,
        extrapolation_dim: Literal["C", "N", "D"] = "C",
        target_multiple: Optional[float] = None,
        verbose: bool = True,
    ) -> ExtrapolationValidationResult:
        """
        Validate scaling law extrapolation using expanding-window cross-validation.

        Args:
            ladders: Array of ladder names/identifiers for each observation.
            N: Array of parameter counts.
            D: Array of token counts.
            loss: Array of loss values.
            min_train_points: Minimum number of training points required per ladder
                to fit a scaling law (default 5).
            min_ladders_per_fold: Minimum number of ladders required per fold
                for meaningful ranking comparison (default 2).
            loss_fn: Loss function for fitting scaling laws.
                Use "log_huber" (default) for validation to avoid systematic bias.
            loss_param: Parameter for loss function.
            parallel: If True, use parallel optimization for fitting.
            validation_points: Optional array of values (in the chosen dimension) to use
                as validation cutoffs. If None, automatically determines folds based on data.
            extrapolation_dim: Dimension to use for defining extrapolation distance:
                - "C": Compute (6*N*D) - extrapolate based on total compute (default)
                - "N": Parameters - extrapolate based on model size
                - "D": Tokens - extrapolate based on training data amount
            target_multiple: If set, only predict at points that are approximately this
                multiple of the max training scale. For example, target_multiple=8.0 means
                predict at ~8x the max training compute/N/D. This standardizes the
                extrapolation distance across folds. If None, predict at all points
                above the cutoff.
            verbose: If True, print progress information.

        Returns:
            ExtrapolationValidationResult with comprehensive metrics.

        Raises:
            ValueError: If insufficient data for validation.
        """
        # Convert to numpy arrays
        ladders = np.asarray(ladders)
        N = np.asarray(N)
        D = np.asarray(D)
        loss = np.asarray(loss)

        # Compute values (approximate FLOPs)
        C = 6 * N * D

        # Clean data
        mask = np.isfinite(N) & np.isfinite(D) & np.isfinite(loss) & (N > 0) & (D > 0) & (loss > 0)
        ladders = ladders[mask]
        N = N[mask]
        D = D[mask]
        loss = loss[mask]
        C = C[mask]

        # Select the dimension for extrapolation
        dim_map = {"C": C, "N": N, "D": D}
        dim_names = {"C": "Compute (6ND)", "N": "Parameters (N)", "D": "Tokens (D)"}
        if extrapolation_dim not in dim_map:
            raise ValueError(f"extrapolation_dim must be 'C', 'N', or 'D', got {extrapolation_dim}")
        scale_dim = dim_map[extrapolation_dim]

        if len(N) < min_train_points + 1:
            raise ValueError(
                f"Need at least {min_train_points + 1} valid data points, got {len(N)}"
            )

        # Get unique ladders
        unique_ladders = np.unique(ladders)
        n_ladders = len(unique_ladders)

        if n_ladders < min_ladders_per_fold:
            raise ValueError(
                f"Need at least {min_ladders_per_fold} ladders for ranking, got {n_ladders}"
            )

        if verbose:
            print(f"Validating extrapolation for {n_ladders} ladders:")
            print(f"Extrapolation dimension: {dim_names[extrapolation_dim]}")
            if target_multiple is not None:
                print(f"Target multiple: {target_multiple}x (fixed extrapolation distance)")
            else:
                print("Target multiple: None (all points above cutoff)")
            for ladder in unique_ladders:
                n_points = np.sum(ladders == ladder)
                print(f"  {ladder}: {n_points} observations")

        # Determine validation points (thresholds in the chosen dimension)
        if validation_points is None:
            # Automatic: use unique values as potential validation points
            unique_vals = np.sort(np.unique(scale_dim))

            # We need at least min_train_points below each threshold
            # and at least 1 point at or above for validation
            valid_thresholds = []
            for val in unique_vals:
                # Points below this value
                n_below = np.sum(scale_dim < val)
                # Check if most ladders have enough training data
                ladders_with_enough = 0
                for ladder in unique_ladders:
                    ladder_mask = ladders == ladder
                    n_train = np.sum((scale_dim < val) & ladder_mask)
                    if n_train >= min_train_points:
                        ladders_with_enough += 1

                if ladders_with_enough >= min_ladders_per_fold and n_below >= min_train_points:
                    valid_thresholds.append(val)

            validation_points = np.array(valid_thresholds)

        else:
            validation_points = np.asarray(validation_points)

        if len(validation_points) == 0:
            raise ValueError(
                "No valid validation points found. "
                f"Need at least {min_train_points} training points per ladder. "
                "Consider reducing min_train_points or adding more data."
            )

        if verbose:
            print(f"\nUsing {len(validation_points)} validation folds")

        # Run validation for each fold
        fold_results: list[ExtrapolationFoldResult] = []

        for fold_idx, cutoff_val in enumerate(
            tqdm(validation_points, desc="Validation folds", disable=not verbose)
        ):
            fold_result = ScalingLawExtrapolationValidator._evaluate_fold(
                fold_idx=fold_idx,
                cutoff_val=cutoff_val,
                ladders=ladders,
                N=N,
                D=D,
                loss=loss,
                scale_dim=scale_dim,
                unique_ladders=unique_ladders,
                min_train_points=min_train_points,
                min_ladders_per_fold=min_ladders_per_fold,
                loss_fn=loss_fn,
                loss_param=loss_param,
                parallel=parallel,
                target_multiple=target_multiple,
            )

            if fold_result is not None:
                fold_results.append(fold_result)

        if len(fold_results) == 0:
            raise ValueError(
                "No valid folds produced. Check that ladders have sufficient "
                "data across multiple compute scales."
            )

        return ExtrapolationValidationResult(
            fold_results=fold_results,
            n_folds=len(fold_results),
            n_ladders=n_ladders,
            ladder_names=list(unique_ladders),
        )

    @staticmethod
    def _evaluate_fold(
        fold_idx: int,
        cutoff_val: float,
        ladders: np.ndarray,
        N: np.ndarray,
        D: np.ndarray,
        loss: np.ndarray,
        scale_dim: np.ndarray,
        unique_ladders: np.ndarray,
        min_train_points: int,
        min_ladders_per_fold: int,
        loss_fn: LossFunctionType,
        loss_param: Optional[float],
        parallel: bool,
        target_multiple: Optional[float] = None,
    ) -> Optional[ExtrapolationFoldResult]:
        """
        Evaluate a single validation fold.

        For each fold:
        1. Fit scaling law per ladder using training data (scale_dim < cutoff)
        2. For target scale levels above cutoff (or at specific multiple if set):
           - Predict loss for each ladder
           - Compare predicted vs actual rankings
        3. Track metrics by extrapolation distance

        Args:
            target_multiple: If set, only evaluate at the target closest to
                max_train_scale * target_multiple. Otherwise, evaluate at all
                points above the cutoff.
        """
        ladders_skipped: list[str] = []
        ladder_fits: dict[str, ChinchillaParametricFit] = {}

        # Compute max training scale for extrapolation distance calculation
        train_mask_all = scale_dim < cutoff_val
        if not np.any(train_mask_all):
            return None
        max_train_scale = float(np.max(scale_dim[train_mask_all]))

        # Step 1: Fit scaling law for each ladder
        for ladder in unique_ladders:
            ladder_mask = ladders == ladder
            train_mask = (scale_dim < cutoff_val) & ladder_mask
            n_train = np.sum(train_mask)

            if n_train < min_train_points:
                ladders_skipped.append(str(ladder))
                continue

            try:
                fit = ChinchillaParametricFit.fit(
                    N=N[train_mask],
                    D=D[train_mask],
                    loss=loss[train_mask],
                    loss_fn=loss_fn,
                    loss_param=loss_param,
                    parallel=parallel,
                )
                ladder_fits[str(ladder)] = fit
            except Exception:
                ladders_skipped.append(str(ladder))
                continue

        # Check if we have enough ladders
        if len(ladder_fits) < min_ladders_per_fold:
            return None

        # Step 2: Get target scale levels
        # Use strict inequality (>) to ensure we're always extrapolating beyond training
        target_mask = scale_dim > max_train_scale
        if not np.any(target_mask):
            return None

        # Get unique target scale values, sorted
        unique_target_vals = np.sort(np.unique(scale_dim[target_mask]))

        # If target_multiple is set, filter to only the closest target
        if target_multiple is not None:
            desired_target = max_train_scale * target_multiple
            # Find the closest actual data point to the desired target
            closest_idx = np.argmin(np.abs(unique_target_vals - desired_target))
            closest_val = unique_target_vals[closest_idx]
            # Check if it's reasonably close (within 50% of the desired multiple)
            actual_multiple = closest_val / max_train_scale
            if actual_multiple < target_multiple * 0.5 or actual_multiple > target_multiple * 2.0:
                # No suitable target at this multiple
                return None
            unique_target_vals = np.array([closest_val])

        # Step 3: Evaluate at each target scale level
        point_results: list[ExtrapolationPointResult] = []

        for target_val in unique_target_vals:
            predicted_losses: dict[str, float] = {}
            actual_losses: dict[str, float] = {}

            # For each fitted ladder, find if it has data at this target scale
            # and make predictions
            target_N_rep = 0.0
            target_D_rep = 0.0

            for ladder_name, fit in ladder_fits.items():
                ladder_mask = ladders == ladder_name

                # Find points for this ladder at exactly this target scale value
                val_mask = (scale_dim == target_val) & ladder_mask

                if not np.any(val_mask):
                    continue

                # If multiple points at same scale, use the first one
                val_idx = np.where(val_mask)[0][0]
                val_N = N[val_idx]
                val_D = D[val_idx]
                val_loss = loss[val_idx]

                # Update representative N, D
                if target_N_rep == 0.0:
                    target_N_rep = float(val_N)
                    target_D_rep = float(val_D)

                # Predict
                pred_loss = fit.predict_loss(np.array([val_N]), np.array([val_D]))[0]
                predicted_losses[ladder_name] = float(pred_loss)
                actual_losses[ladder_name] = float(val_loss)

            # Need enough ladders at this target for meaningful ranking
            if len(predicted_losses) < min_ladders_per_fold:
                continue

            # Compute rankings
            predicted_ranking = sorted(
                predicted_losses.keys(), key=lambda ldr: predicted_losses[ldr]
            )
            actual_ranking = sorted(actual_losses.keys(), key=lambda ldr: actual_losses[ldr])

            # Compute metrics
            ndcg = _compute_ndcg(predicted_ranking, actual_ranking)
            kendall_tau = _compute_kendall_tau(predicted_ranking, actual_ranking)
            spearman = _compute_spearman(predicted_ranking, actual_ranking)

            # Extrapolation distance (in the chosen dimension)
            extrapolation_ratio = float(target_val) / max_train_scale
            log_extrapolation_distance = float(np.log10(extrapolation_ratio))

            point_results.append(
                ExtrapolationPointResult(
                    target_compute=float(target_val),  # Generic "scale" value
                    target_N=target_N_rep,
                    target_D=target_D_rep,
                    extrapolation_ratio=extrapolation_ratio,
                    log_extrapolation_distance=log_extrapolation_distance,
                    predicted_losses=predicted_losses,
                    actual_losses=actual_losses,
                    predicted_ranking=predicted_ranking,
                    actual_ranking=actual_ranking,
                    ndcg=ndcg,
                    kendall_tau=kendall_tau,
                    spearman=spearman,
                    n_ladders_evaluated=len(predicted_losses),
                )
            )

        if not point_results:
            return None

        return ExtrapolationFoldResult(
            fold_idx=fold_idx,
            cutoff_compute=float(cutoff_val),  # Cutoff in chosen dimension
            max_train_compute=max_train_scale,  # Max training value in chosen dimension
            point_results=point_results,
            n_ladders_evaluated=len(ladder_fits),
            ladders_skipped=ladders_skipped,
            ladder_fits=ladder_fits,
        )
