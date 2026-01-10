from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats
from scipy.optimize import minimize

from olmo_core.data.composable.utils import format_token_count
from olmo_core.model_ladder.utils import format_count


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


def print_missing_ladder_points(
    df: pd.DataFrame,
    *,
    ladder_col: str = "ladder",
    size_col: str = "size",
    xC_col: str = "xC",
    expected_xC: list[float] | None = None,
    expected_sizes: list[str] | None = None,
) -> pd.DataFrame:
    """
    Print missing (xC, Model_size) data points for each ladder.

    Args:
        df: DataFrame with ladder data
        ladder_col: Column name for ladder identifier
        size_col: Column name for model size
        xC_col: Column name for Chinchilla multiple
        expected_xC: Expected xC values. Defaults to [0.5, 1, 2, 4, 8]
        expected_sizes: Expected model sizes. Defaults to unique sizes in df

    Returns:
        DataFrame with missing combinations
    """
    if expected_xC is None:
        expected_xC = [0.5, 1.0, 2.0, 4.0, 8.0]
    if expected_sizes is None:
        expected_sizes = sorted(
            df[size_col].unique(), key=lambda x: float(x.replace("M", "e6").replace("B", "e9"))
        )

    # Build full expected grid
    expected = set((xc, size) for xc in expected_xC for size in expected_sizes)

    missing_rows = []
    for ladder_name in sorted(df[ladder_col].unique()):
        ladder_df = df[df[ladder_col] == ladder_name]
        observed = set(zip(ladder_df[xC_col], ladder_df[size_col]))
        missing = expected - observed

        if missing:
            print(f"\n{ladder_name}:")
            # Sort by size then xC
            missing_sorted = sorted(missing, key=lambda x: (expected_sizes.index(x[1]), x[0]))
            for xc, size in missing_sorted:
                print(f"  Missing: {xc}xC @ {size}")
                missing_rows.append({ladder_col: ladder_name, xC_col: xc, size_col: size})
        else:
            print(f"\n{ladder_name}: ✓ Complete")

    return pd.DataFrame(missing_rows)


def _silverman_bandwidth(x: np.ndarray) -> float:
    """
    Compute Silverman's rule-of-thumb bandwidth for kernel density estimation.

    This bandwidth is used for smoothed bootstrap, where we add Gaussian noise
    to resampled data points to reduce discreteness artifacts.

    Args:
        x: Array of data values

    Returns:
        Bandwidth h for Gaussian kernel
    """
    n = len(x)
    sd = np.std(x, ddof=1)
    if sd < 1e-10 or n < 2:
        return 0.0
    return 1.06 * sd * n ** (-1 / 5)


def _extract_pareto_frontier(
    N: np.ndarray,
    D: np.ndarray,
    C: np.ndarray,
    L: np.ndarray,
    n_bins: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract the Pareto frontier: for each compute bin, find the observation with minimum loss.

    This implements the key step of Chinchilla Approach 1: at each compute level, identify
    which (N, D) configuration achieves the lowest loss. Only these optimal points should
    be used for fitting the scaling laws N_opt(C), D_opt(C), and L(C).

    Args:
        N: Array of parameter counts
        D: Array of token counts
        C: Array of compute values
        L: Array of loss values
        n_bins: Number of log-spaced bins to group compute values

    Returns:
        Tuple of (N_frontier, D_frontier, C_frontier, L_frontier) arrays containing
        only the Pareto-optimal points (minimum loss per compute bin)
    """
    log_C = np.log(C)
    log_C_edges = np.linspace(log_C.min(), log_C.max(), n_bins + 1)

    optimal_indices = []
    for i in range(n_bins):
        # Include right edge in last bin
        if i == n_bins - 1:
            bin_mask = (log_C >= log_C_edges[i]) & (log_C <= log_C_edges[i + 1])
        else:
            bin_mask = (log_C >= log_C_edges[i]) & (log_C < log_C_edges[i + 1])

        if bin_mask.any():
            bin_indices = np.where(bin_mask)[0]
            # Find the index with minimum loss in this bin
            best_idx = bin_indices[np.argmin(L[bin_mask])]
            optimal_indices.append(best_idx)

    if len(optimal_indices) == 0:
        raise ValueError("No valid bins found - check data range")

    idx = np.array(optimal_indices)
    return N[idx], D[idx], C[idx], L[idx]


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
    """Compute loss for given parameter count and token count using the Chinchilla scaling law."""
    return E + A / np.power(N, alpha) + B / np.power(D, beta)


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

    # Bootstrap confidence intervals (10th, 90th percentiles)
    E_ci: Optional[tuple[float, float]] = None
    """Bootstrap 10th/90th percentile for E"""
    A_ci: Optional[tuple[float, float]] = None
    """Bootstrap 10th/90th percentile for A"""
    alpha_ci: Optional[tuple[float, float]] = None
    """Bootstrap 10th/90th percentile for alpha"""
    G_ci: Optional[tuple[float, float]] = None
    """Bootstrap 10th/90th percentile for G"""
    a_ci: Optional[tuple[float, float]] = None
    """Bootstrap 10th/90th percentile for a"""
    H_ci: Optional[tuple[float, float]] = None
    """Bootstrap 10th/90th percentile for H"""
    b_ci: Optional[tuple[float, float]] = None
    """Bootstrap 10th/90th percentile for b"""

    def predict_loss(self, C: ArrayLike) -> np.ndarray:
        """Predict loss for given compute values using L(C) = E + A / C^alpha."""
        C = np.asarray(C)
        return self.E + self.A / np.power(C, self.alpha)

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
        bootstrap: bool = False,
        n_bootstrap: int = 100,
        bootstrap_frac: float = 0.8,
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
            bootstrap: If True, compute bootstrap confidence intervals
            n_bootstrap: Number of bootstrap samples (default 100)
            bootstrap_frac: Fraction of data to sample per bootstrap (default 0.8)

        Returns:
            ChinchillaIsoParamFit with fitted parameters (and bootstrap CIs if requested)
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
            L_pred = E_param + A / np.power(C_frontier, alpha)
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
        L_pred = E_fit + A / np.power(C_frontier, alpha)
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

        # Bootstrap confidence intervals
        bootstrap_cis: dict[str, Optional[tuple[float, float]]] = {
            "E_ci": None,
            "A_ci": None,
            "alpha_ci": None,
            "G_ci": None,
            "a_ci": None,
            "H_ci": None,
            "b_ci": None,
        }

        if bootstrap:
            # Smoothed bootstrap: resample with replacement and add Gaussian noise
            # to loss values. This is more principled than smoothing parameter estimates.
            n_samples = len(N_valid)
            sample_size = int(n_samples * bootstrap_frac)
            bootstrap_params: dict[str, list[float]] = {
                "E": [],
                "A": [],
                "alpha": [],
                "G": [],
                "a": [],
                "H": [],
                "b": [],
            }

            rng = np.random.default_rng(42)

            # Compute bandwidth for loss smoothing using Silverman's rule
            loss_bandwidth = _silverman_bandwidth(L_valid)

            for _ in range(n_bootstrap):
                # Sample with replacement from all valid observations
                indices = rng.choice(n_samples, size=sample_size, replace=True)
                N_boot = N_valid[indices]
                D_boot = D_valid[indices]
                C_boot = C_valid[indices]
                # Smoothed bootstrap: add Gaussian noise to loss values
                L_boot = L_valid[indices] + rng.normal(0.0, loss_bandwidth, size=sample_size)
                # Ensure loss stays positive
                L_boot = np.maximum(L_boot, 1e-10)

                try:
                    # Fit on smoothed bootstrap sample
                    boot_fit = cls.fit(
                        N=N_boot,
                        D=D_boot,
                        C=C_boot,
                        loss=L_boot,
                        huber_delta=huber_delta,
                        bootstrap=False,
                    )
                    bootstrap_params["E"].append(boot_fit.E)
                    bootstrap_params["A"].append(boot_fit.A)
                    bootstrap_params["alpha"].append(boot_fit.alpha)
                    bootstrap_params["G"].append(boot_fit.G)
                    bootstrap_params["a"].append(boot_fit.a)
                    bootstrap_params["H"].append(boot_fit.H)
                    bootstrap_params["b"].append(boot_fit.b)
                except (ValueError, RuntimeError):
                    # Skip failed fits
                    continue

            # Compute 10th and 90th percentiles
            for param_name in bootstrap_params:
                if len(bootstrap_params[param_name]) >= 10:
                    p10 = float(np.percentile(bootstrap_params[param_name], 10))
                    p90 = float(np.percentile(bootstrap_params[param_name], 90))
                    bootstrap_cis[f"{param_name}_ci"] = (p10, p90)

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
            **bootstrap_cis,
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
        E_ci_str = f"  [{self.E_ci[0]:.4f}, {self.E_ci[1]:.4f}]" if self.E_ci else ""
        lines.append(f"  Entropy floor (E):     {self.E:.4f}{E_ci_str}")
        a_ci_str = f"  [{self.a_ci[0]:.4f}, {self.a_ci[1]:.4f}]" if self.a_ci else ""
        lines.append(
            f"  Param exponent (a):    {self.a:.4f}{a_ci_str}  {'← Chinchilla: ~0.50' if abs(self.a - 0.5) < 0.1 else ''}"
        )
        b_ci_str = f"  [{self.b_ci[0]:.4f}, {self.b_ci[1]:.4f}]" if self.b_ci else ""
        lines.append(
            f"  Token exponent (b):    {self.b:.4f}{b_ci_str}  {'← Chinchilla: ~0.50' if abs(self.b - 0.5) < 0.1 else ''}"
        )

        # Show additional bootstrap CIs if available
        if any([self.A_ci, self.alpha_ci, self.G_ci, self.H_ci]):
            lines.append("-" * 40)
            lines.append("  Bootstrap 90% CIs (10th-90th percentile):")
            if self.A_ci:
                lines.append(f"    A:     {self.A:.4f}  [{self.A_ci[0]:.4f}, {self.A_ci[1]:.4f}]")
            if self.alpha_ci:
                lines.append(
                    f"    α:     {self.alpha:.4f}  [{self.alpha_ci[0]:.4f}, {self.alpha_ci[1]:.4f}]"
                )
            if self.G_ci:
                lines.append(f"    G:     {self.G:.4e}  [{self.G_ci[0]:.4e}, {self.G_ci[1]:.4e}]")
            if self.H_ci:
                lines.append(f"    H:     {self.H:.4e}  [{self.H_ci[0]:.4e}, {self.H_ci[1]:.4e}]")

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

    def plot(
        self,
        C: ArrayLike,
        loss: ArrayLike,
        N: Optional[ArrayLike] = None,
        D: Optional[ArrayLike] = None,
        title: Optional[str] = None,
        figsize: tuple[int, int] = (14, 10),
    ) -> plt.Figure:
        """
        Plot the scaling law fits against actual data with residuals.

        Creates a 2x2 grid when N and D are provided:
        - Top left: Loss vs Compute
        - Top right: N vs Compute
        - Bottom left: D vs Compute
        - Bottom right: Loss residuals

        All observations are shown as lighter points, while the Pareto frontier
        points (not dominated by any other point in compute-loss space) are highlighted.

        If N and D are not provided, falls back to a simple 2-panel loss plot.

        Args:
            C: Array of compute values (same units used in fitting)
            loss: Array of actual loss values
            N: Optional array of parameter counts (for N_opt fit plot)
            D: Optional array of token counts (for D_opt fit plot)
            title: Optional title for the plot
            figsize: Figure size (width, height)

        Returns:
            matplotlib Figure object
        """
        C = np.asarray(C)
        loss = np.asarray(loss)

        # Generate smooth curve for the fit
        C_smooth = np.logspace(np.log10(C.min()), np.log10(C.max()), 200)
        L_pred_smooth = self.predict_loss(C_smooth)

        # If N and D provided, create 2x2 grid
        if N is not None and D is not None:
            N = np.asarray(N)
            D = np.asarray(D)

            # Extract frontier for highlighting (using Pareto dominance)
            N_frontier, D_frontier, C_frontier, L_frontier = _extract_pareto_frontier_dominance(
                N, D, C, loss
            )

            N_pred_smooth = self.predict_optimal_N(C_smooth)
            D_pred_smooth = self.predict_optimal_D(C_smooth)

            # Compute residuals on frontier points only (what was actually fit)
            L_pred_frontier = self.predict_loss(C_frontier)
            loss_residuals_frontier = L_frontier - L_pred_frontier

            fig, axes = plt.subplots(2, 2, figsize=figsize)

            # Top left: Loss vs Compute
            ax = axes[0, 0]
            # All observations (lighter, smaller)
            ax.scatter(
                C,
                loss,
                alpha=0.3,
                s=30,
                c="gray",
                label="All runs",
                zorder=3,
            )
            # Frontier points (highlighted)
            ax.scatter(
                C_frontier,
                L_frontier,
                alpha=0.9,
                s=70,
                c="tab:blue",
                label="Frontier (used for fit)",
                zorder=5,
                edgecolors="black",
                linewidths=0.5,
            )
            ax.plot(
                C_smooth,
                L_pred_smooth,
                "r-",
                linewidth=2,
                label=f"L(C) = {self.E:.3f} + {self.A:.3f}/C^{self.alpha:.3f}",
            )
            ax.set_xscale("log")
            ax.set_xlabel("Compute (petaFLOPs)", fontsize=11)
            ax.set_ylabel("Loss", fontsize=11)
            r2_str = f" (R²={self.r_squared_loss:.4f})" if self.r_squared_loss else ""
            ax.set_title(f"Loss vs Compute{r2_str}", fontsize=12, fontweight="bold")
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(True, alpha=0.3, which="both")

            # Top right: N vs Compute
            ax = axes[0, 1]
            # All observations (lighter)
            ax.scatter(C, N, alpha=0.3, s=30, c="gray", label="All runs", zorder=3)
            # Frontier points (highlighted)
            ax.scatter(
                C_frontier,
                N_frontier,
                alpha=0.9,
                s=70,
                c="forestgreen",
                label="Frontier (optimal N)",
                zorder=5,
                edgecolors="black",
                linewidths=0.5,
            )
            ax.plot(
                C_smooth,
                N_pred_smooth,
                "r-",
                linewidth=2,
                label=f"N_opt(C) = {self.G:.2e} × C^{self.a:.3f}",
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Compute (petaFLOPs)", fontsize=11)
            ax.set_ylabel("Parameters (N)", fontsize=11)
            r2_str = f" (R²={self.r_squared_N:.4f})" if self.r_squared_N else ""
            ax.set_title(f"Optimal N vs Compute{r2_str}", fontsize=12, fontweight="bold")
            ax.legend(loc="lower right", fontsize=9)
            ax.grid(True, alpha=0.3, which="both")

            # Bottom left: D vs Compute
            ax = axes[1, 0]
            # All observations (lighter)
            ax.scatter(C, D, alpha=0.3, s=30, c="gray", label="All runs", zorder=3)
            # Frontier points (highlighted)
            ax.scatter(
                C_frontier,
                D_frontier,
                alpha=0.9,
                s=70,
                c="steelblue",
                label="Frontier (optimal D)",
                zorder=5,
                edgecolors="black",
                linewidths=0.5,
            )
            ax.plot(
                C_smooth,
                D_pred_smooth,
                "r-",
                linewidth=2,
                label=f"D_opt(C) = {self.H:.2e} × C^{self.b:.3f}",
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Compute (petaFLOPs)", fontsize=11)
            ax.set_ylabel("Tokens (D)", fontsize=11)
            r2_str = f" (R²={self.r_squared_D:.4f})" if self.r_squared_D else ""
            ax.set_title(f"Optimal D vs Compute{r2_str}", fontsize=12, fontweight="bold")
            ax.legend(loc="lower right", fontsize=9)
            ax.grid(True, alpha=0.3, which="both")

            # Bottom right: Loss Residuals (on frontier points only)
            ax = axes[1, 1]
            ax.scatter(
                C_frontier,
                loss_residuals_frontier,
                alpha=0.9,
                s=70,
                c="coral",
                edgecolors="black",
                linewidths=0.5,
            )
            ax.axhline(y=0, color="r", linestyle="--", linewidth=1.5)
            ax.set_xscale("log")
            ax.set_xlabel("Compute (petaFLOPs)", fontsize=11)
            ax.set_ylabel("Loss Residual", fontsize=11)
            ax.set_title("Loss Residuals (frontier points)", fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3, which="both")
            rmse = np.sqrt(np.mean(loss_residuals_frontier**2))
            ax.text(
                0.05,
                0.95,
                f"RMSE: {rmse:.4f}\nn={len(C_frontier)} frontier points",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
            )

            # Overall title
            if title:
                fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
            else:
                fig.suptitle(
                    "Chinchilla Approach 1 Scaling Law Fits", fontsize=14, fontweight="bold"
                )

            plt.tight_layout()
            return fig

        # Fallback: simple 2-panel loss plot when N and D not provided
        # Extract frontier for loss only (need dummy N, D)
        # In this case, just show all points since we can't extract frontier
        L_pred = self.predict_loss(C)
        loss_residuals = loss - L_pred

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(figsize[0], figsize[1] * 0.8), height_ratios=[3, 1], sharex=True
        )
        fig.subplots_adjust(hspace=0.05)

        # Main plot: Loss vs Compute
        ax1.scatter(
            C, loss, alpha=0.7, s=50, label="Observed", zorder=5, edgecolors="black", linewidths=0.5
        )
        ax1.plot(
            C_smooth,
            L_pred_smooth,
            "r-",
            linewidth=2,
            label=f"Fit: L(C) = {self.E:.3f} + {self.A:.3f}/C^{self.alpha:.3f}",
        )
        ax1.set_xscale("log")
        ax1.set_ylabel("Loss", fontsize=12)
        ax1.legend(loc="upper right", fontsize=10)
        ax1.grid(True, alpha=0.3, which="both")

        if title:
            ax1.set_title(title, fontsize=14, fontweight="bold")
        else:
            r2_str = f" (R²={self.r_squared_loss:.4f})" if self.r_squared_loss else ""
            ax1.set_title(
                f"Chinchilla IsoParam Scaling Law{r2_str}", fontsize=14, fontweight="bold"
            )

        # Residual plot
        ax2.scatter(
            C, loss_residuals, alpha=0.7, s=40, c="steelblue", edgecolors="black", linewidths=0.5
        )
        ax2.axhline(y=0, color="r", linestyle="--", linewidth=1.5)
        ax2.set_xscale("log")
        ax2.set_xlabel("Compute (petaFLOPs)", fontsize=12)
        ax2.set_ylabel("Residual", fontsize=12)
        ax2.grid(True, alpha=0.3, which="both")

        # Add residual stats
        rmse = np.sqrt(np.mean(loss_residuals**2))
        ax2.text(
            0.02,
            0.95,
            f"RMSE: {rmse:.4f}",
            transform=ax2.transAxes,
            fontsize=9,
            verticalalignment="top",
        )

        plt.tight_layout()
        return fig

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

    r_squared: Optional[float] = None
    """R-squared of the fit"""

    # Bootstrap confidence intervals (10th, 90th percentiles)
    E_ci: Optional[tuple[float, float]] = None
    """Bootstrap 10th/90th percentile for E"""
    A_ci: Optional[tuple[float, float]] = None
    """Bootstrap 10th/90th percentile for A"""
    alpha_ci: Optional[tuple[float, float]] = None
    """Bootstrap 10th/90th percentile for alpha"""
    B_ci: Optional[tuple[float, float]] = None
    """Bootstrap 10th/90th percentile for B"""
    beta_ci: Optional[tuple[float, float]] = None
    """Bootstrap 10th/90th percentile for beta"""
    a_opt_ci: Optional[tuple[float, float]] = None
    """Bootstrap 10th/90th percentile for a_opt = β/(α+β) (N_opt exponent)"""
    b_opt_ci: Optional[tuple[float, float]] = None
    """Bootstrap 10th/90th percentile for b_opt = α/(α+β) (D_opt exponent)"""

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

    def effective_data_multiplier_at_constant_beta(
        self, other: "ChinchillaParametricFit", tol: float = 0.01
    ) -> float:
        """
        Compute the MARGINAL data scaling efficiency ratio (B/D^β term only).

        Returns k such that self's data term equals other's data term with k× more tokens:
            B_self / D^β = B_other / (k*D)^β

        WARNING: This IGNORES the entropy floor E and parameter term A/N^α!
        If E_self ≠ E_other, the actual losses will NOT be equal even with this multiplier.
        Use predict_loss() for true loss comparisons.

        Args:
            other: The baseline model to compare against.
            tol: Tolerance for beta difference (raises if too different).

        Returns:
            k > 1 means self is more data-efficient (lower B coefficient).
        """
        if abs(self.beta - other.beta) < tol:
            return (other.B / self.B) ** (1 / self.beta)
        raise ValueError(f"Beta values are too different: {self.beta} vs {other.beta}")

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

    def effective_param_multiplier_at_constant_alpha(
        self, other: "ChinchillaParametricFit", tol: float = 0.01
    ) -> float:
        """
        Compute the MARGINAL parameter scaling efficiency ratio (A/N^α term only).

        Returns k such that self's param term equals other's param term with k× more params:
            A_self / N^α = A_other / (k*N)^α

        WARNING: This IGNORES the entropy floor E and data term B/D^β!
        If E_self ≠ E_other, the actual losses will NOT be equal even with this multiplier.
        Use predict_loss() for true loss comparisons.

        Args:
            other: The baseline model to compare against.
            tol: Tolerance for alpha difference (raises if too different).

        Returns:
            k > 1 means self is more parameter-efficient (lower A coefficient).
        """
        if abs(self.alpha - other.alpha) < tol:
            return (other.A / self.A) ** (1 / self.alpha)
        raise ValueError(f"Alpha values are too different: {self.alpha} vs {other.alpha}")

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
        self,
        other: "ChinchillaParametricFit",
        N: float,
        D: float,
        compact: bool = False,
        efficiency_context: Optional[tuple[float, float]] = None,
    ) -> tuple[float, float, str]:
        """
        Compute the ACTUAL loss advantage of this model vs another at a given (N, D) scale.

        Unlike effective_data_multiplier and effective_param_multiplier which only compare
        marginal scaling terms, this compares the full predicted losses including E.

        Args:
            other: The baseline model to compare against.
            N: Parameter count.
            D: Token count.
            compact: If True, return a short interpretation suitable for tables.
            efficiency_context: Optional (data_mult, param_mult) to include in interpretation.
                If provided and compact=True, will flag when efficiency disagrees with loss.

        Returns:
            Tuple of (loss_self, loss_other, interpretation) where:
            - loss_self: Predicted loss for this model
            - loss_other: Predicted loss for the baseline
            - interpretation: Human-readable comparison string
        """
        L_self = self.predict_loss(np.array([N]), np.array([D]))[0]
        L_other = other.predict_loss(np.array([N]), np.array([D]))[0]
        diff = L_other - L_self  # positive = self is better

        if compact:
            # Short format for tables
            has_eff_advantage = False
            if efficiency_context is not None:
                data_mult, param_mult = efficiency_context
                has_eff_advantage = data_mult > 1.0 or param_mult > 1.0

            if diff > 0.005:
                if has_eff_advantage:
                    interp = f"✓ Wins ({100 * diff / L_other:.1f}%)"
                else:
                    interp = f"✓ Lower L ({100 * diff / L_other:.1f}%)"
            elif diff < -0.005:
                if has_eff_advantage:
                    interp = f"⚠ Eff but +{100 * -diff / L_self:.1f}%"
                else:
                    interp = f"✗ Higher L (+{100 * -diff / L_self:.1f}%)"
            else:
                interp = "≈ Similar"
        else:
            # Verbose format
            if diff > 0.01:
                interp = f"This model wins by {diff:.4f} ({100 * diff / L_other:.1f}% lower loss)"
            elif diff < -0.01:
                interp = f"Baseline wins by {-diff:.4f} ({100 * -diff / L_self:.1f}% lower loss)"
            else:
                interp = f"Approximately equal (Δ={diff:.4f})"

        return L_self, L_other, interp

    def residual_diagnostics(
        self,
        N: ArrayLike,
        D: ArrayLike,
        loss: ArrayLike,
        alpha: float = 0.05,
    ) -> "ResidualDiagnostics":
        """
        Compute comprehensive residual diagnostics to validate the scaling law fit.

        Tests for:
        1. Normality (Shapiro-Wilk test)
        2. Homoscedasticity (constant variance across N and D)
        3. Systematic patterns (residuals correlated with log(N) or log(D))
        4. Overall fit quality metrics

        Args:
            N: Array of parameter counts used in fitting.
            D: Array of token counts used in fitting.
            loss: Array of observed loss values.
            alpha: Significance level for statistical tests (default 0.05).

        Returns:
            ResidualDiagnostics object with test results and interpretation.
        """
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

    @classmethod
    def fit(
        cls,
        N: ArrayLike,
        D: ArrayLike,
        loss: ArrayLike,
        huber_delta: float = 1e-3,
        bootstrap: bool = False,
        n_bootstrap: int = 100,
        bootstrap_frac: float = 0.8,
    ) -> "ChinchillaParametricFit":
        """
        Fit the full Chinchilla scaling law: L = E + A/N^alpha + B/D^beta

        Minimizes Huber loss on log(L) using L-BFGS-B, matching the Chinchilla paper methodology.

        Args:
            N: Array of parameter counts
            D: Array of token counts
            loss: Array of loss values
            huber_delta: Delta parameter for Huber loss
            bootstrap: If True, compute bootstrap confidence intervals
            n_bootstrap: Number of bootstrap samples (default 100)
            bootstrap_frac: Fraction of data to sample per bootstrap (default 0.8)

        Returns:
            ChinchillaParametricFit with fitted parameters (and bootstrap CIs if requested)
        """
        N = np.asarray(N)
        D = np.asarray(D)
        loss = np.asarray(loss)

        # Clean data
        mask = np.isfinite(N) & np.isfinite(D) & np.isfinite(loss) & (N > 0) & (D > 0) & (loss > 0)
        N, D, L = N[mask], D[mask], loss[mask]

        if len(N) < 5:
            raise ValueError(f"Need at least 5 valid data points, got {len(N)}")

        log_L = np.log(L)

        def huber_loss(residuals: np.ndarray, delta: float) -> np.ndarray:
            abs_r = np.abs(residuals)
            quadratic = 0.5 * residuals**2
            linear = delta * (abs_r - 0.5 * delta)
            return np.where(abs_r <= delta, quadratic, linear)

        # Data-driven initialization for A and B
        # The loss contribution from each term should be roughly (L - E) / 2
        # So A / N^alpha ≈ (L - E) / 2, giving A ≈ (L - E) / 2 * N^alpha
        E_init = L.min() * 0.8
        alpha_init, beta_init = 0.34, 0.28  # Chinchilla paper values
        N_median, D_median = np.median(N), np.median(D)
        loss_headroom = L.mean() - E_init
        A_init = (loss_headroom / 2) * np.power(N_median, alpha_init)
        B_init = (loss_headroom / 2) * np.power(D_median, beta_init)

        # Fit all parameters including E
        def objective(params: np.ndarray) -> float:
            E_param, A, alpha, B, beta = params
            L_pred = chinchilla_parametric_scaling_law(N, D, E_param, A, alpha, B, beta)
            L_pred = np.maximum(L_pred, 1e-10)
            log_residuals = log_L - np.log(L_pred)
            return np.sum(huber_loss(log_residuals, huber_delta))

        p0 = (E_init, A_init, alpha_init, B_init, beta_init)
        scipy_bounds = [(0.0, L.min()), (1e-10, 1e20), (0.01, 2.0), (1e-10, 1e20), (0.01, 2.0)]

        result = minimize(
            objective,
            p0,
            method="L-BFGS-B",
            bounds=scipy_bounds,
            options={"maxiter": 50000, "ftol": 1e-12},
        )
        E_fit, A, alpha, B, beta = result.x

        # R-squared (computed on log scale to match optimization objective)
        L_pred = chinchilla_parametric_scaling_law(N, D, E_fit, A, alpha, B, beta)
        log_L_pred = np.log(np.maximum(L_pred, 1e-10))
        ss_res = np.sum((log_L - log_L_pred) ** 2)
        ss_tot = np.sum((log_L - log_L.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        # Bootstrap confidence intervals
        bootstrap_cis: dict[str, Optional[tuple[float, float]]] = {
            "E_ci": None,
            "A_ci": None,
            "alpha_ci": None,
            "B_ci": None,
            "beta_ci": None,
            "a_opt_ci": None,
            "b_opt_ci": None,
        }

        if bootstrap:
            # Smoothed bootstrap: resample with replacement and add Gaussian noise
            # to loss values. This is more principled than smoothing parameter estimates.
            n_samples = len(N)
            sample_size = int(n_samples * bootstrap_frac)
            bootstrap_params: dict[str, list[float]] = {
                "E": [],
                "A": [],
                "alpha": [],
                "B": [],
                "beta": [],
                "a_opt": [],
                "b_opt": [],
            }

            rng = np.random.default_rng(42)

            # Compute bandwidth for loss smoothing using Silverman's rule
            loss_bandwidth = _silverman_bandwidth(L)

            for _ in range(n_bootstrap):
                # Sample with replacement
                indices = rng.choice(n_samples, size=sample_size, replace=True)
                N_boot = N[indices]
                D_boot = D[indices]
                # Smoothed bootstrap: add Gaussian noise to loss values
                L_boot = L[indices] + rng.normal(0.0, loss_bandwidth, size=sample_size)
                # Ensure loss stays positive
                L_boot = np.maximum(L_boot, 1e-10)

                try:
                    # Fit on smoothed bootstrap sample
                    boot_fit = cls.fit(
                        N=N_boot,
                        D=D_boot,
                        loss=L_boot,
                        huber_delta=huber_delta,
                        bootstrap=False,
                    )
                    bootstrap_params["E"].append(boot_fit.E)
                    bootstrap_params["A"].append(boot_fit.A)
                    bootstrap_params["alpha"].append(boot_fit.alpha)
                    bootstrap_params["B"].append(boot_fit.B)
                    bootstrap_params["beta"].append(boot_fit.beta)
                    # Compute derived optimal exponents
                    bootstrap_params["a_opt"].append(boot_fit.a_opt)
                    bootstrap_params["b_opt"].append(boot_fit.b_opt)
                except (ValueError, RuntimeError):
                    # Skip failed fits
                    continue

            # Compute 10th and 90th percentiles
            for param_name in bootstrap_params:
                if len(bootstrap_params[param_name]) >= 10:
                    p10 = float(np.percentile(bootstrap_params[param_name], 10))
                    p90 = float(np.percentile(bootstrap_params[param_name], 90))
                    bootstrap_cis[f"{param_name}_ci"] = (p10, p90)

        return cls(E=E_fit, A=A, alpha=alpha, B=B, beta=beta, r_squared=r_squared, **bootstrap_cis)

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
            token_counts = [100e9, 1e12, 5e12, 15e12]  # 100B, 1T, 5T, 15T

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
        E_ci_str = f"  [{self.E_ci[0]:.4f}, {self.E_ci[1]:.4f}]" if self.E_ci else ""
        lines.append(f"  Entropy floor (E):     {self.E:.4f}{E_ci_str}")
        A_ci_str = f"  [{self.A_ci[0]:.4f}, {self.A_ci[1]:.4f}]" if self.A_ci else ""
        lines.append(f"  Param coefficient (A): {self.A:.4f}{A_ci_str}")
        alpha_ci_str = (
            f"  [{self.alpha_ci[0]:.4f}, {self.alpha_ci[1]:.4f}]" if self.alpha_ci else ""
        )
        lines.append(
            f"  Param exponent (α):    {self.alpha:.4f}{alpha_ci_str}  {'← Chinchilla: ~0.34' if abs(self.alpha - 0.34) < 0.05 else ''}"
        )
        B_ci_str = f"  [{self.B_ci[0]:.4f}, {self.B_ci[1]:.4f}]" if self.B_ci else ""
        lines.append(f"  Data coefficient (B):  {self.B:.4f}{B_ci_str}")
        beta_ci_str = f"  [{self.beta_ci[0]:.4f}, {self.beta_ci[1]:.4f}]" if self.beta_ci else ""
        lines.append(
            f"  Data exponent (β):     {self.beta:.4f}{beta_ci_str}  {'← Chinchilla: ~0.28' if abs(self.beta - 0.28) < 0.05 else ''}"
        )

        # Compute-optimal allocation (theoretical)
        # For L = E + A/N^α + B/D^β, optimal allocation: N ∝ C^(β/(α+β)), D ∝ C^(α/(α+β))
        lines.append("-" * 40)
        lines.append("  Compute-optimal allocation (theoretical):")
        a_opt_ci_str = (
            f"  [{self.a_opt_ci[0]:.3f}, {self.a_opt_ci[1]:.3f}]" if self.a_opt_ci else ""
        )
        lines.append(f"    N_opt ∝ C^{self.a_opt:.3f}{a_opt_ci_str}  ← Chinchilla: 0.46")
        b_opt_ci_str = (
            f"  [{self.b_opt_ci[0]:.3f}, {self.b_opt_ci[1]:.3f}]" if self.b_opt_ci else ""
        )
        lines.append(f"    D_opt ∝ C^{self.b_opt:.3f}{b_opt_ci_str}  ← Chinchilla: 0.54")
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

    def plot(
        self,
        N: ArrayLike,
        D: ArrayLike,
        loss: ArrayLike,
        C: Optional[ArrayLike] = None,
        title: Optional[str] = None,
        figsize: tuple[int, int] = (16, 10),
    ) -> plt.Figure:
        """
        Plot the parametric scaling law fit with comprehensive diagnostics.

        Creates a 2x3 grid:
        - Top left: Observed vs Predicted loss (calibration)
        - Top middle: Residuals vs Parameters (log scale)
        - Top right: Residuals vs Tokens (log scale)
        - Bottom left: Residual map in N-D space (color = residual)
        - Bottom middle: Fitted loss surface contours with data overlay and Pareto frontier
        - Bottom right: Residual histogram

        Args:
            N: Array of parameter counts
            D: Array of token counts
            loss: Array of actual loss values
            C: Optional array of compute values. If not provided, uses C = 6*N*D.
            title: Optional title for the plot
            figsize: Figure size (width, height)

        Returns:
            matplotlib Figure object
        """
        N = np.asarray(N)
        D = np.asarray(D)
        loss = np.asarray(loss)
        if C is None:
            C = 6 * N * D
        else:
            C = np.asarray(C)

        L_pred = self.predict_loss(N, D)
        residuals = loss - L_pred
        rmse = np.sqrt(np.mean(residuals**2))

        fig, axes = plt.subplots(2, 3, figsize=figsize)

        # ──────────────────────────────────────────────────────────────────────
        # Top left: Calibration (Observed vs Predicted)
        # ──────────────────────────────────────────────────────────────────────
        ax = axes[0, 0]
        ax.scatter(L_pred, loss, alpha=0.7, s=50, edgecolors="black", linewidths=0.5)
        min_val, max_val = min(L_pred.min(), loss.min()), max(L_pred.max(), loss.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect fit")
        ax.set_xlabel("Predicted Loss", fontsize=11)
        ax.set_ylabel("Observed Loss", fontsize=11)
        ax.set_title("Calibration", fontsize=12, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
        stats_text = f"R² = {self.r_squared:.4f}\nRMSE = {rmse:.4f}" if self.r_squared else ""
        ax.text(
            0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment="top"
        )

        # ──────────────────────────────────────────────────────────────────────
        # Top middle: Residuals vs N (log scale)
        # ──────────────────────────────────────────────────────────────────────
        ax = axes[0, 1]
        ax.scatter(N, residuals, alpha=0.7, s=50, c="steelblue", edgecolors="black", linewidths=0.5)
        ax.axhline(y=0, color="r", linestyle="--", linewidth=1.5)
        ax.set_xscale("log")
        ax.set_xlabel("Parameters (N)", fontsize=11)
        ax.set_ylabel("Residual (Obs - Pred)", fontsize=11)
        ax.set_title("Residuals vs Parameters", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, which="both")

        # ──────────────────────────────────────────────────────────────────────
        # Top right: Residuals vs D (log scale)
        # ──────────────────────────────────────────────────────────────────────
        ax = axes[0, 2]
        ax.scatter(
            D, residuals, alpha=0.7, s=50, c="forestgreen", edgecolors="black", linewidths=0.5
        )
        ax.axhline(y=0, color="r", linestyle="--", linewidth=1.5)
        ax.set_xscale("log")
        ax.set_xlabel("Tokens (D)", fontsize=11)
        ax.set_ylabel("Residual (Obs - Pred)", fontsize=11)
        ax.set_title("Residuals vs Tokens", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, which="both")

        # ──────────────────────────────────────────────────────────────────────
        # Bottom left: Residual map in N-D space
        # ──────────────────────────────────────────────────────────────────────
        ax = axes[1, 0]
        # Use diverging colormap centered at 0
        abs_max = max(abs(residuals.min()), abs(residuals.max()))
        scatter = ax.scatter(
            N,
            D,
            c=residuals,
            cmap="RdBu_r",
            vmin=-abs_max,
            vmax=abs_max,
            s=60,
            edgecolors="black",
            linewidths=0.5,
            alpha=0.8,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Parameters (N)", fontsize=11)
        ax.set_ylabel("Tokens (D)", fontsize=11)
        ax.set_title("Residual Map (N-D Space)", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, which="both")
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Residual", fontsize=10)

        # ──────────────────────────────────────────────────────────────────────
        # Bottom middle: Loss surface contours with data overlay
        # ──────────────────────────────────────────────────────────────────────
        ax = axes[1, 1]
        # Create grid for contour plot
        N_grid = np.logspace(np.log10(N.min()), np.log10(N.max()), 50)
        D_grid = np.logspace(np.log10(D.min()), np.log10(D.max()), 50)
        N_mesh, D_mesh = np.meshgrid(N_grid, D_grid)
        L_mesh = self.predict_loss(N_mesh.ravel(), D_mesh.ravel()).reshape(N_mesh.shape)

        # Plot contours (in log space for visualization)
        contour = ax.contourf(
            np.log10(N_mesh),
            np.log10(D_mesh),
            L_mesh,
            levels=20,
            cmap="viridis",
            alpha=0.8,
        )
        ax.contour(
            np.log10(N_mesh),
            np.log10(D_mesh),
            L_mesh,
            levels=10,
            colors="white",
            linewidths=0.5,
            alpha=0.5,
        )
        # Overlay actual data points, colored by observed loss
        scatter = ax.scatter(
            np.log10(N),
            np.log10(D),
            c=loss,
            cmap="viridis",
            s=60,
            edgecolors="white",
            linewidths=1.5,
            vmin=L_mesh.min(),
            vmax=L_mesh.max(),
        )
        # Plot empirical Pareto frontier (compute-optimal points from data)
        try:
            N_frontier, D_frontier, _, _ = _extract_pareto_frontier(N, D, C, loss, n_bins=15)
            # Sort by N for clean line plotting
            sort_idx = np.argsort(N_frontier)
            ax.plot(
                np.log10(N_frontier[sort_idx]),
                np.log10(D_frontier[sort_idx]),
                "r-",
                linewidth=2.5,
                marker="o",
                markersize=6,
                markerfacecolor="white",
                markeredgecolor="red",
                markeredgewidth=1.5,
                label="Pareto frontier",
            )
            ax.legend(loc="lower right", fontsize=9)
        except ValueError:
            pass  # Not enough data for Pareto frontier
        ax.set_xlabel("log₁₀(Parameters)", fontsize=11)
        ax.set_ylabel("log₁₀(Tokens)", fontsize=11)
        ax.set_title("Fitted Loss Surface", fontsize=12, fontweight="bold")
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label("Loss", fontsize=10)

        # ──────────────────────────────────────────────────────────────────────
        # Bottom right: Residual histogram
        # ──────────────────────────────────────────────────────────────────────
        ax = axes[1, 2]
        n_bins = min(30, max(10, len(residuals) // 5))
        ax.hist(residuals, bins=n_bins, color="coral", edgecolor="black", alpha=0.7, density=True)
        ax.axvline(x=0, color="r", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Residual (Obs - Pred)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title("Residual Distribution", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        # Add normal fit for reference
        mu, std = np.mean(residuals), np.std(residuals)
        x_norm = np.linspace(residuals.min(), residuals.max(), 100)
        y_norm = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_norm - mu) / std) ** 2)
        ax.plot(x_norm, y_norm, "b-", linewidth=2, label=f"Normal(μ={mu:.3f}, σ={std:.3f})")
        ax.legend(fontsize=9)

        # ──────────────────────────────────────────────────────────────────────
        # Overall title
        # ──────────────────────────────────────────────────────────────────────
        if title:
            fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        else:
            fig.suptitle(
                f"L(N,D) = {self.E:.3f} + {self.A:.3f}/N^{self.alpha:.3f} + {self.B:.3f}/D^{self.beta:.3f}",
                fontsize=12,
                y=1.02,
            )

        plt.tight_layout()
        return fig

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

                # Actual loss comparison (includes E) with efficiency context
                _, _, interp = fit.loss_advantage(
                    baseline, N, D, compact=True, efficiency_context=(data_mult, param_mult)
                )

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
