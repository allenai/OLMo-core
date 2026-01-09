from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from olmo_core.data.composable.utils import format_token_count
from olmo_core.model_ladder.utils import format_count


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

    Method (Chinchilla Approach 1):
    1. For each parameter count N, train models to different amounts of data D
    2. Bin observations by compute C (log-spaced)
    3. For each compute bin, find the observation with MINIMUM loss (the Pareto frontier)
    4. Fit power laws for L(C), N_opt(C), and D_opt(C) on these frontier points only

    This matches the Chinchilla paper: "for each FLOP count, we determine which run
    achieves the lowest loss" and then fit on those optimal (N, D) configurations.

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
        n_bins: int = 20,
        bootstrap: bool = False,
        n_bootstrap: int = 100,
        bootstrap_frac: float = 0.8,
    ) -> "ChinchillaIsoParamFit":
        """
        Fit the IsoParam scaling laws using Chinchilla Approach 1.

        This method first extracts the Pareto frontier (minimum loss for each compute bin),
        then fits power laws on those optimal points:

        1. Bin observations by compute (log-spaced)
        2. For each bin, find the observation with minimum loss (the optimal N, D for that compute)
        3. Fit L(C) = E + A / C^alpha on frontier (C, L) pairs (Huber loss in log space)
        4. Fit N_opt(C) = G * C^a on frontier (C, N) pairs (linear regression in log-log space)
        5. Fit D_opt(C) = H * C^b on frontier (C, D) pairs (linear regression in log-log space)

        This matches the Chinchilla paper's Approach 1: "for each FLOP count, we determine
        which run achieves the lowest loss."

        Args:
            N: Array of parameter counts (one per observation)
            D: Array of token counts (one per observation)
            C: Array of compute values (one per observation, e.g., FLOPs or petaFLOPs)
            loss: Array of loss values (one per observation)
            huber_delta: Delta parameter for Huber loss in L(C) fitting
            n_bins: Number of log-spaced bins for grouping compute values (default 20)
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

        # Extract Pareto frontier: minimum loss for each compute bin
        # This is the key step of Chinchilla Approach 1
        N_frontier, D_frontier, C_frontier, L_frontier = _extract_pareto_frontier(
            N_valid, D_valid, C_valid, L_valid, n_bins=n_bins
        )

        if len(N_frontier) < 3:
            raise ValueError(
                f"Need at least 3 frontier points, got {len(N_frontier)}. "
                "Try reducing n_bins or adding more data."
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
                        n_bins=n_bins,
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
        n_bins: int = 20,
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
        points (minimum loss per compute bin, used for fitting) are highlighted.

        If N and D are not provided, falls back to a simple 2-panel loss plot.

        Args:
            C: Array of compute values (same units used in fitting)
            loss: Array of actual loss values
            N: Optional array of parameter counts (for N_opt fit plot)
            D: Optional array of token counts (for D_opt fit plot)
            n_bins: Number of bins for frontier extraction (should match fitting)
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

            # Extract frontier for highlighting
            N_frontier, D_frontier, C_frontier, L_frontier = _extract_pareto_frontier(
                N, D, C, loss, n_bins=n_bins
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
        How much more data-efficient is this model vs another when beta is constant?
        Returns k such that this model with D tokens matches other with k*D tokens.
        """
        if abs(self.beta - other.beta) < tol:
            return (other.B / self.B) ** (1 / self.beta)
        raise ValueError(f"Beta values are too different: {self.beta} vs {other.beta}")

    def effective_data_multiplier(self, other: "ChinchillaParametricFit", D: float) -> float:
        """
        At training size D, how much more data-efficient is this model vs another?
        Returns k such that this model with D tokens matches other with k*D tokens.

        When beta values are similar, k is constant. When they differ, k depends on D.
        """
        return (other.B / self.B) ** (1 / other.beta) * D ** ((self.beta - other.beta) / other.beta)

    def effective_param_multiplier_at_constant_alpha(
        self, other: "ChinchillaParametricFit", tol: float = 0.01
    ) -> float:
        """
        How much more parameter-efficient is this model vs another when alpha is constant?
        Returns k such that this model with N params matches other with k*N params.
        """
        if abs(self.alpha - other.alpha) < tol:
            return (other.A / self.A) ** (1 / self.alpha)
        raise ValueError(f"Alpha values are too different: {self.alpha} vs {other.alpha}")

    def effective_param_multiplier(self, other: "ChinchillaParametricFit", N: float) -> float:
        """
        At training size N, how much more parameter-efficient is this model vs another?
        Returns k such that this model with N params matches other with k*N params.
        """
        return (other.A / self.A) ** (1 / other.alpha) * N ** (
            (self.alpha - other.alpha) / other.alpha
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


def fit_chinchilla_parametric_from_df(
    df: pd.DataFrame,
    N_col: str = "num_params",
    D_col: str = "throughput/total tokens",
    loss_col: str = "eval/lm/pile-validation/CE loss",
    **kwargs,
) -> ChinchillaParametricFit:
    """Fit Chinchilla scaling law from DataFrame."""
    return ChinchillaParametricFit.fit(
        df[N_col].values, df[D_col].values, df[loss_col].values, **kwargs
    )


def fit_chinchilla_isoparam_from_df(
    df: pd.DataFrame,
    N_col: str = "num_params",
    D_col: str = "throughput/total tokens",
    C_col: str = "throughput/total petaflops",
    loss_col: str = "eval/lm/pile-validation/CE loss",
    **kwargs,
) -> ChinchillaIsoParamFit:
    """
    Convenience function to fit Chinchilla Approach 1 scaling law from a DataFrame.

    Args:
        df: DataFrame containing N, D, C, and loss columns
        N_col: Name of the parameter count column
        D_col: Name of the token count column
        C_col: Name of the compute column
        loss_col: Name of the loss column
        **kwargs: Additional arguments passed to ChinchillaIsoParamFit.fit

    Returns:
        ChinchillaIsoParamFit with fitted parameters
    """
    return ChinchillaIsoParamFit.fit(
        N=df[N_col].values,
        D=df[D_col].values,
        C=df[C_col].values,
        loss=df[loss_col].values,
        **kwargs,
    )


def find_crossover_compute(
    fit1: ChinchillaIsoParamFit,
    fit2: ChinchillaIsoParamFit,
    C_min: float = 1e-3,
    C_max: float = 1e12,
) -> Optional[float]:
    """
    Find the compute value where two scaling laws cross over (give equal loss).

    Returns the compute C where fit1.predict(C) == fit2.predict(C), or None if
    they don't cross in the given range.

    Args:
        fit1: First scaling law fit
        fit2: Second scaling law fit
        C_min: Minimum compute to search
        C_max: Maximum compute to search

    Returns:
        Crossover compute value, or None if no crossover exists in range
    """
    from scipy.optimize import brentq

    def loss_diff(log_C):
        C = 10**log_C
        return fit1.predict_loss(np.array([C]))[0] - fit2.predict_loss(np.array([C]))[0]

    # Check if there's a sign change (crossover exists)
    log_C_min, log_C_max = np.log10(C_min), np.log10(C_max)

    try:
        diff_min = loss_diff(log_C_min)
        diff_max = loss_diff(log_C_max)

        # If same sign at both ends, no crossover in range
        if diff_min * diff_max > 0:
            return None

        # Find the crossover point
        log_C_cross = brentq(loss_diff, log_C_min, log_C_max)
        return 10**log_C_cross
    except (ValueError, RuntimeError):
        return None


def find_crossover_parameter_count_at_constant_D(
    fit1: ChinchillaParametricFit,
    fit2: ChinchillaParametricFit,
    D: float,
    N_min: float = 1e6,
    N_max: float = 1e12,
) -> Optional[float]:
    """
    Find the parameter count where two scaling laws cross over (give equal loss) at a fixed data size.

    For L(N, D) = E + A/N^α + B/D^β, finds N where fit1 and fit2 give equal loss.

    Args:
        fit1: First scaling law fit
        fit2: Second scaling law fit
        D: Data size (number of tokens)
        N_min: Minimum parameter count to search
        N_max: Maximum parameter count to search

    Returns:
        Crossover parameter count, or None if no crossover exists in range
    """
    from scipy.optimize import brentq

    D_arr = np.array([D])

    def loss_diff(log_N: float) -> float:
        N = 10**log_N
        N_arr = np.array([N])
        return fit1.predict_loss(N_arr, D_arr)[0] - fit2.predict_loss(N_arr, D_arr)[0]

    log_N_min, log_N_max = np.log10(N_min), np.log10(N_max)

    try:
        diff_min = loss_diff(log_N_min)
        diff_max = loss_diff(log_N_max)

        # If same sign at both ends, no crossover in range
        if diff_min * diff_max > 0:
            return None

        # Find the crossover point
        log_N_cross = brentq(loss_diff, log_N_min, log_N_max)
        return 10**log_N_cross
    except (ValueError, RuntimeError):
        return None


def find_crossover_token_count_at_constant_N(
    fit1: ChinchillaParametricFit,
    fit2: ChinchillaParametricFit,
    N: float,
    D_min: float = 1e9,
    D_max: float = 1e15,
) -> Optional[float]:
    """
    Find the token count where two scaling laws cross over (give equal loss) at a fixed parameter count.

    For L(N, D) = E + A/N^α + B/D^β, finds D where fit1 and fit2 give equal loss.

    Args:
        fit1: First scaling law fit
        fit2: Second scaling law fit
        N: Parameter count
        D_min: Minimum token count to search
        D_max: Maximum token count to search

    Returns:
        Crossover token count, or None if no crossover exists in range
    """
    from scipy.optimize import brentq

    N_arr = np.array([N])

    def loss_diff(log_D: float) -> float:
        D = 10**log_D
        D_arr = np.array([D])
        return fit1.predict_loss(N_arr, D_arr)[0] - fit2.predict_loss(N_arr, D_arr)[0]

    log_D_min, log_D_max = np.log10(D_min), np.log10(D_max)

    try:
        diff_min = loss_diff(log_D_min)
        diff_max = loss_diff(log_D_max)

        # If same sign at both ends, no crossover in range
        if diff_min * diff_max > 0:
            return None

        # Find the crossover point
        log_D_cross = brentq(loss_diff, log_D_min, log_D_max)
        return 10**log_D_cross
    except (ValueError, RuntimeError):
        return None
