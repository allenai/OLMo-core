from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.optimize import curve_fit


@dataclass
class ScalingLawFit(ABC):
    """Results from fitting a scaling law: L = E + A / N^alpha + B / D^beta"""

    @abstractmethod
    def effective_compute_multiplier(self, other: "ScalingLawFit") -> float:
        """Compute how much more effective this scaling law is vs another."""
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        """Return a string representation of the fit."""
        raise NotImplementedError


@dataclass
class ComputeScalingLawFit:
    """Results from fitting a scaling law: L = E + A / C^alpha"""

    E: float  # Irreducible loss (entropy floor)
    A: float  # Scale coefficient
    alpha: float  # Power law exponent
    E_err: Optional[float] = None  # Standard error on E
    A_err: Optional[float] = None  # Standard error on A
    alpha_err: Optional[float] = None  # Standard error on alpha
    r_squared: Optional[float] = None  # Coefficient of determination

    def predict(self, C: np.ndarray) -> np.ndarray:
        """Predict loss for given compute values."""
        return self.E + self.A / np.power(C, self.alpha)

    def effective_compute_multiplier(self, other: "ComputeScalingLawFit") -> float:
        """
        Compute how much more effective this scaling law is vs another.
        Returns multiplier k such that this law at C achieves same loss as other at k*C.
        """
        # At same loss L: E1 + A1/C1^a1 = E2 + A2/C2^a2
        # If we assume same alpha for simplicity, k = (A2/A1)^(1/alpha)
        if abs(self.alpha - other.alpha) < 0.01:
            return (other.A / self.A) ** (1 / self.alpha)
        # For different alphas, return ratio at a reference point
        return other.A / self.A

    def __repr__(self) -> str:
        parts = [f"L = {self.E:.4f} + {self.A:.4f} / C^{self.alpha:.4f}"]
        if self.r_squared is not None:
            parts.append(f"R² = {self.r_squared:.4f}")
        return " | ".join(parts)

    @classmethod
    def fit(
        cls,
        compute: ArrayLike,
        loss: ArrayLike,
        p0: Optional[tuple] = None,
        bounds: Optional[tuple] = None,
    ) -> "ComputeScalingLawFit":
        """
        Fit a Chinchilla-style scaling law: L = E + A / C^alpha

        Args:
            compute: Array of compute values (e.g., petaflops)
            loss: Array of loss values
            p0: Initial guess for (E, A, alpha). If None, will be estimated.
            bounds: Bounds for parameters as ((E_min, A_min, alpha_min), (E_max, A_max, alpha_max)).
                    If None, reasonable defaults will be used.

        Returns:
            ComputeScalingLawFit with fitted parameters and uncertainties
        """
        # Convert to numpy arrays
        compute = np.asarray(compute)
        loss = np.asarray(loss)

        # Clean data
        mask = np.isfinite(compute) & np.isfinite(loss) & (compute > 0) & (loss > 0)
        C = compute[mask]
        L = loss[mask]

        if len(C) < 3:
            raise ValueError(f"Need at least 3 valid data points, got {len(C)}")

        # Default initial guess
        if p0 is None:
            E_init = L.min() * 0.8  # Assume irreducible loss is below minimum observed
            A_init = (L.max() - E_init) * C.min() ** 0.1  # Rough estimate
            alpha_init = 0.1  # Typical value for compute scaling
            p0 = (E_init, A_init, alpha_init)

        # Default bounds
        if bounds is None:
            bounds = (
                [0, 0, 0.01],  # Lower bounds: E >= 0, A >= 0, alpha > 0
                [L.min(), np.inf, 1.0],  # Upper: E < min loss, alpha typically < 1
            )

        # Fit the scaling law
        popt, pcov = curve_fit(cls.loss, (C,), L, p0=p0, bounds=bounds, maxfev=10000)
        E, A, alpha = popt

        # Extract uncertainties from covariance matrix
        perr = np.sqrt(np.diag(pcov))
        E_err, A_err, alpha_err = perr

        # Compute R-squared
        L_pred = cls.loss((C,), E=E, A=A, alpha=alpha)
        ss_res = np.sum((L - L_pred) ** 2)
        ss_tot = np.sum((L - L.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        return cls(
            E=E,
            A=A,
            alpha=alpha,
            E_err=E_err,
            A_err=A_err,
            alpha_err=alpha_err,
            r_squared=r_squared,
        )


@dataclass
class ChinchillaFit:
    """
    Results from fitting the full Chinchilla scaling law:
    L(N, D) = E + A / N^alpha + B / D^beta

    Where:
        N = number of parameters
        D = number of training tokens
        E = irreducible loss (entropy floor)
        A, alpha = parameters scaling coefficients
        B, beta = data scaling coefficients
    """

    E: float  # Irreducible loss
    A: float  # Parameter scale coefficient
    alpha: float  # Parameter scaling exponent
    B: float  # Data scale coefficient
    beta: float  # Data scaling exponent
    r_squared: Optional[float] = None

    # Standard errors (optional)
    E_err: Optional[float] = None
    A_err: Optional[float] = None
    alpha_err: Optional[float] = None
    B_err: Optional[float] = None
    beta_err: Optional[float] = None

    @classmethod
    def loss(
        cls, N: np.ndarray, D: np.ndarray, E: float, A: float, alpha: float, B: float, beta: float
    ) -> np.ndarray:
        """Compute loss for given parameter count and token count."""
        return E + A / np.power(N, alpha) + B / np.power(D, beta)

    def predict(self, N: np.ndarray, D: np.ndarray) -> np.ndarray:
        """Predict loss for given parameter count and token count."""
        return self.loss(N, D, E=self.E, A=self.A, alpha=self.alpha, B=self.B, beta=self.beta)

    def predict_at_xC(
        self, N: np.ndarray, xC: float, tokens_per_param_at_1xC: float = 20.0
    ) -> np.ndarray:
        """Predict loss for given N at a specific xC (Chinchilla multiple)."""
        D = xC * tokens_per_param_at_1xC * N
        return self.loss(N, D, E=self.E, A=self.A, alpha=self.alpha, B=self.B, beta=self.beta)

    def optimal_allocation(
        self, C_flops: float, flops_per_param_token: float = 6.0
    ) -> Tuple[float, float]:
        """
        Given a compute budget C (in FLOPs), return optimal (N, D) allocation.

        Chinchilla optimal: allocate compute such that marginal improvements from
        scaling N vs D are equal.

        Returns (N_opt, D_opt)
        """
        # C = flops_per_param_token * N * D
        # Optimal allocation minimizes L subject to N*D = C/6
        # Taking derivatives and solving: N_opt / D_opt = (alpha * B) / (beta * A)
        ratio = (self.alpha * self.B) / (self.beta * self.A)
        # N * D = C / flops_per_param_token
        # N = ratio * D  =>  ratio * D^2 = C / flops_per_param_token
        D_opt = np.sqrt(C_flops / (flops_per_param_token * ratio))
        N_opt = ratio * D_opt
        return N_opt, D_opt

    def effective_data_multiplier(self, other: "ChinchillaFit") -> float:
        """
        How much more data-efficient is this model vs another?
        Returns k such that this model with D tokens matches other with k*D tokens.
        """
        if abs(self.beta - other.beta) < 0.01:
            return (other.B / self.B) ** (1 / self.beta)
        return other.B / self.B

    def effective_param_multiplier(self, other: "ChinchillaFit") -> float:
        """
        How much more parameter-efficient is this model vs another?
        Returns k such that this model with N params matches other with k*N params.
        """
        if abs(self.alpha - other.alpha) < 0.01:
            return (other.A / self.A) ** (1 / self.alpha)
        return other.A / self.A

    @classmethod
    def fit(
        cls,
        N: ArrayLike,
        D: ArrayLike,
        loss: ArrayLike,
        p0: Optional[tuple] = None,
        bounds: Optional[tuple] = None,
    ) -> "ChinchillaFit":
        """
        Fit the full Chinchilla scaling law: L = E + A/N^alpha + B/D^beta

        Args:
            N: Array of parameter counts
            D: Array of token counts
            loss: Array of loss values
            p0: Initial guess for (E, A, alpha, B, beta)
            bounds: Parameter bounds

        Returns:
            ChinchillaFit with fitted parameters
        """
        N = np.asarray(N)
        D = np.asarray(D)
        loss = np.asarray(loss)

        # Clean data
        mask = np.isfinite(N) & np.isfinite(D) & np.isfinite(loss) & (N > 0) & (D > 0) & (loss > 0)
        N, D, L = N[mask], D[mask], loss[mask]

        if len(N) < 5:
            raise ValueError(f"Need at least 5 valid data points, got {len(N)}")

        # Default initial guess based on Chinchilla paper values
        if p0 is None:
            E_init = L.min() * 0.8
            A_init = 400.0  # Chinchilla: ~400
            alpha_init = 0.34  # Chinchilla: ~0.34
            B_init = 400.0  # Chinchilla: ~400
            beta_init = 0.28  # Chinchilla: ~0.28
            p0 = (E_init, A_init, alpha_init, B_init, beta_init)

        # Default bounds
        if bounds is None:
            bounds = (
                [0, 1, 0.1, 1, 0.1],  # Lower bounds
                [L.min(), 1e6, 1.0, 1e6, 1.0],  # Upper bounds
            )

        # Wrapper for curve_fit (needs single array input)
        def model(X, E, A, alpha, B, beta):
            N, D = X
            return cls.loss(N, D, E, A, alpha, B, beta)

        # Fit
        popt, pcov = curve_fit(model, (N, D), L, p0=p0, bounds=bounds, maxfev=50000)
        E, A, alpha, B, beta = popt

        # Uncertainties
        perr = np.sqrt(np.diag(pcov))

        # R-squared
        L_pred = cls.loss(N, D, *popt)
        ss_res = np.sum((L - L_pred) ** 2)
        ss_tot = np.sum((L - L.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        return cls(
            E=E,
            A=A,
            alpha=alpha,
            B=B,
            beta=beta,
            r_squared=r_squared,
            E_err=perr[0],
            A_err=perr[1],
            alpha_err=perr[2],
            B_err=perr[3],
            beta_err=perr[4],
        )

    def __repr__(self) -> str:
        return (
            f"L = {self.E:.4f} + {self.A:.4f}/N^{self.alpha:.4f} + {self.B:.4f}/D^{self.beta:.4f}"
            f" | R²={self.r_squared:.4f}"
            if self.r_squared
            else ""
        )


def fit_chinchilla_from_df(
    df: pd.DataFrame,
    N_col: str = "num_params",
    D_col: str = "throughput/total tokens",
    loss_col: str = "eval/lm/pile-validation/CE loss",
    **kwargs,
) -> ChinchillaFit:
    """Fit Chinchilla scaling law from DataFrame."""
    return ChinchillaFit.fit(df[N_col].values, df[D_col].values, df[loss_col].values, **kwargs)


def fit_compute_scaling_law_from_df(
    df: pd.DataFrame,
    compute_col: str = "throughput/total petaflops",
    loss_col: str = "eval/lm/pile-validation/CE loss",
    **kwargs,
) -> ComputeScalingLawFit:
    """
    Convenience function to fit scaling law directly from a DataFrame.

    Args:
        df: DataFrame containing compute and loss columns
        compute_col: Name of the compute column
        loss_col: Name of the loss column
        **kwargs: Additional arguments passed to fit_scaling_law

    Returns:
        ScalingLawFit with fitted parameters
    """
    return ComputeScalingLawFit.fit(df[compute_col].values, df[loss_col].values, **kwargs)


def find_crossover_compute(
    fit1: ScalingLawFit,
    fit2: ScalingLawFit,
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
        return fit1.predict(np.array([C]))[0] - fit2.predict(np.array([C]))[0]

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


def fit_chinchilla_shared_E(
    ladder_dfs: dict,
    N_col: str = "num_params",
    D_col: str = "throughput/total tokens",
    loss_col: str = "eval/lm/pile-validation/CE loss",
):
    """
    Fit Chinchilla scaling law with SHARED E (irreducible loss) across all ladders.
    Each ladder gets its own A, α, B, β.

    This is more principled because all variants train on the same data,
    so they should have the same entropy floor.
    """
    ladder_names = list(ladder_dfs.keys())

    # Collect all data
    all_data = {}
    min_loss_overall = float("inf")
    for name, df_ladder in ladder_dfs.items():
        N = df_ladder[N_col].values.astype(float)
        D = df_ladder[D_col].values.astype(float)
        L = df_ladder[loss_col].values.astype(float)
        mask = np.isfinite(N) & np.isfinite(D) & np.isfinite(L) & (N > 0) & (D > 0) & (L > 0)
        all_data[name] = (N[mask], D[mask], L[mask])
        min_loss_overall = min(min_loss_overall, L[mask].min())

    # Parameters: [E_shared, A_0, α_0, B_0, β_0, A_1, α_1, B_1, β_1, ...]
    # Total: 1 shared + 4 * n_ladders

    def loss_fn(params):
        E = params[0]
        total_loss = 0.0
        idx = 1
        for name in ladder_names:
            A, alpha, B, beta = params[idx], params[idx + 1], params[idx + 2], params[idx + 3]
            idx += 4
            N, D, L = all_data[name]
            L_pred = E + A / np.power(N, alpha) + B / np.power(D, beta)
            total_loss += np.sum((L - L_pred) ** 2)
        return total_loss

    # Initial guess - use per-ladder estimates
    E_init = min_loss_overall * 0.9  # Start slightly below min observed
    p0 = [E_init]
    for name in ladder_names:
        N, D, L = all_data[name]
        # Rough initial guesses based on data range
        # At mid-point, L - E ≈ A/N^α + B/D^β
        # Assume roughly equal contribution from each term
        mid_L = np.median(L)
        mid_N = np.median(N)
        mid_D = np.median(D)
        residual = (mid_L - E_init) / 2  # Split between N and D terms

        # A/N^0.34 ≈ residual => A ≈ residual * N^0.34
        A_init = residual * (mid_N**0.34)
        B_init = residual * (mid_D**0.28)

        p0.extend([A_init, 0.34, B_init, 0.28])

    # Bounds
    bounds = [(0.5, min_loss_overall)]  # E must be below min observed loss
    for _ in ladder_names:
        bounds.extend(
            [
                (1, 1e6),  # A
                (0.1, 0.8),  # alpha
                (1, 1e6),  # B
                (0.1, 0.8),  # beta
            ]
        )

    # Optimize
    result = minimize(
        loss_fn, p0, bounds=bounds, method="L-BFGS-B", options={"maxiter": 50000, "ftol": 1e-12}
    )

    # Extract results
    E_shared = result.x[0]

    fits = {}
    idx = 1
    total_ss_res = 0.0
    total_ss_tot = 0.0
    for name in ladder_names:
        A, alpha, B, beta = result.x[idx], result.x[idx + 1], result.x[idx + 2], result.x[idx + 3]
        idx += 4
        N, D, L = all_data[name]
        L_pred = E_shared + A / np.power(N, alpha) + B / np.power(D, beta)
        ss_res = np.sum((L - L_pred) ** 2)
        ss_tot = np.sum((L - L.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        total_ss_res += ss_res
        total_ss_tot += ss_tot
        fits[name] = {"E": E_shared, "A": A, "alpha": alpha, "B": B, "beta": beta, "r_squared": r2}

    overall_r2 = 1 - total_ss_res / total_ss_tot

    return {
        "E_shared": E_shared,
        "fits": fits,
        "overall_r_squared": overall_r2,
        "success": result.success,
    }
