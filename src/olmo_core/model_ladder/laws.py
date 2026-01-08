from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from olmo_core.data.composable.utils import format_token_count
from olmo_core.model_ladder.utils import format_count
from olmo_core.utils import format_float


def chinchilla_parametric_scaling_law(
    N: np.ndarray, D: np.ndarray, E: float, A: float, alpha: float, B: float, beta: float
) -> np.ndarray:
    """Compute loss for given parameter count and token count using the Chinchilla scaling law."""
    return E + A / np.power(N, alpha) + B / np.power(D, beta)


@dataclass
class ChinchillaIsoParamFit:
    """Results from fitting the Chinchilla scaling law with constant parameter count (Approach 1).

    Input: Observations N_i, D_ij, L_ij, C_ij for model size i and data size j

    Output: Power Laws
        L(C) = E + A / C^alpha           (loss as function of compute)
        N_opt(C) = G * C^a               (optimal parameters as function of compute)
        D_opt(C) = H * C^b               (optimal tokens as function of compute)

    Method:
    1. For each parameter count N, train different models to different amounts of data
    2. For each observation (N, D, C, L), use the actual measured compute C
    3. Fit power laws for L(C), N_opt(C), and D_opt(C) directly on observations

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
        E: Optional[float] = None,
        huber_delta: float = 1e-3,
    ) -> "ChinchillaIsoParamFit":
        """
        Fit the IsoParam scaling laws using Chinchilla Approach 1.

        This method fits power laws directly on the provided observations:
        1. L(C) = E + A / C^alpha on all (C, L) pairs (Huber loss in log space)
        2. N_opt(C) = G * C^a on all (C, N) pairs (linear regression in log-log space)
        3. D_opt(C) = H * C^b on all (C, D) pairs (linear regression in log-log space)

        Args:
            N: Array of parameter counts (one per observation)
            D: Array of token counts (one per observation)
            C: Array of compute values (one per observation, e.g., FLOPs or petaFLOPs)
            loss: Array of loss values (one per observation)
            E: If provided, use this fixed value for E (irreducible loss) instead of fitting it
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

        log_C = np.log(C_valid)
        log_L = np.log(L_valid)

        def huber_loss(residuals: np.ndarray, delta: float) -> np.ndarray:
            abs_r = np.abs(residuals)
            return np.where(abs_r <= delta, 0.5 * residuals**2, delta * (abs_r - 0.5 * delta))

        if E is not None:
            # Fit L(C) = E + A / C^alpha with fixed E (only fit A and alpha)
            E_fixed = E

            def loss_objective_fixed_E(params: np.ndarray) -> float:
                A, alpha = params
                L_pred = E_fixed + A / np.power(C_valid, alpha)
                L_pred = np.maximum(L_pred, 1e-10)
                log_residuals = log_L - np.log(L_pred)
                return np.sum(huber_loss(log_residuals, huber_delta))

            A_init = (L_valid.max() - E_fixed) * C_valid.min() ** 0.1
            alpha_init = 0.1

            loss_result = minimize(
                loss_objective_fixed_E,
                [A_init, alpha_init],
                method="L-BFGS-B",
                bounds=[(1e-10, 1e10), (0.01, 1.0)],
            )
            A, alpha = loss_result.x
            E_fit = E_fixed
        else:
            # Fit L(C) = E + A / C^alpha (fit all three parameters)
            def loss_objective(params: np.ndarray) -> float:
                E_param, A, alpha = params
                L_pred = E_param + A / np.power(C_valid, alpha)
                L_pred = np.maximum(L_pred, 1e-10)
                log_residuals = log_L - np.log(L_pred)
                return np.sum(huber_loss(log_residuals, huber_delta))

            E_init = L_valid.min() * 0.8
            A_init = (L_valid.max() - E_init) * C_valid.min() ** 0.1
            alpha_init = 0.1

            loss_result = minimize(
                loss_objective,
                [E_init, A_init, alpha_init],
                method="L-BFGS-B",
                bounds=[(0.0, L_valid.min()), (1e-10, 1e10), (0.01, 1.0)],
            )
            E_fit, A, alpha = loss_result.x

        # R-squared for loss fit
        L_pred = E_fit + A / np.power(C_valid, alpha)
        log_L_pred = np.log(np.maximum(L_pred, 1e-10))
        ss_res_L = np.sum((log_L - log_L_pred) ** 2)
        ss_tot_L = np.sum((log_L - log_L.mean()) ** 2)
        r2_loss = 1 - ss_res_L / ss_tot_L if ss_tot_L > 0 else 0.0

        # 2. Fit N_opt(C) = G * C^a using linear regression in log-log space
        # log(N) = log(G) + a * log(C)
        log_N = np.log(N_valid)
        a, log_G = np.polyfit(log_C, log_N, 1)
        G = np.exp(log_G)

        # R-squared for N fit
        log_N_pred = log_G + a * log_C
        ss_res_N = np.sum((log_N - log_N_pred) ** 2)
        ss_tot_N = np.sum((log_N - log_N.mean()) ** 2)
        r2_N = 1 - ss_res_N / ss_tot_N if ss_tot_N > 0 else 0.0

        # 3. Fit D_opt(C) = H * C^b using linear regression in log-log space
        # log(D) = log(H) + b * log(C)
        log_D = np.log(D_valid)
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
        )

    def report(self, compute_examples: Optional[list[float]] = None) -> str:
        if compute_examples is None:
            compute_examples = [
                1e19,
                1e20,  # GPT-2 1.5B was 8e19 FLOPs
                1e21,
                1e22,  # Llama1 7B was 4e22 FLOPs
                1e23,  # GPT-3 175B was 3e23 FLOPs
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
            lines.append("  ⚠ Tokens/param ratio varies with compute (a ≠ b)")

        # Example predictions
        lines.append("\nCOMPUTE-OPTIMAL PREDICTIONS")
        lines.append("-" * 40)
        lines.append(
            f"  {'Compute':>12}  {'Params':>12}  {'Tokens':>12}  {'Loss':>8}  {'Tok/Param':>10}"
        )
        lines.append(f"  {'-' * 12}  {'-' * 12}  {'-' * 12}  {'-' * 8}  {'-' * 10}")

        for C in compute_examples:
            N_opt = self.predict_optimal_N(np.array([C]))[0]
            D_opt = self.predict_optimal_D(np.array([C]))[0]
            L_pred = format_float(self.predict_loss(np.array([C]))[0])
            tok_per_param = format_float(D_opt / N_opt)
            lines.append(
                f"  {format_float(C)}  {format_count(N_opt)}  {format_token_count(D_opt)}  {L_pred}  {tok_per_param}"
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

    r_squared: Optional[float] = None
    """R-squared of the fit"""

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

    @classmethod
    def fit(
        cls,
        N: ArrayLike,
        D: ArrayLike,
        loss: ArrayLike,
        E: Optional[float] = None,
        huber_delta: float = 1e-3,
    ) -> "ChinchillaParametricFit":
        """
        Fit the full Chinchilla scaling law: L = E + A/N^alpha + B/D^beta

        Minimizes Huber loss on log(L) using L-BFGS-B, matching the Chinchilla paper methodology.

        Args:
            N: Array of parameter counts
            D: Array of token counts
            loss: Array of loss values
            E: If provided, use this fixed value for E (irreducible loss) instead of fitting it
            huber_delta: Delta parameter for Huber loss

        Returns:
            ChinchillaParametricFit with fitted parameters
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

        if E is not None:
            # Fit with fixed E (only fit A, alpha, B, beta)
            E_fixed = E

            def objective_fixed_E(params: np.ndarray) -> float:
                A, alpha, B, beta = params
                L_pred = chinchilla_parametric_scaling_law(N, D, E_fixed, A, alpha, B, beta)
                L_pred = np.maximum(L_pred, 1e-10)
                log_residuals = log_L - np.log(L_pred)
                return np.sum(huber_loss(log_residuals, huber_delta))

            p0 = (400.0, 0.34, 400.0, 0.28)
            scipy_bounds = [(1.0, 1e6), (0.1, 1.0), (1.0, 1e6), (0.1, 1.0)]

            result = minimize(
                objective_fixed_E,
                p0,
                method="L-BFGS-B",
                bounds=scipy_bounds,
                options={"maxiter": 50000, "ftol": 1e-12},
            )
            A, alpha, B, beta = result.x
            E_fit = E_fixed
        else:
            # Fit all parameters including E
            def objective(params: np.ndarray) -> float:
                E_param, A, alpha, B, beta = params
                L_pred = chinchilla_parametric_scaling_law(N, D, E_param, A, alpha, B, beta)
                L_pred = np.maximum(L_pred, 1e-10)
                log_residuals = log_L - np.log(L_pred)
                return np.sum(huber_loss(log_residuals, huber_delta))

            E_init = L.min() * 0.8
            p0 = (E_init, 400.0, 0.34, 400.0, 0.28)
            scipy_bounds = [(0.0, L.min()), (1.0, 1e6), (0.1, 1.0), (1.0, 1e6), (0.1, 1.0)]

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

        return cls(E=E_fit, A=A, alpha=alpha, B=B, beta=beta, r_squared=r_squared)

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
        a_opt = self.beta / (self.alpha + self.beta)
        b_opt = self.alpha / (self.alpha + self.beta)
        lines.append("-" * 40)
        lines.append("  Compute-optimal allocation (theoretical):")
        lines.append(f"    N_opt ∝ C^{a_opt:.3f}  (params scale with compute^{a_opt:.2f})")
        lines.append(f"    D_opt ∝ C^{b_opt:.3f}  (tokens scale with compute^{b_opt:.2f})")
        if abs(a_opt - 0.5) < 0.05 and abs(b_opt - 0.5) < 0.05:
            lines.append("    ✓ Near equal scaling (~0.5 each) matches Chinchilla")
        elif a_opt > b_opt:
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
    C_col: str = "throughput/total petaflops",
    loss_col: str = "eval/lm/pile-validation/CE loss",
    **kwargs,
) -> ChinchillaIsoParamFit:
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
    return ChinchillaIsoParamFit.fit(df[C_col].values, df[loss_col].values, **kwargs)


def fit_chinchilla_parametric_shared_E(
    ladder_dfs: dict[str, pd.DataFrame],
    N_col: str = "num_params",
    D_col: str = "throughput/total tokens",
    loss_col: str = "eval/lm/pile-validation/CE loss",
    **kwargs,
) -> dict[str, ChinchillaParametricFit]:
    """
    Fit Chinchilla parametric scaling law for multiple ladders with a shared E.

    This function:
    1. Fits each ladder independently to estimate E
    2. Computes the mean E across all fits
    3. Refits each ladder with the shared E fixed

    Args:
        ladder_dfs: Dict mapping ladder names to DataFrames
        N_col: Column name for parameter counts
        D_col: Column name for token counts
        loss_col: Column name for loss values
        **kwargs: Additional arguments passed to ChinchillaParametricFit.fit()

    Returns:
        Dict mapping ladder names to ChinchillaParametricFit objects (all with same E)
    """
    # First pass: fit each ladder independently to estimate E
    initial_fits = {}
    for name, df in ladder_dfs.items():
        initial_fits[name] = ChinchillaParametricFit.fit(
            N=df[N_col].values,
            D=df[D_col].values,
            loss=df[loss_col].values,
            **kwargs,
        )

    # Compute shared E as mean of all estimated E values
    shared_E = float(np.mean([fit.E for fit in initial_fits.values()]))

    # Second pass: refit each ladder with fixed shared E
    fits = {}
    for name, df in ladder_dfs.items():
        fits[name] = ChinchillaParametricFit.fit(
            N=df[N_col].values,
            D=df[D_col].values,
            loss=df[loss_col].values,
            E=shared_E,
            **kwargs,
        )
    return fits


def fit_chinchilla_isoparam_shared_E(
    ladder_dfs: dict[str, pd.DataFrame],
    N_col: str = "num_params",
    D_col: str = "throughput/total tokens",
    C_col: str = "throughput/total petaflops",
    loss_col: str = "eval/lm/pile-validation/CE loss",
    **kwargs,
) -> dict[str, ChinchillaIsoParamFit]:
    """
    Fit Chinchilla IsoParam scaling law for multiple ladders with a shared E.

    This function:
    1. Fits each ladder independently to estimate E
    2. Computes the mean E across all fits
    3. Refits each ladder with the shared E fixed

    Args:
        ladder_dfs: Dict mapping ladder names to DataFrames
        N_col: Column name for parameter counts
        D_col: Column name for token counts
        C_col: Column name for compute values
        loss_col: Column name for loss values
        **kwargs: Additional arguments passed to ChinchillaIsoParamFit.fit()

    Returns:
        Dict mapping ladder names to ChinchillaIsoParamFit objects (all with same E)
    """
    # First pass: fit each ladder independently to estimate E
    initial_fits = {}
    for name, df in ladder_dfs.items():
        initial_fits[name] = ChinchillaIsoParamFit.fit(
            N=df[N_col].values,
            D=df[D_col].values,
            C=df[C_col].values,
            loss=df[loss_col].values,
            **kwargs,
        )

    # Compute shared E as mean of all estimated E values
    shared_E = float(np.mean([fit.E for fit in initial_fits.values()]))

    # Second pass: refit each ladder with fixed shared E
    fits = {}
    for name, df in ladder_dfs.items():
        fits[name] = ChinchillaIsoParamFit.fit(
            N=df[N_col].values,
            D=df[D_col].values,
            C=df[C_col].values,
            loss=df[loss_col].values,
            E=shared_E,
            **kwargs,
        )
    return fits


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
