import multiprocessing
import os
from dataclasses import dataclass
from functools import partial
from itertools import product
from typing import NamedTuple, Optional, Protocol

import numpy as np
from numpy.typing import ArrayLike
from tqdm import tqdm


def chinchilla_parametric_scaling_law(
    N: ArrayLike, D: ArrayLike, E: float, A: float, alpha: float, B: float, beta: float
) -> np.ndarray:
    """
    Compute loss for given parameter count and token count using the Chinchilla scaling law.

    L(N, D) = E + A / N^α + B / D^β

    Uses logarithmic transformation with clipping for numerical stability.
    Details: https://github.com/kyo-takano/chinchilla/blob/master/docs/math.md#numerical-stability
    """
    N = np.asarray(N)
    D = np.asarray(D)
    assert np.all(N > 0), "N must be positive"
    assert np.all(D > 0), "D must be positive"
    dtype = np.result_type(N, D, np.float64)
    tiny = np.finfo(dtype).tiny
    max_exp = np.log(np.finfo(dtype).max)  # Limit for exp() to avoid overflow

    # Clip A, B to avoid log(0) = -inf
    A_safe = np.maximum(A, tiny)
    B_safe = np.maximum(B, tiny)

    # Compute in log-space: log(A/N^α) = log(A) - α*log(N)
    log_param_term = np.log(A_safe) - alpha * np.log(N)
    log_data_term = np.log(B_safe) - beta * np.log(D)

    # Clip exponents to avoid overflow in exp()
    log_param_term = np.clip(log_param_term, -max_exp, max_exp)
    log_data_term = np.clip(log_data_term, -max_exp, max_exp)

    param_term = np.exp(log_param_term)
    data_term = np.exp(log_data_term)
    return E + param_term + data_term


class ScalingLawModel(Protocol):
    """Protocol for any scaling law model that can predict loss for a given (N, D) allocation."""

    def predict_loss(self, N: ArrayLike, D: ArrayLike) -> np.ndarray: ...


class ScalingLawModelFitter(Protocol):
    """Protocol for any scaling law model fitter that can fit a scaling law model to data."""

    def fit(self, N: ArrayLike, D: ArrayLike, loss: ArrayLike, **kwargs) -> "ScalingLawModel": ...


class ChinchillaParams(NamedTuple):
    E: float
    A: float
    alpha: float
    B: float
    beta: float

    @property
    def a_opt(self) -> float:
        """Compute-optimal N exponent: N_opt ∝ C^a_opt where a_opt = β/(α+β)."""
        return self.beta / (self.alpha + self.beta)

    @property
    def b_opt(self) -> float:
        """Compute-optimal D exponent: D_opt ∝ C^b_opt where b_opt = α/(α+β)."""
        return self.alpha / (self.alpha + self.beta)

    def predict_loss(self, N: ArrayLike, D: ArrayLike) -> np.ndarray:
        return chinchilla_parametric_scaling_law(
            N, D, self.E, self.A, self.alpha, self.B, self.beta
        )

    def __repr__(self) -> str:
        return (
            f"ChinchillaParams(L(N,D) = {self.E:.6f} + {self.A:.6e}/N^{self.alpha:.4f} + "
            f"{self.B:.6e}/D^{self.beta:.4f}, a_opt={self.a_opt:.4f}, b_opt={self.b_opt:.4f})"
        )


@dataclass
class ChinchillaParametricFit:
    """
    Results from fitting the full Chinchilla N, D parametric scaling law.

    Fits a two-variable power law: L(N, D) = E + A / N^alpha + B / D^beta

    Where:
    - N = number of parameters
    - D = number of training tokens
    """

    fitted_params: ChinchillaParams

    huber_loss: Optional[float] = None
    """Huber loss of the fit (computed on log scale to match optimization objective)."""

    # Stored data from fitting
    _N: Optional[np.ndarray] = None
    _D: Optional[np.ndarray] = None
    _loss: Optional[np.ndarray] = None

    def predict_loss(self, N: ArrayLike, D: ArrayLike) -> np.ndarray:
        """Predict the loss for a given (N, D) allocation."""
        return self.fitted_params.predict_loss(N, D)

    @property
    def residuals(self) -> Optional[np.ndarray]:
        """Residuals (log scale) from the fit on the original data."""
        if self._loss is None or self._N is None or self._D is None:
            return None
        return np.log(self._loss) - np.log(self.predict_loss(self._N, self._D))

    @staticmethod
    def _optimize_single_init(
        init_params: ChinchillaParams,
        scipy_bounds: list[tuple[float, float]],
        N: np.ndarray,
        D: np.ndarray,
        L: np.ndarray,
        weights: Optional[np.ndarray] = None,
        overestimate_penalty: float = 1.0,
    ) -> tuple[float, ChinchillaParams] | None:
        """
        Run a single optimization from given initial parameters.

        :param init_params: Tuple of (E, A, alpha, B, beta) initial values.
        :param scipy_bounds: Bounds for optimization in same order (E, A, alpha, B, beta).
        :param N: Parameter counts.
        :param D: Token counts.
        :param L: Loss values (raw, not log-transformed).
        :param weights: Optional weights for each observation (applied to loss function).
        :param overestimate_penalty: Multiplier for overestimate errors (predicted > actual).
            Values > 1.0 penalize overestimates more than underestimates.
        :returns: Tuple of (loss, ChinchillaParams) if successful, None otherwise.
        """
        try:
            from scipy.optimize import minimize  # type: ignore[reportMissingImports]
        except ImportError:
            raise ImportError("scipy is required to fit scaling laws")

        if weights is None:
            weights = np.ones_like(L)

        log_L = np.log(L)

        def _huber_loss(
            residuals: np.ndarray, delta: float = 1e-3, asymmetry: float = 1.0
        ) -> np.ndarray:
            """
            Apply Huber loss function to residuals with optional asymmetry.

            :param residuals: Array of residuals (actual - predicted in log space).
            :param delta: Threshold for switching between quadratic and linear loss.
            :param asymmetry: Multiplier for negative residuals (overestimates).
                Values > 1.0 penalize overestimates more than underestimates.
            """
            abs_r = np.abs(residuals)
            quadratic = 0.5 * residuals**2
            linear = delta * (abs_r - 0.5 * delta)
            base_loss = np.where(abs_r <= delta, quadratic, linear)
            # Apply asymmetric penalty: negative residuals = overestimates
            asymmetric_weight = np.where(residuals < 0, asymmetry, 1.0)
            return base_loss * asymmetric_weight

        def objective(params: np.ndarray) -> float:
            E_param, A, alpha, B, beta = params
            L_pred = chinchilla_parametric_scaling_law(N, D, E_param, A, alpha, B, beta)
            L_pred = np.maximum(L_pred, 1e-10)
            log_residuals = log_L - np.log(L_pred)
            # Weighted sum of losses
            return np.sum(weights * _huber_loss(log_residuals, asymmetry=overestimate_penalty))

        try:
            result = minimize(
                objective,
                init_params,
                method="L-BFGS-B",
                bounds=scipy_bounds,
                options={"maxiter": 10000},
            )
            if np.isfinite(result.fun):
                return result.fun, ChinchillaParams(*result.x)
        except Exception:
            pass
        return None

    @classmethod
    def fit(
        cls,
        N: ArrayLike,
        D: ArrayLike,
        loss: ArrayLike,
        parallel: bool = True,
        weights: Optional[ArrayLike] = None,
        overestimate_penalty: float = 1.0,
        num_slices: int = 4,
    ) -> "ChinchillaParametricFit":
        """
        Fit the full parametric Chinchilla scaling law: L = E + A/N^alpha + B/D^beta.

        :param N: Array of parameter counts.
        :param D: Array of token counts.
        :param loss: Array of loss values.
        :param parallel: If True, use multiprocessing for grid search optimization.
        :param weights: Optional weights for each observation. Higher weights give more
            importance to those points during fitting. Common choices:
            - np.sqrt(6 * N * D): Weight by sqrt(compute) to emphasize large-scale points
            - None: Uniform weights (default)
        :param overestimate_penalty: Multiplier for overestimate errors in the Huber loss.
            When > 1.0, the loss function penalizes overestimates (predicted > actual)
            more heavily than underestimates. This pushes the fit toward lower loss
            predictions, useful since the parametric scaling law is typically used to capture
            the lower-bound of achievable loss with a given (N, D) allocation. A good choice if
            the goal is to capture the lower-bound of achievable loss is to set this to 10.0.
            Default is 1.0 (symmetric loss).
        :param num_slices: Number of slices to use for grid search along each dimension.
        :returns: :class:`ChinchillaParametricFit` with fitted parameters.
        """
        N = np.asarray(N)
        D = np.asarray(D)
        L = np.asarray(loss)

        assert np.all(np.isfinite(N) & (N > 0)), "N must be finite and positive"
        assert np.all(np.isfinite(D) & (D > 0)), "D must be finite and positive"
        assert np.all(np.isfinite(L) & (L > 0)), "loss must be finite and positive"
        assert len(N) >= 5, f"Need at least 5 data points, got {len(N)}"

        # Normalize weights if provided
        weights_clean: Optional[np.ndarray] = None
        if weights is not None:
            weights_clean = np.asarray(weights)
            weights_clean = weights_clean * len(N) / weights_clean.sum()

        # Grid search over initializations to find the best fit
        L_min = float(L.min())  # E (entropy floor) must be <= minimum observed loss
        lower_bounds = ChinchillaParams(E=0.0, A=1e-10, alpha=0.01, B=1e-10, beta=0.01)
        upper_bounds = ChinchillaParams(E=L_min, A=1e10, alpha=2.0, B=1e10, beta=2.0)
        scipy_bounds = list(zip(lower_bounds, upper_bounds))

        # The fitted parametric model is sensitive to the initial parameter values so it is standard
        # to search over a grid. The original grid used by Chinchilla covers parameter ranges that
        # are much wider than the ranges we typically see in practice. Some are even unrealistic,
        # such as an entropy floor greater than the minimum observed loss. We refine the parameter
        # search space to allow for a more fine-grained search without searching an extremely large grid.
        # NOTE: random search with sobol noise may be a more efficient way to search the parameter space.
        E_grid = np.linspace(0.0, L_min, num_slices)
        A_grid = np.linspace(1, 20, num_slices)
        alpha_grid = np.linspace(0.2, 0.8, num_slices)
        B_grid = np.linspace(1, 20, num_slices)
        beta_grid = np.linspace(0.2, 0.8, num_slices)
        grid: list[ChinchillaParams] = [
            ChinchillaParams(E=E, A=A, alpha=alpha, B=B, beta=beta)
            for E, A, alpha, B, beta in product(E_grid, A_grid, alpha_grid, B_grid, beta_grid)
        ]

        optimize_fn = partial(
            cls._optimize_single_init,
            scipy_bounds=scipy_bounds,
            N=N,
            D=D,
            L=L,
            weights=weights_clean,
            overestimate_penalty=overestimate_penalty,
        )

        results: list[tuple[float, ChinchillaParams]] = []
        if parallel:
            n_workers = os.cpu_count() or 1
            ctx = multiprocessing.get_context("fork")
            with ctx.Pool(n_workers) as pool:
                for res in pool.imap_unordered(optimize_fn, grid):
                    if res is not None:
                        results.append(res)
        else:
            for init_params in grid:
                res = optimize_fn(ChinchillaParams(*init_params))
                if res is not None:
                    results.append(res)

        if not results:
            raise ValueError("All optimization attempts failed")

        # Find the fitted parameters with the lowest loss
        best = min(results, key=lambda x: x[0])
        huber_loss, params = best

        return cls(fitted_params=params, huber_loss=huber_loss, _N=N, _D=D, _loss=L)


@dataclass
class ChinchillaParametricBootstrappedFit:
    """
    Results from bootstrapping the parametric Chinchilla scaling law.

    Provides uncertainty estimates for fitted parameters by resampling the
    original data with replacement and refitting.
    """

    point_estimate: ChinchillaParametricFit
    """The point estimate of the fit from the original data."""

    fits: list[ChinchillaParametricFit]
    """List of fits from bootstrap samples."""

    def predict_loss(self, N: ArrayLike, D: ArrayLike) -> np.ndarray:
        """Predict loss using the point estimate fit."""
        return self.point_estimate.predict_loss(N, D)

    def predict_loss_distribution(
        self, N: ArrayLike, D: ArrayLike, include_observation_noise: bool = True
    ) -> np.ndarray:
        """
        Predict the distribution of loss values for a given (N, D) allocation.

        :param N: Parameter counts.
        :param D: Token counts.
        :param include_observation_noise: If True, adds sampled residuals to the predictions
            to estimate the distribution of future observations (prediction interval).
            If False, returns only the distribution of the mean curve (confidence interval).
        """
        N = np.asarray(N)
        D = np.asarray(D)

        # 1. Get distribution of the mean (parameter uncertainty)
        # Shape: (num_bootstraps, num_points)
        mean_predictions = np.array([fit.predict_loss(N, D) for fit in self.fits])

        if not include_observation_noise:
            return mean_predictions

        # 2. Add observation noise (aleatoric uncertainty)
        residuals = self.point_estimate.residuals
        if residuals is None:
            # Fallback if fit() was called before this field existed or residuals weren't stored
            return mean_predictions

        # Sample residuals with replacement to match the shape of predictions
        # NOTE: assumes the residuals are homoscedastic.
        rng = np.random.default_rng()
        sampled_log_residuals = rng.choice(residuals, size=mean_predictions.shape, replace=True)

        # The residuals in fit() are calculated as log(L_true) - log(L_pred).
        # So L_observed = L_pred * exp(residual)
        return mean_predictions * np.exp(sampled_log_residuals)

    @classmethod
    def fit(
        cls,
        N: ArrayLike,
        D: ArrayLike,
        loss: ArrayLike,
        num_bootstraps: int = 100,
        parallel: bool = True,
        weights: Optional[ArrayLike] = None,
        overestimate_penalty: float = 1.0,
        num_slices: int = 4,
        seed: Optional[int] = None,
        progress_bar: bool = True,
    ) -> "ChinchillaParametricBootstrappedFit":
        """
        Fit the Chinchilla scaling law with bootstrap uncertainty estimation.

        :param N: Array of parameter counts.
        :param D: Array of token counts.
        :param loss: Array of loss values.
        :param num_bootstraps: Number of bootstrap samples to generate.
        :param parallel: If True, use multiprocessing for grid search optimization.
        :param weights: Optional weights for each observation.
        :param overestimate_penalty: Multiplier for overestimate errors in the Huber loss.
        :param num_slices: Number of slices to use for grid search along each dimension.
        :param seed: Random seed for reproducibility.
        :param progress_bar: If True, show a progress bar for the bootstrap fits.
        :returns: A ChinchillaParametricBootstrappedFit with point estimate and bootstrap fits.
        """
        N = np.asarray(N)
        D = np.asarray(D)
        loss = np.asarray(loss)
        n_points = len(N)

        # Fit point estimate on original data
        point_estimate = ChinchillaParametricFit.fit(
            N,
            D,
            loss,
            parallel=parallel,
            weights=weights,
            overestimate_penalty=overestimate_penalty,
            num_slices=num_slices,
        )

        rng = np.random.default_rng(seed)
        bootstrap_fits: list[ChinchillaParametricFit] = []
        for _ in tqdm(range(num_bootstraps), disable=not progress_bar, desc="Bootstrapping"):
            # Resample indices with replacement
            indices = rng.choice(n_points, size=n_points, replace=True)

            N_boot = N[indices]
            D_boot = D[indices]
            loss_boot = loss[indices]
            weights_boot = None if weights is None else np.asarray(weights)[indices]

            try:
                boot_fit = ChinchillaParametricFit.fit(
                    N_boot,
                    D_boot,
                    loss_boot,
                    parallel=parallel,
                    weights=weights_boot,
                    overestimate_penalty=overestimate_penalty,
                    num_slices=num_slices,
                )
                bootstrap_fits.append(boot_fit)
            except ValueError:
                # Skip failed fits (can happen with unlucky resampling)
                continue

        if not bootstrap_fits:
            raise ValueError("All bootstrap fits failed")

        return cls(point_estimate=point_estimate, fits=bootstrap_fits)
