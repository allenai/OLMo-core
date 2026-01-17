import numpy as np

from olmo_core.model_ladder.analysis.scaling_laws import (
    ChinchillaParametricBootstrappedFit,
    ChinchillaParametricFit,
    ChinchillaParams,
    chinchilla_parametric_scaling_law,
)
from olmo_core.utils import seed_all


def test_chinchilla_params_predict_loss():
    """Test predict_loss method."""
    params = ChinchillaParams(E=1.0, A=100.0, alpha=0.5, B=200.0, beta=0.3)
    N = np.array([1e6, 1e7])
    D = np.array([1e9, 1e10])

    loss = params.predict_loss(N, D)

    assert loss.shape == (2,)
    assert np.all(loss > params.E)  # Loss should be above entropy floor
    assert np.all(np.isfinite(loss))


def test_chinchilla_parametric_scaling_law():
    # 1D arrays
    N = np.array([1e6, 1e7, 1e8])
    D = np.array([1e9, 1e10, 1e11])
    loss = chinchilla_parametric_scaling_law(N, D, 1.0, 100.0, 0.5, 200.0, 0.3)
    assert loss.shape == (3,)

    # Broadcasting
    N = np.array([1e6, 1e7])
    D = np.array([1e9])
    loss = chinchilla_parametric_scaling_law(N, D, 1.0, 100.0, 0.5, 200.0, 0.3)
    assert loss.shape == (2,)


def test_chinchilla_parametric_scaling_law_extreme_values():
    N = np.array([1e6])
    D = np.array([1e9])

    # Very large A and B
    loss = chinchilla_parametric_scaling_law(N, D, 1.0, 1e10, 0.5, 1e10, 0.3)
    assert np.all(np.isfinite(loss))
    assert np.all(loss >= 1.0)  # Should be >= E


def test_chinchilla_parametric_scaling_law_tiny_A_B():
    N = np.array([1e6])
    D = np.array([1e9])
    # Use tiny values for A and B
    loss = chinchilla_parametric_scaling_law(N, D, 1.0, 1e-20, 0.5, 1e-20, 0.3)
    assert np.all(np.isfinite(loss))
    # Loss should be approximately E when A and B are tiny
    assert abs(loss[0] - 1.0) < 1e-10


def test_chinchilla_parametric_fit_synthetic_data():
    """Test fitting on synthetic data."""
    # Generate synthetic data from known parameters
    true_params = ChinchillaParams(E=1.0, A=100.0, alpha=0.5, B=200.0, beta=0.3)

    N = np.array([1e6, 2e6, 5e6, 1e7, 2e7, 5e7])
    D = np.array([1e9, 2e9, 5e9, 1e10, 2e10, 5e10])
    true_loss = true_params.predict_loss(N, D)

    # Add small noise
    seed_all(0)
    noisy_loss = true_loss * (1 + 0.01 * np.random.randn(len(N)))

    # Fit the model (small number of slices to speed up test)
    fit = ChinchillaParametricFit.fit(N, D, noisy_loss, num_slices=2)
    assert fit.fitted_params is not None
    assert fit.huber_loss is not None
    assert fit.huber_loss >= 0

    # Check that predictions are reasonable
    pred_loss = fit.predict_loss(N, D)
    assert pred_loss.shape == true_loss.shape
    assert np.all(np.isfinite(pred_loss))

    # Predictions should be close to true values (within reasonable tolerance)
    # Note: With noise and limited grid search, we don't expect perfect recovery
    assert np.allclose(pred_loss, true_loss, rtol=0.1)


def test_chinchilla_parametric_fit_with_weights():
    true_params = ChinchillaParams(E=1.0, A=100.0, alpha=0.5, B=200.0, beta=0.3)

    N = np.array([1e6, 2e6, 5e6, 1e7, 2e7])
    D = np.array([1e9, 2e9, 5e9, 1e10, 2e10])
    true_loss = true_params.predict_loss(N, D)

    np.random.seed(42)
    noisy_loss = true_loss * (1 + 0.01 * np.random.randn(len(N)))

    # Fit with weights (emphasize larger compute points)
    weights = np.sqrt(6 * N * D)
    fit = ChinchillaParametricFit.fit(N, D, noisy_loss, weights=weights, num_slices=2)

    assert fit.fitted_params is not None
    assert fit.huber_loss is not None


def test_chinchilla_parametric_bootstrapped_fit():
    seed_all(0)
    true_params = ChinchillaParams(E=1.0, A=100.0, alpha=0.5, B=200.0, beta=0.3)

    N = np.array([1e6, 2e6, 5e6, 1e7, 2e7, 5e7])
    D = np.array([1e9, 2e9, 5e9, 1e10, 2e10, 5e10])
    true_loss = true_params.predict_loss(N, D)
    noisy_loss = true_loss * (1 + 0.01 * np.random.randn(len(N)))

    # Fit with small number of bootstraps to keep test fast
    boot_fit = ChinchillaParametricBootstrappedFit.fit(
        N, D, noisy_loss, num_bootstraps=3, num_slices=2, progress_bar=False, seed=42
    )

    assert boot_fit.point_estimate is not None
    assert boot_fit.point_estimate.fitted_params is not None
    assert len(boot_fit.fits) == 3

    # Test predict_loss_distribution without observation noise (confidence interval)
    test_N = np.array([1e8])
    test_D = np.array([1e11])
    dist_no_noise = boot_fit.predict_loss_distribution(
        test_N, test_D, include_observation_noise=False
    )
    assert dist_no_noise.shape == (len(boot_fit.fits), 1)
    assert np.all(np.isfinite(dist_no_noise))

    # Test with observation noise (prediction interval)
    dist_with_noise = boot_fit.predict_loss_distribution(
        test_N, test_D, include_observation_noise=True
    )
    assert dist_with_noise.shape == (len(boot_fit.fits), 1)
    assert np.all(np.isfinite(dist_with_noise))
