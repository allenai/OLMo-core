"""Basic smoke tests for plotting functions."""

import numpy as np
import pytest

from olmo_core.model_ladder.analysis.scaling_laws import ChinchillaParams, RolloutSplit

# Check if plotly is available
try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None


def _create_test_split(cutoff_N: float = 100e6) -> RolloutSplit:
    params = ChinchillaParams(E=1.0, A=100.0, alpha=0.5, B=200.0, beta=0.3)

    test_N = np.array([cutoff_N * 2])
    test_D = np.array([2e9])
    test_loss = params.predict_loss(test_N, test_D)

    return RolloutSplit(
        cutoff_variable="N",
        cutoff_value=cutoff_N,
        train_mask=np.array([True]),
        test_mask=np.array([False]),
        model=params,
        train_N=np.array([cutoff_N]),
        train_D=np.array([1e9]),
        train_loss=np.array([params.predict_loss(cutoff_N, 1e9)]),
        test_N=test_N,
        test_D=test_D,
        test_loss=test_loss,
    )


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not installed")
def test_plot_scaling_law_3d_single_split():
    assert go is not None
    from olmo_core.model_ladder.analysis.plotting import plot_scaling_law_3d

    split = _create_test_split()
    fig = plot_scaling_law_3d(split)
    assert fig is not None
    assert isinstance(fig, go.Figure)


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not installed")
def test_plot_scaling_law_3d_multiple_splits():
    assert go is not None
    from olmo_core.model_ladder.analysis.plotting import plot_scaling_law_3d

    splits = [_create_test_split(100e6), _create_test_split(200e6)]
    fig = plot_scaling_law_3d(splits, subtitle="Test Experiment")
    assert fig is not None
    assert isinstance(fig, go.Figure)


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not installed")
def test_plot_scaling_law_3d_empty_splits():
    from olmo_core.model_ladder.analysis.plotting import plot_scaling_law_3d

    with pytest.raises(ValueError, match="splits list cannot be empty"):
        plot_scaling_law_3d([])


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not installed")
def test_plot_scaling_law_3d_comparison():
    assert go is not None
    from olmo_core.model_ladder.analysis.plotting import plot_scaling_law_3d_comparison

    splits_a = [_create_test_split(100e6), _create_test_split(200e6)]
    splits_b = [_create_test_split(100e6), _create_test_split(200e6)]
    fig = plot_scaling_law_3d_comparison(
        ("Experiment A", splits_a), ("Experiment B", splits_b), subtitle="Comparison Test"
    )
    assert fig is not None
    assert isinstance(fig, go.Figure)


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not installed")
def test_plot_scaling_law_3d_comparison_no_common_cutoffs():
    assert go is not None
    from olmo_core.model_ladder.analysis.plotting import plot_scaling_law_3d_comparison

    splits_a = [_create_test_split(100e6)]
    splits_b = [_create_test_split(500e6)]  # Different cutoff
    with pytest.raises(ValueError, match="No common cutoffs"):
        plot_scaling_law_3d_comparison(("Experiment A", splits_a), ("Experiment B", splits_b))
