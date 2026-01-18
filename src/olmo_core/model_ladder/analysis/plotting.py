"""
Plotting utilities for scaling law analysis and visualization.

This module provides interactive 3D visualizations for exploring fitted scaling laws,
their uncertainty bounds, and prediction accuracy.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np

from .scaling_laws import (
    ChinchillaParametricBootstrappedFit,
    ChinchillaParametricFit,
    RolloutSplit,
)


def plot_scaling_law_3d(
    splits: Union[RolloutSplit, list[RolloutSplit]],
    *,
    subtitle: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    width: int = 1000,
    height: int = 900,
    surface_opacity: float = 0.3,
    grid_resolution: int = 30,
    show_bootstrap_points: bool = True,
    show_connecting_lines: bool = True,
    camera_eye: Optional[dict] = None,
):
    """
    Create an interactive 3D visualization of fitted scaling law(s).

    Renders fitted surface(s), actual data points, predictions, and
    (optionally) bootstrap prediction distributions. When multiple splits
    are provided, a dropdown menu allows switching between cutoffs.

    :param splits: A single RolloutSplit or list of RolloutSplit objects.
        Works best with ChinchillaParametricBootstrappedFit for uncertainty visualization.
    :param subtitle: Subtitle shown below the main title (e.g., experiment name).
    :param save_path: If provided, saves the plot as an HTML file to this path.
    :param width: Plot width in pixels.
    :param height: Plot height in pixels.
    :param surface_opacity: Opacity of the fitted surface (0-1).
    :param grid_resolution: Number of points along each axis for the surface grid.
    :param show_bootstrap_points: If True, shows faint points for each bootstrap
        prediction at test locations.
    :param show_connecting_lines: If True, draws lines connecting actual to predicted
        points to visualize prediction errors.
    :param camera_eye: Camera position as dict with x, y, z keys.
        Defaults to dict(x=1.8, y=1.8, z=1.0).
    :returns: A plotly Figure object.

    Example::

        from olmo_core.model_ladder.analysis import (
            ChinchillaParametricBootstrappedFit,
            ScalingLawRollout,
            plot_scaling_law_3d,
        )

        rollout = ScalingLawRollout(N=N, D=D, loss=loss)
        splits = rollout.evaluate(ChinchillaParametricBootstrappedFit.fit)

        # Single split
        fig = plot_scaling_law_3d(splits[0], subtitle="OLMo-3 Baseline")

        # Multiple splits with dropdown
        fig = plot_scaling_law_3d(splits, subtitle="OLMo-3 Baseline")
        fig.show()
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly is required for plotting. Install with: pip install plotly")

    # Normalize to list
    if isinstance(splits, RolloutSplit):
        splits = [splits]

    if not splits:
        raise ValueError("splits list cannot be empty")

    # Compute global axis ranges across all splits for consistent view
    all_N = np.concatenate(
        [np.atleast_1d(s.train_N) for s in splits] + [np.atleast_1d(s.test_N) for s in splits]
    )
    all_D = np.concatenate(
        [np.atleast_1d(s.train_D) for s in splits] + [np.atleast_1d(s.test_D) for s in splits]
    )
    all_loss = np.concatenate(
        [np.atleast_1d(s.train_loss) for s in splits] + [np.atleast_1d(s.test_loss) for s in splits]
    )
    N_min, N_max = all_N.min(), all_N.max()
    D_min, D_max = all_D.min(), all_D.max()

    # Get minimum E (entropy floor) across all models for loss axis minimum
    e_values: list[float] = []
    for s in splits:
        model = s.model
        if isinstance(model, ChinchillaParametricBootstrappedFit):
            e_values.append(model.point_estimate.fitted_params.E)
        elif isinstance(model, ChinchillaParametricFit):
            e_values.append(model.fitted_params.E)
        else:
            raise ValueError(f"Unknown model type: {type(model)}")
    loss_min = min(e_values) if e_values else all_loss.min()

    # Use max of actual data for upper bound with padding
    loss_max = all_loss.max()
    loss_padding = (loss_max - loss_min) * 0.05
    loss_max += loss_padding

    # Create shared grid for all surfaces
    log_N_grid = np.linspace(np.log10(N_min * 0.8), np.log10(N_max * 1.2), grid_resolution)
    log_D_grid = np.linspace(np.log10(D_min * 0.8), np.log10(D_max * 1.2), grid_resolution)
    log_N_mesh, log_D_mesh = np.meshgrid(log_N_grid, log_D_grid)
    N_mesh = 10**log_N_mesh
    D_mesh = 10**log_D_mesh

    fig = go.Figure()

    # Track traces per split for visibility toggling
    traces_per_split: list[int] = []
    cutoff_labels: list[str] = []

    for idx, split in enumerate(splits):
        cutoff_m = int(split.cutoff_value / 1e6)
        cutoff_labels.append(f"{split.cutoff_variable} ≤ {cutoff_m}M")

        model = split.model
        test_N = np.atleast_1d(split.test_N)
        test_D = np.atleast_1d(split.test_D)
        test_loss = np.atleast_1d(split.test_loss)
        point_predictions = np.atleast_1d(model.predict_loss(test_N, test_D))
        visible = idx == 0

        # Compute surface and get E (entropy floor)
        if isinstance(model, ChinchillaParametricBootstrappedFit):
            grid_mean = model.point_estimate.predict_loss(N_mesh.ravel(), D_mesh.ravel()).reshape(
                N_mesh.shape
            )
            bootstrap_predictions = model.predict_loss_distribution(
                test_N, test_D, include_observation_noise=False
            )
            entropy_floor = model.point_estimate.fitted_params.E
        elif isinstance(model, ChinchillaParametricFit):
            grid_mean = model.predict_loss(N_mesh.ravel(), D_mesh.ravel()).reshape(N_mesh.shape)
            bootstrap_predictions = None
            entropy_floor = model.fitted_params.E
        else:
            grid_mean = model.predict_loss(N_mesh.ravel(), D_mesh.ravel()).reshape(N_mesh.shape)
            bootstrap_predictions = None
            entropy_floor = None

        trace_count = 0

        # Entropy floor plane (faint horizontal surface at z=E)
        if entropy_floor is not None:
            floor_z = np.full_like(grid_mean, entropy_floor)
            fig.add_trace(
                go.Surface(
                    x=log_N_mesh,
                    y=log_D_mesh,
                    z=floor_z,
                    colorscale=[[0, "rgba(128,128,128,0.3)"], [1, "rgba(128,128,128,0.3)"]],
                    showscale=False,
                    opacity=0.15,
                    name=f"Entropy Floor (E={entropy_floor:.4f})",
                    hovertemplate=f"Entropy Floor<br>E = {entropy_floor:.4f}<extra></extra>",
                    visible=visible,
                )
            )
            trace_count += 1

        # Fitted surface trace
        fig.add_trace(
            go.Surface(
                x=log_N_mesh,
                y=log_D_mesh,
                z=grid_mean,
                colorscale="Viridis",
                reversescale=True,
                showscale=True,
                colorbar=dict(title="Loss", x=1.02, len=0.8),
                opacity=surface_opacity,
                name="Fitted Surface",
                hovertemplate=(
                    "N: %{customdata[0]:.1f}M<br>"
                    "D: %{customdata[1]:.2f}B<br>"
                    "Loss: %{z:.4f}<extra>Fitted</extra>"
                ),
                customdata=np.stack([N_mesh / 1e6, D_mesh / 1e9], axis=-1),
                visible=visible,
            )
        )
        trace_count += 1

        # Connecting lines
        if show_connecting_lines:
            for i in range(len(test_N)):
                fig.add_trace(
                    go.Scatter3d(
                        x=[np.log10(test_N[i]), np.log10(test_N[i])],
                        y=[np.log10(test_D[i]), np.log10(test_D[i])],
                        z=[test_loss[i], point_predictions[i]],
                        mode="lines",
                        line=dict(color="rgba(255,0,0,0.5)", width=2),
                        showlegend=False,
                        hoverinfo="skip",
                        visible=visible,
                    )
                )
                trace_count += 1

        # Bootstrap points
        if show_bootstrap_points and bootstrap_predictions is not None:
            jitter_scale = 0.005
            all_bootstrap_x = []
            all_bootstrap_y = []
            all_bootstrap_z = []

            for i in range(len(test_N)):
                n_boots = bootstrap_predictions.shape[0]
                jitter_x = np.random.uniform(-jitter_scale, jitter_scale, n_boots)
                jitter_y = np.random.uniform(-jitter_scale, jitter_scale, n_boots)
                all_bootstrap_x.extend((np.log10(test_N[i]) + jitter_x).tolist())
                all_bootstrap_y.extend((np.log10(test_D[i]) + jitter_y).tolist())
                all_bootstrap_z.extend(bootstrap_predictions[:, i].tolist())

            fig.add_trace(
                go.Scatter3d(
                    x=all_bootstrap_x,
                    y=all_bootstrap_y,
                    z=all_bootstrap_z,
                    mode="markers",
                    marker=dict(size=2, color="rgba(100,100,255,0.15)", symbol="circle"),
                    name="Bootstrap Predictions",
                    hoverinfo="skip",
                    visible=visible,
                )
            )
            trace_count += 1

        # Training points
        train_N = np.atleast_1d(split.train_N)
        train_D = np.atleast_1d(split.train_D)
        train_loss = np.atleast_1d(split.train_loss)

        fig.add_trace(
            go.Scatter3d(
                x=np.log10(train_N),
                y=np.log10(train_D),
                z=train_loss,
                mode="markers",
                marker=dict(
                    size=4, color="blue", symbol="circle", line=dict(width=1, color="darkblue")
                ),
                name="Training Data",
                customdata=np.stack([train_N / 1e6, train_D / 1e9, train_loss], axis=-1),
                hovertemplate=(
                    "<b>Training</b><br>"
                    "N: %{customdata[0]:.1f}M<br>"
                    "D: %{customdata[1]:.2f}B<br>"
                    "Loss: %{customdata[2]:.4f}<extra></extra>"
                ),
                visible=visible,
            )
        )
        trace_count += 1

        # Actual test points
        fig.add_trace(
            go.Scatter3d(
                x=np.log10(test_N),
                y=np.log10(test_D),
                z=test_loss,
                mode="markers",
                marker=dict(
                    size=5, color="green", symbol="circle", line=dict(width=1, color="darkgreen")
                ),
                name="Actual Loss (Test)",
                customdata=np.stack([test_N / 1e6, test_D / 1e9, test_loss], axis=-1),
                hovertemplate=(
                    "<b>Actual (Test)</b><br>"
                    "N: %{customdata[0]:.1f}M<br>"
                    "D: %{customdata[1]:.2f}B<br>"
                    "Loss: %{customdata[2]:.4f}<extra></extra>"
                ),
                visible=visible,
            )
        )
        trace_count += 1

        # Predicted points
        abs_errors = point_predictions - test_loss
        ppl_ratio_pct = (2**abs_errors - 1) * 100

        fig.add_trace(
            go.Scatter3d(
                x=np.log10(test_N),
                y=np.log10(test_D),
                z=point_predictions,
                mode="markers",
                marker=dict(
                    size=3, color="red", symbol="diamond", line=dict(width=1, color="darkred")
                ),
                name="Predicted Loss",
                customdata=np.stack(
                    [test_N / 1e6, test_D / 1e9, point_predictions, abs_errors, ppl_ratio_pct],
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>Predicted</b><br>"
                    "N: %{customdata[0]:.1f}M<br>"
                    "D: %{customdata[1]:.2f}B<br>"
                    "Loss: %{customdata[2]:.4f}<br>"
                    "ΔBPB: %{customdata[3]:+.4f} (%{customdata[4]:+.1f}% ppl)<extra></extra>"
                ),
                visible=visible,
            )
        )
        trace_count += 1

        traces_per_split.append(trace_count)

    # Build title helper
    def build_title(label: str) -> str:
        title = f"<b>Training Cutoff: {label}</b>"
        return f"{title}<br><sub>{subtitle}</sub>" if subtitle else title

    if camera_eye is None:
        camera_eye = dict(x=1.8, y=1.8, z=1.0)

    # Compute axis ticks
    n_tick_vals = [
        np.log10(v)
        for v in [60e6, 100e6, 200e6, 400e6, 800e6, 1e9, 3e9, 7e9]
        if N_min * 0.8 <= v <= N_max * 1.2
    ]
    n_tick_text = [
        f"{10**v / 1e6:.0f}M" if 10**v < 1e9 else f"{10**v / 1e9:.0f}B" for v in n_tick_vals
    ]
    d_tick_vals = [
        np.log10(v)
        for v in [1e9, 2e9, 5e9, 10e9, 20e9, 50e9, 100e9, 200e9, 500e9, 1e12]
        if D_min * 0.8 <= v <= D_max * 1.2
    ]
    d_tick_text = [
        f"{10**v / 1e9:.0f}B" if 10**v < 1e12 else f"{10**v / 1e12:.0f}T" for v in d_tick_vals
    ]

    scene = dict(
        xaxis=dict(
            title="N (parameters)",
            tickvals=n_tick_vals,
            ticktext=n_tick_text,
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)",
        ),
        yaxis=dict(
            title="D (tokens)",
            tickvals=d_tick_vals,
            ticktext=d_tick_text,
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)",
        ),
        zaxis=dict(
            title="Loss", showgrid=True, gridcolor="rgba(0,0,0,0.1)", range=[loss_min, loss_max]
        ),
        camera=dict(eye=camera_eye),
        aspectmode="cube",
    )

    legend = dict(
        x=0.02,
        y=0.98,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="rgba(0,0,0,0.3)",
        borderwidth=1,
    )

    # Single split: simple layout without slider
    if len(splits) == 1:
        fig.update_layout(
            title=dict(text=build_title(cutoff_labels[0]), x=0.5, font=dict(size=16)),
            scene=scene,
            width=width,
            height=height,
            showlegend=True,
            legend=legend,
            margin=dict(l=0, r=0, t=60, b=0),
        )
    else:
        # Multiple splits: add slider for switching
        total_traces = sum(traces_per_split)
        steps = []
        trace_start = 0
        for label, n_traces in zip(cutoff_labels, traces_per_split):
            visibility = [False] * total_traces
            for i in range(trace_start, trace_start + n_traces):
                visibility[i] = True

            steps.append(
                dict(
                    label=label.split("≤")[1].strip(),  # Just show "100M" etc.
                    method="update",
                    args=[{"visible": visibility}, {"title.text": build_title(label)}],
                )
            )
            trace_start += n_traces

        fig.update_layout(
            title=dict(text=build_title(cutoff_labels[0]), x=0.5, font=dict(size=16)),
            sliders=[
                dict(
                    active=0,
                    currentvalue=dict(
                        prefix="Training Cutoff: N ≤ ",
                        visible=True,
                        xanchor="center",
                    ),
                    pad=dict(t=50, b=10),
                    len=0.9,
                    x=0.05,
                    xanchor="left",
                    steps=steps,
                )
            ],
            scene=scene,
            width=width,
            height=height,
            showlegend=True,
            legend=legend,
            margin=dict(l=0, r=0, t=60, b=60),
        )

    if save_path is not None:
        fig.write_html(str(save_path))

    return fig


def plot_scaling_law_3d_comparison(
    exp_a: tuple[str, list[RolloutSplit]],
    exp_b: tuple[str, list[RolloutSplit]],
    *,
    subtitle: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    width: int = 1200,
    height: int = 900,
    surface_opacity: float = 0.3,
    grid_resolution: int = 30,
    camera_eye: Optional[dict] = None,
):
    """
    Compare two scaling laws side-by-side with a cutoff slider.

    Creates an interactive 3D visualization comparing two experiments' fitted
    scaling law surfaces. A slider lets you move through training cutoffs.

    :param exp_a: Tuple of (name, splits) for the first experiment.
    :param exp_b: Tuple of (name, splits) for the second experiment.
    :param subtitle: Subtitle shown below the main title.
    :param save_path: If provided, saves the plot as an HTML file to this path.
    :param width: Plot width in pixels.
    :param height: Plot height in pixels.
    :param surface_opacity: Opacity of the fitted surfaces (0-1).
    :param grid_resolution: Number of points along each axis for the surface grid.
    :param camera_eye: Camera position as dict with x, y, z keys.
    :returns: A plotly Figure object.

    Example::

        from olmo_core.model_ladder.analysis import (
            ChinchillaParametricBootstrappedFit,
            ScalingLawRollout,
            plot_scaling_law_3d_comparison,
        )

        baseline_splits = ScalingLawRollout(N=N, D=D, loss=baseline_loss).evaluate(...)
        intervention_splits = ScalingLawRollout(N=N, D=D, loss=interv_loss).evaluate(...)

        fig = plot_scaling_law_3d_comparison(
            ("Baseline", baseline_splits),
            ("Intervention", intervention_splits),
            subtitle="Effect of intervention on scaling"
        )
        fig.show()
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly is required for plotting. Install with: pip install plotly")

    name_a, splits_a = exp_a
    name_b, splits_b = exp_b

    # Map cutoff (in millions) to split
    def get_cutoffs(splits: list[RolloutSplit]) -> dict[int, RolloutSplit]:
        return {int(s.cutoff_value / 1e6): s for s in splits}

    cutoff_map_a = get_cutoffs(splits_a)
    cutoff_map_b = get_cutoffs(splits_b)

    # Use cutoffs present in both experiments
    common_cutoffs = sorted(set(cutoff_map_a.keys()) & set(cutoff_map_b.keys()))
    if not common_cutoffs:
        raise ValueError(
            f"No common cutoffs between experiments. "
            f"{name_a} has: {sorted(cutoff_map_a.keys())}, "
            f"{name_b} has: {sorted(cutoff_map_b.keys())}"
        )

    # Compute global axis ranges
    all_splits = splits_a + splits_b
    all_N = np.concatenate(
        [np.atleast_1d(s.train_N) for s in all_splits]
        + [np.atleast_1d(s.test_N) for s in all_splits]
    )
    all_D = np.concatenate(
        [np.atleast_1d(s.train_D) for s in all_splits]
        + [np.atleast_1d(s.test_D) for s in all_splits]
    )
    all_loss = np.concatenate(
        [np.atleast_1d(s.train_loss) for s in all_splits]
        + [np.atleast_1d(s.test_loss) for s in all_splits]
    )
    N_min, N_max = all_N.min(), all_N.max()
    D_min, D_max = all_D.min(), all_D.max()

    # Get E values for loss axis
    e_values: list[float] = []
    for s in all_splits:
        model = s.model
        if isinstance(model, ChinchillaParametricBootstrappedFit):
            e_values.append(model.point_estimate.fitted_params.E)
        elif isinstance(model, ChinchillaParametricFit):
            e_values.append(model.fitted_params.E)
    loss_min = min(e_values) if e_values else all_loss.min()
    loss_max = all_loss.max()
    loss_padding = (loss_max - loss_min) * 0.05
    loss_max += loss_padding

    # Create grid
    log_N_grid = np.linspace(np.log10(N_min * 0.8), np.log10(N_max * 1.2), grid_resolution)
    log_D_grid = np.linspace(np.log10(D_min * 0.8), np.log10(D_max * 1.2), grid_resolution)
    log_N_mesh, log_D_mesh = np.meshgrid(log_N_grid, log_D_grid)
    N_mesh = 10**log_N_mesh
    D_mesh = 10**log_D_mesh

    # Colors for the two experiments
    colors = [
        ("rgba(0,119,187,{a})", "rgb(0,119,187)"),  # Blue
        ("rgba(238,119,51,{a})", "rgb(238,119,51)"),  # Orange
    ]

    fig = go.Figure()

    # Track traces per cutoff for visibility toggling
    traces_per_cutoff: list[int] = []

    # Add traces for each cutoff
    for cutoff_idx, cutoff in enumerate(common_cutoffs):
        visible = cutoff_idx == 0
        trace_count = 0

        for exp_idx, (exp_name, cutoff_map) in enumerate(
            [(name_a, cutoff_map_a), (name_b, cutoff_map_b)]
        ):
            split = cutoff_map[cutoff]
            color_template, solid_color = colors[exp_idx]
            model = split.model

            # Compute surface
            if isinstance(model, ChinchillaParametricBootstrappedFit):
                grid_mean = model.point_estimate.predict_loss(
                    N_mesh.ravel(), D_mesh.ravel()
                ).reshape(N_mesh.shape)
            else:
                grid_mean = model.predict_loss(N_mesh.ravel(), D_mesh.ravel()).reshape(N_mesh.shape)

            # Surface
            fig.add_trace(
                go.Surface(
                    x=log_N_mesh,
                    y=log_D_mesh,
                    z=grid_mean,
                    colorscale=[
                        [0, color_template.format(a=0.6)],
                        [1, color_template.format(a=0.9)],
                    ],
                    showscale=False,
                    opacity=surface_opacity,
                    name=exp_name,
                    legendgroup=exp_name,
                    hovertemplate=(
                        f"<b>{exp_name}</b><br>"
                        "N: %{customdata[0]:.1f}M<br>"
                        "D: %{customdata[1]:.2f}B<br>"
                        "Loss: %{z:.4f}<extra></extra>"
                    ),
                    customdata=np.stack([N_mesh / 1e6, D_mesh / 1e9], axis=-1),
                    visible=visible,
                )
            )
            trace_count += 1

            # Training points (squares)
            train_N = np.atleast_1d(split.train_N)
            train_D = np.atleast_1d(split.train_D)
            train_loss = np.atleast_1d(split.train_loss)

            fig.add_trace(
                go.Scatter3d(
                    x=np.log10(train_N),
                    y=np.log10(train_D),
                    z=train_loss,
                    mode="markers",
                    marker=dict(
                        size=4,
                        color=solid_color,
                        symbol="square",
                        opacity=0.7,
                        line=dict(width=1, color="rgba(0,0,0,0.3)"),
                    ),
                    name=f"{exp_name} Train",
                    legendgroup=exp_name,
                    customdata=np.stack([train_N / 1e6, train_D / 1e9, train_loss], axis=-1),
                    hovertemplate=(
                        f"<b>{exp_name} Train</b><br>"
                        "N: %{customdata[0]:.1f}M<br>"
                        "D: %{customdata[1]:.2f}B<br>"
                        "Loss: %{customdata[2]:.4f}<extra></extra>"
                    ),
                    visible=visible,
                )
            )
            trace_count += 1

            # Test points (circles)
            test_N = np.atleast_1d(split.test_N)
            test_D = np.atleast_1d(split.test_D)
            test_loss = np.atleast_1d(split.test_loss)

            fig.add_trace(
                go.Scatter3d(
                    x=np.log10(test_N),
                    y=np.log10(test_D),
                    z=test_loss,
                    mode="markers",
                    marker=dict(size=5, color=solid_color, symbol="circle"),
                    name=f"{exp_name} Test",
                    legendgroup=exp_name,
                    customdata=np.stack([test_N / 1e6, test_D / 1e9, test_loss], axis=-1),
                    hovertemplate=(
                        f"<b>{exp_name} Test</b><br>"
                        "N: %{customdata[0]:.1f}M<br>"
                        "D: %{customdata[1]:.2f}B<br>"
                        "Loss: %{customdata[2]:.4f}<extra></extra>"
                    ),
                    visible=visible,
                )
            )
            trace_count += 1

        traces_per_cutoff.append(trace_count)

    # Build slider steps
    total_traces = sum(traces_per_cutoff)
    slider_steps = []
    for cutoff_idx, cutoff in enumerate(common_cutoffs):
        visibility = [False] * total_traces
        trace_start = sum(traces_per_cutoff[:cutoff_idx])
        for i in range(trace_start, trace_start + traces_per_cutoff[cutoff_idx]):
            visibility[i] = True

        slider_steps.append(
            dict(
                label=f"{cutoff}M",
                method="update",
                args=[
                    {"visible": visibility},
                    {"title.text": _build_comparison_title(name_a, name_b, cutoff, subtitle)},
                ],
            )
        )

    if camera_eye is None:
        camera_eye = dict(x=1.8, y=1.8, z=1.0)

    # Axis ticks
    n_tick_vals = [
        np.log10(v)
        for v in [60e6, 100e6, 200e6, 400e6, 800e6, 1e9, 3e9, 7e9]
        if N_min * 0.8 <= v <= N_max * 1.2
    ]
    n_tick_text = [
        f"{10**v / 1e6:.0f}M" if 10**v < 1e9 else f"{10**v / 1e9:.0f}B" for v in n_tick_vals
    ]
    d_tick_vals = [
        np.log10(v)
        for v in [1e9, 2e9, 5e9, 10e9, 20e9, 50e9, 100e9, 200e9, 500e9, 1e12]
        if D_min * 0.8 <= v <= D_max * 1.2
    ]
    d_tick_text = [
        f"{10**v / 1e9:.0f}B" if 10**v < 1e12 else f"{10**v / 1e12:.0f}T" for v in d_tick_vals
    ]

    fig.update_layout(
        title=dict(
            text=_build_comparison_title(name_a, name_b, common_cutoffs[0], subtitle),
            x=0.5,
            font=dict(size=16),
        ),
        sliders=[
            dict(
                active=0,
                currentvalue=dict(prefix="Cutoff: N ≤ ", visible=True, xanchor="center"),
                pad=dict(t=50, b=10),
                len=0.9,
                x=0.05,
                xanchor="left",
                steps=slider_steps,
            )
        ],
        scene=dict(
            xaxis=dict(
                title="N (parameters)",
                tickvals=n_tick_vals,
                ticktext=n_tick_text,
                showgrid=True,
                gridcolor="rgba(0,0,0,0.1)",
            ),
            yaxis=dict(
                title="D (tokens)",
                tickvals=d_tick_vals,
                ticktext=d_tick_text,
                showgrid=True,
                gridcolor="rgba(0,0,0,0.1)",
            ),
            zaxis=dict(
                title="Loss",
                showgrid=True,
                gridcolor="rgba(0,0,0,0.1)",
                range=[loss_min, loss_max],
            ),
            camera=dict(eye=camera_eye),
            aspectmode="cube",
        ),
        width=width,
        height=height,
        showlegend=True,
        legend=dict(
            x=0.98,
            y=0.98,
            xanchor="right",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.3)",
            borderwidth=1,
        ),
        margin=dict(l=0, r=0, t=60, b=60),
    )

    if save_path is not None:
        fig.write_html(str(save_path))

    return fig


def _build_comparison_title(exp_a: str, exp_b: str, cutoff: int, subtitle: Optional[str]) -> str:
    """Build title for comparison plot."""
    title = f"<b>{exp_a} vs {exp_b}</b> (Cutoff: N ≤ {cutoff}M)"
    if subtitle:
        title = f"{title}<br><sub>{subtitle}</sub>"
    return title
