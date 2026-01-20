"""
Visualization functions for scaling law fits.
"""

from typing import TYPE_CHECKING, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from olmo_core.model_ladder.scaling_laws import (
        ChinchillaIsoParamFit,
        ChinchillaParametricBootstrapFit,
        ChinchillaParametricFit,
        ChinchillaParametricWildClusterBootstrapFit,
    )

    # Type alias for bootstrap-like fits (both naive and wild cluster)
    BootstrapFitType = (
        ChinchillaParametricBootstrapFit | ChinchillaParametricWildClusterBootstrapFit
    )


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


def plot_isoparam_fit(
    fit: "ChinchillaIsoParamFit",
    C: Optional[ArrayLike] = None,
    loss: Optional[ArrayLike] = None,
    N: Optional[ArrayLike] = None,
    D: Optional[ArrayLike] = None,
    title: Optional[str] = None,
    figsize: tuple[int, int] = (14, 10),
) -> plt.Figure:
    """
    Plot the IsoParam scaling law fits against actual data with residuals.

    Creates a 2x2 grid when N and D are provided (or stored from fit):
    - Top left: Loss vs Compute
    - Top right: N vs Compute
    - Bottom left: D vs Compute
    - Bottom right: Loss residuals

    All observations are shown as lighter points, while the Pareto frontier
    points (not dominated by any other point in compute-loss space) are highlighted.

    If N and D are not available, falls back to a simple 2-panel loss plot.

    Args:
        fit: The ChinchillaIsoParamFit object to plot.
        C: Array of compute values. If None, uses stored values from fit().
        loss: Array of actual loss values. If None, uses stored values from fit().
        N: Optional array of parameter counts. If None, uses stored values from fit().
        D: Optional array of token counts. If None, uses stored values from fit().
        title: Optional title for the plot
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object
    """
    from olmo_core.model_ladder.scaling_laws import _extract_pareto_frontier_dominance

    # Use stored values if not provided
    if C is None:
        if fit._C is None:
            raise ValueError("C not provided and no stored values from fit()")
        C = fit._C
    if loss is None:
        if fit._loss is None:
            raise ValueError("loss not provided and no stored values from fit()")
        loss = fit._loss
    if N is None:
        N = fit._N
    if D is None:
        D = fit._D

    C = np.asarray(C)
    loss = np.asarray(loss)

    # Generate smooth curve for the fit
    C_smooth = np.logspace(np.log10(C.min()), np.log10(C.max()), 200)
    L_pred_smooth = fit.predict_loss(C_smooth)

    # If N and D provided, create 2x2 grid
    if N is not None and D is not None:
        N = np.asarray(N)
        D = np.asarray(D)

        # Extract frontier for highlighting (using Pareto dominance)
        N_frontier, D_frontier, C_frontier, L_frontier = _extract_pareto_frontier_dominance(
            N, D, C, loss
        )

        N_pred_smooth = fit.predict_optimal_N(C_smooth)
        D_pred_smooth = fit.predict_optimal_D(C_smooth)

        # Compute residuals on frontier points only (what was actually fit)
        L_pred_frontier = fit.predict_loss(C_frontier)
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
            label=f"L(C) = {fit.E:.3f} + {fit.A:.3f}/C^{fit.alpha:.3f}",
        )
        ax.set_xscale("log")
        ax.set_xlabel("Compute (petaFLOPs)", fontsize=11)
        ax.set_ylabel("Loss", fontsize=11)
        r2_str = f" (R²={fit.r_squared_loss:.4f})" if fit.r_squared_loss else ""
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
            label=f"N_opt(C) = {fit.G:.2e} × C^{fit.a:.3f}",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Compute (petaFLOPs)", fontsize=11)
        ax.set_ylabel("Parameters (N)", fontsize=11)
        r2_str = f" (R²={fit.r_squared_N:.4f})" if fit.r_squared_N else ""
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
            label=f"D_opt(C) = {fit.H:.2e} × C^{fit.b:.3f}",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Compute (petaFLOPs)", fontsize=11)
        ax.set_ylabel("Tokens (D)", fontsize=11)
        r2_str = f" (R²={fit.r_squared_D:.4f})" if fit.r_squared_D else ""
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
            fig.suptitle("Chinchilla Approach 1 Scaling Law Fits", fontsize=14, fontweight="bold")

        plt.tight_layout()
        return fig

    # Fallback: simple 2-panel loss plot when N and D not provided
    # Extract frontier for loss only (need dummy N, D)
    # In this case, just show all points since we can't extract frontier
    L_pred = fit.predict_loss(C)
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
        label=f"Fit: L(C) = {fit.E:.3f} + {fit.A:.3f}/C^{fit.alpha:.3f}",
    )
    ax1.set_xscale("log")
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3, which="both")

    if title:
        ax1.set_title(title, fontsize=14, fontweight="bold")
    else:
        r2_str = f" (R²={fit.r_squared_loss:.4f})" if fit.r_squared_loss else ""
        ax1.set_title(f"Chinchilla IsoParam Scaling Law{r2_str}", fontsize=14, fontweight="bold")

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


def plot_parametric_fit(
    fit: "ChinchillaParametricFit",
    N: Optional[ArrayLike] = None,
    D: Optional[ArrayLike] = None,
    loss: Optional[ArrayLike] = None,
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
        fit: The ChinchillaParametricFit object to plot.
        N: Array of parameter counts. If None, uses stored values from fit().
        D: Array of token counts. If None, uses stored values from fit().
        loss: Array of actual loss values. If None, uses stored values from fit().
        C: Array of compute values. If None, uses stored values or computes 6*N*D.
        title: Optional title for the plot
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object
    """

    # Use stored values if not provided
    if N is None:
        if fit._N is None:
            raise ValueError("N not provided and no stored values from fit()")
        N = fit._N
    if D is None:
        if fit._D is None:
            raise ValueError("D not provided and no stored values from fit()")
        D = fit._D
    if loss is None:
        if fit._loss is None:
            raise ValueError("loss not provided and no stored values from fit()")
        loss = fit._loss

    N = np.asarray(N)
    D = np.asarray(D)
    loss = np.asarray(loss)
    if C is None:
        C = fit._C if fit._C is not None else 6 * N * D
    else:
        C = np.asarray(C)

    L_pred = fit.predict_loss(N, D)
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
    stats_text = f"R² = {fit.r_squared:.4f}\nRMSE = {rmse:.4f}" if fit.r_squared else ""
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment="top")

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
    ax.scatter(D, residuals, alpha=0.7, s=50, c="forestgreen", edgecolors="black", linewidths=0.5)
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
    L_mesh = fit.predict_loss(N_mesh.ravel(), D_mesh.ravel()).reshape(N_mesh.shape)

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
            f"L(N,D) = {fit.E:.3f} + {fit.A:.3f}/N^{fit.alpha:.3f} + {fit.B:.3f}/D^{fit.beta:.3f}",
            fontsize=12,
            y=1.02,
        )

    plt.tight_layout()
    return fig


def plot_bootstrap_fit(
    fit: "BootstrapFitType",
    N: Optional[ArrayLike] = None,
    D: Optional[ArrayLike] = None,
    loss: Optional[ArrayLike] = None,
    confidence: float = 0.90,
    title: Optional[str] = None,
    figsize: tuple[int, int] = (16, 12),
    show_bootstrap_samples: int = 50,
) -> plt.Figure:
    """
    Plot comprehensive diagnostics for a bootstrap ensemble scaling law fit.

    Works with both ChinchillaParametricBootstrapFit (naive bootstrap) and
    ChinchillaParametricWildClusterBootstrapFit (wild cluster bootstrap).

    Creates a 2x3 grid:
    - Top row: Loss vs N, Loss vs D, Loss vs Compute (compute-optimal curve)
    - Bottom row: Parameter distributions, Parameter covariance, Residual distribution

    Args:
        fit: Bootstrap fit object (naive or wild cluster).
        N: Array of parameter counts. If None, uses stored values from fit.
        D: Array of token counts. If None, uses stored values from fit.
        loss: Array of actual loss values. If None, uses stored values from fit.
        confidence: Confidence level for intervals (default 0.90).
        title: Optional title for the plot.
        figsize: Figure size (width, height).
        show_bootstrap_samples: Number of bootstrap curves to show in spaghetti plot.

    Returns:
        matplotlib Figure object
    """
    # Use stored values if not provided
    if N is None:
        if fit._N is None:
            raise ValueError("N not provided and no stored values from fit")
        N = fit._N
    if D is None:
        if fit._D is None:
            raise ValueError("D not provided and no stored values from fit")
        D = fit._D
    if loss is None:
        if fit._loss is None:
            raise ValueError("loss not provided and no stored values from fit")
        loss = fit._loss

    N = np.asarray(N)
    D = np.asarray(D)
    loss = np.asarray(loss)
    C = 6 * N * D  # Compute in FLOPs (C = 6ND approximation)

    ci_pct = int(confidence * 100)

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # ──────────────────────────────────────────────────────────────────────
    # Top left: Loss vs N (Parameters) at multiple token counts
    # ──────────────────────────────────────────────────────────────────────
    ax = axes[0, 0]

    # Create smooth N range for prediction curves
    N_smooth = np.logspace(np.log10(N.min()), np.log10(N.max()), 100)

    # Plot curves for multiple token counts: 100B, 1T, 10T
    token_counts = [100e9, 1e12, 10e12]
    token_labels = ["100B", "1T", "10T"]
    cmap_lines = plt.colormaps.get_cmap("viridis")
    line_colors = [cmap_lines(i / (len(token_counts) - 1)) for i in range(len(token_counts))]

    for D_target, label, color in zip(token_counts, token_labels, line_colors):
        D_for_curve = np.full_like(N_smooth, D_target)
        L_mean_N, L_lower_N, L_upper_N = fit.predict_loss_interval(
            N_smooth, D_for_curve, confidence=confidence
        )
        ax.fill_between(N_smooth, L_lower_N, L_upper_N, alpha=0.15, color=color)
        ax.plot(N_smooth, L_mean_N, "-", linewidth=2, color=color, label=f"D={label}")

    # Plot actual data, colored by D
    scatter = ax.scatter(
        N,
        loss,
        c=np.log10(D),
        cmap="viridis",
        s=50,
        edgecolors="black",
        linewidths=0.5,
        alpha=0.8,
        zorder=5,
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("log₁₀(D)", fontsize=9)

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (N)", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("Loss vs Parameters (by token count)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    # ──────────────────────────────────────────────────────────────────────
    # Top middle: Loss vs D (Tokens) at multiple model sizes
    # ──────────────────────────────────────────────────────────────────────
    ax = axes[0, 1]

    # Create smooth D range for prediction curves
    D_smooth = np.logspace(np.log10(D.min()), np.log10(D.max()), 100)

    # Plot curves for multiple model sizes: 1B, 7B, 32B, 70B
    model_sizes = [1e9, 7e9, 32e9, 70e9]
    model_labels = ["1B", "7B", "32B", "70B"]
    cmap_lines = plt.colormaps.get_cmap("plasma")
    line_colors = [cmap_lines(i / (len(model_sizes) - 1)) for i in range(len(model_sizes))]

    for N_target, label, color in zip(model_sizes, model_labels, line_colors):
        N_for_curve = np.full_like(D_smooth, N_target)
        L_mean_D, L_lower_D, L_upper_D = fit.predict_loss_interval(
            N_for_curve, D_smooth, confidence=confidence
        )
        ax.fill_between(D_smooth, L_lower_D, L_upper_D, alpha=0.15, color=color)
        ax.plot(D_smooth, L_mean_D, "-", linewidth=2, color=color, label=f"N={label}")

    # Plot actual data, colored by N
    scatter = ax.scatter(
        D,
        loss,
        c=np.log10(N),
        cmap="plasma",
        s=50,
        edgecolors="black",
        linewidths=0.5,
        alpha=0.8,
        zorder=5,
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("log₁₀(N)", fontsize=9)

    ax.set_xscale("log")
    ax.set_xlabel("Tokens (D)", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("Loss vs Tokens (by model size)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    # ──────────────────────────────────────────────────────────────────────
    # Top right: Loss vs Compute (compute-optimal curve)
    # ──────────────────────────────────────────────────────────────────────
    ax = axes[0, 2]

    # Create smooth compute range
    C_smooth = np.logspace(np.log10(C.min()), np.log10(C.max()), 100)

    # Compute optimal N and D for each compute level using C = 6ND
    # N_opt ∝ C^a_opt, D_opt ∝ C^b_opt where a_opt + b_opt = 1
    if fit.point_estimate:
        a_opt = fit.point_estimate.a_opt
    else:
        a_opt = fit.beta_mean / (fit.alpha_mean + fit.beta_mean)

    # Use a reference point to anchor the scaling
    # N_opt = k_N * C^a_opt, D_opt = k_D * C^b_opt, with 6 * N_opt * D_opt = C
    # This gives: 6 * k_N * k_D * C^(a_opt + b_opt) = C
    # Since a_opt + b_opt = 1: k_N * k_D = 1/6
    # Choose k_N and k_D to pass through median of data
    C_ref = np.median(C)
    N_ref = np.median(N)
    k_N = N_ref / (C_ref**a_opt)
    N_opt_smooth = k_N * C_smooth**a_opt
    D_opt_smooth = C_smooth / (6 * N_opt_smooth)

    # Get predictions along compute-optimal path
    L_mean_C, L_lower_C, L_upper_C = fit.predict_loss_interval(
        N_opt_smooth, D_opt_smooth, confidence=confidence
    )

    # Plot confidence band
    ax.fill_between(
        C_smooth,
        L_lower_C,
        L_upper_C,
        alpha=0.3,
        color="steelblue",
        label=f"{ci_pct}% CI (optimal)",
    )
    ax.plot(C_smooth, L_mean_C, "b-", linewidth=2, label="Compute-optimal")

    # Plot bootstrap curves along optimal path
    n_show = min(show_bootstrap_samples, len(fit.fits))
    indices = np.linspace(0, len(fit.fits) - 1, n_show, dtype=int)
    for i, idx in enumerate(indices):
        boot_fit = fit.fits[idx]
        L_boot = boot_fit.predict_loss(N_opt_smooth, D_opt_smooth)
        ax.plot(
            C_smooth,
            L_boot,
            alpha=0.1,
            color="steelblue",
            linewidth=0.5,
        )

    # Plot actual data
    ax.scatter(
        C,
        loss,
        alpha=0.7,
        s=50,
        c="coral",
        edgecolors="black",
        linewidths=0.5,
        label="Observed",
        zorder=5,
    )

    ax.set_xscale("log")
    ax.set_xlabel("Compute C = 6ND (FLOPs)", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title(f"Loss vs Compute (a_opt={a_opt:.2f})", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    # ──────────────────────────────────────────────────────────────────────
    # Bottom left: Parameter distributions (violin plots)
    # ──────────────────────────────────────────────────────────────────────
    ax = axes[1, 0]

    # Collect parameter data for violin plot
    param_data = [
        fit.alpha_distribution,
        fit.beta_distribution,
    ]
    param_names = ["α (params)", "β (data)"]

    parts = ax.violinplot(param_data, positions=[0, 1], showmeans=True, showmedians=True)

    # Style the violins
    for pc in parts["bodies"]:
        pc.set_facecolor("steelblue")
        pc.set_alpha(0.7)
    parts["cmeans"].set_color("red")
    parts["cmedians"].set_color("black")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(param_names, fontsize=11)
    ax.set_ylabel("Exponent Value", fontsize=11)
    ax.set_title("Parameter Distributions", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Add stats annotation
    stats_text = (
        f"α = {fit.alpha_mean:.3f} ± {fit.alpha_std:.3f}\n"
        f"β = {fit.beta_mean:.3f} ± {fit.beta_std:.3f}\n"
        f"n = {fit.n_bootstrap} samples"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # ──────────────────────────────────────────────────────────────────────
    # Bottom middle: Parameter covariance (α vs β)
    # ──────────────────────────────────────────────────────────────────────
    ax = axes[1, 1]

    scatter = ax.scatter(
        fit.alpha_distribution,
        fit.beta_distribution,
        c=fit.r_squared_distribution,
        cmap="viridis",
        s=40,
        alpha=0.7,
        edgecolors="black",
        linewidths=0.3,
    )

    # Mark point estimate
    if fit.point_estimate:
        ax.scatter(
            fit.point_estimate.alpha,
            fit.point_estimate.beta,
            s=200,
            marker="*",
            c="red",
            edgecolors="black",
            linewidths=1,
            zorder=10,
            label="Point estimate",
        )

    # Add α = β diagonal line (Chinchilla hypothesis: equal scaling for N and D)
    ax_min = min(fit.alpha_distribution.min(), fit.beta_distribution.min()) * 0.9
    ax_max = max(fit.alpha_distribution.max(), fit.beta_distribution.max()) * 1.1
    ax.plot([ax_min, ax_max], [ax_min, ax_max], "k--", alpha=0.5, linewidth=1.5, label="α = β")

    ax.set_xlabel("α (parameter exponent)", fontsize=11)
    ax.set_ylabel("β (data exponent)", fontsize=11)
    ax.set_title("Parameter Covariance", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("R²", fontsize=10)

    # Add correlation annotation
    corr = np.corrcoef(fit.alpha_distribution, fit.beta_distribution)[0, 1]
    ax.text(
        0.05,
        0.05,
        f"ρ = {corr:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # ──────────────────────────────────────────────────────────────────────
    # Bottom right: Residual distribution
    # ──────────────────────────────────────────────────────────────────────
    ax = axes[1, 2]

    # Get residuals for point estimate
    if fit.point_estimate:
        L_pred_point = fit.point_estimate.predict_loss(N, D)
        residuals_point = loss - L_pred_point

        ax.hist(
            residuals_point,
            bins=min(30, max(10, len(residuals_point) // 3)),
            color="coral",
            edgecolor="black",
            alpha=0.7,
            density=True,
            label="Point estimate residuals",
        )

        # Add stats
        rmse = np.sqrt(np.mean(residuals_point**2))
        ax.axvline(x=0, color="r", linestyle="--", linewidth=1.5)

        stats_text = (
            f"RMSE = {rmse:.4f}\n"
            f"Mean = {np.mean(residuals_point):.4f}\n"
            f"Std = {np.std(residuals_point):.4f}"
        )
        ax.text(
            0.95,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Overlay normal fit
        mu, std = np.mean(residuals_point), np.std(residuals_point)
        if std > 0:
            x_norm = np.linspace(residuals_point.min() - std, residuals_point.max() + std, 100)
            y_norm = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_norm - mu) / std) ** 2)
            ax.plot(x_norm, y_norm, "b-", linewidth=2, label=f"Normal(μ={mu:.3f}, σ={std:.3f})")

    ax.set_xlabel("Residual (Observed - Predicted)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Residual Distribution", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ──────────────────────────────────────────────────────────────────────
    # Overall title
    # ──────────────────────────────────────────────────────────────────────
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    else:
        fig.suptitle(
            f"Bootstrap Scaling Law ({fit.n_bootstrap} samples): "
            f"α={fit.alpha_mean:.3f}±{fit.alpha_std:.3f}, β={fit.beta_mean:.3f}±{fit.beta_std:.3f}",
            fontsize=12,
            y=1.02,
        )

    plt.tight_layout()
    return fig


def plot_bootstrap_parameters(
    fit: "BootstrapFitType",
    confidence: float = 0.90,
    figsize: tuple[int, int] = (14, 8),
) -> plt.Figure:
    """
    Plot detailed parameter distributions from bootstrap ensemble.

    Works with both ChinchillaParametricBootstrapFit and ChinchillaParametricWildClusterBootstrapFit.

    Creates a 2x3 grid showing distribution of each parameter (E, A, α, B, β)
    plus derived quantity a_opt (compute-optimal N exponent).

    Args:
        fit: The ChinchillaParametricBootstrapFit object.
        confidence: Confidence level for interval shading.
        figsize: Figure size.

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    params = [
        ("E", fit.E_distribution, "Entropy Floor (E)", "forestgreen"),
        ("A", fit.A_distribution, "Param Coefficient (A)", "steelblue"),
        ("α", fit.alpha_distribution, "Param Exponent (α)", "coral"),
        ("B", fit.B_distribution, "Data Coefficient (B)", "mediumpurple"),
        ("β", fit.beta_distribution, "Data Exponent (β)", "goldenrod"),
    ]

    # Derived: compute-optimal exponents
    a_opt = fit.beta_distribution / (fit.alpha_distribution + fit.beta_distribution)

    params.append(("a_opt", a_opt, "Compute-Optimal N Exp (a_opt)", "teal"))

    lower_pct = (1 - confidence) / 2 * 100
    upper_pct = (1 + confidence) / 2 * 100
    ci_pct = int(confidence * 100)

    for idx, (name, dist, label, color) in enumerate(params):
        ax = axes[idx // 3, idx % 3]

        # Histogram
        ax.hist(dist, bins=30, color=color, edgecolor="black", alpha=0.7, density=True)

        # CI bounds
        lower = np.percentile(dist, lower_pct)
        upper = np.percentile(dist, upper_pct)
        median = np.median(dist)

        ax.axvline(
            x=median, color="black", linestyle="-", linewidth=2, label=f"Median: {median:.4f}"
        )
        ax.axvline(x=lower, color="red", linestyle="--", linewidth=1.5)
        ax.axvline(x=upper, color="red", linestyle="--", linewidth=1.5, label=f"{ci_pct}% CI")

        # Shade CI region
        ylim = ax.get_ylim()
        ax.fill_betweenx(ylim, lower, upper, alpha=0.2, color="red")
        ax.set_ylim(ylim)

        # Point estimate reference if available
        if fit.point_estimate and name in ["E", "A", "α", "B", "β"]:
            attr_name = {"α": "alpha", "β": "beta"}.get(name, name)
            point_val = getattr(fit.point_estimate, attr_name)
            ax.axvline(
                x=point_val,
                color="blue",
                linestyle=":",
                linewidth=2,
                label=f"Point est: {point_val:.4f}",
            )
        elif fit.point_estimate and name == "a_opt":
            point_val = fit.point_estimate.a_opt
            ax.axvline(
                x=point_val,
                color="blue",
                linestyle=":",
                linewidth=2,
                label=f"Point est: {point_val:.4f}",
            )

        ax.set_xlabel(name, fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Bootstrap Parameter Distributions (n={fit.n_bootstrap})",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    return fig


def plot_bootstrap_prediction_comparison(
    fits: dict[str, "BootstrapFitType"],
    N_targets: Optional[list[float]] = None,
    D_targets: Optional[list[float]] = None,
    D_range: Optional[ArrayLike] = None,
    N_range: Optional[ArrayLike] = None,
    confidence: float = 0.90,
    figsize: tuple[int, int] = (14, 10),
) -> plt.Figure:
    """
    Compare predictions from multiple bootstrap fits across model sizes and token counts.

    Works with both ChinchillaParametricBootstrapFit and ChinchillaParametricWildClusterBootstrapFit.
    Useful for comparing different experimental conditions or data subsets.

    Creates a 2x2 grid:
    - Top left: Loss vs Tokens at multiple model sizes (N_targets)
    - Top right: Loss vs Parameters at multiple token counts (D_targets)
    - Bottom left: Uncertainty vs Tokens
    - Bottom right: Uncertainty vs Parameters

    Args:
        fits: Dictionary mapping names to bootstrap fits.
        N_targets: List of target parameter counts (default: [7B, 32B, 70B]).
        D_targets: List of target token counts (default: [100B, 1T, 5T, 15T]).
        D_range: Range of token counts for x-axis (default: 100B to 20T).
        N_range: Range of parameter counts for x-axis (default: 1B to 100B).
        confidence: Confidence level for intervals.
        figsize: Figure size.

    Returns:
        matplotlib Figure object
    """
    if N_targets is None:
        N_targets = [7e9, 32e9, 70e9]  # 7B, 32B, 70B
    if D_targets is None:
        D_targets = [100e9, 1e12, 5e12, 15e12]  # 100B, 1T, 5T, 15T
    if D_range is None:
        D_range = np.logspace(11, 13.3, 100)  # 100B to 20T
    if N_range is None:
        N_range = np.logspace(9, 11, 100)  # 1B to 100B
    D_range = np.asarray(D_range)
    N_range = np.asarray(N_range)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fit_cmap = plt.colormaps.get_cmap("tab10")
    fit_colors = [fit_cmap(i) for i in range(len(fits))]

    ci_pct = int(confidence * 100)

    # ──────────────────────────────────────────────────────────────────────
    # Top left: Loss vs Tokens at multiple N targets
    # ──────────────────────────────────────────────────────────────────────
    ax = axes[0, 0]

    # Use different line styles for different N targets
    linestyles = ["-", "--", "-.", ":"]

    for (name, fit), color in zip(fits.items(), fit_colors):
        for i, N_target in enumerate(N_targets):
            N_arr = np.full_like(D_range, N_target)
            mean, lower, upper = fit.predict_loss_interval(N_arr, D_range, confidence=confidence)

            ls = linestyles[i % len(linestyles)]
            ax.fill_between(D_range, lower, upper, alpha=0.08, color=color)
            ax.plot(D_range, mean, ls, linewidth=2, color=color, alpha=0.7 + 0.1 * i)

    # Add N target legend
    for i, N_target in enumerate(N_targets):
        ax.plot(
            [], [], linestyles[i % len(linestyles)], color="gray", label=f"N={N_target / 1e9:.0f}B"
        )

    ax.set_xscale("log")
    ax.set_xlabel("Training Tokens (D)", fontsize=11)
    ax.set_ylabel("Predicted Loss", fontsize=11)
    ax.set_title("Loss vs Tokens (by model size)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")

    # ──────────────────────────────────────────────────────────────────────
    # Top right: Loss vs Parameters at multiple D targets
    # ──────────────────────────────────────────────────────────────────────
    ax = axes[0, 1]

    for (name, fit), color in zip(fits.items(), fit_colors):
        for i, D_target in enumerate(D_targets):
            D_arr = np.full_like(N_range, D_target)
            mean, lower, upper = fit.predict_loss_interval(N_range, D_arr, confidence=confidence)

            ls = linestyles[i % len(linestyles)]
            ax.fill_between(N_range, lower, upper, alpha=0.08, color=color)
            ax.plot(N_range, mean, ls, linewidth=2, color=color, alpha=0.7 + 0.1 * i)

    # Add D target legend
    for i, D_target in enumerate(D_targets):
        D_label = f"{D_target / 1e12:.0f}T" if D_target >= 1e12 else f"{D_target / 1e9:.0f}B"
        ax.plot([], [], linestyles[i % len(linestyles)], color="gray", label=f"D={D_label}")

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (N)", fontsize=11)
    ax.set_ylabel("Predicted Loss", fontsize=11)
    ax.set_title("Loss vs Parameters (by token count)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")

    # ──────────────────────────────────────────────────────────────────────
    # Bottom left: Uncertainty vs Tokens
    # ──────────────────────────────────────────────────────────────────────
    ax = axes[1, 0]

    for (name, fit), color in zip(fits.items(), fit_colors):
        for i, N_target in enumerate(N_targets):
            N_arr = np.full_like(D_range, N_target)
            std = fit.predict_loss_std(N_arr, D_range)

            ls = linestyles[i % len(linestyles)]
            ax.plot(D_range, std, ls, linewidth=2, color=color, alpha=0.7 + 0.1 * i)

    ax.set_xscale("log")
    ax.set_xlabel("Training Tokens (D)", fontsize=11)
    ax.set_ylabel("Prediction Std", fontsize=11)
    ax.set_title("Uncertainty vs Tokens", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")

    # ──────────────────────────────────────────────────────────────────────
    # Bottom right: Uncertainty vs Parameters
    # ──────────────────────────────────────────────────────────────────────
    ax = axes[1, 1]

    for (name, fit), color in zip(fits.items(), fit_colors):
        for i, D_target in enumerate(D_targets):
            D_arr = np.full_like(N_range, D_target)
            std = fit.predict_loss_std(N_range, D_arr)

            ls = linestyles[i % len(linestyles)]
            ax.plot(N_range, std, ls, linewidth=2, color=color, alpha=0.7 + 0.1 * i)

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (N)", fontsize=11)
    ax.set_ylabel("Prediction Std", fontsize=11)
    ax.set_title("Uncertainty vs Parameters", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")

    # Add fit name legend to bottom right
    for (name, _), color in zip(fits.items(), fit_colors):
        ax.plot([], [], "-", color=color, linewidth=2, label=name)
    ax.legend(fontsize=9, loc="upper right")

    fig.suptitle(
        f"Bootstrap Fit Comparison ({ci_pct}% CI)",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    return fig


def plot_bootstrap_rankings(
    fits: dict[str, "BootstrapFitType"],
    N_range: Optional[ArrayLike] = None,
    D_func: Optional[Callable[[float], float]] = None,
    N_heatmap: float = 7e9,
    confidence: float = 0.90,
    figsize: tuple[int, int] = (16, 10),
) -> plt.Figure:
    """
    Plot rankings of multiple bootstrap fits across scales.

    Creates a visualization showing:
    - Left: Predicted loss vs N for all models (with CI bands)
    - Right: Win probability heatmap (P(row < col) at a specific scale)

    Args:
        fits: Dictionary mapping names to bootstrap fits.
        N_range: Parameter counts for x-axis (default: 1B to 100B).
        D_func: Function mapping N to D. Default: D = 20*N (20 tokens per param).
        N_heatmap: Parameter count for the pairwise comparison heatmap (default: 7B).
            The D value is computed using D_func(N_heatmap).
        confidence: Confidence level for intervals (default 0.90).
        figsize: Figure size.

    Returns:
        matplotlib Figure object
    """
    if len(fits) < 2:
        raise ValueError("Need at least 2 fits to compare")

    if N_range is None:
        N_range = np.logspace(9, 11, 50)  # 1B to 100B
    N_range = np.asarray(N_range)

    def default_d_func(n: float) -> float:
        return 20 * n  # 20 tokens per param

    d_func = D_func if D_func is not None else default_d_func
    D_range = np.array([d_func(n) for n in N_range])

    names = list(fits.keys())
    n_fits = len(fits)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    cmap = plt.colormaps.get_cmap("tab10")
    colors = {name: cmap(i) for i, name in enumerate(names)}

    ci_pct = int(confidence * 100)

    # Get predictions for all fits
    n_samples = min(f.n_bootstrap for f in fits.values())
    predictions: dict[str, np.ndarray] = {}
    for name, fit in fits.items():
        predictions[name] = np.array(
            [fit.fits[i].predict_loss(N_range, D_range) for i in range(n_samples)]
        )

    # ──────────────────────────────────────────────────────────────────────
    # Left: Predicted loss vs N for all models (with CI bands)
    # ──────────────────────────────────────────────────────────────────────
    ax = axes[0]

    for name in names:
        color = colors[name]
        preds = predictions[name]
        mean = np.mean(preds, axis=0)
        lower = np.percentile(preds, (1 - confidence) / 2 * 100, axis=0)
        upper = np.percentile(preds, (1 + confidence) / 2 * 100, axis=0)

        ax.fill_between(N_range, lower, upper, alpha=0.15, color=color)
        ax.plot(N_range, mean, "-", linewidth=2, color=color, label=name)

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (N)", fontsize=11)
    ax.set_ylabel("Predicted Loss", fontsize=11)
    ax.set_title(f"Predicted Loss vs Scale ({ci_pct}% CI)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")

    # ──────────────────────────────────────────────────────────────────────
    # Right: Pairwise win probability heatmap
    # ──────────────────────────────────────────────────────────────────────
    ax = axes[1]

    # Use specified scale for pairwise comparison
    N_eval = N_heatmap
    D_eval = d_func(N_eval)
    N_arr = np.array([N_eval])
    D_arr = np.array([D_eval])

    # Get predictions at this scale
    preds_at_scale = {}
    for name, fit in fits.items():
        preds_at_scale[name] = np.array(
            [fit.fits[i].predict_loss(N_arr, D_arr)[0] for i in range(n_samples)]
        )

    # Build pairwise win probability matrix
    prob_matrix = np.zeros((n_fits, n_fits))
    sig_matrix = np.zeros((n_fits, n_fits), dtype=bool)

    for i, name_a in enumerate(names):
        for j, name_b in enumerate(names):
            if i == j:
                prob_matrix[i, j] = 0.5
            else:
                # P(A < B)
                diff = preds_at_scale[name_b] - preds_at_scale[name_a]
                prob_matrix[i, j] = np.mean(diff > 0)
                # Significant if CI excludes 0
                lower = np.percentile(diff, (1 - confidence) / 2 * 100)
                sig_matrix[i, j] = lower > 0

    # Create heatmap
    im = ax.imshow(prob_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(n_fits))
    ax.set_xticklabels(names, fontsize=9, rotation=45, ha="right")
    ax.set_yticks(range(n_fits))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Compared to", fontsize=11)
    ax.set_ylabel("Model", fontsize=11)
    # Format N and D for title
    from olmo_core.data.composable.utils import format_token_count
    from olmo_core.model_ladder.utils import format_count

    N_label = format_count(int(N_eval))
    D_label = format_token_count(int(D_eval))
    ax.set_title(
        f"P(row < col) @ {N_label}/{D_label} ({ci_pct}% sig)", fontsize=12, fontweight="bold"
    )

    # Add text annotations
    for i in range(n_fits):
        for j in range(n_fits):
            if i == j:
                text = "—"
                text_color = "gray"
            else:
                prob = prob_matrix[i, j]
                sig = sig_matrix[i, j]
                text = f"{prob:.0%}" + ("**" if sig else "")
                text_color = "white" if prob < 0.3 or prob > 0.7 else "black"
            ax.text(j, i, text, ha="center", va="center", color=text_color, fontsize=8)

    plt.colorbar(im, ax=ax, label="P(row wins)", shrink=0.8)

    fig.suptitle(
        f"Bootstrap Fit Rankings ({n_fits} models)",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    return fig
