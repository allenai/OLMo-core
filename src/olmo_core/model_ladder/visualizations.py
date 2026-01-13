"""
Visualization functions for scaling law fits.
"""

from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from olmo_core.model_ladder.scaling_laws import (
        ChinchillaIsoParamFit,
        ChinchillaParametricFit,
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
