#!/usr/bin/env python3
"""
Chinchilla Scaling Law Fitting Script

Fits the parametric Chinchilla scaling law L(N, D) = E + A/N^α + B/D^β
to model ladder results stored in pickle files.

Usage:
    # Fit a single ladder
    python fit_chinchilla_scaling_laws.py --ladder olmo3:~/Downloads/olmo3-ladder

    # Compare multiple ladders
    python fit_chinchilla_scaling_laws.py \
        --ladder olmo3:~/Downloads/olmo3-ladder \
        --ladder hybrid-gdn:~/Downloads/hybrid-gdn-ladder

    # With bootstrap uncertainty and 2D plots
    python fit_chinchilla_scaling_laws.py \
        --ladder olmo3:~/Downloads/olmo3-ladder \
        --bootstrap 100 \
        --plot

    # With rollout cross-validation and 3D interactive plots
    python fit_chinchilla_scaling_laws.py \
        --ladder olmo3:~/Downloads/olmo3-ladder \
        --rollout \
        --plot-3d

    # Predict loss for a target model
    python fit_chinchilla_scaling_laws.py \
        --ladder olmo3:~/Downloads/olmo3-ladder \
        --predict-n 7e9 --predict-d 140e9

    # Export everything to a directory
    python fit_chinchilla_scaling_laws.py \
        --ladder olmo3:~/Downloads/olmo3-ladder \
        --ladder hybrid-gdn:~/Downloads/hybrid-gdn-ladder \
        --bootstrap 100 \
        --rollout \
        --plot --plot-3d \
        --output ~/results/scaling-analysis
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from olmo_core.model_ladder.analysis import (
    ChinchillaParametricBootstrappedFit,
    ChinchillaParametricFit,
    ScalingLawRollout,
    evaluate_rollout,
    plot_scaling_law_3d,
    plot_scaling_law_3d_comparison,
)

# Default loss column to use for fitting
DEFAULT_LOSS_COLUMN = "eval/lm/c4_en/CE loss"

# Alternative loss columns to try if default not found
FALLBACK_LOSS_COLUMNS = [
    "eval/lm/pile/CE loss",
    "eval/lm/dolma_common-crawl/CE loss",
    "eval/lm/wikitext_103/CE loss",
    "train/CE loss (log scale)",
]

# Corrected non-embedding parameter counts for each model type and size.
# These override the values from pickle files which may include embedding params.
# Format: {ladder_name_pattern: {size: non_embedding_params}}
CORRECTED_PARAM_COUNTS = {
    # Transformer models
    "olmo3-1": {
        "60M": 57_422_208,
        "100M": 101_736_960,
        "190M": 190_354_176,
        "370M": 371_262_464,
        "600M": 547_964_160,
        "760M": 758_220_288,
    },
    "olmo3-2": {
        "60M": 57_422_208,
        "100M": 101_736_960,
        "190M": 190_354_176,
        "370M": 371_262_464,
        "600M": 547_964_160,
        "760M": 758_220_288,
    },
    "olmo3-3": {
        "60M": 57_422_208,
        "100M": 101_736_960,
        "190M": 190_354_176,
        "370M": 371_262_464,
        "600M": 547_964_160,
        "760M": 758_220_288,
    },
    "pure-gdn": {
        "60M": 78_045_696,
        "100M": 139_771_584,
        "190M": 275_789_856,
        "370M": 573_609_472,
        "600M": 779_794_176,
    },
    "hybrid-gdn": {
        "60M": 72_889_824,
        "100M": 130_262_928,
        "190M": 254_430_936,
        "370M": 523_022_720,
        "600M": 721_836_672,
        "760M": 947_913_600,
        "1B": 1_481_856_384,
    },
    "hybrid-gdn-middle": {
        "60M": 70_311_888,
        "100M": 127_093_376,
        "190M": 247_311_296,
        "370M": 510_376_032,
        "600M": 707_347_296,
    },
    "hybrid-gdn-half": {
        "60M": 67_733_952,
        "100M": 120_754_272,
        "190M": 233_072_016,
        "370M": 472_435_968,
        "600M": 663_879_168,
    },
    "hybrid-gdn-eight": {
        "60M": 75_467_760,
        "100M": 133_432_480,
        "190M": 261_550_576,
        "370M": 548_316_096,
        "600M": 750_815_424,
        "760M": 979_529_152,
    },
    "pure-mamba": {
        "60M": 59_837_760,
        "100M": 109_746_048,
        "190M": 207_115_584,
    },
    "hybrid-mamba": {
        "60M": 59_837_760,
        "100M": 107_743_776,
        "190M": 202_925_232,
    },
}


def get_corrected_param_count(ladder_name: str, size: str) -> Optional[int]:
    """
    Look up the corrected non-embedding parameter count for a given ladder and size.

    Args:
        ladder_name: Name of the ladder (e.g., "transformer", "hybrid-gdn")
        size: Model size string (e.g., "60M", "190M")

    Returns:
        Corrected parameter count if found, None otherwise.
    """
    ladder_lower = ladder_name.lower()

    # Try exact match first
    if ladder_lower in CORRECTED_PARAM_COUNTS:
        size_map = CORRECTED_PARAM_COUNTS[ladder_lower]
        if size in size_map:
            return size_map[size]

    # Try substring matches for common patterns
    for pattern, size_map in CORRECTED_PARAM_COUNTS.items():
        if pattern in ladder_lower:
            if size in size_map:
                return size_map[size]

    return None


def find_loss_column(df: pd.DataFrame, preferred: Optional[str] = None) -> str:
    """Find an appropriate loss column in the dataframe."""
    candidates = [preferred] if preferred else []
    candidates.extend([DEFAULT_LOSS_COLUMN] + FALLBACK_LOSS_COLUMNS)

    for col in candidates:
        if col and col in df.columns:
            return col

    # Try to find any CE loss column
    ce_loss_cols = [c for c in df.columns if "CE loss" in c and "log" not in c.lower()]
    if ce_loss_cols:
        return ce_loss_cols[0]

    raise ValueError(f"No suitable loss column found. Available columns: {list(df.columns)}")


def load_ladder_data(
    ladder_dir: Path,
    ladder_name: str,
    loss_column: Optional[str] = None,
    verbose: bool = True,
    use_all_checkpoints: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load ladder results from pickle files for Chinchilla fitting.

    Args:
        ladder_dir: Directory containing metrics_*.pkl files
        ladder_name: Name of the ladder (used for parameter count corrections)
        loss_column: Specific loss column to use (auto-detected if None)
        verbose: Print loading progress
        use_all_checkpoints: If True, use all checkpoints from each model size.
            If False, only use the final checkpoint (legacy behavior).

    Returns:
        N: Array of parameter counts (corrected non-embedding counts if available)
        D: Array of token counts
        L: Array of loss values
        F: Array of FLOPs (in petaflops, from throughput/total petaflops column)
        sizes: List of size names (e.g., ["190M", "370M", ...] or ["190M@20", "190M@40", ...])
    """
    pkl_files = sorted(ladder_dir.glob("metrics_*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No metrics_*.pkl files found in {ladder_dir}")

    N_list, D_list, L_list, F_list, sizes = [], [], [], [], []
    used_loss_col = None
    flops_col = "throughput/total petaflops"

    for pkl_path in pkl_files:
        size = pkl_path.stem.replace("metrics_", "")

        try:
            df = pd.read_pickle(pkl_path)
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not load {pkl_path}: {e}")
            continue

        if df.empty:
            if verbose:
                print(f"  Warning: {pkl_path} is empty, skipping")
            continue

        # Find the loss column once
        if used_loss_col is None:
            used_loss_col = find_loss_column(df, loss_column)
            if verbose:
                print(f"  Using loss column: {used_loss_col}")

        # Get parameter count (should be constant for all rows in the file)
        num_params = df["num_params"].iloc[0]
        if num_params is None or pd.isna(num_params):
            if verbose:
                print(f"  Warning: No num_params for {size}, skipping")
            continue

        # Apply correction for non-embedding parameter count if available
        corrected_params = get_corrected_param_count(ladder_name, size)
        if corrected_params is not None:
            if verbose:
                print(
                    f"  {size}: Using corrected non-embedding params: {corrected_params/1e6:.1f}M (was {num_params/1e6:.1f}M)"
                )
            num_params = corrected_params

        if use_all_checkpoints:
            # Use all checkpoints from this model size
            checkpoints_added = 0
            for _, row in df.iterrows():
                tokens = row.get("tokens")
                loss = row.get(used_loss_col)
                flops = row.get(flops_col)

                if tokens is None or pd.isna(tokens):
                    continue
                if loss is None or pd.isna(loss):
                    continue

                N_list.append(float(num_params))
                D_list.append(float(tokens))
                L_list.append(float(loss))
                # FLOPs: use measured value if available, otherwise estimate as 6*N*D
                if flops is not None and not pd.isna(flops):
                    # Convert from petaflops to raw FLOPs
                    F_list.append(float(flops) * 1e15)
                else:
                    # Estimate FLOPs as 6*N*D (standard approximation)
                    F_list.append(6.0 * float(num_params) * float(tokens))
                # Label with D/N ratio for identification
                dn_ratio = int(round(tokens / num_params))
                sizes.append(f"{size}@{dn_ratio}")
                checkpoints_added += 1

            if verbose and checkpoints_added > 0:
                print(f"  {size}: N={num_params/1e6:.0f}M, {checkpoints_added} checkpoints")
        else:
            # Legacy behavior: only use final checkpoint
            final_row = df[df["step"] == df["step"].max()].iloc[0]

            tokens = final_row.get("tokens")
            if tokens is None or pd.isna(tokens):
                if verbose:
                    print(f"  Warning: No tokens for {size}, skipping")
                continue

            loss = final_row.get(used_loss_col)
            if loss is None or pd.isna(loss):
                if verbose:
                    print(f"  Warning: No loss value for {size}, skipping")
                continue

            flops = final_row.get(flops_col)

            N_list.append(float(num_params))
            D_list.append(float(tokens))
            L_list.append(float(loss))
            # FLOPs: use measured value if available, otherwise estimate as 6*N*D
            if flops is not None and not pd.isna(flops):
                F_list.append(float(flops) * 1e15)
            else:
                F_list.append(6.0 * float(num_params) * float(tokens))
            sizes.append(size)

            if verbose:
                print(f"  {size}: N={num_params/1e6:.0f}M, D={tokens/1e9:.1f}B, L={loss:.4f}")

    if len(N_list) < 5:
        raise ValueError(f"Need at least 5 data points for fitting, got {len(N_list)}")

    if verbose:
        print(f"  Total data points: {len(N_list)}")

    return np.array(N_list), np.array(D_list), np.array(L_list), np.array(F_list), sizes


def fit_ladder(
    name: str,
    ladder_dir: Path,
    loss_column: Optional[str] = None,
    use_bootstrap: bool = False,
    num_bootstraps: int = 100,
    weight_by_compute: bool = True,
    overestimate_penalty: float = 1.0,
    verbose: bool = True,
    use_all_checkpoints: bool = True,
) -> Tuple[
    ChinchillaParametricFit,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
    Optional[ChinchillaParametricBootstrappedFit],
]:
    """
    Fit Chinchilla scaling law to a ladder.

    Returns:
        fit: The fitted scaling law
        N, D, L, F: The data arrays used for fitting (params, tokens, loss, FLOPs)
        sizes: List of size names
        bootstrap_fit: Bootstrap fit if requested, else None
    """
    print(f"\n{'='*60}")
    print(f"Fitting: {name}")
    print(f"Directory: {ladder_dir}")
    print(f"{'='*60}")

    N, D, L, F, sizes = load_ladder_data(
        ladder_dir, name, loss_column, verbose, use_all_checkpoints=use_all_checkpoints
    )

    # Compute weights
    weights = None
    if weight_by_compute:
        # Weight by sqrt(compute) to emphasize larger-scale runs
        weights = np.sqrt(6 * N * D)
        if verbose:
            print(f"\n  Weighting by sqrt(compute)")

    if verbose:
        print(f"\n  Fitting with {len(N)} data points...")
        if overestimate_penalty != 1.0:
            print(f"  Overestimate penalty: {overestimate_penalty}")

    if use_bootstrap:
        bootstrap_fit = ChinchillaParametricBootstrappedFit.fit(
            N,
            D,
            L,
            num_bootstraps=num_bootstraps,
            weights=weights,
            overestimate_penalty=overestimate_penalty,
            progress_bar=verbose,
        )
        # Return the point estimate for consistency
        return bootstrap_fit.point_estimate, N, D, L, F, sizes, bootstrap_fit
    else:
        fit = ChinchillaParametricFit.fit(
            N,
            D,
            L,
            weights=weights,
            overestimate_penalty=overestimate_penalty,
        )
        return fit, N, D, L, F, sizes, None


def fit_rollout(
    name: str,
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    use_bootstrap: bool = False,
    num_bootstraps: int = 100,
    weight_by_compute: bool = True,
    overestimate_penalty: float = 1.0,
    verbose: bool = True,
) -> ScalingLawRollout:
    """
    Fit scaling law with rollout cross-validation.

    Returns:
        ScalingLawRollout with multiple splits for evaluation.
    """
    if verbose:
        print(f"\n  Fitting rollout cross-validation for {name}...")

    weights = None
    if weight_by_compute:
        weights = np.sqrt(6 * N * D)

    fit_fn = (
        ChinchillaParametricBootstrappedFit.fit if use_bootstrap else ChinchillaParametricFit.fit
    )
    fit_kwargs = {
        "overestimate_penalty": overestimate_penalty,
        "progress_bar": True,
    }
    if use_bootstrap:
        fit_kwargs["num_bootstraps"] = num_bootstraps

    rollout = ScalingLawRollout.fit(
        N=N,
        D=D,
        L=L,
        fit_fn=fit_fn,
        weights=weights,
        split_by="N",
        min_points_train=5,
        min_groups_train=3,
        **fit_kwargs,
    )

    if verbose:
        print(f"  Created {len(rollout.splits)} rollout splits")

    return rollout


def print_fit_results(
    name: str,
    fit: ChinchillaParametricFit,
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    sizes: List[str],
    bootstrap_fit: Optional[ChinchillaParametricBootstrappedFit] = None,
):
    """Print detailed results from a scaling law fit."""
    params = fit.fitted_params

    print(f"\n{'─'*60}")
    print(f"Results for: {name}")
    print(f"{'─'*60}")

    print("\nFitted Chinchilla Parameters:")
    print(f"  L(N, D) = E + A/N^α + B/D^β")
    print(f"  E (entropy floor) = {params.E:.6f}")
    print(f"  A = {params.A:.6e}")
    print(f"  α (alpha) = {params.alpha:.4f}")
    print(f"  B = {params.B:.6e}")
    print(f"  β (beta) = {params.beta:.4f}")

    print(f"\nCompute-Optimal Scaling:")
    print(f"  a_opt = {params.a_opt:.4f}  (N_opt ∝ C^a_opt)")
    print(f"  b_opt = {params.b_opt:.4f}  (D_opt ∝ C^b_opt)")
    print(f"  → For optimal allocation: N/D ratio scales as C^({params.a_opt - params.b_opt:.4f})")

    # Compare to Chinchilla paper values (a=0.50, b=0.50 for equal scaling)
    chinchilla_a, chinchilla_b = 0.50, 0.50
    print(f"\n  Chinchilla paper: a_opt={chinchilla_a}, b_opt={chinchilla_b}")
    if params.a_opt > chinchilla_a + 0.05:
        print(f"  → Your data suggests scaling parameters MORE than Chinchilla")
    elif params.a_opt < chinchilla_a - 0.05:
        print(f"  → Your data suggests scaling data MORE than Chinchilla")
    else:
        print(f"  → Similar to Chinchilla's equal N/D scaling")

    # Show fit quality
    predicted = fit.predict_loss(N, D)
    residuals = L - predicted
    rel_errors = np.abs(residuals) / L * 100

    print(f"\nFit Quality:")
    print(f"  Huber loss: {fit.huber_loss:.6f}")
    print(f"  Mean absolute error: {np.mean(np.abs(residuals)):.6f}")
    print(f"  Mean relative error: {np.mean(rel_errors):.2f}%")
    print(f"  Max relative error: {np.max(rel_errors):.2f}%")

    # Per-size breakdown
    print(f"\nPer-Size Fit:")
    print(f"  {'Size':<8} {'Actual':>10} {'Predicted':>10} {'Error':>10}")
    print(f"  {'-'*38}")
    for i, size in enumerate(sizes):
        print(f"  {size:<8} {L[i]:>10.4f} {predicted[i]:>10.4f} {rel_errors[i]:>9.2f}%")

    # Bootstrap confidence intervals
    if bootstrap_fit is not None:
        print(f"\nBootstrap Confidence Intervals (from {len(bootstrap_fit.fits)} samples):")
        all_params = np.array(
            [
                (
                    f.fitted_params.E,
                    f.fitted_params.A,
                    f.fitted_params.alpha,
                    f.fitted_params.B,
                    f.fitted_params.beta,
                )
                for f in bootstrap_fit.fits
            ]
        )
        param_names = ["E", "A", "α", "B", "β"]
        for i, pname in enumerate(param_names):
            low, high = np.percentile(all_params[:, i], [2.5, 97.5])
            median = np.median(all_params[:, i])
            if pname in ["A", "B"]:
                print(f"  {pname}: {median:.4e} [{low:.4e}, {high:.4e}]")
            else:
                print(f"  {pname}: {median:.4f} [{low:.4f}, {high:.4f}]")


def print_rollout_evaluation(name: str, rollout: ScalingLawRollout):
    """Print rollout cross-validation evaluation results."""
    evaluation = evaluate_rollout(rollout)

    print(f"\n{'─'*60}")
    print(f"Rollout Cross-Validation: {name}")
    print(f"{'─'*60}")

    print("\nOverall Metrics:")
    print(f"  Mean perplexity error: {evaluation.overall_mean_ppl_error:.2f}%")
    print(f"  Weighted perplexity error: {evaluation.overall_weighted_mean_ppl_error:.2f}%")
    print(f"  Mean relative BPB error: {evaluation.overall_mean_relative_error:.2f}%")
    print(f"  Mean signed error: {evaluation.overall_mean_signed_error:+.4f} (+ = overestimate)")

    print("\nPer-Split Results:")
    print(f"  {'Cutoff':>10} {'N_test':>8} {'PPL Err':>10} {'Rel Err':>10} {'Signed':>10}")
    print(f"  {'-'*48}")
    for split_eval in evaluation.split_evaluations:
        cutoff_m = split_eval.cutoff_value / 1e6
        print(
            f"  {cutoff_m:>9.0f}M {split_eval.n_test_points:>8} "
            f"{split_eval.mean_ppl_error:>9.2f}% {split_eval.mean_relative_error:>9.2f}% "
            f"{split_eval.mean_signed_error:>+9.4f}"
        )


def print_comparison(fits: Dict[str, Tuple]):
    """Print comparison of multiple ladder fits."""
    if len(fits) < 2:
        return

    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")

    # Table header
    print(f"\n{'Ladder':<20} {'E':>8} {'α':>8} {'β':>8} {'a_opt':>8} {'b_opt':>8}")
    print("-" * 60)

    for name, (fit, N, D, L, F, sizes, bootstrap) in fits.items():
        p = fit.fitted_params
        print(
            f"{name:<20} {p.E:>8.4f} {p.alpha:>8.4f} {p.beta:>8.4f} {p.a_opt:>8.4f} {p.b_opt:>8.4f}"
        )

    # Compare predictions at a common scale
    print(f"\nPredicted Loss at Common Scales:")
    test_points = [
        (1e9, 20e9, "1B @ 20B tokens"),
        (3e9, 60e9, "3B @ 60B tokens"),
        (7e9, 140e9, "7B @ 140B tokens"),
        (13e9, 260e9, "13B @ 260B tokens"),
    ]

    print(f"{'Scale':<20}", end="")
    for name in fits.keys():
        print(f" {name:>15}", end="")
    print()
    print("-" * (20 + 16 * len(fits)))

    for n, d, label in test_points:
        print(f"{label:<20}", end="")
        for name, (fit, *_) in fits.items():
            pred = fit.predict_loss(n, d)
            print(f" {pred:>15.4f}", end="")
        print()


def plot_fits_2d(
    fits: Dict[str, Tuple],
    output_path: Optional[Path] = None,
    show: bool = True,
):
    """Generate 2D matplotlib plots for scaling law fits."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("Warning: matplotlib not installed, skipping 2D plots")
        return

    n_fits = len(fits)
    if n_fits == 0:
        return

    # Use a nice style
    plt.style.use("seaborn-v0_8-whitegrid")

    # Color palette - distinct colors for each ladder
    color_palette = ["#2ecc71", "#e74c3c", "#3498db", "#9b59b6", "#f39c12", "#1abc9c", "#e67e22"]
    colors = {name: color_palette[i % len(color_palette)] for i, name in enumerate(fits.keys())}

    # Create a cleaner 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Chinchilla Scaling Law Analysis", fontsize=16, fontweight="bold", y=0.98)

    # ============================================================
    # Top Left: Loss vs Parameters (all ladders, with fitted curves)
    # ============================================================
    ax = axes[0, 0]

    # Determine N range from data
    all_N = np.concatenate([N for (_, N, *_) in fits.values()])
    N_min, N_max = all_N.min(), all_N.max()
    N_range = np.logspace(np.log10(N_min * 0.8), np.log10(N_max * 1.2), 200)

    for name, (fit, N, D, L, *_) in fits.items():
        color = colors[name]

        # Get unique model sizes and their mean D/N ratio for the fitted curve
        unique_N = np.unique(N)
        mean_dn_ratios = []
        for n_val in unique_N:
            mask = N == n_val
            mean_dn_ratios.append(np.mean(D[mask]) / n_val)
        overall_dn_ratio = np.mean(mean_dn_ratios)

        # Plot smooth fitted curve using average D/N ratio
        D_curve = N_range * overall_dn_ratio
        loss_curve = fit.predict_loss(N_range, D_curve)
        ax.plot(N_range, loss_curve, color=color, linewidth=2.5, alpha=0.8, label=f"{name} (fit)")

        # Plot actual data points (scatter only, no lines)
        ax.scatter(N, L, color=color, s=40, alpha=0.6, edgecolors="white", linewidth=0.5, zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel("Non-embedding Parameters", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Scaling Laws: Loss vs Parameters", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.tick_params(labelsize=10)

    # ============================================================
    # Top Right: Loss vs Tokens for each model size
    # ============================================================
    ax = axes[0, 1]

    for name, (fit, N, D, L, *_) in fits.items():
        color = colors[name]

        # Group by unique N values and plot loss vs D
        unique_N = np.unique(N)
        for n_val in unique_N:
            mask = N == n_val
            D_subset = D[mask]
            L_subset = L[mask]

            # Sort by D for cleaner visualization
            sort_idx = np.argsort(D_subset)
            ax.scatter(
                D_subset[sort_idx],
                L_subset[sort_idx],
                color=color,
                s=30,
                alpha=0.6,
                edgecolors="white",
                linewidth=0.3,
            )
        del fit  # unused in this loop

    # Create legend manually (one entry per ladder)
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[name], markersize=10, label=name)
        for name in fits.keys()
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=10, framealpha=0.9)

    ax.set_xscale("log")
    ax.set_xlabel("Tokens", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Loss vs Training Tokens", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=10)

    # ============================================================
    # Bottom Left: Compute-optimal scaling comparison
    # ============================================================
    ax = axes[1, 0]

    # Plot at Chinchilla-optimal D = 20N
    N_range_compute = np.logspace(7, 11, 200)
    D_chinchilla = 20 * N_range_compute

    for name, (fit, *_) in fits.items():
        color = colors[name]
        loss_pred = fit.predict_loss(N_range_compute, D_chinchilla)
        ax.plot(
            N_range_compute,
            loss_pred,
            linewidth=2.5,
            label=f"{name} (α={fit.fitted_params.alpha:.2f}, β={fit.fitted_params.beta:.2f})",
            color=color,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Non-embedding Parameters", fontsize=12)
    ax.set_ylabel("Loss (at Chinchilla-optimal D=20N)", fontsize=12)
    ax.set_title("Compute-Optimal Scaling Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.tick_params(labelsize=10)

    # Add reference lines for common model sizes
    ref_sizes = [(1e8, "100M"), (1e9, "1B"), (1e10, "10B")]
    for n_ref, label in ref_sizes:
        if N_range_compute.min() < n_ref < N_range_compute.max():
            ax.axvline(x=n_ref, color="gray", linestyle=":", alpha=0.5, linewidth=1)
            ax.text(n_ref, ax.get_ylim()[1], f" {label}", fontsize=8, alpha=0.7, va="top")

    # ============================================================
    # Bottom Right: Fit residuals
    # ============================================================
    ax = axes[1, 1]

    for name, (fit, N, D, L, *_) in fits.items():
        color = colors[name]
        predicted = fit.predict_loss(N, D)
        residuals_pct = (L - predicted) / L * 100
        ax.scatter(N, residuals_pct, s=40, label=name, color=color, alpha=0.6, edgecolors="white", linewidth=0.5)

    ax.axhline(y=0, color="black", linestyle="-", linewidth=1.5, alpha=0.8)
    ax.axhline(y=1, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(y=-1, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    ax.set_xscale("log")
    ax.set_xlabel("Non-embedding Parameters", fontsize=12)
    ax.set_ylabel("Relative Error (%)", fontsize=12)
    ax.set_title("Fit Residuals: (Actual - Predicted) / Actual", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.tick_params(labelsize=10)

    # Add shaded region for "good fit" zone
    ylim = ax.get_ylim()
    max_abs = max(abs(ylim[0]), abs(ylim[1]))
    ax.set_ylim(-max_abs * 1.1, max_abs * 1.1)
    ax.fill_between(
        ax.get_xlim(), -1, 1, alpha=0.1, color="green", label="±1% zone"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        fig.savefig(output_path / "chinchilla_scaling_fits_2d.png", dpi=200, bbox_inches="tight", facecolor="white")
        print(f"\nSaved 2D plot to: {output_path / 'chinchilla_scaling_fits_2d.png'}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_fits_3d(
    rollouts: Dict[str, ScalingLawRollout],
    output_path: Optional[Path] = None,
    show: bool = True,
):
    """Generate interactive 3D Plotly visualizations."""
    if not rollouts:
        print("No rollouts to plot")
        return

    rollout_list = list(rollouts.items())

    # Single rollout: use plot_scaling_law_3d
    if len(rollout_list) == 1:
        name, rollout = rollout_list[0]
        safe_name = name.lower().replace(" ", "_")
        save_path = output_path / f"{safe_name}_3d.html" if output_path else None
        fig = plot_scaling_law_3d(
            rollout,
            subtitle=name,
            save_path=save_path,
        )
        if show:
            fig.show()
        if output_path:
            print(f"Saved 3D plot to: {save_path}")

    # Two rollouts: use comparison plot
    elif len(rollout_list) == 2:
        (name_a, rollout_a), (name_b, rollout_b) = rollout_list
        fig = plot_scaling_law_3d_comparison(
            (name_a, rollout_a),
            (name_b, rollout_b),
            subtitle="Scaling Law Comparison",
            save_path=output_path / "scaling_comparison_3d.html" if output_path else None,
        )
        if show:
            fig.show()
        if output_path:
            print(f"Saved 3D comparison to: {output_path / 'scaling_comparison_3d.html'}")

    # More than two: generate individual plots
    else:
        for name, rollout in rollout_list:
            safe_name = name.lower().replace(" ", "_")
            fig = plot_scaling_law_3d(
                rollout,
                subtitle=name,
                save_path=output_path / f"{safe_name}_3d.html" if output_path else None,
            )
            if show:
                fig.show()
            if output_path:
                print(f"Saved 3D plot to: {output_path / f'{safe_name}_3d.html'}")


def plot_isoflop_curves(
    fits: Dict[str, Tuple],
    output_path: Optional[Path] = None,
    show: bool = True,
):
    """Plot iso-FLOP curves showing optimal N vs D tradeoffs."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed, skipping iso-FLOP plots")
        return

    if not fits:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(fits)))

    # FLOPs = 6 * N * D (approximate)
    flop_budgets = [1e18, 1e19, 1e20, 1e21, 1e22]  # 1e18 to 1e22 FLOPs

    # Left: Loss vs N for different FLOP budgets
    ax = axes[0]
    for idx, (name, (fit, N, D, L, F, sizes, bootstrap)) in enumerate(fits.items()):
        for flop_idx, flops in enumerate(flop_budgets):
            N_range = np.logspace(7, 11, 100)
            D_range = flops / (6 * N_range)
            # Only plot where D is reasonable (> 1B tokens)
            valid = D_range > 1e9
            if not np.any(valid):
                continue

            loss_pred = fit.predict_loss(N_range[valid], D_range[valid])
            alpha = 0.3 + 0.7 * (flop_idx / len(flop_budgets))
            label = f"{name} ({flops:.0e} FLOPs)" if flop_idx == len(flop_budgets) - 1 else None
            ax.plot(
                N_range[valid],
                loss_pred,
                color=colors[idx],
                alpha=alpha,
                linewidth=2 if flop_idx == len(flop_budgets) - 1 else 1,
                label=label,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (N)", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("Iso-FLOP Curves\n(lighter = less compute)", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: Optimal N vs FLOP budget
    ax = axes[1]
    flop_range = np.logspace(17, 23, 100)

    for idx, (name, (fit, N, D, L, F, sizes, bootstrap)) in enumerate(fits.items()):
        params = fit.fitted_params
        # Compute-optimal: N_opt ∝ C^a_opt, where C = 6*N*D
        # For a given FLOP budget F = 6*N*D, optimal N scales as F^a_opt
        # This is approximate; exact solution requires solving the optimality conditions
        N_opt = (flop_range / 6) ** params.a_opt * (1 / 20) ** (params.a_opt * params.b_opt)

        ax.plot(
            flop_range,
            N_opt,
            linewidth=2,
            label=f"{name} (a={params.a_opt:.3f})",
            color=colors[idx],
        )

    # Add Chinchilla reference line
    N_chinchilla = (flop_range / 6) ** 0.5 * (1 / 20) ** 0.25
    ax.plot(flop_range, N_chinchilla, "k--", linewidth=1, label="Chinchilla (a=0.5)", alpha=0.7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute Budget (FLOPs)", fontsize=11)
    ax.set_ylabel("Optimal Parameters (N)", fontsize=11)
    ax.set_title("Compute-Optimal Model Size", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path / "isoflop_curves.png", dpi=150, bbox_inches="tight")
        print(f"\nSaved iso-FLOP plot to: {output_path / 'isoflop_curves.png'}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_loss_vs_flops(
    fits: Dict[str, Tuple],
    output_path: Optional[Path] = None,
    show: bool = True,
):
    """Plot loss vs FLOPs for all models across all ladders."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("Warning: matplotlib not installed, skipping loss vs FLOPs plot")
        return

    if not fits:
        return

    # Use a nice style
    plt.style.use("seaborn-v0_8-whitegrid")

    # Color palette - distinct colors for each ladder
    color_palette = ["#2ecc71", "#e74c3c", "#3498db", "#9b59b6", "#f39c12", "#1abc9c", "#e67e22", "#34495e", "#e91e63", "#00bcd4"]
    colors = {name: color_palette[i % len(color_palette)] for i, name in enumerate(fits.keys())}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Loss vs Training FLOPs", fontsize=16, fontweight="bold", y=0.98)

    # ============================================================
    # Left: Loss vs FLOPs (scatter plot with all data points)
    # ============================================================
    ax = axes[0]

    for name, (fit, N, D, L, F, sizes, bootstrap) in fits.items():
        color = colors[name]
        # F is in raw FLOPs, convert to PetaFLOPs for plotting
        F_peta = F / 1e15
        ax.scatter(F_peta, L, color=color, s=40, alpha=0.6, edgecolors="white", linewidth=0.5, label=name)

    ax.set_xscale("log")
    ax.set_xlabel("Training FLOPs (PetaFLOPs)", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("All Checkpoints", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3)

    # ============================================================
    # Right: Loss vs FLOPs with fitted lines per model size
    # ============================================================
    ax = axes[1]

    for name, (fit, N, D, L, F, sizes, bootstrap) in fits.items():
        color = colors[name]
        F_peta = F / 1e15

        # Group by unique N values and plot loss vs FLOPs for each size
        unique_N = np.unique(N)
        for n_val in unique_N:
            mask = N == n_val
            F_subset = F_peta[mask]
            L_subset = L[mask]

            # Sort by FLOPs for cleaner lines
            sort_idx = np.argsort(F_subset)
            ax.plot(
                F_subset[sort_idx],
                L_subset[sort_idx],
                color=color,
                linewidth=1.5,
                alpha=0.7,
                marker='o',
                markersize=4,
            )

    # Create legend manually (one entry per ladder)
    legend_handles = [
        Line2D([0], [0], marker="o", color=colors[name], markersize=8, label=name, linewidth=1.5)
        for name in fits.keys()
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.9)

    ax.set_xscale("log")
    ax.set_xlabel("Training FLOPs (PetaFLOPs)", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Per Model Size", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        save_path = output_path / "loss_vs_flops.png"
        fig.savefig(str(save_path), dpi=200, bbox_inches="tight", facecolor="white")
        print(f"\nSaved loss vs FLOPs plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def predict_loss_for_target(
    fits: Dict[str, Tuple],
    target_n: float,
    target_d: float,
):
    """Predict loss for a target (N, D) allocation."""
    print(f"\n{'='*60}")
    print(f"Loss Predictions for N={target_n:.2e}, D={target_d:.2e}")
    print(f"(Chinchilla ratio: D/N = {target_d/target_n:.1f})")
    print(f"{'='*60}")

    for name, (fit, N, D, L, F, sizes, bootstrap) in fits.items():
        pred = fit.predict_loss(target_n, target_d)
        print(f"\n{name}:")
        print(f"  Predicted loss: {pred:.4f}")

        if bootstrap is not None:
            loss_dist = bootstrap.predict_loss_distribution(target_n, target_d)
            ci_low, ci_high = np.percentile(loss_dist, [2.5, 97.5])
            print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")


def main():
    parser = argparse.ArgumentParser(
        description="Fit Chinchilla scaling laws to model ladder results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--ladder",
        action="append",
        required=True,
        metavar="NAME:PATH",
        help="Ladder to fit. Format: name:path (can specify multiple times)",
    )
    parser.add_argument(
        "--loss-column",
        type=str,
        default=None,
        help=f"Loss column to use (default: {DEFAULT_LOSS_COLUMN})",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        metavar="N",
        help="Number of bootstrap samples for uncertainty estimation (0 = no bootstrap)",
    )
    parser.add_argument(
        "--rollout",
        action="store_true",
        help="Perform rollout cross-validation to evaluate extrapolation",
    )
    parser.add_argument(
        "--no-weight",
        action="store_true",
        help="Don't weight by compute (default: weight by sqrt(6*N*D))",
    )
    parser.add_argument(
        "--overestimate-penalty",
        type=float,
        default=1.0,
        help="Penalty for overestimates in loss function (>1 penalizes overestimates more)",
    )
    parser.add_argument(
        "--predict-n",
        type=float,
        default=None,
        help="Predict loss for this many parameters",
    )
    parser.add_argument(
        "--predict-d",
        type=float,
        default=None,
        help="Predict loss for this many tokens",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate 2D matplotlib visualization plots",
    )
    parser.add_argument(
        "--plot-3d",
        action="store_true",
        help="Generate interactive 3D Plotly visualizations (requires --rollout)",
    )
    parser.add_argument(
        "--plot-isoflop",
        action="store_true",
        help="Generate iso-FLOP curve plots",
    )
    parser.add_argument(
        "--plot-flops",
        action="store_true",
        help="Generate loss vs FLOPs plots",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for plots and results",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )
    parser.add_argument(
        "--final-checkpoints-only",
        action="store_true",
        help="Only use final checkpoint from each model size (legacy behavior). "
        "By default, all checkpoints are used for better fitting.",
    )

    args = parser.parse_args()

    # Parse ladder arguments
    ladders = {}
    for spec in args.ladder:
        if ":" in spec:
            name, path = spec.split(":", 1)
        else:
            path = spec
            name = Path(path).name
        ladders[name] = Path(path).expanduser()

    # Fit each ladder
    fits: Dict[str, Tuple] = {}
    rollouts: Dict[str, ScalingLawRollout] = {}

    for name, ladder_dir in ladders.items():
        try:
            result = fit_ladder(
                name=name,
                ladder_dir=ladder_dir,
                loss_column=args.loss_column,
                use_bootstrap=args.bootstrap > 0,
                num_bootstraps=args.bootstrap,
                weight_by_compute=not args.no_weight,
                overestimate_penalty=args.overestimate_penalty,
                verbose=not args.quiet,
                use_all_checkpoints=not args.final_checkpoints_only,
            )
            fit, N, D, L, F, sizes, bootstrap = result
            fits[name] = (fit, N, D, L, F, sizes, bootstrap)

            print_fit_results(name, fit, N, D, L, sizes, bootstrap)

            # Fit rollout if requested
            if args.rollout or args.plot_3d:
                rollout = fit_rollout(
                    name=name,
                    N=N,
                    D=D,
                    L=L,
                    use_bootstrap=args.bootstrap > 0,
                    num_bootstraps=max(args.bootstrap, 50),  # At least 50 for rollout
                    weight_by_compute=not args.no_weight,
                    overestimate_penalty=args.overestimate_penalty,
                    verbose=not args.quiet,
                )
                rollouts[name] = rollout
                print_rollout_evaluation(name, rollout)

        except Exception as e:
            print(f"\nError fitting {name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    if not fits:
        print("\nNo successful fits!")
        return

    # Print comparison if multiple ladders
    print_comparison(fits)

    # Predict for target if specified
    if args.predict_n is not None and args.predict_d is not None:
        predict_loss_for_target(fits, args.predict_n, args.predict_d)
    elif args.predict_n is not None:
        target_d = 20 * args.predict_n
        print(f"\n(Using Chinchilla-optimal D = 20 × N = {target_d:.2e})")
        predict_loss_for_target(fits, args.predict_n, target_d)

    # Create output directory if needed
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)

    # Generate 2D plots
    if args.plot:
        plot_fits_2d(fits, args.output, show=(args.output is None))

    # Generate iso-FLOP plots
    if args.plot_isoflop:
        plot_isoflop_curves(fits, args.output, show=(args.output is None))

    # Generate loss vs FLOPs plots
    if args.plot_flops:
        plot_loss_vs_flops(fits, args.output, show=(args.output is None))

    # Generate 3D plots
    if args.plot_3d:
        if not rollouts:
            print("\nWarning: --plot-3d requires --rollout, skipping 3D plots")
        else:
            plot_fits_3d(rollouts, args.output, show=(args.output is None))

    # Save results
    if args.output:
        # Save fitted parameters as CSV
        rows = []
        for name, (fit, N, D, L, F, sizes, bootstrap) in fits.items():
            p = fit.fitted_params
            rows.append(
                {
                    "ladder": name,
                    "E": p.E,
                    "A": p.A,
                    "alpha": p.alpha,
                    "B": p.B,
                    "beta": p.beta,
                    "a_opt": p.a_opt,
                    "b_opt": p.b_opt,
                    "huber_loss": fit.huber_loss,
                }
            )
        pd.DataFrame(rows).to_csv(args.output / "chinchilla_fits.csv", index=False)
        print(f"\nSaved fit parameters to: {args.output / 'chinchilla_fits.csv'}")

        # Save per-point data
        all_data = []
        for name, (fit, N, D, L, F, sizes, bootstrap) in fits.items():
            predicted = fit.predict_loss(N, D)
            for i in range(len(N)):
                all_data.append(
                    {
                        "ladder": name,
                        "size": sizes[i],
                        "N": N[i],
                        "D": D[i],
                        "FLOPs": F[i],
                        "FLOPs_petaflops": F[i] / 1e15,
                        "actual_loss": L[i],
                        "predicted_loss": predicted[i],
                        "error": L[i] - predicted[i],
                        "rel_error_pct": (L[i] - predicted[i]) / L[i] * 100,
                    }
                )
        pd.DataFrame(all_data).to_csv(args.output / "chinchilla_predictions.csv", index=False)
        print(f"Saved predictions to: {args.output / 'chinchilla_predictions.csv'}")

        # Save rollout evaluations if available
        if rollouts:
            rollout_rows = []
            for name, rollout in rollouts.items():
                evaluation = evaluate_rollout(rollout)
                for split_eval in evaluation.split_evaluations:
                    rollout_rows.append(
                        {
                            "ladder": name,
                            "cutoff_M": split_eval.cutoff_value / 1e6,
                            "n_test_points": split_eval.n_test_points,
                            "mean_ppl_error_pct": split_eval.mean_ppl_error,
                            "mean_relative_error_pct": split_eval.mean_relative_error,
                            "mean_signed_error": split_eval.mean_signed_error,
                        }
                    )
            pd.DataFrame(rollout_rows).to_csv(args.output / "rollout_evaluation.csv", index=False)
            print(f"Saved rollout evaluation to: {args.output / 'rollout_evaluation.csv'}")


if __name__ == "__main__":
    main()
