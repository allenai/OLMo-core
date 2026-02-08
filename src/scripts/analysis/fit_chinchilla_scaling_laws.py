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
from olmo_core.model_ladder.analysis.model_specs import OLMO3_SPECS_BY_NAME, compute_specs_for_size
from olmo_core.model_ladder.analysis.model_specs import fmt as fmt_number

# Default loss column to use for fitting
DEFAULT_LOSS_COLUMN = "eval/lm/c4_en/CE loss"

# Alternative loss columns to try if default not found
FALLBACK_LOSS_COLUMNS = [
    "eval/lm/pile/CE loss",
    "eval/lm/dolma_common-crawl/CE loss",
    "eval/lm/wikitext_103/CE loss",
    "train/CE loss (log scale)",
]

# Legacy non-embedding parameter counts for architectures that cannot yet be
# computed from ModelSpec (Mamba variants).  All other architectures (OLMo3,
# hybrid-GDN, pure-GDN) are now computed dynamically via compute_specs_for_size().
_LEGACY_PARAM_COUNTS = {
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

# Human-readable display names for each ladder.
# Used in plots, tables, and printed output.
DISPLAY_NAMES: Dict[str, str] = {
    "olmo3": "Olmo 3",
    "olmo3-1": "Olmo 3 v1",
    "olmo3-2": "Olmo 3 v2",
    "olmo3-3": "Olmo 3 v3",
    "pure-gdn": "Pure GDN",
    "hybrid-gdn": "Hybrid GDN (1/4)",
    "hybrid-gdn-half": "Hybrid GDN (1/2)",
    "hybrid-gdn-eight": "Hybrid GDN (1/8)",
    "hybrid-gdn-middle": "Hybrid GDN (Middle)",
    "pure-mamba": "Pure Mamba",
    "hybrid-mamba": "Hybrid Mamba",
}


def get_display_name(ladder_name: str) -> str:
    """Return a human-readable display name for a ladder, falling back to the raw name."""
    return DISPLAY_NAMES.get(ladder_name.lower(), ladder_name)


def shade_color(hex_color: str, fraction: float) -> Tuple[float, float, float]:
    """Blend a hex color with white. fraction=0 → white, fraction=1 → original color."""
    r = int(hex_color[1:3], 16) / 255.0
    g = int(hex_color[3:5], 16) / 255.0
    b = int(hex_color[5:7], 16) / 255.0
    return (1 - fraction + fraction * r, 1 - fraction + fraction * g, 1 - fraction + fraction * b)


def get_corrected_param_count(ladder_name: str, size: str) -> Optional[int]:
    """
    Get the non-embedding parameter count for a given ladder and size.

    First tries to compute from architecture specs via :func:`compute_specs_for_size`.
    Falls back to ``_LEGACY_PARAM_COUNTS`` for architectures that cannot be computed
    (Mamba variants).

    Args:
        ladder_name: Name of the ladder (e.g., "olmo3-1", "hybrid-gdn")
        size: Model size string (e.g., "60M", "190M")

    Returns:
        Non-embedding parameter count if found/computable, None otherwise.
    """
    # Try computing from architecture specs first
    computed = compute_specs_for_size(ladder_name, size)
    if computed is not None:
        return computed["non_embed_params"]

    # Fallback to legacy hardcoded values (Mamba variants)
    ladder_lower = ladder_name.lower()
    if ladder_lower in _LEGACY_PARAM_COUNTS:
        size_map = _LEGACY_PARAM_COUNTS[ladder_lower]
        if size in size_map:
            return size_map[size]

    for pattern, size_map in _LEGACY_PARAM_COUNTS.items():
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
    simple_flops: bool = False,
    post_decay_only: bool = True,
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
        simple_flops: If True, use 6*N*D approximation for FLOPs instead of logged values.
        post_decay_only: If True (default), only use post-decay checkpoints
            (D/N = 10, 20, 40, 80, 160, ...). If False, also include pre-decay.

    Returns:
        N: Array of parameter counts (corrected non-embedding counts if available)
        D: Array of token counts
        L: Array of loss values
        F: Array of FLOPs (raw FLOPs, either from logged values or 6*N*D approximation)
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

        computed = compute_specs_for_size(ladder_name, size)
        if not computed:
            if verbose:
                print(f"  Warning: Could not compute specs for {size}, skipping")
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
        original_num_params = num_params

        # Apply correction for non-embedding parameter count if available
        corrected_params = get_corrected_param_count(ladder_name, size)
        if corrected_params is not None:
            if verbose:
                print(
                    f"  {size}: Using corrected non-embedding params: {corrected_params/1e6:.1f}M (was {num_params/1e6:.1f}M)"
                )
            num_params = corrected_params

        # Canonical D/N values for snapping (absorb batch-size rounding jitter).
        # Always snap against the full set so pre-decay checkpoints don't get
        # mis-snapped to post-decay values.  Then filter by post_decay_only.
        if ladder_name in ["hybrid-gdn", "hybrid-gdn-eight", "hybrid-gdn-half"] and size == "760M":
            _POST_DECAY_DN = [9, 18, 36, 72, 144]
            _PRE_DECAY_DN = [8, 17, 34, 68, 137]
        else:
            _POST_DECAY_DN = [10, 20, 40, 80, 160]
            _PRE_DECAY_DN = [9, 19, 38, 76, 152]
        _ALL_DN = sorted(_POST_DECAY_DN + _PRE_DECAY_DN)
        _KEEP_DN = set(_POST_DECAY_DN) if post_decay_only else set(_ALL_DN)

        if use_all_checkpoints:
            # Use all checkpoints from this model size
            checkpoints_added = 0
            num_params = computed["non_embed_params"]
            flops_per_token = computed["total_flops_per_token"]  # forward-pass FLOPs per token (2 * MACs)

            for _, row in df.iterrows():
                tokens = row.get("tokens")
                loss = row.get(used_loss_col)

                if tokens is None or pd.isna(tokens):
                    continue
                if loss is None or pd.isna(loss):
                    continue

                # Snap D/N to nearest canonical value, then keep only desired checkpoints.
                raw_dn = tokens / original_num_params if original_num_params else 0
                if raw_dn <= 0:
                    continue
                dn_ratio = min(_ALL_DN, key=lambda c: abs(c - raw_dn))
                if dn_ratio not in _KEEP_DN:
                    if verbose:
                        print(
                            f"    SKIP {size}: step={row.get('step')}, "
                            f"tokens={tokens:.3e}, raw_dn={raw_dn:.2f}, "
                            f"snapped={dn_ratio}, loss={loss:.4f} (not in _KEEP_DN)"
                        )
                    continue

                if verbose:
                    print(
                        f"    KEEP {size}: step={row.get('step')}, "
                        f"tokens={tokens:.3e}, raw_dn={raw_dn:.2f}, "
                        f"snapped={dn_ratio}, loss={loss:.4f}"
                    )

                N_list.append(float(num_params))
                D_list.append(float(tokens))
                L_list.append(float(loss))
                # Training FLOPs = 3 * forward_flops_per_token * tokens
                # (1 forward + ~2 backward passes)
                if simple_flops:
                    F_list.append(6.0 * float(num_params) * float(tokens))
                else:
                    F_list.append(3.0 * flops_per_token * float(tokens))
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
            # FLOPs: use 6*N*D approximation if requested, otherwise use measured value
            if simple_flops:
                F_list.append(6.0 * float(num_params) * float(tokens))
            elif flops is not None and not pd.isna(flops):
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
    simple_flops: bool = False,
    num_slices: int = 4,
    post_decay_only: bool = True,
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
    display = get_display_name(name)
    print(f"\n{'='*60}")
    print(f"Fitting: {display}")
    print(f"Directory: {ladder_dir}")
    print(f"{'='*60}")

    N, D, L, F, sizes = load_ladder_data(
        ladder_dir,
        name,
        loss_column,
        verbose,
        use_all_checkpoints=use_all_checkpoints,
        simple_flops=simple_flops,
        post_decay_only=post_decay_only,
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
            num_slices=num_slices,
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
            num_slices=num_slices,
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
    num_slices: int = 4,
) -> ScalingLawRollout:
    """
    Fit scaling law with rollout cross-validation.

    Returns:
        ScalingLawRollout with multiple splits for evaluation.
    """
    if verbose:
        print(f"\n  Fitting rollout cross-validation for {get_display_name(name)}...")

    weights = None
    if weight_by_compute:
        weights = np.sqrt(6 * N * D)

    fit_fn = (
        ChinchillaParametricBootstrappedFit.fit if use_bootstrap else ChinchillaParametricFit.fit
    )
    fit_kwargs = {
        "overestimate_penalty": overestimate_penalty,
        "num_slices": num_slices,
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

    display = get_display_name(name)
    print(f"\n{'─'*60}")
    print(f"Results for: {display}")
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

    display = get_display_name(name)
    print(f"\n{'─'*60}")
    print(f"Rollout Cross-Validation: {display}")
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
    display_names = {name: get_display_name(name) for name in fits}
    max_name_len = max(len(d) for d in display_names.values())
    col_w = max(max_name_len, 20)
    print(f"\n{'Ladder':<{col_w}} {'E':>8} {'α':>8} {'β':>8} {'a_opt':>8} {'b_opt':>8}")
    print("-" * (col_w + 40))

    for name, (fit, N, D, L, F, sizes, bootstrap) in fits.items():
        p = fit.fitted_params
        print(
            f"{display_names[name]:<{col_w}} {p.E:>8.4f} {p.alpha:>8.4f} {p.beta:>8.4f} {p.a_opt:>8.4f} {p.b_opt:>8.4f}"
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
        print(f" {display_names[name]:>15}", end="")
    print()
    print("-" * (20 + 16 * len(fits)))

    for n, d, label in test_points:
        print(f"{label:<20}", end="")
        for name, (fit, *_) in fits.items():
            pred = fit.predict_loss(n, d)
            print(f" {pred:>15.4f}", end="")
        print()


# =============================================================================
# Solve helpers for efficiency projections
# =============================================================================


def solve_d_for_target_loss(fit: ChinchillaParametricFit, target_loss: float, N: float) -> float:
    """
    Solve L(N, D) = target_loss for D analytically.

    L(N, D) = E + A/N^α + B/D^β = target_loss
    => D = (B / (target_loss - E - A/N^α))^(1/β)

    Returns float('inf') if target_loss is unreachable at this N.
    """
    p = fit.fitted_params
    remainder = target_loss - p.E - p.A / (N**p.alpha)
    if remainder <= 0:
        return float("inf")
    return (p.B / remainder) ** (1.0 / p.beta)


def solve_n_for_target_loss(fit: ChinchillaParametricFit, target_loss: float, D: float) -> float:
    """
    Solve L(N, D) = target_loss for N analytically.

    L(N, D) = E + A/N^α + B/D^β = target_loss
    => N = (A / (target_loss - E - B/D^β))^(1/α)

    Returns float('inf') if target_loss is unreachable at this D.
    """
    p = fit.fitted_params
    remainder = target_loss - p.E - p.B / (D**p.beta)
    if remainder <= 0:
        return float("inf")
    return (p.A / remainder) ** (1.0 / p.alpha)


# =============================================================================
# Domain-specific data loading and fitting
# =============================================================================


def load_ladder_domain_data(
    ladder_dir: Path,
    ladder_name: str,
    domains: Optional[List[str]] = None,
    verbose: bool = True,
    post_decay_only: bool = True,
    simple_flops: bool = False,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]]:
    """
    Load ladder results with per-domain BPB metrics for domain-specific scaling law fitting.

    Args:
        ladder_dir: Directory containing metrics_*.pkl files
        ladder_name: Name of the ladder (used for parameter count corrections)
        domains: Which domain clusters to extract (default: Math_BPB, Code_BPB, QA_BPB)
        verbose: Print loading progress
        post_decay_only: If True, only use post-decay checkpoints
        simple_flops: If True, use 6*N*D approximation for FLOPs

    Returns:
        Dict mapping domain name -> (N, D, L_domain, F, sizes)
    """
    from olmo_core.model_ladder.analysis.metrics import (
        BASE_EASY_SUITE,
        aggregate_base_easy_cluster_for_row,
        find_bpb_columns,
    )

    if domains is None:
        domains = ["Math_BPB", "Code_BPB", "QA_BPB"]

    pkl_files = sorted(ladder_dir.glob("metrics_*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No metrics_*.pkl files found in {ladder_dir}")

    # Accumulators per domain
    domain_data: Dict[str, Dict[str, list]] = {
        d: {"N": [], "D": [], "L": [], "F": [], "sizes": []} for d in domains
    }

    _POST_DECAY_DN = [10, 20, 40, 80, 160]
    _PRE_DECAY_DN = [9, 19, 38, 76, 152]
    _ALL_DN = sorted(_POST_DECAY_DN + _PRE_DECAY_DN)
    _KEEP_DN = set(_POST_DECAY_DN) if post_decay_only else set(_ALL_DN)

    # Pre-compute column mappings once per first file
    col_maps: Dict[str, Optional[Dict[str, str]]] = {d: None for d in domains}

    for pkl_path in pkl_files:
        size = pkl_path.stem.replace("metrics_", "")
        try:
            df = pd.read_pickle(pkl_path)
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not load {pkl_path}: {e}")
            continue
        if df.empty:
            continue

        num_params = df["num_params"].iloc[0]
        if num_params is None or pd.isna(num_params):
            continue
        original_num_params = num_params

        corrected_params = get_corrected_param_count(ladder_name, size)
        if corrected_params is not None:
            num_params = corrected_params

        # Build column mappings on first file
        for domain in domains:
            if col_maps[domain] is None and domain in BASE_EASY_SUITE:
                task_patterns = BASE_EASY_SUITE[domain]["tasks"]
                col_maps[domain] = find_bpb_columns(df, task_patterns)

        for _, row in df.iterrows():
            tokens = row.get("tokens")
            if tokens is None or pd.isna(tokens):
                continue

            raw_dn = tokens / original_num_params if original_num_params else 0
            if raw_dn <= 0:
                continue
            dn_ratio = min(_ALL_DN, key=lambda c: abs(c - raw_dn))
            if dn_ratio not in _KEEP_DN:
                continue

            flops = row.get("throughput/total petaflops")
            if simple_flops:
                f_val = 6.0 * float(num_params) * float(tokens)
            elif flops is not None and not pd.isna(flops):
                f_val = float(flops) * 1e15
            else:
                f_val = 6.0 * float(num_params) * float(tokens)

            for domain in domains:
                if domain not in BASE_EASY_SUITE:
                    continue
                bpb = aggregate_base_easy_cluster_for_row(
                    row,
                    df.columns,
                    domain,
                    BASE_EASY_SUITE[domain],
                    precomputed_col_map=col_maps[domain],
                )
                if bpb is not None and bpb > 0:
                    domain_data[domain]["N"].append(float(num_params))
                    domain_data[domain]["D"].append(float(tokens))
                    domain_data[domain]["L"].append(float(bpb))
                    domain_data[domain]["F"].append(f_val)
                    domain_data[domain]["sizes"].append(f"{size}@{dn_ratio}")

    # Convert to arrays
    result: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]] = {}
    for domain in domains:
        dd = domain_data[domain]
        if len(dd["N"]) >= 5:
            result[domain] = (
                np.array(dd["N"]),
                np.array(dd["D"]),
                np.array(dd["L"]),
                np.array(dd["F"]),
                dd["sizes"],
            )
            if verbose:
                print(f"  {domain}: {len(dd['N'])} data points")
        else:
            if verbose:
                print(f"  {domain}: only {len(dd['N'])} data points, skipping (need >= 5)")

    return result


def fit_domain_ladders(
    name: str,
    ladder_dir: Path,
    domains: Optional[List[str]] = None,
    weight_by_compute: bool = True,
    overestimate_penalty: float = 1.0,
    num_slices: int = 4,
    verbose: bool = True,
    post_decay_only: bool = True,
    simple_flops: bool = False,
) -> Dict[
    str, Tuple[ChinchillaParametricFit, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]
]:
    """
    Fit scaling laws per domain for a single ladder.

    Returns:
        Dict mapping domain name -> (fit, N, D, L, F, sizes)
    """
    display = get_display_name(name)
    if verbose:
        print(f"\n  Fitting domain-specific scaling laws for {display}...")

    domain_data = load_ladder_domain_data(
        ladder_dir,
        name,
        domains=domains,
        verbose=verbose,
        post_decay_only=post_decay_only,
        simple_flops=simple_flops,
    )

    results: Dict[
        str,
        Tuple[ChinchillaParametricFit, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]],
    ] = {}
    for domain, (N, D, L, F, sizes) in domain_data.items():
        weights = np.sqrt(6 * N * D) if weight_by_compute else None
        try:
            fit = ChinchillaParametricFit.fit(
                N,
                D,
                L,
                weights=weights,
                overestimate_penalty=overestimate_penalty,
                num_slices=num_slices,
            )
            results[domain] = (fit, N, D, L, F, sizes)
            if verbose:
                p = fit.fitted_params
                print(
                    f"    {domain}: E={p.E:.4f}, A={p.A:.4e}, α={p.alpha:.4f}, "
                    f"B={p.B:.4e}, β={p.beta:.4f}, R²={fit.r_squared:.4f}"
                )
        except Exception as e:
            if verbose:
                print(f"    {domain}: fitting failed: {e}")

    return results


# =============================================================================
# LaTeX table generation
# =============================================================================


def _param_ci_str(bootstrap, param_name: str, point_value: float, fmt: str = ".4f") -> str:
    """Format a parameter value with bootstrap CI below in smaller font using makecell."""
    val_str = f"{point_value:{fmt}}"
    if bootstrap is None:
        ci_line = r"{\color{red}$\pm$ 0.0}"
    else:
        bootstrap_values = [getattr(f.fitted_params, param_name) for f in bootstrap.fits]
        lo = np.percentile(bootstrap_values, 2.5)
        hi = np.percentile(bootstrap_values, 97.5)
        ci_half = (hi - lo) / 2
        ci_line = f"$\\pm$ {ci_half:{fmt}}"
    return r"\makecell{" + val_str + r" \\ {\scriptsize " + ci_line + "}}"


def _param_ci_str_sci(bootstrap, param_name: str, point_value: float) -> str:
    """Format a parameter value (scientific notation) with bootstrap CI below in smaller font."""
    val_str = f"{point_value:.2e}"
    if bootstrap is None:
        ci_line = r"{\color{red}$\pm$ 0.0}"
    else:
        bootstrap_values = [getattr(f.fitted_params, param_name) for f in bootstrap.fits]
        lo = np.percentile(bootstrap_values, 2.5)
        hi = np.percentile(bootstrap_values, 97.5)
        ci_half = (hi - lo) / 2
        ci_line = f"$\\pm$ {ci_half:.2e}"
    return r"\makecell{" + val_str + r" \\ {\scriptsize " + ci_line + "}}"


def _derived_ci_str(bootstrap, attr_name: str, point_value: float, fmt: str = ".4f") -> str:
    """Format a derived property (a_opt/b_opt) with bootstrap CI below in smaller font."""
    val_str = f"{point_value:{fmt}}"
    if bootstrap is None:
        ci_line = r"{\color{red}$\pm$ 0.0}"
    else:
        bootstrap_values = [getattr(f.fitted_params, attr_name) for f in bootstrap.fits]
        lo = np.percentile(bootstrap_values, 2.5)
        hi = np.percentile(bootstrap_values, 97.5)
        ci_half = (hi - lo) / 2
        ci_line = f"$\\pm$ {ci_half:{fmt}}"
    return r"\makecell{" + val_str + r" \\ {\scriptsize " + ci_line + "}}"


def generate_scaling_params_latex_table(fits: Dict[str, Tuple]) -> str:
    """
    Generate LaTeX table of fitted scaling law parameters (Table 1).

    Columns: Architecture, E, A, α, B, β, a_opt, b_opt, R²
    With bootstrap confidence intervals when available.
    """
    lines = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"    \centering")
    lines.append(
        r"    \caption{Fitted Chinchilla scaling law parameters for "
        r"$L(N, D) = E + A/N^\alpha + B/D^\beta$."
    )
    lines.append(
        r"    95\% bootstrap CI shown below each estimate "
        r"({\color{red}red} placeholders indicate no bootstrap was run).}"
    )
    lines.append(r"    \label{tab:scaling-law-parameters}")
    lines.append(r"    \small")
    lines.append(r"    \renewcommand{\arraystretch}{1.2}")
    lines.append(r"    \begin{tabular}{l*{8}{c}}")
    lines.append(r"        \toprule")
    lines.append(
        r"        Architecture & $E$ & $A$ & $\alpha$ & $B$ & $\beta$ "
        r"& $a_\text{opt}$ & $b_\text{opt}$ & $R^2$ \\"
    )
    lines.append(r"        \midrule")

    for name, (fit, N, D, L, F, sizes, bootstrap) in fits.items():
        p = fit.fitted_params
        r2 = fit.r_squared
        r2_str = f"{r2:.4f}" if r2 is not None else "---"
        display = get_display_name(name)

        E_str = _param_ci_str(bootstrap, "E", p.E)
        A_str = _param_ci_str_sci(bootstrap, "A", p.A)
        alpha_str = _param_ci_str(bootstrap, "alpha", p.alpha)
        B_str = _param_ci_str_sci(bootstrap, "B", p.B)
        beta_str = _param_ci_str(bootstrap, "beta", p.beta)
        a_opt_str = _derived_ci_str(bootstrap, "a_opt", p.a_opt)
        b_opt_str = _derived_ci_str(bootstrap, "b_opt", p.b_opt)

        lines.append(
            f"        {display} & {E_str} & {A_str} & {alpha_str} "
            f"& {B_str} & {beta_str} & {a_opt_str} & {b_opt_str} & {r2_str} \\\\"
        )

    # Add efficiency ratio row if exactly 2 ladders
    ladder_names = list(fits.keys())
    if len(ladder_names) == 2:
        fit_a = fits[ladder_names[0]][0]
        fit_b = fits[ladder_names[1]][0]
        pa, pb = fit_a.fitted_params, fit_b.fitted_params
        a_ratio = pa.A / pb.A if pb.A > 0 else float("inf")
        b_ratio = pa.B / pb.B if pb.B > 0 else float("inf")
        lines.append(r"        \midrule")
        lines.append(
            f"        Ratio ({get_display_name(ladder_names[0])} / "
            f"{get_display_name(ladder_names[1])}) "
            f"& --- & {a_ratio:.2f}$\\times$ & --- "
            f"& {b_ratio:.2f}$\\times$ & --- & --- & --- & --- \\\\"
        )

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def generate_model_config_latex_table(ladder_names: List[str]) -> str:
    """
    Generate LaTeX table of model configurations for scaling ladder (Table 2).

    Includes architecture-independent config (d_model, n_heads, n_layers)
    plus per-ladder non-embedding parameter counts.
    """
    # Collect all sizes across ladders by probing which sizes are computable
    all_sizes = set()
    size_order = ["60M", "100M", "190M", "370M", "600M", "760M", "1B"]
    for name in ladder_names:
        for size in size_order:
            if get_corrected_param_count(name, size) is not None:
                all_sizes.add(size)
        # Also check legacy dict for Mamba sizes not in size_order
        for pattern, size_map in _LEGACY_PARAM_COUNTS.items():
            if pattern in name.lower():
                all_sizes.update(size_map.keys())
    sizes = [s for s in size_order if s in all_sizes]

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"    \centering")
    lines.append(r"    \caption{Model configurations used in scaling ladder experiments.}")
    lines.append(r"    \label{tab:scaling-ladder-configs}")

    # Build column spec: Size + 3 config cols + one params column per ladder + D/N
    n_ladders = len(ladder_names)
    col_spec = "l" + "rrr" + "r" * n_ladders + "r"
    lines.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"        \toprule")

    # Header
    header_parts = [r"Size", r"$d$", r"$h$", r"$l$"]
    for name in ladder_names:
        display = get_display_name(name).replace("_", r"\_")
        header_parts.append(f"$N$ ({display})")
    header_parts.append("D/N")
    lines.append("        " + " & ".join(header_parts) + r" \\")
    lines.append(r"        \midrule")

    for size in sizes:
        # Model config columns (shared across architectures)
        spec = OLMO3_SPECS_BY_NAME.get(size)
        if spec is not None:
            row_parts = [size, str(spec.d_model), str(spec.n_heads), str(spec.n_layers)]
        else:
            row_parts = [size, "---", "---", "---"]

        for name in ladder_names:
            params = get_corrected_param_count(name, size)
            if params is not None:
                row_parts.append(f"{params/1e6:.1f}M")
            else:
                row_parts.append("---")
        # Max D/N ratio used
        row_parts.append("160")
        lines.append("        " + " & ".join(row_parts) + r" \\")

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_efficiency_latex_table(
    fits: Dict[str, Tuple],
    target_losses: Optional[List[float]] = None,
    reference_n: float = 7e9,
) -> str:
    """
    Generate LaTeX table of efficiency gains at key loss targets (Table 3).

    For each target loss, computes how many tokens each architecture needs
    at a reference model size to reach that loss.
    """
    ladder_names = list(fits.keys())
    if len(ladder_names) < 2:
        return "% Need at least 2 ladders for efficiency comparison"

    # Auto-determine target losses from the data if not provided
    if target_losses is None:
        # Use a range of losses that span the data
        all_L = np.concatenate([L for (_, _, _, L, _, _, _) in fits.values()])
        l_min, l_max = all_L.min(), all_L.max()
        target_losses = np.linspace(l_max * 0.95, l_min * 1.05, 5).tolist()
        target_losses = [round(val, 3) for val in target_losses]

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"    \centering")
    lines.append(
        r"    \caption{Projected efficiency gains at key loss milestones. "
        r"Loss targets are 5 linearly-spaced values spanning the range of observed training losses.}"
    )
    lines.append(r"    \label{tab:scaling-efficiency-gains}")
    lines.append(r"    \begin{tabular}{r" + "rr" * len(ladder_names) + r"r}")
    lines.append(r"        \toprule")

    # Header row 1
    header1_parts = [r"Target"]
    for name in ladder_names:
        display = get_display_name(name).replace("_", r"\_")
        header1_parts.append(r"\multicolumn{2}{c}{" + display + "}")
    header1_parts.append("")
    lines.append("        " + " & ".join(header1_parts) + r" \\")

    # Header row 2
    header2_parts = [r"Loss"]
    for _ in ladder_names:
        header2_parts.extend(["$D$ (M)", "$N$ (M)"])
    header2_parts.append("Savings")
    lines.append("        " + " & ".join(header2_parts) + r" \\")
    lines.append(r"        \midrule")

    for target_loss in target_losses:
        row_parts = [f"{target_loss:.3f}"]
        d_values = []
        for name in ladder_names:
            fit = fits[name][0]
            d_needed = solve_d_for_target_loss(fit, target_loss, reference_n)
            n_needed = solve_n_for_target_loss(fit, target_loss, 20 * reference_n)
            d_values.append(d_needed)
            if np.isfinite(d_needed):
                row_parts.append(f"{d_needed/1e6:.0f}")
            else:
                row_parts.append(r"$\infty$")
            if np.isfinite(n_needed):
                row_parts.append(f"{n_needed/1e6:.0f}")
            else:
                row_parts.append(r"$\infty$")

        # Data savings ratio (first ladder / second ladder)
        if (
            len(d_values) >= 2
            and np.isfinite(d_values[0])
            and np.isfinite(d_values[1])
            and d_values[1] > 0
        ):
            ratio = d_values[0] / d_values[1]
            row_parts.append(f"{ratio:.1f}$\\times$")
        else:
            row_parts.append("---")

        lines.append("        " + " & ".join(row_parts) + r" \\")

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# =============================================================================
# pgfplots/TikZ figure generation
# =============================================================================


def _escape_latex(s: str) -> str:
    """Escape special LaTeX characters in text strings."""
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")


# Import LaTeX plot styles from the metrics analysis script
try:
    from ladder_metrics_analysis import LATEX_PLOT_STYLES, _get_latex_style
except ImportError:
    # Fallback styles if ladder_metrics_analysis is not importable (AI2 color scheme)
    LATEX_PLOT_STYLES: Dict[str, Dict[str, str]] = {  # type: ignore[no-redef]
        "olmo3": {
            "color_def": r"\definecolor{clrTransformer}{HTML}{012E59}",
            "color_name": "clrTransformer",
            "mark": "square*",
            "mark_options": "fill=clrTransformer",
            "line_style": "",
        },
        "olmo3-1": {
            "color_def": r"\definecolor{clrTransformer}{HTML}{012E59}",
            "color_name": "clrTransformer",
            "mark": "square*",
            "mark_options": "fill=clrTransformer",
            "line_style": "",
        },
        "olmo3-2": {
            "color_def": r"\definecolor{clrTransformerV2}{HTML}{265ED4}",
            "color_name": "clrTransformerV2",
            "mark": "square*",
            "mark_options": "fill=clrTransformerV2",
            "line_style": "",
        },
        "olmo3-3": {
            "color_def": r"\definecolor{clrTransformerV3}{HTML}{00D5FF}",
            "color_name": "clrTransformerV3",
            "mark": "square*",
            "mark_options": "fill=clrTransformerV3",
            "line_style": "",
        },
        "pure-gdn": {
            "color_def": r"\definecolor{clrGDN}{HTML}{FF9100}",
            "color_name": "clrGDN",
            "mark": "triangle*",
            "mark_options": "fill=clrGDN",
            "line_style": "",
        },
        "pure-mamba": {
            "color_def": r"\definecolor{clrMamba}{HTML}{B86800}",
            "color_name": "clrMamba",
            "mark": "diamond*",
            "mark_options": "fill=clrMamba",
            "line_style": "",
        },
        "hybrid-gdn": {
            "color_def": r"\definecolor{clrHybGDN}{HTML}{F0529C}",
            "color_name": "clrHybGDN",
            "mark": "*",
            "mark_options": "fill=clrHybGDN",
            "line_style": "",
        },
        "hybrid-gdn-half": {
            "color_def": r"\definecolor{clrHybGDNHalf}{HTML}{C4387E}",
            "color_name": "clrHybGDNHalf",
            "mark": "square",
            "mark_options": "draw=clrHybGDNHalf, thick",
            "line_style": "dashed",
        },
        "hybrid-gdn-eight": {
            "color_def": r"\definecolor{clrHybGDNEight}{HTML}{A02060}",
            "color_name": "clrHybGDNEight",
            "mark": "triangle",
            "mark_options": "draw=clrHybGDNEight, thick",
            "line_style": "densely dashed",
        },
        "hybrid-gdn-middle": {
            "color_def": r"\definecolor{clrHybGDNMid}{HTML}{009BB8}",
            "color_name": "clrHybGDNMid",
            "mark": "pentagon*",
            "mark_options": "fill=clrHybGDNMid",
            "line_style": "densely dotted",
        },
        "hybrid-mamba": {
            "color_def": r"\definecolor{clrHybMamba}{HTML}{265ED4}",
            "color_name": "clrHybMamba",
            "mark": "diamond",
            "mark_options": "draw=clrHybMamba, thick",
            "line_style": "densely dotted",
        },
        "hybrid-gdn-middle-no-final": {
            "color_def": r"\definecolor{clrHybGDNMidNoFinal}{HTML}{007A94}",
            "color_name": "clrHybGDNMidNoFinal",
            "mark": "pentagon",
            "mark_options": "draw=clrHybGDNMidNoFinal, thick",
            "line_style": "densely dotted",
        },
    }

    _FALLBACK_COLORS = [
        ("clrFallbackA", "012E59"),
        ("clrFallbackB", "FF9100"),
        ("clrFallbackC", "F0529C"),
        ("clrFallbackD", "265ED4"),
    ]
    _FALLBACK_MARKS = ["*", "square*", "triangle*", "diamond*"]

    def _get_latex_style(ladder_name: str, idx: int) -> Dict[str, str]:  # type: ignore[misc]
        key = ladder_name.lower()
        if key in LATEX_PLOT_STYLES:
            return LATEX_PLOT_STYLES[key]
        fb_color_name, fb_hex = _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)]
        fb_mark = _FALLBACK_MARKS[idx % len(_FALLBACK_MARKS)]
        return {
            "color_def": f"\\definecolor{{{fb_color_name}}}{{HTML}}{{{fb_hex}}}",
            "color_name": fb_color_name,
            "mark": fb_mark,
            "mark_options": f"fill={fb_color_name}" if fb_mark.endswith("*") else "",
            "line_style": "dashed" if idx % 2 else "",
        }


def _emit_color_defs(ladder_names: List[str]) -> str:
    """Emit definecolor commands for all ladders."""
    seen = set()
    lines = []
    for idx, name in enumerate(ladder_names):
        style = _get_latex_style(name, idx)
        if style["color_name"] not in seen:
            lines.append(style["color_def"])
            seen.add(style["color_name"])
    return "\n".join(lines)


def _make_coordinate_table(xs: np.ndarray, ys: np.ndarray) -> str:
    """Generate pgfplots coordinate pairs from arrays."""
    pairs = []
    for x, y in zip(xs, ys):
        if np.isfinite(x) and np.isfinite(y):
            pairs.append(f"({x:.6e},{y:.6e})")
    return " ".join(pairs)


def generate_paper_figure_1(fits: Dict[str, Tuple], log_loss: bool = False) -> str:
    """
    Generate Figure 1: Main Scaling Law Fits (2x2 pgfplots groupplot).

    Supports any number of architectures (all overlaid in each panel).

    Panel A: Loss vs FLOPs (all architectures)
    Panel B: Loss vs Parameters (all architectures)
    Panel C: Loss vs Tokens / data budget (all architectures)
    Panel D: Residuals (all architectures)

    Args:
        fits: Dict mapping ladder name to (fit, N, D, L, F, sizes, bootstrap).
        log_loss: If True, use log scale on the y-axis for loss panels (a-c),
            making the power-law relationships appear linear.
    """
    ladder_names = list(fits.keys())
    loss_label = "Loss (log)" if log_loss else "Loss"
    ymode_str = "    ymode=log,\n" if log_loss else ""
    fig_suffix = "-log" if log_loss else ""
    lines = []
    lines.append(f"% Figure 1{fig_suffix}: Main Scaling Law Fits")
    lines.append("% Generated by fit_chinchilla_scaling_laws.py")
    lines.append(r"% Requires: \usepackage{pgfplots}, \usepgfplotslibrary{groupplots}")
    lines.append("")
    lines.append(_emit_color_defs(ladder_names))
    lines.append("")
    lines.append(r"\begin{figure*}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\begin{tikzpicture}")
    lines.append(r"\begin{groupplot}[")
    lines.append(r"    group style={")
    lines.append(r"        group size=2 by 2,")
    lines.append(r"        horizontal sep=1.8cm,")
    lines.append(r"        vertical sep=1.8cm,")
    lines.append(r"    },")
    lines.append(r"    width=0.48\textwidth,")
    lines.append(r"    height=0.38\textwidth,")
    lines.append(r"    grid=major,")
    lines.append(r"    grid style={gray!30},")
    lines.append(r"    tick label style={font=\small},")
    lines.append(r"    label style={font=\small},")
    legend_name = f"scalinglegend{fig_suffix.replace('-', '')}"
    lines.append(f"    legend to name={legend_name},")
    lines.append(r"    legend style={")
    lines.append(r"        font=\footnotesize,")
    lines.append(r"        cells={anchor=west},")
    lines.append(f"        legend columns={len(ladder_names)},")
    lines.append(r"        draw=none,")
    lines.append(r"        column sep=8pt,")
    lines.append(r"    },")
    lines.append(r"]")

    # Panel A: Loss vs FLOPs (all architectures overlaid)
    lines.append("")
    lines.append("% Panel A: Loss vs FLOPs")
    lines.append(r"\nextgroupplot[")
    lines.append(r"    xmode=log,")
    if ymode_str:
        lines.append(ymode_str.rstrip("\n"))
    lines.append(r"    xlabel={FLOPs (PetaFLOPs)},")
    lines.append(f"    ylabel={{{loss_label}}},")
    lines.append(r"    title={(a) Loss vs Compute},")
    lines.append(r"]")

    for idx, name in enumerate(ladder_names):
        fit, N, D, L, F, sizes, bootstrap = fits[name]
        style = _get_latex_style(name, idx)
        display = _escape_latex(get_display_name(name))

        F_peta = F / 1e15

        # Scatter: data points
        lines.append(
            f"\\addplot[only marks, mark={style['mark']}, "
            f"{style['color_name']}, mark size=1.5pt, "
            f"mark options={{{style['mark_options']}}}, opacity=0.5, forget plot] "
            f"coordinates {{{_make_coordinate_table(F_peta, L)}}};"
        )

        # Fitted curve
        unique_N = np.unique(N)
        mean_dn = np.mean([np.mean(D[N == n]) / n for n in unique_N])
        flop_ratio = np.median(F / (6.0 * N * D))
        N_sweep = np.logspace(np.log10(N.min()), np.log10(N.max()), 150)
        D_sweep = N_sweep * mean_dn
        F_sweep_peta = (6.0 * N_sweep * D_sweep * flop_ratio) / 1e15
        L_sweep = fit.predict_loss(N_sweep, D_sweep)
        line_style = f", {style['line_style']}" if style.get("line_style") else ""
        lines.append(
            f"\\addplot[{style['color_name']}, thick, no markers{line_style}] "
            f"coordinates {{{_make_coordinate_table(F_sweep_peta, L_sweep)}}};"
        )
        lines.append(f"\\addlegendentry{{{display}}}")

    # Panel B: Loss vs Parameters (all overlaid)
    lines.append("")
    lines.append("% Panel B: Loss vs Parameters")
    lines.append(r"\nextgroupplot[")
    lines.append(r"    xmode=log,")
    if log_loss:
        lines.append(r"    ymode=log,")
    lines.append(r"    xlabel={Parameters},")
    lines.append(f"    ylabel={{{loss_label}}},")
    lines.append(r"    title={(b) Loss vs Parameters},")
    lines.append(r"]")

    for idx, name in enumerate(ladder_names):
        fit, N, D, L, F, sizes, bootstrap = fits[name]
        style = _get_latex_style(name, idx)

        unique_N = np.unique(N)
        mean_dn = np.mean([np.mean(D[N == n]) / n for n in unique_N])
        N_range = np.logspace(np.log10(N.min() * 0.8), np.log10(N.max() * 1.2), 150)
        D_curve = N_range * mean_dn
        L_curve = fit.predict_loss(N_range, D_curve)

        # Scatter
        lines.append(
            f"\\addplot[only marks, mark={style['mark']}, "
            f"{style['color_name']}, mark size=1.5pt, "
            f"mark options={{{style['mark_options']}}}, opacity=0.5, forget plot] "
            f"coordinates {{{_make_coordinate_table(N, L)}}};"
        )
        # Fitted curve (forget plot — legend already added in panel A)
        line_style = f", {style['line_style']}" if style.get("line_style") else ""
        lines.append(
            f"\\addplot[{style['color_name']}, thick, no markers{line_style}, forget plot] "
            f"coordinates {{{_make_coordinate_table(N_range, L_curve)}}};"
        )

    # Panel C: Loss vs Tokens (all overlaid, data efficiency comparison)
    lines.append("")
    lines.append("% Panel C: Loss vs Tokens (data efficiency comparison)")
    lines.append(r"\nextgroupplot[")
    lines.append(r"    xmode=log,")
    if log_loss:
        lines.append(r"    ymode=log,")
    lines.append(r"    xlabel={Training Tokens},")
    lines.append(f"    ylabel={{{loss_label}}},")
    lines.append(r"    title={(c) Loss vs Data Budget},")
    lines.append(r"]")

    # Use a shared median N across all ladders for fair comparison
    all_unique_N = np.unique(np.concatenate([np.unique(fits[n][1]) for n in ladder_names]))
    shared_median_N = np.median(all_unique_N)
    all_D = np.concatenate([fits[n][2] for n in ladder_names])
    D_range = np.logspace(np.log10(all_D.min() * 0.8), np.log10(all_D.max() * 1.2), 150)

    for idx, name in enumerate(ladder_names):
        fit, N, D, L, F, sizes, bootstrap = fits[name]
        style = _get_latex_style(name, idx)

        L_curve = fit.predict_loss(shared_median_N, D_range)

        # Scatter
        lines.append(
            f"\\addplot[only marks, mark={style['mark']}, "
            f"{style['color_name']}, mark size=1.5pt, "
            f"mark options={{{style['mark_options']}}}, opacity=0.5, forget plot] "
            f"coordinates {{{_make_coordinate_table(D, L)}}};"
        )
        # Fitted curve (forget plot — legend already added in panel A)
        line_style = f", {style['line_style']}" if style.get("line_style") else ""
        lines.append(
            f"\\addplot[{style['color_name']}, thick, no markers{line_style}, forget plot] "
            f"coordinates {{{_make_coordinate_table(D_range, L_curve)}}};"
        )

    # Panel D: Residuals (all architectures)
    lines.append("")
    lines.append("% Panel D: Fit Residuals")
    lines.append(r"\nextgroupplot[")
    lines.append(r"    xmode=log,")
    lines.append(r"    xlabel={Parameters},")
    lines.append(r"    ylabel={Relative Error (\%)},")
    lines.append(r"    title={(d) Fit Residuals},")
    lines.append(r"]")

    lines.append(
        r"\draw[black, thick] (axis cs:\pgfkeysvalueof{/pgfplots/xmin},0) -- "
        r"(axis cs:\pgfkeysvalueof{/pgfplots/xmax},0);"
    )

    for idx, name in enumerate(ladder_names):
        fit, N, D, L, F, sizes, bootstrap = fits[name]
        style = _get_latex_style(name, idx)

        predicted = fit.predict_loss(N, D)
        residuals_pct = (L - predicted) / L * 100

        # forget plot — legend already added in panel A
        lines.append(
            f"\\addplot[only marks, mark={style['mark']}, "
            f"{style['color_name']}, mark size=1.5pt, "
            f"mark options={{{style['mark_options']}}}, opacity=0.6, forget plot] "
            f"coordinates {{{_make_coordinate_table(N, residuals_pct)}}};"
        )

    lines.append("")
    lines.append(r"\end{groupplot}")
    lines.append(r"\end{tikzpicture}")
    lines.append(r"\vspace{0.2em}\\")
    lines.append(f"\\ref{{{legend_name}}}")

    # Build caption with fitted parameter annotations
    log_note = " Loss axes use log scale." if log_loss else ""
    caption_parts = [
        r"\caption{Scaling law fits $L(N,D) = E + A/N^\alpha + B/D^\beta$ "
        r"for all architectures. "
    ]
    for name in ladder_names:
        fit = fits[name][0]
        p = fit.fitted_params
        r2 = fit.r_squared
        display = _escape_latex(get_display_name(name))
        r2_str = f"{r2:.3f}" if r2 is not None else "?"
        caption_parts.append(
            f"{display}: $\\alpha={p.alpha:.3f}$, $\\beta={p.beta:.3f}$, $R^2={r2_str}$. "
        )
    caption_parts.append(
        r"\textbf{(a)} Loss vs compute. "
        r"\textbf{(b)} Loss vs parameter count. "
        r"\textbf{(c)} Loss vs data budget. "
        r"\textbf{(d)} Fit residuals."
        + log_note
        + "}"
    )
    lines.append("".join(caption_parts))
    lines.append(f"\\label{{fig:scaling-law-fit{fig_suffix}}}")
    lines.append(r"\end{figure*}")
    return "\n".join(lines)


def generate_paper_figure_2(
    fits: Dict[str, Tuple],
    target_loss: Optional[float] = None,
) -> str:
    """
    Generate Figure 2: Efficiency Projections.

    Shows tokens-to-target-loss vs model size for both architectures,
    with shaded savings region.
    """
    ladder_names = list(fits.keys())
    if len(ladder_names) < 2:
        return "% Need at least 2 ladders for efficiency projection"

    # Auto-determine target loss (median of observed losses across all ladders)
    if target_loss is None:
        all_L = np.concatenate([fits[n][3] for n in ladder_names])
        target_loss = float(np.median(all_L))

    lines = []
    lines.append("% Figure 2: Efficiency Projections")
    lines.append("% Generated by fit_chinchilla_scaling_laws.py")
    lines.append(r"% Requires: \usepackage{pgfplots}")
    lines.append("")
    lines.append(_emit_color_defs(ladder_names))
    lines.append("")
    lines.append(r"\begin{figure}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\begin{tikzpicture}")
    lines.append(r"\begin{axis}[")
    lines.append(r"    xmode=log,")
    lines.append(r"    ymode=log,")
    lines.append(r"    width=0.85\textwidth,")
    lines.append(r"    height=0.55\textwidth,")
    lines.append(r"    xlabel={Model Size (Parameters)},")
    lines.append(f"    ylabel={{Tokens to reach loss $={target_loss:.3f}$}},")
    lines.append(r"    grid=major,")
    lines.append(r"    grid style={gray!30},")
    lines.append(r"    legend style={")
    lines.append(r"        font=\footnotesize,")
    lines.append(r"        cells={anchor=west},")
    lines.append(r"        at={(0.5,-0.15)},")
    lines.append(r"        anchor=north,")
    lines.append(f"        legend columns={len(ladder_names)},")
    lines.append(r"        draw=none,")
    lines.append(r"        column sep=8pt,")
    lines.append(r"    },")
    lines.append(r"    tick label style={font=\small},")
    lines.append(r"    label style={font=\small},")
    lines.append(r"]")

    N_range = np.logspace(7.5, 10.85, 200)  # ~30M to 70B
    all_d_curves = {}

    for idx, name in enumerate(ladder_names):
        fit = fits[name][0]
        style = _get_latex_style(name, idx)
        display = _escape_latex(get_display_name(name))

        d_needed = np.array([solve_d_for_target_loss(fit, target_loss, n) for n in N_range])
        valid = np.isfinite(d_needed) & (d_needed > 0)
        all_d_curves[name] = (N_range, d_needed, valid)

        if np.any(valid):
            line_style = f", {style['line_style']}" if style.get("line_style") else ""
            lines.append(
                f"\\addplot[{style['color_name']}, thick, no markers{line_style}] "
                f"coordinates {{{_make_coordinate_table(N_range[valid], d_needed[valid])}}};"
            )
            lines.append(f"\\addlegendentry{{{display}}}")

    # Shaded region between the two curves
    if len(all_d_curves) >= 2:
        names = list(all_d_curves.keys())
        N_a, d_a, valid_a = all_d_curves[names[0]]
        N_b, d_b, valid_b = all_d_curves[names[1]]
        both_valid = valid_a & valid_b
        if np.any(both_valid):
            style_a = _get_latex_style(names[0], 0)
            # Fill between: forward path of curve a, backward path of curve b
            N_fwd = N_a[both_valid]
            d_upper = np.maximum(d_a[both_valid], d_b[both_valid])
            d_lower = np.minimum(d_a[both_valid], d_b[both_valid])
            fill_coords = np.concatenate([N_fwd, N_fwd[::-1]])
            fill_d = np.concatenate([d_upper, d_lower[::-1]])
            lines.append(
                f"\\addplot[fill={style_a['color_name']}!15, draw=none, forget plot] "
                f"coordinates {{{_make_coordinate_table(fill_coords, fill_d)}}} -- cycle;"
            )

    # Vertical line at 7B
    lines.append(
        r"\draw[dashed, gray] (axis cs:7e9,\pgfkeysvalueof{/pgfplots/ymin}) -- "
        r"(axis cs:7e9,\pgfkeysvalueof{/pgfplots/ymax});"
    )
    lines.append(
        r"\node[font=\footnotesize, anchor=south] at (axis cs:7e9,"
        r"\pgfkeysvalueof{/pgfplots/ymax}) {7B};"
    )

    lines.append(r"\end{axis}")
    lines.append(r"\end{tikzpicture}")
    lines.append(
        r"\caption{Projected data requirements across scales "
        f"(target loss $= {target_loss:.3f}$, "
        r"selected as the median of all observed training losses). "
        r"Extrapolating from scaling ladder experiments, the hybrid architecture "
        r"requires substantially fewer training tokens to reach equivalent loss. "
        r"Shaded region shows the data savings.}"
    )
    lines.append(r"\label{fig:scaling-efficiency-projections}")
    lines.append(r"\end{figure}")
    return "\n".join(lines)


def generate_paper_figure_3(
    domain_fits: Dict[str, Dict[str, Tuple]],
) -> str:
    """
    Generate Figure 3: Per-Domain Scaling Laws (Math, Code, QA).

    Args:
        domain_fits: {ladder_name: {domain: (fit, N, D, L, F, sizes)}}
    """
    # Determine which domains are present across all ladders
    all_domains = set()
    for ladder_domains in domain_fits.values():
        all_domains.update(ladder_domains.keys())
    domains = [d for d in ["Math_BPB", "Code_BPB", "QA_BPB"] if d in all_domains]

    if not domains:
        return "% No domain-specific scaling law data available"

    ladder_names = list(domain_fits.keys())
    n_panels = len(domains)

    lines = []
    lines.append("% Figure 3: Per-Domain Scaling Laws")
    lines.append("% Generated by fit_chinchilla_scaling_laws.py")
    lines.append(r"% Requires: \usepackage{pgfplots}, \usepgfplotslibrary{groupplots}")
    lines.append("")
    lines.append(_emit_color_defs(ladder_names))
    lines.append("")
    lines.append(r"\begin{figure}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\begin{tikzpicture}")
    lines.append(r"\begin{groupplot}[")
    lines.append(f"    group style={{group size={n_panels} by 1, horizontal sep=1.5cm}},")
    lines.append(r"    width=0.33\textwidth,")
    lines.append(r"    height=0.28\textwidth,")
    lines.append(r"    xmode=log,")
    lines.append(r"    grid=major,")
    lines.append(r"    grid style={gray!30},")
    lines.append(r"    tick label style={font=\tiny},")
    lines.append(r"    label style={font=\small},")
    lines.append(r"    legend to name=domainlegend,")
    lines.append(r"    legend style={")
    lines.append(r"        font=\footnotesize,")
    lines.append(r"        cells={anchor=west},")
    lines.append(f"        legend columns={len(ladder_names)},")
    lines.append(r"        draw=none,")
    lines.append(r"        column sep=8pt,")
    lines.append(r"    },")
    lines.append(r"]")

    for domain_idx, domain in enumerate(domains):
        domain_clean = domain.replace("_BPB", "")
        lines.append("")
        lines.append(f"% Panel: {domain_clean}")
        lines.append(r"\nextgroupplot[")
        lines.append(r"    ymode=log,")
        lines.append(r"    xlabel={Training Tokens},")
        if domain_idx == 0:
            lines.append(r"    ylabel={BPB},")
        lines.append(f"    title={{{_escape_latex(domain_clean)}}},")
        lines.append(r"]")

        for ladder_idx, name in enumerate(ladder_names):
            style = _get_latex_style(name, ladder_idx)
            display = _escape_latex(get_display_name(name))

            if domain not in domain_fits.get(name, {}):
                continue

            fit, N, D, L, F, sizes = domain_fits[name][domain]

            # Scatter (always forget plot)
            lines.append(
                f"\\addplot[only marks, mark={style['mark']}, "
                f"{style['color_name']}, mark size=1pt, "
                f"mark options={{{style['mark_options']}}}, opacity=0.5, forget plot] "
                f"coordinates {{{_make_coordinate_table(D, L)}}};"
            )

            # Fitted curve — add legend entry only in the first panel
            D_range = np.logspace(np.log10(D.min()), np.log10(D.max()), 100)
            median_N = np.median(np.unique(N))
            L_curve = fit.predict_loss(median_N, D_range)
            line_style = f", {style['line_style']}" if style.get("line_style") else ""
            if domain_idx == 0:
                lines.append(
                    f"\\addplot[{style['color_name']}, thick, no markers{line_style}] "
                    f"coordinates {{{_make_coordinate_table(D_range, L_curve)}}};"
                )
                lines.append(f"\\addlegendentry{{{display}}}")
            else:
                lines.append(
                    f"\\addplot[{style['color_name']}, thick, no markers{line_style}, forget plot] "
                    f"coordinates {{{_make_coordinate_table(D_range, L_curve)}}};"
                )

        # Annotate domain-specific B coefficients for all ladders
        b_parts = []
        for lname in ladder_names:
            if domain in domain_fits.get(lname, {}):
                b_val = domain_fits[lname][domain][0].fitted_params.B
                short = _escape_latex(get_display_name(lname))
                b_parts.append(f"$B_{{{short}}}={b_val:.2e}$")
        if len(b_parts) >= 2:
            lines.append(
                f"\\node[font=\\tiny, anchor=south east, align=right] "
                f"at (rel axis cs:0.95,0.05) {{{', '.join(b_parts)}}};"
            )

    lines.append("")
    lines.append(r"\end{groupplot}")
    lines.append(r"\end{tikzpicture}")
    lines.append(r"\vspace{0.2em}\\")
    lines.append(r"\ref{domainlegend}")
    lines.append(
        r"\caption{Domain-specific scaling laws show the hybrid advantage is consistent "
        r"across Math, Code, and QA domains.}"
    )
    lines.append(r"\label{fig:scaling-law-domains}")
    lines.append(r"\end{figure}")
    return "\n".join(lines)


def generate_paper_figure_4(fits: Dict[str, Tuple]) -> str:
    """
    Generate Figure 4: Scaling Law Residuals (2-panel diagnostic).

    Panel A: Residual scatter (predicted loss on x-axis, actual-predicted on y-axis)
    Panel B: Mean absolute relative error (%) per model size — shows fit quality is
             consistent across scales. This is more robust than per-size R² which is
             ill-defined when each size has only a few D/N checkpoints.
    """
    ladder_names = list(fits.keys())

    lines = []
    lines.append("% Figure 4: Scaling Law Residuals")
    lines.append("% Generated by fit_chinchilla_scaling_laws.py")
    lines.append(r"% Requires: \usepackage{pgfplots}, \usepgfplotslibrary{groupplots}")
    lines.append("")
    lines.append(_emit_color_defs(ladder_names))
    lines.append("")
    lines.append(r"\begin{figure}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\begin{tikzpicture}")
    lines.append(r"\begin{groupplot}[")
    lines.append(r"    group style={group size=2 by 1, horizontal sep=2cm},")
    lines.append(r"    width=0.45\textwidth,")
    lines.append(r"    height=0.35\textwidth,")
    lines.append(r"    grid=major,")
    lines.append(r"    grid style={gray!30},")
    lines.append(r"    tick label style={font=\small},")
    lines.append(r"    label style={font=\small},")
    lines.append(r"    legend to name=residuallegend,")
    lines.append(r"    legend style={")
    lines.append(r"        font=\footnotesize,")
    lines.append(r"        cells={anchor=west},")
    lines.append(f"        legend columns={len(ladder_names)},")
    lines.append(r"        draw=none,")
    lines.append(r"        column sep=8pt,")
    lines.append(r"    },")
    lines.append(r"]")

    # Panel A: Residual scatter
    lines.append("")
    lines.append("% Panel A: Residual scatter")
    lines.append(r"\nextgroupplot[")
    lines.append(r"    xlabel={Predicted Loss},")
    lines.append(r"    ylabel={Relative Error (\%)},")
    lines.append(r"    title={(a) Fit Residuals},")
    lines.append(r"]")

    # Zero line
    lines.append(
        r"\draw[black, thick] (axis cs:\pgfkeysvalueof{/pgfplots/xmin},0) -- "
        r"(axis cs:\pgfkeysvalueof{/pgfplots/xmax},0);"
    )

    for idx, name in enumerate(ladder_names):
        fit, N, D, L, F, sizes, bootstrap = fits[name]
        style = _get_latex_style(name, idx)
        display = _escape_latex(get_display_name(name))

        predicted = fit.predict_loss(N, D)
        residuals_pct = (L - predicted) / L * 100

        lines.append(
            f"\\addplot[only marks, mark={style['mark']}, "
            f"{style['color_name']}, mark size=1.5pt, "
            f"mark options={{{style['mark_options']}}}, opacity=0.6] "
            f"coordinates {{{_make_coordinate_table(predicted, residuals_pct)}}};"
        )
        lines.append(f"\\addlegendentry{{{display}}}")

    # Panel B: Mean absolute relative error (%) per model size
    lines.append("")
    lines.append("% Panel B: Mean absolute relative error by model size")
    lines.append(r"\nextgroupplot[")
    lines.append(r"    xlabel={Parameters},")
    lines.append(r"    ylabel={Mean $|$Relative Error$|$ (\%)},")
    lines.append(r"    title={(b) Error by Scale},")
    lines.append(r"    ymin=0,")
    lines.append(r"    xmode=log,")
    lines.append(r"]")

    for idx, name in enumerate(ladder_names):
        fit, N, D, L, F, sizes, bootstrap = fits[name]
        style = _get_latex_style(name, idx)

        predicted = fit.predict_loss(N, D)
        abs_rel_err = np.abs((L - predicted) / L) * 100

        # Group by unique N and compute mean error per group
        unique_N = np.sort(np.unique(N))
        n_vals = []
        err_vals = []
        for n_val in unique_N:
            mask = N == n_val
            if np.sum(mask) < 1:
                continue
            n_vals.append(n_val)
            err_vals.append(float(np.mean(abs_rel_err[mask])))

        if n_vals:
            n_arr = np.array(n_vals)
            err_arr = np.array(err_vals)
            line_style = f", {style['line_style']}" if style.get("line_style") else ""
            # forget plot — legend already added in panel A
            lines.append(
                f"\\addplot[{style['color_name']}, thick, mark={style['mark']}, "
                f"mark size=2pt, mark options={{{style['mark_options']}}}{line_style}, forget plot] "
                f"coordinates {{{_make_coordinate_table(n_arr, err_arr)}}};"
            )

    # Reference line at 1%
    lines.append(
        r"\draw[dashed, gray] (axis cs:\pgfkeysvalueof{/pgfplots/xmin},1) -- "
        r"(axis cs:\pgfkeysvalueof{/pgfplots/xmax},1);"
    )
    lines.append(
        r"\node[font=\tiny, gray, anchor=west] at "
        r"(axis cs:\pgfkeysvalueof{/pgfplots/xmin},1.05) {1\%};"
    )

    lines.append("")
    lines.append(r"\end{groupplot}")
    lines.append(r"\end{tikzpicture}")
    lines.append(r"\vspace{0.2em}\\")
    lines.append(r"\ref{residuallegend}")

    # Caption with overall R² values
    r2_parts = []
    for name in ladder_names:
        r2 = fits[name][0].r_squared
        display = _escape_latex(get_display_name(name))
        r2_str = f"{r2:.4f}" if r2 is not None else "?"
        r2_parts.append(f"{display} $R^2={r2_str}$")
    lines.append(
        r"\caption{Scaling law fit diagnostics. "
        r"\textbf{(a)} Residuals are small and unbiased for all architectures. "
        r"\textbf{(b)} Mean absolute relative error stays below 1\% across all model sizes, "
        r"confirming reliable fits. "
        f"Overall: {'; '.join(r2_parts)}.}}"
    )
    lines.append(r"\label{fig:scaling-law-residuals}")
    lines.append(r"\end{figure}")
    return "\n".join(lines)


def plot_fits_2d(
    fits: Dict[str, Tuple],
    output_path: Optional[Path] = None,
    show: bool = True,
    log_loss: bool = False,
):
    """Generate 2D matplotlib plots for scaling law fits.

    Args:
        fits: Dict mapping ladder name to (fit, N, D, L, F, sizes, bootstrap).
        output_path: Directory to save the plot to.
        show: Whether to display the plot interactively.
        log_loss: If True, use log scale on the y-axis for loss panels,
            making the power-law relationships appear linear.
    """
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

    # Human-readable display names
    display_names = {name: get_display_name(name) for name in fits}

    # Color palette - distinct colors for each ladder
    color_palette = ["#2ecc71", "#e74c3c", "#3498db", "#9b59b6", "#f39c12", "#1abc9c", "#e67e22"]
    colors = {name: color_palette[i % len(color_palette)] for i, name in enumerate(fits.keys())}

    # Create a cleaner 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    title_suffix = " (log-loss)" if log_loss else ""
    fig.suptitle(f"Chinchilla Scaling Law Analysis{title_suffix}", fontsize=16, fontweight="bold", y=0.98)

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
        ax.plot(
            N_range,
            loss_curve,
            color=color,
            linewidth=2.5,
            alpha=0.8,
            label=f"{display_names[name]} (fit)",
        )

        # Plot actual data points (scatter only, no lines)
        ax.scatter(N, L, color=color, s=40, alpha=0.6, edgecolors="white", linewidth=0.5, zorder=5)

    ax.set_xscale("log")
    if log_loss:
        ax.set_yscale("log")
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
        base_color = colors[name]

        # Group by unique N values and plot loss vs D
        unique_N = np.sort(np.unique(N))
        n_sizes = len(unique_N)
        for size_idx, n_val in enumerate(unique_N):
            # Gradient: lightest for smallest model, full color for largest
            frac = 0.3 + 0.7 * size_idx / max(n_sizes - 1, 1)
            shade = shade_color(base_color, frac)

            mask = N == n_val
            D_subset = D[mask]
            L_subset = L[mask]

            # Sort by D for cleaner visualization
            sort_idx = np.argsort(D_subset)
            ax.scatter(
                D_subset[sort_idx],
                L_subset[sort_idx],
                color=shade,
                s=30,
                alpha=0.7,
                edgecolors="white",
                linewidth=0.3,
            )

            # Fitted curve spanning the data range
            D_curve = np.logspace(np.log10(D_subset.min()), np.log10(D_subset.max()), 100)
            L_curve = fit.predict_loss(n_val, D_curve)
            ax.plot(D_curve, L_curve, color=shade, linewidth=1.5, alpha=0.8)

    # Primary legend: one entry per ladder
    ladder_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors[name],
            markersize=10,
            label=display_names[name],
        )
        for name in fits.keys()
    ]
    first_legend = ax.legend(handles=ladder_handles, loc="upper right", fontsize=9, framealpha=0.9)
    ax.add_artist(first_legend)

    # Secondary legend: gradient key using canonical size buckets (60M, 100M, etc.)
    canonical_sizes = ["60M", "100M", "190M", "370M", "600M", "760M"]
    # Determine which canonical sizes appear in the data
    present_sizes = []
    for _, (_, N_data, *_) in fits.items():
        for n_val in np.unique(N_data):
            # Snap to nearest canonical bucket
            n_m = n_val / 1e6
            if n_m < 80:
                label = "60M"
            elif n_m < 150:
                label = "100M"
            elif n_m < 280:
                label = "190M"
            elif n_m < 480:
                label = "370M"
            elif n_m < 680:
                label = "600M"
            else:
                label = "760M"
            if label not in present_sizes:
                present_sizes.append(label)
    # Sort by canonical order
    present_sizes = [s for s in canonical_sizes if s in present_sizes]
    n_all = len(present_sizes)
    size_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=shade_color("#555555", 0.3 + 0.7 * i / max(n_all - 1, 1)),
            markersize=7,
            label=present_sizes[i],
        )
        for i in range(n_all)
    ]
    ax.legend(
        handles=size_handles,
        loc="lower left",
        fontsize=8,
        framealpha=0.9,
        title="Model size",
        title_fontsize=8,
    )

    ax.set_xscale("log")
    if log_loss:
        ax.set_yscale("log")
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
            label=f"{display_names[name]} (α={fit.fitted_params.alpha:.2f}, β={fit.fitted_params.beta:.2f})",
            color=color,
        )

    ax.set_xscale("log")
    if log_loss:
        ax.set_yscale("log")
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
        ax.scatter(
            N,
            residuals_pct,
            s=40,
            label=display_names[name],
            color=color,
            alpha=0.6,
            edgecolors="white",
            linewidth=0.5,
        )

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
    ax.fill_between(ax.get_xlim(), -1, 1, alpha=0.1, color="green", label="±1% zone")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        suffix = "_log" if log_loss else ""
        save_name = f"chinchilla_scaling_fits_2d{suffix}.png"
        fig.savefig(
            output_path / save_name,
            dpi=200,
            bbox_inches="tight",
            facecolor="white",
        )
        print(f"\nSaved 2D plot to: {output_path / save_name}")

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
            subtitle=get_display_name(name),
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
            (get_display_name(name_a), rollout_a),
            (get_display_name(name_b), rollout_b),
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
                subtitle=get_display_name(name),
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
        from matplotlib.lines import Line2D
    except ImportError:
        print("Warning: matplotlib not installed, skipping iso-FLOP plots")
        return

    if not fits:
        return

    plt.style.use("seaborn-v0_8-whitegrid")

    display_names = {name: get_display_name(name) for name in fits}
    color_palette = [
        "#2ecc71",
        "#e74c3c",
        "#3498db",
        "#9b59b6",
        "#f39c12",
        "#1abc9c",
        "#e67e22",
        "#34495e",
        "#e91e63",
        "#00bcd4",
    ]
    colors = {name: color_palette[i % len(color_palette)] for i, name in enumerate(fits.keys())}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # FLOPs = 6 * N * D (approximate)
    flop_budgets = [1e18, 1e19, 1e20, 1e21, 1e22]  # 1e18 to 1e22 FLOPs
    n_budgets = len(flop_budgets)
    # Gradient fractions: lightest (0.25) for smallest budget, full color (1.0) for largest
    gradient_fractions = [0.25 + 0.75 * i / (n_budgets - 1) for i in range(n_budgets)]

    # Left: Loss vs N for different FLOP budgets
    ax = axes[0]

    for name, (fit, N, D, L, F, sizes, bootstrap) in fits.items():
        base_color = colors[name]
        for flop_idx, flops in enumerate(flop_budgets):
            N_range = np.logspace(7, 11, 100)
            D_range = flops / (6 * N_range)
            # Only plot where D is reasonable (> 1B tokens)
            valid = D_range > 1e9
            if not np.any(valid):
                continue

            loss_pred = fit.predict_loss(N_range[valid], D_range[valid])
            curve_color = shade_color(base_color, gradient_fractions[flop_idx])
            lw = 1.0 + 1.5 * (flop_idx / (n_budgets - 1))  # thinner for small budgets
            ax.plot(
                N_range[valid],
                loss_pred,
                color=curve_color,
                linewidth=lw,
            )

    # Primary legend: one entry per ladder
    ladder_handles = [
        Line2D([0], [0], color=colors[name], linewidth=2, label=display_names[name])
        for name in fits.keys()
    ]
    # Secondary legend: gradient key showing FLOP budgets (use gray gradient)
    budget_handles = [
        Line2D(
            [0],
            [0],
            color=shade_color("#555555", gradient_fractions[i]),
            linewidth=1.0 + 1.5 * (i / (n_budgets - 1)),
            label=f"{flop_budgets[i]:.0e} FLOPs",
        )
        for i in range(n_budgets)
    ]
    first_legend = ax.legend(handles=ladder_handles, loc="upper right", fontsize=9, framealpha=0.9)
    ax.add_artist(first_legend)
    ax.legend(
        handles=budget_handles,
        loc="lower left",
        fontsize=8,
        framealpha=0.9,
        title="Compute budget",
        title_fontsize=8,
    )

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (N)", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("Iso-FLOP Curves", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Right: Optimal N vs FLOP budget
    ax = axes[1]
    flop_range = np.logspace(17, 23, 100)

    for name, (fit, N, D, L, F, sizes, bootstrap) in fits.items():
        color = colors[name]
        params = fit.fitted_params
        # Compute-optimal: N_opt ∝ C^a_opt, where C = 6*N*D
        # For a given FLOP budget F = 6*N*D, optimal N scales as F^a_opt
        # This is approximate; exact solution requires solving the optimality conditions
        N_opt = (flop_range / 6) ** params.a_opt * (1 / 20) ** (params.a_opt * params.b_opt)

        ax.plot(
            flop_range,
            N_opt,
            linewidth=2,
            label=f"{display_names[name]} (a={params.a_opt:.3f})",
            color=color,
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
    log_loss: bool = False,
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
    # Human-readable display names
    display_names = {name: get_display_name(name) for name in fits}

    color_palette = [
        "#2ecc71",
        "#e74c3c",
        "#3498db",
        "#9b59b6",
        "#f39c12",
        "#1abc9c",
        "#e67e22",
        "#34495e",
        "#e91e63",
        "#00bcd4",
    ]
    colors = {name: color_palette[i % len(color_palette)] for i, name in enumerate(fits.keys())}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    title_suffix = " (log-loss)" if log_loss else ""
    fig.suptitle(f"Loss vs Training FLOPs{title_suffix}", fontsize=16, fontweight="bold", y=0.98)

    # ============================================================
    # Left: Loss vs FLOPs (scatter + fitted curve per ladder)
    # ============================================================
    ax = axes[0]

    for name, (fit, N, D, L, F, sizes, bootstrap) in fits.items():
        color = colors[name]
        # F is in raw FLOPs, convert to PetaFLOPs for plotting
        F_peta = F / 1e15
        ax.scatter(F_peta, L, color=color, s=40, alpha=0.5, edgecolors="white", linewidth=0.5)

        # Fitted curve: sweep FLOPs using the ladder's average D/N ratio
        # Correct for measured vs 6*N*D FLOPs difference
        unique_N = np.unique(N)
        mean_dn_ratios = [np.mean(D[N == n_val]) / n_val for n_val in unique_N]
        overall_dn_ratio = np.mean(mean_dn_ratios)
        flop_ratio = np.median(F / (6.0 * N * D))
        N_sweep = np.logspace(np.log10(N.min()), np.log10(N.max()), 200)
        D_sweep = N_sweep * overall_dn_ratio
        F_sweep_peta = (6.0 * N_sweep * D_sweep * flop_ratio) / 1e15
        L_sweep = fit.predict_loss(N_sweep, D_sweep)
        ax.plot(
            F_sweep_peta,
            L_sweep,
            color=color,
            linewidth=2.5,
            alpha=0.8,
            label=f"{display_names[name]} (fit)",
        )

    ax.set_xscale("log")
    if log_loss:
        ax.set_yscale("log")
    ax.set_xlabel("Training FLOPs (PetaFLOPs)", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("All Checkpoints", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3)

    # ============================================================
    # Right: Loss vs FLOPs per model size (scatter + fitted curves)
    # ============================================================
    ax = axes[1]

    for name, (fit, N, D, L, F, sizes, bootstrap) in fits.items():
        base_color = colors[name]
        F_peta = F / 1e15

        # Group by unique N values: scatter data + fitted curve per size
        unique_N = np.sort(np.unique(N))
        n_sizes = len(unique_N)
        for size_idx, n_val in enumerate(unique_N):
            frac = 0.3 + 0.7 * size_idx / max(n_sizes - 1, 1)
            cur_shade = shade_color(base_color, frac)

            mask = N == n_val
            F_subset = F_peta[mask]
            L_subset = L[mask]
            D_subset = D[mask]

            ax.scatter(
                F_subset,
                L_subset,
                color=cur_shade,
                s=25,
                alpha=0.7,
                edgecolors="white",
                linewidth=0.3,
            )

            # Fitted curve: sweep D over the data range, use actual F/D ratio to
            # compute FLOPs on the x-axis so the curve aligns with the data points
            D_range = np.logspace(np.log10(D_subset.min()), np.log10(D_subset.max()), 100)
            # Use the median measured-F / (6*N*D) ratio to correct for architecture differences
            f_measured = F[mask]
            f_approx = 6.0 * n_val * D_subset
            flop_ratio = np.median(f_measured / f_approx)
            F_curve_peta = (6.0 * n_val * D_range * flop_ratio) / 1e15
            L_curve = fit.predict_loss(n_val, D_range)
            ax.plot(F_curve_peta, L_curve, color=cur_shade, linewidth=1.5, alpha=0.8)

    # Create legend manually (one entry per ladder)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=colors[name],
            markersize=8,
            label=display_names[name],
            linewidth=1.5,
        )
        for name in fits.keys()
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.9)

    ax.set_xscale("log")
    if log_loss:
        ax.set_yscale("log")
    ax.set_xlabel("Training FLOPs (PetaFLOPs)", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Per Model Size", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        suffix = "_log" if log_loss else ""
        save_path = output_path / f"loss_vs_flops{suffix}.png"
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
        print(f"\n{get_display_name(name)}:")
        print(f"  Predicted loss: {pred:.4f}")

        if bootstrap is not None:
            loss_dist = bootstrap.predict_loss_distribution(target_n, target_d)
            ci_low, ci_high = np.percentile(loss_dist, [2.5, 97.5])
            print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")


def compare_specs_for_ladder(ladder_name: str, ladder_dir: Path) -> None:
    """
    Print a diagnostic comparison of computed vs. logged parameter counts and FLOPs.

    For each model size in the ladder, compares:
    - Computed non-embedding params (from architecture specs) vs. logged num_params from pickle
    - Computed forward-pass FLOPs/token vs. logged training FLOPs (using 3x factor for fwd+bwd)
    """
    pkl_files = sorted(ladder_dir.glob("metrics_*.pkl"))
    if not pkl_files:
        print(f"  No metrics_*.pkl files found in {ladder_dir}")
        return

    display = get_display_name(ladder_name)
    print(f"\n{'='*90}")
    print(f"  Spec Comparison: {display} ({ladder_name})")
    print(f"{'='*90}")

    # Header
    print(
        f"  {'Size':<6}  {'Computed N':>12}  {'Logged N':>12}  {'Diff':>10}  "
        f"{'Comp FLOP/tok':>14}  {'Log FLOP/tok':>14}  {'FLOP Diff':>10}"
    )
    print(f"  {'-'*84}")

    flops_col = "throughput/total petaflops"

    for pkl_path in pkl_files:
        size = pkl_path.stem.replace("metrics_", "")
        try:
            df = pd.read_pickle(pkl_path)
        except Exception:
            continue
        if df.empty:
            continue

        logged_params = df["num_params"].iloc[0]
        if logged_params is None or pd.isna(logged_params):
            continue
        logged_params = float(logged_params)

        computed = compute_specs_for_size(ladder_name, size)

        # Param comparison
        if computed is not None:
            comp_params = computed["non_embed_params"]
            param_diff_pct = (comp_params - logged_params) / logged_params * 100
            comp_params_str = fmt_number(comp_params)
            param_diff_str = f"{param_diff_pct:+.2f}%"
        else:
            comp_params_str = "N/A"
            param_diff_str = "---"

        logged_params_str = fmt_number(logged_params)

        # FLOP comparison: use the final row (most tokens) for best estimate
        comp_flops_str = "N/A"
        log_flops_str = "N/A"
        flop_diff_str = "---"

        if computed is not None:
            comp_flops_per_tok = computed["total_flops_per_token"]
            comp_flops_str = fmt_number(comp_flops_per_tok)

            # Try to derive logged FLOPs per token from total training FLOPs
            final_row = df[df["step"] == df["step"].max()].iloc[0]
            logged_pflops = final_row.get(flops_col)
            tokens = final_row.get("tokens")
            if (
                logged_pflops is not None
                and not pd.isna(logged_pflops)
                and tokens is not None
                and not pd.isna(tokens)
                and float(tokens) > 0
            ):
                # logged = total training FLOPs; estimate fwd-only per-token: total / (tokens * 3)
                logged_total_flops = float(logged_pflops) * 1e15
                logged_flops_per_tok = logged_total_flops / (float(tokens) * 3.0)
                log_flops_str = fmt_number(logged_flops_per_tok)
                flop_diff_pct = (
                    (comp_flops_per_tok - logged_flops_per_tok) / logged_flops_per_tok * 100
                )
                flop_diff_str = f"{flop_diff_pct:+.2f}%"

        print(
            f"  {size:<6}  {comp_params_str:>12}  {logged_params_str:>12}  {param_diff_str:>10}  "
            f"{comp_flops_str:>14}  {log_flops_str:>14}  {flop_diff_str:>10}"
        )

    print()


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
    parser.add_argument(
        "--include-pre-decay",
        action="store_true",
        help="Include pre-decay checkpoints in addition to post-decay. "
        "By default, only post-decay checkpoints (D/N = 10, 20, 40, 80, 160, ...) are used.",
    )
    parser.add_argument(
        "--simple-flops",
        action="store_true",
        help="Use simple 6*N*D approximation for FLOPs instead of logged values. "
        "This can be useful when comparing models with different architectures.",
    )
    parser.add_argument(
        "--compare-specs",
        action="store_true",
        help="Print a diagnostic comparison of computed vs. logged parameter counts "
        "and FLOPs for each ladder before fitting.",
    )
    parser.add_argument(
        "--num-slices",
        type=int,
        default=4,
        help="Number of grid slices per dimension for optimizer initialization. "
        "Total initializations = num_slices^5 (default: 4 → 1024 inits). "
        "Increase to 5-8 for more thorough search at higher compute cost.",
    )
    parser.add_argument(
        "--paper-figures",
        action="store_true",
        help="Generate all paper figures (pgfplots/TikZ .tex files) for section 4.1",
    )
    parser.add_argument(
        "--paper-tables",
        action="store_true",
        help="Generate all paper tables (LaTeX .tex files) for section 4.1",
    )
    parser.add_argument(
        "--domain-fit",
        action="store_true",
        help="Also fit per-domain scaling laws (Math_BPB, Code_BPB, QA_BPB)",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="Which domains to fit (default: Math_BPB Code_BPB QA_BPB)",
    )
    parser.add_argument(
        "--target-losses",
        nargs="+",
        type=float,
        default=None,
        help="Target losses for efficiency table (default: auto-determined from data)",
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

    # Compare computed vs. logged specs if requested
    if args.compare_specs:
        for name, ladder_dir in ladders.items():
            compare_specs_for_ladder(name, ladder_dir)

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
                simple_flops=args.simple_flops,
                num_slices=args.num_slices,
                post_decay_only=not args.include_pre_decay,
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
                    num_bootstraps=max(args.bootstrap, 5),  # At least 50 for rollout
                    weight_by_compute=not args.no_weight,
                    overestimate_penalty=args.overestimate_penalty,
                    verbose=not args.quiet,
                    num_slices=args.num_slices,
                )
                rollouts[name] = rollout
                print_rollout_evaluation(name, rollout)

        except Exception as e:
            print(f"\nError fitting {get_display_name(name)}: {e}")
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
        plot_fits_2d(fits, args.output, show=(args.output is None), log_loss=True)

    # Generate iso-FLOP plots
    if args.plot_isoflop:
        plot_isoflop_curves(fits, args.output, show=(args.output is None))

    # Generate loss vs FLOPs plots
    if args.plot_flops:
        plot_loss_vs_flops(fits, args.output, show=(args.output is None))
        plot_loss_vs_flops(fits, args.output, show=(args.output is None), log_loss=True)

    # Generate 3D plots
    if args.plot_3d:
        if not rollouts:
            print("\nWarning: --plot-3d requires --rollout, skipping 3D plots")
        else:
            plot_fits_3d(rollouts, args.output, show=(args.output is None))

    # =================================================================
    # Paper figures and tables (pgfplots/TikZ + LaTeX)
    # =================================================================

    # Domain-specific scaling law fitting
    domain_fits_all: Dict[str, Dict[str, Tuple]] = {}
    if args.domain_fit or args.paper_figures:
        for name, ladder_dir in ladders.items():
            try:
                domain_results = fit_domain_ladders(
                    name=name,
                    ladder_dir=ladder_dir,
                    domains=args.domains,
                    weight_by_compute=not args.no_weight,
                    overestimate_penalty=args.overestimate_penalty,
                    num_slices=args.num_slices,
                    verbose=not args.quiet,
                    post_decay_only=not args.include_pre_decay,
                    simple_flops=args.simple_flops,
                )
                if domain_results:
                    domain_fits_all[name] = domain_results
            except Exception as e:
                print(f"\nError fitting domains for {get_display_name(name)}: {e}")
                import traceback

                traceback.print_exc()

    # Generate paper figures
    if args.paper_figures:
        output_dir = args.output or Path(".")
        output_dir.mkdir(parents=True, exist_ok=True)
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Figure 1: Main Scaling Law Fits (linear and log-loss versions)
        fig1 = generate_paper_figure_1(fits)
        fig1_path = figures_dir / "scaling-law-fit.tex"
        with open(fig1_path, "w") as f:
            f.write(fig1)
        print(f"\nSaved Figure 1 to: {fig1_path}")

        fig1_log = generate_paper_figure_1(fits, log_loss=True)
        fig1_log_path = figures_dir / "scaling-law-fit-log.tex"
        with open(fig1_log_path, "w") as f:
            f.write(fig1_log)
        print(f"Saved Figure 1 (log-loss) to: {fig1_log_path}")

        # Figure 2: Efficiency Projections
        fig2 = generate_paper_figure_2(fits)
        fig2_path = figures_dir / "scaling-efficiency-projections.tex"
        with open(fig2_path, "w") as f:
            f.write(fig2)
        print(f"Saved Figure 2 to: {fig2_path}")

        # Figure 3: Per-Domain Scaling Laws
        if domain_fits_all:
            fig3 = generate_paper_figure_3(domain_fits_all)
            fig3_path = figures_dir / "scaling-law-domains.tex"
            with open(fig3_path, "w") as f:
                f.write(fig3)
            print(f"Saved Figure 3 to: {fig3_path}")
        else:
            print("Skipping Figure 3: no domain-specific fits available")

        # Figure 4: Scaling Law Residuals
        fig4 = generate_paper_figure_4(fits)
        fig4_path = figures_dir / "scaling-law-residuals.tex"
        with open(fig4_path, "w") as f:
            f.write(fig4)
        print(f"Saved Figure 4 to: {fig4_path}")

    # Generate paper tables
    if args.paper_tables:
        output_dir = args.output or Path(".")
        output_dir.mkdir(parents=True, exist_ok=True)
        tables_dir = output_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)

        # Table 1: Fitted Scaling Law Parameters
        table1 = generate_scaling_params_latex_table(fits)
        table1_path = tables_dir / "scaling-law-parameters.tex"
        with open(table1_path, "w") as f:
            f.write(table1)
        print(f"\nSaved Table 1 to: {table1_path}")

        # Table 2: Model Configurations
        table2 = generate_model_config_latex_table(list(ladders.keys()))
        table2_path = tables_dir / "scaling-ladder-configs.tex"
        with open(table2_path, "w") as f:
            f.write(table2)
        print(f"Saved Table 2 to: {table2_path}")

        # Table 3: Efficiency Gains
        table3 = generate_efficiency_latex_table(fits, target_losses=args.target_losses)
        table3_path = tables_dir / "scaling-efficiency-gains.tex"
        with open(table3_path, "w") as f:
            f.write(table3)
        print(f"Saved Table 3 to: {table3_path}")

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
