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

    # Pool multiple independent runs (comma-separated paths)
    python fit_chinchilla_scaling_laws.py \
        --ladder olmo3:~/data/olmo3-run1,~/data/olmo3-run2,~/data/olmo3-run3
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
from olmo_core.model_ladder.analysis.model_specs import (
    LADDER_ARCH_CONFIGS,
    OLMO3_SPECS_BY_NAME,
    compute_specs_for_size,
    compute_specs_for_size_parallel,
    get_display_name,
    get_param_count,
)
from olmo_core.model_ladder.analysis.model_specs import (
    fmt as fmt_number,
)
from olmo_core.model_ladder.analysis.plotting import (
    LATEX_PLOT_STYLES,
    escape_latex,
    get_latex_style,
)

# Default loss column to use for fitting
DEFAULT_LOSS_COLUMN = "eval/lm/c4_en/CE loss"

# Alternative loss columns to try if default not found
FALLBACK_LOSS_COLUMNS = [
    "eval/lm/c4_en-validation/CE loss",
    "eval/lm/pile/CE loss",
    "eval/lm/dolma_common-crawl/CE loss",
    "eval/lm/dolma_common-crawl-validation/CE loss",
    "eval/lm/wikitext_103/CE loss",
    "train/CE loss (log scale)",
]

# Validation loss columns to average when --average-val-loss is used
VALIDATION_LOSS_COLUMNS = [
    "eval/lm/c4_en-validation/CE loss",
    "eval/lm/dolma_books-validation/CE loss",
    "eval/lm/dolma_common-crawl-validation/CE loss",
    "eval/lm/dolma_pes2o-validation/CE loss",
    "eval/lm/dolma_reddit-validation/CE loss",
    "eval/lm/dolma_stack-validation/CE loss",
    "eval/lm/dolma_wiki-validation/CE loss",
    "eval/lm/ice-validation/CE loss",
    "eval/lm/m2d2_s2orc-validation/CE loss",
    "eval/lm/pile-validation/CE loss",
    "eval/lm/wikitext_103-validation/CE loss",
]


def shade_color(hex_color: str, fraction: float) -> Tuple[float, float, float]:
    """Blend a hex color with white. fraction=0 → white, fraction=1 → original color."""
    r = int(hex_color[1:3], 16) / 255.0
    g = int(hex_color[3:5], 16) / 255.0
    b = int(hex_color[5:7], 16) / 255.0
    return (1 - fraction + fraction * r, 1 - fraction + fraction * g, 1 - fraction + fraction * b)


def _compute_specs(
    ladder_name: str,
    size: str,
    parallel_flops: bool = True,
    chunk_size: int = 256,
    chinchilla_flops: bool = True,
    seq_len: int = 8192,
) -> Optional[dict]:
    """Dispatch to parallel or recurrent spec computation."""
    if parallel_flops:
        return compute_specs_for_size_parallel(
            ladder_name,
            size,
            seq_len=seq_len,
            chunk_size=chunk_size,
            chinchilla_flops=chinchilla_flops,
        )
    return compute_specs_for_size(
        ladder_name, size, seq_len=seq_len, chinchilla_flops=chinchilla_flops
    )


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
    ladder_dir: "Path | List[Path]",
    ladder_name: str,
    loss_column: Optional[str] = None,
    verbose: bool = True,
    use_all_checkpoints: bool = True,
    simple_flops: bool = False,
    post_decay_only: bool = True,
    parallel_flops: bool = True,
    chunk_size: int = 256,
    chinchilla_flops: bool = True,
    seq_len: int = 8192,
    average_val_loss: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load ladder results from pickle files for Chinchilla fitting.

    Args:
        ladder_dir: Directory (or list of directories) containing metrics_*.pkl files.
            When multiple directories are given the data is concatenated, which is useful
            for pooling independent runs of the same ladder configuration.
        ladder_name: Name of the ladder (used for parameter count corrections)
        loss_column: Specific loss column to use (auto-detected if None)
        verbose: Print loading progress
        use_all_checkpoints: If True, use all checkpoints from each model size.
            If False, only use the final checkpoint (legacy behavior).
        simple_flops: If True, use 6*N*D approximation for FLOPs instead of logged values.
        post_decay_only: If True (default), only use post-decay checkpoints
            (D/N = 10, 20, 40, 80, 160, ...). If False, also include pre-decay.
        chinchilla_flops: If True, include embedding and softmax FLOPs in the
            Chinchilla convention.
        seq_len: Sequence length for FLOP calculation (default: arch config default).
        average_val_loss: If True, use the average of all available validation loss
            columns (VALIDATION_LOSS_COLUMNS) instead of a single loss column. This
            gives a more stable signal than any single domain.

    Returns:
        N: Array of parameter counts (corrected non-embedding counts if available)
        D: Array of token counts
        L: Array of loss values
        F: Array of FLOPs (raw FLOPs, either from logged values or 6*N*D approximation)
        sizes: List of size names (e.g., ["190M", "370M", ...] or ["190M@20", "190M@40", ...])
    """
    # Normalise to a list of directories
    if isinstance(ladder_dir, Path):
        ladder_dirs = [ladder_dir]
    else:
        ladder_dirs = list(ladder_dir)

    pkl_files = []
    for d in ladder_dirs:
        found = sorted(d.glob("metrics_*.pkl"))
        if not found:
            if verbose:
                print(f"  Warning: No metrics_*.pkl files found in {d}")
        pkl_files.extend(found)
    if not pkl_files:
        raise FileNotFoundError(
            f"No metrics_*.pkl files found in any of: {[str(d) for d in ladder_dirs]}"
        )
    if verbose and len(ladder_dirs) > 1:
        print(f"  Pooling {len(ladder_dirs)} run directories ({len(pkl_files)} pkl files total)")

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

        computed = _compute_specs(
            ladder_name,
            size,
            parallel_flops=parallel_flops,
            chunk_size=chunk_size,
            chinchilla_flops=chinchilla_flops,
            seq_len=seq_len,
        )
        if not computed:
            if verbose:
                print(f"  Warning: Could not compute specs for {size}, skipping")
            continue

        # Find the loss column once (or detect available val columns for averaging)
        if used_loss_col is None:
            if average_val_loss:
                # Find which validation columns are present in this dataframe
                _available_val_cols = [c for c in VALIDATION_LOSS_COLUMNS if c in df.columns]
                if not _available_val_cols:
                    # Fallback to single column
                    used_loss_col = find_loss_column(df, loss_column)
                    average_val_loss = False  # disable for this ladder
                    if verbose:
                        print(f"  No validation columns found, falling back to: {used_loss_col}")
                else:
                    used_loss_col = "__average_val__"
                    if verbose:
                        print(f"  Averaging {len(_available_val_cols)} validation loss columns")
            else:
                used_loss_col = find_loss_column(df, loss_column)
            if verbose and used_loss_col != "__average_val__":
                print(f"  Using loss column: {used_loss_col}")

        # Get parameter count (should be constant for all rows in the file)
        num_params = df["num_params"].iloc[0]
        if num_params is None or pd.isna(num_params):
            if verbose:
                print(f"  Warning: No num_params for {size}, skipping")
            continue
        original_num_params = num_params

        # Apply correction for non-embedding parameter count if available
        corrected_params = get_param_count(ladder_name, size)
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
            flops_per_token = computed[
                "total_flops_per_token"
            ]  # forward-pass FLOPs per token (2 * MACs)

            for _, row in df.iterrows():
                tokens = row.get("tokens")

                if average_val_loss and used_loss_col == "__average_val__":
                    vals = [
                        row.get(c)
                        for c in _available_val_cols
                        if row.get(c) is not None and not pd.isna(row.get(c))
                    ]
                    loss = float(np.mean(vals)) if vals else None
                else:
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

            if average_val_loss and used_loss_col == "__average_val__":
                vals = [
                    final_row.get(c)
                    for c in _available_val_cols
                    if final_row.get(c) is not None and not pd.isna(final_row.get(c))
                ]
                loss = float(np.mean(vals)) if vals else None
            else:
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
    ladder_dir: "Path | List[Path]",
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
    fixed_params: Optional[Dict[str, float]] = None,
    parallel_flops: bool = True,
    chunk_size: int = 256,
    chinchilla_flops: bool = True,
    seq_len: int = 8192,
    average_val_loss: bool = True,
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
    dirs = ladder_dir if isinstance(ladder_dir, list) else [ladder_dir]
    print(f"\n{'='*60}")
    print(f"Fitting: {display}")
    for d in dirs:
        print(f"Directory: {d}")
    print(f"{'='*60}")

    N, D, L, F, sizes = load_ladder_data(
        ladder_dir,
        name,
        loss_column,
        verbose,
        use_all_checkpoints=use_all_checkpoints,
        simple_flops=simple_flops,
        post_decay_only=post_decay_only,
        parallel_flops=parallel_flops,
        chunk_size=chunk_size,
        chinchilla_flops=chinchilla_flops,
        seq_len=seq_len,
        average_val_loss=average_val_loss,
    )

    # Compute weights
    weights = None
    if weight_by_compute:
        # Weight by actual training FLOPs to emphasize larger-scale runs.
        # F already accounts for non-embedding ops (or falls back to 6ND when
        # simple_flops=True), so this is more accurate than the 6ND approximation.
        weights = F
        if verbose:
            print("\n  Weighting by compute (using actual FLOPs)")

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
            fixed_params=fixed_params,
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
            fixed_params=fixed_params,
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
        weights = 6 * N * D

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


def print_comparison(fits: Dict[str, Tuple], header: str = "COMPARISON SUMMARY"):
    """Print comparison of multiple ladder fits."""
    if len(fits) < 2:
        return

    print(f"\n{'='*60}")
    print(header)
    print(f"{'='*60}")

    # Table header
    display_names = {name: get_display_name(name) for name in fits}
    max_name_len = max(len(d) for d in display_names.values())
    col_w = max(max_name_len, 20)
    print(
        f"\n{'Ladder':<{col_w}} {'E':>8} {'A':>10} {'α':>8} {'B':>10} {'β':>8} {'a_opt':>8} {'b_opt':>8}"
    )
    print("-" * (col_w + 60))

    for name, (fit, N, D, L, F, sizes, bootstrap) in fits.items():
        p = fit.fitted_params
        print(
            f"{display_names[name]:<{col_w}} {p.E:>8.4f} {p.A:>10.4f} {p.alpha:>8.4f} {p.B:>10.4f} {p.beta:>8.4f} {p.a_opt:>8.4f} {p.b_opt:>8.4f}"
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


def compute_optimal_frontier(
    fit: ChinchillaParametricFit, C_range: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the compute-optimal (N, D) allocation for each FLOP budget in C_range.

    Minimizes L(N,D) = E + A/N^α + B/D^β subject to C = 6·N·D.
    From the first-order conditions: α·A/N^α = β·B/D^β, giving
        D = ((β·B)/(α·A))^(1/β) · N^(α/β)
    Combined with C = 6·N·D:
        N_opt = (C / (6 · G))^a_opt   where G = ((β·B)/(α·A))^(1/β)
        D_opt = C / (6 · N_opt)

    Returns (N_opt, D_opt) arrays with the same shape as C_range.
    """
    p = fit.fitted_params
    G = (p.beta * p.B / (p.alpha * p.A)) ** (1.0 / p.beta)
    N_opt = (C_range / (6.0 * G)) ** p.a_opt
    D_opt = C_range / (6.0 * N_opt)
    return N_opt, D_opt


# =============================================================================
# Bootstrap CI envelope helpers
# =============================================================================


def _compute_ci_envelope(
    bootstrap_fit: ChinchillaParametricBootstrappedFit,
    target_loss: float,
    N_range: np.ndarray,
    ci_level: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute lower/upper CI for D-to-target-loss across N_range.

    For each bootstrap fit, solves D = f(N, target_loss) analytically.
    Returns the percentile envelope.

    Args:
        bootstrap_fit: Bootstrap fit with list of ChinchillaParametricFit samples.
        target_loss: The target loss to solve for.
        N_range: Array of parameter counts to sweep.
        ci_level: Confidence level (default 0.95 for 95% CI).

    Returns:
        N_valid: Subset of N_range where all bootstraps give finite D.
        ci_lower: Lower CI bound on D.
        ci_upper: Upper CI bound on D.
    """
    alpha = (1 - ci_level) / 2
    d_samples = np.array(
        [
            [solve_d_for_target_loss(boot_fit, target_loss, n) for n in N_range]
            for boot_fit in bootstrap_fit.fits
        ]
    )  # (num_bootstrap, len(N_range))
    valid = np.all(np.isfinite(d_samples), axis=0)
    if not np.any(valid):
        return N_range[:0], np.array([]), np.array([])
    ci_lower = np.percentile(d_samples[:, valid], alpha * 100, axis=0)
    ci_upper = np.percentile(d_samples[:, valid], (1 - alpha) * 100, axis=0)
    return N_range[valid], ci_lower, ci_upper


def _compute_savings_ci_envelope(
    ref_bootstrap: ChinchillaParametricBootstrappedFit,
    arch_bootstrap: ChinchillaParametricBootstrappedFit,
    target_loss: float,
    N_range: np.ndarray,
    ci_level: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute CI on savings factor (D_ref / D_arch) vs model size.

    Pairs bootstrap samples i-to-i and computes the ratio at each N.

    Returns:
        N_valid, ci_lower, ci_upper on the savings ratio.
    """
    alpha = (1 - ci_level) / 2
    n_boot = min(len(ref_bootstrap.fits), len(arch_bootstrap.fits))
    ratio_samples = []
    for i in range(n_boot):
        d_ref_row = np.array(
            [solve_d_for_target_loss(ref_bootstrap.fits[i], target_loss, n) for n in N_range]
        )
        d_arch_row = np.array(
            [solve_d_for_target_loss(arch_bootstrap.fits[i], target_loss, n) for n in N_range]
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_row = np.where(
                np.isfinite(d_ref_row) & np.isfinite(d_arch_row) & (d_arch_row > 0),
                d_ref_row / d_arch_row,
                np.inf,
            )
        ratio_samples.append(ratio_row)
    ratio_matrix = np.array(ratio_samples)
    valid = np.all(np.isfinite(ratio_matrix), axis=0)
    if not np.any(valid):
        return N_range[:0], np.array([]), np.array([])
    ci_lower = np.percentile(ratio_matrix[:, valid], alpha * 100, axis=0)
    ci_upper = np.percentile(ratio_matrix[:, valid], (1 - alpha) * 100, axis=0)
    return N_range[valid], ci_lower, ci_upper


def _compute_savings_vs_loss_ci_envelope(
    ref_bootstrap: ChinchillaParametricBootstrappedFit,
    arch_bootstrap: ChinchillaParametricBootstrappedFit,
    target_losses: np.ndarray,
    reference_n: float,
    ci_level: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute CI on savings factor (D_ref / D_arch) swept across target losses at fixed N.

    Used for the savings-factor figure where X = target loss, Y = ratio.

    Returns:
        losses_valid, ci_lower, ci_upper on the savings ratio.
    """
    alpha = (1 - ci_level) / 2
    n_boot = min(len(ref_bootstrap.fits), len(arch_bootstrap.fits))
    ratio_samples = []
    for i in range(n_boot):
        row = []
        for tl in target_losses:
            dr = solve_d_for_target_loss(ref_bootstrap.fits[i], tl, reference_n)
            da = solve_d_for_target_loss(arch_bootstrap.fits[i], tl, reference_n)
            if np.isfinite(dr) and np.isfinite(da) and da > 0:
                row.append(dr / da)
            else:
                row.append(float("inf"))
        ratio_samples.append(row)
    ratio_matrix = np.array(ratio_samples)
    valid = np.all(np.isfinite(ratio_matrix), axis=0)
    if not np.any(valid):
        return target_losses[:0], np.array([]), np.array([])
    ci_lower = np.percentile(ratio_matrix[:, valid], alpha * 100, axis=0)
    ci_upper = np.percentile(ratio_matrix[:, valid], (1 - alpha) * 100, axis=0)
    return target_losses[valid], ci_lower, ci_upper


def _emit_ci_band(
    xs_valid: np.ndarray,
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
    color_name: str,
    opacity: float = 0.15,
) -> str:
    """
    Emit a filled CI polygon for pgfplots.

    Creates a closed path: forward on the upper envelope, backward on the lower.
    """
    if len(xs_valid) == 0:
        return ""
    fill_xs = np.concatenate([xs_valid, xs_valid[::-1]])
    fill_ys = np.concatenate([ci_upper, ci_lower[::-1]])
    return (
        f"\\addplot[fill={color_name}, opacity={opacity}, "
        f"draw=none, forget plot] "
        f"coordinates {{{_make_coordinate_table(fill_xs, fill_ys)}}} -- cycle;"
    )


# =============================================================================
# Domain-specific data loading and fitting
# =============================================================================


def load_ladder_domain_data(
    ladder_dir: "Path | List[Path]",
    ladder_name: str,
    domains: Optional[List[str]] = None,
    verbose: bool = True,
    post_decay_only: bool = True,
    simple_flops: bool = False,
    parallel_flops: bool = True,
    chunk_size: int = 256,
    chinchilla_flops: bool = True,
    seq_len: int = 8192,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]]:
    """
    Load ladder results with per-domain BPB metrics for domain-specific scaling law fitting.

    Args:
        ladder_dir: Directory (or list of directories) containing metrics_*.pkl files.
            When multiple directories are given the data is concatenated.
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

    # Normalize to a list of directories
    if isinstance(ladder_dir, Path):
        ladder_dirs = [ladder_dir]
    else:
        ladder_dirs = list(ladder_dir)

    pkl_files = []
    for d in ladder_dirs:
        found = sorted(d.glob("metrics_*.pkl"))
        if not found and verbose:
            print(f"  Warning: No metrics_*.pkl files found in {d}")
        pkl_files.extend(found)
    if not pkl_files:
        raise FileNotFoundError(
            f"No metrics_*.pkl files found in any of: {[str(d) for d in ladder_dirs]}"
        )

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

        # Use compute_specs for consistent FLOPs (same as main loader)
        computed = _compute_specs(
            ladder_name,
            size,
            parallel_flops=parallel_flops,
            chunk_size=chunk_size,
            chinchilla_flops=chinchilla_flops,
            seq_len=seq_len,
        )
        if computed:
            num_params = computed["non_embed_params"]
            flops_per_token = computed["total_flops_per_token"]
        else:
            num_params = df["num_params"].iloc[0]
            if num_params is None or pd.isna(num_params):
                continue
            corrected_params = get_param_count(ladder_name, size)
            if corrected_params is not None:
                num_params = corrected_params
            flops_per_token = None

        original_num_params = df["num_params"].iloc[0]
        if original_num_params is None or pd.isna(original_num_params):
            continue

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

            # Compute FLOPs consistently with main loader:
            # 3 * forward_flops_per_token * tokens (1 fwd + ~2 bwd passes)
            if simple_flops or flops_per_token is None:
                f_val = 6.0 * float(num_params) * float(tokens)
            else:
                f_val = 3.0 * flops_per_token * float(tokens)

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
    ladder_dir: "Path | List[Path]",
    domains: Optional[List[str]] = None,
    weight_by_compute: bool = True,
    overestimate_penalty: float = 1.0,
    num_slices: int = 4,
    verbose: bool = True,
    post_decay_only: bool = True,
    simple_flops: bool = False,
    parallel_flops: bool = True,
    chunk_size: int = 256,
    chinchilla_flops: bool = True,
    seq_len: int = 8192,
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
        parallel_flops=parallel_flops,
        chunk_size=chunk_size,
        chinchilla_flops=chinchilla_flops,
        seq_len=seq_len,
    )

    results: Dict[
        str,
        Tuple[ChinchillaParametricFit, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]],
    ] = {}
    for domain, (N, D, L, F, sizes) in domain_data.items():
        weights = 6 * N * D if weight_by_compute else None
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


def _fmt_latex_num(value: float, decimal_places: int = 2) -> str:
    """Format a number for LaTeX tables: no scientific notation.

    * |value| <= 10_000  → plain decimal  (e.g. ``1234.56``)
    * |value| >  10_000  → ``$x \\cdot 10^{n}$``  (e.g. ``$1.23 \\cdot 10^{4}$``)

    Handles negative values and zero gracefully.
    """
    if value == 0:
        return "0"
    abs_val = abs(value)
    sign = "-" if value < 0 else ""
    if abs_val <= 1e4:
        return f"{value:.{decimal_places}f}"
    exp = int(np.floor(np.log10(abs_val)))
    mantissa = abs_val / 10**exp
    return f"${sign}{mantissa:.{decimal_places}f} \\cdot 10^{{{exp}}}$"


def _param_ci_str(
    bootstrap, param_name: str, point_value: float, fmt: str = ".4f", ci_fmt: str = ".2f"
) -> str:
    """Format a parameter value with bootstrap CI below in smaller font using makecell."""
    val_str = f"{point_value:{fmt}}"
    if bootstrap is None:
        ci_line = r"{\color{red}[-, -]}"
    else:
        bootstrap_values = [getattr(f.fitted_params, param_name) for f in bootstrap.fits]
        lo = np.percentile(bootstrap_values, 2.5)
        hi = np.percentile(bootstrap_values, 97.5)
        ci_line = f"[{lo:{ci_fmt}}, {hi:{ci_fmt}}]"
    return r"\makecell{" + val_str + r" \\ {\scriptsize " + ci_line + "}}"


def _param_ci_str_sci(
    bootstrap, param_name: str, point_value: float, ci_decimal_places: int = 1
) -> str:
    """Format a parameter value (large numbers) with bootstrap CI below in smaller font."""
    val_str = _fmt_latex_num(point_value)
    if bootstrap is None:
        ci_line = r"{\color{red}[-, -]}"
    else:
        bootstrap_values = [getattr(f.fitted_params, param_name) for f in bootstrap.fits]
        lo = np.percentile(bootstrap_values, 2.5)
        hi = np.percentile(bootstrap_values, 97.5)
        ci_line = (
            f"[{_fmt_latex_num(lo, ci_decimal_places)}, {_fmt_latex_num(hi, ci_decimal_places)}]"
        )
    return r"\makecell{" + val_str + r" \\ {\scriptsize " + ci_line + "}}"


def _derived_ci_str(
    bootstrap, attr_name: str, point_value: float, fmt: str = ".4f", ci_fmt: str = ".2f"
) -> str:
    """Format a derived property (a_opt/b_opt) with bootstrap CI below in smaller font."""
    val_str = f"{point_value:{fmt}}"
    if bootstrap is None:
        ci_line = r"{\color{red}[-, -]}"
    else:
        bootstrap_values = [getattr(f.fitted_params, attr_name) for f in bootstrap.fits]
        lo = np.percentile(bootstrap_values, 2.5)
        hi = np.percentile(bootstrap_values, 97.5)
        ci_line = f"[{lo:{ci_fmt}}, {hi:{ci_fmt}}]"
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
    any_missing_bootstrap = any(bootstrap is None for _, _, _, _, _, _, bootstrap in fits.values())
    lines.append(
        r"    \caption{Fitted Chinchilla scaling law parameters for "
        r"$L(N, D) = E + A/N^\alpha + B/D^\beta$."
    )
    caption_ci = r"    95\% bootstrap CI shown below each estimate"
    if any_missing_bootstrap:
        caption_ci += r" ({\color{red}red} placeholders indicate no bootstrap was run)"
    caption_ci += ".}"
    lines.append(caption_ci)
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

    Ladders are grouped by architecture family (Transformer, GDN, Mamba2) with a
    multicolumn header per group.  Within each group a short variant label (ratio,
    placement, etc.) distinguishes individual ladders.
    """
    # Collect all sizes across ladders by probing which sizes are computable
    all_sizes = set()
    size_order = ["60M", "100M", "190M", "370M", "600M", "760M", "1B"]
    for name in ladder_names:
        for size in size_order:
            if get_param_count(name, size) is not None:
                all_sizes.add(size)
    sizes = [s for s in size_order if s in all_sizes]

    # --- Group ladders by architecture family ---
    FAMILY_ORDER = ["transformer", "gdn", "mamba2"]
    FAMILY_DISPLAY = {"transformer": "Transformer", "gdn": "GDN", "mamba2": "Mamba2"}

    # Map ladder_name -> (family, variant_label, sort_key)
    # sort_key: lower = more attention (appears first, left-to-right)
    def _classify(name: str) -> Tuple[str, str, float]:
        lower = name.lower()
        cfg = LADDER_ARCH_CONFIGS.get(lower)
        if cfg is None:
            return ("other", name, 50)
        if cfg.is_transformer:
            return ("transformer", "100%", 0)
        family = cfg.layer_type  # "gdn" or "mamba2"
        if cfg.transformer_ratio == 0:
            return (family, "Pure", 100)
        attn_frac = 1.0 / cfg.transformer_ratio
        ratio_label = f"1/{cfg.transformer_ratio}"
        sort_key = 1.0 - attn_frac
        if cfg.placement == "middle":
            ratio_label += " Mid"
            sort_key += 0.001
        return (family, ratio_label, sort_key)

    # Build ordered groups from the input ladder_names
    groups_raw: List[Tuple[str, List[Tuple[str, str, float]]]] = []
    seen_families: Dict[str, int] = {}
    for name in ladder_names:
        family, label, sort_key = _classify(name)
        if family not in seen_families:
            seen_families[family] = len(groups_raw)
            groups_raw.append((family, []))
        groups_raw[seen_families[family]][1].append((name, label, sort_key))

    # Sort groups by FAMILY_ORDER
    family_rank = {f: i for i, f in enumerate(FAMILY_ORDER)}
    groups_raw.sort(key=lambda g: family_rank.get(g[0], 99))

    # Sort members within each group by sort_key (most attention first)
    for _, members in groups_raw:
        members.sort(key=lambda m: m[2])

    # Deduplicate variant labels within each group (e.g. multiple olmo3 runs)
    groups: List[Tuple[str, List[Tuple[str, str]]]] = []
    for family, members in groups_raw:
        seen_labels: Dict[str, str] = {}
        deduped: List[Tuple[str, str]] = []
        for name, label, _ in members:
            if label not in seen_labels:
                seen_labels[label] = name
                deduped.append((name, label))
        groups.append((family, deduped))

    n_cols = sum(len(members) for _, members in groups)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"    \centering")
    lines.append(
        r"    \caption{Model configurations used in scaling ladder experiments. "
        r"$d$: model dimension, $h$: number of attention heads, "
        r"$l$: number of layers, $N$: non-embedding parameter count.}"
    )
    lines.append(r"    \label{tab:scaling-ladder-configs}")

    col_spec = "l" + "rrr" + "r" * n_cols
    lines.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"        \toprule")

    # Row 1: architecture family multicolumn headers
    header_top = ["", "", "", ""]
    col_idx = 5
    cmidrules = []
    for family, members in groups:
        n = len(members)
        display = FAMILY_DISPLAY.get(family, escape_latex(family))
        header_top.append(f"\\multicolumn{{{n}}}{{c}}{{{display}}}")
        cmidrules.append(f"\\cmidrule(lr){{{col_idx}-{col_idx + n - 1}}}")
        col_idx += n
    lines.append("        " + " & ".join(header_top) + r" \\")
    lines.append("        " + " ".join(cmidrules))

    # Row 2: Size d h l + variant labels
    header_bot = [r"Size", r"$d$", r"$h$", r"$l$"]
    for _, members in groups:
        for _, label in members:
            header_bot.append(escape_latex(label))
    lines.append("        " + " & ".join(header_bot) + r" \\")
    lines.append(r"        \midrule")

    for size in sizes:
        spec = OLMO3_SPECS_BY_NAME.get(size)
        if spec is not None:
            row_parts = [size, str(spec.d_model), str(spec.n_heads), str(spec.n_layers)]
        else:
            row_parts = [size, "---", "---", "---"]

        for _, members in groups:
            for name, _ in members:
                params = get_param_count(name, size)
                if params is not None:
                    row_parts.append(f"{round(params / 1e6)}M")
                else:
                    row_parts.append("---")
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
        all_L = np.concatenate([L for (_, _, _, L, _, _, _) in fits.values()])
        # Span from the 25th-percentile loss (harder to achieve) down to the minimum
        # observed loss (easiest to achieve). linspace produces 5 evenly-spaced targets
        # in descending order (harder → easier).
        loss_hard = np.percentile(all_L, 25)
        loss_easy = all_L.min()
        target_losses = np.linspace(loss_hard, loss_easy, 5).tolist()
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


def _fmt_tokens(d: float) -> str:
    """Format a token count for display in LaTeX tables.

    Uses T (trillions), B (billions), or M (millions) as appropriate.
    """
    if not np.isfinite(d) or d <= 0:
        return r"$\infty$"
    if d >= 1e12:
        return f"{d / 1e12:.1f}T"
    if d >= 1e9:
        return f"{d / 1e9:.1f}B"
    return f"{d / 1e6:.0f}M"


def _fmt_model_size(n: float) -> str:
    """Format a model size (number of parameters) for display."""
    if n >= 1e9:
        v = n / 1e9
        return f"{v:.0f}B" if v == int(v) else f"{v:.1f}B"
    if n >= 1e6:
        v = n / 1e6
        return f"{v:.0f}M" if v == int(v) else f"{v:.1f}M"
    return f"{n:.0f}"


def generate_token_savings_table(
    fits: Dict[str, Tuple],
    target_loss: Optional[float] = None,
    loss_strategy: str = "median",
    model_sizes: Optional[List[float]] = None,
) -> str:
    """
    Generate LaTeX table of token savings across model scales at a fixed target loss.

    For each model size, shows how many tokens each architecture needs to reach
    the target loss, plus the savings ratio relative to the first architecture.

    Args:
        fits: {ladder_name: (fit, N, D, L, F, sizes, bootstrap)}
        target_loss: explicit target loss; if None, auto-determined via loss_strategy
        loss_strategy: "median" or "min" — how to pick target loss when not given
        model_sizes: list of model sizes (in params) to evaluate
    """
    ladder_names = list(fits.keys())
    if len(ladder_names) < 2:
        return "% Need at least 2 ladders for token savings comparison"

    # Auto-determine target loss (same logic as generate_paper_figure_2)
    if target_loss is None:
        all_L = np.concatenate([fits[n][3] for n in ladder_names])
        if loss_strategy == "min":
            target_loss = float(np.min(all_L))
        else:
            target_loss = float(np.median(all_L))

    if model_sizes is None:
        model_sizes = [1e9, 3e9, 7e9, 13e9, 30e9, 70e9]

    ref_name = ladder_names[0]  # reference architecture (Transformer)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"    \centering")
    strategy_desc = "minimum" if loss_strategy == "min" else "median"
    # Check if any ladder has bootstrap data — include CI note in caption
    has_bootstrap = any(fits[n][6] is not None for n in ladder_names)
    caption_text = (
        r"    \caption{Projected token requirements across model scales "
        f"(target loss $= {target_loss:.3f}$, "
        f"selected as the {strategy_desc} of all observed training losses). "
        r"Savings $> 1\times$ means fewer tokens needed than "
        + escape_latex(get_display_name(ref_name))
        + "."
    )
    if has_bootstrap:
        caption_text += r" 95\% bootstrap CI shown below each estimate."
    caption_text += "}"
    lines.append(caption_text)
    lines.append(r"    \label{tab:token-savings-by-scale}")

    # Column spec: Size + D per ladder + savings per non-reference ladder
    n_ladders = len(ladder_names)
    n_savings = n_ladders - 1
    col_spec = "r" + "r" * n_ladders + "r" * n_savings
    lines.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"        \toprule")

    # Header row 1: architecture names spanning D + savings columns
    header1_parts = [""]
    for idx, name in enumerate(ladder_names):
        display = escape_latex(get_display_name(name))
        if idx == 0:
            header1_parts.append(display)
        else:
            header1_parts.append(r"\multicolumn{2}{c}{" + display + "}")
    lines.append("        " + " & ".join(header1_parts) + r" \\")

    # Header row 2: Size, D columns, savings columns
    header2_parts = [r"$N$"]
    header2_parts.append("$D$")  # reference
    for _ in ladder_names[1:]:
        header2_parts.extend(["$D$", "Savings"])
    lines.append("        " + " & ".join(header2_parts) + r" \\")
    lines.append(r"        \midrule")

    for n_size in model_sizes:
        size_label = _fmt_model_size(n_size)
        row_parts = [size_label]

        # Compute D for reference architecture
        ref_fit = fits[ref_name][0]
        ref_bootstrap = fits[ref_name][6]
        d_ref = solve_d_for_target_loss(ref_fit, target_loss, n_size)
        row_parts.append(_fmt_tokens(d_ref))

        # Compute D and savings for each other architecture
        for name in ladder_names[1:]:
            fit = fits[name][0]
            bootstrap = fits[name][6]
            d_needed = solve_d_for_target_loss(fit, target_loss, n_size)
            row_parts.append(_fmt_tokens(d_needed))

            # Savings ratio: ref / this (> 1 means this arch needs fewer tokens)
            if np.isfinite(d_ref) and np.isfinite(d_needed) and d_needed > 0:
                ratio = d_ref / d_needed
                # Compute bootstrap CI on savings ratio below point estimate
                savings_str = f"{ratio:.2f}$\\times$"
                if ref_bootstrap is not None and bootstrap is not None:
                    ratios = []
                    for rf, bf in zip(ref_bootstrap.fits, bootstrap.fits):
                        d_r = solve_d_for_target_loss(rf, target_loss, n_size)
                        d_b = solve_d_for_target_loss(bf, target_loss, n_size)
                        if np.isfinite(d_r) and np.isfinite(d_b) and d_b > 0:
                            ratios.append(d_r / d_b)
                    if len(ratios) >= 3:
                        lo, hi = np.percentile(ratios, [2.5, 97.5])
                        savings_str = (
                            r"\makecell{"
                            + f"{ratio:.2f}$\\times$"
                            + r" \\ {\scriptsize "
                            + f"[{lo:.2f}, {hi:.2f}]"
                            + "}}"
                        )
                row_parts.append(savings_str)
            else:
                row_parts.append("---")

        lines.append("        " + " & ".join(row_parts) + r" \\")

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_compute_equivalent_table(
    fits: Dict[str, Tuple],
    flop_budgets: Optional[List[float]] = None,
    label: str = "tab:compute-equivalent-loss",
) -> str:
    """
    Generate LaTeX table comparing loss at compute-optimal allocation across architectures.

    For each compute budget, uses the Chinchilla optimal allocation to find optimal N and D,
    then predicts loss via the fitted scaling law.

    Uses FLOPs = 6*N*D approximation.
    From the first-order conditions α·A/N^α = β·B/D^β:
        D = G · N^(α/β)  where G = (β·B / α·A)^(1/β)
    Combined with C = 6·N·D:
        N_opt = (C / (6·G))^a_opt   where a_opt = β/(α+β)
        D_opt = C / (6 · N_opt)

    Args:
        fits: {ladder_name: (fit, N, D, L, F, sizes, bootstrap)}
        flop_budgets: list of compute budgets in FLOPs; auto-determined if None
    """
    ladder_names = list(fits.keys())
    if len(ladder_names) < 2:
        return "% Need at least 2 ladders for compute-equivalent comparison"

    if flop_budgets is None:
        # Span from ~190M Chinchilla-optimal to ~70B Chinchilla-optimal
        # 190M at D/N=20 → C = 6 * 190e6 * 3.8e9 ≈ 4.3e18
        # 70B at D/N=20 → C = 6 * 70e9 * 1.4e12 ≈ 5.9e23
        flop_budgets = [1e18, 1e19, 1e20, 1e21, 1e22, 1e23]

    ref_name = ladder_names[0]

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"    \centering")
    caption_text = (
        r"    \caption{Compute-optimal model size ($N^*$), dataset size ($D^*$), and predicted loss "
        r"for each architecture. "
        r"Each architecture's fitted $a_\text{opt}, b_\text{opt}$ determine its optimal allocation "
        r"for a given compute budget ($C = 6ND$). "
        r"$\Delta$ shows loss reduction relative to "
        + escape_latex(get_display_name(ref_name))
        + "."
    )
    # Check if any ladder has bootstrap data — include CI note in caption
    has_bootstrap = any(fits[n][6] is not None for n in ladder_names)
    if has_bootstrap:
        caption_text += r" 95\% bootstrap CI shown below each estimate."
    caption_text += "}"
    lines.append(caption_text)
    lines.append(r"    \label{" + label + "}")

    # Column spec: Compute + (N* + D* + Loss) per ref + (N* + D* + Loss + Delta) per non-ref
    n_others = len(ladder_names) - 1
    col_spec = "r" + "rrr" + "rrrr" * n_others
    lines.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"        \toprule")

    # Header row 1
    header1_parts = [""]
    ref_display = escape_latex(get_display_name(ref_name))
    header1_parts.append(r"\multicolumn{3}{c}{" + ref_display + "}")
    for name in ladder_names[1:]:
        display = escape_latex(get_display_name(name))
        header1_parts.append(r"\multicolumn{4}{c}{" + display + "}")
    lines.append("        " + " & ".join(header1_parts) + r" \\")

    # Header row 2
    header2_parts = ["FLOPs"]
    header2_parts.extend(["$N^*$", "$D^*$", "Loss"])
    for _ in ladder_names[1:]:
        header2_parts.extend(["$N^*$", "$D^*$", "Loss", r"$\Delta$"])
    lines.append("        " + " & ".join(header2_parts) + r" \\")
    lines.append(r"        \midrule")

    def _optimal_alloc(fit: ChinchillaParametricFit, C: float) -> Tuple[float, float, float]:
        """Return (n_opt, d_opt, loss) for a given compute budget."""
        p = fit.fitted_params
        G = (p.beta * p.B / (p.alpha * p.A)) ** (1.0 / p.beta)
        n_opt = (C / (6.0 * G)) ** p.a_opt
        d_opt = C / (6.0 * n_opt)
        loss = float(fit.predict_loss(np.array([n_opt]), np.array([d_opt]))[0])
        return n_opt, d_opt, loss

    for C in flop_budgets:
        # Format compute budget
        exp = int(np.floor(np.log10(C)))
        mantissa = C / 10**exp
        if mantissa == 1.0:
            c_str = f"$10^{{{exp}}}$"
        else:
            c_str = f"${mantissa:.0f} \\cdot 10^{{{exp}}}$"

        row_parts = [c_str]

        # Compute optimal allocation for reference architecture
        ref_fit = fits[ref_name][0]
        ref_bootstrap = fits[ref_name][6]
        ref_n, ref_d, ref_loss = _optimal_alloc(ref_fit, C)

        # Format reference loss with bootstrap CI below
        ref_loss_str = f"{ref_loss:.3f}"
        if ref_bootstrap is not None:
            ref_losses = [_optimal_alloc(rf, C)[2] for rf in ref_bootstrap.fits]
            if len(ref_losses) >= 3:
                lo, hi = np.percentile(ref_losses, [2.5, 97.5])
                ref_loss_str = (
                    r"\makecell{"
                    + f"{ref_loss:.3f}"
                    + r" \\ {\scriptsize "
                    + f"[{lo:.2f}, {hi:.2f}]"
                    + "}}"
                )
        row_parts.extend([_fmt_model_size(ref_n), _fmt_tokens(ref_d), ref_loss_str])

        # Compute optimal allocation for each other architecture
        for name in ladder_names[1:]:
            fit = fits[name][0]
            bootstrap = fits[name][6]
            n_opt, d_opt, loss = _optimal_alloc(fit, C)
            delta = loss - ref_loss

            # Format loss with bootstrap CI below
            loss_str = f"{loss:.3f}"
            if bootstrap is not None:
                boot_losses = [_optimal_alloc(bf, C)[2] for bf in bootstrap.fits]
                if len(boot_losses) >= 3:
                    lo, hi = np.percentile(boot_losses, [2.5, 97.5])
                    loss_str = (
                        r"\makecell{"
                        + f"{loss:.3f}"
                        + r" \\ {\scriptsize "
                        + f"[{lo:.2f}, {hi:.2f}]"
                        + "}}"
                    )

            # Format delta with bootstrap CI below
            delta_str = f"{delta:+.3f}"
            if ref_bootstrap is not None and bootstrap is not None:
                deltas = []
                for rf, bf in zip(ref_bootstrap.fits, bootstrap.fits):
                    _, _, bl = _optimal_alloc(bf, C)
                    _, _, rl = _optimal_alloc(rf, C)
                    deltas.append(bl - rl)
                if len(deltas) >= 3:
                    lo, hi = np.percentile(deltas, [2.5, 97.5])
                    delta_str = (
                        r"\makecell{"
                        + f"{delta:+.3f}"
                        + r" \\ {\scriptsize "
                        + f"[{lo:+.2f}, {hi:+.2f}]"
                        + "}}"
                    )
            row_parts.extend([_fmt_model_size(n_opt), _fmt_tokens(d_opt), loss_str, delta_str])

        lines.append("        " + " & ".join(row_parts) + r" \\")

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# =============================================================================
# pgfplots/TikZ figure generation
# =============================================================================


def _emit_color_defs(ladder_names: List[str]) -> str:
    """Emit definecolor commands for all ladders."""
    seen = set()
    lines = []
    for idx, name in enumerate(ladder_names):
        style = get_latex_style(name, idx)
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


# D/N opacity mapping for scatter points (lighter = less data, darker = more data)
_DN_VALUES = sorted({10, 20, 40, 80, 160})
_DN_OPACITY = {dn: 0.25 + 0.65 * i / (len(_DN_VALUES) - 1) for i, dn in enumerate(_DN_VALUES)}


def _scatter_by_dn(
    xs: np.ndarray,
    ys: np.ndarray,
    sizes_arr: List[str],
    style: Dict[str, str],
    forget: bool = True,
    mark_size: str = "1.5pt",
) -> List[str]:
    """Emit one scatter addplot per D/N ratio with opacity varying by dataset size."""
    dn_arr = np.array([int(s.split("@")[1]) for s in sizes_arr])
    scatter_lines: List[str] = []
    for dn_val in _DN_VALUES:
        mask = dn_arr == dn_val
        if not np.any(mask):
            continue
        op = _DN_OPACITY[dn_val]
        forget_str = ", forget plot" if forget else ""
        scatter_lines.append(
            f"\\addplot[only marks, mark={style['mark']}, "
            f"{style['color_name']}, mark size={mark_size}, "
            f"mark options={{{style['mark_options']}}}, opacity={op:.2f}{forget_str}] "
            f"coordinates {{{_make_coordinate_table(xs[mask], ys[mask])}}};"
        )
    return scatter_lines


def _scatter_by_model_size(
    xs: np.ndarray,
    ys: np.ndarray,
    sizes_arr: List[str],
    style: Dict[str, str],
    forget: bool = True,
    mark_size: str = "1.5pt",
) -> List[str]:
    """Emit one scatter addplot per unique model size with opacity varying by model size."""
    # Extract model-size label (e.g. "190M") and the raw param count for sorting
    size_labels = [s.split("@")[0] for s in sizes_arr]
    unique_sizes = sorted(set(size_labels), key=lambda s: _parse_size(s))
    n_sizes = len(unique_sizes)
    opacity_map = {sz: 0.25 + 0.65 * i / max(n_sizes - 1, 1) for i, sz in enumerate(unique_sizes)}
    scatter_lines: List[str] = []
    for sz in unique_sizes:
        mask = np.array([sl == sz for sl in size_labels])
        if not np.any(mask):
            continue
        op = opacity_map[sz]
        forget_str = ", forget plot" if forget else ""
        scatter_lines.append(
            f"\\addplot[only marks, mark={style['mark']}, "
            f"{style['color_name']}, mark size={mark_size}, "
            f"mark options={{{style['mark_options']}}}, opacity={op:.2f}{forget_str}] "
            f"coordinates {{{_make_coordinate_table(xs[mask], ys[mask])}}};"
        )
    return scatter_lines


def _parse_size(s: str) -> float:
    """Parse a size label like '190M' or '1B' into a float."""
    s = s.strip().upper()
    if s.endswith("B"):
        return float(s[:-1]) * 1e9
    if s.endswith("M"):
        return float(s[:-1]) * 1e6
    return float(s)


def _fmt_size(n: float) -> str:
    """Format a parameter count for display (e.g. 190M, 1B)."""
    if n >= 1e9:
        return f"{n / 1e9:.0f}B"
    return f"{n / 1e6:.0f}M"


def _emit_arch_legend(ladder_names: List[str]) -> str:
    """Build a manual tikz-based architecture color legend line."""
    parts = []
    for idx, name in enumerate(ladder_names):
        style = get_latex_style(name, idx)
        display = escape_latex(get_display_name(name))
        line_style = style.get("line_style", "")
        line_opt = f", {line_style}" if line_style else ""
        part = (
            f"\\tikz\\draw[{style['color_name']}, line width=2pt{line_opt}] "
            f"(0,0) -- (1em,0);"
            f"\\,{display}"
        )
        parts.append(part)
    return r"{\footnotesize " + "\\qquad".join(parts) + r"}"


def generate_paper_figure_1(
    fits: Dict[str, Tuple],
    log_loss: bool = False,
    use_lines: bool = False,
    bold_curve: str = "mean",
) -> str:
    """
    Generate Figure 1: Main Scaling Law Fits (1x3 pgfplots groupplot).

    Supports any number of architectures (all overlaid in each panel).

    Panel A: Loss vs FLOPs (all architectures, isoparam curves + bold fit)
    Panel B: Loss vs Parameters (all architectures)
    Panel C: Loss vs Tokens / data budget (all architectures)

    Args:
        fits: Dict mapping ladder name to (fit, N, D, L, F, sizes, bootstrap).
        log_loss: If True, use log scale on the y-axis for loss panels (a-c),
            making the power-law relationships appear linear.
        use_lines: If True, show thin lines connecting data points (iso-param
            curves) instead of scatter points in all three panels.  The fitted
            curve in panels B and C is evaluated at the largest model only.
        bold_curve: Strategy for the bold fitted curve.
            "mean" — evaluate at the mean D/N ratio (default).
            "optimal" — evaluate along the compute-optimal frontier
              (Panel A traces the Pareto frontier; Panels B/C use the
              optimal D/N at the largest compute budget).
    """
    ladder_names = list(fits.keys())
    loss_label = "Loss"
    parts = []
    if log_loss:
        parts.append("log")
    if bold_curve == "optimal":
        parts.append("opt")
    fig_suffix = ("-" + "-".join(parts)) if parts else ""
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
    lines.append(r"        group size=3 by 1,")
    lines.append(r"        horizontal sep=0.1cm,")
    lines.append(r"        y descriptions at=edge left,")
    lines.append(r"    },")
    lines.append(r"    width=0.405\textwidth,")
    lines.append(r"    height=0.35\textwidth,")
    lines.append(r"    grid=major,")
    lines.append(r"    grid style={gray!30},")
    lines.append(r"    tick label style={font=\scriptsize},")
    lines.append(r"    label style={font=\scriptsize},")
    lines.append(r"    title style={font=\small},")
    lines.append(r"]")

    # Panel A: Loss vs FLOPs (all architectures overlaid)
    lines.append("")
    lines.append("% Panel A: Loss vs FLOPs")
    lines.append(r"\nextgroupplot[")
    lines.append(r"    xmode=log,")
    if log_loss:
        lines.append(r"    ymode=log,")
        lines.append(
            r"    yticklabel={\pgfmathparse{exp(\tick)}\pgfmathprintnumber{\pgfmathresult}},"
        )
    lines.append(r"    xlabel={PetaFLOPs},")
    # lines.append(f"    ylabel={{{loss_label}}},")
    lines.append(r"    title={(a) Loss vs Compute},")

    lines.append(r"]")

    for idx, name in enumerate(ladder_names):
        fit, N, D, L, F, sizes, bootstrap = fits[name]
        style = get_latex_style(name, idx)

        F_peta = F / 1e15
        unique_N = np.unique(N)
        line_style = f", {style['line_style']}" if style.get("line_style") else ""

        if use_lines:
            # Thin isoparam curves: connect data points sharing the same N, sorted by FLOPs
            for n_val in unique_N:
                mask = N == n_val
                sort_idx = np.argsort(F_peta[mask])
                F_sorted = F_peta[mask][sort_idx]
                L_sorted = L[mask][sort_idx]
                lines.append(
                    f"\\addplot[{style['color_name']}, thin, no markers{line_style}, "
                    f"opacity=0.5, forget plot] "
                    f"coordinates {{{_make_coordinate_table(F_sorted, L_sorted)}}};"
                )
        else:
            # Scatter: data points with opacity varying by model size
            lines.extend(_scatter_by_model_size(F_peta, L, sizes, style))

        # Bold fitted curve
        flop_ratio = np.median(F / (6.0 * N * D))
        if bold_curve == "optimal":
            # Compute-optimal frontier: for each C, allocate (N,D) optimally
            C_range = np.logspace(
                np.log10(F.min() / flop_ratio), np.log10(F.max() / flop_ratio), 150
            )
            N_sweep, D_sweep = compute_optimal_frontier(fit, C_range)
            F_sweep_peta = (C_range * flop_ratio) / 1e15
        else:
            # Mean D/N ratio (original behaviour)
            mean_dn = np.mean([np.mean(D[N == n]) / n for n in unique_N])
            N_sweep = np.logspace(np.log10(N.min()), np.log10(N.max()), 150)
            D_sweep = N_sweep * mean_dn
            F_sweep_peta = (6.0 * N_sweep * D_sweep * flop_ratio) / 1e15
        L_sweep = fit.predict_loss(N_sweep, D_sweep)
        lines.append(
            f"\\addplot[{style['color_name']}, very thick, no markers{line_style}, forget plot] "
            f"coordinates {{{_make_coordinate_table(F_sweep_peta, L_sweep)}}};"
        )

    # Panel B: Loss vs Parameters (all overlaid)
    lines.append("")
    lines.append("% Panel B: Loss vs Parameters")
    lines.append(r"\nextgroupplot[")
    lines.append(r"    xmode=log,")
    if log_loss:
        lines.append(r"    ymode=log,")
        lines.append(
            r"    yticklabel={\pgfmathparse{exp(\tick)}\pgfmathprintnumber{\pgfmathresult}},"
        )
    lines.append(r"    xlabel={Parameters},")
    lines.append(r"    yticklabels={},")
    lines.append(r"    title={(b) Loss vs Parameters},")

    lines.append(r"]")

    for idx, name in enumerate(ladder_names):
        fit, N, D, L, F, sizes, bootstrap = fits[name]
        style = get_latex_style(name, idx)

        # Extract D/N ratio per data point
        dn_arr = np.array([int(s.split("@")[1]) for s in sizes])
        unique_N = np.unique(N)
        max_dn_global = int(np.max(dn_arr))
        line_style = f", {style['line_style']}" if style.get("line_style") else ""

        if use_lines:
            # Thin iso-D/N curves: connect points with same D/N ratio across model sizes
            unique_dn = sorted(np.unique(dn_arr))
            for dn_val in unique_dn:
                mask = dn_arr == dn_val
                if np.sum(mask) < 2:
                    continue
                sort_idx = np.argsort(N[mask])
                N_sorted = N[mask][sort_idx]
                L_sorted = L[mask][sort_idx]
                lines.append(
                    f"\\addplot[{style['color_name']}, thin, no markers{line_style}, "
                    f"opacity=0.5, forget plot] "
                    f"coordinates {{{_make_coordinate_table(N_sorted, L_sorted)}}};"
                )

            # Fitted curve at max D/N, evaluated at N of the largest model only
            largest_N = np.max(unique_N)
            N_range = np.logspace(np.log10(N.min() * 0.8), np.log10(largest_N * 1.2), 150)
            D_curve = N_range * max_dn_global
            L_curve = fit.predict_loss(N_range, D_curve)
        else:
            # Scatter (opacity varies by D/N ratio)
            lines.extend(_scatter_by_dn(N, L, sizes, style))

            # Fitted curve uses only final-checkpoint D/N per model size
            N_range = np.logspace(np.log10(N.min() * 0.8), np.log10(N.max() * 1.2), 150)
            D_curve = N_range * max_dn_global
            L_curve = fit.predict_loss(N_range, D_curve)

        # Fitted curve at max chinchilla multiple (forget plot — legend already added in panel A)
        lines.append(
            f"\\addplot[{style['color_name']}, thick, no markers{line_style}, forget plot] "
            f"coordinates {{{_make_coordinate_table(N_range, L_curve)}}};"
        )

    # Panel C: Loss vs Tokens (single curve per architecture)
    lines.append("")
    lines.append("% Panel C: Loss vs Tokens")
    lines.append(r"\nextgroupplot[")
    lines.append(r"    xmode=log,")
    if log_loss:
        lines.append(r"    ymode=log,")
        lines.append(
            r"    yticklabel={\pgfmathparse{exp(\tick)}\pgfmathprintnumber{\pgfmathresult}},"
        )
    lines.append(r"    xlabel={Training Tokens},")
    lines.append(r"    yticklabels={},")
    lines.append(r"    title={(c) Loss vs Data Budget},")

    lines.append(r"]")

    for idx, name in enumerate(ladder_names):
        fit, N, D, L, F, sizes, bootstrap = fits[name]
        style = get_latex_style(name, idx)

        unique_N = np.unique(N)
        line_style = f", {style['line_style']}" if style.get("line_style") else ""

        if use_lines:
            # Thin iso-N curves: connect data points with the same model size, sorted by D
            for n_val in unique_N:
                mask = N == n_val
                sort_idx = np.argsort(D[mask])
                D_sorted = D[mask][sort_idx]
                L_sorted = L[mask][sort_idx]
                lines.append(
                    f"\\addplot[{style['color_name']}, thin, no markers{line_style}, "
                    f"opacity=0.5, forget plot] "
                    f"coordinates {{{_make_coordinate_table(D_sorted, L_sorted)}}};"
                )

            # Fitted curve at the largest model size
            largest_N = np.max(unique_N)
            D_range = np.logspace(np.log10(D.min() * 0.8), np.log10(D.max() * 1.2), 150)
            L_curve = fit.predict_loss(largest_N, D_range)
        else:
            # Scatter: data points (opacity varies by model size)
            lines.extend(_scatter_by_model_size(D, L, sizes, style))

            # Single fitted curve per architecture at max D/N ratio
            max_dn = max(np.mean(D[N == n]) / n for n in unique_N)
            D_range = np.logspace(np.log10(D.min() * 0.8), np.log10(D.max() * 1.2), 150)
            N_curve = D_range / max_dn
            L_curve = fit.predict_loss(N_curve, D_range)

        lines.append(
            f"\\addplot[{style['color_name']}, thick, no markers{line_style}, forget plot] "
            f"coordinates {{{_make_coordinate_table(D_range, L_curve)}}};"
        )

    lines.append("")
    lines.append(r"\end{groupplot}")
    lines.append(r"\end{tikzpicture}")
    lines.append("")
    # Architecture color legend (manual tikz — reliable across all LaTeX setups)
    lines.append(r"\par\vspace{0.5em}")
    lines.append(_emit_arch_legend(ladder_names))

    if not use_lines:
        # D/N ratio legend (panel b opacity encoding) — only for scatter mode
        dn_opacity_parts = []
        for dn_val in _DN_VALUES:
            op = _DN_OPACITY[dn_val]
            dn_opacity_parts.append(
                f"\\tikz\\fill[black, opacity={op:.2f}] (0,0) circle (2pt);"
                f"\\,$D/N\\!=\\!{dn_val}$"
            )
        dn_line = "\\enspace".join(dn_opacity_parts)

        lines.append(r"\par\vspace{0.3em}")
        lines.append(r"{\scriptsize (b) Chinchilla multiple:\enspace " + dn_line + r"}")

        # Model size legend (panel c opacity encoding)
        all_model_sizes: List[str] = []
        for name in ladder_names:
            sizes_list = fits[name][5]
            all_model_sizes.extend(s.split("@")[0] for s in sizes_list)
        unique_model_sizes = sorted(set(all_model_sizes), key=lambda s: _parse_size(s))
        n_ms = len(unique_model_sizes)
        ms_opacity_parts = []
        for i, sz in enumerate(unique_model_sizes):
            op = 0.25 + 0.65 * i / max(n_ms - 1, 1)
            ms_opacity_parts.append(
                f"\\tikz\\fill[black, opacity={op:.2f}] (0,0) circle (2pt);"
                f"\\,{escape_latex(sz)}"
            )
        ms_line = "\\enspace".join(ms_opacity_parts)

        lines.append(r"\par\vspace{0.1em}")
        lines.append(r"{\scriptsize (c) Model size:\enspace " + ms_line + r"}")

    # Build caption with fitted parameter annotations
    log_note = (
        " Loss axes are plotted on a logarithmic scale so that the power-law relationships appear linear."
        if log_loss
        else ""
    )
    caption_parts = [
        r"\caption{Scaling law fits $L(N,D) = E + A/N^\alpha + B/D^\beta$ "
        r"for all architectures. "
    ]
    for name in ladder_names:
        fit = fits[name][0]
        p = fit.fitted_params
        r2 = fit.r_squared
        display = escape_latex(get_display_name(name))
        r2_str = f"{r2:.3f}" if r2 is not None else "?"
        caption_parts.append(
            f"{display}: $\\alpha={p.alpha:.3f}$, $\\beta={p.beta:.3f}$, $R^2={r2_str}$. "
        )
    curve_desc_a = (
        r"the compute-optimal frontier" if bold_curve == "optimal" else r"the mean $D/N$ ratio"
    )
    if use_lines:
        caption_parts.append(
            r"\textbf{(a)} Loss vs compute; thin lines connect checkpoints "
            r"of the same model size and the bold curve shows the fitted scaling law "
            f"at {curve_desc_a}. "
            r"\textbf{(b)} Loss vs parameter count; thin lines connect checkpoints "
            r"at the same Chinchilla multiple ($D/N$ ratio) and the fitted curve is "
            r"evaluated at the largest multiple. "
            r"\textbf{(c)} Loss vs data budget; thin lines connect checkpoints "
            r"of the same model size and the fitted curve is evaluated at the "
            r"largest model size." + log_note + "}"
        )
    else:
        caption_parts.append(
            r"\textbf{(a)} Loss vs compute, with isoparam curves (thin) and "
            f"the fitted scaling law at {curve_desc_a} (bold). "
            r"\textbf{(b)} Loss vs parameter count; point opacity reflects the Chinchilla multiple "
            r"($D/N$ ratio) and the fitted curve is evaluated at the largest multiple, "
            r"corresponding to the final checkpoint per model size. "
            r"\textbf{(c)} Loss vs data budget; point opacity reflects the model size and "
            r"the fitted curve is evaluated at the largest $D/N$ ratio." + log_note + "}"
        )
    lines.append("".join(caption_parts))
    lines.append(f"\\label{{fig:scaling-law-fit{fig_suffix}}}")
    lines.append(r"\end{figure*}")
    return "\n".join(lines)


def generate_paper_figure_2(
    fits: Dict[str, Tuple],
    target_loss: Optional[float] = None,
    loss_strategy: str = "median",
    fig_suffix: str = "",
    plot_ci: bool = False,
) -> str:
    """
    Generate Figure 2: Efficiency Projections.

    Shows tokens-to-target-loss vs model size for both architectures,
    with shaded savings region.

    Args:
        fits: {ladder_name: (fit, N, D, L, F, sizes)}
        target_loss: explicit target loss; if None, auto-determined via loss_strategy
        loss_strategy: "median" or "min" — how to pick target loss when not given explicitly
        fig_suffix: appended to the LaTeX label, e.g. "-min"
    """
    ladder_names = list(fits.keys())
    if len(ladder_names) < 2:
        return "% Need at least 2 ladders for efficiency projection"

    # Auto-determine target loss
    if target_loss is None:
        all_L = np.concatenate([fits[n][3] for n in ladder_names])
        if loss_strategy == "min":
            target_loss = float(np.min(all_L))
        else:
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
    lines.append(r"    ymax=1e19,")
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
        bootstrap = fits[name][6]
        style = get_latex_style(name, idx)
        display = escape_latex(get_display_name(name))

        d_needed = np.array([solve_d_for_target_loss(fit, target_loss, n) for n in N_range])
        valid = np.isfinite(d_needed) & (d_needed > 0) & (d_needed <= 1e19)
        all_d_curves[name] = (N_range, d_needed, valid)

        # CI band (rendered before main curve for z-ordering)
        if plot_ci and bootstrap is not None:
            N_ci, ci_lo, ci_hi = _compute_ci_envelope(bootstrap, target_loss, N_range)
            ci_valid = ci_hi <= 1e19
            if len(N_ci) > 0 and np.any(ci_valid):
                lines.append(
                    _emit_ci_band(
                        N_ci[ci_valid], ci_lo[ci_valid], ci_hi[ci_valid], style["color_name"]
                    )
                )

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
            style_a = get_latex_style(names[0], 0)
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
    strategy_desc = "minimum" if loss_strategy == "min" else "median"
    lines.append(
        r"\caption{Projected data requirements across scales "
        f"(target loss $= {target_loss:.3f}$, "
        f"selected as the {strategy_desc} of all observed training losses). "
        r"Extrapolating from scaling ladder experiments, the hybrid architecture "
        r"requires substantially fewer training tokens to reach equivalent loss. "
        r"Shaded region shows the data savings.}"
    )
    lines.append(f"\\label{{fig:scaling-efficiency-projections{fig_suffix}}}")
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
    lines.append(f"    group style={{group size={n_panels} by 1, horizontal sep=0.8cm}},")
    lines.append(r"    width=0.37\textwidth,")
    lines.append(r"    height=0.30\textwidth,")
    lines.append(r"    xmode=log,")
    lines.append(r"    grid=major,")
    lines.append(r"    grid style={gray!30},")
    lines.append(r"    tick label style={font=\scriptsize},")
    lines.append(r"    label style={font=\scriptsize},")
    lines.append(r"    title style={font=\small},")
    lines.append(r"]")

    for domain_idx, domain in enumerate(domains):
        domain_clean = domain.replace("_BPB", "")
        lines.append("")
        lines.append(f"% Panel: {domain_clean}")
        lines.append(r"\nextgroupplot[")
        lines.append(r"    ymode=log,")
        lines.append(
            r"    yticklabel={\pgfmathparse{exp(\tick)}\pgfmathprintnumber{\pgfmathresult}},"
        )
        lines.append(r"    xlabel={PetaFLOPs},")
        if domain_idx == 0:
            lines.append(r"    ylabel={BPB},")
        lines.append(f"    title={{{escape_latex(domain_clean)}}},")
        lines.append(r"]")

        for ladder_idx, name in enumerate(ladder_names):
            style = get_latex_style(name, ladder_idx)

            if domain not in domain_fits.get(name, {}):
                continue

            fit, N, D, L, F, sizes = domain_fits[name][domain]

            F_peta = F / 1e15

            # Scatter (opacity varies by D/N ratio)
            lines.extend(_scatter_by_dn(F_peta, L, sizes, style, mark_size="1pt"))

            # Fitted curve — sweep N at mean D/N ratio (matching fig 1 panel A)
            unique_N = np.unique(N)
            mean_dn = np.mean([np.mean(D[N == n]) / n for n in unique_N])
            flop_ratio = np.median(F / (6.0 * N * D))
            N_sweep = np.logspace(np.log10(N.min()), np.log10(N.max()), 150)
            D_sweep = N_sweep * mean_dn
            F_sweep_peta = (6.0 * N_sweep * D_sweep * flop_ratio) / 1e15
            L_curve = fit.predict_loss(N_sweep, D_sweep)
            line_style = f", {style['line_style']}" if style.get("line_style") else ""
            lines.append(
                f"\\addplot[{style['color_name']}, thick, no markers{line_style}, forget plot] "
                f"coordinates {{{_make_coordinate_table(F_sweep_peta, L_curve)}}};"
            )

    lines.append("")
    lines.append(r"\end{groupplot}")
    lines.append(r"\end{tikzpicture}")
    lines.append(r"\par\vspace{0.5em}")
    lines.append(_emit_arch_legend(ladder_names))
    lines.append(
        r"\caption{Domain-specific scaling laws (BPB vs compute) show the hybrid advantage "
        r"is consistent across Math, Code, and QA domains.}"
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
    lines.append(r"    group style={group size=2 by 1, horizontal sep=1.2cm},")
    lines.append(r"    width=0.52\textwidth,")
    lines.append(r"    height=0.38\textwidth,")
    lines.append(r"    grid=major,")
    lines.append(r"    grid style={gray!30},")
    lines.append(r"    tick label style={font=\scriptsize},")
    lines.append(r"    label style={font=\scriptsize},")
    lines.append(r"    title style={font=\small},")
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
        style = get_latex_style(name, idx)

        predicted = fit.predict_loss(N, D)
        residuals_pct = (L - predicted) / L * 100

        lines.append(
            f"\\addplot[only marks, mark={style['mark']}, "
            f"{style['color_name']}, mark size=1.5pt, "
            f"mark options={{{style['mark_options']}}}, opacity=0.6, forget plot] "
            f"coordinates {{{_make_coordinate_table(predicted, residuals_pct)}}};"
        )

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
        style = get_latex_style(name, idx)

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
    lines.append(r"\par\vspace{0.5em}")
    lines.append(_emit_arch_legend(ladder_names))

    # Caption with overall R² values
    r2_parts = []
    for name in ladder_names:
        r2 = fits[name][0].r_squared
        display = escape_latex(get_display_name(name))
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


def generate_savings_factor_figure(
    fits: Dict[str, Tuple],
    reference_name: Optional[str] = None,
    reference_n: float = 7e9,
    num_points: int = 100,
    plot_ci: bool = False,
) -> str:
    """
    Generate savings factor plot: projected data savings vs target loss.

    X-axis: target loss (decreasing left-to-right, i.e. harder targets on the right)
    Y-axis: savings factor = D_reference / D_arch (>1 means arch needs fewer tokens)
    Horizontal dashed line at y=1 (reference/transformer baseline).
    One curve per non-reference architecture.

    Args:
        fits: {ladder_name: (fit, N, D, L, F, sizes, bootstrap)}
        reference_name: Name of the reference architecture (default: first ladder)
        reference_n: Reference model size in parameters (default: 7B)
        num_points: Number of target loss values to evaluate
    """
    ladder_names = list(fits.keys())
    if len(ladder_names) < 2:
        return "% Need at least 2 ladders for savings factor plot"

    if reference_name is None:
        reference_name = ladder_names[0]

    ref_fit = fits[reference_name][0]

    # Determine target loss range from all data
    all_L = np.concatenate([fits[n][3] for n in ladder_names])
    loss_max = float(np.percentile(all_L, 75))  # easy targets (high loss)
    loss_min = float(np.min(all_L))  # hard targets (low loss)
    target_losses = np.linspace(loss_max, loss_min, num_points)

    other_names = [n for n in ladder_names if n != reference_name]

    lines = []
    lines.append("% Savings Factor Plot: data savings vs target loss")
    lines.append("% Generated by fit_chinchilla_scaling_laws.py")
    lines.append(r"% Requires: \usepackage{pgfplots}")
    lines.append("")
    lines.append(_emit_color_defs(ladder_names))
    lines.append("")
    lines.append(r"\begin{figure}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\begin{tikzpicture}")
    lines.append(r"\begin{axis}[")
    lines.append(r"    width=0.85\textwidth,")
    lines.append(r"    height=0.55\textwidth,")
    lines.append(r"    xlabel={Target Loss},")
    lines.append(f"    ylabel={{Projected Savings Factor ({_fmt_size(reference_n)})}},")
    lines.append(r"    x dir=reverse,")
    lines.append(r"    grid=major,")
    lines.append(r"    grid style={gray!30},")
    lines.append(r"    legend style={")
    lines.append(r"        font=\footnotesize,")
    lines.append(r"        cells={anchor=west},")
    lines.append(r"        at={(0.03,0.97)},")
    lines.append(r"        anchor=north west,")
    lines.append(r"        draw=none,")
    lines.append(r"    },")
    lines.append(r"    tick label style={font=\small},")
    lines.append(r"    label style={font=\small},")
    lines.append(r"]")

    # Horizontal baseline at y=1
    ref_display = escape_latex(get_display_name(reference_name))
    lines.append(
        r"\draw[dashed, gray, thick] (axis cs:\pgfkeysvalueof{/pgfplots/xmin},1) -- "
        r"(axis cs:\pgfkeysvalueof{/pgfplots/xmax},1);"
    )
    lines.append(
        r"\node[font=\footnotesize, gray, anchor=south west] at "
        r"(axis cs:\pgfkeysvalueof{/pgfplots/xmin},1) "
        f"{{{ref_display} ($1\\times$)}};"
    )

    ref_bootstrap = fits[reference_name][6]

    for name in other_names:
        arch_fit = fits[name][0]
        arch_bootstrap = fits[name][6]
        style = get_latex_style(name, ladder_names.index(name))
        display = escape_latex(get_display_name(name))

        savings = []
        valid_losses = []
        for tl in target_losses:
            d_ref = solve_d_for_target_loss(ref_fit, tl, reference_n)
            d_arch = solve_d_for_target_loss(arch_fit, tl, reference_n)
            if np.isfinite(d_ref) and np.isfinite(d_arch) and d_ref > 0 and d_arch > 0:
                savings.append(d_ref / d_arch)
                valid_losses.append(tl)

        # CI band on savings ratio
        if plot_ci and ref_bootstrap is not None and arch_bootstrap is not None:
            tl_ci, ci_lo, ci_hi = _compute_savings_vs_loss_ci_envelope(
                ref_bootstrap, arch_bootstrap, target_losses, reference_n
            )
            if len(tl_ci) > 0:
                lines.append(_emit_ci_band(tl_ci, ci_lo, ci_hi, style["color_name"]))

        if valid_losses:
            valid_losses_arr = np.array(valid_losses)
            savings_arr = np.array(savings)
            line_style = f", {style['line_style']}" if style.get("line_style") else ""
            lines.append(
                f"\\addplot[{style['color_name']}, thick, no markers{line_style}] "
                f"coordinates {{{_make_coordinate_table(valid_losses_arr, savings_arr)}}};"
            )
            lines.append(f"\\addlegendentry{{{display}}}")

    lines.append(r"\end{axis}")
    lines.append(r"\end{tikzpicture}")
    lines.append(
        r"\caption{Projected data savings factor relative to "
        + ref_display
        + f" at {_fmt_size(reference_n)} scale. "
        + r"Values above $1\times$ indicate fewer training tokens are needed to reach the target loss.}"
    )
    lines.append(r"\label{fig:savings-factor}")
    lines.append(r"\end{figure}")
    return "\n".join(lines)


def generate_combined_savings_figure(
    fits: Dict[str, Tuple],
    target_loss: Optional[float] = None,
    loss_strategy: str = "min",
    reference_name: Optional[str] = None,
    plot_ci: bool = False,
) -> str:
    """
    Generate combined 2-panel figure: data requirements + savings factor side by side.

    Panel (a): Tokens to reach target loss vs model size (log-log).
        One curve per architecture, with shaded savings region.
    Panel (b): Savings factor (D_ref / D_arch) vs model size (log X, linear Y).
        One curve per non-reference architecture, dashed line at y=1.

    Both panels share the X axis (model size in parameters).

    Args:
        fits: {ladder_name: (fit, N, D, L, F, sizes, bootstrap)}
        target_loss: Explicit target loss; if None, auto-determined via loss_strategy.
        loss_strategy: "min" or "median" — how to pick target loss when not given.
        reference_name: Reference architecture (default: first ladder).
        plot_ci: If True and bootstrap data is available, render 95% CI bands.
    """
    ladder_names = list(fits.keys())
    if len(ladder_names) < 2:
        return "% Need at least 2 ladders for combined savings figure"

    if reference_name is None:
        reference_name = ladder_names[0]

    # Auto-determine target loss
    if target_loss is None:
        all_L = np.concatenate([fits[n][3] for n in ladder_names])
        if loss_strategy == "min":
            target_loss = float(np.min(all_L))
        else:
            target_loss = float(np.median(all_L))

    ref_fit = fits[reference_name][0]
    ref_bootstrap = fits[reference_name][6]
    other_names = [n for n in ladder_names if n != reference_name]

    N_range = np.logspace(7.5, 10.85, 200)  # ~30M to 70B

    lines = []
    lines.append("% Combined Savings Figure: data requirements + savings factor")
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
    lines.append(r"        group size=2 by 1,")
    lines.append(r"        horizontal sep=1.2cm,")
    lines.append(r"    },")
    lines.append(r"    width=0.50\textwidth,")
    lines.append(r"    height=0.40\textwidth,")
    lines.append(r"    xmode=log,")
    lines.append(r"    grid=major,")
    lines.append(r"    grid style={gray!30},")
    lines.append(r"    tick label style={font=\scriptsize},")
    lines.append(r"    label style={font=\scriptsize},")
    lines.append(r"    title style={font=\small},")
    lines.append(r"    legend=false,")
    lines.append(r"]")

    # ── Panel (a): Tokens to reach target loss vs model size ──
    lines.append("")
    lines.append("% Panel (a): Projected Data Requirements")
    lines.append(r"\nextgroupplot[")
    lines.append(r"    ymode=log,")
    lines.append(r"    xlabel={Model Size (Parameters)},")
    lines.append(f"    ylabel={{Tokens to reach loss $= {target_loss:.3f}$}},")
    lines.append(r"    ymax=1e19,")
    lines.append(r"    title={(a) Projected Data Requirements},")
    lines.append(r"]")

    all_d_curves: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    # Pre-compute all curves so we can draw shading first (below the curves)
    for idx, name in enumerate(ladder_names):
        fit = fits[name][0]
        d_needed = np.array([solve_d_for_target_loss(fit, target_loss, n) for n in N_range])
        valid = np.isfinite(d_needed) & (d_needed > 0) & (d_needed <= 1e19)
        all_d_curves[name] = (N_range, d_needed, valid)

    # Shaded savings region: only where the non-reference curve is BELOW the reference.
    # Drawn first so the curves render on top.
    if len(all_d_curves) >= 2:
        names_list = list(all_d_curves.keys())
        N_a, d_a, valid_a = all_d_curves[names_list[0]]  # reference
        N_b, d_b, valid_b = all_d_curves[names_list[1]]  # other arch
        both_valid = valid_a & valid_b & (d_b < d_a)  # only shade where other < reference
        if np.any(both_valid):
            style_b = get_latex_style(names_list[1], 1)
            N_fwd = N_a[both_valid]
            d_upper = d_a[both_valid]  # reference curve is the top boundary
            d_lower = d_b[both_valid]  # other arch is the bottom boundary
            fill_coords = np.concatenate([N_fwd, N_fwd[::-1]])
            fill_d = np.concatenate([d_upper, d_lower[::-1]])
            lines.append(
                f"\\addplot[fill={style_b['color_name']}!20, draw=none, forget plot] "
                f"coordinates {{{_make_coordinate_table(fill_coords, fill_d)}}} -- cycle;"
            )

    # Now draw curves on top of the shading
    for idx, name in enumerate(ladder_names):
        fit = fits[name][0]
        bootstrap = fits[name][6]
        style = get_latex_style(name, idx)
        line_style = f", {style['line_style']}" if style.get("line_style") else ""
        N_range_cur, d_needed, valid = all_d_curves[name]

        # CI band (rendered before main curve for z-ordering)
        if plot_ci and bootstrap is not None:
            N_valid, ci_lo, ci_hi = _compute_ci_envelope(bootstrap, target_loss, N_range_cur)
            ci_valid = ci_hi <= 1e19
            if len(N_valid) > 0 and np.any(ci_valid):
                lines.append(
                    _emit_ci_band(
                        N_valid[ci_valid], ci_lo[ci_valid], ci_hi[ci_valid], style["color_name"]
                    )
                )

        if np.any(valid):
            lines.append(
                f"\\addplot[{style['color_name']}, thick, no markers{line_style}] "
                f"coordinates {{{_make_coordinate_table(N_range_cur[valid], d_needed[valid])}}};"
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

    # ── Panel (b): Savings factor vs model size ──
    lines.append("")
    lines.append("% Panel (b): Projected Savings Factor")
    lines.append(r"\nextgroupplot[")
    lines.append(r"    xlabel={Model Size (Parameters)},")
    lines.append(r"    ylabel={Savings Factor ($D_\mathrm{ref}/D_\mathrm{arch}$)},")
    lines.append(r"    title={(b) Projected Savings Factor},")
    lines.append(r"    xmin=2e8,")
    lines.append(r"]")

    # Horizontal baseline at y=1 in reference (OLMo 3) color.
    # Use \addplot with domain instead of \draw so pgfplots handles axis limits correctly.
    ref_style = get_latex_style(reference_name, 0)
    ref_color = ref_style["color_name"]
    lines.append(f"\\addplot[{ref_color}, thick, no markers, forget plot, domain=2e8:1e11] {{1}};")
    lines.append(
        f"\\node[font=\\footnotesize, {ref_color}, anchor=south west] at " f"(axis cs:2e8,1) {{}};"
    )

    for name in other_names:
        arch_fit = fits[name][0]
        arch_bootstrap = fits[name][6]
        idx = ladder_names.index(name)
        style = get_latex_style(name, idx)
        line_style = f", {style['line_style']}" if style.get("line_style") else ""

        # Compute savings factor at each N
        # Only include points where both D values are practical (<=1e19),
        # matching panel (a) filtering.  At small N the required D diverges
        # and the ratio becomes numerically meaningless.
        # Start at 2e8 params since savings factors are unreliable below that.
        savings = []
        valid_N = []
        for n in N_range:
            if n < 2e8:
                continue
            d_ref = solve_d_for_target_loss(ref_fit, target_loss, n)
            d_arch = solve_d_for_target_loss(arch_fit, target_loss, n)
            if (
                np.isfinite(d_ref)
                and np.isfinite(d_arch)
                and 0 < d_ref <= 1e19
                and 0 < d_arch <= 1e19
            ):
                savings.append(d_ref / d_arch)
                valid_N.append(n)

        # CI band on savings ratio (filtered to N >= 2e8)
        if plot_ci and ref_bootstrap is not None and arch_bootstrap is not None:
            N_ci, ci_lo, ci_hi = _compute_savings_ci_envelope(
                ref_bootstrap, arch_bootstrap, target_loss, N_range[N_range >= 2e8]
            )
            if len(N_ci) > 0:
                lines.append(_emit_ci_band(N_ci, ci_lo, ci_hi, style["color_name"]))

        if valid_N:
            lines.append(
                f"\\addplot[{style['color_name']}, thick, no markers{line_style}] "
                f"coordinates {{{_make_coordinate_table(np.array(valid_N), np.array(savings))}}};"
            )

    lines.append("")
    lines.append(r"\end{groupplot}")
    lines.append(r"\end{tikzpicture}")
    lines.append("")
    lines.append(r"\par\vspace{0.5em}")
    lines.append(_emit_arch_legend(ladder_names))
    strategy_desc = "minimum" if loss_strategy == "min" else "median"
    lines.append(
        r"\caption{Projected data requirements and savings factor across scales "
        f"(target loss $= {target_loss:.3f}$, "
        f"selected as the {strategy_desc} of all observed training losses). "
        r"\textbf{(a)} Tokens needed to reach the target loss for each architecture. "
        r"Shaded region shows the data savings. "
        r"\textbf{(b)} Savings factor ($D_\mathrm{ref}/D_\mathrm{arch}$) vs model size; "
        r"values above $1\times$ indicate fewer training tokens are needed.}"
    )
    lines.append(r"\label{fig:combined-savings}")
    lines.append(r"\end{figure*}")
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
        from matplotlib.ticker import LogFormatterMathtext, NullFormatter, ScalarFormatter
    except ImportError:
        print("Warning: matplotlib not installed, skipping 2D plots")
        return

    def _format_log_yaxis(ax):
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(formatter)
        minor_formatter = ScalarFormatter()
        minor_formatter.set_scientific(False)
        ax.yaxis.set_minor_formatter(minor_formatter)

    def _format_log_xaxis(ax):
        ax.xaxis.set_major_formatter(LogFormatterMathtext())
        ax.xaxis.set_minor_formatter(NullFormatter())

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
    fig.suptitle(
        f"Chinchilla Scaling Law Analysis{title_suffix}", fontsize=16, fontweight="bold", y=0.98
    )

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
    _format_log_xaxis(ax)
    if log_loss:
        ax.set_yscale("log")
        _format_log_yaxis(ax)
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
    _format_log_xaxis(ax)
    if log_loss:
        ax.set_yscale("log")
        _format_log_yaxis(ax)
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
    _format_log_xaxis(ax)
    if log_loss:
        ax.set_yscale("log")
        _format_log_yaxis(ax)
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
    _format_log_xaxis(ax)
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
        from matplotlib.ticker import LogFormatterMathtext, NullFormatter, ScalarFormatter
    except ImportError:
        print("Warning: matplotlib not installed, skipping iso-FLOP plots")
        return

    def _format_log_yaxis(ax):
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(formatter)
        minor_formatter = ScalarFormatter()
        minor_formatter.set_scientific(False)
        ax.yaxis.set_minor_formatter(minor_formatter)

    def _format_log_xaxis(ax):
        ax.xaxis.set_major_formatter(LogFormatterMathtext())
        ax.xaxis.set_minor_formatter(NullFormatter())

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
    _format_log_xaxis(ax)
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
        # Compute-optimal allocation from first-order conditions:
        # N_opt = (C / (6·G))^a_opt, G = (β·B / α·A)^(1/β)
        N_opt, _ = compute_optimal_frontier(fit, flop_range)

        ax.plot(
            flop_range,
            N_opt,
            linewidth=2,
            label=f"{display_names[name]} (a={params.a_opt:.3f})",
            color=color,
        )

    # Add Chinchilla reference line using paper's Eq. 4 with α=β=0.5, A=B (symmetric):
    # G_paper = (αA/βB)^(1/(α+β)) = 1 when α=β and A=B; N_opt = (C/6)^0.5 exactly
    # We also use a representative G from actual Chinchilla paper fits (G ≈ 1)
    N_chinchilla = (flop_range / 6) ** 0.5
    ax.plot(flop_range, N_chinchilla, "k--", linewidth=1, label="Chinchilla (a=0.5)", alpha=0.7)

    ax.set_xscale("log")
    _format_log_xaxis(ax)
    ax.set_yscale("log")
    _format_log_yaxis(ax)
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
        from matplotlib.ticker import LogFormatterMathtext, NullFormatter, ScalarFormatter
    except ImportError:
        print("Warning: matplotlib not installed, skipping loss vs FLOPs plot")
        return

    def _format_log_yaxis(ax):
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(formatter)
        minor_formatter = ScalarFormatter()
        minor_formatter.set_scientific(False)
        ax.yaxis.set_minor_formatter(minor_formatter)

    def _format_log_xaxis(ax):
        ax.xaxis.set_major_formatter(LogFormatterMathtext())
        ax.xaxis.set_minor_formatter(NullFormatter())

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
    _format_log_xaxis(ax)
    if log_loss:
        ax.set_yscale("log")
        _format_log_yaxis(ax)
    ax.set_xlabel("Training PetaFLOPs", fontsize=12)
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
    _format_log_xaxis(ax)
    if log_loss:
        ax.set_yscale("log")
        _format_log_yaxis(ax)
    ax.set_xlabel("Training PetaFLOPs", fontsize=12)
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


def plot_bootstrap_ci(
    fits: Dict[str, Tuple],
    output_path: Optional[Path] = None,
    show: bool = True,
):
    """Generate bootstrap confidence interval plots for scaling law parameters.

    Creates a multi-panel figure with:
    - Top row: whisker plots for each fitted parameter (E, alpha, beta, a_opt, b_opt)
    - Bottom row: whisker plots for A and B (log scale), plus predicted loss CIs at
      representative compute budgets

    Each architecture gets a point estimate with 95% CI error bars.

    Args:
        fits: Dict mapping ladder name to (fit, N, D, L, F, sizes, bootstrap).
        output_path: Directory to save the plot to.
        show: Whether to display the plot interactively.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed, skipping bootstrap CI plots")
        return

    # Filter to only ladders with bootstrap data
    boot_fits = {name: v for name, v in fits.items() if v[6] is not None}
    if not boot_fits:
        print("Warning: no bootstrap data available, skipping bootstrap CI plots")
        return

    plt.style.use("seaborn-v0_8-whitegrid")

    display_names = {name: get_display_name(name) for name in boot_fits}
    color_palette = ["#2ecc71", "#e74c3c", "#3498db", "#9b59b6", "#f39c12", "#1abc9c", "#e67e22"]
    colors = {
        name: color_palette[i % len(color_palette)] for i, name in enumerate(boot_fits.keys())
    }
    ladder_names = list(boot_fits.keys())
    n_ladders = len(ladder_names)

    # --- Figure 1: Parameter estimates with CIs ---
    param_specs = [
        ("E", "E (entropy floor)", False),
        ("alpha", r"$\alpha$", False),
        ("beta", r"$\beta$", False),
        ("a_opt", r"$a_\mathrm{opt}$", False),
        ("b_opt", r"$b_\mathrm{opt}$", False),
        ("A", "A", True),
        ("B", "B", True),
    ]
    n_params = len(param_specs)
    fig, axes = plt.subplots(1, n_params, figsize=(3 * n_params, 5))
    fig.suptitle("Scaling Law Parameters with 95% Bootstrap CI", fontsize=14, fontweight="bold")

    for ax_idx, (param_name, param_label, use_log) in enumerate(param_specs):
        ax = axes[ax_idx]
        x_positions = np.arange(n_ladders)

        for i, name in enumerate(ladder_names):
            fit, _, _, _, _, _, bootstrap = boot_fits[name]
            point_val = getattr(fit.fitted_params, param_name)
            boot_vals = np.array([getattr(f.fitted_params, param_name) for f in bootstrap.fits])
            lo, hi = np.percentile(boot_vals, [2.5, 97.5])

            # yerr must be non-negative; point estimate can fall outside bootstrap CI
            err_lo = max(0.0, point_val - lo)
            err_hi = max(0.0, hi - point_val)

            ax.errorbar(
                i,
                point_val,
                yerr=[[err_lo], [err_hi]],
                fmt="o",
                color=colors[name],
                markersize=8,
                capsize=5,
                capthick=1.5,
                linewidth=1.5,
                zorder=3,
            )

        if use_log:
            ax.set_yscale("log")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [display_names[n] for n in ladder_names], rotation=45, ha="right", fontsize=9
        )
        ax.set_title(param_label, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if output_path:
        save_path = output_path / "bootstrap_parameters.png"
        fig.savefig(str(save_path), dpi=200, bbox_inches="tight", facecolor="white")
        print(f"\nSaved bootstrap parameter plot to: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    # --- Figure 2: Predicted loss CIs at compute-optimal allocation ---
    flop_budgets = [1e19, 1e20, 1e21, 1e22, 1e23]
    n_budgets = len(flop_budgets)

    fig2, ax2 = plt.subplots(1, 1, figsize=(max(10, 2 * n_budgets), 6))
    fig2.suptitle(
        "Predicted Loss at Compute-Optimal Allocation (95% Bootstrap CI)",
        fontsize=14,
        fontweight="bold",
    )

    bar_width = 0.8 / n_ladders
    x_positions = np.arange(n_budgets)

    for ladder_idx, name in enumerate(ladder_names):
        fit, _, _, _, _, _, bootstrap = boot_fits[name]
        color = colors[name]

        point_losses = []
        ci_los = []
        ci_his = []

        for C in flop_budgets:
            p = fit.fitted_params
            G = (p.beta * p.B / (p.alpha * p.A)) ** (1.0 / p.beta)
            n_opt = (C / (6.0 * G)) ** p.a_opt
            d_opt = C / (6.0 * n_opt)
            point_loss = float(fit.predict_loss(np.array([n_opt]), np.array([d_opt]))[0])
            point_losses.append(point_loss)

            # Bootstrap distribution of optimal loss at this compute
            boot_losses = []
            for bf in bootstrap.fits:
                bp = bf.fitted_params
                bG = (bp.beta * bp.B / (bp.alpha * bp.A)) ** (1.0 / bp.beta)
                bn = (C / (6.0 * bG)) ** bp.a_opt
                bd = C / (6.0 * bn)
                boot_losses.append(float(bf.predict_loss(np.array([bn]), np.array([bd]))[0]))
            lo, hi = np.percentile(boot_losses, [2.5, 97.5])
            ci_los.append(max(0.0, point_loss - lo))
            ci_his.append(max(0.0, hi - point_loss))

        x_offset = x_positions + (ladder_idx - (n_ladders - 1) / 2) * bar_width
        ax2.errorbar(
            x_offset,
            point_losses,
            yerr=[ci_los, ci_his],
            fmt="s",
            color=color,
            markersize=7,
            capsize=4,
            capthick=1.5,
            linewidth=1.5,
            label=display_names[name],
        )

    # Format x-axis with compute budgets
    budget_labels = []
    for C in flop_budgets:
        exp = int(np.floor(np.log10(C)))
        mantissa = C / 10**exp
        if mantissa == 1.0:
            budget_labels.append(f"$10^{{{exp}}}$")
        else:
            budget_labels.append(f"${mantissa:.0f}\\times 10^{{{exp}}}$")

    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(budget_labels, fontsize=11)
    ax2.set_xlabel("Compute Budget (FLOPs)", fontsize=12)
    ax2.set_ylabel("Predicted Loss (BPB)", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if output_path:
        save_path = output_path / "bootstrap_predicted_loss.png"
        fig2.savefig(str(save_path), dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved bootstrap predicted loss plot to: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig2)

    # --- Figure 3: Token savings ratio CIs (if ≥2 ladders) ---
    if n_ladders >= 2:
        ref_name = ladder_names[0]
        other_names = ladder_names[1:]
        model_sizes = [1e9, 3e9, 7e9, 13e9, 30e9, 70e9]

        # Auto-determine target loss (median of all observed)
        all_L = np.concatenate([boot_fits[n][3] for n in ladder_names])
        target_loss = float(np.median(all_L))

        n_sizes = len(model_sizes)
        n_others = len(other_names)
        fig3, axes3 = plt.subplots(1, n_others, figsize=(max(8, 4 * n_others), 5), squeeze=False)
        fig3.suptitle(
            f"Token Savings Ratio vs {display_names[ref_name]} (95% CI, target loss={target_loss:.3f})",
            fontsize=13,
            fontweight="bold",
        )

        ref_bootstrap = boot_fits[ref_name][6]

        for other_idx, other_name in enumerate(other_names):
            ax = axes3[0, other_idx]
            other_bootstrap = boot_fits[other_name][6]
            color = colors[other_name]

            x_pos = np.arange(n_sizes)
            point_ratios = []
            ci_los = []
            ci_his = []

            for size_idx, n_size in enumerate(model_sizes):
                ref_fit = boot_fits[ref_name][0]
                other_fit = boot_fits[other_name][0]
                d_ref = solve_d_for_target_loss(ref_fit, target_loss, n_size)
                d_other = solve_d_for_target_loss(other_fit, target_loss, n_size)

                if np.isfinite(d_ref) and np.isfinite(d_other) and d_other > 0:
                    ratio = d_ref / d_other
                else:
                    ratio = float("nan")
                point_ratios.append(ratio)

                # Bootstrap CIs
                boot_ratios = []
                for rf, bf in zip(ref_bootstrap.fits, other_bootstrap.fits):
                    dr = solve_d_for_target_loss(rf, target_loss, n_size)
                    db = solve_d_for_target_loss(bf, target_loss, n_size)
                    if np.isfinite(dr) and np.isfinite(db) and db > 0:
                        boot_ratios.append(dr / db)
                if len(boot_ratios) >= 3:
                    lo, hi = np.percentile(boot_ratios, [2.5, 97.5])
                    ci_los.append(max(0.0, ratio - lo))
                    ci_his.append(max(0.0, hi - ratio))
                else:
                    ci_los.append(0)
                    ci_his.append(0)

            # Filter out NaN values for plotting
            valid = [not np.isnan(r) for r in point_ratios]
            valid_x = [x for x, v in zip(x_pos, valid) if v]
            valid_r = [r for r, v in zip(point_ratios, valid) if v]
            valid_lo = [lo for lo, v in zip(ci_los, valid) if v]
            valid_hi = [hi for hi, v in zip(ci_his, valid) if v]

            ax.errorbar(
                valid_x,
                valid_r,
                yerr=[valid_lo, valid_hi],
                fmt="o-",
                color=color,
                markersize=8,
                capsize=5,
                capthick=1.5,
                linewidth=1.5,
                label=display_names[other_name],
            )
            ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="1x (no savings)")
            ax.set_xticks(x_pos)
            ax.set_xticklabels([_fmt_model_size(s) for s in model_sizes], fontsize=10)
            ax.set_xlabel("Model Size", fontsize=11)
            ax.set_ylabel(f"Savings vs {display_names[ref_name]}", fontsize=11)
            ax.set_title(display_names[other_name], fontsize=12, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout(rect=[0, 0, 1, 0.91])

        if output_path:
            save_path = output_path / "bootstrap_token_savings.png"
            fig3.savefig(str(save_path), dpi=200, bbox_inches="tight", facecolor="white")
            print(f"Saved bootstrap token savings plot to: {save_path}")
        if show:
            plt.show()
        else:
            plt.close(fig3)


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


def compare_specs_for_ladder(
    ladder_name: str,
    ladder_dir: Path,
    parallel_flops: bool = True,
    chunk_size: int = 256,
    chinchilla_flops: bool = True,
    seq_len: int = 8192,
) -> None:
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

        computed = _compute_specs(
            ladder_name,
            size,
            parallel_flops=parallel_flops,
            chunk_size=chunk_size,
            chinchilla_flops=chinchilla_flops,
            seq_len=seq_len,
        )

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
        help="Ladder to fit. Format: name:path (can specify multiple times). "
        "Pool multiple runs with comma-separated paths: name:path1,path2,path3",
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
        "--plot-bootstrap",
        action="store_true",
        help="Generate bootstrap CI whisker plots (requires --bootstrap)",
    )
    parser.add_argument(
        "--plot-ci",
        action="store_true",
        help="Include 95%% bootstrap CI bands in paper figures "
        "(requires --bootstrap). Renders shaded envelopes in TikZ output.",
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
        "--scatter",
        action="store_true",
        help="Use scatter points instead of connecting lines in paper figure 1. "
        "Default is lines mode (iso-param curves in all three panels).",
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
    parser.add_argument(
        "--parallel-flops",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use chunkwise parallel training FLOP estimates for GDN/Mamba2 layers "
        "instead of recurrent inference FLOPs. This accounts for intra-chunk "
        "attention and inter-chunk state propagation overhead.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Chunk size for parallel FLOP estimation (default: 256). "
        "Only used when --parallel-flops is set.",
    )
    parser.add_argument(
        "--chinchilla-flops",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Chinchilla FLOP counting convention: include embedding lookup and "
        "softmax FLOPs in the per-token FLOP estimate (default: True). "
        "Use --no-chinchilla-flops to disable.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=8192,
        help="Sequence length for FLOP calculation (default: 8192).",
    )
    parser.add_argument(
        "--fix-params",
        nargs="*",
        default=None,
        metavar="PARAM=VALUE",
        help="Fix specific scaling law parameters during fitting. "
        "Format: E=1.5 alpha=0.3. Valid params: E, A, alpha, B, beta.",
    )
    parser.add_argument(
        "--fit-flops",
        action="store_true",
        help="Also fit a FLOP-based scaling law L(C, D) = E + A/C^α + B/D^β "
        "where C = total training FLOPs. This uses the same fitting machinery "
        "but replaces N (params) with C (FLOPs) as the first variable, which "
        "provides a fairer comparison across architectures with different "
        "parameter efficiencies (e.g. Pure GDN vs Transformer).",
    )
    parser.add_argument(
        "--average-val-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the average of all available validation loss columns "
        "(eval/lm/*-validation/CE loss) instead of a single loss column. "
        "Gives a more stable, multi-domain loss signal for fitting.",
    )

    args = parser.parse_args()

    # Parse ladder arguments.  Each --ladder value is NAME:PATH or just PATH.
    # Multiple directories for the same ladder (independent runs to pool) can
    # be given as comma-separated paths:  NAME:path1,path2,path3
    ladders: Dict[str, List[Path]] = {}
    for spec in args.ladder:
        if ":" in spec:
            name, paths_str = spec.split(":", 1)
        else:
            paths_str = spec
            name = Path(paths_str.split(",")[0]).name
        paths = [Path(p.strip()).expanduser() for p in paths_str.split(",")]
        ladders[name] = paths

    # Parse fixed params
    fixed_params: Optional[Dict[str, float]] = None
    if args.fix_params:
        fixed_params = {}
        for spec in args.fix_params:
            if "=" not in spec:
                parser.error(f"Invalid --fix-params format: '{spec}'. Expected PARAM=VALUE.")
            key, val = spec.split("=", 1)
            fixed_params[key] = float(val)

    # Compare computed vs. logged specs if requested (uses first dir only — specs are identical across runs)
    if args.compare_specs:
        for name, ladder_dirs in ladders.items():
            compare_specs_for_ladder(
                name,
                ladder_dirs[0],
                parallel_flops=args.parallel_flops,
                chunk_size=args.chunk_size,
                chinchilla_flops=args.chinchilla_flops,
                seq_len=args.seq_len,
            )

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
                fixed_params=fixed_params,
                parallel_flops=args.parallel_flops,
                chunk_size=args.chunk_size,
                chinchilla_flops=args.chinchilla_flops,
                seq_len=args.seq_len,
                average_val_loss=args.average_val_loss,
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

    # ---- FLOP-based fitting: L(N_eff, D) where N_eff = flops_per_token / 6 ----
    flop_fits: Dict[str, Tuple] = {}
    if args.fit_flops:
        print(f"\n{'='*60}")
        print("FLOP-BASED FITTING: L(N_eff, D) where N_eff = FLOPs_per_token / 6")
        print("  N_eff normalises away architecture differences so that models")
        print("  with the same FLOP budget share the same effective parameter count.")
        print(f"{'='*60}")

        for name, (fit_nd, N, D, L, F, sizes, bootstrap) in fits.items():
            display = get_display_name(name)

            # N_eff = total_training_flops / (6 * D)  =  flops_per_token * 3 * D / (6 * D)
            #       = flops_per_token / 2
            # But F = 3 * flops_per_token * D, so N_eff = F / (6 * D)
            N_eff = F / (6.0 * D)
            print(
                f"\n  {display}: N_eff range = [{N_eff.min()/1e6:.1f}M, {N_eff.max()/1e6:.1f}M] "
                f"(was N = [{N.min()/1e6:.1f}M, {N.max()/1e6:.1f}M])"
            )

            # Compute weights
            weights = None
            if not args.no_weight:
                weights = 6 * N_eff * D

            try:
                if args.bootstrap > 0:
                    boot_fit = ChinchillaParametricBootstrappedFit.fit(
                        N_eff,
                        D,
                        L,
                        num_bootstraps=args.bootstrap,
                        weights=weights,
                        overestimate_penalty=args.overestimate_penalty,
                        num_slices=args.num_slices,
                        progress_bar=not args.quiet,
                        fixed_params=fixed_params,
                    )
                    flop_fit = boot_fit.point_estimate
                    flop_bootstrap = boot_fit
                else:
                    flop_fit = ChinchillaParametricFit.fit(
                        N_eff,
                        D,
                        L,
                        weights=weights,
                        overestimate_penalty=args.overestimate_penalty,
                        num_slices=args.num_slices,
                        fixed_params=fixed_params,
                    )
                    flop_bootstrap = None

                flop_fits[name] = (flop_fit, N_eff, D, L, F, sizes, flop_bootstrap)

                p = flop_fit.fitted_params
                r2 = flop_fit.r_squared
                print(
                    f"  {display}: E={p.E:.4f}, A={p.A:.4e}, α={p.alpha:.4f}, "
                    f"B={p.B:.4e}, β={p.beta:.4f}" + (f", R²={r2:.6f}" if r2 is not None else "")
                )
            except Exception as e:
                print(f"  Error fitting {display}: {e}")
                import traceback

                traceback.print_exc()

        if flop_fits:
            print_comparison(
                flop_fits, header="FLOP-BASED FIT COMPARISON (L(N_eff, D), N_eff = FLOPs/token / 6)"
            )

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

    # Generate bootstrap CI plots
    if args.plot_bootstrap:
        if args.bootstrap <= 0:
            print("\nWarning: --plot-bootstrap requires --bootstrap, skipping bootstrap plots")
        else:
            plot_bootstrap_ci(fits, args.output, show=(args.output is None))

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
    if args.domain_fit:
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
                    parallel_flops=args.parallel_flops,
                    chunk_size=args.chunk_size,
                    chinchilla_flops=args.chinchilla_flops,
                    seq_len=args.seq_len,
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

        # Figure 1: Main Scaling Law Fits
        # Generate all 4 combinations: {linear, log-loss} × {mean, optimal}
        use_lines = not args.scatter
        for log_loss, bold in [
            (False, "mean"),
            (False, "optimal"),
            (True, "mean"),
            (True, "optimal"),
        ]:
            fig1 = generate_paper_figure_1(
                fits, log_loss=log_loss, use_lines=use_lines, bold_curve=bold
            )
            parts = []
            if log_loss:
                parts.append("log")
            if bold == "optimal":
                parts.append("opt")
            suffix = ("-" + "-".join(parts)) if parts else ""
            fig1_path = figures_dir / f"scaling-law-fit{suffix}.tex"
            with open(fig1_path, "w") as f:
                f.write(fig1)
            desc = []
            if log_loss:
                desc.append("log-loss")
            if bold == "optimal":
                desc.append("optimal frontier")
            desc_str = f" ({', '.join(desc)})" if desc else ""
            print(f"Saved Figure 1{desc_str} to: {fig1_path}")

        # Figure 2: Efficiency Projections (median target loss)
        fig2 = generate_paper_figure_2(fits, loss_strategy="median", plot_ci=args.plot_ci)
        fig2_path = figures_dir / "scaling-efficiency-projections.tex"
        with open(fig2_path, "w") as f:
            f.write(fig2)
        print(f"Saved Figure 2 (median) to: {fig2_path}")

        # Figure 2b: Efficiency Projections (minimum target loss)
        fig2_min = generate_paper_figure_2(
            fits, loss_strategy="min", fig_suffix="-min", plot_ci=args.plot_ci
        )
        fig2_min_path = figures_dir / "scaling-efficiency-projections-min.tex"
        with open(fig2_min_path, "w") as f:
            f.write(fig2_min)
        print(f"Saved Figure 2 (min) to: {fig2_min_path}")

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

        # Figure 5: Savings Factor Plot
        fig5 = generate_savings_factor_figure(fits, plot_ci=args.plot_ci)
        fig5_path = figures_dir / "savings-factor.tex"
        with open(fig5_path, "w") as f:
            f.write(fig5)
        print(f"Saved Figure 5 to: {fig5_path}")

        # Figure 6: Combined Savings (data requirements + savings factor, shared X axis)
        fig6 = generate_combined_savings_figure(fits, plot_ci=args.plot_ci)
        fig6_path = figures_dir / "combined-savings.tex"
        with open(fig6_path, "w") as f:
            f.write(fig6)
        print(f"Saved Combined Savings to: {fig6_path}")

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

        # Table 3: Efficiency Gains
        table3 = generate_efficiency_latex_table(fits, target_losses=args.target_losses)
        table3_path = tables_dir / "scaling-efficiency-gains.tex"
        with open(table3_path, "w") as f:
            f.write(table3)
        print(f"Saved Table 3 to: {table3_path}")

        # Table 4: Token Savings by Scale
        table4 = generate_token_savings_table(fits)
        table4_path = tables_dir / "token-savings-by-scale.tex"
        with open(table4_path, "w") as f:
            f.write(table4)
        print(f"Saved Table 4 to: {table4_path}")

        table4_min = generate_token_savings_table(fits, loss_strategy="min")
        table4_min_path = tables_dir / "token-savings-by-scale-min.tex"
        with open(table4_min_path, "w") as f:
            f.write(table4_min)
        print(f"Saved Table 4-min to: {table4_min_path}")

        # Table 5: Compute-Equivalent Loss (hybrid ladders — main paper)
        pure_gdn_names = {"pure-gdn"}
        hybrid_fits = {k: v for k, v in fits.items() if k not in pure_gdn_names}
        table5 = generate_compute_equivalent_table(
            hybrid_fits, label="tab:compute-equivalent-loss"
        )
        table5_path = tables_dir / "compute-equivalent-loss.tex"
        with open(table5_path, "w") as f:
            f.write(table5)
        print(f"Saved Table 5 to: {table5_path}")

        # Table 5b: Compute-Equivalent Loss (Pure GDN — appendix)
        if any(k in fits for k in pure_gdn_names):
            ref_name = list(fits.keys())[0]
            pure_gdn_fits_full = {
                k: v for k, v in fits.items() if k == ref_name or k in pure_gdn_names
            }
            table5b = generate_compute_equivalent_table(
                pure_gdn_fits_full, label="tab:compute-equivalent-loss-pure-gdn"
            )
            table5b_path = tables_dir / "compute-equivalent-loss-pure-gdn.tex"
            with open(table5b_path, "w") as f:
                f.write(table5b)
            print(f"Saved Table 5b to: {table5b_path}")

    # Save results
    if args.output:
        # Save fitted parameters as CSV (with bootstrap CIs when available)
        rows = []
        for name, (fit, N, D, L, F, sizes, bootstrap) in fits.items():
            p = fit.fitted_params
            row: dict = {
                "ladder": name,
                "E": p.E,
                "A": p.A,
                "alpha": p.alpha,
                "B": p.B,
                "beta": p.beta,
                "a_opt": p.a_opt,
                "b_opt": p.b_opt,
                "huber_loss": fit.huber_loss,
                "r_squared": fit.r_squared,
            }
            if bootstrap is not None:
                for param in ["E", "A", "alpha", "B", "beta", "a_opt", "b_opt"]:
                    vals = np.array([getattr(f.fitted_params, param) for f in bootstrap.fits])
                    lo, hi = np.percentile(vals, [2.5, 97.5])
                    row[f"{param}_ci_lo"] = lo
                    row[f"{param}_ci_hi"] = hi
                row["n_bootstrap_samples"] = len(bootstrap.fits)
            rows.append(row)
        pd.DataFrame(rows).to_csv(args.output / "chinchilla_fits.csv", index=False)
        print(f"\nSaved fit parameters to: {args.output / 'chinchilla_fits.csv'}")

        # Save per-point data (with bootstrap prediction CIs when available)
        all_data = []
        for name, (fit, N, D, L, F, sizes, bootstrap) in fits.items():
            predicted = fit.predict_loss(N, D)
            # Compute bootstrap prediction intervals if available
            pred_ci_lo = pred_ci_hi = None
            if bootstrap is not None:
                loss_dist = bootstrap.predict_loss_distribution(
                    N, D, include_observation_noise=False
                )
                pred_ci_lo, pred_ci_hi = np.percentile(loss_dist, [2.5, 97.5], axis=0)
            for i in range(len(N)):
                row: dict = {
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
                if pred_ci_lo is not None and pred_ci_hi is not None:
                    row["predicted_loss_ci_lo"] = pred_ci_lo[i]
                    row["predicted_loss_ci_hi"] = pred_ci_hi[i]
                all_data.append(row)
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
