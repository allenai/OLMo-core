"""
Plot qnorm/knorm magnitude and comparison results from qknorm_results/.

For each model:
  - Magnitude plots: l2_norm and mean_abs over time, per-layer and averaged.
  - Comparison plots: pct_increased/decreased/same over time, per-layer and averaged.
"""

import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path("qknorm_results")
OUTPUT_DIR = Path("qknorm_results/plots")


def load_magnitudes(filepath: Path) -> pd.DataFrame:
    """Load a magnitudes TSV, skipping comment lines."""
    return pd.read_csv(filepath, sep="\t", comment="#")


def load_comparisons(filepath: Path) -> pd.DataFrame:
    """Load a comparisons TSV, skipping comment lines."""
    return pd.read_csv(filepath, sep="\t", comment="#")


def extract_layer(param_name: str) -> int:
    """Extract layer number from parameter name like blocks.12.attention.q_norm.weight."""
    m = re.search(r"blocks\.(\d+)\.", param_name)
    return int(m.group(1)) if m else -1


def extract_norm_type(param_name: str) -> str:
    """Return 'q_norm' or 'k_norm'."""
    if "q_norm" in param_name:
        return "q_norm"
    return "k_norm"


def plot_magnitudes_per_layer(df: pd.DataFrame, model_name: str, norm_type: str):
    """Plot l2_norm and mean_abs over steps for each layer, for one norm type."""
    sub = df[df["norm_type"] == norm_type].copy()
    layers = sorted(sub["layer"].unique())
    steps = sorted(sub["step"].unique())

    # Use a colormap spanning layers
    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(layers) - 1, 1)) for i in range(len(layers))]

    for metric, ylabel in [("l2_norm", "L2 Norm"), ("mean_abs", "Mean |w|")]:
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, layer in enumerate(layers):
            layer_data = sub[sub["layer"] == layer].sort_values("step")
            ax.plot(layer_data["step"], layer_data[metric], color=colors[i],
                    alpha=0.6, linewidth=0.8)

        # Add colorbar for layer index
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(layers), vmax=max(layers)))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Layer")

        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{model_name} — {norm_type} {ylabel} by layer")
        ax.set_xlim(min(steps), max(steps))

        fname = f"{model_name}_{norm_type}_{metric}_by_layer.png"
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / fname, dpi=150)
        plt.close(fig)


def plot_magnitudes_averaged(df: pd.DataFrame, model_name: str):
    """Plot average l2_norm and mean_abs across all layers, q_norm vs k_norm."""
    for metric, ylabel in [("l2_norm", "L2 Norm"), ("mean_abs", "Mean |w|")]:
        fig, ax = plt.subplots(figsize=(10, 5))

        for norm_type, color, ls in [("q_norm", "tab:blue", "-"), ("k_norm", "tab:red", "--")]:
            sub = df[df["norm_type"] == norm_type]
            avg = sub.groupby("step")[metric].mean().sort_index()
            ax.plot(avg.index, avg.values, color=color, linestyle=ls, linewidth=2, label=norm_type)

            # Also show min/max band
            lo = sub.groupby("step")[metric].min().sort_index()
            hi = sub.groupby("step")[metric].max().sort_index()
            ax.fill_between(avg.index, lo.values, hi.values, color=color, alpha=0.12)

        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{model_name} — {ylabel} averaged across layers (band = min/max)")
        ax.legend()

        fname = f"{model_name}_avg_{metric}.png"
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / fname, dpi=150)
        plt.close(fig)


def plot_comparisons_per_layer(df: pd.DataFrame, model_name: str, norm_type: str):
    """Plot pct_increased / pct_decreased / pct_same over step pairs, per layer."""
    sub = df[df["norm_type"] == norm_type].copy()
    layers = sorted(sub["layer"].unique())

    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(layers) - 1, 1)) for i in range(len(layers))]

    for metric, ylabel in [
        ("pct_increased", "% Weights Increased"),
        ("pct_decreased", "% Weights Decreased"),
        ("pct_same", "% Weights Same (within tol)"),
        ("mean_change", "Mean Change"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, layer in enumerate(layers):
            layer_data = sub[sub["layer"] == layer].sort_values("step_b")
            ax.plot(layer_data["step_b"], layer_data[metric], color=colors[i],
                    alpha=0.6, linewidth=0.8)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(layers), vmax=max(layers)))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Layer")

        ax.set_xlabel("Step (end of interval)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{model_name} — {norm_type} {ylabel} by layer")

        fname = f"{model_name}_{norm_type}_{metric}_by_layer.png"
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / fname, dpi=150)
        plt.close(fig)


def plot_comparisons_averaged(df: pd.DataFrame, model_name: str):
    """Plot averaged comparison metrics across all layers, q_norm vs k_norm."""
    for metric, ylabel in [
        ("pct_increased", "% Weights Increased"),
        ("pct_decreased", "% Weights Decreased"),
        ("pct_same", "% Weights Same (within tol)"),
        ("mean_change", "Mean Change"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))

        for norm_type, color, ls in [("q_norm", "tab:blue", "-"), ("k_norm", "tab:red", "--")]:
            sub = df[df["norm_type"] == norm_type]
            avg = sub.groupby("step_b")[metric].mean().sort_index()
            ax.plot(avg.index, avg.values, color=color, linestyle=ls, linewidth=2, label=norm_type)

            lo = sub.groupby("step_b")[metric].min().sort_index()
            hi = sub.groupby("step_b")[metric].max().sort_index()
            ax.fill_between(avg.index, lo.values, hi.values, color=color, alpha=0.12)

        ax.set_xlabel("Step (end of interval)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{model_name} — {ylabel} averaged across layers (band = min/max)")
        ax.legend()

        fname = f"{model_name}_avg_{metric}.png"
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / fname, dpi=150)
        plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Discover models from filenames: <model>_qknorm_magnitudes.tsv
    mag_files = sorted(RESULTS_DIR.glob("*_qknorm_magnitudes.tsv"))

    for mag_path in mag_files:
        model_name = mag_path.name.replace("_qknorm_magnitudes.tsv", "")
        comp_path = mag_path.parent / f"{model_name}_qknorm_comparisons.tsv"

        if not comp_path.exists():
            print(f"Skipping {model_name}: no comparisons file found")
            continue

        print(f"\nProcessing model: {model_name}")

        # Load and enrich magnitudes
        mag_df = load_magnitudes(mag_path)
        mag_df["layer"] = mag_df["parameter_name"].apply(extract_layer)
        mag_df["norm_type"] = mag_df["parameter_name"].apply(extract_norm_type)

        # Load and enrich comparisons
        comp_df = load_comparisons(comp_path)
        comp_df["layer"] = comp_df["parameter_name"].apply(extract_layer)
        comp_df["norm_type"] = comp_df["parameter_name"].apply(extract_norm_type)

        # Magnitude plots
        print("  Plotting magnitudes by layer...")
        plot_magnitudes_per_layer(mag_df, model_name, "q_norm")
        plot_magnitudes_per_layer(mag_df, model_name, "k_norm")
        print("  Plotting magnitudes averaged...")
        plot_magnitudes_averaged(mag_df, model_name)

        # Comparison plots
        print("  Plotting comparisons by layer...")
        plot_comparisons_per_layer(comp_df, model_name, "q_norm")
        plot_comparisons_per_layer(comp_df, model_name, "k_norm")
        print("  Plotting comparisons averaged...")
        plot_comparisons_averaged(comp_df, model_name)

    print(f"\nAll plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
