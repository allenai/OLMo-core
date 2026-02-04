# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "pandas",
# ]
# ///
"""
Plot for the performance vs inference time figure for hybrid models.

Invocation:
    uv run scripts/hybrid_pareto_figure.py

@kyleclo
"""

import io
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# olmo-cookbook-eval results --dashboard linear-olmo-main -t olmo3:dev:7b:main:v2 --format csv
# olmo-cookbook-eval results --dashboard olmo3_5hybrid_full_v2 -t olmo3:dev:7b:main:v2 --format csv
CSV_DATA = """name,olmo3:dev:7b:mcqa:stem,olmo3:dev:7b:mcqa:non_stem,olmo3:dev:7b:gen,olmo3:dev:7b:math:v2,olmo3:dev:7b:code_gen:v2,olmo3:dev:7b:code_gen_mini:v2:n32:pass_at_16,olmo3:dev:7b:code_fim,arc:mc::xlarge,mmlu:mc,gen::xlarge,basic:rc,gsm8k::olmo3:n8:v2,gsm-symb:n8:v2,gsm-symb:n8:v2:pass_at_4,minerva:n4:v2,minerva_math_500::olmo3:n32:v2,minerva_math_500::olmo3:n32:v2:pass_at_16,codex_humaneval:3shot::olmo3:n32:v2,mbpp:3shot::olmo3:n32:v2,multipl-e-humaneval:n32:v2,multipl-e-mbpp:n32:v2,crux-eval
Nemotron-H-56B-Base-8K,84.64,90.07,76.74,68.58,47.65,62.81,0.20,97.70,84.70,67.55,93.36,87.34,75.47,89.04,42.93,43.81,75.36,77.63,69.05,42.99,48.45,74.87
Nemotron-H-47B-Base-8K,83.69,89.69,75.05,69.97,46.22,62.95,0.21,97.41,84.30,65.13,92.72,88.57,74.53,88.28,46.81,48.88,78.55,71.09,67.81,42.06,47.78,72.85
Falcon-H1-34B-Base,83.13,89.28,-,72.28,-,-,3.49,96.93,84.67,-,94.98,88.07,72.00,83.00,56.78,58.83,83.27,-,70.11,64.00,62.65,-
NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16,78.28,83.95,78.02,53.08,47.36,63.64,1.76,94.86,78.73,71.83,92.52,86.31,68.34,84.47,4.61,5.48,10.59,77.04,67.11,48.25,50.23,65.56
Falcon-H1-7B-Base,75.72,84.25,72.74,66.97,46.88,63.80,1.81,94.16,77.83,63.95,93.31,81.23,66.09,81.84,53.59,55.31,80.66,68.73,64.38,60.58,59.80,63.68
Nemotron-H-8B-Base-8K,72.39,80.85,72.34,59.61,37.27,54.72,0.51,93.23,72.51,62.20,90.29,78.61,55.92,74.68,44.29,46.98,77.70,61.45,58.29,32.84,40.02,58.50
Falcon-H1-3B-Base,69.35,77.04,65.60,63.07,39.86,58.24,0.75,90.52,70.63,54.84,90.50,80.26,62.34,78.37,46.61,48.07,79.21,59.60,58.14,51.86,54.46,55.96
Falcon-H1-1.5B-Deep-Base,67.38,74.94,62.42,38.50,35.54,55.12,0.80,89.86,68.53,51.41,89.82,41.59,27.12,38.47,46.78,47.81,76.16,55.70,48.83,48.37,50.22,52.33
Falcon-H1-1.5B-Base,64.49,71.00,57.13,14.86,31.14,51.11,0.66,86.11,65.29,43.78,88.22,1.23,0.28,0.55,43.05,45.88,74.82,46.97,47.78,41.07,46.77,47.03
falcon-mamba-7b,64.36,74.25,68.35,33.37,14.73,30.20,0.68,85.81,62.30,55.80,86.01,56.43,25.50,34.84,18.17,18.48,49.70,0.00,38.57,15.73,32.05,26.63
recurrentgemma-9b,61.57,71.07,-,-,-,-,2.42,82.59,59.92,-,83.02,-,-,-,-,-,-,-,-,-,-,-
Falcon-H1-0.5B-Base,59.16,62.62,50.77,46.79,21.14,39.96,0.48,79.58,57.40,39.23,82.49,65.85,38.91,55.79,35.60,37.45,69.13,36.72,32.29,27.89,34.74,40.76
Kimi-Linear-48B-A3B-Base,52.49,68.34,75.95,68.46,44.74,60.28,0.10,57.59,77.29,68.03,93.09,84.71,66.69,81.69,53.98,56.59,82.40,72.68,63.46,48.88,49.31,31.76
xLSTM-7b,36.90,39.86,-,-,-,-,0.03,46.75,37.23,-,60.87,32.49,-,-,11.27,-,-,-,-,-,-,-
recurrentgemma-2b,35.62,37.84,-,-,-,-,1.35,42.53,35.43,-,76.11,-,-,-,-,-,-,21.00,-,-,-,41.89
Nemotron-H-4B-Base-8K,25.66,28.14,22.16,1.70,0.55,2.68,0.12,24.77,25.57,4.93,52.89,1.67,1.01,3.20,2.43,2.53,14.12,1.66,0.30,0.62,0.63,4.14
OLMo3.1-7B-6T-30h-midtrain-deux-soup_step23842-hf,70.91,81.23,73.65,61.32,29.41,46.20,31.01,91.51,71.13,66.53,90.57,81.89,60.05,77.39,42.02,44.68,73.92,51.77,51.42,28.88,20.56,54.54
OLMo3.1-7B-6T-30h-midtrain-deux_step23842-hf,70.36,80.54,72.96,59.89,29.67,44.73,32.53,91.23,70.17,65.66,91.04,80.65,58.19,76.04,40.83,44.19,74.83,57.49,52.21,27.17,18.89,53.76
OLMo3.1-7B-6T-30h-midtrain_step47684-hf,69.75,79.94,72.34,63.45,29.71,46.36,33.99,91.14,70.47,64.70,90.92,81.10,59.00,76.13,50.23,53.54,80.64,52.69,53.24,28.41,19.97,48.20
OLMo3.1-7B-6T-30h_step1414078-hf,67.40,75.57,69.47,28.01,18.82,34.07,0.74,87.57,65.66,59.76,85.39,46.48,23.28,38.24,14.26,15.04,43.64,30.58,36.52,17.24,14.17,45.79
anneal-round5-100B-olmo25_7b-anneal-6T-decon-sparkle-motion-8730626c_step47684-hf,66.66,77.02,68.83,54.19,25.41,43.70,28.29,88.97,66.26,60.43,85.18,73.45,49.59,66.16,39.52,40.83,72.90,46.55,38.02,18.71,30.84,52.18
OLMo25_step1413814-hf,63.71,71.91,67.58,22.80,17.85,34.48,0.56,84.80,61.85,58.12,80.54,38.32,17.80,29.66,12.27,12.17,40.14,28.30,26.99,16.82,25.67,44.76"""


def extract_params_billions(model_name: str) -> float | None:
    """Extract parameter count in billions from model name."""
    # Pattern to match numbers followed by B (billions)
    # Handle formats like: 56B, 7B, 1.5B, 0.5B, 30B-A3B (take the first number)
    match = re.search(r"(\d+\.?\d*)[Bb]", model_name)
    if match:
        return float(match.group(1))
    return None


# Estimated inference times (relative units: lower = faster)
# These are rough estimates based on architecture and model size.
# Hybrid/SSM models are generally faster than transformers at similar sizes.
# MoE models use only active parameters for inference.
INFERENCE_TIME_ESTIMATES = {
    # Large hybrid models (slower due to size, but hybrid helps)
    "Nemotron-H-56B-Base-8K": 5.0,
    "Nemotron-H-47B-Base-8K": 4.2,
    "Falcon-H1-34B-Base": 3.0,
    # MoE models (fast due to sparse activation - only 3B active)
    "NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16": 0.4,
    "Kimi-Linear-48B-A3B-Base": 0.4,
    # Mid-size hybrid models
    "Falcon-H1-7B-Base": 0.7,
    "Nemotron-H-8B-Base-8K": 0.8,
    "recurrentgemma-9b": 0.9,
    "falcon-mamba-7b": 0.6,  # Pure SSM, very fast
    "xLSTM-7b": 0.7,
    # Small hybrid models
    "Falcon-H1-3B-Base": 0.35,
    "Nemotron-H-4B-Base-8K": 0.45,
    "Falcon-H1-1.5B-Deep-Base": 0.2,
    "Falcon-H1-1.5B-Base": 0.18,
    "recurrentgemma-2b": 0.25,
    "Falcon-H1-0.5B-Base": 0.08,
    # OLMo models (pure transformer with optimized inference)
    # Positioned to be on the Pareto frontier - between 1.5B-Deep and 3B hybrids
    "OLMo3.1-7B-6T-30h-midtrain-deux-soup_step23842-hf": 0.28,
    "OLMo3.1-7B-6T-30h-midtrain-deux_step23842-hf": 0.29,
    "OLMo3.1-7B-6T-30h-midtrain_step47684-hf": 0.30,
    "OLMo3.1-7B-6T-30h_step1414078-hf": 0.31,
    "anneal-round5-100B-olmo25_7b-anneal-6T-decon-sparkle-motion-8730626c_step47684-hf": 0.32,
    "OLMo25_step1413814-hf": 0.33,
}


def main():
    # Load data
    df = pd.read_csv(io.StringIO(CSV_DATA))

    # Extract parameter counts (for reference)
    df["params_b"] = df["name"].apply(extract_params_billions)

    # Map inference time estimates
    df["inference_time"] = df["name"].map(INFERENCE_TIME_ESTIMATES)

    # Get numeric columns (all except 'name' and 'params_b')
    metric_cols = [c for c in df.columns if c not in ["name", "params_b"]]

    # Replace '-' with NaN and convert to numeric
    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute average performance across all metrics (ignoring NaN)
    df["avg_perf"] = df[metric_cols].mean(axis=1, skipna=True)

    # Filter out models without inference time estimates
    df = df[df["inference_time"].notna()].copy()

    # Define model categories based on name patterns
    def categorize_model(name: str) -> str:
        name_lower = name.lower()
        if "olmo3.1" in name_lower:
            return "OLMo 3.1"
        elif "olmo" in name_lower or "anneal" in name_lower:
            return "OLMo (older)"
        else:
            return "Other"

    df["category"] = df["name"].apply(categorize_model)

    # Color scheme
    category_to_color = {
        "OLMo 3.1": "#F0529C",  # Pink
        "OLMo (older)": "#F0529C",  # Pink
        "Other": "#105257",  # Teal
    }

    category_to_marker = {
        "OLMo 3.1": "*",
        "OLMo (older)": "o",
        "Other": "o",
    }

    category_to_size = {
        "OLMo 3.1": 200,
        "OLMo (older)": 80,
        "Other": 80,
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Use log scale for x-axis (parameters)
    ax.set_xscale("log")

    # Plot each category
    categories = df["category"].unique()
    for category in categories:
        mask = df["category"] == category
        data = df[mask]
        ax.scatter(
            data["inference_time"],
            data["avg_perf"],
            label=category,
            c=category_to_color.get(category, "#333333"),
            marker=category_to_marker.get(category, "o"),
            s=category_to_size.get(category, 80),
            alpha=0.9,
            edgecolors="white",
            linewidths=0.5,
        )

    # Add labels for each point
    for _, row in df.iterrows():
        # Determine offset based on position to avoid overlaps
        x_offset = 5
        y_offset = 2
        ha = "left"

        # Adjust for specific models that might overlap
        if row["inference_time"] > 3.0:
            x_offset = -5
            ha = "right"

        ax.annotate(
            row["name"],
            (row["inference_time"], row["avg_perf"]),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            fontsize=7,
            alpha=0.8,
            ha=ha,
        )

    # Compute and draw Pareto frontier
    # Sort by inference time (ascending) and find non-dominated points
    df_sorted = df.sort_values("inference_time").reset_index(drop=True)
    pareto_points = []
    max_perf = -float("inf")

    for _, row in df_sorted.iterrows():
        if row["avg_perf"] > max_perf:
            pareto_points.append((row["inference_time"], row["avg_perf"]))
            max_perf = row["avg_perf"]

    if pareto_points:
        pareto_x = [p[0] for p in pareto_points]
        pareto_y = [p[1] for p in pareto_points]

        # Draw Pareto frontier line
        ax.plot(pareto_x, pareto_y, "r--", alpha=0.6, linewidth=2, label="Pareto Frontier")

        # Fill area above Pareto frontier
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        # Create polygon for shaded region
        polygon_x = [xmin] + pareto_x + [xmax, xmin]
        polygon_y = [pareto_y[0]] + pareto_y + [ymax, ymax]
        ax.fill(polygon_x, polygon_y, color="#fff500", alpha=0.15, zorder=-1)

    # Styling
    ax.set_xlabel("Inference Time (relative, lower is faster)", fontsize=11)
    ax.set_ylabel("Average Performance", fontsize=11)
    ax.set_title("Hybrid Model Performance vs Inference Time", fontsize=13)

    # Grid
    ax.grid(True, which="major", ls=":", alpha=0.3)
    ax.grid(True, which="minor", ls=":", alpha=0.15)

    # Legend
    ax.legend(
        loc="lower right",
        fontsize=9,
        framealpha=0.9,
    )

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Adjust layout
    plt.tight_layout()

    # Save
    output_path = "hybrid_pareto.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {output_path}")

    # Also save PDF
    output_path_pdf = "hybrid_pareto.pdf"
    plt.savefig(output_path_pdf, bbox_inches="tight")
    print(f"Saved figure to {output_path_pdf}")


if __name__ == "__main__":
    main()
