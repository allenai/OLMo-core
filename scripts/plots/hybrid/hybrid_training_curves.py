# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "wandb",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "pyarrow",
# ]
# ///
"""
Plot training curves (training time vs performance) for hybrid models.

Compares:
- Baseline: OLMo-3-1025-7B (stage 1 + stage 2)
- Hybrid: OLMo3.1-7B-6T-30h

Invocation:
    uv run scripts/hybrid_training_curves.py

@kyleclo
"""

import os
import sys
import wandb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict
import re

# Add olmo-cookbook to path for AI2 theme
sys.path.insert(0, "/Users/kylel/ai2/olmo-cookbook/olmo-3-plots")
from ai2_theme import apply_ai2_theme, AI2_COLORS

# Apply AI2 matplotlib theme
apply_ai2_theme()

# Initialize API with higher timeout
API_TIMEOUT = 120

# W&B run configurations
RUNS_CONFIG = {
    "OLMo 3 7B Base": [
        {"entity": "ai2-llm", "project": "Olmo-3-1025-7B", "group": "Olmo-3-1025-7B-stage-1"},
        {"entity": "ai2-llm", "project": "Olmo-3-1025-7B", "group": "Olmo-3-1025-7B-stage-2"},
    ],
    "OLMo 3.2 7B Base": [
        {"entity": "ai2-llm", "project": "linear-rnns", "group": "OLMo3.1-7B-6T-30h"},
    ],
}

# Metrics we want to extract (patterns)
METRIC_PATTERNS = {
    # MMLU MC format - CE loss and accuracy
    "mmlu_stem_ce": r"eval/downstream/mmlu_stem.*mc.*CE loss",
    "mmlu_humanities_ce": r"eval/downstream/mmlu_humanities.*mc.*CE loss",
    "mmlu_social_ce": r"eval/downstream/mmlu_social.*mc.*CE loss",
    "mmlu_other_ce": r"eval/downstream/mmlu_other.*mc.*CE loss",
    "mmlu_stem_acc": r"eval/downstream/mmlu_stem.*mc.*Accuracy",
    "mmlu_humanities_acc": r"eval/downstream/mmlu_humanities.*mc.*Accuracy",
    "mmlu_social_acc": r"eval/downstream/mmlu_social.*mc.*Accuracy",
    "mmlu_other_acc": r"eval/downstream/mmlu_other.*mc.*Accuracy",
    # BPB metrics
    "arc_challenge_bpb": r"eval/downstream/arc_challenge.*bpb.*\(BPB",
    "arc_easy_bpb": r"eval/downstream/arc_easy.*bpb.*\(BPB",
    "hellaswag_bpb": r"eval/downstream/hellaswag.*bpb.*\(BPB",
    # RC (reading comprehension) accuracy metrics
    "arc_challenge_acc": r"eval/downstream/arc_challenge.*rc.*Accuracy",
    "arc_easy_acc": r"eval/downstream/arc_easy.*rc.*Accuracy",
    "hellaswag_acc": r"eval/downstream/hellaswag.*rc.*Accuracy",
    # Language model perplexity
    "wikitext_ppl": r"eval/lm/wikitext.*PPL",
    "dolma_cc_ce": r"eval/lm/dolma_common-crawl.*CE loss",
}


def get_relevant_keys_from_run(run) -> list[str]:
    """Get the list of relevant metric keys from a run's summary (fast lookup)."""
    # Always include these
    keys = ["_step", "throughput/total tokens"]

    # Check run.summary_metrics or run.summary for available keys
    try:
        summary_keys = list(run.summary.keys())
    except Exception:
        return keys

    for key in summary_keys:
        for pattern in METRIC_PATTERNS.values():
            if re.search(pattern, key, re.IGNORECASE):
                keys.append(key)
                break

    return list(set(keys))


def fetch_run_data_streaming(
    api: wandb.Api, entity: str, project: str, group: str, use_cache: bool = True
) -> pd.DataFrame:
    """Fetch data using scan_history (streaming, memory efficient, faster).

    Caches each run individually so progress is preserved if script is interrupted.
    """
    print(f"  Fetching runs from {entity}/{project}, group: {group}", flush=True)

    try:
        runs = list(api.runs(f"{entity}/{project}", filters={"group": group}))
        print(f"    Found {len(runs)} runs", flush=True)
    except Exception as e:
        print(f"    Error listing runs: {e}", flush=True)
        return pd.DataFrame()

    all_data = []
    for run in runs:
        print(f"    Run: {run.name} (id: {run.id})", flush=True)

        # Check per-run cache first
        run_cache_path = get_run_cache_path(run.id)
        if use_cache and os.path.exists(run_cache_path):
            try:
                history = pd.read_parquet(run_cache_path)
                all_data.append(history)
                print(f"      Loaded from cache: {len(history)} rows", flush=True)
                continue
            except Exception as e:
                print(f"      Cache read failed, re-fetching: {e}", flush=True)

        try:
            # Get only the keys we care about (much faster than fetching all columns)
            keys = get_relevant_keys_from_run(run)
            print(f"      Fetching {len(keys)} keys: {keys[:5]}...", flush=True)

            if len(keys) <= 2:
                print(f"      No eval metrics found in summary", flush=True)
                continue

            # Use scan_history - streams data, much faster than history()
            rows = []
            for row in run.scan_history(keys=keys, page_size=10000):
                rows.append(row)

            if not rows:
                print(f"      No history found", flush=True)
                continue

            history = pd.DataFrame(rows)
            history["run_name"] = run.name
            history["run_id"] = run.id

            # Cache this run immediately (survives interruption)
            history.to_parquet(run_cache_path)
            print(f"      Cached to {run_cache_path}", flush=True)

            all_data.append(history)
            print(f"      Found {len(history.columns)} columns with {len(history)} rows", flush=True)

        except Exception as e:
            print(f"      Error: {e}", flush=True)
            continue

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def fetch_run_data(
    api: wandb.Api, entity: str, project: str, group: str, use_cache: bool = True
) -> pd.DataFrame:
    """Fetch data for all runs in a group."""
    return fetch_run_data_streaming(api, entity, project, group, use_cache=use_cache)


CACHE_DIR = ".wandb_cache"


def get_cache_path(model_name: str) -> str:
    """Get the cache file path for a model."""
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    return f"hybrid_data_{safe_name}.csv"


def get_run_cache_path(run_id: str) -> str:
    """Get the cache file path for a single run."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"run_{run_id}.parquet")


def fetch_all_data(runs_config: dict, use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """Fetch data for all model configurations.

    Args:
        runs_config: Configuration dict for runs to fetch
        use_cache: If True, load from cached CSV files if they exist
    """
    results = {}
    need_to_fetch = []

    # Check cache first
    for model_name in runs_config.keys():
        cache_path = get_cache_path(model_name)
        if use_cache and os.path.exists(cache_path):
            print(f"\nLoading cached data for: {model_name} from {cache_path}", flush=True)
            df = pd.read_csv(cache_path)
            results[model_name] = df
            print(f"  Loaded {len(df)} rows", flush=True)
        else:
            need_to_fetch.append(model_name)

    if not need_to_fetch:
        print("\nAll data loaded from cache!", flush=True)
        return results

    print("Creating W&B API client...", flush=True)
    api = wandb.Api(timeout=API_TIMEOUT)
    print("API client created", flush=True)

    for model_name in need_to_fetch:
        run_configs = runs_config[model_name]
        print(f"\nFetching data for: {model_name}", flush=True)
        all_dfs = []

        for config in run_configs:
            df = fetch_run_data(
                api,
                entity=config["entity"],
                project=config["project"],
                group=config["group"],
                use_cache=use_cache,
            )
            if not df.empty:
                df["stage"] = config["group"]
                all_dfs.append(df)

        if all_dfs:
            results[model_name] = pd.concat(all_dfs, ignore_index=True)
            # Save to cache immediately
            cache_path = get_cache_path(model_name)
            results[model_name].to_csv(cache_path, index=False)
            print(f"  Cached to {cache_path}", flush=True)

    return results


def get_tokens_col(df: pd.DataFrame) -> str | None:
    """Find the total tokens column."""
    for col in df.columns:
        if "total tokens" in col.lower():
            return col
    return None


def aggregate_mmlu_ce(df: pd.DataFrame) -> pd.Series:
    """Aggregate MMLU CE loss across subjects (lower is better)."""
    mmlu_cols = [c for c in df.columns if "mmlu" in c.lower() and "ce loss" in c.lower()]
    if not mmlu_cols:
        return pd.Series(dtype=float)
    return df[mmlu_cols].mean(axis=1, skipna=True)


def aggregate_mmlu_bpb(df: pd.DataFrame) -> pd.Series:
    """Aggregate MMLU as BPB (macro-averaged across subjects).

    Converts CE loss to BPB: BPB = CE_loss / ln(2)
    """
    mmlu_cols = [c for c in df.columns if "mmlu" in c.lower() and "ce loss" in c.lower() and "v2" not in c.lower()]
    if not mmlu_cols:
        return pd.Series(dtype=float)
    ce_loss = df[mmlu_cols].mean(axis=1, skipna=True)
    return ce_loss / np.log(2)


def aggregate_mmlu_accuracy(df: pd.DataFrame) -> pd.Series:
    """Aggregate MMLU accuracy (macro-averaged across subjects)."""
    mmlu_cols = [c for c in df.columns if "mmlu" in c.lower() and "length-normalized accuracy" in c.lower() and "v2" not in c.lower()]
    if not mmlu_cols:
        return pd.Series(dtype=float)
    return df[mmlu_cols].mean(axis=1, skipna=True)


def get_metric_col(df: pd.DataFrame, pattern: str) -> str | None:
    """Find a column matching the pattern."""
    for col in df.columns:
        if re.search(pattern, col, re.IGNORECASE):
            return col
    return None


def plot_training_curves(data: dict[str, pd.DataFrame], output_prefix: str = "hybrid_training"):
    """Create line plots of training time vs performance."""

    print("\n" + "=" * 60)
    print("Creating plots...")
    print("=" * 60)

    # Discover available metrics
    for model_name, df in data.items():
        print(f"\n{model_name} - Available columns:")
        eval_cols = [c for c in df.columns if "eval" in c.lower()]
        for col in sorted(eval_cols)[:20]:
            print(f"  {col}")

    # Colors for models (using AI2 theme)
    colors = {
        "OLMo 3 7B Base": AI2_COLORS["teal"],
        "OLMo 3.2 7B Base": AI2_COLORS["pink"],
    }

    # Define metrics to plot
    # Format: (title, metric_type, pattern) - pattern is None for aggregations
    metrics_config = [
        ("Dolma Common Crawl CE Loss", "single", r"dolma_common-crawl.*CE loss"),
        ("MMLU Accuracy (Macro Avg)", "mmlu_acc_aggregate", None),  # Aggregated MMLU accuracy
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes = axes.flatten()

    for idx, (title, metric_type, pattern) in enumerate(metrics_config):
        ax = axes[idx]

        for model_name, df in data.items():
            # Get x-axis (use total tokens for better comparability)
            x_col = get_tokens_col(df)
            if x_col is None:
                x_col = "_step"
            if x_col not in df.columns:
                print(f"  No x-axis column for {model_name}")
                continue

            # Get y-axis metric
            if metric_type == "mmlu_ce_aggregate":
                y_data = aggregate_mmlu_ce(df)
                if y_data.empty or y_data.isna().all():
                    print(f"  No {title} metric for {model_name}")
                    continue
            elif metric_type == "mmlu_bpb_aggregate":
                y_data = aggregate_mmlu_bpb(df)
                if y_data.empty or y_data.isna().all():
                    print(f"  No {title} metric for {model_name}")
                    continue
            elif metric_type == "mmlu_acc_aggregate":
                y_data = aggregate_mmlu_accuracy(df)
                if y_data.empty or y_data.isna().all():
                    print(f"  No {title} metric for {model_name}")
                    continue
            else:
                y_col = get_metric_col(df, pattern)
                if y_col is None:
                    print(f"  No {title} metric for {model_name}")
                    continue
                y_data = df[y_col]

            # Create plot dataframe
            plot_df = pd.DataFrame({
                "step": df[x_col],
                "metric": y_data,
            }).dropna()

            if plot_df.empty:
                print(f"  No valid data for {model_name} - {title}")
                continue

            # Aggregate by step (in case of multiple runs)
            agg_df = plot_df.groupby("step")["metric"].mean().reset_index()
            agg_df = agg_df.sort_values("step")

            # Plot
            ax.plot(
                agg_df["step"],
                agg_df["metric"],
                label=model_name,
                color=colors.get(model_name, "#666666"),
                linewidth=2,
                alpha=0.8,
            )

            print(f"  Plotted {model_name} - {title}: {len(agg_df)} points")

        ax.set_xlabel("Tokens Trained")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(loc="best")

        # Format x-axis with T/B suffixes for tokens
        ax.xaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, p: f"{x/1e12:.1f}T" if x >= 1e12 else f"{x/1e9:.0f}B" if x >= 1e9 else f"{x/1e6:.0f}M"
        ))

    plt.tight_layout()

    # Save figures
    plt.savefig(f"{output_prefix}.png", dpi=300, bbox_inches="tight")
    print(f"\nSaved: {output_prefix}.png")
    plt.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    print(f"Saved: {output_prefix}.pdf")


def clear_cache():
    """Clear all cached run data."""
    import shutil
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        print(f"Cleared cache directory: {CACHE_DIR}")
    # Also clear model-level CSVs
    for model_name in RUNS_CONFIG.keys():
        cache_path = get_cache_path(model_name)
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print(f"Removed: {cache_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot training curves from W&B data")
    parser.add_argument("--refresh", action="store_true", help="Force refresh data from W&B (ignore cache)")
    parser.add_argument("--plot-only", action="store_true", help="Only plot from cached data, don't fetch")
    parser.add_argument("--clear-cache", action="store_true", help="Clear all cached data and exit")
    args = parser.parse_args()

    if args.clear_cache:
        clear_cache()
        return

    print("=" * 60, flush=True)
    print("Fetching W&B data for training curves", flush=True)
    print("=" * 60, flush=True)

    use_cache = not args.refresh

    if not args.plot_only:
        # Login to wandb
        wandb.login()
        print("Logged in to W&B", flush=True)

    # Fetch data (or load from cache)
    data = fetch_all_data(RUNS_CONFIG, use_cache=use_cache)

    if not data:
        print("\nNo data fetched! Check your W&B access and run configurations.")
        return

    # Create plots
    plot_training_curves(data)


if __name__ == "__main__":
    main()
