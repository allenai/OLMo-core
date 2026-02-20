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
Plot extended training curves for hybrid models with many downstream eval metrics.

Compares:
- Baseline: OLMo-3-1025-7B (stage 1 + stage 2)
- Hybrid: OLMo3.1-7B-6T-30h

Invocation:
    uv run scripts/plots/hybrid/hybrid_training_curves_extended.py
    uv run scripts/plots/hybrid/hybrid_training_curves_extended.py --plot-only
    uv run scripts/plots/hybrid/hybrid_training_curves_extended.py --refresh

@kyleclo
"""

import os
import sys
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add olmo-cookbook to path for AI2 theme
sys.path.insert(0, "/Users/kylel/ai2/olmo-cookbook/olmo-3-plots")
from ai2_theme import apply_ai2_theme, AI2_COLORS

apply_ai2_theme()

API_TIMEOUT = 120

# W&B run configurations (same as original)
RUNS_CONFIG = {
    "OLMo 3 7B Base": [
        {"entity": "ai2-llm", "project": "Olmo-3-1025-7B", "group": "Olmo-3-1025-7B-stage-1"},
        {"entity": "ai2-llm", "project": "Olmo-3-1025-7B", "group": "Olmo-3-1025-7B-stage-2"},
    ],
    "OLMo 3.2 7B Base": [
        {"entity": "ai2-llm", "project": "linear-rnns", "group": "OLMo3.1-7B-6T-30h"},
    ],
}

# Extended metric patterns for all downstream eval metrics we want.
# We request exact key substrings to match against run.summary keys and history columns.
# For each benchmark, we pull all available metric types (accuracy, BPB, CE loss).
METRIC_PATTERNS = {
    # ---- ARC Challenge (MC format) ----
    "arc_challenge_mc_acc": r"eval/downstream/arc_challenge_test_mc_5shot_fast \(accuracy\)$",
    "arc_challenge_mc_ce": r"eval/downstream/arc_challenge_test_mc_5shot_fast \(CE loss\)$",
    "arc_challenge_bpb": r"eval/downstream/arc_challenge_test_bpb_5shot \(BPB\)$",
    # ---- ARC Easy (MC format) ----
    "arc_easy_mc_acc": r"eval/downstream/arc_easy_test_mc_5shot_fast \(accuracy\)$",
    "arc_easy_mc_ce": r"eval/downstream/arc_easy_test_mc_5shot_fast \(CE loss\)$",
    "arc_easy_bpb": r"eval/downstream/arc_easy_test_bpb_5shot \(BPB\)$",
    # ---- HellaSwag ----
    "hellaswag_bpb": r"eval/downstream/hellaswag_bpb_5shot \(BPB\)$",
    # ---- Codex (BPB only) ----
    "codex_humaneval_bpb": r"eval/downstream/codex_humaneval_gold_bpb_3shot \(BPB\)$",
    "codex_mbpp_bpb": r"eval/downstream/codex_mbpp_gold_bpb_3shot \(BPB\)$",
    # ---- Minerva Math (BPB only) ----
    "minerva_math_bpb": r"eval/downstream/minerva_math_500_gold_bpb_0shot \(BPB\)$",
}

CACHE_DIR = ".wandb_cache_extended"


def get_run_cache_path(run_id: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"run_{run_id}.parquet")


def get_model_cache_path(model_name: str) -> str:
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    return f"hybrid_extended_data_{safe_name}.csv"


def get_relevant_keys_from_run(run) -> list[str]:
    """Match metric patterns against a run's summary to find exact key names."""
    keys = ["_step", "throughput/total tokens"]
    try:
        summary_keys = list(run.summary.keys())
    except Exception:
        return keys

    for key in summary_keys:
        for pattern in METRIC_PATTERNS.values():
            if re.search(pattern, key):
                keys.append(key)
                break

    return list(set(keys))


def fetch_run_data(api, entity: str, project: str, group: str, use_cache: bool = True) -> pd.DataFrame:
    """Fetch data for all runs in a group using scan_history with per-run caching."""
    import wandb

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

        run_cache_path = get_run_cache_path(run.id)
        if use_cache and os.path.exists(run_cache_path):
            try:
                history = pd.read_parquet(run_cache_path)
                all_data.append(history)
                print(f"      Loaded from cache: {len(history)} rows, {len(history.columns)} cols", flush=True)
                continue
            except Exception as e:
                print(f"      Cache read failed, re-fetching: {e}", flush=True)

        try:
            keys = get_relevant_keys_from_run(run)
            print(f"      Found {len(keys)} relevant keys from summary", flush=True)

            if len(keys) <= 2:
                print(f"      No eval metrics found in summary, skipping", flush=True)
                continue

            # Stream history with scan_history (memory efficient)
            rows = []
            for row in run.scan_history(keys=keys, page_size=10000):
                rows.append(row)

            if not rows:
                print(f"      No history rows found", flush=True)
                continue

            history = pd.DataFrame(rows)
            history["run_name"] = run.name
            history["run_id"] = run.id

            # Cache immediately
            history.to_parquet(run_cache_path)
            print(f"      Fetched {len(history)} rows, {len(history.columns)} cols. Cached.", flush=True)
            all_data.append(history)

        except Exception as e:
            print(f"      Error: {e}", flush=True)
            continue

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def fetch_all_data(use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """Fetch data for all models, with model-level CSV caching."""
    results = {}
    need_to_fetch = []

    for model_name in RUNS_CONFIG.keys():
        cache_path = get_model_cache_path(model_name)
        if use_cache and os.path.exists(cache_path):
            print(f"\nLoading cached data for: {model_name} from {cache_path}", flush=True)
            df = pd.read_csv(cache_path)
            results[model_name] = df
            print(f"  Loaded {len(df)} rows, {len(df.columns)} cols", flush=True)
        else:
            need_to_fetch.append(model_name)

    if not need_to_fetch:
        print("\nAll data loaded from cache!", flush=True)
        _merge_mmlu_from_original(results)
        return results

    import wandb

    wandb.login()
    api = wandb.Api(timeout=API_TIMEOUT)

    for model_name in need_to_fetch:
        run_configs = RUNS_CONFIG[model_name]
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
            combined = pd.concat(all_dfs, ignore_index=True)
            results[model_name] = combined
            cache_path = get_model_cache_path(model_name)
            combined.to_csv(cache_path, index=False)
            print(f"  Cached {len(combined)} rows to {cache_path}", flush=True)

    # Merge MMLU data from original training curves cache (which already has MMLU columns)
    _merge_mmlu_from_original(results)

    return results


def _merge_mmlu_from_original(results: dict[str, pd.DataFrame]):
    """Merge MMLU columns from the original training curves cache into extended data."""
    ORIGINAL_CACHE_NAMES = {
        "OLMo 3 7B Base": "hybrid_data_OLMo_3_7B_Base.csv",
        "OLMo 3.2 7B Base": "hybrid_data_OLMo_3.2_7B_Base.csv",
    }
    for model_name, df in results.items():
        orig_path = ORIGINAL_CACHE_NAMES.get(model_name)
        if orig_path and os.path.exists(orig_path):
            orig_df = pd.read_csv(orig_path)
            mmlu_cols = [c for c in orig_df.columns if "mmlu" in c.lower() and c not in df.columns]
            if mmlu_cols:
                tok_col = None
                for col in df.columns:
                    if "total tokens" in col.lower():
                        tok_col = col
                        break
                if tok_col and tok_col in orig_df.columns:
                    merge_cols = [tok_col] + mmlu_cols
                    orig_subset = orig_df[merge_cols].drop_duplicates(subset=[tok_col])
                    results[model_name] = df.merge(orig_subset, on=tok_col, how="left")
                    print(f"  Merged {len(mmlu_cols)} MMLU columns into {model_name}", flush=True)


def get_tokens_col(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if "total tokens" in col.lower():
            return col
    return None


def get_metric_col(df: pd.DataFrame, pattern: str) -> str | None:
    for col in df.columns:
        if re.search(pattern, col):
            return col
    return None


def print_available_metrics(data: dict[str, pd.DataFrame]):
    """Print all available eval metrics across models."""
    print("\n" + "=" * 60)
    print("Available downstream eval metrics")
    print("=" * 60)
    for model_name, df in data.items():
        eval_cols = sorted([c for c in df.columns if "eval/downstream" in c.lower()])
        n_nonnull = {c: df[c].notna().sum() for c in eval_cols}
        print(f"\n{model_name} ({len(eval_cols)} downstream metrics):")
        for col in eval_cols:
            print(f"  [{n_nonnull[col]:3d} pts] {col}")


def compute_efficiency(data, metric_type, pattern, lower_is_better, baseline_name, target_name):
    """Compute token efficiency percentage for sorting. Returns pct_fewer or -inf if not computable."""
    for model_name, df in data.items():
        pass  # just need to iterate

    curves = {}
    for model_name, df in data.items():
        x_col = None
        for col in df.columns:
            if "total tokens" in col.lower():
                x_col = col
                break
        if x_col is None:
            x_col = "_step"
        if x_col not in df.columns:
            continue

        if metric_type == "basic_skills_acc_aggregate":
            bs_cols = [c for c in df.columns if "basic_skills" in c.lower() and "(accuracy)" in c]
            if not bs_cols:
                continue
            y_data = df[bs_cols].mean(axis=1, skipna=True)
        elif metric_type == "mmlu_acc_aggregate":
            mmlu_cols = [c for c in df.columns if "mmlu" in c.lower() and "length-normalized accuracy" in c.lower() and "v2" not in c.lower()]
            if not mmlu_cols:
                continue
            y_data = df[mmlu_cols].mean(axis=1, skipna=True)
        else:
            y_col = None
            for col in df.columns:
                if re.search(pattern, col):
                    y_col = col
                    break
            if y_col is None:
                continue
            y_data = df[y_col]

        plot_df = pd.DataFrame({"step": df[x_col], "metric": y_data}).dropna()
        if plot_df.empty:
            continue
        agg_df = plot_df.groupby("step")["metric"].mean().reset_index().sort_values("step")
        curves[model_name] = agg_df

    if baseline_name not in curves or target_name not in curves:
        return float("-inf")

    baseline_df = curves[baseline_name]
    target_df = curves[target_name]
    baseline_end_tokens = baseline_df["step"].iloc[-1]
    baseline_end_metric = baseline_df["metric"].iloc[-1]
    target_steps = target_df["step"].values
    target_metrics = target_df["metric"].values

    if lower_is_better:
        mask = target_metrics <= baseline_end_metric
    else:
        mask = target_metrics >= baseline_end_metric

    if not mask.any():
        return float("-inf")

    first_idx = np.where(mask)[0][0]
    if first_idx > 0:
        x0, x1 = target_steps[first_idx - 1], target_steps[first_idx]
        y0, y1 = target_metrics[first_idx - 1], target_metrics[first_idx]
        if y1 != y0:
            frac = (baseline_end_metric - y0) / (y1 - y0)
            target_tokens_at_match = x0 + frac * (x1 - x0)
        else:
            target_tokens_at_match = x0
    else:
        target_tokens_at_match = target_steps[first_idx]

    return (1 - target_tokens_at_match / baseline_end_tokens) * 100


def plot_training_curves(data: dict[str, pd.DataFrame], output_prefix: str = "hybrid_training_extended"):
    """Create multi-panel training curves for all extended metrics."""
    from matplotlib.patheffects import withStroke

    colors = {
        "OLMo 3 7B Base": AI2_COLORS["teal"],
        "OLMo 3.2 7B Base": AI2_COLORS["pink"],
    }

    baseline_name = "OLMo 3 7B Base"
    target_name = "OLMo 3.2 7B Base"

    # Define plot panels: (display_title, metric_type, pattern_or_key, lower_is_better)
    metrics_config = [
        ("ARC Challenge (Acc)", "single", r"arc_challenge_test_mc_5shot_fast \(accuracy\)$", False),
        ("ARC Easy (Acc)", "single", r"arc_easy_test_mc_5shot_fast \(accuracy\)$", False),
        ("HellaSwag (BPB)", "single", r"hellaswag_bpb_5shot \(BPB\)$", True),
        ("Codex HumanEval (BPB)", "single", r"codex_humaneval_gold_bpb_3shot \(BPB\)$", True),
        ("Codex MBPP (BPB)", "single", r"codex_mbpp_gold_bpb_3shot \(BPB\)$", True),
        ("Minerva Math 500 (BPB)", "single", r"minerva_math_500_gold_bpb_0shot \(BPB\)$", True),
    ]

    # Sort by token efficiency (most efficient first)
    efficiencies = []
    for title, metric_type, pattern, lower_is_better in metrics_config:
        eff = compute_efficiency(data, metric_type, pattern, lower_is_better, baseline_name, target_name)
        efficiencies.append((eff, (title, metric_type, pattern, lower_is_better)))
    efficiencies.sort(key=lambda x: -x[0])
    metrics_config = [item for _, item in efficiencies]

    n_rows = 3
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    axes = axes.flatten()

    for idx, (title, metric_type, pattern, lower_is_better) in enumerate(metrics_config):
        ax = axes[idx]
        curves = {}

        for model_name, df in data.items():
            x_col = get_tokens_col(df)
            if x_col is None:
                x_col = "_step"
            if x_col not in df.columns:
                continue

            # Get y-axis
            if metric_type == "mmlu_acc_aggregate":
                mmlu_cols = [c for c in df.columns if "mmlu" in c.lower() and "length-normalized accuracy" in c.lower() and "v2" not in c.lower()]
                if not mmlu_cols:
                    print(f"  No MMLU accuracy cols for {model_name}")
                    continue
                y_data = df[mmlu_cols].mean(axis=1, skipna=True)
            elif metric_type == "basic_skills_acc_aggregate":
                bs_cols = [c for c in df.columns if "basic_skills" in c.lower() and "(accuracy)" in c]
                if not bs_cols:
                    print(f"  No basic skills accuracy cols for {model_name}")
                    continue
                y_data = df[bs_cols].mean(axis=1, skipna=True)
            else:
                y_col = get_metric_col(df, pattern)
                if y_col is None:
                    print(f"  No {title} metric for {model_name}")
                    continue
                y_data = df[y_col]

            plot_df = pd.DataFrame({"step": df[x_col], "metric": y_data}).dropna()
            if plot_df.empty:
                continue

            agg_df = plot_df.groupby("step")["metric"].mean().reset_index().sort_values("step")
            curves[model_name] = agg_df

            ax.plot(
                agg_df["step"],
                agg_df["metric"],
                label=model_name,
                color=colors.get(model_name, "#666666"),
                linewidth=2,
                alpha=0.8,
            )
            print(f"  Plotted {model_name} - {title}: {len(agg_df)} points")

        # Draw token efficiency arrow
        if baseline_name in curves and target_name in curves:
            baseline_df = curves[baseline_name]
            target_df = curves[target_name]

            baseline_end_tokens = baseline_df["step"].iloc[-1]
            baseline_end_metric = baseline_df["metric"].iloc[-1]

            target_steps = target_df["step"].values
            target_metrics = target_df["metric"].values

            if lower_is_better:
                mask = target_metrics <= baseline_end_metric
            else:
                mask = target_metrics >= baseline_end_metric

            if mask.any():
                first_idx = np.where(mask)[0][0]
                if first_idx > 0:
                    x0, x1 = target_steps[first_idx - 1], target_steps[first_idx]
                    y0, y1 = target_metrics[first_idx - 1], target_metrics[first_idx]
                    if y1 != y0:
                        frac = (baseline_end_metric - y0) / (y1 - y0)
                        target_tokens_at_match = x0 + frac * (x1 - x0)
                    else:
                        target_tokens_at_match = x0
                else:
                    target_tokens_at_match = target_steps[first_idx]

                pct_fewer = (1 - target_tokens_at_match / baseline_end_tokens) * 100

                if pct_fewer > 0:
                    ax.annotate(
                        "",
                        xy=(target_tokens_at_match, baseline_end_metric),
                        xytext=(baseline_end_tokens, baseline_end_metric),
                        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.5, shrinkA=0, shrinkB=0),
                    )
                    mid_tokens = (baseline_end_tokens + target_tokens_at_match) / 2
                    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                    label_y_offset = -0.015 * y_range if lower_is_better else 0.005 * y_range
                    label_va = "top" if lower_is_better else "bottom"
                    ax.text(
                        mid_tokens,
                        baseline_end_metric + label_y_offset,
                        f"{pct_fewer:.0f}%",
                        ha="center",
                        va=label_va,
                        fontsize=10,
                        fontweight="bold",
                        color="#333333",
                        path_effects=[withStroke(linewidth=4, foreground="white")],
                    )
                    print(f"  {title}: {pct_fewer:.0f}% fewer tokens")

        ax.set_xlabel("Tokens Trained")
        ax.set_ylabel(title.split("(")[0].strip())
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, p: f"{x/1e12:.1f}T" if x >= 1e12 else f"{x/1e9:.0f}B" if x >= 1e9 else f"{x/1e6:.0f}M"
        ))

    # Hide unused axes
    for idx in range(len(metrics_config), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{output_prefix}.png", dpi=300, bbox_inches="tight")
    print(f"\nSaved: {output_prefix}.png")
    plt.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    print(f"Saved: {output_prefix}.pdf")


def clear_cache():
    import shutil
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        print(f"Cleared cache directory: {CACHE_DIR}")
    for model_name in RUNS_CONFIG.keys():
        cache_path = get_model_cache_path(model_name)
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print(f"Removed: {cache_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot extended training curves from W&B data")
    parser.add_argument("--refresh", action="store_true", help="Force refresh data from W&B (ignore cache)")
    parser.add_argument("--plot-only", action="store_true", help="Only plot from cached data, don't fetch")
    parser.add_argument("--list-metrics", action="store_true", help="List available metrics and exit")
    parser.add_argument("--clear-cache", action="store_true", help="Clear all cached data and exit")
    args = parser.parse_args()

    if args.clear_cache:
        clear_cache()
        return

    use_cache = not args.refresh
    data = fetch_all_data(use_cache=use_cache)

    if not data:
        print("\nNo data fetched!")
        return

    if args.list_metrics:
        print_available_metrics(data)
        return

    print_available_metrics(data)
    plot_training_curves(data)


if __name__ == "__main__":
    main()
