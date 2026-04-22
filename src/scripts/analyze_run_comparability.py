"""
Analyze comparability of pretraining and LC extension runs against OLMo25 baseline.

Usage:
    python src/scripts/analyze_run_comparability.py
"""

import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import wandb

from olmo_core.utils import flatten_dict, prepare_cli_environment

# Keys that are always acceptable to differ (run identity / bookkeeping)
ALWAYS_IGNORE = {
    "trainer.callbacks.wandb.group",
    "trainer.callbacks.wandb.name",
    "trainer.save_folder",
}
ALWAYS_IGNORE_PATTERNS = [
    re.compile(r".*work_dir.*"),
]

# Keys acceptable to differ in LC extension comparisons but NOT pretraining
LC_ONLY_IGNORE = {
    "trainer.load_path",
}

# --- Run data from models_info.csv ---
# (run_name, pretraining_group_url, lc_run_url)
RUNS = [
    ("OLMo25",                                       "https://wandb.ai/ai2-llm/olmo3/groups/OLMo25",                                                         "https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/ci4kk6cf"),
    ("OLMo-2.5-noswa",                               "https://wandb.ai/ai2-llm/olmo3/groups/OLMo-2.5-noswa",                                                 "https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/sg3nke52"),
    ("OLMo29",                                       "https://wandb.ai/ai2-llm/olmo3/groups/OLMo29",                                                         "https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/ilo6rxfy"),
    ("LlamaClone-swa-fixed-init",                    "https://wandb.ai/ai2-llm/olmo3/groups/LlamaClone-swa-fixed-init",                                      "https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/mfhixmo8"),
    ("OLMo2.8-correctdata-noheadnorm",               "https://wandb.ai/ai2-llm/olmo3/groups/OLMo2.8-correctdata-noheadnorm",                                 "https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/8v18qwgd"),
    ("LlamaClone-8B",                                "https://wandb.ai/ai2-llm/olmo3/groups/LlamaClone-8B",                                                  "https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/oz13lw04"),
    ("LlamaClone-8B-gqa-16",                         "https://wandb.ai/ai2-llm/olmo3/groups/LlamaClone-8B-gqa-16",                                           "https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/22fkmnjk"),
    ("LlamaClone-8B-gqa-4",                          "https://wandb.ai/ai2-llm/olmo3/groups/LlamaClone-8B-gqa-4",                                            "https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/7iuy7k9t"),
    ("LlamaClone-8B-nogqa",                          "https://wandb.ai/ai2-llm/olmo3/groups/LlamaClone-8B-nogqa",                                            "https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/8sqdb1qd"),
    ("OLMo2.5-float8",                               "https://wandb.ai/ai2-llm/olmo3/groups/OLMo2.5-float8",                                                 "https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/yl17npuy"),
    ("OLMo2.5-plus-float8-same-init",                "https://wandb.ai/ai2-llm/olmo3/groups/OLMo2.5-plus-float8-same-init",                                  "https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/bxxkah46"),
    ("OLMo2.5-plus-gqa-fixed-max5T",                 "https://wandb.ai/ai2-llm/olmo3/groups/OLMo2.5-plus-gqa-fixed-max5T",                                   "https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/q4m37ql2"),
    ("OLMo2.5-plus-headnorm-test",                   "https://wandb.ai/ai2-llm/olmo3/groups/OLMo2.5-plus-headnorm-test",                                     "https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/n2vi9o8c"),
    ("LlamaClone-plus-qknorm-reordered-norm-fixed-reload", "https://wandb.ai/ai2-llm/olmo3/groups/LlamaClone-plus-qknorm-reordered-norm-fixed-reload",       "https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/3id4nl98"),
    ("OLMo-2.5-preorder-no-qk",                      "https://wandb.ai/ai2-llm/olmo3/groups/OLMo-2.5-preorder-no-qk",                                       "https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/zgu3wzw0"),
    ("OLMo-2.5-preorder",                            "https://wandb.ai/ai2-llm/olmo3/groups/OLMo-2.5-preorder",                                              "https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/jen0xpu2"),
    ("LlamaClone-plus-qknorm",                       "https://wandb.ai/ai2-llm/olmo3/groups/LlamaClone-plus-qknorm",                                         "https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/624m9b04"),
    ("LlamaClone-swa",                               "https://wandb.ai/ai2-llm/olmo3/groups/LlamaClone-swa",                                                 "https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/0eaui6s9"),
    ("OLMo2.5-gqa",                                  "https://wandb.ai/ai2-llm/olmo3/groups/OLMo2.5-gqa",                                                    "https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/vigsl0ts"),
    ("OLMo2.5-plus-gqa-fixed",                       "https://wandb.ai/ai2-llm/olmo3/groups/OLMo2.5-plus-gqa-fixed",                                         "https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/qg1ivql7"),
    ("OLMo2.5-half-context-fixed-init",              "https://wandb.ai/ai2-llm/olmo3/groups/OLMo2.5-half-context-fixed-init",                                "https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/7o0fze7s"),
    ("OLMo2.5-halfcontext",                          "https://wandb.ai/ai2-llm/olmo3/groups/OLMo2.5-halfcontext",                                            "https://wandb.ai/ai2-llm/olmo3-7b-long-context/runs/5xtv5mp6"),
]

GROUP_URL_RE = re.compile(r"^https?://wandb\.ai/([^/]+)/([^/]+)/groups/([^/?#]+)")
RUN_URL_RE = re.compile(r"^https?://wandb\.ai/([^/]+)/([^/]+)/runs/([^/?#]+)")


def parse_group_url(url: str) -> tuple[str, str, str]:
    m = GROUP_URL_RE.match(url)
    if not m:
        raise ValueError(f"Could not parse group URL: {url!r}")
    return m.group(1), m.group(2), m.group(3)


def parse_run_url(url: str) -> str:
    m = RUN_URL_RE.match(url)
    if not m:
        raise ValueError(f"Could not parse run URL: {url!r}")
    return f"{m.group(1)}/{m.group(2)}/{m.group(3)}"


def get_best_run_from_group(api: wandb.Api, entity: str, project: str, group: str) -> wandb.apis.public.Run:
    """Return the last finished run in the group, falling back to the last run of any state."""
    # Try finished runs first, ordered by most recent
    runs = api.runs(
        f"{entity}/{project}",
        filters={"group": group, "state": "finished"},
        order="-created_at",
        per_page=5,
    )
    runs_list = list(runs)
    if not runs_list:
        # Fall back to any state, most recent
        runs = api.runs(f"{entity}/{project}", filters={"group": group}, order="-created_at", per_page=1)
        runs_list = list(runs)
    if not runs_list:
        raise ValueError(f"No runs found for group {group!r} in {entity}/{project}")
    # Fetch full run object (the listing doesn't include rawconfig)
    chosen = runs_list[0]
    print(f"    -> picked run {chosen.id} (state={chosen.state}, created={chosen.created_at})")
    return api.run(f"{entity}/{project}/{chosen.id}")


def should_ignore(key: str, lc_mode: bool) -> bool:
    if key in ALWAYS_IGNORE:
        return True
    if lc_mode and key in LC_ONLY_IGNORE:
        return True
    for pat in ALWAYS_IGNORE_PATTERNS:
        if pat.match(key):
            return True
    return False


@dataclass
class ComparisonResult:
    run_name: str
    baseline_run_id: str
    comparison_run_id: str
    left_only: dict = field(default_factory=dict)
    right_only: dict = field(default_factory=dict)
    differences: dict = field(default_factory=dict)

    @property
    def has_unexpected_diffs(self) -> bool:
        return bool(self.left_only or self.right_only or self.differences)


def compare_configs(
    baseline_config: dict,
    other_config: dict,
    lc_mode: bool,
) -> tuple[dict, dict, dict]:
    left_only = {
        k: baseline_config[k]
        for k in baseline_config.keys() - other_config.keys()
        if not should_ignore(k, lc_mode)
    }
    right_only = {
        k: other_config[k]
        for k in other_config.keys() - baseline_config.keys()
        if not should_ignore(k, lc_mode)
    }
    differences = {
        k: (baseline_config[k], other_config[k])
        for k in baseline_config.keys() & other_config.keys()
        if baseline_config[k] != other_config[k] and not should_ignore(k, lc_mode)
    }
    return left_only, right_only, differences


def format_report_section(results: list[ComparisonResult], section_title: str) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append(section_title)
    lines.append("=" * 70)

    clean = [r for r in results if not r.has_unexpected_diffs]
    flagged = [r for r in results if r.has_unexpected_diffs]

    lines.append(f"\nCLEAN runs (no unexpected differences from OLMo25 baseline):")
    if clean:
        for r in clean:
            lines.append(f"  ✓ {r.run_name}  [{r.comparison_run_id}]")
    else:
        lines.append("  (none)")

    lines.append(f"\nFLAGGED runs (unexpected differences found):")
    if flagged:
        for r in flagged:
            lines.append(f"\n  ✗ {r.run_name}  [{r.comparison_run_id}]")
            if r.left_only:
                lines.append("    Keys only in OLMo25 baseline:")
                for k, v in sorted(r.left_only.items()):
                    lines.append(f"      - {k}: {v}")
            if r.right_only:
                lines.append("    Keys only in this run:")
                for k, v in sorted(r.right_only.items()):
                    lines.append(f"      + {k}: {v}")
            if r.differences:
                lines.append("    Differing values (baseline → this run):")
                for k, (lv, rv) in sorted(r.differences.items()):
                    lines.append(f"      ~ {k}:")
                    lines.append(f"          baseline: {lv}")
                    lines.append(f"          this run: {rv}")
    else:
        lines.append("  (none)")

    return "\n".join(lines)


def main():
    api = wandb.Api()

    print("Fetching baseline (OLMo25) pretraining run...", flush=True)
    baseline_pt_entity, baseline_pt_project, baseline_pt_group = parse_group_url(RUNS[0][1])
    baseline_pt_run = get_best_run_from_group(api, baseline_pt_entity, baseline_pt_project, baseline_pt_group)
    baseline_pt_config = flatten_dict(baseline_pt_run._attrs["rawconfig"])
    print(f"  Baseline pretraining run: {baseline_pt_run.id} ({baseline_pt_run.name})")

    print("Fetching baseline (OLMo25) LC extension run...", flush=True)
    baseline_lc_path = parse_run_url(RUNS[0][2])
    baseline_lc_run = api.run(baseline_lc_path)
    baseline_lc_config = flatten_dict(baseline_lc_run._attrs["rawconfig"])
    print(f"  Baseline LC run: {baseline_lc_run.id} ({baseline_lc_run.name})")

    pt_results: list[ComparisonResult] = []
    lc_results: list[ComparisonResult] = []

    # Skip index 0 (baseline itself)
    for run_name, pt_group_url, lc_run_url in RUNS[1:]:
        print(f"\nProcessing: {run_name}", flush=True)

        # --- Pretraining ---
        try:
            pt_entity, pt_project, pt_group = parse_group_url(pt_group_url)
            pt_run = get_best_run_from_group(api, pt_entity, pt_project, pt_group)
            pt_config = flatten_dict(pt_run._attrs["rawconfig"])
            left_only, right_only, differences = compare_configs(baseline_pt_config, pt_config, lc_mode=False)
            pt_results.append(ComparisonResult(
                run_name=run_name,
                baseline_run_id=baseline_pt_run.id,
                comparison_run_id=pt_run.id,
                left_only=left_only,
                right_only=right_only,
                differences=differences,
            ))
            print(f"  Pretraining: {pt_run.id} — {'CLEAN' if not (left_only or right_only or differences) else 'FLAGGED'}")
        except Exception as e:
            print(f"  Pretraining ERROR: {e}")
            pt_results.append(ComparisonResult(
                run_name=run_name,
                baseline_run_id=baseline_pt_run.id,
                comparison_run_id=f"ERROR: {e}",
                differences={"_error": (None, str(e))},
            ))

        # --- LC extension ---
        try:
            lc_path = parse_run_url(lc_run_url)
            lc_run = api.run(lc_path)
            lc_config = flatten_dict(lc_run._attrs["rawconfig"])
            left_only, right_only, differences = compare_configs(baseline_lc_config, lc_config, lc_mode=True)
            lc_results.append(ComparisonResult(
                run_name=run_name,
                baseline_run_id=baseline_lc_run.id,
                comparison_run_id=lc_run.id,
                left_only=left_only,
                right_only=right_only,
                differences=differences,
            ))
            print(f"  LC extension: {lc_run.id} — {'CLEAN' if not (left_only or right_only or differences) else 'FLAGGED'}")
        except Exception as e:
            print(f"  LC extension ERROR: {e}")
            lc_results.append(ComparisonResult(
                run_name=run_name,
                baseline_run_id=baseline_lc_run.id,
                comparison_run_id=f"ERROR: {e}",
                differences={"_error": (None, str(e))},
            ))

    print("\n\n")
    print(format_report_section(pt_results, "PRETRAINING RUNS — comparison vs OLMo25 baseline"))
    print()
    print(format_report_section(lc_results, "LC EXTENSION RUNS — comparison vs OLMo25 baseline"))


if __name__ == "__main__":
    prepare_cli_environment()
    main()
