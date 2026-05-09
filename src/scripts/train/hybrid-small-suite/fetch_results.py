#!/usr/bin/env python3
"""Fetch and parse olmo-eval results from a Beaker group.

Usage:
    uv run src/scripts/train/hybrid-small-suite/fetch_results.py --group yashasbls-hybrid-small-downstream-evals --completed-only --workspace ai2/linear-rnns
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from tqdm import tqdm



GROUP = "hybrid-small-downstream-evals"  # olmo-eval group name by default
OUTPUT_DIR = Path(__file__).parent / "results"

def sanitize_group_name(group: str) -> str:
    # Replace slashes and spaces with underscores
    return group.replace("/", "_").replace(" ", "_")


def get_experiment_arguments(exp: dict) -> list:
    """Extract the launch command arguments from a Beaker experiment."""
    jobs = exp.get("jobs", [])
    if not jobs:
        return []
    return jobs[0].get("execution", {}).get("spec", {}).get("arguments", [])


def extract_olmo_eval_group(metrics: dict, exp: dict) -> str:
    # Try to get from metrics first
    group = metrics.get("experiment_group")
    if group:
        return group
    # Try to parse from launch command
    cmd = get_experiment_arguments(exp)
    if isinstance(cmd, list):
        for i, arg in enumerate(cmd):
            if arg in ("--group", "--experiment-group") and i + 1 < len(cmd):
                return cmd[i + 1]
    return "unknown"


def run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: {' '.join(cmd)}\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    return result.stdout



def get_workspace_experiments(workspace: str) -> list[dict]:
    # Fetch all experiments in a Beaker workspace
    out = run(["beaker", "workspace", "experiments", "--format", "json", workspace])
    return json.loads(out)


def get_successful_job(exp: dict) -> Optional[dict]:
    """Return the first job with exitCode=0, or None."""
    for job in exp.get("jobs", []):
        st = job.get("status", {})
        if "finalized" in st and st.get("exitCode") == 0:
            return job
    return None


def is_olmo_eval_experiment(exp: dict) -> bool:
    # Only include experiments whose launch command is 'olmo-eval'
    cmd = get_experiment_arguments(exp)
    return isinstance(cmd, list) and len(cmd) > 0 and cmd[0] == "olmo-eval"


def is_completed(exp: dict) -> bool:
    """Return True if the experiment has at least one successful job."""
    return get_successful_job(exp) is not None


def get_job_status(exp: dict) -> str:
    jobs = exp.get("jobs", [])
    if not jobs:
        return "no_jobs"
    # Report based on the best job: succeeded if any job exited 0
    if get_successful_job(exp):
        return "succeeded"
    # Otherwise report status of the latest job
    status = jobs[-1].get("status", {})
    if "finalized" in status:
        exit_code = status.get("exitCode", -1)
        return f"failed(exit={exit_code})"
    if "exited" in status:
        return f"exited(code={status.get('exitCode', '?')})"
    if "started" in status:
        return "running"
    if "scheduled" in status:
        return "scheduled"
    return "unknown"


def download_results(exp: dict, output_dir: Path) -> Optional[Path]:
    exp_id = exp["id"]
    exp_dir = output_dir / exp_id
    if exp_dir.exists() and (exp_dir / "main" / "metrics.json").exists():
        return exp_dir

    # Find the successful job's result dataset
    job = get_successful_job(exp)
    if job is None:
        tqdm.write(f"  [warn] No successful job found for {exp_id}")
        return None
    dataset_id = job.get("result", {}).get("beaker")
    if not dataset_id:
        tqdm.write(f"  [warn] Successful job for {exp_id} has no result dataset")
        return None

    tmp_dir = output_dir / f".{exp_id}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["beaker", "dataset", "fetch", "--output", str(tmp_dir / "main"), dataset_id],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        tqdm.write(f"  [warn] Could not download dataset {dataset_id} for {exp_id}: {result.stderr.strip()}")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        return None
    if not (tmp_dir / "main" / "metrics.json").exists():
        tqdm.write(f"  [warn] Downloaded {exp_id} (dataset {dataset_id}) but no metrics.json found")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        return None
    if exp_dir.exists():
        shutil.rmtree(exp_dir)
    tmp_dir.rename(exp_dir)
    return exp_dir


def parse_metrics(exp_dir: Path) -> Optional[dict]:
    metrics_file = exp_dir / "main" / "metrics.json"
    if not metrics_file.exists():
        return None
    with open(metrics_file) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Fetch olmo-eval results for a specific olmo-eval group.")
    parser.add_argument("--group", default=GROUP, help="olmo-eval group name")
    parser.add_argument("--workspace", default="yashasbls", help="Beaker workspace (user namespace) to fetch experiments from")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="Base output directory (results will be organized by group)")
    parser.add_argument("--completed-only", action="store_true", help="Skip experiments still running")
    parser.add_argument("--no-download", action="store_true", help="Only list, don't download")
    parser.add_argument("--workers", type=int, default=32, help="Parallel download workers")
    args = parser.parse_args()

    group_dir = args.output / sanitize_group_name(args.group)
    group_dir.mkdir(parents=True, exist_ok=True)

    combined_path = group_dir / "all_results.json"

    # Load previously known experiment IDs from all_results.json
    prev_exp_ids = set()
    if combined_path.exists():
        with open(combined_path) as f:
            prev_results = json.load(f)
        prev_exp_ids = {r["experiment_id"] for r in prev_results}

    print(f"Fetching all experiments from Beaker workspace: {args.workspace}")
    all_experiments = get_workspace_experiments(args.workspace)
    experiments = [e for e in all_experiments if is_olmo_eval_experiment(e)]
    print(f"Found {len(experiments)} olmo-eval experiments (out of {len(all_experiments)} total)")

    # Check for new completed experiments not yet in all_results.json
    new_experiments = [
        e for e in experiments
        if e["id"] not in prev_exp_ids
        and (not args.completed_only or get_job_status(e) == "succeeded")
    ]

    if not new_experiments and prev_exp_ids:
        print(f"[skip] No new experiments since last run ({len(prev_exp_ids)} already tracked).")
        return

    print(f"{len(new_experiments)} new experiments to process\n")

    # Filter to all succeeded experiments for building the full results file
    to_process = [
        e for e in experiments
        if args.completed_only and get_job_status(e) == "succeeded"
        or not args.completed_only
    ]

    def process_one(exp: dict) -> Optional[dict]:
        exp_id = exp["id"]
        exp_name = exp.get("name", exp_id)
        status = get_job_status(exp)

        exp_dir = download_results(exp, group_dir)
        if exp_dir is None:
            return None

        metrics = parse_metrics(exp_dir)
        if metrics is None:
            tqdm.write(f"  [warn] no metrics.json for {exp_id}")
            return None

        olmo_group = extract_olmo_eval_group(metrics, exp)
        if olmo_group != args.group:
            return None

        label = "[new] " if exp_id not in prev_exp_ids else ""
        if label:
            tqdm.write(f"  {label}{exp_name}  id={exp_id}  status={status}")

        if args.no_download:
            return None

        summary = metrics.get("summary", {})
        model = metrics.get("config", {}).get("provider", {}).get("model", "unknown")
        return {
            "experiment_id": exp_id,
            "experiment_name": exp_name,
            "experiment_group": olmo_group,
            "model": model,
            "status": status,
            "summary": summary,
            "tasks": metrics.get("tasks", []),
        }

    all_results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, exp): exp for exp in to_process}
        with tqdm(total=len(futures), unit="exp") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    all_results.append(result)
                pbar.update(1)

    if not args.no_download and all_results:
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved {len(all_results)} results to {combined_path}")


if __name__ == "__main__":
    main()
