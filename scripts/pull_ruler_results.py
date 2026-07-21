#!/usr/bin/env python3
"""
Pull RULER results for my recent Beaker eval jobs into a tidy CSV.

This talks to Beaker directly -- it does NOT use the olmo-cookbook-eval
dashboard / oe-eval-datalake service, which has been decommissioned (its
endpoints now just 404 to the generic allenai.org page). Instead it lists
Beaker experiments in a workspace, matches RULER job names, and downloads
each job's ``metrics-all.jsonl`` result file straight from its Beaker result
dataset (``beaker experiment results``).

RULER jobs are launched by ./launch_long_context_evals.sh via
olmo-cookbook-eval, which names each Beaker experiment
``lmeval-<run_name>-on-<first_task>-<N>tasks-<hex>`` (oe-eval's own naming
convention). We match that pattern to find RULER jobs and recover the model
name (``run_name``) and context length (from the first task's ``__<len>``
suffix).

For every matched model x length it emits one row of
    modelname,4k,8k,16k,32k,64k,128k
where each value is the aggregate ``ruler_all__<len>::suite`` score (the full
13-subtask suite averaged at that context length), scaled to a percentage.

A job is considered done once its ``metrics-all.jsonl`` contains that
aggregate row -- NOT based on the experiment's Beaker exit code. Jobs
launched before the datalake-push fix (see olmo-cookbook's evaluate_checkpoint)
exit 1 even on a fully successful eval, because the run crashes trying to
push to the dead datalake *after* metrics are already written; exit code is
therefore not a reliable success signal here.

Results are *merged* into the output CSV: existing rows are preserved, models
from this run are added or updated in place (new non-empty cells override,
but a new run never blanks out a value the file already had), and models not
seen in this run are left untouched.

Usage:
    ./pull_ruler_results.py                  # -> ../results/ruler_results.csv
    ./pull_ruler_results.py -o out.csv       # custom output path
    ./pull_ruler_results.py -w ai2/flex2     # different workspace
    ./pull_ruler_results.py -u amandab       # different launching user
    ./pull_ruler_results.py -d 7             # look back 7 days

Env vars:
    WORKSPACE        Beaker workspace (default: ai2/flex2)
"""

import argparse
import csv
import datetime
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Context lengths (in K) we look for, mapped to the CSV column label.
LENGTHS_K = [4, 8, 16, 32, 64, 128]

# oe-eval's beaker experiment naming convention (make_exp_name in
# oe_eval_internal/utilities/launch_utils.py): lmeval-<model>-on-<task>-<hex>,
# where for a multi-task RULER job <task> is "<first_subtask>-<N>tasks".
NAME_RE = re.compile(r"^lmeval-(?P<model>.+?)-on-ruler_[a-z0-9_]+__(?P<length>\d+)-\d+tasks-[0-9a-f]+$")


def parse_ts(t: str) -> datetime.datetime:
    """Parse a Beaker ISO-8601 timestamp (trailing 'Z') into an aware datetime."""
    return datetime.datetime.fromisoformat(t.replace("Z", "+00:00"))


def list_experiments(workspace: str) -> list[dict]:
    """Return all experiments in ``workspace`` as parsed JSON (newest first)."""
    proc = subprocess.run(
        ["beaker", "workspace", "experiments", workspace, "--format", "json"],
        stdout=subprocess.PIPE,
        check=True,
    )
    return json.loads(proc.stdout)


def find_ruler_jobs(experiments: list[dict], user: str, days: int) -> list[dict]:
    """
    Filter ``experiments`` to RULER jobs launched by ``user`` in the last
    ``days`` days, keeping only the most recent run per (model, length).
    """
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)

    candidates = []
    for exp in experiments:
        if exp.get("author", {}).get("name") != user:
            continue
        if parse_ts(exp["created"]) < cutoff:
            continue
        m = NAME_RE.match(exp["name"])
        if not m:
            continue
        exp = {**exp, "_model": m["model"], "_length": int(m["length"])}
        candidates.append(exp)

    candidates.sort(key=lambda e: e["created"], reverse=True)
    deduped: dict[tuple, dict] = {}
    for exp in candidates:
        deduped.setdefault((exp["_model"], exp["_length"]), exp)
    return list(deduped.values())


def fetch_aggregate_score(experiment_id: str) -> float | None:
    """
    Download just the metrics files from an experiment's Beaker result
    dataset and return the full-suite aggregate primary_score, or None if the
    aggregate row isn't present (job still running, or failed before
    finishing all subtasks).
    """
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run(
            ["beaker", "experiment", "results", experiment_id, "-o", tmp, "--prefix", "metrics", "--quiet"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            print(
                f"  !! could not fetch results for {experiment_id}: {proc.stderr.decode().strip()}",
                file=sys.stderr,
            )
            return None

        metrics_files = list(Path(tmp).rglob("metrics-all.jsonl"))
        if not metrics_files:
            return None

        for line in metrics_files[0].read_text().splitlines():
            row = json.loads(line)
            if row.get("task_name", "").startswith("ruler_all__"):
                return row["metrics"]["primary_score"]
    return None


def tidy(jobs: list[dict]) -> "dict[str, dict[str, str]]":
    """Fetch scores for ``jobs`` and reshape into model -> {label: score}."""
    scores: dict[str, dict[str, str]] = {}
    for i, exp in enumerate(jobs, 1):
        label = f"{exp['_length'] // 1024}k"
        print(f"[{i}/{len(jobs)}] {exp['_model']} @ {label} ({exp['name']})...", file=sys.stderr)
        score = fetch_aggregate_score(exp["id"])
        if score is None:
            print("  !! no aggregate score found (job may still be running)", file=sys.stderr)
            continue
        scores.setdefault(exp["_model"], {})[label] = f"{score * 100:.2f}"
    return scores


def read_existing(path: str) -> "dict[str, dict[str, str]]":
    """Read a previously written ruler CSV into model -> {label: score} (or {})."""
    if not os.path.exists(path):
        return {}
    labels = [f"{k}k" for k in LENGTHS_K]
    scores: dict[str, dict[str, str]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            model = row.get("modelname")
            if not model:
                continue
            scores[model] = {label: row[label] for label in labels if row.get(label, "") not in ("", "-")}
    return scores


def merge(existing: "dict[str, dict[str, str]]", new: "dict[str, dict[str, str]]") -> list[list[str]]:
    """
    Merge ``new`` scores into ``existing``, preserving every existing model.

    New non-empty cells override existing ones; existing cells are kept where
    the new fetch has no value. Existing models keep their original order;
    freshly seen models are appended.
    """
    labels = [f"{k}k" for k in LENGTHS_K]
    order = list(existing.keys()) + [m for m in new if m not in existing]
    header = ["modelname"] + labels
    out = [header]
    for model in order:
        merged = {**existing.get(model, {}), **new.get(model, {})}
        out.append([model] + [merged.get(label, "-") for label in labels])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-w", "--workspace", default=os.environ.get("WORKSPACE", "ai2/flex2"), help="Beaker workspace."
    )
    parser.add_argument(
        "-u", "--user", default=os.environ.get("USER", "amandab"), help="Launching Beaker user."
    )
    parser.add_argument("-d", "--days", type=int, default=3, help="Look back this many days.")
    parser.add_argument(
        "-o",
        "--output",
        default=str(Path(__file__).resolve().parent.parent / "results" / "ruler_results.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    experiments = list_experiments(args.workspace)
    jobs = find_ruler_jobs(experiments, args.user, args.days)
    if not jobs:
        sys.exit(f"No RULER jobs for user '{args.user}' in the last {args.days} days.")

    print(f"Found {len(jobs)} RULER job(s); fetching results...", file=sys.stderr)
    new = tidy(jobs)
    existing = read_existing(args.output)
    table = merge(existing, new)

    with open(args.output, "w", newline="") as f:
        csv.writer(f).writerows(table)

    added = [m for m in new if m not in existing]
    print(
        f"Wrote {len(table) - 1} models to {args.output} "
        f"({len(added)} new, {len(new) - len(added)} updated, "
        f"{len(table) - 1 - len(new)} untouched).",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
