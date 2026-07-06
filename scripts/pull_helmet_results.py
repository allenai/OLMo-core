#!/usr/bin/env python3
"""
Pull HELMET results for my recent Beaker eval jobs into a tidy CSV.

This finds every HELMET evaluation launched by a given user (default: the
``$USER``/``amandab``) in a Beaker workspace within the last N days, fetches the
per-length aggregate scores via ``../ai2-helmet/fetch_results.py``, and emits one
row per model:

    modelname,8K avg,16K avg,32k avg,64k avg

where each value is the ``helmet_average`` at that context length (8192, 16384,
32768, 65536 tokens).

HELMET jobs are launched by ../ai2-helmet/gantry_eval.sh, which sets a Beaker
experiment description of the form "Evaluating <model> on HELMET[ (nothink)]".
We use that description to identify HELMET jobs and to give each row a name.
Only experiments that *succeeded* are pulled; failed / canceled / running jobs
are skipped. When the same model was evaluated more than once, the most recent
successful run wins.

Results are *merged* into the output CSV: existing rows are preserved, models
from this run are added or updated in place (new non-empty cells override, but a
new run never blanks out a value the file already had), and models not seen in
this run are left untouched.

Usage:
    ./pull_helmet_results.py                       # -> ../results/helmet_results.csv
    ./pull_helmet_results.py -o out.csv            # custom output path
    ./pull_helmet_results.py -w ai2/flex2          # different workspace
    ./pull_helmet_results.py -u amandab            # different launching user
    ./pull_helmet_results.py -d 7                   # look back 7 days

Env vars:
    WORKSPACE        Beaker workspace (default: ai2/flex2)
    HELMET_DIR       ai2-helmet checkout containing fetch_results.py
                     (default: ../../ai2-helmet, i.e. a sibling of OLMo-core)
"""

import argparse
import csv
import datetime
import io
import json
import os
import subprocess
import sys
from pathlib import Path

# HELMET context lengths in tokens, mapped to the CSV column label we emit.
LENGTHS = [(8192, "8K avg"), (16384, "16K avg"), (32768, "32k avg"), (65536, "64k avg")]


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


def find_helmet_jobs(experiments: list[dict], user: str, days: int) -> list[dict]:
    """
    Filter ``experiments`` to succeeded HELMET jobs launched by ``user`` in the
    last ``days`` days, keeping only the most recent run per model description.
    """
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)

    candidates = []
    for exp in experiments:
        if exp.get("author", {}).get("name") != user:
            continue
        if parse_ts(exp["created"]) < cutoff:
            continue
        description = exp.get("description") or ""
        if "HELMET" not in description.upper():
            continue
        jobs = exp.get("jobs") or []
        if not jobs or jobs[0].get("status", {}).get("exitCode") != 0:
            continue  # only succeeded runs
        candidates.append(exp)

    # Dedup by description (which encodes model + think/nothink), newest wins.
    candidates.sort(key=lambda e: e["created"], reverse=True)
    deduped: dict[str, dict] = {}
    for exp in candidates:
        deduped.setdefault(exp["description"], exp)
    return list(deduped.values())


def fetch_results(experiment_ids: list[str], helmet_dir: Path) -> list[dict]:
    """Run ai2-helmet/fetch_results.py for the given experiments; return its rows."""
    fetch_script = helmet_dir / "fetch_results.py"
    if not fetch_script.exists():
        sys.exit(f"Could not find {fetch_script}. Set HELMET_DIR to an ai2-helmet checkout.")

    cmd = [str(fetch_script)]
    for eid in experiment_ids:
        cmd += ["-e", eid]
    proc = subprocess.run(cmd, cwd=helmet_dir, stdout=subprocess.PIPE, text=True, check=True)
    return list(csv.DictReader(io.StringIO(proc.stdout)))


def fetch_all(jobs: list[dict], helmet_dir: Path) -> list[dict]:
    """
    Fetch results for ``jobs``, keeping think and nothink runs distinct.

    fetch_results.py names each row by ``model_config.model``, which does NOT
    encode the "(nothink)" eval marker, so think and nothink runs of the same
    checkpoint would otherwise collide. We fetch the two groups separately and
    append a ``_nothink`` suffix to the nothink rows.
    """
    think = [j["id"] for j in jobs if "nothink" not in (j.get("description") or "").lower()]
    nothink = [j["id"] for j in jobs if "nothink" in (j.get("description") or "").lower()]

    rows = []
    if think:
        rows += fetch_results(think, helmet_dir)
    for row in fetch_results(nothink, helmet_dir) if nothink else []:
        row = dict(row)
        row["model"] = f"{row['model']}_nothink"
        rows.append(row)
    return rows


def tidy(rows: list[dict]) -> "dict[str, dict[str, str]]":
    """Reshape fetch_results rows (one per model x length) into model -> {label: score}."""
    scores: dict[str, dict[str, str]] = {}
    length_label = {length: label for length, label in LENGTHS}
    for row in rows:
        model = row["model"]
        label = length_label.get(int(row["input_max_length"]))
        if label is None:
            continue  # a context length we don't track
        value = row.get("helmet_average", "")
        if value != "":
            scores.setdefault(model, {})[label] = value
    return scores


def read_existing(path: str) -> "dict[str, dict[str, str]]":
    """Read a previously written helmet CSV into model -> {label: score} (or {})."""
    if not os.path.exists(path):
        return {}
    scores: dict[str, dict[str, str]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            model = row.get("modelname")
            if not model:
                continue
            scores[model] = {
                label: row[label] for _, label in LENGTHS if row.get(label, "") != ""
            }
    return scores


def merge(existing: "dict[str, dict[str, str]]", new: "dict[str, dict[str, str]]") -> list[list[str]]:
    """
    Merge ``new`` scores into ``existing``, preserving every existing model.

    New non-empty cells override existing ones; existing cells are kept where the
    new fetch has no value. Existing models keep their original order; freshly
    seen models are appended.
    """
    order = list(existing.keys()) + [m for m in new if m not in existing]
    header = ["modelname"] + [label for _, label in LENGTHS]
    out = [header]
    for model in order:
        merged = {**existing.get(model, {}), **new.get(model, {})}
        out.append([model] + [merged.get(label, "") for _, label in LENGTHS])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
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
        default=str(Path(__file__).resolve().parent.parent / "results" / "helmet_results.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--helmet-dir",
        default=os.environ.get(
            "HELMET_DIR", str(Path(__file__).resolve().parent.parent.parent / "ai2-helmet")
        ),
        help="ai2-helmet checkout with fetch_results.py.",
    )
    args = parser.parse_args()

    experiments = list_experiments(args.workspace)
    jobs = find_helmet_jobs(experiments, args.user, args.days)
    if not jobs:
        sys.exit(f"No succeeded HELMET jobs for user '{args.user}' in the last {args.days} days.")

    print(f"Found {len(jobs)} HELMET model(s); fetching results...", file=sys.stderr)
    new = tidy(fetch_all(jobs, Path(args.helmet_dir)))
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
