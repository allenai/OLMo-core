#!/usr/bin/env python3
"""
Pull all RULER results from an olmo-cookbook-eval dashboard into a tidy CSV.

For every model on the dashboard it emits one row of
    modelname,4k,8k,16k,32k,64k
where each value is the aggregate ruler:Nk score (the full 13-subtask suite
averaged at that context length). The per-subtask ``::std`` columns that the
cookbook CLI emits are dropped.

The RULER evals are launched by ./launch_long_context_evals.sh, which reports to
the ``memory-LC`` dashboard by default.

Usage:
    ./pull_ruler_results.py                 # -> ../results/ruler_results.csv
    ./pull_ruler_results.py -o out.csv      # custom output path
    ./pull_ruler_results.py -d some-board    # different dashboard
    DASHBOARD=foo ./pull_ruler_results.py    # env-var override

Env vars:
    DASHBOARD        dashboard name (default: memory-LC)
    COOKBOOK_BIN     path to the olmo-cookbook-eval executable
    COOKBOOK_DIR     olmo-cookbook checkout (its .venv/bin/olmo-cookbook-eval
                     is used if COOKBOOK_BIN is unset; default: ../../olmo-cookbook,
                     i.e. a sibling of OLMo-core)
"""

import argparse
import csv
import io
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Context lengths to pull, in K. Each maps to a ruler:Nk task group.
LENGTHS_K = [4, 8, 16, 32, 64]


def resolve_cookbook_bin() -> str:
    """Find the olmo-cookbook-eval executable."""
    if bin_ := os.environ.get("COOKBOOK_BIN"):
        return bin_
    cookbook_dir = Path(
        os.environ.get(
            "COOKBOOK_DIR", Path(__file__).resolve().parent.parent.parent / "olmo-cookbook"
        )
    )
    venv_bin = cookbook_dir / ".venv" / "bin" / "olmo-cookbook-eval"
    if venv_bin.exists():
        return str(venv_bin)
    if found := shutil.which("olmo-cookbook-eval"):
        return found
    sys.exit(
        "Could not find olmo-cookbook-eval. Set COOKBOOK_BIN, or COOKBOOK_DIR to an "
        f"olmo-cookbook checkout with a .venv (looked in {venv_bin})."
    )


def fetch_dashboard_csv(cookbook_bin: str, dashboard: str) -> str:
    """Run the cookbook CLI and return its raw CSV (stdout only)."""
    cmd = [cookbook_bin, "results", "-d", dashboard, "-f", "csv"]
    for k in LENGTHS_K:
        cmd += ["-t", f"ruler:{k}k"]
    # tqdm progress goes to stderr; let it pass through to our stderr so the
    # user sees progress while stdout stays clean CSV.
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, check=True)
    return proc.stdout


def tidy(raw_csv: str) -> list[list[str]]:
    """Keep only name + aggregate ruler:Nk columns, renamed to Nk."""
    reader = csv.DictReader(io.StringIO(raw_csv))
    if reader.fieldnames is None or "name" not in reader.fieldnames:
        sys.exit("Unexpected CSV from cookbook CLI (no 'name' column):\n" + raw_csv[:500])

    # Only keep lengths that actually appear on the dashboard.
    present = [k for k in LENGTHS_K if f"ruler:{k}k" in reader.fieldnames]
    header = ["modelname"] + [f"{k}k" for k in present]

    rows = [header]
    for row in reader:
        rows.append([row["name"]] + [row.get(f"ruler:{k}k", "") for k in present])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-d", "--dashboard", default=os.environ.get("DASHBOARD", "memory-LC"), help="Dashboard name."
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(Path(__file__).resolve().parent.parent / "results" / "ruler_results.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    cookbook_bin = resolve_cookbook_bin()
    raw = fetch_dashboard_csv(cookbook_bin, args.dashboard)
    rows = tidy(raw)

    with open(args.output, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    print(f"Wrote {len(rows) - 1} models x {len(rows[0]) - 1} lengths to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
