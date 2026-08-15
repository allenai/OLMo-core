#!/usr/bin/env python3
"""Print the true status of Beaker experiments, one per line.

    python debug/ctc_vllm_validation/beaker/job_status.py <EXP-ID> [<EXP-ID> ...]
    python debug/ctc_vllm_validation/beaker/job_status.py --ledger <LAUNCH_LEDGER.tsv>

Exists because ``beaker experiment get``'s TABLE view lies -- it shows "running" for canceled jobs
-- so the only truth is ``jobs[-1].status`` in the JSON, and reading that inline from zsh means a
nested-quoting python -c that has broken more than once. The status dict has no single state field:
it carries timestamps (created/scheduled/started/finalized), so the state is *which keys exist*.

``exitCode`` is reported separately from the state on purpose: for evals ``exitCode=0`` does NOT
mean success -- a job whose inputs were all MISSING skips everything and still exits 0.
"""

import argparse
import csv
import json
import subprocess
import sys
from typing import List


def state_of(status: dict) -> str:
    """Reduce a Beaker job status dict to one word.

    :param status: The ``jobs[-1].status`` dict.

    :returns: One of ``FINAL`` / ``run`` / ``sched`` / ``queue``.
    """
    if "finalized" in status:
        return "FINAL"
    if "started" in status:
        return "run"
    if "scheduled" in status:
        return "sched"
    return "queue"


def rows(exp_ids: List[str]) -> None:
    """Print one status line per experiment id.

    :param exp_ids: Beaker experiment ids.
    """
    for exp in exp_ids:
        try:
            out = subprocess.run(
                ["beaker", "experiment", "get", exp, "--format", "json"],
                capture_output=True, text=True, timeout=90,
            )
            data = json.loads(out.stdout)[0]
        except Exception as e:  # noqa: BLE001 - a lookup failure is a status, not a crash
            print(f"  {exp:28s} LOOKUP-FAILED {type(e).__name__}")
            continue
        jobs = data.get("jobs") or []
        status = jobs[-1].get("status", {}) if jobs else {}
        code = status.get("exitCode", "")
        print(f"  {data['name'][:50]:52s} {state_of(status):5s} exit={code}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("exp_ids", nargs="*")
    ap.add_argument("--ledger", help="TSV with an experiment_id column")
    args = ap.parse_args()

    ids = list(args.exp_ids)
    if args.ledger:
        with open(args.ledger) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                e = (row.get("experiment_id") or "").strip()
                if e and e.startswith("01"):
                    ids.append(e)
    if not ids:
        sys.exit("no experiment ids given")
    rows(ids)


if __name__ == "__main__":
    main()
