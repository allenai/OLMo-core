"""Small state queries for ``poll_fast2k.sh``.

    python debug/ds64_fast2k/pending.py needs_eval   # runs that finished but have no eval yet
    python debug/ds64_fast2k/pending.py summary      # ALLDONE, or "PENDING <n>"
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from launch_fast2k import STATE  # noqa: E402

st = json.load(open(STATE)) if os.path.exists(STATE) else {"runs": {}, "evals": {}}
runs, evals = st["runs"], st["evals"]
if sys.argv[1] == "needs_eval":
    print(",".join(n for n, r in runs.items() if r.get("state") == "DONE" and n not in evals))
else:
    bad = [n for n, r in runs.items() if r.get("state") not in ("DONE", "FAILED")]
    bad += [n for n, r in runs.items()
            if r.get("state") == "DONE" and evals.get(n, {}).get("state") not in ("DONE", "FAILED")]
    print("ALLDONE" if not bad else "PENDING %d" % len(bad))
