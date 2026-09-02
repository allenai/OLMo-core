"""List the study's Beaker workloads (name prefix fs-) with their job state.
    python debug/flop_scaling/beaker_status.py [prefix] [--limit N]
The beaker CLI has no experiment listing and names carry a random suffix, so this scans the
workspace through the Python client (beaker v2 API)."""
import sys
from beaker import Beaker

prefix = next((a for a in sys.argv[1:] if not a.startswith("--")), "fs-")
limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 400
b = Beaker.from_env()
ws = b.workspace.get("ai2/flex2")
for w in b.workload.list(workspace=ws, limit=limit):
    ex = w.experiment
    name = ex.name if ex else ""
    if not name.startswith(prefix):
        continue
    j = b.workload.get_latest_job(w)
    st = "?" if j is None else ("finalized" if j.status.finalized.seconds else ("running" if j.status.started.seconds else "scheduled"))
    print(f"{ex.id}\t{name}\t{st}\t{'' if j is None else j.status.exit_code}")
