"""Print the comma list of ``-warm`` arms whose source run has finished and that are not launched yet.

    python debug/ds64_fast2k/warm_ready.py xhdr17-warm,xhdr50-warm
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from launch_fast2k import STATE, TASK, WARM_FROM, run_name  # noqa: E402

state = json.load(open(STATE)) if os.path.exists(STATE) else {"runs": {}}
budget = sys.argv[2] if len(sys.argv) > 2 else "4M"
ready = [a for a in sys.argv[1].split(",")
         if state["runs"].get("f2k-" + WARM_FROM[a][1], {}).get("state") == "DONE"
         and run_name(TASK, a, budget) not in state["runs"]]
print(",".join(ready))
