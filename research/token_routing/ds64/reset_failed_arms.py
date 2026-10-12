"""
Drop FAILED run entries from the ds64 orchestrator state so the next cycle relaunches them.

The orchestrator only launches a run whose name is absent from ``state["runs"]`` and never retries a
``FAILED`` one, so an arm that died from a fixable cause (a micro-batch that trips the 900 s NCCL
watchdog, say) stays dead for the life of the process. Removing its entry is the supported way back.

Run this ONLY while the orchestrator is stopped -- it holds the state in memory and would overwrite
the edit on its next save.

    pkill -f orchestrate_ds64.py
    python research/token_routing/ds64/reset_failed_arms.py --arms ohdr17          # --dry-run to preview
    DS64_GEN=3 setsid nohup python research/token_routing/ds64/orchestrate_ds64.py \
        >> research/token_routing/ds64/orchestrator_ds64.log 2>&1 &

:returns: writes the pruned state back in place after a timestamped backup.
"""

import argparse
import json
import os
import shutil
import subprocess
import time

D = os.path.dirname(os.path.abspath(__file__))
STATE = f"{D}/orchestrator_ds64_state.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="", help="comma list of arms to reset; default = every FAILED run")
    ap.add_argument("--tasks", default="", help="comma list of tasks to restrict to")
    ap.add_argument("--runs", default="",
                    help="comma list of exact run names to drop regardless of state (e.g. to "
                         "relaunch queued runs under a different GPU count)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    st = json.load(open(STATE))
    arms = {a for a in args.arms.split(",") if a}
    tasks = {t for t in args.tasks.split(",") if t}
    named = {n for n in args.runs.split(",") if n}
    if named:
        drop = [n for n in named if n in st["runs"]]
        for n in named - set(drop):
            print(f"  !! not in state, skipped: {n}")
    else:
        drop = [n for n, r in st["runs"].items()
                if r.get("state") == "FAILED"
                and (not arms or r.get("arm") in arms)
                and (not tasks or r.get("task") in tasks)]

    if not drop:
        print("nothing to reset")
        return
    for n in sorted(drop):
        print(f"  reset {n}  (was {st['runs'][n].get('state')}, arm={st['runs'][n].get('arm')})")
    if args.dry_run:
        print(f"\n--dry-run: {len(drop)} entries would be removed; state untouched")
        return

    # only the write path needs the orchestrator stopped: it holds state in memory and saves over us
    # `pgrep -f` matches ANY command line containing the name -- including our own caller's, when
    # the reset and the restart are typed in one shell command. Ignore our own process tree, or the
    # guard fires spuriously and silently skips the edit.
    mine = {os.getpid(), os.getppid()}
    hits = subprocess.run(["pgrep", "-f", "orchestrate_ds64.py"], capture_output=True, text=True)
    others = [int(x) for x in hits.stdout.split() if x.isdigit() and int(x) not in mine]
    if others:
        raise SystemExit(
            f"orchestrator is still running (pid {others[0]}) -- stop it first, or it will "
            "overwrite this edit"
        )

    bak = f"{STATE}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(STATE, bak)
    for n in drop:
        st["runs"].pop(n, None)
        st["evals"].pop(n, None)
    # TWO latched flags, not one. `done` ends the loop; `finishing` is set when the campaign looks
    # complete and makes the NEXT finished harvest set `done` again -- so clearing only `done` buys
    # one cycle and then exits ALL_DONE with new work still pending (seen 2026-09-09: 12 fresh runs
    # abandoned 12 minutes after launch).
    st["done"] = False
    st["finishing"] = False
    json.dump(st, open(STATE, "w"), indent=1)
    print(f"\nremoved {len(drop)} entries; backup at {bak}")
    print("restart:  DS64_GEN=3 setsid nohup python research/token_routing/ds64/orchestrate_ds64.py "
          ">> research/token_routing/ds64/orchestrator_ds64.log 2>&1 &")


if __name__ == "__main__":
    main()
