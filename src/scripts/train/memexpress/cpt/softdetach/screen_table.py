"""Screening table for the soft-detach CPT strategies (records/softdetach-cpt-plan.md): reads the
dev-loss driver JSONs under debug/devloss_grid/results_screen (+ the earlier grid dirs for schemes
already scored there) and prints strategy x rung -> CE, dCE vs full, compaction, selection cost.

    python src/scripts/train/memexpress/cpt/softdetach/screen_table.py [--roots a,b] [--task cpt80]
"""
import argparse, glob, json, os
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *([".."] * 6)))
ORDER = ["full", "k0", "rand20", "fl20", "first64", "first128", "rule20", "grad20", "attnrow20", "fl64", "gold_fl64"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", default=f"{REPO}/debug/devloss_grid/results_screen,{REPO}/debug/devloss_grid/results,{REPO}/debug/devloss_grid/results_fl20")
    ap.add_argument("--task", default="cpt80", help="cpt80, or 'ctc' = mean over the CTC rows")
    a = ap.parse_args()
    cells = {}  # (task, rung) -> {scheme: summary}
    for root in a.roots.split(","):
        for f in sorted(glob.glob(os.path.join(root, "*.json"))):
            d = json.load(open(f))
            key = (d["task"].replace("ctc_", ""), d["rung"])
            c = cells.setdefault(key, {})
            for s, m in d["summary"].items():
                # a driver warning names its selector ("grad: ...", "attnrow: ...") -> attach to that scheme only
                w = [x for x in d.get("warnings", []) if s.startswith(x.split(":")[0])]
                c.setdefault(s, dict(m, warnings=w, eval_size=d["eval_size"]))
    rungs = ["2k", "8k", "32k"]
    if a.task == "cpt80":
        tasks = ["cpt80"]
    else:
        tasks = sorted({t for t, _ in cells if t != "cpt80"})
    print(f"task(s): {tasks}")
    hdr = f"{'scheme':10} " + " | ".join(f"{r:>7} {'dCE':>7} {'x':>5} {'sel':>4}" for r in rungs)
    print(hdr)
    for s in ORDER:
        parts = []
        for r in rungs:
            ces, dces, comps, sels, n = [], [], [], [], 0
            for t in tasks:
                c = cells.get((t, r), {})
                if s in c and "full" in c:
                    ces.append(c[s]["ce"]); dces.append(c[s]["ce"] - c["full"]["ce"]); comps.append(c[s]["compaction"])
                    sels.append(c[s].get("sel_cost", 0.0) or 0.0); n += 1
            if n:
                parts.append(f"{np.mean(ces):7.3f} {np.mean(dces):+7.3f} {np.mean(comps):5.2f} {np.mean(sels):4.1f}" + ("" if n == len(tasks) else f"*{n}"))
            else:
                parts.append(f"{'—':>7} {'':>7} {'':>5} {'':>4}")
        print(f"{s:10} " + " | ".join(parts))
    warns = {(k, s): c[s]["warnings"] for k, c in cells.items() for s in c if c[s].get("warnings")}
    for (k, s), w in sorted(warns.items()):
        if s in ORDER:
            print(f"  ⚠ {k[0]}@{k[1]} {s}: {w}")


if __name__ == "__main__":
    main()
