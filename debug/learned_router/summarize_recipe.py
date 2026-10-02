"""One row per task for a canonical-recipe run vs gold_fl20p8_noslot at 2k (the iteration log's table).

    python debug/learned_router/summarize_recipe.py --res results_router_v5 --name v5_s0

Per task: the bar (T2/T, dCE); the router at per-row EXACT T2/T = the bar's (``@c``) and at the
val-matched global threshold (``_c<c_B>``), each paired vs the bar +- SE; train/val paired vs the exact
heuristic (``diag``); the summarize_vsfl verdict over all router points. eval_size 16 -- flagged.
"""
import argparse
import glob
import json
import os

from summarize_rel import test_points

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = os.path.join(os.path.dirname(HERE), "devloss_grid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results_router_v5")
    ap.add_argument("--name", default="v5_s0")
    a = ap.parse_args()
    tally = {}
    print("| task | bar ×c ΔCE | exact T2/T (@c) paired ± SE | val-matched ×c paired ± SE | train / val paired vs heur | verdict |")
    print("|---|---|---|---|---|---|")
    for f in sorted(glob.glob(os.path.join(GRID, a.res, f"*_{a.name}_2k.json"))):
        t = os.path.basename(f)[: -len(f"_{a.name}_2k.json")]
        tp = test_points(f)
        at = [p for p in tp["points"] if "@c" in p["scheme"]]
        vm = [p for p in tp["points"] if "_c" in p["scheme"] and "@c" not in p["scheme"]]
        vm = min(vm, key=lambda p: abs(p["comp"] - tp["cB"])) if vm else None
        run = os.path.join(HERE, "runs", t, f"{a.name}_rhobar.json")
        dg = json.load(open(run)).get("diag", {}) if os.path.exists(run) else {}
        fm = lambda p: (f"×{p['comp']:.3f} {p['paired']:+.3f} ± {p['se']:.3f} (med {p['median']:+.3f}, max {p['max']:+.3f}, "  # noqa: E731
                        f"{p['n_bad']} rows >0.1)")
        tv = (f"{dg['train']['paired']:+.3f} / {dg['val']['paired']:+.3f}" if dg.get("train") else "—")
        print(f"| {t} | ×{tp['cB']:.3f} {tp['dB']:+.3f} | {fm(at[0]) if at else '—'} | {fm(vm) if vm else '—'} | {tv} | **{tp['verdict']}** |")
        tally[tp["verdict"]] = tally.get(tp["verdict"], 0) + 1
    print(f"\n{tally}  (eval_size 16 per task ⚠)")


if __name__ == "__main__":
    main()
