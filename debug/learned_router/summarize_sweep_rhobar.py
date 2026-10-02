"""Summarise rho-matched e2e runs (any prefix, e.g. the per-row residual sweep) vs the exact heuristic on
train/val and vs gold_fl20p8_noslot on the test rows.

    python debug/learned_router/summarize_sweep_rhobar.py --task rerank --prefix e2eres --res results_router_resid

Per run ``runs/<task>/<prefix>*_rhobar.json``: train / val paired dCE vs the heuristic at equal T2/T (``diag``),
train-val gap, val gold keep; test file ``<res>/<task>_<name minus _rhobar>_2k.json`` -> router at the bar's
T2/T (paired +- SE) and the summarize_vsfl verdict. eval_size 16 -- flagged.
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
    ap.add_argument("--task", required=True)
    ap.add_argument("--prefix", default="e2eres")
    ap.add_argument("--res", default="results_router_resid")
    a = ap.parse_args()
    print("| run | train paired vs heur | val paired vs heur | gap | val gold keep r/h | test @c_B ×c ΔCE (paired ± SE) | test @0.7c_B | verdict |")
    print("|---|---|---|---|---|---|---|---|")
    for f in sorted(glob.glob(os.path.join(HERE, "runs", a.task, f"{a.prefix}*_rhobar.json"))):
        r = json.load(open(f))
        nm = os.path.basename(f)[: -len("_rhobar.json")]
        dg = r.get("diag") or {}
        tr, va = dg.get("train", {}), dg.get("val", {})
        tf = os.path.join(GRID, a.res, f"{a.task}_{nm}_2k.json")
        tp = test_points(tf) if os.path.exists(tf) else None
        pts = sorted([p for p in (tp["points"] if tp else []) if "_c" in p["scheme"]], key=lambda p: -p["comp"])
        fm = lambda p: f"×{p['comp']:.3f} {p['dce']:+.3f} ({p['paired']:+.3f} ± {p['se']:.3f})"  # noqa: E731
        gk, gh = va.get("gold_keep_router"), va.get("gold_keep_heuristic")
        if tp is None:
            print(f"| {nm} | pending |")
            continue
        nan = float("nan")
        print(f"| {nm} | {tr.get('paired', nan):+.3f} ± {tr.get('paired_se') or nan:.3f} | {va.get('paired', nan):+.3f} ± "
              f"{va.get('paired_se') or nan:.3f} | {va.get('paired', nan) - tr.get('paired', nan):+.3f} | "
              f"{'—' if gk is None else f'{gk:.2f}/{gh:.2f}'} | {fm(pts[0]) if pts else '—'} | {fm(pts[1]) if len(pts) > 1 else '—'} | "
              f"{tp['verdict']} (bar ×{tp['cB']:.3f} {tp['dB']:+.3f}) |")

if __name__ == "__main__":
    main()
