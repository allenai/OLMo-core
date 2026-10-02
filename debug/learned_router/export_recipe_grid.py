"""Export one canonical-recipe router per task into dev-loss-grid format with canonical scheme names.

    python debug/learned_router/export_recipe_grid.py --label v5 --spec rerank=results_router_v5:v5_s0 ... 

For each task, the recipe's results file (``<res>/<task>_<name>_2k.json``) is copied into
``debug/devloss_grid/results_router_recipe/<task>_2k.json`` with only ``full``, ``gold_fl20p8_noslot`` and two
router rows renamed:
  ``router_<label>@exact`` -- per-row top-k at exactly the bar's T2/T (``router_<name>_rhobar@c<x>``, from the
                              same file or from ``results_router_atc/<task>_<atc>_2k.json``)
  ``router_<label>@val``   -- the label-free global threshold matched on val rows to the bar's T2/T
                              (``router_<name>_rhobar_c<x>`` with x closest to the bar's T2/T).
"""
import argparse
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = os.path.join(os.path.dirname(HERE), "devloss_grid")
BAR = "gold_fl20p8_noslot"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="v5")
    ap.add_argument("--spec", nargs="+", required=True, help="task=res_dir:name[:atc_stem]")
    ap.add_argument("--out", default=os.path.join(GRID, "results_router_recipe"))
    ap.add_argument("--file-tag", default="", help="suffix for the output file stem (several labels side by side)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for sp in a.spec:
        task, rest = sp.split("=")
        parts = rest.split(":")
        res, name = parts[0], parts[1]
        atc = parts[2] if len(parts) > 2 else None
        d = json.load(open(os.path.join(GRID, res, f"{task}_{name}_2k.json")))
        cB = float(np.mean(d["per_row"][BAR]["compaction"]))
        keep = {"full": "full", BAR: BAR}
        vals = [s for s in d["per_row"] if re.fullmatch(rf"router_{re.escape(name)}_rhobar_c[0-9.]+", s)]
        if vals:
            keep[min(vals, key=lambda s: abs(float(s.rsplit("_c", 1)[1]) - cB))] = f"router_{a.label}@val"
        ex = [s for s in d["per_row"] if s.startswith(f"router_{name}_rhobar@c")]
        src_ex = d
        if not ex and atc:
            src_ex = json.load(open(os.path.join(GRID, "results_router_atc", f"{task}_{atc}_2k.json")))
            ex = [s for s in src_ex["per_row"] if "@c" in s]
            assert np.allclose(src_ex["per_row"]["full"]["ce"], d["per_row"]["full"]["ce"], atol=1e-3), f"{task}: test rows differ"
        out = dict(d)
        out["per_row"], out["summary"] = {}, {}
        for s_old, s_new in keep.items():
            out["per_row"][s_new] = d["per_row"][s_old]
            out["summary"][s_new] = d["summary"][s_old]
        pairs = [s for s in d["per_row"] if s == f"router_{name}_rhobar@pair"]
        if pairs:  # paired budget: per row, as many routed tokens as the bar keeps on that row
            out_pair = (d["per_row"][pairs[0]], d["summary"][pairs[0]])
        else:
            out_pair = None
        if ex:
            s_old = min(ex, key=lambda s: abs(float(s.rsplit("@c", 1)[1]) - cB))
            out["per_row"][f"router_{a.label}@exact"] = src_ex["per_row"][s_old]
            out["summary"][f"router_{a.label}@exact"] = src_ex["summary"][s_old]
        if out_pair is not None:
            out["per_row"][f"router_{a.label}@pair"], out["summary"][f"router_{a.label}@pair"] = out_pair
        out["schemes"] = {v: d["schemes"].get(k) for k, v in keep.items()}
        out["recipe_export"] = {"label": a.label, "source": f"{res}/{task}_{name}_2k.json", "exact_from": None if src_ex is d else atc}
        json.dump(out, open(os.path.join(a.out, f"{task}{a.file_tag}_2k.json"), "w"), indent=1)
        print(f"{task}: {sorted(out['per_row'])}")


if __name__ == "__main__":
    main()
