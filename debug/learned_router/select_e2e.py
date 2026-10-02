"""Post-hoc selection among the end-to-end rho routers of one task, on the VAL rows only.

    python debug/learned_router/select_e2e.py --task nq

For tau in {0.05, 0.1, 0.2}: the most compact router (final-epoch val T2/T, exact hard removal,
deterministic gate) whose val mean dCE <= max(tau * mean CE_full(val), 0.02 nats); keep-all
(compaction 1, dCE 0) is always an allowed candidate. Writes ``weights/<task>/e2e_tau<tau>.pt`` (a
copy of the chosen rho router, or a keep-all router: bias +30, all weights 0) and
``runs/<task>/e2e_selection.json``.
"""
import argparse
import glob
import json
import os
import re
import shutil

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
FLOOR = 0.02
TAUS = ("0.05", "0.1", "0.2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    a = ap.parse_args()
    runs = {}
    for f in glob.glob(os.path.join(HERE, "runs", a.task, "e2e_rho*.json")):
        m = re.fullmatch(r"e2e_rho([0-9.]+)\.json", os.path.basename(f))
        if m:
            runs[m.group(1)] = json.load(open(f))
    if not runs:
        raise SystemExit(f"no e2e runs for {a.task}")
    cef = next(iter(runs.values()))["ce_full_val_mean"]
    cands = [{"name": "keep_all", "comp": 1.0, "dce": 0.0}] + [
        {"name": f"e2e_rho{r}", "comp": v["final_val"]["comp"], "dce": v["final_val"]["dce"], "keep": v["final_val"]["keep"]}
        for r, v in sorted(runs.items(), key=lambda kv: float(kv[0]))]
    sel = {"task": a.task, "ce_full_val_mean": cef, "floor": FLOOR, "candidates": cands, "selected": {}}
    wdir = os.path.join(HERE, "weights", a.task)
    for tau in TAUS:
        tol = max(float(tau) * cef, FLOOR)
        ok = [c for c in cands if c["dce"] <= tol]
        pick = min(ok, key=lambda c: (c["comp"], c["dce"]))
        dst = os.path.join(wdir, f"e2e_tau{tau}.pt")
        if pick["name"] == "keep_all":
            st = torch.load(glob.glob(os.path.join(wdir, "e2e_rho*.pt"))[0], map_location="cpu")
            for k in ("w_pos", "w_gold", "w_emb"):
                st[k] = torch.zeros_like(st[k])
            st["b"] = torch.full_like(st["b"], 30.0)
            torch.save(st, dst)
        else:
            shutil.copyfile(os.path.join(wdir, pick["name"] + ".pt"), dst)
        sel["selected"][tau] = {"tol": tol, **pick}
        print(f"[select] {a.task} tau {tau}: tol {tol:.4f} -> {pick['name']} (val T2/T {pick['comp']:.3f}, dCE {pick['dce']:+.4f})")
    json.dump(sel, open(os.path.join(HERE, "runs", a.task, "e2e_selection.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
