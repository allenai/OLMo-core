"""Held-out test of the frozen-checkpoint predictors of trained keep-rule quality.

The combined score (z(-gold leak) + z(-dCE)) was picked on 17 ds64 rules (predict_points_2026-10-21.json). Here its
z-scaling is FROZEN from those 17 (per checkpoint) and applied, unchanged, to ds64 arms that were not used to pick it
(gsimheld_*.json). Outcome = best measured ds64 delta f1 vs dense at matched FLOPs (extrapolated only if no measured
point). Tasks without gold labels (oolong) get leak 0: there is no gold to leak.

    python research/token_routing/learned_router/gradprobe/heldout_predict.py --ds64-html <saved Soft-KV page> \
        --fit predict_points_2026-10-21.json --out heldout_points_2026-10-21.json /path/gsimheld_*_8k_r24.json
"""

import argparse
import glob
import json
import os

import numpy as np

# probe rule name -> (ds64 arm(s), keeps gold whole?)
MAP = {
    "contradiction": {"runs8.3": (["runs08"], True), "runs2.8": (["runs03"], True)},
    "nq": {"gb33.3": (["kvgb"], False), "kv16.7": (["nqcc17"], True), "kv8.3": (["nqcc08"], True)},
    "outlier": {"ck32": (["ck32"], False), "flk32": (["flk32"], False), "kv16.7": (["kv17"], True),
                "gb50": (["kvgb50"], False), "gbh50": (["xh2k50"], False), "gbh8.3": (["cc08"], False),
                "gbh2.8": (["cc03"], False)},
    # ohdr08 and occ08 have the same mask (they differ only in the slot construction): two outcomes, one x
    "oolong": {"gbh33.3": (["ohdr33"], False), "gbh16.7": (["ohdr17"], False), "gbh8.3": (["ohdr08", "occ08"], False),
               "gbh0": (["occ00"], False)},
}


def outcomes(html_path):
    h = open(html_path).read()
    i = h.index("const D = ") + len("const D = ")
    D, _ = json.JSONDecoder().raw_decode(h[i:])
    out = {}
    for t in D["tasks"]:
        for a in t["arms"]:
            pts = [p for p in a["points"] if p.get("delta") is not None]
            if pts:
                meas = [p for p in pts if not p["extrapolated"]]
                best = max(meas or pts, key=lambda p: p["delta"])
                out[(t["task"], a["arm"])] = dict(y=round(best["delta"], 4), budget=best["budget"], extrap=not meas)
    return out


def leak_whole(row, q):
    gold = set(row["gold_docs"])
    if not gold:
        return 0.0
    L, K = np.asarray(row["doc_len"], float), np.asarray(q["doc_kept"], float)
    n = min(len(L), len(K))
    w = (K[:n] >= L[:n]).astype(float)
    ok = L[:n] > 0
    g = [w[i] for i in range(n) if i in gold and ok[i]]
    ng = [w[i] for i in range(n) if i not in gold and ok[i]]
    if not g or not ng:
        return float("nan")
    ga, na = np.array(g)[:, None], np.array(ng)[None, :]
    return float(((ga > na).sum() + 0.5 * (ga == na).sum()) / (ga.size * na.size) - 0.5)


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds64-html", required=True)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    Y = outcomes(a.ds64_html)
    fit = [p for p in json.load(open(a.fit)) if p["rule"] != "bar"]
    pts = []
    for f in a.files:
        d = json.load(open(f))
        ck = "setA32k" if "setA32k" in os.path.basename(f) else "per-task"
        task = os.path.basename(f).split("_")[2]
        acc = {}
        for row in d["rows"]:
            for q in row["points"]:
                if q["rule"] in MAP.get(task, {}):
                    acc.setdefault(q["rule"], []).append((q["cos"], q["dce"], leak_whole(row, q), q["rho"]))
        for rule, v in acc.items():
            arr = np.array(v, float)
            arms, gf = MAP[task][rule]
            for arm in arms:
                pts.append(dict(ckpt=ck, task=task, rule=arm, probe_rule=rule, gold_forced=gf, n=len(v),
                                keep=round(float(arr[:, 3].mean()), 3), cos=round(float(np.median(arr[:, 0])), 3),
                                dce=round(float(np.mean(arr[:, 1])), 3), leak=round(float(np.nanmedian(arr[:, 2])), 3),
                                heldout=True, **Y[(task, arm)]))
    json.dump(pts, open(a.out, "w"), indent=0)
    print(f"{len(pts)} held-out points -> {a.out}\n")
    print("| checkpoint | set | n | cos | -dCE | -leak | combined (frozen z) |")
    print("|---|---|---|---|---|---|---|")
    for ck in ("per-task", "setA32k"):
        F = [p for p in fit if p["ckpt"] == ck]
        if not F:
            continue
        mu = {k: np.mean([-p[k] for p in F]) for k in ("leak", "dce")}
        sd = {k: np.std([-p[k] for p in F]) + 1e-9 for k in ("leak", "dce")}
        comb = lambda p: (-p["leak"] - mu["leak"]) / sd["leak"] + (-p["dce"] - mu["dce"]) / sd["dce"]
        for name, S in (("held-out only", [p for p in pts if p["ckpt"] == ck]),
                        ("original 17", F),
                        ("all", F + [p for p in pts if p["ckpt"] == ck])):
            if len(S) < 3:
                continue
            y = [p["y"] for p in S]
            row = [spearman([s * p[k] for p in S], y) for k, s in (("cos", 1), ("dce", -1), ("leak", -1))]
            row.append(spearman([comb(p) for p in S], y))
            print(f"| {ck} | {name} | {len(S)} | " + " | ".join(f"{v:.2f}" for v in row) + " |")


if __name__ == "__main__":
    main()
