"""
Pareto analysis for olmo3_cptdrop: dev CE (eval_devce / merge_devce JSONs) vs training compute, and the FLOP
multiplier m = PF_dense(CE_arm) / PF_arm read off the measured dense curve at the same context length.

Training PF per run = sum over its segments of steps x measured PF/step of that segment's arm (the smoke FLOP
meter; refreshed from flops.json of finished runs when present). Using per-step costs makes main runs, decay
branches (which resume a main run's step counter) and handoff runs (drop steps then dense steps) consistent.

The dense curve is piecewise log-linear in PF between measured dense points; an arm whose CE is below every dense
point gets m reported as a LOWER bound ("> m at the largest dense point"), never an extrapolation; above every
dense point gets an upper bound.

    python analyze.py --devce-dir DIR [--devce-dir DIR2 ...] --out-json pareto.json --out-md pareto.md
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re

# Measured PF per optimizer step (524,288 tokens): FLOP-meter smoke runs on cubbins, 2026-10-21.
PF_PER_STEP = {
    ("D", "32k"): 36.558, ("IRAND", "32k"): 6.726, ("IFL", "32k"): 8.137, ("C1", "32k"): 32.547,
}
# Estimates until measured (relative to dense at the same ctx): keep 10% / 5% I-fl, and 64k dense.
EST = {
    ("IFL10", "32k"): 0.09 * 36.558, ("IFL05", "32k"): 0.046 * 36.558,
    ("D", "64k"): 1.13 * 36.558,
}
EST[("IFL10", "64k")] = 0.073 * EST[("D", "64k")]
EST[("IFL05", "64k")] = 0.036 * EST[("D", "64k")]
EST[("IFL", "64k")] = 0.19 * EST[("D", "64k")]


def segments(label: str):
    """[(arm_key, ctx, steps)] for a run label (o3cd- prefix stripped). None if not a training run."""
    ctx = "64k" if "x64" in label or label.startswith("D64") else "32k"
    m = re.fullmatch(r"(D|DLOW|D64|D64LOW|D64e|D64lr\d+|Dlr\d+|DB\d+lr\d+)-(\d+)(-s\d+)?", label)
    if m:
        return [("D", ctx, int(m.group(2)))]
    m = re.fullmatch(r"(IRAND|IFL)(?:lr\d+)?-(\d+)", label)
    if m:
        return [(m.group(1), ctx, int(m.group(2)))]
    m = re.fullmatch(r"(IFL10|IFL05|IFL25)(x64)?-(\d+)", label)
    if m:
        arm = "IFL" if m.group(1) == "IFL25" else m.group(1)
        return [(arm, ctx, int(m.group(3)))]
    m = re.fullmatch(r"C-(\d+)", label)
    if m:
        b = int(m.group(1))
        return [("C1", ctx, b // 2), ("D", ctx, b // 2)]
    m = re.fullmatch(r"HO(?:lr\d+)?-(\d+)", label)
    if m:
        b = int(m.group(1))
        s0 = {120: 390, 240: 585, 480: 1170}[b]
        ds = round(b - s0 * 0.184)
        return [("IRAND", ctx, s0), ("D", ctx, ds)]
    return None


def pf_of(segs, table):
    tot, est = 0.0, False
    for arm, ctx, steps in segs:
        k = (arm, ctx)
        if k in table:
            tot += steps * table[k]
        elif k in EST:
            tot += steps * EST[k]
            est = True
        else:
            raise KeyError(k)
    return tot, est


def family(label: str, ref: str) -> str:
    """'dense' for the reference dense curve (regex `ref` on the label prefix); every other dense LR / variant is
    its own family, so the multiplier is always read off one dense recipe (2026-10-22: peak LR 2e-5, Dlr20/D64lr20)."""
    pre = re.sub(r"-\d+(-s\d+)?$", "", label)
    return "dense" if re.fullmatch(ref, pre) else pre


def interp_pf(dense, ce):
    """PF at which the dense curve reaches `ce` (dense: [(pf, ce)] sorted by pf). Returns (pf, kind)."""
    pts = sorted(dense)
    # dense CE should fall with PF; use the monotone envelope so noise can't create two crossings
    env, best = [], math.inf
    for pf, c in pts:
        best = min(best, c)
        env.append((pf, best))
    if ce < env[-1][1]:
        return env[-1][0], "lower_bound"   # arm beats every dense point: m > PF_max / PF_arm
    if ce >= env[0][1]:
        return env[0][0], "upper_bound"
    for (p0, c0), (p1, c1) in zip(env, env[1:]):
        if c1 <= ce <= c0 and c0 != c1:
            t = (c0 - ce) / (c0 - c1)
            return math.exp(math.log(p0) + t * (math.log(p1) - math.log(p0))), "interp"
    return env[0][0], "upper_bound"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devce-dir", action="append", required=True)
    ap.add_argument("--shard", default=None, help="dev shard name (default: dev32k_1024 / dev64k_512 by ctx)")
    ap.add_argument("--bin", default="overall", help="overall or a position bin, e.g. 16k-32k")
    ap.add_argument("--ref", default=r"Dlr20|D64lr20", help="regex for the reference dense family's label prefix")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    a = ap.parse_args()
    table = dict(PF_PER_STEP)
    rows = []
    for d in a.devce_dir:
        for f in glob.glob(os.path.join(d, "*.json")):
            j = json.load(open(f))
            if "summary" not in j or "label" not in j:
                continue
            label = j["label"].replace("o3cd-", "")
            dev = os.path.basename(j["dev"].rstrip("/"))
            segs = segments(label)
            if segs is None or not dev.startswith("dev"):
                continue
            ctx = segs[0][1]
            if a.shard and dev != a.shard:
                continue
            if not a.shard and dev != ("dev64k_512" if ctx == "64k" else "dev32k_1024"):
                continue
            s = j["summary"]["overall"] if a.bin == "overall" else j["summary"]["bins"].get(a.bin)
            if s is None:
                continue
            pf, est = pf_of(segs, table)
            rows.append({"label": label, "family": family(label, a.ref), "ctx": ctx, "pf": pf, "pf_estimated": est,
                         "ce": s["ce"], "row_se": s.get("row_se"), "eval_size": j["eval_size"], "file": f})
    out = []
    for ctx in sorted({r["ctx"] for r in rows}):
        dense = [(r["pf"], r["ce"]) for r in rows if r["ctx"] == ctx and r["family"] == "dense"]
        for r in sorted((r for r in rows if r["ctx"] == ctx), key=lambda r: (r["family"], r["pf"])):
            if r["family"] != "dense" and len(dense) >= 2:
                pfd, kind = interp_pf(dense, r["ce"])
                r["m"] = pfd / r["pf"]
                r["m_kind"] = kind
            out.append(r)
    json.dump(out, open(a.out_json, "w"), indent=1)
    with open(a.out_md, "w") as f:
        f.write(f"| ctx | run | family | train PF | dev CE ({a.bin}) | m vs dense | note |\n|---|---|---|---|---|---|---|\n")
        for r in out:
            m = r.get("m")
            ms = "" if m is None else (f"> {m:.1f}" if r["m_kind"] == "lower_bound" else
                                       f"< {m:.1f}" if r["m_kind"] == "upper_bound" else f"{m:.1f}")
            note = ("PF estimated; " if r["pf_estimated"] else "") + f"eval_size={r['eval_size']} rows"
            f.write(f"| {r['ctx']} | {r['label']} | {r['family']} | {r['pf']:.0f} | {r['ce']:.4f} | {ms} | {note} |\n")
    print(open(a.out_md).read())


if __name__ == "__main__":
    main()
