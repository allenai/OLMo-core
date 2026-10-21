"""Does a frozen-checkpoint metric predict which keep rules TRAIN well?

Reads grad_similarity.py JSONs (gsimpred*_{task}_8k_r*.json) and joins each rule to its measured training
outcome (ds64 campaign + tokdrop_lr; OUTCOME below). Per rule and checkpoint it reports the median gradient cosine
(dropped vs full context), median log10 norm ratio, mean dCE, and three mask-only gold-leak scores:
  leak_frac   AUC of per-doc kept FRACTION, gold vs non-gold (stored per point as gold_leak_auc)
  leak_count  AUC of per-doc kept token COUNT (the length the model actually sees)
  leak_whole  AUC of "doc kept whole" (all body tokens kept)
then, for each metric, how well it separates rules that won from rules that lost (AUC over rules, 0.5 = no signal).

    python research/token_routing/learned_router/gradprobe/predict_report.py /path/to/gsimpredq_*_8k_r24.json
"""

import json
import os
import sys

import numpy as np

# (task, rule name in the JSON) -> (outcome, evidence). W = trained Pareto/near-parity winner, L = lost or collapsed.
OUTCOME = {
    ("contradiction", "ck16"): ("W", "ds64 chk16: 0.910 @64M vs dense, dominates dense-32M at 1.8x fewer FLOPs"),
    ("contradiction", "ck32"): ("W", "ds64 chk32: dominates dense-32M at 2.4x fewer FLOPs"),
    ("contradiction", "ck64"): ("W", "ds64 chk64: dominates dense-16M at 2.2x fewer FLOPs"),
    ("contradiction", "hdr33"): ("W", "ds64 hdr33: hdr33-32M beats dense-16M at 1.7x fewer FLOPs"),
    ("contradiction", "hdr17"): ("L", "ds64 hdr17: -0.04 vs dense at matched FLOPs (64M)"),
    ("contradiction", "hdr8"): ("L", "ds64 hdr08: -0.28 at matched FLOPs"),
    ("contradiction", "hdr3"): ("L", "ds64 hdr03: -0.59 at matched FLOPs"),
    ("contradiction", "goldhfl"): ("L", "tokdrop_lr bar: 8k <=0.07 vs dense 0.89 at every LR"),
    ("nq", "kv33"): ("W", "ds64 kv33: 0.878 vs dense-32M 0.865 at 2.5x fewer FLOPs (nq frontier)"),
    ("nq", "kv17"): ("L", "ds64 kv17: below kv33 at every rung, -0.19 at 64k vs dense same budget"),
    ("nq", "ck32"): ("L", "ds64 nqk32: -0.04..-0.23 vs dense same budget"),
    ("nq", "ck64"): ("L", "ds64 nqk64: -0.02..-0.17 vs dense same budget"),
    ("nq", "goldhfl"): ("L", "tokdrop_lr bar: 8k 0.742 vs dense 0.898 (best LRs)"),
    ("outlier", "flk64"): ("W", "ds64 flk64: largest matched-FLOP gain (+0.32 interpolated)"),
    ("outlier", "ck64"): ("W", "ds64 ck64-32M dominates dense at 4.0x fewer FLOPs"),
    ("outlier", "ck16"): ("L", "ds64 ck16: -0.49 at 16k vs dense same budget"),
    ("outlier", "ck0"): ("L", "ds64 cc00 (headers only): -0.78 at 8k"),
    ("outlier", "kv33"): ("L", "ds64 kv33: -0.78 at 8k (gold-forcing collapse)"),
    ("outlier", "gb33"): ("L", "ds64 kvgb: -0.60 at 8k"),
    ("outlier", "goldhfl"): ("L", "tokdrop_lr bar: 8k <=0.07 vs dense 0.30"),
}


def auc(pos, neg):
    """P(pos > neg) + 0.5 P(tie)."""
    pos, neg = np.asarray(pos, float)[:, None], np.asarray(neg, float)[None, :]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    return float(((pos > neg).sum() + 0.5 * (pos == neg).sum()) / (pos.size * neg.size))


def doc_leaks(row, pt):
    gold = set(row["gold_docs"])
    L, K = np.asarray(row.get("doc_len", []), float), np.asarray(pt.get("doc_kept", []), float)
    n = min(len(L), len(K))
    if not gold or n == 0:
        return float("nan"), float("nan")
    L, K = L[:n], K[:n]
    ok = L > 0
    g = np.array([i in gold for i in range(n)]) & ok
    ng = ~np.array([i in gold for i in range(n)]) & ok
    whole = (K >= L).astype(float)
    return auc(K[g], K[ng]), auc(whole[g], whole[ng])


def main(paths):
    recs = []
    for p in paths:
        d = json.load(open(p))
        ck = "setA32k" if "setA32k" in os.path.basename(p) else "per-task"
        task = d["task"] if isinstance(d.get("task"), str) else os.path.basename(p).split("_")[2]
        per_rule = {}
        for row in d["rows"]:
            for pt in row["points"]:
                if pt["rule"] in ("hfl", "goldhfl") and abs((pt.get("x") or 0) - 20) > 1e-9:
                    continue
                lc, lw = doc_leaks(row, pt)
                per_rule.setdefault(pt["rule"], []).append(
                    (pt["cos"], np.log10(max(pt["norm_ratio"], 1e-30)), pt["dce"], pt.get("gold_leak_auc", np.nan), lc, lw,
                     pt["rho"]))
        for rule, v in per_rule.items():
            a = np.array(v, float)
            out, ev = OUTCOME.get((task, rule), ("?", ""))
            recs.append(dict(ckpt=ck, task=task, rule=rule, outcome=out, n=len(v), keep=float(np.mean(a[:, 6])),
                             cos=float(np.median(a[:, 0])), lognorm=float(np.median(a[:, 1])), dce=float(np.mean(a[:, 2])),
                             leak_frac=float(np.nanmedian(np.abs(a[:, 3] - 0.5))) if np.isfinite(a[:, 3]).any() else np.nan,
                             leak_count=float(np.nanmedian(a[:, 4] - 0.5)) if np.isfinite(a[:, 4]).any() else np.nan,
                             leak_whole=float(np.nanmedian(a[:, 5] - 0.5)) if np.isfinite(a[:, 5]).any() else np.nan,
                             evidence=ev))
    recs.sort(key=lambda r: (r["ckpt"], r["task"], r["outcome"], -r["cos"]))
    print("| ckpt | task | rule | trained outcome | keep | grad cos | log10 norm ratio | mean dCE | leak: count AUC-0.5 |"
          " leak: whole AUC-0.5 | rows |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in recs:
        print(f"| {r['ckpt']} | {r['task']} | {r['rule']} | {r['outcome']} | {r['keep']:.2f} | {r['cos']:.2f} | "
              f"{r['lognorm']:+.2f} | {r['dce']:+.3f} | {r['leak_count']:+.2f} | {r['leak_whole']:+.2f} | {r['n']} |")
    # does each metric separate W from L? (higher-is-better orientation stated per metric)
    print("\nAUC separating trained winners (W) from losers (L); 0.5 = no signal, 1.0 = perfect:\n")
    print("| metric (oriented) | per-task ckpts | setA32k | pooled |")
    print("|---|---|---|---|")
    mets = [("grad cos (higher = predicted good)", lambda r: r["cos"]),
            ("-|log norm ratio| (closer to 1x = good)", lambda r: -abs(r["lognorm"])),
            ("-dCE (dev-loss parity = good)", lambda r: -r["dce"]),
            ("-leak count (less visible-length leak = good)", lambda r: -max(r["leak_count"], 0) if np.isfinite(r["leak_count"]) else np.nan),
            ("-leak whole (fewer gold-only whole docs = good)", lambda r: -max(r["leak_whole"], 0) if np.isfinite(r["leak_whole"]) else np.nan)]
    for name, f in mets:
        cells = []
        for sel in ("per-task", "setA32k", None):
            rr = [r for r in recs if r["outcome"] in "WL" and r["outcome"] != "?" and (sel is None or r["ckpt"] == sel)]
            w = [f(r) for r in rr if r["outcome"] == "W" and np.isfinite(f(r))]
            l_ = [f(r) for r in rr if r["outcome"] == "L" and np.isfinite(f(r))]
            cells.append(f"{auc(w, l_):.2f} ({len(w)}W/{len(l_)}L)")
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
