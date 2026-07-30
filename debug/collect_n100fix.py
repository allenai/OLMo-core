"""Collect the n100fix eval results into one table (f1 + binomial SE, eval_size)."""
import json, glob, math, os

LOGS = "/scratch/users/prasann/attn_explore_logs"
rows = []
for f in sorted(glob.glob(f"{LOGS}/attn-explore-n100fix-*_native.json")):
    d = json.load(open(f))
    c = d.get("contradiction", {})
    name = os.path.basename(f).replace("attn-explore-", "").replace("_native.json", "")
    es = d.get("eval_size") or d.get("n") or 488
    rows.append((name, c.get("f1", -1), c.get("exact_match", -1), c.get("parse_rate", -1), es))

hdr = "%-28s %7s %7s %7s %6s  %s" % ("run", "f1", "+/-SE", "em", "parse", "eval_size")
print(hdr)
print("-" * len(hdr))
for n, f1, em, pr, es in sorted(rows, key=lambda r: -r[1]):
    se = math.sqrt(max(f1, 1e-9) * (1 - f1) / es) if 0 <= f1 <= 1 else 0.0
    print("%-28s %7.3f %7.3f %7.3f %6.2f  %d" % (n, f1, se, em, pr, es))
