"""
Paired comparison of two eval_devce.py JSONs scored on the same shard (same rows, same order).

For each bin (and overall): token-weighted CE of A and B, delta = B - A, and the paired SE of the
delta from the per-row CE differences (rows are the independent unit; tokens within a row are not).
Negative delta = B is better.

    python compare_devce.py stage2_yarn4_dev32k.json stage3_yarn8_dev32k.json [--md]
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np


def load(path: str) -> dict:
    d = json.load(open(path))
    d["_sums"] = np.asarray(d["per_row"]["loss_sum"], dtype=np.float64)
    d["_cnts"] = np.asarray(d["per_row"]["tokens"], dtype=np.int64)
    return d


def paired(a: dict, b: dict) -> list:
    # the shard may sit at different paths on different clusters; its metadata must agree
    for k in ("eval_size", "row_len", "shard_meta"):
        if a.get(k) != b.get(k):
            raise SystemExit(f"different shards ({k}: {a.get(k)} vs {b.get(k)}): {a['dev']} vs {b['dev']}")
    if a["row_ids"] != b["row_ids"] or [x["name"] for x in a["bins"]] != [x["name"] for x in b["bins"]]:
        raise SystemExit("row ids or bins differ: not a paired comparison")
    if not np.array_equal(a["_cnts"], b["_cnts"]):
        raise SystemExit("per-row token counts differ: the two runs did not score the same targets")
    cols = [("overall", a["_sums"].sum(1), b["_sums"].sum(1), a["_cnts"].sum(1))]
    cols += [(x["name"], a["_sums"][:, i], b["_sums"][:, i], a["_cnts"][:, i]) for i, x in enumerate(a["bins"])]
    out = []
    for name, sa, sb, c in cols:
        ok = c > 0
        d = sb[ok] / c[ok] - sa[ok] / c[ok]
        k = int(ok.sum())
        out.append({"bin": name, "rows": k, "tokens": int(c.sum()),
                    "ce_a": float(sa.sum() / c.sum()), "ce_b": float(sb.sum() / c.sum()),
                    "delta": float(sb.sum() / c.sum() - sa.sum() / c.sum()),
                    "row_mean_delta": float(d.mean()),
                    "paired_se": float(d.std(ddof=1) / math.sqrt(k)) if k > 1 else float("nan"),
                    "frac_rows_b_better": float((d < 0).mean())})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--md", action="store_true", help="markdown table")
    ap.add_argument("--json", help="also write the table here")
    args = ap.parse_args(argv)
    a, b = load(args.a), load(args.b)
    rows = paired(a, b)
    la, lb = a.get("label") or args.a, b.get("label") or args.b
    flag = f"  ⚠ eval_size={a['eval_size']} rows only (<500)" if a["eval_size"] < 500 else ""
    print(f"A = {la}   B = {lb}   shard {a['dev']}   eval_size={a['eval_size']} rows{flag}")
    if args.md:
        print("| target pos | tokens | CE A | CE B | Δ (B−A) | paired SE | rows B better |")
        print("|---|---|---|---|---|---|---|")
        for r in rows:
            print(f"| {r['bin']} | {r['tokens']:,} | {r['ce_a']:.4f} | {r['ce_b']:.4f} | {r['delta']:+.4f} | "
                  f"±{r['paired_se']:.4f} | {r['frac_rows_b_better']:.0%} |")
    else:
        for r in rows:
            print(f"{r['bin']:>8}  tok {r['tokens']:>10,}  A {r['ce_a']:.4f}  B {r['ce_b']:.4f}  "
                  f"d {r['delta']:+.4f} ±{r['paired_se']:.4f}  B-better {r['frac_rows_b_better']:.0%}")
    if args.json:
        json.dump({"a": args.a, "b": args.b, "eval_size": a["eval_size"], "rows": rows}, open(args.json, "w"), indent=1)
    return rows


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
