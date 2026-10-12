"""
Collect every ``layer_soft_probe.py`` result JSON into
``research/token_routing/pooled_kv/layer_probe/results.csv`` and print one table per
(task, rung, keep, mode, slot-source, stale-keys) group.

    python research/token_routing/pooled_kv/layer_probe/collect.py '/net/sneetches/data/prasann/layer_probe/*.json'
"""

import csv
import glob
import json
import os
import sys

COLS = [
    "task",
    "rung",
    "eval_size",
    "keep",
    "mode",
    "slot_source",
    "stale_keys",
    "prefix_real",
    "pattern",
    "n_soft_layers",
    "n_layers",
    "layers",
    "ce",
    "ce_se",
    "dce_soft",
    "dce_soft_se",
    "top1",
    "top1_se",
    "kl",
    "top1_soft",
    "kl_soft",
    "em",
    "em_se",
    "flop_saved_vs_full",
    "flop_saved_vs_soft",
    "mem_saved_vs_full",
    "mem_saved_vs_soft",
    "sec_per_row",
    "source",
]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.csv")


def g(p, *keys, default=""):
    for k in keys:
        if k in p:
            v = p[k]
            return round(v, 4) if isinstance(v, float) else v
    return default


def rows_from(path):
    d = json.load(open(path))
    for name, p in d["patterns"].items():
        yield {
            "task": d["task"],
            "rung": d["rung"],
            "eval_size": d["eval_size"],
            "keep": d["keep"],
            "mode": d.get("mode", "layer_soft"),
            "slot_source": d["slot_source"],
            "stale_keys": d.get("stale_keys", "keep"),
            "prefix_real": d.get("prefix_real", "none"),
            "pattern": name,
            "n_soft_layers": p["n_soft_layers"],
            "n_layers": d["n_layers"],
            "layers": " ".join(str(i) for i in (p["layers"] or [])),
            "ce": g(p, "ce_mean"),
            "ce_se": g(p, "ce_se"),
            "dce_soft": g(p, "dce_soft_mean"),
            "dce_soft_se": g(p, "dce_soft_se"),
            "top1": g(p, "top1_mean"),
            "top1_se": g(p, "top1_se"),
            "kl": g(p, "kl_mean"),
            "top1_soft": g(p, "top1_soft_mean"),
            "kl_soft": g(p, "kl_soft_mean"),
            "em": g(p, "correct_mean"),
            "em_se": g(p, "correct_se"),
            # older files only carried the vs-FULL flop columns
            "flop_saved_vs_full": g(p, "flop_full_mean", "flop_tot_mean"),
            "flop_saved_vs_soft": g(p, "flop_soft_mean"),
            "mem_saved_vs_full": g(p, "mem_full_mean"),
            "mem_saved_vs_soft": g(p, "mem_soft_mean"),
            "sec_per_row": g(p, "sec_per_row"),
            "source": os.path.basename(path),
        }


def main():
    paths = []
    for a in sys.argv[1:]:
        paths += sorted(glob.glob(a))
    if not paths:
        paths = sorted(glob.glob("/net/sneetches/data/prasann/layer_probe/*.json"))
    rows = []
    for p in paths:
        try:
            rows += list(rows_from(p))
        except Exception as e:  # a partially written file
            print(f"skip {p}: {e}")
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows from {len(paths)} files)\n")

    groups = {}
    for r in rows:
        key = (
            r["task"],
            r["rung"],
            r["keep"],
            r["mode"],
            r["slot_source"],
            r["stale_keys"],
            r["eval_size"],
        )
        groups.setdefault(key, []).append(r)

    def f(v, w=8, p=3):
        return f"{v:{w}.{p}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}"

    for k, rs in sorted(groups.items()):
        print(
            f"== task={k[0]} rung={k[1]} keep={k[2]} mode={k[3]} slot_source={k[4]} "
            f"stale_keys={k[5]} eval_size={k[6]}"
            + ("   ** eval_size < 500: probe, not a headline number **" if int(k[6]) < 500 else "")
        )
        print(
            f"{'pattern':16} {'#lyr':>4} {'CE':>16} {'dCE|SOFT':>9} {'top1|FULL':>16} "
            f"{'KL|FULL':>8} {'EM':>14} {'flop/FULL':>9} {'flop/SOFT':>9} {'mem/FULL':>8} {'mem/SOFT':>8}"
        )
        for r in sorted(rs, key=lambda r: (r["n_soft_layers"], r["pattern"])):
            print(
                f"{r['pattern']:16} {r['n_soft_layers']:4d} {r['ce']:7.3f}+-{r['ce_se']:<7.3f} "
                f"{f(r['dce_soft'], 9)} {r['top1']:7.3f}+-{r['top1_se']:<7.3f} {f(r['kl'])} "
                f"{r['em']:6.3f}+-{r['em_se']:<6.3f} {f(r['flop_saved_vs_full'], 9)} "
                f"{f(r['flop_saved_vs_soft'], 9)} {f(r['mem_saved_vs_full'])} {f(r['mem_saved_vs_soft'])}"
            )
        print()


if __name__ == "__main__":
    main()
