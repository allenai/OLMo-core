"""Per-example overfit ceiling (``oracle_ls.py``) vs the bar and the frontier pick, on the same test64 rows.

    python debug/learned_router/layerskip/summarize_oracle.py [--tasks nq,scifact,...] [--json out.json]

Per task and tau: median / min hard FLOP fraction of (i) token-only and (ii) token + layer oracles (and
how many rows meet tau after the hard check), the bar's median FLOPs on the same rows, and the frontier
pick for that tier (``frontier.json``) on the same rows (median FLOPs, rows meeting tau). Plus what the
oracle keeps (gold body / non-gold body / non-gold id region / markers, layer skip rates) and the wall
time per example. Non-deployable lower bound: the oracle sees the answer.
"""
import argparse
import glob
import json
import os

import numpy as np

import summarize_frontier as SF

HERE = os.path.dirname(os.path.abspath(__file__))


def frontier_rows(task, tau, rows):
    fj = json.load(open(os.path.join(HERE, "frontier.json")))
    rec = fj[task]["tiers"][f"{tau:g}"]
    r = SF.load(task)
    full = np.asarray(r["full"]["test64"])
    if rec["label"] is None:
        return rec["candidate"], [1.0] * len(rows), [0.0] * len(rows)
    c = r["configs"][rec["label"]]
    if rec["keep"] >= 1.0:
        ce, fl = c["splits"]["test64"]["tok"], c["splits"]["test64"]["tok_flops"]
    else:
        pt = c["points"][f"{rec['keep']:g}"]["test64"]
        ce, fl = pt["ce"], pt["flops"]
    return rec["candidate"], [fl[i] for i in rows], [ce[i] - full[i] for i in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="nq,scifact,outlier,contradiction,rerank,strmatch,niah,textgroups")
    ap.add_argument("--rung", default="2k")
    ap.add_argument("--variant", default="", help="'' = free selection; 'fid8' = --force-id-prefix 8 runs")
    ap.add_argument("--json", default=os.path.join(HERE, "oracle_summary.json"))
    a = ap.parse_args()
    out = {}
    for task in a.tasks.split(","):
        rows = []
        for f in sorted(glob.glob(os.path.join(HERE, "runs", task, f"oracle{'_' + a.variant if a.variant else ''}_{a.rung}_r*.json"))):
            rows += json.load(open(f))["rows"]
        if not rows:
            print(f"== {task}: no oracle rows")
            continue
        rows.sort(key=lambda r: r["row"])
        ids = [r["row"] for r in rows]
        bar_fl = [r["bar"]["flops"] for r in rows]
        bar_d = [r["bar"]["dce"] for r in rows]
        t = {"rows": ids, "bar_flops_median": float(np.median(bar_fl)), "bar_dce": bar_d,
             "sec_per_row_median": float(np.median([r["row_sec"] for r in rows])), "arms": {}}
        print(f"\n== {task} {a.rung}: {len(rows)} rows {ids} ⚠ per-example; bar FLOPs median ×{t['bar_flops_median']:.3f} "
              f"(dCE {np.round(bar_d, 3).tolist()}); wall {t['sec_per_row_median']:.0f} s/row")
        for tau in (0.02, 0.05):
            line = f"   tau {tau}: "
            for arm in ("tok", "tok+layer"):
                k = f"{arm}@{tau:g}"
                fl = [r["arms"][k]["flops"] for r in rows]
                ok = int(sum(bool(r["arms"][k]["meets"]) for r in rows))
                ks = {s: float(np.nanmedian([r["arms"][k]["keeps"][s] for r in rows]))
                      for s in ("gold_body", "nongold_body", "nongold_id_region", "markers", "skip_attn", "skip_gdn", "skip_early", "skip_late")}
                t["arms"][k] = {"flops_median": float(np.median(fl)), "flops_min": float(np.min(fl)), "flops": fl, "meets": ok, "keeps_median": ks}
                line += f"{arm}: median ×{np.median(fl):.3f} min ×{np.min(fl):.3f} ({ok}/{len(rows)} meet) | "
            for arm, allowed in (("tok", ("tok",)), ("tok+layer", ("tok", "tok+layer"))):
                env = []
                for r in rows:
                    c_ = [r["arms"][f"{x}@{tt:g}"]["flops"] for x in allowed for tt in (0.02, 0.05) if r["arms"][f"{x}@{tt:g}"]["dce"] <= tau]
                    env.append(min(c_) if c_ else 1.0)
                t["arms"][f"{arm}@{tau:g}"]["envelope_flops_median"] = float(np.median(env))
                t["arms"][f"{arm}@{tau:g}"]["envelope_flops"] = env
                line += f"envelope {arm} ×{np.median(env):.3f} | "
            if os.path.exists(os.path.join(HERE, "frontier.json")) and a.rung == "2k":
                cand, ffl, fd = frontier_rows(task, tau, ids)
                fok = int(sum(bool(d <= tau) for d in fd))
                fd = [float(d) for d in fd]
                ffl = [float(x) for x in ffl]
                t[f"frontier@{tau:g}"] = {"candidate": cand, "flops_median": float(np.median(ffl)), "meets": fok, "dce": fd}
                line += f"frontier {cand}: median ×{np.median(ffl):.3f} ({fok}/{len(rows)} meet)"
            print(line)
        for k, v in t["arms"].items():
            ks = v["keeps_median"]
            print(f"   keeps {k:14s}: gold {ks['gold_body']:.2f} non-gold {ks['nongold_body']:.2f} ids {ks['nongold_id_region']:.2f} "
                  f"markers {ks['markers']:.2f} | skip attn {ks['skip_attn']:.2f} gdn {ks['skip_gdn']:.2f} early {ks['skip_early']:.2f} late {ks['skip_late']:.2f}")
        out[task] = t
    if out:
        for tau in (0.02, 0.05):
            for arm in ("tok", "tok+layer"):
                m = [v["arms"][f"{arm}@{tau:g}"]["flops_median"] for v in out.values()]
                me = [v["arms"][f"{arm}@{tau:g}"]["envelope_flops_median"] for v in out.values()]
                print(f"ACROSS tasks tau {tau} {arm}: median-of-medians ×{np.median(m):.3f} (= {1 / np.median(m):.1f}× fewer); "
                      f"envelope ×{np.median(me):.3f} (= {1 / np.median(me):.1f}×)")
            if a.rung == "2k":
                fm = [v[f"frontier@{tau:g}"]["flops_median"] for v in out.values() if f"frontier@{tau:g}" in v]
                print(f"ACROSS tasks tau {tau} frontier pick: median-of-medians ×{np.median(fm):.3f}")
        print(f"ACROSS tasks bar: median-of-medians ×{np.median([v['bar_flops_median'] for v in out.values()]):.3f}")
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
