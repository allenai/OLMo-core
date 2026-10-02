"""Bigger 8k / 32k test sets for length transfer: staged rung rows past the grid's own test rows (8k rows
16-63, 32k rows 8-63), dropping any row that shares a query / gold-doc text / (query, answer) hash with ANY
router train or val row (``router_{train,val}_{e2e,stg}``, all rungs). The grid's test rows are not train
data, so overlap with them is allowed. Writes ``<data>/testlong/<subset>/rung_<tok>.jsonl``.

    python debug/learned_router/build_testlong.py --tasks textgroups,nq,scifact,outlier,rerank
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "devloss_grid"))
from fetch_train_rows import OUT_ROOT, STAGED, _read_jsonl, row_keys  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    a = ap.parse_args()
    from ctc_devloss_grid import ROSTER

    rep = {}
    for key in a.tasks.split(","):
        row = ROSTER[f"ctc_{key}"]
        spec, sub = row["spec"], row["subset"]
        uq, uqa, ug = {}, set(), set()
        for sp in ("router_train_e2e", "router_val_e2e", "router_train_stg", "router_val_stg"):
            for path in glob.glob(os.path.join(STAGED, sp, sub, "rung_*.jsonl")):
                for ex in _read_jsonl(path, 10_000):
                    q, qa, g, _, _ = row_keys(ex, spec)
                    uq[q] = uq.get(q, 0) + 1
                    uqa.add(qa)
                    ug |= g
        templ = {q for q, c in uq.items() if c >= 3}
        rep[key] = {}
        for path in sorted(glob.glob(os.path.join(STAGED, sub, "rung_*.jsonl"))):
            tok = int(os.path.basename(path)[5:-6])
            if tok <= 2560:
                continue
            start = 16 if tok <= 8192 else 8
            kept, drops = [], 0
            for ex in _read_jsonl(path, 64)[start:]:
                q, qa, g, _, hg = row_keys(ex, spec)
                if (q in uq and q not in templ) or (hg and g & ug) or (not hg and qa in uqa):
                    drops += 1
                    continue
                kept.append(ex)
            out = os.path.join(OUT_ROOT, "testlong", sub, os.path.basename(path))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w") as f:
                for ex in kept:
                    f.write(json.dumps(ex) + "\n")
            rep[key][tok] = {"from_row": start, "kept": len(kept), "dropped": drops}
            print(f"{key:12s} rung {tok}: staged rows {start}-63 -> kept {len(kept)}, dropped {drops}", flush=True)
    json.dump(rep, open(os.path.join(HERE, "split_report_testlong.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
