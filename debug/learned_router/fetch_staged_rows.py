"""Router TRAIN / VAL rows from the SAME staged rung files as the grid's test rows (same builder, same regime),
proven disjoint from every test row at every rung.

    python debug/learned_router/fetch_staged_rows.py --tasks niah,absence,fiqa [--split-suffix _stg]

Why (2026-09-30): the HF r2k split of niah is not homogeneous -- rows 0-63 and 320-383 use a short-haystack
regime (~66 chars/doc) and the rest a long one (~140-200); the grid's test rows are the first 16 staged rows
(short regime), but ``fetch_train_rows.py`` drew train/val from rows 64+ (long regime). Here candidates are
staged 2k rows ``[--start, 64)`` (the test rows are the first 16/16/8 of the 2k/8k/32k files), each dropped
on any query / gold-text / (query, answer) hash match with any test row of any staged rung file (the same
keys as ``fetch_train_rows.py``). The first ``--n-train`` kept rows -> ``router_train<suffix>``, the next
``--n-val`` -> ``router_val<suffix>``. Audit reads + small writes on the staging node over /net.
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
    ap.add_argument("--start", type=int, default=16)
    ap.add_argument("--n-train", type=int, default=32)
    ap.add_argument("--n-val", type=int, default=16)
    ap.add_argument("--split-suffix", default="_stg")
    a = ap.parse_args()
    from ctc_devloss_grid import ROSTER

    report = {}
    for key in a.tasks.split(","):
        row = ROSTER[f"ctc_{key}"]
        spec = row["spec"]
        tq, tqa, tg = {}, set(), set()
        staged = sorted(glob.glob(os.path.join(STAGED, row["subset"], "rung_*.jsonl")))
        n_test = 0
        for path in staged:
            tok = int(os.path.basename(path)[len("rung_"):-len(".jsonl")])
            for ex in _read_jsonl(path, 16 if tok <= 8192 else 8):  # the grid's test rows of that rung
                q, qa, g, _, _ = row_keys(ex, spec)
                tq[q] = tq.get(q, 0) + 1
                tqa.add(qa)
                tg |= g
                n_test += 1
        templated = {q for q, c in tq.items() if c >= 3}
        p2 = sorted(p for p in staged if int(os.path.basename(p)[5:-6]) <= 2560)[0]
        cands = _read_jsonl(p2, 64)[a.start:]
        kept, drops = [], {"query": 0, "gold": 0, "qa": 0}
        for j, ex in enumerate(cands):
            q, qa, g, _, has_gold = row_keys(ex, spec)
            why = "query" if (q in tq and q not in templated) else ("gold" if has_gold and g & tg else ("qa" if not has_gold and qa in tqa else None))
            if why:
                drops[why] += 1
                continue
            ex["_router_src"] = {"split": "staged", "file": os.path.basename(p2), "index": a.start + j}
            kept.append(ex)
        tr, va = kept[: a.n_train], kept[a.n_train : a.n_train + a.n_val]
        for split, rows_ in ((f"router_train{a.split_suffix}", tr), (f"router_val{a.split_suffix}", va)):
            out = os.path.join(OUT_ROOT, split, row["subset"], os.path.basename(p2))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out + ".part", "w") as f:
                for ex in rows_:
                    f.write(json.dumps(ex) + "\n")
            os.replace(out + ".part", out)
        report[key] = {"test_rows_hashed": n_test, "test_files": [os.path.basename(x) for x in staged], "candidates": len(cands),
                       "dropped": drops, "kept": len(kept), "train": len(tr), "val": len(va),
                       "train_src_index": [e["_router_src"]["index"] for e in tr], "val_src_index": [e["_router_src"]["index"] for e in va]}
        print(f"{key:12s} {len(cands)} cands (staged rows {a.start}-63 of {os.path.basename(p2)}), dropped {drops}, kept {len(kept)} "
              f"-> train {len(tr)} val {len(va)}; test rows hashed {n_test}", flush=True)
    json.dump(report, open(os.path.join(HERE, f"split_report{a.split_suffix}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
