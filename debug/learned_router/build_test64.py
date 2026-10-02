"""A BIGGER test set (default 64 rows per task at 2k) for final verdicts, disjoint from every row the routers
or the grid have touched. Eval only.

    python debug/learned_router/build_test64.py [--tasks ...] [--n 64]

Candidates come from the cached HF ``r2k`` split (``PrasannSinghal/ctc-suite-eval``) past every block in use
(staged 0-63, router val 64-95, router train 96-191), from ``--start`` (default 192). niah takes rows
320-383 instead: the split's second short-haystack block, i.e. the same regime as its staged test rows.
Each candidate is dropped on any query / gold-text / (query, answer) hash match against the staged rows of
every rung (first 64), ``router_{train,val}_{e2e,stg}``, and the rows already kept (the keys of
``fetch_train_rows.py``). Kept rows go to ``<data>/test64/<subset>/rung_<tok>.jsonl``, the layout
``load_examples`` reads, so ``--eval-data-root <data>/test64`` scores them. Writes ``split_report_test64.json``.
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "devloss_grid"))
from fetch_train_rows import CACHE, OUT_ROOT, STAGED, _read_jsonl, row_keys, rung_tokens  # noqa: E402

TASKS = "nq,contradiction,scifact,strmatch,outlier,fiqa,msmarco,obliq,oolong,outlier_amzn,reorder,qdmatch_hpqa,rerank,textgroups,grouping,niah,absence"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=TASKS)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--start", type=int, default=192)
    a = ap.parse_args()
    import pyarrow.parquet as pq

    from ctc_devloss_grid import ROSTER

    report = {}
    for key in a.tasks.split(","):
        row = ROSTER[f"ctc_{key}"]
        spec, sub = row["spec"], row["subset"]
        used_q, used_qa, used_g, n_used = {}, set(), set(), 0
        files = sorted(glob.glob(os.path.join(STAGED, sub, "rung_*.jsonl")))
        for sp in ("router_train_e2e", "router_val_e2e", "router_train_stg", "router_val_stg"):
            files += sorted(glob.glob(os.path.join(STAGED, sp, sub, "rung_*.jsonl")))
        for path in files:
            for ex in _read_jsonl(path, 10_000):
                q, qa, g, _, _ = row_keys(ex, spec)
                used_q[q] = used_q.get(q, 0) + 1
                used_qa.add(qa)
                used_g |= g
                n_used += 1
        templated = {q for q, c in used_q.items() if c >= 3}
        tok = rung_tokens(row, "2k")
        pq_path = sorted(glob.glob(os.path.join(CACHE, "datasets--PrasannSinghal--ctc-suite-eval", "snapshots", "*", "data", sub, "r2k.parquet")))[-1]
        tab = pq.read_table(pq_path)
        start = 320 if key == "niah" else a.start
        cands = tab.slice(start, tab.num_rows - start).to_pylist()
        kept, drops = [], {"query": 0, "gold": 0, "qa": 0}
        for j, ex in enumerate(cands):
            if len(kept) >= a.n:
                break
            for mk in ("meta", "_meta"):
                if isinstance(ex.get(mk), str):
                    try:
                        ex[mk] = json.loads(ex[mk])
                    except ValueError:
                        pass
            q, qa, g, _, has_gold = row_keys(ex, spec)
            why = "query" if (q in used_q and q not in templated) else ("gold" if has_gold and g & used_g else ("qa" if not has_gold and qa in used_qa else None))
            if why:
                drops[why] += 1
                continue
            used_q[q] = used_q.get(q, 0) + 1  # also disjoint within the new set
            used_qa.add(qa)
            used_g |= g
            ex["_router_src"] = {"split": "r2k", "index": start + j, "set": "test64"}
            kept.append(ex)
        out = os.path.join(OUT_ROOT, "test64", sub, f"rung_{tok}.jsonl")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out + ".part", "w") as f:
            for ex in kept:
                f.write(json.dumps(ex) + "\n")
        os.replace(out + ".part", out)
        import numpy as np

        chars = float(np.mean([np.mean([len((d or {}).get("text") or "") for d in (ex.get("documents") or [{}])]) for ex in kept]))
        report[key] = {"hf_rows_from": start, "kept": len(kept), "dropped": drops, "rows_hashed_against": n_used,
                       "src_index": [e["_router_src"]["index"] for e in kept], "mean_doc_chars": chars}
        print(f"{key:13s} rows {start}+: kept {len(kept)}, dropped {drops}, hashed against {n_used} used rows; mean doc chars {chars:.0f} -> {out}", flush=True)
    json.dump(report, open(os.path.join(HERE, "split_report_test64.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
