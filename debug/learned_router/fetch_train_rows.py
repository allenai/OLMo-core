"""Fetch router TRAIN / VAL rows for the learned token router and prove they are disjoint from every
dev-loss-grid TEST row.

    python debug/learned_router/fetch_train_rows.py --tasks nq,outlier,contradiction,oolong

Login node only (compute nodes have no egress). Candidates are rows ``[--start, --start + --pool)``
of the HF ``r2k`` / ``r8k`` splits (``PrasannSinghal/ctc-suite-eval``); the grid's test rows are the
FIRST 16/16/8 rows of the 2k/8k/32k files and the staged copies hold the first 64, so every
candidate index is already past the staged block. Index is NOT trusted: rung files reuse the same
underlying examples at different lengths, so each candidate is hashed against the first 64 rows of
EVERY rung (a superset of the test rows) and dropped on any of

* ``query``  -- same query text, unless the query is a template (shared by >= 3 staged rows),
* ``gold``   -- any gold document text shared with any staged row's gold set,
* ``qa``     -- (query, answer) match, for tasks without a gold subset (oolong).

Kept rows are split per rung: the first ``--n-train`` -> ``router_train``, the next ``--n-val`` ->
``router_val`` (``<root>/<split>/<subset>/rung_<tok>.jsonl``, the layout ``load_examples`` reads).
Writes ``split_report.json`` next to this file.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "devloss_grid"))

STAGED = "/net/sneetches/data/prasann/devloss_grid/data"  # first 64 rows of every rung (audit read)
OUT_ROOT = "/net/sneetches/data/prasann/devloss_grid/data"  # node-local /data of the staging node
CACHE = os.environ.get(
    "ROUTER_HF_CACHE",
    "/tmp/claude-3018/-accounts-projects-berkeleynlp-prasann-projects-OLMo-core/"
    "b25b0e58-1d93-45d2-a076-a44e278feae6/scratchpad/hf_cache",
)
RUNG_TOK = {"2k": 2048, "8k": 8192, "32k": 32768}


def _norm(s) -> str:
    return " ".join(str(s).split()).lower()


def _h(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def row_keys(ex: dict, spec: str):
    from ctc_devloss_grid import gold_docs  # noqa: E402

    q = _norm(" || ".join(ex.get("queries") or []))
    a = _norm(" || ".join(str(x) for x in (ex.get("answers") or [])))
    docs = ex.get("documents") or []
    g = gold_docs(spec, ex)
    gtexts = {_h(_norm(docs[i].get("text") if isinstance(docs[i], dict) else docs[i])) for i in (g or ())}
    dtexts = {_h(_norm(d.get("text") if isinstance(d, dict) else d)) for d in docs}
    return _h(q), _h(q + "##" + a), gtexts, dtexts, g is not None


def _read_jsonl(path: str, n: int):
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                ex = json.loads(line)
                out.append(ex["ex"] if "ex" in ex and "documents" not in ex else ex)
                if len(out) >= n:
                    break
    return out


def rung_tokens(row: dict, rung: str) -> int:
    return row.get("rung_alias", {}).get(rung, RUNG_TOK[rung])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="manifest keys, comma separated")
    ap.add_argument("--start", type=int, default=64)
    ap.add_argument("--pool", type=int, default=64)
    ap.add_argument("--n-train", type=int, default=16, help="per rung")
    ap.add_argument("--n-val", type=int, default=8, help="per rung")
    ap.add_argument("--rungs", default="2k,8k")
    ap.add_argument("--split-suffix", default="", help="e.g. _e2e -> router_train_e2e / router_val_e2e")
    ap.add_argument("--val-first", action="store_true", help="fill val before train (short splits keep a val set)")
    ap.add_argument("--report", default="split_report.json")
    a = ap.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from ctc_devloss_grid import ROSTER

    manifest = json.load(open(os.path.join(os.path.dirname(HERE), "devloss_grid", "manifest.json")))
    rep_path = os.path.join(HERE, a.report)
    report = json.load(open(rep_path)) if os.path.exists(rep_path) else {}
    for key in a.tasks.split(","):
        t = manifest["tasks"][key]
        row = ROSTER[f"ctc_{key}"]
        spec = row["spec"]
        # ---- test-side keys: first 64 staged rows of EVERY rung ----
        tq, tqa, tg, td = {}, set(), set(), set()
        n_test = 0
        # every staged rung file of the subset (2k/8k/32k, and the 16k stand-ins of absence/reorder)
        staged = sorted(glob.glob(os.path.join(STAGED, row["subset"], "rung_*.jsonl")))
        assert staged, f"no staged rung files for {row['subset']}"
        for path in staged:
            for ex in _read_jsonl(path, 64):
                q, qa, g, d, _ = row_keys(ex, spec)
                tq[q] = tq.get(q, 0) + 1
                tqa.add(qa)
                tg |= g
                td |= d
                n_test += 1
        templated = {q for q, c in tq.items() if c >= 3}
        rep = {"spec": spec, "test_rows_hashed": n_test, "test_files": [os.path.basename(x) for x in staged], "templated_queries": len(templated), "rungs": {}}
        for rung in a.rungs.split(","):
            tok = rung_tokens(row, rung)
            p = hf_hub_download("PrasannSinghal/ctc-suite-eval", f"data/{row['subset']}/r{rung}.parquet",
                                repo_type="dataset", cache_dir=CACHE)
            tab = pq.read_table(p)
            cands = tab.slice(a.start, a.pool).to_pylist()
            kept, drops, doc_overlap = [], {"query": 0, "gold": 0, "qa": 0}, []
            for j, ex in enumerate(cands):
                for mk in ("meta", "_meta"):
                    if isinstance(ex.get(mk), str):
                        try:
                            ex[mk] = json.loads(ex[mk])
                        except ValueError:
                            pass
                q, qa, g, d, has_gold = row_keys(ex, spec)
                why = None
                if q in tq and q not in templated:
                    why = "query"
                elif has_gold and g & tg:
                    why = "gold"
                elif not has_gold and qa in tqa:
                    why = "qa"
                if why:
                    drops[why] += 1
                    continue
                doc_overlap.append(len(d & td) / max(1, len(d)))
                ex["_router_src"] = {"split": f"r{rung}", "index": a.start + j}
                kept.append(ex)
            if a.val_first:
                va, tr = kept[: a.n_val], kept[a.n_val : a.n_val + a.n_train]
            else:
                tr, va = kept[: a.n_train], kept[a.n_train : a.n_train + a.n_val]
            for split, rows_ in ((f"router_train{a.split_suffix}", tr), (f"router_val{a.split_suffix}", va)):
                out = os.path.join(OUT_ROOT, split, row["subset"], f"rung_{tok}.jsonl")
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with open(out + ".part", "w") as f:
                    for ex in rows_:
                        f.write(json.dumps(ex) + "\n")
                os.replace(out + ".part", out)
            rep["rungs"][rung] = {
                "hf_rows": [a.start, a.start + a.pool], "hf_total": tab.num_rows, "candidates": len(cands),
                "dropped": drops, "kept": len(kept), "train": len(tr), "val": len(va),
                "train_src_index": [e["_router_src"]["index"] for e in tr],
                "val_src_index": [e["_router_src"]["index"] for e in va],
                "distractor_doc_overlap_mean": (sum(doc_overlap) / len(doc_overlap)) if doc_overlap else None,
            }
            print(f"{key:14} {rung:>3}: {len(cands)} cands, dropped {drops}, kept {len(kept)} -> train {len(tr)} "
                  f"val {len(va)}; distractor-doc overlap with test rows {rep['rungs'][rung]['distractor_doc_overlap_mean']}",
                  flush=True)
            if len(tr) < a.n_train or len(va) < a.n_val:
                print(f"!! {key}@{rung}: short of rows after the disjointness filter", flush=True)
        report[key] = rep
    json.dump(report, open(rep_path, "w"), indent=1)
    print(f"wrote {rep_path}")


if __name__ == "__main__":
    main()
