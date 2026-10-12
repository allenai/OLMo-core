"""Fetch the 2k/8k/32k rungs of every CTC-suite row from the public HF dataset
(PrasannSinghal/ctc-suite-eval, parquet at data/<subset>/<rung>.parquet) and write the first
N rows of each as JSONL in the suite's local layout: <root>/<subset>/rung_<tokens>.jsonl
(contradiction_iid aliases r2k -> 2560). Rows keep the dataset's own schema. Login node only
(compute nodes have no egress)."""
import json, os, sys, time
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/scratch/users/prasann/devloss_grid_data"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 64
SUBSETS = ["fiqa", "nq", "hotpotqa", "qdmatch_fiqa", "qdmatch_nq", "qdmatch_hpqa", "outlier_amzn",
           "outlier", "outlier_fixedM", "oolong", "grouping", "absence_gutenberg", "xabsence", "rerank",
           "msmarco", "reorder", "obliq_twitter", "niah", "contradiction_iid", "strmatch", "textgroups",
           "scifact"]
RUNGS = {"r2k": 2048, "r8k": 8192, "r32k": 32768}
ALIAS = {"contradiction_iid": {"r2k": 2560}}
summary = {}
for s in SUBSETS:
    for r, tok in RUNGS.items():
        tok = ALIAS.get(s, {}).get(r, tok)
        out = f"{ROOT}/{s}/rung_{tok}.jsonl"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        t0 = time.time()
        try:
            p = hf_hub_download("PrasannSinghal/ctc-suite-eval", f"data/{s}/{r}.parquet", repo_type="dataset")
            t = pq.read_table(p)
            n_total = t.num_rows
            rows = t.slice(0, N).to_pylist()
            with open(out, "w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            summary[f"{s}/{r}"] = {"path": out, "rows": len(rows), "total": n_total, "split": r,
                                   "hf_file": f"data/{s}/{r}.parquet"}
            print(f"OK   {s:20s} {r:5s} -> {out} ({len(rows)}/{n_total} rows, {time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            summary[f"{s}/{r}"] = {"path": None, "error": repr(e)[:300], "split": r}
            print(f"FAIL {s:20s} {r:5s}: {repr(e)[:200]}", flush=True)
json.dump(summary, open(f"{ROOT}/fetch_summary.json", "w"), indent=1)
print("done", flush=True)
