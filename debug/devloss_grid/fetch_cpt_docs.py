"""Fetch long raw-text pretraining documents for the cpt80 column from the public
geodesic-research/dolma3_longmino_mix_500k_sample (a text sample of the dolma3 longmino mix amandab's
CPT runs read in tokenized form). Keeps docs long enough for the 32k rung (>= 140k chars ~ 32k+ tokens)
plus a mid tier for 8k, writes cpt80/long_docs.jsonl with fields {text, source, n_chars}."""
import json, sys, os
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
OUT = "/scratch/users/prasann/devloss_grid_data/cpt80"
os.makedirs(OUT, exist_ok=True)
long_docs, mid_docs = [], []
for i in range(7):
    p = hf_hub_download("geodesic-research/dolma3_longmino_mix_500k_sample", f"data/train-0000{i}-of-00007.parquet", repo_type="dataset")
    t = pq.read_table(p)
    cols = t.column_names
    print("shard", i, t.num_rows, cols, flush=True)
    textcol = "text" if "text" in cols else cols[0]
    for row in t.to_pylist():
        txt = row.get(textcol) or ""
        n = len(txt)
        rec = {"text": txt, "n_chars": n, "source": str(row.get("source") or row.get("id") or f"shard{i}")}
        if n >= 140_000 and len(long_docs) < 96:
            long_docs.append(rec)
        elif 40_000 <= n < 140_000 and len(mid_docs) < 96:
            mid_docs.append(rec)
    print(f"  long={len(long_docs)} mid={len(mid_docs)}", flush=True)
    if len(long_docs) >= 96 and len(mid_docs) >= 96:
        break
with open(f"{OUT}/long_docs.jsonl", "w") as f:
    for r in long_docs + mid_docs:
        f.write(json.dumps(r) + "\n")
print("wrote", f"{OUT}/long_docs.jsonl", len(long_docs), "long (>=140k chars),", len(mid_docs), "mid (40k-140k)", flush=True)
