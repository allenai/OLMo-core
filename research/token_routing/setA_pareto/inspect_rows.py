"""Print each setA task's row schema and gold fields (first row), and compare a task's new vs shipped
shard metadata. Diagnostic for setA_gold_sidecars.py.

    python research/token_routing/setA_pareto/inspect_rows.py --root <set root> --new-dir shards_qwen3_5_gold \
        --ref-dir shards_qwen35_256k --tasks qdmatch_nq,reorder,oolong,rerank
"""
import argparse
import glob
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--new-dir", required=True)
ap.add_argument("--ref-dir", required=True)
ap.add_argument("--tasks", required=True)
a = ap.parse_args()
for t in a.tasks.split(","):
    p = os.path.join(a.root, "per_task", t, "train.jsonl")
    with open(p) as f:
        ex = json.loads(f.readline())
    ex = ex["ex"] if "ex" in ex and "documents" not in ex else ex
    short = {k: (v if len(json.dumps(v)) < 300 else f"<{type(v).__name__} len={len(v) if hasattr(v, '__len__') else '?'}>")
             for k, v in ex.items()}
    print(f"== {t} keys={sorted(ex)}\n   {json.dumps(short)[:1500]}")
    for d in (a.ref_dir, a.new_dir):
        m = json.load(open(os.path.join(a.root, d, t, "metadata.json")))
        parts = sorted(os.path.basename(x) for x in glob.glob(os.path.join(a.root, d, t, "*.npy")))
        print(f"   {d}: rows={m.get('num_instances')} tokens={m.get('num_tokens')} dropped={m.get('num_dropped')} "
              f"max_len={m.get('max_example_len')} parts={len(parts)} first={parts[:1]} last={parts[-1:]}")
    print(f"   per_task rows in jsonl: {sum(1 for _ in open(p))}  mtime jsonl={os.path.getmtime(p):.0f} "
          f"ref meta mtime={os.path.getmtime(os.path.join(a.root, a.ref_dir, t, 'metadata.json')):.0f}")
