"""Show the gold self-check separates the broken pairing from the fixed one.

A guard nobody has seen fail is not evidence. This scores each shard instance's OWN gold answer --
the span its label mask marks -- against the example it gets paired with, under both pairings:

  * positional zip (what the original grader did):  SRC[i]
  * src_index map  (the fix):                       SRC[src_index[i]]

Correct pairing must self-score 1.0 by construction. Whatever the positional zip scores is the
ceiling the old results table was measured against -- a perfect model could not have exceeded it.
CPU only, no model, seconds to run.
"""
import inspect, json, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, "/scratch/users/prasann/hyb_sft/ctcsrc")
SH = "/scratch/users/prasann/ctc_hybridish_sft/shards_long32k"
SRC_P = "/scratch/users/prasann/ctc_hybridish_sft/long/mix_long.jsonl"
PER_TASK = int(sys.argv[1]) if len(sys.argv) > 1 else 40

from ctc.tasks import load_all
from ctc.format import registry
load_all()
T2S = {"nq": "retrieval", "qdmatch_nq": "qdmatch"}
def spec_for(t): return registry.get(T2S.get(t, t))

def score_any(sp, parsed, ex):
    second = (list(inspect.signature(sp.score).parameters) + ["", ""])[1]
    if "example" in second:
        return sp.score(parsed, ex)
    for k in ("gold_doc_indices", "gold_pairs", "gold_order", "answers"):
        if ex.get(k):
            return sp.score(parsed, ex[k])
    raise KeyError(sp.name)

meta = json.load(open(f"{SH}/metadata.json"))
EOS = meta["eos_token_id"]
SRC_ROW = json.load(open(f"{SH}/src_index.json"))["src_index"]

ids = np.fromfile(f"{SH}/token_ids_part_000000.npy", dtype=np.uint32, count=80_000_000)
b = np.flatnonzero(ids == EOS) + 1
starts = np.concatenate([[0], b[:-1]])
msk = np.fromfile(f"{SH}/labels_mask_000000.npy", dtype=np.bool_, count=80_000_000)
N = len(b)

need = max(SRC_ROW[:N]) + 1
SRC = []
with open(SRC_P) as f:
    for i, line in enumerate(f):
        if i >= need:
            break
        SRC.append(json.loads(line))

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(meta["tokenizer"])

by_task = defaultdict(list)
for i in range(N):
    by_task[SRC[SRC_ROW[i]].get("_task", "?")].append(i)

print(f"\n{'task':<15} {'positional zip (old)':>22} {'src_index map (fix)':>22}")
print("-" * 61)
tot_old, tot_new, n_all = 0.0, 0.0, 0
for t, idxs in sorted(by_task.items()):
    idxs = idxs[:PER_TASK]
    sp = spec_for(t)
    old, new = [], []
    for i in idxs:
        s, e = int(starts[i]), int(b[i])
        cut = int(np.argmax(msk[s:e]))
        gold_text = tok.decode(ids[s + cut : e].astype(np.int64).tolist(), skip_special_tokens=True)
        for pairing, acc in ((SRC[i], old), (SRC[SRC_ROW[i]], new)):
            nd = len(pairing.get("documents", []) or [])
            try:
                p = sp.parse(gold_text, nd)
                acc.append(0.0 if p is None else float(score_any(sp, p, pairing)[sp.primary_metric]))
            except Exception:
                acc.append(0.0)
    mo, mn = float(np.mean(old)), float(np.mean(new))
    tot_old += sum(old); tot_new += sum(new); n_all += len(idxs)
    print(f"{t:<15} {mo:>22.4f} {mn:>22.4f}")
print("-" * 61)
print(f"{'MEAN':<15} {tot_old/n_all:>22.4f} {tot_new/n_all:>22.4f}   (eval_size={n_all})")
print("\nLeft column is the CEILING the old table was scored against: gold itself could not beat it.")
print("Right column is 1.0 by construction when the pairing is right.")
