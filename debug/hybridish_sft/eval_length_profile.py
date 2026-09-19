"""Actual token lengths of the graded eval slice, per task and per length bucket.

The mix carries only a `_band: 2k-32k` label and no per-row rung, and rung labels do not match
realized token counts anyway, so this measures the shards directly: the eval slice is the first N
instances of each task under the src_index map -- the same slice the grader takes.
"""
import json, sys
from collections import defaultdict
import numpy as np

SH = "/scratch/users/prasann/ctc_hybridish_sft/shards_long32k"
SRC_P = "/scratch/users/prasann/ctc_hybridish_sft/long/mix_long.jsonl"
PER_TASK = int(sys.argv[1]) if len(sys.argv) > 1 else 40

meta = json.load(open(f"{SH}/metadata.json"))
EOS = meta["eos_token_id"]
SRC_ROW = json.load(open(f"{SH}/src_index.json"))["src_index"]
ids = np.fromfile(f"{SH}/token_ids_part_000000.npy", dtype=np.uint32, count=80_000_000)
b = np.flatnonzero(ids == EOS) + 1
starts = np.concatenate([[0], b[:-1]])
N = len(b)
lens = (b - starts)

need = max(SRC_ROW[:N]) + 1
task_of = {}
with open(SRC_P) as f:
    for i, line in enumerate(f):
        if i >= need: break
        task_of[i] = json.loads(line).get("_task", "?")

by_task = defaultdict(list)
for i in range(N):
    by_task[task_of[SRC_ROW[i]]].append(i)

EDGES = [2048, 4096, 8192, 16384, 32768]
NAMES = ["<=2k", "2-4k", "4-8k", "8-16k", "16-32k"]
def bucket(L):
    for e, nm in zip(EDGES, NAMES):
        if L <= e: return nm
    return "16-32k"

hdr = f"{'task':<15}" + "".join(f"{n:>8}" for n in NAMES) + f"{'median':>9}{'max':>8}"
print(hdr); print("-" * len(hdr))
totals = defaultdict(int)
for t, idxs in sorted(by_task.items()):
    idxs = idxs[:PER_TASK]
    L = [int(lens[i]) for i in idxs]
    c = defaultdict(int)
    for x in L:
        c[bucket(x)] += 1; totals[bucket(x)] += 1
    print(f"{t:<15}" + "".join(f"{c[n]:>8}" for n in NAMES)
          + f"{int(np.median(L)):>9}{max(L):>8}")
print("-" * len(hdr))
print(f"{'ALL (n=' + str(sum(totals.values())) + ')':<15}"
      + "".join(f"{totals[n]:>8}" for n in NAMES))
print(f"{'% of eval':<15}" + "".join(f"{100*totals[n]/sum(totals.values()):>7.0f}%" for n in NAMES))

print(f"\nGrader bands used in the results table map to these as:")
print(f"  <=4k   = {NAMES[0]} + {NAMES[1]}   ({totals['<=2k']+totals['2-4k']} of {sum(totals.values())})")
print(f"  <=16k  = {NAMES[2]} + {NAMES[3]}   ({totals['4-8k']+totals['8-16k']})  <- really 4k-16k, mislabelled")
print(f"  >16k   = {NAMES[4]}            ({totals['16-32k']})")
