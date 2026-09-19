"""Does shard instance i correspond to source row i?

The converter drops rows it cannot fit (``skipped_above_max_seq_len``) and writes no index sidecar,
so the shards are a SUBSEQUENCE of the source JSONL, not a copy of it. Any consumer that zips them
positionally pairs each generation with some other example's gold from the first dropped row onward.

Rather than keyword-matching, this derives the instruction->task map from the data: each task's
alpaca ``### Instruction:`` block is a fixed string, so the distinct blocks ARE the task labels. The
map is learned from the aligned prefix (before the first drop), then used to score the rest, and the
shard task sequence is greedily matched against the source task sequence as a subsequence -- which
recovers exactly which source rows were dropped.
"""
import json, sys
from collections import Counter, defaultdict
import numpy as np

SH = "/scratch/users/prasann/ctc_hybridish_sft/shards_long32k"
SRC = "/scratch/users/prasann/ctc_hybridish_sft/long/mix_long.jsonl"
N_PROBE = int(sys.argv[1]) if len(sys.argv) > 1 else 400

meta = json.load(open(f"{SH}/metadata.json"))
EOS = meta["eos_token_id"]
ids = np.fromfile(f"{SH}/token_ids_part_000000.npy", dtype=np.uint32, count=60_000_000)
b = np.flatnonzero(ids == EOS) + 1
starts = np.concatenate([[0], b[:-1]])
inst = list(zip(starts.tolist(), b.tolist()))[:N_PROBE]

src = []
with open(SRC) as f:
    for i, line in enumerate(f):
        if i >= N_PROBE:
            break
        src.append(json.loads(line))

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(meta["tokenizer"])

def instruction_of(i):
    s, _ = inst[i]
    head = tok.decode(ids[s : s + 260].tolist())
    a = head.find("### Instruction:")
    b_ = head.find("### Input:")
    return head[a + 16 : b_].strip()[:150] if a >= 0 and b_ > a else head[:150].strip()

shard_instr = [instruction_of(i) for i in range(len(inst))]
src_task = [r.get("_task", "?") for r in src]

# Learn instruction -> task from the majority vote over the whole probe. Even with drift, the
# correct pairing is the modal one: drops are ~5%, so the aligned majority dominates.
votes = defaultdict(Counter)
for ins, t in zip(shard_instr, src_task):
    votes[ins][t] += 1
instr2task = {ins: c.most_common(1)[0][0] for ins, c in votes.items()}
print(f"[map] {len(instr2task)} distinct instruction blocks -> tasks: "
      f"{sorted(set(instr2task.values()))}\n")

shard_task = [instr2task[i] for i in shard_instr]

mism = [i for i in range(len(shard_task)) if shard_task[i] != src_task[i]]
print(f"[POSITIONAL ZIP] {len(mism)}/{len(shard_task)} shard instances carry a DIFFERENT task "
      f"than SRC[i]['_task']")
print(f"[POSITIONAL ZIP] first mismatch at instance {mism[0] if mism else None}")

# Greedy subsequence match: walk the source, consuming a shard instance whenever the task agrees.
# If the shards really are src-minus-drops, this consumes every shard instance in order.
si, dropped = 0, []
for j, t in enumerate(src_task):
    if si < len(shard_task) and shard_task[si] == t:
        si += 1
    else:
        dropped.append(j)
print(f"\n[SUBSEQUENCE] matched {si}/{len(shard_task)} shard instances against {len(src_task)} "
      f"source rows, skipping {len(dropped)}")
print(f"[SUBSEQUENCE] inferred dropped source rows (first 20): {dropped[:20]}")

lens = [e - s for s, e in inst]
print(f"\n[LENGTHS] shard instances: p50={int(np.median(lens))} max={max(lens)} "
      f"(cap was {meta['max_seq_len']})")
print(f"[DROPS] metadata: skipped_above_max_seq_len={meta['skipped_above_max_seq_len']} of "
      f"{meta['skipped_above_max_seq_len'] + meta['num_instances']} source rows "
      f"({100*meta['skipped_above_max_seq_len']/(meta['skipped_above_max_seq_len']+meta['num_instances']):.1f}%)")
