"""Reconstruct the shard-instance -> source-row map, and prove it by exact token-length match.

The converter drops any row whose tokenized length exceeds --max-seq-len and records only a COUNT,
never which rows. So the shards are source-minus-drops and nothing on disk says where the holes are.

This re-runs the converter's own prompt build + tokenization over the head of the source file,
applies the same length filter, and checks that the surviving lengths equal the shard instance
lengths element-for-element. An exact match proves both the drop model and the recovered map -- and
the map is what a grader needs in order to pair a generation with its own gold.

Writes ``src_index.json`` (shard instance i -> source row src_index[i]) next to the shards.
"""
import json, sys, os
import numpy as np

sys.path.insert(0, "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/src/scripts/data/hybridish")
sys.path.insert(0, "/scratch/users/prasann/hyb_sft/ctcsrc")

SH = "/scratch/users/prasann/ctc_hybridish_sft/shards_long32k"
SRC = "/scratch/users/prasann/ctc_hybridish_sft/long/mix_long.jsonl"
N_PROBE = int(sys.argv[1]) if len(sys.argv) > 1 else 500

from convert_ctc_to_sft_completion import spec_name_for, tokenize_completion
from olmo_core.data.corpus_reasoning_prompts import build_prompt
from transformers import AutoTokenizer

meta = json.load(open(f"{SH}/metadata.json"))
EOS, CAP = meta["eos_token_id"], meta["max_seq_len"]
tok = AutoTokenizer.from_pretrained(meta["tokenizer"])

ids = np.fromfile(f"{SH}/token_ids_part_000000.npy", dtype=np.uint32, count=80_000_000)
b = np.flatnonzero(ids == EOS) + 1
starts = np.concatenate([[0], b[:-1]])
shard_len = (b - starts).tolist()

kept_len, kept_src, dropped = [], [], []
with open(SRC) as f:
    for j, line in enumerate(f):
        if j >= N_PROBE:
            break
        ex = json.loads(line)
        task = ex.get("_task")
        prompt, answer = build_prompt(
            ex, task=spec_name_for(task), query_position=meta["query_position"],
            use_alpaca=True, cot_mode=ex.get("_cot_mode", "none"),
        )
        r = tokenize_completion(tok, prompt, answer, EOS, meta["train_on_eos"])
        if r is None:
            dropped.append((j, task, "bad"))
            continue
        n = int(r[0].size)
        if n > CAP:
            dropped.append((j, task, n))
            continue
        kept_len.append(n)
        kept_src.append(j)

m = min(len(kept_len), len(shard_len))
exact = kept_len[:m] == shard_len[:m]
print(f"[probe] {N_PROBE} source rows -> {len(kept_len)} kept, {len(dropped)} dropped")
print(f"[verify] reconstructed lengths == shard lengths over first {m}: {exact}")
if not exact:
    bad = next(i for i in range(m) if kept_len[i] != shard_len[i])
    print(f"[verify] first divergence at instance {bad}: "
          f"reconstructed {kept_len[bad]} vs shard {shard_len[bad]}")

print(f"\n[drops] first 15 dropped source rows (row, task, tokens):")
for d in dropped[:15]:
    print(f"    {d}")

print(f"\n[impact] positional zip SRC[i] is correct only for i < {kept_src.index(next(j for j in kept_src if j != kept_src.index(j)))if False else (dropped[0][0] if dropped else 'n/a')}")
off = [i for i in range(m) if kept_src[i] != i]
print(f"[impact] {len(off)}/{m} shard instances map to a source row OTHER than their own index")
if off:
    print(f"[impact] drift starts at instance {off[0]}; by instance {m-1} the offset is "
          f"{kept_src[m-1] - (m-1)} rows")

out = f"{SH}/src_index.json"
json.dump({"note": "shard instance i was built from source row src_index[i]",
           "src_jsonl": SRC, "probe_rows": N_PROBE,
           "src_index": kept_src, "dropped": [d[0] for d in dropped]}, open(out, "w"))
print(f"\n[wrote] {out}")
