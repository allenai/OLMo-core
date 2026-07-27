"""Drop training examples that collide with the oolong 8k/16k/32k EVAL rungs.

The 8k/16k/32k rungs in eval500_v2 were filtered only against the *v1* oolong train split, so the
newly generated 2k->256k oolong train pool overlaps them (measured 21.6% / 11.4% / 3.6% of each
rung). Two ways to fix it:

  (a) regenerate those three eval rungs disjoint from the new train pool, or
  (b) remove the offending examples from the TRAIN pool.

(b) is chosen. The 8k/16k/32k rungs are the ones every PRIOR oolong result was measured on -- keep
them fixed and comparability across the whole result history is preserved; regenerate them and every
historical oolong number silently refers to a different eval set. The train side loses ~0.9% of its
examples, which costs nothing.

Writes a filtered copy of each pool shard; the oolong shards must then be re-tokenized.
"""
import glob, json, os, sys

EVAL = "/data/prasann/xlong5/eval/oolong"
POOL = "/data/prasann/xlong5/pools/oolong"
OUT = "/data/prasann/xlong5/pools_oolong_clean"

def sig(ex):
    q = (ex.get("queries") or [""])[0]
    b = ex["documents"][0]["text"] if ex.get("documents") else ""
    return q + "||" + b[:200]

evalkeys = set()
for ctx in (2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144):
    p = f"{EVAL}/oolong_test_synth_ctx{ctx}_spliteval.jsonl"
    if os.path.exists(p):
        n = 0
        for line in open(p):
            if line.strip():
                evalkeys.add(sig(json.loads(line))); n += 1
        print(f"  eval ctx{ctx}: {n} examples")
print(f"eval signature set: {len(evalkeys)}")

os.makedirs(OUT, exist_ok=True)
kept = dropped = 0
for f in sorted(glob.glob(POOL + "/*/*.jsonl")):
    rel = os.path.relpath(f, POOL).replace("/", "__")
    with open(os.path.join(OUT, rel), "w") as w:
        for line in open(f):
            if not line.strip():
                continue
            ex = json.loads(line)
            if sig(ex) in evalkeys:
                dropped += 1
            else:
                w.write(line); kept += 1
print(f"\nkept={kept} dropped={dropped} ({100*dropped/max(1,kept+dropped):.2f}%) -> {OUT}")
