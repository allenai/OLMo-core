#!/bin/bash
# Turn the generated OOLONG band pool into a CTC-suite training shard: verify bands -> combine ->
# decontaminate against the CTC oolong eval rungs -> tokenize -> verify chunk layout.
#
# Run AFTER the gen array (gen_oolong_ctc_bands.sbatch) finishes.
#   sbatch debug/ctc_modelscale/finalize_oolong_ctc.sbatch
#SBATCH --job-name=oolong-finalize
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=sneetches
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=/data/prasann/ctc_oolong_ctc/logs/finalize_%j.log

set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
ROOT=/data/prasann/ctc_oolong_ctc
# Pools are read from /scratch, not a node's /data: cubbins and mooney (the only two nodes with the
# cached oolong-synth dataset) both went `drng` with Prolog errors mid-build, so the generated
# shards were rescued off mooney to shared storage.
POOLS=/scratch/users/prasann/ctc_oolong_ctc_pools
COMBINED=$ROOT/oolong_ctc_bands_combined.jsonl
CLEAN=$ROOT/oolong_ctc_bands_clean.jsonl
OUTDIR=$ROOT/shards/oolong_train_ctc
EVALDIR=/scratch/users/prasann/ctc_suite_staged/eval_rungs/oolong

source /usr/local/linux/miniforge-3.13/etc/profile.d/conda.sh 2>/dev/null \
  || source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate /data/prasann/conda/envs/corpus-reasoning-olmo
export PYTHONPATH=$REPO/src HF_HOME=/data/prasann/hf_cache
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

echo "=== HOST=$(hostname) START=$(date '+%F %T') ==="

echo "=== 1. capacity-trap check: did each band realize its requested length? ==="
python "$REPO/debug/ctc_modelscale/check_realized_bands.py" --pool-root "$POOLS" --sample 60
rc=$?
[ $rc -ne 0 ] && { echo "!!! band check FAILED -- refusing to build a shard from a short pool"; exit $rc; }

echo "=== 2. combine band shards, ENFORCING the even split ==="
# The whole point of this build is an even split across the CTC rung bands, so the combine step
# takes min(count) over bands and samples exactly that many from each -- it does not just `cat`.
# One of the 20 generation array tasks (band 2 shard 3) was held by an admin when cubbins/mooney
# went into `drng`, so band 2 has 3 shards where the others have 4; without this, "evenly split"
# would quietly become 4000/4000/3000/4000/4000.
python - <<PY
import glob, json, os, random
pools = "$POOLS"
by_band = {}
for b in range(5):
    rows = []
    for f in sorted(glob.glob(os.path.join(pools, f"band{b}_shard*", "*.jsonl"))):
        with open(f) as fh:
            rows.extend(l for l in fh if l.strip())
    by_band[b] = rows
    print(f"  band {b}: {len(rows)} examples from "
          f"{len(glob.glob(os.path.join(pools, f'band{b}_shard*')))} shards")
per = min(len(v) for v in by_band.values())
print(f"  -> even split at {per}/band = {per*5} total")
rng = random.Random(1234)
with open("$COMBINED", "w") as w:
    for b in range(5):
        for line in rng.sample(by_band[b], per):
            w.write(line)
PY
echo "  combined: $(wc -l < "$COMBINED") examples"

echo "=== 3. decontaminate against the CTC oolong eval rungs ==="
# Same signature as debug/xlong_5task/decontaminate_oolong_train.py (query + first 200 chars of the
# packed stream). Seeds alone do not guarantee disjointness -- the 5task build measured 21.6/11.4/3.6%
# overlap at 8k/16k/32k before filtering. Drop from TRAIN, never regenerate the eval: the rungs are
# what every prior oolong number was measured on.
python - <<PY
import glob, json, os
def sig(ex):
    q = (ex.get("queries") or [""])[0]
    b = ex["documents"][0]["text"] if ex.get("documents") else ""
    return q + "||" + b[:200]

evalkeys = set()
for p in sorted(glob.glob("$EVALDIR/rung_*.jsonl")):
    n = 0
    for line in open(p):
        if line.strip():
            evalkeys.add(sig(json.loads(line))); n += 1
    print(f"  {os.path.basename(p)}: {n} eval examples")
print(f"  eval signature set: {len(evalkeys)}")

kept = dropped = 0
with open("$CLEAN", "w") as w:
    for line in open("$COMBINED"):
        if not line.strip():
            continue
        if sig(json.loads(line)) in evalkeys:
            dropped += 1
        else:
            w.write(line); kept += 1
pct = 100 * dropped / max(1, kept + dropped)
print(f"  kept={kept} dropped={dropped} ({pct:.2f}%)")
PY
rc=$?
[ $rc -ne 0 ] && { echo "!!! decontamination FAILED"; exit $rc; }

echo "=== 4. tokenize to a CTC-suite shard ==="
# query_position=after matches BOTH the 5task-mix setup this pool's source/setup comes from and the
# standing directive that new runs train query-after. Eval must therefore also render `after`;
# numbers are NOT comparable to the July `both`-mode oolong row (which was leaky anyway).
# cot_mode=none matches the CTC suite (the converter's DEFAULT is 'plan' -- do not rely on it).
# NO --item-regex: the default is the escaped r"\|\|"; a bare '||' is the leak.
rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"
python "$REPO/src/scripts/data/convert_unified_to_document_landmark.py" \
  --emit dense --task oolong --chunk-by line --cot-mode none \
  --marker-set qwen3_5 --tokenizer Qwen/Qwen3.5-0.8B-Base \
  --query-position after --seq-len 40960 --shard-tokens 60000000 \
  --input-jsonl "$CLEAN" --out-dir "$OUTDIR"
rc=$?
echo "  convert rc=$rc"
[ $rc -ne 0 ] && exit $rc

echo "=== 5. verify chunk layout (must be CLEAN -- 0 inter-chunk FREE tokens) ==="
python "$REPO/debug/ctc_modelscale/scan_oolong_chunks.py" --shard "$OUTDIR" --max-examples 400

echo "=== 6. part/byte consistency (the qdmatch_nq truncation class) ==="
python - <<PY
import glob, json, os
d = "$OUTDIR"
tok = sorted(glob.glob(os.path.join(d, "token_ids_part_*.npy")))
lab = sorted(glob.glob(os.path.join(d, "labels_mask_part_*.npy")))
m = json.load(open(os.path.join(d, "metadata.json")))
tb = sum(os.path.getsize(p) for p in tok); lb = sum(os.path.getsize(p) for p in lab)
print(f"token parts={len(tok)} bytes={tb}")
print(f"label parts={len(lab)} bytes={lb}")
print(f"num_instances={m['num_instances']} max_example_len={m['max_example_len']} "
      f"item_regex={m.get('item_regex')!r} query_position={m.get('query_position')!r}")
print("VERDICT:", "PASS" if (len(tok) == len(lab) and tb == m["num_tokens"] * 4
                             and lb == m["num_tokens"]) else "FAIL -- truncated")
PY
echo "=== DONE $(date '+%F %T') ==="
