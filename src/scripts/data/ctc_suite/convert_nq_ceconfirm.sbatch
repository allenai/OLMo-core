#!/bin/bash
#SBATCH --job-name=nq_ceconfirm_conv
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --account=site
#SBATCH --nodelist=mooney
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/users/prasann/ctc_suite_logs/conv_nq_ceconfirm_%j.log
# CE-CONFIRM convert: mirror convert_ctc_pilot_dense_cubbins.sbatch for nq exactly
# (TASK=retrieval CHUNK=document marker-set qwen3_5 seq-len 4096), on the
# CE-filtered regen. Harvest 6 shards -> 2500 rows -> tokenized dense shard.
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
SRC=/data/prasann/ctc_suite_data/nq_ceconfirm
HARVEST=/data/prasann/ctc_suite_data/pilot_train/nq_ceconfirm.jsonl
OUTDIR=/data/prasann/ctc_suite/data/nq_ceconfirm_train
CONV=$REPO/src/scripts/data/convert_unified_to_document_landmark.py
mkdir -p "$(dirname "$HARVEST")"

# Harvest: concat all shards, trim to 2500 (comparable to the collapsed pilot run).
cat "$SRC"/*.jsonl | head -n 2500 > "$HARVEST"
echo "harvested $(wc -l < "$HARVEST") rows -> $HARVEST"

source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate /data/prasann/conda/envs/corpus-reasoning-olmo
export PYTHONPATH="$REPO/src" HF_HOME=/data/prasann/hf_cache TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"
python "$CONV" --emit dense --task retrieval --chunk-by document --cot-mode none \
  --marker-set qwen3_5 --tokenizer Qwen/Qwen3.5-0.8B-Base \
  --seq-len 4096 --shard-tokens 20000000 \
  --input-jsonl "$HARVEST" --out-dir "$OUTDIR"
echo "convert rc=$? $(date '+%T')"

echo "=== VERIFY box markers $(date '+%T') ==="
python3 - "$OUTDIR" <<'PY'
import sys, glob, json, numpy as np
d = sys.argv[1]
try:
    meta = json.load(open(d + "/metadata.json"))
    print("metadata:", {k: meta.get(k) for k in ("marker_set","doc_start_id","doc_end_id","eos_token_id","task")})
except Exception as e:
    print("NO/BAD metadata.json:", e)
fs = sorted(glob.glob(d + "/token_ids_part_*.npy"))
if not fs:
    print("NO SHARDS"); sys.exit(1)
DS, DE, EOS = 248049, 248050, 248044
a = np.concatenate([np.fromfile(f, dtype=np.uint32) for f in fs])
ds=int((a==DS).sum()); de=int((a==DE).sum()); eos=int((a==EOS).sum())
ok = ds>0 and ds==de
print(f"toks={len(a)} examples(eos)={eos} box_start={ds} box_end={de} OK={ok}")
print("ALL_OK" if ok else "FAILED")
PY
echo "=== DONE $(date '+%T') ==="