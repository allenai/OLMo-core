#!/bin/bash
# CTC-suite OLMo-3 arm, hotpotqa (O(N) retrieval): build a ~10k ladder-matched training shard with
# the OLMo/dolma2 marker set.
#
# NO regeneration needed. The BUILD_MATRIX §2 recipe (n in 11/24/50/100/205, bridge, hard-negs n/10,
# seed 42) was already run for the Qwen arm, and its RAW unified JSONL survives on cubbins' node-local
# /data -- only the tokenized shard is Qwen-specific. So this subsamples 2000 per n-value from the
# existing 4000-per-n pool (10k total, discrete-uniform over n, exactly the documented convention)
# and re-tokenizes with the OLMo tokenizer + --marker-set olmo3.
#
#   sbatch --partition=jsteinhardt --qos=preemptive --account=site -w cubbins \
#     src/scripts/data/ctc_suite/convert_hpqa_olmo3_cubbins.sbatch
#SBATCH --job-name=hpqa-olmo3-conv
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/data/prasann/joblogs/hpqa_olmo3_conv_%j.log

set -uo pipefail
MIRROR=/scratch/users/prasann/ctc_suite_logs
mkdir -p "$MIRROR" 2>/dev/null || true
_S=/data/prasann/joblogs/hpqa_olmo3_conv_${SLURM_JOB_ID}.log
_D=$MIRROR/hpqa_olmo3_conv_${SLURM_JOB_ID}.log
( while true; do cp -f "$_S" "$_D" 2>/dev/null; sleep 20; done ) & _MP=$!
trap 'cp -f "$_S" "$_D" 2>/dev/null; kill $_MP 2>/dev/null' EXIT

REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$ENV" ] || ENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$REPO/src"
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HOME=/data/prasann/home_cache
mkdir -p "$HOME" 2>/dev/null || true

RAW=${RAW:-/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/ctc_multifamily/hpqa_raw}
TOKENIZER=/scratch/users/prasann/hf_models/Olmo-3-1025-7B-docchunk
OUT=/scratch/users/prasann/ctc_suite_staged/shards_olmo3/hotpotqa_train
TMP=/data/prasann/ctc_suite_olmo3_tmp
CONV=$REPO/src/scripts/data/convert_unified_to_document_landmark.py
mkdir -p "$OUT" "$TMP"
PER_N=${PER_N:-2000}

echo "=== HOST=$(hostname) START=$(date '+%F %T') per_n=$PER_N ==="
ls -la "$RAW" 2>&1 | head -10

# Discrete-uniform over the 5 ladder rungs: 2000 examples from each n-value file.
python - "$RAW" "$TMP/hotpotqa_10k.jsonl" "$PER_N" <<'PY'
import glob, json, os, random, sys
data, out, per_n = sys.argv[1], sys.argv[2], int(sys.argv[3])
files = sorted(glob.glob(os.path.join(data, "*.jsonl")))
if not files:
    sys.exit(f"FATAL: no hotpotqa pool files under {data}")
rng = random.Random(42)
total = 0
with open(out, "w") as w:
    for f in files:
        lines = [ln for ln in open(f) if ln.strip()]
        keep = rng.sample(lines, per_n) if len(lines) > per_n else lines
        ndocs = json.loads(keep[0]).get("documents", [])
        print(f"  {os.path.basename(f)}: pool={len(lines)} kept={len(keep)} n_docs={len(ndocs)}")
        for ln in keep:
            w.write(ln if ln.endswith("\n") else ln + "\n")
        total += len(keep)
print(f"wrote {total} examples -> {out}")
PY
RC=$?
[ $RC -ne 0 ] && { echo "FATAL: subsample failed rc=$RC"; exit $RC; }

python -u "$CONV" --task retrieval \
  --input-jsonl "$TMP/hotpotqa_10k.jsonl" \
  --out-dir "$OUT" \
  --marker-set olmo3 --tokenizer "$TOKENIZER" \
  --emit dense --chunk-by document --cot-mode none --seq-len 32768 --num-proc 8
RC=$?
echo "--- metadata ---"; cat "$OUT/metadata.json"
echo "=== DONE rc=$RC $(date '+%F %T') ==="
