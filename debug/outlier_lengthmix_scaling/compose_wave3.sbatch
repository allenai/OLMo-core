#!/bin/bash
#SBATCH --job-name=lm-wave3-arms
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --nodelist=mooney
#SBATCH --output=/data/prasann/joblogs/lm_wave3_arms_%j.log
#SBATCH --chdir=/data/prasann
set -uo pipefail
export HOME=/data/prasann/home HF_HOME=/data/prasann/hf_cache TMPDIR=/data/prasann/tmp
GENSRC=/data/prasann/repo/OLMo-core/src
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-eval/bin/python
WORK=/data/prasann/outlier_lengthmix
$PY /data/prasann/repo/OLMo-core/compose_arms_lengthmix.py --work "$WORK" || exit 1
tokarm () {
  local ARM=$1; local OUT="$WORK/arms_tokenized/$ARM"
  [ -s "$OUT/metadata.json" ] && { echo "[skip] $ARM"; return 0; }
  mkdir -p "$OUT"
  PYTHONPATH="$GENSRC" $PY "$GENSRC/corpus_reasoning/data/convert_unified_to_sft.py" \
    --task outlier --input "$WORK/arms/$ARM.jsonl" --out-dir "$OUT" \
    --tokenizer Qwen/Qwen3.5-0.8B-Base --max-seq-len 40960 --query-position after \
    --eos 248044 || touch "$WORK/arms_tokenized/FAILED_$ARM"
}
tokarm msb_1000 & tokarm msb_2000 & tokarm p8k_2000 & tokarm msb_8000 &
wait
tokarm msb_16000 & tokarm p8k_16000 & tokarm p8k_32000 &
wait
ls "$WORK"/arms_tokenized/FAILED_* 2>/dev/null && exit 1
export AWS_PROFILE=S3
aws s3 sync "$WORK/arms_tokenized" s3://ai2-llm/checkpoints/prasanns/outlier_lengthmix/arms --only-show-errors
echo "=== wave3 arms DONE $(date '+%F %T') ==="
