#!/bin/bash
# Step 2/2 of Beaker staging for the 512k/1M/2M ULTRA rungs: gantry job that syncs S3 -> weka.
# Weka is not reachable from Berkeley, and an S3 push ALONE is not enough -- every rung would log
# MISSING at eval time while the job still exits 0.
#
# Step 1 (cubbins /data -> S3) is debug/xlong_5task/upload_ultra_to_s3.sbatch.
set -uo pipefail
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
cd /accounts/projects/berkeleynlp/prasann/projects/OLMo-core

WORK='
set -e
export PATH=/opt/conda/bin:$PATH
if ! command -v aws >/dev/null 2>&1; then
  echo "installing awscli into conda env..."; python -m pip install -q awscli
fi
AWS=$(command -v aws); echo "using aws: $AWS"
mkdir -p ~/.aws
printf "%s" "$AWS_CREDS" > ~/.aws/credentials
printf "%s" "$AWS_CFG" > ~/.aws/config
export AWS_PROFILE=S3
S3=s3://ai2-llm/checkpoints/prasanns/xlong5_2k256k_qwen35/eval
WK=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/xlong5_2k256k_qwen35/eval

echo "=== sync ultra rungs into the existing eval bundle ==="
"$AWS" s3 sync "$S3" "$WK"
echo "sync rc=$?"

echo "=== VERIFY ON WEKA (weka side, not S3) ==="
fail=0
# The ultra rungs must be present AND >=500 examples. Counting lines on weka is the only check
# that proves the bytes actually landed -- an S3-side listing would pass even if weka were empty.
for pat in \
  "contra/*_xlong_512k.jsonl" "contra/*_xlong_1M.jsonl" "contra/*_xlong_2M.jsonl" \
  "nq/*_xlong_512k.jsonl" "nq/*_xlong_1M.jsonl" "nq/*_xlong_2M.jsonl" \
  "outlier/*_xlong_512k.jsonl" "outlier/*_xlong_1M.jsonl" "outlier/*_xlong_2M.jsonl" \
  "rerank/*_xlong_512k.jsonl" "rerank/*_xlong_1M.jsonl" "rerank/*_xlong_2M.jsonl" \
  "oolong/oolong_test_synth_ctx524288_spliteval.jsonl" \
  "oolong/oolong_test_synth_ctx1048576_spliteval.jsonl" \
  "oolong/oolong_test_synth_ctx2097152_spliteval.jsonl" ; do
  hit=$(ls $WK/$pat 2>/dev/null | head -1)
  if [ -z "$hit" ]; then
    echo "  MISSING: $pat"; fail=1; continue
  fi
  n=$(wc -l < "$hit")
  printf "  %-56s eval_size=%s\n" "$(basename "$hit")" "$n"
  [ "$n" -ge 500 ] || { echo "    ^ BELOW the 500 floor"; fail=1; }
done

echo "=== full ladder now visible per task ==="
for t in contra nq outlier rerank oolong; do
  echo "  $t: $(ls $WK/$t/*.jsonl 2>/dev/null | wc -l) jsonl"
done
du -sh "$WK" 2>/dev/null
[ "$fail" = 0 ] && echo "ULTRA_WEKA_SYNC_OK" || { echo "ULTRA_WEKA_SYNC_INCOMPLETE"; exit 4; }
'

gantry run --name xlong5-ultra-weka-sync -w ai2/flex2 -b ai2/oe-other \
  --cluster 'ai2/*-cirrascale*' --gpus 0 --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --weka oe-training-default:/weka/oe-training-default \
  --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
  --no-python --allow-dirty --timeout 0 --yes -- bash -c "$WORK"
