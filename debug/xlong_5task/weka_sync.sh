#!/bin/bash
# Step 2/2 of weka staging for the 2k->256k 5-task build: gantry job that syncs S3 -> weka.
# Weka is not reachable from Berkeley, and an S3 push ALONE is not enough -- every file would log
# MISSING at train/eval time while the job still exits 0.
#
# Step 1 (local /data -> S3) is debug/xlong_5task/upload_to_s3.sbatch.
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
S3=s3://ai2-llm/checkpoints/prasanns/xlong5_2k256k_qwen35
WK=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/xlong5_2k256k_qwen35

echo "=== sync shards + eval ==="
"$AWS" s3 sync "$S3" "$WK"
echo "sync rc=$?"

echo "=== VERIFY ON WEKA (weka side, not S3) ==="
fail=0
for v in shards shards_full; do
for t in contradiction nq oolong outlier rerank; do
  d="$WK/$v/${t}_train"
  n=$(ls "$d" 2>/dev/null | wc -l)
  tok=$(ls "$d"/token_ids_part_*.npy 2>/dev/null | wc -l)
  msk=$(ls "$d"/labels_mask_part_*.npy 2>/dev/null | wc -l)
  meta=$([ -f "$d/metadata.json" ] && echo yes || echo NO)
  echo "  $v/${t}_train: files=$n token_parts=$tok mask_parts=$msk metadata=$meta"
  { [ "$tok" -gt 0 ] && [ "$tok" -eq "$msk" ] && [ "$meta" = yes ]; } || { echo "    ^ INCOMPLETE"; fail=1; }
done
done
echo "  eval rungs: $(find "$WK/eval" -name "*.jsonl" 2>/dev/null | wc -l) jsonl"
find "$WK/eval" -name "*.jsonl" 2>/dev/null | sort | while read f; do
  echo "    $(wc -l < "$f") $(basename "$f")"
done
[ -f "$WK/README.md" ] && echo "  README.md present" || { echo "  README.md MISSING"; fail=1; }
du -sh "$WK" 2>/dev/null
[ "$fail" = 0 ] && echo "WEKA_SYNC_OK" || { echo "WEKA_SYNC_INCOMPLETE"; exit 4; }
'

gantry run --name xlong5-weka-sync2 -w ai2/flex2 -b ai2/oe-other \
  --cluster 'ai2/*-cirrascale*' --gpus 0 --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --weka oe-training-default:/weka/oe-training-default \
  --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
  --no-python --allow-dirty --timeout 0 --yes -- bash -c "$WORK"
