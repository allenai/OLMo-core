#!/bin/bash
# Step 2/2 of Beaker staging for the 512k/1M/2M ULTRA rungs: gantry job that syncs S3 -> weka.
# Weka is not reachable from Berkeley, and an S3 push ALONE is not enough -- every rung would log
# MISSING at eval time while the job still exits 0.
#
# Step 1 (cubbins /data -> S3) is debug/xlong_5task/upload_ultra_to_s3.sbatch.
#
# ⚠ TARGET ROOT: _eval_bundle_eval500_v2, NOT xlong5_2k256k_qwen35/eval.
# run_beaker_multirung_eval.sh sets EVAL500_ROOT="$WEKA_LLM/checkpoints/prasanns/_eval_bundle_eval500_v2"
# for LADDER_VERSION=v2 (its default; the live 256k jobs pass no override), so that bundle is what
# eval actually reads. xlong5_2k256k_qwen35/eval is the BUILD output bundle -- staging there looks
# successful and leaves every new rung invisible to eval. This script was pointed at the build
# bundle on its first run (2026-07-29) for exactly that reason; fixed here.
#
# The ultra rungs sit alongside the existing 64k/128k/256k rungs because eval_lc_native.py resolves
# them by size-labelled glob (*_xlong_{size}.jsonl). One file per size => unambiguous. Do NOT copy
# this bundle's 64k/128k/256k files in from elsewhere: the two bundles were calibrated differently
# (contra 256k is n6408 here vs n4944 in the build bundle) and a second file for one size would
# silently change which one sorted()[0] picks.
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
S3=s3://ai2-llm/checkpoints/prasanns/_eval_bundle_eval500_v2
WK=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/_eval_bundle_eval500_v2

echo "=== sync ultra rungs into the LIVE eval root (EVAL500_ROOT for v2) ==="
"$AWS" s3 sync "$S3" "$WK"
echo "sync rc=$?"

echo "=== VERIFY ON WEKA (weka side, not S3) ==="
fail=0
# Line-count on weka is the only check that proves the bytes actually landed; an S3-side listing
# would pass even if weka were empty.
for pat in \
  "contra/*_xlong_512k.jsonl" "contra/*_xlong_1M.jsonl" "contra/*_xlong_2M.jsonl" \
  "nq/*_xlong_512k.jsonl" "nq/*_xlong_1M.jsonl" "nq/*_xlong_2M.jsonl" \
  "outlier/*_xlong_512k.jsonl" "outlier/*_xlong_1M.jsonl" "outlier/*_xlong_2M.jsonl" \
  "rerank/*_xlong_512k.jsonl" "rerank/*_xlong_1M.jsonl" "rerank/*_xlong_2M.jsonl" \
  "oolong/oolong_test_synth_ctx524288_spliteval.jsonl" \
  "oolong/oolong_test_synth_ctx1048576_spliteval.jsonl" \
  "oolong/oolong_test_synth_ctx2097152_spliteval.jsonl" ; do
  n_hits=$(ls $WK/$pat 2>/dev/null | wc -l)
  if [ "$n_hits" = 0 ]; then echo "  MISSING: $pat"; fail=1; continue; fi
  # >1 file for a single size means sorted()[0] silently decides the rung -- treat as a failure.
  if [ "$n_hits" -gt 1 ]; then echo "  AMBIGUOUS ($n_hits files): $pat"; fail=1; fi
  hit=$(ls $WK/$pat 2>/dev/null | head -1)
  n=$(wc -l < "$hit")
  printf "  %-56s eval_size=%s\n" "$(basename "$hit")" "$n"
  [ "$n" -ge 500 ] || { echo "    ^ BELOW the 500 floor"; fail=1; }
done

echo "=== the 64k..2M ladder as eval will now see it ==="
for t in contra nq outlier rerank; do
  echo "  $t:"
  for s in 64k 128k 256k 512k 1M 2M; do
    f=$(ls $WK/$t/*_xlong_${s}.jsonl 2>/dev/null | head -1)
    [ -n "$f" ] && printf "    %-5s %s\n" "$s" "$(basename "$f")" || printf "    %-5s (none)\n" "$s"
  done
done
echo "  oolong:"
for c in 8192 16384 32768 65536 131072 262144 524288 1048576 2097152; do
  f=$WK/oolong/oolong_test_synth_ctx${c}_spliteval.jsonl
  [ -f "$f" ] && printf "    ctx%-8s eval_size=%s\n" "$c" "$(wc -l < "$f")" || printf "    ctx%-8s (none)\n" "$c"
done
du -sh "$WK" 2>/dev/null
[ "$fail" = 0 ] && echo "ULTRA_WEKA_SYNC_OK" || { echo "ULTRA_WEKA_SYNC_INCOMPLETE"; exit 4; }
'

gantry run --name xlong5-ultra-weka-sync-liveroot -w ai2/flex2 -b ai2/oe-other \
  --cluster 'ai2/*-cirrascale*' --gpus 0 --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --weka oe-training-default:/weka/oe-training-default \
  --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
  --no-python --allow-dirty --timeout 0 --yes -- bash -c "$WORK"
