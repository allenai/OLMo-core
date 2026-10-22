#!/bin/bash
# Small 4-task fast8k shard for the Lambda LR sweep (tokdrop_lr), shipped to S3 so it can be pulled onto a
# Berkeley node and streamed into a Lambda node's /tmp (Lambda has no weka and a full /accounts quota).
#
#   ROWS=488 bash research/token_routing/ds64_fast8k/build_mt4_small_s3_beaker.sh
#   ROWS="976 1464" bash ...   # several sizes in one job; first-N rows, so each is a superset of the smaller
#
# ROWS rows per task (488 = 4M nominal tokens at 8192/row; x4 tasks = 1,952 rows = 122 steps at 16 rows/step,
# the September fast8k 16M step count). Writes $WEKA/ds64/fast8k/shards/mt4s<ROWS> and syncs it, plus the 8k/2k
# eval rung files of the four tasks, to $DST.
set -uo pipefail
ROWS="${ROWS:-488}"
WEKA=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
DST="${DST:-s3://ai2-llm/checkpoints/prasanns/_transfer/tokdrop_lr}"
gantry run --name "fast8k-mt4s${ROWS// /-}-s3-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other \
  --cluster ai2/phobos --cluster ai2/hammond --cluster 'ai2/jupiter*' --cluster 'ai2/ceres*' \
  --gpus 0 --cpus 4 --memory 32GiB --priority urgent --min-runtime 1h \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 --install false \
  --weka oe-training-default:/weka/oe-training-default \
  --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
  --env WEKA="$WEKA" --env ROWS="$ROWS" --env DST="$DST" \
  --allow-dirty --timeout 0 --yes -- bash -c '
set -uo pipefail
PYB=/opt/conda/bin/python
F=$WEKA/ds64/fast8k
mkdir -p ~/.aws && printf "%s\n" "$AWS_CREDS" > ~/.aws/credentials && printf "%s\n" "$AWS_CFG" > ~/.aws/config
export AWS_PROFILE=S3
pip install -q awscli >/dev/null 2>&1 || { echo "!!! awscli install failed"; exit 1; }
for R in $ROWS; do
  OUT=$F/shards/mt4s$R
  if [ -s $OUT/metadata.json ]; then echo "[skip] $OUT built"; else
    $PYB research/token_routing/ds64_fast8k/merge_tasks.py --out $OUT --rows-per-src $R --name mt4 \
      --src $F/shards/contradiction_g16M $F/shards/nq_g16M $F/shards/outlier_g16M $F/shards/oolong_g16M_eg || { echo "!!! merge failed $R"; exit 1; }
  fi
  aws s3 sync --quiet $OUT $DST/shards/mt4s$R || { echo "!!! sync failed $R"; exit 1; }
  echo "[merge] mt4s$R synced"
done
for f in _eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n190_k3.jsonl \
         _eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n100_k3.jsonl \
         outlier_lengthmix/eval_rungs/nq/rung_8192.jsonl outlier_lengthmix/eval_rungs/nq/rung_2048.jsonl \
         outlier_lengthmix/eval_rungs/outlier/rung_8192.jsonl outlier_lengthmix/eval_rungs/outlier/rung_2048.jsonl \
         _eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx8192_spliteval.jsonl \
         _eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx2048_spliteval.jsonl; do
  aws s3 cp --quiet $WEKA/$f $DST/eval/$f || { echo "!!! eval copy failed $f"; exit 1; }
done
aws s3 ls --recursive $DST | awk "{s+=\$3} END {print \"S3 objects:\", NR, \"bytes:\", s}"
echo S3_DONE'
