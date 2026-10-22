#!/bin/bash
# Small 11-task screening shard for the universal-keep-rule screen (tokdrop_lr): ROWS rows per setA task, sampled
# uniformly among rows <= MAXLEN tokens (the 2k-8k rungs), with gold sidecars (setA32k gold re-tokenization,
# research/token_routing/setA_pareto/), shipped to S3 for staging onto Lambda node /tmp.
#
#   ROWS=240 bash research/token_routing/ds64_fast8k/build_mt11_small_s3_beaker.sh
#   ROWS="480 720" bash ...   # several sizes per job (uniform samples with the same seed: NOT nested)
set -uo pipefail
ROWS="${ROWS:-240}"; MAXLEN="${MAXLEN:-10240}"
WEKA=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
DST="${DST:-s3://ai2-llm/checkpoints/prasanns/_transfer/tokdrop_lr}"
gantry run --name "fast8k-mt11s${ROWS// /-}-s3-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other \
  --cluster 'ai2/jupiter*' --cluster 'ai2/ceres*' --cluster ai2/phobos --cluster ai2/hammond \
  --gpus 0 --cpus 4 --memory 64GiB --priority urgent --min-runtime 1h \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 --install false \
  --weka oe-training-default:/weka/oe-training-default \
  --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
  --env WEKA="$WEKA" --env ROWS="$ROWS" --env MAXLEN="$MAXLEN" --env DST="$DST" \
  --allow-dirty --timeout 0 --yes -- bash -c '
set -uo pipefail
PYB=/opt/conda/bin/python
SRC=$WEKA/ctc_sft_sets/setA_max20_evaliid/shards_qwen3_5_gold
mkdir -p ~/.aws && printf "%s\n" "$AWS_CREDS" > ~/.aws/credentials && printf "%s\n" "$AWS_CFG" > ~/.aws/config
export AWS_PROFILE=S3
pip install -q awscli >/dev/null 2>&1 || { echo "!!! awscli install failed"; exit 1; }
for R in $ROWS; do
  OUT=$WEKA/ds64/fast8k/shards/mt11s$R
  if [ -s $OUT/metadata.json ]; then echo "[skip] $OUT built"; else
    $PYB research/token_routing/ds64_fast8k/merge_tasks.py --out $OUT --name mt11 --rows-per-src $R \
      --max-row-len $MAXLEN --sample-seed 20261022 \
      --src $(for t in nq hotpotqa qdmatch_nq outlier oolong contradiction xabsence reorder rerank strmatch grouping; do echo $SRC/$t; done) || { echo "!!! merge failed $R"; exit 1; }
  fi
  aws s3 sync --quiet $OUT $DST/shards/mt11s$R || { echo "!!! sync failed $R"; exit 1; }
  echo "[merge] mt11s$R synced"
done
aws s3 ls --recursive $DST/shards | grep mt11 | awk "{s+=\$3} END {print \"S3 objects:\", NR, \"bytes:\", s}"
echo S3_DONE'
