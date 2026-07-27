#!/bin/bash
# Per-(task, arm) rung-ladder eval for the OLMo-3 CTC arm, run ON BEAKER (the Beaker-trained
# checkpoints live on weka and are far too large to pull back to Berkeley).
#
# One gantry job = one checkpoint x the 2k/4k/8k/16k ladder, looping run_rung_eval.py. Results are
# written to weka and relayed to S3 at the end so this host can pull the few-KB JSONs.
#
#   bash debug/ctc_olmo3/beaker_eval.sh <ckpt-dir-on-weka> <dense|chunked> <arm> <task> <eval-dir>
# e.g.
#   bash debug/ctc_olmo3/beaker_eval.sh \
#     /weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_suite/ckpts/ctc-olmo3-7b-contra-cmix-... \
#     chunked chunked-mix contradiction contradiction
set -uo pipefail
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$HOME/.local/bin:$PATH
cd /accounts/projects/berkeleynlp/prasann/projects/OLMo-core

CKPT="${1:?ckpt dir on weka}"
VARIANT="${2:?dense|chunked}"
ARM="${3:?full|chunked-mix}"
TASK="${4:-contradiction}"
EVALDIR="${5:-contradiction}"
NAME="olmo3-eval-$(echo "$ARM-$TASK" | tr '_' '-')-$(date +%H%M%S)"

WORK='
set -uo pipefail
export PYTHONPATH=$PWD/src
export TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
WK=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_olmo3
OUT=$WK/results
mkdir -p "$OUT"
echo "ckpt=CKPT_SUB variant=VARIANT_SUB arm=ARM_SUB task=TASK_SUB"
ls -d CKPT_SUB || { echo "FATAL: checkpoint MISSING"; exit 3; }
ls CKPT_SUB/config.json || { echo "FATAL: config.json MISSING"; exit 3; }
RC_ALL=0
for RUNG in 2048 4096 8192 16384; do
  JSONL=$WK/eval_rungs/EVALDIR_SUB/rung_${RUNG}.jsonl
  if [ ! -f "$JSONL" ]; then echo "MISSING eval rung $JSONL"; RC_ALL=1; continue; fi
  PORT=$((29000 + RANDOM % 1000))
  echo "########## rung=$RUNG port=$PORT $(date -u +%T) ##########"
  python -u -m scripts.eval.ctc_suite.run_rung_eval \
    --task TASK_SUB --ckpt CKPT_SUB --variant VARIANT_SUB --arm ARM_SUB \
    --rung-tokens $RUNG --eval-jsonl "$JSONL" \
    --model-scale olmo3-7b --nproc 8 --master-port $PORT \
    --tokenizer $WK/tokenizer \
    --doc-start-id 100266 --doc-end-id 100267 --eos-token-id 100257 \
    --small-eval-ok --out-root "$OUT"
  RC=$?; echo "---- rung $RUNG rc=$RC ----"
  [ $RC -ne 0 ] && RC_ALL=$RC
done
echo "=== relay results to S3 ==="
mkdir -p ~/.aws
printf "%s" "$AWS_CREDS" > ~/.aws/credentials
printf "%s" "$AWS_CFG" > ~/.aws/config
AWS=$(command -v aws || echo /opt/conda/bin/aws)
AWS_PROFILE=S3 "$AWS" s3 sync "$OUT" s3://ai2-llm/checkpoints/prasanns/ctc_olmo3/results --only-show-errors
echo "EVAL_LADDER_DONE rc=$RC_ALL"
'
WORK="${WORK//CKPT_SUB/$CKPT}"
WORK="${WORK//VARIANT_SUB/$VARIANT}"
WORK="${WORK//ARM_SUB/$ARM}"
WORK="${WORK//TASK_SUB/$TASK}"
WORK="${WORK//EVALDIR_SUB/$EVALDIR}"

gantry run --name "$NAME" -w ai2/flex2 -b ai2/oe-other \
  --cluster ai2/jupiter-cirrascale-2 --gpus 8 --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --weka oe-training-default:/weka/oe-training-default \
  --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
  --env-secret WANDB_API_KEY=PRASANNS_WANDB_API_KEY \
  --install true --allow-dirty --timeout 0 --yes -- bash -c "$WORK"
