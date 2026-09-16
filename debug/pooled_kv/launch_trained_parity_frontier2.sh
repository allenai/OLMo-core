#!/usr/bin/env bash
# Launch the eval-time CE-parity check on the two NEW ds64 first-k frontier arms
# (contradiction chk32-32M, outlier ck64-32M; records/ds64-overnight-2026-09-14.md 09-15 16:40/21:30
# entries), one 1-GPU gantry job per checkpoint. Mirrors debug/pooled_kv/launch_trained_parity.sh.
# outlier has no 2k eval rung file (eval_side_slot_probe.py:EVAL_JSONL only has 8k/16k/32k) so its
# job substitutes 16k for 2k.
set -uo pipefail
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$HOME/.local/bin:$PATH
cd /accounts/projects/berkeleynlp/prasann/projects/OLMo-core

launch() {
  local name="$1" ckpt="$2" rungs="$3" rows="$4" gen="$5"
  local work='
set -uo pipefail
export PATH=/opt/conda/bin:$PATH
export PYTHONPATH=$PWD/src
export TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
python -u debug/pooled_kv/trained_parity_check.py \
  --ckpt-name '"$ckpt"' \
  --rungs '"$rungs"' --rows '"$rows"' --gen-rows '"$gen"' \
  --work /results/tp_work_'"$ckpt"' --out /results/tp_'"$ckpt"'.json
RC=$?
echo "TRAINED_PARITY_DONE ckpt='"$ckpt"' rc=$RC"
exit $RC
'
  gantry run --name "$name" -w ai2/flex2 -b ai2/oe-other \
    --cluster ai2/ceres-cirrascale --cluster ai2/saturn-cirrascale --cluster ai2/jupiter-cirrascale-2 \
    --gpus 1 --priority urgent \
    --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
    --weka oe-training-default:/weka/oe-training-default \
    --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
    --install "pip install -e '.[all]' 2>&1 | tail -8" \
    --allow-dirty --timeout 0 --yes -- bash -c "$work"
}

launch "tp-chk32-32M" ds64-contradiction-chk32-b128f3-u32M "2k,8k,32k"    "240,240,120" "48,48,24"
launch "tp-ck64-32M"  ds64-outlier-ck64-b128f3-u32M        "8k,16k,32k"   "240,240,120" "48,48,24"
