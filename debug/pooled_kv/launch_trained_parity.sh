#!/usr/bin/env bash
# Launch the eval-time CE-parity check (debug/pooled_kv/trained_parity_check.py) on the ds64
# campaign's winning arms, one 1-GPU gantry job per checkpoint (oolong needs two: occ00 and
# ohdr08). See records/trained-parity-winning-arms.md for the results table and verdicts.
#
#   bash debug/pooled_kv/launch_trained_parity.sh [smoke]
#
# `smoke` runs a tiny (8-row, 1-rung) sanity pass per arm instead of the full rung set -- use it
# to catch a bug before spending the full ~1-2 GPU-hours/arm.
set -uo pipefail
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$HOME/.local/bin:$PATH
cd /accounts/projects/berkeleynlp/prasann/projects/OLMo-core

MODE="${1:-full}"
if [ "$MODE" = "smoke" ]; then
  RUNGS="2k"; ROWS="8"; GEN="2"; TAG="smoke"
else
  RUNGS="2k,8k,32k"; ROWS="240,240,120"; GEN="48,48,24"; TAG=""
fi

launch() {
  local name="$1" ckpt="$2"
  # NOTE: must run under /opt/conda/bin/python, and PATH must resolve `python` there too for the
  # driver's own subprocess shard-conversion call -- gantry's --install populates the baked image's
  # SYSTEM python (/opt/conda), but the job executes under a separate, empty /gantry-runtime/.venv,
  # so a bare `python -u ...` silently runs against an env with nothing installed (looks like
  # success in Beaker's exitCode unless the WORK script itself also propagates $RC -- see below).
  # `.[all]` (not the bare package) because Qwen3.5's GatedDeltaNet needs `fla`.  Both bugs cost a
  # full round of false-"succeeded" jobs on 2026-09-15/16 -- see records/trained-parity-winning-arms.md §3.
  local work='
set -uo pipefail
export PATH=/opt/conda/bin:$PATH
export PYTHONPATH=$PWD/src
export TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
python -u debug/pooled_kv/trained_parity_check.py \
  --ckpt-name '"$ckpt"' \
  --rungs '"$RUNGS"' --rows '"$ROWS"' --gen-rows '"$GEN"' \
  --work /results/tp_work_'"$ckpt"' --out /results/tp_'"$ckpt"'.json \
  --tag '"$TAG"'
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

launch "tp-${TAG:+$TAG-}occ00"  ds64-oolong-occ00-b128f3-u16M
launch "tp-${TAG:+$TAG-}ohdr08" ds64-oolong-ohdr08-b128f3-u16M
launch "tp-${TAG:+$TAG-}hdr33"  ds64-contradiction-hdr33-b128f3-u64M
launch "tp-${TAG:+$TAG-}kv33"   ds64-nq-kv33-b128f3-u64M
