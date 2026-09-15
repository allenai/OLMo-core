#!/usr/bin/env bash
# Local (Berkeley, no weka) run of debug/pooled_kv/trained_parity_check.py, pinned to horton where
# the transfer agent is staging model-only checkpoint copies + eval rung mirrors (see
# records/trained-parity-winning-arms.md). Waits for its OWN checkpoint's model_and_optim to
# appear on /data before running, so all arms can be submitted up front.
#
#   sbatch --export=ALL,CKPT_NAME=ds64-contradiction-hdr33-b128f3-u64M \
#     debug/pooled_kv/run_trained_parity_local.sbatch
#   sbatch --export=ALL,CKPT_NAME=ds64-nq-kv33-b128f3-u64M \
#     debug/pooled_kv/run_trained_parity_local.sbatch
#   sbatch --export=ALL,CKPT_NAME=ds64-oolong-occ00-b128f3-u16M \
#     debug/pooled_kv/run_trained_parity_local.sbatch   # needs the oolong_u16M training shard too
#
# Env overrides: RUNGS (default 2k,8k,32k), ROWS (240,240,120), GEN (48,48,24), TAG, WAIT_MAX_S
# (default 5400s checkpoint-wait timeout).
#SBATCH --job-name=tp-local
#SBATCH --account=site
#SBATCH --partition=berkeleynlp
#SBATCH --qos=preemptive_high_sewonm
#SBATCH --nodelist=horton
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/data/prasann/joblogs/tp-local_%x_%j.log

set -uo pipefail
: "${CKPT_NAME:?set CKPT_NAME, e.g. ds64-contradiction-hdr33-b128f3-u64M}"
RUNGS="${RUNGS:-2k,8k,32k}"
ROWS="${ROWS:-240,240,120}"
GEN="${GEN:-48,48,24}"
TAG="${TAG:-}"
WAIT_MAX_S="${WAIT_MAX_S:-5400}"

REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
cd "$REPO"
source src/scripts/local_env.sh

PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
case "$PY" in /data/*) ;; *) echo "!!! interpreter not node-local"; exit 1;; esac
export PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false

CKPT_ROOT=/data/prasann/ckpts/$CKPT_NAME
mkdir -p /data/prasann/joblogs /data/prasann/tp_results
WORKDIR=/data/prasann/tp_work_$CKPT_NAME
mkdir -p "$WORKDIR"

echo "waiting for $CKPT_ROOT*/model_and_optim/config.json ..."
WAITED=0
while true; do
  CK=$(ls -d "$CKPT_ROOT"*/model_and_optim "$CKPT_ROOT"*/step*/model_and_optim 2>/dev/null | sort -V | tail -1)
  [ -n "$CK" ] && [ -f "$CK/../config.json" -o -f "$CK/config.json" ] && break
  if [ "$WAITED" -ge "$WAIT_MAX_S" ]; then
    echo "FATAL: $CKPT_NAME checkpoint never appeared on /data after ${WAITED}s"; exit 3
  fi
  [ $((WAITED % 300)) -eq 0 ] && echo "  ...still waiting (${WAITED}s)"
  sleep 30; WAITED=$((WAITED + 30))
done
echo "checkpoint present after ${WAITED}s: $CK"

$PY -u debug/pooled_kv/trained_parity_check.py \
  --ckpt-name "$CKPT_NAME" --ckpt "$CKPT_ROOT" \
  --rungs "$RUNGS" --rows "$ROWS" --gen-rows "$GEN" \
  --eval-dir /data/prasann/ds64_eval --slot-stop-shard-root /data/prasann/ds64_shards \
  --work "$WORKDIR" --out "/data/prasann/tp_results/tp_${CKPT_NAME}.json" --weka-out none \
  --tag "$TAG"
RC=$?
echo "TRAINED_PARITY_LOCAL_DONE ckpt=$CKPT_NAME rc=$RC"
exit $RC
