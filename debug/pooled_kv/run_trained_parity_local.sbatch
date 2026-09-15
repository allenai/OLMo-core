#!/usr/bin/env bash
# Local (Berkeley, no weka) run of debug/pooled_kv/trained_parity_check.py. Checkpoints + eval
# rung mirrors are staged by a transfer agent to horton's node-local /data
# (/data/prasann/ckpts/<run>/, /data/prasann/ds64_eval/<task>/). This launcher can run EITHER on
# horton directly, OR on any other node with a free GPU (checked via `scontrol show node`) by doing
# a ONE-TIME bulk copy from horton's /data (read cross-node via /net, at NFS speed, but a single
# multi-GB read at job start -- not repeated job I/O) into the target node's own node-local /data.
# See records/trained-parity-winning-arms.md.
#
#   sbatch --partition=jsteinhardt --qos=preemptive_high --account=site --nodelist=cubbins \
#     --export=ALL,CKPT_NAME=ds64-contradiction-hdr33-b128f3-u64M,SRC_NODE=horton \
#     debug/pooled_kv/run_trained_parity_local.sbatch
#
# Env overrides: RUNGS (default 2k,8k,32k), ROWS (48,48,48 -- fast first pass), GEN (0,0,0 -- CE
# only first pass; set to e.g. 48,48,24 for the generation-inclusive follow-up), TAG, SRC_NODE
# (default horton -- where the transfer agent stages files), WAIT_MAX_S (default 5400s).
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
ROWS="${ROWS:-48,48,48}"
GEN="${GEN:-0,0,0}"
TAG="${TAG:-}"
SRC_NODE="${SRC_NODE:-horton}"
WAIT_MAX_S="${WAIT_MAX_S:-5400}"
THIS_NODE=$(hostname -s)

REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
cd "$REPO"
source src/scripts/local_env.sh

PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
case "$PY" in /data/*) ;; *) echo "!!! interpreter not node-local"; exit 1;; esac
export PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false

CKPT_ROOT=/data/prasann/ckpts/$CKPT_NAME
mkdir -p /data/prasann/joblogs /data/prasann/tp_results /data/prasann/ckpts /data/prasann/ds64_eval
WORKDIR=/data/prasann/tp_work_$CKPT_NAME
mkdir -p "$WORKDIR"

echo "running on $THIS_NODE, checkpoints staged on $SRC_NODE (waiting up to ${WAIT_MAX_S}s)"
WAITED=0
while true; do
  # bootstrap: local /data copy missing but the SRC_NODE (e.g. horton) already has it -> one-time
  # bulk cp via /net (never repeated job I/O -- this loop only fires the copy once).
  SRC_CK=$(ls -d "/net/$SRC_NODE/data/prasann/ckpts/$CKPT_NAME"*/model_and_optim 2>/dev/null | sort -V | tail -1)
  if [ -n "$SRC_CK" ] && [ ! -e "$CKPT_ROOT"*/model_and_optim ] 2>/dev/null; then
    SRC_RUN=$(dirname "$SRC_CK")
    DST_RUN="/data/prasann/ckpts/$(basename "$SRC_RUN")"
    if [ ! -d "$DST_RUN/model_and_optim" ]; then
      echo "bootstrap: copying $SRC_RUN -> $DST_RUN (one-time, $(date +%T))"
      mkdir -p "$DST_RUN"
      cp -r "$SRC_RUN"/. "$DST_RUN"/ && echo "bootstrap copy done $(date +%T)"
    fi
  fi
  CK=$(ls -d "$CKPT_ROOT"*/model_and_optim "$CKPT_ROOT"*/step*/model_and_optim 2>/dev/null | sort -V | tail -1)
  [ -n "$CK" ] && [ -f "$CK/../config.json" -o -f "$CK/config.json" ] && break
  if [ "$WAITED" -ge "$WAIT_MAX_S" ]; then
    echo "FATAL: $CKPT_NAME checkpoint never appeared (local or on $SRC_NODE) after ${WAITED}s"; exit 3
  fi
  [ $((WAITED % 300)) -eq 0 ] && echo "  ...still waiting (${WAITED}s)"
  sleep 30; WAITED=$((WAITED + 30))
done
echo "checkpoint present after ${WAITED}s: $CK"

# eval-rung + slot-stop-shard mirrors: same bootstrap, once, from SRC_NODE if not already local.
for TASK_DIR in contradiction oolong nq outlier; do
  if [ ! -d "/data/prasann/ds64_eval/$TASK_DIR" ] && [ -d "/net/$SRC_NODE/data/prasann/ds64_eval/$TASK_DIR" ]; then
    echo "bootstrap: copying ds64_eval/$TASK_DIR from $SRC_NODE ($(date +%T))"
    cp -r "/net/$SRC_NODE/data/prasann/ds64_eval/$TASK_DIR" /data/prasann/ds64_eval/ && echo "  done $(date +%T)"
  fi
done
if [ ! -d /data/prasann/ds64_shards ] && [ -d "/net/$SRC_NODE/data/prasann/ds64_shards" ]; then
  echo "bootstrap: copying ds64_shards from $SRC_NODE ($(date +%T))"
  cp -r "/net/$SRC_NODE/data/prasann/ds64_shards" /data/prasann/ds64_shards && echo "  done $(date +%T)"
fi

$PY -u debug/pooled_kv/trained_parity_check.py \
  --ckpt-name "$CKPT_NAME" --ckpt "$CKPT_ROOT" \
  --rungs "$RUNGS" --rows "$ROWS" --gen-rows "$GEN" \
  --eval-dir /data/prasann/ds64_eval --slot-stop-shard-root /data/prasann/ds64_shards \
  --work "$WORKDIR" --out "/data/prasann/tp_results/tp_${CKPT_NAME}${TAG:+_$TAG}.json" --weka-out none \
  --tag "$TAG"
RC=$?
echo "TRAINED_PARITY_LOCAL_DONE ckpt=$CKPT_NAME rc=$RC"
exit $RC
