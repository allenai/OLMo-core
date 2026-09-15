#!/usr/bin/env bash
# Accuracy conditioned on GOLD OUTPUT LENGTH for the 256k dense vs 256k compressive-LM pair,
# as a Beaker CPU gantry job.
#
# The inputs are the <task>_multirung.generations.jsonl sidecars the multirung ladder already
# wrote next to each checkpoint on weka -- no GPU, no checkpoint load, no re-decoding. weka is
# not mounted at Berkeley, which is the only reason this needs a job at all.
#
# The full report is printed to stdout as well as written to weka, so it can be pulled straight
# out of the beaker logs without a second job:
#     beaker experiment results <id>       # or: beaker job logs <job-id>
#
# Usage:
#   src/scripts/train/memexpress/singletask_ladder/gold_length/launch_gold_length_gantry.sh
#   EVAL_DIRS='eval,eval_xlong,eval_xlong256k,eval_yarn2-256k' analysis/.../launch_gold_length_gantry.sh
#   RUNGS='2k,3k,8k,16k,32k' NAME=gold-len-short-rungs analysis/.../launch_gold_length_gantry.sh
#
# Overridable: CLUSTER WORKSPACE BUDGET WEKA PRIORITY CPUS NAME IMAGE DENSE COMPRESSIVE
#              EVAL_DIRS LADDER_VERSION RUNGS TASKS TOKENIZER OUT_DIR ITEM_EDGES TOKEN_EDGES
#              INCLUDE_FIXED
#
# PREREQ: commit AND push first -- gantry clones the repo at your pushed HEAD, so an unpushed
# analysis script means the job runs without it (beaker.md golden rule 1).
set -euo pipefail

# CPU job -> multi-cluster eager-first. Jupiter alone is strict-priority backfill and can queue a
# 0-GPU job for 10+ minutes (beaker.md golden rule 4).
CLUSTER="${CLUSTER:-ai2/neptune,ai2/ceres,ai2/saturn,ai2/jupiter}"
WORKSPACE="${WORKSPACE:-ai2/flex2}"
BUDGET="${BUDGET:-ai2/oe-other}"
WEKA="${WEKA:-oe-training-default}"
PRIORITY="${PRIORITY:-urgent}"      # ALWAYS urgent (CLAUDE.md / beaker.md golden rule 2)
CPUS="${CPUS:-4}"
NAME="${NAME:-gold-length-256k-dense-vs-compressive}"
IMAGE="${IMAGE:-tylerr/olmo-core-tch291cu128-2025-11-25}"

CKPT_ROOT="/weka/${WEKA}/ai2-llm/checkpoints/amandab"
DENSE="${DENSE:-${CKPT_ROOT}/q35-4b-dense-xlong5-dolci25-256k}"
COMPRESSIVE="${COMPRESSIVE:-${CKPT_ROOT}/q35-4b-fastcomplm-xlong5-dolci25-256k}"

# Native-RoPE main evals only. The eval_yarn2-*/eval_yarn4-* dirs serve a DIFFERENT RoPE scaling,
# so pooling them with the native dirs would compare two serving configs inside one bucket.
EVAL_DIRS="${EVAL_DIRS:-eval,eval_xlong,eval_xlong256k}"
LADDER_VERSION="${LADDER_VERSION:-v2}"
RUNGS="${RUNGS:-}"
TASKS="${TASKS:-}"
ITEM_EDGES="${ITEM_EDGES:-1,2,3,4,6,11}"
TOKEN_EDGES="${TOKEN_EDGES:-1,2,4,8,16,32,64}"
INCLUDE_FIXED="${INCLUDE_FIXED:-0}"   # 1 -> also fold rerank (fixed top-10 output) into the macro

# Staged on weka by src/scripts/data/stage_tokenizers_weka.py, so the job never touches the HF hub
# (a single Xet 429 aborts snapshot_download outright). The script degrades to a word count and
# says so in the report if this path is missing, rather than failing the run.
TOKENIZER="${TOKENIZER:-/weka/${WEKA}/amandab/tokenizers/Qwen__Qwen3.5-0.8B}"

STAMP="$(date -u +%Y%m%d-%H%M%S)"
OUT_DIR="${OUT_DIR:-/weka/${WEKA}/ai2-llm/checkpoints/amandab/_analysis/gold_length/${STAMP}}"

CLUSTER_ARGS=()
IFS=',' read -ra _CLUSTERS <<< "${CLUSTER}"
for c in "${_CLUSTERS[@]}"; do CLUSTER_ARGS+=(--cluster "$c"); done

FIXED_FLAG=""
[ "${INCLUDE_FIXED}" = "1" ] && FIXED_FLAG="--include-fixed-output-tasks"

read -r -d '' JOB <<'EOS' || true
set -euo pipefail
echo "=== gold-length-conditioned accuracy | host=$(hostname) START=$(date -u '+%F %T')Z ==="
echo "    dense       = $DENSE"
echo "    compressive = $COMPRESSIVE"
echo "    eval dirs   = $EVAL_DIRS (ladder $LADDER_VERSION)"
echo "    out         = $OUT_DIR"

# Fail loudly and early if the generations sidecars are not where we think they are: an empty
# glob otherwise reads downstream as "no length effect" instead of "no input".
found=0
for root in "$DENSE" "$COMPRESSIVE"; do
  echo "--- sidecars under $root:"
  n=$(find "$root" -maxdepth 2 -name '*_multirung*.generations.jsonl' 2>/dev/null | wc -l | tr -d ' ')
  find "$root" -maxdepth 2 -name '*_multirung*.generations.jsonl' 2>/dev/null \
    | sort | while read -r f; do printf '  %10s lines  %s\n' "$(wc -l < "$f")" "$f"; done
  echo "    -> $n file(s)"
  found=$((found + n))
done
if [ "$found" -eq 0 ]; then
  echo "FATAL: no *_multirung*.generations.jsonl under either checkpoint root." >&2
  echo "       The ladder writes them next to the result JSON; check the eval tag dirs." >&2
  exit 1
fi

[ -d "$TOKENIZER" ] || echo "[warn] tokenizer dir $TOKENIZER missing -- report will fall back to word counts"

python -u src/scripts/train/memexpress/singletask_ladder/gold_length/gold_length_conditioned_accuracy.py \
  --model "dense=$DENSE" \
  --model "compressive=$COMPRESSIVE" \
  --eval-dirs "$EVAL_DIRS" \
  --ladder-version "$LADDER_VERSION" \
  --rungs "$RUNGS" \
  --tasks "$TASKS" \
  --tokenizer "$TOKENIZER" \
  --item-edges "$ITEM_EDGES" \
  --token-edges "$TOKEN_EDGES" \
  $FIXED_FLAG \
  --out-dir "$OUT_DIR"

echo "=== DONE $(date -u '+%F %T')Z -- report also at $OUT_DIR/report.md ==="
EOS

gantry run \
  --name "${NAME}" \
  --description "Accuracy by gold output length: 256k dense vs 256k compressive-LM" \
  --workspace "${WORKSPACE}" \
  --budget "${BUDGET}" \
  "${CLUSTER_ARGS[@]}" \
  --beaker-image "${IMAGE}" \
  --weka "${WEKA}:/weka/${WEKA}" \
  --cpus "${CPUS}" \
  --gpus 0 \
  --priority "${PRIORITY}" \
  --install true \
  --timeout 0 \
  --env "DENSE=${DENSE}" \
  --env "COMPRESSIVE=${COMPRESSIVE}" \
  --env "EVAL_DIRS=${EVAL_DIRS}" \
  --env "LADDER_VERSION=${LADDER_VERSION}" \
  --env "RUNGS=${RUNGS}" \
  --env "TASKS=${TASKS}" \
  --env "TOKENIZER=${TOKENIZER}" \
  --env "ITEM_EDGES=${ITEM_EDGES}" \
  --env "TOKEN_EDGES=${TOKEN_EDGES}" \
  --env "FIXED_FLAG=${FIXED_FLAG}" \
  --env "OUT_DIR=${OUT_DIR}" \
  --yes \
  -- bash -c "${JOB}"

echo
echo "Launched ${NAME} (priority ${PRIORITY}). Report -> ${OUT_DIR}/report.md"
echo "Pull it without weka:  beaker experiment results <experiment-id>   (report is on stdout too)"
