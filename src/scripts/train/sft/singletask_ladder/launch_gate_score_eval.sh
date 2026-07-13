#!/bin/bash
# Submit the ON-BEAKER landmark GATE-SCORE analysis eval: run the 5 SFT tasks (contra, nq, rerank,
# outlier, oolong) at 50 examples/task for ONE landmark checkpoint, logging every decode step's
# landmark gate scores (olmo_core.nn.attention.landmark_gate_analysis) so we can measure how peaky
# the gate-score distribution is. One Beaker job per task; fully on Beaker (code+data+checkpoint from
# weka). Modeled on launch_beaker_multirung_eval.sh.
#
# Each job decodes with hard top-k retrieval at landmark_top_k_fraction=0.1 (the GenerationConfig
# default: keep the top 10% of landmark blocks per step) and writes, per task, per GPU worker, a JSONL
# gate-score log to  $GATE_LOG_DIR/gate_<RUN_NAME>_<task>.rank<N>  -- readable on weka after the run.
# --gate-log-all records EVERY candidate block's score each step (the full distribution), not just the
# top-k kept ones. Set TOPK=<int> to instead pin a fixed landmark_top_k_blocks count.
#
# Usage:
#   CKPT=/weka/oe-training-default/ai2-llm/checkpoints/amandab/<run>/stepNNNN \
#   GATE_LOG_DIR=/weka/oe-training-default/ai2-llm/checkpoints/amandab/<run>/gate_scores \
#   bash src/scripts/train/sft/singletask_ladder/launch_gate_score_eval.sh <RUN_NAME>
#
#   DRY=1 ... bash .../launch_gate_score_eval.sh <RUN_NAME>          # build, don't submit
#   TASKS="contra nq" ... bash .../launch_gate_score_eval.sh <RUN>   # subset of tasks
#
# Env:
#   CKPT          (required) absolute weka step dir of the landmark checkpoint to analyze.
#   GATE_LOG_DIR  (required) absolute weka dir for the gate-score JSONL logs (created on-node).
#   RUN_NAME      (arg 1, default "gatescore") label used in job names, result dir, and log filenames.
#   TASKS         default "contra nq rerank outlier oolong" (the 5 SFT tasks).
#   VARIANT       default "landmark" (gate logging is landmark/compressive only).
#   TOPK          default "" (use the eval's landmark_top_k_fraction=0.1 default). Set to an int to
#                 pin a fixed landmark_top_k_blocks count instead.
#   PROMPT_FORMAT default "chat" (matches SFT training); use "raw" for BASE/CPT checkpoints.
#   MAX_TEST      default 50 ; CLUSTER default ai2/jupiter ; PRIORITY default urgent ; NGPU default 2.
set -uo pipefail

RUN_NAME="${1:-gatescore}"
CKPT="${CKPT:?set CKPT=<absolute weka step dir of the landmark checkpoint>}"
GATE_LOG_DIR="${GATE_LOG_DIR:?set GATE_LOG_DIR=<absolute weka dir for gate-score logs>}"

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && git rev-parse --show-toplevel)}"
LAUNCHER="$REPO/src/scripts/train/sft/singletask_ladder/run_q4b_beaker_multirung_eval.py"
CLUSTER="${CLUSTER:-ai2/jupiter}"
TASKS="${TASKS:-contra nq rerank outlier oolong}"
VARIANT="${VARIANT:-landmark}"
TOPK="${TOPK:-}"   # empty -> eval's landmark_top_k_fraction=0.1 default; set to an int to pin blocks
MAX_TEST="${MAX_TEST:-50}"
PROMPT_FORMAT="${PROMPT_FORMAT:-chat}"   # chat=SFT (matches training) | raw=BASE/CPT | alpaca=legacy
PRIORITY="${PRIORITY:-urgent}"   # urgent everywhere except holmes (see memory)
NGPU="${NGPU:-2}"
DRY_FLAG=""; [ "${DRY:-0}" = "1" ] && DRY_FLAG="--dry-run"

cd "$REPO"
export PYTHONPATH="$REPO/src"

# Pin a fixed block count only if TOPK is set; empty TOPK -> eval uses landmark_top_k_fraction=0.1.
TOPK_FLAG=""; [ -n "$TOPK" ] && TOPK_FLAG="--landmark-top-k-blocks $TOPK"

echo "=== Beaker gate-score eval | run=$RUN_NAME variant=$VARIANT topk=${TOPK:-fraction=0.1} max_test=$MAX_TEST"
echo "    ckpt=$CKPT"
echo "    gate_log_dir=$GATE_LOG_DIR"
echo "    cluster=$CLUSTER priority=$PRIORITY tasks=[$TASKS] dry=${DRY:-0} ==="
n=0
for task in $TASKS; do
  n=$((n+1))
  echo "--- [$n] $task ---"
  python "$LAUNCHER" "$RUN_NAME" "$CLUSTER" \
    --task "$task" --variant "$VARIANT" --ckpt "$CKPT" \
    --results-dir "$GATE_LOG_DIR/results" --prompt-format "$PROMPT_FORMAT" \
    --max-test "$MAX_TEST" --ngpu "$NGPU" --priority "$PRIORITY" \
    $TOPK_FLAG \
    --gate-log-dir "$GATE_LOG_DIR" --gate-log-all \
    $DRY_FLAG
done
echo "=== done: $n gate-score eval jobs (dry=${DRY:-0}) ==="
