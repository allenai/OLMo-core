#!/bin/bash
# LENGTH GENERALIZATION: score a 4B DENSE checkpoint past the 32k context it was trained on.
#
# One job per task; rungs are the 64k/128k files built by expand_ctc_rung.py, which are strict
# nested supersets of that task's own 32k rung (same 500 questions, same gold, same corpus, only
# more distractors). The 2k-32k points come from the published 4B table -- this job only adds the
# two out-of-training-range points.
#
# Both rungs sit under Qwen3.5's 262,144 native position ceiling, so these are native-RoPE
# measurements: no YaRN serving copy (run-evals rule 2 applies only from 256k up).
#
#   TASK=contradiction CKPT=/data/prasann/ctc_suite/ckpts/... sbatch lengthgen_sweep.sbatch
#SBATCH --job-name=ctcms-lengthgen
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=10:00:00
#SBATCH --output=/data/prasann/joblogs/ctcms_lengthgen_%x_%j.log

set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core

export TASK="${TASK:?set TASK}"
export EVAL_TASK_DIR="${EVAL_TASK_DIR:-$TASK}"
export CKPT="${CKPT:?set CKPT}"
export RUNGS="${RUNGS:-65536 131072}"
export EVAL_ROOT="${EVAL_ROOT:-/scratch/users/prasann/ctc_suite_staged/eval_rungs_xlong}"
export MODEL_SCALE="${MODEL_SCALE:-qwen3.5-4b}"
export ARM="${ARM:-full}"
# A single 131k-token sequence is ~4GB of KV on this hybrid; the sweep default (0.35) would leave
# almost no room to batch. This job owns its GPU, so take most of it.
export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.85}"

echo "=== length-gen HOST=$(hostname) TASK=$TASK RUNGS=$RUNGS EVAL_ROOT=$EVAL_ROOT ==="
for R in $RUNGS; do
  F="$EVAL_ROOT/$EVAL_TASK_DIR/rung_${R}.jsonl"
  [ -f "$F" ] && echo "  rung $R: $(wc -l < "$F") examples  $(du -h "$F" | cut -f1)" \
              || echo "  rung $R: MISSING $F"
done

bash "$REPO/debug/ctc_vllm_validation/sweep_task_vllm.sh"
echo "=== length-gen DONE rc=$? $(date '+%F %T') ==="
