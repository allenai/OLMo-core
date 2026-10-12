#!/bin/bash
#SBATCH --job-name=vllm-contra-ladder
#SBATCH --account=site
#SBATCH --partition=berkeleynlp
#SBATCH --qos=preemptive
#SBATCH --nodelist=horton
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/data/prasann/joblogs/vllm_contra_ladder_%j.log
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" rev-parse --show-toplevel)}
REPO=$REPO_ROOT
source "$REPO/recipes/env/local_env.sh"
export PYTHONPATH="$REPO/research/common:${PYTHONPATH:-}"  # corpus_reasoning.* lives in research/common
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_HOME=/usr/local/cuda-12.8
PY=/data/prasann/ctc_vllm_venv/bin/python
VDIR=$REPO/research/landmark/ctc_vllm_validation
SERVE=/data/prasann/ctc_suite/vllm_serving_4b_v3/ctc-s5-contra-full-4b
BASE_SNAP=/data/prasann/hf_models/Qwen3.5-4B-Base
STAGE=/scratch/users/prasann/ctc_suite_staged/eval_rungs/contradiction
RUNGS="${RUNGS:-4096 8192 16384 32768}"

echo "=== HOST=$(hostname) START=$(date '+%F %T') RUNGS=$RUNGS ==="
for RUNG in $RUNGS; do
  echo ""
  echo "############ RUNG=$RUNG $(date '+%F %T') ############"
  PREFILLS=$VDIR/prefills_contra${RUNG}_offline.json
  RESP=$VDIR/responses_contra${RUNG}_vllm.json
  GRADE=$VDIR/grade_contra${RUNG}_vllm.json
  python -u "$VDIR/build_prefills.py" \
    --tokenizer "$BASE_SNAP" --contra-data "$STAGE/rung_${RUNG}.jsonl" \
    --max-test-samples 100000 --cot-mode none --out "$PREFILLS" \
    || { echo "!!! rung=$RUNG PREFILLS FAILED"; continue; }

  # run_vllm_eval auto-sizes max_model_len from the ACTUAL longest prefill (nominal rung
  # underestimates it badly); pass a small floor + high gpu-mem-util for long-context KV.
  PATH=$CUDA_HOME/bin:$PATH $PY -u "$VDIR/run_vllm_eval.py" \
    --hf-model "$SERVE" --prefills "$PREFILLS" --mode full \
    --max-new-tokens 512 --max-model-len 8192 --gpu-mem-util 0.85 \
    --out "$RESP" \
    || { echo "!!! rung=$RUNG VLLM EVAL FAILED"; continue; }

  $PY -u "$VDIR/grade_responses.py" \
    --responses "$RESP" --contra-data "$STAGE/rung_${RUNG}.jsonl" \
    --max-test-samples 100000 --out "$GRADE"
  echo "--- rung=$RUNG grade ---"; cat "$GRADE"
done
echo ""
echo "=== LADDER DONE $(date '+%F %T') ==="
echo "=== SUMMARY ==="
for RUNG in 2048 $RUNGS; do
  g=$VDIR/grade_contra${RUNG}_vllm.json
  [ -f "$g" ] && echo -n "rung_$RUNG: " && $PY -c "import json,sys; d=json.load(open('$g'))['contradiction']; print(f\"f1={d['f1']:.3f} p={d['precision']:.3f} r={d['recall']:.3f} em={d['exact_match']:.3f} parse={d['parse_rate']:.3f}\")" 2>/dev/null
done
