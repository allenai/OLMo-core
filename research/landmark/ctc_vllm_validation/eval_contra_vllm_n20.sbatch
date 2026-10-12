#!/bin/bash
#SBATCH --job-name=vllm-contra-n20
#SBATCH --account=site
#SBATCH --partition=berkeleynlp
#SBATCH --qos=preemptive
#SBATCH --nodelist=horton
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/data/prasann/joblogs/vllm_contra_n20_%j.log
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" rev-parse --show-toplevel)}
REPO=$REPO_ROOT
# local_env.sh sets REPO/PATH/PYTHONPATH + the corpus-reasoning-olmo env python (as `python`)
source "$REPO/recipes/env/local_env.sh"
export PYTHONPATH="$REPO/research/common:${PYTHONPATH:-}"  # corpus_reasoning.* lives in research/common
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_HOME=/usr/local/cuda-12.8
PY=/data/prasann/ctc_vllm_venv/bin/python
VDIR=$REPO/research/landmark/ctc_vllm_validation
OUTDIR=$VDIR/n20_verify
SERVE=/data/prasann/ctc_suite/vllm_serving_4b_v3/ctc-s5-contra-full-4b
BASE_SNAP=/data/prasann/hf_models/Qwen3.5-4B-Base
# N=20 docs eval set -- the paper's easiest point, per the dense-contra investigation
# (see records/ dense-contra-eval-data-leak note). Staged to a debug path only -- does NOT
# touch S3 roster grades.
RUNG_JSONL=/scratch/users/prasann/ctc_suite_staged/eval_rungs/contradiction_eval_pubmed_both_n20_k3.jsonl
PREFILLS=$OUTDIR/prefills_contra_n20_offline.json

echo "=== HOST=$(hostname) START=$(date '+%F %T') ==="
echo "=== build prefills (corpus-reasoning-olmo python, offline, local tokenizer) ==="
python -u "$VDIR/build_prefills.py" \
  --tokenizer "$BASE_SNAP" --contra-data "$RUNG_JSONL" \
  --max-test-samples 100000 --cot-mode none --out "$PREFILLS" || { echo "PREFILLS FAILED"; exit 1; }
export PATH=$CUDA_HOME/bin:$PATH

echo "=== vLLM full-arm eval (contra n20) ==="
$PY -u "$VDIR/run_vllm_eval.py" \
  --hf-model "$SERVE" --prefills "$PREFILLS" --mode full \
  --max-new-tokens 512 --max-model-len 8192 \
  --out "$OUTDIR/responses_contra_n20_vllm.json" || { echo "VLLM EVAL FAILED"; exit 1; }

echo "=== grade ==="
$PY -u "$VDIR/grade_responses.py" \
  --responses "$OUTDIR/responses_contra_n20_vllm.json" --contra-data "$RUNG_JSONL" \
  --max-test-samples 100000 --out "$OUTDIR/grade_contra_n20_vllm.json"
echo "=== DONE $(date '+%F %T') ==="
echo "--- grade ---"; cat "$OUTDIR/grade_contra_n20_vllm.json"
