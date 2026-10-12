#!/bin/bash
# GPU-only step: general driver (build_prefills_any -> run_vllm_eval --task -> grade_any) on
# the ALREADY-PREPARED outlier serving copy. Prefill build is CPU/fast so it's kept in this job
# (it needs the corpus-reasoning-olmo env, not the vllm venv) but it's short; the GPU-heavy part
# is the vLLM load (first-ever load on a fresh node does a ~5-13min GDN JIT compile) + generate.
#SBATCH --job-name=gate-outlier-2048-gpu-mooney
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=mooney
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/data/prasann/joblogs/gate_outlier_2048_gpu_mooney_%j.log
set -uo pipefail

REPO_ROOT=${REPO_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" rev-parse --show-toplevel)}
REPO=$REPO_ROOT
VDIR="$REPO/research/landmark/ctc_vllm_validation"
# TODO(port): depends on debug/ctc_vllm_validation/general/ (build_prefills_any.py, grade_any.py), not ported
GDIR="$VDIR/general"

TASK=outlier
SERVE=/data/prasann/ctc_suite/vllm_serving_4b/ctc-4b-outlier-full
BASE_SNAP=/data/prasann/hf_models/Qwen3.5-4B-Base
RUNG_JSONL=/scratch/users/prasann/ctc_suite_staged/eval_rungs/outlier/rung_2048.jsonl
VLLM_PY=/data/prasann/ctc_vllm_venv/bin/python

echo "=== HOST=$(hostname) START=$(date '+%F %T') TASK=$TASK ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

source "$REPO/recipes/env/local_env.sh"
export PYTHONPATH="$REPO/research/common:${PYTHONPATH:-}"  # corpus_reasoning.* lives in research/common

if [ ! -f "$SERVE/model.safetensors" ]; then
  echo "!!! serving copy not found at $SERVE -- stage_export job must complete first"; exit 1
fi

# ---- 3. build prefills (corpus-reasoning-olmo env python, CPU-only) ----
PREFILLS=$VDIR/prefills_outlier_2048_4b_mooney.json
echo "=== building prefills $(date '+%F %T') ==="
python "$GDIR/build_prefills_any.py" \
  --tokenizer "$BASE_SNAP" --task "$TASK" --eval-data "$RUNG_JSONL" \
  --max-test-samples 100000 --out "$PREFILLS"
rc=$?
if [ $rc -ne 0 ]; then echo "!!! build_prefills_any FAILED rc=$rc"; exit 1; fi

# ---- 4. run vLLM (venv python, GPU) ----
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_HOME=/usr/local/cuda-12.8
RESP=$VDIR/responses_outlier_2048_vllm_mooney.json
echo "=== vLLM full-attention eval $(date '+%F %T') ==="
PATH=$CUDA_HOME/bin:$PATH $VLLM_PY -u "$VDIR/run_vllm_eval.py" \
  --hf-model "$SERVE" --prefills "$PREFILLS" --mode full --task "$TASK" \
  --max-new-tokens 256 --max-model-len 4096 --gpu-mem-util 0.85 \
  --out "$RESP"
rc=$?
if [ $rc -ne 0 ]; then echo "!!! run_vllm_eval FAILED rc=$rc"; exit 1; fi

# ---- 5. grade ----
GRADE=$VDIR/grade_outlier_2048_vllm_mooney.json
echo "=== grading $(date '+%F %T') ==="
python "$GDIR/grade_any.py" \
  --responses "$RESP" --eval-data "$RUNG_JSONL" --task "$TASK" \
  --max-test-samples 100000 --out "$GRADE"
rc=$?
if [ $rc -ne 0 ]; then echo "!!! grade_any FAILED rc=$rc"; exit 1; fi

echo "=== DONE $(date '+%F %T') ==="
echo "--- $GRADE ---"
cat "$GRADE"
