#!/bin/bash
# vLLM DENSE (full-attention) rung sweep for contradiction, reusing the PROVEN
# build_prefills.py / run_vllm_eval.py / grade_responses.py trio (validated at 0.8B,
# smoke-tested at 4B). One fresh vLLM load per rung.
#
# Usage: CKPT=/data/prasann/ctc_suite/ckpts_4b/ctc-s5-contra-full-4b \
#        RUNGS="2048 8192 32768 4096 16384" sbatch ... sweep_contra_vllm.sbatch
set -uo pipefail

REPO_ROOT=${REPO_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" rev-parse --show-toplevel)}
REPO=$REPO_ROOT
VDIR="$REPO/research/landmark/ctc_vllm_validation"

CKPT="${CKPT:?set CKPT=/data/prasann/ctc_suite/ckpts_4b/ctc-s5-contra-full-4b}"
RUNGS="${RUNGS:-2048 8192 32768 4096 16384}"
MODEL_SCALE="${MODEL_SCALE:-qwen3.5-4b}"
ARM="${ARM:-full}"
BASE_MODEL_ID="${BASE_MODEL_ID:-Qwen/Qwen3.5-4B-Base}"
MAXNEW=512

export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
source "$REPO/recipes/env/local_env.sh"
export PYTHONPATH="$REPO/research/common:${PYTHONPATH:-}"  # corpus_reasoning.* lives in research/common
unset TRANSFORMERS_CACHE
export HF_HOME=/data/prasann/hf_dl_4b
mkdir -p "$HF_HOME"
export PYTHONPATH="/scratch/users/prasann/tf_qwen35_target:$PYTHONPATH"

CKPT_NAME=$(basename "$CKPT")
HF_EXPORT="/data/prasann/ctc_suite/hf_exports_4b/$CKPT_NAME"
VLLM_PY=/scratch/users/prasann/conda/envs/corpus-reasoning-eval/bin/python

echo "=== HOST=$(hostname) TASK=contradiction CKPT=$CKPT_NAME START=$(date '+%F %T') ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

if [ ! -f "$HF_EXPORT/config.json" ]; then
  echo "=== exporting $CKPT -> HF $(date '+%F %T') ==="
  # TODO(port): depends on src/corpus_reasoning/train/export_olmo_to_hf.py, not ported
  python "$REPO/src/corpus_reasoning/train/export_olmo_to_hf.py" \
    --save-folder "$CKPT" --ckpt "$CKPT" \
    --hf-out "$HF_EXPORT" --base-model "$BASE_MODEL_ID"
else
  echo "HF export already exists: $HF_EXPORT"
fi
ls -la "$HF_EXPORT"

FAIL=0
for RUNG in $RUNGS; do
  EVAL_JSONL="/scratch/users/prasann/ctc_suite_staged/eval_rungs/contradiction/rung_${RUNG}.jsonl"
  if [ ! -f "$EVAL_JSONL" ]; then
    echo "!!! rung=$RUNG missing eval file $EVAL_JSONL -- skipping"
    continue
  fi
  echo "=== rung=$RUNG $(date '+%F %T') ==="
  MAXLEN=$((RUNG + MAXNEW + 1024))

  PREFILLS="$VDIR/prefills_contradiction_${RUNG}_4b.json"
  python "$VDIR/build_prefills.py" \
    --tokenizer "$HF_EXPORT" --contra-data "$EVAL_JSONL" \
    --max-test-samples 100000 --cot-mode none --out "$PREFILLS"
  rc=$?; if [ $rc -ne 0 ]; then echo "!!! rung=$RUNG build_prefills FAILED rc=$rc"; FAIL=1; continue; fi

  RESPONSES="$VDIR/responses_contradiction_${RUNG}_4b.json"
  $VLLM_PY "$VDIR/run_vllm_eval.py" \
    --hf-model "$HF_EXPORT" --prefills "$PREFILLS" --mode full \
    --max-new-tokens "$MAXNEW" --max-model-len "$MAXLEN" \
    --out "$RESPONSES"
  rc=$?; if [ $rc -ne 0 ]; then echo "!!! rung=$RUNG run_vllm_eval FAILED rc=$rc"; FAIL=1; continue; fi

  GRADE="$VDIR/grade_contradiction_${RUNG}_4b.json"
  python "$VDIR/grade_responses.py" \
    --responses "$RESPONSES" --contra-data "$EVAL_JSONL" \
    --max-test-samples 100000 --out "$GRADE"
  rc=$?; if [ $rc -ne 0 ]; then echo "!!! rung=$RUNG grade FAILED rc=$rc"; FAIL=1; continue; fi

  GEN_S=$(python -c "import json; print(json.load(open('$RESPONSES'))['gen_seconds'])")
  LOAD_S=$(python -c "import json; print(json.load(open('$RESPONSES'))['load_seconds'])")
  python "$VDIR/write_vllm_result.py" \
    --grade-json "$GRADE" --task contradiction --ckpt "$CKPT" --eval-jsonl "$EVAL_JSONL" \
    --model-scale "$MODEL_SCALE" --arm "$ARM" --rung-tokens "$RUNG" \
    --gen-seconds "$GEN_S" --load-seconds "$LOAD_S"
  rc=$?; if [ $rc -ne 0 ]; then echo "!!! rung=$RUNG write_result FAILED rc=$rc"; FAIL=1; fi
done
echo "=== DONE $(date '+%F %T') FAIL=$FAIL ==="
exit $FAIL
