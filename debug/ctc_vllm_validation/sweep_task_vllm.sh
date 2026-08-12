#!/bin/bash
# Generic vLLM DENSE (full-attention) rung sweep for one CTC-suite task, reusing
# build_prefills_generic.py / run_vllm_eval_generic.py / grade_responses_generic.py /
# write_vllm_result.py. One fresh vLLM load per rung (simple, proven-per-rung pattern
# from pipeline_4b.sh) -- fine for a fire-and-forget sweep since generation itself is
# fast once loaded (vLLM continuous batching).
#
# Usage: TASK=hotpotqa CKPT=/data/prasann/ctc_suite/ckpts/ctc-4b-hotpotqa-full \
#        EVAL_TASK_DIR=hotpotqa RUNGS="2048 8192 4096 16384" sbatch ... sweep_task_vllm.sbatch
set -uo pipefail

REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
VDIR="$REPO/debug/ctc_vllm_validation"

TASK="${TASK:?set TASK=hotpotqa|oolong|outlier|... (a TASK_CFG/TASK_ALIASES key)}"
CKPT="${CKPT:?set CKPT=/data/prasann/ctc_suite/ckpts/<ckpt-dir>}"
EVAL_TASK_DIR="${EVAL_TASK_DIR:-$TASK}"
# Root holding <task>/rung_<tokens>.jsonl. Defaults to the standard 2k-32k ladder; the
# length-generalization study points it at `eval_rungs_xlong`, whose rungs are built by
# debug/ctc_modelscale/expand_ctc_rung.py and are named by MEASURED tokens.
# ⚠ Do NOT "simplify" this by copying xlong rungs into eval_rungs/: contradiction already has a
# rung_131072.jsonl there built from a contaminated filler pool (47.1 tok/doc of FEVER/wiki
# against 15.6 for its own ladder), and a copy would silently overwrite or shadow it.
EVAL_ROOT="${EVAL_ROOT:-/scratch/users/prasann/ctc_suite_staged/eval_rungs}"
RUNGS="${RUNGS:-2048 8192 32768 4096 16384}"
MODEL_SCALE="${MODEL_SCALE:-qwen3.5-4b}"
ARM="${ARM:-full}"
BASE_MODEL_ID="${BASE_MODEL_ID:-Qwen/Qwen3.5-4B-Base}"

# Allow network egress for export_olmo_to_hf.py's one-time hybrid-config lookup (same
# escape hatch pipeline_4b.sh uses) -- override BEFORE sourcing local_env.sh.
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
source "$REPO/src/scripts/local_env.sh"
unset TRANSFORMERS_CACHE
export HF_HOME=/data/prasann/hf_dl_4b
mkdir -p "$HF_HOME"
# Shadow transformers build with real qwen3_5 support for the export step only (the
# corpus-reasoning-olmo env's transformers predates it) -- same as convert_both.sh/pipeline_4b.sh.
export PYTHONPATH="/scratch/users/prasann/tf_qwen35_target:$PYTHONPATH"

CKPT_NAME=$(basename "$CKPT")
HF_EXPORT="/data/prasann/ctc_suite/hf_exports_4b/$CKPT_NAME"
VLLM_PY=/scratch/users/prasann/conda/envs/corpus-reasoning-eval/bin/python

echo "=== HOST=$(hostname) TASK=$TASK CKPT=$CKPT_NAME START=$(date '+%F %T') ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

if [ ! -f "$HF_EXPORT/config.json" ]; then
  echo "=== exporting $CKPT -> HF $(date '+%F %T') ==="
  python "$REPO/src/corpus_reasoning/train/export_olmo_to_hf.py" \
    --save-folder "$CKPT" --ckpt "$CKPT" \
    --hf-out "$HF_EXPORT" --base-model "$BASE_MODEL_ID"
else
  echo "HF export already exists: $HF_EXPORT"
fi
ls -la "$HF_EXPORT"

FAIL=0
for RUNG in $RUNGS; do
  EVAL_JSONL="$EVAL_ROOT/$EVAL_TASK_DIR/rung_${RUNG}.jsonl"
  if [ ! -f "$EVAL_JSONL" ]; then
    echo "!!! rung=$RUNG missing eval file $EVAL_JSONL -- skipping"
    continue
  fi
  echo "=== rung=$RUNG $(date '+%F %T') ==="
  MAXNEW=$(python -c "
from corpus_reasoning.eval.eval_lc_native_docchunk import TASK_ALIASES, TASK_CFG
t = TASK_ALIASES.get('$TASK', '$TASK')
print(TASK_CFG[t]['max_new'])
")
  MAXLEN=$((RUNG + MAXNEW + 1024))

  PREFILLS="$VDIR/prefills_${TASK}_${RUNG}_4b.json"
  python "$VDIR/build_prefills_generic.py" \
    --tokenizer "$HF_EXPORT" --task "$TASK" --eval-data "$EVAL_JSONL" \
    --max-test-samples 100000 --out "$PREFILLS"
  rc=$?; if [ $rc -ne 0 ]; then echo "!!! rung=$RUNG build_prefills FAILED rc=$rc"; FAIL=1; continue; fi

  RESPONSES="$VDIR/responses_${TASK}_${RUNG}_4b.json"
  $VLLM_PY "$VDIR/run_vllm_eval_generic.py" \
    --hf-model "$HF_EXPORT" --prefills "$PREFILLS" --mode full \
    --max-new-tokens "$MAXNEW" --max-model-len "$MAXLEN" \
    --out "$RESPONSES"
  rc=$?; if [ $rc -ne 0 ]; then echo "!!! rung=$RUNG run_vllm_eval FAILED rc=$rc"; FAIL=1; continue; fi

  GRADE="$VDIR/grade_${TASK}_${RUNG}_4b.json"
  python "$VDIR/grade_responses_generic.py" \
    --responses "$RESPONSES" --eval-data "$EVAL_JSONL" --task "$TASK" \
    --max-test-samples 100000 --out "$GRADE"
  rc=$?; if [ $rc -ne 0 ]; then echo "!!! rung=$RUNG grade FAILED rc=$rc"; FAIL=1; continue; fi

  GEN_S=$(python -c "import json; print(json.load(open('$RESPONSES'))['gen_seconds'])")
  LOAD_S=$(python -c "import json; print(json.load(open('$RESPONSES'))['load_seconds'])")
  python "$VDIR/write_vllm_result.py" \
    --grade-json "$GRADE" --task "$TASK" --ckpt "$CKPT" --eval-jsonl "$EVAL_JSONL" \
    --model-scale "$MODEL_SCALE" --arm "$ARM" --rung-tokens "$RUNG" \
    --gen-seconds "$GEN_S" --load-seconds "$LOAD_S"
  rc=$?; if [ $rc -ne 0 ]; then echo "!!! rung=$RUNG write_result FAILED rc=$rc"; FAIL=1; fi
done
echo "=== DONE $(date '+%F %T') FAIL=$FAIL ==="
exit $FAIL
