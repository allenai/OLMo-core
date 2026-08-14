#!/bin/bash
# GPU step: general driver (build_prefills_any -> run_vllm_eval --task -> grade_any) over ALL
# requested rungs for one task, fully node-local (code, env, venv, data, checkpoints, serving
# copy all under /data/prasann -- zero /scratch or /accounts touches). Writes small grade JSON
# results to node-local /data/prasann/ctc_suite_vllm_results/<task>/, then does ONE final rsync
# of just those small json files back to the shared repo's results/ctc_suite/ tree (the
# deliverable location -- not a repeated hot-path touch).
#
# Usage: TASK=outlier RUNGS="2048 4096 8192 16384 32768" bash gpu_eval_task.sh
#
# TASK names the checkpoint/rung-dir/serving-copy (matches the S3 checkpoint's logical name).
# EVAL_TASK is the value passed to --task on the driver scripts -- these are NOT always the
# same string: several of our checkpoint/rung-dir names are NOT valid run_rung_eval.py --task
# values (they're data-source labels, not catalog keys/aliases). Defaults to TASK if unset.
# Known corrections (discovered 2026-07-21 when 5/18 fan-out tasks failed fast with
# "not in [...]" from build_prefills_any.py):
#   niah              -> niah_contradiction   (alias -> retrieval)
#   scifact           -> beir_scifact         (alias -> retrieval)
#   outlier_amzn      -> outlier_amazon       (alias -> outlier)
#   absence_gutenberg -> absence              (canonical)
#   qdmatch_hpqa      -> qdmatch              (canonical; the HPQA variant comes from --eval-data)
set -uo pipefail

TASK="${TASK:?set TASK=outlier|niah|...}"
EVAL_TASK="${EVAL_TASK:-$TASK}"
# RUNG_TASK names the eval_rungs/<dir> to read from -- NOT always == TASK. Discovered
# 2026-07-21: grouping_labeled has no eval_rungs/grouping_labeled/ dir of its own -- it shares
# eval_rungs/grouping/. Defaults to TASK.
RUNG_TASK="${RUNG_TASK:-$TASK}"
RUNGS="${RUNGS:-2048 4096 8192 16384 32768}"
KEEP_RESPONSES="${KEEP_RESPONSES:-0}"

LOCAL_REPO=/data/prasann/repo/OLMo-core
# SERVE / BASE_SNAP / OUTDIR are overridable so the SAME driver can eval a non-4B checkpoint (the
# model-scale study's 0.8B and 2B runs) without a forked copy. Unset, they resolve to exactly the
# historical 4B paths, so every existing caller is bit-unchanged.
# BASE_SNAP must match the checkpoint's SCALE: it supplies the tokenizer for prompt building, and
# more importantly the per-scale `vision_config` the serving copy is built from (0.8B is 768-wide /
# 12-deep; 2B and 4B are 1024/24). Pointing a 0.8B eval at the 4B snapshot builds the wrong vision
# tower and the load fails in vLLM's weight tracker.
SERVE="${SERVE:-/data/prasann/ctc_suite/vllm_serving_4b/ctc-4b-${TASK}-full}"
BASE_SNAP="${BASE_SNAP:-/data/prasann/hf_models/Qwen3.5-4B-Base}"
# RUNGDIR is overridable so the length-generalization study can point at the separately-built
# `eval_rungs_lengthgen` tree (64k/128k rungs calibrated on REAL prefill length) without copying
# files into the main ladder, where a mis-sized rung would silently contaminate the 2k-32k results.
RUNGDIR="${RUNGDIR:-/data/prasann/ctc_suite_staged/eval_rungs/${RUNG_TASK}}"
VLLM_PY=/data/prasann/ctc_vllm_venv/bin/python
OUTDIR="${OUTDIR:-/data/prasann/ctc_suite_vllm_results/${TASK}}"
mkdir -p "$OUTDIR"

# CRITICAL (root-caused 2026-07-21, comprehensive fix): HOME/FLASHINFER_CACHE_DIR alone were
# NOT enough -- the job's cwd defaulted to /accounts/.../OLMo-core (sbatch inherits the
# submission dir when --chdir is unset -- see gpu.sbatch's --chdir), and CONDA_ENVS_PATH leaked
# in from the ambient system profile pointing at /scratch. JIT-compile subprocesses
# (humming-kernels' nvrtc build, triton's cc1 launcher compile) inherited that cwd and wedged in
# nfs_wait_bit_killable under load. This self-healing preamble forces EVERY cache/tmp/config
# path a python/pip/nvcc/triton/flashinfer process could touch onto /data -- materializes once
# per node (no NFS distribution needed) and is safe to re-source.
PREAMBLE=/data/prasann/node_local_env.sh
if [ ! -f "$PREAMBLE" ]; then
  cat > "$PREAMBLE" <<'ENVEOF'
export HOME=/data/prasann/home
export TMPDIR=/data/prasann/tmp
export XDG_CACHE_HOME=/data/prasann/xdg_cache
export XDG_CONFIG_HOME=/data/prasann/xdg_config
export FLASHINFER_CACHE_DIR=/data/prasann/flashinfer_cache
export CUDA_CACHE_PATH=/data/prasann/cuda_cache
export TRITON_CACHE_DIR=/data/prasann/triton_cache
export TORCHINDUCTOR_CACHE_DIR=/data/prasann/inductor_cache
export HF_HOME=/data/prasann/hf
export TRANSFORMERS_CACHE=/data/prasann/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PIP_CACHE_DIR=/data/prasann/pip_cache
export PYTHONPATH=/data/prasann/repo/OLMo-core/src
export CUDA_HOME=/usr/local/cuda-12.8
# /system/linux/aws-cli/... is the compute-node aws CLI (login-node /usr/local/linux/bin
# is not on compute nodes); without it the grade-push `aws s3 cp` below silently no-ops
# (caught by its own `|| echo non-fatal`) and the S3 roster goes stale unnoticed.
export PATH="$CUDA_HOME/bin:/data/prasann/conda/envs/corpus-reasoning-olmo/bin:/data/prasann/ctc_vllm_venv/bin:/system/linux/aws-cli/v2/2.1.5/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS=ignore
export CONDA_ENVS_PATH=/data/prasann/conda/envs
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_PROMPT_MODIFIER CONDA_PKGS_DIRS 2>/dev/null
ENVEOF
fi
source "$PREAMBLE"
mkdir -p "$HOME" "$HOME/.cache" "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" \
  "$FLASHINFER_CACHE_DIR" "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
  "$HF_HOME" "$PIP_CACHE_DIR"
# AWS creds: moving HOME to /data broke the ~/.aws lookup (AWS CLI resolves it from
# $HOME, and the real creds live under the NFS home). Copy once, self-healing.
if [ ! -f "$HOME/.aws/credentials" ]; then
  mkdir -p "$HOME/.aws"
  cp /accounts/projects/berkeleynlp/prasann/.aws/config /accounts/projects/berkeleynlp/prasann/.aws/credentials "$HOME/.aws/" 2>/dev/null
  chmod 600 "$HOME/.aws/config" "$HOME/.aws/credentials" 2>/dev/null
fi

echo "=== [gpu_eval:$TASK] HOST=$(hostname) START=$(date '+%F %T') RUNGS=$RUNGS TP=${TP:-1} ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

if [ ! -f "$SERVE/model.safetensors" ]; then
  echo "!!! [gpu_eval:$TASK] serving copy not found at $SERVE -- prep_task.sh must run first"; exit 1
fi

FAIL=0
for RUNG in $RUNGS; do
  RUNG_JSONL="$RUNGDIR/rung_${RUNG}.jsonl"
  if [ ! -f "$RUNG_JSONL" ]; then
    echo "!!! [gpu_eval:$TASK] rung=$RUNG missing $RUNG_JSONL -- skipping"; FAIL=1; continue
  fi
  echo ""
  echo "############ [gpu_eval:$TASK] RUNG=$RUNG $(date '+%F %T') ############"

  PREFILLS="$OUTDIR/prefills_${RUNG}.json"
  # QUERY_POSITION must match the training shard's metadata.json. Unset -> build_prefills_any's
  # "both" default, which is what every pre-2026-08 checkpoint was trained with; new runs train
  # "after" and are scored garbage under "both" (the contradiction 0.559-vs-0.946 failure).
  python "$LOCAL_REPO/debug/ctc_vllm_validation/general/build_prefills_any.py" \
    --tokenizer "$BASE_SNAP" --task "$EVAL_TASK" --eval-data "$RUNG_JSONL" \
    ${COT_MODE:+--cot-mode "$COT_MODE"} \
    ${QUERY_POSITION:+--query-position "$QUERY_POSITION"} \
    --max-test-samples 100000 --out "$PREFILLS" \
    || { echo "!!! [gpu_eval:$TASK] rung=$RUNG build_prefills FAILED"; FAIL=1; continue; }
  # Verify the flag TOOK EFFECT rather than trusting that it was passed -- build_prefills_any
  # records the value it actually rendered with, so this compares intent against the artifact.
  ACTUAL_QP=$(python -c "import json;print(json.load(open('$PREFILLS')).get('query_position'))")
  echo "[gpu_eval:$TASK] rung=$RUNG rendered query_position=$ACTUAL_QP (requested='${QUERY_POSITION:-<default both>}')"
  if [ -n "${QUERY_POSITION:-}" ] && [ "$ACTUAL_QP" != "$QUERY_POSITION" ]; then
    echo "!!! [gpu_eval:$TASK] rung=$RUNG query_position MISMATCH: wanted '$QUERY_POSITION' got '$ACTUAL_QP' -- stale node repo?"
    FAIL=1; continue
  fi

  RESP="$OUTDIR/responses_${RUNG}.json"
  # TP shards the KV cache across GPUs. Default 1 leaves every historical caller bit-unchanged.
  # The 2k-32k ladder fits on one H200 with room to spare; the length-gen rungs do not have that
  # margin -- this hybrid keeps KV only on its 8 full-attention layers (4 kv heads x 256 head_dim
  # x 2 x 2 bytes x 8 = 32 KiB/token), so a 133k-token prompt is ~4.4 GiB of KV *per sequence*
  # and vLLM sizes its concurrency from whatever pool is left after weights (~20 GiB here).
  # ⚠ TP MUST DIVIDE ALL FOUR HEAD COUNTS, not just num_attention_heads: this config carries
  # num_attention_heads=16, num_key_value_heads=4, linear_num_key_heads=16,
  # linear_num_value_heads=32 (the GDN layers), plus the grafted vision tower's num_heads=16.
  # num_key_value_heads=4 is the binding one -- TP=8 would not divide it.
  # Ask SLURM for the matching --gres; TP>1 with fewer visible GPUs dies at engine init.
  PATH=$CUDA_HOME/bin:$PATH $VLLM_PY -u "$LOCAL_REPO/debug/ctc_vllm_validation/run_vllm_eval.py" \
    --hf-model "$SERVE" --prefills "$PREFILLS" --mode full --task "$EVAL_TASK" \
    --max-new-tokens "${MAX_NEW_TOKENS:-256}" --max-model-len 4096 --gpu-mem-util "${GPU_MEM_UTIL:-0.85}" \
    --tensor-parallel-size "${TP:-1}" \
    --out "$RESP" \
    || { echo "!!! [gpu_eval:$TASK] rung=$RUNG run_vllm_eval FAILED"; FAIL=1; continue; }

  GRADE="$OUTDIR/grade_${RUNG}.json"
  python "$LOCAL_REPO/debug/ctc_vllm_validation/general/grade_any.py" \
    --responses "$RESP" --eval-data "$RUNG_JSONL" --task "$EVAL_TASK" \
    --max-test-samples 100000 --out "$GRADE" \
    || { echo "!!! [gpu_eval:$TASK] rung=$RUNG grade FAILED"; FAIL=1; continue; }

  echo "--- [gpu_eval:$TASK] rung=$RUNG grade ---"; cat "$GRADE"

  # push this rung's small grade json to S3 immediately (not a hot-path touch -- one small
  # object per rung, and gives the coordinator a live view without waiting for the sweep to finish)
  AWS_PROFILE=S3 aws s3 cp "$GRADE" \
    "s3://ai2-llm/checkpoints/prasanns/_transfer/ctc_dense_results/${TASK}/rung_${RUNG}.json" \
    --only-show-errors || echo "!!! [gpu_eval:$TASK] rung=$RUNG S3 result push FAILED (non-fatal)"

  # append to the node-local per-task status fragment (one file per task -> no cross-job races)
  METRIC_NAME=$(python -c "import json; d=json.load(open('$GRADE')); print(d.get('metric_name'))")
  METRIC_VALUE=$(python -c "import json; d=json.load(open('$GRADE')); print(d.get('metric_value'))")
  PARSE_RATE=$(python -c "import json; d=json.load(open('$GRADE')); print(d.get('parse_rate'))")
  echo "| $TASK | $RUNG | $METRIC_NAME | $METRIC_VALUE | $PARSE_RATE | $(hostname) | $(date '+%F %T') |" \
    >> "$OUTDIR/status_fragment.md"

  # free the (large) prefill/response artifacts once graded -- keep node-local disk light,
  # unless KEEP_RESPONSES=1 (diagnostic runs that need the raw generations inspected).
  if [ "$KEEP_RESPONSES" != "1" ]; then
    rm -f "$PREFILLS" "$RESP"
  fi
done

echo ""
echo "=== [gpu_eval:$TASK] SWEEP DONE $(date '+%F %T') FAIL=$FAIL ==="

# ---- final, one-time, SMALL write-back to the shared repo (deliverable location, not hot path).
#
# ⚠ THE DESTINATION MUST INCLUDE MODEL IDENTITY, NOT JUST $TASK. It used to be
# `$ACCOUNTS_RESULTS/${TASK}/`, keyed on the task alone, so EVERY checkpoint evaluated on a task
# overwrote the previous one in place. On 2026-08-12 the 2B model-scale run silently destroyed the
# 4B suite's contradiction ladder that way: the grade files kept the suite's directory name while
# carrying the 2B model's numbers, which is worse than losing them outright -- the result looked
# like a 4B reference and was not one. The 4B dense contradiction rungs had to be re-run.
#
# The serving-copy basename is the model identity the grader already records in `hf_model`, so
# deriving the subdirectory from it keeps the on-disk layout consistent with the file contents.
# The old flat per-task path is left readable but is never written again.
ACCOUNTS_RESULTS=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/ctc_vllm_validation/node_local_results
MODEL_ID=$(basename "${SERVE:-unknown-model}")
DEST="$ACCOUNTS_RESULTS/${TASK}__${MODEL_ID}"
mkdir -p "$DEST"
# Refuse to clobber a DIFFERENT model's grades even under the new scheme (e.g. if two runs share a
# serving-copy name). Cheap check, and the failure it prevents is silent and unrecoverable.
for g in "$OUTDIR"/grade_*.json; do
  [ -f "$g" ] || continue
  EXIST="$DEST/$(basename "$g")"
  if [ -f "$EXIST" ]; then
    PREV=$(python3 -c "import json;print(json.load(open('$EXIST')).get('hf_model',''))" 2>/dev/null)
    NEW=$(python3 -c "import json;print(json.load(open('$g')).get('hf_model',''))" 2>/dev/null)
    if [ -n "$PREV" ] && [ "$PREV" != "$NEW" ]; then
      echo "[gpu_eval:$TASK] REFUSING to overwrite $(basename "$g"): existing is from $PREV, new is from $NEW"
      continue
    fi
  fi
  cp "$g" "$DEST/"
done
if [ "$KEEP_RESPONSES" = "1" ]; then
  cp "$OUTDIR"/responses_*.json "$DEST/" 2>/dev/null
fi
[ -f "$OUTDIR/status_fragment.md" ] && cp "$OUTDIR/status_fragment.md" "$ACCOUNTS_RESULTS/roster_status_${TASK}.md"
echo "[gpu_eval:$TASK] copied grade jsons + status fragment -> $ACCOUNTS_RESULTS"
exit $FAIL
