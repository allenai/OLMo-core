#!/usr/bin/env bash
# One gantry job per (checkpoint, rung) running eval_pipeline_cu129_apt.sh on Beaker.
#
# Every CTC eval launched from this host has so far been an ad-hoc inline `gantry run ... -- bash
# -c` typed at the prompt. That is how five separate wrong-but-confident results happened in one
# night: a chunked checkpoint scored with MODE=full, a checkpoint scored against a ladder its
# training shard never matched, EVAL_TASK left at the logical task name when the catalog key
# differs (fiqa -> beir_fiqa, qdmatch_fiqa -> qdmatch), --weka omitted so the job silently fell
# back to a nonexistent S3 path, and three artifacts pushed to one S3 key so a later 100-example
# probe overwrote a finished 500-example run. Each of those is a flag on the command line, so each
# of them is preventable by having exactly one command line. This is it.
#
#   CKPT=<weka-ckpt-dirname> MODE=chunked TASK=fiqa EVAL_TASK=beir_fiqa \
#   RESULT_DIR=ctcms-fiqa-cmix-2b RUNGS="2048 4096" bash debug/ctc_vllm_validation/beaker/launch_eval.sh
#
# ── THE FOUR THINGS THAT MUST AGREE, AND WHY EACH IS CHECKED HERE ──────────────────────────────
#   MODE        must match the mask the checkpoint TRAINED with (dense arm -> full, chunked-mix
#               and pure-chunked arms -> chunked). The baseline any of these gets compared against
#               was itself graded with its own matching mode, so a mismatch is not a comparison.
#               Required explicitly -- no default, because the safe-looking default (full) is the
#               one that silently produced zeros.
#   EVAL_TASK   the run_vllm_eval/grade_any CATALOG KEY, which is not always TASK. Defaults to
#               TASK, and the known aliases are asserted below rather than left to memory.
#   LADDER      TASK also selects the rung tree, so the eval ladder is the training shard's own
#               ladder by construction -- pass RUNG_TASK to override when they legitimately differ
#               (contradiction trains on contradiction_train and evals on contradiction_iid).
#   RESULT_DIR  the S3 prefix. Per-run, never shared: results carry no arm/scale field, so two runs
#               sharing a prefix silently overwrite each other rung by rung.
set -uo pipefail
cd /accounts/projects/berkeleynlp/prasann/projects/OLMo-core
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$HOME/.local/bin:$PATH

CKPT="${CKPT:?exact weka checkpoint dirname under ctc_suite/ckpts/ -- READ IT FROM THE LAUNCH LOG, never reconstruct the timestamp}"
TASK="${TASK:?logical task name (selects the rung tree)}"
MODE="${MODE:?full for a dense-trained arm, chunked for chunked-mix/pure-chunked}"
RESULT_DIR="${RESULT_DIR:?S3 prefix under _transfer/, unique per run}"
RUNG_TASK="${RUNG_TASK:-$TASK}"
# EXPORTER picks the checkpoint->HF path inside the pipeline. Explicit, never sniffed.
#   qwen  (default): Qwen3/Qwen3.5 checkpoints via export_olmo_to_hf.py -- unchanged.
#   noswa: OLMo-3 sliding-window-FREE (--model-scale 7b-noswa) checkpoints via
#          export_noswa_to_hf.py, mapped onto the Olmo2 HF class. Use for ctc-olmo3ns* /
#          ctc-olmo3nsc* DENSE arms only.
# ⚠ There is NO vLLM path for the GDN/hybrid arms (ctc-olmohyb-*): olmo-core exports those through
# save_hf_hybrid_model and vLLM has no loader for the result. Those stay on the native evaluator
# (debug/ctc_crossfamily/eval_olmo_beaker.sh) -- that is a real limitation, not an oversight.
EXPORTER="${EXPORTER:-qwen}"
case "$EXPORTER" in qwen|noswa) ;; *) echo "FATAL: EXPORTER must be qwen|noswa, got '$EXPORTER'"; exit 2 ;; esac
if [ "$EXPORTER" = "noswa" ]; then
  case "$CKPT" in
    *olmohyb*) echo "FATAL: '$CKPT' looks like a GDN/hybrid checkpoint; there is no vLLM path for it. Use the native evaluator."; exit 2 ;;
  esac
  # The SWA-keeping arm (--model-scale 7b, run names ctc-olmo3-7b-*) exports to Olmo3Config, not
  # Olmo2Config, and has never been served here. Only the noswa arms are covered.
  case "$CKPT" in
    ctc-olmo3ns*) ;;
    *) echo "WARNING: '$CKPT' is not a ctc-olmo3ns* name -- EXPORTER=noswa assumes a sliding-window-FREE olmo3 checkpoint. The pipeline asserts model_type==olmo2 after export and will fail loudly if it is not."; ;;
  esac
fi
RUNGS="${RUNGS:-2048 4096 8192 16384 32768}"
RUNG_TREE="${RUNG_TREE:-ctc_eval_rungs}"
# Rungs packed into ONE gantry job. Jupiter allocation turnover is high enough that queueing time
# dominates a short eval, so N rungs in one job = one wait instead of N. The pipeline is re-entrant
# per rung (its WORK dir is keyed on TASK+RUNG), so packing only repeats the per-rung setup -- it
# changes scheduling, never a number. Every rung still gets its own ledger row and its own S3 key.
# A rung that fails does NOT abort its jobmates; the job exits non-zero if any rung failed.
RUNGS_PER_JOB="${RUNGS_PER_JOB:-2}"
CODE_TARBALL="${CODE_TARBALL:-ctc_eval_code_2026-08-13.tar.gz}"
# ⚠ CHUNKED MODE IS INCOMPATIBLE WITH TP>1 -- AND FAILS SILENTLY, NOT LOUDLY.
# run_vllm_eval installs the document-chunk mask as an IN-PROCESS monkey-patch and relies on
# VLLM_ENABLE_V1_MULTIPROCESSING=0 to keep the model in the driver process. TP>1 forces vLLM to
# spawn Worker_TP0/Worker_TP1 subprocesses anyway, so the patch never reaches the model: the run
# completes, reports mode=chunked, and is UNMASKED. Measured: fiqa cmix at 2048 returned
# gold_id_f1=0.9165619047619048 under MODE=chunked TP=2 -- bit-identical to the same checkpoint
# scored dense, with patch_debug calls=0.
# So chunked defaults to TP=1, and an explicit TP>1 with MODE=chunked is refused below.
TP="${TP:-$([ "${MODE:-}" = chunked ] && echo 1 || echo 2)}"
# ⚠ DEEP RUNGS MUST BE PINNED TO H100s. The default cluster list includes ai2/neptune, whose
# nodes are L40S with 46GB -- not the 80GB an H100 gives. A 128k rung that lands there dies
# mid-generation while the 64k rung on the SAME node succeeds, and the packed job can still
# report exit 0, so the missing rung is SILENT (measured: fiqa rung_131755, job
# 01M0FTZT5W0AX5ME8BZ1X526BH, "NVIDIA L40S, 46068 MiB").
# For any rung >= 64k pass:  CLUSTERS="--cluster ai2/jupiter"
PRIORITY="${PRIORITY:-urgent}"
LOGD="${LOGD:-debug/ctc_vllm_validation/beaker/launches}"
NAME_PREFIX="${NAME_PREFIX:-ev}"
REF="${REF:-$(git rev-parse HEAD)}"
# ⚠ PIN THE IMAGE. A hand-rolled `gantry run` without --beaker-image builds a uv venv from the
# repo's pyproject, whose torch refuses the cluster driver ("NVIDIA driver too old, found version
# 12080"). --no-python keeps that venv from shadowing the image's python; the pipeline builds its
# own vLLM venv on top either way.
IMAGE="${IMAGE:-tylerr/olmo-core-tch291cu128-2025-11-25}"

# The HF base the pipeline rebuilds the model from. It MUST match the checkpoint's model scale --
# export_olmo_to_hf loads the distcp into a model built from this id, so a mismatch is a tensor
# shape mismatch. Derived from MODEL_SCALE so callers name the scale, not a HF repo path.
declare -A BASE_FOR_SCALE=(
  [0.8b]=Qwen/Qwen3.5-0.8B-Base [2b]=Qwen/Qwen3.5-2B-Base [4b]=Qwen/Qwen3.5-4B-Base [9b]=Qwen/Qwen3.5-9B-Base
)
MODEL_SCALE="${MODEL_SCALE:-4b}"
BASE_MODEL_ID="${BASE_MODEL_ID:-${BASE_FOR_SCALE[$MODEL_SCALE]:-}}"
# EXPORTER=noswa rebuilds the model from the checkpoint's OWN config.json (olmo-core
# TransformerConfig), so there is no HF base to name and MODEL_SCALE=7b needs no table entry.
if [ "$EXPORTER" != "noswa" ]; then
  [ -n "$BASE_MODEL_ID" ] || { echo "FATAL: no HF base for MODEL_SCALE=$MODEL_SCALE; pass BASE_MODEL_ID"; exit 2; }
fi

# ⚠ GANTRY SHIPS THE PUSHED COMMIT, NOT THE WORKING TREE. --allow-dirty silences the check but
# does not upload anything, so a local-only fix to the pipeline runs as whatever is on the remote.
if ! git diff --quiet HEAD -- debug/ctc_vllm_validation/beaker/eval_pipeline_cu129_apt.sh; then
  echo "FATAL: eval_pipeline_cu129_apt.sh has uncommitted changes; commit+push before launching"; exit 2
fi
# Same rule for the noswa exporter: gantry clones the PUSHED commit, so an untracked or
# uncommitted export_noswa_to_hf.py is simply absent in the container and the job dies at the
# export step -- after ~30 minutes of venv build.
if [ "$EXPORTER" = "noswa" ]; then
  git ls-files --error-unmatch debug/ctc_olmo_hybrid/export_noswa_to_hf.py >/dev/null 2>&1 || {
    echo "FATAL: debug/ctc_olmo_hybrid/export_noswa_to_hf.py is untracked; git add + commit + push it"; exit 2; }
  git diff --quiet HEAD -- debug/ctc_olmo_hybrid/export_noswa_to_hf.py || {
    echo "FATAL: export_noswa_to_hf.py has uncommitted changes; commit+push before launching"; exit 2; }
fi
if ! git merge-base --is-ancestor "$REF" "origin/$(git rev-parse --abbrev-ref HEAD)" 2>/dev/null; then
  echo "FATAL: $REF is not on the remote branch -- push first (gantry clones from the remote)"; exit 2
fi

case "$MODE" in full|chunked) ;; *) echo "FATAL: MODE must be full or chunked, got '$MODE'"; exit 2 ;; esac
# The in-process chunk patch masks EVERY attention layer; the olmo3 chunked arms mask only the
# full-attention layers. That mismatch returns a well-formed number, not an error -- refuse it.
if [ "$EXPORTER" = "noswa" ] && [ "$MODE" = "chunked" ]; then
  echo "FATAL: MODE=chunked is not supported for EXPORTER=noswa; use the native evaluator"; exit 2
fi
if [ "$MODE" = "chunked" ] && [ "$TP" != "1" ]; then
  echo "FATAL: MODE=chunked requires TP=1 (the mask patch cannot reach vLLM worker processes); got TP=$TP"; exit 2
fi

# The catalog keys that differ from the logical task name. Getting one of these wrong does not
# crash -- grade_any scores against the wrong task's parser and returns a plausible number.
declare -A CATALOG_ALIAS=( [fiqa]=beir_fiqa [qdmatch_fiqa]=qdmatch [qdmatch_nq]=qdmatch [qdmatch_hpqa]=qdmatch [absence_gutenberg]=absence [outlier_fixedM]=outlier )
# ⚠ outlier_fixedM IS NOT A CATALOG KEY -- it is the fix-k RUNG TREE scored by the `outlier`
# grader (confirmed against the cmix baseline's grade JSON, which records task/scorer=outlier).
# Left unaliased it would default to EVAL_TASK=outlier_fixedM and grade_any would fail to
# resolve a parser. Its rungs also live OUTSIDE the default tree: pass
# RUNG_TREE=ctc_eval_rungs_fixedM (staged from sneetches 2026-08-19).
EVAL_TASK="${EVAL_TASK:-${CATALOG_ALIAS[$TASK]:-$TASK}}"

mkdir -p "$LOGD"
LEDGER="$LOGD/LAUNCH_LEDGER.tsv"
[ -f "$LEDGER" ] || printf 'launched_at\tckpt\ttask\teval_task\trung_task\tmode\trung\tresult_dir\texperiment_id\n' > "$LEDGER"

echo "ckpt=$CKPT task=$TASK eval_task=$EVAL_TASK rung_task=$RUNG_TASK mode=$MODE scale=$MODEL_SCALE exporter=$EXPORTER base=${BASE_MODEL_ID:-<n/a>} tp=$TP -> $RESULT_DIR"
set -- $RUNGS
while [ $# -gt 0 ]; do
  GROUP=""
  for _ in $(seq 1 "$RUNGS_PER_JOB"); do
    [ $# -eq 0 ] && break
    GROUP="${GROUP:+$GROUP }$1"; shift
  done
  TAG=$(echo "$GROUP" | tr ' ' '-')
  NAME="${NAME_PREFIX}-$(echo "$TASK-$MODE-r$TAG" | tr '_' '-')"
  LOG="$LOGD/${RESULT_DIR//\//_}_r${TAG}.log"
  gantry run --name "$NAME" \
    -w ai2/flex2 -b ai2/oe-other \
    ${CLUSTERS:---cluster ai2/jupiter --cluster ai2/ceres --cluster ai2/neptune} \
    --ref "$REF" --gpus "$TP" --priority "$PRIORITY" \
    --beaker-image "$IMAGE" \
    --weka oe-training-default:/weka/oe-training-default \
    --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
    --env "TASK=$RUNG_TASK" --env "EVAL_TASK=$EVAL_TASK" --env "RUNG_LIST=$GROUP" \
    --env "CKPT_NAME=$CKPT" --env "MODE=$MODE" --env "TP=$TP" \
    --env "RUNG_TREE=$RUNG_TREE" --env "CODE_TARBALL=$CODE_TARBALL" \
    --env "RESULT_DIR=$RESULT_DIR" --env "EXPORTER=$EXPORTER" \
    ${BASE_MODEL_ID:+--env BASE_MODEL_ID=$BASE_MODEL_ID} \
    ${TOKENIZER_DIR:+--env TOKENIZER_DIR=$TOKENIZER_DIR} \
    ${NOSWA_MAX_SEQ_LEN:+--env NOSWA_MAX_SEQ_LEN=$NOSWA_MAX_SEQ_LEN} \
    ${NOSWA_DTYPE:+--env NOSWA_DTYPE=$NOSWA_DTYPE} \
    ${EXTRA_STOP_TOKEN_IDS:+--env EXTRA_STOP_TOKEN_IDS=$EXTRA_STOP_TOKEN_IDS} \
    ${MAX_TEST_SAMPLES:+--env MAX_TEST_SAMPLES=$MAX_TEST_SAMPLES} \
    ${QUERY_POSITION:+--env QUERY_POSITION=$QUERY_POSITION} \
    ${COT_MODE:+--env COT_MODE=$COT_MODE} \
    ${GPU_MEM_UTIL:+--env GPU_MEM_UTIL=$GPU_MEM_UTIL} \
    ${VENV_CACHE_KEY:+--env VENV_CACHE_KEY=$VENV_CACHE_KEY} \
    `# ^ VENV_CACHE_KEY passthrough. The pipeline's apt CUDA-toolkit block (which exports` \
    `# CUDA_HOME) lives INSIDE 'if [ ! -x $VENV/bin/python ]', so a venv cache HIT skips it and the` \
    `# job dies later at 'CUDA_HOME: unbound variable' under set -u. The first job of a wave misses` \
    `# the cache, succeeds, uploads it -- and poisons every job launched after it. Until that block` \
    `# is hoisted out of the if, pass a UNIQUE VENV_CACHE_KEY per job to force the working build` \
    `# path (costs ~19 min of venv build; a shared new key just re-poisons on the second job).` \
    --no-python --allow-dirty --timeout 0 --yes \
    -- bash -c '
REPO=$(find / -maxdepth 3 -iname pyproject.toml 2>/dev/null | grep -v /opt/conda | grep -v /root/.cache | head -1 | xargs -r dirname)
rc=0
for R in $RUNG_LIST; do
  echo "=== PACKED RUNG $R of [$RUNG_LIST] ==="
  RUNG=$R bash "$REPO/debug/ctc_vllm_validation/beaker/eval_pipeline_cu129_apt.sh" || { echo "!! rung $R FAILED"; rc=1; }
done
exit $rc
' > "$LOG" 2>&1
  E=$(grep -oE 'beaker\.org/ex/[A-Z0-9]+' "$LOG" | head -1 | sed 's#.*/##')
  for R in $GROUP; do
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$(date -Iseconds)" "$CKPT" "$TASK" "$EVAL_TASK" "$RUNG_TASK" "$MODE" "$R" "$RESULT_DIR" \
      "${E:-SUBMIT-FAILED}" >> "$LEDGER"
  done
  echo "  rungs [$GROUP] -> ${E:-FAILED (see $LOG)}"
done
