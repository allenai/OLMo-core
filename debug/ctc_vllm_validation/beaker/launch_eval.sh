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
RUNGS="${RUNGS:-2048 4096 8192 16384 32768}"
RUNG_TREE="${RUNG_TREE:-ctc_eval_rungs}"
CODE_TARBALL="${CODE_TARBALL:-ctc_eval_code_2026-08-13.tar.gz}"
TP="${TP:-2}"
PRIORITY="${PRIORITY:-urgent}"
LOGD="${LOGD:-debug/ctc_vllm_validation/beaker/launches}"
NAME_PREFIX="${NAME_PREFIX:-ev}"
REF="${REF:-$(git rev-parse HEAD)}"
# ⚠ PIN THE IMAGE. A hand-rolled `gantry run` without --beaker-image builds a uv venv from the
# repo's pyproject, whose torch refuses the cluster driver ("NVIDIA driver too old, found version
# 12080"). --no-python keeps that venv from shadowing the image's python; the pipeline builds its
# own vLLM venv on top either way.
IMAGE="${IMAGE:-tylerr/olmo-core-tch291cu128-2025-11-25}"

# ⚠ GANTRY SHIPS THE PUSHED COMMIT, NOT THE WORKING TREE. --allow-dirty silences the check but
# does not upload anything, so a local-only fix to the pipeline runs as whatever is on the remote.
if ! git diff --quiet HEAD -- debug/ctc_vllm_validation/beaker/eval_pipeline_cu129_apt.sh; then
  echo "FATAL: eval_pipeline_cu129_apt.sh has uncommitted changes; commit+push before launching"; exit 2
fi
if ! git merge-base --is-ancestor "$REF" "origin/$(git rev-parse --abbrev-ref HEAD)" 2>/dev/null; then
  echo "FATAL: $REF is not on the remote branch -- push first (gantry clones from the remote)"; exit 2
fi

case "$MODE" in full|chunked) ;; *) echo "FATAL: MODE must be full or chunked, got '$MODE'"; exit 2 ;; esac

# The catalog keys that differ from the logical task name. Getting one of these wrong does not
# crash -- grade_any scores against the wrong task's parser and returns a plausible number.
declare -A CATALOG_ALIAS=( [fiqa]=beir_fiqa [qdmatch_fiqa]=qdmatch [qdmatch_nq]=qdmatch [qdmatch_hpqa]=qdmatch [absence_gutenberg]=absence )
EVAL_TASK="${EVAL_TASK:-${CATALOG_ALIAS[$TASK]:-$TASK}}"

mkdir -p "$LOGD"
LEDGER="$LOGD/LAUNCH_LEDGER.tsv"
[ -f "$LEDGER" ] || printf 'launched_at\tckpt\ttask\teval_task\trung_task\tmode\trung\tresult_dir\texperiment_id\n' > "$LEDGER"

echo "ckpt=$CKPT task=$TASK eval_task=$EVAL_TASK rung_task=$RUNG_TASK mode=$MODE tp=$TP -> $RESULT_DIR"
for R in $RUNGS; do
  NAME="${NAME_PREFIX}-$(echo "$TASK-$MODE-r$R" | tr '_' '-')"
  LOG="$LOGD/${RESULT_DIR//\//_}_r${R}.log"
  gantry run --name "$NAME" \
    -w ai2/flex2 -b ai2/oe-other \
    --cluster ai2/jupiter --cluster ai2/ceres --cluster ai2/neptune \
    --ref "$REF" --gpus "$TP" --priority "$PRIORITY" \
    --beaker-image "$IMAGE" \
    --weka oe-training-default:/weka/oe-training-default \
    --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
    --env "TASK=$RUNG_TASK" --env "EVAL_TASK=$EVAL_TASK" --env "RUNG=$R" \
    --env "CKPT_NAME=$CKPT" --env "MODE=$MODE" --env "TP=$TP" \
    --env "RUNG_TREE=$RUNG_TREE" --env "CODE_TARBALL=$CODE_TARBALL" \
    --env "RESULT_DIR=$RESULT_DIR" \
    ${MAX_TEST_SAMPLES:+--env MAX_TEST_SAMPLES=$MAX_TEST_SAMPLES} \
    ${QUERY_POSITION:+--env QUERY_POSITION=$QUERY_POSITION} \
    ${COT_MODE:+--env COT_MODE=$COT_MODE} \
    ${GPU_MEM_UTIL:+--env GPU_MEM_UTIL=$GPU_MEM_UTIL} \
    --no-python --allow-dirty --timeout 0 --yes \
    -- bash -c '
REPO=$(find / -maxdepth 3 -iname pyproject.toml 2>/dev/null | grep -v /opt/conda | grep -v /root/.cache | head -1 | xargs -r dirname)
bash "$REPO/debug/ctc_vllm_validation/beaker/eval_pipeline_cu129_apt.sh"
' > "$LOG" 2>&1
  E=$(grep -oE 'beaker\.org/ex/[A-Z0-9]+' "$LOG" | head -1 | sed 's#.*/##')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -Iseconds)" "$CKPT" "$TASK" "$EVAL_TASK" "$RUNG_TASK" "$MODE" "$R" "$RESULT_DIR" \
    "${E:-SUBMIT-FAILED}" >> "$LEDGER"
  echo "  rung $R -> ${E:-FAILED (see $LOG)}"
done
