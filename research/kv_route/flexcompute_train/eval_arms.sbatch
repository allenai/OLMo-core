#!/bin/bash
# Score the flexcompute ladder's checkpoints on `contradiction_iid` (500 examples per rung).
#
# Native olmo_core generate, dense box-marker emitter (`--variant dense`) -- the emitter the runs
# were TRAINED on. The routers are re-enabled from each checkpoint's own config.json by
# `post_build_hook_from_config`, so a routed arm is scored WITH its routing; a checkpoint that
# carries router weights would fail the strict load otherwise, so this cannot silently score a
# dense model either.
#
#   RUNS="fxt-g1-dense fxt-g1-aggressive" sbatch research/kv_route/flexcompute_train/eval_arms.sbatch
#
# CAVEAT on the eval set: contradiction_iid was built disjoint from the `single_task_ladders_20k`
# contradiction TRAIN file, not from the ctc-data seed-pool short-heavy arm these runs train on.
# Numbers are comparable ACROSS arms (identical eval, identical exposure); treat the absolute
# level as "held-out-ish", not as an audited IID split.
#SBATCH --job-name=fxteval
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=cubbins
#SBATCH --nodes=1
#SBATCH --gres=gpu:H200:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=300G
#SBATCH --time=10:00:00
#SBATCH --output=/data/prasann/joblogs/fxteval_%x_%j.log
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || git -C "${SLURM_SUBMIT_DIR:-.}" rev-parse --show-toplevel)}
REPO="${REPO:-$REPO_ROOT}"
RUNS="${RUNS:?set RUNS=\"fxt-g1-dense ...\"}"
NGPU="${NGPU:-4}"
# `sbatch --export` splits on COMMAS, so a comma inside RUNGS silently truncates it to the
# first rung (2026-09-09, job 3523841 ran "2k" only). Pass rungs with "+" as the separator.
RUNGS="${RUNGS:-2k+8k}"   # the mix is 45% 2k / 27% 4k / 16% 8k; 16k/32k cost ~3x the wall-clock
RUNGS="${RUNGS//+/,}"
MAX_TEST="${MAX_TEST:-500}"
MAX_LENGTH="${MAX_LENGTH:-40960}"
WORK_ROOT=/data/prasann/flexcompute_train
OUT_DIR="$WORK_ROOT/eval"
RUNGS_LOCAL="$WORK_ROOT/eval_rungs/contradiction_iid"
RUNGS_NFS=/scratch/users/prasann/ctc_suite_staged/eval_rungs/contradiction_iid
# /scratch hit 100%% of its 500G quota on 2026-09-09, so nothing here writes to it --
# the rung files are only READ from there, and the mirror is node-local.
MIRROR=/data/prasann/flexcompute_train/logs
mkdir -p "$OUT_DIR" "$RUNGS_LOCAL" "$MIRROR"

_SRC_LOG="/data/prasann/joblogs/fxteval_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
_mirror() { cp -f "$_SRC_LOG" "$MIRROR/eval_${SLURM_JOB_ID}.log" 2>/dev/null || true
            cp -f "$OUT_DIR"/*.json "$MIRROR/" 2>/dev/null || true; }
( while true; do _mirror; sleep 30; done ) & _MIRROR_PID=$!
trap '_mirror; kill $_MIRROR_PID 2>/dev/null' EXIT

ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$REPO/recipes/eval/legacy_native:$REPO/research/common:$REPO/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/data/prasann/triton_cache TMPDIR=/data/prasann/tmp
# BUG WORKAROUND (2026-09-09, job 3523878): every block-skip checkpoint dies in
# `olmo_core/nn/attention/kv_route.py:_compacted_block_mask` with
#   RuntimeError: The size of tensor a (19) must match the size of tensor b (20) at dim 2
# Cause: `_masked_attention` sets `Kp = min(T, _round_up(counts.max(), 1024))`. The `min(T, ...)`
# clamp makes Kp equal to T, and at eval T is an arbitrary prompt length, not a multiple of
# FLEX_BLOCK_SIZE. `k_first = pos_k[:, ::bs]` then has ceil(Kp/bs) blocks while
# `k_last = pos_k[:, bs-1::bs]` has floor(Kp/bs), and the two are AND-ed. In TRAINING Kp is always
# block-aligned (T = 65536, and round_up(.., 1024) is a multiple of 128), which is why this path
# has never failed before. Block skipping routes its key mask through kv_route_attention, so this
# hits arms with the KV router OFF too.
# KV_ROUTE_BLOCKMASK=dense selects the REFERENCE mask builder (`create_block_mask`) instead. Its
# documented cost is an 8.6 GB dense mask at 65k x 131k -- but at these eval lengths (<= 8k) it is
# ~130 MB per layer and transient, so the workaround is free here.
export KV_ROUTE_BLOCKMASK="${KV_ROUTE_BLOCKMASK:-dense}"

# Stage the rung files node-local once (/scratch is ~5 MB/s NFS).
for f in rung_2560.jsonl rung_8192.jsonl rung_16384.jsonl rung_32768.jsonl; do
  [ -s "$RUNGS_LOCAL/$f" ] || cp "$RUNGS_NFS/$f" "$RUNGS_LOCAL/$f" || { echo "FATAL: no $f"; exit 3; }
done
export DC_RUNG_FILES="{\"contradiction\": {\"2k\": \"$RUNGS_LOCAL/rung_2560.jsonl\", \
\"8k\": \"$RUNGS_LOCAL/rung_8192.jsonl\", \"16k\": \"$RUNGS_LOCAL/rung_16384.jsonl\", \
\"32k\": \"$RUNGS_LOCAL/rung_32768.jsonl\"}}"

echo "=== HOST=$(hostname) RUNS='$RUNS' RUNGS=$RUNGS MAX_TEST=$MAX_TEST START=$(date '+%F %T') ==="
cd "$REPO"
for RUN in $RUNS; do
  # train_ctc_suite.py's post-fit model-only export writes config.json + model_and_optim/ into the
  # save folder ITSELF, not into a step<N>/ subdirectory -- unlike the mid-run checkpointer. The
  # step*/ glob alone silently skipped every arm (2026-09-09, job 3523841), so try the run root
  # first and keep the step glob as the fallback for a run that saved mid-flight.
  CKPT=""
  [ -f "$WORK_ROOT/ckpts/$RUN/model_and_optim/.metadata" ] && CKPT="$WORK_ROOT/ckpts/$RUN"
  for d in $(ls -d "$WORK_ROOT/ckpts/$RUN"/step*/ 2>/dev/null | sed 's#/$##' | sort -V); do
    [ -f "$d/model_and_optim/.metadata" ] && CKPT="$d"
  done
  if [ -z "$CKPT" ]; then echo "=== SKIP $RUN: no complete step dir ==="; continue; fi
  OUT="$OUT_DIR/${RUN}_contra_iid.json"
  echo "=== [$RUN] CKPT=$CKPT -> $OUT $(date '+%F %T') ==="
  PORT=$((26000 + RANDOM % 2000))
  "$ENV/bin/python" -m torch.distributed.run --nproc_per_node="$NGPU" --master_port="$PORT" \
    recipes/eval/legacy_native/ctc_eval/eval/eval_lc_native_docchunk_ladder.py \
    --variant dense --model-path "$CKPT" --out "$OUT" \
    --tokenizer /data/prasann/hf_models/Qwen3.5-0.8B-Base --family qwen3_5 \
    --tasks contradiction --rungs "$RUNGS" \
    --max-test-samples "$MAX_TEST" --max-length "$MAX_LENGTH" \
    --query-position after --cot-mode none
  echo "=== [$RUN] rc=$? $(date '+%F %T') ==="
  [ -f "$OUT" ] && cat "$OUT"
done
echo "=== ALL EVALS DONE $(date '+%F %T') ==="
