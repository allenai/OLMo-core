#!/bin/bash
# Export an olmo-core Qwen3-4B checkpoint -> HF, then eval RULER (2 smallest ctx) + contradiction
# (size 100) with a single vLLM load. Runs on 1 GPU on a jsteinhardt node.
#
#   # baseline CPT model (model-only distcp dir that directly contains model_and_optim/):
#   sbatch --job-name=eval-base --nodelist=cubbins \
#     --export=ALL,NAME=q4b-cpt-base,CKPT=/scratch/users/prasann/olmo_ckpts/q4b-dense-cpt-step2385-modelonly \
#     src/scripts/train/sft/run_eval_q4b_lc.sbatch
#
#   # a trained run (save folder containing step<N>/ -> auto-latest):
#   sbatch --job-name=eval-f70 --nodelist=cubbins \
#     --export=ALL,NAME=q4b-cptmix-contra-f70-lr5e5,SAVE_FOLDER=/scratch/users/prasann/olmo_ckpts/q4b-cptmix-contra-f70-lr5e5 \
#     src/scripts/train/sft/run_eval_q4b_lc.sbatch
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=cubbins
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/.eval_%x_%j.log

set -uo pipefail
CR=/scratch/users/prasann/corpus-reasoning
NAME="${NAME:?set NAME}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Base}"
HF_OUT="${HF_OUT:-/scratch/users/prasann/hf_exports/${NAME}_hf}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
RULER_LENGTHS="${RULER_LENGTHS:-L1024,L2048}"
echo "=== HOST=$(hostname) NAME=$NAME START=$(date '+%F %T') ==="

source /system/linux/miniforge-3.13/etc/profile.d/conda.sh

# Trust SLURM's GPU allocation (nvidia-smi is not cgroup-isolated here, so a node-wide scan would
# collide with co-located jobs). Only fall back to scanning if SLURM gave us nothing.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  mapfile -t CLEAN < <(nvidia-smi --query-gpu=memory.used,uuid --format=csv,noheader,nounits | awk -F', ' '$1+0 < 2000 {print $2}')
  export CUDA_VISIBLE_DEVICES="${CLEAN[0]}"
fi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

cd "$CR"
export PYTHONPATH="$CR"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# ---- 1) export olmo distcp -> HF (corpus-reasoning-olmo env; uses the scratch olmo_core) ----
# Stage the checkpoint to node-local /dev/shm first: reading the distcp (esp. the dotfile
# .metadata) directly off /scratch NFS intermittently fails with "not a distributed checkpoint
# folder" on compute nodes (NFS dotfile visibility). A local copy is reliable and faster.
if [ ! -f "$HF_OUT/config.json" ]; then
  if [ -n "${CKPT:-}" ]; then SRC_STEP="$CKPT"; else
    SRC_STEP=$(ls -d "${SAVE_FOLDER:?set SAVE_FOLDER or CKPT}"/step* 2>/dev/null | sort -t p -k2 -n | tail -1)
  fi
  [ -z "${SRC_STEP:-}" ] && { echo "no checkpoint dir found"; exit 2; }
  STAGE=/dev/shm/${NAME}_ckpt
  echo "=== staging $SRC_STEP/model_and_optim -> $STAGE (node-local) ==="
  rm -rf "$STAGE"; mkdir -p "$STAGE"
  # NFS dir-entry cache can hide freshly-written files on a compute node; retry with refreshes.
  ok=""
  for attempt in 1 2 3 4 5; do
    ls -la "$SRC_STEP/model_and_optim/" >/dev/null 2>&1   # force NFS dentry refresh
    cp -rf "$SRC_STEP/model_and_optim" "$STAGE/" 2>/dev/null
    if [ -f "$STAGE/model_and_optim/.metadata" ] && ls "$STAGE"/model_and_optim/*.distcp >/dev/null 2>&1; then
      ok=1; break
    fi
    echo "  stage attempt $attempt incomplete (NFS lag), retrying in 10s..."; rm -rf "$STAGE/model_and_optim"; sleep 10
  done
  [ -z "$ok" ] && { echo "STAGE FAILED: cannot read distcp from $SRC_STEP over NFS"; exit 2; }
  ls -la "$STAGE/model_and_optim/" | head

  conda activate /scratch/users/prasann/conda/envs/corpus-reasoning-olmo
  echo "=== export -> $HF_OUT ==="
  python -u scripts/train/export_olmo_to_hf.py --save-folder "$STAGE" --ckpt "$STAGE" \
    --hf-out "$HF_OUT" --base-model "$BASE_MODEL"
  rc=$?; rm -rf "$STAGE"
  [ $rc -ne 0 ] && { echo "EXPORT FAILED rc=$rc"; exit $rc; }
  conda deactivate
else
  echo "=== HF export already exists at $HF_OUT, skipping export ==="
fi

# ---- 2) eval RULER + contradiction (corpus-reasoning-eval env; vLLM) ----
conda activate /scratch/users/prasann/conda/envs/corpus-reasoning-eval
OUT="$CR/outputs/eval_results/${NAME}_lc.json"
echo "=== eval -> $OUT ==="
python -u scripts/eval/eval_lc_fast.py \
  --base-model "$HF_OUT" --out "$OUT" \
  --max-test-samples "$MAX_SAMPLES" --ruler-lengths "$RULER_LENGTHS" --max-model-len 16384
echo "=== DONE $(date '+%F %T') rc=$? ==="