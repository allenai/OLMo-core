#!/usr/bin/env bash
# Stage-2 per-(token, layer) SKIP router on top of the frozen token router
# (layerskip/train_layerskip.py): trains one router per (variant, target), val-calibrated threshold,
# test-row eval + baselines IN process -> layerskip/runs/<task>/<NAME>.json.
#   TASK=nq TOK_ROUTER=debug/learned_router/weights/nq/e2ecal2_rho0.1_c0.13.pt TARGETS=0.75,0.5,0.25 \
#     sbatch debug/learned_router/layerskip/run_layerskip_local.sbatch
# Env: TASK, TOK_ROUTER, TARGETS, VARIANTS (perlayer), EPOCHS (40), NAME (ls), SELFTEST (GPU test
# first), TRAIN_EXTRA (extra trainer args).
#SBATCH --job-name=layerskip
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --exclude=mooney,cubbins,mcfuzz,balrog
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=/data/prasann/joblogs/layerskip_%j.log

set -uo pipefail
: "${TASK:?set TASK=<manifest key>}"
# a node without /data/prasann (balrog) cannot take --output there (slurm opens it before the script
# runs): submit with --output=/tmp/layerskip_%j.log and the job moves its own log onto node-local /data
mkdir -p /data/prasann/joblogs 2>/dev/null
case "$(readlink -f /proc/$$/fd/1)" in /data/prasann/joblogs/*) ;; *)
  [ -d /data/prasann/joblogs ] && exec >>"/data/prasann/joblogs/layerskip_${SLURM_JOB_ID:-$$}.log" 2>&1;; esac
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
cd "$REPO"
echo "== layerskip $TASK on $(hostname -s) job ${SLURM_JOB_ID:-?} ($(date +%T))"
source src/scripts/local_env.sh
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
case "$PY" in /data/*) ;; *) echo "!!! interpreter not node-local"; exit 1;; esac
mkdir -p /data/prasann/joblogs /data/prasann/conda/envs
LOCK=/data/prasann/.router_bootstrap.lock
if ! "$PY" -c "import urllib.parse, torch, olmo_core" 2>/dev/null; then
  ( flock 9
    if ! "$PY" -c "import urllib.parse, torch, olmo_core" 2>/dev/null; then
      echo ".. bootstrapping conda env from sneetches ($(date +%T))"
      rsync -a /net/sneetches/data/prasann/conda/envs/corpus-reasoning-olmo/ /data/prasann/conda/envs/corpus-reasoning-olmo/ && echo ".. env bootstrapped ($(date +%T))"
    fi ) 9>"$LOCK"
  "$PY" -c "import urllib.parse, torch, olmo_core" 2>/dev/null || { echo "!!! env bootstrap failed"; exit 98; }
fi
export HF_HOME=/data/prasann/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONWARNINGS=ignore DEVLOSS_SKIP_CHECKS=${DEVLOSS_SKIP_CHECKS:-1}
export TMPDIR=/data/prasann/tmp TRITON_CACHE_DIR=/data/prasann/triton_cache XDG_CACHE_HOME=/data/prasann/xdg_cache TORCHINDUCTOR_CACHE_DIR=/data/prasann/inductor_cache
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR"
[ -d /usr/local/cuda-12.8 ] && { export CUDA_HOME=/usr/local/cuda-12.8; export PATH=$CUDA_HOME/bin:$PATH; }
TOK=/data/prasann/hf_cache/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68
ROOT=/data/prasann/devloss_grid
SRC=/net/sneetches/data/prasann/devloss_grid
MANIFEST=$REPO/debug/devloss_grid/manifest.json
read -r CKPT FMT DRIVER_TASK < <("$PY" - "$MANIFEST" "$TASK" <<'PYEOF'
import json, sys
t = json.load(open(sys.argv[1]))["tasks"][sys.argv[2]]
assert not t.get("dropped") and t.get("ckpt"), f"{sys.argv[2]} is dropped / has no checkpoint"
print(t["ckpt"], "distcp" if t["ckpt_format"] == "distcp" else "hf", f"ctc_{sys.argv[2]}")
PYEOF
)
[ -n "${CKPT:-}" ] || { echo "!!! no checkpoint for $TASK"; exit 1; }
sentinel_ok() { if [ "$2" = "distcp" ]; then [ -f "$1/model_and_optim/.metadata" ] || [ -f "$1/step"*"/model_and_optim/.metadata" ]; else [ -f "$1/.text_export_done" ]; fi; }
if [ "$(hostname -s)" != "sneetches" ]; then
  avail_gb=$(( $(df --output=avail -k /data | tail -1) / 1048576 ))
  [ "$avail_gb" -ge 300 ] || { echo "!!! $(hostname -s):/data has only ${avail_gb} GB free"; exit 99; }
  ( flock 9
    mkdir -p "$ROOT" /data/prasann/hf_cache/hub
    [ -f "$TOK/tokenizer.json" ] || rsync -a /net/sneetches/data/prasann/hf_cache/hub/models--Qwen--Qwen3.5-0.8B-Base /data/prasann/hf_cache/hub/
    rsync -a "$SRC/data/" "$ROOT/data/" && echo ".. data synced from sneetches ($(date +%T))"
    if ! sentinel_ok "$CKPT" "$FMT"; then
      rel=${CKPT#$ROOT/}; t0=$SECONDS; echo ".. bootstrapping $rel ($(date +%T))"
      mkdir -p "$CKPT" && rsync -a --exclude .text_export_done --exclude .metadata "$SRC/$rel/" "$CKPT/" \
        && rsync -a --include '.text_export_done' --include '*/' --include '.metadata' --exclude '*' "$SRC/$rel/" "$CKPT/" \
        && echo ".. bootstrapped $rel in $((SECONDS-t0))s"
    fi ) 9>"$LOCK"
fi
[ -f "$TOK/tokenizer.json" ] || { echo "!!! tokenizer missing"; exit 1; }
sentinel_ok "$CKPT" "$FMT" || { echo "!!! checkpoint $CKPT incomplete"; exit 1; }
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader


if [ -n "${SELFTEST:-}" ]; then
  echo "== GPU self-test ($(date +%T))"
  TRITON_CACHE_DIR=/data/prasann/tmp/triton_selftest_${SLURM_JOB_ID:-$$} "$PY" -u debug/learned_router/layerskip/test_layerskip_gpu.py || { echo "!! self-test FAILED"; exit 3; }
fi
: "${TOK_ROUTER:?set TOK_ROUTER=<frozen token-router weights>}"
export WANDB_DIR=/data/prasann/wandb; mkdir -p "$WANDB_DIR"  # keep run files off /accounts
echo "== train $TASK variants=${VARIANTS:-perlayer} targets=${TARGETS:-0.75,0.5,0.25} ($(date +%T))"
"$PY" -u debug/learned_router/layerskip/train_layerskip.py --task "$TASK" --ckpt "$CKPT" --ckpt-format "$FMT" \
    --train-root "$ROOT/data/router_train_e2e" --val-root "$ROOT/data/router_val_e2e" --eval-data-root "$ROOT/data" \
    --tokenizer "$TOK" --tok-router "$TOK_ROUTER" --rungs "${RUNGS_TRAIN:-2k}" --n-train "${N_TRAIN:-16}" --n-val "${N_VAL:-32}" \
    --val-extra-from-train "${VAL_EXTRA:-32}" --eval-rows "${EVAL_ROWS:-16}" --targets "${TARGETS:-0.75,0.5,0.25}" \
    --variants "${VARIANTS:-perlayer}" --epochs "${EPOCHS:-40}" --name "${NAME:-ls}" \
    --wandb-group "${WANDB_GROUP:-router-layerskip}" ${TRAIN_EXTRA:-} || { echo "!! training FAILED rc=$?"; exit 2; }
echo "== layerskip $TASK finished ($(date))"
