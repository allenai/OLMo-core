#!/usr/bin/env bash
# Oracle-then-classifier pilot (debug/learned_router/oracle/): runs $SCRIPT (oracle.py | classify.py) for $TASK with $ARGS,
# after the same node bootstrap as run_e2e_local.sbatch (env, tokenizer, data, checkpoint onto node-local /data).
#   TASK=nq SCRIPT=oracle.py ARGS="--seeds 3" sbatch --partition=jsteinhardt --qos=preemptive_high --nodelist=cubbins debug/learned_router/oracle/run_oracle.sbatch
#SBATCH --job-name=router-oracle
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --exclude=mooney,mcfuzz,balrog,lorax,sneetches
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/data/prasann/joblogs/router_oracle_%j.log

set -uo pipefail
: "${TASK:?set TASK=<manifest key>}"
# a node without /data/prasann (balrog) cannot take --output there (slurm opens it before the script
# runs): submit with --output=/tmp/router_oracle_%j.log and the job moves its own log onto node-local /data
mkdir -p /data/prasann/joblogs 2>/dev/null
case "$(readlink -f /proc/$$/fd/1)" in /data/prasann/joblogs/*) ;; *)
  [ -d /data/prasann/joblogs ] && exec >>"/data/prasann/joblogs/router_oracle_${SLURM_JOB_ID:-$$}.log" 2>&1;; esac
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
cd "$REPO"
echo "== router $TASK on $(hostname -s) job ${SLURM_JOB_ID:-?} ($(date +%T))"
# twin jobs may be queued on two partitions: the first to start takes the task, the other exits
TASK_LOCK="$REPO/debug/learned_router/runs/.lock_oracle_${TASK}_${LOCK_TAG:-all}"; mkdir -p "$REPO/debug/learned_router/runs"
if [ "${NO_TASK_LOCK:-0}" != "1" ]; then
  if ! mkdir "$TASK_LOCK" 2>/dev/null; then
    # a scancel'led owner can leave its lock behind: reclaim it when the owner job is no longer queued
    owner=$(cat "$TASK_LOCK/owner" 2>/dev/null); ojob=${owner#*:}
    # ... or when the owner is THIS job id (a preempted-and-requeued job finds its own earlier lock)
    if [ -n "$ojob" ] && { [ "$ojob" = "${SLURM_JOB_ID:-}" ] || ! squeue -j "$ojob" -h 2>/dev/null | grep -q .; }; then
      echo ".. reclaiming stale lock of $owner"; rm -rf "$TASK_LOCK"; mkdir "$TASK_LOCK" 2>/dev/null || { echo "-- lost the race for $TASK; exiting"; exit 0; }
    else echo "-- $TASK owned by another job ($owner); exiting"; exit 0; fi
  fi
  echo "$(hostname -s):${SLURM_JOB_ID:-?}" > "$TASK_LOCK/owner"; trap 'rm -rf "$TASK_LOCK"' EXIT; trap 'exit 143' TERM INT
fi
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

XTASKS="${XTASKS:-}"
for xt in ${XTASKS//,/ }; do
  read -r XC XF < <("$PY" -c "import json,sys; t=json.load(open('$MANIFEST'))['tasks']['$xt']; print(t['ckpt'], 'distcp' if t['ckpt_format']=='distcp' else 'hf')")
  if [ "$(hostname -s)" != "sneetches" ] && ! sentinel_ok "$XC" "$XF"; then
    ( flock 9; rel=${XC#$ROOT/}; echo ".. bootstrapping $rel for XTASKS ($(date +%T))"
      mkdir -p "$XC" && rsync -a --exclude .text_export_done --exclude .metadata "$SRC/$rel/" "$XC/" \
        && rsync -a --include '.text_export_done' --include '*/' --include '.metadata' --exclude '*' "$SRC/$rel/" "$XC/" ) 9>"$LOCK"
  fi
  sentinel_ok "$XC" "$XF" || { echo "!!! XTASKS checkpoint $XC incomplete"; exit 1; }
done
[ -n "${XTASKS:-}" ] && TRAIN_EXTRA="--xtasks $XTASKS --xtask-name ${XTASK_NAME:-xtask} ${TRAIN_EXTRA:-}"
[ -f "$TOK/tokenizer.json" ] || { echo "!!! tokenizer missing"; exit 1; }
sentinel_ok "$CKPT" "$FMT" || { echo "!!! checkpoint $CKPT incomplete"; exit 1; }
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
echo "== $SCRIPT $TASK ($(date +%T)) ARGS: ${ARGS:-}"
"$PY" -u debug/learned_router/oracle/$SCRIPT --task "$TASK" --ckpt "$CKPT" --ckpt-format "$FMT" \
    --train-root "$ROOT/data/router_train_e2e" --val-root "$ROOT/data/router_val_e2e" --tokenizer "$TOK" \
    --eval-data-root "$ROOT/data" ${ARGS:-} || { echo "!! $SCRIPT FAILED rc=$?"; exit 2; }
echo "== done ($(date +%T))"
