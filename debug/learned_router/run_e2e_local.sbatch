#!/usr/bin/env bash
# END-TO-END learned token router (train_e2e_router.py): target-keep Lagrangian, one router per rho in
# $RHOS, then select_e2e.py (val) and the grid driver on the test rows -> results_router_e2e/.
#   TASK=nq RHOS=0.1,0.2 N_TRAIN=32 N_VAL=16 EPOCHS=3 sbatch debug/learned_router/run_e2e_local.sbatch
# Env: TASK, RHOS, N_TRAIN/N_VAL (per rung), EPOCHS, RUNGS_TRAIN (2k,8k), TRAIN (1), EVAL (1),
# EVAL_RUNGS, ROWS, EXTRA_SCHEMES, TRAIN_EXTRA, RES_DIR, LOCK_TAG (twin-job lock suffix).
#SBATCH --job-name=router-e2e
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --exclude=mooney,cubbins,mcfuzz,balrog,lorax
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/data/prasann/joblogs/router_e2e_%j.log

set -uo pipefail
: "${TASK:?set TASK=<manifest key>}"
# a node without /data/prasann (balrog) cannot take --output there (slurm opens it before the script
# runs): submit with --output=/tmp/router_e2e_%j.log and the job moves its own log onto node-local /data
mkdir -p /data/prasann/joblogs 2>/dev/null
case "$(readlink -f /proc/$$/fd/1)" in /data/prasann/joblogs/*) ;; *)
  [ -d /data/prasann/joblogs ] && exec >>"/data/prasann/joblogs/router_e2e_${SLURM_JOB_ID:-$$}.log" 2>&1;; esac
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
cd "$REPO"
echo "== router $TASK on $(hostname -s) job ${SLURM_JOB_ID:-?} ($(date +%T))"
# twin jobs may be queued on two partitions: the first to start takes the task, the other exits
TASK_LOCK="$REPO/debug/learned_router/runs/.lock_e2e_${TASK}_${LOCK_TAG:-all}"; mkdir -p "$REPO/debug/learned_router/runs"
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
# XTASKS (cross-task router): every listed task's checkpoint must be on this node too
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

if [ -n "${SELFTEST:-}" ]; then
  echo "== GPU self-test ($(date +%T))"
  TRITON_CACHE_DIR=/data/prasann/tmp/triton_selftest_${SLURM_JOB_ID:-$$} "$PY" -u debug/learned_router/test_relax_gpu.py || { echo "!! self-test FAILED"; exit 3; }
fi
RHOS="${RHOS:-0.05,0.1,0.2,0.3,0.5}"
NP="${NAME_PREFIX:-e2e}"  # run/weights name prefix: <NP>_rho<rho>
RES="${RES_DIR:-$REPO/debug/devloss_grid/results_router_e2e}"; mkdir -p "$RES"
TODO=""
RUNDIR=$TASK; [ -n "${XTASKS:-}" ] && RUNDIR=${XTASK_NAME:-xtask}
for r in ${RHOS//,/ }; do [ -f "$REPO/debug/learned_router/runs/$RUNDIR/${NP}_rho$r.json" ] || TODO+="${TODO:+,}$r"; done
[ "${FORCE_TRAIN:-0}" = "1" ] && TODO=$RHOS
if [ "${TRAIN:-1}" = "1" ] && [ -n "$TODO" ]; then
  echo "== train $TASK rhos=$TODO ($(date +%T))"
  "$PY" -u debug/learned_router/train_e2e_router.py --task "$TASK" --ckpt "$CKPT" --ckpt-format "$FMT" \
      --train-root "$ROOT/data/router_train${SPLIT_SUFFIX:-_e2e}" --val-root "$ROOT/data/router_val${SPLIT_SUFFIX:-_e2e}" --tokenizer "$TOK" \
      --rungs "${RUNGS_TRAIN:-2k,8k}" --n-train "${N_TRAIN:-96}" --n-val "${N_VAL:-32}" --rhos "$TODO" \
      --epochs "${EPOCHS:-10}" --name "${NP}_rho{rho}" --wandb-group "${WANDB_GROUP:-router-e2e}" ${TRAIN_EXTRA:-} || { echo "!! training FAILED rc=$?"; exit 2; }
fi
[ "${EVAL:-1}" = "1" ] || exit 0
[ "$NP" = "e2e" ] && { "$PY" -u debug/learned_router/select_e2e.py --task "$TASK" || echo "!! selection failed"; }
SCHEMES=full
for r in ${RHOS//,/ }; do [ -f "$REPO/debug/learned_router/weights/$TASK/${NP}_rho$r.pt" ] && SCHEMES+=",router_${NP}_rho$r"; done
[ "$NP" = "e2e" ] && for t in 0.05 0.1 0.2; do [ -f "$REPO/debug/learned_router/weights/$TASK/e2e_tau$t.pt" ] && SCHEMES+=",router_e2e_tau$t"; done
SCHEMES+=",${EXTRA_SCHEMES:-gold_fl20p8_noslot,gold_only_noslot,gold_rand20p8_noslot}"
rows_for() { if [ -n "${ROWS:-}" ]; then echo "$ROWS"; elif [ "$1" = "32k" ]; then echo 8; else echo 16; fi; }
EVAL_RUNGS="${EVAL_RUNGS:-2k,8k,32k}"
# every rung in ONE process (model loaded once); cells whose JSON exists are skipped
todo_r=""; todo_n=""
for r in ${EVAL_RUNGS//,/ }; do
  [ -f "$RES/${TASK}_${r}.json" ] && [ "${OVERWRITE:-0}" != "1" ] && { echo "-- skip $TASK@$r (exists)"; continue; }
  todo_r+="${todo_r:+,}$r"; todo_n+="${todo_n:+,}$(rows_for $r)"
done
if [ -n "$todo_r" ]; then
  mkdir -p "$ROOT/work_router_e2e"; t0=$SECONDS
  echo "== eval $TASK@$todo_r rows=$todo_n schemes=$SCHEMES ($(date +%T))"
  "$PY" -u debug/devloss_grid/ctc_devloss_grid.py --task "$DRIVER_TASK" --rung "$todo_r" --rows "$todo_n" \
      --ckpt "$CKPT" --ckpt-format "$FMT" --data-root "$ROOT/data" --tokenizer "$TOK" --attn-backend flash_2 \
      --schemes "$SCHEMES" --out "$ROOT/work_router_e2e/${TASK}_{rung}.json"
  rc=$?
  for r in ${todo_r//,/ }; do
    tmp="$ROOT/work_router_e2e/${TASK}_${r}.json"
    if [ -f "$tmp" ]; then cp "$tmp" "$RES/${TASK}_${r}.json.part" && mv "$RES/${TASK}_${r}.json.part" "$RES/${TASK}_${r}.json" && echo "== done $TASK@$r"
    else echo "!! eval FAILED $TASK@$r rc=$rc"; printf '%s\t%s\t%s\t%s\t%s\n' "$(date +%FT%T)" "$TASK" "$r" "$rc" "$(hostname -s):${SLURM_JOB_ID:-?}" >> "$RES/FAILED.tsv"; fi
  done
  echo "== eval of $todo_r took $((SECONDS-t0))s"
fi
echo "== router-e2e $TASK finished ($(date))"
