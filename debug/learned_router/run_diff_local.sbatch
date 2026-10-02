#!/usr/bin/env bash
# DIFFERENTIABLE learned token router, ONE task per job on ONE GPU: train one router per tau in
# $TAUS (train_diff_router.py: relaxed removal, hard-concrete gates, task-relative tolerance), then
# score the router_diff_tau<tau> schemes (exact hard removal) on the dev-loss grid's standard test
# rows (16/16/8 at 2k/8k/32k) -> debug/devloss_grid/results_router_diff/.
#
#   TASK=nq sbatch debug/learned_router/run_diff_local.sbatch
#   TASK=nq TAUS=0.1 RUNGS_TRAIN=2k EPOCHS=6 SELFTEST=1 EVAL_RUNGS=2k ROWS=4 RES_DIR=/data/prasann/router_smoke sbatch ...  # debug
#
# Env: TASK (required), TAUS (0.05,0.1,0.2), FLOOR (0.005), EPOCHS (40), RUNGS_TRAIN (2k,8k), SELFTEST
# (run test_relax_gpu.py first), GAP_ROWS (3 per rung), TRAIN (1), EVAL (1), EVAL_RUNGS, ROWS,
# EXTRA_SCHEMES (baselines re-scored for paired deltas; the REINFORCE router at its val-selected
# lambda is added when its weights exist), TRAIN_EXTRA (extra trainer args), RES_DIR. Mirrors debug/devloss_grid/run_grid_local.sbatch: node-local env/caches/checkpoints,
# bootstrap from sneetches (flock-serialised so parallel jobs on one node copy once), logs on /data.
# NEVER lands on cubbins/mcfuzz while rhys_gould holds them (preemptive_high would evict his preemptive
# jobs); balrog is under a MAINT reservation. Excluded.
#SBATCH --job-name=router-diff
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --exclude=mooney,cubbins,mcfuzz,balrog,lorax
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/data/prasann/joblogs/router_diff_%j.log

set -uo pipefail
: "${TASK:?set TASK=<manifest key>}"
# a node without /data/prasann (balrog) cannot take --output there (slurm opens it before the script
# runs): submit with --output=/tmp/router_diff_%j.log and the job moves its own log onto node-local /data
mkdir -p /data/prasann/joblogs 2>/dev/null
case "$(readlink -f /proc/$$/fd/1)" in /data/prasann/joblogs/*) ;; *)
  [ -d /data/prasann/joblogs ] && exec >>"/data/prasann/joblogs/router_diff_${SLURM_JOB_ID:-$$}.log" 2>&1;; esac
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
cd "$REPO"
echo "== router $TASK on $(hostname -s) job ${SLURM_JOB_ID:-?} ($(date +%T))"
# twin jobs may be queued on two partitions: the first to start takes the task, the other exits
TASK_LOCK="$REPO/debug/learned_router/runs/.lock_diff_${TASK}"; mkdir -p "$REPO/debug/learned_router/runs"
if [ "${NO_TASK_LOCK:-0}" != "1" ]; then
  if ! mkdir "$TASK_LOCK" 2>/dev/null; then
    # a scancel'led owner can leave its lock behind: reclaim it when the owner job is no longer queued
    owner=$(cat "$TASK_LOCK/owner" 2>/dev/null); ojob=${owner#*:}
    if [ -n "$ojob" ] && ! squeue -j "$ojob" -h 2>/dev/null | grep -q .; then
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
export HF_HOME=/data/prasann/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONWARNINGS=ignore
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

TAUS="${TAUS:-0.05,0.1,0.2}"
RES="${RES_DIR:-$REPO/debug/devloss_grid/results_router_diff}"; mkdir -p "$RES"
if [ -n "${SELFTEST:-}" ]; then
  echo "== GPU self-test (tiny hybrid: p=1, finite differences, conv gap) ($(date +%T))"
  # fresh Triton cache: a stale per-node cache gave a NaN plain forward of the tiny fp32 model (sneetches)
  TRITON_CACHE_DIR=/data/prasann/tmp/triton_selftest_${SLURM_JOB_ID:-$$} "$PY" -u debug/learned_router/test_relax_gpu.py || { echo "!! self-test FAILED"; exit 3; }
fi
TODO=""
for t in ${TAUS//,/ }; do [ -f "$REPO/debug/learned_router/runs/$TASK/diff_tau$t.json" ] || TODO+="${TODO:+,}$t"; done
[ "${FORCE_TRAIN:-0}" = "1" ] && TODO=$TAUS
if [ "${TRAIN:-1}" = "1" ] && [ -n "$TODO" ]; then
  echo "== train $TASK taus=$TODO ($(date +%T))"
  "$PY" -u debug/learned_router/train_diff_router.py --task "$TASK" --ckpt "$CKPT" --ckpt-format "$FMT" \
      --train-root "$ROOT/data/router_train" --val-root "$ROOT/data/router_val" --tokenizer "$TOK" \
      --rungs "${RUNGS_TRAIN:-2k,8k}" --taus "$TODO" --floor "${FLOOR:-0.005}" --epochs "${EPOCHS:-40}" \
      --gap-rows "${GAP_ROWS:-3}" ${TRAIN_EXTRA:-} || { echo "!! training FAILED rc=$?"; exit 2; }
fi
[ "${EVAL:-1}" = "1" ] || exit 0
SCHEMES=full
for t in ${TAUS//,/ }; do SCHEMES+=",router_diff_tau$t"; done
# the REINFORCE router at its val-selected lambda (headline.json), for a paired comparison
RL_SEL=$("$PY" -c "import json,sys; h=json.load(open('$REPO/debug/learned_router/headline.json')); print(h.get('$TASK',{}).get('selected') or '')" 2>/dev/null)
[ -n "$RL_SEL" ] && [ -f "$REPO/debug/learned_router/weights/$TASK/$RL_SEL.pt" ] && SCHEMES+=",router_$RL_SEL"
SCHEMES+=",${EXTRA_SCHEMES:-gold_rand20p8_noslot,gold_fl20p8_noslot,gold_first20,gold_only_noslot}"
rows_for() { if [ -n "${ROWS:-}" ]; then echo "$ROWS"; elif [ "$1" = "32k" ]; then echo 8; else echo 16; fi; }
EVAL_RUNGS="${EVAL_RUNGS:-2k,8k,32k}"
for r in ${EVAL_RUNGS//,/ }; do
  out="$RES/${TASK}_${r}.json"
  [ -f "$out" ] && [ "${OVERWRITE:-0}" != "1" ] && { echo "-- skip $TASK@$r (exists)"; continue; }
  tmp="$ROOT/work_router_diff/${TASK}_${r}.json"; mkdir -p "$ROOT/work_router_diff"
  echo "== eval $TASK@$r rows=$(rows_for $r) schemes=$SCHEMES ($(date +%T))"; t0=$SECONDS
  "$PY" -u debug/devloss_grid/ctc_devloss_grid.py --task "$DRIVER_TASK" --rung "$r" --rows "$(rows_for $r)" \
      --ckpt "$CKPT" --ckpt-format "$FMT" --data-root "$ROOT/data" --tokenizer "$TOK" --attn-backend flash_2 \
      --schemes "$SCHEMES" --out "$tmp"
  rc=$?
  if [ $rc -eq 0 ] && [ -f "$tmp" ]; then cp "$tmp" "$out.part" && mv "$out.part" "$out" && echo "== done $TASK@$r in $((SECONDS-t0))s"
  else echo "!! eval FAILED $TASK@$r rc=$rc"; printf '%s\t%s\t%s\t%s\t%s\n' "$(date +%FT%T)" "$TASK" "$r" "$rc" "$(hostname -s):${SLURM_JOB_ID:-?}" >> "$RES/FAILED.tsv"; fi
done
echo "== router-diff $TASK finished ($(date))"
