#!/usr/bin/env bash
# Compute-vs-loss frontier (layerskip/frontier.py): frozen v6a token routers (TOK_CFG = label=weights:f,...)
# -> shared layer-skip router at keep 0.75 / 0.5 (warm) -> runs/<task>/<NAME>.json; optionally (GRIDSKIP=1)
# the grid's heuristic layer-skip schemes on test64 -> devloss_grid/results_layerskip_t64/<task>_2k.json.
#   TASK=nq TOK_CFG=bar1=v6avg_s0_rhobar:1.0 NAME=fr_bar sbatch debug/learned_router/layerskip/run_frontier_local.sbatch
# Env: TASK, TOK_CFG, NAME, STG (1 = _stg train/val rows), EPOCHS (20), GRIDSKIP, EXTRA.
#SBATCH --job-name=lsfront
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --exclude=mooney,cubbins,mcfuzz,balrog
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/data/prasann/joblogs/lsfront_%j.log

set -uo pipefail
: "${TASK:?set TASK=<manifest key>}"
# a node without /data/prasann (balrog) cannot take --output there (slurm opens it before the script
# runs): submit with --output=/tmp/lsfront_%j.log and the job moves its own log onto node-local /data
mkdir -p /data/prasann/joblogs 2>/dev/null
case "$(readlink -f /proc/$$/fd/1)" in /data/prasann/joblogs/*) ;; *)
  [ -d /data/prasann/joblogs ] && exec >>"/data/prasann/joblogs/lsfront_${SLURM_JOB_ID:-$$}.log" 2>&1;; esac
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



if [ -n "${ORACLE:-}" ]; then  # ORACLE=1: per-example overfit ceiling (oracle_ls.py); OR_ROWS, OR_RUNG, OR_EXTRA
  export WANDB_DIR=/data/prasann/wandb; mkdir -p "$WANDB_DIR"
  DR="$ROOT/data/test64"; [ "${OR_RUNG:-2k}" != "2k" ] && DR="$ROOT/data"
  "$PY" -u debug/learned_router/layerskip/oracle_ls.py --task "$TASK" --ckpt "$CKPT" --ckpt-format "$FMT" --data-root "$DR" \
      --rung "${OR_RUNG:-2k}" --rows "${OR_ROWS:-0-5}" --tokenizer "$TOK" --wandb-group "${WANDB_GROUP:-router-layerskip}" ${OR_EXTRA:-} \
      || { echo "!! oracle FAILED rc=$?"; exit 2; }
  exit 0
fi
if [ -n "${LONGCHECK:-}" ]; then  # LONGCHECK="<tok weights:f> <layerskip weights>"; LC_RUNG (32k), LC_REF (extra token routers)
  set -- $LONGCHECK
  "$PY" -u debug/learned_router/layerskip/longcheck.py --task "$TASK" --ckpt "$CKPT" --ckpt-format "$FMT" --data-root "$ROOT/data/testlong" \
      --rung "${LC_RUNG:-32k}" --tokenizer "$TOK" --tok "$1" --ls "$2" --tok-ref "${LC_REF:-}" \
      --out "$REPO/debug/learned_router/layerskip/runs/$TASK/longcheck_${LC_RUNG:-32k}.json" || { echo "!! longcheck FAILED"; exit 2; }
  exit 0
fi
if [ -n "${FAST:-}" ]; then  # FAST="<weights:f> <layerskip weights>": fastskip.py; FS_RUNG, FS_BATCH, FS_ROWS
  set -- $FAST
  DR="$ROOT/data/test64"; [ "${FS_RUNG:-2k}" != "2k" ] && DR="$ROOT/data"
  "$PY" -u debug/learned_router/layerskip/fastskip.py --task "$TASK" --ckpt "$CKPT" --ckpt-format "$FMT" --data-root "$DR" \
      --rung "${FS_RUNG:-2k}" --batch "${FS_BATCH:-1}" --rows "${FS_ROWS:-8}" --tokenizer "$TOK" --tok "$1" --ls "$2" ${FS_EXTRA:-} \
      --out "$REPO/debug/learned_router/layerskip/runs/$TASK/fastskip_${FS_RUNG:-2k}_b${FS_BATCH:-1}.json" || { echo "!! fastskip FAILED"; exit 2; }
  exit 0
fi
if [ -n "${WALLCLOCK:-}" ]; then  # WALLCLOCK="<weights:f> <layerskip weights rel. to layerskip/>"
  set -- $WALLCLOCK
  "$PY" -u debug/learned_router/layerskip/wallclock.py --task "$TASK" --ckpt "$CKPT" --ckpt-format "$FMT" \
      --test64-root "$ROOT/data/test64" --tokenizer "$TOK" --tok "$1" --ls "$2" --rows "${WC_ROWS:-16}" --batch "${WC_BATCH:-1}" --rung "${WC_RUNG:-2k}" --data-root "$ROOT/data" \
      --out "$REPO/debug/learned_router/layerskip/runs/$TASK/wallclock_${NAME:-wc}_b${WC_BATCH:-1}_${WC_RUNG:-2k}.json" || { echo "!! wallclock FAILED"; exit 2; }
  exit 0
fi
: "${TOK_CFG:?set TOK_CFG=label=weights_name:f,...}"
export WANDB_DIR=/data/prasann/wandb; mkdir -p "$WANDB_DIR"
SFX=_e2e; [ "${STG:-0}" = "1" ] && SFX=_stg
echo "== frontier $TASK tok=$TOK_CFG ($(date +%T))"
"$PY" -u debug/learned_router/layerskip/frontier.py --task "$TASK" --ckpt "$CKPT" --ckpt-format "$FMT" \
    --train-root "$ROOT/data/router_train$SFX" --val-root "$ROOT/data/router_val$SFX" \
    --test64-root "$ROOT/data/test64" --test16-root "$ROOT/data" --tokenizer "$TOK" --tok "$TOK_CFG" \
    --epochs "${EPOCHS:-20}" --name "${NAME:-fr}" --wandb-group "${WANDB_GROUP:-router-layerskip}" ${EXTRA:-} || { echo "!! frontier FAILED rc=$?"; exit 2; }
if [ "${GRIDSKIP:-0}" = "1" ]; then
  RES=$REPO/debug/devloss_grid/results_layerskip_t64; mkdir -p "$RES" "$ROOT/work_lsfront"
  if [ ! -f "$RES/${TASK}_2k.json" ]; then
    echo "== grid heuristic layer-skip schemes on test64 ($(date +%T))"
    "$PY" -u debug/devloss_grid/ctc_devloss_grid.py --task "$DRIVER_TASK" --rung 2k --rows 64 --ckpt "$CKPT" --ckpt-format "$FMT" \
        --data-root "$ROOT/data/test64" --tokenizer "$TOK" --attn-backend flash_2 \
        --schemes full,gold_fl20p8_noslot,gold_fl20p8,gold_fl20p8_skipodd,gold_fl20p8_skipeven,gold_fl20p8_skipgdn2 \
        --out "$ROOT/work_lsfront/${TASK}_{rung}.json" && cp "$ROOT/work_lsfront/${TASK}_2k.json" "$RES/${TASK}_2k.json.part" \
        && mv "$RES/${TASK}_2k.json.part" "$RES/${TASK}_2k.json" && echo "== grid skip schemes done"
  fi
fi
echo "== frontier $TASK finished ($(date))"
