#!/bin/bash
# One olmo3_cptdrop arm family on a Beaker 8-GPU node (launched by launch_beaker.sh via gantry, repo checkout = cwd).
# Pulls data + the stage-2 base from the S3 hub to local disk, trains with the SAME run_arm.sbatch recipe as
# Berkeley, scores every final export with eval_devce.py (dev32k_1024 + short4k_v2), then pushes eval JSONs
# and model-only exports to s3://ai2-llm/checkpoints/prasanns/_transfer/olmo3_cptdrop/{devce,runs}/.
#   FAMILY = C120 | C240 | C480 | IFL | IRAND
set -euo pipefail
FAMILY="${1:?FAMILY}"
# ROOT on weka (one dir per family: no two jobs ever write the same file); JIT caches on node-local /tmp.
export CODE="$PWD" PY="$(command -v python)" TMPBASE=/tmp/o3cd_tmp
export ROOT="${ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/prasanns/olmo3_cptdrop/beaker_$FAMILY}"
mkdir -p "$ROOT/data" "$ROOT/base" "$ROOT/runs" "$ROOT/logs" "$ROOT/devce" ~/.aws
printf '%s\n' "$AWS_CREDS" > ~/.aws/credentials
printf '%s\n' "$AWS_CFG" > ~/.aws/config
export AWS_PROFILE=S3
log() { echo "[beaker-job $(date '+%F %T')] $*"; }
S3SYNC="$CODE/research/token_routing/olmo3_cptdrop/s3_sync.py"
ARM_SH="$CODE/research/token_routing/olmo3_cptdrop/run_arm.sbatch"
df -h "$ROOT" | tail -1
nvidia-smi -L

case "$FAMILY" in
  C*) NEED="c8k train32k_head" ;;
  IFL|IRAND) NEED="train32k" ;;
  *) echo "unknown FAMILY $FAMILY"; exit 2 ;;
esac
for d in $NEED dev32k_1024 short4k_v2; do "$PY" -u "$S3SYNC" down "$ROOT/data/$d" "data/$d" | tail -1; done
"$PY" -u "$S3SYNC" down "$ROOT/base/stage2-step47684" base/stage2-step47684 | tail -1

FINALS=()
run() {  # name, then env for run_arm
  local name=$1; shift
  log "START $name"
  set +e +o pipefail
  env "$@" RUN="$name" bash "$ARM_SH" 2>&1 | tee "$ROOT/logs/$name.log" | grep -E "INFO\s+\[step=|train/CE loss|Error|error|Traceback" --line-buffered
  local rc=${PIPESTATUS[0]}
  set -e -o pipefail
  log "END $name rc=$rc"
  [ "$rc" -eq 0 ] || exit "$rc"
}
case "$FAMILY" in
  C120|C240|C480)
    B=${FAMILY#C}; H=$((B / 2))
    run "o3cd-C-$B-p1" ARM=C1 MAX_STEPS_TOTAL=$B HARD_STOP=$H
    run "o3cd-C-$B" ARM=C2 MAX_STEPS_TOTAL=$H BRANCH_FROM="$ROOT/runs/o3cd-C-$B-p1/step$H" BRANCH_OPTIM_ONLY=1 WARMUP=0 DECAY_FRAC=0.2
    FINALS=("o3cd-C-$B") ;;
  IFL)
    run o3cd-IFL-2160 ARM=IFL MAX_STEPS_TOTAL=2160 SAVE_INTERVAL=486
    run o3cd-IFL-540 ARM=IFL MAX_STEPS_TOTAL=540 BRANCH_FROM="$ROOT/runs/o3cd-IFL-2160/step486"
    run o3cd-IFL-1080 ARM=IFL MAX_STEPS_TOTAL=1080 BRANCH_FROM="$ROOT/runs/o3cd-IFL-2160/step972"
    FINALS=(o3cd-IFL-540 o3cd-IFL-1080 o3cd-IFL-2160) ;;
  IRAND)
    run o3cd-IRAND-2600 ARM=IRAND MAX_STEPS_TOTAL=2600 SAVE_INTERVAL=585
    run o3cd-IRAND-650 ARM=IRAND MAX_STEPS_TOTAL=650 BRANCH_FROM="$ROOT/runs/o3cd-IRAND-2600/step585"
    run o3cd-IRAND-1300 ARM=IRAND MAX_STEPS_TOTAL=1300 BRANCH_FROM="$ROOT/runs/o3cd-IRAND-2600/step1170"
    FINALS=(o3cd-IRAND-650 o3cd-IRAND-1300 o3cd-IRAND-2600) ;;
esac

# Score each final export (YaRN x4, as trained); one GPU per (run, shard), all in parallel.
export PYTHONPATH="$CODE/src:$CODE/research/token_routing/olmo3_cptdrop" HF_HOME="$ROOT/hf_cache" TMPDIR="$TMPBASE"
g=0; pids=()
for r in "${FINALS[@]}"; do
  for sh in dev32k_1024 short4k_v2; do
    CUDA_VISIBLE_DEVICES=$g "$PY" -u "$CODE/research/token_routing/olmo3_cptdrop/eval_devce.py" \
      --ckpt "$ROOT/runs/$r" --yarn-factor 4 --dev "$ROOT/data/$sh" --label "$r" \
      --out "$ROOT/devce/${r}_${sh}.json" > "$ROOT/logs/devce-${r}-${sh}.log" 2>&1 &
    pids+=($!); g=$(( (g + 1) % 8 ))
  done
done
ok=1; for p in "${pids[@]}"; do wait "$p" || ok=0; done
log "evals done ok=$ok"; ls -la "$ROOT/devce"
"$PY" -u "$S3SYNC" up "$ROOT/devce" devce | tail -1
for r in "${FINALS[@]}"; do
  "$PY" -u "$S3SYNC" up "$ROOT/runs/$r/model_and_optim" "runs/$r/model_and_optim" | tail -1
  for f in config.json flops.json provenance.json; do
    [ -f "$ROOT/runs/$r/$f" ] && mkdir -p "$ROOT/up/$r" && cp "$ROOT/runs/$r/$f" "$ROOT/up/$r/"
  done
  [ -d "$ROOT/up/$r" ] && "$PY" -u "$S3SYNC" up "$ROOT/up/$r" "runs/$r/meta" | tail -1
done
"$PY" -u "$S3SYNC" up "$ROOT/logs" "logs/beaker-$FAMILY" | tail -1
[ "$ok" -eq 1 ] || { log "some evals failed"; exit 3; }
log "DONE $FAMILY"
