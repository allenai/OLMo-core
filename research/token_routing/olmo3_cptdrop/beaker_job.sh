#!/bin/bash
# One olmo3_cptdrop arm family on a Beaker 8-GPU node (launched by launch_beaker.sh via gantry, repo checkout = cwd).
# Pulls data + the stage-2 base from the S3 hub to local disk, trains with the SAME run_arm.sbatch recipe as
# Berkeley, scores every final export with eval_devce.py (dev32k_1024 + short4k_v2), then pushes eval JSONs
# and model-only exports to s3://ai2-llm/checkpoints/prasanns/_transfer/olmo3_cptdrop/{devce,runs}/.
#   FAMILY = C120 | C240 | C480 | IFL | IRAND | GEN
#   GEN reads env: NAME (run prefix), ARM (D|IRAND|IFL), KEEP (optional), CTX (32k|64k), TOP (steps of the main
#   WSD run), SAVE (save interval, optional), BRANCHES ("B@S,B@S" decay branches to B steps from stepS)
set -euo pipefail
FAMILY="${1:?FAMILY}"
# ROOT on weka (one dir per family: no two jobs ever write the same file); JIT caches on node-local /tmp.
# gantry --no-python: use the image's own interpreter (torch 2.9.1 / flash-attn 2.8.2). Under --install true gantry
# runs an EMPTY venv (no torch, no pip).
export CODE="$PWD" PY=/opt/conda/bin/python TMPBASE=/tmp/o3cd_tmp
# GEN jobs differ only by NAME: key the dir on it so concurrent GEN jobs never share a data folder
export ROOT="${ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/prasanns/olmo3_cptdrop/beaker_${NAME:-$FAMILY}}"
mkdir -p "$ROOT/data" "$ROOT/base" "$ROOT/runs" "$ROOT/logs" "$ROOT/devce" ~/.aws
printf '%s\n' "$AWS_CREDS" > ~/.aws/credentials
printf '%s\n' "$AWS_CFG" > ~/.aws/config
export AWS_PROFILE=S3
# The image lacks boto3 and some olmo-core deps; install those only (pip install -e . would pull torch>=2.10).
"$PY" -m pip install -q boto3 "dataclass-extensions>=0.3.0" "cached-path>=1.7.2" bettermap importlib_resources
PYTHONPATH="$CODE/src:$CODE/recipes/train/research_trainer:$CODE/recipes/train:$CODE" "$PY" -c \
  "import torch, flash_attn, olmo_core; from olmo_core.nn.attention import position_window; import train_ctc_suite; print('imports ok', torch.__version__, flash_attn.__version__)"
[ "$FAMILY" = CHECK ] && { echo "CHECK ok"; exit 0; }
log() { echo "[beaker-job $(date '+%F %T')] $*"; }
S3SYNC="$CODE/research/token_routing/olmo3_cptdrop/s3_sync.py"
ARM_SH="$CODE/research/token_routing/olmo3_cptdrop/run_arm.sbatch"
df -h "$ROOT" | tail -1
nvidia-smi -L

CTX="${CTX:-32k}"; export CTX
case "$FAMILY" in
  C1*|C2*|C4*) NEED="c8k_ee train32k_head_ee" ;;
  IFL|IRAND) NEED="train32k_ee" ;;
  GEN) NEED=$([ "$CTX" = 64k ] && echo train64k_ee || echo train32k_ee) ;;
  *) echo "unknown FAMILY $FAMILY"; exit 2 ;;
esac
if [ "$CTX" = 64k ]; then DEV=dev64k_512; YF=8; else DEV=dev32k_1024; YF=4; fi
for d in $NEED $DEV short4k_v2; do "$PY" -u "$S3SYNC" down "$ROOT/data/$d" "data/$d" | tail -1; done
"$PY" -u "$S3SYNC" down "$ROOT/base/stage2-step47684" base/stage2-step47684 | tail -1
# Training shards (*_ee) must END every row with EOS and have no other EOS: the trainer splits at EOS, and both a
# stale shard (livelock on empty epochs) and EOS-first rows (degenerate 1-token document -> soft-token hang) broke
# jobs on 2026-10-22. Eval shards (dev*, short*) are read row by row and START with EOS.
"$PY" - "$ROOT/data" $NEED $DEV short4k_v2 <<'PYEOF'
import json, sys
import numpy as np
root = sys.argv[1]
for d in sys.argv[2:]:
    m = json.load(open(f"{root}/{d}/metadata.json")); L = m["max_example_len"]; eos = m["eos_token_id"]
    a = np.memmap(f"{root}/{d}/token_ids_part_0000.npy", dtype=np.uint32, mode="r").reshape(-1, L)
    col = -1 if d.endswith("_ee") else 0
    f = float((a[:, col] == eos).mean())
    n_eos = int((a == eos).sum())
    print(f"data check {d}: {a.shape[0]} rows x {L}, EOS at {'end' if col else 'start'} {f:.4f}, total EOS {n_eos}", flush=True)
    if f < 1.0 or n_eos != a.shape[0]:
        sys.exit(f"{d}: rows do not have exactly one EOS at the {'end' if col else 'start'} -- refusing to train")
PYEOF

FINALS=()
run() {  # name, then env for run_arm
  local name=$1; shift
  log "START $name"
  set +e +o pipefail
  env "$@" RUN="$name" bash "$ARM_SH" 2>&1 | tee "$ROOT/logs/$name.log" | grep -E "\[step=[0-9]|train/CE loss|Error|error|Traceback" --line-buffered
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
  GEN)
    : "${NAME:?}" "${ARM:?}" "${TOP:?}"
    run "$NAME-$TOP" ARM="$ARM" MAX_STEPS_TOTAL="$TOP" ${SAVE:+SAVE_INTERVAL=$SAVE}
    FINALS=("$NAME-$TOP")
    BL="${BRANCHES:-}"; for br in ${BL//,/ }; do
      B=${br%@*}; S0=${br#*@}
      run "$NAME-$B" ARM="$ARM" MAX_STEPS_TOTAL="$B" BRANCH_FROM="$ROOT/runs/$NAME-$TOP/step$S0"
      FINALS+=("$NAME-$B")
    done ;;
esac

# Score each final export with the 8-GPU row-chunked scorer (YaRN as trained).
SPECS=""; for r in "${FINALS[@]}"; do SPECS="$SPECS ${r#o3cd-}:runs/$r:$YF"; done
ok=1
set +e +o pipefail
SPECS="$SPECS" SHARDS="$DEV:$([ "$DEV" = dev64k_512 ] && echo 64 || echo 128) short4k_v2:256" \
  CODE="$CODE" PY="$PY" ROOT="$ROOT" TMPBASE="$TMPBASE" \
  bash "$CODE/research/token_routing/olmo3_cptdrop/devce_multi.sh" 2>&1 | tee "$ROOT/logs/devce.log" | grep -E "merged|WARN"
[ "${PIPESTATUS[0]}" -eq 0 ] || ok=0
set -e -o pipefail
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
