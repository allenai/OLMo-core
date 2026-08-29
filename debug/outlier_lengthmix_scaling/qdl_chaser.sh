#!/bin/bash
# qdmatch smoke chaser: when a qd train finishes on Beaker, push ckpt to S3 relay and eval
# locally on mooney (both variants local — Beaker eval bundle has no qdmatch rungs).
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
D=$REPO/debug/outlier_lengthmix_scaling
STATE=$D/chaser_state; mkdir -p "$STATE"
LOGD=$D/launches
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
cd "$REPO"

JOBS="
lmx-full-q2k1250-qd-4b|full
lmx-full-q2k2500-qd-4b|full
lmx-full-q2k10000-qd-4b|full
lmx-full-q2k20000-qd-4b|full
lmx-full-q8k1000-qd-4b|full
lmx-full-q8k2000-qd-4b|full
lmx-full-q8k8000-qd-4b|full
lmx-slm-q2k20000-qd-4b|slm
lmx-slm-q8k8000-qd-4b|slm
"
exid () { grep -Eo 'beaker.org/ex/[A-Z0-9]+' "$LOGD/$1.log" 2>/dev/null | head -1 | sed 's|.*/||'; }
state () {
  beaker experiment get "$1" --format json 2>/dev/null | python3 -c "
import json,sys
try: e=json.load(sys.stdin)[0]
except Exception: print('UNKNOWN'); raise SystemExit
st=(e.get('jobs') or [{}])[-1].get('status',{})
print('CANCELED' if st.get('canceled') else ('DONE:'+str(st.get('exitCode'))) if st.get('finalized') else 'RUNNING' if st.get('started') else 'SCHEDULED')"
}
fire_push () {
  local RUN=$1 Y=/tmp/qdpush_$$.yaml
  sed "s/RUNPLACEHOLDER/$RUN/g" "$D/push_template.yaml" > "$Y"
  beaker experiment create "$Y" --name "push-$RUN-$(date +%H%M)" --workspace ai2/flex2 2>&1 \
    | grep -Eo 'ex/[A-Z0-9]+' | head -1 | sed 's|ex/||'
  rm -f "$Y"
}
fire_local_eval () { # run variant
  local RUN=$1 VAR=$2 BS=2 SPD=""
  [ "$VAR" = slm ] && { BS=1; SPD="OLMO_LANDMARK_SPARSE_DECODE=1 "; }
  sbatch --job-name="cev-$RUN" -w mooney --qos=preemptive_high --account=site \
    --gres=gpu:H200:2 --cpus-per-task=16 --mem=200G --time=05:00:00 \
    --output=/data/prasann/joblogs/cev_${RUN}_%j.log \
    --partition=jsteinhardt --wrap='
set -e
export HOME=/data/prasann/home TMPDIR=/data/prasann/tmp AWS_PROFILE=S3
export TRITON_CACHE_DIR=/data/prasann/triton_cache
RUN='"$RUN"'
DST=/data/prasann/ctc_suite/ckpts/$RUN
mkdir -p "$DST"
aws s3 sync "s3://ai2-llm/checkpoints/prasanns/outlier_lengthmix/ckpt_relay/$RUN" "$DST" --only-show-errors || [ $? -eq 2 ]
ls "$DST/config.json" "$DST/model_and_optim/.metadata" || { echo "ckpt incomplete"; exit 1; }
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
cd /scratch/users/prasann/corpus-reasoning
export PYTHONPATH=$REPO/src:$REPO/src/scripts:/scratch/users/prasann/corpus-reasoning
export EVAL500_ROOT=/data/prasann/qdmatch_lengthmix/eval_rungs
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
'"$SPD"'$PY -m torch.distributed.run --nproc_per_node=2 --master_port=$((20000 + SLURM_JOB_ID % 10000)) $REPO/src/scripts/ctc_eval/eval/eval_lc_native.py \
  --model-path "$DST" --out "$DST/eval_qdmatch_multirung.json" --tokenizer Qwen/Qwen3.5-0.8B \
  --max-length 40960 --max-test-samples 600 --batch-size '"$BS"' --skip-ruler --skip-gen \
  --landmark-mem-id 248200 --landmark-pad-id 248203 --eos-token-id 248044 \
  --prompt-format chat --query-position after \
  --ladder --ladder-tasks qdmatch_nq --ladder-rungs 3k,8k
cp "$DST/eval_qdmatch_multirung.json" /scratch/users/prasann/stl_eval_results/${RUN}-chat_qdmatch_multirung.json
rm -rf "$DST"
echo EVAL-OK'
}

for i in $(seq 1 100); do
  echo "$JOBS" | while IFS='|' read -r RUN VAR; do
    [ -z "$RUN" ] && continue
    SF="$STATE/$RUN.qdl.state"
    ST=$(cat "$SF" 2>/dev/null || echo TRAIN)
    case "$ST" in
      EVAL|GAVE_UP*) continue ;;
      TRAIN)
        EX=$(exid "$RUN"); [ -z "$EX" ] && continue
        S=$(state "$EX")
        if [[ "$S" == "DONE:0" ]]; then
          P=$(fire_push "$RUN")
          [ -n "$P" ] && { echo "PUSH:$P" > "$SF"; echo "[qdchaser] $RUN pushed ($P)"; }
        elif [[ "$S" == DONE:* || "$S" == CANCELED ]]; then
          echo "[qdchaser] $RUN TRAIN ended badly: $S"; echo "GAVE_UP:train-$S" > "$SF"
        fi ;;
      PUSH:*)
        P=${ST#PUSH:}; S=$(state "$P")
        if [[ "$S" == "DONE:0" || "$S" == "DONE:2" ]]; then
          fire_local_eval "$RUN" "$VAR" && echo EVAL > "$SF" && echo "[qdchaser] $RUN eval submitted"
        elif [[ "$S" == DONE:* || "$S" == CANCELED ]]; then
          echo "[qdchaser] $RUN PUSH failed: $S"; echo "GAVE_UP:push-$S" > "$SF"
        fi ;;
    esac
  done
  N=$(grep -l '^EVAL$' "$STATE"/lmx-*-qd-4b.qdl.state 2>/dev/null | wc -l)
  echo "[qdchaser tick $i] $N/9 chained $(date '+%T')"
  [ "$N" -ge 9 ] && break
  sleep 300
done
echo "[qdchaser] exiting"
