#!/bin/bash
# Chase the 16k anchors (+32k big arm when it exists) into evals. Same state machine as
# overnight_chaser, fixed ports, generations preserved, post-eval ckpt cleanup.
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
D=$REPO/debug/outlier_lengthmix_scaling
STATE=$D/chaser_state; LOGD=$D/launches
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
cd "$REPO"; export PYTHONPATH="$REPO/src"
JOBS="
lmx-full-p16k250-lropt-4b|full|3k,8k,16k,32k
lmx-full-p16k1000-lropt-4b|full|3k,8k,16k,32k
lmx-full-p16k4000-lropt-4b|full|3k,8k,16k,32k
lmx-slm-p16k4000-lropt-4b|slm|3k,8k,16k
lmx-full-p32k8000-lropt-4b|full|3k,8k,16k,32k
lmx-slm-p16k8000-lropt-4b|slm|3k,8k,16k
lmx-slm-p16k16000-lropt-4b|slm|3k,8k,16k
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
savedir () { beaker experiment spec "$2" 2>/dev/null | grep -Eo "${1}-[0-9]{8}T[0-9]{6}-[0-9]{4}" | head -1; }
fire_dense_eval () {
  local RUN=$1 EX=$2 SD
  SD=$(savedir "$RUN" "$EX"); [ -z "$SD" ] && return 1
  timeout 300 $PY src/scripts/train/memexpress/singletask_ladder/run_q4b_beaker_multirung_eval.py \
    "$RUN" ai2/jupiter-cirrascale-2 --task outlier --variant dense \
    --ckpt "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_suite/ckpts/$SD" \
    --query-position after --prompt-format chat --ladder-version v3 \
    --tokenizer Qwen/Qwen3.5-0.8B > "$LOGD/eval-$RUN.log" 2>&1
  grep -q 'beaker.org/ex/' "$LOGD/eval-$RUN.log"
}
fire_push () {
  local RUN=$1 Y=/tmp/apush_$$.yaml
  sed "s/RUNPLACEHOLDER/$RUN/g" "$D/push_template.yaml" > "$Y"
  beaker experiment create "$Y" --name "push-$RUN-$(date +%H%M)" --workspace ai2/flex2 2>&1 \
    | grep -Eo 'ex/[A-Z0-9]+' | head -1 | sed 's|ex/||'
  rm -f "$Y"
}
fire_local_eval () {
  local RUN=$1 RUNGS=$2
  sbatch --job-name="aev-$RUN" -w mooney --qos=preemptive_high --account=site \
    --gres=gpu:H200:2 --cpus-per-task=16 --mem=200G --time=05:00:00 \
    --output=/data/prasann/joblogs/aev_${RUN}_%j.log \
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
export EVAL500_ROOT=/data/prasann/outlier_lengthmix/eval500_local/outlier
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
OLMO_LANDMARK_SPARSE_DECODE=1 $PY -m torch.distributed.run --nproc_per_node=2 --master_port=$((20000 + SLURM_JOB_ID % 10000)) $REPO/src/scripts/ctc_eval/eval/eval_lc_native.py \
  --model-path "$DST" --out "$DST/eval_outlier_multirung.json" --tokenizer Qwen/Qwen3.5-0.8B \
  --max-length 40960 --max-test-samples 600 --batch-size 1 --skip-ruler --skip-gen \
  --landmark-mem-id 248200 --landmark-pad-id 248203 --eos-token-id 248044 \
  --prompt-format chat --query-position after \
  --ladder --ladder-tasks outlier --ladder-rungs '"$RUNGS"'
cp "$DST/eval_outlier_multirung.json" /scratch/users/prasann/stl_eval_results/${RUN}-chat_outlier_multirung.json
cp "$DST"/eval_outlier_multirung*generations* /scratch/users/prasann/stl_eval_results/${RUN}-chat_outlier_multirung.generations.jsonl 2>/dev/null || true
rm -rf "$DST"
rm -rf "$DST"
echo EVAL-OK'
}
for i in $(seq 1 150); do
  echo "$JOBS" | while IFS='|' read -r RUN VAR RUNGS; do
    [ -z "$RUN" ] && continue
    SF="$STATE/$RUN.anchor.state"
    ST=$(cat "$SF" 2>/dev/null || echo TRAIN)
    case "$ST" in
      EVAL|GAVE_UP*) continue ;;
      TRAIN)
        EX=$(exid "$RUN"); [ -z "$EX" ] && continue
        S=$(state "$EX")
        if [[ "$S" == "DONE:0" ]]; then
          if [ "$VAR" = full ]; then
            fire_dense_eval "$RUN" "$EX" && echo EVAL > "$SF" || echo "GAVE_UP:eval-launch" > "$SF"
          else
            P=$(fire_push "$RUN"); [ -n "$P" ] && echo "PUSH:$P" > "$SF"
          fi
        elif [[ "$S" == DONE:* || "$S" == CANCELED ]]; then
          echo "GAVE_UP:train-$S" > "$SF"; echo "[anchor] $RUN train ended badly: $S"
        fi ;;
      PUSH:*)
        P=${ST#PUSH:}; S=$(state "$P")
        if [[ "$S" == "DONE:0" || "$S" == "DONE:2" ]]; then
          fire_local_eval "$RUN" "$RUNGS" && echo EVAL > "$SF"
        elif [[ "$S" == DONE:* || "$S" == CANCELED ]]; then
          echo "GAVE_UP:push-$S" > "$SF"
        fi ;;
    esac
  done
  N=$(grep -l '^EVAL$' "$STATE"/*.anchor.state 2>/dev/null | wc -l)
  echo "[anchor tick $i] $N/7 chained $(date '+%T')"
  [ "$N" -ge 7 ] && break
  sleep 300
done
echo "[anchor] exiting"
