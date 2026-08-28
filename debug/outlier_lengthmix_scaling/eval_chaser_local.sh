#!/bin/bash
# Watch mooney for finished lmx-*-loc checkpoints; submit one local multirung eval per run
# (mooney-pinned, qos=preemptive so it coexists with training's preemptive_high GPU cap).
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
cd "$REPO"
RUNS="lmx-full-lr2e5-4b-loc lmx-full-lr5e5-4b-loc lmx-full-lr1p2e4-4b-loc lmx-slm-lr2e5-4b-loc lmx-slm-lr5e5-4b-loc lmx-slm-lr1p2e4-4b-loc"
declare -A DONE
for i in $(seq 1 90); do
  alldone=1
  for RUN in $RUNS; do
    [ "${DONE[$RUN]:-}" = 1 ] && continue
    STEP=$(ls -d /net/mooney/data/prasann/ctc_suite/ckpts/$RUN/step* 2>/dev/null | sort -V | tail -1)
    if [ -n "$STEP" ] && [ -e "$STEP/model_and_optim/.metadata" ]; then
      VARIANT=dense; [[ "$RUN" == *slm* ]] && VARIANT=landmark
      LOCAL_STEP="/data/prasann/ctc_suite/ckpts/$RUN/$(basename "$STEP")"
      echo "[chaser] $RUN ready ($LOCAL_STEP) -> eval variant=$VARIANT"
      sbatch --job-name="ev-$RUN" -w mooney \
        --export=ALL,TASK=outlier,VARIANT=$VARIANT,CKPT=$LOCAL_STEP,TOKENIZER=Qwen/Qwen3.5-0.8B,RUNGS=3k,MAX_TEST=600,NGPU=2 \
        --gres=gpu:H200:2 --time=02:00:00 \
        src/scripts/train/memexpress/singletask_ladder/run_q4b_stl_multirung_eval.sbatch
      sbatch --job-name="ev8k-$RUN" -w mooney \
        --export=ALL,TASK=outlier,VARIANT=$VARIANT,CKPT=$LOCAL_STEP,TOKENIZER=Qwen/Qwen3.5-0.8B,RUNGS=8k,MAX_TEST=600,NGPU=2 \
        --gres=gpu:H200:2 --time=02:00:00 \
        src/scripts/train/memexpress/singletask_ladder/run_q4b_stl_multirung_eval.sbatch
      DONE[$RUN]=1
    else
      alldone=0
    fi
  done
  [ "$alldone" = 1 ] && { echo "[chaser] all 6 runs evaluated-or-submitted"; break; }
  sleep 60
done
echo "[chaser] exit $(date '+%T')"
