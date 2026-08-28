#!/bin/bash
# Chase the 4 LR-bracket-extension runs; submit fixed-recipe evals as checkpoints appear.
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
cd "$REPO"
RUNS="lmx-full-lr1e5-4b-loc lmx-slm-lr1e5-4b-loc lmx-full-lr5e6-4b-loc lmx-slm-lr5e6-4b-loc"
donelist=""
for i in $(seq 1 120); do
  alldone=1
  for RUN in $RUNS; do
    echo "$donelist" | grep -q "$RUN" && continue
    ROOT=/net/mooney/data/prasann/ctc_suite/ckpts/$RUN
    if [ -e "$ROOT/model_and_optim/.metadata" ] && [ -e "$ROOT/config.json" ]; then
      EXTRA=""
      case "$RUN" in *slm*) VARIANT=landmark; EXTRA=",LANDMARK_MEM_ID=248200,LANDMARK_PAD_ID=248203";; *) VARIANT=dense;; esac
      echo "[chaser-ext] $RUN ready -> eval"
      sbatch --job-name="ev-$RUN" -w mooney --qos=preemptive_high --account=site \
        --export=ALL,TASK=outlier,VARIANT=$VARIANT,CKPT=/data/prasann/ctc_suite/ckpts/$RUN,RUN_OVERRIDE=$RUN,TOKENIZER=Qwen/Qwen3.5-0.8B,RUNGS=3k+8k,MAX_TEST=600,NGPU=2,EVAL500_ROOT=/data/prasann/outlier_lengthmix/eval500_local/outlier$EXTRA \
        --gres=gpu:H200:2 --time=03:00:00 \
        src/scripts/train/memexpress/singletask_ladder/run_q4b_stl_multirung_eval.sbatch >/dev/null
      donelist="$donelist $RUN"
    else
      alldone=0
    fi
  done
  [ "$alldone" = 1 ] && { echo "[chaser-ext] all 4 submitted"; break; }
  sleep 90
done
