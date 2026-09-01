#!/bin/bash
# Sharpen wave: wait for arm build (3487369), weka-sync, launch the 12 arm-dependent trains.
# (The 4 mix_s{64,160}M seed-2 replicates were launched directly -- their arms were already synced.)
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
cd "$REPO"
for i in $(seq 1 150); do
  ST=$(sacct -j 3487369 --format=State --noheader 2>/dev/null | head -1 | tr -d ' ')
  echo "[shp $i] build=$ST $(date '+%T')"
  [[ "$ST" == "COMPLETED" ]] && break
  [[ "$ST" == FAILED* || "$ST" == CANCELLED* || "$ST" == TIMEOUT ]] && { echo "[shp] build FAILED"; exit 1; }
  sleep 120
done
[[ "$(sacct -j 3487369 --format=State --noheader 2>/dev/null | head -1 | tr -d ' ')" == "COMPLETED" ]] || exit 1
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
NAME=outlier-lm-weka-sync19 PRIORITY=urgent \
  S3_PREFIX=s3://ai2-llm/checkpoints/prasanns/outlier_lengthmix \
  DEST_REL=ai2-llm/checkpoints/prasanns/outlier_lengthmix \
  bash src/scripts/train/memexpress/singletask_ladder/stage_eval500_v2_to_weka_gantry.sh 2>&1 | tail -1
sleep 600
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
export PYTHONPATH="$REPO/src"
WEKA_ROOT=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
BASE="$WEKA_ROOT/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim"
LOGD=debug/outlier_lengthmix_scaling/launches
launch () { local ARM=$1 TASK=$2 SEQ=$3 variant=$4 LR=$5 RUN=$6; shift 6
  echo "[shp] launch $RUN"
  timeout 300 $PY -u src/scripts/train/memexpress/ctc_suite/beaker_ctc_suite.py \
    --task "$TASK" --variant "$variant" --model-scale 4b --model-family qwen3_5 \
    --run-name "$RUN" --num-nodes 1 --epochs 1 --seq-len "$SEQ" --lr "$LR" "$@" \
    --global-batch 8 --micro-batch-instances 1 \
    --data-root "$WEKA_ROOT/outlier_lengthmix/arms/$ARM" \
    --base-checkpoint "$BASE" --wandb-group outlier-lengthmix-checks \
    launch > "$LOGD/${RUN}.log" 2>&1 &
  sleep 4
}
launch p32k_4000   outlier   65536 full 5e-6 lmx-full-p32k4000-4b --pack
launch q32k_16000  qdmatch   32768 full 5e-6 lmx-full-q32k16000-qd-4b --pack
launch qmix_s64M   qdmatch   65536 full 5e-6 lmx-full-qmixs64M-qd-4b --pack
launch qmix_s160M  qdmatch   65536 full 5e-6 lmx-full-qmixs160M-qd-4b --pack
launch qmix_s320M  qdmatch   65536 full 5e-6 lmx-full-qmixs320M-qd-4b --pack
launch qmix_s320M  qdmatch   65536 sparselandmark 1e-5 lmx-slm-qmixs320M-qd-4b
launch nmix_s16M   retrieval 65536 full 5e-6 lmx-full-nmixs16M-nq-4b --pack
launch nmix_s32M   retrieval 65536 full 5e-6 lmx-full-nmixs32M-nq-4b --pack
launch nmix_s48M   retrieval 65536 full 5e-6 lmx-full-nmixs48M-nq-4b --pack
launch nmix_s48M   retrieval 65536 sparselandmark 1e-5 lmx-slm-nmixs48M-nq-4b
launch nqD32k_4000 retrieval 65536 full 5e-6 lmx-full-nqD32k4000-4b --pack
launch nqD64k_2000 retrieval 66560 full 5e-6 lmx-full-nqD64k2000-4b
wait
echo "[shp] all 12 submitted $(date '+%T')"
