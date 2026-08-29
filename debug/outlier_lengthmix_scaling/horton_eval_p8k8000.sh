#!/bin/bash
# Pull the 3 sparse 8k-arm ckpts from S3 to horton /data and eval there (separate QOS pool
# from mooney). Waits for the batch push job first.
set -uo pipefail
PUSH=01M13B169CC09WEJDNDC5187JA
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
cd "$REPO"
state () {
  beaker experiment get "$1" --format json 2>/dev/null | python3 -c "
import json,sys
e=json.load(sys.stdin)[0]
st=(e.get('jobs') or [{}])[-1].get('status',{})
print('CANCELED' if st.get('canceled') else ('DONE:'+str(st.get('exitCode'))) if st.get('finalized') else 'RUNNING' if st.get('started') else 'SCHEDULED')"
}
for i in $(seq 1 160); do
  S=$(state "$PUSH"); echo "[h8k $i] push=$S $(date '+%T')"
  [[ "$S" == "DONE:0" || "$S" == "DONE:2" ]] && break
  [[ "$S" == DONE:* || "$S" == CANCELED ]] && { echo "[h8k] push FAILED"; exit 1; }
  sleep 45
done
S=$(state "$PUSH"); [[ "$S" == "DONE:0" || "$S" == "DONE:2" ]] || { echo "[h8k] timeout"; exit 1; }

for RUN in lmx-slm-p8k8000-4b; do
  sbatch --job-name="evh-$RUN" --partition=berkeleynlp --qos=preemptive_high_sewonm -w horton \
    --gres=gpu:H200:2 --cpus-per-task=16 --mem=200G --time=03:00:00 \
    --output=/data/prasann/joblogs/evh_${RUN}_%j.log --wrap='
set -e
export HOME=/data/prasann/home TMPDIR=/data/prasann/tmp AWS_PROFILE=S3
export TRITON_CACHE_DIR=/data/prasann/triton_cache
mkdir -p /data/prasann/joblogs /data/prasann/outlier_lengthmix
RUN='"$RUN"'
DST=/data/prasann/outlier_lengthmix/ckpts/$RUN
mkdir -p "$DST"
aws s3 sync "s3://ai2-llm/checkpoints/prasanns/outlier_lengthmix/ckpt_relay/$RUN" "$DST" --only-show-errors
ls "$DST/config.json" "$DST/model_and_optim/.metadata"
E5=/data/prasann/outlier_lengthmix/eval500_local/outlier
mkdir -p "$E5/outlier"
aws s3 sync s3://ai2-llm/checkpoints/prasanns/outlier_lengthmix/eval_rungs/outlier "$E5/tmp_rungs" --only-show-errors
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
CRV=/scratch/users/prasann/corpus-reasoning
export PYTHONPATH=$REPO/src:$REPO/src/scripts:$CRV
export EVAL500_ROOT=$E5
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
# build the v2-named ladder files locally if absent (instant, same builder+pool via NFS pkl read)
if [ ! -s "$E5/outlier/outlier_wiki100w_n55_k3_eval_600.jsonl" ]; then
  PYTHONPATH=$REPO/src /scratch/users/prasann/conda/envs/corpus-reasoning-eval/bin/python \
    $REPO/src/corpus_reasoning/data/build_v2_outlier_ladder.py --num-examples 600 \
    --pool-cache /net/mooney/data/prasann/single_task_ladders_20k/wiki100w_article_pool_smoke.pkl \
    --out-root "$E5"
fi
cd "$CRV"
OLMO_LANDMARK_SPARSE_DECODE=1 $PY -m torch.distributed.run --nproc_per_node=2 --master_port=$((20000 + SLURM_JOB_ID % 10000)) $REPO/src/scripts/ctc_eval/eval/eval_lc_native.py \
  --model-path "$DST" --out "$DST/eval_outlier_multirung.json" --tokenizer Qwen/Qwen3.5-0.8B \
  --max-length 40960 --max-test-samples 600 --batch-size 1 --skip-ruler --skip-gen \
  --landmark-mem-id 248200 --landmark-pad-id 248203 --eos-token-id 248044 \
  --prompt-format chat --query-position after \
  --ladder --ladder-tasks outlier --ladder-rungs 3k,8k
cp "$DST/eval_outlier_multirung.json" /scratch/users/prasann/stl_eval_results/${RUN}-chat_outlier_multirung.json
echo EVAL-OK'
  sleep 2
done
echo "[h8k] 3 horton evals submitted"
