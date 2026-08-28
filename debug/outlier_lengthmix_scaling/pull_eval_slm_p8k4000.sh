#!/bin/bash
# Wait for the weka->S3 push of lmx-slm-p8k4000-4b, pull to mooney /data, eval locally
# (fixed recipe + fast sparse decode).
set -uo pipefail
PUSH=01M12QX4F3S7PAYQAWSGF0PGPN
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
cd "$REPO"
state () {
  beaker experiment get "$1" --format json 2>/dev/null | python3 -c "
import json,sys
e=json.load(sys.stdin)[0]
st=(e.get('jobs') or [{}])[-1].get('status',{})
print('CANCELED' if st.get('canceled') else ('DONE:'+str(st.get('exitCode'))) if st.get('finalized') else 'RUNNING' if st.get('started') else 'SCHEDULED')"
}
for i in $(seq 1 40); do
  S=$(state "$PUSH"); echo "[pull $i] push=$S $(date '+%T')"
  # aws s3 sync exits 2 on skip-warnings (dangling wandb symlinks); payload still synced --
  # the pull step hard-verifies config.json + .metadata, so treat DONE:2 as pass.
  [[ "$S" == "DONE:0" || "$S" == "DONE:2" ]] && break
  [[ "$S" == DONE:* || "$S" == CANCELED ]] && { echo "[pull] push FAILED"; exit 1; }
  sleep 45
done
S=$(state "$PUSH"); [[ "$S" == "DONE:0" || "$S" == "DONE:2" ]] || { echo "[pull] timeout ($S)"; exit 1; }

echo "[pull] pulling to mooney + launching eval"
sbatch --job-name=pull-ev-slm-p8k4000 -w mooney --qos=preemptive_high --account=site \
  --gres=gpu:H200:2 --cpus-per-task=16 --mem=200G --time=03:00:00 \
  --output=/data/prasann/joblogs/pull_ev_slm_p8k4000_%j.log \
  --partition=jsteinhardt --wrap='
set -e
export HOME=/data/prasann/home TMPDIR=/data/prasann/tmp AWS_PROFILE=S3
DST=/data/prasann/ctc_suite/ckpts/lmx-slm-p8k4000-4b
mkdir -p "$DST"
aws s3 sync s3://ai2-llm/checkpoints/prasanns/outlier_lengthmix/ckpt_slm_p8k4000 "$DST" --only-show-errors
ls "$DST/config.json" "$DST/model_and_optim/.metadata" || { echo "ckpt incomplete"; exit 1; }
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
cd /scratch/users/prasann/corpus-reasoning
export PYTHONPATH=$REPO/src:$REPO/src/scripts:/scratch/users/prasann/corpus-reasoning
export EVAL500_ROOT=/data/prasann/outlier_lengthmix/eval500_local/outlier
export TRITON_CACHE_DIR=/data/prasann/triton_cache
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
OLMO_LANDMARK_SPARSE_DECODE=1 $PY -m torch.distributed.run --nproc_per_node=2 --master_port=29631 $REPO/src/scripts/ctc_eval/eval/eval_lc_native.py \
  --model-path "$DST" --out "$DST/eval/outlier_multirung.json" --tokenizer Qwen/Qwen3.5-0.8B \
  --max-length 40960 --max-test-samples 600 --batch-size 1 --skip-ruler --skip-gen \
  --landmark-mem-id 248200 --landmark-pad-id 248203 --eos-token-id 248044 \
  --prompt-format chat --query-position after \
  --ladder --ladder-tasks outlier --ladder-rungs 3k,8k
cp "$DST/eval/outlier_multirung.json" /scratch/users/prasann/stl_eval_results/lmx-slm-p8k4000-4b_outlier_multirung.json
echo EVAL-OK'
echo "[pull] eval job submitted"
