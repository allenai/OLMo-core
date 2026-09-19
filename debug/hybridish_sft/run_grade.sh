#!/usr/bin/env bash
# Grade one hybridish SFT checkpoint on the 2k-32k CTC mix, 4-GPU DDP, on the Berkeley cluster.
#
#   bash debug/hybridish_sft/run_grade.sh sft_4to1_ml_hf sft4to1_fixed [eval_per_task]
#
# The interpreter and the transformers shadow MUST be node-local /data paths: the same venv exists
# on NFS and importing torch from there parks the process in nfs_wait_bit_killable for minutes at
# ~0% CPU, which reads as a hung GPU.
set -euo pipefail

CKPT_NAME="${1:?usage: run_grade.sh <ckpt-dir-name> <tag> [eval_per_task]}"
TAG="${2:?}"
N="${3:-500}"
NODE="${NODE:-horton}"
GPUS="${GPUS:-4}"

REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
LOG="$REPO/debug/hybridish_sft/grade_${TAG}.log"

srun --partition=berkeleynlp --nodelist="$NODE" --gres=gpu:"$GPUS" \
     --cpus-per-task=16 --mem=200G --time=4:00:00 \
     bash -lc "
set -euo pipefail
PY=/data/prasann/ctc_vllm_venv/bin/python
[ -x \"\$PY\" ] || { echo 'FATAL_NO_VENV: no node-local interpreter on '\$(hostname); exit 97; }
export PYTHONPATH=/data/prasann/tf514_shadow
export HOME=/data/prasann/hometmp TMPDIR=/data/prasann/tmp
export TOKENIZERS_PARALLELISM=false
echo \"[grade] \$(hostname) \$(date +%T)\"
\$PY -m torch.distributed.run --nproc-per-node=$GPUS --master-port=\$((29000 + RANDOM % 1000)) \
  $REPO/debug/hybridish_sft/grade_ctc_mix.py \
  --ckpt /data/prasann/hybridish/$CKPT_NAME \
  --shards /scratch/users/prasann/ctc_hybridish_sft/shards_long32k \
  --src-jsonl /scratch/users/prasann/ctc_hybridish_sft/long/mix_long.jsonl \
  --plugin /data/prasann/hybridish/plugin_src \
  --ctcsrc /scratch/users/prasann/hyb_sft/ctcsrc \
  --eval-per-task $N --train-n 0 --tag $TAG --max-new 24 --grade-bs 2
echo \"[grade] done \$(date +%T)\"
" > "$LOG" 2>&1

echo "log: $LOG"
tail -25 "$LOG"
