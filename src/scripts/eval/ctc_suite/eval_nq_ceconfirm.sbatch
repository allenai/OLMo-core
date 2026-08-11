#!/bin/bash
#SBATCH --job-name=nq-ceconfirm-eval
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=mooney
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/data/prasann/joblogs/nq_ceconfirm_eval_%j.log
# CE-CONFIRM eval: EXACT mirror of the collapsed nq pilot eval (task=retrieval,
# variant=dense, rung 2048, nproc 1, max-length 4096, max-new-tokens 64, same
# eval jsonl) but for the CE-filtered ckpt, into a SEPARATE out-root so nothing
# overwrites the original nq.json / raw_eval.
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$REPO/src"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
cd "$REPO"
MASTER_PORT=$((29000 + RANDOM % 1000))
python -u -m scripts.eval.ctc_suite.run_rung_eval \
  --task retrieval \
  --ckpt /data/prasann/ctc_suite/ckpts/ctc-1ep-nq-ceconfirm \
  --variant dense \
  --rung-tokens 2048 \
  --eval-jsonl /data/prasann/ctc_suite_data/eval_rungs_pilot/nq/rung_2048.jsonl \
  --arm full \
  --model-scale qwen3.5-0.8b \
  --nproc 1 \
  --master-port "$MASTER_PORT" \
  --out-root "$REPO/results/ctc_suite/pilot_2k/raw_eval_ceconfirm" \
  --small-eval-ok
echo "=== EVAL DONE $(date '+%F %T') rc=$? ==="