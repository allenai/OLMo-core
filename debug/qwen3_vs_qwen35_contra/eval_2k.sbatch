#!/bin/bash
# Eval a 2k-context contradiction sanity checkpoint (native olmo-core backend -- the path we want
# to validate). Pinned to the node holding the checkpoint on /data. One model per invocation.
#   sbatch -w horton --export=ALL,MODEL=qwen35,CKPT=/data/prasann/ctc_suite/ckpts/q35-4b-contra-2k-full-sanity eval_2k.sbatch
#   sbatch -w lorax  --export=ALL,MODEL=qwen3, CKPT=/data/prasann/ctc_suite/ckpts/q3-4b-contra-2k-full-sanity  eval_2k.sbatch
#SBATCH --job-name=contra2k-eval
#SBATCH --partition=berkeleynlp
#SBATCH --nodes=1
#SBATCH --gres=gpu:H200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/data/prasann/joblogs/contra2k_eval_%x_%j.log
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$ENV" ] || ENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$REPO/src"
export HF_HOME=/scratch/users/prasann/huggingface-cache
unset TRANSFORMERS_CACHE   # stale profile value -> repo-id offline lookups miss
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/data/prasann/triton_cache
mkdir -p "$TRITON_CACHE_DIR" 2>/dev/null || true
cd "$REPO"

MODEL="${MODEL:?set MODEL=qwen3|qwen35}"
CKPT="${CKPT:?set CKPT=<save_folder with config.json + model_and_optim>}"
EVAL_JSONL=/scratch/users/prasann/ctc_suite_staged/eval_rungs/contradiction/rung_2048.jsonl
[ -f "$CKPT/model_and_optim/.metadata" ] || { echo "FATAL: $CKPT/model_and_optim/.metadata missing"; exit 2; }

if [ "$MODEL" = "qwen35" ]; then
  SCALE=qwen3.5-4b; TOK=/scratch/users/prasann/hf_models/Qwen3.5-4B-Base
  IDS="--doc-start-id 248049 --doc-end-id 248050 --eos-token-id 248044"
else
  SCALE=qwen3-4b;   TOK=/scratch/users/prasann/hf_models/Qwen3-4B-Base  # flat dir, offline-safe (same tokenizer as Qwen3-4B)
  IDS="--doc-start-id 151648 --doc-end-id 151649 --eos-token-id 151643"
fi

echo "=== eval MODEL=$MODEL scale=$SCALE ckpt=$CKPT $(date) ==="
python -m scripts.eval.ctc_suite.run_rung_eval \
  --task contradiction --ckpt "$CKPT" \
  --variant dense --arm full --model-scale "$SCALE" \
  --rung-tokens 2048 --eval-jsonl "$EVAL_JSONL" \
  --tokenizer "$TOK" $IDS \
  --max-length 6144 --small-eval-ok --nproc 2 \
  --master-port $((29500 + SLURM_JOB_ID % 2000))
echo "=== DONE rc=$? $(date). result+gens under results/ctc_suite/contradiction/ ==="
