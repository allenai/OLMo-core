#!/bin/bash
# Contradiction-ONLY native eval (no ruler/nq) of a local Qwen3-0.6B n20 SFT checkpoint on mooney.
# Uses the same eval_lc_native.py that produced the 4B contra numbers, but pointed at the n20 eval
# set (apples-to-apples with the n20 SFT) and Qwen3-0.6B tokenizer. 1 GPU.
#   sbatch --export=ALL,MODEL_PATH=/data/prasann/olmo_ckpts/<run>/step750,OUT_NAME=q06b-dense-contra-n20,VARIANT=dense \
#     src/scripts/train/sft/eval_q06b_contra_n20_native.sbatch
#SBATCH --job-name=q06b-contra-eval
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=mooney
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/data/prasann/joblogs/q06b_contra_eval_%x_%j.log
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
CR=/scratch/users/prasann/corpus-reasoning
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH}"
OUT_NAME="${OUT_NAME:?set OUT_NAME}"
VARIANT="${VARIANT:-dense}"           # dense | landmark | compressive
CONTRA_DATA="${CONTRA_DATA:-data/contradiction_eval_pubmed_both_n20_k3.jsonl}"
# Our local SFT checkpointer does NOT embed config.json in the step dir, but eval_lc_native.py's
# generation-module .build() needs it. The SFT never changes architecture, so a matching olmo_core
# base config.json works. Use the flash_2-backed 64k base configs (the raw base config has
# backend=None -> TorchAttentionBackend, which can't KV-cache during generate). Always overwrite so a
# previously-copied wrong config is corrected. BASE_CONFIG overrides the per-VARIANT default.
case "$VARIANT" in
  landmark)     DEF_BASE_CONFIG=/data/prasann/olmo_ckpts/q06_bases/fastlm/config.json ;;
  compressive)  DEF_BASE_CONFIG=/data/prasann/olmo_ckpts/q06_bases/comp/config.json ;;
  # dilated ("Hierarchical K" rotating-dilation mask): flash_2-backed dense base config with the
  # dilated_sliding_window attention fields injected (see make_dilated_config below) so the eval
  # applies the SAME positional mask the model was trained with (not full attention).
  dilated)      DEF_BASE_CONFIG=/data/prasann/olmo_ckpts/q06_bases/dilated/config.json ;;
  *)            DEF_BASE_CONFIG=/data/prasann/olmo_ckpts/q06_bases/dense/config.json ;;
esac
BASE_CONFIG="${BASE_CONFIG:-$DEF_BASE_CONFIG}"
echo "seeding step-dir config.json from $BASE_CONFIG"
cp "$BASE_CONFIG" "$MODEL_PATH/config.json"
TOK="${TOK:-Qwen/Qwen3-0.6B}"
MAXLEN="${MAXLEN:-4096}"
# landmark/compressive native decode can't be batched/left-padded -> bs=1 (per singletask_ladder/EVAL.md)
case "$VARIANT" in
  landmark|compressive) BS=1 ;;
  *)                    BS="${BS:-16}" ;;
esac

OLMOENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$OLMOENV" ] || OLMOENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
# Direct PATH instead of `conda activate` -- activate hangs for minutes in D-state on NFS lock
# contention (same reason the training sbatch uses direct PATH). The env bin on PATH is all we need.
export PATH="$OLMOENV/bin:$PATH"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/src:$CR"
cd "$CR"
echo "HOST=$(hostname) MODEL=$MODEL_PATH VARIANT=$VARIANT BS=$BS DATA=$CONTRA_DATA START=$(date '+%F %T')"
python -u scripts/eval/eval_lc_native.py \
  --model-path "$MODEL_PATH" \
  --out "outputs/eval_results/${OUT_NAME}_native.json" \
  --tokenizer "$TOK" --max-length "$MAXLEN" \
  --skip-ruler \
  --contra-data "$CONTRA_DATA" \
  --nq-data /nonexistent_skip_nq \
  --batch-size "$BS"
echo "DONE $(date '+%T') rc=$? -> outputs/eval_results/${OUT_NAME}_native.json"
