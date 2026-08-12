#!/bin/bash
# Re-run the FULL arm after `Transformer.disable_document_chunk_attention` was added, plus the
# mechanism probe. Before the fix, `--attn full` left the chunked mask on and produced 500/500
# generations byte-identical to the chunked arm; if the fix works these two must now differ.
#SBATCH --job-name=ctc-eval-regrade
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=mooney
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=/data/prasann/joblogs/ctc-eval-regrade-%j.log
set -euo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/newolmocore/OLMo-core
WORK=/data/prasann/ctc_eval_loop
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
export HOME=$WORK/home TMPDIR=$WORK/tmp TRITON_CACHE_DIR=$WORK/triton HF_HOME=$WORK/hf HF_HUB_OFFLINE=1
cd "$REPO"; export PYTHONPATH=src:ctc/src

echo "=== probe: is the mask applied, and does full now differ from chunked? $(date +%T) ==="
$PY debug/eval_loop_close/probe_mask_applied.py --ckpt $WORK/ckpt/step30 --tokenizer $WORK/tok \
    --data $WORK/data/mathmatch/eval_2k.jsonl

echo ""
echo "=== ARM E (full), re-run  $(date +%T) ==="
$PY debug/eval_loop_close/ctc_eval_mathmatch.py --data $WORK/data/mathmatch/eval_2k.jsonl \
    --ckpt $WORK/ckpt/step30 --tasks mathmatch --backend native --tokenizer $WORK/tok \
    --max-length 4096 --out $WORK/results --overwrite \
    --attn full --query-position after --tag full
echo "=== ARM E EXIT=$? ==="
echo "=== DONE $(date +%T) ==="
