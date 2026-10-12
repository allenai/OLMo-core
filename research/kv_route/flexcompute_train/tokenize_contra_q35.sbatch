#!/bin/bash
# Re-tokenize the short-heavy contradiction arm for Qwen3.5 (marker set qwen3_5).
#
# The composed short-heavy JSONL arms from the FLOP-scaling build
# (research/ffn_compute/flop_scaling/build_shortheavy_data.sbatch, shape 45/27/16/8/4 % at 2k/4k/8k/16k/32k)
# were only ever tokenized for Qwen3 (`marker_set: qwen3`, ids 151648/151649/151643). The
# flexcompute training ladder runs Qwen3.5-4B, whose reserved ids are 248049/248050/248044, so
# the same JSONL has to be re-tokenized. Nothing about the DATA changes -- same rows, same
# shape, same seed -- only the tokenizer.
#
#   ARM=contradiction_sh16M sbatch research/kv_route/flexcompute_train/tokenize_contra_q35.sbatch
#SBATCH --job-name=fxt-tok
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=cubbins
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/data/prasann/joblogs/fxt_tok_%x_%j.log
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || git -C "${SLURM_SUBMIT_DIR:-.}" rev-parse --show-toplevel)}
REPO="${REPO:-$REPO_ROOT}"
ARM="${ARM:-contradiction_sh16M}"
WORK=/data/prasann/flexcompute_train
IN="$WORK/arms/${ARM}.jsonl"
OUT="$WORK/shards/${ARM}_q35"
TOKENIZER="${TOKENIZER:-/data/prasann/hf_models/Qwen3.5-0.8B-Base}"

ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$REPO/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
export TMPDIR=/data/prasann/tmp
mkdir -p "$OUT" "$TMPDIR"
echo "=== HOST=$(hostname) ARM=$ARM IN=$IN OUT=$OUT START=$(date '+%F %T') ==="
[ -s "$IN" ] || { echo "FATAL: missing $IN"; exit 2; }
wc -l "$IN"

python "$REPO/recipes/data/convert_unified_to_document_landmark.py" \
  --input-jsonl "$IN" --task contradiction --out-dir "$OUT" --emit dense \
  --marker-set qwen3_5 --tokenizer "$TOKENIZER" --seq-len 40960 \
  --query-position after --cot-mode none --chunk-by document --num-proc 16
rc=$?
echo "--- metadata ---"; cat "$OUT/metadata.json"
echo "=== DONE rc=$rc $(date '+%F %T') ==="
exit $rc
