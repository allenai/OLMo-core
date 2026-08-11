#!/bin/bash
# CTC-suite xabsence data build: assemble 20k mixed-n train + per-rung eval from the
# 659-pair paraphrase pool, then tokenize the joint train file to a Qwen3.5 dense
# document-chunked shard. CPU-only (no GPU) so it coexists with GPU jobs on cubbins.
#
#   sbatch src/scripts/data/ctc_suite/build_xabsence_20k_cubbins.sbatch
#
# Token->rung mapping (BUILD_MATRIX row 22, P18->2k confirmed by the 0.6B pilot shard
# max_example_len=2051): P18->2k P39->4k P81->8k P165->16k P333->32k. The job MEASURES
# each eval-P's converted max_example_len and prints it so the rung labels can be
# corrected if a measurement deviates.
#
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=cubbins
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=ctc-xabsence-build
#SBATCH --output=/data/prasann/joblogs/ctc_xabsence_build_%A.log

set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
POOL=/scratch/users/prasann/corpus-reasoning/data/xabsence_pool_pubmed.jsonl
GEN=$REPO/src/corpus_reasoning/data/generate_xabsence_data.py
CONV=$REPO/src/scripts/data/convert_unified_to_document_landmark.py
RAW=/data/prasann/ctc_suite_data/xabsence/raw
OUT=/data/prasann/ctc_suite/shards_final
STAGE=/scratch/users/prasann/ctc_suite_staged/eval_rungs/xabsence
mkdir -p "$RAW" "$OUT" "$STAGE" /data/prasann/joblogs
# node-local /data logs are unreadable from the login host -> mirror the SLURM log to
# shared /scratch every 20s (and once more at exit) so progress is visible off-node.
SCRATCH_LOGDIR=/scratch/users/prasann/ctc_suite_logs
mkdir -p "$SCRATCH_LOGDIR"
SLURM_LOG=/data/prasann/joblogs/ctc_xabsence_build_${SLURM_JOB_ID}.log
MIRROR=$SCRATCH_LOGDIR/ctc_xabsence_build_${SLURM_JOB_ID}.log
( while true; do cp -f "$SLURM_LOG" "$MIRROR" 2>/dev/null; sleep 20; done ) &
MIRROR_PID=$!
trap 'cp -f "$SLURM_LOG" "$MIRROR" 2>/dev/null; kill $MIRROR_PID 2>/dev/null' EXIT

echo "=== HOST=$(hostname) START=$(date '+%F %T') POOL_LINES=$(wc -l < "$POOL") ==="
source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate /data/prasann/conda/envs/corpus-reasoning-olmo
export PYTHONPATH="$REPO/src" HF_HOME=/data/prasann/hf_cache TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

# ---- 1. assemble per-P (train 4000 + eval 500) ----
for P in 18 39 81 165 333; do
  echo "--- assemble P=$P ---"
  python "$GEN" --pool "$POOL" --num-pairs $P --num-unmatched 3 \
    --num-train 4000 --num-eval 500 --src-tag pubmed --output-dir "$RAW" --seed 42
done

# ---- 2. cat train -> joint mixed-n 20k ----
JOINT="$RAW/xabsence_train_20k_mixedn.jsonl"
cat "$RAW"/xabsence_train_pubmed_p18_k3.jsonl \
    "$RAW"/xabsence_train_pubmed_p39_k3.jsonl \
    "$RAW"/xabsence_train_pubmed_p81_k3.jsonl \
    "$RAW"/xabsence_train_pubmed_p165_k3.jsonl \
    "$RAW"/xabsence_train_pubmed_p333_k3.jsonl > "$JOINT"
echo "joint train lines=$(wc -l < "$JOINT")"

# ---- 3. convert joint train shard ----
OUTDIR="$OUT/xabsence_train"
rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"
python "$CONV" --emit dense --task xabsence --chunk-by document --cot-mode none \
  --marker-set qwen3_5 --tokenizer Qwen/Qwen3.5-0.8B-Base \
  --seq-len 40960 --shard-tokens 60000000 \
  --input-jsonl "$JOINT" --out-dir "$OUTDIR"
echo "train convert rc=$?"
python3 - "$OUTDIR" <<'PY'
import sys, json
m = json.load(open(sys.argv[1]+"/metadata.json"))
print("TRAIN meta:", {k: m.get(k) for k in (
    "task","num_instances","num_dropped","max_example_len","min_example_len",
    "marker_set","doc_start_id","doc_end_id","eos_token_id")})
PY

# ---- 4. measure each eval-P token length + stage eval JSONL by rung label ----
declare -A RUNG=( [18]=2048 [39]=4096 [81]=8192 [165]=16384 [333]=32768 )
for P in 18 39 81 165 333; do
  EV="$RAW/xabsence_eval_pubmed_p${P}_k3.jsonl"
  MDIR="$OUT/_xabsence_evalmeasure_p${P}"
  rm -rf "$MDIR"; mkdir -p "$MDIR"
  python "$CONV" --emit dense --task xabsence --chunk-by document --cot-mode none \
    --marker-set qwen3_5 --tokenizer Qwen/Qwen3.5-0.8B-Base \
    --seq-len 40960 --shard-tokens 60000000 \
    --input-jsonl "$EV" --out-dir "$MDIR" >/dev/null 2>&1
  STATS=$(python3 - "$MDIR" <<'PY'
import sys, json
m = json.load(open(sys.argv[1]+"/metadata.json"))
print(f"max={m.get('max_example_len')} min={m.get('min_example_len')} n={m.get('num_instances')} dropped={m.get('num_dropped')}")
PY
)
  echo "EVAL P=$P rung=${RUNG[$P]} $STATS"
  cp "$EV" "$STAGE/rung_${RUNG[$P]}.jsonl"
  rm -rf "$MDIR"
done
echo "=== staged eval rungs ==="; ls -la "$STAGE"
echo "=== DONE $(date '+%F %T') ==="
