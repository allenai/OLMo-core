#!/bin/bash
# CTC suite FINAL 20k dense document-chunked tokenization, Qwen3.5 marker set.
# Mirrors convert_ctc_p0_dense_cubbins.sbatch but for the FINAL 20k mixed-n pools
# under /data/prasann/ctc_suite_data/<task>/final20k/ (built by gen20k_batch{A,B,C,D}),
# plus reuse-pool tasks (nq/msmarco/grouping_labeled) whose inputs pre-exist.
# seq-len 40960 (fits 32k; cap16k tasks are naturally short so no waste).
# Output -> /data/prasann/ctc_suite/shards_final/<task>_train/.
#
#   sbatch --array=0-17 src/scripts/data/ctc_suite/convert_ctc_final_dense_cubbins.sbatch
#
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=cubbins
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --job-name=ctc-final-conv
#SBATCH --output=/data/prasann/joblogs/ctc_final_conv_%a_%A.log

set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
DATA=/data/prasann/ctc_suite_data
OUT=/data/prasann/ctc_suite/shards_final
CONV=$REPO/src/scripts/data/convert_unified_to_document_landmark.py
mkdir -p "$OUT" /data/prasann/joblogs

# index: name
TASKS=(nq msmarco grouping_labeled niah scifact fiqa outlier_amzn textgroups \
       absence_gutenberg strmatch qdmatch_nq qdmatch_hpqa mathmatch cycle groups4 \
       helmet_qa helmet_summ msmarco_rerank obliq_retrieval)
NAME="${TASKS[$SLURM_ARRAY_TASK_ID]}"

# per-task: TASK key, CHUNK, and INPUT file(s)
FINAL=""
case "$NAME" in
  nq)               TASK=retrieval;  CHUNK=document; FILES=(/scratch/users/prasann/nq_p10_20k/nq_train_k25-202_clean.jsonl) ;;
  msmarco)          TASK=retrieval;  CHUNK=document; FILES=("$DATA/rerank_pool/msmarco_trainhn_train_k20-315_20000.jsonl") ;;
  msmarco_rerank)   TASK=rerank;     CHUNK=document; FILES=("$DATA/rerank_pool/msmarco_trainhn_train_k20-315_20000.jsonl") ;;
  grouping_labeled) TASK=grouping_labeled; CHUNK=document; FILES=("$DATA"/grouping_pool/openalex_grouping_n*_levels_train_4000.jsonl) ;;
  niah)             TASK=retrieval;  CHUNK=document; FILES=("$DATA/niah/final20k/niah_train_20k_mixedn.jsonl") ;;
  scifact)          TASK=retrieval;  CHUNK=document; FILES=("$DATA/scifact/final20k/scifact_train_20k_mixedn.jsonl") ;;
  fiqa)             TASK=retrieval;  CHUNK=document; FILES=("$DATA/fiqa/final20k/fiqa_train_20k_mixedn.jsonl") ;;
  outlier_amzn)     TASK=outlier;    CHUNK=document; FILES=("$DATA/outlier_amzn/final20k/outlier_amzn_train_20k_mixedn.jsonl") ;;
  textgroups)       TASK=textgroups; CHUNK=document; FILES=("$DATA/textgroups/final20k/textgroups_train_20k_mixedn.jsonl") ;;
  absence_gutenberg)TASK=absence;    CHUNK=document; FILES=("$DATA/absence_gutenberg/final20k/absence_gutenberg_train_20k_mixedn.jsonl") ;;
  strmatch)         TASK=strmatch;   CHUNK=document; FILES=("$DATA/strmatch/final20k/strmatch_train_20k_mixedn.jsonl") ;;
  qdmatch_nq)       TASK=qdmatch;    CHUNK=document; FILES=("$DATA/qdmatch_nq/final20k/qdmatch_nq_train_20k_mixedn.jsonl") ;;
  qdmatch_hpqa)     TASK=qdmatch;    CHUNK=document; FILES=("$DATA/qdmatch_hpqa/final20k/qdmatch_hpqa_train_20k_mixedn.jsonl") ;;
  mathmatch)        TASK=mathmatch;  CHUNK=document; FILES=("$DATA/mathmatch/final20k/mathmatch_train_20k_mixedn.jsonl") ;;
  cycle)            TASK=cycle;      CHUNK=document; FILES=("$DATA/cycle/final20k/cycle_train_20k_mixedn.jsonl") ;;
  groups4)          TASK=groups4;    CHUNK=document; FILES=("$DATA/groups4/final20k/groups4_train_20k_mixedn.jsonl") ;;
  helmet_qa)        TASK=qa;         CHUNK=document; FILES=("$DATA/helmet_qa/final20k/helmet_qa_train_20k_mixedn.jsonl") ;;
  helmet_summ)      TASK=summarization; CHUNK=document; FILES=("$DATA/helmet_summ/final20k/helmet_summ_train_20k_mixedn.jsonl") ;;
  obliq_retrieval)  TASK=retrieval;  CHUNK=document; FILES=(/scratch/users/prasann/corpus-reasoning/data/obliq_mix4_train_1797.jsonl) ;;
  *) echo "unknown idx $SLURM_ARRAY_TASK_ID -> $NAME"; exit 1 ;;
esac

echo "=== HOST=$(hostname) NAME=$NAME TASK=$TASK START=$(date '+%F %T') ==="
for f in "${FILES[@]}"; do [ -f "$f" ] || { echo "  MISSING INPUT: $f"; exit 2; }; done
echo "  inputs: ${FILES[*]}  total_lines=$(cat "${FILES[@]}" | wc -l)"

source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate /data/prasann/conda/envs/corpus-reasoning-olmo
export PYTHONPATH="$REPO/src" HF_HOME=/data/prasann/hf_cache TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

OUTDIR="$OUT/${NAME}_train"
rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"
python "$CONV" --emit dense --task "$TASK" --chunk-by "$CHUNK" --cot-mode none \
  --marker-set qwen3_5 --tokenizer Qwen/Qwen3.5-0.8B-Base \
  --seq-len 40960 --shard-tokens 60000000 \
  --input-jsonl "${FILES[@]}" --out-dir "$OUTDIR"
echo "  convert rc=$? $(date '+%T')"

echo "=== VERIFY ($NAME) ==="
python3 - "$OUTDIR" <<'PY'
import sys, glob, json, numpy as np
d = sys.argv[1]
try:
    meta = json.load(open(d+"/metadata.json"))
    print("meta:", {k: meta.get(k) for k in ("marker_set","doc_start_id","doc_end_id","eos_token_id")})
except Exception as e:
    print("NO metadata.json:", e)
fs = sorted(glob.glob(d+"/token_ids_part_*.npy"))
if not fs: print("NO SHARDS"); sys.exit(1)
DS,DE,EOS = 248049,248050,248044
a = np.concatenate([np.fromfile(f, dtype=np.uint32) for f in fs])
ds=int((a==DS).sum()); de=int((a==DE).sum()); eos=int((a==EOS).sum())
print(f"toks={len(a)} examples(eos)={eos} box_start={ds} box_end={de} avg_docs/ex={ds/max(1,eos):.1f} OK={ds>0 and ds==de}")
print("ALL_OK" if (ds>0 and ds==de) else "FAILED")
PY
echo "=== DONE $NAME $(date '+%T') ==="
