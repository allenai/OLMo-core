#!/bin/bash
# CTC suite PILOT dense document-chunked tokenization, Qwen3.5 marker set, run
# NODE-LOCAL on cubbins against the pilot-scale (~2.5k train @ n(2k)) pools under
# /data/prasann/ctc_suite_data/<task>/ built 2026-07-19 (see BUILD_MATRIX.md
# "PILOT data build" log). Mirrors convert_ctc_p0_dense_cubbins.sbatch but at
# --seq-len 4096 (pilot) instead of 40960 (final) and single-n files (no rungs).
#
# Output -> /data/prasann/ctc_suite/shards_pilot/<task>_train/.
#
#   sbatch --array=0-19 src/scripts/data/ctc_suite/convert_ctc_pilot_dense_cubbins.sbatch
#   sbatch --array=5    src/scripts/data/ctc_suite/convert_ctc_pilot_dense_cubbins.sbatch   # single task
#
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=cubbins
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --job-name=ctc-pilot-conv
#SBATCH --output=/data/prasann/joblogs/ctc_pilot_conv_%a_%A.log

set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
DATA=/data/prasann/ctc_suite_data
OUT=/data/prasann/ctc_suite/shards_pilot
CONV=$REPO/src/scripts/data/convert_unified_to_document_landmark.py
mkdir -p "$OUT" /data/prasann/joblogs

TASKS=(niah absence_pubmed qdmatch_nq qdmatch_hpqa qdmatch_obliq grouping_labeled \
       mathmatch cycle groups4 textgroups strmatch outlier_amzn helmet_qa helmet_summ \
       xabsence scifact fiqa msmarco msmarco_rerank nq obliq_retrieval absence_gutenberg)
NAME="${TASKS[$SLURM_ARRAY_TASK_ID]}"

# All train inputs are pre-harvested (trimmed to <=2500 rows) by
# harvest_pilot_all.py into a single canonical file per task:
#   $DATA/pilot_train/<task>.jsonl
# (msmarco_rerank reuses msmarco's harvested file under a different --task key.)
PILOT_TRAIN=$DATA/pilot_train
case "$NAME" in
  niah)              TASK=retrieval; CHUNK=document ;;
  absence_pubmed)    TASK=absence; CHUNK=document ;;
  qdmatch_nq)        TASK=qdmatch; CHUNK=document ;;
  qdmatch_hpqa)      TASK=qdmatch; CHUNK=document ;;
  qdmatch_obliq)     TASK=qdmatch; CHUNK=document ;;
  grouping_labeled)  TASK=grouping_labeled; CHUNK=document ;;
  mathmatch)         TASK=mathmatch; CHUNK=document ;;
  cycle)             TASK=cycle; CHUNK=document ;;
  groups4)           TASK=groups4; CHUNK=document ;;
  textgroups)        TASK=textgroups; CHUNK=document ;;
  strmatch)          TASK=strmatch; CHUNK=document ;;
  outlier_amzn)      TASK=outlier; CHUNK=document ;;
  helmet_qa)         TASK=qa; CHUNK=document ;;
  helmet_summ)       TASK=summarization; CHUNK=document ;;
  xabsence)          TASK=xabsence; CHUNK=document ;;
  scifact)           TASK=retrieval; CHUNK=document ;;
  fiqa)               TASK=retrieval; CHUNK=document ;;
  msmarco)           TASK=retrieval; CHUNK=document ;;
  msmarco_rerank)    TASK=rerank; CHUNK=document ;;
  nq)                TASK=retrieval; CHUNK=document ;;
  obliq_retrieval)   TASK=retrieval; CHUNK=document ;;
  absence_gutenberg) TASK=absence; CHUNK=document ;;
  *)
    echo "unknown task index $SLURM_ARRAY_TASK_ID -> $NAME"; exit 1 ;;
esac
if [ "$NAME" = "msmarco_rerank" ]; then
  FILES=("$PILOT_TRAIN/msmarco.jsonl")
else
  FILES=("$PILOT_TRAIN/$NAME.jsonl")
fi

echo "=== HOST=$(hostname) NAME=$NAME TASK=$TASK CHUNK=$CHUNK FILES=${#FILES[@]} START=$(date '+%F %T') ==="
if [ "${#FILES[@]}" -eq 0 ]; then
  echo "  NO INPUT FILES MATCHED for $NAME"; exit 1
fi
for f in "${FILES[@]}"; do
  [ -f "$f" ] || { echo "  MISSING INPUT: $f"; exit 1; }
done

source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate /data/prasann/conda/envs/corpus-reasoning-olmo
export PYTHONPATH="$REPO/src" HF_HOME=/data/prasann/hf_cache TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

OUTDIR="$OUT/${NAME}_train"
rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"

python "$CONV" --emit dense --task "$TASK" --chunk-by "$CHUNK" --cot-mode none \
  --marker-set qwen3_5 --tokenizer Qwen/Qwen3.5-0.8B-Base \
  --seq-len 4096 --shard-tokens 20000000 \
  --input-jsonl "${FILES[@]}" --out-dir "$OUTDIR"
rc=$?
echo "  convert rc=$rc $(date '+%T')"

echo "=== VERIFY box markers ($NAME) $(date '+%T') ==="
python3 - "$OUTDIR" <<'PY'
import sys, glob, json, numpy as np
d = sys.argv[1]
meta_path = d + "/metadata.json"
try:
    meta = json.load(open(meta_path))
    print("metadata:", {k: meta.get(k) for k in ("marker_set", "doc_start_id", "doc_end_id", "eos_token_id")})
except Exception as e:
    print("NO/BAD metadata.json:", e)
fs = sorted(glob.glob(d + "/token_ids_part_*.npy"))
if not fs:
    print("NO SHARDS"); sys.exit(1)
DS, DE, EOS = 248049, 248050, 248044
a = np.concatenate([np.fromfile(f, dtype=np.uint32) for f in fs])
ds = int((a == DS).sum()); de = int((a == DE).sum()); eos = int((a == EOS).sum())
ok = ds > 0 and ds == de
print(f"toks={len(a)} examples(eos)={eos} box_start={ds} box_end={de} OK={ok}")
print("ALL_OK" if ok else "FAILED")
PY
echo "=== DONE $NAME $(date '+%T') ==="
