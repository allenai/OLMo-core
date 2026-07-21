#!/bin/bash
# Box-marker document-chunked DENSE tokenization for the single-task ladder (v2), run NODE-LOCAL on
# mooney against the LATEST unified JSONL sources under /data/prasann/single_task_ladders_20k/.
# Wraps src/scripts/data/convert_unified_to_document_landmark.py (--emit dense, --cot-mode none,
# --seq-len 40960). Output -> /data/prasann/single_task_docchunk_v2/<task>_dense/. Per-task
# (task name + chunk-by) matches the ladder40k provenance (convert_docchunk_5task_gantry.sh):
#   contra->contradiction/document  nq->retrieval/document  oolong->oolong/line
#   outlier->outlier/document        rerank->rerank/document
#
#   sbatch --job-name=dc-v2 --export=ALL,TASKS="contra nq oolong outlier" \
#     src/scripts/data/convert_docchunk_singletask_v2_local.sbatch
#   # rerank (after CE regen):
#   sbatch --job-name=dc-v2-rr --export=ALL,TASKS="rerank" src/scripts/data/convert_docchunk_singletask_v2_local.sbatch
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive
#SBATCH --nodelist=mooney
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/.dcv2_%x_%j.log

set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
SRC="${SRC:-/data/prasann/single_task_ladders_20k}"
# Overridable so a REBUILD lands in a NEW root instead of silently redefining an existing one.
# The shards under single_task_docchunk_v2/ were tokenized 2026-06-30, i.e. BEFORE
# abccf2837 ("docchunk: fix FREE-token leak in chunk wrapping + drop titles by default"), so any
# model trained on them learned with the `Document [N] (Title: ...)` label OUTSIDE the box (global
# FREE role) while today's eval builds the label INSIDE the chunk -- a train/eval mismatch that
# reads as a modeling result. Rebuild into e.g. single_task_docchunk_v2fix and re-point the
# launcher's DOCCHUNK_DATA_ROOT rather than overwriting a path other runs already cite.
OUT="${OUT:-/data/prasann/single_task_docchunk_v2}"
CONV=$REPO/src/scripts/data/convert_unified_to_document_landmark.py
TASKS="${TASKS:-contra nq oolong outlier}"
echo "=== HOST=$(hostname) TASKS='$TASKS' START=$(date '+%F %T') ==="

source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
OLMOENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$OLMOENV" ] || OLMOENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
conda activate "$OLMOENV"
export PYTHONPATH="$REPO/src" HF_HOME=/data/prasann/hf_cache TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p "$OUT"

# task -> "unified_task chunk_by relative_jsonl_path"
declare -A SPEC=(
 [contra]="contradiction document contradiction/gen_full/contradiction_train_pubmed_realistic_n50-950_k3.jsonl"
 [nq]="retrieval document nq/nq_train_k20-200_combined_aligned.jsonl"
 [oolong]="oolong line oolong/raw/oolong_ladder_train_combined.jsonl"
 [outlier]="outlier document outlier/outlier_wiki100w_contin_n14-220_k3_20000.jsonl"
 [rerank]="rerank document rerank/ce_gen/msmarco_trainhn_train_k20-315_20000.jsonl"
)

for name in $TASKS; do
  read -r task chunk jsonl <<< "${SPEC[$name]}"
  extra=""; [ "$chunk" = line ] && extra="--item-regex ||"
  outdir="$OUT/${name}_dense"
  src="$SRC/$jsonl"
  echo "=== [$name] task=$task chunk=$chunk src=$src -> $outdir $(date '+%T') ==="
  [ -f "$src" ] || { echo "  SOURCE MISSING: $src -- SKIP"; continue; }
  rm -rf "$outdir"; mkdir -p "$outdir"
  python "$CONV" --emit dense --task "$task" --chunk-by "$chunk" $extra --cot-mode none \
    --seq-len 40960 --shard-tokens 50000000 --input-jsonl "$src" --out-dir "$outdir"
  rc=$?; echo "  convert rc=$rc $(date '+%T')"
done

echo "=== VERIFY box markers $(date '+%T') ==="
python3 - "$OUT" $TASKS <<'PY'
import sys,glob,numpy as np
out=sys.argv[1]; tasks=sys.argv[2:]
EOS=151643; DS=151648; DE=151649
allok=True
for name in tasks:
    d=f"{out}/{name}_dense"; fs=sorted(glob.glob(d+"/token_ids_part_*.npy"))
    if not fs: print(f"[{name}] NO SHARDS"); allok=False; continue
    a=np.concatenate([np.fromfile(f,dtype=np.uint32) for f in fs])
    ds=int((a==DS).sum()); de=int((a==DE).sum()); eos=int((a==EOS).sum())
    ok = ds>0 and ds==de
    allok = allok and ok
    print(f"[{name}] toks={len(a)} examples(eos)={eos} box_start={ds} box_end={de} OK={ok}")
print("ALL_OK" if allok else "SOME_FAILED")
PY
echo "=== DONE $(date '+%T') ==="
